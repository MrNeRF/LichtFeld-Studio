/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/cuda/sh_layout.cuh"
#include "core/scene.hpp"
#include "core/sh_value_quant.hpp"
#include "core/sh_value_quant_kernels.hpp"
#include "core/splat_data.hpp"
#include "core/splat_exportable_storage.hpp"
#include "core/tensor.hpp"
#include "io/exporter.hpp"
#include "io/loader.hpp"
#include "lfs/training/live_model_mutation_guard.hpp"
#include "lfs/training/sh_value_codec.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include "lfs/training/vram_ledger.hpp"
#include "training/optimizer/adam_optimizer.hpp"
#include "training/rasterization/fast_rasterizer.hpp"
#include "training/rasterization/fastgs/rasterization/include/rasterization_config.h"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <filesystem>
#include <glm/mat4x4.hpp>
#include <gtest/gtest.h>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lfs::core;
using namespace lfs::training;

namespace {

    constexpr size_t kN = 256; // one full quant block
    constexpr int kShDegree = 3;

    SplatData make_random_sh3(const size_t n, const uint32_t seed = 42) {
        auto means = Tensor::zeros({n, size_t{3}}, Device::CUDA, DataType::Float32);
        auto sh0 = Tensor::zeros({n, size_t{1}, size_t{3}}, Device::CUDA, DataType::Float32);
        auto shN_can = Tensor::zeros({n, size_t{15}, size_t{3}}, Device::CUDA, DataType::Float32);
        auto scaling = Tensor::zeros({n, size_t{3}}, Device::CUDA, DataType::Float32);
        auto rotation = Tensor::zeros({n, size_t{4}}, Device::CUDA, DataType::Float32);
        auto opacity = Tensor::zeros({n, size_t{1}}, Device::CUDA, DataType::Float32);

        {
            std::mt19937 rng(seed);
            std::normal_distribution<float> nd(0.0f, 0.15f);
            auto cpu = shN_can.cpu();
            auto* p = cpu.ptr<float>();
            for (size_t i = 0; i < n * 15 * 3; ++i)
                p[i] = nd(rng);
            shN_can = cpu.to(Device::CUDA);

            auto rcpu = rotation.cpu();
            auto* r = rcpu.ptr<float>();
            for (size_t i = 0; i < n; ++i)
                r[i * 4] = 1.0f;
            rotation = rcpu.to(Device::CUDA);
        }

        return SplatData(kShDegree, means, sh0, shN_can, scaling, rotation, opacity, 1.0f);
    }

    [[nodiscard]] double mse_tensors(const Tensor& a, const Tensor& b) {
        EXPECT_EQ(a.numel(), b.numel());
        auto ac = a.cpu().contiguous();
        auto bc = b.cpu().contiguous();
        const auto* pa = ac.ptr<float>();
        const auto* pb = bc.ptr<float>();
        double mse = 0.0;
        for (size_t i = 0; i < a.numel(); ++i) {
            const double e = static_cast<double>(pa[i]) - static_cast<double>(pb[i]);
            mse += e * e;
        }
        return mse / static_cast<double>(a.numel());
    }

    [[nodiscard]] double psnr_from_mse(double mse) {
        if (mse <= 0.0)
            return 100.0;
        // peak = 1.0 for SH coeff range proxy
        return 10.0 * std::log10(1.0 / mse);
    }

    Tensor retag_external(Tensor tensor, std::string kind) {
        const TensorShape shape = tensor.shape();
        const auto device = tensor.device();
        const auto dtype = tensor.dtype();
        const size_t capacity = tensor.capacity();
        const cudaStream_t stream = tensor.stream();
        auto owner = std::make_shared<Tensor>(std::move(tensor));
        return Tensor::from_external_owner(owner->data_ptr(),
                                           shape,
                                           device,
                                           dtype,
                                           owner,
                                           capacity,
                                           stream,
                                           std::move(kind));
    }

    void retag_viewer_external(SplatData& splat,
                               const std::string& shN_kind = "vulkan_external_buffer",
                               const std::string& bounds_kind = "vulkan_external_buffer") {
        splat.means_raw() = retag_external(std::move(splat.means_raw()), "vulkan_external_buffer");
        splat.sh0_raw() = retag_external(std::move(splat.sh0_raw()), "vulkan_external_buffer");
        splat.shN_raw() = retag_external(std::move(splat.shN_raw()), shN_kind);
        splat.shN_value_bounds() =
            retag_external(std::move(splat.shN_value_bounds()), bounds_kind);
        splat.scaling_raw() =
            retag_external(std::move(splat.scaling_raw()), "vulkan_external_buffer");
        splat.rotation_raw() =
            retag_external(std::move(splat.rotation_raw()), "vulkan_external_buffer");
        splat.opacity_raw() =
            retag_external(std::move(splat.opacity_raw()), "vulkan_external_buffer");
    }

    SplatTensorAllocator counting_viewer_allocator(int& allocation_calls) {
        return [&allocation_calls](TensorShape shape,
                                   const size_t capacity,
                                   const DataType dtype,
                                   std::string_view) {
            ++allocation_calls;
            Tensor backing = Tensor::zeros_direct(shape, capacity, Device::CUDA, dtype);
            return retag_external(std::move(backing), "vulkan_external_buffer");
        };
    }

} // namespace

TEST(ShValueStorageTest, GpuEncodeDecodeRoundtripLowMse) {
    sh_value::set_sh_value_quant_enabled_for_testing(true);
    auto splat = make_random_sh3(kN);
    const auto before = splat.shN_canonical().cpu().contiguous();

    ASSERT_TRUE(sh_value::apply_shN_value_quant(splat));
    ASSERT_TRUE(splat.shN_value_quantized());
    ASSERT_TRUE(splat.shN_value_bounds().is_valid());

    // Expand back and compare to original via float4 decode.
    ASSERT_TRUE(sh_value::ensure_shN_fp32_for_mutation(splat));
    ASSERT_FALSE(splat.shN_value_quantized());
    const auto after = splat.shN_canonical().cpu().contiguous();

    const double mse = mse_tensors(before, after);
    EXPECT_LT(mse, 1e-6) << "MSE=" << mse;
    EXPECT_GT(psnr_from_mse(mse), 55.0) << "PSNR from MSE=" << psnr_from_mse(mse);

    sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
}

TEST(ShValueStorageTest, CanonicalExportIsFp32BitCompat) {
    sh_value::set_sh_value_quant_enabled_for_testing(true);
    auto splat = make_random_sh3(64);
    const auto ref = splat.shN_canonical().cpu().contiguous();
    ASSERT_TRUE(sh_value::apply_shN_value_quant(splat));

    const auto deq = splat.shN_canonical();
    EXPECT_EQ(deq.dtype(), DataType::Float32);
    EXPECT_EQ(deq.ndim(), 3u);
    EXPECT_EQ(deq.shape()[0], 64u);
    EXPECT_EQ(deq.shape()[1], 15u);
    EXPECT_EQ(deq.shape()[2], 3u);

    const auto deq_cpu = splat.shN_canonical_cpu();
    EXPECT_EQ(deq_cpu.device(), Device::CPU);
    EXPECT_EQ(deq_cpu.dtype(), DataType::Float32);

    const double mse = mse_tensors(ref, deq.cpu());
    EXPECT_LT(mse, 1e-6);
    EXPECT_GT(psnr_from_mse(mse), 55.0);

    sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
}

TEST(ShValueStorageTest, Q16CloneCarriesBoundsAndDecodesIdentically) {
    sh_value::set_sh_value_quant_enabled_for_testing(true);
    auto source = make_random_sh3(kN);
    ASSERT_TRUE(sh_value::apply_shN_value_quant(source));
    source.set_active_sh_degree(1);

    auto clone = std::make_unique<SplatData>(
        source.get_max_sh_degree(),
        source.means_raw().clone(),
        source.sh0_raw().clone(),
        source.clone_shN_storage(),
        source.scaling_raw().clone(),
        source.rotation_raw().clone(),
        source.opacity_raw().clone(),
        source.get_scene_scale(),
        SplatData::ShNLayout::Swizzled);
    clone->shN_value_bounds() = source.shN_value_bounds().clone();
    clone->set_active_sh_degree(source.get_active_sh_degree());
    clone->set_max_sh_degree(source.get_max_sh_degree());

    ASSERT_TRUE(clone->shN_value_quantized());
    EXPECT_EQ(clone->shN_value_bounds().shape(), source.shN_value_bounds().shape());
    const auto source_canonical = source.shN_canonical().cpu().contiguous();
    const auto clone_canonical = clone->shN_canonical().cpu().contiguous();
    EXPECT_EQ(source_canonical.numel(), clone_canonical.numel());
    EXPECT_LT(mse_tensors(source_canonical, clone_canonical), 1e-12);

    sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
}

TEST(ShValueStorageTest, ViewerExternalBindAcceptsCompleteQ16PairWithoutRehome) {
    sh_value::set_sh_value_quant_enabled_for_testing(true);
    auto splat = make_random_sh3(64);
    ASSERT_TRUE(sh_value::apply_shN_value_quant(splat));
    retag_viewer_external(splat);

    int allocation_calls = 0;
    const auto result = lfs::io::migrateSplatTensorsToAllocator(
        splat, counting_viewer_allocator(allocation_calls));

    ASSERT_TRUE(result.has_value()) << result.error().format();
    EXPECT_EQ(allocation_calls, 0);
    EXPECT_TRUE(splat.shN_value_quantized());
    EXPECT_EQ(splat.shN_raw().external_storage_kind(), "vulkan_external_buffer");
    EXPECT_EQ(splat.shN_value_bounds().external_storage_kind(), "vulkan_external_buffer");
    sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
}

TEST(ShValueStorageTest, ViewerExternalBindRehomesDegradedQ16BoundsAsPair) {
    sh_value::set_sh_value_quant_enabled_for_testing(true);
    auto splat = make_random_sh3(64);
    ASSERT_TRUE(sh_value::apply_shN_value_quant(splat));
    const Tensor reference = splat.shN_canonical().cpu().contiguous();
    retag_viewer_external(splat, "vulkan_external_buffer", "degraded_external_buffer");

    int allocation_calls = 0;
    const auto result = lfs::io::migrateSplatTensorsToAllocator(
        splat, counting_viewer_allocator(allocation_calls));

    ASSERT_TRUE(result.has_value()) << result.error().format();
    EXPECT_EQ(allocation_calls, 7);
    ASSERT_TRUE(splat.shN_value_quantized());
    EXPECT_EQ(splat.shN_raw().external_storage_kind(), "vulkan_external_buffer");
    EXPECT_EQ(splat.shN_value_bounds().external_storage_kind(), "vulkan_external_buffer");
    EXPECT_LT(mse_tensors(reference, splat.shN_canonical().cpu()), 1e-12);
    sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
}

TEST(ShValueStorageTest, ViewerExternalBindRehomesDegradedQ16CodesAsPair) {
    sh_value::set_sh_value_quant_enabled_for_testing(true);
    auto splat = make_random_sh3(64);
    ASSERT_TRUE(sh_value::apply_shN_value_quant(splat));
    const Tensor reference = splat.shN_canonical().cpu().contiguous();
    retag_viewer_external(splat, "degraded_external_buffer", "vulkan_external_buffer");

    int allocation_calls = 0;
    const auto result = lfs::io::migrateSplatTensorsToAllocator(
        splat, counting_viewer_allocator(allocation_calls));

    ASSERT_TRUE(result.has_value()) << result.error().format();
    EXPECT_EQ(allocation_calls, 7);
    ASSERT_TRUE(splat.shN_value_quantized());
    EXPECT_EQ(splat.shN_raw().external_storage_kind(), "vulkan_external_buffer");
    EXPECT_EQ(splat.shN_value_bounds().external_storage_kind(), "vulkan_external_buffer");
    EXPECT_LT(mse_tensors(reference, splat.shN_canonical().cpu()), 1e-12);
    sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
}

TEST(ShValueStorageTest, Q16DeletedMaskSceneMergeAndPlyExport) {
    sh_value::set_sh_value_quant_enabled_for_testing(true);
    auto splat = make_random_sh3(64);
    ASSERT_TRUE(sh_value::apply_shN_value_quant(splat));
    ASSERT_TRUE(splat.shN_value_quantized());
    ASSERT_EQ(splat.shN_raw().dtype(), DataType::Float16);

    std::vector<bool> deleted(64, false);
    deleted[1] = true;
    deleted[17] = true;
    deleted[63] = true;
    splat.deleted() = Tensor::from_vector(deleted, {deleted.size()}, Device::CPU).to(Device::CUDA);
    ASSERT_TRUE(splat.has_deleted_mask());

    auto merged = Scene::mergeSplatsWithTransforms(
        {{&splat, glm::mat4{1.0f}}}, Scene::MergeStorageMode::Clone);
    ASSERT_NE(merged, nullptr);
    ASSERT_EQ(merged->size(), 61);

    const auto output_path =
        std::filesystem::temp_directory_path() / "lfs_q16_deleted_merge_export.ply";
    std::filesystem::remove(output_path);
    const auto save_result = lfs::io::save_ply(
        *merged, {.output_path = output_path, .binary = true, .async = false});
    ASSERT_TRUE(save_result.has_value()) << save_result.error().message;
    ASSERT_TRUE(std::filesystem::exists(output_path));
    EXPECT_GT(std::filesystem::file_size(output_path), 0u);
    std::filesystem::remove(output_path);

    sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
}

TEST(ShValueStorageTest, DensifyExpandCommitPreservesValues) {
    sh_value::set_sh_value_quant_enabled_for_testing(true);
    auto splat = make_random_sh3(kN);
    ASSERT_TRUE(sh_value::apply_shN_value_quant(splat));
    const auto ref = splat.shN_canonical().cpu().contiguous();

    // densify window: expand → (float-native ops would go here) → commit
    ASSERT_TRUE(sh_value::ensure_shN_fp32_for_mutation(splat));
    ASSERT_EQ(splat.shN().dtype(), DataType::Float32);
    ASSERT_TRUE(sh_value::commit_shN_after_mutation(splat));
    ASSERT_TRUE(splat.shN_value_quantized());

    const double mse = mse_tensors(ref, splat.shN_canonical().cpu());
    EXPECT_LT(mse, 1e-6);
    EXPECT_GT(psnr_from_mse(mse), 55.0);

    sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
}

TEST(ShValueStorageTest, ScopeExitCommitContainsAllocatorFailure) {
    sh_value::set_sh_value_quant_enabled_for_testing(true);
    auto splat = make_random_sh3(16);
    int allocation_attempts = 0;
    splat.set_tensor_allocator(
        [&](TensorShape, size_t, DataType, std::string_view) -> Tensor {
            ++allocation_attempts;
            throw std::runtime_error("injected SH commit allocation failure");
        });

    {
        LiveModelMutationGuard mutation_scope("ScopeExitCommitContainsAllocatorFailure");
        sh_value::ShNCommitGuard commit_guard(
            splat, /*expanded=*/true, "ScopeExitCommitContainsAllocatorFailure");
    }

    EXPECT_EQ(allocation_attempts, 1);
    EXPECT_EQ(splat.shN().dtype(), DataType::Float32);
    EXPECT_FALSE(splat.shN_value_quantized());
    sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
}

TEST(ShValueStorageTest, KernelEncodeDecodeMatchesHost) {
    sh_value::set_sh_value_quant_enabled_for_testing(true);
    constexpr size_t n = 64;
    constexpr uint32_t rest = 15;
    const auto n_cells = sh_value::n_value_cells_per_prim(rest);
    const auto n_u16 = sh_value::sh_value_u16_count(n, rest);
    const auto n_bounds = sh_value::n_bounds_for_prims(n);
    const auto n_floats = sh_swizzled_float_count(n, rest);

    // Build float4-swizzled source with known pattern on active cells only.
    // Pad floats (48−45 per prim in the float4 layout) stay zero — encode/decode
    // only touch n_cells = coeffs_rest*3 pad-dropped cells.
    Tensor src = Tensor::zeros({n_floats}, Device::CUDA, DataType::Float32);
    {
        auto cpu = src.cpu();
        auto* p = cpu.ptr<float>();
        const auto slots = sh_float4_slots_for_rest(rest);
        for (size_t prim = 0; prim < n; ++prim) {
            for (std::uint32_t c = 0; c < n_cells; ++c) {
                const std::uint32_t slot = c / 4u;
                const std::uint32_t comp = c % 4u;
                if (slot >= slots)
                    break;
                const size_t f4_idx =
                    static_cast<size_t>(sh_swizzled_index(static_cast<std::uint32_t>(prim),
                                                          slot, rest)) *
                        4u +
                    comp;
                p[f4_idx] = static_cast<float>(static_cast<int>(c % 17) - 8) * 0.05f;
            }
        }
        src = cpu.to(Device::CUDA);
    }

    Tensor u16 = Tensor::zeros({n_u16}, Device::CUDA, DataType::Float16);
    Tensor bounds = Tensor::zeros({n_bounds * 2}, Device::CUDA, DataType::Float32);
    Tensor dst = Tensor::zeros({n_floats}, Device::CUDA, DataType::Float32);

    sh_value::encode_shN_float4_to_u16(
        src.ptr<float>(),
        reinterpret_cast<std::uint16_t*>(u16.data_ptr()),
        bounds.ptr<float>(),
        n, rest, nullptr);
    sh_value::decode_shN_u16_to_float4(
        reinterpret_cast<const std::uint16_t*>(u16.data_ptr()),
        bounds.ptr<float>(),
        dst.ptr<float>(),
        n, rest, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const double mse = mse_tensors(src, dst);
    EXPECT_LT(mse, 1e-6) << "kernel RT MSE=" << mse;
    EXPECT_GT(psnr_from_mse(mse), 55.0);

    sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
}

TEST(ShValueStorageTest, LedgerBpsUnder307WithJoint) {
    sh_value::set_sh_value_quant_enabled_for_testing(true);

    // Large-N asymptotic: use N=1024 so bounds amortize.
    constexpr size_t n = 1024;
    auto splat = make_random_sh3(n);
    splat._densification_info = Tensor::zeros({size_t{2}, n}, Device::CUDA, DataType::Float32);
    ASSERT_TRUE(sh_value::apply_shN_value_quant(splat));

    AdamOptimizer optimizer(splat, AdamConfig{});
    optimizer.allocate_gradients();
    const auto ledger = compute_training_state_ledger(splat, &optimizer);

    EXPECT_LE(ledger.bytes_per_splat, 307.0) << "B/splat=" << ledger.bytes_per_splat;
    // params ~146, optim ~152, densify 8 → ~306
    EXPECT_GT(ledger.bytes_per_splat, 290.0);

    sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
}

// Grow N across a 256-row block boundary, re-encode, then run FastGS forward.
TEST(ShValueStorageTest, PostDensifyReencodeThenFastGSForward) {
    sh_value::set_sh_value_quant_enabled_for_testing(true);

    constexpr size_t kCap = 2048;
    constexpr size_t kN0 = 250;
    constexpr size_t kAppend = 40; // → 290 crosses 256 bounds block

    auto splat = make_random_sh3(kN0);
    splat.means().reserve(kCap);
    splat.sh0().reserve(kCap);
    splat.scaling_raw().reserve(kCap);
    splat.rotation_raw().reserve(kCap);
    splat.opacity_raw().reserve(kCap);
    {
        const auto rest = static_cast<uint32_t>(splat.max_sh_coeffs_rest());
        const auto cap_f = sh_swizzled_float_count(kCap, rest);
        if (splat.shN().capacity() < cap_f) {
            auto grown = Tensor::zeros_direct(splat.shN().shape(), cap_f, Device::CUDA);
            if (splat.shN().numel() > 0) {
                cudaMemcpy(grown.ptr<float>(), splat.shN().ptr<float>(),
                           splat.shN().numel() * sizeof(float), cudaMemcpyDeviceToDevice);
            }
            grown.set_name("splat.shN");
            splat.shN() = std::move(grown);
        }
    }

    ASSERT_TRUE(sh_value::apply_shN_value_quant(splat));
    {
        const auto rest = static_cast<uint32_t>(splat.max_sh_coeffs_rest());
        EXPECT_GE(splat.shN().capacity(), sh_value::sh_value_u16_count(kCap, rest));
        EXPECT_GE(splat.shN_value_bounds().capacity(),
                  sh_value::n_bounds_for_prims(kCap) * 2);
    }

    AdamConfig cfg{};
    cfg.initial_capacity = kCap;
    AdamOptimizer opt(splat, cfg);
    opt.allocate_gradients(kCap);

    std::vector<float> R_data = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    std::vector<float> T_data = {0, 0, 4};
    auto R = Tensor::from_blob(R_data.data(), {3, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto T = Tensor::from_blob(T_data.data(), {3}, Device::CPU, DataType::Float32).to(Device::CUDA);
    Camera camera(R, T, 100.f, 100.f, 32.f, 32.f, Tensor(), Tensor(), CameraModelType::PINHOLE,
                  "test", "", std::filesystem::path{}, 64, 64, 0);
    Tensor bg = Tensor::zeros({3}, Device::CUDA);

    {
        auto r = fast_rasterize_forward(camera, splat, bg, 0, 0, 0, 0, false);
        ASSERT_TRUE(r.has_value()) << lfs::format_for_developer(r.error());
    }

    ASSERT_TRUE(sh_value::ensure_shN_fp32_for_mutation(splat));
    const auto rest = static_cast<uint32_t>(splat.max_sh_coeffs_rest());
    const size_t n1 = kN0 + kAppend;
    {
        auto append_means = Tensor::zeros({kAppend, size_t{3}}, Device::CUDA);
        {
            auto cpu = append_means.cpu();
            auto* p = cpu.ptr<float>();
            for (size_t i = 0; i < kAppend; ++i) {
                p[i * 3 + 0] = static_cast<float>(i) * 0.05f - 0.5f;
            }
            append_means = cpu.to(Device::CUDA);
        }
        opt.add_new_params(ParamType::Means, append_means, true);
        opt.add_new_params(ParamType::Sh0,
                           Tensor::full({kAppend, size_t{1}, size_t{3}}, 0.25f, Device::CUDA), true);
        opt.add_new_params(ParamType::Scaling,
                           Tensor::full({kAppend, size_t{3}}, -2.0f, Device::CUDA), true);
        std::vector<float> rot(kAppend * 4, 0.f);
        for (size_t i = 0; i < kAppend; ++i)
            rot[i * 4] = 1.f;
        opt.add_new_params(
            ParamType::Rotation,
            Tensor::from_blob(rot.data(), {kAppend, size_t{4}}, Device::CPU, DataType::Float32)
                .to(Device::CUDA),
            true);
        opt.add_new_params(ParamType::Opacity,
                           Tensor::full({kAppend, size_t{1}}, 2.0f, Device::CUDA), true);
    }
    ASSERT_EQ(static_cast<size_t>(splat.size()), n1);
    {
        const size_t needed = sh_swizzled_float_count(n1, rest);
        auto& shN = splat.shN();
        if (shN.numel() < needed) {
            if (shN.capacity() < needed) {
                auto grown = Tensor::zeros_direct(
                    shN.shape(), sh_swizzled_float_count(kCap, rest), Device::CUDA);
                if (shN.numel() > 0) {
                    cudaMemcpy(grown.ptr<float>(), shN.ptr<float>(),
                               shN.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
                }
                grown.set_name("splat.shN");
                shN = std::move(grown);
            }
            shN.append_zeros(needed - shN.numel());
        }
        opt.extend_state_for_new_params(ParamType::ShN, kAppend);
    }

    ASSERT_TRUE(sh_value::commit_shN_after_mutation(splat));
    ASSERT_TRUE(splat.shN_value_quantized());
    EXPECT_EQ(static_cast<size_t>(splat.shN().numel()), sh_value::sh_value_u16_count(n1, rest));
    EXPECT_GE(splat.shN().capacity(), sh_value::sh_value_u16_count(kCap, rest));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    {
        auto r = fast_rasterize_forward(camera, splat, bg, 0, 0, 0, 0, false);
        ASSERT_TRUE(r.has_value()) << lfs::format_for_developer(r.error());
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess)
            << "illegal address after post-densify re-encode";
        opt.zero_grad(100);
        auto grad_out = Tensor::ones_like(r->first.image).mul(0.01f);
        ASSERT_NO_THROW(fast_rasterize_backward(r->second, grad_out, splat, opt, {}, {},
                                                DensificationType::None, 100));
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        auto r2 = fast_rasterize_forward(camera, splat, bg, 0, 0, 0, 0, false);
        ASSERT_TRUE(r2.has_value()) << lfs::format_for_developer(r2.error());
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }

    sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
}

// GUI exportable q16 SH densify expand → append → re-encode, then
// FastGS forward/backward must not illegal-address. Headless pool q16 already
// has PostDensifyReencodeThenFastGSForward; this is the packed SoA path the
// viewport zero-copy gate missed (gate ran -i 800 without a full densify).
TEST(ShValueStorageTest, ExportableQ16DensifyThenFastGSForward) {
    sh_value::set_sh_value_quant_enabled_for_testing(true);

    constexpr size_t kN0 = 512;
    constexpr size_t kAppend = 300; // crosses several 256-bounds blocks
    constexpr size_t kCap = 2048;   // exportable committed capacity
    constexpr int kShDegree = 3;
    const auto rest = static_cast<uint32_t>(sh_rest_coefficients_for_degree(kShDegree));

    auto storage_result = SplatExportableStorage::create(kCap, kShDegree, /*device=*/0, kCap * 4);
    if (!storage_result) {
        GTEST_SKIP() << "exportable create failed: " << storage_result.error();
    }
    auto storage = std::make_shared<SplatExportableStorage>(std::move(*storage_result));
    auto allocator = storage->make_allocator();

    auto seed = make_random_sh3(kN0);
    Tensor means = allocator(TensorShape({kN0, 3}), kCap, DataType::Float32, "SplatData.means");
    Tensor scaling = allocator(TensorShape({kN0, 3}), kCap, DataType::Float32, "SplatData.scaling");
    Tensor rotation = allocator(TensorShape({kN0, 4}), kCap, DataType::Float32, "SplatData.rotation");
    Tensor opacity = allocator(TensorShape({kN0, 1}), kCap, DataType::Float32, "SplatData.opacity");
    Tensor sh0 = allocator(TensorShape({kN0, 1, 3}), kCap, DataType::Float32, "SplatData.sh0");
    means.copy_from(seed.means_raw());
    scaling.copy_from(seed.scaling_raw());
    rotation.copy_from(seed.rotation_raw());
    opacity.copy_from(seed.opacity_raw());
    sh0.copy_from(seed.sh0_raw());
    const size_t n_floats = sh_swizzled_float_count(kN0, rest);
    const size_t cap_floats = sh_swizzled_float_count(kCap, rest);
    Tensor shN_float = Tensor::zeros_direct(TensorShape({n_floats}), cap_floats, Device::CUDA);
    shN_float.copy_from(seed.shN_raw());
    SplatData model(kShDegree, std::move(means), std::move(sh0), std::move(shN_float),
                    std::move(scaling), std::move(rotation), std::move(opacity), 1.0f,
                    SplatData::ShNLayout::Swizzled);
    model.set_tensor_allocator(allocator);
    model.set_active_sh_degree(0);

    ASSERT_TRUE(sh_value::apply_shN_value_quant(model));
    ASSERT_TRUE(model.shN_value_quantized());
    EXPECT_EQ(model.shN().external_storage_kind(), "splat.exportable");
    EXPECT_EQ(model.shN_value_bounds().external_storage_kind(), "splat.exportable");

    AdamConfig cfg{};
    cfg.initial_capacity = kCap;
    AdamOptimizer opt(model, cfg);
    opt.allocate_gradients(kCap);

    std::vector<float> R_data = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    std::vector<float> T_data = {0, 0, 4};
    auto R = Tensor::from_blob(R_data.data(), {3, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto T = Tensor::from_blob(T_data.data(), {3}, Device::CPU, DataType::Float32).to(Device::CUDA);
    Camera camera(R, T, 100.f, 100.f, 32.f, 32.f, Tensor(), Tensor(), CameraModelType::PINHOLE,
                  "test", "", std::filesystem::path{}, 64, 64, 0);
    Tensor bg = Tensor::zeros({3}, Device::CUDA);

    {
        auto r = fast_rasterize_forward(camera, model, bg, 0, 0, 0, 0, false);
        ASSERT_TRUE(r.has_value()) << lfs::format_for_developer(r.error());
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }

    ASSERT_TRUE(sh_value::ensure_shN_fp32_for_mutation(model));
    ASSERT_EQ(model.shN().dtype(), DataType::Float32);
    EXPECT_FALSE(model.shN_value_quantized());
    EXPECT_FALSE(model.shN_value_bounds().is_valid());

    const size_t n1 = kN0 + kAppend;
    {
        auto append_means = Tensor::zeros({kAppend, size_t{3}}, Device::CUDA);
        opt.add_new_params(ParamType::Means, append_means, true);
        opt.add_new_params(ParamType::Sh0,
                           Tensor::full({kAppend, size_t{1}, size_t{3}}, 0.25f, Device::CUDA), true);
        opt.add_new_params(ParamType::Scaling,
                           Tensor::full({kAppend, size_t{3}}, -2.0f, Device::CUDA), true);
        std::vector<float> rot(kAppend * 4, 0.f);
        for (size_t i = 0; i < kAppend; ++i)
            rot[i * 4] = 1.f;
        opt.add_new_params(
            ParamType::Rotation,
            Tensor::from_blob(rot.data(), {kAppend, size_t{4}}, Device::CPU, DataType::Float32)
                .to(Device::CUDA),
            true);
        opt.add_new_params(ParamType::Opacity,
                           Tensor::full({kAppend, size_t{1}}, 2.0f, Device::CUDA), true);
    }
    ASSERT_EQ(static_cast<size_t>(model.size()), n1);
    {
        const size_t needed = sh_swizzled_float_count(n1, rest);
        auto& shN = model.shN();
        if (shN.numel() < needed) {
            if (shN.capacity() < needed) {
                auto grown = Tensor::zeros_direct(
                    shN.shape(), sh_swizzled_float_count(kCap, rest), Device::CUDA);
                if (shN.numel() > 0) {
                    cudaMemcpy(grown.ptr<float>(), shN.ptr<float>(),
                               shN.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
                }
                grown.set_name("splat.shN");
                shN = std::move(grown);
            }
            shN.append_zeros(needed - shN.numel());
        }
        opt.extend_state_for_new_params(ParamType::ShN, kAppend);
    }

    ASSERT_TRUE(sh_value::commit_shN_after_mutation(model));
    ASSERT_TRUE(model.shN_value_quantized());
    EXPECT_EQ(model.shN().external_storage_kind(), "splat.exportable");
    EXPECT_EQ(model.shN_value_bounds().external_storage_kind(), "splat.exportable");
    EXPECT_EQ(static_cast<size_t>(model.shN().numel()),
              sh_value_quant::sh_value_u16_count(n1, rest));
    EXPECT_GE(model.shN().capacity(), sh_value_quant::sh_value_u16_count(kCap, rest));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Exercise every active SH degree after the densification commit.
    for (int active = 0; active <= kShDegree; ++active) {
        model.set_active_sh_degree(active);
        ASSERT_TRUE(model.shN_value_quantized()) << "q16 must stay resident after densify commit";
        auto r = fast_rasterize_forward(camera, model, bg, 0, 0, 0, 0, false);
        ASSERT_TRUE(r.has_value()) << "active_sh=" << active << " "
                                   << lfs::format_for_developer(r.error());
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess)
            << "illegal address after exportable q16 densify, active_sh=" << active;
        opt.zero_grad(100);
        auto grad_out = Tensor::ones_like(r->first.image).mul(0.01f);
        ASSERT_NO_THROW(fast_rasterize_backward(r->second, grad_out, model, opt, {}, {},
                                                DensificationType::MRNF, 100));
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess)
            << "illegal address in backward after exportable q16 densify active_sh=" << active;
    }

    sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
}

// SH degree updates must remain collision-safe with densification and growth.
// Representation is declared state, and codec commit is the sole q16 writer.

namespace {
    std::vector<std::uint8_t> snapshot_bytes(const Tensor& t) {
        auto c = t.cpu().contiguous();
        const auto* p = static_cast<const std::uint8_t*>(c.data_ptr());
        const size_t nbytes = c.numel() * (c.dtype() == DataType::Float16 ? 2 : 4);
        return {p, p + nbytes};
    }
} // namespace

TEST(ShDegreeCollisionTest, Q16DegreeUpIsStorageNoOpAllDegrees) {
    sh_value::set_sh_value_quant_enabled_for_testing(true);
    auto splat = make_random_sh3(kN);
    ASSERT_TRUE(sh_value::apply_shN_value_quant(splat));
    splat.set_active_sh_degree(0);

    const auto codes_before = snapshot_bytes(splat.shN());
    const auto bounds_before = snapshot_bytes(splat.shN_value_bounds());

    for (int d = 0; d <= kShDegree; ++d) {
        splat.set_active_sh_degree(d);
        EXPECT_EQ(splat.get_active_sh_degree(), d);
    }
    EXPECT_EQ(snapshot_bytes(splat.shN()), codes_before)
        << "degree-up mutated q16 codes";
    EXPECT_EQ(snapshot_bytes(splat.shN_value_bounds()), bounds_before)
        << "degree-up mutated q16 bounds";
    sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
}

TEST(ShDegreeCollisionTest, DegreeUpInsideOpenMutationWindowBothOrders) {
    sh_value::set_sh_value_quant_enabled_for_testing(true);
    for (const bool increment_before_commit : {true, false}) {
        auto splat = make_random_sh3(kN);
        ASSERT_TRUE(sh_value::apply_shN_value_quant(splat));
        splat.set_active_sh_degree(1);
        const auto ref = splat.shN_canonical().cpu().contiguous();

        ASSERT_TRUE(sh_value::ensure_shN_fp32_for_mutation(splat));
        if (increment_before_commit) {
            splat.increment_sh_degree(); // inside the open float window
            ASSERT_TRUE(sh_value::commit_shN_after_mutation(splat));
        } else {
            ASSERT_TRUE(sh_value::commit_shN_after_mutation(splat));
            splat.increment_sh_degree(); // immediately after commit, same boundary
        }
        ASSERT_TRUE(splat.shN_value_quantized());
        const auto after = splat.shN_canonical().cpu().contiguous();
        const double mse = mse_tensors(ref, after);
        EXPECT_LT(mse, 1e-6) << "order=" << increment_before_commit << " MSE=" << mse;
    }
    sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
}

TEST(ShDegreeCollisionTest, DegreeUpWithGrownMeansCapacitySameBoundary) {
    // Densify grow raises means.capacity before/while codes grow. A degree-up on
    // the same boundary must either no-op (consistent q16) or fail loud — never
    // silently rewrite codes using float-topology sizing.
    sh_value::set_sh_value_quant_enabled_for_testing(true);
    constexpr size_t kCapGrow = kN * 2;
    auto splat = make_random_sh3(kN);
    ASSERT_TRUE(sh_value::apply_shN_value_quant(splat));

    // Simulate the densify float window with capacity growth mid-flight.
    ASSERT_TRUE(sh_value::ensure_shN_fp32_for_mutation(splat));
    splat.means().reserve(kCapGrow);
    splat.increment_sh_degree(); // degree-up lands mid-window with cap grown
    ASSERT_TRUE(sh_value::commit_shN_after_mutation(splat));
    ASSERT_TRUE(splat.shN_value_quantized());

    // Post-commit codes/bounds must be sized for the CURRENT n at max-degree
    // layout and survive further degree flips untouched.
    const auto codes = snapshot_bytes(splat.shN());
    splat.set_active_sh_degree(0);
    splat.set_active_sh_degree(kShDegree);
    EXPECT_EQ(snapshot_bytes(splat.shN()), codes);
    sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
}

TEST(ShDegreeCollisionTest, InconsistentQ16StorageFailsLoudNotSilentRepair) {
    sh_value::set_sh_value_quant_enabled_for_testing(true);
    auto splat = make_random_sh3(kN);
    ASSERT_TRUE(sh_value::apply_shN_value_quant(splat));
    // Corrupt the invariant: drop bounds while codes stay q16-shaped.
    splat.shN_value_bounds() = Tensor{};
    EXPECT_THROW(splat.set_active_sh_degree(1), std::runtime_error);
    sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
}

TEST(ShDegreeCollisionTest, MaxDegreeChangeOnQ16RelayoutsViaCanonical) {
    // A max-degree change on resident q16 runs the safe sequence internally:
    // decode -> fp32 relayout at the new topology -> leave unquantized for the
    // codec to requantize. Values of the kept coefficients survive exactly
    // (within codec tolerance); no byte-level guessing.
    sh_value::set_sh_value_quant_enabled_for_testing(true);
    auto splat = make_random_sh3(kN);
    ASSERT_TRUE(sh_value::apply_shN_value_quant(splat));
    const auto ref = splat.shN_canonical().cpu().contiguous(); // [N, 15, 3] deg3

    splat.set_max_sh_degree(kShDegree - 1); // 15 -> 8 rest coefficients
    EXPECT_FALSE(splat.shN_value_quantized());
    const auto down = splat.shN_canonical().cpu().contiguous();
    ASSERT_EQ(down.shape()[1], 8u);
    const auto ref_kept = ref.slice(1, 0, 8).contiguous();
    EXPECT_LT(mse_tensors(ref_kept, down), 1e-6);

    // Requantization after the relayout works and roundtrips.
    ASSERT_TRUE(sh_value::apply_shN_value_quant(splat));
    ASSERT_TRUE(splat.shN_value_quantized());
    const auto requant = splat.shN_canonical().cpu().contiguous();
    EXPECT_LT(mse_tensors(down, requant), 1e-6);
    sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
}

// Force densification and degree growth at the same exportable q16 boundary,
// across SH degrees 0..3, with capacity growth mid-window. The model must leave q16
// resident after commit (no multi-iter float densify window) and survive FastGS.
TEST(ShDegreeCollisionTest, ExportableDegreeUpGrowSameBoundaryAllDegrees) {
    sh_value::set_sh_value_quant_enabled_for_testing(true);

    constexpr size_t kN0 = 512;
    constexpr size_t kAppend = 250;
    constexpr size_t kCap = 4096;
    const auto rest = static_cast<uint32_t>(sh_rest_coefficients_for_degree(kShDegree));

    auto storage_result = SplatExportableStorage::create(kCap, kShDegree, /*device=*/0, kCap * 2);
    if (!storage_result) {
        GTEST_SKIP() << "exportable create failed: " << storage_result.error();
    }
    auto storage = std::make_shared<SplatExportableStorage>(std::move(*storage_result));
    auto allocator = storage->make_allocator();

    auto seed = make_random_sh3(kN0, /*seed=*/0xC011);
    Tensor means = allocator(TensorShape({kN0, 3}), kCap, DataType::Float32, "SplatData.means");
    Tensor scaling = allocator(TensorShape({kN0, 3}), kCap, DataType::Float32, "SplatData.scaling");
    Tensor rotation = allocator(TensorShape({kN0, 4}), kCap, DataType::Float32, "SplatData.rotation");
    Tensor opacity = allocator(TensorShape({kN0, 1}), kCap, DataType::Float32, "SplatData.opacity");
    Tensor sh0 = allocator(TensorShape({kN0, 1, 3}), kCap, DataType::Float32, "SplatData.sh0");
    means.copy_from(seed.means_raw());
    scaling.copy_from(seed.scaling_raw());
    rotation.copy_from(seed.rotation_raw());
    opacity.copy_from(seed.opacity_raw());
    sh0.copy_from(seed.sh0_raw());
    const size_t n_floats = sh_swizzled_float_count(kN0, rest);
    const size_t cap_floats = sh_swizzled_float_count(kCap, rest);
    Tensor shN_float = Tensor::zeros_direct(TensorShape({n_floats}), cap_floats, Device::CUDA);
    shN_float.copy_from(seed.shN_raw());
    SplatData model(kShDegree, std::move(means), std::move(sh0), std::move(shN_float),
                    std::move(scaling), std::move(rotation), std::move(opacity), 1.0f,
                    SplatData::ShNLayout::Swizzled);
    model.set_tensor_allocator(allocator);
    model.set_active_sh_degree(0);

    ASSERT_TRUE(sh_value::apply_shN_value_quant(model));
    ASSERT_TRUE(model.shN_value_quantized());

    AdamConfig cfg{};
    cfg.initial_capacity = kCap;
    AdamOptimizer opt(model, cfg);
    opt.allocate_gradients(kCap);

    std::vector<float> R_data = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    std::vector<float> T_data = {0, 0, 4};
    auto R = Tensor::from_blob(R_data.data(), {3, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto T = Tensor::from_blob(T_data.data(), {3}, Device::CPU, DataType::Float32).to(Device::CUDA);
    Camera camera(R, T, 100.f, 100.f, 32.f, 32.f, Tensor(), Tensor(), CameraModelType::PINHOLE,
                  "coll", "", std::filesystem::path{}, 64, 64, 0);
    Tensor bg = Tensor::zeros({3}, Device::CUDA);

    // Two densify+degree-up cycles (simulates degree schedule colliding with refine).
    for (int cycle = 0; cycle < 2; ++cycle) {
        const size_t n_before = static_cast<size_t>(model.size());
        ASSERT_TRUE(sh_value::ensure_shN_fp32_for_mutation(model));
        ASSERT_EQ(model.shN().dtype(), DataType::Float32);
        EXPECT_FALSE(model.shN_value_quantized());

        // Capacity grow mid float window (exportable means grow-in-place).
        model.means().reserve(std::min(kCap, n_before + kAppend * 2));

        {
            auto append_means = Tensor::zeros({kAppend, size_t{3}}, Device::CUDA);
            opt.add_new_params(ParamType::Means, append_means, true);
            opt.add_new_params(ParamType::Sh0,
                               Tensor::full({kAppend, size_t{1}, size_t{3}}, 0.1f, Device::CUDA), true);
            opt.add_new_params(ParamType::Scaling,
                               Tensor::full({kAppend, size_t{3}}, -2.0f, Device::CUDA), true);
            std::vector<float> rot(kAppend * 4, 0.f);
            for (size_t i = 0; i < kAppend; ++i)
                rot[i * 4] = 1.f;
            opt.add_new_params(
                ParamType::Rotation,
                Tensor::from_blob(rot.data(), {kAppend, size_t{4}}, Device::CPU, DataType::Float32)
                    .to(Device::CUDA),
                true);
            opt.add_new_params(ParamType::Opacity,
                               Tensor::full({kAppend, size_t{1}}, 2.0f, Device::CUDA), true);
        }
        const size_t n_after = static_cast<size_t>(model.size());
        {
            const size_t needed = sh_swizzled_float_count(n_after, rest);
            auto& shN = model.shN();
            if (shN.numel() < needed) {
                if (shN.capacity() < needed) {
                    auto grown = Tensor::zeros_direct(
                        shN.shape(), sh_swizzled_float_count(kCap, rest), Device::CUDA);
                    if (shN.numel() > 0) {
                        cudaMemcpy(grown.ptr<float>(), shN.ptr<float>(),
                                   shN.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
                    }
                    grown.set_name("splat.shN");
                    shN = std::move(grown);
                }
                shN.append_zeros(needed - shN.numel());
            }
            opt.extend_state_for_new_params(ParamType::ShN, kAppend);
        }

        // Degree-up mid-window (same iteration as grow) — must not corrupt storage.
        model.increment_sh_degree();

        ASSERT_TRUE(sh_value::commit_shN_after_mutation(model));
        ASSERT_TRUE(model.shN_value_quantized())
            << "q16 must be resident after densify commit (no lingering float window)";
        EXPECT_EQ(model.shN().external_storage_kind(), "splat.exportable");
        EXPECT_EQ(model.shN_value_bounds().external_storage_kind(), "splat.exportable");
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        // Ledger + storage: pad-dropped q16 residency after commit (not float).
        const auto ledger = compute_training_state_ledger(model, &opt);
        EXPECT_EQ(ledger.live_splats, n_after);
        EXPECT_GT(ledger.params_bytes, 0u);
        const size_t q16_cells = sh_value_quant::sh_value_u16_count(n_after, rest);
        EXPECT_EQ(static_cast<size_t>(model.shN().numel()), q16_cells);
        EXPECT_EQ(model.shN().dtype(), DataType::Float16);

        for (int active = 0; active <= kShDegree; ++active) {
            model.set_active_sh_degree(active);
            ASSERT_TRUE(model.shN_value_quantized()) << "active=" << active << " cycle=" << cycle;
            auto r = fast_rasterize_forward(camera, model, bg, 0, 0, 0, 0, false);
            ASSERT_TRUE(r.has_value()) << "cycle=" << cycle << " active=" << active << " "
                                       << lfs::format_for_developer(r.error());
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess)
                << "illegal address cycle=" << cycle << " active=" << active;
        }
    }

    sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
}

// Cadence-misalign proxy: repeated densify windows with degree flips at every
// boundary (interval-style). Storage remains q16 after each commit.
TEST(ShDegreeCollisionTest, MisalignedCadenceDensifyDegreeSweep) {
    sh_value::set_sh_value_quant_enabled_for_testing(true);
    auto splat = make_random_sh3(kN, /*seed=*/0xCAD3);
    ASSERT_TRUE(sh_value::apply_shN_value_quant(splat));
    splat.set_active_sh_degree(0);

    // Simulated step schedule: refine every 100, degree every 250/333/1000-style
    // offsets — force degree-up on both refining and non-refining boundaries.
    const int degree_intervals[] = {1000, 250, 333, 100};
    for (int interval : degree_intervals) {
        for (int iter = 1; iter <= 2000; iter += 50) {
            const bool refining = (iter % 100 == 0) && iter > 500 && iter < 15000;
            if (refining) {
                ASSERT_TRUE(sh_value::ensure_shN_fp32_for_mutation(splat));
                // grow capacity mid-window on a subset of refining steps
                if (iter % 200 == 0) {
                    splat.means().reserve(static_cast<size_t>(splat.size()) + 64);
                }
                if (iter % interval == 0) {
                    splat.increment_sh_degree();
                }
                ASSERT_TRUE(sh_value::commit_shN_after_mutation(splat));
                ASSERT_TRUE(splat.shN_value_quantized())
                    << "interval=" << interval << " iter=" << iter;
            } else if (iter % interval == 0) {
                // Non-refining degree bump: pure flag flip on resident q16.
                const auto codes = snapshot_bytes(splat.shN());
                splat.increment_sh_degree();
                EXPECT_EQ(snapshot_bytes(splat.shN()), codes)
                    << "degree bump mutated codes interval=" << interval
                    << " iter=" << iter;
                ASSERT_TRUE(splat.shN_value_quantized());
            }
        }
        // Reset active degree for next interval sweep without touching storage.
        splat.set_active_sh_degree(0);
        ASSERT_TRUE(splat.shN_value_quantized());
    }
    sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
}

// Crossing stop_refine must keep q16 resident on both sides of the refinement
// freeze.
TEST(ShDegreeCollisionTest, StopRefineCrossingAlwaysCommitQ16Throughout) {
    sh_value::set_sh_value_quant_enabled_for_testing(true);
    auto splat = make_random_sh3(kN, /*seed=*/0x57A8);
    ASSERT_TRUE(sh_value::apply_shN_value_quant(splat));
    splat.set_active_sh_degree(0);

    // Scaled-invariant schedule (DEFAULT ratios): start_refine=500, refine_every=100,
    // stop_refine=1500 stand-in for 15000 under steps_scaler=0.1 semantics.
    constexpr int kStartRefine = 500;
    constexpr int kRefineEvery = 100;
    constexpr int kStopRefine = 1500;
    constexpr int kShInterval = 1000;

    int densify_commits = 0;
    for (int iter = 1; iter <= kStopRefine + kRefineEvery; ++iter) {
        const bool refining =
            iter < kStopRefine && iter > kStartRefine && (iter % kRefineEvery == 0);
        if (refining) {
            ASSERT_TRUE(sh_value::ensure_shN_fp32_for_mutation(splat));
            EXPECT_FALSE(splat.shN_value_quantized()) << "iter=" << iter;
            if (iter % kShInterval == 0) {
                splat.increment_sh_degree();
            }
            // Always-commit (no multi-iter float window).
            ASSERT_TRUE(sh_value::commit_shN_after_mutation(splat));
            ASSERT_TRUE(splat.shN_value_quantized()) << "post-commit iter=" << iter;
            ++densify_commits;
        } else if (iter % kShInterval == 0) {
            splat.increment_sh_degree();
            ASSERT_TRUE(splat.shN_value_quantized()) << "degree-up iter=" << iter;
        }

        // Topology-freeze safety net (mirrors MRNF::post_backward stop_refine).
        if (iter == kStopRefine) {
            if (splat.shN().is_valid() && splat.shN().dtype() == DataType::Float32) {
                ASSERT_TRUE(sh_value::commit_shN_after_mutation(splat));
            }
            ASSERT_TRUE(splat.shN_value_quantized())
                << "q16 must be resident at stop_refine boundary";
        }
        if (iter > kStopRefine) {
            ASSERT_TRUE(splat.shN_value_quantized())
                << "q16 must remain after stop_refine iter=" << iter;
        }
    }
    EXPECT_GT(densify_commits, 0);
    // Cross stop_refine: no further densify windows leave float behind.
    EXPECT_TRUE(splat.shN_value_quantized());
    EXPECT_EQ(splat.shN().dtype(), DataType::Float16);
    sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
}
