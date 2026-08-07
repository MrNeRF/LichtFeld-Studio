/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * WO-G3 / Phase 2.1 — SH value quant storage + densify bridge + export dequant.
 *
 * TDD: GPU encode/decode roundtrip, densify expand/commit, shN_canonical fp32,
 * and render-equivalence PSNR > 55 dB (synthetic SH evaluation proxy).
 */

#include "core/camera.hpp"
#include "core/cuda/sh_layout.cuh"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "lfs/training/joint_adam_codec.hpp"
#include "lfs/training/sh_value_codec.hpp"
#include "lfs/training/sh_value_quant_kernels.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include "lfs/training/vram_ledger.hpp"
#include "training/optimizer/adam_optimizer.hpp"
#include "training/rasterization/fast_rasterizer.hpp"
#include "training/rasterization/fastgs/rasterization/include/rasterization_config.h"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <random>
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
    joint_adam::set_joint_codec_enabled_for_testing(true);
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
    joint_adam::set_joint_codec_enabled_for_testing(std::nullopt);
}

// ISS-2.1 / WO-G6: G3 crash repro — grow N across 256-block boundary, re-encode, FastGS forward.
TEST(ShValueStorageTest, PostDensifyReencodeThenFastGSForward) {
    joint_adam::set_joint_codec_enabled_for_testing(true);
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
            << "illegal address after post-densify re-encode (ISS-2.1)";
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
    joint_adam::set_joint_codec_enabled_for_testing(std::nullopt);
}
