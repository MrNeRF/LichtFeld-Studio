/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/cuda/sh_layout.cuh"
#include "core/scene.hpp"
#include "core/sh_value_quant.hpp"
#include "core/splat_data.hpp"
#include "core/splat_exportable_storage.hpp"
#include "core/tensor.hpp"
#include "io/exporter.hpp"
#include "io/formats/ply.hpp"
#include "io/loader.hpp"
#include "lfs/training/live_model_mutation_guard.hpp"
#include "lfs/training/sh_value_codec.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include "lfs/training/vram_ledger.hpp"
#include "training/optimizer/adam_optimizer.hpp"
#include "training/rasterization/fast_rasterizer.hpp"
#include "training/rasterization/fastgs/rasterization/include/rasterization_config.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/mat4x4.hpp>
#include <gtest/gtest.h>
#include <numeric>
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
        auto means = Tensor::zeros({n, size_t{3}}, Device::GPU, DataType::Float32);
        auto sh0 = Tensor::zeros({n, size_t{1}, size_t{3}}, Device::GPU, DataType::Float32);
        auto shN_can = Tensor::zeros({n, size_t{15}, size_t{3}}, Device::GPU, DataType::Float32);
        auto scaling = Tensor::zeros({n, size_t{3}}, Device::GPU, DataType::Float32);
        auto rotation = Tensor::zeros({n, size_t{4}}, Device::GPU, DataType::Float32);
        auto opacity = Tensor::zeros({n, size_t{1}}, Device::GPU, DataType::Float32);

        {
            std::mt19937 rng(seed);
            std::normal_distribution<float> nd(0.0f, 0.15f);
            auto cpu = shN_can.cpu();
            auto* p = cpu.ptr<float>();
            for (size_t i = 0; i < n * 15 * 3; ++i)
                p[i] = nd(rng);
            shN_can = cpu.to(Device::GPU);

            auto rcpu = rotation.cpu();
            auto* r = rcpu.ptr<float>();
            for (size_t i = 0; i < n; ++i)
                r[i * 4] = 1.0f;
            rotation = rcpu.to(Device::GPU);
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

    void expect_tensors_bitwise_equal(const Tensor& a, const Tensor& b, const char* name) {
        auto ac = a.cpu().contiguous();
        auto bc = b.cpu().contiguous();
        ASSERT_EQ(ac.dtype(), bc.dtype()) << name;
        ASSERT_EQ(ac.ndim(), bc.ndim()) << name;
        ASSERT_EQ(ac.numel(), bc.numel()) << name;
        ASSERT_EQ(ac.shape().str(), bc.shape().str()) << name;
        ASSERT_EQ(std::memcmp(ac.data_ptr(), bc.data_ptr(), ac.bytes()), 0) << name;
    }

    [[nodiscard]] float max_abs_diff(const Tensor& a, const Tensor& b) {
        auto ac = a.cpu().contiguous();
        auto bc = b.cpu().contiguous();
        EXPECT_EQ(ac.numel(), bc.numel());
        EXPECT_EQ(ac.dtype(), DataType::Float32);
        EXPECT_EQ(bc.dtype(), DataType::Float32);
        const auto* pa = ac.ptr<float>();
        const auto* pb = bc.ptr<float>();
        float m = 0.0f;
        for (size_t i = 0; i < ac.numel(); ++i) {
            const float d = std::fabs(pa[i] - pb[i]);
            if (d > m)
                m = d;
        }
        return m;
    }

    void fill_float_noise(Tensor& t, const uint32_t seed) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        auto cpu = t.cpu();
        auto* p = cpu.ptr<float>();
        for (size_t i = 0; i < cpu.numel(); ++i)
            p[i] = nd(rng);
        t = cpu.to(Device::GPU);
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
            Tensor backing = Tensor::zeros_direct(shape, capacity, Device::GPU, dtype);
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

TEST(ShValueStorageTest, Q16CanonicalCpuMatchesDevicePath) {
    sh_value::set_sh_value_quant_enabled_for_testing(true);
    auto splat = make_random_sh3(kN);
    ASSERT_TRUE(sh_value::apply_shN_value_quant(splat));
    ASSERT_TRUE(splat.shN_value_quantized());

    const auto cpu = splat.shN_canonical_cpu();
    const auto device_cpu = splat.shN_canonical().cpu();
    EXPECT_EQ(cpu.device(), Device::CPU);
    EXPECT_EQ(cpu.dtype(), DataType::Float32);
    expect_tensors_bitwise_equal(cpu, device_cpu, "q16 canonical cpu vs device");

    sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
}

TEST(ShValueStorageTest, IeeeF16CanonicalCpuMatchesDevicePath) {
    auto splat = make_random_sh3(64);
    splat.shN() = splat.shN().to(DataType::Float16);
    ASSERT_TRUE(splat.shN_ieee_f16());
    ASSERT_FALSE(splat.shN_value_quantized());

    const auto cpu = splat.shN_canonical_cpu();
    const auto device_cpu = splat.shN_canonical().cpu();
    EXPECT_EQ(cpu.device(), Device::CPU);
    EXPECT_EQ(cpu.dtype(), DataType::Float32);
    expect_tensors_bitwise_equal(cpu, device_cpu, "ieee-f16 canonical cpu vs device");
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
    splat.deleted() = Tensor::from_vector(deleted, {deleted.size()}, Device::CPU).to(Device::GPU);
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

TEST(ShValueStorageTest, Q16BorrowSingleIdentityMergePreservesQuantAndPly) {
    sh_value::set_sh_value_quant_enabled_for_testing(true);
    auto splat = make_random_sh3(64);
    ASSERT_TRUE(sh_value::apply_shN_value_quant(splat));
    ASSERT_TRUE(splat.shN_value_quantized());
    ASSERT_EQ(splat.shN_raw().dtype(), DataType::Float16);

    auto merged = Scene::mergeSplatsWithTransforms(
        {{&splat, glm::mat4{1.0f}}}, Scene::MergeStorageMode::BorrowSingleIdentity);
    ASSERT_NE(merged, nullptr);
    EXPECT_EQ(merged->shN_raw().dtype(), DataType::Float16);
    EXPECT_TRUE(merged->shN_value_quantized());
    EXPECT_TRUE(merged->shN_value_bounds().is_valid());
    EXPECT_GT(merged->shN_value_bounds().numel(), 0u);
    expect_tensors_bitwise_equal(splat.shN_canonical_cpu(), merged->shN_canonical_cpu(),
                                 "borrow ephemeral shN");

    const auto source_path =
        std::filesystem::temp_directory_path() / "lfs_q16_borrow_source.ply";
    const auto merged_path =
        std::filesystem::temp_directory_path() / "lfs_q16_borrow_merged.ply";
    std::filesystem::remove(source_path);
    std::filesystem::remove(merged_path);

    const auto save_source = lfs::io::save_ply(
        splat, {.output_path = source_path, .binary = true, .async = false});
    ASSERT_TRUE(save_source.has_value()) << save_source.error().message;
    const auto save_merged = lfs::io::save_ply(
        *merged, {.output_path = merged_path, .binary = true, .async = false});
    ASSERT_TRUE(save_merged.has_value()) << save_merged.error().message;

    auto loaded_source = lfs::io::load_ply(source_path);
    auto loaded_merged = lfs::io::load_ply(merged_path);
    ASSERT_TRUE(loaded_source.has_value()) << lfs::format_for_developer(loaded_source.error());
    ASSERT_TRUE(loaded_merged.has_value()) << lfs::format_for_developer(loaded_merged.error());

    const SplatData& src_pc = loaded_source->value;
    const SplatData& merged_pc = loaded_merged->value;
    expect_tensors_bitwise_equal(src_pc.means_raw(), merged_pc.means_raw(), "ply means");
    expect_tensors_bitwise_equal(src_pc.sh0_raw(), merged_pc.sh0_raw(), "ply sh0");
    expect_tensors_bitwise_equal(src_pc.shN_canonical_cpu(), merged_pc.shN_canonical_cpu(),
                                 "ply shN");
    expect_tensors_bitwise_equal(src_pc.opacity_raw(), merged_pc.opacity_raw(), "ply opacity");
    expect_tensors_bitwise_equal(src_pc.scaling_raw(), merged_pc.scaling_raw(), "ply scaling");
    expect_tensors_bitwise_equal(src_pc.rotation_raw(), merged_pc.rotation_raw(), "ply rotation");

    std::filesystem::remove(source_path);
    std::filesystem::remove(merged_path);
    sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
}

TEST(ShValueStorageTest, Q16MultiSourceIdentityMergeDecodesFloat) {
    sh_value::set_sh_value_quant_enabled_for_testing(true);
    auto a = make_random_sh3(64, /*seed=*/0xA101);
    auto b = make_random_sh3(64, /*seed=*/0xB202);
    fill_float_noise(a.means_raw(), 0xA103);
    fill_float_noise(a.sh0_raw(), 0xA104);
    fill_float_noise(a.opacity_raw(), 0xA105);
    fill_float_noise(b.means_raw(), 0xB203);
    fill_float_noise(b.sh0_raw(), 0xB204);
    fill_float_noise(b.opacity_raw(), 0xB205);

    ASSERT_TRUE(sh_value::apply_shN_value_quant(a));
    ASSERT_TRUE(sh_value::apply_shN_value_quant(b));
    ASSERT_TRUE(a.shN_value_quantized());
    ASSERT_TRUE(b.shN_value_quantized());

    const auto a_canon = a.shN_canonical_cpu();
    const auto b_canon = b.shN_canonical_cpu();
    const auto a_means = a.means_raw().clone();
    const auto a_sh0 = a.sh0_raw().clone();
    const auto a_opacity = a.opacity_raw().clone();
    const auto b_means = b.means_raw().clone();
    const auto b_sh0 = b.sh0_raw().clone();
    const auto b_opacity = b.opacity_raw().clone();

    auto merged = Scene::mergeSplatsWithTransforms(
        {{&a, glm::mat4{1.0f}}, {&b, glm::mat4{1.0f}}}, Scene::MergeStorageMode::Clone);
    ASSERT_NE(merged, nullptr);
    ASSERT_EQ(merged->size(), 128);
    EXPECT_EQ(merged->shN_raw().dtype(), DataType::Float32);
    EXPECT_FALSE(merged->shN_value_quantized());

    const auto merged_canon = merged->shN_canonical_cpu();
    expect_tensors_bitwise_equal(merged_canon.slice(0, 0, 64).contiguous(), a_canon,
                                 "identity merge a shN");
    expect_tensors_bitwise_equal(merged_canon.slice(0, 64, 128).contiguous(), b_canon,
                                 "identity merge b shN");
    expect_tensors_bitwise_equal(merged->means_raw().slice(0, 0, 64).contiguous(), a_means,
                                 "identity merge a means");
    expect_tensors_bitwise_equal(merged->means_raw().slice(0, 64, 128).contiguous(), b_means,
                                 "identity merge b means");
    expect_tensors_bitwise_equal(merged->sh0_raw().slice(0, 0, 64).contiguous(), a_sh0,
                                 "identity merge a sh0");
    expect_tensors_bitwise_equal(merged->sh0_raw().slice(0, 64, 128).contiguous(), b_sh0,
                                 "identity merge b sh0");
    expect_tensors_bitwise_equal(merged->opacity_raw().slice(0, 0, 64).contiguous(), a_opacity,
                                 "identity merge a opacity");
    expect_tensors_bitwise_equal(merged->opacity_raw().slice(0, 64, 128).contiguous(), b_opacity,
                                 "identity merge b opacity");

    sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
}

TEST(ShValueStorageTest, Q16MultiSourceTranslationMergeDecodesFloat) {
    sh_value::set_sh_value_quant_enabled_for_testing(true);
    auto a = make_random_sh3(64, /*seed=*/0xA301);
    auto b = make_random_sh3(64, /*seed=*/0xB302);
    fill_float_noise(b.means_raw(), 0xB303);

    ASSERT_TRUE(sh_value::apply_shN_value_quant(a));
    ASSERT_TRUE(sh_value::apply_shN_value_quant(b));
    ASSERT_TRUE(a.shN_value_quantized());
    ASSERT_TRUE(b.shN_value_quantized());

    const auto b_canon = b.shN_canonical_cpu();
    auto expected_b_means = b.means_raw().cpu().contiguous();
    {
        auto* p = expected_b_means.ptr<float>();
        for (size_t i = 0; i < 64; ++i)
            p[i * 3 + 0] += 1.0f;
    }

    const glm::mat4 translation = glm::translate(glm::mat4{1.0f}, glm::vec3{1.0f, 0.0f, 0.0f});
    auto merged = Scene::mergeSplatsWithTransforms(
        {{&a, glm::mat4{1.0f}}, {&b, translation}}, Scene::MergeStorageMode::Clone);
    ASSERT_NE(merged, nullptr);
    ASSERT_EQ(merged->size(), 128);
    EXPECT_EQ(merged->shN_raw().dtype(), DataType::Float32);
    EXPECT_FALSE(merged->shN_value_quantized());

    const auto merged_canon = merged->shN_canonical_cpu();
    expect_tensors_bitwise_equal(merged_canon.slice(0, 64, 128).contiguous(), b_canon,
                                 "translation merge b shN");
    EXPECT_LT(max_abs_diff(merged->means_raw().slice(0, 64, 128).contiguous(), expected_b_means),
              1e-5f);

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
    Tensor src = Tensor::zeros({n_floats}, Device::GPU, DataType::Float32);
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
        src = cpu.to(Device::GPU);
    }

    Tensor u16 = Tensor::zeros({n_u16}, Device::GPU, DataType::Float16);
    Tensor bounds = Tensor::zeros({n_bounds * 2}, Device::GPU, DataType::Float32);
    Tensor dst = Tensor::zeros({n_floats}, Device::GPU, DataType::Float32);

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
    splat._densification_info = Tensor::zeros({size_t{2}, n}, Device::GPU, DataType::Float32);
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
            auto grown = Tensor::zeros_direct(splat.shN().shape(), cap_f, Device::GPU);
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
    auto R = Tensor::from_blob(R_data.data(), {3, 3}, Device::CPU, DataType::Float32).to(Device::GPU);
    auto T = Tensor::from_blob(T_data.data(), {3}, Device::CPU, DataType::Float32).to(Device::GPU);
    Camera camera(R, T, 100.f, 100.f, 32.f, 32.f, Tensor(), Tensor(), CameraModelType::PINHOLE,
                  "test", "", std::filesystem::path{}, 64, 64, 0);
    Tensor bg = Tensor::zeros({3}, Device::GPU);

    {
        auto r = fast_rasterize_forward(camera, splat, bg, 0, 0, 0, 0, false);
        ASSERT_TRUE(r.has_value()) << lfs::format_for_developer(r.error());
    }

    ASSERT_TRUE(sh_value::ensure_shN_fp32_for_mutation(splat));
    const auto rest = static_cast<uint32_t>(splat.max_sh_coeffs_rest());
    const size_t n1 = kN0 + kAppend;
    {
        auto append_means = Tensor::zeros({kAppend, size_t{3}}, Device::GPU);
        {
            auto cpu = append_means.cpu();
            auto* p = cpu.ptr<float>();
            for (size_t i = 0; i < kAppend; ++i) {
                p[i * 3 + 0] = static_cast<float>(i) * 0.05f - 0.5f;
            }
            append_means = cpu.to(Device::GPU);
        }
        opt.add_new_params(ParamType::Means, append_means, true);
        opt.add_new_params(ParamType::Sh0,
                           Tensor::full({kAppend, size_t{1}, size_t{3}}, 0.25f, Device::GPU), true);
        opt.add_new_params(ParamType::Scaling,
                           Tensor::full({kAppend, size_t{3}}, -2.0f, Device::GPU), true);
        std::vector<float> rot(kAppend * 4, 0.f);
        for (size_t i = 0; i < kAppend; ++i)
            rot[i * 4] = 1.f;
        opt.add_new_params(
            ParamType::Rotation,
            Tensor::from_blob(rot.data(), {kAppend, size_t{4}}, Device::CPU, DataType::Float32)
                .to(Device::GPU),
            true);
        opt.add_new_params(ParamType::Opacity,
                           Tensor::full({kAppend, size_t{1}}, 2.0f, Device::GPU), true);
    }
    ASSERT_EQ(static_cast<size_t>(splat.size()), n1);
    {
        const size_t needed = sh_swizzled_float_count(n1, rest);
        auto& shN = splat.shN();
        if (shN.numel() < needed) {
            if (shN.capacity() < needed) {
                auto grown = Tensor::zeros_direct(
                    shN.shape(), sh_swizzled_float_count(kCap, rest), Device::GPU);
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
    Tensor shN_float = Tensor::zeros_direct(TensorShape({n_floats}), cap_floats, Device::GPU);
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
    auto R = Tensor::from_blob(R_data.data(), {3, 3}, Device::CPU, DataType::Float32).to(Device::GPU);
    auto T = Tensor::from_blob(T_data.data(), {3}, Device::CPU, DataType::Float32).to(Device::GPU);
    Camera camera(R, T, 100.f, 100.f, 32.f, 32.f, Tensor(), Tensor(), CameraModelType::PINHOLE,
                  "test", "", std::filesystem::path{}, 64, 64, 0);
    Tensor bg = Tensor::zeros({3}, Device::GPU);

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
        auto append_means = Tensor::zeros({kAppend, size_t{3}}, Device::GPU);
        opt.add_new_params(ParamType::Means, append_means, true);
        opt.add_new_params(ParamType::Sh0,
                           Tensor::full({kAppend, size_t{1}, size_t{3}}, 0.25f, Device::GPU), true);
        opt.add_new_params(ParamType::Scaling,
                           Tensor::full({kAppend, size_t{3}}, -2.0f, Device::GPU), true);
        std::vector<float> rot(kAppend * 4, 0.f);
        for (size_t i = 0; i < kAppend; ++i)
            rot[i * 4] = 1.f;
        opt.add_new_params(
            ParamType::Rotation,
            Tensor::from_blob(rot.data(), {kAppend, size_t{4}}, Device::CPU, DataType::Float32)
                .to(Device::GPU),
            true);
        opt.add_new_params(ParamType::Opacity,
                           Tensor::full({kAppend, size_t{1}}, 2.0f, Device::GPU), true);
    }
    ASSERT_EQ(static_cast<size_t>(model.size()), n1);
    {
        const size_t needed = sh_swizzled_float_count(n1, rest);
        auto& shN = model.shN();
        if (shN.numel() < needed) {
            if (shN.capacity() < needed) {
                auto grown = Tensor::zeros_direct(
                    shN.shape(), sh_swizzled_float_count(kCap, rest), Device::GPU);
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
    Tensor shN_float = Tensor::zeros_direct(TensorShape({n_floats}), cap_floats, Device::GPU);
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
    auto R = Tensor::from_blob(R_data.data(), {3, 3}, Device::CPU, DataType::Float32).to(Device::GPU);
    auto T = Tensor::from_blob(T_data.data(), {3}, Device::CPU, DataType::Float32).to(Device::GPU);
    Camera camera(R, T, 100.f, 100.f, 32.f, 32.f, Tensor(), Tensor(), CameraModelType::PINHOLE,
                  "coll", "", std::filesystem::path{}, 64, 64, 0);
    Tensor bg = Tensor::zeros({3}, Device::GPU);

    // Two densify+degree-up cycles (simulates degree schedule colliding with refine).
    for (int cycle = 0; cycle < 2; ++cycle) {
        const size_t n_before = static_cast<size_t>(model.size());
        ASSERT_TRUE(sh_value::ensure_shN_fp32_for_mutation(model));
        ASSERT_EQ(model.shN().dtype(), DataType::Float32);
        EXPECT_FALSE(model.shN_value_quantized());

        // Capacity grow mid float window (exportable means grow-in-place).
        model.means().reserve(std::min(kCap, n_before + kAppend * 2));

        {
            auto append_means = Tensor::zeros({kAppend, size_t{3}}, Device::GPU);
            opt.add_new_params(ParamType::Means, append_means, true);
            opt.add_new_params(ParamType::Sh0,
                               Tensor::full({kAppend, size_t{1}, size_t{3}}, 0.1f, Device::GPU), true);
            opt.add_new_params(ParamType::Scaling,
                               Tensor::full({kAppend, size_t{3}}, -2.0f, Device::GPU), true);
            std::vector<float> rot(kAppend * 4, 0.f);
            for (size_t i = 0; i < kAppend; ++i)
                rot[i * 4] = 1.f;
            opt.add_new_params(
                ParamType::Rotation,
                Tensor::from_blob(rot.data(), {kAppend, size_t{4}}, Device::CPU, DataType::Float32)
                    .to(Device::GPU),
                true);
            opt.add_new_params(ParamType::Opacity,
                               Tensor::full({kAppend, size_t{1}}, 2.0f, Device::GPU), true);
        }
        const size_t n_after = static_cast<size_t>(model.size());
        {
            const size_t needed = sh_swizzled_float_count(n_after, rest);
            auto& shN = model.shN();
            if (shN.numel() < needed) {
                if (shN.capacity() < needed) {
                    auto grown = Tensor::zeros_direct(
                        shN.shape(), sh_swizzled_float_count(kCap, rest), Device::GPU);
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

namespace {

    std::vector<std::uint16_t> copy_u16_device(const Tensor& t) {
        EXPECT_TRUE(t.is_valid());
        std::vector<std::uint16_t> out(static_cast<size_t>(t.numel()));
        if (out.empty()) {
            return out;
        }
        EXPECT_EQ(cudaMemcpy(out.data(), t.data_ptr(), out.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        return out;
    }

    std::vector<float> copy_f32_device(const Tensor& t) {
        EXPECT_TRUE(t.is_valid());
        std::vector<float> out(static_cast<size_t>(t.numel()));
        if (out.empty()) {
            return out;
        }
        EXPECT_EQ(cudaMemcpy(out.data(), t.data_ptr(), out.size() * sizeof(float),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        return out;
    }

    void append_dummy_attr_rows(SplatData& splat, const size_t k) {
        auto append = [&](Tensor& t) {
            if (!t.is_valid() || t.ndim() == 0) {
                return;
            }
            auto dims = t.shape().dims();
            dims[0] = k;
            Tensor extra = Tensor::zeros(TensorShape(dims), t.device(), t.dtype());
            t = Tensor::cat({t, extra}, 0);
        };
        append(splat.means_raw());
        append(splat.sh0_raw());
        append(splat.scaling_raw());
        append(splat.rotation_raw());
        append(splat.opacity_raw());
    }

    void run_refine_like_shN_mutation(SplatData& splat,
                                      const Tensor& dup_idx,
                                      const Tensor& zero_idx,
                                      const bool full_expand) {
        LiveModelMutationGuard guard("refine_like_shN_mutation");
        const size_t old_n = static_cast<size_t>(splat.size());
        if (full_expand) {
            ASSERT_TRUE(sh_value::ensure_shN_fp32_for_mutation(splat));
            ASSERT_FALSE(splat.shN_value_quantized());
        } else {
            ASSERT_TRUE(splat.shN_value_quantized());
        }
        Tensor child;
        sh_value::gather_shN_to_canonical(splat, dup_idx, child, old_n);
        append_dummy_attr_rows(splat, dup_idx.numel());
        if (full_expand) {
            sh_value::append_canonical_to_shN(splat, child, old_n);
            sh_value::zero_shN_at_indices(splat, zero_idx);
            ASSERT_TRUE(sh_value::commit_shN_after_mutation(splat));
        } else {
            sh_value::ShNMutationBatch batch(splat);
            batch.append(child, old_n);
            batch.zero(zero_idx);
            batch.flush();
        }
        ASSERT_TRUE(splat.shN_value_quantized());
    }

} // namespace

TEST(ShValueStorageTest, ChunkedRefineMutationTouchedBlocksMatchOldUntouchedStayPristine) {
    // Old full-expand commit_shN_after_mutation re-encodes every block from already
    // quantized values, so untouched blocks are not idempotent. The chunked path
    // must leave those blocks bit-identical to the pre-mutation snapshot and match
    // the old path only on blocks that contain dest rows.
    sh_value::set_sh_value_quant_enabled_for_testing(true);
    constexpr size_t n = 70000; // not a multiple of 256 or 32
    auto splat = make_random_sh3(n, /*seed=*/20260830);
    ASSERT_TRUE(sh_value::apply_shN_value_quant(splat));
    ASSERT_TRUE(splat.shN_value_quantized());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto rest = static_cast<uint32_t>(splat.max_sh_coeffs_rest());
    const size_t n_dup = n / 20;  // 5%
    const size_t n_zero = n / 50; // 2%
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(20260830);
    std::shuffle(order.begin(), order.end(), rng);

    std::vector<int> dup_host(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(n_dup));
    std::vector<int> zero_host(order.begin() + static_cast<std::ptrdiff_t>(n_dup),
                               order.begin() + static_cast<std::ptrdiff_t>(n_dup + n_zero));
    auto dup_idx = Tensor::from_vector(dup_host, TensorShape({n_dup}), Device::GPU)
                       .to(DataType::Int64);
    auto zero_idx = Tensor::from_vector(zero_host, TensorShape({n_zero}), Device::GPU)
                        .to(DataType::Int64);

    const auto orig_codes = copy_u16_device(splat.shN());
    const auto orig_bounds = copy_f32_device(splat.shN_value_bounds());
    const size_t orig_n_blocks = sh_value_quant::n_bounds_for_prims(n);
    ASSERT_EQ(orig_bounds.size(), orig_n_blocks * 2);

    auto splat_old = splat.clone();
    auto splat_new = splat.clone();
    run_refine_like_shN_mutation(splat_old, dup_idx, zero_idx, /*full_expand=*/true);
    run_refine_like_shN_mutation(splat_new, dup_idx, zero_idx, /*full_expand=*/false);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_TRUE(splat_old.shN_value_quantized());
    ASSERT_TRUE(splat_new.shN_value_quantized());
    const auto codes_old = copy_u16_device(splat_old.shN());
    const auto codes_new = copy_u16_device(splat_new.shN());
    const auto bounds_old = copy_f32_device(splat_old.shN_value_bounds());
    const auto bounds_new = copy_f32_device(splat_new.shN_value_bounds());
    ASSERT_EQ(codes_old.size(), codes_new.size());
    ASSERT_EQ(bounds_old.size(), bounds_new.size());

    const size_t new_n = n + n_dup;
    ASSERT_EQ(static_cast<size_t>(splat_old.size()), new_n);
    ASSERT_EQ(static_cast<size_t>(splat_new.size()), new_n);
    ASSERT_EQ(codes_old.size(), sh_value_quant::sh_value_u16_count(new_n, rest));

    const size_t n_blocks = sh_value_quant::n_bounds_for_prims(new_n);
    ASSERT_EQ(bounds_old.size(), n_blocks * 2);
    std::vector<char> touched(n_blocks, 0);
    const size_t first_append_block = n / static_cast<size_t>(sh_value_quant::kBlockSize);
    const size_t last_block = (new_n - 1) / static_cast<size_t>(sh_value_quant::kBlockSize);
    for (size_t b = first_append_block; b <= last_block; ++b) {
        touched[b] = 1;
    }
    for (int idx : zero_host) {
        touched[static_cast<size_t>(idx) / static_cast<size_t>(sh_value_quant::kBlockSize)] = 1;
    }

    auto cells_match = [&](const std::vector<std::uint16_t>& a,
                           const std::vector<std::uint16_t>& b,
                           const size_t block,
                           const size_t n_prims) {
        const size_t bs = static_cast<size_t>(sh_value_quant::kBlockSize);
        const size_t p0 = block * bs;
        const size_t p1 = std::min(n_prims, p0 + bs);
        const size_t cell0 = sh_value_quant::sh_value_u16_count(p0, rest);
        const size_t cell1 = sh_value_quant::sh_value_u16_count(p1, rest);
        if (cell1 > a.size() || cell1 > b.size()) {
            ADD_FAILURE() << "cell range OOB block " << block << " cell1=" << cell1
                          << " a=" << a.size() << " b=" << b.size();
            return false;
        }
        return std::equal(a.begin() + static_cast<std::ptrdiff_t>(cell0),
                          a.begin() + static_cast<std::ptrdiff_t>(cell1),
                          b.begin() + static_cast<std::ptrdiff_t>(cell0));
    };

    size_t untouched = 0;
    size_t old_drifted = 0;
    for (size_t b = 0; b < orig_n_blocks; ++b) {
        if (touched[b]) {
            continue;
        }
        ++untouched;
        EXPECT_EQ(bounds_new[b * 2], orig_bounds[b * 2]) << "new bounds lo block " << b;
        EXPECT_EQ(bounds_new[b * 2 + 1], orig_bounds[b * 2 + 1]) << "new bounds hi block " << b;
        EXPECT_TRUE(cells_match(orig_codes, codes_new, b, n))
            << "new path mutated untouched block " << b;

        const bool bounds_drift = bounds_old[b * 2] != orig_bounds[b * 2] ||
                                  bounds_old[b * 2 + 1] != orig_bounds[b * 2 + 1];
        const bool cells_drift = !cells_match(orig_codes, codes_old, b, n);
        if (bounds_drift || cells_drift) {
            ++old_drifted;
        }
    }
    EXPECT_GT(untouched, 0u);
    EXPECT_GT(old_drifted, 0u)
        << "old full-expand path must drift on at least one untouched block "
           "(re-encode from decoded q16 is not idempotent; this is why new != old globally)";

    // Touched blocks contain appended / overlaid / zeroed dest rows. Chunked cells+bounds
    // must match the old full-expand path bit-identically. If this fails, the helpers
    // are wrong — do not weaken.
    const size_t bs = static_cast<size_t>(sh_value_quant::kBlockSize);
    size_t n_touched = 0;
    for (size_t b = 0; b < n_blocks; ++b) {
        if (!touched[b]) {
            continue;
        }
        ++n_touched;
        size_t zeros_in_block = 0;
        for (int idx : zero_host) {
            if (static_cast<size_t>(idx) / bs == b) {
                ++zeros_in_block;
            }
        }
        const bool append_overlap = b >= first_append_block && b <= last_block;
        EXPECT_EQ(bounds_new[b * 2], bounds_old[b * 2])
            << "touched bounds lo block " << b << " zeros=" << zeros_in_block
            << " append_overlap=" << append_overlap;
        EXPECT_EQ(bounds_new[b * 2 + 1], bounds_old[b * 2 + 1])
            << "touched bounds hi block " << b << " zeros=" << zeros_in_block
            << " append_overlap=" << append_overlap;
        EXPECT_TRUE(cells_match(codes_old, codes_new, b, new_n))
            << "touched block " << b
            << " must be bit-identical between chunked and full-expand paths"
            << " zeros=" << zeros_in_block << " append_overlap=" << append_overlap;
    }
    EXPECT_GT(n_touched, 0u);

    std::vector<int> mutated_host;
    mutated_host.reserve(n_dup + n_zero);
    for (size_t i = 0; i < n_dup; ++i) {
        mutated_host.push_back(static_cast<int>(n + i));
    }
    mutated_host.insert(mutated_host.end(), zero_host.begin(), zero_host.end());
    auto mutated_idx =
        Tensor::from_vector(mutated_host, TensorShape({mutated_host.size()}), Device::GPU)
            .to(DataType::Int64);
    Tensor decoded_old;
    Tensor decoded_new;
    {
        LiveModelMutationGuard guard("decode_mutated_shN_rows");
        sh_value::gather_shN_to_canonical(splat_old, mutated_idx, decoded_old, new_n);
        sh_value::gather_shN_to_canonical(splat_new, mutated_idx, decoded_new, new_n);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    expect_tensors_bitwise_equal(decoded_old, decoded_new, "decoded mutated shN rows");

    sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
}
