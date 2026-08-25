/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/error.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "io/formats/ply.hpp"
#include "io/loader.hpp"
#include "lfs/training/sh_value_storage.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace lfs::core;

namespace {

    void require_cuda() {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
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

    struct AllocCall {
        std::string name;
        DataType dtype;
    };

    SplatTensorAllocator recording_allocator(std::vector<AllocCall>& calls) {
        return [&calls](TensorShape shape,
                        const size_t capacity,
                        const DataType dtype,
                        const std::string_view name) {
            calls.push_back(AllocCall{std::string{name}, dtype});
            Tensor backing = Tensor::zeros_direct(shape, capacity, Device::CUDA, dtype);
            return retag_external(std::move(backing), "vulkan_external_buffer");
        };
    }

    [[nodiscard]] bool called_for_float_shN(const std::vector<AllocCall>& calls) {
        return std::any_of(calls.begin(), calls.end(), [](const AllocCall& call) {
            return call.name == "SplatData.shN" && call.dtype == DataType::Float32;
        });
    }

    class ScopedPlyQ16BandPrims {
    public:
        explicit ScopedPlyQ16BandPrims(const std::size_t band_prims) {
            lfs::io::set_ply_q16_band_prims_for_tests(band_prims);
        }

        ~ScopedPlyQ16BandPrims() {
            lfs::io::set_ply_q16_band_prims_for_tests(0);
        }

        ScopedPlyQ16BandPrims(const ScopedPlyQ16BandPrims&) = delete;
        ScopedPlyQ16BandPrims& operator=(const ScopedPlyQ16BandPrims&) = delete;
    };

    [[nodiscard]] std::filesystem::path gd_ply_path() {
        return std::filesystem::path(PROJECT_ROOT_PATH) / "gd.ply";
    }

    void skip_if_missing_gd_ply() {
        if (!std::filesystem::exists(gd_ply_path())) {
            GTEST_SKIP() << "Missing test asset: " << gd_ply_path();
        }
    }

    SplatData load_ply_q16(std::vector<AllocCall>& calls) {
        lfs::io::LoadOptions options;
        options.splat_tensor_allocator = recording_allocator(calls);
        options.shN_q16 = true;
        auto loaded = lfs::io::load_ply(gd_ply_path(), options);
        EXPECT_TRUE(loaded.has_value()) << lfs::format_for_developer(loaded.error());
        if (!loaded.has_value()) {
            return {};
        }
        return std::move(loaded->value);
    }

    SplatData load_ply_then_quant(std::vector<AllocCall>& calls) {
        lfs::io::LoadOptions options;
        options.splat_tensor_allocator = recording_allocator(calls);
        auto loaded = lfs::io::load_ply(gd_ply_path(), options);
        EXPECT_TRUE(loaded.has_value()) << lfs::format_for_developer(loaded.error());
        if (!loaded.has_value()) {
            return {};
        }
        SplatData model = std::move(loaded->value);
        EXPECT_TRUE(lfs::training::sh_value::apply_shN_value_quant(model));
        return model;
    }

    void expect_q16_byte_identical(const SplatData& expected, const SplatData& actual) {
        ASSERT_TRUE(expected.shN_value_quantized());
        ASSERT_TRUE(actual.shN_value_quantized());
        ASSERT_EQ(expected.size(), actual.size());
        ASSERT_EQ(expected.get_max_sh_degree(), actual.get_max_sh_degree());
        ASSERT_EQ(expected.get_active_sh_degree(), actual.get_active_sh_degree());

        const Tensor codes_a = expected.shN_raw().cpu();
        const Tensor codes_b = actual.shN_raw().cpu();
        ASSERT_EQ(codes_a.dtype(), DataType::Float16);
        ASSERT_EQ(codes_b.dtype(), DataType::Float16);
        ASSERT_EQ(codes_a.numel(), codes_b.numel());
        ASSERT_EQ(codes_a.bytes(), codes_b.bytes());
        EXPECT_EQ(std::memcmp(codes_a.data_ptr(), codes_b.data_ptr(), codes_a.bytes()), 0);

        const Tensor bounds_a = expected.shN_value_bounds().cpu();
        const Tensor bounds_b = actual.shN_value_bounds().cpu();
        ASSERT_EQ(bounds_a.dtype(), DataType::Float32);
        ASSERT_EQ(bounds_b.dtype(), DataType::Float32);
        ASSERT_EQ(bounds_a.numel(), bounds_b.numel());
        ASSERT_EQ(bounds_a.bytes(), bounds_b.bytes());
        EXPECT_EQ(std::memcmp(bounds_a.data_ptr(), bounds_b.data_ptr(), bounds_a.bytes()), 0);
    }

} // namespace

TEST(PlyQ16StreamedEncode, GdPlyMatchesSingleShotAndSkipsFloatShN) {
    require_cuda();
    skip_if_missing_gd_ply();

    std::vector<AllocCall> calls_single_shot;
    SplatData single_shot = load_ply_then_quant(calls_single_shot);
    ASSERT_TRUE(single_shot.shN_value_quantized());
    EXPECT_TRUE(called_for_float_shN(calls_single_shot));

    std::vector<AllocCall> calls_q16;
    SplatData streamed = load_ply_q16(calls_q16);
    ASSERT_TRUE(streamed.shN_value_quantized());
    EXPECT_FALSE(called_for_float_shN(calls_q16));
    EXPECT_FALSE(lfs::training::sh_value::apply_shN_value_quant(streamed));

    expect_q16_byte_identical(single_shot, streamed);
}

TEST(PlyQ16StreamedEncode, BandBoundaryMatchesSingleShot) {
    require_cuda();
    skip_if_missing_gd_ply();

    std::vector<AllocCall> calls_single_shot;
    SplatData single_shot = load_ply_then_quant(calls_single_shot);
    ASSERT_TRUE(single_shot.shN_value_quantized());

    ScopedPlyQ16BandPrims band(4096);
    std::vector<AllocCall> calls_q16;
    SplatData streamed = load_ply_q16(calls_q16);
    ASSERT_TRUE(streamed.shN_value_quantized());
    EXPECT_FALSE(called_for_float_shN(calls_q16));

    expect_q16_byte_identical(single_shot, streamed);
}
