/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/cuda/mask_normalization.cuh"
#include "core/cuda/mask_normalization.hpp"
#include "core/image_io.hpp"
#include "cuda_backend_test.hpp"
#include "io/pipelined_image_loader.hpp"
#include "licht_test_support.hpp"
#include "mask_test_utils.hpp"
#include <algorithm>
#include <bit>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace {
    using namespace lfs::core;
    constexpr size_t width = 129, height = 97;
    class MaskCacheEndpoints : public lfs::test::CudaBackendTest {
    protected:
        lfs::test::licht::TemporaryDirectory temp{"mask-cache-endpoints"};
        std::vector<uint8_t> values;
        void SetUp() override {
            LFS_CUDA_BACKEND_OR_RETURN();
            constexpr uint8_t bands[]{0, 1, 63, 127, 128, 129, 200, 249, 250, 251, 254, 255};
            values.resize(width * height);
            for (size_t y = 0; y < height; ++y)
                for (size_t x = 0; x < width; ++x)
                    values[y * width + x] = bands[x * 12 / width];
            const std::vector<uint8_t> rgb(width * height * 3, 96);
            ASSERT_TRUE(save_png(temp.path / "image.png", rgb.data(), width, height, 3, 8, 0));
            ASSERT_TRUE(save_png(temp.path / "mask.png", values.data(), width, height, 1, 8, 0));
        }
        auto config() {
            lfs::io::PipelinedLoaderConfig c;
            c.prefetch_count = 1;
            c.output_queue_size = 2;
            c.jpeg_batch_size = 1;
            c.decoder_pool_size = 1;
            c.io_threads = 1;
            c.cold_process_threads = 1;
            c.max_cache_bytes = 8 * 1024 * 1024;
            c.min_spill_free_bytes = 0; // the spilled visit must not depend on free space in the temp dir
            return c;
        }
        auto request() {
            lfs::io::ImageRequest r;
            r.path = temp.path / "image.png";
            r.mask_path = temp.path / "mask.png";
            r.params.resize_factor = 1;
            r.params.max_width = 0;
            return r;
        }
    };
} // namespace

TEST_F(MaskCacheEndpoints, EveryByteHasOneValueAcrossColdHotAndSpilledVisits) {
    constexpr size_t u16_count = 65536;
    std::vector<uint16_t> wide(u16_count);
    for (size_t i = 0; i < u16_count; ++i)
        wide[i] = static_cast<uint16_t>(i);
    std::vector<float> wide_actual(u16_count);
    uint16_t* device_input = nullptr;
    float* device_output = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_input), u16_count * sizeof(uint16_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_output), u16_count * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_input, wide.data(), u16_count * sizeof(uint16_t), cudaMemcpyHostToDevice), cudaSuccess);
    lfs::core::cuda::normalize_mask_u16(device_input, device_output, u16_count, nullptr);
    ASSERT_EQ(cudaMemcpy(wide_actual.data(), device_output, u16_count * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(device_input);
    cudaFree(device_output);
    for (size_t i = 0; i < wide.size(); ++i) {
        const float expected = lfs::test::mask::exact_ratio(static_cast<uint32_t>(i), 65535);
        EXPECT_EQ(std::bit_cast<uint32_t>(lfs::core::cuda::normalize_mask_u16_value(static_cast<uint16_t>(i))), std::bit_cast<uint32_t>(expected));
        EXPECT_EQ(std::bit_cast<uint32_t>(wide_actual[i]), std::bit_cast<uint32_t>(expected));
    }
    auto byte_values = lfs::test::mask::byte_ramp(256);
    auto byte_host = Tensor::empty({byte_values.size()}, Device::CPU, DataType::UInt8);
    std::copy(byte_values.begin(), byte_values.end(), byte_host.ptr<uint8_t>());
    const auto byte_device = byte_host.to(Device::CUDA);
    const auto byte_actual = lfs::core::cuda::normalize_mask_bytes(byte_device).cpu().to_vector();
    for (size_t i = 0; i < byte_values.size(); ++i) {
        const float expected = lfs::test::mask::exact_ratio(static_cast<uint32_t>(i), 255);
        EXPECT_EQ(std::bit_cast<uint32_t>(lfs::core::cuda::normalize_mask_byte(byte_values[i])), std::bit_cast<uint32_t>(expected));
        EXPECT_EQ(std::bit_cast<uint32_t>(byte_actual[i]), std::bit_cast<uint32_t>(expected));
    }
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = static_cast<uint8_t>(i % 256);
    ASSERT_TRUE(save_png(temp.path / "mask.png", values.data(), width, height, 1, 8, 0));
    for (const bool invert : {false, true}) {
        lfs::io::PipelinedImageLoader loader(config());
        auto r = request();
        r.mask_params = {.invert = invert, .threshold = 0.f};
        for (size_t visit = 0; visit < 3; ++visit) {
            SCOPED_TRACE(invert);
            SCOPED_TRACE(visit);
            if (visit == 2) {
                const auto stats = loader.get_stats();
                EXPECT_GT(loader.release_host_cache(stats.jpeg_cache_bytes), 0u);
                EXPECT_GT(loader.get_stats().spill_cache_entries, 0u);
            }
            r.sequence_id = visit;
            loader.prefetch({r});
            const auto ready = loader.get();
            ASSERT_TRUE(ready.error.empty()) << ready.error;
            ASSERT_TRUE(ready.mask);
            const auto actual = ready.mask->cpu().to_vector();
            size_t mismatches = 0;
            std::vector<float> expected(actual.size());
            for (size_t i = 0; i < actual.size(); ++i) {
                const float normalized = static_cast<float>(static_cast<double>(values[i]) / 255.0);
                expected[i] = invert ? 1.f - normalized : normalized;
                mismatches += std::bit_cast<uint32_t>(actual[i]) != std::bit_cast<uint32_t>(expected[i]);
            }
            EXPECT_EQ(mismatches, 0u);
        }
    }
}
