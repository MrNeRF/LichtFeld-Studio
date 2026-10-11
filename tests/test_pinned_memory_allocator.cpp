/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/image_io.hpp"
#include "core/pinned_memory_allocator.hpp"
#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <limits>
#include <memory>

namespace lfs::core {

    class PinnedMemoryAllocatorTest : public ::testing::Test {
    protected:
        void TearDown() override {
            auto& allocator = PinnedMemoryAllocator::instance();
            allocator.empty_cache();
            allocator.set_cache_limit_for_testing(initial_cache_limit_);
        }

        static size_t round_size(const size_t bytes) {
            return PinnedMemoryAllocator::round_size(bytes);
        }

        static constexpr size_t MIB = 1024 * 1024;
        const size_t initial_cache_limit_ = PinnedMemoryAllocator::instance().cache_limit_bytes();
    };

    TEST_F(PinnedMemoryAllocatorTest, PreservesSmallAndMediumBuckets) {
        EXPECT_EQ(round_size(0), 0);
        EXPECT_EQ(round_size(4095), 4095);
        EXPECT_EQ(round_size(4096), 4096);
        EXPECT_EQ(round_size(4097), 4608);
        EXPECT_EQ(round_size(MIB - 1), MIB);
        EXPECT_EQ(round_size(MIB), MIB);
        EXPECT_EQ(round_size(MIB + 1), 2 * MIB);
        EXPECT_EQ(round_size(64 * MIB + 1), 128 * MIB);
        EXPECT_EQ(round_size(128 * MIB), 128 * MIB);
    }

    TEST_F(PinnedMemoryAllocatorTest, BoundsLargeBucketWaste) {
        EXPECT_EQ(round_size(128 * MIB + 1), 192 * MIB);
        EXPECT_EQ(round_size(192 * MIB), 192 * MIB);
        EXPECT_EQ(round_size(192 * MIB + 1), 256 * MIB);
        EXPECT_EQ(round_size(256 * MIB + 1), 320 * MIB);
        EXPECT_EQ(round_size(4096 * MIB + 103 * MIB), 4224 * MIB);
    }

    TEST_F(PinnedMemoryAllocatorTest, DoesNotOverflowLargeBuckets) {
        constexpr auto maximum = std::numeric_limits<size_t>::max();
        constexpr size_t granularity = 64 * MIB;
        constexpr size_t last_bucket = maximum - (granularity - 1);
        EXPECT_EQ(round_size(last_bucket - 1), last_bucket);
        EXPECT_EQ(round_size(last_bucket), last_bucket);
        EXPECT_EQ(round_size(last_bucket + 1), last_bucket + 1);
        EXPECT_EQ(round_size(maximum), maximum);
    }

    TEST_F(PinnedMemoryAllocatorTest, ReusesLargeImageBucket) {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
            GTEST_SKIP() << "CUDA device unavailable";
        const auto path = std::filesystem::path(PROJECT_ROOT_PATH) / "data/bicycle/images/_DSC8679.JPG";
        if (!std::filesystem::exists(path))
            GTEST_SKIP() << "Real image unavailable";
        auto [data, width, height, channels] = load_image(path, 1, 0);
        const std::unique_ptr<unsigned char, decltype(&free_image)> pixels(data, &free_image);
        ASSERT_NE(data, nullptr);
        const size_t elements = static_cast<size_t>(width) * height * channels;
        const size_t bytes = elements * sizeof(float);
        ASSERT_GT(bytes, 128 * MIB);
        const size_t bucket = ((bytes - 1) / (64 * MIB) + 1) * (64 * MIB);

        auto& allocator = PinnedMemoryAllocator::instance();
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        allocator.empty_cache();
        allocator.set_cache_limit_for_testing(512 * MIB);
        const auto before = allocator.get_stats();
        const auto release = [&](float* ptr) { allocator.deallocate(ptr); };
        std::unique_ptr<float, decltype(release)> buffer(static_cast<float*>(allocator.allocate(bytes)), release);
        ASSERT_NE(buffer, nullptr);
        ASSERT_TRUE(allocator.is_cuda_host_allocation(buffer.get()));
        EXPECT_EQ(allocator.get_stats().allocated_bytes - before.allocated_bytes, bucket);
        std::transform(data, data + elements, buffer.get(), [](const unsigned char value) { return value / 255.0f; });
        auto* const original = buffer.get();
        buffer.reset();
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_EQ(allocator.get_stats().cached_bytes, bucket);

        buffer.reset(static_cast<float*>(allocator.allocate(bytes - sizeof(float))));
        ASSERT_EQ(buffer.get(), original);
        const auto reused = allocator.get_stats();
        EXPECT_EQ(reused.cache_hits, before.cache_hits + 1);
        EXPECT_EQ(reused.num_allocs, before.num_allocs + 1);
        EXPECT_EQ(reused.cached_bytes, 0);
        EXPECT_EQ(reused.allocated_bytes - before.allocated_bytes, bucket);
        for (size_t i = 0; i + 1 < elements; ++i) {
            ASSERT_EQ(buffer.get()[i], data[i] / 255.0f) << i;
        }
        buffer.reset();
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        allocator.empty_cache();
        EXPECT_EQ(allocator.get_stats().allocated_bytes, before.allocated_bytes);
        EXPECT_EQ(allocator.get_stats().cached_bytes, 0);
    }

} // namespace lfs::core
