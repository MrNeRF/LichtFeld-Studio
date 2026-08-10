/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor/internal/gpu_slab_allocator.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include "core/tensor/internal/size_bucketed_pool.hpp"
#include <cuda_runtime.h>
#include <gtest/gtest.h>

using namespace lfs::core;

TEST(AllocatorPolicyTest, SlabReserveScalesBySizeClass) {
    constexpr size_t KiB = 1024;
    constexpr size_t MiB = 1024 * KiB;

    size_t all_classes_first_touch = 0;
    for (size_t i = 0; i < GPUSlabAllocator::NUM_SIZE_CLASSES; ++i) {
        const size_t block_size = GPUSlabAllocator::get_block_size(i);
        const size_t slab_size = GPUSlabAllocator::slab_size_for_class(i);
        EXPECT_GE(slab_size, block_size);
        EXPECT_EQ(slab_size % block_size, 0u);
        EXPECT_GE(slab_size / block_size, 32u);
        all_classes_first_touch += slab_size;
    }

    EXPECT_EQ(GPUSlabAllocator::slab_size_for_class(0), 256 * KiB);
    EXPECT_EQ(GPUSlabAllocator::slab_size_for_class(10), 8 * MiB);
    EXPECT_LT(all_classes_first_touch, 64 * MiB);
}

TEST(AllocatorPolicyTest, BucketCacheBudgetStaysBoundedOnLargeGpus) {
    constexpr size_t MiB = 1024 * 1024;
    constexpr size_t GiB = 1024 * MiB;

    EXPECT_EQ(SizeBucketedPool::cache_budget_for_total_memory(2 * GiB), 64 * MiB);
    EXPECT_GT(SizeBucketedPool::cache_budget_for_total_memory(8 * GiB), 64 * MiB);
    EXPECT_LT(SizeBucketedPool::cache_budget_for_total_memory(8 * GiB), 128 * MiB);
    EXPECT_EQ(SizeBucketedPool::cache_budget_for_total_memory(24 * GiB), 256 * MiB);
    EXPECT_EQ(SizeBucketedPool::cache_budget_for_total_memory(48 * GiB), 256 * MiB);
}

// get_bucket_size remains correct (bucket_size(x) >= x). get_bucket_index must
// NOT clamp to NUM_BUCKETS-1 — oversize requests bypass the cache entirely.
TEST(AllocatorPolicyTest, BucketIndexBypassesCachePastTableAndSizeContractHolds) {
    constexpr size_t GiB = 1024ULL * 1024 * 1024;
    // Sweep includes sizes that formerly alias into bucket 127 (≥ ~60 GiB).
    const size_t sizes[] = {
        1 * GiB,
        8 * GiB,
        16 * GiB,
        32 * GiB,
        59 * GiB,
        60 * GiB,
        70 * GiB,
        100 * GiB,
    };
    for (const size_t req : sizes) {
        const size_t bucket_size = SizeBucketedPool::get_bucket_size(req);
        EXPECT_GE(bucket_size, req) << "get_bucket_size must round up for " << req;
        const size_t idx = SizeBucketedPool::get_bucket_index(bucket_size);
        if (idx >= SizeBucketedPool::NUM_BUCKETS) {
            EXPECT_TRUE(SizeBucketedPool::bucket_index_bypasses_cache(idx));
            // try_allocate_cached / cache_free must no-op (return null/false)
            // without handing back a smaller block from bucket 127.
            auto& pool = SizeBucketedPool::instance();
            EXPECT_EQ(pool.try_allocate_cached(req), nullptr);
            EXPECT_FALSE(pool.cache_free(reinterpret_cast<void*>(0x1), req));
        } else {
            EXPECT_FALSE(SizeBucketedPool::bucket_index_bypasses_cache(idx));
            EXPECT_LT(idx, SizeBucketedPool::NUM_BUCKETS);
        }
    }
    // Explicit aliasing pair: two sizes that under the old clamp shared 127.
    const size_t a = SizeBucketedPool::get_bucket_size(60 * GiB);
    const size_t b = SizeBucketedPool::get_bucket_size(70 * GiB);
    EXPECT_GE(a, 60 * GiB);
    EXPECT_GE(b, 70 * GiB);
    EXPECT_TRUE(SizeBucketedPool::bucket_index_bypasses_cache(
        SizeBucketedPool::get_bucket_index(a)));
    EXPECT_TRUE(SizeBucketedPool::bucket_index_bypasses_cache(
        SizeBucketedPool::get_bucket_index(b)));
}

TEST(AllocatorPolicyTest, OversizedSingletonDoesNotEscapeBucketCacheBudget) {
    constexpr size_t MiB = 1024 * 1024;
    auto& pool = SizeBucketedPool::instance();
    pool.trim_cache();
    pool.set_cache_budget_for_testing(1 * MiB);

    // The first use is deliberately probationary. A second miss makes this a
    // reusable bucket and exercises global budget enforcement on deallocation.
    for (int attempt = 0; attempt < 2; ++attempt) {
        void* ptr = pool.allocate(2 * MiB);
        EXPECT_NE(ptr, nullptr);
        if (ptr) {
            pool.deallocate(ptr, 2 * MiB);
        }
    }

    EXPECT_EQ(pool.stats().bytes_cached.load(std::memory_order_relaxed), 0u);
    pool.trim_cache();
    pool.set_cache_budget_for_testing(0);
}

TEST(AllocatorPolicyTest, LiveBucketWasteTracksOnlyOutstandingAllocations) {
    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    if (device_count == 0) {
        GTEST_SKIP() << "No CUDA device available";
    }

    constexpr size_t MiB = 1024 * 1024;
    auto& pool = SizeBucketedPool::instance();
    const auto baseline =
        pool.stats().live_rounding_waste.load(std::memory_order_relaxed);

    void* exact = pool.allocate(MiB);
    ASSERT_NE(exact, nullptr);
    EXPECT_EQ(pool.stats().live_rounding_waste.load(std::memory_order_relaxed),
              baseline);
    pool.deallocate(exact, MiB);
    EXPECT_EQ(pool.stats().live_rounding_waste.load(std::memory_order_relaxed),
              baseline);

    constexpr size_t request = MiB + 1;
    const size_t expected_waste = SizeBucketedPool::get_bucket_size(request) - request;
    void* rounded = pool.allocate(request);
    ASSERT_NE(rounded, nullptr);
    EXPECT_EQ(pool.stats().live_rounding_waste.load(std::memory_order_relaxed),
              baseline + expected_waste);
    pool.deallocate(rounded, request);
    EXPECT_EQ(pool.stats().live_rounding_waste.load(std::memory_order_relaxed),
              baseline);
}

TEST(AllocatorPolicyTest, ExactAsyncBypassesBucketCache) {
    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    if (device_count == 0) {
        GTEST_SKIP() << "No CUDA device available";
    }

    constexpr size_t MiB = 1024 * 1024;
    constexpr size_t request = MiB + 17;
    auto& bucket_pool = SizeBucketedPool::instance();
    const auto cached_before =
        bucket_pool.stats().bytes_cached.load(std::memory_order_relaxed);
    const auto waste_before =
        bucket_pool.stats().live_rounding_waste.load(std::memory_order_relaxed);

    void* ptr = nullptr;
    {
        CudaMemoryPool::LabelGuard label_guard("test.exact_async");
        ptr = allocate_cuda_storage(
            request, nullptr, CudaStorageMode::ExactAsync,
            "test.exact_async", "allocator-policy-test");
    }
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(bucket_pool.stats().bytes_cached.load(std::memory_order_relaxed),
              cached_before);
    EXPECT_EQ(
        bucket_pool.stats().live_rounding_waste.load(std::memory_order_relaxed),
        waste_before);

    safe_cuda_pool_deallocate(ptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(bucket_pool.stats().bytes_cached.load(std::memory_order_relaxed),
              cached_before);
    EXPECT_EQ(
        bucket_pool.stats().live_rounding_waste.load(std::memory_order_relaxed),
        waste_before);
}
