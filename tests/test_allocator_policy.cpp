/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor/internal/gpu_slab_allocator.hpp"
#include "core/tensor/internal/size_bucketed_pool.hpp"
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
