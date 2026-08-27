/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/alloc_counter.hpp"
#include "core/tensor.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

using namespace lfs::core;

namespace {

    // Size large enough to land in the size-bucketed pool (above slab threshold)
    // so free → re-alloc of the same size is a cache hit, not a slab growth.
    constexpr size_t kBucketElems = 1024 * 1024; // 4 MiB of float32

    void cuda_warmup() {
        auto warm = Tensor::zeros({256}, Device::CUDA, DataType::Float32);
        ASSERT_TRUE(warm.is_valid());
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        (void)warm;
    }

} // namespace

// ---------------------------------------------------------------------------
// Task 0.1 — real device allocation counter
// Counts only cudaMalloc / cudaMallocAsync / VMM physical commits issued by
// the pool tiers, zeros_direct/reserve direct allocs, and the rasterizer arena.
// Pool cache hits must NOT increment the counter.
// ---------------------------------------------------------------------------

TEST(AllocCounterTest, SnapshotAndDeltaApiExists) {
    const auto snap = alloc_counter::snapshot();
    const auto delta = alloc_counter::delta_since(snap);
    EXPECT_EQ(delta, 0u);
    EXPECT_GE(alloc_counter::total(), snap);
}

TEST(AllocCounterTest, FreshLargeTensorIncrementsCounter) {
    cuda_warmup();

    const auto snap = alloc_counter::snapshot();
    auto t = Tensor::zeros({kBucketElems}, Device::CUDA, DataType::Float32);
    ASSERT_TRUE(t.is_valid());
    ASSERT_EQ(t.bytes(), kBucketElems * sizeof(float));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto delta = alloc_counter::delta_since(snap);
    EXPECT_GE(delta, 1u) << "fresh large CUDA tensor must issue a real driver alloc";
}

TEST(AllocCounterTest, PoolCacheHitDoesNotIncrement) {
    cuda_warmup();

    // Seed the bucketed cache with a free of this exact size.
    {
        auto seed = Tensor::zeros({kBucketElems}, Device::CUDA, DataType::Float32);
        ASSERT_TRUE(seed.is_valid());
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }
    // Destructor has returned the block to SizeBucketedPool.
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto snap = alloc_counter::snapshot();
    auto hit = Tensor::zeros({kBucketElems}, Device::CUDA, DataType::Float32);
    ASSERT_TRUE(hit.is_valid());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto delta = alloc_counter::delta_since(snap);
    EXPECT_EQ(delta, 0u) << "re-allocation of a pool-cached size must not call the driver";
}

TEST(AllocCounterTest, ZerosDirectIncrementsCounter) {
    cuda_warmup();

    // zeros_direct always uses CudaStorageMode::Direct (cudaMalloc), never the pool.
    const auto snap = alloc_counter::snapshot();
    auto t = Tensor::zeros_direct({4096, 3}, /*capacity=*/4096, Device::CUDA, DataType::Float32);
    ASSERT_TRUE(t.is_valid());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto delta = alloc_counter::delta_since(snap);
    EXPECT_GE(delta, 1u) << "zeros_direct must count as a real device allocation";
}
