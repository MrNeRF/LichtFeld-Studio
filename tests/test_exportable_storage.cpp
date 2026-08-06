/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include "core/exportable_storage.hpp"
#include "core/splat_exportable_storage.hpp"
#include "diagnostics/vram_profiler.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

using namespace lfs::core;

namespace {

    void require_cuda() {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
    }

    void fill_device_pattern(void* device_ptr, std::size_t floats, float base) {
        std::vector<float> host(floats);
        for (std::size_t i = 0; i < floats; ++i) {
            host[i] = base + static_cast<float>(i);
        }
        ASSERT_EQ(cudaMemcpy(device_ptr, host.data(), floats * sizeof(float), cudaMemcpyHostToDevice),
                  cudaSuccess);
    }

    void expect_device_pattern(const void* device_ptr, std::size_t floats, float base) {
        std::vector<float> host(floats);
        ASSERT_EQ(cudaMemcpy(host.data(), device_ptr, floats * sizeof(float), cudaMemcpyDeviceToHost),
                  cudaSuccess);
        for (std::size_t i = 0; i < floats; ++i) {
            EXPECT_FLOAT_EQ(host[i], base + static_cast<float>(i)) << "index " << i;
        }
    }

} // namespace

TEST(ExportableStorageTest, ImmediateDestroyLeavesCudaUsable) {
    require_cuda();

    constexpr std::size_t BLOCK_BYTES = 1 << 20;
    auto block_result = allocateExportableDeviceBlock(BLOCK_BYTES, 0, false);
    if (!block_result) {
        FAIL() << block_result.error();
    }

    auto block = std::move(*block_result);
    ASSERT_NE(block, nullptr);
    ASSERT_NE(block->device_ptr, nullptr);
    block.reset();

    constexpr std::size_t PROBE_BYTES = 4096;
    void* probe = nullptr;
    LFS_CUDA_CHECK_MSG_ARGS(
        cudaMalloc(&probe, PROBE_BYTES),
        reinterpret_cast<uintptr_t>(probe),
        0,
        PROBE_BYTES,
        "allocating unrelated CUDA probe dst={} src={} bytes={}",
        probe,
        static_cast<const void*>(nullptr),
        PROBE_BYTES);
    LFS_CUDA_CHECK_MSG_ARGS(
        cudaMemset(probe, 0xa5, PROBE_BYTES),
        reinterpret_cast<uintptr_t>(probe),
        0,
        PROBE_BYTES,
        "writing unrelated CUDA probe dst={} src={} bytes={}",
        probe,
        static_cast<const void*>(nullptr),
        PROBE_BYTES);
    LFS_CUDA_CHECK_MSG_ARGS(
        cudaDeviceSynchronize(),
        reinterpret_cast<uintptr_t>(probe),
        0,
        PROBE_BYTES,
        "synchronizing unrelated CUDA probe dst={} src={} bytes={}",
        probe,
        static_cast<const void*>(nullptr),
        PROBE_BYTES);
    LFS_CUDA_CHECK_MSG_ARGS(
        cudaFree(probe),
        0,
        reinterpret_cast<uintptr_t>(probe),
        PROBE_BYTES,
        "freeing unrelated CUDA probe dst={} src={} bytes={}",
        static_cast<const void*>(nullptr),
        probe,
        PROBE_BYTES);
}

// ---------------------------------------------------------------------------
// Phase 5.1 — exportable splat block grows with live N
// ---------------------------------------------------------------------------

TEST(SplatExportableStorageTest, CreateTracksLiveCapacityNotMaxCap) {
    require_cuda();

    constexpr std::size_t kLive = 1024;
    constexpr std::size_t kMaxCap = 5'000'000;
    constexpr int kShDegree = 3;

    const std::size_t live_bytes = SplatExportableStorage::layoutBytes(kLive, kShDegree);
    const std::size_t max_bytes = SplatExportableStorage::layoutBytes(kMaxCap, kShDegree);
    ASSERT_GT(max_bytes, live_bytes);
    ASSERT_GT(max_bytes - live_bytes, 100ull << 20); // hundreds of MiB at 5M SH3

    auto storage_result = SplatExportableStorage::create(kLive, kShDegree, 0, kMaxCap);
    if (!storage_result) {
        FAIL() << storage_result.error();
    }
    auto storage = std::move(*storage_result);
    ASSERT_TRUE(storage.valid());
    EXPECT_EQ(storage.capacity(), kLive);
    EXPECT_EQ(storage.reservedCapacity(), kMaxCap);
    EXPECT_GE(storage.block->size, live_bytes);
    // Committed physical must track live N, not max_cap.
    EXPECT_LT(storage.block->size, max_bytes / 4);

    const auto snap = lfs::diagnostics::VramProfiler::instance().snapshot();
    EXPECT_EQ(snap.process.exportable_splat_bytes, storage.block->size);
    EXPECT_LT(snap.process.exportable_splat_bytes, max_bytes / 4);
}

TEST(SplatExportableStorageTest, GrowPreservesDataAndTracksBytes) {
    require_cuda();

    constexpr std::size_t kInitial = 256;
    constexpr std::size_t kGrown = 512;
    constexpr std::size_t kReserve = 4096;
    constexpr int kShDegree = 3;

    auto storage_result = SplatExportableStorage::create(kInitial, kShDegree, 0, kReserve);
    if (!storage_result) {
        FAIL() << storage_result.error();
    }
    auto storage = std::move(*storage_result);
    ASSERT_TRUE(storage.valid());
    const void* const stable_ptr = storage.block->device_ptr;
    const std::size_t bytes_before = storage.block->size;
    const auto gen_before = storage.generation();

    // Write a known pattern into the means region (first kInitial * 3 floats).
    constexpr std::size_t kMeansFloats = kInitial * 3;
    fill_device_pattern(storage.block->device_ptr, kMeansFloats, 10.0f);

    // Also stamp scaling region so relocation is tested for a non-zero offset.
    void* scaling_ptr =
        static_cast<char*>(storage.block->device_ptr) + storage.region_offsets[SplatExportableStorage::Scaling];
    constexpr std::size_t kScalingFloats = kInitial * 3;
    fill_device_pattern(scaling_ptr, kScalingFloats, 100.0f);

    auto grew = storage.grow(kGrown);
    if (!grew) {
        FAIL() << grew.error();
    }
    ASSERT_TRUE(*grew);
    EXPECT_EQ(storage.capacity(), kGrown);
    EXPECT_EQ(storage.block->device_ptr, stable_ptr) << "device_ptr must stay stable across grow";
    EXPECT_GT(storage.generation(), gen_before);
    EXPECT_GE(storage.block->size, bytes_before);

    const auto snap = lfs::diagnostics::VramProfiler::instance().snapshot();
    EXPECT_EQ(snap.process.exportable_splat_bytes, storage.block->size);
    EXPECT_GE(snap.process.exportable_splat_bytes, SplatExportableStorage::layoutBytes(kGrown, kShDegree));

    // (a) old data intact after growth (means at offset 0, scaling at new offset).
    expect_device_pattern(storage.block->device_ptr, kMeansFloats, 10.0f);
    void* scaling_after =
        static_cast<char*>(storage.block->device_ptr) + storage.region_offsets[SplatExportableStorage::Scaling];
    expect_device_pattern(scaling_after, kScalingFloats, 100.0f);

    // Idempotent: grow to same capacity is a no-op.
    auto grew_again = storage.grow(kGrown);
    ASSERT_TRUE(grew_again.has_value());
    EXPECT_FALSE(*grew_again);
}

TEST(SplatExportableStorageTest, TensorViewsValidAfterGrowViaRebind) {
    require_cuda();

    constexpr std::size_t kInitial = 128;
    constexpr std::size_t kGrown = 256;
    constexpr int kShDegree = 0; // no shN rest; keeps the fixture minimal

    auto storage_result = SplatExportableStorage::create(kInitial, kShDegree, 0, kGrown * 2);
    if (!storage_result) {
        FAIL() << storage_result.error();
    }
    auto storage = std::move(*storage_result);
    auto allocator = storage.make_allocator();

    // Build a tiny SplatData backed by exportable storage.
    const size_t n = 64;
    Tensor means = allocator(TensorShape({n, 3}), kInitial, DataType::Float32, "SplatData.means");
    Tensor scaling = allocator(TensorShape({n, 3}), kInitial, DataType::Float32, "SplatData.scaling");
    Tensor rotation = allocator(TensorShape({n, 4}), kInitial, DataType::Float32, "SplatData.rotation");
    Tensor opacity = allocator(TensorShape({n, 1}), kInitial, DataType::Float32, "SplatData.opacity");
    Tensor sh0 = allocator(TensorShape({n, 1, 3}), kInitial, DataType::Float32, "SplatData.sh0");
    Tensor shN; // degree 0: empty rest

    // Fill means with identity pattern.
    {
        std::vector<float> host(n * 3);
        for (size_t i = 0; i < n * 3; ++i)
            host[i] = static_cast<float>(i + 1);
        ASSERT_EQ(cudaMemcpy(means.ptr<float>(), host.data(), host.size() * sizeof(float),
                             cudaMemcpyHostToDevice),
                  cudaSuccess);
    }

    EXPECT_EQ(means.capacity(), kInitial);
    EXPECT_EQ(means.external_storage_kind(), "splat.exportable");

    // Grow storage (region offsets for non-means change).
    auto grew = storage.grow(kGrown);
    if (!grew) {
        FAIL() << grew.error();
    }
    ASSERT_TRUE(*grew);

    // Rebuild tensor views via rebind API (c).
    SplatData model(/*max_sh_degree=*/kShDegree,
                    std::move(means),
                    std::move(sh0),
                    std::move(shN),
                    std::move(scaling),
                    std::move(rotation),
                    std::move(opacity),
                    /*scene_scale=*/1.0f,
                    SplatData::ShNLayout::Swizzled);

    auto rebound = storage.rebindSplatData(model);
    if (!rebound) {
        FAIL() << rebound.error();
    }

    EXPECT_EQ(model.means_raw().capacity(), kGrown);
    EXPECT_EQ(model.means_raw().external_storage_kind(), "splat.exportable");
    EXPECT_EQ(model.scaling_raw().capacity(), kGrown);

    // Means data preserved.
    {
        std::vector<float> host(n * 3);
        ASSERT_EQ(cudaMemcpy(host.data(), model.means_raw().ptr<float>(), host.size() * sizeof(float),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        for (size_t i = 0; i < n * 3; ++i) {
            EXPECT_FLOAT_EQ(host[i], static_cast<float>(i + 1));
        }
    }

    // Allocator after grow hands out views at new capacity.
    Tensor means2 = storage.make_allocator()(
        TensorShape({n, 3}), /*requested max_cap-like*/ kGrown * 10, DataType::Float32, "SplatData.means");
    EXPECT_EQ(means2.capacity(), kGrown) << "allocator must clamp to committed capacity";
}

TEST(SplatExportableStorageTest, GrowthCapacityHelper) {
    EXPECT_EQ(SplatExportableStorage::growthCapacity(100), 150u);
    EXPECT_EQ(SplatExportableStorage::growthCapacity(100, 120), 120u);
    EXPECT_EQ(SplatExportableStorage::growthCapacity(100, 50), 50u);
    EXPECT_EQ(SplatExportableStorage::growthCapacity(0, 5'000'000), 1u);
}
