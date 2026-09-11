/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Tasks 3.3 + 3.4 + 3.7 — allocator hygiene
//  3.3 route bare cudaMalloc op temps through the pool
//  3.4 free fully-empty slabs on trim_cached_memory
//  3.7 empty CUDA tensors use null-owner path (no 1-byte CUDA sentinel)

#include "core/alloc_counter.hpp"
#include "core/tensor.hpp"
#include "core/tensor/internal/gpu_slab_allocator.hpp"
#include "core/tensor/internal/memory_pool.hpp"

#include <atomic>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using namespace lfs::core;

namespace {

    std::atomic<void*> g_failed_reclaim_ptr{nullptr};

    cudaError_t fail_slab_reclaim_for_testing(void* ptr) {
        void* expected = nullptr;
        g_failed_reclaim_ptr.compare_exchange_strong(expected, ptr);
        return cudaErrorMemoryAllocation;
    }

    class ReclaimHookReset {
    public:
        explicit ReclaimHookReset(GPUSlabAllocator& allocator)
            : allocator_(allocator) {}
        ~ReclaimHookReset() { allocator_.set_reclaim_free_fn_for_testing(nullptr); }

    private:
        GPUSlabAllocator& allocator_;
    };

    void cuda_ok() {
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }

} // namespace

// ---------------------------------------------------------------------------
// 3.7 — empty CUDA tensor must not touch the slab allocator
// ---------------------------------------------------------------------------

TEST(AllocatorHygiene, EmptyCudaTensorDoesNotAllocateSlabBlock) {
    // Warm the pool so later zeros/empty of real sizes do not confound counts.
    {
        auto warm = Tensor::zeros({256}, Device::CUDA, DataType::Float32);
        ASSERT_TRUE(warm.is_valid());
        cuda_ok();
    }

    const uint64_t slab_allocs_before =
        GPUSlabAllocator::instance().stats().alloc_count.load(std::memory_order_relaxed);

    {
        auto empty = Tensor::empty({0}, Device::CUDA, DataType::Float32);
        EXPECT_TRUE(empty.is_valid());
        EXPECT_EQ(empty.numel(), 0u);
        EXPECT_EQ(empty.data_ptr(), nullptr);

        auto clone = empty.clone();
        EXPECT_TRUE(clone.is_valid());
        EXPECT_EQ(clone.numel(), 0u);
        EXPECT_EQ(clone.data_ptr(), nullptr);

        // Multi-dim empty (e.g. {0, 3}) same rule
        auto empty2d = Tensor::empty({0, 3}, Device::CUDA, DataType::Float32);
        EXPECT_EQ(empty2d.numel(), 0u);
        EXPECT_EQ(empty2d.data_ptr(), nullptr);
    }
    cuda_ok();

    const uint64_t slab_allocs_after =
        GPUSlabAllocator::instance().stats().alloc_count.load(std::memory_order_relaxed);

    // Pre-fix: empty() allocates a 1-byte CUDA sentinel through the slab →
    // alloc_count advances. Post-fix: null-owner path → no slab touch.
    EXPECT_EQ(slab_allocs_after, slab_allocs_before)
        << "empty CUDA tensors must not allocate a 1-byte CUDA sentinel";
}

// ---------------------------------------------------------------------------
// 3.4 — fully-empty slabs return to the driver on trim_cached_memory
// ---------------------------------------------------------------------------

TEST(AllocatorHygiene, TrimCachedMemoryFreesFullyEmptySlabs) {
    auto& pool = CudaMemoryPool::instance();
    auto& slab = GPUSlabAllocator::instance();

    // Start from a reclaimed baseline (may still hold non-empty slabs from other
    // tests sharing the process-global pool).
    pool.trim_cached_memory();
    cuda_ok();
    const size_t reserved_baseline = slab.stats().total_slab_memory;

    // Touch the smallest size class with enough live blocks to force a slab
    // growth if none is present, then drop every live block.
    {
        std::vector<Tensor> hold;
        hold.reserve(2048);
        for (int i = 0; i < 2048; ++i) {
            // 256 floats = 1 KiB → size class above 256 B, still in slab range.
            hold.push_back(Tensor::zeros({256}, Device::CUDA, DataType::Float32));
            ASSERT_TRUE(hold.back().is_valid());
        }
        cuda_ok();
    }
    cuda_ok();

    const size_t reserved_peak = slab.stats().total_slab_memory;
    ASSERT_GT(reserved_peak, 0u) << "expected slab growth for small tensors";
    ASSERT_GE(reserved_peak, reserved_baseline);

    // Pre-fix: trim only merges free lists into virgin; slabs stay reserved
    // until process shutdown. Post-fix: fully-empty slabs are cudaFree'd.
    pool.trim_cached_memory();
    cuda_ok();

    const size_t reserved_after = slab.stats().total_slab_memory;
    EXPECT_LT(reserved_after, reserved_peak)
        << "trim_cached_memory must free fully-empty slabs (peak="
        << reserved_peak << " after=" << reserved_after << ")";
}

TEST(AllocatorHygiene, FailedSlabReclaimRestoresOwnershipAndAccounting) {
    auto& pool = CudaMemoryPool::instance();
    auto& slab = GPUSlabAllocator::instance();

    pool.trim_cached_memory();
    cuda_ok();
    {
        auto tensor = Tensor::zeros({64}, Device::CUDA, DataType::Float32);
        ASSERT_TRUE(tensor.is_valid());
        cuda_ok();
    }
    cuda_ok();

    const size_t reserved_before = slab.stats().total_slab_memory;
    ASSERT_GT(reserved_before, 0u);

    g_failed_reclaim_ptr.store(nullptr, std::memory_order_release);
    slab.set_reclaim_free_fn_for_testing(&fail_slab_reclaim_for_testing);
    ReclaimHookReset reset_hook(slab);
    pool.trim_cached_memory();

    void* const failed_ptr = g_failed_reclaim_ptr.load(std::memory_order_acquire);
    ASSERT_NE(failed_ptr, nullptr);
    EXPECT_TRUE(slab.owns_pointer(failed_ptr));
    EXPECT_EQ(slab.stats().total_slab_memory, reserved_before)
        << "a failed driver free must keep the slab budget charged";

    // The slab stays quarantined until allocator shutdown because CUDA can
    // report an older asynchronous error from cudaFree, making reuse unsafe.
}

// ---------------------------------------------------------------------------
// 3.3 — op temps route through the pool (count_nonzero correctness + hygiene)
// ---------------------------------------------------------------------------

TEST(AllocatorHygiene, CountNonzeroPoolPathCorrectAndReusable) {
    // Correctness after routing d_count through CudaMemoryPool instead of bare
    // cudaMalloc/cudaFree.
    auto zeros = Tensor::zeros({128}, Device::CUDA, DataType::Float32);
    EXPECT_EQ(zeros.count_nonzero(), 0u);

    auto ones = Tensor::ones({64}, Device::CUDA, DataType::Float32);
    EXPECT_EQ(ones.count_nonzero(), 64u);

    // Mixed: set a few nonzeros via host upload
    std::vector<float> host(32, 0.0f);
    host[3] = 1.0f;
    host[7] = 2.0f;
    host[31] = -1.0f;
    auto mixed = Tensor::from_vector(host, {32}, Device::CUDA);
    EXPECT_EQ(mixed.count_nonzero(), 3u);

    // Second call reuses the pooled d_count block (no extra driver growth once warm).
    {
        // Seed any residual free-list for sizeof(size_t)
        (void)mixed.count_nonzero();
        cuda_ok();
        const auto snap = alloc_counter::snapshot();
        EXPECT_EQ(mixed.count_nonzero(), 3u);
        EXPECT_EQ(ones.count_nonzero(), 64u);
        cuda_ok();
        EXPECT_EQ(alloc_counter::delta_since(snap), 0u)
            << "steady count_nonzero must not issue driver allocs after warm-up";
    }
}

TEST(AllocatorHygiene, EmptyAndCountNonzeroCompose) {
    auto empty = Tensor::empty({0}, Device::CUDA, DataType::Float32);
    EXPECT_EQ(empty.count_nonzero(), 0u);
}
