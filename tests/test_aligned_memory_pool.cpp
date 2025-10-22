/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/aligned_memory_pool.hpp"
#include <cuda_runtime.h>
#include <gtest/gtest.h>

using namespace lfs::core;

class AlignedMemoryPoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Warmup GPU
        int device;
        cudaGetDevice(&device);
        cudaDeviceSynchronize();
    }
};

// ============= Basic Allocation Tests =============

TEST_F(AlignedMemoryPoolTest, BasicAllocation) {
    size_t bytes = 1024;
    size_t alignment = 16;

    void* ptr = AlignedMemoryPool::instance().allocate_aligned(bytes, alignment);

    ASSERT_NE(ptr, nullptr);

    // Check alignment
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    EXPECT_EQ(addr % alignment, 0) << "Pointer not aligned to " << alignment << " bytes";

    // Test we can write to it
    cudaMemset(ptr, 0, bytes);
    cudaDeviceSynchronize();

    AlignedMemoryPool::instance().deallocate_aligned(ptr);
}

TEST_F(AlignedMemoryPoolTest, CacheLineAlignment) {
    size_t bytes = 4096;
    size_t alignment = 128; // Cache line size

    void* ptr = AlignedMemoryPool::instance().allocate_aligned(bytes, alignment);

    ASSERT_NE(ptr, nullptr);

    // Check 128-byte alignment
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    EXPECT_EQ(addr % alignment, 0) << "Pointer not aligned to cache line (128 bytes)";

    AlignedMemoryPool::instance().deallocate_aligned(ptr);
}

TEST_F(AlignedMemoryPoolTest, Float4Alignment) {
    size_t bytes = 256;
    size_t alignment = 16; // float4 requires 16-byte alignment

    void* ptr = AlignedMemoryPool::instance().allocate_aligned(bytes, alignment);

    ASSERT_NE(ptr, nullptr);

    // Check 16-byte alignment for float4
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    EXPECT_EQ(addr % 16, 0) << "Pointer not aligned for float4 operations";

    // Test we can use float4 operations
    cudaMemset(ptr, 0, bytes);
    cudaDeviceSynchronize();

    AlignedMemoryPool::instance().deallocate_aligned(ptr);
}

// ============= Multiple Allocations =============

TEST_F(AlignedMemoryPoolTest, MultipleAllocations) {
    const int num_allocs = 100;
    std::vector<void*> ptrs;
    size_t alignment = 128;

    for (int i = 0; i < num_allocs; ++i) {
        size_t bytes = 1024 + i * 64; // Varying sizes
        void* ptr = AlignedMemoryPool::instance().allocate_aligned(bytes, alignment);

        ASSERT_NE(ptr, nullptr);

        // Check alignment
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        EXPECT_EQ(addr % alignment, 0) << "Allocation " << i << " not aligned";

        ptrs.push_back(ptr);
    }

    // Deallocate all
    for (void* ptr : ptrs) {
        AlignedMemoryPool::instance().deallocate_aligned(ptr);
    }
}

// ============= Smart Allocation Tests =============

TEST_F(AlignedMemoryPoolTest, SmartAllocationSmallTensor) {
    size_t bytes = 2048; // < 4KB threshold
    size_t threshold = 4096;
    size_t alignment = 128;

    void* ptr = AlignedMemoryPool::instance().allocate_smart(bytes, nullptr, threshold, alignment);

    ASSERT_NE(ptr, nullptr);

    // For small allocations, alignment is NOT guaranteed (uses regular pool)
    // So we just test that it works
    cudaMemset(ptr, 0, bytes);
    cudaDeviceSynchronize();

    AlignedMemoryPool::instance().deallocate_smart(ptr);
}

TEST_F(AlignedMemoryPoolTest, SmartAllocationLargeTensor) {
    size_t bytes = 8192; // > 4KB threshold
    size_t threshold = 4096;
    size_t alignment = 128;

    void* ptr = AlignedMemoryPool::instance().allocate_smart(bytes, nullptr, threshold, alignment);

    ASSERT_NE(ptr, nullptr);

    // For large allocations, alignment IS guaranteed
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    EXPECT_EQ(addr % alignment, 0) << "Large allocation not aligned";

    AlignedMemoryPool::instance().deallocate_smart(ptr);
}

// ============= Alignment Validation Tests =============

TEST_F(AlignedMemoryPoolTest, VariousAlignments) {
    std::vector<size_t> alignments = {16, 32, 64, 128, 256, 512};
    size_t bytes = 1024;

    for (size_t alignment : alignments) {
        void* ptr = AlignedMemoryPool::instance().allocate_aligned(bytes, alignment);

        ASSERT_NE(ptr, nullptr) << "Allocation failed for alignment " << alignment;

        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        EXPECT_EQ(addr % alignment, 0) << "Not aligned to " << alignment << " bytes";

        AlignedMemoryPool::instance().deallocate_aligned(ptr);
    }
}

TEST_F(AlignedMemoryPoolTest, InvalidAlignmentReturnsNull) {
    size_t bytes = 1024;
    size_t alignment = 17; // Not a power of 2

    void* ptr = AlignedMemoryPool::instance().allocate_aligned(bytes, alignment);

    // Should return nullptr for invalid alignment
    EXPECT_EQ(ptr, nullptr) << "Should return nullptr for non-power-of-2 alignment";
}

// ============= Memory Correctness Tests =============

TEST_F(AlignedMemoryPoolTest, MemoryZeroing) {
    size_t bytes = 1024;
    size_t alignment = 128;

    void* ptr = AlignedMemoryPool::instance().allocate_aligned(bytes, alignment);
    ASSERT_NE(ptr, nullptr);

    // Zero memory
    cudaMemset(ptr, 0, bytes);
    cudaDeviceSynchronize();

    // Copy to host and verify
    std::vector<char> host_data(bytes);
    cudaMemcpy(host_data.data(), ptr, bytes, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < bytes; ++i) {
        EXPECT_EQ(host_data[i], 0) << "Byte " << i << " is not zero";
    }

    AlignedMemoryPool::instance().deallocate_aligned(ptr);
}

TEST_F(AlignedMemoryPoolTest, MemoryPatternWrite) {
    size_t bytes = 256;
    size_t alignment = 16;

    void* ptr = AlignedMemoryPool::instance().allocate_aligned(bytes, alignment);
    ASSERT_NE(ptr, nullptr);

    // Write pattern from host
    std::vector<float> pattern(bytes / sizeof(float));
    for (size_t i = 0; i < pattern.size(); ++i) {
        pattern[i] = static_cast<float>(i);
    }

    cudaMemcpy(ptr, pattern.data(), bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // Read back and verify
    std::vector<float> result(bytes / sizeof(float));
    cudaMemcpy(result.data(), ptr, bytes, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < pattern.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], pattern[i]) << "Mismatch at index " << i;
    }

    AlignedMemoryPool::instance().deallocate_aligned(ptr);
}

// ============= Statistics Tests =============

TEST_F(AlignedMemoryPoolTest, Statistics) {
    // Allocate multiple blocks
    std::vector<void*> ptrs;
    size_t alignment = 128;

    for (int i = 0; i < 5; ++i) {
        void* ptr = AlignedMemoryPool::instance().allocate_aligned(1024, alignment);
        ptrs.push_back(ptr);
    }

    // Get stats
    std::string stats = AlignedMemoryPool::instance().get_stats();

    // Should mention active allocations
    EXPECT_NE(stats.find("Active allocations: 5"), std::string::npos)
        << "Stats should show 5 active allocations";

    // Cleanup
    for (void* ptr : ptrs) {
        AlignedMemoryPool::instance().deallocate_aligned(ptr);
    }
}

// ============= Performance Hint Tests =============

TEST_F(AlignedMemoryPoolTest, AlignmentOverheadReasonable) {
    size_t bytes = 4096;
    size_t alignment = 128;

    void* ptr = AlignedMemoryPool::instance().allocate_aligned(bytes, alignment);
    ASSERT_NE(ptr, nullptr);

    // The overhead should be less than alignment - 1 bytes
    // We can't directly measure this without accessing internals,
    // but we can verify the allocation succeeds

    std::string stats = AlignedMemoryPool::instance().get_stats();

    // Just verify stats are available
    EXPECT_FALSE(stats.empty());

    AlignedMemoryPool::instance().deallocate_aligned(ptr);
}

// ============= Edge Cases =============

TEST_F(AlignedMemoryPoolTest, ZeroByteAllocation) {
    void* ptr = AlignedMemoryPool::instance().allocate_aligned(0, 128);

    // Should return nullptr for zero-byte allocation
    EXPECT_EQ(ptr, nullptr);
}

TEST_F(AlignedMemoryPoolTest, VeryLargeAllocation) {
    size_t bytes = 1024 * 1024 * 100; // 100 MB
    size_t alignment = 128;

    void* ptr = AlignedMemoryPool::instance().allocate_aligned(bytes, alignment);

    // May or may not succeed depending on available GPU memory
    if (ptr != nullptr) {
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        EXPECT_EQ(addr % alignment, 0) << "Large allocation not aligned";

        AlignedMemoryPool::instance().deallocate_aligned(ptr);
    }
}
