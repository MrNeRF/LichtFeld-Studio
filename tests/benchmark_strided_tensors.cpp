/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/tensor.hpp"
#include <chrono>
#include <gtest/gtest.h>
#include <iomanip>
#include <vector>

using namespace lfs::core;
using namespace std::chrono;

// Benchmark helper
template <typename Func>
double benchmark(Func func, int warmup_iters = 10, int bench_iters = 100) {
    // Warmup
    for (int i = 0; i < warmup_iters; ++i) {
        func();
    }

    // Benchmark
    auto start = high_resolution_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
        func();
    }
    auto end = high_resolution_clock::now();

    auto duration_ns = duration_cast<nanoseconds>(end - start).count();
    return static_cast<double>(duration_ns) / bench_iters;
}

class StridedTensorBenchmark : public ::testing::Test {
protected:
    void SetUp() override {
        // Seed for reproducibility
        Tensor::manual_seed(42);
    }

    void print_time(const std::string& label, double time_ns) {
        std::cout << "  " << std::left << std::setw(50) << label << " | ";
        if (time_ns < 1000) {
            std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(2)
                      << time_ns << " ns\n";
        } else if (time_ns < 1000000) {
            std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(2)
                      << time_ns / 1000.0 << " μs\n";
        } else {
            std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(2)
                      << time_ns / 1000000.0 << " ms\n";
        }
    }
};

// ============================================================================
// Test 1: Transpose operations on large image-like tensors
// ============================================================================

TEST_F(StridedTensorBenchmark, Transpose1080pImage) {
    std::cout << "\n=== Transpose 1080p RGBA Image (1920×1080×4) ===\n";

    auto img_cpu = Tensor::rand({1920, 1080, 4}, Device::CPU);
    auto img_gpu = img_cpu.to(Device::CUDA);

    // CPU transpose - should be instant (zero-copy)
    double cpu_time = benchmark([&]() {
        auto t = img_cpu.transpose(0, 1);
        return t.numel(); // Prevent optimization
    });
    print_time("CPU: Transpose (zero-copy)", cpu_time);

    // Verify it's actually zero-copy
    auto transposed_cpu = img_cpu.transpose(0, 1);
    EXPECT_TRUE(transposed_cpu.is_view());
    EXPECT_FALSE(transposed_cpu.owns_memory());
    EXPECT_FALSE(transposed_cpu.is_contiguous());

    // GPU transpose - should be instant (zero-copy)
    double gpu_time = benchmark([&]() {
        auto t = img_gpu.transpose(0, 1);
        return t.numel();
    });
    print_time("GPU: Transpose (zero-copy)", gpu_time);

    auto transposed_gpu = img_gpu.transpose(0, 1);
    EXPECT_TRUE(transposed_gpu.is_view());
    EXPECT_FALSE(transposed_gpu.owns_memory());

    // GPU transpose + materialize - should be much slower
    double gpu_materialize_time = benchmark([&]() {
        auto t = img_gpu.transpose(0, 1).contiguous();
        return t.numel();
    },
                                            5, 20); // Fewer iterations for expensive operation
    print_time("GPU: Transpose + Materialize", gpu_materialize_time);

    // Materialized should be contiguous and own memory
    auto materialized = img_gpu.transpose(0, 1).contiguous();
    EXPECT_TRUE(materialized.is_contiguous());
    EXPECT_TRUE(materialized.owns_memory());

    // Zero-copy should be orders of magnitude faster
    EXPECT_LT(gpu_time, gpu_materialize_time / 1000.0)
        << "Zero-copy transpose should be >1000× faster than materialization";
}

// ============================================================================
// Test 2: Slice operations
// ============================================================================

TEST_F(StridedTensorBenchmark, SliceOperations) {
    std::cout << "\n=== Slice Operations (Large Tensor 10000×1000) ===\n";

    auto large_cpu = Tensor::rand({10000, 1000}, Device::CPU);
    auto large_gpu = large_cpu.to(Device::CUDA);

    // CPU slice - should be instant
    double cpu_time = benchmark([&]() {
        auto s = large_cpu.slice(0, 1000, 2000);
        return s.numel();
    });
    print_time("CPU: Slice rows [1000:2000]", cpu_time);

    auto sliced_cpu = large_cpu.slice(0, 1000, 2000);
    EXPECT_TRUE(sliced_cpu.is_view());
    EXPECT_FALSE(sliced_cpu.owns_memory());
    EXPECT_EQ(sliced_cpu.shape()[0], 1000);
    EXPECT_EQ(sliced_cpu.shape()[1], 1000);
    EXPECT_EQ(sliced_cpu.storage_offset(), 1000 * 1000); // Offset to row 1000

    // GPU slice - should be instant
    double gpu_time = benchmark([&]() {
        auto s = large_gpu.slice(0, 1000, 2000);
        return s.numel();
    });
    print_time("GPU: Slice rows [1000:2000]", gpu_time);

    auto sliced_gpu = large_gpu.slice(0, 1000, 2000);
    EXPECT_TRUE(sliced_gpu.is_view());
    EXPECT_FALSE(sliced_gpu.owns_memory());

    // Slice should be nearly instant (< 1 microsecond)
    EXPECT_LT(gpu_time, 1000.0) << "Slice should take < 1 μs";
}

// ============================================================================
// Test 3: Chained view operations
// ============================================================================

TEST_F(StridedTensorBenchmark, ChainedViewOperations) {
    std::cout << "\n=== Chained View Operations ===\n";

    auto tensor_gpu = Tensor::rand({256, 256, 64}, Device::CUDA);

    // Multiple views chained - all should be instant
    double chain_time = benchmark([&]() {
        auto t = tensor_gpu.transpose(0, 1).slice(1, 100, 200).transpose(1, 2);
        return t.numel();
    });
    print_time("GPU: 3 chained views (transpose+slice+transpose)", chain_time);

    // Verify the chain produces correct views
    auto chained = tensor_gpu.transpose(0, 1).slice(1, 100, 200).transpose(1, 2);
    EXPECT_TRUE(chained.is_view());
    EXPECT_FALSE(chained.owns_memory());
    EXPECT_FALSE(chained.is_contiguous());

    // Same operations with materialization - much slower
    double chain_materialize_time = benchmark([&]() {
        auto t = tensor_gpu.transpose(0, 1).slice(1, 100, 200).transpose(1, 2).contiguous();
        return t.numel();
    },
                                              5, 20);
    print_time("GPU: 3 chained views + materialize", chain_materialize_time);

    auto materialized = tensor_gpu.transpose(0, 1).slice(1, 100, 200).transpose(1, 2).contiguous();
    EXPECT_TRUE(materialized.is_contiguous());
    EXPECT_TRUE(materialized.owns_memory());

    // Chained views should be much faster than materialization
    EXPECT_LT(chain_time, chain_materialize_time / 100.0)
        << "Chained views should be >100× faster than materialization";
}

// ============================================================================
// Test 4: Memory allocation verification
// ============================================================================

TEST_F(StridedTensorBenchmark, MemoryAllocationVerification) {
    std::cout << "\n=== Memory Allocation Verification ===\n";

    auto base_tensor = Tensor::rand({512, 512, 16}, Device::CUDA);

    std::cout << "Base tensor: " << base_tensor.str() << "\n";
    std::cout << "  - owns_memory: " << base_tensor.owns_memory() << "\n";
    std::cout << "  - is_view: " << base_tensor.is_view() << "\n";
    std::cout << "  - is_contiguous: " << base_tensor.is_contiguous() << "\n";

    EXPECT_TRUE(base_tensor.owns_memory());
    EXPECT_FALSE(base_tensor.is_view());
    EXPECT_TRUE(base_tensor.is_contiguous());

    // Transpose should create view without allocating
    auto transposed = base_tensor.transpose(0, 1);
    std::cout << "\nAfter transpose: " << transposed.str() << "\n";
    std::cout << "  - owns_memory: " << transposed.owns_memory() << "\n";
    std::cout << "  - is_view: " << transposed.is_view() << "\n";
    std::cout << "  - is_contiguous: " << transposed.is_contiguous() << "\n";
    std::cout << "  - strides: [";
    for (size_t i = 0; i < transposed.strides().size(); ++i) {
        if (i > 0)
            std::cout << ", ";
        std::cout << transposed.strides()[i];
    }
    std::cout << "]\n";
    std::cout << "  - storage_offset: " << transposed.storage_offset() << "\n";

    EXPECT_FALSE(transposed.owns_memory());
    EXPECT_TRUE(transposed.is_view());
    EXPECT_FALSE(transposed.is_contiguous());
    EXPECT_EQ(transposed.storage_offset(), 0);

    // Slice should create view with storage offset
    auto sliced = base_tensor.slice(0, 100, 200);
    std::cout << "\nAfter slice [100:200]: " << sliced.str() << "\n";
    std::cout << "  - owns_memory: " << sliced.owns_memory() << "\n";
    std::cout << "  - is_view: " << sliced.is_view() << "\n";
    std::cout << "  - is_contiguous: " << sliced.is_contiguous() << "\n";
    std::cout << "  - storage_offset: " << sliced.storage_offset() << "\n";

    EXPECT_FALSE(sliced.owns_memory());
    EXPECT_TRUE(sliced.is_view());
    EXPECT_TRUE(sliced.is_contiguous());                // Row slice is contiguous
    EXPECT_EQ(sliced.storage_offset(), 100 * 512 * 16); // Offset to row 100

    // Contiguous should materialize and own memory
    auto materialized = transposed.contiguous();
    std::cout << "\nAfter contiguous(): " << materialized.str() << "\n";
    std::cout << "  - owns_memory: " << materialized.owns_memory() << "\n";
    std::cout << "  - is_view: " << materialized.is_view() << "\n";
    std::cout << "  - is_contiguous: " << materialized.is_contiguous() << "\n";
    std::cout << "  - strides: [";
    for (size_t i = 0; i < materialized.strides().size(); ++i) {
        if (i > 0)
            std::cout << ", ";
        std::cout << materialized.strides()[i];
    }
    std::cout << "]\n";

    EXPECT_TRUE(materialized.owns_memory());
    EXPECT_FALSE(materialized.is_view());
    EXPECT_TRUE(materialized.is_contiguous());
}

// ============================================================================
// Test 5: Correctness verification
// ============================================================================

TEST_F(StridedTensorBenchmark, CorrectnessVerification) {
    std::cout << "\n=== Correctness Verification ===\n";

    auto test_tensor = Tensor::empty({3, 4}, Device::CPU);
    float* data = test_tensor.ptr<float>();
    for (int i = 0; i < 12; ++i) {
        data[i] = static_cast<float>(i);
    }

    std::cout << "Original tensor (3×4): " << test_tensor.str() << "\n";
    std::cout << "  Data: ";
    for (int i = 0; i < 12; ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << "\n";

    auto transposed_test = test_tensor.transpose(0, 1);
    std::cout << "\nTransposed view (4×3): " << transposed_test.str() << "\n";
    std::cout << "  is_view: " << transposed_test.is_view() << ", is_contiguous: " << transposed_test.is_contiguous() << "\n";

    auto materialized_test = transposed_test.contiguous();
    std::cout << "\nMaterialized (4×3): " << materialized_test.str() << "\n";
    std::cout << "  owns_memory: " << materialized_test.owns_memory() << ", is_contiguous: " << materialized_test.is_contiguous() << "\n";

    // Verify correctness
    auto transposed_vec = transposed_test.to_vector();
    auto materialized_vec = materialized_test.to_vector();

    ASSERT_EQ(transposed_vec.size(), materialized_vec.size())
        << "View and materialized tensor should have same size";

    for (size_t i = 0; i < transposed_vec.size(); ++i) {
        EXPECT_FLOAT_EQ(transposed_vec[i], materialized_vec[i])
            << "Mismatch at index " << i;
    }

    std::cout << "\n✓ Correctness verified: Strided view and materialized tensor match!\n";
}

// ============================================================================
// Test 6: Performance comparison summary
// ============================================================================

TEST_F(StridedTensorBenchmark, PerformanceSummary) {
    std::cout << "\n╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║              STRIDED TENSOR PERFORMANCE SUMMARY                  ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n\n";

    auto img = Tensor::rand({1920, 1080, 4}, Device::CUDA);
    auto large = Tensor::rand({10000, 1000}, Device::CUDA);
    auto vol = Tensor::rand({256, 256, 64}, Device::CUDA);

    double transpose_time = benchmark([&]() { return img.transpose(0, 1).numel(); });
    double slice_time = benchmark([&]() { return large.slice(0, 1000, 2000).numel(); });
    double chain_time = benchmark([&]() {
        return vol.transpose(0, 1).slice(1, 100, 200).transpose(1, 2).numel();
    });

    std::cout << "Zero-Copy Operations Performance:\n";
    std::cout << "  • Transpose (1080p):     " << std::fixed << std::setprecision(2)
              << transpose_time << " ns (INSTANT)\n";
    std::cout << "  • Slice (large tensor):  " << std::fixed << std::setprecision(2)
              << slice_time << " ns (INSTANT)\n";
    std::cout << "  • Chained views (3 ops): " << std::fixed << std::setprecision(2)
              << chain_time << " ns (INSTANT)\n\n";

    std::cout << "Expected Use Cases:\n";
    std::cout << "  • Neural network training: Frequent transpose for weight matrices\n";
    std::cout << "  • Batch processing: Slice tensors without copying\n";
    std::cout << "  • Data augmentation: Multiple view operations without memory overhead\n";
    std::cout << "  • Gaussian splatting: Transpose point cloud data for rendering\n\n";

    std::cout << "Memory Savings:\n";
    std::cout << "  • Each view operation: 0 bytes allocated (vs. full tensor copy)\n";
    std::cout << "  • Materialize only when needed for compute kernels\n\n";

    // All operations should be fast
    EXPECT_LT(transpose_time, 10000.0) << "Transpose should be < 10 μs";
    EXPECT_LT(slice_time, 10000.0) << "Slice should be < 10 μs";
    EXPECT_LT(chain_time, 10000.0) << "Chained ops should be < 10 μs";
}
