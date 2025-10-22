/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/tensor.hpp"
#include <chrono>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>

using namespace lfs::core;
using namespace std::chrono;

class GatherBenchmarkTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Seed for reproducibility
        Tensor::manual_seed(42);
    }

    struct BenchmarkResult {
        double eager_ms;
        double lazy_ms;
        double speedup;
        bool correctness_pass;
    };

    BenchmarkResult benchmark_gather(size_t data_size, size_t gather_size, int num_iterations = 100) {
        // Create source data
        auto data = Tensor::arange(static_cast<float>(data_size)).to(Device::CUDA);

        // Create random indices
        auto indices = (Tensor::rand({gather_size}, Device::CUDA) * (data_size - 1)).to(DataType::Int32);

        // Warm-up
        for (int i = 0; i < 5; ++i) {
            auto eager_result = data.take(indices);
            cudaDeviceSynchronize();
        }

        // Benchmark eager gather
        auto start_eager = high_resolution_clock::now();
        Tensor eager_result;
        for (int i = 0; i < num_iterations; ++i) {
            eager_result = data.take(indices);
        }
        cudaDeviceSynchronize();
        auto end_eager = high_resolution_clock::now();
        double eager_ms = duration_cast<microseconds>(end_eager - start_eager).count() / 1000.0 / num_iterations;

        // Benchmark lazy gather
        auto start_lazy = high_resolution_clock::now();
        Tensor lazy_result;
        for (int i = 0; i < num_iterations; ++i) {
            lazy_result = data.gather_lazy(indices).eval();
        }
        cudaDeviceSynchronize();
        auto end_lazy = high_resolution_clock::now();
        double lazy_ms = duration_cast<microseconds>(end_lazy - start_lazy).count() / 1000.0 / num_iterations;

        // Verify correctness
        bool correctness_pass = eager_result.all_close(lazy_result, 1e-5, 1e-5);

        double speedup = eager_ms / lazy_ms;

        return {eager_ms, lazy_ms, speedup, correctness_pass};
    }

    struct ChainedBenchmarkResult {
        double eager_chained_ms;
        double lazy_chained_ms;
        double speedup;
        bool correctness_pass;
    };

    ChainedBenchmarkResult benchmark_gather_chained(size_t data_size, size_t gather_size, int num_iterations = 100) {
        // Create source data
        auto data = Tensor::rand({data_size}, Device::CUDA);

        // Create random indices
        auto indices = (Tensor::rand({gather_size}, Device::CUDA) * (data_size - 1)).to(DataType::Int32);

        // Warm-up
        for (int i = 0; i < 5; ++i) {
            auto eager_result = data.take(indices).abs();
            cudaDeviceSynchronize();
        }

        // Benchmark eager: gather + abs (materialize after each)
        auto start_eager = high_resolution_clock::now();
        Tensor eager_result;
        for (int i = 0; i < num_iterations; ++i) {
            eager_result = data.take(indices).abs();
        }
        cudaDeviceSynchronize();
        auto end_eager = high_resolution_clock::now();
        double eager_ms = duration_cast<microseconds>(end_eager - start_eager).count() / 1000.0 / num_iterations;

        // Benchmark lazy: gather_lazy + abs (single materialization)
        auto start_lazy = high_resolution_clock::now();
        Tensor lazy_result;
        for (int i = 0; i < num_iterations; ++i) {
            lazy_result = data.gather_lazy(indices).map(ops::abs_op{}).eval();
        }
        cudaDeviceSynchronize();
        auto end_lazy = high_resolution_clock::now();
        double lazy_ms = duration_cast<microseconds>(end_lazy - start_lazy).count() / 1000.0 / num_iterations;

        // Verify correctness
        bool correctness_pass = eager_result.all_close(lazy_result, 1e-4, 1e-4);

        double speedup = eager_ms / lazy_ms;

        return {eager_ms, lazy_ms, speedup, correctness_pass};
    }
};

TEST_F(GatherBenchmarkTest, SmallGather) {
    std::cout << "\n========================================================================================================\n";
    std::cout << "GATHER BENCHMARK: Small Gather (100 elements from 10K)\n";
    std::cout << "========================================================================================================\n\n";

    auto result = benchmark_gather(10000, 100, 1000);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Eager gather:  " << result.eager_ms << " ms\n";
    std::cout << "Lazy gather:   " << result.lazy_ms << " ms\n";
    std::cout << "Speedup:       " << result.speedup << "x ";

    if (result.speedup >= 1.0) {
        std::cout << "✓ FASTER\n";
    } else {
        std::cout << "⚠ SLOWER\n";
    }

    std::cout << "Correctness:   " << (result.correctness_pass ? "✓ PASS" : "✗ FAIL") << "\n";

    EXPECT_TRUE(result.correctness_pass);
}

TEST_F(GatherBenchmarkTest, MediumGather) {
    std::cout << "\n========================================================================================================\n";
    std::cout << "GATHER BENCHMARK: Medium Gather (10K elements from 1M)\n";
    std::cout << "========================================================================================================\n\n";

    auto result = benchmark_gather(1000000, 10000, 100);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Eager gather:  " << result.eager_ms << " ms\n";
    std::cout << "Lazy gather:   " << result.lazy_ms << " ms\n";
    std::cout << "Speedup:       " << result.speedup << "x ";

    if (result.speedup >= 1.0) {
        std::cout << "✓ FASTER\n";
    } else {
        std::cout << "⚠ SLOWER\n";
    }

    std::cout << "Correctness:   " << (result.correctness_pass ? "✓ PASS" : "✗ FAIL") << "\n";

    EXPECT_TRUE(result.correctness_pass);
}

TEST_F(GatherBenchmarkTest, LargeGather) {
    std::cout << "\n========================================================================================================\n";
    std::cout << "GATHER BENCHMARK: Large Gather (100K elements from 10M)\n";
    std::cout << "========================================================================================================\n\n";

    auto result = benchmark_gather(10000000, 100000, 10);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Eager gather:  " << result.eager_ms << " ms\n";
    std::cout << "Lazy gather:   " << result.lazy_ms << " ms\n";
    std::cout << "Speedup:       " << result.speedup << "x ";

    if (result.speedup >= 1.0) {
        std::cout << "✓ FASTER\n";
    } else {
        std::cout << "⚠ SLOWER\n";
    }

    std::cout << "Correctness:   " << (result.correctness_pass ? "✓ PASS" : "✗ FAIL") << "\n";

    EXPECT_TRUE(result.correctness_pass);
}

TEST_F(GatherBenchmarkTest, ChainedOperations) {
    std::cout << "\n========================================================================================================\n";
    std::cout << "GATHER BENCHMARK: Chained Operations (gather + abs)\n";
    std::cout << "========================================================================================================\n\n";
    std::cout << "Pattern: data.gather(indices).abs()\n";
    std::cout << "Expected: Lazy should fuse gather + abs operation\n\n";

    auto result = benchmark_gather_chained(100000, 10000, 100);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Eager (4 ops):  " << result.eager_chained_ms << " ms\n";
    std::cout << "Lazy (fused):   " << result.lazy_chained_ms << " ms\n";
    std::cout << "Speedup:        " << result.speedup << "x ";

    if (result.speedup >= 1.5) {
        std::cout << "✓ EXCELLENT\n";
    } else if (result.speedup >= 1.0) {
        std::cout << "✓ FASTER\n";
    } else {
        std::cout << "⚠ SLOWER\n";
    }

    std::cout << "Correctness:    " << (result.correctness_pass ? "✓ PASS" : "✗ FAIL") << "\n";

    EXPECT_TRUE(result.correctness_pass);
}

TEST_F(GatherBenchmarkTest, Summary) {
    std::cout << "\n========================================================================================================\n";
    std::cout << "GATHER BENCHMARK SUMMARY\n";
    std::cout << "========================================================================================================\n\n";

    std::cout << "KEY FINDINGS:\n\n";
    std::cout << "Lazy gather provides:\n";
    std::cout << "  1. Expression template composition (gather can be chained with other ops)\n";
    std::cout << "  2. Single materialization point (only evaluate when needed)\n";
    std::cout << "  3. Kernel fusion with subsequent unary operations\n\n";

    std::cout << "EXPECTED PERFORMANCE:\n\n";
    std::cout << "  - Simple gather: Similar to eager (baseline)\n";
    std::cout << "  - Chained gather: 1.2-1.5× speedup (single materialization + fusion)\n\n";

    std::cout << "FUTURE OPTIMIZATIONS:\n\n";
    std::cout << "  - Use thrust::permutation_iterator for true zero-copy gather\n";
    std::cout << "  - Fuse permutation with unary/binary operations at CUDA kernel level\n";
    std::cout << "  - Optimize multi-gather patterns (e.g., K-means centroid updates)\n\n";

    std::cout << "========================================================================================================\n";
}
