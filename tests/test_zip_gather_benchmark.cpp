/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/tensor.hpp"
#include <chrono>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>

using namespace lfs::core;
using namespace std::chrono;

class ZipGatherBenchmarkTest : public ::testing::Test {
protected:
    void SetUp() override {
        Tensor::manual_seed(42);
    }

    struct BenchmarkResult {
        double separate_ms;
        double zipped_ms;
        double speedup;
        bool correctness_pass;
    };

    // Benchmark gathering from 2 tensors
    BenchmarkResult benchmark_zip_gather_2(size_t data_size, size_t gather_size, int num_iterations = 100) {
        // Create source data tensors (e.g., positions and colors)
        auto data1 = Tensor::rand({data_size}, Device::CUDA);
        auto data2 = Tensor::rand({data_size}, Device::CUDA);

        // Create random indices
        auto indices = (Tensor::rand({gather_size}, Device::CUDA) * (data_size - 1)).to(DataType::Int32);

        // Warm-up
        for (int i = 0; i < 5; ++i) {
            auto r1 = data1.take(indices);
            auto r2 = data2.take(indices);
            cudaDeviceSynchronize();
        }

        // Benchmark separate gather (2 kernel launches)
        auto start_separate = high_resolution_clock::now();
        Tensor result1_sep, result2_sep;
        for (int i = 0; i < num_iterations; ++i) {
            result1_sep = data1.take(indices);
            result2_sep = data2.take(indices);
        }
        cudaDeviceSynchronize();
        auto end_separate = high_resolution_clock::now();
        double separate_ms = duration_cast<microseconds>(end_separate - start_separate).count() / 1000.0 / num_iterations;

        // Benchmark zipped gather (1 fused kernel)
        auto start_zipped = high_resolution_clock::now();
        Tensor result1_zip, result2_zip;
        for (int i = 0; i < num_iterations; ++i) {
            // Allocate outputs
            result1_zip = Tensor::empty({gather_size}, Device::CUDA, DataType::Float32);
            result2_zip = Tensor::empty({gather_size}, Device::CUDA, DataType::Float32);

            // Single fused gather
            tensor_ops::launch_zip_gather_2(
                data1.ptr<float>(), data2.ptr<float>(),
                indices.ptr<int>(),
                result1_zip.ptr<float>(), result2_zip.ptr<float>(),
                data_size, gather_size,
                1, 1, // strides
                nullptr);
        }
        cudaDeviceSynchronize();
        auto end_zipped = high_resolution_clock::now();
        double zipped_ms = duration_cast<microseconds>(end_zipped - start_zipped).count() / 1000.0 / num_iterations;

        // Verify correctness
        bool correctness_pass = result1_sep.all_close(result1_zip, 1e-5, 1e-5) &&
                                result2_sep.all_close(result2_zip, 1e-5, 1e-5);

        double speedup = separate_ms / zipped_ms;

        return {separate_ms, zipped_ms, speedup, correctness_pass};
    }

    // Benchmark gathering from 3 tensors
    BenchmarkResult benchmark_zip_gather_3(size_t data_size, size_t gather_size, int num_iterations = 100) {
        // Create source data tensors (e.g., positions, colors, scales)
        auto data1 = Tensor::rand({data_size}, Device::CUDA);
        auto data2 = Tensor::rand({data_size}, Device::CUDA);
        auto data3 = Tensor::rand({data_size}, Device::CUDA);

        // Create random indices
        auto indices = (Tensor::rand({gather_size}, Device::CUDA) * (data_size - 1)).to(DataType::Int32);

        // Warm-up
        for (int i = 0; i < 5; ++i) {
            auto r1 = data1.take(indices);
            auto r2 = data2.take(indices);
            auto r3 = data3.take(indices);
            cudaDeviceSynchronize();
        }

        // Benchmark separate gather (3 kernel launches)
        auto start_separate = high_resolution_clock::now();
        Tensor result1_sep, result2_sep, result3_sep;
        for (int i = 0; i < num_iterations; ++i) {
            result1_sep = data1.take(indices);
            result2_sep = data2.take(indices);
            result3_sep = data3.take(indices);
        }
        cudaDeviceSynchronize();
        auto end_separate = high_resolution_clock::now();
        double separate_ms = duration_cast<microseconds>(end_separate - start_separate).count() / 1000.0 / num_iterations;

        // Benchmark zipped gather (1 fused kernel)
        auto start_zipped = high_resolution_clock::now();
        Tensor result1_zip, result2_zip, result3_zip;
        for (int i = 0; i < num_iterations; ++i) {
            // Allocate outputs
            result1_zip = Tensor::empty({gather_size}, Device::CUDA, DataType::Float32);
            result2_zip = Tensor::empty({gather_size}, Device::CUDA, DataType::Float32);
            result3_zip = Tensor::empty({gather_size}, Device::CUDA, DataType::Float32);

            // Single fused gather
            tensor_ops::launch_zip_gather_3(
                data1.ptr<float>(), data2.ptr<float>(), data3.ptr<float>(),
                indices.ptr<int>(),
                result1_zip.ptr<float>(), result2_zip.ptr<float>(), result3_zip.ptr<float>(),
                data_size, gather_size,
                1, 1, 1, // strides
                nullptr);
        }
        cudaDeviceSynchronize();
        auto end_zipped = high_resolution_clock::now();
        double zipped_ms = duration_cast<microseconds>(end_zipped - start_zipped).count() / 1000.0 / num_iterations;

        // Verify correctness
        bool correctness_pass = result1_sep.all_close(result1_zip, 1e-5, 1e-5) &&
                                result2_sep.all_close(result2_zip, 1e-5, 1e-5) &&
                                result3_sep.all_close(result3_zip, 1e-5, 1e-5);

        double speedup = separate_ms / zipped_ms;

        return {separate_ms, zipped_ms, speedup, correctness_pass};
    }
};

TEST_F(ZipGatherBenchmarkTest, TwoTensors_Small) {
    std::cout << "\n========================================================================================================\n";
    std::cout << "ZIP GATHER BENCHMARK: 2 Tensors - Small (100 elements from 10K)\n";
    std::cout << "========================================================================================================\n\n";

    auto result = benchmark_zip_gather_2(10000, 100, 1000);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Separate gather (2 kernels):  " << result.separate_ms << " ms\n";
    std::cout << "Zipped gather (1 kernel):      " << result.zipped_ms << " ms\n";
    std::cout << "Speedup:                       " << result.speedup << "x ";

    if (result.speedup >= 1.5) {
        std::cout << "✓ EXCELLENT\n";
    } else if (result.speedup >= 1.2) {
        std::cout << "✓ FASTER\n";
    } else if (result.speedup >= 1.0) {
        std::cout << "~ SIMILAR\n";
    } else {
        std::cout << "⚠ SLOWER\n";
    }

    std::cout << "Correctness:                   " << (result.correctness_pass ? "✓ PASS" : "✗ FAIL") << "\n";

    EXPECT_TRUE(result.correctness_pass);
}

TEST_F(ZipGatherBenchmarkTest, TwoTensors_Medium) {
    std::cout << "\n========================================================================================================\n";
    std::cout << "ZIP GATHER BENCHMARK: 2 Tensors - Medium (10K elements from 1M)\n";
    std::cout << "========================================================================================================\n\n";

    auto result = benchmark_zip_gather_2(1000000, 10000, 100);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Separate gather (2 kernels):  " << result.separate_ms << " ms\n";
    std::cout << "Zipped gather (1 kernel):      " << result.zipped_ms << " ms\n";
    std::cout << "Speedup:                       " << result.speedup << "x ";

    if (result.speedup >= 1.5) {
        std::cout << "✓ EXCELLENT\n";
    } else if (result.speedup >= 1.2) {
        std::cout << "✓ FASTER\n";
    } else if (result.speedup >= 1.0) {
        std::cout << "~ SIMILAR\n";
    } else {
        std::cout << "⚠ SLOWER\n";
    }

    std::cout << "Correctness:                   " << (result.correctness_pass ? "✓ PASS" : "✗ FAIL") << "\n";

    EXPECT_TRUE(result.correctness_pass);
}

TEST_F(ZipGatherBenchmarkTest, ThreeTensors_Medium) {
    std::cout << "\n========================================================================================================\n";
    std::cout << "ZIP GATHER BENCHMARK: 3 Tensors - Medium (10K elements from 1M)\n";
    std::cout << "========================================================================================================\n\n";

    auto result = benchmark_zip_gather_3(1000000, 10000, 100);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Separate gather (3 kernels):  " << result.separate_ms << " ms\n";
    std::cout << "Zipped gather (1 kernel):      " << result.zipped_ms << " ms\n";
    std::cout << "Speedup:                       " << result.speedup << "x ";

    if (result.speedup >= 2.0) {
        std::cout << "✓ EXCELLENT\n";
    } else if (result.speedup >= 1.5) {
        std::cout << "✓ FASTER\n";
    } else if (result.speedup >= 1.0) {
        std::cout << "~ SIMILAR\n";
    } else {
        std::cout << "⚠ SLOWER\n";
    }

    std::cout << "Correctness:                   " << (result.correctness_pass ? "✓ PASS" : "✗ FAIL") << "\n";

    EXPECT_TRUE(result.correctness_pass);
}

TEST_F(ZipGatherBenchmarkTest, Summary) {
    std::cout << "\n========================================================================================================\n";
    std::cout << "ZIP GATHER BENCHMARK SUMMARY\n";
    std::cout << "========================================================================================================\n\n";

    std::cout << "KEY FINDINGS:\n\n";
    std::cout << "Zip gather provides:\n";
    std::cout << "  1. Single kernel launch vs multiple (reduces overhead)\n";
    std::cout << "  2. Single pass through indices (better cache utilization)\n";
    std::cout << "  3. Coalesced memory access for all tensors\n";
    std::cout << "  4. Reduced memory bandwidth (fetch indices once)\n\n";

    std::cout << "EXPECTED PERFORMANCE:\n\n";
    std::cout << "  - 2 tensors: 1.5-2× speedup vs separate gather\n";
    std::cout << "  - 3 tensors: 2-2.5× speedup vs separate gather\n\n";

    std::cout << "USE CASES:\n\n";
    std::cout << "  - Gather positions AND colors from point cloud\n";
    std::cout << "  - Gather means, scales, rotations for Gaussian subsets\n";
    std::cout << "  - K-means centroid gathering (position + features)\n";
    std::cout << "  - Any multi-field data structure indexing\n\n";

    std::cout << "========================================================================================================\n";
}
