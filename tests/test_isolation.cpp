/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Standalone timing test - compile with our build system
#include "core/tensor.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include <chrono>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
#include <torch/torch.h>

using namespace lfs::core;

TEST(IsolationTest, ReductionAfterTorchInit) {
    // Test that runs after PyTorch tensor creation to see interference
    // This simulates the benchmark conditions

    // Create a PyTorch tensor first (this initializes PyTorch's CUDA context)
    auto torch_tensor = torch::rand({1024, 1024}, torch::kCUDA);
    cudaDeviceSynchronize();

    const int iterations = 100;

    // Now create and test our tensor
    auto tensor = Tensor::rand({1024, 1024}, Device::CUDA);
    cudaDeviceSynchronize();

    // Warmup
    for (int i = 0; i < 20; i++) {
        auto result = tensor.sum({1}, false);
    }
    cudaDeviceSynchronize();

    // Benchmark with sync per iteration
    double total_us = 0.0;
    for (int i = 0; i < iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = tensor.sum({1}, false);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        total_us += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    std::cout << "\n[AFTER TORCH INIT] Sum dim=1 (sync each): " << (total_us / iterations) << " us/iter" << std::endl;
}

TEST(IsolationTest, BothTensorsAlive) {
    // Test with BOTH tensors existing simultaneously (like the benchmark)
    const int iterations = 100;

    // Create BOTH tensors first (like ReductionBenchmarkTest does)
    auto tensor_custom = Tensor::rand({1024, 1024}, Device::CUDA);
    auto tensor_torch = torch::rand({1024, 1024}, torch::kCUDA);
    cudaDeviceSynchronize();

    // Warmup
    for (int i = 0; i < 20; i++) {
        auto result = tensor_custom.sum({1}, false);
    }
    cudaDeviceSynchronize();

    // Benchmark with sync per iteration
    double total_us = 0.0;
    for (int i = 0; i < iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = tensor_custom.sum({1}, false);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        total_us += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    std::cout << "\n[BOTH ALIVE] Sum dim=1 (sync each): " << (total_us / iterations) << " us/iter" << std::endl;
}

TEST(IsolationTest, ReductionDim0Timing) {
    const int iterations = 100;

    // Create test tensor
    auto tensor = Tensor::rand({1024, 1024}, Device::CUDA);
    cudaDeviceSynchronize();

    // Warmup
    for (int i = 0; i < 20; i++) {
        auto result = tensor.sum({0}, false); // Reduce along dim 0
    }
    cudaDeviceSynchronize();

    // Benchmark with sync per iteration
    double total_us = 0.0;
    for (int i = 0; i < iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = tensor.sum({0}, false); // Reduce along dim 0
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        total_us += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    std::cout << "\nSum dim=0 (1024x1024, sync each): " << (total_us / iterations) << " us/iter" << std::endl;
}

TEST(IsolationTest, Dim0CorrectnessCheck) {
    // Simple correctness test for dim=0 reduction with transpose optimization
    auto tensor = Tensor::ones({100, 100, 10}, Device::CUDA);

    // Sum along dim 0 should give [100, 10] filled with 100.0
    auto result = tensor.sum({0}, false);

    std::cout << "\nDim0 correctness test:" << std::endl;
    std::cout << "  Input shape: [100, 100, 10]" << std::endl;
    std::cout << "  Result shape: [" << result.shape()[0] << ", " << result.shape()[1] << "]" << std::endl;

    auto vec = result.to_vector();
    std::cout << "  First 5 values (should all be 100.0): ";
    for (int i = 0; i < 5; i++) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;

    // Verify values
    bool correct = true;
    for (size_t i = 0; i < vec.size(); i++) {
        if (std::abs(vec[i] - 100.0f) > 0.01f) {
            std::cout << "  ERROR at index " << i << ": " << vec[i] << " != 100.0" << std::endl;
            correct = false;
            break;
        }
    }
    EXPECT_TRUE(correct);
}

TEST(IsolationTest, PermuteContiguousDebug) {
    // Manually trace the steps to find where doubling occurs
    auto t = Tensor::ones({100, 100, 10}, Device::CUDA);
    cudaDeviceSynchronize();

    std::cout << "\n=== Step 1: Original tensor ===" << std::endl;
    std::cout << "Shape: [" << t.shape()[0] << ", " << t.shape()[1] << ", " << t.shape()[2] << "]" << std::endl;
    std::cout << "Strides: [" << t.strides()[0] << ", " << t.strides()[1] << ", " << t.strides()[2] << "]" << std::endl;
    auto v1 = t.to_vector();
    float sum1 = 0;
    for (auto x : v1)
        sum1 += x;
    std::cout << "Sum of all values: " << sum1 << " (expected: 100000)" << std::endl;

    // Step 2: Permute (zero-copy view)
    std::vector<int> perm = {1, 2, 0};
    auto permuted = t.permute(perm);
    cudaDeviceSynchronize();

    std::cout << "\n=== Step 2: After permute({1,2,0}) ===" << std::endl;
    std::cout << "Shape: [" << permuted.shape()[0] << ", " << permuted.shape()[1] << ", " << permuted.shape()[2] << "]" << std::endl;
    std::cout << "Strides: [" << permuted.strides()[0] << ", " << permuted.strides()[1] << ", " << permuted.strides()[2] << "]" << std::endl;
    std::cout << "is_contiguous: " << (permuted.is_contiguous() ? "true" : "false") << std::endl;
    auto v2 = permuted.to_vector();
    float sum2 = 0;
    for (auto x : v2)
        sum2 += x;
    std::cout << "Sum of all values: " << sum2 << " (expected: 100000)" << std::endl;

    // Step 3: Contiguous (actual data copy)
    auto contig = permuted.contiguous();
    cudaDeviceSynchronize();

    std::cout << "\n=== Step 3: After contiguous() ===" << std::endl;
    std::cout << "Shape: [" << contig.shape()[0] << ", " << contig.shape()[1] << ", " << contig.shape()[2] << "]" << std::endl;
    std::cout << "Strides: [" << contig.strides()[0] << ", " << contig.strides()[1] << ", " << contig.strides()[2] << "]" << std::endl;
    std::cout << "is_contiguous: " << (contig.is_contiguous() ? "true" : "false") << std::endl;
    auto v3 = contig.to_vector();
    float sum3 = 0;
    for (auto x : v3)
        sum3 += x;
    std::cout << "Sum of all values: " << sum3 << " (expected: 100000)" << std::endl;
    std::cout << "numel: " << contig.numel() << std::endl;

    // Step 4: Sum along last dimension
    auto reduced = contig.sum({2}, false);
    cudaDeviceSynchronize();

    std::cout << "\n=== Step 4: After sum({2}) ===" << std::endl;
    std::cout << "Shape: [" << reduced.shape()[0] << ", " << reduced.shape()[1] << "]" << std::endl;
    auto v4 = reduced.to_vector();
    std::cout << "First 5 values (expected 100.0): ";
    for (int i = 0; i < 5 && i < (int)v4.size(); i++) {
        std::cout << v4[i] << " ";
    }
    std::cout << std::endl;
    float sum4 = 0;
    for (auto x : v4)
        sum4 += x;
    std::cout << "Sum of all output values: " << sum4 << " (expected: 100000)" << std::endl;

    EXPECT_NEAR(v4[0], 100.0f, 0.1f);
}

TEST(IsolationTest, ReductionTiming) {
    const int iterations = 100;

    // Create test tensor
    auto tensor = Tensor::rand({1024, 1024}, Device::CUDA);
    cudaDeviceSynchronize();

    // Warmup
    for (int i = 0; i < 20; i++) {
        auto result = tensor.sum({1}, false);
    }
    cudaDeviceSynchronize();

    // Benchmark with sync per iteration
    double total_us = 0.0;
    for (int i = 0; i < iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = tensor.sum({1}, false);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        total_us += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    std::cout << "\nSum dim=1 (1024x1024, sync each): " << (total_us / iterations) << " us/iter" << std::endl;

    // Benchmark batched
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        auto result = tensor.sum({1}, false);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    total_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Sum dim=1 (1024x1024, batched): " << (total_us / iterations) << " us/iter" << std::endl;

    // GPU timing
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    cudaEventRecord(start_event);
    for (int i = 0; i < iterations; i++) {
        auto result = tensor.sum({1}, false);
    }
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);

    float ms;
    cudaEventElapsedTime(&ms, start_event, stop_event);
    std::cout << "Sum dim=1 (1024x1024, GPU time): " << (ms / iterations * 1000) << " us/iter" << std::endl;
}
