/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "training/kernels/mcmc_kernels.hpp"
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <numeric>
#include <unordered_map>

using namespace lfs::core;
using namespace lfs::training::mcmc;

class MCMCHistogramTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure CUDA is available
        int device_count;
        cudaGetDeviceCount(&device_count);
        ASSERT_GT(device_count, 0) << "No CUDA devices available";
    }

    // Helper: upload int64 vector to GPU
    Tensor upload_int64_vector(const std::vector<int64_t>& data) {
        std::vector<int> data_int(data.begin(), data.end());
        Tensor tensor = Tensor::from_vector(data_int, TensorShape({data.size()}), Device::CUDA);
        return tensor.to(DataType::Int64);
    }

    // Helper: download int32 tensor from GPU
    std::vector<int32_t> download_int32_tensor(const Tensor& tensor) {
        std::vector<float> result_float = tensor.to(DataType::Float32).to_vector();
        return std::vector<int32_t>(result_float.begin(), result_float.end());
    }

    // CPU reference implementation for verification
    std::vector<int32_t> count_occurrences_cpu(const std::vector<int64_t>& indices) {
        std::unordered_map<int64_t, int32_t> count_map;

        // Count occurrences
        for (int64_t idx : indices) {
            count_map[idx]++;
        }

        // Build output array
        std::vector<int32_t> counts(indices.size());
        for (size_t i = 0; i < indices.size(); i++) {
            counts[i] = count_map[indices[i]];
        }

        return counts;
    }

    // Performance measurement helper
    template <typename Func>
    double measure_time_ms(Func&& func, int warmup_iters = 2, int measure_iters = 5) {
        // Warmup
        for (int i = 0; i < warmup_iters; i++) {
            func();
        }
        cudaDeviceSynchronize();

        // Measure
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < measure_iters; i++) {
            func();
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        return elapsed_ms / measure_iters;
    }
};

// Test 1: Correctness with small input
TEST_F(MCMCHistogramTest, SmallInput_Correctness) {
    const size_t n = 100;

    // Create test indices with some duplicates
    std::vector<int64_t> h_indices = {
        5, 10, 5, 20, 10, 5, 30, 20, 10, 5, // indices 5,10,20,30 with various counts
        15, 15, 25, 25, 25, 35, 40, 40, 40, 40};

    // Upload to GPU
    Tensor indices = upload_int64_vector(h_indices);
    Tensor output_counts = Tensor::empty({h_indices.size()}, Device::CUDA, DataType::Int32);

    // Run optimized kernel
    launch_count_occurrences_fast(
        indices.ptr<int64_t>(),
        output_counts.ptr<int32_t>(),
        h_indices.size(),
        nullptr);

    // Download results
    std::vector<int32_t> result = download_int32_tensor(output_counts);

    // Compute expected counts
    std::vector<int32_t> expected = count_occurrences_cpu(h_indices);

    // Verify
    ASSERT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < result.size(); i++) {
        EXPECT_EQ(result[i], expected[i])
            << "Mismatch at position " << i
            << " for index " << h_indices[i];
    }
}

// Test 2: Correctness with realistic MCMC scenario (500k samples from 10M Gaussians)
TEST_F(MCMCHistogramTest, RealisticMCMC_Correctness) {
    const size_t n_samples = 500000;    // 500k samples
    const int64_t max_index = 10000000; // 10M Gaussians

    // Generate realistic sampled indices (multinomial sampling pattern)
    std::vector<int64_t> h_indices(n_samples);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int64_t> dist(0, max_index - 1);

    for (size_t i = 0; i < n_samples; i++) {
        h_indices[i] = dist(rng);
    }

    // Upload to GPU
    Tensor indices = upload_int64_vector(h_indices);
    Tensor output_counts = Tensor::empty({n_samples}, Device::CUDA, DataType::Int32);

    // Run optimized kernel
    launch_count_occurrences_fast(
        indices.ptr<int64_t>(),
        output_counts.ptr<int32_t>(),
        n_samples,
        nullptr);

    // Download results
    std::vector<int32_t> result = download_int32_tensor(output_counts);

    // Compute expected counts (sample verification - check a few random indices)
    std::unordered_map<int64_t, int32_t> count_map;
    for (int64_t idx : h_indices) {
        count_map[idx]++;
    }

    // Verify random samples
    std::uniform_int_distribution<size_t> sample_dist(0, n_samples - 1);
    for (int i = 0; i < 1000; i++) {
        size_t pos = sample_dist(rng);
        int64_t idx = h_indices[pos];
        EXPECT_EQ(result[pos], count_map[idx])
            << "Mismatch at position " << pos << " for index " << idx;
    }
}

// Test 3: Performance comparison - OLD vs NEW approach
TEST_F(MCMCHistogramTest, Performance_RealisticScale) {
    const size_t n_samples = 500000;
    const int64_t N = 10000000; // 10M Gaussians

    // Generate realistic indices
    std::vector<int64_t> h_indices(n_samples);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int64_t> dist(0, N - 1);

    for (size_t i = 0; i < n_samples; i++) {
        h_indices[i] = dist(rng);
    }

    Tensor indices = upload_int64_vector(h_indices);

    std::cout << "\n========================================\n";
    std::cout << "PERFORMANCE COMPARISON\n";
    std::cout << "n_samples = " << n_samples << ", N = " << N << "\n";
    std::cout << "========================================\n";

    // OLD APPROACH (emulate the slow index_add_ + index_select)
    double old_time_ms = measure_time_ms([&]() {
        // Step 1: Create massive N-sized array (40MB allocation)
        Tensor ratios = Tensor::ones({N}, Device::CUDA, DataType::Int32);

        // Step 2: index_add_ (slow scattered writes)
        ratios = ratios.index_add_(0, indices, Tensor::ones({n_samples}, Device::CUDA, DataType::Int32));

        // Step 3: index_select (gather back)
        Tensor result = ratios.index_select(0, indices);
    },
                                         1, 3); // Fewer iterations since it's slow

    // NEW APPROACH (optimized)
    Tensor output_counts = Tensor::empty({n_samples}, Device::CUDA, DataType::Int32);
    double new_time_ms = measure_time_ms([&]() {
        launch_count_occurrences_fast(
            indices.ptr<int64_t>(),
            output_counts.ptr<int32_t>(),
            n_samples,
            nullptr);
    });

    std::cout << "OLD approach (index_add_ + index_select): " << old_time_ms << " ms\n";
    std::cout << "NEW approach (optimized histogram):       " << new_time_ms << " ms\n";
    std::cout << "SPEEDUP:                                   " << (old_time_ms / new_time_ms) << "x\n";
    std::cout << "========================================\n\n";

    // Verify speedup is significant
    EXPECT_LT(new_time_ms, old_time_ms * 0.1)
        << "Expected at least 10x speedup, got " << (old_time_ms / new_time_ms) << "x";
}

// Test 4: Edge cases
TEST_F(MCMCHistogramTest, EdgeCases) {
    // All same index
    {
        std::vector<int64_t> h_indices(1000, 42);
        Tensor indices = upload_int64_vector(h_indices);
        Tensor output = Tensor::empty({1000}, Device::CUDA, DataType::Int32);

        launch_count_occurrences_fast(indices.ptr<int64_t>(), output.ptr<int32_t>(), 1000, nullptr);

        std::vector<int32_t> result = download_int32_tensor(output);
        for (int32_t count : result) {
            EXPECT_EQ(count, 1000) << "All indices should have count 1000";
        }
    }

    // All unique indices
    {
        std::vector<int64_t> h_indices(1000);
        std::iota(h_indices.begin(), h_indices.end(), 0); // 0, 1, 2, ..., 999

        Tensor indices = upload_int64_vector(h_indices);
        Tensor output = Tensor::empty({1000}, Device::CUDA, DataType::Int32);

        launch_count_occurrences_fast(indices.ptr<int64_t>(), output.ptr<int32_t>(), 1000, nullptr);

        std::vector<int32_t> result = download_int32_tensor(output);
        for (int32_t count : result) {
            EXPECT_EQ(count, 1) << "All unique indices should have count 1";
        }
    }
}

// Test 5: Stress test with very large input (10M samples)
TEST_F(MCMCHistogramTest, StressTest_10MSamples) {
    const size_t n_samples = 10000000; // 10M samples
    const int64_t max_index = 5000000; // 5M Gaussians

    std::cout << "\n========================================\n";
    std::cout << "STRESS TEST: 10M samples from 5M Gaussians\n";
    std::cout << "========================================\n";

    // Generate indices
    std::vector<int64_t> h_indices(n_samples);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int64_t> dist(0, max_index - 1);

    for (size_t i = 0; i < n_samples; i++) {
        h_indices[i] = dist(rng);
    }

    Tensor indices = upload_int64_vector(h_indices);
    Tensor output_counts = Tensor::empty({n_samples}, Device::CUDA, DataType::Int32);

    // Measure performance
    auto start = std::chrono::high_resolution_clock::now();

    launch_count_occurrences_fast(
        indices.ptr<int64_t>(),
        output_counts.ptr<int32_t>(),
        n_samples,
        nullptr);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Processing time: " << elapsed_ms << " ms\n";
    std::cout << "Throughput: " << (n_samples / elapsed_ms / 1000.0) << " M elements/sec\n";
    std::cout << "========================================\n\n";

    // Verify a sample of results
    std::vector<int32_t> result = download_int32_tensor(output_counts);
    std::unordered_map<int64_t, int32_t> count_map;
    for (int64_t idx : h_indices) {
        count_map[idx]++;
    }

    std::uniform_int_distribution<size_t> sample_dist(0, n_samples - 1);
    for (int i = 0; i < 1000; i++) {
        size_t pos = sample_dist(rng);
        EXPECT_EQ(result[pos], count_map[h_indices[pos]]);
    }

    // Performance expectation: should complete in under 50ms
    EXPECT_LT(elapsed_ms, 50.0) << "Expected processing time < 50ms for 10M samples";
}

// Test 6: Integration test - full MCMC densification pattern
TEST_F(MCMCHistogramTest, Integration_MCMCDensificationPattern) {
    const size_t N = 2000000;     // 2M Gaussians
    const size_t n_dead = 100000; // 100k dead gaussians to relocate

    std::cout << "\n========================================\n";
    std::cout << "INTEGRATION TEST: MCMC Densification\n";
    std::cout << "N = " << N << ", n_dead = " << n_dead << "\n";
    std::cout << "========================================\n";

    // Simulate multinomial sampling pattern
    std::vector<int64_t> h_sampled_idxs(n_dead);
    std::mt19937 rng(42);

    // Realistic pattern: most samples from a smaller subset (power-law distribution)
    std::uniform_int_distribution<int64_t> dist(0, N / 2); // Sample from first half more
    for (size_t i = 0; i < n_dead; i++) {
        h_sampled_idxs[i] = dist(rng);
    }

    Tensor sampled_idxs = upload_int64_vector(h_sampled_idxs);
    Tensor ratios = Tensor::empty({n_dead}, Device::CUDA, DataType::Int32);

    // Measure full pattern (count + clamp)
    auto start = std::chrono::high_resolution_clock::now();

    launch_count_occurrences_fast(
        sampled_idxs.ptr<int64_t>(),
        ratios.ptr<int32_t>(),
        n_dead,
        nullptr);

    // Clamp to [1, 51] (as in real code)
    ratios = ratios.clamp(1, 51);

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Total time (count + clamp): " << elapsed_ms << " ms\n";
    std::cout << "========================================\n\n";

    // Verify counts are in valid range [1, 51]
    std::vector<int32_t> result = download_int32_tensor(ratios);
    for (size_t i = 0; i < result.size(); i++) {
        EXPECT_GE(result[i], 1) << "Count must be >= 1";
        EXPECT_LE(result[i], 51) << "Count must be <= 51 after clamp";
    }

    // Expected: should complete in under 10ms
    EXPECT_LT(elapsed_ms, 10.0) << "Expected total time < 10ms";
}
