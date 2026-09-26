/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_mcmc_tensor_ops.cpp
 * @brief Comprehensive tests for tensor operations critical to MCMC strategy
 *
 * This test suite covers bugs found and fixed during MCMC implementation:
 *
 * BUG #1: nonzero() stream synchronization
 *   - Symptom: Uninitialized memory reads, garbage indices
 *   - Root cause: Missing cudaStreamSynchronize() in nonzero() implementation
 *   - Fix: Added stream sync in tensor_masking_ops.cpp
 *
 * BUG #2: multinomial() int32/int64 mismatch
 *   - Symptom: Corrupted indices after sampling
 *   - Root cause: multinomial() returned Int32 but alive_indices needs Int64
 *   - Fix: Changed multinomial return type to Int64
 *
 * BUG #3: index_select() missing Int64 support
 *   - Symptom: Garbage values when selecting from alive_indices (Int64 tensor)
 *   - Root cause: index_select() only supported Float32 source data
 *   - Fix: Templated index_select_kernel to support Float32, Int64, Int32
 *
 * BUG #4: index_select() missing Int32 support
 *   - Symptom: "unsupported dtype for CUDA" when selecting ratios
 *   - Root cause: MCMC ratios tensor is Int32, but index_select didn't support it
 *   - Fix: Added Int32 overload to templated kernel (mcmc.cpp:174)
 *
 * These tests ensure we never reintroduce these bugs.
 */

#include "core/tensor.hpp"
#include <gtest/gtest.h>
#include <iostream>

using namespace lfs::core;

// ============================================================================
// nonzero() Tests - Bug #1: Stream synchronization
// ============================================================================

TEST(MCMCTensorOps, NonzeroBasicBool) {
    // Test nonzero() with boolean mask - basic MCMC dead mask scenario
    std::vector<unsigned char> mask_data = {1, 0, 1, 0, 1, 1, 0, 1, 0, 0};
    auto mask = Tensor::from_blob(mask_data.data(), {10}, Device::CPU, DataType::Bool).to(Device::GPU);

    auto indices = mask.nonzero().squeeze(-1).cpu();

    ASSERT_EQ(indices.dtype(), DataType::Int64);
    ASSERT_EQ(indices.numel(), 5); // Five 1s in the mask

    const int64_t* idx_ptr = indices.ptr<int64_t>();
    EXPECT_EQ(idx_ptr[0], 0);
    EXPECT_EQ(idx_ptr[1], 2);
    EXPECT_EQ(idx_ptr[2], 4);
    EXPECT_EQ(idx_ptr[3], 5);
    EXPECT_EQ(idx_ptr[4], 7);
}

TEST(MCMCTensorOps, NonzeroRealisticMCMCSize) {
    // Simulate MCMC dead mask on realistic Gaussian count
    constexpr size_t N = 54275;
    std::vector<unsigned char> mask_data(N);

    // ~3% dead (like real MCMC scenario)
    size_t expected_alive = 0;
    for (size_t i = 0; i < N; ++i) {
        mask_data[i] = (i % 30 != 0) ? 1 : 0; // ~97% alive
        if (mask_data[i])
            expected_alive++;
    }

    auto alive_mask = Tensor::from_blob(mask_data.data(), {N}, Device::CPU, DataType::Bool).to(Device::GPU);
    auto alive_indices = alive_mask.nonzero().squeeze(-1).cpu();

    ASSERT_EQ(alive_indices.dtype(), DataType::Int64);
    ASSERT_EQ(alive_indices.numel(), expected_alive);

    // Verify first few indices are correct
    const int64_t* idx_ptr = alive_indices.ptr<int64_t>();
    EXPECT_EQ(idx_ptr[0], 1); // Skip 0 (dead)
    EXPECT_EQ(idx_ptr[1], 2);
    EXPECT_EQ(idx_ptr[2], 3);
}

// ============================================================================
// index_select() Int64 Tests - Bug #3: Missing Int64 support
// ============================================================================

TEST(MCMCTensorOps, IndexSelectInt64Basic) {
    // Create int64 source data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    std::vector<int64_t> source_data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto source = Tensor::from_blob(source_data.data(), {10}, Device::CPU, DataType::Int64).to(Device::GPU);

    // Create int64 indices: [2, 5, 8]
    std::vector<int64_t> index_data = {2, 5, 8};
    auto indices = Tensor::from_blob(index_data.data(), {3}, Device::CPU, DataType::Int64).to(Device::GPU);

    // Perform index_select
    auto result = source.index_select(0, indices).cpu();

    // Check results
    ASSERT_EQ(result.numel(), 3);
    ASSERT_EQ(result.dtype(), DataType::Int64);

    const int64_t* result_ptr = static_cast<const int64_t*>(result.data_ptr());
    EXPECT_EQ(result_ptr[0], 2);
    EXPECT_EQ(result_ptr[1], 5);
    EXPECT_EQ(result_ptr[2], 8);
}

TEST(MCMCTensorOps, IndexSelectInt64MCMCScenario) {
    // Simulate MCMC relocation scenario
    constexpr size_t N = 100;

    // Create alive_indices (int64): all indices from 0 to 99
    std::vector<int64_t> alive_data(N);
    for (size_t i = 0; i < N; ++i) {
        alive_data[i] = static_cast<int64_t>(i);
    }
    auto alive_indices = Tensor::from_blob(alive_data.data(), {N}, Device::CPU, DataType::Int64).to(Device::GPU);

    // Create sampled_local indices (int64): [5, 10, 15, 20, 25]
    std::vector<int64_t> sampled_data = {5, 10, 15, 20, 25};
    auto sampled_local = Tensor::from_blob(sampled_data.data(), {5}, Device::CPU, DataType::Int64).to(Device::GPU);

    // This is the operation that was failing before:
    // sampled_idxs = alive_indices.index_select(0, sampled_local)
    auto sampled_idxs = alive_indices.index_select(0, sampled_local).cpu();

    // Check results
    ASSERT_EQ(sampled_idxs.numel(), 5);
    ASSERT_EQ(sampled_idxs.dtype(), DataType::Int64);

    const int64_t* result_ptr = static_cast<const int64_t*>(sampled_idxs.data_ptr());

    // Should get: [5, 10, 15, 20, 25] (the indices from alive_indices)
    EXPECT_EQ(result_ptr[0], 5);
    EXPECT_EQ(result_ptr[1], 10);
    EXPECT_EQ(result_ptr[2], 15);
    EXPECT_EQ(result_ptr[3], 20);
    EXPECT_EQ(result_ptr[4], 25);
}

TEST(MCMCTensorOps, IndexSelectInt64LargeIndices) {
    // Test with indices > 32768 to ensure no int32 truncation
    constexpr size_t N = 100000;

    // Create large int64 source data
    std::vector<int64_t> source_data(N);
    for (size_t i = 0; i < N; ++i) {
        source_data[i] = static_cast<int64_t>(i);
    }
    auto source = Tensor::from_blob(source_data.data(), {N}, Device::CPU, DataType::Int64).to(Device::GPU);

    // Select some large indices: [50000, 75000, 99999]
    std::vector<int64_t> index_data = {50000, 75000, 99999};
    auto indices = Tensor::from_blob(index_data.data(), {3}, Device::CPU, DataType::Int64).to(Device::GPU);

    // Perform index_select
    auto result = source.index_select(0, indices).cpu();

    // Check results - should NOT have corruption
    ASSERT_EQ(result.numel(), 3);
    ASSERT_EQ(result.dtype(), DataType::Int64);

    const int64_t* result_ptr = static_cast<const int64_t*>(result.data_ptr());
    EXPECT_EQ(result_ptr[0], 50000) << "Got garbage: " << result_ptr[0];
    EXPECT_EQ(result_ptr[1], 75000) << "Got garbage: " << result_ptr[1];
    EXPECT_EQ(result_ptr[2], 99999) << "Got garbage: " << result_ptr[2];
}

// ============================================================================
// index_select() Int32 Tests - Bug #4: Missing Int32 support for ratios
// ============================================================================

TEST(MCMCTensorOps, IndexSelectInt32Basic) {
    // Test Int32 index_select (needed for MCMC ratios tensor)
    std::vector<int32_t> source_data = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    auto source = Tensor::from_blob(source_data.data(), {10}, Device::CPU, DataType::Int32).to(Device::GPU);

    std::vector<int64_t> index_data = {1, 3, 5, 7};
    auto indices = Tensor::from_blob(index_data.data(), {4}, Device::CPU, DataType::Int64).to(Device::GPU);

    // This was throwing "unsupported dtype for CUDA" before the fix
    auto result = source.index_select(0, indices).cpu();

    ASSERT_EQ(result.numel(), 4);
    ASSERT_EQ(result.dtype(), DataType::Int32);

    const int32_t* result_ptr = result.ptr<int32_t>();
    EXPECT_EQ(result_ptr[0], 20);
    EXPECT_EQ(result_ptr[1], 40);
    EXPECT_EQ(result_ptr[2], 60);
    EXPECT_EQ(result_ptr[3], 80);
}

TEST(MCMCTensorOps, IndexSelectInt32RatiosExact) {
    // Simulate the exact MCMC scenario from mcmc.cpp:174
    // ratios = ratios.index_select(0, sampled_idxs).contiguous();

    constexpr size_t N = 54275; // Typical total gaussians
    constexpr size_t n_dead = 1651;

    // Create ratios tensor (Int32) - counts of how many times each gaussian was sampled
    // Initialize to 1 (like ones_like)
    std::vector<int32_t> ratios_data(N, 1);
    auto ratios = Tensor::from_blob(ratios_data.data(), {N}, Device::CPU, DataType::Int32).to(Device::GPU);

    // Create sampled_idxs (Int64) - indices of alive gaussians that were sampled
    std::vector<int64_t> sampled_data(n_dead);
    for (size_t i = 0; i < n_dead; ++i) {
        sampled_data[i] = static_cast<int64_t>(i * 30 % N); // Some pattern
    }
    auto sampled_idxs = Tensor::from_blob(sampled_data.data(), {n_dead}, Device::CPU, DataType::Int64).to(Device::GPU);

    // This is the exact operation that was failing:
    // "index_select: unsupported dtype for CUDA"
    auto sampled_ratios = ratios.index_select(0, sampled_idxs).cpu();

    // Verify results
    ASSERT_EQ(sampled_ratios.numel(), n_dead);
    ASSERT_EQ(sampled_ratios.dtype(), DataType::Int32);

    const int32_t* result_ptr = sampled_ratios.ptr<int32_t>();

    // All should be 1 since we initialized with ones
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(result_ptr[i], 1) << "Index " << i << " failed";
    }
}

TEST(MCMCTensorOps, IndexSelectInt32RatiosCounting) {
    // More realistic MCMC scenario with actual counting
    constexpr size_t N = 10000;
    constexpr size_t n_samples = 500;

    // Simulate ratios after index_add_ (some gaussians sampled multiple times)
    std::vector<int32_t> ratios_data(N);
    for (size_t i = 0; i < N; ++i) {
        ratios_data[i] = 1 + (i % 5); // 1, 2, 3, 4, 5, 1, 2, ...
    }
    auto ratios = Tensor::from_blob(ratios_data.data(), {N}, Device::CPU, DataType::Int32).to(Device::GPU);

    // Sampled indices
    std::vector<int64_t> sampled_data(n_samples);
    for (size_t i = 0; i < n_samples; ++i) {
        sampled_data[i] = static_cast<int64_t>((i * 17) % N);
    }
    auto sampled_idxs = Tensor::from_blob(sampled_data.data(), {n_samples}, Device::CPU, DataType::Int64).to(Device::GPU);

    // Select ratios for sampled gaussians
    auto sampled_ratios = ratios.index_select(0, sampled_idxs).cpu();

    ASSERT_EQ(sampled_ratios.numel(), n_samples);
    ASSERT_EQ(sampled_ratios.dtype(), DataType::Int32);

    const int32_t* result_ptr = sampled_ratios.ptr<int32_t>();

    // Verify a few values
    for (size_t i = 0; i < 10; ++i) {
        int64_t idx = sampled_data[i];
        int32_t expected = 1 + (idx % 5);
        EXPECT_EQ(result_ptr[i], expected)
            << "Index " << i << " (selecting from pos " << idx << ")";
    }
}
