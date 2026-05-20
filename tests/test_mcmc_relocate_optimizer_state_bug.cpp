/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_mcmc_relocate_optimizer_state_bug.cpp
 * @brief Regression coverage for MCMC relocation optimizer state reset.
 *
 * When relocate_gs() copies parameters from sampled_indices to dead_indices:
 * 1. Source rows are adjusted in-place and must lose stale momentum.
 * 2. Destination dead rows receive fresh parameters and must also lose stale momentum.
 */

#include "core/logger.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "optimizer/adam_optimizer.hpp"
#include <gtest/gtest.h>

using namespace lfs::training;
using namespace lfs::core;

class MCMCRelocateOptimizerStateTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create small SplatData with 10 Gaussians
        n_gaussians = 10;

        // Initialize with simple values
        auto means = Tensor::zeros({n_gaussians, 3}, Device::CUDA);
        auto sh0 = Tensor::zeros({n_gaussians, 1, 3}, Device::CUDA);  // [N, 1, 3]
        auto shN = Tensor::zeros({n_gaussians, 15, 3}, Device::CUDA); // degree 3: 15 coefficients, 3 channels
        auto scaling_raw = Tensor::zeros({n_gaussians, 3}, Device::CUDA);
        auto rotation_raw = Tensor::ones({n_gaussians, 4}, Device::CUDA); // normalized quaternion [1,0,0,0]
        auto opacity_raw = Tensor::zeros({n_gaussians, 1}, Device::CUDA);

        splat_data = std::make_unique<SplatData>(
            3, // sh_degree
            std::move(means),
            std::move(sh0),
            std::move(shN),
            std::move(scaling_raw),
            std::move(rotation_raw),
            std::move(opacity_raw),
            1.0f // scene_scale
        );

        // Create optimizer and allocate gradients
        AdamConfig config;
        config.lr = 0.01f;
        config.beta1 = 0.9f;
        config.beta2 = 0.999f;
        config.eps = 1e-15f;

        optimizer = std::make_unique<AdamOptimizer>(*splat_data, config);
        optimizer->allocate_gradients();
    }

    size_t n_gaussians;
    std::unique_ptr<SplatData> splat_data;
    std::unique_ptr<AdamOptimizer> optimizer;
};

TEST_F(MCMCRelocateOptimizerStateTest, ResetBothSourceAndDestinationRows) {
    std::cout << "\n=== Testing CORRECT Behavior (Reset Both Indices) ===" << std::endl;

    // Build momentum (same as previous test)
    for (int iter = 0; iter < 5; iter++) {
        auto grad_means = Tensor::ones({n_gaussians, 3}, Device::CUDA) * 0.1f;
        optimizer->get_grad(ParamType::Means) = grad_means;
        optimizer->step(iter);
        optimizer->zero_grad(iter);
    }

    // Verify momentum exists
    auto means_state = optimizer->get_state(ParamType::Means);
    auto exp_avg_before = means_state->exp_avg.cpu();
    float* exp_avg_before_data = exp_avg_before.ptr<float>();

    float total_momentum_before = 0.0f;
    for (size_t i = 0; i < n_gaussians * 3; i++) {
        total_momentum_before += std::abs(exp_avg_before_data[i]);
    }
    std::cout << "  Total momentum BEFORE: " << total_momentum_before << std::endl;
    EXPECT_GT(total_momentum_before, 0.0f);

    // Setup indices
    std::vector<int64_t> dead_idx_vec = {8, 9};
    std::vector<int64_t> sampled_idx_vec = {2, 3};
    auto dead_indices = Tensor::from_blob(dead_idx_vec.data(), {2}, Device::CPU, DataType::Int64).cuda();
    auto sampled_indices = Tensor::from_blob(sampled_idx_vec.data(), {2}, Device::CPU, DataType::Int64).cuda();

    // Get pointers
    const int64_t* sampled_indices_gpu_ptr = sampled_indices.template ptr<int64_t>();
    const int64_t* dead_indices_gpu_ptr = dead_indices.template ptr<int64_t>();

    // CORRECT implementation: Reset BOTH sampled and dead indices
    std::cout << "\n  Resetting optimizer state at sampled indices..." << std::endl;
    optimizer->relocate_params_at_indices_gpu(ParamType::Means,
                                              sampled_indices_gpu_ptr,
                                              static_cast<size_t>(sampled_indices.numel()));

    std::cout << "  Resetting optimizer state at dead indices." << std::endl;
    optimizer->relocate_params_at_indices_gpu(ParamType::Means,
                                              dead_indices_gpu_ptr,
                                              static_cast<size_t>(dead_indices.numel()));

    // Check that BOTH indices have zero momentum
    auto exp_avg_after = optimizer->get_state(ParamType::Means)->exp_avg.cpu();
    float* exp_avg_after_data = exp_avg_after.ptr<float>();

    bool all_reset = true;
    for (int idx : dead_idx_vec) {
        for (int j = 0; j < 3; j++) {
            if (std::abs(exp_avg_after_data[idx * 3 + j]) > 1e-6f) {
                all_reset = false;
            }
        }
    }
    for (int idx : sampled_idx_vec) {
        for (int j = 0; j < 3; j++) {
            if (std::abs(exp_avg_after_data[idx * 3 + j]) > 1e-6f) {
                all_reset = false;
            }
        }
    }

    std::cout << "\n  Both sampled AND dead indices have zero momentum: " << (all_reset ? "YES" : "NO") << std::endl;
    EXPECT_TRUE(all_reset) << "CORRECT behavior: Both sampled and dead indices should have zero momentum!";

    std::cout << "\nCorrect behavior verified." << std::endl;
    std::cout << "When both sampled_indices and dead_indices have their optimizer state reset," << std::endl;
    std::cout << "the relocated Gaussians start with fresh momentum matching their parameters." << std::endl;
}

// Note: main() is provided by test_main.cpp
