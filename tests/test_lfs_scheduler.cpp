/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "cuda_backend_test.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "optimizer/scheduler.hpp"
#include <cmath>
#include <gtest/gtest.h>

using namespace lfs::core;
using namespace lfs::training;

namespace {

    class LfsSchedulerTest : public lfs::test::CudaBackendTest {};

    // Helper function to create a simple SplatData for testing
    SplatData create_test_splat_data(size_t n_points = 10) {
        auto means = Tensor::randn({n_points, 3}, Device::GPU);
        auto sh0 = Tensor::randn({n_points, 1, 3}, Device::GPU);
        auto shN = Tensor::randn({n_points, 15, 3}, Device::GPU);
        auto scaling = Tensor::randn({n_points, 3}, Device::GPU);
        auto rotation = Tensor::randn({n_points, 4}, Device::GPU);
        auto opacity = Tensor::randn({n_points, 1}, Device::GPU);

        SplatData splat_data(3, means, sh0, shN, scaling, rotation, opacity, 1.0f);
        // Note: gradients are allocated by AdamOptimizer, not SplatData
        return splat_data;
    }

    // ===================================================================================
    // ExponentialLR Tests
    // ===================================================================================

    TEST_F(LfsSchedulerTest, ExponentialLR_Basic) {
        // Create optimizer with initial LR
        auto splat_data = create_test_splat_data(10);
        AdamConfig config;
        config.lr = 1.0f;
        AdamOptimizer optimizer(splat_data, config);

        // Create scheduler with gamma=0.9
        double gamma = 0.9;
        ExponentialLR scheduler(optimizer, gamma);

        // Check initial LR
        EXPECT_FLOAT_EQ(optimizer.get_lr(), 1.0f);

        // Step 1: lr = 1.0 * 0.9 = 0.9
        scheduler.step();
        EXPECT_NEAR(optimizer.get_lr(), 0.9f, 1e-5f);

        // Step 2: lr = 0.9 * 0.9 = 0.81
        scheduler.step();
        EXPECT_NEAR(optimizer.get_lr(), 0.81f, 1e-5f);

        // Step 3: lr = 0.81 * 0.9 = 0.729
        scheduler.step();
        EXPECT_NEAR(optimizer.get_lr(), 0.729f, 1e-5f);
    }

    TEST_F(LfsSchedulerTest, ExponentialLR_MultipleSteps) {
        auto splat_data = create_test_splat_data(10);
        AdamConfig config;
        config.lr = 0.001f;
        AdamOptimizer optimizer(splat_data, config);

        double gamma = 0.99;
        ExponentialLR scheduler(optimizer, gamma);

        // Take 100 steps
        float initial_lr = optimizer.get_lr();
        for (int i = 0; i < 100; i++) {
            scheduler.step();
        }

        // After 100 steps: lr = 0.001 * 0.99^100
        float expected_lr = initial_lr * std::pow(gamma, 100);
        EXPECT_NEAR(optimizer.get_lr(), expected_lr, 1e-7f);
    }

    TEST_F(LfsSchedulerTest, ExponentialLR_RapidDecay) {
        auto splat_data = create_test_splat_data(10);
        AdamConfig config;
        config.lr = 1.0f;
        AdamOptimizer optimizer(splat_data, config);

        // Rapid decay with gamma=0.5
        double gamma = 0.5;
        ExponentialLR scheduler(optimizer, gamma);

        for (int i = 0; i < 10; i++) {
            scheduler.step();
        }

        // After 10 steps: lr = 1.0 * 0.5^10 = 1/1024 ≈ 0.0009766
        float expected_lr = std::pow(0.5, 10);
        EXPECT_NEAR(optimizer.get_lr(), expected_lr, 1e-6f);
    }

    TEST_F(LfsSchedulerTest, ExponentialLR_NoDecay) {
        auto splat_data = create_test_splat_data(10);
        AdamConfig config;
        config.lr = 0.001f;
        AdamOptimizer optimizer(splat_data, config);

        // No decay with gamma=1.0
        double gamma = 1.0;
        ExponentialLR scheduler(optimizer, gamma);

        float initial_lr = optimizer.get_lr();
        for (int i = 0; i < 100; i++) {
            scheduler.step();
        }

        // LR should remain unchanged
        EXPECT_FLOAT_EQ(optimizer.get_lr(), initial_lr);
    }

    // ===================================================================================
    // Integration Tests - Scheduler with Optimizer
    // ===================================================================================

    TEST_F(LfsSchedulerTest, Integration_ExponentialLR_WithOptimization) {
        auto splat_data = create_test_splat_data(10);
        AdamConfig config;
        config.lr = 0.1f;
        AdamOptimizer optimizer(splat_data, config);
        optimizer.allocate_gradients();

        double gamma = 0.95;
        ExponentialLR scheduler(optimizer, gamma);

        // Simulate training loop
        for (int iter = 0; iter < 10; iter++) {
            // Simulate gradients
            optimizer.get_grad(lfs::training::ParamType::Means) = Tensor::ones(splat_data.means().shape(), Device::GPU);

            // Optimize
            optimizer.step(iter);

            // Update LR
            scheduler.step();

            // Verify LR is decaying
            float expected_lr = 0.1f * std::pow(gamma, iter + 1);
            EXPECT_NEAR(optimizer.get_lr(), expected_lr, 1e-6f);

            // Zero gradients
            optimizer.zero_grad(iter);
        }
    }

    // ===================================================================================
    // Stress Tests
    // ===================================================================================

    TEST_F(LfsSchedulerTest, StressTest_ManySteps) {
        auto splat_data = create_test_splat_data(10);
        AdamConfig config;
        config.lr = 1.0f;
        AdamOptimizer optimizer(splat_data, config);

        double gamma = 0.9999;
        ExponentialLR scheduler(optimizer, gamma);

        // Run for many iterations
        for (int i = 0; i < 10000; i++) {
            scheduler.step();
        }

        // Verify LR
        float expected_lr = std::pow(gamma, 10000);
        EXPECT_NEAR(optimizer.get_lr(), expected_lr, 1e-5f);

        // LR should still be positive and reasonable
        EXPECT_GT(optimizer.get_lr(), 0.0f);
        EXPECT_LT(optimizer.get_lr(), 1.0f);
    }

    TEST_F(LfsSchedulerTest, StressTest_VerySmallLR) {
        auto splat_data = create_test_splat_data(10);
        AdamConfig config;
        config.lr = 1e-10f;
        AdamOptimizer optimizer(splat_data, config);

        double gamma = 0.5;
        ExponentialLR scheduler(optimizer, gamma);

        // Even with very small LR, scheduler should work
        for (int i = 0; i < 10; i++) {
            scheduler.step();
        }

        float expected_lr = 1e-10f * std::pow(0.5, 10);
        EXPECT_NEAR(optimizer.get_lr(), expected_lr, 1e-16f);
    }

    TEST_F(LfsSchedulerTest, StressTest_MultipleSchedulers) {
        auto splat_data = create_test_splat_data(10);
        AdamConfig config;
        config.lr = 1.0f;
        AdamOptimizer optimizer(splat_data, config);

        // Create multiple schedulers (only one should be used at a time)
        double gamma1 = 0.95;
        ExponentialLR scheduler1(optimizer, gamma1);

        for (int i = 0; i < 5; i++) {
            scheduler1.step();
        }

        float lr_after_first = optimizer.get_lr();
        EXPECT_NEAR(lr_after_first, std::pow(gamma1, 5), 1e-5f);

        // Switch to a new scheduler (different gamma)
        double gamma2 = 0.9;
        ExponentialLR scheduler2(optimizer, gamma2);

        for (int i = 0; i < 5; i++) {
            scheduler2.step();
        }

        // Should continue from current LR
        float expected_final_lr = lr_after_first * std::pow(gamma2, 5);
        EXPECT_NEAR(optimizer.get_lr(), expected_final_lr, 1e-5f);
    }

    // ===================================================================================
    // Realistic Training Scenario Tests
    // ===================================================================================

} // anonymous namespace
