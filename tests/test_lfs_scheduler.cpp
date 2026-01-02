/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "optimizer/scheduler.hpp"
#include <cmath>
#include <gtest/gtest.h>

using namespace lfs::core;
using namespace lfs::training;

namespace {

    // Helper function to create a simple SplatData for testing
    SplatData create_test_splat_data(size_t n_points = 10) {
        auto means = Tensor::randn({n_points, 3}, Device::CUDA);
        auto sh0 = Tensor::randn({n_points, 1, 3}, Device::CUDA);
        auto shN = Tensor::randn({n_points, 15, 3}, Device::CUDA);
        auto scaling = Tensor::randn({n_points, 3}, Device::CUDA);
        auto rotation = Tensor::randn({n_points, 4}, Device::CUDA);
        auto opacity = Tensor::randn({n_points, 1}, Device::CUDA);

        SplatData splat_data(3, means, sh0, shN, scaling, rotation, opacity, 1.0f);
        // Note: gradients are allocated by AdamOptimizer, not SplatData
        return splat_data;
    }

    // ===================================================================================
    // ExponentialLR Tests
    // ===================================================================================

    TEST(LfsSchedulerTest, ExponentialLR_Basic) {
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

    TEST(LfsSchedulerTest, ExponentialLR_MultipleSteps) {
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

    TEST(LfsSchedulerTest, ExponentialLR_RapidDecay) {
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

    TEST(LfsSchedulerTest, ExponentialLR_NoDecay) {
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
    // WarmupExponentialLR Tests - Basic Functionality
    // ===================================================================================

    TEST(LfsSchedulerTest, WarmupExponentialLR_NoWarmup) {
        auto splat_data = create_test_splat_data(10);
        AdamConfig config;
        config.lr = 1.0f;
        AdamOptimizer optimizer(splat_data, config);

        // No warmup, just exponential decay
        double gamma = 0.9;
        WarmupExponentialLR scheduler(optimizer, gamma, /*warmup_steps=*/0);

        EXPECT_FLOAT_EQ(optimizer.get_lr(), 1.0f);

        // Should immediately start decaying
        scheduler.step();
        EXPECT_NEAR(optimizer.get_lr(), 0.9f, 1e-5f);

        scheduler.step();
        EXPECT_NEAR(optimizer.get_lr(), 0.81f, 1e-5f);
    }

    TEST(LfsSchedulerTest, WarmupExponentialLR_LinearWarmup) {
        auto splat_data = create_test_splat_data(10);
        AdamConfig config;
        config.lr = 1.0f;
        AdamOptimizer optimizer(splat_data, config);

        // Linear warmup from 0.1 to 1.0 over 10 steps
        double gamma = 1.0; // No decay, just test warmup
        int warmup_steps = 10;
        double warmup_start_factor = 0.1;
        WarmupExponentialLR scheduler(optimizer, gamma, warmup_steps, warmup_start_factor);

        EXPECT_EQ(scheduler.get_step(), 0);

        // Step 1: progress = 1/10 = 0.1, factor = 0.1 + 0.9*0.1 = 0.19, lr = 0.19
        scheduler.step();
        EXPECT_EQ(scheduler.get_step(), 1);
        float expected_lr = 1.0f * (0.1f + 0.9f * 0.1f);
        EXPECT_NEAR(optimizer.get_lr(), expected_lr, 1e-5f);

        // Step 5: progress = 5/10 = 0.5, factor = 0.1 + 0.9*0.5 = 0.55, lr = 0.55
        for (int i = 1; i < 5; i++)
            scheduler.step();
        EXPECT_EQ(scheduler.get_step(), 5);
        expected_lr = 1.0f * (0.1f + 0.9f * 0.5f);
        EXPECT_NEAR(optimizer.get_lr(), expected_lr, 1e-5f);

        // Step 10: progress = 10/10 = 1.0, factor = 0.1 + 0.9*1.0 = 1.0, lr = 1.0
        for (int i = 5; i < 10; i++)
            scheduler.step();
        EXPECT_EQ(scheduler.get_step(), 10);
        EXPECT_NEAR(optimizer.get_lr(), 1.0f, 1e-5f);

        // After warmup, with gamma=1.0, LR stays constant
        scheduler.step();
        EXPECT_EQ(scheduler.get_step(), 11);
        EXPECT_NEAR(optimizer.get_lr(), 1.0f, 1e-5f);
    }

    TEST(LfsSchedulerTest, WarmupExponentialLR_WarmupThenDecay) {
        auto splat_data = create_test_splat_data(10);
        AdamConfig config;
        config.lr = 1.0f;
        AdamOptimizer optimizer(splat_data, config);

        // Warmup for 5 steps, then decay with gamma=0.9
        double gamma = 0.9;
        int warmup_steps = 5;
        double warmup_start_factor = 0.5;
        WarmupExponentialLR scheduler(optimizer, gamma, warmup_steps, warmup_start_factor);

        // Warmup phase (steps 1-5)
        for (int i = 0; i < warmup_steps; i++) {
            scheduler.step();
        }
        // At step 5: progress=1.0, factor=0.5+0.5*1.0=1.0, lr=1.0
        EXPECT_NEAR(optimizer.get_lr(), 1.0f, 1e-5f);

        // Decay phase starts
        // Step 6: decay_steps = 6-5 = 1, lr = 1.0 * 0.9^1 = 0.9
        scheduler.step();
        EXPECT_NEAR(optimizer.get_lr(), 0.9f, 1e-5f);

        // Step 7: decay_steps = 7-5 = 2, lr = 1.0 * 0.9^2 = 0.81
        scheduler.step();
        EXPECT_NEAR(optimizer.get_lr(), 0.81f, 1e-5f);

        // Step 10: decay_steps = 10-5 = 5, lr = 1.0 * 0.9^5 = 0.59049
        for (int i = 0; i < 3; i++)
            scheduler.step();
        float expected_lr = std::pow(0.9, 5);
        EXPECT_NEAR(optimizer.get_lr(), expected_lr, 1e-5f);
    }

    TEST(LfsSchedulerTest, WarmupExponentialLR_FullStartFactor) {
        auto splat_data = create_test_splat_data(10);
        AdamConfig config;
        config.lr = 0.001f;
        AdamOptimizer optimizer(splat_data, config);

        // Warmup with start_factor=1.0 means no warmup effect
        double gamma = 0.95;
        int warmup_steps = 10;
        double warmup_start_factor = 1.0;
        WarmupExponentialLR scheduler(optimizer, gamma, warmup_steps, warmup_start_factor);

        float initial_lr = optimizer.get_lr();

        // During warmup, LR should stay constant (factor = 1.0 + 0*progress = 1.0)
        for (int i = 0; i < warmup_steps; i++) {
            scheduler.step();
            EXPECT_NEAR(optimizer.get_lr(), initial_lr, 1e-7f);
        }

        // After warmup, decay starts
        scheduler.step();
        EXPECT_NEAR(optimizer.get_lr(), initial_lr * 0.95f, 1e-7f);
    }

    TEST(LfsSchedulerTest, WarmupExponentialLR_ZeroStartFactor) {
        auto splat_data = create_test_splat_data(10);
        AdamConfig config;
        config.lr = 1.0f;
        AdamOptimizer optimizer(splat_data, config);

        // Warmup from 0 to 1.0 over 4 steps
        double gamma = 1.0;
        int warmup_steps = 4;
        double warmup_start_factor = 0.0;
        WarmupExponentialLR scheduler(optimizer, gamma, warmup_steps, warmup_start_factor);

        // Step 1: progress = 0.25, factor = 0.0 + 1.0*0.25 = 0.25, lr = 0.25
        scheduler.step();
        EXPECT_NEAR(optimizer.get_lr(), 0.25f, 1e-5f);

        // Step 2: progress = 0.5, factor = 0.5, lr = 0.5
        scheduler.step();
        EXPECT_NEAR(optimizer.get_lr(), 0.5f, 1e-5f);

        // Step 3: progress = 0.75, factor = 0.75, lr = 0.75
        scheduler.step();
        EXPECT_NEAR(optimizer.get_lr(), 0.75f, 1e-5f);

        // Step 4: progress = 1.0, factor = 1.0, lr = 1.0
        scheduler.step();
        EXPECT_NEAR(optimizer.get_lr(), 1.0f, 1e-5f);
    }

    // ===================================================================================
    // WarmupExponentialLR Tests - Edge Cases
    // ===================================================================================

    TEST(LfsSchedulerTest, WarmupExponentialLR_SingleStepWarmup) {
        auto splat_data = create_test_splat_data(10);
        AdamConfig config;
        config.lr = 1.0f;
        AdamOptimizer optimizer(splat_data, config);

        double gamma = 0.9;
        int warmup_steps = 1;
        double warmup_start_factor = 0.5;
        WarmupExponentialLR scheduler(optimizer, gamma, warmup_steps, warmup_start_factor);

        // Step 1: progress = 1.0, factor = 0.5 + 0.5*1.0 = 1.0, lr = 1.0
        scheduler.step();
        EXPECT_NEAR(optimizer.get_lr(), 1.0f, 1e-5f);

        // Step 2: decay starts, lr = 1.0 * 0.9^1 = 0.9
        scheduler.step();
        EXPECT_NEAR(optimizer.get_lr(), 0.9f, 1e-5f);
    }

    TEST(LfsSchedulerTest, WarmupExponentialLR_LongWarmup) {
        auto splat_data = create_test_splat_data(10);
        AdamConfig config;
        config.lr = 0.01f;
        AdamOptimizer optimizer(splat_data, config);

        double gamma = 0.99;
        int warmup_steps = 1000;
        double warmup_start_factor = 0.01;
        WarmupExponentialLR scheduler(optimizer, gamma, warmup_steps, warmup_start_factor);

        // Step through entire warmup
        for (int i = 0; i < warmup_steps; i++) {
            scheduler.step();
        }

        // At end of warmup, should be at initial LR
        EXPECT_NEAR(optimizer.get_lr(), 0.01f, 1e-7f);

        // Verify some intermediate points
        AdamOptimizer optimizer2(splat_data, config);
        WarmupExponentialLR scheduler2(optimizer2, gamma, warmup_steps, warmup_start_factor);

        // At step 500: progress = 0.5, factor = 0.01 + 0.99*0.5 = 0.505, lr = 0.00505
        for (int i = 0; i < 500; i++)
            scheduler2.step();
        float expected_lr = 0.01f * (0.01f + 0.99f * 0.5f);
        EXPECT_NEAR(optimizer2.get_lr(), expected_lr, 1e-7f);
    }

    // ===================================================================================
    // Integration Tests - Scheduler with Optimizer
    // ===================================================================================

    TEST(LfsSchedulerTest, Integration_ExponentialLR_WithOptimization) {
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
            optimizer.get_grad(lfs::training::ParamType::Means) = Tensor::ones(splat_data.means().shape(), Device::CUDA);

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

    TEST(LfsSchedulerTest, Integration_WarmupExponentialLR_WithOptimization) {
        auto splat_data = create_test_splat_data(10);
        AdamConfig config;
        config.lr = 0.01f;
        AdamOptimizer optimizer(splat_data, config);
        optimizer.allocate_gradients();

        double gamma = 0.98;
        int warmup_steps = 5;
        double warmup_start_factor = 0.1;
        WarmupExponentialLR scheduler(optimizer, gamma, warmup_steps, warmup_start_factor);

        // Simulate training loop
        for (int iter = 0; iter < 20; iter++) {
            // Simulate gradients
            optimizer.get_grad(lfs::training::ParamType::Means) = Tensor::randn(splat_data.means().shape(), Device::CUDA);

            // Optimize
            optimizer.step(iter);

            // Update LR
            scheduler.step();

            // Verify LR follows expected schedule
            float expected_lr;
            if (iter + 1 <= warmup_steps) {
                // Warmup phase
                float progress = static_cast<float>(iter + 1) / warmup_steps;
                float factor = warmup_start_factor + (1.0f - warmup_start_factor) * progress;
                expected_lr = 0.01f * factor;
            } else {
                // Decay phase
                int decay_steps = (iter + 1) - warmup_steps;
                expected_lr = 0.01f * std::pow(gamma, decay_steps);
            }

            EXPECT_NEAR(optimizer.get_lr(), expected_lr, 1e-6f) << "Iteration: " << iter;

            // Zero gradients
            optimizer.zero_grad(iter);
        }
    }

    // ===================================================================================
    // Stress Tests
    // ===================================================================================

    TEST(LfsSchedulerTest, StressTest_ManySteps) {
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

    TEST(LfsSchedulerTest, StressTest_VerySmallLR) {
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

    TEST(LfsSchedulerTest, StressTest_MultipleSchedulers) {
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

    TEST(LfsSchedulerTest, RealisticScenario_GaussianSplatting) {
        // Simulate a realistic Gaussian Splatting training scenario
        auto splat_data = create_test_splat_data(1000);
        AdamConfig config;
        config.lr = 1.6e-4f; // Typical initial LR for Gaussian Splatting
        AdamOptimizer optimizer(splat_data, config);

        // Typical schedule: warmup for 100 steps, then decay
        double gamma = std::pow(0.01, 1.0 / 30000.0); // Decay to 1% over 30k steps
        int warmup_steps = 100;
        double warmup_start_factor = 0.01;
        WarmupExponentialLR scheduler(optimizer, gamma, warmup_steps, warmup_start_factor);

        // Verify warmup phase
        for (int i = 0; i < warmup_steps; i++) {
            scheduler.step();
        }
        EXPECT_NEAR(optimizer.get_lr(), config.lr, 1e-8f);

        // Verify some points in decay phase
        int steps_to_check[] = {1000, 5000, 10000, 30000};
        for (int target_step : steps_to_check) {
            AdamOptimizer temp_optimizer(splat_data, config);
            WarmupExponentialLR temp_scheduler(temp_optimizer, gamma, warmup_steps, warmup_start_factor);

            for (int i = 0; i < target_step; i++) {
                temp_scheduler.step();
            }

            float expected_lr;
            int decay_steps = target_step - warmup_steps;
            expected_lr = config.lr * std::pow(gamma, decay_steps);

            EXPECT_NEAR(temp_optimizer.get_lr(), expected_lr, 1e-8f)
                << "Failed at step " << target_step;
        }
    }

} // anonymous namespace
