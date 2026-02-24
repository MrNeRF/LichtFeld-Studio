/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_gradient_accumulation.cpp
 * @brief Test gradient accumulation behavior in regularization losses
 *
 * This test verifies that regularization gradients properly ACCUMULATE
 * on top of existing gradients rather than overwriting them.
 */

#include "core/tensor.hpp"
#include "training/losses/regularization.hpp"
#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

using namespace lfs::core;
using namespace lfs::training::losses;

class GradientAccumulationTest : public ::testing::Test {
protected:
    void SetUp() override {
        spdlog::set_level(spdlog::level::info);
        Tensor::manual_seed(42);
    }
};

TEST_F(GradientAccumulationTest, OpacityRegularization_AccumulatesGradients) {
    spdlog::info("=== Testing Opacity Regularization Gradient Accumulation ===");

    constexpr size_t N = 1000;
    constexpr float weight = 0.01f;

    // Create test data
    auto opacity = Tensor::randn({N, 1}, Device::CUDA);

    // Test 1: Apply regularization to ZERO gradients
    spdlog::info("--- Test 1: Regularization on zero gradients ---");
    auto opacity_grad_1 = Tensor::zeros_like(opacity);

    OpacityRegularization::Params params{.weight = weight};
    auto result_1 = OpacityRegularization::forward(opacity, opacity_grad_1, params);
    ASSERT_TRUE(result_1.has_value());

    float grad_norm_1 = opacity_grad_1.abs().sum().item<float>();
    float grad_max_1 = opacity_grad_1.abs().max().item<float>();
    spdlog::info("After regularization: norm={:.6e}, max={:.6e}", grad_norm_1, grad_max_1);

    // Test 2: Apply regularization to EXISTING gradients (should accumulate)
    spdlog::info("--- Test 2: Regularization on existing gradients (should accumulate) ---");
    auto opacity_grad_2 = Tensor::ones_like(opacity) * 0.1f; // Pre-existing gradients
    auto opacity_grad_2_before = opacity_grad_2.clone();

    auto result_2 = OpacityRegularization::forward(opacity, opacity_grad_2, params);
    ASSERT_TRUE(result_2.has_value());

    // Element-wise check: at least one element must have changed
    auto element_diff = (opacity_grad_2 - opacity_grad_2_before).abs().max().item<float>();
    spdlog::info("Max element-wise change: {:.6e}", element_diff);

    EXPECT_GT(element_diff, 0.0f)
        << "Gradients should accumulate (at least one element must change)!";

    // Test 3: Verify the accumulated gradient equals the sum
    spdlog::info("--- Test 3: Verify accumulation is additive ---");
    auto expected_grad = Tensor::ones_like(opacity) * 0.1f + opacity_grad_1;

    auto diff = (opacity_grad_2 - expected_grad).abs();
    float max_diff = diff.max().item<float>();
    spdlog::info("Max difference from expected: {:.6e}", max_diff);

    EXPECT_LT(max_diff, 1e-5f) << "Accumulated gradient should equal sum of components";
}

TEST_F(GradientAccumulationTest, ScaleRegularization_AccumulatesGradients) {
    spdlog::info("=== Testing Scale Regularization Gradient Accumulation ===");

    constexpr size_t N = 1000;
    constexpr size_t D = 3;
    constexpr float weight = 0.01f;

    // Create test data
    auto scaling = Tensor::randn({N, D}, Device::CUDA);

    // Test 1: Apply regularization to ZERO gradients
    spdlog::info("--- Test 1: Regularization on zero gradients ---");
    auto scaling_grad_1 = Tensor::zeros_like(scaling);

    ScaleRegularization::Params params{.weight = weight};
    auto result_1 = ScaleRegularization::forward(scaling, scaling_grad_1, params);
    ASSERT_TRUE(result_1.has_value());

    float grad_norm_1 = scaling_grad_1.abs().sum().item<float>();
    float grad_max_1 = scaling_grad_1.abs().max().item<float>();
    spdlog::info("After regularization: norm={:.6e}, max={:.6e}", grad_norm_1, grad_max_1);

    // Test 2: Apply regularization to EXISTING gradients (should accumulate)
    spdlog::info("--- Test 2: Regularization on existing gradients (should accumulate) ---");
    auto scaling_grad_2 = Tensor::ones_like(scaling) * 0.1f; // Pre-existing gradients

    float grad_norm_before = scaling_grad_2.abs().sum().item<float>();
    spdlog::info("Before regularization: norm={:.6e}", grad_norm_before);

    auto result_2 = ScaleRegularization::forward(scaling, scaling_grad_2, params);
    ASSERT_TRUE(result_2.has_value());

    float grad_norm_after = scaling_grad_2.abs().sum().item<float>();
    float grad_max_after = scaling_grad_2.abs().max().item<float>();
    spdlog::info("After regularization: norm={:.6e}, max={:.6e}", grad_norm_after, grad_max_after);

    // The gradient norm should INCREASE (accumulation) not stay the same or decrease
    spdlog::info("Gradient norm change: {:.6e} -> {:.6e} (diff: {:.6e})",
                 grad_norm_before, grad_norm_after, grad_norm_after - grad_norm_before);

    EXPECT_GT(grad_norm_after, grad_norm_before)
        << "Gradients should accumulate (increase), not overwrite!";

    // Test 3: Verify the accumulated gradient equals the sum
    spdlog::info("--- Test 3: Verify accumulation is additive ---");
    auto expected_grad = Tensor::ones_like(scaling) * 0.1f + scaling_grad_1;

    auto diff = (scaling_grad_2 - expected_grad).abs();
    float max_diff = diff.max().item<float>();
    spdlog::info("Max difference from expected: {:.6e}", max_diff);

    EXPECT_LT(max_diff, 1e-5f) << "Accumulated gradient should equal sum of components";
}

TEST_F(GradientAccumulationTest, MultipleAccumulations) {
    spdlog::info("=== Testing Multiple Sequential Gradient Accumulations ===");

    constexpr size_t N = 1000;
    constexpr float weight = 0.01f;

    auto opacity = Tensor::randn({N, 1}, Device::CUDA);
    auto opacity_grad = Tensor::zeros_like(opacity);

    OpacityRegularization::Params params{.weight = weight};

    // Apply regularization multiple times - should accumulate each time
    spdlog::info("Applying regularization 5 times sequentially:");

    float prev_norm = 0.0f;
    for (int i = 0; i < 5; ++i) {
        auto result = OpacityRegularization::forward(opacity, opacity_grad, params);
        ASSERT_TRUE(result.has_value()) << "Iteration " << i << " failed";

        float curr_norm = opacity_grad.abs().sum().item<float>();
        spdlog::info("  Iteration {}: gradient norm = {:.6e}", i, curr_norm);

        if (i > 0) {
            EXPECT_GT(curr_norm, prev_norm)
                << "Gradient should keep accumulating at iteration " << i;
        }

        prev_norm = curr_norm;
    }
}

TEST_F(GradientAccumulationTest, RasterGradient_ThenRegularization) {
    spdlog::info("=== Testing Simulated Rasterizer + Regularization Flow ===");

    constexpr size_t N = 1000;
    constexpr float reg_weight = 0.01f;

    auto opacity = Tensor::randn({N, 1}, Device::CUDA);
    auto opacity_grad = Tensor::zeros_like(opacity);

    // Simulate large gradients from rasterizer backward
    spdlog::info("Step 1: Simulate rasterizer backward (large gradients)");
    auto raster_grads = Tensor::randn({N, 1}, Device::CUDA) * 0.01f; // Typical rasterizer magnitude
    opacity_grad = opacity_grad + raster_grads;

    float grad_after_raster = opacity_grad.abs().sum().item<float>();
    float grad_max_after_raster = opacity_grad.abs().max().item<float>();
    spdlog::info("After rasterizer: norm={:.6e}, max={:.6e}",
                 grad_after_raster, grad_max_after_raster);

    // Apply regularization on top
    spdlog::info("Step 2: Apply opacity regularization (should accumulate)");
    OpacityRegularization::Params params{.weight = reg_weight};
    auto result = OpacityRegularization::forward(opacity, opacity_grad, params);
    ASSERT_TRUE(result.has_value());

    float grad_after_reg = opacity_grad.abs().sum().item<float>();
    float grad_max_after_reg = opacity_grad.abs().max().item<float>();
    spdlog::info("After regularization: norm={:.6e}, max={:.6e}",
                 grad_after_reg, grad_max_after_reg);

    // Regularization should have added MORE gradient on top
    spdlog::info("Gradient increase from regularization: {:.6e}",
                 grad_after_reg - grad_after_raster);

    EXPECT_GT(grad_after_reg, grad_after_raster * 0.99f) // Allow for small numerical variance
        << "Regularization should add gradient, not overwrite!";

    // The increase should be small but non-zero (regularization is a small term)
    float rel_increase = (grad_after_reg - grad_after_raster) / grad_after_raster;
    spdlog::info("Relative increase: {:.2f}%", rel_increase * 100.0f);

    EXPECT_GT(rel_increase, -0.01f) // Should not decrease (allowing tiny numerical error)
        << "Regularization should not decrease total gradient significantly";
}
