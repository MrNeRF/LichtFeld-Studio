/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <cuda_runtime.h>
#include <cmath>
#include "kernels/regularization.cuh"

/**
 * Regularization CUDA Kernel Tests
 *
 * These tests verify that our custom CUDA kernels for regularization
 * match PyTorch's autograd results exactly. The kernels apply chain rule
 * gradients through parameter transformations:
 * - Scaling: exp(_scaling)
 * - Opacity: sigmoid(_opacity)
 */

class RegularizationTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available, skipping test";
        }
    }
};

// =============================================================================
// SCALING TESTS (exp transformation)
// =============================================================================

TEST_F(RegularizationTest, ExpKernel_MatchesAutograd_Loss) {
    const int n = 10000;
    const float weight = 0.1f;

    // Create raw scaling parameter [N, 3] - 3 scales per Gaussian
    auto scaling_raw = torch::randn({n, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                          .requires_grad_(true);

    // Method 1: Our CUDA kernel
    auto scaling_raw_cuda = scaling_raw.clone().detach().requires_grad_(true);
    float cuda_loss = gs::regularization::compute_exp_l1_regularization_with_grad_cuda(
        scaling_raw_cuda, weight);

    // Method 2: PyTorch autograd (ground truth)
    auto scaling_raw_autograd = scaling_raw.clone().detach().requires_grad_(true);
    auto scaling = torch::exp(scaling_raw_autograd);
    auto loss = weight * scaling.mean();
    float autograd_loss = loss.item<float>();
    loss.backward();

    // Compare loss values
    float loss_diff = std::abs(cuda_loss - autograd_loss);
    EXPECT_LT(loss_diff, 1e-5) << "Loss mismatch: CUDA=" << cuda_loss
                               << ", PyTorch=" << autograd_loss;

    std::cout << "Exp kernel loss: CUDA=" << cuda_loss << ", PyTorch=" << autograd_loss
              << ", diff=" << loss_diff << " ✓" << std::endl;
}

TEST_F(RegularizationTest, ExpKernel_MatchesAutograd_Gradients) {
    const int n = 10000;
    const float weight = 0.1f;

    // Create raw scaling parameter [N, 3] - 3 scales per Gaussian
    auto scaling_raw = torch::randn({n, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                          .requires_grad_(true);

    // Method 1: Our CUDA kernel
    auto scaling_raw_cuda = scaling_raw.clone().detach().requires_grad_(true);
    gs::regularization::compute_exp_l1_regularization_with_grad_cuda(
        scaling_raw_cuda, weight);
    auto cuda_grad = scaling_raw_cuda.grad();

    // Method 2: PyTorch autograd (ground truth)
    auto scaling_raw_autograd = scaling_raw.clone().detach().requires_grad_(true);
    auto scaling = torch::exp(scaling_raw_autograd);
    auto loss = weight * scaling.mean();
    loss.backward();
    auto autograd_grad = scaling_raw_autograd.grad();

    // Compare gradients
    ASSERT_TRUE(cuda_grad.defined()) << "CUDA gradient not computed";
    ASSERT_TRUE(autograd_grad.defined()) << "Autograd gradient not computed";

    auto grad_diff = (cuda_grad - autograd_grad).abs();
    float max_diff = grad_diff.max().item<float>();
    float mean_diff = grad_diff.mean().item<float>();

    EXPECT_LT(max_diff, 1e-5) << "Max gradient difference too large: " << max_diff;
    EXPECT_LT(mean_diff, 1e-6) << "Mean gradient difference too large: " << mean_diff;

    std::cout << "Exp kernel gradients: max_diff=" << max_diff
              << ", mean_diff=" << mean_diff << " ✓" << std::endl;
}

TEST_F(RegularizationTest, ExpKernel_DifferentSizes) {
    const float weight = 0.1f;
    std::vector<int> sizes = {100, 1000, 10000, 100000};

    for (int n : sizes) {
        auto scaling_raw = torch::randn({n, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                              .requires_grad_(true);

        // CUDA kernel
        auto scaling_raw_cuda = scaling_raw.clone().detach().requires_grad_(true);
        float cuda_loss = gs::regularization::compute_exp_l1_regularization_with_grad_cuda(
            scaling_raw_cuda, weight);

        // PyTorch autograd
        auto scaling_raw_autograd = scaling_raw.clone().detach().requires_grad_(true);
        auto scaling = torch::exp(scaling_raw_autograd);
        auto loss = weight * scaling.mean();
        float autograd_loss = loss.item<float>();
        loss.backward();

        // Compare
        float loss_diff = std::abs(cuda_loss - autograd_loss);
        EXPECT_LT(loss_diff, 1e-5) << "Loss mismatch for size " << n;

        auto grad_diff = (scaling_raw_cuda.grad() - scaling_raw_autograd.grad()).abs();
        float max_diff = grad_diff.max().item<float>();
        EXPECT_LT(max_diff, 1e-5) << "Failed for size " << n << ": max_diff=" << max_diff;
    }

    std::cout << "Exp kernel tested across " << sizes.size() << " sizes ✓" << std::endl;
}

TEST_F(RegularizationTest, ExpKernel_GradientAccumulation) {
    const int n = 1000;
    const float weight = 0.1f;

    auto scaling_raw = torch::randn({n, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                          .requires_grad_(true);

    // Simulate existing gradient from main loss
    scaling_raw.mutable_grad() = torch::ones_like(scaling_raw) * 0.5f;
    auto initial_grad = scaling_raw.grad().clone();

    // Apply CUDA kernel (should accumulate)
    gs::regularization::compute_exp_l1_regularization_with_grad_cuda(
        scaling_raw, weight);

    // Verify accumulation with PyTorch
    auto expected_reg_grad = (weight / static_cast<float>(scaling_raw.numel())) * torch::exp(scaling_raw.clone().detach());
    auto expected_total_grad = initial_grad + expected_reg_grad;

    auto grad_diff = (scaling_raw.grad() - expected_total_grad).abs();
    float max_diff = grad_diff.max().item<float>();

    EXPECT_LT(max_diff, 1e-5) << "Gradient accumulation failed: max_diff=" << max_diff;

    std::cout << "Exp kernel gradient accumulation ✓" << std::endl;
}

// =============================================================================
// OPACITY TESTS (sigmoid transformation)
// =============================================================================

TEST_F(RegularizationTest, SigmoidKernel_MatchesAutograd_Loss) {
    const int n = 10000;
    const float weight = 0.1f;

    // Create raw opacity parameter [N, 1]
    auto opacity_raw = torch::randn({n, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                          .requires_grad_(true);

    // Method 1: Our CUDA kernel
    auto opacity_raw_cuda = opacity_raw.clone().detach().requires_grad_(true);
    float cuda_loss = gs::regularization::compute_sigmoid_l1_regularization_with_grad_cuda(
        opacity_raw_cuda, weight);

    // Method 2: PyTorch autograd (ground truth)
    auto opacity_raw_autograd = opacity_raw.clone().detach().requires_grad_(true);
    auto opacity = torch::sigmoid(opacity_raw_autograd).squeeze(-1);
    auto loss = weight * opacity.mean();
    float autograd_loss = loss.item<float>();
    loss.backward();

    // Compare loss values
    float loss_diff = std::abs(cuda_loss - autograd_loss);
    EXPECT_LT(loss_diff, 1e-5) << "Loss mismatch: CUDA=" << cuda_loss
                               << ", PyTorch=" << autograd_loss;

    std::cout << "Sigmoid kernel loss: CUDA=" << cuda_loss << ", PyTorch=" << autograd_loss
              << ", diff=" << loss_diff << " ✓" << std::endl;
}

TEST_F(RegularizationTest, SigmoidKernel_MatchesAutograd_Gradients) {
    const int n = 10000;
    const float weight = 0.1f;

    // Create raw opacity parameter [N, 1]
    auto opacity_raw = torch::randn({n, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                          .requires_grad_(true);

    // Method 1: Our CUDA kernel
    auto opacity_raw_cuda = opacity_raw.clone().detach().requires_grad_(true);
    gs::regularization::compute_sigmoid_l1_regularization_with_grad_cuda(
        opacity_raw_cuda, weight);
    auto cuda_grad = opacity_raw_cuda.grad();

    // Method 2: PyTorch autograd (ground truth)
    auto opacity_raw_autograd = opacity_raw.clone().detach().requires_grad_(true);
    auto opacity = torch::sigmoid(opacity_raw_autograd).squeeze(-1);
    auto loss = weight * opacity.mean();
    loss.backward();
    auto autograd_grad = opacity_raw_autograd.grad();

    // Compare gradients
    ASSERT_TRUE(cuda_grad.defined()) << "CUDA gradient not computed";
    ASSERT_TRUE(autograd_grad.defined()) << "Autograd gradient not computed";

    auto grad_diff = (cuda_grad - autograd_grad).abs();
    float max_diff = grad_diff.max().item<float>();
    float mean_diff = grad_diff.mean().item<float>();

    EXPECT_LT(max_diff, 1e-5) << "Max gradient difference too large: " << max_diff;
    EXPECT_LT(mean_diff, 1e-6) << "Mean gradient difference too large: " << mean_diff;

    std::cout << "Sigmoid kernel gradients: max_diff=" << max_diff
              << ", mean_diff=" << mean_diff << " ✓" << std::endl;
}

TEST_F(RegularizationTest, SigmoidKernel_DifferentSizes) {
    const float weight = 0.1f;
    std::vector<int> sizes = {100, 1000, 10000, 100000};

    for (int n : sizes) {
        auto opacity_raw = torch::randn({n, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                              .requires_grad_(true);

        // CUDA kernel
        auto opacity_raw_cuda = opacity_raw.clone().detach().requires_grad_(true);
        float cuda_loss = gs::regularization::compute_sigmoid_l1_regularization_with_grad_cuda(
            opacity_raw_cuda, weight);

        // PyTorch autograd
        auto opacity_raw_autograd = opacity_raw.clone().detach().requires_grad_(true);
        auto opacity = torch::sigmoid(opacity_raw_autograd).squeeze(-1);
        auto loss = weight * opacity.mean();
        float autograd_loss = loss.item<float>();
        loss.backward();

        // Compare
        float loss_diff = std::abs(cuda_loss - autograd_loss);
        EXPECT_LT(loss_diff, 1e-5) << "Loss mismatch for size " << n;

        auto grad_diff = (opacity_raw_cuda.grad() - opacity_raw_autograd.grad()).abs();
        float max_diff = grad_diff.max().item<float>();
        EXPECT_LT(max_diff, 1e-5) << "Failed for size " << n << ": max_diff=" << max_diff;
    }

    std::cout << "Sigmoid kernel tested across " << sizes.size() << " sizes ✓" << std::endl;
}

TEST_F(RegularizationTest, SigmoidKernel_GradientAccumulation) {
    const int n = 1000;
    const float weight = 0.1f;

    auto opacity_raw = torch::randn({n, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                          .requires_grad_(true);

    // Simulate existing gradient from main loss
    opacity_raw.mutable_grad() = torch::ones_like(opacity_raw) * 0.5f;
    auto initial_grad = opacity_raw.grad().clone();

    // Apply CUDA kernel (should accumulate)
    gs::regularization::compute_sigmoid_l1_regularization_with_grad_cuda(
        opacity_raw, weight);

    // Verify accumulation with PyTorch
    auto opacity_detached = torch::sigmoid(opacity_raw.clone().detach()).squeeze(-1);
    auto expected_reg_grad = (weight / static_cast<float>(opacity_detached.numel())) *
                             (opacity_detached * (1.0f - opacity_detached)).unsqueeze(-1);
    auto expected_total_grad = initial_grad + expected_reg_grad;

    auto grad_diff = (opacity_raw.grad() - expected_total_grad).abs();
    float max_diff = grad_diff.max().item<float>();

    EXPECT_LT(max_diff, 1e-5) << "Gradient accumulation failed: max_diff=" << max_diff;

    std::cout << "Sigmoid kernel gradient accumulation ✓" << std::endl;
}

// =============================================================================
// ZERO WEIGHT TESTS
// =============================================================================

TEST_F(RegularizationTest, ZeroWeight_NoGradients) {
    const int n = 1000;
    const float weight = 0.0f;

    // Test exp kernel
    {
        auto scaling_raw = torch::randn({n, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                              .requires_grad_(true);
        float loss = gs::regularization::compute_exp_l1_regularization_with_grad_cuda(
            scaling_raw, weight);

        EXPECT_FLOAT_EQ(loss, 0.0f) << "Loss should be zero when weight is zero";
        EXPECT_FALSE(scaling_raw.grad().defined()) << "Gradient should not be defined for zero weight";
    }

    // Test sigmoid kernel
    {
        auto opacity_raw = torch::randn({n, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                              .requires_grad_(true);
        float loss = gs::regularization::compute_sigmoid_l1_regularization_with_grad_cuda(
            opacity_raw, weight);

        EXPECT_FLOAT_EQ(loss, 0.0f) << "Loss should be zero when weight is zero";
        EXPECT_FALSE(opacity_raw.grad().defined()) << "Gradient should not be defined for zero weight";
    }

    std::cout << "Zero weight test ✓" << std::endl;
}
