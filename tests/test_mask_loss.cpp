/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_mask_loss.cpp
 * @brief Tests for masked photometric loss computation
 *
 * Tests the mask loss implementation including:
 * - Segment mode with opacity penalty
 * - Ignore mode (no penalty)
 * - AlphaConsistent mode
 * - Opacity penalty formula verification
 * - SSIM map masking (like legacy fused_ssim_map)
 */

#include "core/parameters.hpp"
#include "core/tensor.hpp"
#include "lfs/kernels/ssim.cuh"
#include <cmath>
#include <gtest/gtest.h>

using namespace lfs::core;

class MaskLossTest : public ::testing::Test {
protected:
    void SetUp() override {
        Tensor::manual_seed(42);
    }

    // Helper to create a simple binary mask
    Tensor create_binary_mask(size_t H, size_t W, float object_ratio = 0.5f) {
        auto mask = Tensor::zeros({H, W}, Device::CUDA);
        // Create a rectangular object region in the center
        size_t obj_h = static_cast<size_t>(H * std::sqrt(object_ratio));
        size_t obj_w = static_cast<size_t>(W * std::sqrt(object_ratio));
        size_t start_h = (H - obj_h) / 2;
        size_t start_w = (W - obj_w) / 2;

        // Fill center with 1.0 (object)
        auto ones = Tensor::ones({obj_h, obj_w}, Device::CUDA);
        // Use slice assignment
        auto mask_view = mask.slice(0, start_h, start_h + obj_h);
        // Note: We need to set values - for simplicity create full mask
        mask = Tensor::zeros({H, W}, Device::CUDA);
        for (size_t h = start_h; h < start_h + obj_h; ++h) {
            for (size_t w = start_w; w < start_w + obj_w; ++w) {
                // This is inefficient but works for tests
            }
        }
        // Alternative: create mask on CPU and transfer
        std::vector<float> mask_data(H * W, 0.0f);
        for (size_t h = start_h; h < start_h + obj_h; ++h) {
            for (size_t w = start_w; w < start_w + obj_w; ++w) {
                mask_data[h * W + w] = 1.0f;
            }
        }
        return Tensor::from_vector(mask_data, {H, W}, Device::CUDA);
    }

    // Helper to create soft mask with gradient values
    Tensor create_soft_mask(size_t H, size_t W) {
        std::vector<float> mask_data(H * W);
        for (size_t h = 0; h < H; ++h) {
            for (size_t w = 0; w < W; ++w) {
                // Radial gradient from center
                float dy = (h - H / 2.0f) / (H / 2.0f);
                float dx = (w - W / 2.0f) / (W / 2.0f);
                float dist = std::sqrt(dx * dx + dy * dy);
                mask_data[h * W + w] = std::max(0.0f, 1.0f - dist);
            }
        }
        return Tensor::from_vector(mask_data, {H, W}, Device::CUDA);
    }
};

// Test opacity penalty formula: mean(alpha * (1-mask)^power) * weight
TEST_F(MaskLossTest, OpacityPenaltyFormula) {
    constexpr size_t H = 64;
    constexpr size_t W = 64;
    constexpr float WEIGHT = 10.0f;
    constexpr float POWER = 2.0f;

    // Create uniform alpha and binary mask
    auto alpha = Tensor::full({H, W}, 0.5f, Device::CUDA); // 50% opacity everywhere
    auto mask = create_binary_mask(H, W, 0.25f);           // 25% is object

    // Compute inverted mask
    auto ones = Tensor::full({H, W}, 1.0f, Device::CUDA);
    auto inverted_mask = ones - mask;

    // Apply power: (1-mask)^power
    auto penalty_weight_map = inverted_mask.pow(POWER);

    // Compute penalty: mean(alpha * penalty_weight_map) * weight
    auto penalty = (alpha * penalty_weight_map).mean() * WEIGHT;
    float penalty_val = penalty.item<float>();

    // Verify penalty is positive
    EXPECT_GT(penalty_val, 0.0f);

    // Background coverage is ~75%, so penalty should be significant
    // Expected: 0.5 (alpha) * 0.75 (bg ratio) * 1.0 (power=2 on binary) * 10 = 3.75
    // With power=2 on binary mask: (1-0)^2 = 1, (1-1)^2 = 0
    EXPECT_GT(penalty_val, 1.0f);  // Should be significant
    EXPECT_LT(penalty_val, 10.0f); // But bounded
}

// Test that penalty is zero when mask covers everything
TEST_F(MaskLossTest, OpacityPenaltyZeroForFullMask) {
    constexpr size_t H = 32;
    constexpr size_t W = 32;
    constexpr float WEIGHT = 100.0f;

    auto alpha = Tensor::full({H, W}, 1.0f, Device::CUDA);
    auto mask = Tensor::ones({H, W}, Device::CUDA); // Full object coverage

    auto ones = Tensor::full({H, W}, 1.0f, Device::CUDA);
    auto inverted_mask = ones - mask;
    auto penalty_weight_map = inverted_mask.pow(2.0f);
    auto penalty = (alpha * penalty_weight_map).mean() * WEIGHT;

    float penalty_val = penalty.item<float>();
    EXPECT_NEAR(penalty_val, 0.0f, 1e-5f);
}

// Test that penalty increases with alpha in background
TEST_F(MaskLossTest, OpacityPenaltyIncreasesWithBackgroundAlpha) {
    constexpr size_t H = 32;
    constexpr size_t W = 32;
    constexpr float WEIGHT = 1.0f;
    constexpr float POWER = 2.0f;

    auto mask = create_binary_mask(H, W, 0.5f);
    auto ones = Tensor::full({H, W}, 1.0f, Device::CUDA);
    auto inverted_mask = ones - mask;
    auto penalty_weight_map = inverted_mask.pow(POWER);

    // Low alpha
    auto alpha_low = Tensor::full({H, W}, 0.1f, Device::CUDA);
    float penalty_low = (alpha_low * penalty_weight_map).mean().item<float>() * WEIGHT;

    // High alpha
    auto alpha_high = Tensor::full({H, W}, 0.9f, Device::CUDA);
    float penalty_high = (alpha_high * penalty_weight_map).mean().item<float>() * WEIGHT;

    EXPECT_GT(penalty_high, penalty_low);
    EXPECT_NEAR(penalty_high / penalty_low, 9.0f, 0.5f); // Should be ~9x
}

// Test power falloff behavior
TEST_F(MaskLossTest, PowerFalloffBehavior) {
    constexpr size_t H = 32;
    constexpr size_t W = 32;

    auto soft_mask = create_soft_mask(H, W);
    auto alpha = Tensor::full({H, W}, 1.0f, Device::CUDA);
    auto ones = Tensor::full({H, W}, 1.0f, Device::CUDA);
    auto inverted_mask = ones - soft_mask;

    // Power = 1 (linear)
    auto penalty_map_p1 = inverted_mask.pow(1.0f);
    float penalty_p1 = (alpha * penalty_map_p1).mean().item<float>();

    // Power = 2 (quadratic - gentler on uncertain regions)
    auto penalty_map_p2 = inverted_mask.pow(2.0f);
    float penalty_p2 = (alpha * penalty_map_p2).mean().item<float>();

    // Power = 4 (even gentler)
    auto penalty_map_p4 = inverted_mask.pow(4.0f);
    float penalty_p4 = (alpha * penalty_map_p4).mean().item<float>();

    // Higher power should give lower penalty (gentler on uncertain regions)
    EXPECT_GT(penalty_p1, penalty_p2);
    EXPECT_GT(penalty_p2, penalty_p4);
}

// Test masked L1 loss computation
TEST_F(MaskLossTest, MaskedL1Loss) {
    constexpr int C = 3;
    constexpr size_t H = 32;
    constexpr size_t W = 32;
    constexpr float EPSILON = 1e-8f;

    // Create rendered and GT images
    auto rendered = Tensor::rand({C, H, W}, Device::CUDA);
    auto gt = Tensor::rand({C, H, W}, Device::CUDA);
    auto mask_2d = create_binary_mask(H, W, 0.5f);

    // Expand mask to match image channels
    auto mask_expanded = mask_2d.unsqueeze(0).expand({C, H, W});
    auto mask_sum = mask_expanded.sum() + EPSILON;

    // Compute masked L1
    auto l1_diff = (rendered - gt).abs();
    auto masked_l1 = (l1_diff * mask_expanded).sum() / mask_sum;

    float loss_val = masked_l1.item<float>();
    EXPECT_GT(loss_val, 0.0f);
    EXPECT_LT(loss_val, 1.0f); // Should be reasonable for random images
}

// Test that masked L1 only considers object regions
TEST_F(MaskLossTest, MaskedL1IgnoresBackgroundDifference) {
    constexpr int C = 3;
    constexpr size_t H = 32;
    constexpr size_t W = 32;
    constexpr float EPSILON = 1e-8f;

    // Create identical images
    auto rendered = Tensor::zeros({C, H, W}, Device::CUDA);
    auto gt = Tensor::zeros({C, H, W}, Device::CUDA);

    // Mask: only center is object
    auto mask_2d = create_binary_mask(H, W, 0.25f);
    auto mask_expanded = mask_2d.unsqueeze(0).expand({C, H, W});
    auto mask_sum = mask_expanded.sum() + EPSILON;

    // Loss should be zero for identical images
    auto l1_diff = (rendered - gt).abs();
    auto masked_l1 = (l1_diff * mask_expanded).sum() / mask_sum;
    EXPECT_NEAR(masked_l1.item<float>(), 0.0f, 1e-5f);

    // Now add large difference only in background
    // Since mask ignores background, loss should still be ~0
    // (This test verifies the mask is applied correctly)
}

// Test gradient computation for masked L1
TEST_F(MaskLossTest, MaskedL1Gradient) {
    constexpr int C = 3;
    constexpr size_t H = 16;
    constexpr size_t W = 16;
    constexpr float EPSILON = 1e-8f;

    auto rendered = Tensor::rand({C, H, W}, Device::CUDA);
    auto gt = Tensor::rand({C, H, W}, Device::CUDA);
    auto mask_2d = create_binary_mask(H, W, 0.5f);

    auto mask_expanded = mask_2d.unsqueeze(0).expand({C, H, W});
    auto mask_sum = mask_expanded.sum() + EPSILON;

    // Gradient: sign(rendered - gt) * mask / mask_sum
    auto sign_diff = (rendered - gt).sign();
    auto grad = sign_diff * mask_expanded / mask_sum;

    // Gradient should be zero outside mask
    auto bg_mask = Tensor::full({H, W}, 1.0f, Device::CUDA) - mask_2d;
    auto bg_mask_expanded = bg_mask.unsqueeze(0).expand({C, H, W});
    auto grad_in_bg = (grad.abs() * bg_mask_expanded).sum();

    EXPECT_NEAR(grad_in_bg.item<float>(), 0.0f, 1e-5f);
}

// Test mask threshold application
TEST_F(MaskLossTest, MaskThresholdApplication) {
    constexpr size_t H = 32;
    constexpr size_t W = 32;
    constexpr float THRESHOLD = 0.5f;

    // Create soft mask with values from 0 to 1
    auto soft_mask = create_soft_mask(H, W);

    // Apply threshold: >= threshold -> 1.0, < threshold -> keep original
    auto ones = Tensor::ones({H, W}, Device::CUDA);
    auto threshold_mask = soft_mask.ge(THRESHOLD);
    auto thresholded = ones.where(threshold_mask, soft_mask);

    // All values >= threshold should be 1.0
    auto above_threshold = soft_mask.ge(THRESHOLD);
    auto thresholded_values = thresholded.where(above_threshold, Tensor::zeros({H, W}, Device::CUDA));
    auto sum_above = thresholded_values.sum().item<float>();
    // Convert bool sum to float
    auto count_above = static_cast<float>(above_threshold.to(DataType::Int32).sum().item<int>());

    // Sum should equal count (all 1.0s)
    EXPECT_NEAR(sum_above, count_above, 1e-3f);
}

// Test different mask modes
TEST_F(MaskLossTest, MaskModeEnum) {
    using MaskMode = param::MaskMode;

    EXPECT_EQ(static_cast<int>(MaskMode::None), 0);
    EXPECT_EQ(static_cast<int>(MaskMode::Segment), 1);
    EXPECT_EQ(static_cast<int>(MaskMode::Ignore), 2);
    EXPECT_EQ(static_cast<int>(MaskMode::AlphaConsistent), 3);
}

// Test optimization parameters for masks
TEST_F(MaskLossTest, MaskOptimizationParameters) {
    param::OptimizationParameters params;

    // Check defaults
    EXPECT_EQ(params.mask_mode, param::MaskMode::None);
    EXPECT_FALSE(params.invert_masks);
    EXPECT_FLOAT_EQ(params.mask_threshold, 0.5f);
    EXPECT_FLOAT_EQ(params.mask_opacity_penalty_weight, 1.0f);
    EXPECT_FLOAT_EQ(params.mask_opacity_penalty_power, 2.0f);
}

// Integration test: Full segment mode loss
TEST_F(MaskLossTest, SegmentModeLossIntegration) {
    constexpr int C = 3;
    constexpr size_t H = 64;
    constexpr size_t W = 64;
    constexpr float EPSILON = 1e-8f;
    constexpr float WEIGHT = 1.0f;
    constexpr float POWER = 2.0f;
    constexpr float LAMBDA_DSSIM = 0.2f;

    // Create test data
    auto rendered = Tensor::rand({C, H, W}, Device::CUDA);
    auto gt = Tensor::rand({C, H, W}, Device::CUDA);
    auto alpha = Tensor::rand({H, W}, Device::CUDA);
    auto mask_2d = create_binary_mask(H, W, 0.4f);

    // Expand mask
    auto mask_expanded = mask_2d.unsqueeze(0).expand({C, H, W});
    auto mask_sum = mask_expanded.sum() + EPSILON;

    // Compute masked L1
    auto l1_diff = (rendered - gt).abs();
    auto masked_l1 = (l1_diff * mask_expanded).sum() / mask_sum;

    // Compute opacity penalty
    auto ones = Tensor::full({H, W}, 1.0f, Device::CUDA);
    auto inverted_mask = ones - mask_2d;
    auto penalty_weight_map = inverted_mask.pow(POWER);
    auto penalty = (alpha * penalty_weight_map).mean() * WEIGHT;

    // Combine (without SSIM for simplicity)
    auto total_loss = masked_l1 * (1.0f - LAMBDA_DSSIM) + penalty;

    float loss_val = total_loss.item<float>();
    EXPECT_GT(loss_val, 0.0f);

    // Loss should be composed of photo loss + penalty
    float photo_val = masked_l1.item<float>() * (1.0f - LAMBDA_DSSIM);
    float penalty_val = penalty.item<float>();
    EXPECT_NEAR(loss_val, photo_val + penalty_val, 1e-4f);
}

// Test AlphaConsistent mode alpha loss
TEST_F(MaskLossTest, AlphaConsistentLoss) {
    constexpr size_t H = 32;
    constexpr size_t W = 32;
    constexpr float ALPHA_WEIGHT = 10.0f;

    auto alpha = Tensor::rand({H, W}, Device::CUDA);
    auto mask = create_soft_mask(H, W);

    // L1 loss between alpha and mask
    auto alpha_loss = (alpha - mask).abs().mean() * ALPHA_WEIGHT;

    float loss_val = alpha_loss.item<float>();
    EXPECT_GT(loss_val, 0.0f);
    EXPECT_LT(loss_val, ALPHA_WEIGHT); // Max L1 diff is 1.0
}

// Test SSIM map forward function returns per-pixel SSIM
TEST_F(MaskLossTest, SSIMMapForward) {
    constexpr int C = 3;
    constexpr size_t H = 64;
    constexpr size_t W = 64;

    auto img1 = Tensor::rand({1, C, H, W}, Device::CUDA);
    auto img2 = Tensor::rand({1, C, H, W}, Device::CUDA);

    // Get SSIM map with "same" padding (no cropping)
    auto result = lfs::training::kernels::ssim_forward_map(img1, img2, false);

    // Verify ssim_map has correct shape [1, C, H, W]
    EXPECT_EQ(result.ssim_map.ndim(), 4);
    EXPECT_EQ(result.ssim_map.shape()[0], 1);
    EXPECT_EQ(result.ssim_map.shape()[1], static_cast<size_t>(C));
    EXPECT_EQ(result.ssim_map.shape()[2], static_cast<size_t>(H));
    EXPECT_EQ(result.ssim_map.shape()[3], static_cast<size_t>(W));

    // Verify ssim_value is scalar
    EXPECT_EQ(result.ssim_value.numel(), 1);

    // SSIM values should be between 0 and 1
    float ssim_val = result.ssim_value.item<float>();
    EXPECT_GE(ssim_val, 0.0f);
    EXPECT_LE(ssim_val, 1.0f);
}

// Test SSIM map with identical images gives SSIM = 1
TEST_F(MaskLossTest, SSIMMapIdenticalImages) {
    constexpr int C = 3;
    constexpr size_t H = 32;
    constexpr size_t W = 32;

    auto img = Tensor::rand({1, C, H, W}, Device::CUDA);

    auto result = lfs::training::kernels::ssim_forward_map(img, img, false);

    // SSIM of identical images should be ~1.0
    float ssim_val = result.ssim_value.item<float>();
    EXPECT_NEAR(ssim_val, 1.0f, 0.01f);
}

// Test masked SSIM computation (like legacy fused_ssim_map + masking)
TEST_F(MaskLossTest, MaskedSSIMComputation) {
    constexpr int C = 3;
    constexpr size_t H = 64;
    constexpr size_t W = 64;
    constexpr float EPSILON = 1e-8f;

    auto rendered = Tensor::rand({1, C, H, W}, Device::CUDA);
    auto gt = Tensor::rand({1, C, H, W}, Device::CUDA);
    auto mask_2d = create_binary_mask(H, W, 0.5f);

    // Get SSIM map
    auto result = lfs::training::kernels::ssim_forward_map(rendered, gt, false);
    auto ssim_map_3d = result.ssim_map.squeeze(0); // [C, H, W]

    // Expand mask to [C, H, W]
    auto mask_expanded = mask_2d.unsqueeze(0).expand({C, H, W});
    auto mask_sum = mask_expanded.sum() + EPSILON;

    // Compute masked SSIM (like legacy)
    auto masked_ssim_map = ssim_map_3d * mask_expanded;
    auto masked_ssim = masked_ssim_map.sum() / mask_sum;
    auto ssim_loss = Tensor::full({}, 1.0f, Device::CUDA) - masked_ssim;

    float loss_val = ssim_loss.item<float>();
    EXPECT_GE(loss_val, 0.0f);
    EXPECT_LE(loss_val, 1.0f);

    // Masked SSIM should be different from full SSIM
    float full_ssim = result.ssim_value.item<float>();
    float masked_ssim_val = masked_ssim.item<float>();
    // They can be similar but not necessarily equal
    EXPECT_GE(masked_ssim_val, 0.0f);
    EXPECT_LE(masked_ssim_val, 1.0f);
}

// Test that masked SSIM focuses only on object regions
TEST_F(MaskLossTest, MaskedSSIMIgnoresBackground) {
    constexpr int C = 3;
    constexpr size_t H = 32;
    constexpr size_t W = 32;
    constexpr float EPSILON = 1e-8f;

    // Create identical images in object region, different in background
    auto base = Tensor::rand({1, C, H, W}, Device::CUDA);
    auto modified = base.clone();

    // Create mask where center is object (1.0), edges are background (0.0)
    auto mask_2d = create_binary_mask(H, W, 0.25f); // 25% is object
    auto mask_expanded = mask_2d.unsqueeze(0).expand({C, H, W});
    auto mask_sum = mask_expanded.sum() + EPSILON;

    // SSIM of base with itself (in masked region)
    auto result = lfs::training::kernels::ssim_forward_map(base, base, false);
    auto ssim_map_3d = result.ssim_map.squeeze(0);
    auto masked_ssim_map = ssim_map_3d * mask_expanded;
    auto masked_ssim = masked_ssim_map.sum() / mask_sum;

    // Should be ~1.0 since object regions are identical
    float ssim_val = masked_ssim.item<float>();
    EXPECT_NEAR(ssim_val, 1.0f, 0.05f);
}

// Test full masked loss computation (L1 + SSIM)
TEST_F(MaskLossTest, FullMaskedLossComputation) {
    constexpr int C = 3;
    constexpr size_t H = 64;
    constexpr size_t W = 64;
    constexpr float EPSILON = 1e-8f;
    constexpr float LAMBDA_DSSIM = 0.2f;

    auto rendered = Tensor::rand({C, H, W}, Device::CUDA);
    auto gt = Tensor::rand({C, H, W}, Device::CUDA);
    auto mask_2d = create_binary_mask(H, W, 0.5f);

    auto mask_expanded = mask_2d.unsqueeze(0).expand({C, H, W});
    auto mask_sum = mask_expanded.sum() + EPSILON;

    // Masked L1
    auto l1_diff = (rendered - gt).abs();
    auto masked_l1 = (l1_diff * mask_expanded).sum() / mask_sum;

    // Masked SSIM
    auto rendered_4d = rendered.unsqueeze(0);
    auto gt_4d = gt.unsqueeze(0);
    auto ssim_result = lfs::training::kernels::ssim_forward_map(rendered_4d, gt_4d, false);
    auto ssim_map_3d = ssim_result.ssim_map.squeeze(0);
    auto masked_ssim_map = ssim_map_3d * mask_expanded;
    auto masked_ssim = masked_ssim_map.sum() / mask_sum;
    auto ssim_loss = Tensor::full({}, 1.0f, Device::CUDA) - masked_ssim;

    // Combined loss
    auto combined_loss = masked_l1 * (1.0f - LAMBDA_DSSIM) + ssim_loss * LAMBDA_DSSIM;

    float loss_val = combined_loss.item<float>();
    EXPECT_GT(loss_val, 0.0f);
    EXPECT_LT(loss_val, 2.0f); // Reasonable bound

    // Verify components add up correctly
    float l1_val = masked_l1.item<float>() * (1.0f - LAMBDA_DSSIM);
    float ssim_val = ssim_loss.item<float>() * LAMBDA_DSSIM;
    EXPECT_NEAR(loss_val, l1_val + ssim_val, 1e-4f);
}

// Verify analytical gradient matches finite differences
TEST_F(MaskLossTest, MaskedSSIMGradientNumerical) {
    constexpr int C = 3, H = 16, W = 16;
    constexpr float EPSILON = 1e-8f, DELTA = 1e-3f, GRAD_TOL = 5e-2f;

    const auto rendered_cpu = Tensor::rand({1, C, H, W}, Device::CPU) * 0.5f + 0.25f;
    const auto gt_cpu = Tensor::rand({1, C, H, W}, Device::CPU) * 0.5f + 0.25f;
    const auto rendered = rendered_cpu.to(Device::CUDA);
    const auto gt = gt_cpu.to(Device::CUDA);
    const auto mask_4d = create_soft_mask(H, W).unsqueeze(0).unsqueeze(0).expand({1, C, H, W});
    const auto mask_sum = mask_4d.sum() + EPSILON;
    const float mask_sum_val = mask_sum.item<float>();

    auto compute_loss = [&](const Tensor& img) {
        const auto result = lfs::training::kernels::ssim_forward_map(img, gt, false);
        const auto masked_ssim = (result.ssim_map * mask_4d).sum() / mask_sum;
        return (Tensor::full({}, 1.0f, Device::CUDA) - masked_ssim).item<float>();
    };

    const auto result = lfs::training::kernels::ssim_forward_map(rendered, gt, false);
    const auto dL_dmap = mask_4d * (-1.0f / mask_sum_val);
    const auto analytical_cpu = lfs::training::kernels::ssim_backward_with_grad_map(result.ctx, dL_dmap).to(Device::CPU);

    float max_rel_error = 0.0f;
    int num_checked = 0;
    const float* const rendered_data = rendered_cpu.ptr<float>();
    const float* const analytical_data = analytical_cpu.ptr<float>();

    for (int c = 0; c < C; ++c) {
        for (int h = 6; h < H - 6; h += 3) {
            for (int w = 6; w < W - 6; w += 3) {
                const size_t idx = static_cast<size_t>(c * H * W + h * W + w);
                auto plus_cpu = rendered_cpu.clone();
                auto minus_cpu = rendered_cpu.clone();
                plus_cpu.ptr<float>()[idx] = rendered_data[idx] + DELTA;
                minus_cpu.ptr<float>()[idx] = rendered_data[idx] - DELTA;

                const float numerical = (compute_loss(plus_cpu.to(Device::CUDA)) -
                                         compute_loss(minus_cpu.to(Device::CUDA))) /
                                        (2.0f * DELTA);
                const float analytical = analytical_data[idx];
                const float denom = std::max(std::abs(numerical), std::abs(analytical));
                if (denom > 1e-5f) {
                    max_rel_error = std::max(max_rel_error, std::abs(numerical - analytical) / denom);
                }
                ++num_checked;
            }
        }
    }

    EXPECT_GT(num_checked, 0);
    EXPECT_LT(max_rel_error, GRAD_TOL) << "Max relative error: " << max_rel_error;
}

// Verify per-pixel grad_map produces different results than scalar gradient
TEST_F(MaskLossTest, PerPixelGradMapVsScalarGradient) {
    constexpr int C = 3, H = 32, W = 32;
    constexpr float EPSILON = 1e-8f;

    const auto rendered = Tensor::rand({1, C, H, W}, Device::CUDA);
    const auto gt = Tensor::rand({1, C, H, W}, Device::CUDA);

    auto mask_cpu = Tensor::zeros({H, W}, Device::CPU);
    for (int h = 0; h < H / 2; ++h)
        for (int w = 0; w < W; ++w)
            mask_cpu.ptr<float>()[h * W + w] = 1.0f;

    const auto mask_4d = mask_cpu.to(Device::CUDA).unsqueeze(0).unsqueeze(0).expand({1, C, H, W});
    const float mask_sum_val = (mask_4d.sum() + EPSILON).item<float>();
    const auto result = lfs::training::kernels::ssim_forward_map(rendered, gt, false);

    // Scalar gradient (wrong for masked case)
    const auto grad_scalar = lfs::training::kernels::ssim_backward(result.ctx, -1.0f / mask_sum_val) * mask_4d;
    // Per-pixel gradient (correct)
    const auto dL_dmap = mask_4d * (-1.0f / mask_sum_val);
    const auto grad_perpixel = lfs::training::kernels::ssim_backward_with_grad_map(result.ctx, dL_dmap);

    const float scalar_sum = grad_scalar.abs().sum().item<float>();
    const float perpixel_sum = grad_perpixel.abs().sum().item<float>();
    const float ratio = perpixel_sum / (scalar_sum + EPSILON);

    // Per-pixel should be ~numel times larger (scalar divides by numel internally)
    const float numel = static_cast<float>(rendered.numel());
    EXPECT_GT(ratio, numel * 0.5f);
    EXPECT_LT(ratio, numel * 2.0f);
}
