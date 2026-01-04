/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_fused_l1_ssim.cpp
 * @brief Tests for fused L1+SSIM loss kernel
 *
 * Verifies correctness by comparing fused kernel output against reference
 * implementations that compute L1 and SSIM separately.
 */

#include <gtest/gtest.h>

#include "core/tensor.hpp"
#include "lfs/kernels/l1_loss.cuh"
#include "lfs/kernels/ssim.cuh"
#include "training/losses/photometric_loss.hpp"
#include <cmath>
#include <cuda_runtime.h>

using namespace lfs::core;
using namespace lfs::training::kernels;

class FusedL1SSIMTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure CUDA is available
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA device available";
        }
    }

    // Reference implementation: compute L1 + SSIM loss correctly
    // IMPORTANT: The fused kernel computes PER-PIXEL combined loss for ALL pixels,
    // then crops to valid region (5 pixels from each edge) before taking mean.
    // loss_map[i] = l1_weight * |img1[i] - img2[i]| + ssim_weight * (1 - SSIM[i])
    // total_loss = mean(crop(loss_map))
    std::pair<float, Tensor> compute_reference_loss_and_grad(
        const Tensor& img1, const Tensor& img2, float ssim_weight, bool apply_valid_padding = true) {

        auto img1_4d = img1.ndim() == 3 ? img1.unsqueeze(0) : img1;
        auto img2_4d = img2.ndim() == 3 ? img2.unsqueeze(0) : img2;

        const float l1_weight = 1.0f - ssim_weight;
        const int H = static_cast<int>(img1_4d.shape()[2]);
        const int W = static_cast<int>(img1_4d.shape()[3]);

        // Get per-pixel L1 loss (full image)
        auto l1_map = (img1_4d - img2_4d).abs();  // [N, C, H, W]

        // Get per-pixel SSIM map for FULL image (no cropping in ssim_forward_map)
        auto ssim_result = ssim_forward_map(img1_4d, img2_4d, /*apply_valid_padding=*/false);
        auto ssim_map = ssim_result.ssim_map;  // [N, C, H, W]

        // Compute per-pixel combined loss map (full image)
        auto dssim_map = Tensor::ones(TensorShape(ssim_map.shape().dims()), Device::CUDA) - ssim_map;
        auto combined_loss_map = l1_map * l1_weight + dssim_map * ssim_weight;

        // Apply valid padding by cropping before mean (same as fused kernel)
        Tensor loss_region;
        size_t numel_for_grad;
        if (apply_valid_padding && H > 10 && W > 10) {
            loss_region = combined_loss_map.slice(2, 5, H - 5).slice(3, 5, W - 5);
            numel_for_grad = loss_region.numel();
        } else {
            loss_region = combined_loss_map;
            numel_for_grad = combined_loss_map.numel();
        }

        // Total loss is the mean of the (cropped) loss map
        Tensor loss_mean = loss_region.mean();
        float combined_loss = loss_mean.item<float>();

        // Gradient computation:
        // The fused kernel backward:
        // 1. Creates dL_dmap with 1/numel in valid region, 0 elsewhere
        // 2. Computes SSIM gradient via convolution
        // 3. Adds L1 gradient (sign * l1_weight * dL_dmap)

        // Create gradient map matching fused kernel's approach
        auto dL_dmap = Tensor::zeros(combined_loss_map.shape(), Device::CUDA);
        float grad_per_pixel = 1.0f / static_cast<float>(numel_for_grad);
        if (apply_valid_padding && H > 10 && W > 10) {
            auto cropped = dL_dmap.slice(2, 5, H - 5).slice(3, 5, W - 5);
            cropped.fill_(grad_per_pixel, nullptr);
        } else {
            dL_dmap.fill_(grad_per_pixel, nullptr);
        }
        cudaDeviceSynchronize();

        // L1 gradient: sign(img1 - img2) * l1_weight * dL_dmap
        auto sign_diff = (img1_4d - img2_4d).sign();
        auto l1_grad = sign_diff * l1_weight * dL_dmap;

        // SSIM gradient: need to backprop with -ssim_weight (since loss = 1 - ssim)
        auto ssim_dL_dmap = dL_dmap * (-ssim_weight);
        auto ssim_grad = ssim_backward_with_grad_map(ssim_result.ctx, ssim_dL_dmap);

        auto combined_grad = l1_grad + ssim_grad;

        return std::make_pair(combined_loss, combined_grad);
    }
};

// Test basic fused forward correctness
TEST_F(FusedL1SSIMTest, ForwardMatchesReference) {
    const int N = 1, C = 3, H = 64, W = 64;
    auto img1 = Tensor::randn({N, C, H, W}, Device::CUDA);
    auto img2 = Tensor::randn({N, C, H, W}, Device::CUDA);

    const float ssim_weight = 0.2f;

    // Fused kernel
    FusedL1SSIMWorkspace workspace;
    auto [fused_loss, fused_ctx] = fused_l1_ssim_forward(img1, img2, ssim_weight, workspace, true);
    float fused_loss_value = fused_loss.item<float>();

    // Reference
    auto [ref_loss_value, ref_grad] = compute_reference_loss_and_grad(img1, img2, ssim_weight);

    float diff = std::abs(fused_loss_value - ref_loss_value);
    float rel_diff = diff / ref_loss_value * 100.0f;
    std::cout << "64x64: Fused=" << fused_loss_value << " Ref=" << ref_loss_value
              << " Diff=" << diff << " (" << rel_diff << "%)" << std::endl;

    // Compare loss values (allow small floating point tolerance)
    EXPECT_NEAR(fused_loss_value, ref_loss_value, 1e-4f)
        << "Fused loss: " << fused_loss_value << ", Reference: " << ref_loss_value;
}

// Test backward correctness
TEST_F(FusedL1SSIMTest, BackwardMatchesReference) {
    const int N = 1, C = 3, H = 64, W = 64;
    auto img1 = Tensor::randn({N, C, H, W}, Device::CUDA);
    auto img2 = Tensor::randn({N, C, H, W}, Device::CUDA);

    const float ssim_weight = 0.2f;

    // Fused kernel
    FusedL1SSIMWorkspace workspace;
    auto [fused_loss, fused_ctx] = fused_l1_ssim_forward(img1, img2, ssim_weight, workspace, true);
    auto fused_grad = fused_l1_ssim_backward(fused_ctx, workspace);

    // Reference
    auto [ref_loss, ref_grad] = compute_reference_loss_and_grad(img1, img2, ssim_weight);

    // Compare gradients
    auto diff = (fused_grad - ref_grad).abs();
    float max_diff = diff.max().item<float>();
    float mean_diff = diff.mean().item<float>();

    std::cout << "Gradient: MaxDiff=" << max_diff << " MeanDiff=" << mean_diff << std::endl;

    EXPECT_LT(max_diff, 1e-3f) << "Max gradient difference: " << max_diff;
    EXPECT_LT(mean_diff, 1e-5f) << "Mean gradient difference: " << mean_diff;
}

// Test various SSIM weights
TEST_F(FusedL1SSIMTest, VariousSSIMWeights) {
    const int N = 1, C = 3, H = 64, W = 64;
    auto img1 = Tensor::randn({N, C, H, W}, Device::CUDA);
    auto img2 = Tensor::randn({N, C, H, W}, Device::CUDA);

    std::vector<float> weights = {0.1f, 0.2f, 0.5f, 0.8f, 0.9f};

    for (float ssim_weight : weights) {
        FusedL1SSIMWorkspace workspace;
        auto [fused_loss, fused_ctx] = fused_l1_ssim_forward(img1, img2, ssim_weight, workspace, true);
        float fused_loss_value = fused_loss.item<float>();

        auto [ref_loss_value, ref_grad] = compute_reference_loss_and_grad(img1, img2, ssim_weight);

        EXPECT_NEAR(fused_loss_value, ref_loss_value, 1e-4f)
            << "Failed for ssim_weight=" << ssim_weight;
    }
}

// Test with valid padding disabled
TEST_F(FusedL1SSIMTest, NoValidPadding) {
    const int N = 1, C = 3, H = 64, W = 64;
    auto img1 = Tensor::randn({N, C, H, W}, Device::CUDA);
    auto img2 = Tensor::randn({N, C, H, W}, Device::CUDA);

    const float ssim_weight = 0.2f;

    FusedL1SSIMWorkspace workspace;
    auto [loss, ctx] = fused_l1_ssim_forward(img1, img2, ssim_weight, workspace, false);
    auto grad = fused_l1_ssim_backward(ctx, workspace);

    // Just verify no crash and reasonable output
    EXPECT_FALSE(std::isnan(loss.item<float>()));
    EXPECT_FALSE(std::isinf(loss.item<float>()));
    EXPECT_GT(loss.item<float>(), 0.0f);

    float grad_max = grad.abs().max().item<float>();
    EXPECT_FALSE(std::isnan(grad_max));
    EXPECT_FALSE(std::isinf(grad_max));
}

// Test 3D input (no batch dimension)
TEST_F(FusedL1SSIMTest, ThreeDimensionalInput) {
    const int C = 3, H = 64, W = 64;
    auto img1 = Tensor::randn({C, H, W}, Device::CUDA);
    auto img2 = Tensor::randn({C, H, W}, Device::CUDA);

    const float ssim_weight = 0.2f;

    FusedL1SSIMWorkspace workspace;
    auto [loss, ctx] = fused_l1_ssim_forward(img1, img2, ssim_weight, workspace, true);
    auto grad = fused_l1_ssim_backward(ctx, workspace);

    EXPECT_FALSE(std::isnan(loss.item<float>()));
    EXPECT_EQ(grad.ndim(), 4); // Should be expanded to 4D
}

// Test larger image sizes
TEST_F(FusedL1SSIMTest, LargerImageSize) {
    const int N = 1, C = 3, H = 256, W = 256;
    auto img1 = Tensor::randn({N, C, H, W}, Device::CUDA);
    auto img2 = Tensor::randn({N, C, H, W}, Device::CUDA);

    const float ssim_weight = 0.2f;

    FusedL1SSIMWorkspace workspace;
    auto [loss, ctx] = fused_l1_ssim_forward(img1, img2, ssim_weight, workspace, true);
    auto grad = fused_l1_ssim_backward(ctx, workspace);

    auto [ref_loss, ref_grad] = compute_reference_loss_and_grad(img1, img2, ssim_weight);

    EXPECT_NEAR(loss.item<float>(), ref_loss, 1e-4f);
}

// Test identical images (loss should be 0)
TEST_F(FusedL1SSIMTest, IdenticalImages) {
    const int N = 1, C = 3, H = 64, W = 64;
    auto img = Tensor::randn({N, C, H, W}, Device::CUDA);

    const float ssim_weight = 0.2f;

    FusedL1SSIMWorkspace workspace;
    auto [loss, ctx] = fused_l1_ssim_forward(img, img, ssim_weight, workspace, true);

    // L1 = 0, SSIM = 1, so loss = (1-w)*0 + w*(1-1) = 0
    EXPECT_NEAR(loss.item<float>(), 0.0f, 1e-5f);
}

// Test workspace reuse
TEST_F(FusedL1SSIMTest, WorkspaceReuse) {
    const int N = 1, C = 3, H = 64, W = 64;
    const float ssim_weight = 0.2f;

    FusedL1SSIMWorkspace workspace;

    // First call
    auto img1a = Tensor::randn({N, C, H, W}, Device::CUDA);
    auto img2a = Tensor::randn({N, C, H, W}, Device::CUDA);
    auto [loss1, ctx1] = fused_l1_ssim_forward(img1a, img2a, ssim_weight, workspace, true);
    auto grad1 = fused_l1_ssim_backward(ctx1, workspace);

    // Second call with same workspace
    auto img1b = Tensor::randn({N, C, H, W}, Device::CUDA);
    auto img2b = Tensor::randn({N, C, H, W}, Device::CUDA);
    auto [loss2, ctx2] = fused_l1_ssim_forward(img1b, img2b, ssim_weight, workspace, true);
    auto grad2 = fused_l1_ssim_backward(ctx2, workspace);

    // Both should produce valid results
    EXPECT_FALSE(std::isnan(loss1.item<float>()));
    EXPECT_FALSE(std::isnan(loss2.item<float>()));

    // Results should be different since inputs differ
    EXPECT_NE(loss1.item<float>(), loss2.item<float>());
}

// Test PhotometricLoss uses fused kernel
TEST_F(FusedL1SSIMTest, PhotometricLossUsesFusedKernel) {
    const int C = 3, H = 64, W = 64;
    auto rendered = Tensor::randn({C, H, W}, Device::CUDA);
    auto gt = Tensor::randn({C, H, W}, Device::CUDA);

    lfs::training::losses::PhotometricLoss loss_fn;
    lfs::training::losses::PhotometricLoss::Params params{.lambda_dssim = 0.2f};

    auto result = loss_fn.forward(rendered, gt, params);
    ASSERT_TRUE(result.has_value());

    auto [loss, ctx] = *result;
    EXPECT_FALSE(std::isnan(loss.item<float>()));
    EXPECT_FALSE(std::isnan(ctx.grad_image.abs().max().item<float>()));
}

// ============================================================================
// Masked Fused L1+SSIM Tests
// ============================================================================

class MaskedFusedL1SSIMTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA device available";
        }
    }

    // Reference implementation for masked loss
    std::pair<float, Tensor> compute_reference_masked_loss(
        const Tensor& img1, const Tensor& img2, const Tensor& mask, float ssim_weight) {

        constexpr float EPSILON = lfs::training::kernels::SSIM_EPSILON;

        auto img1_4d = img1.ndim() == 3 ? img1.unsqueeze(0) : img1;
        auto img2_4d = img2.ndim() == 3 ? img2.unsqueeze(0) : img2;

        const int N = static_cast<int>(img1_4d.shape()[0]);
        const int C = static_cast<int>(img1_4d.shape()[1]);
        const int H = static_cast<int>(img1_4d.shape()[2]);
        const int W = static_cast<int>(img1_4d.shape()[3]);

        const float l1_weight = 1.0f - ssim_weight;

        // Expand mask to [N, C, H, W]
        auto mask_2d = mask.ndim() == 3 ? mask.squeeze(0) : mask;
        auto mask_expanded = mask_2d.unsqueeze(0).unsqueeze(0).expand({N, C, H, W});
        auto mask_sum = mask_expanded.sum() + EPSILON;

        // Masked L1 loss
        auto l1_diff = (img1_4d - img2_4d).abs();
        auto masked_l1_loss = ((l1_diff * mask_expanded).sum() / mask_sum).item<float>();

        // Masked L1 gradient
        auto sign_diff = (img1_4d - img2_4d).sign();
        auto l1_grad = sign_diff * mask_expanded / mask_sum;

        // Masked SSIM
        auto ssim_result = ssim_forward_map(img1_4d, img2_4d, false);
        auto ssim_map = ssim_result.ssim_map;
        auto masked_ssim = ((ssim_map * mask_expanded).sum() / mask_sum).item<float>();
        float ssim_loss = 1.0f - masked_ssim;

        // SSIM gradient
        auto dL_dmap = mask_expanded * (-1.0f) / mask_sum;
        auto ssim_grad = ssim_backward_with_grad_map(ssim_result.ctx, dL_dmap);

        // Combined
        float combined_loss = l1_weight * masked_l1_loss + ssim_weight * ssim_loss;
        auto combined_grad = l1_grad * l1_weight + ssim_grad * ssim_weight;

        return {combined_loss, combined_grad};
    }
};

TEST_F(MaskedFusedL1SSIMTest, ForwardBasic) {
    const int N = 1, C = 3, H = 64, W = 64;
    auto img1 = Tensor::randn({N, C, H, W}, Device::CUDA);
    auto img2 = Tensor::randn({N, C, H, W}, Device::CUDA);

    // Create a mask with some regions masked out
    auto mask = Tensor::ones({H, W}, Device::CUDA);
    // Mask out a region
    auto mask_view = mask.slice(0, 0, H / 2).slice(1, 0, W / 2);
    mask_view.fill_(0.0f, nullptr);
    cudaDeviceSynchronize();

    const float ssim_weight = 0.2f;

    MaskedFusedL1SSIMWorkspace workspace;
    auto [loss, ctx] = masked_fused_l1_ssim_forward(img1, img2, mask, ssim_weight, workspace);

    EXPECT_FALSE(std::isnan(loss.item<float>()));
    EXPECT_FALSE(std::isinf(loss.item<float>()));
    EXPECT_GT(loss.item<float>(), 0.0f);
}

TEST_F(MaskedFusedL1SSIMTest, BackwardBasic) {
    const int N = 1, C = 3, H = 64, W = 64;
    auto img1 = Tensor::randn({N, C, H, W}, Device::CUDA);
    auto img2 = Tensor::randn({N, C, H, W}, Device::CUDA);
    auto mask = Tensor::ones({H, W}, Device::CUDA);

    const float ssim_weight = 0.2f;

    MaskedFusedL1SSIMWorkspace workspace;
    auto [loss, ctx] = masked_fused_l1_ssim_forward(img1, img2, mask, ssim_weight, workspace);
    auto grad = masked_fused_l1_ssim_backward(ctx, workspace);

    EXPECT_EQ(grad.shape()[0], N);
    EXPECT_EQ(grad.shape()[1], C);
    EXPECT_EQ(grad.shape()[2], H);
    EXPECT_EQ(grad.shape()[3], W);

    float grad_max = grad.abs().max().item<float>();
    EXPECT_FALSE(std::isnan(grad_max));
    EXPECT_FALSE(std::isinf(grad_max));

    auto [ref_loss, ref_grad] = compute_reference_masked_loss(img1, img2, mask, ssim_weight);

    auto diff = (grad - ref_grad).abs();
    float max_diff = diff.max().item<float>();
    float mean_diff = diff.mean().item<float>();

    EXPECT_LT(max_diff, 1e-2f) << "Max gradient difference: " << max_diff;
    EXPECT_LT(mean_diff, 1e-4f) << "Mean gradient difference: " << mean_diff;
}

TEST_F(MaskedFusedL1SSIMTest, FullMaskMatchesReference) {
    const int N = 1, C = 3, H = 64, W = 64;
    auto img1 = Tensor::randn({N, C, H, W}, Device::CUDA);
    auto img2 = Tensor::randn({N, C, H, W}, Device::CUDA);
    auto mask = Tensor::ones({H, W}, Device::CUDA);

    const float ssim_weight = 0.2f;

    // Fused masked kernel
    MaskedFusedL1SSIMWorkspace workspace;
    auto [fused_loss, fused_ctx] = masked_fused_l1_ssim_forward(img1, img2, mask, ssim_weight, workspace);

    // Reference
    auto [ref_loss, ref_grad] = compute_reference_masked_loss(img1, img2, mask, ssim_weight);

    // Compare (tolerance is higher due to different computation order)
    EXPECT_NEAR(fused_loss.item<float>(), ref_loss, 1e-3f)
        << "Fused: " << fused_loss.item<float>() << ", Reference: " << ref_loss;
}

TEST_F(MaskedFusedL1SSIMTest, PartialMask) {
    const int N = 1, C = 3, H = 64, W = 64;
    auto img1 = Tensor::randn({N, C, H, W}, Device::CUDA);
    auto img2 = Tensor::randn({N, C, H, W}, Device::CUDA);

    // Create checkerboard mask
    auto mask = Tensor::zeros({H, W}, Device::CUDA);
    for (int y = 0; y < H; y += 2) {
        for (int x = 0; x < W; x += 2) {
            auto pixel = mask.slice(0, y, y + 1).slice(1, x, x + 1);
            pixel.fill_(1.0f, nullptr);
        }
    }
    cudaDeviceSynchronize();

    const float ssim_weight = 0.2f;

    MaskedFusedL1SSIMWorkspace workspace;
    auto [loss, ctx] = masked_fused_l1_ssim_forward(img1, img2, mask, ssim_weight, workspace);
    auto grad = masked_fused_l1_ssim_backward(ctx, workspace);

    EXPECT_FALSE(std::isnan(loss.item<float>()));
    EXPECT_FALSE(std::isnan(grad.abs().max().item<float>()));
}

TEST_F(MaskedFusedL1SSIMTest, AllZeroMask) {
    const int N = 1, C = 3, H = 64, W = 64;
    auto img1 = Tensor::randn({N, C, H, W}, Device::CUDA);
    auto img2 = Tensor::randn({N, C, H, W}, Device::CUDA);
    auto mask = Tensor::zeros({H, W}, Device::CUDA);

    const float ssim_weight = 0.2f;

    MaskedFusedL1SSIMWorkspace workspace;
    auto [loss, ctx] = masked_fused_l1_ssim_forward(img1, img2, mask, ssim_weight, workspace);

    // With all-zero mask, loss should be ~0 (or very small due to epsilon)
    EXPECT_LT(loss.item<float>(), 1e-5f);
}

TEST_F(MaskedFusedL1SSIMTest, WorkspaceReuse) {
    const int N = 1, C = 3, H = 64, W = 64;
    const float ssim_weight = 0.2f;
    auto mask = Tensor::ones({H, W}, Device::CUDA);

    MaskedFusedL1SSIMWorkspace workspace;

    // Multiple calls with same workspace
    for (int i = 0; i < 3; ++i) {
        auto img1 = Tensor::randn({N, C, H, W}, Device::CUDA);
        auto img2 = Tensor::randn({N, C, H, W}, Device::CUDA);

        auto [loss, ctx] = masked_fused_l1_ssim_forward(img1, img2, mask, ssim_weight, workspace);
        auto grad = masked_fused_l1_ssim_backward(ctx, workspace);

        EXPECT_FALSE(std::isnan(loss.item<float>()));
        EXPECT_FALSE(std::isnan(grad.abs().max().item<float>()));
    }
}

// ============================================================================
// Performance benchmark (not a correctness test)
// ============================================================================

TEST_F(FusedL1SSIMTest, DISABLED_PerformanceBenchmark) {
    const int N = 1, C = 3, H = 1080, W = 1920;
    const int iterations = 100;
    const float ssim_weight = 0.2f;

    auto img1 = Tensor::randn({N, C, H, W}, Device::CUDA);
    auto img2 = Tensor::randn({N, C, H, W}, Device::CUDA);

    // Warmup
    {
        FusedL1SSIMWorkspace workspace;
        for (int i = 0; i < 10; ++i) {
            auto [loss, ctx] = fused_l1_ssim_forward(img1, img2, ssim_weight, workspace, true);
            auto grad = fused_l1_ssim_backward(ctx, workspace);
        }
        cudaDeviceSynchronize();
    }

    // Benchmark fused kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    FusedL1SSIMWorkspace fused_workspace;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        auto [loss, ctx] = fused_l1_ssim_forward(img1, img2, ssim_weight, fused_workspace, true);
        auto grad = fused_l1_ssim_backward(ctx, fused_workspace);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float fused_time;
    cudaEventElapsedTime(&fused_time, start, stop);

    // Benchmark separate kernels
    SSIMWorkspace ssim_workspace;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        // L1
        auto l1_diff = (img1 - img2).abs();
        auto l1_loss = l1_diff.mean();
        auto sign_diff = (img1 - img2).sign();
        auto l1_grad = sign_diff / static_cast<float>(img1.numel());

        // SSIM
        auto [ssim_val, ssim_ctx] = ssim_forward(img1, img2, ssim_workspace, true);
        auto ssim_grad = ssim_backward(ssim_ctx, ssim_workspace, -1.0f);

        // Combine
        auto grad = l1_grad * (1.0f - ssim_weight) + ssim_grad * ssim_weight;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float separate_time;
    cudaEventElapsedTime(&separate_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    float speedup = separate_time / fused_time;
    std::cout << "Fused kernel time: " << fused_time / iterations << " ms/iter\n";
    std::cout << "Separate kernels time: " << separate_time / iterations << " ms/iter\n";
    std::cout << "Speedup: " << speedup << "x\n";

    EXPECT_GT(speedup, 1.2f);
}
