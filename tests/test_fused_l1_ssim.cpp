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
#include "core/tensor_upload.hpp"
#include "cuda_backend_test.hpp"
#include "lfs/kernels/l1_loss.cuh"
#include "lfs/kernels/ssim.cuh"
#include "lfs/training/ops/photometric_cuda.hpp"
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <limits>
#include <thread>
#include <vector>

using namespace lfs::core;
using namespace lfs::training::kernels;

namespace {

    Tensor reference_ssim_gradient(const Tensor& image, const Tensor& target, const Tensor& map_gradient) {
        const auto x = image.cpu().contiguous().to_vector();
        const auto y = target.cpu().contiguous().to_vector();
        const auto upstream = map_gradient.cpu().contiguous().to_vector();
        const int h = static_cast<int>(image.shape()[2]);
        const int w = static_cast<int>(image.shape()[3]);
        const size_t plane_size = static_cast<size_t>(h) * w;
        std::vector<double> gradient(x.size(), 0.0);
        std::array<double, 11> gaussian;
        double total = 0.0;
        for (int i = -5; i <= 5; ++i) {
            gaussian[i + 5] = std::exp(-i * i / (2.0 * 1.5 * 1.5));
            total += gaussian[i + 5];
        }
        for (auto& weight : gaussian) {
            weight /= total;
        }

        // Differentiate the Gaussian-window SSIM formula on the host, with zero padding.
        for (size_t base = 0; base < x.size(); base += plane_size) {
            for (int row = 0; row < h; ++row) {
                for (int col = 0; col < w; ++col) {
                    const double scale = upstream[base + row * w + col];
                    if (scale == 0.0) {
                        continue;
                    }
                    double mx = 0.0, my = 0.0, xx = 0.0, yy = 0.0, xy = 0.0;
                    for (int dy = -5; dy <= 5; ++dy) {
                        for (int dx = -5; dx <= 5; ++dx) {
                            const int r = row + dy, c = col + dx;
                            if (r < 0 || r >= h || c < 0 || c >= w) {
                                continue;
                            }
                            const size_t i = base + r * w + c;
                            const double weight = gaussian[dy + 5] * gaussian[dx + 5];
                            mx += weight * x[i];
                            my += weight * y[i];
                            xx += weight * x[i] * x[i];
                            yy += weight * y[i] * y[i];
                            xy += weight * x[i] * y[i];
                        }
                    }
                    const double a = 2.0 * mx * my + 0.0001;
                    const double b = 2.0 * (xy - mx * my) + 0.0009;
                    const double c = mx * mx + my * my + 0.0001;
                    const double d = xx - mx * mx + yy - my * my + 0.0009;
                    const double value = a * b / (c * d);
                    const double d_mean = 2.0 * my * (b - a) / (c * d) + 2.0 * mx * value * (1.0 / d - 1.0 / c);
                    const double d_square = -value / d;
                    const double d_product = 2.0 * a / (c * d);
                    for (int dy = -5; dy <= 5; ++dy) {
                        for (int dx = -5; dx <= 5; ++dx) {
                            const int r = row + dy, c = col + dx;
                            if (r < 0 || r >= h || c < 0 || c >= w) {
                                continue;
                            }
                            const size_t i = base + r * w + c;
                            const double weight = gaussian[dy + 5] * gaussian[dx + 5];
                            gradient[i] += scale * weight * (d_mean + 2.0 * x[i] * d_square + y[i] * d_product);
                        }
                    }
                }
            }
        }
        return Tensor::from_vector(std::vector<float>(gradient.begin(), gradient.end()), image.shape(), image.device());
    }

} // namespace

class FusedL1SSIMTest : public lfs::test::CudaBackendTest {
protected:
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
        auto l1_map = (img1_4d - img2_4d).abs(); // [N, C, H, W]

        // Get per-pixel SSIM map for FULL image (no cropping in ssim_forward_map)
        auto ssim_result = ssim_forward_map(img1_4d, img2_4d, /*apply_valid_padding=*/false);
        auto ssim_map = ssim_result.ssim_map; // [N, C, H, W]

        // Compute per-pixel combined loss map (full image)
        auto dssim_map = Tensor::ones(TensorShape(ssim_map.shape().dims()), Device::GPU) - ssim_map;
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
        auto dL_dmap = Tensor::zeros(combined_loss_map.shape(), Device::GPU);
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
        auto ssim_grad = reference_ssim_gradient(img1_4d, img2_4d, ssim_dL_dmap);

        auto combined_grad = l1_grad + ssim_grad;

        return std::make_pair(combined_loss, combined_grad);
    }
};

// Test basic fused forward correctness
TEST_F(FusedL1SSIMTest, ForwardMatchesReference) {
    const int N = 1, C = 3, H = 64, W = 64;
    auto img1 = Tensor::randn({N, C, H, W}, Device::GPU);
    auto img2 = Tensor::randn({N, C, H, W}, Device::GPU);

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
    auto img1 = Tensor::randn({N, C, H, W}, Device::GPU);
    auto img2 = Tensor::randn({N, C, H, W}, Device::GPU);

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
    auto img1 = Tensor::randn({N, C, H, W}, Device::GPU);
    auto img2 = Tensor::randn({N, C, H, W}, Device::GPU);

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
    auto img1 = Tensor::randn({N, C, H, W}, Device::GPU);
    auto img2 = Tensor::randn({N, C, H, W}, Device::GPU);

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

TEST_F(FusedL1SSIMTest, ErrorMapForwardMatchesSSIMReduction) {
    const int N = 1, C = 3, H = 64, W = 64;
    auto img1 = Tensor::randn({N, C, H, W}, Device::GPU);
    auto img2 = Tensor::randn({N, C, H, W}, Device::GPU);

    auto ssim_result = ssim_forward_map(img1, img2, /*apply_valid_padding=*/false);
    auto expected_error = Tensor::empty({H, W}, Device::GPU);
    launch_ssim_to_error_map(ssim_result.ssim_map, expected_error);

    SSIMMapWorkspace workspace;
    Tensor actual_error;
    ssim_error_map_forward(img1, img2, workspace, actual_error);

    auto diff = (actual_error - expected_error).abs();
    EXPECT_LT(diff.max().item<float>(), 1e-6f);
    EXPECT_LT(diff.mean().item<float>(), 1e-7f);
}

TEST_F(FusedL1SSIMTest, FusedChannelMeanMapSupportsInPlaceErrorMap) {
    const int N = 1, C = 3, H = 64, W = 64;
    auto img1 = Tensor::randn({N, C, H, W}, Device::GPU);
    auto img2 = Tensor::randn({N, C, H, W}, Device::GPU);

    auto full_ssim = ssim_forward_map(img1, img2, /*apply_valid_padding=*/false);
    auto expected_error = Tensor::empty({H, W}, Device::GPU);
    launch_ssim_to_error_map(full_ssim.ssim_map, expected_error);

    FusedL1SSIMWorkspace workspace;
    auto [loss, ctx] = fused_l1_ssim_forward(img1, img2, 0.2f, workspace, false);
    (void)loss;
    (void)ctx;

    ASSERT_EQ(workspace.ssim_map.shape()[0], static_cast<size_t>(1));
    ASSERT_EQ(workspace.ssim_map.shape()[1], static_cast<size_t>(1));
    ASSERT_EQ(workspace.ssim_map.shape()[2], static_cast<size_t>(H));
    ASSERT_EQ(workspace.ssim_map.shape()[3], static_cast<size_t>(W));

    auto inplace_error = workspace.ssim_map.reshape({H, W});
    launch_ssim_to_error_map(workspace.ssim_map, inplace_error);

    auto diff = (inplace_error - expected_error).abs();
    EXPECT_LT(diff.max().item<float>(), 1e-6f);
    EXPECT_LT(diff.mean().item<float>(), 1e-7f);
}

// Test 3D input (no batch dimension)
TEST_F(FusedL1SSIMTest, ThreeDimensionalInput) {
    const int C = 3, H = 64, W = 64;
    auto img1 = Tensor::randn({C, H, W}, Device::GPU);
    auto img2 = Tensor::randn({C, H, W}, Device::GPU);

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
    auto img1 = Tensor::randn({N, C, H, W}, Device::GPU);
    auto img2 = Tensor::randn({N, C, H, W}, Device::GPU);

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
    auto img = Tensor::randn({N, C, H, W}, Device::GPU);

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
    auto img1a = Tensor::randn({N, C, H, W}, Device::GPU);
    auto img2a = Tensor::randn({N, C, H, W}, Device::GPU);
    auto [loss1, ctx1] = fused_l1_ssim_forward(img1a, img2a, ssim_weight, workspace, true);
    auto grad1 = fused_l1_ssim_backward(ctx1, workspace);
    // loss tensor aliases workspace.reduction_result — capture before reuse.
    const float loss1_value = loss1.item<float>();
    const float grad1_norm = grad1.abs().sum().item<float>();

    // Second call with same workspace
    auto img1b = Tensor::randn({N, C, H, W}, Device::GPU);
    auto img2b = Tensor::randn({N, C, H, W}, Device::GPU);
    auto [loss2, ctx2] = fused_l1_ssim_forward(img1b, img2b, ssim_weight, workspace, true);
    auto grad2 = fused_l1_ssim_backward(ctx2, workspace);
    const float loss2_value = loss2.item<float>();
    const float grad2_norm = grad2.abs().sum().item<float>();

    // Both should produce valid results
    EXPECT_FALSE(std::isnan(loss1_value));
    EXPECT_FALSE(std::isnan(loss2_value));

    // Results should be different since inputs differ
    EXPECT_NE(loss1_value, loss2_value);
    EXPECT_NE(grad1_norm, grad2_norm);
}

TEST_F(FusedL1SSIMTest, PhotometricOpsUsesFusedKernel) {
    const int C = 3, H = 64, W = 64;
    auto rendered = Tensor::randn({C, H, W}, Device::GPU);
    auto gt = Tensor::randn({C, H, W}, Device::GPU);

    const auto& ops = lfs::training::cuda_photometric_ops();
    lfs::gpu_ops::PhotoSaved saved{.backend = ops.create()};
    lfs::core::Tensor loss;
    lfs::core::Tensor grad;
    lfs::core::Tensor grad_raw;
    const lfs::gpu_ops::PhotoParams params{
        .path = lfs::gpu_ops::PhotoPath::Fused,
        .ssim_weight = 0.2f,
        .valid_padding = true,
    };
    ops.evaluate(saved, rendered, {}, gt, {}, params, loss, grad, grad_raw);
    EXPECT_FALSE(std::isnan(loss.item<float>()));
    EXPECT_FALSE(std::isnan(grad.abs().max().item<float>()));
}

TEST_F(FusedL1SSIMTest, RejectsInvalidImageContractsBeforeKernelLaunch) {
    auto valid = Tensor::zeros({1, 3, 16, 16}, Device::GPU);
    FusedL1SSIMWorkspace workspace;

    EXPECT_THROW(
        (void)fused_l1_ssim_forward(
            Tensor::zeros({3, 16}, Device::GPU), valid, 0.2f, workspace, true),
        std::exception);
    EXPECT_THROW(
        (void)fused_l1_ssim_forward(
            Tensor::zeros({1, 3, 8, 16}, Device::GPU), valid, 0.2f, workspace, true),
        std::exception);
    EXPECT_THROW(
        (void)fused_l1_ssim_forward(
            valid, valid.to(DataType::Int32), 0.2f, workspace, true),
        std::exception);
    EXPECT_THROW(
        (void)fused_l1_ssim_forward(
            valid, valid, std::numeric_limits<float>::quiet_NaN(), workspace, true),
        std::exception);
    EXPECT_THROW(
        (void)fused_l1_ssim_forward(
            Tensor::empty({0, 3, 16, 16}, Device::GPU),
            Tensor::empty({0, 3, 16, 16}, Device::GPU),
            0.2f, workspace, true),
        std::exception);

    const auto& ops = lfs::training::cuda_photometric_ops();
    lfs::gpu_ops::PhotoSaved saved{.backend = ops.create()};
    lfs::core::Tensor loss;
    lfs::core::Tensor grad;
    lfs::core::Tensor grad_raw;
    EXPECT_THROW(
        ops.evaluate(
            saved, valid, {}, valid.cpu(), {},
            {.path = lfs::gpu_ops::PhotoPath::Fused, .ssim_weight = 0.2f, .valid_padding = true},
            loss, grad, grad_raw),
        std::exception);
    EXPECT_THROW(
        ops.evaluate(
            saved, valid, {}, valid,
            {},
            {.path = lfs::gpu_ops::PhotoPath::Fused,
             .ssim_weight = std::numeric_limits<float>::infinity(),
             .valid_padding = true},
            loss, grad, grad_raw),
        std::exception);
}

TEST_F(FusedL1SSIMTest, UInt8TargetMatchesFloatReference) {
    const int N = 1, C = 3, H = 64, W = 64;
    auto pred = Tensor::rand({N, C, H, W}, Device::GPU);
    auto gt_float = Tensor::rand({N, C, H, W}, Device::GPU);
    auto gt_u8 = (gt_float * 255.0f).clamp(0.0f, 255.0f).to(DataType::UInt8);
    auto gt_quant = gt_u8.to(DataType::Float32) / 255.0f;

    const float ssim_weight = 0.2f;
    FusedL1SSIMWorkspace workspace;
    auto [loss, ctx] = fused_l1_ssim_forward(pred, gt_u8, ssim_weight, workspace, true);
    auto grad = fused_l1_ssim_backward(ctx, workspace);

    auto [ref_loss, ref_grad] = compute_reference_loss_and_grad(pred, gt_quant, ssim_weight);
    EXPECT_NEAR(loss.item<float>(), ref_loss, 1e-4f);

    auto diff = (grad - ref_grad).abs();
    EXPECT_LT(diff.max().item<float>(), 1e-3f);
    EXPECT_LT(diff.mean().item<float>(), 1e-5f);
}

// ============================================================================
// Masked Fused L1+SSIM Tests
// ============================================================================

class MaskedFusedL1SSIMTest : public lfs::test::CudaBackendTest {
protected:
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
        if (mask_2d.dtype() == DataType::UInt8 || mask_2d.dtype() == DataType::Bool) {
            mask_2d = mask_2d.to(DataType::Float32);
        }
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
        auto ssim_grad = reference_ssim_gradient(img1_4d, img2_4d, dL_dmap);

        // Combined
        float combined_loss = l1_weight * masked_l1_loss + ssim_weight * ssim_loss;
        auto combined_grad = l1_grad * l1_weight + ssim_grad * ssim_weight;

        return {combined_loss, combined_grad};
    }
};

TEST_F(MaskedFusedL1SSIMTest, ForwardBasic) {
    const int N = 1, C = 3, H = 64, W = 64;
    auto img1 = Tensor::randn({N, C, H, W}, Device::GPU);
    auto img2 = Tensor::randn({N, C, H, W}, Device::GPU);

    // Create a mask with some regions masked out
    auto mask = Tensor::ones({H, W}, Device::GPU).to(DataType::UInt8);
    // Mask out a region
    auto mask_view = mask.slice(0, 0, H / 2).slice(1, 0, W / 2);
    mask_view.zero_();
    cudaDeviceSynchronize();

    const float ssim_weight = 0.2f;

    MaskedFusedL1SSIMWorkspace workspace;
    auto [loss, ctx] = masked_fused_l1_ssim_forward(img1, img2, mask, ssim_weight, workspace);

    EXPECT_FALSE(std::isnan(loss.item<float>()));
    EXPECT_FALSE(std::isinf(loss.item<float>()));
    EXPECT_GT(loss.item<float>(), 0.0f);
}

TEST_F(MaskedFusedL1SSIMTest, SoftWeightsUseWeightedMeanNormalization) {
    constexpr int N = 1;
    constexpr int C = 3;
    constexpr int H = 12;
    constexpr int W = 12;
    constexpr float ssim_weight = 0.0f;

    std::vector<float> prediction_data(static_cast<size_t>(N) * C * H * W);
    std::vector<float> weight_data(static_cast<size_t>(H) * W);
    double weighted_sum = 0.0;
    double weight_sum = 0.0;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const size_t pixel = static_cast<size_t>(y) * W + x;
            const float pixel_weight = static_cast<float>((x + 2 * y) % 5) * 0.25f;
            weight_data[pixel] = pixel_weight;
            weight_sum += pixel_weight;
            for (int c = 0; c < C; ++c) {
                const float value = 0.01f * static_cast<float>(1 + c + x + 2 * y);
                prediction_data[static_cast<size_t>(c) * H * W + pixel] = value;
                weighted_sum += static_cast<double>(value) * pixel_weight;
            }
        }
    }
    const float expected = static_cast<float>(
        weighted_sum /
        (weight_sum * C + lfs::training::kernels::SSIM_EPSILON));

    const auto prediction = Tensor::from_vector(
        prediction_data, {N, C, H, W}, Device::GPU);
    const auto target = Tensor::zeros({N, C, H, W}, Device::GPU);
    const auto weight = Tensor::from_vector(
        weight_data, {H, W}, Device::GPU);

    MaskedFusedL1SSIMWorkspace workspace;
    const auto [loss, ctx] =
        masked_fused_l1_ssim_forward(prediction, target, weight, ssim_weight, workspace);
    const auto grad = masked_fused_l1_ssim_backward(ctx, workspace);

    EXPECT_NEAR(loss.item<float>(), expected, 1.0e-6f);
    EXPECT_FLOAT_EQ(
        grad.to(Device::CPU).to_vector()[static_cast<size_t>(H) * W - 1],
        weight_data.back() /
            static_cast<float>(weight_sum * C + lfs::training::kernels::SSIM_EPSILON));
}

TEST_F(MaskedFusedL1SSIMTest, AllOneWeightMatchesUnmaskedTinyImage) {
    constexpr int N = 1;
    constexpr int C = 3;
    constexpr int H = 8;
    constexpr int W = 8;
    constexpr float ssim_weight = 0.2f;

    const auto prediction = Tensor::rand({N, C, H, W}, Device::GPU);
    const auto target = Tensor::rand({N, C, H, W}, Device::GPU);
    const auto weight = Tensor::ones({H, W}, Device::GPU);

    FusedL1SSIMWorkspace unmasked_workspace;
    auto [unmasked_loss, unmasked_ctx] =
        fused_l1_ssim_forward(
            prediction, target, ssim_weight, unmasked_workspace,
            /*apply_valid_padding=*/true);
    const auto unmasked_grad =
        fused_l1_ssim_backward(unmasked_ctx, unmasked_workspace);

    MaskedFusedL1SSIMWorkspace masked_workspace;
    auto [masked_loss, masked_ctx] =
        masked_fused_l1_ssim_forward(
            prediction, target, weight, ssim_weight, masked_workspace);
    const auto masked_grad =
        masked_fused_l1_ssim_backward(masked_ctx, masked_workspace);

    EXPECT_NEAR(masked_loss.item<float>(), unmasked_loss.item<float>(), 1.0e-6f);
    const auto difference = (masked_grad - unmasked_grad).abs();
    EXPECT_LT(difference.max().item<float>(), 1.0e-6f);
}

TEST_F(MaskedFusedL1SSIMTest, SoftBoundaryWeightsMatchSsimReference) {
    constexpr int N = 1;
    constexpr int C = 3;
    constexpr int H = 24;
    constexpr int W = 24;
    constexpr float ssim_weight = 0.35f;

    const auto prediction = Tensor::rand({N, C, H, W}, Device::GPU);
    const auto target = Tensor::rand({N, C, H, W}, Device::GPU);
    auto weight = Tensor::ones({H, W}, Device::GPU);
    weight.slice(0, 0, 5).fill_(0.15f);
    weight.slice(1, W - 5, W).fill_(0.4f);

    MaskedFusedL1SSIMWorkspace workspace;
    auto [loss, ctx] =
        masked_fused_l1_ssim_forward(
            prediction, target, weight, ssim_weight, workspace);
    const auto gradient = masked_fused_l1_ssim_backward(ctx, workspace);
    const auto [reference_loss, reference_gradient] =
        compute_reference_masked_loss(prediction, target, weight, ssim_weight);

    EXPECT_NEAR(loss.item<float>(), reference_loss, 1.0e-3f);
    const auto difference = (gradient - reference_gradient).abs();
    EXPECT_LT(difference.max().item<float>(), 1.0e-2f);
    EXPECT_LT(difference.mean().item<float>(), 1.0e-4f);
}

TEST_F(MaskedFusedL1SSIMTest, BackwardBasic) {
    const int N = 1, C = 3, H = 64, W = 64;
    auto img1 = Tensor::randn({N, C, H, W}, Device::GPU);
    auto img2 = Tensor::randn({N, C, H, W}, Device::GPU);
    auto mask = Tensor::ones({H, W}, Device::GPU).to(DataType::UInt8);

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
    auto img1 = Tensor::randn({N, C, H, W}, Device::GPU);
    auto img2 = Tensor::randn({N, C, H, W}, Device::GPU);
    auto mask = Tensor::ones({H, W}, Device::GPU).to(DataType::UInt8);

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
    auto img1 = Tensor::randn({N, C, H, W}, Device::GPU);
    auto img2 = Tensor::randn({N, C, H, W}, Device::GPU);

    // Create checkerboard mask
    auto mask = Tensor::zeros({H, W}, Device::GPU);
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
    auto img1 = Tensor::randn({N, C, H, W}, Device::GPU);
    auto img2 = Tensor::randn({N, C, H, W}, Device::GPU);
    auto mask = Tensor::zeros({H, W}, Device::GPU);

    const float ssim_weight = 0.2f;

    MaskedFusedL1SSIMWorkspace workspace;
    auto [loss, ctx] = masked_fused_l1_ssim_forward(img1, img2, mask, ssim_weight, workspace);
    const auto grad = masked_fused_l1_ssim_backward(ctx, workspace);

    EXPECT_LT(loss.item<float>(), 1e-5f);
    EXPECT_EQ(grad.abs().max().item<float>(), 0.0f);
}

TEST_F(MaskedFusedL1SSIMTest, WorkspaceReuse) {
    const int N = 1, C = 3, H = 64, W = 64;
    const float ssim_weight = 0.2f;
    auto mask = Tensor::ones({H, W}, Device::GPU);

    MaskedFusedL1SSIMWorkspace workspace;

    // Multiple calls with same workspace
    for (int i = 0; i < 3; ++i) {
        auto img1 = Tensor::randn({N, C, H, W}, Device::GPU);
        auto img2 = Tensor::randn({N, C, H, W}, Device::GPU);

        auto [loss, ctx] = masked_fused_l1_ssim_forward(img1, img2, mask, ssim_weight, workspace);
        auto grad = masked_fused_l1_ssim_backward(ctx, workspace);

        EXPECT_FALSE(std::isnan(loss.item<float>()));
        EXPECT_FALSE(std::isnan(grad.abs().max().item<float>()));
    }
}

TEST_F(MaskedFusedL1SSIMTest, UInt8TargetMatchesFloatReference) {
    const int N = 1, C = 3, H = 64, W = 64;
    auto pred = Tensor::rand({N, C, H, W}, Device::GPU);
    auto gt_float = Tensor::rand({N, C, H, W}, Device::GPU);
    auto gt_u8 = (gt_float * 255.0f).clamp(0.0f, 255.0f).to(DataType::UInt8);
    auto gt_quant = gt_u8.to(DataType::Float32) / 255.0f;
    auto mask = Tensor::ones({H, W}, Device::GPU);
    mask.slice(0, 0, H / 2).slice(1, 0, W / 2).fill_(0.0f, nullptr);
    cudaDeviceSynchronize();

    const float ssim_weight = 0.2f;
    MaskedFusedL1SSIMWorkspace workspace;
    auto [loss, ctx] = masked_fused_l1_ssim_forward(pred, gt_u8, mask, ssim_weight, workspace);
    auto grad = masked_fused_l1_ssim_backward(ctx, workspace);

    auto [ref_loss, ref_grad] = compute_reference_masked_loss(pred, gt_quant, mask, ssim_weight);
    EXPECT_NEAR(loss.item<float>(), ref_loss, 1e-3f);

    auto diff = (grad - ref_grad).abs();
    EXPECT_LT(diff.max().item<float>(), 1e-2f);
    EXPECT_LT(diff.mean().item<float>(), 1e-4f);
}

TEST_F(MaskedFusedL1SSIMTest, UInt8TargetAndMaskMatchFloatReference) {
    const int N = 1, C = 3, H = 64, W = 64;
    auto pred = Tensor::rand({N, C, H, W}, Device::GPU);
    auto gt_float = Tensor::rand({N, C, H, W}, Device::GPU);
    auto gt_u8 = (gt_float * 255.0f).clamp(0.0f, 255.0f).to(DataType::UInt8);
    auto gt_quant = gt_u8.to(DataType::Float32) / 255.0f;
    auto mask_float = Tensor::ones({H, W}, Device::GPU);
    mask_float.slice(0, 0, H / 2).slice(1, 0, W / 2).fill_(0.0f, nullptr);
    auto mask_u8 = mask_float.to(DataType::UInt8);
    cudaDeviceSynchronize();

    const float ssim_weight = 0.2f;
    MaskedFusedL1SSIMWorkspace workspace;
    auto [loss, ctx] = masked_fused_l1_ssim_forward(pred, gt_u8, mask_u8, ssim_weight, workspace);
    auto grad = masked_fused_l1_ssim_backward(ctx, workspace);

    auto [ref_loss, ref_grad] = compute_reference_masked_loss(pred, gt_quant, mask_float, ssim_weight);
    EXPECT_NEAR(loss.item<float>(), ref_loss, 1e-3f);

    auto diff = (grad - ref_grad).abs();
    EXPECT_LT(diff.max().item<float>(), 1e-2f);
    EXPECT_LT(diff.mean().item<float>(), 1e-4f);
}

// ============================================================================
// Decoupled appearance-loss tests
// ============================================================================

TEST_F(FusedL1SSIMTest, DecoupledMatchesStandardWhenCorrectedEqualsRaw) {
    const int N = 1, C = 3, H = 64, W = 64;
    const float ssim_weight = 0.2f;

    auto raw = Tensor::randn({N, C, H, W}, Device::GPU).abs() + 0.1f;
    auto corrected = raw.clone();
    auto gt = Tensor::randn({N, C, H, W}, Device::GPU).abs() + 0.1f;

    FusedL1SSIMWorkspace standard_workspace;
    auto [standard_loss, standard_ctx] =
        fused_l1_ssim_forward(raw, gt, ssim_weight, standard_workspace, true);
    auto standard_grad = fused_l1_ssim_backward(standard_ctx, standard_workspace);

    DecoupledFusedL1SSIMWorkspace decoupled_workspace;
    auto [decoupled_loss, decoupled_ctx] =
        decoupled_fused_l1_ssim_forward(corrected, raw, gt, ssim_weight, decoupled_workspace, true);
    auto decoupled_grads = decoupled_fused_l1_ssim_backward(decoupled_ctx, decoupled_workspace);
    auto combined_grad = decoupled_grads.grad_corrected + decoupled_grads.grad_raw;

    EXPECT_NEAR(decoupled_loss.item<float>(), standard_loss.item<float>(), 1e-4f);

    auto diff = (combined_grad - standard_grad).abs();
    EXPECT_LT(diff.max().item<float>(), 1e-3f);
    EXPECT_LT(diff.mean().item<float>(), 1e-5f);
}

TEST_F(MaskedFusedL1SSIMTest, DecoupledMatchesStandardWhenCorrectedEqualsRaw) {
    const int N = 1, C = 3, H = 64, W = 64;
    const float ssim_weight = 0.2f;

    auto raw = Tensor::randn({N, C, H, W}, Device::GPU).abs() + 0.1f;
    auto corrected = raw.clone();
    auto gt = Tensor::randn({N, C, H, W}, Device::GPU).abs() + 0.1f;
    auto mask = Tensor::ones({H, W}, Device::GPU).to(DataType::UInt8);

    MaskedFusedL1SSIMWorkspace standard_workspace;
    auto [standard_loss, standard_ctx] =
        masked_fused_l1_ssim_forward(raw, gt, mask, ssim_weight, standard_workspace);
    auto standard_grad = masked_fused_l1_ssim_backward(standard_ctx, standard_workspace);

    MaskedDecoupledFusedL1SSIMWorkspace decoupled_workspace;
    auto [decoupled_loss, decoupled_ctx] =
        masked_decoupled_fused_l1_ssim_forward(corrected, raw, gt, mask, ssim_weight, decoupled_workspace);
    auto decoupled_grads =
        masked_decoupled_fused_l1_ssim_backward(decoupled_ctx, decoupled_workspace);
    auto combined_grad = decoupled_grads.grad_corrected + decoupled_grads.grad_raw;

    EXPECT_NEAR(decoupled_loss.item<float>(), standard_loss.item<float>(), 1e-4f);

    auto diff = (combined_grad - standard_grad).abs();
    EXPECT_LT(diff.max().item<float>(), 1e-3f);
    EXPECT_LT(diff.mean().item<float>(), 1e-5f);
}

TEST_F(MaskedFusedL1SSIMTest, DecoupledAllZeroWeightReturnsZeroGradients) {
    constexpr int N = 1;
    constexpr int C = 3;
    constexpr int H = 16;
    constexpr int W = 16;
    const auto corrected = Tensor::rand({N, C, H, W}, Device::GPU);
    const auto raw = Tensor::rand({N, C, H, W}, Device::GPU);
    const auto target = Tensor::rand({N, C, H, W}, Device::GPU);
    const auto weight = Tensor::zeros({H, W}, Device::GPU);

    MaskedDecoupledFusedL1SSIMWorkspace workspace;
    auto [loss, ctx] = masked_decoupled_fused_l1_ssim_forward(
        corrected, raw, target, weight, 0.2f, workspace);
    const auto gradients =
        masked_decoupled_fused_l1_ssim_backward(ctx, workspace);

    EXPECT_LT(loss.item<float>(), 1.0e-5f);
    EXPECT_EQ(gradients.grad_corrected.abs().max().item<float>(), 0.0f);
    EXPECT_EQ(gradients.grad_raw.abs().max().item<float>(), 0.0f);
}

TEST_F(FusedL1SSIMTest, DecoupledRoutesContrastStructureGradientToRawBranch) {
    const int N = 1, C = 3, H = 64, W = 64;
    const float ssim_weight = 1.0f;
    auto gt = Tensor::linspace(-1.0f, 1.0f, N * C * H * W, Device::GPU).reshape({N, C, H, W});
    auto corrected = gt.clone();
    auto raw = gt * -0.6f;

    DecoupledFusedL1SSIMWorkspace workspace;
    auto [loss, ctx] =
        decoupled_fused_l1_ssim_forward(corrected, raw, gt, ssim_weight, workspace, true);
    auto grads = decoupled_fused_l1_ssim_backward(ctx, workspace);

    EXPECT_GT(loss.item<float>(), 0.0f);
    EXPECT_LT(grads.grad_corrected.abs().max().item<float>(), 1e-4f);
    EXPECT_GT(grads.grad_raw.abs().max().item<float>(), 1e-4f);
}

TEST_F(FusedL1SSIMTest, BackwardFillFollowsIndependentQueue) {
    constexpr size_t h = 24, w = 28;
    std::vector<float> pixels(3 * h * w), target(pixels.size());
    for (size_t i = 0; i < pixels.size(); ++i) {
        pixels[i] = static_cast<float>(i % 251) / 256.0f;
        target[i] = static_cast<float>((i * 13) % 251) / 256.0f;
    }
    for (const bool crop : {false, true}) {
        Tensor reference;
        for (const bool independent : {false, true}) {
            TensorWorkQueue queue(GpuBackend::CUDA, independent ? TensorWorkQueue::Mode::Independent
                                                                : TensorWorkQueue::Mode::LegacyOrdered);
            TensorWorkQueue::Scope scope(queue);
            auto image = Tensor::from_vector(pixels, {1, 3, h, w}, Device::GPU);
            auto truth = Tensor::from_vector(target, {1, 3, h, w}, Device::GPU);
            SSIMWorkspace workspace;
            auto [loss, context] = ssim_forward(image, truth, workspace, crop);
            queue.wait();
            if (independent) {
                // Legacy clears and fills can overtake a pending workspace write.
                queue.enqueue_host_callback([](void*) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(30));
                },
                                            nullptr);
            }
            workspace.dL_dmap.fill_(-0.5f, getCurrentCUDAStream());
            auto gradient = ssim_backward(context, workspace, 0.75f);
            queue.wait();
            const auto map = workspace.dL_dmap.cpu().to_vector();
            const float expected = 0.75f / static_cast<float>(3 * (crop ? (h - 10) * (w - 10) : h * w));
            for (size_t i = 0; i < map.size(); ++i) {
                const size_t row = (i / w) % h, col = i % w;
                const bool valid = !crop || (row >= 5 && row < h - 5 && col >= 5 && col < w - 5);
                ASSERT_EQ(map[i], valid ? expected : 0.0f) << "crop=" << crop << " index=" << i;
            }
            auto host = gradient.cpu().contiguous();
            if (independent) {
                ASSERT_EQ(host.bytes(), reference.bytes());
                EXPECT_EQ(std::memcmp(host.data_ptr(), reference.data_ptr(), host.bytes()), 0);
            } else {
                reference = std::move(host);
            }
        }
    }
}

TEST_F(FusedL1SSIMTest, ReusedWorkspaceFollowsExecutionQueue) {
    TensorWorkQueue producer(GpuBackend::CUDA);
    TensorWorkQueue consumer(GpuBackend::CUDA);
    TensorWorkQueue::Scope scope(producer);
    auto image = Tensor::full({1, 3, 24, 28}, 0.25f, Device::GPU);
    auto target = Tensor::full({1, 3, 24, 28}, 0.75f, Device::GPU);
    FusedL1SSIMWorkspace workspace;
    const auto first = fused_l1_ssim_forward(image, target, 0.2f, workspace);
    const float reference = first.first.item<float>();
    {
        TensorWorkQueue::Scope next(consumer);
        const auto result = fused_l1_ssim_forward(image, target, 0.2f, workspace);
        EXPECT_EQ(workspace.ssim_map.stream(), consumer.native_handle());
        EXPECT_EQ(workspace.reduction_result.stream(), consumer.native_handle());
        EXPECT_EQ(result.first.item<float>(), reference);
        const auto gradient = fused_l1_ssim_backward(result.second, workspace);
        EXPECT_EQ(gradient.stream(), consumer.native_handle());
        consumer.wait();
    }
}
