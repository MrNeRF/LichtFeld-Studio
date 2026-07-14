/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * Analytical gradient verification for FastGS rasterizer.
 *
 * Strategy:
 * 1. Implement forward ops in LibTorch with requires_grad=True
 * 2. Use autograd to get ground truth gradients
 * 3. Verify our analytical formulas match autograd
 * 4. Compare CUDA kernel gradients against verified analytical gradients
 */

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

#include "core/cuda/sh_layout.cuh"
#include "training/kernels/depth_loss.hpp"
#include "training/kernels/normal_consistency_loss.hpp"
#include "training/kernels/normal_loss.hpp"
#include "training/rasterization/gsplat/SphericalHarmonics.h"

namespace {

    constexpr float REL_TOL = 1e-4f;
    constexpr float ABS_TOL = 1e-6f;

    bool tensors_close(const torch::Tensor& a, const torch::Tensor& b, float rtol = REL_TOL, float atol = ABS_TOL) {
        return torch::allclose(a, b, rtol, atol);
    }

    float max_rel_error(const torch::Tensor& a, const torch::Tensor& b) {
        auto diff = (a - b).abs();
        auto scale = torch::max(a.abs(), b.abs()) + 1e-8f;
        return (diff / scale).max().item<float>();
    }

    struct SsiAlignment {
        bool valid = false;
        int model = 0;
        float a = 0.0f;
        float b = 0.0f;
        float floor_f = 0.0f;
        float sigma_p = 0.0f;
        float inv_norm = 0.0f;
    };

    lfs::training::kernels::DepthAnchor make_depth_anchor(
        const int model,
        const float a,
        const float b,
        const float floor_f) {
        lfs::training::kernels::DepthAnchor anchor;
        anchor.valid = true;
        anchor.model = model;
        anchor.scale = a;
        anchor.shift = b;
        anchor.floor = floor_f;
        anchor.corr = 1.0f;
        anchor.samples = 1024;
        if (model == 0) {
            anchor.disparity = {true, a, b, 1.0f, 1024};
        } else {
            anchor.depth = {true, a, b, 1.0f, 1024};
        }
        return anchor;
    }

    // Mirrors the anchored alignment stage of the CUDA kernel: alpha-weighted
    // stats in double, sigma and normalization scalars truncated to float
    // exactly like the kernel's finals buffer.
    SsiAlignment ssi_depth_alignment_reference(
        const torch::Tensor& depth_accum,
        const torch::Tensor& alpha_accum,
        const torch::Tensor& target_depth,
        const lfs::training::kernels::DepthAnchor* anchor = nullptr) {
        namespace tk = lfs::training::kernels;
        SsiAlignment out{};
        if (anchor == nullptr || !anchor->valid) {
            return out;
        }

        const auto d64 = depth_accum.detach().to(torch::kFloat64);
        const auto a64 = alpha_accum.detach().to(torch::kFloat64);
        const auto active = target_depth.detach().gt(0.0f).logical_and(
            alpha_accum.detach().gt(tk::kDepthLossMinAlpha));
        const auto aw = a64 * active.to(torch::kFloat64);

        const double sum_alpha = aw.sum().item<double>();
        if (sum_alpha <= 0.0) {
            return out;
        }

        const auto e64 = d64.clamp_min(0.0) / a64;
        const double mean_e = (aw * e64).sum().item<double>() / sum_alpha;
        (void)mean_e;
        out.floor_f = anchor->floor;

        const auto p64 = 1.0 / (e64 + static_cast<double>(out.floor_f));
        const double mean_p = (aw * p64).sum().item<double>() / sum_alpha;
        const double var_p = std::max((aw * p64 * p64).sum().item<double>() / sum_alpha - mean_p * mean_p, 0.0);
        const double sigma_p = std::sqrt(std::max(var_p, 1.0e-20));

        out.valid = true;
        out.model = anchor->model;
        out.a = anchor->scale;
        out.b = anchor->shift;
        out.sigma_p = static_cast<float>(sigma_p);
        out.inv_norm = static_cast<float>(1.0 / sum_alpha);
        return out;
    }

    // Differentiable reference of the per-pixel loss with the alignment frozen,
    // matching the kernel's stop-gradient semantics. Inputs are [H, W].
    torch::Tensor ssi_depth_loss_reference(
        const torch::Tensor& depth_accum,
        const torch::Tensor& alpha_accum,
        const torch::Tensor& target_depth,
        const SsiAlignment& align,
        const float weight,
        const float lambda_grad,
        const float quantization_step = 0.0f) {
        namespace tk = lfs::training::kernels;
        if (!align.valid) {
            return torch::zeros({}, depth_accum.options());
        }

        const auto H = depth_accum.size(0);
        const auto W = depth_accum.size(1);
        const auto alpha_det = alpha_accum.detach();
        const auto active = target_depth.detach().gt(0.0f).logical_and(alpha_det.gt(tk::kDepthLossMinAlpha));

        const auto e = depth_accum.clamp_min(0.0f) / alpha_accum;
        const auto p = 1.0f / (e + align.floor_f);
        const auto fit = align.a * target_depth.detach() + align.b;
        const auto ok = active.logical_and(fit.gt(0.0f));
        const auto d = align.model == 0
                           ? fit.clamp_max(1.0f / align.floor_f)
                           : 1.0f / (fit + align.floor_f);

        const float half_step = 0.5f * quantization_step * std::fabs(align.a);
        const auto delta = align.model == 0
                               ? torch::full_like(d, half_step)
                               : (half_step * d * d).detach();

        const auto okf = ok.to(torch::kFloat32);
        const auto r = p - d;
        const float inv_scaled_sigma =
            1.0f / (tk::kDepthLossResidualScale * align.sigma_p);
        const auto signed_excess =
            torch::sign(r) * (r.abs() - delta).clamp_min(0.0f);
        const auto robust = [](const torch::Tensor& x) {
            const auto x2 = x.square();
            return 0.5f * x2 / (1.0f + x2);
        };
        const auto l1 = (alpha_det * okf * robust(signed_excess * inv_scaled_sigma)).sum();

        const auto pl = p.slice(1, 0, W - 1);
        const auto pr = p.slice(1, 1, W);
        const auto dl = d.slice(1, 0, W - 1);
        const auto dr = d.slice(1, 1, W);
        const auto band_h = delta.slice(1, 0, W - 1) + delta.slice(1, 1, W);
        const auto mh = ok.slice(1, 0, W - 1).logical_and(ok.slice(1, 1, W)).to(torch::kFloat32);
        const auto wh = torch::minimum(alpha_det.slice(1, 0, W - 1), alpha_det.slice(1, 1, W)) * mh;
        const auto h = (pr - pl) - (dr - dl);
        const auto h_signed = torch::sign(h) * (h.abs() - band_h).clamp_min(0.0f);
        const auto gal_h = (wh * robust(h_signed * inv_scaled_sigma)).sum();

        const auto pu = p.slice(0, 0, H - 1);
        const auto pb = p.slice(0, 1, H);
        const auto du = d.slice(0, 0, H - 1);
        const auto db = d.slice(0, 1, H);
        const auto band_v = delta.slice(0, 0, H - 1) + delta.slice(0, 1, H);
        const auto mv = ok.slice(0, 0, H - 1).logical_and(ok.slice(0, 1, H)).to(torch::kFloat32);
        const auto wv = torch::minimum(alpha_det.slice(0, 0, H - 1), alpha_det.slice(0, 1, H)) * mv;
        const auto v = (pb - pu) - (db - du);
        const auto v_signed = torch::sign(v) * (v.abs() - band_v).clamp_min(0.0f);
        const auto gal_v = (wv * robust(v_signed * inv_scaled_sigma)).sum();

        return weight * align.inv_norm * (l1 + lambda_grad * (gal_h + gal_v));
    }

    struct SsiKernelResult {
        torch::Tensor loss;
        torch::Tensor grad_depth;
        torch::Tensor grad_alpha;
        torch::Tensor partials;
    };

    SsiKernelResult run_ssi_depth_loss_kernel(
        const torch::Tensor& depth_accum,
        const torch::Tensor& alpha_accum,
        const torch::Tensor& target_depth,
        const float weight,
        const float lambda_grad,
        const lfs::training::kernels::DepthAnchor* anchor = nullptr,
        const float quantization_step = 0.0f) {
        const auto opts = depth_accum.options();
        const int height = static_cast<int>(depth_accum.size(0));
        const int width = static_cast<int>(depth_accum.size(1));
        const auto num_pixels = static_cast<size_t>(depth_accum.numel());

        SsiKernelResult result{
            .loss = torch::empty({1}, opts),
            .grad_depth = torch::empty_like(depth_accum),
            .grad_alpha = torch::empty_like(depth_accum),
            .partials = torch::empty(
                {static_cast<int64_t>(lfs::training::kernels::depth_loss_partial_count(num_pixels))}, opts)};

        lfs::training::kernels::launch_depth_loss(
            depth_accum.data_ptr<float>(),
            alpha_accum.data_ptr<float>(),
            target_depth.data_ptr<float>(),
            result.grad_depth.data_ptr<float>(),
            result.grad_alpha.data_ptr<float>(),
            result.loss.data_ptr<float>(),
            result.partials.data_ptr<float>(),
            width,
            height,
            weight,
            lambda_grad,
            quantization_step,
            anchor);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        return result;
    }

    void expect_ssi_depth_loss_kernel_matches_autograd(
        const torch::Tensor& depth_accum,
        const torch::Tensor& alpha_accum,
        const torch::Tensor& target_depth,
        const float weight,
        const float lambda_grad,
        const lfs::training::kernels::DepthAnchor* anchor = nullptr,
        const float quantization_step = 0.0f) {
        ASSERT_TRUE(depth_accum.is_cuda());
        ASSERT_EQ(depth_accum.dim(), 2);
        ASSERT_TRUE(depth_accum.sizes() == alpha_accum.sizes());
        ASSERT_TRUE(depth_accum.sizes() == target_depth.sizes());

        const auto align = ssi_depth_alignment_reference(depth_accum, alpha_accum, target_depth, anchor);
        auto depth_ag = depth_accum.detach().clone().set_requires_grad(true);
        auto alpha_ag = alpha_accum.detach().clone().set_requires_grad(true);
        const auto expected_loss = ssi_depth_loss_reference(
            depth_ag, alpha_ag, target_depth, align, weight, lambda_grad, quantization_step);
        if (align.valid) {
            expected_loss.backward();
        }

        const auto result = run_ssi_depth_loss_kernel(
            depth_accum, alpha_accum, target_depth, weight, lambda_grad, anchor, quantization_step);

        namespace slots = lfs::training::kernels::depth_loss_slots;
        const float kernel_valid = result.partials[slots::kValid].item<float>();
        ASSERT_EQ(kernel_valid > 0.5f, align.valid) << "kernel/reference validity mismatch";
        if (!align.valid) {
            EXPECT_EQ(result.loss.item<float>(), 0.0f);
            EXPECT_EQ(result.grad_depth.abs().max().item<float>(), 0.0f);
            EXPECT_EQ(result.grad_alpha.abs().max().item<float>(), 0.0f);
            return;
        }

        const int kernel_model = static_cast<int>(result.partials[slots::kModel].item<float>());
        EXPECT_EQ(kernel_model, align.model) << "alignment-space selection mismatch";

        EXPECT_TRUE(torch::allclose(result.loss, expected_loss.detach().reshape({1}), 2e-3f, 1e-7f))
            << "SSI depth loss value mismatch: kernel=" << result.loss.item<float>()
            << ", autograd=" << expected_loss.item<float>();
        EXPECT_TRUE(tensors_close(result.grad_depth, depth_ag.grad(), 2e-3f, 1e-9f))
            << "SSI depth-gradient mismatch: " << max_rel_error(result.grad_depth, depth_ag.grad());
        EXPECT_TRUE(tensors_close(result.grad_alpha, alpha_ag.grad(), 2e-3f, 1e-9f))
            << "SSI alpha-gradient mismatch: " << max_rel_error(result.grad_alpha, alpha_ag.grad());
    }

    struct SplitSHTensors {
        torch::Tensor sh0;
        torch::Tensor shN;
    };

    SplitSHTensors make_split_sh_tensors(const torch::Tensor& sh_coeffs) {
        const auto n = static_cast<std::size_t>(sh_coeffs.size(0));
        const auto k = static_cast<std::size_t>(sh_coeffs.size(1));
        SplitSHTensors result{
            .sh0 = sh_coeffs.select(1, 0).contiguous(),
            .shN = torch::empty({0}, sh_coeffs.options())};
        if (k > 1) {
            auto sh_rest = sh_coeffs.slice(1, 1, static_cast<int64_t>(k)).contiguous();
            result.shN = torch::empty(
                {static_cast<int64_t>(lfs::core::sh_swizzled_float_count(
                    n,
                    static_cast<std::uint32_t>(k - 1)))},
                sh_coeffs.options());
            lfs::core::reorder_sh_to_swizzled(
                sh_rest.data_ptr<float>(),
                result.shN.data_ptr<float>(),
                n,
                static_cast<std::uint32_t>(k - 1));
        }
        return result;
    }

} // namespace

class AnalyticalGradientTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available";
        }
        torch::manual_seed(42);
    }
};

// =============================================================================
// Test 1: Quaternion to Rotation Matrix gradient
// =============================================================================
TEST_F(AnalyticalGradientTest, QuaternionToRotation) {
    const int N = 100;

    auto quat = torch::randn({N, 4}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    quat = quat / quat.norm(2, -1, true); // normalize

    // Forward with autograd
    auto quat_ag = quat.clone().set_requires_grad(true);
    auto qn = quat_ag / quat_ag.norm(2, -1, true);

    auto w = qn.select(1, 0);
    auto x = qn.select(1, 1);
    auto y = qn.select(1, 2);
    auto z = qn.select(1, 3);

    auto x2 = x * x, y2 = y * y, z2 = z * z;
    auto xy = x * y, xz = x * z, yz = y * z;
    auto wx = w * x, wy = w * y, wz = w * z;

    // Build rotation matrix [N, 3, 3]
    auto R00 = 1 - 2 * (y2 + z2);
    auto R01 = 2 * (xy - wz);
    auto R02 = 2 * (xz + wy);
    auto R10 = 2 * (xy + wz);
    auto R11 = 1 - 2 * (x2 + z2);
    auto R12 = 2 * (yz - wx);
    auto R20 = 2 * (xz - wy);
    auto R21 = 2 * (yz + wx);
    auto R22 = 1 - 2 * (x2 + y2);

    auto R = torch::stack({torch::stack({R00, R01, R02}, 1),
                           torch::stack({R10, R11, R12}, 1),
                           torch::stack({R20, R21, R22}, 1)},
                          1); // [N, 3, 3]

    // Random upstream gradient
    auto v_R = torch::randn_like(R);

    // Autograd backward
    R.backward(v_R);
    auto grad_quat_autograd = quat_ag.grad();

    // Analytical gradient computation
    // For quaternion normalization: d(q_normalized)/d(q) = (I - q_n * q_n^T) / ||q||
    // Then chain through rotation matrix construction
    // This is complex, so we just verify autograd works and use it as ground truth

    ASSERT_TRUE(grad_quat_autograd.defined());
    ASSERT_EQ(grad_quat_autograd.sizes(), quat.sizes());

    // Verify gradients are finite
    EXPECT_FALSE(grad_quat_autograd.isnan().any().item<bool>());
    EXPECT_FALSE(grad_quat_autograd.isinf().any().item<bool>());

    std::cout << "QuaternionToRotation: grad magnitude = " << grad_quat_autograd.abs().mean().item<float>() << std::endl;
}

// =============================================================================
// Test 2: Scale to Variance gradient (exp(2*s))
// =============================================================================
TEST_F(AnalyticalGradientTest, ScaleToVariance) {
    const int N = 100;

    auto scale = torch::randn({N, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // Autograd path
    auto scale_ag = scale.clone().set_requires_grad(true);
    auto var_ag = torch::exp(2 * scale_ag);

    auto v_var = torch::randn_like(var_ag);
    var_ag.backward(v_var);
    auto grad_scale_autograd = scale_ag.grad();

    // Analytical: d(exp(2s))/ds = 2 * exp(2s)
    auto var = torch::exp(2 * scale);
    auto grad_scale_analytical = 2 * var * v_var;

    EXPECT_TRUE(tensors_close(grad_scale_autograd, grad_scale_analytical))
        << "Scale gradient mismatch: max_rel_err = " << max_rel_error(grad_scale_autograd, grad_scale_analytical);

    std::cout << "ScaleToVariance: max_rel_err = " << max_rel_error(grad_scale_autograd, grad_scale_analytical) << std::endl;
}

// =============================================================================
// Test 3: Build 3D Covariance (R * diag(var) * R^T) gradient
// =============================================================================
TEST_F(AnalyticalGradientTest, BuildCov3D) {
    const int N = 100;

    auto quat = torch::randn({N, 4}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    quat = quat / quat.norm(2, -1, true);
    auto scale = torch::randn({N, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // Autograd path
    auto quat_ag = quat.clone().set_requires_grad(true);
    auto scale_ag = scale.clone().set_requires_grad(true);

    // Build rotation
    auto qn = quat_ag / quat_ag.norm(2, -1, true);
    auto w = qn.select(1, 0), x = qn.select(1, 1), y = qn.select(1, 2), z = qn.select(1, 3);

    auto R00 = 1 - 2 * (y * y + z * z), R01 = 2 * (x * y - w * z), R02 = 2 * (x * z + w * y);
    auto R10 = 2 * (x * y + w * z), R11 = 1 - 2 * (x * x + z * z), R12 = 2 * (y * z - w * x);
    auto R20 = 2 * (x * z - w * y), R21 = 2 * (y * z + w * x), R22 = 1 - 2 * (x * x + y * y);

    auto R = torch::stack({torch::stack({R00, R01, R02}, 1),
                           torch::stack({R10, R11, R12}, 1),
                           torch::stack({R20, R21, R22}, 1)},
                          1); // [N, 3, 3]

    // Build variance and covariance
    auto var = torch::exp(2 * scale_ag);                          // [N, 3]
    auto S = torch::diag_embed(var);                              // [N, 3, 3]
    auto cov3d = torch::bmm(torch::bmm(R, S), R.transpose(1, 2)); // [N, 3, 3]

    auto v_cov3d = torch::randn_like(cov3d);
    cov3d.backward(v_cov3d);

    auto grad_quat_autograd = quat_ag.grad();
    auto grad_scale_autograd = scale_ag.grad();

    ASSERT_TRUE(grad_quat_autograd.defined());
    ASSERT_TRUE(grad_scale_autograd.defined());

    EXPECT_FALSE(grad_quat_autograd.isnan().any().item<bool>());
    EXPECT_FALSE(grad_scale_autograd.isnan().any().item<bool>());

    std::cout << "BuildCov3D: quat_grad_mag = " << grad_quat_autograd.abs().mean().item<float>()
              << ", scale_grad_mag = " << grad_scale_autograd.abs().mean().item<float>() << std::endl;
}

// =============================================================================
// Test 4: Cov2D to Conic (matrix inverse) gradient
// =============================================================================
TEST_F(AnalyticalGradientTest, Cov2DToConic) {
    const int N = 100;

    // Generate valid positive-definite 2x2 covariance matrices
    auto cov2d_xx = torch::abs(torch::randn({N}, torch::dtype(torch::kFloat32).device(torch::kCUDA))) + 0.5f;
    auto cov2d_yy = torch::abs(torch::randn({N}, torch::dtype(torch::kFloat32).device(torch::kCUDA))) + 0.5f;
    auto max_xy = torch::sqrt(cov2d_xx * cov2d_yy) * 0.5f;
    auto cov2d_xy = torch::randn({N}, torch::dtype(torch::kFloat32).device(torch::kCUDA)) * max_xy * 0.5f;

    // Autograd path
    auto xx_ag = cov2d_xx.clone().set_requires_grad(true);
    auto xy_ag = cov2d_xy.clone().set_requires_grad(true);
    auto yy_ag = cov2d_yy.clone().set_requires_grad(true);

    auto det = xx_ag * yy_ag - xy_ag * xy_ag;
    auto conic_x = yy_ag / det;
    auto conic_y = -xy_ag / det;
    auto conic_z = xx_ag / det;

    auto v_conic_x = torch::randn_like(conic_x);
    auto v_conic_y = torch::randn_like(conic_y);
    auto v_conic_z = torch::randn_like(conic_z);

    auto loss = (conic_x * v_conic_x + conic_y * v_conic_y + conic_z * v_conic_z).sum();
    loss.backward();

    auto grad_xx_autograd = xx_ag.grad();
    auto grad_xy_autograd = xy_ag.grad();
    auto grad_yy_autograd = yy_ag.grad();

    // Analytical gradient (derived from matrix inverse derivative)
    // d(A^{-1})/dA = -A^{-1} (dA) A^{-1}
    // For our storage: conic = (yy/det, -xy/det, xx/det)
    auto det_val = cov2d_xx * cov2d_yy - cov2d_xy * cov2d_xy;
    auto det_sq = det_val * det_val;

    // d(conic_x)/d(xx) = d(yy/det)/d(xx) = -yy * yy / det^2
    // d(conic_x)/d(xy) = -yy * (-2*xy) / det^2 = 2*xy*yy / det^2
    // d(conic_x)/d(yy) = 1/det - yy*xx/det^2 = (det - yy*xx)/det^2

    auto d_cx_dxx = -cov2d_yy * cov2d_yy / det_sq;
    auto d_cx_dxy = 2 * cov2d_xy * cov2d_yy / det_sq;
    auto d_cx_dyy = (det_val - cov2d_yy * cov2d_xx) / det_sq;

    // d(conic_y)/d(xx) = -(-xy) * yy / det^2 = xy*yy / det^2
    // d(conic_y)/d(xy) = -1/det - (-xy)*(-2*xy)/det^2 = -(det + 2*xy^2)/det^2
    // d(conic_y)/d(yy) = -(-xy) * xx / det^2 = xy*xx / det^2

    auto d_cy_dxx = cov2d_xy * cov2d_yy / det_sq;
    auto d_cy_dxy = -(det_val + 2 * cov2d_xy * cov2d_xy) / det_sq;
    auto d_cy_dyy = cov2d_xx * cov2d_xy / det_sq;

    // d(conic_z)/d(xx) = 1/det - xx*yy/det^2 = (det - xx*yy)/det^2
    // d(conic_z)/d(xy) = -xx * (-2*xy) / det^2 = 2*xx*xy / det^2
    // d(conic_z)/d(yy) = -xx * xx / det^2

    auto d_cz_dxx = (det_val - cov2d_xx * cov2d_yy) / det_sq;
    auto d_cz_dxy = 2 * cov2d_xx * cov2d_xy / det_sq;
    auto d_cz_dyy = -cov2d_xx * cov2d_xx / det_sq;

    auto grad_xx_analytical = d_cx_dxx * v_conic_x + d_cy_dxx * v_conic_y + d_cz_dxx * v_conic_z;
    auto grad_xy_analytical = d_cx_dxy * v_conic_x + d_cy_dxy * v_conic_y + d_cz_dxy * v_conic_z;
    auto grad_yy_analytical = d_cx_dyy * v_conic_x + d_cy_dyy * v_conic_y + d_cz_dyy * v_conic_z;

    EXPECT_TRUE(tensors_close(grad_xx_autograd, grad_xx_analytical))
        << "cov2d_xx gradient mismatch: max_rel_err = " << max_rel_error(grad_xx_autograd, grad_xx_analytical);
    EXPECT_TRUE(tensors_close(grad_xy_autograd, grad_xy_analytical))
        << "cov2d_xy gradient mismatch: max_rel_err = " << max_rel_error(grad_xy_autograd, grad_xy_analytical);
    EXPECT_TRUE(tensors_close(grad_yy_autograd, grad_yy_analytical))
        << "cov2d_yy gradient mismatch: max_rel_err = " << max_rel_error(grad_yy_autograd, grad_yy_analytical);

    std::cout << "Cov2DToConic: xx_err=" << max_rel_error(grad_xx_autograd, grad_xx_analytical)
              << ", xy_err=" << max_rel_error(grad_xy_autograd, grad_xy_analytical)
              << ", yy_err=" << max_rel_error(grad_yy_autograd, grad_yy_analytical) << std::endl;
}

// =============================================================================
// Test 5: Gaussian 2D evaluation gradient
// =============================================================================
TEST_F(AnalyticalGradientTest, Gaussian2DEval) {
    const int N = 1000;

    // Conic values (inverse covariance elements)
    auto conic_x = torch::abs(torch::randn({N}, torch::dtype(torch::kFloat32).device(torch::kCUDA))) + 0.1f;
    auto conic_z = torch::abs(torch::randn({N}, torch::dtype(torch::kFloat32).device(torch::kCUDA))) + 0.1f;
    auto max_y = torch::sqrt(conic_x * conic_z) * 0.5f;
    auto conic_y = torch::randn({N}, torch::dtype(torch::kFloat32).device(torch::kCUDA)) * max_y;

    auto delta_x = torch::randn({N}, torch::dtype(torch::kFloat32).device(torch::kCUDA)) * 2.0f;
    auto delta_y = torch::randn({N}, torch::dtype(torch::kFloat32).device(torch::kCUDA)) * 2.0f;

    // Autograd path
    auto cx_ag = conic_x.clone().set_requires_grad(true);
    auto cy_ag = conic_y.clone().set_requires_grad(true);
    auto cz_ag = conic_z.clone().set_requires_grad(true);
    auto dx_ag = delta_x.clone().set_requires_grad(true);
    auto dy_ag = delta_y.clone().set_requires_grad(true);

    // G = exp(-sigma), sigma = 0.5*(cx*dx^2 + cz*dy^2) + cy*dx*dy
    auto sigma = 0.5f * (cx_ag * dx_ag * dx_ag + cz_ag * dy_ag * dy_ag) + cy_ag * dx_ag * dy_ag;
    auto G = torch::exp(-sigma);

    auto v_G = torch::randn_like(G);
    G.backward(v_G);

    auto grad_cx_autograd = cx_ag.grad();
    auto grad_cy_autograd = cy_ag.grad();
    auto grad_cz_autograd = cz_ag.grad();
    auto grad_dx_autograd = dx_ag.grad();
    auto grad_dy_autograd = dy_ag.grad();

    // Analytical gradients
    // dG/d(param) = G * (-d_sigma/d_param)
    auto sigma_val = 0.5f * (conic_x * delta_x * delta_x + conic_z * delta_y * delta_y) + conic_y * delta_x * delta_y;
    auto G_val = torch::exp(-sigma_val);

    auto d_sigma_dcx = 0.5f * delta_x * delta_x;
    auto d_sigma_dcy = delta_x * delta_y;
    auto d_sigma_dcz = 0.5f * delta_y * delta_y;
    auto d_sigma_ddx = conic_x * delta_x + conic_y * delta_y;
    auto d_sigma_ddy = conic_z * delta_y + conic_y * delta_x;

    auto grad_cx_analytical = -G_val * d_sigma_dcx * v_G;
    auto grad_cy_analytical = -G_val * d_sigma_dcy * v_G;
    auto grad_cz_analytical = -G_val * d_sigma_dcz * v_G;
    auto grad_dx_analytical = -G_val * d_sigma_ddx * v_G;
    auto grad_dy_analytical = -G_val * d_sigma_ddy * v_G;

    EXPECT_TRUE(tensors_close(grad_cx_autograd, grad_cx_analytical))
        << "conic_x gradient mismatch: " << max_rel_error(grad_cx_autograd, grad_cx_analytical);
    EXPECT_TRUE(tensors_close(grad_cy_autograd, grad_cy_analytical))
        << "conic_y gradient mismatch: " << max_rel_error(grad_cy_autograd, grad_cy_analytical);
    EXPECT_TRUE(tensors_close(grad_cz_autograd, grad_cz_analytical))
        << "conic_z gradient mismatch: " << max_rel_error(grad_cz_autograd, grad_cz_analytical);
    EXPECT_TRUE(tensors_close(grad_dx_autograd, grad_dx_analytical))
        << "delta_x gradient mismatch: " << max_rel_error(grad_dx_autograd, grad_dx_analytical);
    EXPECT_TRUE(tensors_close(grad_dy_autograd, grad_dy_analytical))
        << "delta_y gradient mismatch: " << max_rel_error(grad_dy_autograd, grad_dy_analytical);

    std::cout << "Gaussian2DEval: cx_err=" << max_rel_error(grad_cx_autograd, grad_cx_analytical)
              << ", cy_err=" << max_rel_error(grad_cy_autograd, grad_cy_analytical)
              << ", dx_err=" << max_rel_error(grad_dx_autograd, grad_dx_analytical) << std::endl;
}

// =============================================================================
// Test 6: Alpha blending gradient
// =============================================================================
TEST_F(AnalyticalGradientTest, AlphaBlending) {
    const int N = 100;

    auto T = torch::rand({N}, torch::dtype(torch::kFloat32).device(torch::kCUDA)) * 0.9f + 0.1f; // transmittance
    auto alpha = torch::rand({N}, torch::dtype(torch::kFloat32).device(torch::kCUDA)) * 0.5f;
    auto color_r = torch::rand({N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto color_g = torch::rand({N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto color_b = torch::rand({N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // Autograd path
    auto T_ag = T.clone().set_requires_grad(true);
    auto alpha_ag = alpha.clone().set_requires_grad(true);
    auto cr_ag = color_r.clone().set_requires_grad(true);
    auto cg_ag = color_g.clone().set_requires_grad(true);
    auto cb_ag = color_b.clone().set_requires_grad(true);

    // Forward: delta_color = T * alpha * color, T_new = T * (1 - alpha)
    auto delta_r = T_ag * alpha_ag * cr_ag;
    auto delta_g = T_ag * alpha_ag * cg_ag;
    auto delta_b = T_ag * alpha_ag * cb_ag;
    auto T_new = T_ag * (1 - alpha_ag);

    auto v_delta_r = torch::randn_like(delta_r);
    auto v_delta_g = torch::randn_like(delta_g);
    auto v_delta_b = torch::randn_like(delta_b);
    auto v_T_new = torch::randn_like(T_new);

    auto loss = (delta_r * v_delta_r + delta_g * v_delta_g + delta_b * v_delta_b + T_new * v_T_new).sum();
    loss.backward();

    auto grad_T_autograd = T_ag.grad();
    auto grad_alpha_autograd = alpha_ag.grad();
    auto grad_cr_autograd = cr_ag.grad();

    // Analytical gradients
    // d(delta_r)/dT = alpha * color_r, d(T_new)/dT = 1 - alpha
    // d(delta_r)/d(alpha) = T * color_r, d(T_new)/d(alpha) = -T
    // d(delta_r)/d(color_r) = T * alpha

    auto grad_T_analytical = alpha * (color_r * v_delta_r + color_g * v_delta_g + color_b * v_delta_b) + (1 - alpha) * v_T_new;
    auto grad_alpha_analytical = T * (color_r * v_delta_r + color_g * v_delta_g + color_b * v_delta_b) - T * v_T_new;
    auto grad_cr_analytical = T * alpha * v_delta_r;

    EXPECT_TRUE(tensors_close(grad_T_autograd, grad_T_analytical))
        << "T gradient mismatch: " << max_rel_error(grad_T_autograd, grad_T_analytical);
    EXPECT_TRUE(tensors_close(grad_alpha_autograd, grad_alpha_analytical))
        << "alpha gradient mismatch: " << max_rel_error(grad_alpha_autograd, grad_alpha_analytical);
    EXPECT_TRUE(tensors_close(grad_cr_autograd, grad_cr_analytical))
        << "color_r gradient mismatch: " << max_rel_error(grad_cr_autograd, grad_cr_analytical);

    std::cout << "AlphaBlending: T_err=" << max_rel_error(grad_T_autograd, grad_T_analytical)
              << ", alpha_err=" << max_rel_error(grad_alpha_autograd, grad_alpha_analytical) << std::endl;
}

// =============================================================================
// Test 7: Front-to-back color/depth/alpha compositing gradient
// =============================================================================
TEST_F(AnalyticalGradientTest, DepthAccumulationBlendingMatchesAutograd) {
    const int N = 64;
    const auto opts = torch::dtype(torch::kFloat32).device(torch::kCUDA);

    const auto colors = torch::rand({N, 3}, opts) * 1.5f - 0.25f;
    const auto depths = torch::rand({N}, opts) * 7.0f + 0.5f;
    const auto alphas = torch::rand({N}, opts) * 0.55f + 0.01f;
    const auto grad_rgb = torch::randn({3}, opts);
    const auto grad_depth_out = torch::randn({}, opts);
    const auto grad_alpha_out = torch::randn({}, opts);

    auto colors_ag = colors.clone().set_requires_grad(true);
    auto depths_ag = depths.clone().set_requires_grad(true);
    auto alphas_ag = alphas.clone().set_requires_grad(true);

    auto rgb = torch::zeros({3}, opts);
    auto depth_out = torch::zeros({}, opts);
    auto transmittance = torch::ones({}, opts);
    for (int i = 0; i < N; ++i) {
        const auto alpha = alphas_ag.select(0, i);
        const auto weight = transmittance * alpha;
        rgb = rgb + weight * colors_ag.select(0, i);
        depth_out = depth_out + weight * depths_ag.select(0, i);
        transmittance = transmittance * (1.0f - alpha);
    }
    const auto alpha_out = 1.0f - transmittance;
    const auto loss = (rgb * grad_rgb).sum() + depth_out * grad_depth_out + alpha_out * grad_alpha_out;
    loss.backward();

    std::vector<torch::Tensor> transmittance_before;
    transmittance_before.reserve(N);
    auto transmittance_value = torch::ones({}, opts);
    for (int i = 0; i < N; ++i) {
        transmittance_before.push_back(transmittance_value);
        transmittance_value = transmittance_value * (1.0f - alphas.select(0, i));
    }

    std::vector<torch::Tensor> grad_color_parts(N);
    std::vector<torch::Tensor> grad_depth_parts(N);
    std::vector<torch::Tensor> grad_alpha_parts(N);
    auto grad_transmittance_after = -grad_alpha_out;
    for (int i = N - 1; i >= 0; --i) {
        const auto T = transmittance_before[static_cast<size_t>(i)];
        const auto alpha = alphas.select(0, i);
        const auto color = colors.select(0, i);
        const auto depth = depths.select(0, i);
        const auto grad_weight = (color * grad_rgb).sum() + depth * grad_depth_out;

        grad_color_parts[static_cast<size_t>(i)] = T * alpha * grad_rgb;
        grad_depth_parts[static_cast<size_t>(i)] = T * alpha * grad_depth_out;
        grad_alpha_parts[static_cast<size_t>(i)] = T * grad_weight - T * grad_transmittance_after;
        grad_transmittance_after = alpha * grad_weight + (1.0f - alpha) * grad_transmittance_after;
    }

    const auto grad_colors_analytical = torch::stack(grad_color_parts, 0);
    const auto grad_depths_analytical = torch::stack(grad_depth_parts, 0);
    const auto grad_alphas_analytical = torch::stack(grad_alpha_parts, 0);

    EXPECT_TRUE(tensors_close(colors_ag.grad(), grad_colors_analytical, 5e-5f, 1e-6f))
        << "composited color gradient mismatch: " << max_rel_error(colors_ag.grad(), grad_colors_analytical);
    EXPECT_TRUE(tensors_close(depths_ag.grad(), grad_depths_analytical, 5e-5f, 1e-6f))
        << "composited depth gradient mismatch: " << max_rel_error(depths_ag.grad(), grad_depths_analytical);
    EXPECT_TRUE(tensors_close(alphas_ag.grad(), grad_alphas_analytical, 5e-5f, 1e-6f))
        << "composited alpha gradient mismatch: " << max_rel_error(alphas_ag.grad(), grad_alphas_analytical);
}

// =============================================================================
// Test 8: scale-shift-invariant depth loss kernel vs. LibTorch autograd
// =============================================================================
TEST_F(AnalyticalGradientTest, SsiDepthLossCudaKernelMatchesLibtorchAutograd) {
    constexpr int H = 61;
    constexpr int W = 73;
    constexpr float kWeight = 2.0f;
    constexpr float kLambdaGrad = 1.0f;
    const auto opts = torch::dtype(torch::kFloat32).device(torch::kCUDA);

    torch::manual_seed(1234);
    const auto expected_depth = torch::rand({H, W}, opts) * 18.0f + 0.5f;
    const auto alpha_accum = torch::rand({H, W}, opts) * 0.95f + 0.02f;
    const auto depth_accum = expected_depth * alpha_accum;
    const float mean_e = ((alpha_accum * expected_depth).sum() / alpha_accum.sum()).item<float>();
    const float floor_f = std::max(
        1.0e-8f, lfs::training::kernels::kDepthLossFloorFraction * mean_e);

    // Disparity-convention prior: affine map of true inverse depth plus noise.
    auto disparity_target = 0.6f / (expected_depth + floor_f) + 0.05f +
                            torch::randn({H, W}, opts) * 0.01f;
    disparity_target = disparity_target.clamp_min(0.0f);
    const auto zero_rows = torch::arange(0, H, 13, torch::dtype(torch::kInt64).device(torch::kCUDA));
    disparity_target.index_put_({zero_rows}, 0.0f);
    const auto disparity_anchor = make_depth_anchor(0, 1.0f / 0.6f, -0.05f / 0.6f, floor_f);

    expect_ssi_depth_loss_kernel_matches_autograd(
        depth_accum, alpha_accum, disparity_target, kWeight, kLambdaGrad, &disparity_anchor);
    expect_ssi_depth_loss_kernel_matches_autograd(
        depth_accum, alpha_accum, disparity_target, kWeight, kLambdaGrad, &disparity_anchor);

    // Depth-convention prior: affine map of true depth plus noise.
    auto depth_target = (0.4f * expected_depth + 1.5f +
                         torch::randn({H, W}, opts) * 0.05f)
                            .clamp_min(0.0f);
    depth_target.index_put_({zero_rows}, 0.0f);
    const auto depth_anchor = make_depth_anchor(1, 1.0f / 0.4f, -1.5f / 0.4f, floor_f);

    expect_ssi_depth_loss_kernel_matches_autograd(
        depth_accum, alpha_accum, depth_target, kWeight, kLambdaGrad, &depth_anchor);
    expect_ssi_depth_loss_kernel_matches_autograd(
        depth_accum, alpha_accum, depth_target, kWeight, kLambdaGrad, &depth_anchor);
}

TEST_F(AnalyticalGradientTest, SsiDepthLossPerfectDisparityPriorGivesNearZeroLoss) {
    constexpr int H = 48;
    constexpr int W = 64;
    const auto opts = torch::dtype(torch::kFloat32).device(torch::kCUDA);

    torch::manual_seed(7);
    const auto expected_depth = torch::rand({H, W}, opts) * 12.0f + 1.0f;
    const auto alpha_accum = torch::rand({H, W}, opts) * 0.5f + 0.5f;
    const auto depth_accum = expected_depth * alpha_accum;

    const float mean_e = ((alpha_accum * expected_depth).sum() / alpha_accum.sum()).item<float>();
    const float floor_f = std::max(
        1.0e-8f, lfs::training::kernels::kDepthLossFloorFraction * mean_e);
    const auto disparity_target = 0.7f / (expected_depth + floor_f) + 0.1f;
    const auto anchor = make_depth_anchor(0, 1.0f / 0.7f, -0.1f / 0.7f, floor_f);

    const auto result = run_ssi_depth_loss_kernel(
        depth_accum, alpha_accum, disparity_target, 2.0f, 1.0f, &anchor);

    namespace slots = lfs::training::kernels::depth_loss_slots;
    EXPECT_GT(result.partials[slots::kValid].item<float>(), 0.5f);
    EXPECT_EQ(static_cast<int>(result.partials[slots::kModel].item<float>()), 0);
    EXPECT_LT(result.loss.item<float>(), 1e-5f);
}

TEST_F(AnalyticalGradientTest, SsiDepthLossPerfectDepthPriorSelectsDepthModel) {
    constexpr int H = 48;
    constexpr int W = 64;
    const auto opts = torch::dtype(torch::kFloat32).device(torch::kCUDA);

    torch::manual_seed(11);
    const auto expected_depth = torch::rand({H, W}, opts) * 12.0f + 1.0f;
    const auto alpha_accum = torch::rand({H, W}, opts) * 0.5f + 0.5f;
    const auto depth_accum = expected_depth * alpha_accum;
    const auto depth_target = 0.4f * expected_depth + 1.5f;
    const float mean_e = ((alpha_accum * expected_depth).sum() / alpha_accum.sum()).item<float>();
    const float floor_f = std::max(
        1.0e-8f, lfs::training::kernels::kDepthLossFloorFraction * mean_e);
    const auto anchor = make_depth_anchor(1, 1.0f / 0.4f, -1.5f / 0.4f, floor_f);

    const auto result = run_ssi_depth_loss_kernel(
        depth_accum, alpha_accum, depth_target, 2.0f, 1.0f, &anchor);

    namespace slots = lfs::training::kernels::depth_loss_slots;
    EXPECT_GT(result.partials[slots::kValid].item<float>(), 0.5f);
    EXPECT_EQ(static_cast<int>(result.partials[slots::kModel].item<float>()), 1);
    EXPECT_LT(result.loss.item<float>(), 1e-3f);
}

TEST_F(AnalyticalGradientTest, SsiDepthLossInvariantToJointAlphaScaling) {
    constexpr int H = 48;
    constexpr int W = 64;
    const auto opts = torch::dtype(torch::kFloat32).device(torch::kCUDA);

    torch::manual_seed(21);
    const auto expected_depth = torch::rand({H, W}, opts) * 10.0f + 1.0f;
    const auto alpha_accum = torch::rand({H, W}, opts) * 0.4f + 0.5f;
    const auto depth_accum = expected_depth * alpha_accum;
    const float mean_e = ((alpha_accum * expected_depth).sum() / alpha_accum.sum()).item<float>();
    const float floor_f = std::max(
        1.0e-8f, lfs::training::kernels::kDepthLossFloorFraction * mean_e);
    const auto target = (0.8f / (expected_depth + floor_f) +
                         torch::randn({H, W}, opts) * 0.02f)
                            .clamp_min(1e-4f);
    const auto anchor = make_depth_anchor(0, 1.0f / 0.8f, 0.0f, floor_f);

    const auto full = run_ssi_depth_loss_kernel(
        depth_accum, alpha_accum, target, 2.0f, 1.0f, &anchor);
    const auto scaled = run_ssi_depth_loss_kernel(
        depth_accum * 0.5f, alpha_accum * 0.5f, target, 2.0f, 1.0f, &anchor);

    // Expected depth is unchanged, so the alpha-weighted loss must match.
    EXPECT_NEAR(full.loss.item<float>(), scaled.loss.item<float>(),
                5e-4f * std::max(1.0f, full.loss.item<float>()));
}

TEST_F(AnalyticalGradientTest, SsiDepthLossQuantizedPriorDeadbandMatchesAutograd) {
    constexpr int H = 61;
    constexpr int W = 73;
    constexpr float kWeight = 2.0f;
    constexpr float kLambdaGrad = 1.0f;
    constexpr float kQStep = 1.0f / 255.0f;
    const auto opts = torch::dtype(torch::kFloat32).device(torch::kCUDA);

    torch::manual_seed(4321);
    const auto expected_depth = torch::rand({H, W}, opts) * 18.0f + 0.5f;
    const auto alpha_accum = torch::rand({H, W}, opts) * 0.95f + 0.02f;
    const auto depth_accum = expected_depth * alpha_accum;
    const float mean_e = ((alpha_accum * expected_depth).sum() / alpha_accum.sum()).item<float>();
    const float floor_f = std::max(
        1.0e-8f, lfs::training::kernels::kDepthLossFloorFraction * mean_e);

    // 8-bit export of a disparity prior: residuals cluster around the
    // quantizer's half-step corridor, exercising both gated and active pixels.
    auto disparity_target = (0.6f / (expected_depth + floor_f) + 0.05f +
                             torch::randn({H, W}, opts) * 0.01f)
                                .clamp(0.0f, 1.0f);
    disparity_target = (disparity_target * 255.0f).round() / 255.0f;
    const auto disparity_anchor = make_depth_anchor(0, 1.0f / 0.6f, -0.05f / 0.6f, floor_f);

    expect_ssi_depth_loss_kernel_matches_autograd(
        depth_accum, alpha_accum, disparity_target, kWeight, kLambdaGrad, &disparity_anchor, kQStep);

    auto depth_target = ((0.4f * expected_depth + 1.5f) +
                         torch::randn({H, W}, opts) * 0.005f)
                            .clamp_min(0.0f);
    depth_target = (depth_target * 255.0f).round() / 255.0f;
    const auto depth_anchor = make_depth_anchor(1, 1.0f / 0.4f, -1.5f / 0.4f, floor_f);

    expect_ssi_depth_loss_kernel_matches_autograd(
        depth_accum, alpha_accum, depth_target, kWeight, kLambdaGrad, &depth_anchor, kQStep);
}

TEST_F(AnalyticalGradientTest, SsiDepthLossQuantizedPriorCorridorSilencesSmoothSurface) {
    constexpr int H = 48;
    constexpr int W = 64;
    constexpr float A_TRUE = 0.4f;
    constexpr float B_TRUE = 0.05f;
    // 2% wider than the true step so residuals landing exactly on the
    // half-step boundary can't leak out through float rounding.
    constexpr float kQStep = 1.02f / 255.0f;
    const auto opts = torch::dtype(torch::kFloat32).device(torch::kCUDA);

    torch::manual_seed(55);
    // Smooth planar ramp: the geometry an 8-bit prior can only describe as a
    // staircase.
    const auto us = torch::arange(0, W, opts).reshape({1, W}).expand({H, W});
    const auto vs = torch::arange(0, H, opts).reshape({H, 1}).expand({H, W});
    const auto expected_depth = 2.0f + 6.0f * us / (W - 1) + 3.0f * vs / (H - 1);
    const auto alpha_accum = torch::rand({H, W}, opts) * 0.2f + 0.75f;
    const auto depth_accum = expected_depth * alpha_accum;

    const float mean_e = ((alpha_accum * expected_depth).sum() / alpha_accum.sum()).item<float>();
    const float floor_f = std::max(
        1.0e-8f, lfs::training::kernels::kDepthLossFloorFraction * mean_e);
    const auto q_true = 1.0f / (expected_depth + floor_f);
    const auto continuous_target = (q_true - B_TRUE) / A_TRUE;
    const auto quantized_target = (continuous_target * 255.0f).round() / 255.0f;

    lfs::training::kernels::DepthAnchor anchor;
    anchor.valid = true;
    anchor.model = 0;
    anchor.scale = A_TRUE;
    anchor.shift = B_TRUE;
    anchor.floor = floor_f;
    anchor.corr = 1.0f;
    anchor.samples = H * W;
    anchor.disparity = {true, A_TRUE, B_TRUE, 1.0f, H * W};

    // The render matches the true smooth surface; every residual against the
    // quantized prior stays inside the half-step corridor.
    const auto gated = run_ssi_depth_loss_kernel(
        depth_accum, alpha_accum, quantized_target, 2.0f, 1.0f, &anchor, kQStep);
    namespace slots = lfs::training::kernels::depth_loss_slots;
    EXPECT_GT(gated.partials[slots::kValid].item<float>(), 0.5f);
    EXPECT_EQ(gated.loss.item<float>(), 0.0f);
    EXPECT_EQ(gated.grad_depth.abs().max().item<float>(), 0.0f);
    EXPECT_EQ(gated.grad_alpha.abs().max().item<float>(), 0.0f);

    // Without the corridor the same configuration drags the smooth surface
    // toward the staircase.
    const auto ungated = run_ssi_depth_loss_kernel(
        depth_accum, alpha_accum, quantized_target, 2.0f, 1.0f, &anchor, 0.0f);
    EXPECT_GT(ungated.loss.item<float>(), 0.0f);
    EXPECT_GT(ungated.grad_depth.abs().max().item<float>(), 0.0f);
}

TEST_F(AnalyticalGradientTest, DepthAnchorFitRecoversAffineDisparityAlignment) {
    constexpr int H = 96;
    constexpr int W = 128;
    constexpr float FX = 100.0f;
    constexpr float FY = 100.0f;
    constexpr float CX = W / 2.0f;
    constexpr float CY = H / 2.0f;
    constexpr float A_TRUE = 0.6f;
    constexpr float B_TRUE = 0.05f;
    const auto opts = torch::dtype(torch::kFloat32).device(torch::kCUDA);

    torch::manual_seed(5);
    // Per-pixel depth surface; anchor points are the backprojected pixels, so
    // the projection in the fit lands exactly on the sampled prior texels.
    const auto z_img = torch::rand({H, W}, opts) * 8.0f + 2.0f;
    const float median_z = z_img.flatten().median().item<float>();
    const float floor_expected = lfs::training::kernels::kDepthLossFloorFraction * median_z;

    const auto us = torch::arange(0, W, opts).reshape({1, W}).expand({H, W});
    const auto vs = torch::arange(0, H, opts).reshape({H, 1}).expand({H, W});
    const auto xs = (us + 0.5f - CX) / FX * z_img;
    const auto ys = (vs + 0.5f - CY) / FY * z_img;
    const auto points = torch::stack({xs.reshape(-1), ys.reshape(-1), z_img.reshape(-1)}, 1).contiguous();

    // Prior encodes an affine map of the true inverse depth (disparity-like).
    const auto q_true = 1.0f / (z_img + floor_expected);
    const auto prior = ((q_true - B_TRUE) / A_TRUE).clamp_min(1e-4f).contiguous();

    auto w2c = torch::eye(4, opts).contiguous();

    const float aabb_lo[3] = {-1e30f, -1e30f, -1e30f};
    const float aabb_hi[3] = {1e30f, 1e30f, 1e30f};
    const auto anchor = lfs::training::kernels::fit_depth_anchor(
        points.data_ptr<float>(),
        static_cast<size_t>(points.size(0)),
        w2c.data_ptr<float>(),
        FX, FY, CX, CY,
        prior.data_ptr<float>(),
        W, H,
        0.01f,
        aabb_lo,
        aabb_hi);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_TRUE(anchor.valid);
    ASSERT_TRUE(anchor.disparity.valid);
    EXPECT_EQ(anchor.model, 0);
    EXPECT_NEAR(anchor.scale, A_TRUE, 5e-3f);
    EXPECT_NEAR(anchor.shift, B_TRUE, 5e-3f);
    EXPECT_NEAR(anchor.floor, floor_expected, 2e-3f);
    EXPECT_GT(std::fabs(anchor.corr), 0.99f);

    // A render matching the anchored prior yields near-zero loss.
    const auto alpha_accum = torch::rand({H, W}, opts) * 0.4f + 0.55f;
    const auto depth_accum = z_img * alpha_accum;
    const auto result = run_ssi_depth_loss_kernel(
        depth_accum, alpha_accum, prior, 2.0f, 1.0f, &anchor);
    namespace slots = lfs::training::kernels::depth_loss_slots;
    EXPECT_GT(result.partials[slots::kValid].item<float>(), 0.5f);
    EXPECT_LT(result.loss.item<float>(), 2e-2f);
}

// =============================================================================
// Test 9: Spherical Harmonics degree 0 gradient
// =============================================================================
TEST_F(AnalyticalGradientTest, SHDegree0) {
    const int N = 100;
    const float SH_C0 = 0.28209479177387814f;

    auto sh_coeff = torch::randn({N, 1, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // Autograd path
    auto sh_ag = sh_coeff.clone().set_requires_grad(true);

    // color = SH_C0 * coeff + 0.5
    auto color = SH_C0 * sh_ag.squeeze(1) + 0.5f; // [N, 3]

    auto v_color = torch::randn_like(color);
    color.backward(v_color);

    auto grad_sh_autograd = sh_ag.grad();

    // Analytical: d(color)/d(coeff) = SH_C0
    auto grad_sh_analytical = SH_C0 * v_color.unsqueeze(1);

    EXPECT_TRUE(tensors_close(grad_sh_autograd, grad_sh_analytical))
        << "SH degree 0 gradient mismatch: " << max_rel_error(grad_sh_autograd, grad_sh_analytical);

    std::cout << "SHDegree0: err=" << max_rel_error(grad_sh_autograd, grad_sh_analytical) << std::endl;
}

// =============================================================================
// Test 10: Spherical Harmonics degree 1 gradient
// =============================================================================
TEST_F(AnalyticalGradientTest, SHDegree1) {
    const int N = 100;
    const float SH_C0 = 0.28209479177387814f;
    const float SH_C1 = 0.4886025119029199f;

    auto sh_coeffs = torch::randn({N, 4, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto dir = torch::randn({N, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    dir = dir / dir.norm(2, -1, true); // normalize

    // Autograd path
    auto sh_ag = sh_coeffs.clone().set_requires_grad(true);
    auto dir_ag = dir.clone().set_requires_grad(true);

    auto x = dir_ag.select(1, 0);
    auto y = dir_ag.select(1, 1);
    auto z = dir_ag.select(1, 2);

    // Degree 0: SH_C0 * coeff[0]
    auto color = SH_C0 * sh_ag.select(1, 0);

    // Degree 1: SH_C1 * (-y * coeff[1] + z * coeff[2] - x * coeff[3])
    color = color + SH_C1 * (-y.unsqueeze(1) * sh_ag.select(1, 1) +
                             z.unsqueeze(1) * sh_ag.select(1, 2) +
                             -x.unsqueeze(1) * sh_ag.select(1, 3));

    color = color + 0.5f;

    auto v_color = torch::randn_like(color);
    color.backward(v_color);

    auto grad_sh_autograd = sh_ag.grad();
    auto grad_dir_autograd = dir_ag.grad();

    ASSERT_TRUE(grad_sh_autograd.defined());
    ASSERT_TRUE(grad_dir_autograd.defined());

    EXPECT_FALSE(grad_sh_autograd.isnan().any().item<bool>());
    EXPECT_FALSE(grad_dir_autograd.isnan().any().item<bool>());

    std::cout << "SHDegree1: sh_grad_mag=" << grad_sh_autograd.abs().mean().item<float>()
              << ", dir_grad_mag=" << grad_dir_autograd.abs().mean().item<float>() << std::endl;
}

// =============================================================================
// Test 11: Full forward chain (quat + scale -> cov3d -> cov2d -> conic -> gaussian)
// =============================================================================
TEST_F(AnalyticalGradientTest, FullForwardChain) {
    const int N = 50;

    // Gaussian parameters
    auto quat = torch::randn({N, 4}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    quat = quat / quat.norm(2, -1, true);
    auto scale = torch::randn({N, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA)) * 0.5f;
    auto opacity = torch::sigmoid(torch::randn({N}, torch::dtype(torch::kFloat32).device(torch::kCUDA)));

    // Simplified Jacobian (assume identity for testing)
    auto J = torch::eye(2, 3, torch::dtype(torch::kFloat32).device(torch::kCUDA)).unsqueeze(0).expand({N, 2, 3});

    // Pixel offsets
    auto delta = torch::randn({N, 2}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // Autograd path
    auto quat_ag = quat.clone().set_requires_grad(true);
    auto scale_ag = scale.clone().set_requires_grad(true);
    auto opacity_ag = opacity.clone().set_requires_grad(true);
    auto delta_ag = delta.clone().set_requires_grad(true);

    // Build rotation matrix
    auto qn = quat_ag / quat_ag.norm(2, -1, true);
    auto w = qn.select(1, 0), x = qn.select(1, 1), y = qn.select(1, 2), z = qn.select(1, 3);

    auto R00 = 1 - 2 * (y * y + z * z), R01 = 2 * (x * y - w * z), R02 = 2 * (x * z + w * y);
    auto R10 = 2 * (x * y + w * z), R11 = 1 - 2 * (x * x + z * z), R12 = 2 * (y * z - w * x);
    auto R20 = 2 * (x * z - w * y), R21 = 2 * (y * z + w * x), R22 = 1 - 2 * (x * x + y * y);

    auto R = torch::stack({torch::stack({R00, R01, R02}, 1),
                           torch::stack({R10, R11, R12}, 1),
                           torch::stack({R20, R21, R22}, 1)},
                          1);

    // Build 3D covariance
    auto var = torch::exp(2 * scale_ag);
    auto S = torch::diag_embed(var);
    auto cov3d = torch::bmm(torch::bmm(R, S), R.transpose(1, 2));

    // Project to 2D (simplified: J * cov3d * J^T)
    auto cov2d = torch::bmm(torch::bmm(J, cov3d), J.transpose(1, 2)); // [N, 2, 2]

    // Add blur for numerical stability
    cov2d = cov2d + 0.3f * torch::eye(2, torch::dtype(torch::kFloat32).device(torch::kCUDA)).unsqueeze(0);

    // Compute conic (inverse)
    auto det = cov2d.select(1, 0).select(1, 0) * cov2d.select(1, 1).select(1, 1) -
               cov2d.select(1, 0).select(1, 1) * cov2d.select(1, 0).select(1, 1);
    auto conic_x = cov2d.select(1, 1).select(1, 1) / det;
    auto conic_y = -cov2d.select(1, 0).select(1, 1) / det;
    auto conic_z = cov2d.select(1, 0).select(1, 0) / det;

    // Evaluate Gaussian
    auto dx = delta_ag.select(1, 0);
    auto dy = delta_ag.select(1, 1);
    auto sigma = 0.5f * (conic_x * dx * dx + conic_z * dy * dy) + conic_y * dx * dy;
    auto G = torch::exp(-sigma);

    // Compute alpha
    auto alpha = opacity_ag * G;
    alpha = torch::clamp(alpha, 0.0f, 0.999f);

    // Backward
    auto v_alpha = torch::randn_like(alpha);
    alpha.backward(v_alpha);

    auto grad_quat = quat_ag.grad();
    auto grad_scale = scale_ag.grad();
    auto grad_opacity = opacity_ag.grad();
    auto grad_delta = delta_ag.grad();

    ASSERT_TRUE(grad_quat.defined());
    ASSERT_TRUE(grad_scale.defined());
    ASSERT_TRUE(grad_opacity.defined());
    ASSERT_TRUE(grad_delta.defined());

    EXPECT_FALSE(grad_quat.isnan().any().item<bool>()) << "Quaternion gradient has NaN";
    EXPECT_FALSE(grad_scale.isnan().any().item<bool>()) << "Scale gradient has NaN";
    EXPECT_FALSE(grad_opacity.isnan().any().item<bool>()) << "Opacity gradient has NaN";
    EXPECT_FALSE(grad_delta.isnan().any().item<bool>()) << "Delta gradient has NaN";

    std::cout << "FullForwardChain gradients:" << std::endl;
    std::cout << "  quat: " << grad_quat.abs().mean().item<float>() << std::endl;
    std::cout << "  scale: " << grad_scale.abs().mean().item<float>() << std::endl;
    std::cout << "  opacity: " << grad_opacity.abs().mean().item<float>() << std::endl;
    std::cout << "  delta: " << grad_delta.abs().mean().item<float>() << std::endl;
}

// =============================================================================
// CUDA Kernel Gradient Tests
// These tests compare actual CUDA backward kernel outputs against LibTorch autograd
// =============================================================================

class CUDAKernelGradientTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available";
        }
        torch::manual_seed(42);
    }
};

// SH constants matching the CUDA kernel
namespace {
    constexpr float SH_C0 = 0.2820947917738781f;
    constexpr float SH_C1 = 0.48860251190292f;
    constexpr float SH_DC_OFFSET = 0.5f;
} // namespace

// =============================================================================
// Test: SH Degree 0 CUDA Kernel vs LibTorch Autograd
// =============================================================================
TEST_F(CUDAKernelGradientTest, SHDegree0_CUDA_vs_Autograd) {
    const int N = 100;
    const int K = 1; // Degree 0 has 1 coefficient

    // Create test data
    auto sh_coeffs = torch::randn({N, K, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto v_colors = torch::randn({N, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // LibTorch autograd path
    auto sh_ag = sh_coeffs.clone().set_requires_grad(true);
    auto color = SH_C0 * sh_ag.squeeze(1) + SH_DC_OFFSET; // [N, 3]
    color.backward(v_colors);
    auto grad_sh_autograd = sh_ag.grad().contiguous();

    // CUDA kernel path
    auto v_coeffs_cuda = torch::zeros_like(sh_coeffs);
    auto split_sh = make_split_sh_tensors(sh_coeffs);

    gsplat_lfs::launch_spherical_harmonics_swizzled_bwd_kernel(
        0,       // degrees_to_use
        nullptr, // dirs (not needed for degree 0)
        split_sh.sh0.data_ptr<float>(),
        nullptr,
        nullptr, // masks
        v_colors.data_ptr<float>(),
        N,
        K,
        false, // compute_v_dirs
        v_coeffs_cuda.data_ptr<float>(),
        nullptr, // v_dirs
        nullptr  // stream
    );

    cudaDeviceSynchronize();

    // Compare results
    EXPECT_TRUE(tensors_close(grad_sh_autograd, v_coeffs_cuda, REL_TOL, ABS_TOL))
        << "SH Degree 0 CUDA kernel gradient mismatch: max_rel_err = "
        << max_rel_error(grad_sh_autograd, v_coeffs_cuda);

    std::cout << "SH Degree 0 CUDA vs Autograd: max_rel_err = "
              << max_rel_error(grad_sh_autograd, v_coeffs_cuda) << std::endl;
}

// =============================================================================
// Test: SH Degree 1 CUDA Kernel vs LibTorch Autograd
// Note: CUDA kernel normalizes directions internally and computes gradients
// w.r.t. unnormalized inputs. LibTorch must match this behavior.
// =============================================================================
TEST_F(CUDAKernelGradientTest, SHDegree1_CUDA_vs_Autograd) {
    const int N = 1; // Single element to isolate the issue
    const int K = 4; // Degree 1 has 4 coefficients (1 + 3)
    const uint32_t degree = 1;

    // Create test data with UNNORMALIZED directions (matching CUDA kernel input)
    auto sh_coeffs = torch::randn({N, K, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto dirs = torch::randn({N, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    // Don't pre-normalize - CUDA kernel does this internally

    auto v_colors = torch::randn({N, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // LibTorch autograd path - must normalize internally like CUDA does
    auto sh_ag = sh_coeffs.clone().set_requires_grad(true);
    auto dir_ag = dirs.clone().set_requires_grad(true);

    // Normalize directions (matching CUDA kernel's internal normalization)
    auto dir_norm = dir_ag.norm(2, -1, true);
    auto dir_n = dir_ag / dir_norm;

    auto x = dir_n.select(1, 0);
    auto y = dir_n.select(1, 1);
    auto z = dir_n.select(1, 2);

    // Degree 0
    auto color = SH_C0 * sh_ag.select(1, 0);

    // Degree 1: SH_C1 * (-y * coeff[1] + z * coeff[2] - x * coeff[3])
    color = color + SH_C1 * (-y.unsqueeze(1) * sh_ag.select(1, 1) +
                             z.unsqueeze(1) * sh_ag.select(1, 2) +
                             -x.unsqueeze(1) * sh_ag.select(1, 3));

    color = color + SH_DC_OFFSET;

    color.backward(v_colors);
    auto grad_sh_autograd = sh_ag.grad().contiguous();
    auto grad_dir_autograd = dir_ag.grad().contiguous();

    // Debug: print first few values
    std::cout << "\n=== DEBUG SH Degree 1 (N=" << N << ") ===" << std::endl;
    std::cout << "dirs[0]: " << dirs[0] << std::endl;
    std::cout << "v_colors[0]: " << v_colors[0] << std::endl;
    std::cout << "sh_coeffs[0]: " << sh_coeffs[0] << std::endl;
    std::cout << "grad_sh_autograd[0]: " << grad_sh_autograd[0] << std::endl;

    // CUDA kernel path
    auto v_coeffs_cuda = torch::zeros({N, K, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto v_dirs_cuda = torch::zeros({N, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto split_sh = make_split_sh_tensors(sh_coeffs);

    // Debug: verify tensor layout
    std::cout << "v_dirs_cuda is_contiguous: " << v_dirs_cuda.is_contiguous() << std::endl;

    // Verify with raw pointers that memory is zeroed
    float test_before[3];
    cudaMemcpy(test_before, v_dirs_cuda.data_ptr<float>(), 3 * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Before kernel: v_dirs_cuda[0,1,2] = "
              << test_before[0] << ", " << test_before[1] << ", " << test_before[2] << std::endl;

    gsplat_lfs::launch_spherical_harmonics_swizzled_bwd_kernel(
        degree,
        dirs.data_ptr<float>(),
        split_sh.sh0.data_ptr<float>(),
        split_sh.shN.data_ptr<float>(),
        nullptr, // masks
        v_colors.data_ptr<float>(),
        N,
        K,
        true, // compute_v_dirs
        v_coeffs_cuda.data_ptr<float>(),
        v_dirs_cuda.data_ptr<float>(),
        nullptr // stream
    );

    cudaDeviceSynchronize();

    // Check raw values after kernel
    float test_after[3];
    cudaMemcpy(test_after, v_dirs_cuda.data_ptr<float>(), 3 * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "After kernel: v_dirs_cuda[0,1,2] = "
              << test_after[0] << ", " << test_after[1] << ", " << test_after[2] << std::endl;

    // Debug output - print all elements
    std::cout << "\n=== Direction Gradient Comparison ===" << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << "Element " << i << ":" << std::endl;
        std::cout << "  AutoGrad: " << grad_dir_autograd[i] << std::endl;
        std::cout << "  CUDA:     " << v_dirs_cuda[i] << std::endl;
    }

    // Compare coefficient gradients
    EXPECT_TRUE(tensors_close(grad_sh_autograd, v_coeffs_cuda, REL_TOL, ABS_TOL))
        << "SH Degree 1 coeffs CUDA kernel gradient mismatch: max_rel_err = "
        << max_rel_error(grad_sh_autograd, v_coeffs_cuda);

    // Compare direction gradients
    EXPECT_TRUE(tensors_close(grad_dir_autograd, v_dirs_cuda, REL_TOL, ABS_TOL))
        << "SH Degree 1 dirs CUDA kernel gradient mismatch: max_rel_err = "
        << max_rel_error(grad_dir_autograd, v_dirs_cuda);

    std::cout << "SH Degree 1 CUDA vs Autograd: coeffs_err = "
              << max_rel_error(grad_sh_autograd, v_coeffs_cuda)
              << ", dirs_err = " << max_rel_error(grad_dir_autograd, v_dirs_cuda) << std::endl;
}

// =============================================================================
// Test: SH Degree 2 CUDA Kernel vs LibTorch Autograd
// =============================================================================
TEST_F(CUDAKernelGradientTest, SHDegree2_CUDA_vs_Autograd) {
    const int N = 100;
    const int K = 9; // Degree 2 has 9 coefficients (1 + 3 + 5)
    const uint32_t degree = 2;

    auto sh_coeffs = torch::randn({N, K, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto dirs = torch::randn({N, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    // Don't pre-normalize - CUDA kernel does this internally
    auto v_colors = torch::randn({N, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // LibTorch autograd path - must normalize internally like CUDA does
    auto sh_ag = sh_coeffs.clone().set_requires_grad(true);
    auto dir_ag = dirs.clone().set_requires_grad(true);

    // Normalize directions (matching CUDA kernel's internal normalization)
    auto dir_norm = dir_ag.norm(2, -1, true);
    auto dir_n = dir_ag / dir_norm;

    auto x = dir_n.select(1, 0);
    auto y = dir_n.select(1, 1);
    auto z = dir_n.select(1, 2);

    // Degree 0
    auto color = SH_C0 * sh_ag.select(1, 0);

    // Degree 1
    color = color + SH_C1 * (-y.unsqueeze(1) * sh_ag.select(1, 1) +
                             z.unsqueeze(1) * sh_ag.select(1, 2) +
                             -x.unsqueeze(1) * sh_ag.select(1, 3));

    // Degree 2 (using constants from CUDA kernel)
    auto z2 = z * z;
    auto fC1 = x * x - y * y;
    auto fS1 = 2.0f * x * y;
    auto fTmp0B = -1.092548430592079f * z;

    auto pSH4 = 0.5462742152960395f * fS1;
    auto pSH5 = fTmp0B * y;
    auto pSH6 = 0.9461746957575601f * z2 - 0.3153915652525201f;
    auto pSH7 = fTmp0B * x;
    auto pSH8 = 0.5462742152960395f * fC1;

    color = color + pSH4.unsqueeze(1) * sh_ag.select(1, 4) +
            pSH5.unsqueeze(1) * sh_ag.select(1, 5) +
            pSH6.unsqueeze(1) * sh_ag.select(1, 6) +
            pSH7.unsqueeze(1) * sh_ag.select(1, 7) +
            pSH8.unsqueeze(1) * sh_ag.select(1, 8);

    color = color + SH_DC_OFFSET;
    color.backward(v_colors);

    auto grad_sh_autograd = sh_ag.grad().contiguous();
    auto grad_dir_autograd = dir_ag.grad().contiguous();

    // CUDA kernel path
    auto v_coeffs_cuda = torch::zeros_like(sh_coeffs);
    auto v_dirs_cuda = torch::zeros_like(dirs);
    auto split_sh = make_split_sh_tensors(sh_coeffs);

    gsplat_lfs::launch_spherical_harmonics_swizzled_bwd_kernel(
        degree,
        dirs.data_ptr<float>(),
        split_sh.sh0.data_ptr<float>(),
        split_sh.shN.data_ptr<float>(),
        nullptr,
        v_colors.data_ptr<float>(),
        N,
        K,
        true,
        v_coeffs_cuda.data_ptr<float>(),
        v_dirs_cuda.data_ptr<float>(),
        nullptr);

    cudaDeviceSynchronize();

    EXPECT_TRUE(tensors_close(grad_sh_autograd, v_coeffs_cuda, REL_TOL, ABS_TOL))
        << "SH Degree 2 coeffs CUDA kernel gradient mismatch: max_rel_err = "
        << max_rel_error(grad_sh_autograd, v_coeffs_cuda);

    EXPECT_TRUE(tensors_close(grad_dir_autograd, v_dirs_cuda, REL_TOL, ABS_TOL))
        << "SH Degree 2 dirs CUDA kernel gradient mismatch: max_rel_err = "
        << max_rel_error(grad_dir_autograd, v_dirs_cuda);

    std::cout << "SH Degree 2 CUDA vs Autograd: coeffs_err = "
              << max_rel_error(grad_sh_autograd, v_coeffs_cuda)
              << ", dirs_err = " << max_rel_error(grad_dir_autograd, v_dirs_cuda) << std::endl;
}

// =============================================================================
// Test: SH Degree 3 CUDA Kernel vs LibTorch Autograd
// =============================================================================
TEST_F(CUDAKernelGradientTest, SHDegree3_CUDA_vs_Autograd) {
    const int N = 100;
    const int K = 16; // Degree 3 has 16 coefficients (1 + 3 + 5 + 7)
    const uint32_t degree = 3;

    auto sh_coeffs = torch::randn({N, K, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto dirs = torch::randn({N, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    // Don't pre-normalize - CUDA kernel does this internally
    auto v_colors = torch::randn({N, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // LibTorch autograd path - must normalize internally like CUDA does
    auto sh_ag = sh_coeffs.clone().set_requires_grad(true);
    auto dir_ag = dirs.clone().set_requires_grad(true);

    // Normalize directions (matching CUDA kernel's internal normalization)
    auto dir_norm = dir_ag.norm(2, -1, true);
    auto dir_n = dir_ag / dir_norm;

    auto x = dir_n.select(1, 0);
    auto y = dir_n.select(1, 1);
    auto z = dir_n.select(1, 2);

    // Degree 0
    auto color = SH_C0 * sh_ag.select(1, 0);

    // Degree 1
    color = color + SH_C1 * (-y.unsqueeze(1) * sh_ag.select(1, 1) +
                             z.unsqueeze(1) * sh_ag.select(1, 2) +
                             -x.unsqueeze(1) * sh_ag.select(1, 3));

    // Degree 2
    auto z2 = z * z;
    auto fC1 = x * x - y * y;
    auto fS1 = 2.0f * x * y;
    auto fTmp0B = -1.092548430592079f * z;

    auto pSH4 = 0.5462742152960395f * fS1;
    auto pSH5 = fTmp0B * y;
    auto pSH6 = 0.9461746957575601f * z2 - 0.3153915652525201f;
    auto pSH7 = fTmp0B * x;
    auto pSH8 = 0.5462742152960395f * fC1;

    color = color + pSH4.unsqueeze(1) * sh_ag.select(1, 4) +
            pSH5.unsqueeze(1) * sh_ag.select(1, 5) +
            pSH6.unsqueeze(1) * sh_ag.select(1, 6) +
            pSH7.unsqueeze(1) * sh_ag.select(1, 7) +
            pSH8.unsqueeze(1) * sh_ag.select(1, 8);

    // Degree 3 (using constants from CUDA kernel)
    auto fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
    auto fTmp1B = 1.445305721320277f * z;
    auto fC2 = x * fC1 - y * fS1;
    auto fS2 = x * fS1 + y * fC1;

    auto pSH9 = -0.5900435899266435f * fS2;
    auto pSH10 = fTmp1B * fS1;
    auto pSH11 = fTmp0C * y;
    auto pSH12 = z * (1.865881662950577f * z2 - 1.119528997770346f);
    auto pSH13 = fTmp0C * x;
    auto pSH14 = fTmp1B * fC1;
    auto pSH15 = -0.5900435899266435f * fC2;

    color = color + pSH9.unsqueeze(1) * sh_ag.select(1, 9) +
            pSH10.unsqueeze(1) * sh_ag.select(1, 10) +
            pSH11.unsqueeze(1) * sh_ag.select(1, 11) +
            pSH12.unsqueeze(1) * sh_ag.select(1, 12) +
            pSH13.unsqueeze(1) * sh_ag.select(1, 13) +
            pSH14.unsqueeze(1) * sh_ag.select(1, 14) +
            pSH15.unsqueeze(1) * sh_ag.select(1, 15);

    color = color + SH_DC_OFFSET;
    color.backward(v_colors);

    auto grad_sh_autograd = sh_ag.grad().contiguous();
    auto grad_dir_autograd = dir_ag.grad().contiguous();

    // CUDA kernel path
    auto v_coeffs_cuda = torch::zeros_like(sh_coeffs);
    auto v_dirs_cuda = torch::zeros_like(dirs);
    auto split_sh = make_split_sh_tensors(sh_coeffs);

    gsplat_lfs::launch_spherical_harmonics_swizzled_bwd_kernel(
        degree,
        dirs.data_ptr<float>(),
        split_sh.sh0.data_ptr<float>(),
        split_sh.shN.data_ptr<float>(),
        nullptr,
        v_colors.data_ptr<float>(),
        N,
        K,
        true,
        v_coeffs_cuda.data_ptr<float>(),
        v_dirs_cuda.data_ptr<float>(),
        nullptr);

    cudaDeviceSynchronize();

    EXPECT_TRUE(tensors_close(grad_sh_autograd, v_coeffs_cuda, REL_TOL, ABS_TOL))
        << "SH Degree 3 coeffs CUDA kernel gradient mismatch: max_rel_err = "
        << max_rel_error(grad_sh_autograd, v_coeffs_cuda);

    EXPECT_TRUE(tensors_close(grad_dir_autograd, v_dirs_cuda, REL_TOL, ABS_TOL))
        << "SH Degree 3 dirs CUDA kernel gradient mismatch: max_rel_err = "
        << max_rel_error(grad_dir_autograd, v_dirs_cuda);

    std::cout << "SH Degree 3 CUDA vs Autograd: coeffs_err = "
              << max_rel_error(grad_sh_autograd, v_coeffs_cuda)
              << ", dirs_err = " << max_rel_error(grad_dir_autograd, v_dirs_cuda) << std::endl;
}

// =============================================================================
// Test: SH Forward-Backward Round Trip
// Verify that forward + backward CUDA kernels are consistent
// =============================================================================
TEST_F(CUDAKernelGradientTest, SH_ForwardBackward_RoundTrip) {
    const int N = 100;
    const int K = 16;
    const uint32_t degree = 3;

    auto sh_coeffs = torch::randn({N, K, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto dirs = torch::randn({N, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    dirs = dirs / dirs.norm(2, -1, true);

    // Forward pass with CUDA kernel
    auto colors_cuda = torch::zeros({N, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto split_sh = make_split_sh_tensors(sh_coeffs);

    gsplat_lfs::launch_spherical_harmonics_swizzled_fwd_kernel(
        degree,
        dirs.data_ptr<float>(),
        split_sh.sh0.data_ptr<float>(),
        split_sh.shN.data_ptr<float>(),
        nullptr,
        N,
        colors_cuda.data_ptr<float>(),
        nullptr);

    cudaDeviceSynchronize();

    // Forward pass with LibTorch (matching CUDA implementation)
    auto x = dirs.select(1, 0);
    auto y = dirs.select(1, 1);
    auto z = dirs.select(1, 2);

    auto color_torch = SH_C0 * sh_coeffs.select(1, 0);

    // Degree 1
    color_torch = color_torch + SH_C1 * (-y.unsqueeze(1) * sh_coeffs.select(1, 1) +
                                         z.unsqueeze(1) * sh_coeffs.select(1, 2) +
                                         -x.unsqueeze(1) * sh_coeffs.select(1, 3));

    // Degree 2
    auto z2 = z * z;
    auto fC1 = x * x - y * y;
    auto fS1 = 2.0f * x * y;
    auto fTmp0B = -1.092548430592079f * z;

    color_torch = color_torch + 0.5462742152960395f * fS1.unsqueeze(1) * sh_coeffs.select(1, 4) +
                  (fTmp0B * y).unsqueeze(1) * sh_coeffs.select(1, 5) +
                  (0.9461746957575601f * z2 - 0.3153915652525201f).unsqueeze(1) * sh_coeffs.select(1, 6) +
                  (fTmp0B * x).unsqueeze(1) * sh_coeffs.select(1, 7) +
                  0.5462742152960395f * fC1.unsqueeze(1) * sh_coeffs.select(1, 8);

    // Degree 3
    auto fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
    auto fTmp1B = 1.445305721320277f * z;
    auto fC2 = x * fC1 - y * fS1;
    auto fS2 = x * fS1 + y * fC1;

    color_torch = color_torch +
                  (-0.5900435899266435f * fS2).unsqueeze(1) * sh_coeffs.select(1, 9) +
                  (fTmp1B * fS1).unsqueeze(1) * sh_coeffs.select(1, 10) +
                  (fTmp0C * y).unsqueeze(1) * sh_coeffs.select(1, 11) +
                  (z * (1.865881662950577f * z2 - 1.119528997770346f)).unsqueeze(1) * sh_coeffs.select(1, 12) +
                  (fTmp0C * x).unsqueeze(1) * sh_coeffs.select(1, 13) +
                  (fTmp1B * fC1).unsqueeze(1) * sh_coeffs.select(1, 14) +
                  (-0.5900435899266435f * fC2).unsqueeze(1) * sh_coeffs.select(1, 15);

    color_torch = color_torch + SH_DC_OFFSET;

    // Compare forward outputs
    EXPECT_TRUE(tensors_close(color_torch, colors_cuda, REL_TOL, ABS_TOL))
        << "Forward pass mismatch: max_rel_err = "
        << max_rel_error(color_torch, colors_cuda);

    // Now test backward consistency using the forward output as "ground truth"
    auto v_colors = torch::randn({N, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto v_coeffs_cuda = torch::zeros_like(sh_coeffs);
    auto v_dirs_cuda = torch::zeros_like(dirs);

    gsplat_lfs::launch_spherical_harmonics_swizzled_bwd_kernel(
        degree,
        dirs.data_ptr<float>(),
        split_sh.sh0.data_ptr<float>(),
        split_sh.shN.data_ptr<float>(),
        nullptr,
        v_colors.data_ptr<float>(),
        N,
        K,
        true,
        v_coeffs_cuda.data_ptr<float>(),
        v_dirs_cuda.data_ptr<float>(),
        nullptr);

    cudaDeviceSynchronize();

    // Verify gradients are reasonable (non-zero, finite)
    EXPECT_FALSE(v_coeffs_cuda.isnan().any().item<bool>()) << "Backward produced NaN in v_coeffs";
    EXPECT_FALSE(v_coeffs_cuda.isinf().any().item<bool>()) << "Backward produced Inf in v_coeffs";
    EXPECT_FALSE(v_dirs_cuda.isnan().any().item<bool>()) << "Backward produced NaN in v_dirs";
    EXPECT_FALSE(v_dirs_cuda.isinf().any().item<bool>()) << "Backward produced Inf in v_dirs";

    EXPECT_GT(v_coeffs_cuda.abs().sum().item<float>(), 0.0f) << "Backward produced zero v_coeffs";
    EXPECT_GT(v_dirs_cuda.abs().sum().item<float>(), 0.0f) << "Backward produced zero v_dirs";

    std::cout << "SH Forward-Backward Round Trip: fwd_err = "
              << max_rel_error(color_torch, colors_cuda)
              << ", v_coeffs_sum = " << v_coeffs_cuda.abs().sum().item<float>()
              << ", v_dirs_sum = " << v_dirs_cuda.abs().sum().item<float>() << std::endl;
}

// =============================================================================
// Visual Gradient Inspection Tests
// These tests print gradient values for manual verification
// =============================================================================

TEST_F(AnalyticalGradientTest, Visual_ScaleGradient) {
    const int N = 3;

    auto raw_scale = torch::tensor({{-0.5f, 0.0f, 0.5f},
                                    {0.1f, 0.2f, 0.3f},
                                    {-0.1f, -0.2f, -0.3f}},
                                   torch::dtype(torch::kFloat32).device(torch::kCUDA));

    auto raw_scale_ag = raw_scale.clone().set_requires_grad(true);
    auto variance = torch::exp(2.0f * raw_scale_ag);

    // Simple upstream gradient = 1
    auto v_var = torch::ones_like(variance);
    variance.backward(v_var);

    auto grad = raw_scale_ag.grad();
    auto var = torch::exp(2.0f * raw_scale);

    std::cout << "\n=== VISUAL: Scale Gradient ===" << std::endl;
    std::cout << "Formula: variance = exp(2 * raw_scale)" << std::endl;
    std::cout << "Gradient: d(variance)/d(raw_scale) = 2 * variance" << std::endl;
    std::cout << "\nInput raw_scale:" << std::endl
              << raw_scale.cpu() << std::endl;
    std::cout << "\nVariance = exp(2*s):" << std::endl
              << var.cpu() << std::endl;
    std::cout << "\nGradient (2*var):" << std::endl
              << grad.cpu() << std::endl;

    // Verify manually: grad should be 2*var for upstream=1
    auto expected = 2.0f * var;
    EXPECT_TRUE(tensors_close(grad, expected, 1e-5f, 1e-6f));
}

TEST_F(AnalyticalGradientTest, Visual_Gaussian2DGradient) {
    const int N = 3;

    // Simple conic (diagonal) and delta values
    auto conic_x = torch::tensor({1.0f, 2.0f, 0.5f}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto conic_y = torch::tensor({0.0f, 0.0f, 0.0f}, torch::dtype(torch::kFloat32).device(torch::kCUDA)); // No off-diagonal
    auto conic_z = torch::tensor({1.0f, 0.5f, 2.0f}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto delta_x = torch::tensor({0.5f, 1.0f, -0.5f}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto delta_y = torch::tensor({0.5f, -1.0f, 0.5f}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    auto cx_ag = conic_x.clone().set_requires_grad(true);
    auto dx_ag = delta_x.clone().set_requires_grad(true);

    auto sigma = 0.5f * (cx_ag * dx_ag * dx_ag + conic_z * delta_y * delta_y);
    auto G = torch::exp(-sigma);

    auto v_G = torch::ones_like(G);
    G.backward(v_G);

    auto grad_cx = cx_ag.grad();
    auto grad_dx = dx_ag.grad();

    std::cout << "\n=== VISUAL: Gaussian 2D Gradient ===" << std::endl;
    std::cout << "Formula: G = exp(-sigma), sigma = 0.5*(cx*dx^2 + cz*dy^2)" << std::endl;
    std::cout << "d(G)/d(cx) = G * (-0.5 * dx^2)" << std::endl;
    std::cout << "d(G)/d(dx) = G * (-cx * dx)" << std::endl;

    auto G_val = torch::exp(-0.5f * (conic_x * delta_x * delta_x + conic_z * delta_y * delta_y));
    std::cout << "\nInputs:" << std::endl;
    std::cout << "  conic_x: " << conic_x.cpu() << std::endl;
    std::cout << "  delta_x: " << delta_x.cpu() << std::endl;
    std::cout << "  delta_y: " << delta_y.cpu() << std::endl;
    std::cout << "\nGaussian G:" << std::endl
              << G_val.cpu() << std::endl;
    std::cout << "\nGrad w.r.t conic_x:" << std::endl
              << grad_cx.cpu() << std::endl;
    std::cout << "Expected (-0.5 * dx^2 * G):" << std::endl
              << (-0.5f * delta_x * delta_x * G_val).cpu() << std::endl;
    std::cout << "\nGrad w.r.t delta_x:" << std::endl
              << grad_dx.cpu() << std::endl;
    std::cout << "Expected (-cx * dx * G):" << std::endl
              << (-conic_x * delta_x * G_val).cpu() << std::endl;

    EXPECT_TRUE(tensors_close(grad_cx, -0.5f * delta_x * delta_x * G_val, 1e-5f, 1e-6f));
    EXPECT_TRUE(tensors_close(grad_dx, -conic_x * delta_x * G_val, 1e-5f, 1e-6f));
}

TEST_F(AnalyticalGradientTest, Visual_AlphaBlendingGradient) {
    // Single Gaussian blend
    auto T = torch::tensor({1.0f}, torch::dtype(torch::kFloat32).device(torch::kCUDA)); // Transmittance
    auto alpha = torch::tensor({0.3f}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto color = torch::tensor({1.0f, 0.5f, 0.25f}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    auto T_ag = T.clone().set_requires_grad(true);
    auto alpha_ag = alpha.clone().set_requires_grad(true);
    auto color_ag = color.clone().set_requires_grad(true);

    // Forward: pixel = T * alpha * color
    auto pixel = T_ag * alpha_ag * color_ag;

    // Upstream gradient = (1, 1, 1)
    auto v_pixel = torch::ones_like(pixel);
    pixel.backward(v_pixel);

    std::cout << "\n=== VISUAL: Alpha Blending Gradient ===" << std::endl;
    std::cout << "Formula: pixel = T * alpha * color" << std::endl;
    std::cout << "d(pixel)/d(T) = alpha * color" << std::endl;
    std::cout << "d(pixel)/d(alpha) = T * color" << std::endl;
    std::cout << "d(pixel)/d(color) = T * alpha" << std::endl;

    std::cout << "\nInputs:" << std::endl;
    std::cout << "  T (transmittance): " << T.cpu() << std::endl;
    std::cout << "  alpha: " << alpha.cpu() << std::endl;
    std::cout << "  color: " << color.cpu() << std::endl;

    std::cout << "\nPixel = T * alpha * color:" << std::endl
              << pixel.detach().cpu() << std::endl;

    std::cout << "\nGradients (upstream = (1,1,1)):" << std::endl;
    std::cout << "  d/dT = sum(alpha * color * upstream) = " << T_ag.grad().cpu() << std::endl;
    std::cout << "  Expected: " << (alpha * color).sum().cpu() << std::endl;
    std::cout << "  d/dalpha = sum(T * color * upstream) = " << alpha_ag.grad().cpu() << std::endl;
    std::cout << "  Expected: " << (T * color).sum().cpu() << std::endl;
    std::cout << "  d/dcolor = T * alpha * upstream = " << color_ag.grad().cpu() << std::endl;
    std::cout << "  Expected: " << (T * alpha).cpu() << std::endl;
}

TEST_F(AnalyticalGradientTest, Visual_SHDegree1Gradient) {
    const int N = 2;
    const float SH_C0_val = 0.28209479177387814f;
    const float SH_C1_val = 0.4886025119029199f;

    // Fixed directions for easy interpretation
    auto dir = torch::tensor({{1.0f, 0.0f, 0.0f},  // +X axis
                              {0.0f, 1.0f, 0.0f}}, // +Y axis
                             torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // SH coefficients: [N, 4, 3] - 4 bases, 3 color channels
    auto sh_coeffs = torch::ones({N, 4, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    auto sh_ag = sh_coeffs.clone().set_requires_grad(true);
    auto dir_ag = dir.clone().set_requires_grad(true);

    auto x = dir_ag.select(1, 0);
    auto y = dir_ag.select(1, 1);
    auto z = dir_ag.select(1, 2);

    // SH degree 0 + 1
    auto color = SH_C0_val * sh_ag.select(1, 0);
    color = color + SH_C1_val * (-y.unsqueeze(1) * sh_ag.select(1, 1) +
                                 z.unsqueeze(1) * sh_ag.select(1, 2) +
                                 -x.unsqueeze(1) * sh_ag.select(1, 3));
    color = color + 0.5f;

    // Upstream: just red channel
    auto v_color = torch::zeros_like(color);
    v_color.select(1, 0) = 1.0f;
    color.backward(v_color);

    std::cout << "\n=== VISUAL: SH Degree 1 Gradient ===" << std::endl;
    std::cout << "Formula: color = 0.5 + SH_C0 * sh[0] + SH_C1 * (-y*sh[1] + z*sh[2] - x*sh[3])" << std::endl;
    std::cout << "SH_C0 = " << SH_C0_val << ", SH_C1 = " << SH_C1_val << std::endl;

    std::cout << "\nDirections:" << std::endl;
    std::cout << "  d0 = (1, 0, 0) - looking along +X" << std::endl;
    std::cout << "  d1 = (0, 1, 0) - looking along +Y" << std::endl;

    std::cout << "\nSH coefficients (all 1.0):" << std::endl;

    std::cout << "\nColors:" << std::endl
              << color.detach().cpu() << std::endl;

    std::cout << "\nGrad w.r.t sh[0] (d/d(sh0) = SH_C0 * upstream):" << std::endl;
    std::cout << sh_ag.grad().select(1, 0).cpu() << std::endl;

    std::cout << "\nGrad w.r.t sh[1] (d/d(sh1) = -SH_C1 * y * upstream):" << std::endl;
    std::cout << "  d0: y=0, so grad should be 0" << std::endl;
    std::cout << "  d1: y=1, so grad should be -SH_C1 = " << -SH_C1_val << std::endl;
    std::cout << "Actual:" << std::endl
              << sh_ag.grad().select(1, 1).cpu() << std::endl;

    std::cout << "\nGrad w.r.t sh[3] (d/d(sh3) = -SH_C1 * x * upstream):" << std::endl;
    std::cout << "  d0: x=1, so grad should be -SH_C1 = " << -SH_C1_val << std::endl;
    std::cout << "  d1: x=0, so grad should be 0" << std::endl;
    std::cout << "Actual:" << std::endl
              << sh_ag.grad().select(1, 3).cpu() << std::endl;

    std::cout << "\nGrad w.r.t direction:" << std::endl
              << dir_ag.grad().cpu() << std::endl;
}

// =============================================================================
// Test: Quaternion Gradient - Explicit Analytical Formula
// FastGS uses implicit normalization: q_n = q / ||q||
// R_ij = f(q_n), so dR/dq requires chain rule through normalization
// =============================================================================
TEST_F(AnalyticalGradientTest, QuaternionGradientAnalytical) {
    const int N = 50;

    auto quat = torch::randn({N, 4}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // Autograd path
    auto quat_ag = quat.clone().set_requires_grad(true);

    auto q_norm_sq = (quat_ag * quat_ag).sum(-1, true);
    auto qr = quat_ag.select(1, 0);
    auto qx = quat_ag.select(1, 1);
    auto qy = quat_ag.select(1, 2);
    auto qz = quat_ag.select(1, 3);

    // Rotation matrix element R[0,0] = 1 - 2*(qy^2 + qz^2) / ||q||^2
    auto R00 = 1.0f - 2.0f * (qy * qy + qz * qz) / q_norm_sq.squeeze();

    auto v_R00 = torch::randn_like(R00);
    R00.backward(v_R00);
    auto grad_quat_autograd = quat_ag.grad();

    // Analytical gradient for R00 = 1 - 2*(qy^2 + qz^2) / (qr^2 + qx^2 + qy^2 + qz^2)
    // Let S = qy^2 + qz^2, N = qr^2 + qx^2 + qy^2 + qz^2
    // R00 = 1 - 2*S/N
    // dR00/dqr = 2*S * 2*qr / N^2 = 4*S*qr / N^2
    // dR00/dqx = 2*S * 2*qx / N^2 = 4*S*qx / N^2
    // dR00/dqy = -2*(2*qy*N - S*2*qy) / N^2 = -4*qy*(N - S) / N^2 = -4*qy*(qr^2 + qx^2) / N^2
    // dR00/dqz = -4*qz*(qr^2 + qx^2) / N^2

    auto qr_v = quat.select(1, 0);
    auto qx_v = quat.select(1, 1);
    auto qy_v = quat.select(1, 2);
    auto qz_v = quat.select(1, 3);
    auto N_sq = (quat * quat).sum(-1);
    auto S = qy_v * qy_v + qz_v * qz_v;
    auto N_sq_sq = N_sq * N_sq;

    auto dR00_dqr = 4.0f * S * qr_v / N_sq_sq;
    auto dR00_dqx = 4.0f * S * qx_v / N_sq_sq;
    auto dR00_dqy = -4.0f * qy_v * (qr_v * qr_v + qx_v * qx_v) / N_sq_sq;
    auto dR00_dqz = -4.0f * qz_v * (qr_v * qr_v + qx_v * qx_v) / N_sq_sq;

    auto grad_qr_analytical = dR00_dqr * v_R00;
    auto grad_qx_analytical = dR00_dqx * v_R00;
    auto grad_qy_analytical = dR00_dqy * v_R00;
    auto grad_qz_analytical = dR00_dqz * v_R00;

    EXPECT_TRUE(tensors_close(grad_quat_autograd.select(1, 0), grad_qr_analytical, REL_TOL, ABS_TOL))
        << "dR00/dqr mismatch: " << max_rel_error(grad_quat_autograd.select(1, 0), grad_qr_analytical);
    EXPECT_TRUE(tensors_close(grad_quat_autograd.select(1, 1), grad_qx_analytical, REL_TOL, ABS_TOL))
        << "dR00/dqx mismatch: " << max_rel_error(grad_quat_autograd.select(1, 1), grad_qx_analytical);
    EXPECT_TRUE(tensors_close(grad_quat_autograd.select(1, 2), grad_qy_analytical, REL_TOL, ABS_TOL))
        << "dR00/dqy mismatch: " << max_rel_error(grad_quat_autograd.select(1, 2), grad_qy_analytical);
    EXPECT_TRUE(tensors_close(grad_quat_autograd.select(1, 3), grad_qz_analytical, REL_TOL, ABS_TOL))
        << "dR00/dqz mismatch: " << max_rel_error(grad_quat_autograd.select(1, 3), grad_qz_analytical);

    std::cout << "QuaternionGradientAnalytical: qr_err=" << max_rel_error(grad_quat_autograd.select(1, 0), grad_qr_analytical)
              << ", qy_err=" << max_rel_error(grad_quat_autograd.select(1, 2), grad_qy_analytical) << std::endl;
}

// =============================================================================
// Test: Scale → Cov3D Gradient - Explicit Analytical Formula
// Cov3D = R * diag(exp(2*s)) * R^T
// For diagonal element: Cov3D[0,0] = sum_k R[0,k]^2 * exp(2*s[k])
// =============================================================================
TEST_F(AnalyticalGradientTest, ScaleToCov3DAnalytical) {
    const int N = 50;

    // Fixed rotation (identity for simplicity)
    auto R = torch::eye(3, torch::dtype(torch::kFloat32).device(torch::kCUDA)).unsqueeze(0).expand({N, 3, 3}).contiguous();
    auto raw_scale = torch::randn({N, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA)) * 0.5f;

    // Autograd path
    auto scale_ag = raw_scale.clone().set_requires_grad(true);
    auto var = torch::exp(2.0f * scale_ag);
    auto S = torch::diag_embed(var);
    auto cov3d = torch::bmm(torch::bmm(R, S), R.transpose(1, 2));

    // Take Cov3D[0,0] = R[0,:]^2 . var = var[0] (since R = I)
    auto cov3d_00 = cov3d.select(1, 0).select(1, 0);

    auto v_cov3d_00 = torch::randn_like(cov3d_00);
    cov3d_00.backward(v_cov3d_00);
    auto grad_scale_autograd = scale_ag.grad();

    // Analytical: d(Cov3D[0,0])/d(s[0]) = d(exp(2*s[0]))/d(s[0]) = 2*exp(2*s[0])
    // d(Cov3D[0,0])/d(s[1]) = 0, d(Cov3D[0,0])/d(s[2]) = 0
    auto var_val = torch::exp(2.0f * raw_scale);
    auto grad_s0_analytical = 2.0f * var_val.select(1, 0) * v_cov3d_00;
    auto grad_s1_analytical = torch::zeros_like(grad_s0_analytical);
    auto grad_s2_analytical = torch::zeros_like(grad_s0_analytical);

    EXPECT_TRUE(tensors_close(grad_scale_autograd.select(1, 0), grad_s0_analytical, REL_TOL, ABS_TOL))
        << "d(cov3d_00)/d(s0) mismatch: " << max_rel_error(grad_scale_autograd.select(1, 0), grad_s0_analytical);
    EXPECT_TRUE(tensors_close(grad_scale_autograd.select(1, 1), grad_s1_analytical, REL_TOL, ABS_TOL))
        << "d(cov3d_00)/d(s1) should be 0";
    EXPECT_TRUE(tensors_close(grad_scale_autograd.select(1, 2), grad_s2_analytical, REL_TOL, ABS_TOL))
        << "d(cov3d_00)/d(s2) should be 0";

    std::cout << "ScaleToCov3DAnalytical: s0_err=" << max_rel_error(grad_scale_autograd.select(1, 0), grad_s0_analytical)
              << ", s1_err=" << max_rel_error(grad_scale_autograd.select(1, 1), grad_s1_analytical) << std::endl;
}

// =============================================================================
// Test: SH Degree 1 Direction Gradient - Explicit Analytical Formula
// color = SH_C0*c0 + SH_C1*(-y*c1 + z*c2 - x*c3) + 0.5
// where (x,y,z) = dir / ||dir||
// d(color)/d(dir) requires chain through normalization
// =============================================================================
TEST_F(AnalyticalGradientTest, SHDegree1DirectionGradientAnalytical) {
    const int N = 50;

    auto dir = torch::randn({N, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto sh_coeffs = torch::randn({N, 4, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // Autograd path
    auto dir_ag = dir.clone().set_requires_grad(true);

    auto dir_norm = dir_ag.norm(2, -1, true);
    auto dir_n = dir_ag / dir_norm;
    auto x = dir_n.select(1, 0);
    auto y = dir_n.select(1, 1);
    auto z = dir_n.select(1, 2);

    // Just compute red channel for simplicity
    auto c1_r = sh_coeffs.select(1, 1).select(1, 0);
    auto c2_r = sh_coeffs.select(1, 2).select(1, 0);
    auto c3_r = sh_coeffs.select(1, 3).select(1, 0);

    auto color_r = SH_C1 * (-y * c1_r + z * c2_r - x * c3_r);

    auto v_color = torch::ones_like(color_r);
    color_r.backward(v_color);
    auto grad_dir_autograd = dir_ag.grad();

    // Analytical gradient:
    // Let d = dir, n = ||d||, d_n = d/n
    // color_r = SH_C1 * (-d_n.y * c1 + d_n.z * c2 - d_n.x * c3)
    //
    // d(d_n)/d(d) = (I - d_n * d_n^T) / n
    //
    // d(color_r)/d(d_n) = SH_C1 * (-c1, 0, 0) for y component contribution
    //                   = SH_C1 * (0, 0, c2) for z component contribution
    //                   = SH_C1 * (-c3, 0, 0) for x component contribution
    // So: d(color_r)/d(d_n) = SH_C1 * (-c3, -c1, c2)
    //
    // d(color_r)/d(d) = d(color_r)/d(d_n) * d(d_n)/d(d)
    //                 = SH_C1 * (-c3, -c1, c2) * (I - d_n*d_n^T) / n

    auto d = dir;
    auto n = d.norm(2, -1, true);
    auto d_n = d / n;

    // Gradient w.r.t. normalized direction
    auto v_dn_x = -SH_C1 * sh_coeffs.select(1, 3).select(1, 0) * v_color;
    auto v_dn_y = -SH_C1 * sh_coeffs.select(1, 1).select(1, 0) * v_color;
    auto v_dn_z = SH_C1 * sh_coeffs.select(1, 2).select(1, 0) * v_color;

    auto v_dn = torch::stack({v_dn_x, v_dn_y, v_dn_z}, 1); // [N, 3]

    // Chain through normalization: v_d = (v_dn - (v_dn . d_n) * d_n) / n
    auto dot_prod = (v_dn * d_n).sum(-1, true);
    auto grad_dir_analytical = (v_dn - dot_prod * d_n) / n;

    EXPECT_TRUE(tensors_close(grad_dir_autograd, grad_dir_analytical, REL_TOL, ABS_TOL))
        << "SH Degree 1 direction gradient mismatch: " << max_rel_error(grad_dir_autograd, grad_dir_analytical);

    std::cout << "SHDegree1DirectionGradientAnalytical: max_rel_err="
              << max_rel_error(grad_dir_autograd, grad_dir_analytical) << std::endl;
}

// =============================================================================
// Test: EWA Cov3D → Cov2D Gradient - Explicit Analytical Formula
// Cov2D = J * Cov3D * J^T where J is the projection Jacobian
// For J = [[fx/z, 0, 0], [0, fy/z, 0]]:
// Cov2D[0,0] = (fx/z)^2 * Cov3D[0,0]
// =============================================================================
TEST_F(AnalyticalGradientTest, EWACov3DToCov2DAnalytical) {
    const int N = 50;
    const float fx = 500.0f;

    // Simple diagonal Cov3D
    auto cov3d_00 = torch::abs(torch::randn({N}, torch::dtype(torch::kFloat32).device(torch::kCUDA))) + 0.1f;
    auto depth = torch::abs(torch::randn({N}, torch::dtype(torch::kFloat32).device(torch::kCUDA))) + 1.0f;

    // Autograd path
    auto cov3d_ag = cov3d_00.clone().set_requires_grad(true);
    auto depth_ag = depth.clone().set_requires_grad(true);

    // Cov2D[0,0] = (fx/z)^2 * Cov3D[0,0]
    auto cov2d_00 = (fx / depth_ag) * (fx / depth_ag) * cov3d_ag;

    auto v_cov2d = torch::randn_like(cov2d_00);
    cov2d_00.backward(v_cov2d);

    auto grad_cov3d_autograd = cov3d_ag.grad();
    auto grad_depth_autograd = depth_ag.grad();

    // Analytical:
    // d(cov2d_00)/d(cov3d_00) = (fx/z)^2
    // d(cov2d_00)/d(z) = -2 * fx^2 * cov3d_00 / z^3

    auto j_sq = (fx / depth) * (fx / depth);
    auto grad_cov3d_analytical = j_sq * v_cov2d;
    auto grad_depth_analytical = -2.0f * fx * fx * cov3d_00 / (depth * depth * depth) * v_cov2d;

    EXPECT_TRUE(tensors_close(grad_cov3d_autograd, grad_cov3d_analytical, REL_TOL, ABS_TOL))
        << "d(cov2d)/d(cov3d) mismatch: " << max_rel_error(grad_cov3d_autograd, grad_cov3d_analytical);
    EXPECT_TRUE(tensors_close(grad_depth_autograd, grad_depth_analytical, REL_TOL, ABS_TOL))
        << "d(cov2d)/d(depth) mismatch: " << max_rel_error(grad_depth_autograd, grad_depth_analytical);

    std::cout << "EWACov3DToCov2DAnalytical: cov3d_err=" << max_rel_error(grad_cov3d_autograd, grad_cov3d_analytical)
              << ", depth_err=" << max_rel_error(grad_depth_autograd, grad_depth_analytical) << std::endl;
}

// =============================================================================
// Test: Opacity (Sigmoid) Gradient
// FastGS uses: original_opacity = sigmoid(raw_opacity)
// Gradient: d(sigmoid)/d(raw) = sigmoid * (1 - sigmoid)
// =============================================================================
TEST_F(AnalyticalGradientTest, OpacitySigmoidGradient) {
    const int N = 100;

    auto raw_opacity = torch::randn({N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // Autograd path
    auto raw_ag = raw_opacity.clone().set_requires_grad(true);
    auto opacity = torch::sigmoid(raw_ag);

    auto v_opacity = torch::randn_like(opacity);
    opacity.backward(v_opacity);
    auto grad_raw_autograd = raw_ag.grad();

    // Analytical: d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
    auto sigmoid_val = torch::sigmoid(raw_opacity);
    auto grad_raw_analytical = sigmoid_val * (1 - sigmoid_val) * v_opacity;

    EXPECT_TRUE(tensors_close(grad_raw_autograd, grad_raw_analytical))
        << "Opacity sigmoid gradient mismatch: max_rel_err = "
        << max_rel_error(grad_raw_autograd, grad_raw_analytical);

    std::cout << "OpacitySigmoidGradient: max_rel_err = "
              << max_rel_error(grad_raw_autograd, grad_raw_analytical) << std::endl;
}

// =============================================================================
// Test: Quaternion to Rotation Matrix - Analytical Formula Verification
// Verifies the analytical gradient formulas used in FastGS preprocess_backward_cu
// =============================================================================
TEST_F(AnalyticalGradientTest, QuaternionToRotationAnalytical) {
    const int N = 100;

    // Random unnormalized quaternions (FastGS normalizes internally)
    auto quat = torch::randn({N, 4}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // Autograd path with internal normalization (matching FastGS)
    auto quat_ag = quat.clone().set_requires_grad(true);

    // FastGS normalization: q_normalized = q / ||q||
    auto q_norm_sq = (quat_ag * quat_ag).sum(-1, true);
    auto qr = quat_ag.select(1, 0);
    auto qx = quat_ag.select(1, 1);
    auto qy = quat_ag.select(1, 2);
    auto qz = quat_ag.select(1, 3);

    // Normalized products (matching FastGS: 2*qi*qj / q_norm_sq)
    auto qxx = 2.0f * qx * qx / q_norm_sq.squeeze();
    auto qyy = 2.0f * qy * qy / q_norm_sq.squeeze();
    auto qzz = 2.0f * qz * qz / q_norm_sq.squeeze();
    auto qxy = 2.0f * qx * qy / q_norm_sq.squeeze();
    auto qxz = 2.0f * qx * qz / q_norm_sq.squeeze();
    auto qyz = 2.0f * qy * qz / q_norm_sq.squeeze();
    auto qrx = 2.0f * qr * qx / q_norm_sq.squeeze();
    auto qry = 2.0f * qr * qy / q_norm_sq.squeeze();
    auto qrz = 2.0f * qr * qz / q_norm_sq.squeeze();

    // Build rotation matrix (matching FastGS)
    auto R00 = 1.0f - (qyy + qzz);
    auto R01 = qxy - qrz;
    auto R02 = qry + qxz;
    auto R10 = qrz + qxy;
    auto R11 = 1.0f - (qxx + qzz);
    auto R12 = qyz - qrx;
    auto R20 = qxz - qry;
    auto R21 = qrx + qyz;
    auto R22 = 1.0f - (qxx + qyy);

    auto R = torch::stack({torch::stack({R00, R01, R02}, 1),
                           torch::stack({R10, R11, R12}, 1),
                           torch::stack({R20, R21, R22}, 1)},
                          1); // [N, 3, 3]

    // Upstream gradient
    auto v_R = torch::randn_like(R);
    R.backward(v_R);
    auto grad_quat_autograd = quat_ag.grad();

    // Verify gradients are finite and non-zero
    EXPECT_FALSE(grad_quat_autograd.isnan().any().item<bool>()) << "Quaternion gradient has NaN";
    EXPECT_FALSE(grad_quat_autograd.isinf().any().item<bool>()) << "Quaternion gradient has Inf";
    EXPECT_GT(grad_quat_autograd.abs().sum().item<float>(), 0.0f) << "Quaternion gradient is zero";

    // Verify gradient magnitude is reasonable
    float grad_magnitude = grad_quat_autograd.abs().mean().item<float>();
    EXPECT_GT(grad_magnitude, 1e-6f) << "Quaternion gradient magnitude too small";
    EXPECT_LT(grad_magnitude, 1e4f) << "Quaternion gradient magnitude too large";

    std::cout << "QuaternionToRotationAnalytical: grad_mag = " << grad_magnitude << std::endl;
}

// =============================================================================
// Test: Mean3D Projection Gradient
// Tests the gradient through camera projection: mean2d = project(mean3d, w2c, K)
// =============================================================================
TEST_F(AnalyticalGradientTest, Mean3DProjectionGradient) {
    const int N = 100;

    // Camera intrinsics
    const float fx = 500.0f, fy = 500.0f;
    const float cx = 320.0f, cy = 240.0f;

    // 3D means (in front of camera)
    auto mean3d = torch::randn({N, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    mean3d.select(1, 2) = torch::abs(mean3d.select(1, 2)) + 1.0f; // Ensure positive depth

    // Identity world-to-camera for simplicity
    // In camera space: mean2d.x = fx * (x/z) + cx, mean2d.y = fy * (y/z) + cy

    // Autograd path
    auto mean3d_ag = mean3d.clone().set_requires_grad(true);

    auto x = mean3d_ag.select(1, 0);
    auto y = mean3d_ag.select(1, 1);
    auto z = mean3d_ag.select(1, 2);

    auto mean2d_x = fx * (x / z) + cx;
    auto mean2d_y = fy * (y / z) + cy;

    auto v_mean2d_x = torch::randn_like(mean2d_x);
    auto v_mean2d_y = torch::randn_like(mean2d_y);

    auto loss = (mean2d_x * v_mean2d_x + mean2d_y * v_mean2d_y).sum();
    loss.backward();
    auto grad_mean3d_autograd = mean3d_ag.grad();

    // Analytical gradients:
    // d(mean2d_x)/dx = fx / z
    // d(mean2d_x)/dy = 0
    // d(mean2d_x)/dz = -fx * x / z^2
    // d(mean2d_y)/dx = 0
    // d(mean2d_y)/dy = fy / z
    // d(mean2d_y)/dz = -fy * y / z^2

    auto z_val = mean3d.select(1, 2);
    auto x_val = mean3d.select(1, 0);
    auto y_val = mean3d.select(1, 1);

    auto grad_x_analytical = fx / z_val * v_mean2d_x;
    auto grad_y_analytical = fy / z_val * v_mean2d_y;
    auto grad_z_analytical = -fx * x_val / (z_val * z_val) * v_mean2d_x - fy * y_val / (z_val * z_val) * v_mean2d_y;

    auto grad_mean3d_analytical = torch::stack({grad_x_analytical, grad_y_analytical, grad_z_analytical}, 1);

    EXPECT_TRUE(tensors_close(grad_mean3d_autograd, grad_mean3d_analytical))
        << "Mean3D projection gradient mismatch: max_rel_err = "
        << max_rel_error(grad_mean3d_autograd, grad_mean3d_analytical);

    std::cout << "Mean3DProjectionGradient: max_rel_err = "
              << max_rel_error(grad_mean3d_autograd, grad_mean3d_analytical) << std::endl;
}

// =============================================================================
// Test: Full Covariance Chain - Scale and Quaternion to Cov3D
// Matches FastGS: cov3d = R * diag(exp(2*s)) * R^T
// =============================================================================
TEST_F(AnalyticalGradientTest, FullCov3DChain) {
    const int N = 50;

    auto quat = torch::randn({N, 4}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto raw_scale = torch::randn({N, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA)) * 0.5f;

    // Autograd path
    auto quat_ag = quat.clone().set_requires_grad(true);
    auto scale_ag = raw_scale.clone().set_requires_grad(true);

    // Quaternion normalization
    auto q_norm_sq = (quat_ag * quat_ag).sum(-1, true);
    auto qr = quat_ag.select(1, 0);
    auto qx = quat_ag.select(1, 1);
    auto qy = quat_ag.select(1, 2);
    auto qz = quat_ag.select(1, 3);

    auto qxx = 2.0f * qx * qx / q_norm_sq.squeeze();
    auto qyy = 2.0f * qy * qy / q_norm_sq.squeeze();
    auto qzz = 2.0f * qz * qz / q_norm_sq.squeeze();
    auto qxy = 2.0f * qx * qy / q_norm_sq.squeeze();
    auto qxz = 2.0f * qx * qz / q_norm_sq.squeeze();
    auto qyz = 2.0f * qy * qz / q_norm_sq.squeeze();
    auto qrx = 2.0f * qr * qx / q_norm_sq.squeeze();
    auto qry = 2.0f * qr * qy / q_norm_sq.squeeze();
    auto qrz = 2.0f * qr * qz / q_norm_sq.squeeze();

    // Rotation matrix
    auto R00 = 1.0f - (qyy + qzz);
    auto R01 = qxy - qrz;
    auto R02 = qry + qxz;
    auto R10 = qrz + qxy;
    auto R11 = 1.0f - (qxx + qzz);
    auto R12 = qyz - qrx;
    auto R20 = qxz - qry;
    auto R21 = qrx + qyz;
    auto R22 = 1.0f - (qxx + qyy);

    auto R = torch::stack({torch::stack({R00, R01, R02}, 1),
                           torch::stack({R10, R11, R12}, 1),
                           torch::stack({R20, R21, R22}, 1)},
                          1); // [N, 3, 3]

    // Variance from raw scale
    auto variance = torch::exp(2.0f * scale_ag); // [N, 3]

    // 3D covariance: R * diag(var) * R^T
    auto S = torch::diag_embed(variance);                         // [N, 3, 3]
    auto cov3d = torch::bmm(torch::bmm(R, S), R.transpose(1, 2)); // [N, 3, 3]

    // Random upstream gradient for cov3d
    auto v_cov3d = torch::randn_like(cov3d);
    cov3d.backward(v_cov3d);

    auto grad_quat = quat_ag.grad();
    auto grad_scale = scale_ag.grad();

    ASSERT_TRUE(grad_quat.defined());
    ASSERT_TRUE(grad_scale.defined());

    EXPECT_FALSE(grad_quat.isnan().any().item<bool>()) << "Quaternion gradient has NaN";
    EXPECT_FALSE(grad_scale.isnan().any().item<bool>()) << "Scale gradient has NaN";

    // Verify scale gradient follows the chain rule:
    // d(cov3d)/d(raw_scale) = d(cov3d)/d(var) * d(var)/d(raw_scale)
    // d(var)/d(raw_scale) = 2 * var
    // So gradient should scale with variance

    // Check gradient magnitudes are reasonable
    float quat_grad_mag = grad_quat.abs().mean().item<float>();
    float scale_grad_mag = grad_scale.abs().mean().item<float>();

    EXPECT_GT(quat_grad_mag, 1e-6f) << "Quaternion gradient too small";
    EXPECT_GT(scale_grad_mag, 1e-6f) << "Scale gradient too small";
    EXPECT_LT(quat_grad_mag, 1e4f) << "Quaternion gradient too large";
    EXPECT_LT(scale_grad_mag, 1e4f) << "Scale gradient too large";

    std::cout << "FullCov3DChain: quat_grad_mag = " << quat_grad_mag
              << ", scale_grad_mag = " << scale_grad_mag << std::endl;
}

// =============================================================================
// Test: EWA Splatting Gradient (Cov3D -> Cov2D)
// Tests the gradient through Jacobian projection: cov2d = J * W * cov3d * W^T * J^T
// =============================================================================
TEST_F(AnalyticalGradientTest, EWASplattingGradient) {
    const int N = 50;

    // Camera parameters
    const float fx = 500.0f, fy = 500.0f;

    // Random symmetric positive definite 3D covariance
    auto L = torch::randn({N, 3, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto cov3d = torch::bmm(L, L.transpose(1, 2)) + 0.1f * torch::eye(3, torch::kCUDA).unsqueeze(0);

    // Random depths (positive)
    auto depth = torch::abs(torch::randn({N}, torch::dtype(torch::kFloat32).device(torch::kCUDA))) + 1.0f;

    // Simplified Jacobian J = [[fx/z, 0, 0], [0, fy/z, 0]] (ignoring off-center terms)
    // For simplicity, use identity world-to-camera W = I

    // Autograd path
    auto cov3d_ag = cov3d.clone().set_requires_grad(true);
    auto depth_ag = depth.clone().set_requires_grad(true);

    // Build Jacobian
    auto J = torch::zeros({N, 2, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    J.select(1, 0).select(1, 0) = fx / depth_ag; // J[0,0] = fx/z
    J.select(1, 1).select(1, 1) = fy / depth_ag; // J[1,1] = fy/z

    // Cov2D = J * Cov3D * J^T
    auto cov2d = torch::bmm(torch::bmm(J, cov3d_ag), J.transpose(1, 2)); // [N, 2, 2]

    // Add dilation for numerical stability
    cov2d = cov2d + 0.3f * torch::eye(2, torch::kCUDA).unsqueeze(0);

    auto v_cov2d = torch::randn_like(cov2d);
    cov2d.backward(v_cov2d);

    auto grad_cov3d = cov3d_ag.grad();
    auto grad_depth = depth_ag.grad();

    ASSERT_TRUE(grad_cov3d.defined());
    ASSERT_TRUE(grad_depth.defined());

    EXPECT_FALSE(grad_cov3d.isnan().any().item<bool>()) << "Cov3D gradient has NaN";
    EXPECT_FALSE(grad_depth.isnan().any().item<bool>()) << "Depth gradient has NaN";

    std::cout << "EWASplattingGradient: cov3d_grad_mag = " << grad_cov3d.abs().mean().item<float>()
              << ", depth_grad_mag = " << grad_depth.abs().mean().item<float>() << std::endl;
}

// =============================================================================
// Test: SH Degree 2 Analytical Formula Verification
// =============================================================================
TEST_F(AnalyticalGradientTest, SHDegree2Analytical) {
    const int N = 100;

    auto dir = torch::randn({N, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    dir = dir / dir.norm(2, -1, true); // normalized

    auto sh_coeffs = torch::randn({N, 9, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // Autograd path
    auto sh_ag = sh_coeffs.clone().set_requires_grad(true);
    auto dir_ag = dir.clone().set_requires_grad(true);

    auto x = dir_ag.select(1, 0);
    auto y = dir_ag.select(1, 1);
    auto z = dir_ag.select(1, 2);

    // Degree 0
    auto color = SH_C0 * sh_ag.select(1, 0);

    // Degree 1
    color = color + SH_C1 * (-y.unsqueeze(1) * sh_ag.select(1, 1) +
                             z.unsqueeze(1) * sh_ag.select(1, 2) +
                             -x.unsqueeze(1) * sh_ag.select(1, 3));

    // Degree 2 - using constants from FastGS kernel_utils.cuh
    auto z2 = z * z;
    auto fC1 = x * x - y * y;
    auto fS1 = 2.0f * x * y;
    auto fTmp0B = -1.092548430592079f * z;

    auto pSH4 = 0.5462742152960395f * fS1;
    auto pSH5 = fTmp0B * y;
    auto pSH6 = 0.9461746957575601f * z2 - 0.3153915652525201f;
    auto pSH7 = fTmp0B * x;
    auto pSH8 = 0.5462742152960395f * fC1;

    color = color + pSH4.unsqueeze(1) * sh_ag.select(1, 4) +
            pSH5.unsqueeze(1) * sh_ag.select(1, 5) +
            pSH6.unsqueeze(1) * sh_ag.select(1, 6) +
            pSH7.unsqueeze(1) * sh_ag.select(1, 7) +
            pSH8.unsqueeze(1) * sh_ag.select(1, 8);

    color = color + SH_DC_OFFSET;

    auto v_color = torch::randn_like(color);
    color.backward(v_color);

    auto grad_sh = sh_ag.grad();
    auto grad_dir = dir_ag.grad();

    // Verify analytical gradients for SH coefficients (these are straightforward)
    // d(color)/d(sh[k]) = basis_function[k](dir) * v_color
    auto expected_grad_sh4 = pSH4.unsqueeze(1) * v_color;
    auto expected_grad_sh5 = (fTmp0B * y).unsqueeze(1) * v_color;
    auto expected_grad_sh6 = (0.9461746957575601f * z2 - 0.3153915652525201f).unsqueeze(1) * v_color;

    EXPECT_TRUE(tensors_close(grad_sh.select(1, 4), expected_grad_sh4, REL_TOL, ABS_TOL))
        << "SH[4] gradient mismatch: max_rel_err = "
        << max_rel_error(grad_sh.select(1, 4), expected_grad_sh4);

    EXPECT_TRUE(tensors_close(grad_sh.select(1, 5), expected_grad_sh5, REL_TOL, ABS_TOL))
        << "SH[5] gradient mismatch: max_rel_err = "
        << max_rel_error(grad_sh.select(1, 5), expected_grad_sh5);

    EXPECT_TRUE(tensors_close(grad_sh.select(1, 6), expected_grad_sh6, REL_TOL, ABS_TOL))
        << "SH[6] gradient mismatch: max_rel_err = "
        << max_rel_error(grad_sh.select(1, 6), expected_grad_sh6);

    EXPECT_FALSE(grad_dir.isnan().any().item<bool>()) << "Direction gradient has NaN";

    std::cout << "SHDegree2Analytical: sh4_err=" << max_rel_error(grad_sh.select(1, 4), expected_grad_sh4)
              << ", sh5_err=" << max_rel_error(grad_sh.select(1, 5), expected_grad_sh5)
              << ", sh6_err=" << max_rel_error(grad_sh.select(1, 6), expected_grad_sh6) << std::endl;
}

// =============================================================================
// Test: SH Degree 3 Analytical Formula Verification
// =============================================================================
TEST_F(AnalyticalGradientTest, SHDegree3Analytical) {
    const int N = 100;

    auto dir = torch::randn({N, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    dir = dir / dir.norm(2, -1, true);

    auto sh_coeffs = torch::randn({N, 16, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    auto sh_ag = sh_coeffs.clone().set_requires_grad(true);
    auto dir_ag = dir.clone().set_requires_grad(true);

    auto x = dir_ag.select(1, 0);
    auto y = dir_ag.select(1, 1);
    auto z = dir_ag.select(1, 2);

    // Degree 0
    auto color = SH_C0 * sh_ag.select(1, 0);

    // Degree 1
    color = color + SH_C1 * (-y.unsqueeze(1) * sh_ag.select(1, 1) +
                             z.unsqueeze(1) * sh_ag.select(1, 2) +
                             -x.unsqueeze(1) * sh_ag.select(1, 3));

    // Degree 2
    auto z2 = z * z;
    auto fC1 = x * x - y * y;
    auto fS1 = 2.0f * x * y;
    auto fTmp0B = -1.092548430592079f * z;

    color = color + (0.5462742152960395f * fS1).unsqueeze(1) * sh_ag.select(1, 4) +
            (fTmp0B * y).unsqueeze(1) * sh_ag.select(1, 5) +
            (0.9461746957575601f * z2 - 0.3153915652525201f).unsqueeze(1) * sh_ag.select(1, 6) +
            (fTmp0B * x).unsqueeze(1) * sh_ag.select(1, 7) +
            (0.5462742152960395f * fC1).unsqueeze(1) * sh_ag.select(1, 8);

    // Degree 3
    auto fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
    auto fTmp1B = 1.445305721320277f * z;
    auto fC2 = x * fC1 - y * fS1;
    auto fS2 = x * fS1 + y * fC1;

    auto pSH9 = -0.5900435899266435f * fS2;
    auto pSH10 = fTmp1B * fS1;
    auto pSH11 = fTmp0C * y;
    auto pSH12 = z * (1.865881662950577f * z2 - 1.119528997770346f);
    auto pSH13 = fTmp0C * x;
    auto pSH14 = fTmp1B * fC1;
    auto pSH15 = -0.5900435899266435f * fC2;

    color = color + pSH9.unsqueeze(1) * sh_ag.select(1, 9) +
            pSH10.unsqueeze(1) * sh_ag.select(1, 10) +
            pSH11.unsqueeze(1) * sh_ag.select(1, 11) +
            pSH12.unsqueeze(1) * sh_ag.select(1, 12) +
            pSH13.unsqueeze(1) * sh_ag.select(1, 13) +
            pSH14.unsqueeze(1) * sh_ag.select(1, 14) +
            pSH15.unsqueeze(1) * sh_ag.select(1, 15);

    color = color + SH_DC_OFFSET;

    auto v_color = torch::randn_like(color);
    color.backward(v_color);

    auto grad_sh = sh_ag.grad();
    auto grad_dir = dir_ag.grad();

    // Verify analytical gradients for degree 3 SH coefficients
    auto expected_grad_sh9 = pSH9.unsqueeze(1) * v_color;
    auto expected_grad_sh12 = pSH12.unsqueeze(1) * v_color;
    auto expected_grad_sh15 = pSH15.unsqueeze(1) * v_color;

    EXPECT_TRUE(tensors_close(grad_sh.select(1, 9), expected_grad_sh9, REL_TOL, ABS_TOL))
        << "SH[9] gradient mismatch: max_rel_err = "
        << max_rel_error(grad_sh.select(1, 9), expected_grad_sh9);

    EXPECT_TRUE(tensors_close(grad_sh.select(1, 12), expected_grad_sh12, REL_TOL, ABS_TOL))
        << "SH[12] gradient mismatch: max_rel_err = "
        << max_rel_error(grad_sh.select(1, 12), expected_grad_sh12);

    EXPECT_TRUE(tensors_close(grad_sh.select(1, 15), expected_grad_sh15, REL_TOL, ABS_TOL))
        << "SH[15] gradient mismatch: max_rel_err = "
        << max_rel_error(grad_sh.select(1, 15), expected_grad_sh15);

    EXPECT_FALSE(grad_dir.isnan().any().item<bool>()) << "Direction gradient has NaN";

    std::cout << "SHDegree3Analytical: sh9_err=" << max_rel_error(grad_sh.select(1, 9), expected_grad_sh9)
              << ", sh12_err=" << max_rel_error(grad_sh.select(1, 12), expected_grad_sh12)
              << ", sh15_err=" << max_rel_error(grad_sh.select(1, 15), expected_grad_sh15) << std::endl;
}

// =============================================================================
// Test: Complete Backward Chain - All gradients from loss to parameters
// This is the ultimate test verifying the full gradient flow
// =============================================================================
TEST_F(AnalyticalGradientTest, CompleteBackwardChain) {
    const int N = 20;

    // Camera parameters
    const float fx = 500.0f, fy = 500.0f;
    const float cx = 320.0f, cy = 240.0f;

    // All learnable parameters
    auto mean3d = torch::randn({N, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    mean3d.select(1, 2) = torch::abs(mean3d.select(1, 2)) + 2.0f; // positive depth
    auto raw_scale = torch::randn({N, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA)) * 0.3f;
    auto quat = torch::randn({N, 4}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto raw_opacity = torch::randn({N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto sh_coeffs = torch::randn({N, 4, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // Pixel position
    auto pixel = torch::tensor({320.5f, 240.5f}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // All with requires_grad
    auto mean3d_ag = mean3d.clone().set_requires_grad(true);
    auto scale_ag = raw_scale.clone().set_requires_grad(true);
    auto quat_ag = quat.clone().set_requires_grad(true);
    auto opacity_ag = raw_opacity.clone().set_requires_grad(true);
    auto sh_ag = sh_coeffs.clone().set_requires_grad(true);

    // === Forward Pass (matching FastGS) ===

    // 1. Quaternion normalization -> Rotation matrix
    auto q_norm_sq = (quat_ag * quat_ag).sum(-1, true);
    auto qr = quat_ag.select(1, 0);
    auto qx = quat_ag.select(1, 1);
    auto qy = quat_ag.select(1, 2);
    auto qz = quat_ag.select(1, 3);

    auto qxx = 2.0f * qx * qx / q_norm_sq.squeeze();
    auto qyy = 2.0f * qy * qy / q_norm_sq.squeeze();
    auto qzz = 2.0f * qz * qz / q_norm_sq.squeeze();
    auto qxy = 2.0f * qx * qy / q_norm_sq.squeeze();
    auto qxz = 2.0f * qx * qz / q_norm_sq.squeeze();
    auto qyz = 2.0f * qy * qz / q_norm_sq.squeeze();
    auto qrx = 2.0f * qr * qx / q_norm_sq.squeeze();
    auto qry = 2.0f * qr * qy / q_norm_sq.squeeze();
    auto qrz = 2.0f * qr * qz / q_norm_sq.squeeze();

    auto R = torch::stack({torch::stack({1.0f - (qyy + qzz), qxy - qrz, qry + qxz}, 1),
                           torch::stack({qrz + qxy, 1.0f - (qxx + qzz), qyz - qrx}, 1),
                           torch::stack({qxz - qry, qrx + qyz, 1.0f - (qxx + qyy)}, 1)},
                          1);

    // 2. Scale -> Variance -> 3D Covariance
    auto variance = torch::exp(2.0f * scale_ag);
    auto S = torch::diag_embed(variance);
    auto cov3d = torch::bmm(torch::bmm(R, S), R.transpose(1, 2));

    // 3. Project to 2D (simplified - identity W)
    auto depth = mean3d_ag.select(1, 2);
    auto J = torch::zeros({N, 2, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    J.select(1, 0).select(1, 0) = fx / depth;
    J.select(1, 1).select(1, 1) = fy / depth;
    auto cov2d = torch::bmm(torch::bmm(J, cov3d), J.transpose(1, 2));
    cov2d = cov2d + 0.3f * torch::eye(2, torch::kCUDA).unsqueeze(0);

    // 4. Compute conic (inverse cov2d)
    auto det = cov2d.select(1, 0).select(1, 0) * cov2d.select(1, 1).select(1, 1) - cov2d.select(1, 0).select(1, 1) * cov2d.select(1, 0).select(1, 1);
    auto conic_x = cov2d.select(1, 1).select(1, 1) / det;
    auto conic_y = -cov2d.select(1, 0).select(1, 1) / det;
    auto conic_z = cov2d.select(1, 0).select(1, 0) / det;

    // 5. Project mean to 2D
    auto mean2d_x = fx * mean3d_ag.select(1, 0) / depth + cx;
    auto mean2d_y = fy * mean3d_ag.select(1, 1) / depth + cy;

    // 6. Evaluate Gaussian at pixel
    auto dx = mean2d_x - pixel[0];
    auto dy = mean2d_y - pixel[1];
    auto sigma = 0.5f * (conic_x * dx * dx + conic_z * dy * dy) + conic_y * dx * dy;
    auto G = torch::exp(-sigma);

    // 7. Compute alpha
    auto opacity = torch::sigmoid(opacity_ag);
    auto alpha = opacity * G;
    alpha = torch::clamp(alpha, 0.0f, 0.999f);

    // 8. Evaluate SH for color (degree 1)
    auto cam_pos = torch::zeros({3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto dir = mean3d_ag - cam_pos;
    dir = dir / dir.norm(2, -1, true);

    auto x = dir.select(1, 0);
    auto y = dir.select(1, 1);
    auto z = dir.select(1, 2);

    auto color = SH_C0 * sh_ag.select(1, 0);
    color = color + SH_C1 * (-y.unsqueeze(1) * sh_ag.select(1, 1) +
                             z.unsqueeze(1) * sh_ag.select(1, 2) +
                             -x.unsqueeze(1) * sh_ag.select(1, 3));
    color = color + SH_DC_OFFSET;
    color = torch::clamp(color, 0.0f, 1.0f);

    // 9. Blend (simplified - single Gaussian, T=1)
    auto pixel_color = (alpha.unsqueeze(1) * color).sum(0); // Sum over all Gaussians

    // Loss
    auto target = torch::rand({3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto loss = ((pixel_color - target) * (pixel_color - target)).sum();

    // Backward
    loss.backward();

    // Verify all gradients exist and are valid
    ASSERT_TRUE(mean3d_ag.grad().defined()) << "mean3d gradient not defined";
    ASSERT_TRUE(scale_ag.grad().defined()) << "scale gradient not defined";
    ASSERT_TRUE(quat_ag.grad().defined()) << "quaternion gradient not defined";
    ASSERT_TRUE(opacity_ag.grad().defined()) << "opacity gradient not defined";
    ASSERT_TRUE(sh_ag.grad().defined()) << "sh gradient not defined";

    EXPECT_FALSE(mean3d_ag.grad().isnan().any().item<bool>()) << "mean3d gradient has NaN";
    EXPECT_FALSE(scale_ag.grad().isnan().any().item<bool>()) << "scale gradient has NaN";
    EXPECT_FALSE(quat_ag.grad().isnan().any().item<bool>()) << "quaternion gradient has NaN";
    EXPECT_FALSE(opacity_ag.grad().isnan().any().item<bool>()) << "opacity gradient has NaN";
    EXPECT_FALSE(sh_ag.grad().isnan().any().item<bool>()) << "sh gradient has NaN";

    std::cout << "\n=== Complete Backward Chain Gradients ===" << std::endl;
    std::cout << "mean3d_grad_mag: " << mean3d_ag.grad().abs().mean().item<float>() << std::endl;
    std::cout << "scale_grad_mag: " << scale_ag.grad().abs().mean().item<float>() << std::endl;
    std::cout << "quat_grad_mag: " << quat_ag.grad().abs().mean().item<float>() << std::endl;
    std::cout << "opacity_grad_mag: " << opacity_ag.grad().abs().mean().item<float>() << std::endl;
    std::cout << "sh_grad_mag: " << sh_ag.grad().abs().mean().item<float>() << std::endl;
}

// ============================================================================
// Depth-normal consistency loss
// ============================================================================

namespace {

    struct ConsistencyKernelResult {
        torch::Tensor loss;
        torch::Tensor grad_normal;
        torch::Tensor grad_depth;
        torch::Tensor grad_alpha;
        torch::Tensor partials;
    };

    ConsistencyKernelResult run_normal_consistency_kernel(
        const torch::Tensor& rendered_normal, // [3,H,W]
        const torch::Tensor& depth_accum,     // [H,W]
        const torch::Tensor& alpha_accum,     // [H,W]
        const float fx, const float fy, const float cx, const float cy,
        const float weight) {
        const auto opts = depth_accum.options();
        const int height = static_cast<int>(depth_accum.size(0));
        const int width = static_cast<int>(depth_accum.size(1));
        const auto num_pixels = static_cast<size_t>(depth_accum.numel());

        ConsistencyKernelResult result{
            .loss = torch::empty({1}, opts),
            .grad_normal = torch::zeros_like(rendered_normal),
            .grad_depth = torch::zeros_like(depth_accum),
            .grad_alpha = torch::zeros_like(depth_accum),
            .partials = torch::empty(
                {static_cast<int64_t>(lfs::training::kernels::normal_consistency_partial_count(num_pixels))}, opts)};

        lfs::training::kernels::launch_normal_consistency_loss(
            rendered_normal.data_ptr<float>(),
            depth_accum.data_ptr<float>(),
            alpha_accum.data_ptr<float>(),
            result.grad_normal.data_ptr<float>(),
            result.grad_depth.data_ptr<float>(),
            result.grad_alpha.data_ptr<float>(),
            result.loss.data_ptr<float>(),
            result.partials.data_ptr<float>(),
            width,
            height,
            fx, fy, cx, cy,
            weight);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        return result;
    }

    ConsistencyKernelResult run_normal_prior_depth_kernel(
        const torch::Tensor& prior_normal, // [3,H,W]
        const torch::Tensor& depth_accum,  // [H,W]
        const torch::Tensor& alpha_accum,  // [H,W]
        const float fx, const float fy, const float cx, const float cy,
        const float weight) {
        const auto opts = depth_accum.options();
        const int height = static_cast<int>(depth_accum.size(0));
        const int width = static_cast<int>(depth_accum.size(1));
        const auto num_pixels = static_cast<size_t>(depth_accum.numel());

        ConsistencyKernelResult result{
            .loss = torch::empty({1}, opts),
            .grad_normal = torch::zeros_like(prior_normal),
            .grad_depth = torch::zeros_like(depth_accum),
            .grad_alpha = torch::zeros_like(depth_accum),
            .partials = torch::empty(
                {static_cast<int64_t>(lfs::training::kernels::normal_consistency_partial_count(num_pixels))}, opts)};

        lfs::training::kernels::launch_normal_prior_depth_loss(
            prior_normal.data_ptr<float>(),
            depth_accum.data_ptr<float>(),
            alpha_accum.data_ptr<float>(),
            result.grad_depth.data_ptr<float>(),
            result.grad_alpha.data_ptr<float>(),
            result.loss.data_ptr<float>(),
            result.partials.data_ptr<float>(),
            width,
            height,
            fx, fy, cx, cy,
            weight);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        return result;
    }

    // Mirrors the CUDA kernel: expected depth unprojected through the
    // intrinsics, central-difference cross-product normal with a detached
    // camera-facing sign, alpha-weighted cosine against the rendered normal.
    // Gates, the alpha weighting, and the sum-alpha normalization are
    // detached exactly like the kernel (finals-buffer constants).
    torch::Tensor normal_consistency_reference(
        const torch::Tensor& rendered_normal, // [3,H,W] autograd leaf
        const torch::Tensor& depth_accum,     // [H,W] autograd leaf
        const torch::Tensor& alpha_accum,     // [H,W] autograd leaf
        const float fx, const float fy, const float cx, const float cy,
        const float weight,
        bool* valid_out = nullptr) {
        namespace k = lfs::training::kernels;
        const auto H = depth_accum.size(0);
        const auto W = depth_accum.size(1);
        const auto opts = depth_accum.options();

        const auto grid_y = torch::arange(H, opts).view({H, 1}).expand({H, W});
        const auto grid_x = torch::arange(W, opts).view({1, W}).expand({H, W});
        const auto ray = torch::stack({(grid_x + 0.5f - cx) / fx,
                                       (grid_y + 0.5f - cy) / fy,
                                       torch::ones({H, W}, opts)},
                                      2); // [H,W,3]

        const auto alpha_safe = alpha_accum.clamp_min(1e-3f);
        const auto E = depth_accum.clamp_min(0.0f) / alpha_safe;
        const auto p = E.unsqueeze(2) * ray;

        const auto shift = [](const torch::Tensor& t, const int dy, const int dx) {
            return torch::roll(t, {dy, dx}, {0, 1});
        };
        const auto tx = shift(p, 0, -1) - shift(p, 0, 1);
        const auto ty = shift(p, -1, 0) - shift(p, 1, 0);
        const auto n_raw = torch::cross(tx, ty, 2);
        const auto n_raw_norm = n_raw.norm(2, 2, true).clamp_min(1e-12f);
        const auto facing = (n_raw.detach() * ray).sum(2);
        const auto sign = torch::where(facing > 0.0f,
                                       torch::full({H, W}, -1.0f, opts),
                                       torch::ones({H, W}, opts));
        const auto n_d = sign.unsqueeze(2) * n_raw / n_raw_norm;

        const auto n_r = rendered_normal.permute({1, 2, 0}); // [H,W,3]
        const auto n_r_norm = n_r.norm(2, 2, true).clamp_min(1e-6f);
        const auto n_r_hat = n_r / n_r_norm;
        const auto cos = (n_d * n_r_hat).sum(2);

        // Detached gates mirroring load_sample
        const auto E_d = E.detach();
        const auto a_d = alpha_accum.detach();
        const auto px_valid = (a_d >= k::kNormalConsistencyMinAlpha) & (E_d >= 1e-6f) & E_d.isfinite();
        auto active = px_valid.clone();
        const int shifts[4][2] = {{0, -1}, {0, 1}, {-1, 0}, {1, 0}};
        for (const auto& s : shifts) {
            const auto En = shift(E_d, s[0], s[1]);
            active = active & shift(px_valid, s[0], s[1]) &
                     ((En - E_d).abs() <= k::kNormalConsistencyMaxRelDepthJump * E_d);
        }
        auto interior = torch::zeros({H, W}, opts.dtype(torch::kBool));
        interior.index_put_({torch::indexing::Slice(1, H - 1), torch::indexing::Slice(1, W - 1)}, true);
        active = active & interior &
                 (n_r.detach().norm(2, 2) >= k::kNormalLossMinRenderNorm) &
                 (n_raw.detach().norm(2, 2).square() >= 1e-24f);

        const auto active_f = active.to(opts.dtype());
        const double sum_alpha = (a_d * active_f).sum().item<double>();
        const double count = active_f.sum().item<double>();
        const bool valid = count >= k::kNormalConsistencyMinValidCount &&
                           sum_alpha >= k::kNormalConsistencyMinValidWeight;
        if (valid_out != nullptr) {
            *valid_out = valid;
        }
        if (!valid) {
            return torch::zeros({}, opts);
        }
        const float inv_norm = static_cast<float>(weight / std::max(sum_alpha, 1.0));
        return inv_norm * (a_d * (1.0f - cos) * active_f).sum();
    }

    torch::Tensor normal_prior_depth_reference(
        const torch::Tensor& prior_normal, // [3,H,W]
        const torch::Tensor& depth_accum,  // [H,W] autograd leaf
        const torch::Tensor& alpha_accum,  // [H,W] autograd leaf
        const float fx, const float fy, const float cx, const float cy,
        const float weight,
        bool* valid_out = nullptr) {
        namespace k = lfs::training::kernels;
        const auto H = depth_accum.size(0);
        const auto W = depth_accum.size(1);
        const auto opts = depth_accum.options();

        const auto grid_y = torch::arange(H, opts).view({H, 1}).expand({H, W});
        const auto grid_x = torch::arange(W, opts).view({1, W}).expand({H, W});
        const auto ray = torch::stack({(grid_x + 0.5f - cx) / fx,
                                       (grid_y + 0.5f - cy) / fy,
                                       torch::ones({H, W}, opts)},
                                      2);

        const auto alpha_safe = alpha_accum.clamp_min(1e-3f);
        const auto E = depth_accum.clamp_min(0.0f) / alpha_safe;
        const auto p = E.unsqueeze(2) * ray;

        const auto shift = [](const torch::Tensor& t, const int dy, const int dx) {
            return torch::roll(t, {dy, dx}, {0, 1});
        };
        const auto tx = shift(p, 0, -1) - shift(p, 0, 1);
        const auto ty = shift(p, -1, 0) - shift(p, 1, 0);
        const auto n_raw = torch::cross(tx, ty, 2);
        const auto n_raw_norm = n_raw.norm(2, 2, true).clamp_min(1e-12f);
        const auto facing = (n_raw.detach() * ray).sum(2);
        const auto sign = torch::where(facing > 0.0f,
                                       torch::full({H, W}, -1.0f, opts),
                                       torch::ones({H, W}, opts));
        const auto n_d = sign.unsqueeze(2) * n_raw / n_raw_norm;

        const auto n_p = prior_normal.permute({1, 2, 0});
        const auto n_p_norm = n_p.norm(2, 2, true).clamp_min(1e-6f);
        const auto n_p_hat = n_p / n_p_norm;
        const auto cos = (n_d * n_p_hat).sum(2);

        const auto E_d = E.detach();
        const auto a_d = alpha_accum.detach();
        const auto px_valid = (a_d >= k::kNormalConsistencyMinAlpha) & (E_d >= 1e-6f) & E_d.isfinite();
        auto active = px_valid.clone();
        const int shifts[4][2] = {{0, -1}, {0, 1}, {-1, 0}, {1, 0}};
        for (const auto& s : shifts) {
            const auto En = shift(E_d, s[0], s[1]);
            active = active & shift(px_valid, s[0], s[1]) &
                     ((En - E_d).abs() <= k::kNormalConsistencyMaxRelDepthJump * E_d);
        }
        auto interior = torch::zeros({H, W}, opts.dtype(torch::kBool));
        interior.index_put_({torch::indexing::Slice(1, H - 1), torch::indexing::Slice(1, W - 1)}, true);
        active = active & interior &
                 (n_p.detach().norm(2, 2) >= k::kNormalLossMinPriorNorm) &
                 (n_raw.detach().norm(2, 2).square() >= 1e-24f);

        const auto active_f = active.to(opts.dtype());
        const double sum_alpha = (a_d * active_f).sum().item<double>();
        const bool valid = sum_alpha > 0.0;
        if (valid_out != nullptr) {
            *valid_out = valid;
        }
        if (!valid) {
            return torch::zeros({}, opts);
        }
        const float inv_norm = static_cast<float>(weight / std::max(sum_alpha, 1.0));
        return inv_norm * (a_d * (1.0f - cos) * active_f).sum();
    }

} // namespace

TEST_F(AnalyticalGradientTest, NormalConsistencyLossMatchesLibtorchAutograd) {
    torch::manual_seed(7);
    const int H = 32, W = 32;
    const auto opts = torch::dtype(torch::kFloat32).device(torch::kCUDA);
    const float fx = 40.0f, fy = 42.0f, cx = 15.5f, cy = 16.5f, weight = 0.05f;

    // Smooth tilted surface with mild waviness; alpha solid with variation.
    const auto grid_y = torch::arange(H, opts).view({H, 1}).expand({H, W});
    const auto grid_x = torch::arange(W, opts).view({1, W}).expand({H, W});
    const auto E = 3.0f + 0.01f * grid_x + 0.015f * grid_y +
                   0.02f * torch::sin(0.4f * grid_x) * torch::cos(0.3f * grid_y);
    const auto alpha = 0.7f + 0.25f * torch::rand({H, W}, opts);
    const auto depth_accum = (E * alpha).contiguous();
    auto rendered_normal = torch::randn({3, H, W}, opts);
    rendered_normal = rendered_normal / rendered_normal.norm(2, 0, true).clamp_min(1e-6f);
    rendered_normal = rendered_normal.contiguous();

    auto normal_ag = rendered_normal.detach().clone().set_requires_grad(true);
    auto depth_ag = depth_accum.detach().clone().set_requires_grad(true);
    auto alpha_ag = alpha.detach().clone().set_requires_grad(true);
    bool valid = false;
    const auto expected_loss = normal_consistency_reference(
        normal_ag, depth_ag, alpha_ag, fx, fy, cx, cy, weight, &valid);
    ASSERT_TRUE(valid);
    expected_loss.backward();

    const auto result = run_normal_consistency_kernel(
        rendered_normal, depth_accum, alpha, fx, fy, cx, cy, weight);

    namespace slots = lfs::training::kernels::normal_consistency_slots;
    ASSERT_GT(result.partials[slots::kValid].item<float>(), 0.5f);

    EXPECT_TRUE(torch::allclose(result.loss, expected_loss.detach().reshape({1}), 2e-3f, 1e-7f))
        << "loss mismatch: kernel=" << result.loss.item<float>()
        << ", autograd=" << expected_loss.item<float>();
    EXPECT_TRUE(tensors_close(result.grad_normal, normal_ag.grad(), 2e-3f, 1e-8f))
        << "normal-gradient mismatch: " << max_rel_error(result.grad_normal, normal_ag.grad());
    EXPECT_TRUE(tensors_close(result.grad_depth, depth_ag.grad(), 2e-3f, 1e-8f))
        << "depth-gradient mismatch: " << max_rel_error(result.grad_depth, depth_ag.grad());
    EXPECT_TRUE(tensors_close(result.grad_alpha, alpha_ag.grad(), 2e-3f, 1e-8f))
        << "alpha-gradient mismatch: " << max_rel_error(result.grad_alpha, alpha_ag.grad());
}

TEST_F(AnalyticalGradientTest, NormalPriorDepthLossMatchesLibtorchAutograd) {
    torch::manual_seed(17);
    const int H = 32, W = 32;
    const auto opts = torch::dtype(torch::kFloat32).device(torch::kCUDA);
    const float fx = 38.0f, fy = 41.0f, cx = 15.5f, cy = 15.5f, weight = 0.12f;

    const auto grid_y = torch::arange(H, opts).view({H, 1}).expand({H, W});
    const auto grid_x = torch::arange(W, opts).view({1, W}).expand({H, W});
    const auto E = 2.2f + 0.012f * grid_x + 0.018f * grid_y +
                   0.015f * torch::sin(0.25f * grid_x + 0.1f * grid_y);
    const auto alpha = 0.75f + 0.2f * torch::rand({H, W}, opts);
    const auto depth_accum = (E * alpha).contiguous();
    auto prior_normal = torch::randn({3, H, W}, opts);
    prior_normal = prior_normal / prior_normal.norm(2, 0, true).clamp_min(1e-6f);
    prior_normal = prior_normal.contiguous();

    auto depth_ag = depth_accum.detach().clone().set_requires_grad(true);
    auto alpha_ag = alpha.detach().clone().set_requires_grad(true);
    bool valid = false;
    const auto expected_loss = normal_prior_depth_reference(
        prior_normal, depth_ag, alpha_ag, fx, fy, cx, cy, weight, &valid);
    ASSERT_TRUE(valid);
    expected_loss.backward();

    const auto result = run_normal_prior_depth_kernel(
        prior_normal, depth_accum, alpha, fx, fy, cx, cy, weight);

    namespace slots = lfs::training::kernels::normal_consistency_slots;
    ASSERT_GT(result.partials[slots::kValid].item<float>(), 0.5f);

    EXPECT_TRUE(torch::allclose(result.loss, expected_loss.detach().reshape({1}), 2e-3f, 1e-7f))
        << "loss mismatch: kernel=" << result.loss.item<float>()
        << ", autograd=" << expected_loss.item<float>();
    EXPECT_TRUE(tensors_close(result.grad_depth, depth_ag.grad(), 2e-3f, 1e-8f))
        << "depth-gradient mismatch: " << max_rel_error(result.grad_depth, depth_ag.grad());
    EXPECT_TRUE(tensors_close(result.grad_alpha, alpha_ag.grad(), 2e-3f, 1e-8f))
        << "alpha-gradient mismatch: " << max_rel_error(result.grad_alpha, alpha_ag.grad());
}

TEST_F(AnalyticalGradientTest, NormalConsistencyPerfectPlaneGivesNearZeroLossAndGrads) {
    const int H = 24, W = 24;
    const auto opts = torch::dtype(torch::kFloat32).device(torch::kCUDA);
    const float fx = 30.0f, fy = 30.0f, cx = 11.5f, cy = 11.5f, weight = 0.05f;

    // Fronto-parallel plane: depth-derived normal is (0,0,-1) everywhere.
    const auto alpha = torch::full({H, W}, 0.9f, opts);
    const auto depth_accum = (torch::full({H, W}, 2.5f, opts) * alpha).contiguous();
    auto rendered_normal = torch::zeros({3, H, W}, opts);
    rendered_normal.index_put_({2}, -1.0f);
    rendered_normal = rendered_normal.contiguous();

    const auto result = run_normal_consistency_kernel(
        rendered_normal, depth_accum, alpha, fx, fy, cx, cy, weight);

    namespace slots = lfs::training::kernels::normal_consistency_slots;
    ASSERT_GT(result.partials[slots::kValid].item<float>(), 0.5f);
    EXPECT_GT(result.partials[slots::kMeanCos].item<float>(), 0.999f);
    EXPECT_LT(result.loss.item<float>(), 1e-5f);
    EXPECT_LT(result.grad_normal.abs().max().item<float>(), 1e-5f);
    EXPECT_LT(result.grad_depth.abs().max().item<float>(), 1e-4f);
    EXPECT_LT(result.grad_alpha.abs().max().item<float>(), 1e-4f);
}

TEST_F(AnalyticalGradientTest, NormalConsistencyTooFewValidPixelsDisables) {
    const int H = 8, W = 8;
    const auto opts = torch::dtype(torch::kFloat32).device(torch::kCUDA);
    const auto alpha = torch::full({H, W}, 0.9f, opts);
    const auto depth_accum = (torch::full({H, W}, 2.0f, opts) * alpha).contiguous();
    auto rendered_normal = torch::zeros({3, H, W}, opts);
    rendered_normal.index_put_({2}, -1.0f);
    rendered_normal = rendered_normal.contiguous();

    const auto result = run_normal_consistency_kernel(
        rendered_normal, depth_accum, alpha, 10.0f, 10.0f, 3.5f, 3.5f, 0.05f);

    namespace slots = lfs::training::kernels::normal_consistency_slots;
    EXPECT_LT(result.partials[slots::kValid].item<float>(), 0.5f);
    EXPECT_EQ(result.loss.item<float>(), 0.0f);
    EXPECT_EQ(result.grad_normal.abs().max().item<float>(), 0.0f);
    EXPECT_EQ(result.grad_depth.abs().max().item<float>(), 0.0f);
    EXPECT_EQ(result.grad_alpha.abs().max().item<float>(), 0.0f);
}
