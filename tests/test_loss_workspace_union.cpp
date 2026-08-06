/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_loss_workspace_union.cpp
 * @brief Phase 6D.1 — mutually-exclusive L1+SSIM workspaces must share one arena
 *        region (capacity = max(variant), not sum).
 */

#include <gtest/gtest.h>

#include "core/tensor.hpp"
#include "lfs/kernels/ssim.cuh"
#include "training/losses/photometric_loss.hpp"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <vector>

using namespace lfs::core;
using namespace lfs::training::kernels;

namespace {

    size_t tensor_bytes(const Tensor& t) {
        if (!t.is_valid() || t.numel() == 0) {
            return 0;
        }
        return t.bytes();
    }

    size_t fused_bytes(const FusedL1SSIMWorkspace& w) {
        return tensor_bytes(w.ssim_map) + tensor_bytes(w.dm_dmu1) + tensor_bytes(w.dm_dsigma1_sq) +
               tensor_bytes(w.dm_dsigma12) + tensor_bytes(w.grad_img) + tensor_bytes(w.reduction_temp) +
               tensor_bytes(w.reduction_result);
    }

    size_t pure_ssim_bytes(const SSIMWorkspace& w) {
        return tensor_bytes(w.ssim_map) + tensor_bytes(w.dm_dmu1) + tensor_bytes(w.dm_dsigma1_sq) +
               tensor_bytes(w.dm_dsigma12) + tensor_bytes(w.dL_dmap) + tensor_bytes(w.dL_dimg1) +
               tensor_bytes(w.reduction_temp) + tensor_bytes(w.reduction_result);
    }

    size_t decoupled_bytes(const DecoupledFusedL1SSIMWorkspace& w) {
        return tensor_bytes(w.ssim_map) + tensor_bytes(w.app_dm_dmu1) + tensor_bytes(w.raw_dm_dmu1) +
               tensor_bytes(w.raw_dm_dsigma1_sq) + tensor_bytes(w.raw_dm_dsigma12) +
               tensor_bytes(w.grad_corrected) + tensor_bytes(w.grad_raw) +
               tensor_bytes(w.reduction_temp) + tensor_bytes(w.reduction_result);
    }

    size_t masked_bytes(const MaskedFusedL1SSIMWorkspace& w) {
        return tensor_bytes(w.ssim_map) + tensor_bytes(w.dm_dmu1) + tensor_bytes(w.dm_dsigma1_sq) +
               tensor_bytes(w.dm_dsigma12) + tensor_bytes(w.grad_img) + tensor_bytes(w.reduction_temp) +
               tensor_bytes(w.masked_loss) + tensor_bytes(w.mask_sum);
    }

    size_t masked_decoupled_bytes(const MaskedDecoupledFusedL1SSIMWorkspace& w) {
        return tensor_bytes(w.ssim_map) + tensor_bytes(w.app_dm_dmu1) + tensor_bytes(w.raw_dm_dmu1) +
               tensor_bytes(w.raw_dm_dsigma1_sq) + tensor_bytes(w.raw_dm_dsigma12) +
               tensor_bytes(w.grad_corrected) + tensor_bytes(w.grad_raw) +
               tensor_bytes(w.reduction_temp) + tensor_bytes(w.masked_loss) + tensor_bytes(w.mask_sum);
    }

} // namespace

class LossWorkspaceUnionTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA device available";
        }
    }
};

// Documents the pre-6D.1 bug: five independent ensure_size calls retain the sum.
// After the arena lands this still shows the independent-stack path is "sum", while
// SequentialModesThroughArenaStayWithinMax asserts the production union path.
TEST_F(LossWorkspaceUnionTest, IndependentWorkspacesRetainSum) {
    const std::vector<size_t> shape = {1, 3, 64, 96};

    FusedL1SSIMWorkspace fused;
    SSIMWorkspace pure_ssim;
    DecoupledFusedL1SSIMWorkspace decoupled;
    MaskedFusedL1SSIMWorkspace masked;
    MaskedDecoupledFusedL1SSIMWorkspace masked_decoupled;

    fused.ensure_size(shape);
    pure_ssim.ensure_size(shape);
    decoupled.ensure_size(shape);
    masked.ensure_size(shape);
    masked_decoupled.ensure_size(shape);

    const size_t fb = fused_bytes(fused);
    const size_t pb = pure_ssim_bytes(pure_ssim);
    const size_t db = decoupled_bytes(decoupled);
    const size_t mb = masked_bytes(masked);
    const size_t mdb = masked_decoupled_bytes(masked_decoupled);

    const size_t total = fb + pb + db + mb + mdb;
    const size_t max_variant = std::max({fb, pb, db, mb, mdb});
    const size_t slack = 64 * 1024; // 64 KiB alignment / overhead budget

    // Independent path intentionally retains the sum (documents the bug class).
    EXPECT_GT(total, max_variant + slack)
        << "independent retention total=" << total << " max=" << max_variant;
    EXPECT_GE(total, 2 * max_variant)
        << "sum should be well above a single variant";
}

// 6D.1 gate: sequential mode activation through the shared arena keeps capacity
// at max(variant) + slack, not the sum of every mode ever touched.
TEST_F(LossWorkspaceUnionTest, SequentialModesThroughArenaStayWithinMax) {
    const std::vector<size_t> shape = {1, 3, 64, 96};

    // Per-variant sizes via independent ensure (oracle for max).
    FusedL1SSIMWorkspace fused_ref;
    SSIMWorkspace pure_ref;
    DecoupledFusedL1SSIMWorkspace decoupled_ref;
    MaskedFusedL1SSIMWorkspace masked_ref;
    MaskedDecoupledFusedL1SSIMWorkspace masked_decoupled_ref;
    fused_ref.ensure_size(shape);
    pure_ref.ensure_size(shape);
    decoupled_ref.ensure_size(shape);
    masked_ref.ensure_size(shape);
    masked_decoupled_ref.ensure_size(shape);
    const size_t max_variant = std::max({fused_bytes(fused_ref), pure_ssim_bytes(pure_ref),
                                         decoupled_bytes(decoupled_ref), masked_bytes(masked_ref),
                                         masked_decoupled_bytes(masked_decoupled_ref)});
    const size_t slack = 256 * 1024; // arena alignment / grow headroom

    LossWorkspaceArena arena;
    arena.ensure_fused(shape);
    arena.ensure_pure_ssim(shape);
    arena.ensure_decoupled(shape);
    arena.ensure_masked_fused(shape);
    arena.ensure_masked_decoupled(shape);

    const size_t total = arena.allocated_bytes();
    EXPECT_LE(total, max_variant + slack)
        << "arena retained " << total << " bytes after touching all modes; "
        << "max single variant is " << max_variant << " (slack=" << slack << ")";
    EXPECT_GE(total, max_variant)
        << "arena must be large enough for the largest variant";
}

// Loss values must match between arena-backed and independently-allocated fused workspaces.
TEST_F(LossWorkspaceUnionTest, ArenaFusedLossMatchesIndependent) {
    const int N = 1, C = 3, H = 48, W = 48;
    auto img1 = Tensor::randn({N, C, H, W}, Device::CUDA);
    auto img2 = Tensor::randn({N, C, H, W}, Device::CUDA);
    const float ssim_weight = 0.2f;
    const std::vector<size_t> shape = {static_cast<size_t>(N), static_cast<size_t>(C),
                                       static_cast<size_t>(H), static_cast<size_t>(W)};

    FusedL1SSIMWorkspace independent;
    auto [loss_a, ctx_a] = fused_l1_ssim_forward(img1, img2, ssim_weight, independent, true);
    auto grad_a = fused_l1_ssim_backward(ctx_a, independent);

    LossWorkspaceArena arena;
    // Touch a different mode first so views are rebuilt on mode switch.
    arena.ensure_decoupled(shape);
    auto& fused_ws = arena.ensure_fused(shape);
    auto [loss_b, ctx_b] = fused_l1_ssim_forward(img1, img2, ssim_weight, fused_ws, true);
    auto grad_b = fused_l1_ssim_backward(ctx_b, fused_ws);

    EXPECT_NEAR(loss_a.item<float>(), loss_b.item<float>(), 1e-5f);

    auto ga = grad_a.cpu().contiguous();
    auto gb = grad_b.cpu().contiguous();
    ASSERT_EQ(ga.numel(), gb.numel());
    const float* pa = ga.ptr<float>();
    const float* pb = gb.ptr<float>();
    double max_abs = 0.0;
    for (size_t i = 0; i < ga.numel(); ++i) {
        max_abs = std::max(max_abs, static_cast<double>(std::abs(pa[i] - pb[i])));
    }
    EXPECT_LT(max_abs, 1e-5) << "max |grad diff| = " << max_abs;
}

// PhotometricLoss (production owner of fused/pure-SSIM) should expose the shared arena.
TEST_F(LossWorkspaceUnionTest, PhotometricLossExposesSharedArena) {
    lfs::training::losses::PhotometricLoss loss;
    const std::vector<size_t> shape = {1, 3, 32, 48};

    // Drive fused via public forward, then request another mode on the same arena.
    auto rendered = Tensor::randn({32, 48, 3}, Device::CUDA);
    auto gt = Tensor::randn({32, 48, 3}, Device::CUDA);
    lfs::training::losses::PhotometricLoss::Params params{.lambda_dssim = 0.2f};
    auto result = loss.forward(rendered, gt, params);
    ASSERT_TRUE(result.has_value()) << result.error();

    auto& arena = loss.arena();
    const size_t after_fused = arena.allocated_bytes();
    ASSERT_GT(after_fused, 0u);

    arena.ensure_decoupled(shape);
    arena.ensure_masked_fused(shape);
    arena.ensure_pure_ssim(shape);

    const size_t after_many = arena.allocated_bytes();
    // Capacity is grow-only and must not climb to sum-of-modes.
    // max(variant) at 32x48 is well under 16 MiB; sum would be larger.
    EXPECT_LE(after_many, after_fused + 8 * 1024 * 1024)
        << "arena grew too much after mode switches: fused=" << after_fused
        << " after_many=" << after_many;
    EXPECT_LE(after_many, 16ull * 1024 * 1024);
}

// Phase 6D.2 — zero_terms deleted; decoupled layout drops one full image buffer,
// and app-branch grads match the (now removed) zeros-buffer path.
TEST_F(LossWorkspaceUnionTest, ZeroTermsDeletedAndDecoupledGradsStable) {
    const int N = 1, C = 3, H = 48, W = 48;
    const std::vector<size_t> shape = {1, 3, 48, 48};
    const float ssim_weight = 0.2f;

    // Alloc drop: decoupled independent workspace must be smaller than the
    // pre-6D.2 layout that included a full-image zero_terms buffer.
    DecoupledFusedL1SSIMWorkspace ws;
    ws.ensure_size(shape);
    const size_t live = decoupled_bytes(ws);
    const size_t image_f32 = static_cast<size_t>(N * C * H * W) * sizeof(float);
    // Pre-6D.2 fields: ssim_map(C1) + 4 dm + zero_terms + 2 grad + reduce
    // ≈ map + 7*image + reduce. Post: map + 6*image + reduce.
    const size_t map_bytes = static_cast<size_t>(N * 1 * H * W) * sizeof(float);
    const size_t reduce = 1024 * sizeof(float) + sizeof(float);
    const size_t pre_6d2 = map_bytes + 7 * image_f32 + reduce;
    const size_t post_6d2 = map_bytes + 6 * image_f32 + reduce;
    EXPECT_LE(live, post_6d2 + 4096);
    EXPECT_LT(live, pre_6d2);
    EXPECT_GE(pre_6d2 - live, image_f32 - 4096)
        << "expected ~1 full image (~" << image_f32 << " B) drop from zero_terms";

    // Grad equivalence: two independent runs with different workspaces must match
    // (HasSigmaPartials=false is deterministic and replaces zeros).
    auto corrected = Tensor::randn({N, C, H, W}, Device::CUDA);
    auto raw = Tensor::randn({N, C, H, W}, Device::CUDA);
    auto gt = Tensor::randn({N, C, H, W}, Device::CUDA);

    DecoupledFusedL1SSIMWorkspace a, b;
    auto [loss_a, ctx_a] = decoupled_fused_l1_ssim_forward(corrected, raw, gt, ssim_weight, a, true);
    auto grads_a = decoupled_fused_l1_ssim_backward(ctx_a, a);
    const float la = loss_a.item<float>();

    auto [loss_b, ctx_b] = decoupled_fused_l1_ssim_forward(corrected, raw, gt, ssim_weight, b, true);
    auto grads_b = decoupled_fused_l1_ssim_backward(ctx_b, b);
    const float lb = loss_b.item<float>();

    EXPECT_NEAR(la, lb, 1e-6f);

    auto ga = grads_a.grad_corrected.cpu().contiguous();
    auto gb = grads_b.grad_corrected.cpu().contiguous();
    auto ra = grads_a.grad_raw.cpu().contiguous();
    auto rb = grads_b.grad_raw.cpu().contiguous();
    double max_c = 0, max_r = 0;
    for (size_t i = 0; i < ga.numel(); ++i) {
        max_c = std::max(max_c, static_cast<double>(std::abs(ga.ptr<float>()[i] - gb.ptr<float>()[i])));
        max_r = std::max(max_r, static_cast<double>(std::abs(ra.ptr<float>()[i] - rb.ptr<float>()[i])));
    }
    EXPECT_LT(max_c, 1e-6);
    EXPECT_LT(max_r, 1e-6);

    // When corrected == raw, decoupled corrected+raw grads should match standard fused.
    FusedL1SSIMWorkspace fused;
    auto [floss, fctx] = fused_l1_ssim_forward(corrected, gt, ssim_weight, fused, true);
    auto fgrad = fused_l1_ssim_backward(fctx, fused);

    DecoupledFusedL1SSIMWorkspace dec;
    auto [dloss, dctx] = decoupled_fused_l1_ssim_forward(corrected, corrected, gt, ssim_weight, dec, true);
    auto dgrads = decoupled_fused_l1_ssim_backward(dctx, dec);

    EXPECT_NEAR(floss.item<float>(), dloss.item<float>(), 1e-4f);
    // Combined appearance path: grad_corrected + grad_raw ≈ fused grad when raw==corrected.
    auto combined = (dgrads.grad_corrected + dgrads.grad_raw).cpu().contiguous();
    auto fcpu = fgrad.cpu().contiguous();
    double max_combo = 0;
    for (size_t i = 0; i < fcpu.numel(); ++i) {
        max_combo = std::max(max_combo,
                             static_cast<double>(std::abs(combined.ptr<float>()[i] - fcpu.ptr<float>()[i])));
    }
    EXPECT_LT(max_combo, 5e-4) << "decoupled(corrected==raw) vs fused max abs " << max_combo;
}
