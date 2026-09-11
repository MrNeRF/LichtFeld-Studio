/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_loss_workspace_union.cpp
 * @brief Mutually exclusive L1+SSIM workspaces share one arena
 *        region sized exactly to the active variant.
 */

#include <gtest/gtest.h>

#include "core/alloc_counter.hpp"
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

    constexpr size_t align_arena_bytes(const size_t bytes) {
        constexpr size_t alignment = 256;
        return (bytes + alignment - 1) & ~(alignment - 1);
    }

    void expect_exact_active_allocation(const LossWorkspaceArena& arena,
                                        const size_t required) {
        EXPECT_EQ(arena.required_bytes(), required);
        EXPECT_EQ(arena.allocated_bytes(), align_arena_bytes(required));
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

// Five independent workspaces retain the sum of their allocations, while the
// production arena keeps only its active variant.
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

    EXPECT_GT(total, max_variant + slack)
        << "independent retention total=" << total << " max=" << max_variant;
    EXPECT_GE(total, 2 * max_variant)
        << "sum should be well above a single variant";
}

// Variant switches resize in both directions to the active layout exactly.
TEST_F(LossWorkspaceUnionTest, SequentialModesThroughArenaTrackActiveVariantExactly) {
    const std::vector<size_t> shape = {1, 3, 64, 96};

    LossWorkspaceArena arena;
    arena.ensure_fused(shape);
    expect_exact_active_allocation(arena, LossWorkspaceArena::fused_layout_bytes(shape));

    arena.ensure_pure_ssim(shape);
    expect_exact_active_allocation(arena, LossWorkspaceArena::pure_ssim_layout_bytes(shape));

    arena.ensure_decoupled(shape);
    expect_exact_active_allocation(arena, LossWorkspaceArena::decoupled_layout_bytes(shape));

    arena.ensure_masked_fused(shape);
    expect_exact_active_allocation(arena, LossWorkspaceArena::masked_fused_layout_bytes(shape));

    arena.ensure_masked_decoupled(shape);
    expect_exact_active_allocation(
        arena, LossWorkspaceArena::masked_decoupled_layout_bytes(shape));
}

TEST_F(LossWorkspaceUnionTest, MixedResolutionSameVariantIsGrowOnly) {
    const std::vector<size_t> small_shape = {1, 3, 32, 48};
    const std::vector<size_t> large_shape = {1, 3, 96, 128};

    LossWorkspaceArena arena;
    arena.ensure_fused(small_shape);
    expect_exact_active_allocation(
        arena, LossWorkspaceArena::fused_layout_bytes(small_shape));

    arena.ensure_fused(large_shape);
    const size_t large_required = LossWorkspaceArena::fused_layout_bytes(large_shape);
    const size_t high_water = align_arena_bytes(large_required);
    ASSERT_EQ(arena.required_bytes(), large_required);
    ASSERT_EQ(arena.allocated_bytes(), high_water);

    const auto alloc_snapshot = lfs::core::alloc_counter::snapshot();
    for (int i = 0; i < 16; ++i) {
        arena.ensure_fused(small_shape);
        EXPECT_EQ(arena.required_bytes(),
                  LossWorkspaceArena::fused_layout_bytes(small_shape));
        EXPECT_EQ(arena.allocated_bytes(), high_water);

        arena.ensure_fused(large_shape);
        EXPECT_EQ(arena.required_bytes(), large_required);
        EXPECT_EQ(arena.allocated_bytes(), high_water);
    }
    EXPECT_EQ(lfs::core::alloc_counter::delta_since(alloc_snapshot), 0u)
        << "same-variant mixed-resolution rebinding must not commit device memory";

    arena.ensure_fused(small_shape);
    ASSERT_LT(arena.required_bytes(), arena.allocated_bytes());
    arena.shrink_to_required();
    expect_exact_active_allocation(
        arena, LossWorkspaceArena::fused_layout_bytes(small_shape));

    arena.reset();
    EXPECT_EQ(arena.active_kind(), LossWorkspaceArena::Kind::None);
    EXPECT_EQ(arena.required_bytes(), 0u);
    EXPECT_EQ(arena.allocated_bytes(), 0u);
    arena.fused().ensure_size(small_shape);
    expect_exact_active_allocation(
        arena, LossWorkspaceArena::fused_layout_bytes(small_shape));
}

TEST_F(LossWorkspaceUnionTest, ActiveVariantAllocationMatchesRequiredAtTwoShapes) {
    const std::vector<std::vector<size_t>> shapes = {
        {1, 3, 17, 29},
        {2, 3, 64, 96},
    };

    LossWorkspaceArena arena;
    for (const auto& shape : shapes) {
        SCOPED_TRACE(::testing::Message() << "shape=" << shape[0] << "x" << shape[1]
                                          << "x" << shape[2] << "x" << shape[3]);

        arena.ensure_fused(shape);
        expect_exact_active_allocation(arena, LossWorkspaceArena::fused_layout_bytes(shape));

        arena.ensure_pure_ssim(shape);
        expect_exact_active_allocation(arena, LossWorkspaceArena::pure_ssim_layout_bytes(shape));

        arena.ensure_decoupled(shape);
        expect_exact_active_allocation(arena, LossWorkspaceArena::decoupled_layout_bytes(shape));

        arena.ensure_masked_fused(shape);
        expect_exact_active_allocation(arena, LossWorkspaceArena::masked_fused_layout_bytes(shape));

        arena.ensure_masked_decoupled(shape);
        expect_exact_active_allocation(
            arena, LossWorkspaceArena::masked_decoupled_layout_bytes(shape));
    }
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
    ASSERT_EQ(arena.active_kind(), LossWorkspaceArena::Kind::Fused);
    expect_exact_active_allocation(
        arena, LossWorkspaceArena::fused_layout_bytes(arena.active_shape()));

    arena.ensure_decoupled(shape);
    expect_exact_active_allocation(arena, LossWorkspaceArena::decoupled_layout_bytes(shape));
    arena.ensure_masked_fused(shape);
    expect_exact_active_allocation(arena, LossWorkspaceArena::masked_fused_layout_bytes(shape));
    arena.ensure_pure_ssim(shape);
    expect_exact_active_allocation(arena, LossWorkspaceArena::pure_ssim_layout_bytes(shape));
}

// The appearance branch omits sigma partials while preserving gradient results.
TEST_F(LossWorkspaceUnionTest, ZeroTermsDeletedAndDecoupledGradsStable) {
    const int N = 1, C = 3, H = 48, W = 48;
    const std::vector<size_t> shape = {1, 3, 48, 48};
    const float ssim_weight = 0.2f;

    // The reference budget includes a full-image zero_terms buffer; the current
    // layout omits it.
    DecoupledFusedL1SSIMWorkspace ws;
    ws.ensure_size(shape);
    const size_t live = decoupled_bytes(ws);
    const size_t image_f32 = static_cast<size_t>(N * C * H * W) * sizeof(float);
    // Reference fields: ssim_map(C1) + 4 dm + zero_terms + 2 grad + reduce
    // Current: map + four partials + two gradients + reduction.
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

// Pure-SSIM, decoupled, masked, and masked-decoupled paths use fp16 dm_* partials.
// Workspace byte bounds use fp32-partial layouts as reference ceilings.
TEST_F(LossWorkspaceUnionTest, Fp16PartialsWorkspaceBytesAndGradEquiv) {
    const int N = 1, C = 3, H = 48, W = 48;
    const std::vector<size_t> shape = {1, 3, 48, 48};
    const float ssim_weight = 0.2f;
    const size_t image_f32 = static_cast<size_t>(N * C * H * W) * sizeof(float);
    const size_t image_f16 = image_f32 / 2;
    const size_t map_bytes = static_cast<size_t>(N * 1 * H * W) * sizeof(float);

    // Verify partial dtypes and allocation ceilings.
    SSIMWorkspace pure_ws;
    pure_ws.ensure_size(shape);
    EXPECT_EQ(pure_ws.dm_dmu1.dtype(), DataType::Float16);
    EXPECT_EQ(pure_ws.dm_dsigma1_sq.dtype(), DataType::Float16);
    EXPECT_EQ(pure_ws.dm_dsigma12.dtype(), DataType::Float16);
    // Reference: six fp32 images plus reduction. Current: three fp16 partials,
    // three fp32 maps or gradients, and reduction.
    const size_t pure_pre = 6 * image_f32 + 1024 * sizeof(float) + sizeof(float);
    const size_t pure_post = 3 * image_f16 + 3 * image_f32 + 1024 * sizeof(float) + sizeof(float);
    EXPECT_LE(pure_ssim_bytes(pure_ws), pure_post + 4096);
    EXPECT_LT(pure_ssim_bytes(pure_ws), pure_pre);
    EXPECT_GE(pure_pre - pure_ssim_bytes(pure_ws), 3 * image_f16 - 4096);

    DecoupledFusedL1SSIMWorkspace dec_ws;
    dec_ws.ensure_size(shape);
    EXPECT_EQ(dec_ws.app_dm_dmu1.dtype(), DataType::Float16);
    EXPECT_EQ(dec_ws.raw_dm_dmu1.dtype(), DataType::Float16);
    EXPECT_EQ(dec_ws.raw_dm_dsigma1_sq.dtype(), DataType::Float16);
    EXPECT_EQ(dec_ws.raw_dm_dsigma12.dtype(), DataType::Float16);
    // Reference: map, four fp32 partials, two fp32 gradients, and reduction.
    // Current: map + four fp16 partials + two fp32 gradients + reduction.
    const size_t reduce = 1024 * sizeof(float) + sizeof(float);
    const size_t dec_pre = map_bytes + 4 * image_f32 + 2 * image_f32 + reduce;
    const size_t dec_post = map_bytes + 4 * image_f16 + 2 * image_f32 + reduce;
    EXPECT_LE(decoupled_bytes(dec_ws), dec_post + 4096);
    EXPECT_LT(decoupled_bytes(dec_ws), dec_pre);
    EXPECT_GE(dec_pre - decoupled_bytes(dec_ws), 4 * image_f16 - 4096);

    MaskedFusedL1SSIMWorkspace mask_ws;
    mask_ws.ensure_size(shape);
    EXPECT_EQ(mask_ws.dm_dmu1.dtype(), DataType::Float16);
    EXPECT_EQ(mask_ws.dm_dsigma1_sq.dtype(), DataType::Float16);
    EXPECT_EQ(mask_ws.dm_dsigma12.dtype(), DataType::Float16);
    const size_t mask_reduce = 2048 * sizeof(float) + 2 * sizeof(float);
    const size_t mask_pre = map_bytes + 3 * image_f32 + image_f32 + mask_reduce;
    const size_t mask_post = map_bytes + 3 * image_f16 + image_f32 + mask_reduce;
    EXPECT_LE(masked_bytes(mask_ws), mask_post + 4096);
    EXPECT_LT(masked_bytes(mask_ws), mask_pre);

    MaskedDecoupledFusedL1SSIMWorkspace mdec_ws;
    mdec_ws.ensure_size(shape);
    EXPECT_EQ(mdec_ws.app_dm_dmu1.dtype(), DataType::Float16);
    EXPECT_EQ(mdec_ws.raw_dm_dmu1.dtype(), DataType::Float16);
    EXPECT_EQ(mdec_ws.raw_dm_dsigma1_sq.dtype(), DataType::Float16);
    EXPECT_EQ(mdec_ws.raw_dm_dsigma12.dtype(), DataType::Float16);
    const size_t mdec_pre = map_bytes + 4 * image_f32 + 2 * image_f32 + mask_reduce;
    const size_t mdec_post = map_bytes + 4 * image_f16 + 2 * image_f32 + mask_reduce;
    EXPECT_LE(masked_decoupled_bytes(mdec_ws), mdec_post + 4096);
    EXPECT_LT(masked_decoupled_bytes(mdec_ws), mdec_pre);

    // The arena maximum must reflect the compact layouts.
    const size_t arena_max = LossWorkspaceArena::max_variant_bytes(shape);
    // The largest compact variant remains below the six-fp32-image reference.
    EXPECT_LT(arena_max, pure_pre);

    // Gradients must match the fused path within fp16 tolerance.
    auto img1 = Tensor::randn({N, C, H, W}, Device::CUDA);
    auto img2 = Tensor::randn({N, C, H, W}, Device::CUDA);

    FusedL1SSIMWorkspace fused;
    auto [floss, fctx] = fused_l1_ssim_forward(img1, img2, ssim_weight, fused, true);
    auto fgrad = fused_l1_ssim_backward(fctx, fused).cpu().contiguous();

    // Decoupled when corrected==raw: combined grads ≈ fused (both fp16 partials).
    DecoupledFusedL1SSIMWorkspace dec;
    auto [dloss, dctx] = decoupled_fused_l1_ssim_forward(img1, img1, img2, ssim_weight, dec, true);
    auto dgrads = decoupled_fused_l1_ssim_backward(dctx, dec);
    EXPECT_NEAR(floss.item<float>(), dloss.item<float>(), 1e-4f);
    auto dcombo = (dgrads.grad_corrected + dgrads.grad_raw).cpu().contiguous();
    double max_dec = 0;
    for (size_t i = 0; i < fgrad.numel(); ++i) {
        max_dec = std::max(max_dec,
                           static_cast<double>(std::abs(dcombo.ptr<float>()[i] - fgrad.ptr<float>()[i])));
    }
    EXPECT_LT(max_dec, 2e-3) << "decoupled fp16 vs fused max abs " << max_dec;

    // Masked full-ones mask must match unmasked fused within fp16 tol.
    auto ones_mask = Tensor::ones({static_cast<size_t>(H), static_cast<size_t>(W)}, Device::CUDA);
    MaskedFusedL1SSIMWorkspace mws;
    auto [mloss, mctx] = masked_fused_l1_ssim_forward(img1, img2, ones_mask, ssim_weight, mws);
    auto mgrad = masked_fused_l1_ssim_backward(mctx, mws).cpu().contiguous();
    // Masked normalizes by mask_sum (=H*W) vs fused valid-padding mean — use no padding
    // path for a cleaner comparison.
    FusedL1SSIMWorkspace fused_np;
    auto [floss_np, fctx_np] = fused_l1_ssim_forward(img1, img2, ssim_weight, fused_np, false);
    auto fgrad_np = fused_l1_ssim_backward(fctx_np, fused_np).cpu().contiguous();
    EXPECT_NEAR(mloss.item<float>(), floss_np.item<float>(), 1e-4f);
    double max_mask = 0;
    for (size_t i = 0; i < fgrad_np.numel(); ++i) {
        max_mask = std::max(max_mask,
                            static_cast<double>(std::abs(mgrad.ptr<float>()[i] - fgrad_np.ptr<float>()[i])));
    }
    EXPECT_LT(max_mask, 2e-3) << "masked full fp16 vs fused(no pad) max abs " << max_mask;

    // Pure SSIM: deterministic across workspaces; loss finite.
    SSIMWorkspace pure_a, pure_b;
    auto [sloss_a, sctx_a] = ssim_forward(img1, img2, pure_a, true);
    auto sgrad_a = ssim_backward(sctx_a, pure_a, 1.0f);
    auto [sloss_b, sctx_b] = ssim_forward(img1, img2, pure_b, true);
    auto sgrad_b = ssim_backward(sctx_b, pure_b, 1.0f);
    EXPECT_NEAR(sloss_a.item<float>(), sloss_b.item<float>(), 1e-6f);
    auto sa = sgrad_a.cpu().contiguous();
    auto sb = sgrad_b.cpu().contiguous();
    double max_pure = 0;
    for (size_t i = 0; i < sa.numel(); ++i) {
        max_pure = std::max(max_pure,
                            static_cast<double>(std::abs(sa.ptr<float>()[i] - sb.ptr<float>()[i])));
    }
    EXPECT_LT(max_pure, 1e-6);
    EXPECT_TRUE(std::isfinite(sloss_a.item<float>()));
}
