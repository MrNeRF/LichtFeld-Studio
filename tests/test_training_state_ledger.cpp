/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * Phase 0.2 — bytes-per-splat training-state ledger.
 *
 * Hand-computed sizes from docs/analysis/spirulae-comparison/footprint-compare.md §7
 * and SPEED_VRAM_OPTIMIZATION_PLAN §0b (SH degree 3, capacity = live N):
 *
 *   params:   means 12 + rot 16 + scale 12 + opac 4 + sh0 12 + shN 192 = 248 B/splat
 *   optimizer: means 14 + sh0 14 + shN 104 + scale 14 + rot 16 + opac 10 = 172 B/splat
 *   densify:   densification_info [2,N] fp32 = 8 B/splat
 *   grads:     0 (fused FastGS path; allocate_gradients leaves grad empty)
 *   total:     428 B/splat
 */

#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "lfs/training/vram_ledger.hpp"
#include "training/optimizer/adam_optimizer.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

using namespace lfs::core;
using namespace lfs::training;
using lfs::diagnostics::TrainingStateLedger;
using lfs::diagnostics::VramProfiler;

namespace {

    constexpr size_t kN = 32; // multiple of SH reorder block size (32)
    constexpr int kShDegree = 3;

    // Hand-computed B/splat (footprint-compare §7 / plan §0b).
    constexpr size_t kParamsBps = 248;
    constexpr size_t kOptimBps = 172;
    constexpr size_t kDensifyBps = 8;
    constexpr size_t kGradsBps = 0;
    constexpr size_t kTotalBps = 428;

    SplatData make_sh3_splat(const size_t n) {
        auto means = Tensor::zeros({n, size_t{3}}, Device::CUDA, DataType::Float32);
        auto sh0 = Tensor::zeros({n, size_t{1}, size_t{3}}, Device::CUDA, DataType::Float32);
        // Canonical empty rest — constructor swizzles to degree-3 layout.
        auto shN = Tensor::zeros({n, size_t{15}, size_t{3}}, Device::CUDA, DataType::Float32);
        auto scaling = Tensor::zeros({n, size_t{3}}, Device::CUDA, DataType::Float32);
        auto rotation = Tensor::zeros({n, size_t{4}}, Device::CUDA, DataType::Float32);
        // Unit quaternion w=1
        {
            auto cpu = rotation.cpu();
            auto* r = cpu.ptr<float>();
            for (size_t i = 0; i < n; ++i) {
                r[i * 4] = 1.0f;
            }
            rotation = cpu.to(Device::CUDA);
        }
        auto opacity = Tensor::zeros({n, size_t{1}}, Device::CUDA, DataType::Float32);

        SplatData splat(kShDegree, means, sh0, shN, scaling, rotation, opacity, 1.0f);
        // Densify aux: [2, N] fp32 → 8 B/splat (mcmc densification_info layout).
        splat._densification_info =
            Tensor::zeros({size_t{2}, n}, Device::CUDA, DataType::Float32);
        return splat;
    }

} // namespace

// ---------------------------------------------------------------------------
// FAIL-first target: API + hand-computed ledger for synthetic SH3 splat + Adam
// ---------------------------------------------------------------------------

TEST(TrainingStateLedgerTest, SyntheticSh3MatchesFootprintTable) {
    auto splat = make_sh3_splat(kN);
    ASSERT_EQ(splat.size(), kN);
    ASSERT_EQ(splat.get_max_sh_degree(), kShDegree);

    // Sanity: param bytes alone match 248 * N before optimizer is attached.
    {
        const auto params_only = compute_training_state_ledger(splat, nullptr);
        EXPECT_EQ(params_only.live_splats, kN);
        EXPECT_EQ(params_only.params_bytes, kParamsBps * kN);
        EXPECT_EQ(params_only.densify_aux_bytes, kDensifyBps * kN);
        EXPECT_EQ(params_only.optimizer_bytes, 0u);
        EXPECT_EQ(params_only.gradients_or_helpers_bytes, 0u);
    }

    AdamConfig cfg;
    cfg.lr = 1e-3f;
    AdamOptimizer optimizer(splat, cfg);
    optimizer.allocate_gradients(); // moments only; grads stay empty (fused path)

    const TrainingStateLedger ledger = compute_training_state_ledger(splat, &optimizer);

    EXPECT_EQ(ledger.live_splats, kN);
    EXPECT_EQ(ledger.params_bytes, kParamsBps * kN)
        << "means 12 + rot 16 + scale 12 + opac 4 + sh0 12 + shN 192 = 248";
    EXPECT_EQ(ledger.optimizer_bytes, kOptimBps * kN)
        << "uint8 m+v + per-primitive fp32 scales (footprint-compare §3)";
    EXPECT_EQ(ledger.gradients_or_helpers_bytes, kGradsBps * kN)
        << "fused FastGS path keeps no persistent world grads";
    EXPECT_EQ(ledger.densify_aux_bytes, kDensifyBps * kN)
        << "densification_info [2,N] fp32";
    EXPECT_EQ(ledger.total_bytes, kTotalBps * kN);
    EXPECT_DOUBLE_EQ(ledger.bytes_per_splat, static_cast<double>(kTotalBps));
}

TEST(TrainingStateLedgerTest, PublishesIntoVramProfiler) {
    auto& profiler = VramProfiler::instance();
    profiler.setEnabled(true);

    auto splat = make_sh3_splat(kN);
    AdamOptimizer optimizer(splat, AdamConfig{});
    optimizer.allocate_gradients();

    publish_training_state_ledger(splat, &optimizer);

    const auto stored = profiler.trainingStateLedger();
    EXPECT_EQ(stored.live_splats, kN);
    EXPECT_EQ(stored.total_bytes, kTotalBps * kN);
    EXPECT_DOUBLE_EQ(stored.bytes_per_splat, static_cast<double>(kTotalBps));

    const auto snap = profiler.snapshot();
    EXPECT_EQ(snap.training_state.total_bytes, kTotalBps * kN);

    profiler.setEnabled(false);
}
