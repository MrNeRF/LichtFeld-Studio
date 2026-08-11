/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * Bytes-per-splat training-state ledger.
 *
 * Hand-computed sizes for SH degree 3 with capacity equal to live N:
 *
 *   params:   means 12 + rot 16 + scale 12 + opac 4 + sh0 12 + shN 192 = 248 B/splat
 *   densify:   densification_info [2,N] fp32 = 8 B/splat
 *   grads:     0 (fused FastGS path; allocate_gradients leaves grad empty)
 *
 * Optimizer (joint (u,log_s) codec — only path):
 *   non-SH: 14 cells × 4 B = 56 + 5 × ceil(N/256) × 16 bounds
 *   SH:     48 cells × 2 B = 96 + 1 × ceil(N/256) × 16 bounds
 *   At N=32: bounds = 6 × 16 = 96 total → optim = (56+96)*32 + 96 = 4960 (155 B/splat)
 *   Large-N limit ≈ 152 B/splat (swizzled SH pad); unpadded K×3=45 cells → ~146.
 */

#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "lfs/training/joint_adam_codec.hpp"
#include "lfs/training/sh_value_codec.hpp"
#include "lfs/training/sh_value_storage.hpp"
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

    // Hand-computed bytes per splat.
    constexpr size_t kParamsBps = 248;
    constexpr size_t kDensifyBps = 8;
    constexpr size_t kGradsBps = 0;

    // Exact joint optim bytes at N=32 (includes bounds tables).
    [[nodiscard]] size_t joint_optim_bytes(const size_t n) {
        using namespace joint_adam;
        const size_t nb = n_bounds_for_prims(n);
        // non-SH groups: means3, sh0 3, scaling3, rotation4, opacity1
        const size_t non_sh_cells = n * (3 + 3 + 3 + 4 + 1);
        const size_t non_sh_packed = packed_bytes(non_sh_cells, 16);
        const size_t non_sh_bounds = nb * 5 * sizeof(float) * 4; // 5 attrs × float4
        // SH: 48 swizzled float cells × 2 B, 1 bounds table
        const size_t sh_cells = n * 48;
        const size_t sh_packed = packed_bytes(sh_cells, 8);
        const size_t sh_bounds = nb * sizeof(float) * 4;
        return non_sh_packed + non_sh_bounds + sh_packed + sh_bounds;
    }

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

TEST(TrainingStateLedgerTest, SyntheticSh3MatchesFootprintTable) {
    // Keep SH values fp32 for this baseline footprint table (248 params).
    sh_value::set_sh_value_quant_enabled_for_testing(false);

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
    const size_t expected_optim = joint_optim_bytes(kN);
    const size_t expected_total = kParamsBps * kN + expected_optim + kDensifyBps * kN;

    EXPECT_EQ(ledger.live_splats, kN);
    EXPECT_EQ(ledger.params_bytes, kParamsBps * kN)
        << "means 12 + rot 16 + scale 12 + opac 4 + sh0 12 + shN 192 = 248";
    EXPECT_EQ(ledger.optimizer_bytes, expected_optim)
        << "joint (u,log_s) 16-bit non-SH + 8-bit SH + float4 bounds/256";
    EXPECT_EQ(ledger.gradients_or_helpers_bytes, kGradsBps * kN)
        << "fused FastGS path keeps no persistent world grads";
    EXPECT_EQ(ledger.densify_aux_bytes, kDensifyBps * kN)
        << "densification_info [2,N] fp32";
    EXPECT_EQ(ledger.total_bytes, expected_total);
    EXPECT_DOUBLE_EQ(ledger.bytes_per_splat,
                     static_cast<double>(expected_total) / static_cast<double>(kN));

    sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
}

TEST(TrainingStateLedgerTest, PublishesIntoVramProfiler) {
    sh_value::set_sh_value_quant_enabled_for_testing(false);
    auto& profiler = VramProfiler::instance();
    profiler.setEnabled(true);

    auto splat = make_sh3_splat(kN);
    AdamOptimizer optimizer(splat, AdamConfig{});
    optimizer.allocate_gradients();

    publish_training_state_ledger(splat, &optimizer);

    const size_t expected_total = kParamsBps * kN + joint_optim_bytes(kN) + kDensifyBps * kN;
    const auto stored = profiler.trainingStateLedger();
    EXPECT_EQ(stored.live_splats, kN);
    EXPECT_EQ(stored.total_bytes, expected_total);
    EXPECT_DOUBLE_EQ(stored.bytes_per_splat,
                     static_cast<double>(expected_total) / static_cast<double>(kN));

    const auto snap = profiler.snapshot();
    EXPECT_EQ(snap.training_state.total_bytes, expected_total);

    profiler.setEnabled(false);
    sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
}

TEST(TrainingStateLedgerTest, ShValueQuantDropsParamsTo146) {
    sh_value::set_sh_value_quant_enabled_for_testing(true);

    auto splat = make_sh3_splat(kN);
    // The unquantized model uses 248 parameter bytes per splat.
    {
        const auto before = compute_training_state_ledger(splat, nullptr);
        EXPECT_EQ(before.params_bytes, kParamsBps * kN);
    }
    ASSERT_TRUE(sh_value::apply_shN_value_quant(splat));
    ASSERT_TRUE(splat.shN_value_quantized());

    AdamOptimizer optimizer(splat, AdamConfig{});
    optimizer.allocate_gradients();
    const auto ledger = compute_training_state_ledger(splat, &optimizer);

    // 56 non-SH + 90 shN + bounds (~0) = 146; joint optim ~152; densify 8 → ~306
    constexpr size_t kParamsQ16 = 146;                     // 56 + 90 pad-dropped
    EXPECT_LE(ledger.params_bytes, (kParamsQ16 + 1) * kN); // allow tiny bounds
    EXPECT_GE(ledger.params_bytes, kParamsQ16 * kN);
    EXPECT_LT(ledger.params_bytes, kParamsBps * kN);
    // Total large-N ≈ 306; at N=32 bounds inflate optim a bit.
    EXPECT_LT(ledger.bytes_per_splat, 320.0);
    EXPECT_GT(ledger.bytes_per_splat, 280.0);

    sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
}
