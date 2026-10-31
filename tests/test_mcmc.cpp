/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "lfs/training/joint_adam_codec.hpp"
#include "lfs/training/sh_value_codec.hpp"
#include "training/strategies/improved_gs_plus.hpp"
#include "training/strategies/mcmc.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>

using namespace lfs::core;
using namespace lfs::training;

namespace {
    // Joint (u,log_s) is the only Adam codec — tests assert joint moment state.
    SplatData create_test_splat_data(const int n_gaussians = 100) {
        std::vector<float> means_data(n_gaussians * 3, 0.0f);
        std::vector<float> sh0_data(n_gaussians * 3, 0.5f);
        std::vector<float> shN_data(n_gaussians * 48, 0.0f);
        std::vector<float> scaling_data(n_gaussians * 3, -2.0f);
        std::vector<float> rotation_data(n_gaussians * 4, 0.0f);
        std::vector<float> opacity_data(n_gaussians, 0.5f);

        for (int i = 0; i < n_gaussians; ++i) {
            rotation_data[i * 4 + 0] = 1.0f;
        }

        auto means = Tensor::from_vector(means_data, TensorShape({static_cast<size_t>(n_gaussians), 3}), Device::CUDA);
        auto sh0 = Tensor::from_vector(sh0_data, TensorShape({static_cast<size_t>(n_gaussians), 3}), Device::CUDA);
        auto shN = Tensor::from_vector(shN_data, TensorShape({static_cast<size_t>(n_gaussians), 48}), Device::CUDA);
        auto scaling = Tensor::from_vector(scaling_data, TensorShape({static_cast<size_t>(n_gaussians), 3}), Device::CUDA);
        auto rotation = Tensor::from_vector(rotation_data, TensorShape({static_cast<size_t>(n_gaussians), 4}), Device::CUDA);
        auto opacity = Tensor::from_vector(opacity_data, TensorShape({static_cast<size_t>(n_gaussians), 1}), Device::CUDA);

        return SplatData(3, means, sh0, shN, scaling, rotation, opacity, 1.0f);
    }

    Tensor make_mask(const int total, const int count_true) {
        auto mask = Tensor::zeros_bool({static_cast<size_t>(total)}, Device::CPU);
        auto* mask_ptr = mask.ptr<unsigned char>();
        for (int i = 0; i < count_true; ++i) {
            mask_ptr[i] = 1;
        }
        return mask.to(Device::CUDA);
    }

} // namespace

TEST(MCMCTest, RemoveGaussiansSoftDeletesRows) {
    auto splat_data = create_test_splat_data(50);
    MCMC strategy(splat_data);

    param::OptimizationParameters opt_params;
    opt_params.iterations = 100;
    opt_params.max_cap = 64;
    strategy.initialize(opt_params);

    auto* means_state = strategy.get_optimizer().get_state_mutable(ParamType::Means);
    ASSERT_NE(means_state, nullptr);
    ASSERT_TRUE(means_state->is_joint());
    // grad is allocated lazily via get_grad(); force allocation before fill/assertions.
    strategy.get_optimizer().get_grad(ParamType::Means);
    ASSERT_TRUE(means_state->grad.is_valid());
    means_state->grad.fill_(3.0f);

    auto mask = make_mask(50, 10);
    strategy.remove_gaussians(mask);

    EXPECT_EQ(strategy.get_model().size(), 50);
    EXPECT_TRUE(strategy.get_model().has_deleted_mask());
    EXPECT_EQ(strategy.get_model().visible_count(), 40);

    const auto deleted = strategy.get_model().deleted().to(DataType::Int32).cpu().to_vector();
    ASSERT_EQ(deleted.size(), 50);
    for (int i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(deleted[i], 1.0f);
    }
    for (int i = 10; i < 50; ++i) {
        EXPECT_FLOAT_EQ(deleted[i], 0.0f);
    }

    const auto rotations = strategy.get_model().rotation_raw().cpu().to_vector();
    ASSERT_EQ(rotations.size(), 50 * 4);
    for (int i = 0; i < 10 * 4; ++i) {
        EXPECT_FLOAT_EQ(rotations[i], 0.0f);
    }

    // Joint moments remain live after soft-delete; grads for deleted rows are zeroed.
    EXPECT_TRUE(means_state->is_joint());
    EXPECT_TRUE(means_state->exp_avg.is_valid());
    EXPECT_TRUE(means_state->joint_bounds.is_valid());
    const auto grad = means_state->grad.cpu().to_vector();
    ASSERT_EQ(grad.size(), 50 * 3);
    for (int i = 0; i < 10 * 3; ++i) {
        EXPECT_FLOAT_EQ(grad[i], 0.0f);
    }
}

TEST(CropDampingStrategyTest, McmcRejectedRowsAreNeverSampledAtZeroScale) {
    auto splat_data = create_test_splat_data(8);
    MCMC strategy(splat_data);

    auto opt_params = param::OptimizationParameters::mcmc_defaults();
    opt_params.iterations = 100;
    opt_params.max_cap = 16;
    strategy.initialize(opt_params);
    strategy._error_score_max = Tensor::ones({8}, Device::CUDA);

    auto crop_mask = make_mask(8, 1);
    strategy.get_optimizer().set_crop_score_mask(crop_mask);
    strategy.get_optimizer().set_cropbox_lr_scale(0.0f);
    const auto damped_weights = strategy.get_sampling_weights();
    const auto damped_cpu = damped_weights.cpu().to_vector();
    ASSERT_EQ(damped_cpu.size(), 8u);
    EXPECT_FLOAT_EQ(damped_cpu[0], 0.0f);
    EXPECT_GT(damped_cpu[1], 0.0f);

    const auto samples = Tensor::multinomial(damped_weights, 256, true).to_vector_int64();
    EXPECT_TRUE(std::none_of(samples.begin(), samples.end(), [](const int64_t index) {
        return index == 0;
    }));

    strategy.get_optimizer().set_cropbox_lr_scale(1.0f);
    const auto unit_scale_weights = strategy.get_sampling_weights().cpu().to_vector();
    strategy.get_optimizer().set_crop_score_mask({});
    const auto unmasked_weights = strategy.get_sampling_weights().cpu().to_vector();
    EXPECT_EQ(unit_scale_weights, unmasked_weights);
}

TEST(CropDampingStrategyTest, IgsPlusRejectedRowsAreNeverSampledAtZeroScale) {
    auto splat_data = create_test_splat_data(8);
    ImprovedGSPlus strategy(splat_data);

    auto opt_params = param::OptimizationParameters::igs_plus_defaults();
    opt_params.iterations = 100;
    opt_params.start_refine = 0;
    opt_params.stop_refine = 80;
    opt_params.refine_every = 10;
    opt_params.max_cap = 16;
    strategy.initialize(opt_params);

    auto crop_mask = make_mask(8, 1);
    strategy.get_optimizer().set_crop_score_mask(crop_mask);
    strategy.get_optimizer().set_cropbox_lr_scale(0.0f);
    const auto scores = Tensor::ones({8}, Device::CUDA);
    const auto damped_scores = strategy.damp_densification_scores(scores);
    const auto damped_cpu = damped_scores.cpu().to_vector();
    ASSERT_EQ(damped_cpu.size(), 8u);
    EXPECT_FLOAT_EQ(damped_cpu[0], 0.0f);
    EXPECT_GT(damped_cpu[1], 0.0f);

    const auto samples = Tensor::multinomial(damped_scores, 256, true).to_vector_int64();
    EXPECT_TRUE(std::none_of(samples.begin(), samples.end(), [](const int64_t index) {
        return index == 0;
    }));

    strategy.get_optimizer().set_cropbox_lr_scale(1.0f);
    const auto unit_scale_scores = strategy.damp_densification_scores(scores).cpu().to_vector();
    strategy.get_optimizer().set_crop_score_mask({});
    const auto unmasked_scores = strategy.damp_densification_scores(scores).cpu().to_vector();
    EXPECT_EQ(unit_scale_scores, unmasked_scores);
}

TEST(ImprovedGSPlusTest, InitializesWhenStopRefinePrecedesStartRefine) {
    auto splat_data = create_test_splat_data(8);
    ImprovedGSPlus strategy(splat_data);

    auto opt_params = param::OptimizationParameters::igs_plus_defaults();
    opt_params.iterations = 400;
    opt_params.start_refine = 500;
    opt_params.stop_refine = 400;
    opt_params.max_cap = 16;
    EXPECT_NO_THROW(strategy.initialize(opt_params));
}

TEST(ImprovedGSPlusTest, RefineWithoutEdgeScoresSkipsDensification) {
    auto splat_data = create_test_splat_data(8);
    ImprovedGSPlus strategy(splat_data);

    auto opt_params = param::OptimizationParameters::igs_plus_defaults();
    opt_params.iterations = 1000;
    opt_params.start_refine = 100;
    opt_params.stop_refine = 800;
    opt_params.refine_every = 100;
    opt_params.max_cap = 16;
    strategy.initialize(opt_params);

    constexpr int kIter = 200;
    ASSERT_TRUE(strategy.is_refining(kIter));
    lfs::training::RenderOutput render_output;
    strategy.pre_step(kIter, render_output);
    EXPECT_NO_THROW(strategy.post_backward(kIter, render_output));
    EXPECT_EQ(splat_data.size(), 8u);
}

// MCMC and IGS+ read only the error row, so they keep densification_info as
// [N]; a two-row tensor from an older checkpoint is folded down on the next step.
TEST(ImprovedGSPlusTest, ErrorRowStrategiesKeepOneDensificationRow) {
    constexpr size_t n = 8;
    const auto check = [n](auto& strategy, SplatData& splat_data, param::OptimizationParameters opt_params) {
        opt_params.iterations = 1000;
        opt_params.start_refine = 500;
        opt_params.stop_refine = 800;
        opt_params.max_cap = 16;
        strategy.initialize(opt_params);
        splat_data._densification_info = Tensor::ones({2, n}, Device::CUDA);
        lfs::training::RenderOutput render_output;
        strategy.pre_step(1, render_output);
        strategy.post_backward(1, render_output);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const auto& info = splat_data._densification_info;
        ASSERT_EQ(info.ndim(), 1u);
        ASSERT_EQ(info.shape()[0], n);
        EXPECT_EQ(info.abs().max().item<float>(), 0.f) << "the fold clears the error row";
    };
    {
        SCOPED_TRACE("MCMC");
        auto splat_data = create_test_splat_data(n);
        MCMC strategy(splat_data);
        check(strategy, splat_data, param::OptimizationParameters::mcmc_defaults());
    }
    {
        SCOPED_TRACE("IGS+");
        auto splat_data = create_test_splat_data(n);
        ImprovedGSPlus strategy(splat_data);
        check(strategy, splat_data, param::OptimizationParameters::igs_plus_defaults());
    }
}

// The split stages children in bounded chunks. A budget past one chunk must
// still give every parent exactly one child carrying the parent's colour.
TEST(ImprovedGSPlusTest, ChunkedSplitGivesEachParentOneChild) {
    constexpr size_t n = 1'200'000; // more SH3 children than one 256 MiB chunk holds
    // sh0 holds the exact parent id; shN holds values within q16 range of it.
    const auto low_tag = [](const size_t i) { return static_cast<float>(i % 1024) / 1024.0f; };
    const auto high_tag = [](const size_t i) { return -static_cast<float>((i / 1024) % 1024) / 1024.0f; };
    std::vector<float> means(n * 3, 0.0f), sh0(n * 3, 0.0f), shN(n * 45, 0.0f);
    std::vector<float> scaling(n * 3), rotation(n * 4, 0.0f), opacity(n, 0.5f);
    for (size_t i = 0; i < n; ++i) {
        sh0[i * 3] = static_cast<float>(i);
        shN[i * 45] = low_tag(i);
        shN[i * 45 + 44] = high_tag(i);
        scaling[i * 3] = -1.0f;
        scaling[i * 3 + 1] = -2.0f;
        scaling[i * 3 + 2] = -3.0f;
        rotation[i * 4] = 1.0f;
    }
    SplatData splat_data(3,
                         Tensor::from_vector(means, TensorShape({n, 3}), Device::CUDA),
                         Tensor::from_vector(sh0, TensorShape({n, 1, 3}), Device::CUDA),
                         Tensor::from_vector(shN, TensorShape({n, 15, 3}), Device::CUDA),
                         Tensor::from_vector(scaling, TensorShape({n, 3}), Device::CUDA),
                         Tensor::from_vector(rotation, TensorShape({n, 4}), Device::CUDA),
                         Tensor::from_vector(opacity, TensorShape({n, 1}), Device::CUDA),
                         1.0f);
    ImprovedGSPlus strategy(splat_data);
    auto opt_params = param::OptimizationParameters::igs_plus_defaults();
    opt_params.iterations = 100;
    opt_params.start_refine = 0;
    opt_params.stop_refine = 80;
    opt_params.refine_every = 10;
    opt_params.max_cap = static_cast<int>(2 * n);
    strategy.initialize(opt_params);

    strategy.densify_for_test(Tensor::ones({n}, Device::CUDA), static_cast<int64_t>(n));

    ASSERT_EQ(splat_data.size(), 2 * n);
    const auto sh0_out = splat_data.sh0().cpu().to_vector();
    const auto shN_out = splat_data.shN_canonical_cpu().to_vector();
    std::vector<std::uint8_t> split(n, 0);
    for (size_t row = n; row < 2 * n; ++row) {
        const auto parent = static_cast<size_t>(sh0_out[row * 3]);
        ASSERT_LT(parent, n);
        ASSERT_EQ(split[parent]++, 0) << "parent " << parent << " split twice";
        ASSERT_NEAR(shN_out[row * 45], low_tag(parent), 1e-4f) << "row " << row;
        ASSERT_NEAR(shN_out[row * 45 + 44], high_tag(parent), 1e-4f) << "row " << row;
    }
}

TEST(ImprovedGSPlusTest, FinalIterationPreservesOpacityAtResetBoundary) {
    for (const int iterations : {3000, 6000, 9000, 12000}) {
        SCOPED_TRACE(iterations);
        auto splat_data = create_test_splat_data(8);
        ImprovedGSPlus strategy(splat_data);
        auto params = param::OptimizationParameters::igs_plus_defaults();
        params.iterations = iterations;
        params.max_cap = 16;
        strategy.initialize(params);
        const auto before = splat_data.opacity_raw().to_vector();

        RenderOutput render_output;
        strategy.pre_step(iterations, render_output);
        strategy.post_backward(iterations, render_output);
        strategy.step(iterations);

        EXPECT_EQ(splat_data.opacity_raw().to_vector(), before);
        EXPECT_EQ(splat_data.size(), 8u);
    }
}

TEST(ImprovedGSPlusTest, OpacityResetSchedulePreservesExistingBoundaries) {
    // Non-final reset, adjacent steps, zero, and the refinement cutoff.
    for (const int iter : {0, 2999, 3000, 3001, 5990, 6000, 15000, 18000}) {
        SCOPED_TRACE(iter);
        auto splat_data = create_test_splat_data(8);
        ImprovedGSPlus strategy(splat_data);
        auto params = param::OptimizationParameters::igs_plus_defaults();
        params.iterations = iter == 5990 ? 5990 : 30000;
        params.max_cap = 16;
        strategy.initialize(params);
        splat_data.set_frozen_ranges({{0, 1}});
        const std::vector<float> opacity{0.5f, -4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
        splat_data.opacity_raw() = Tensor::from_vector(opacity, {8, 1}, Device::CUDA);
        auto expected = opacity;
        if (iter > 0 && iter < params.stop_refine && iter % params.reset_every == 0) {
            const float reset_logit = std::log(0.1f / 0.9f);
            for (size_t row = 1; row < expected.size(); ++row)
                expected[row] = std::min(expected[row], reset_logit);
        }

        RenderOutput render_output;
        strategy.pre_step(iter, render_output);
        strategy.post_backward(iter, render_output);

        EXPECT_EQ(splat_data.opacity_raw().to_vector(), expected);
    }
}

TEST(ImprovedGSPlusTest, OpacityResetUsesUpdatedIterationLimit) {
    auto splat_data = create_test_splat_data(8);
    ImprovedGSPlus strategy(splat_data);
    auto params = param::OptimizationParameters::igs_plus_defaults();
    params.max_cap = 16;
    strategy.initialize(params);
    params.iterations = 6000;
    strategy.set_optimization_params(params);
    const auto before = splat_data.opacity_raw().to_vector();
    RenderOutput render_output;
    strategy.post_backward(6000, render_output);
    EXPECT_EQ(splat_data.opacity_raw().to_vector(), before);

    params.iterations = 9000;
    strategy.set_optimization_params(params);
    strategy.post_backward(6000, render_output);
    const auto after = splat_data.opacity_raw().to_vector();
    for (const float value : after)
        EXPECT_FLOAT_EQ(value, std::log(0.1f / 0.9f));
}

TEST(MCMCTest, RelocateClearsDeletedMaskOnReusedRows) {
    auto splat_data = create_test_splat_data(12);
    MCMC strategy(splat_data);

    param::OptimizationParameters opt_params;
    opt_params.iterations = 100;
    opt_params.max_cap = 24;
    strategy.initialize(opt_params);

    auto* means_state = strategy.get_optimizer().get_state_mutable(ParamType::Means);
    ASSERT_NE(means_state, nullptr);
    ASSERT_TRUE(means_state->is_joint());

    strategy.remove_gaussians(make_mask(12, 3));
    ASSERT_TRUE(strategy.get_model().has_deleted_mask());
    EXPECT_EQ(strategy.get_model().visible_count(), 9);

    strategy.get_model().refresh_deleted_count();
    EXPECT_EQ(strategy.get_model().deleted_count(), 3u);

    const int relocated = strategy.relocate_gs_test();
    EXPECT_EQ(relocated, 3);
    EXPECT_EQ(strategy.get_model().visible_count(), 12);
    strategy.get_model().refresh_deleted_count();
    EXPECT_EQ(strategy.get_model().deleted_count(), 0u);

    const auto deleted = strategy.get_model().deleted().to(DataType::Int32).cpu().to_vector();
    ASSERT_EQ(deleted.size(), 12);
    for (float value : deleted) {
        EXPECT_FLOAT_EQ(value, 0.0f);
    }

    // Joint moments stay valid after relocate (zero-encode path).
    EXPECT_TRUE(means_state->is_joint());
    EXPECT_TRUE(means_state->exp_avg.is_valid());
    EXPECT_TRUE(means_state->joint_bounds.is_valid());
}

// Relocation runs on the full model at the VRAM limit, so its temporaries
// must stay a small fraction of the model: no per-row index lists sized for
// every Gaussian, no extra copies of the sampling weights.
TEST(MCMCTest, RelocateTransientStaysSmall) {
    constexpr int n = 2'000'000;
    auto splat_data = create_test_splat_data(n);
    MCMC strategy(splat_data);

    param::OptimizationParameters opt_params;
    opt_params.iterations = 100;
    opt_params.max_cap = n;
    strategy.initialize(opt_params);

    // Every fourth row dead, spread over the model.
    std::vector<float> opacity(n, 0.5f);
    for (int i = 0; i < n; i += 4) {
        opacity[i] = -20.0f;
    }
    strategy.get_model().opacity_raw().copy_from(
        Tensor::from_vector(opacity, TensorShape({static_cast<size_t>(n), 1}), Device::CUDA));

    int device = 0;
    cudaMemPool_t pool = nullptr;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    ASSERT_EQ(cudaDeviceGetDefaultMemPool(&pool, device), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::uint64_t used_before = 0;
    std::uint64_t reset = 0;
    ASSERT_EQ(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &reset), cudaSuccess);
    ASSERT_EQ(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used_before), cudaSuccess);

    EXPECT_EQ(strategy.relocate_gs_test(), n / 4);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::uint64_t used_high = 0;
    ASSERT_EQ(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &used_high), cudaSuccess);
    // SH rows are staged in fixed-size chunks; everything else scales with N.
    const size_t transient = used_high - used_before;
    RecordProperty("relocate_transient_bytes", std::to_string(transient));
    // Sampling holds the dead mask, weights, CDF and two index lists (~13 B/G).
    EXPECT_LT(transient, size_t{16} * n + (size_t{96} << 20));
}

TEST(MCMCTest, RelocateGrowsRatioWorkspaceWhenMaxCapIsDisabled) {
    auto splat_data = create_test_splat_data(12);
    MCMC strategy(splat_data);

    param::OptimizationParameters opt_params;
    opt_params.iterations = 100;
    opt_params.max_cap = 0;
    strategy.initialize(opt_params);

    strategy.remove_gaussians(make_mask(12, 3));
    EXPECT_EQ(strategy.relocate_gs_test(), 3);
    EXPECT_EQ(strategy.get_model().visible_count(), 12);
}

TEST(MCMCTest, AddNewGaussiansExtendsDeletedMask) {
    auto splat_data = create_test_splat_data(8);
    MCMC strategy(splat_data);

    param::OptimizationParameters opt_params;
    opt_params.iterations = 100;
    opt_params.max_cap = 16;
    strategy.initialize(opt_params);

    strategy.remove_gaussians(make_mask(8, 2));
    ASSERT_TRUE(strategy.get_model().has_deleted_mask());
    ASSERT_EQ(strategy.get_model().visible_count(), 6);

    auto sampled_idxs = Tensor::from_vector(
        std::vector<int>{2, 3},
        TensorShape({2}),
        Device::CUDA);
    const int added = strategy.add_new_gs_with_indices_test(sampled_idxs);

    EXPECT_EQ(added, 2);
    EXPECT_EQ(strategy.get_model().size(), 10);
    EXPECT_EQ(strategy.get_model().visible_count(), 8);

    const auto deleted = strategy.get_model().deleted().to(DataType::Int32).cpu().to_vector();
    ASSERT_EQ(deleted.size(), 10);
    EXPECT_FLOAT_EQ(deleted[0], 1.0f);
    EXPECT_FLOAT_EQ(deleted[1], 1.0f);
    for (int i = 2; i < 10; ++i) {
        EXPECT_FLOAT_EQ(deleted[i], 0.0f);
    }
}

TEST(MCMCTest, RelocationWithTinyCropWeightsNeverCopiesFrozenParents) {
    auto splat_data = create_test_splat_data(16);
    std::vector<float> positions(16 * 3, 0.0f);
    for (size_t row = 0; row < 16; ++row)
        positions[3 * row] = static_cast<float>(row);
    splat_data.means() = Tensor::from_vector(positions, {16, 3}, Device::CUDA);
    splat_data.set_frozen_ranges({{0, 4}});
    MCMC strategy(splat_data);
    auto params = param::OptimizationParameters::mcmc_defaults();
    params.max_cap = 32;
    strategy.initialize(params);
    strategy.get_optimizer().set_crop_score_mask(make_mask(16, 16));
    // Initial error scores are floored to 1e-12; keep the resulting weight normal.
    strategy.get_optimizer().set_cropbox_lr_scale(
        1.01f * std::numeric_limits<float>::min() / 1e-12f);
    std::vector<bool> dead(16, false);
    for (size_t row = 4; row < 15; ++row)
        dead[row] = true;
    strategy.remove_gaussians(Tensor::from_vector(dead, {16}, Device::CUDA));
    ASSERT_EQ(strategy.relocate_gs_test(), 11);
    const auto result = splat_data.means().to_vector();
    for (size_t row = 0; row < 4; ++row)
        EXPECT_FLOAT_EQ(result[3 * row], static_cast<float>(row));
    for (size_t row = 4; row < 16; ++row)
        EXPECT_FLOAT_EQ(result[3 * row], 15.0f);
}
