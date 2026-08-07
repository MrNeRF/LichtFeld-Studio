/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "lfs/training/joint_adam_codec.hpp"
#include "lfs/training/sh_value_codec.hpp"
#include "training/strategies/improved_gs_plus.hpp"
#include "training/strategies/mcmc.hpp"
#include <algorithm>
#include <gtest/gtest.h>
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
    strategy.get_optimizer().set_crop_damping_mask(crop_mask);
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
    strategy.get_optimizer().set_crop_damping_mask({});
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
    strategy.get_optimizer().set_crop_damping_mask(crop_mask);
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
    strategy.get_optimizer().set_crop_damping_mask({});
    const auto unmasked_scores = strategy.damp_densification_scores(scores).cpu().to_vector();
    EXPECT_EQ(unit_scale_scores, unmasked_scores);
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

    const int relocated = strategy.relocate_gs_test();
    EXPECT_EQ(relocated, 3);
    EXPECT_EQ(strategy.get_model().visible_count(), 12);

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
