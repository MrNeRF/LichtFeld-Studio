/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

class MRNFStrategyTest_EdgeGuidanceFactorPrefersHigherPrecomputedEdgeScores_Test;
class MRNFStrategyTest_GrowAndSplitResetsOptimizerStateForParents_Test;
class MRNFStrategyTest_SHDegree0KeepsShNEmptyAndFusedAdamUsableAfterGrowth_Test;
class MRNFStrategyTest_GrowAndSplitUsesIgsPlusSplitRule_Test;
class MRNFStrategyTest_GrowAndSplitWithoutMaxCapExtendsBookkeepingMasks_Test;
class MRNFStrategyTest_GrowAndSplitReplacementSkipsZeroWeightCandidates_Test;
class MRNFStrategyTest_GrowAndSplitReusesFreeSlotsBeforeAppending_Test;
class MRNFStrategyTest_SerializeRoundTripPreservesFreeMask_Test;
class MRNFStrategyTest_SerializeRoundTripPreservesLrScheduleState_Test;
class MRNFStrategyTest_DeserializeResizesTransientBuffersToLoadedModel_Test;
class MRNFStrategyTest_SetOptimizationParamsRecomputesDecayFromCurrentState_Test;
class MRNFStrategyTest_DegenerateBoundsStayInvalidAndKeepFiniteMeanLearningRate_Test;
class MRNFStrategyTest_LineBoundsUseFiniteSceneScaleForMeanLearningRate_Test;

#include "core/cuda/sh_layout.cuh"
#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "training/strategies/mrnf.hpp"

#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <sstream>
#include <vector>

using namespace lfs::core;
using namespace lfs::training;

namespace {

    SplatData create_mrnf_test_splat_data(const int n_gaussians = 10, const int sh_degree = 3) {
        const size_t n = static_cast<size_t>(n_gaussians);
        std::vector<float> means_data(n_gaussians * 3, 0.0f);
        for (int i = 0; i < n_gaussians; ++i) {
            means_data[i * 3 + 0] = static_cast<float>(i);
        }

        std::vector<float> sh0_data(n_gaussians * 3, 0.5f);
        std::vector<float> scaling_data(n_gaussians * 3, 0.0f);
        std::vector<float> rotation_data(n_gaussians * 4, 0.0f);
        std::vector<float> opacity_data(n_gaussians, 0.0f);
        const size_t sh_rest = sh_rest_coefficients_for_degree(sh_degree);

        for (int i = 0; i < n_gaussians; ++i) {
            rotation_data[i * 4 + 0] = 1.0f; // identity quaternion
        }

        auto means = Tensor::from_vector(means_data, TensorShape({n, 3}), Device::CUDA);
        auto sh0 = Tensor::from_vector(sh0_data, TensorShape({n, 1, 3}), Device::CUDA);
        auto shN = Tensor::zeros(TensorShape({n, sh_rest, 3}), Device::CUDA);
        auto scaling = Tensor::from_vector(scaling_data, TensorShape({n, 3}), Device::CUDA);
        auto rotation = Tensor::from_vector(rotation_data, TensorShape({n, 4}), Device::CUDA);
        auto opacity = Tensor::from_vector(opacity_data, TensorShape({n, 1}), Device::CUDA);

        return SplatData(sh_degree, means, sh0, shN, scaling, rotation, opacity, 1.0f);
    }

} // namespace

TEST(MRNFStrategyTest, EdgeGuidanceFactorPrefersHigherPrecomputedEdgeScores) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);

    param::OptimizationParameters opt_params;
    opt_params.iterations = 10'000;
    opt_params.refine_every = 100;
    opt_params.sh_degree_interval = 10'000;
    opt_params.max_cap = 32;
    opt_params.use_edge_map = true;

    strategy.initialize(opt_params);

    std::vector<float> edge_scores_data(10, 0.0f);
    edge_scores_data[0] = 1.0f;
    edge_scores_data[1] = 10.0f;
    strategy._precomputed_edge_scores =
        Tensor::from_vector(edge_scores_data, TensorShape({10}), Device::CUDA);
    strategy._edge_precompute_valid = true;

    const auto guidance = strategy.edge_guidance_factor().cpu();
    const float* guidance_ptr = guidance.ptr<float>();

    EXPECT_NEAR(guidance_ptr[2], 1.0f, 1e-5f);
    EXPECT_GT(guidance_ptr[0], 1.0f);
    EXPECT_GT(guidance_ptr[1], guidance_ptr[0]);
}

TEST(MRNFStrategyTest, DegenerateBoundsStayInvalidAndKeepFiniteMeanLearningRate) {
    auto splat_data = create_mrnf_test_splat_data(1);
    MRNF strategy(splat_data);

    auto opt_params = param::OptimizationParameters::mrnf_defaults();
    opt_params.iterations = 10'000;
    opt_params.max_cap = 32;
    strategy.initialize(opt_params);

    EXPECT_FALSE(strategy._bounds_valid);
    const float mean_lr = strategy.get_optimizer().get_param_lr(ParamType::Means);
    EXPECT_TRUE(std::isfinite(mean_lr));
    EXPECT_GT(mean_lr, 0.0f);
}

TEST(MRNFStrategyTest, LineBoundsUseFiniteSceneScaleForMeanLearningRate) {
    auto splat_data = create_mrnf_test_splat_data(10);
    MRNF strategy(splat_data);

    auto opt_params = param::OptimizationParameters::mrnf_defaults();
    opt_params.iterations = 10'000;
    opt_params.max_cap = 32;
    strategy.initialize(opt_params);

    EXPECT_TRUE(strategy._bounds_valid);
    EXPECT_GT(strategy._bounds.median_size, 0.0f);
    const float mean_lr = strategy.get_optimizer().get_param_lr(ParamType::Means);
    EXPECT_TRUE(std::isfinite(mean_lr));
    EXPECT_GT(mean_lr, 0.0f);
}

TEST(MRNFStrategyTest, RemoveGaussiansKeepsOptimizerStateUsable) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);

    auto opt_params = param::OptimizationParameters::mrnf_defaults();
    opt_params.iterations = 10'000;
    opt_params.sh_degree_interval = 10'000;
    opt_params.max_cap = 32;

    strategy.initialize(opt_params);
    splat_data._densification_info = Tensor::ones({2, static_cast<size_t>(splat_data.size())}, Device::CUDA);

    const auto mask = Tensor::from_vector(
        std::vector<bool>{false, true, false, true, false, false, false, false, false, false},
        TensorShape({10}),
        Device::CUDA);

    strategy.remove_gaussians(mask);

    ASSERT_EQ(splat_data.size(), 8u);
    ASSERT_TRUE(splat_data._densification_info.is_valid());
    EXPECT_EQ(splat_data._densification_info.shape()[1], 8u);

    EXPECT_NO_THROW({
        auto& means_grad = strategy.get_optimizer().get_grad(ParamType::Means);
        EXPECT_EQ(means_grad.shape()[0], 8u);
    });
    EXPECT_NO_THROW({
        auto& opacity_grad = strategy.get_optimizer().get_grad(ParamType::Opacity);
        EXPECT_EQ(opacity_grad.shape()[0], 8u);
    });
}

TEST(MRNFStrategyTest, QuantizedShNFirstMomentStartsAtSignedZeroPoint) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);

    auto opt_params = param::OptimizationParameters::mrnf_defaults();
    opt_params.iterations = 10'000;
    opt_params.sh_degree_interval = 10'000;
    opt_params.max_cap = 32;

    strategy.initialize(opt_params);

    const auto* shN_state = strategy.get_optimizer().get_state(ParamType::ShN);
    ASSERT_NE(shN_state, nullptr);
    ASSERT_TRUE(shN_state->exp_avg.is_valid());
    ASSERT_TRUE(shN_state->exp_avg_sq.is_valid());
    ASSERT_EQ(shN_state->exp_avg.dtype(), DataType::UInt8);
    ASSERT_EQ(shN_state->exp_avg_sq.dtype(), DataType::UInt8);

    const auto exp_avg_cpu = shN_state->exp_avg.cpu();
    const auto exp_avg_sq_cpu = shN_state->exp_avg_sq.cpu();
    const auto* exp_avg = exp_avg_cpu.ptr<std::uint8_t>();
    const auto* exp_avg_sq = exp_avg_sq_cpu.ptr<std::uint8_t>();
    for (size_t i = 0; i < exp_avg_cpu.numel(); ++i) {
        EXPECT_EQ(exp_avg[i], static_cast<std::uint8_t>(128));
    }
    for (size_t i = 0; i < exp_avg_sq_cpu.numel(); ++i) {
        EXPECT_EQ(exp_avg_sq[i], static_cast<std::uint8_t>(0));
    }
}

TEST(MRNFStrategyTest, RemoveGaussiansCompactsQuantizedAdamScalesAndPreservesShNDtype) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);

    auto opt_params = param::OptimizationParameters::mrnf_defaults();
    opt_params.iterations = 10'000;
    opt_params.sh_degree_interval = 10'000;
    opt_params.max_cap = 32;

    strategy.initialize(opt_params);

    auto set_scale_rows = [](AdamParamState* state, const size_t rows, const float offset) {
        ASSERT_NE(state, nullptr);
        std::vector<float> first(rows);
        std::vector<float> second(rows);
        for (size_t i = 0; i < rows; ++i) {
            first[i] = offset + static_cast<float>(i);
            second[i] = offset + 100.0f + static_cast<float>(i);
        }
        state->exp_avg_scale = Tensor::from_vector(first, TensorShape({rows}), Device::CUDA);
        state->exp_avg_sq_scale = Tensor::from_vector(second, TensorShape({rows}), Device::CUDA);
    };

    constexpr size_t initial_rows = 10;
    set_scale_rows(strategy.get_optimizer().get_state_mutable(ParamType::Means), initial_rows, 10.0f);
    set_scale_rows(strategy.get_optimizer().get_state_mutable(ParamType::ShN), initial_rows, 20.0f);

    const auto remove_mask = Tensor::from_vector(
        std::vector<bool>{false, true, false, true, false, false, false, false, false, false},
        TensorShape({initial_rows}),
        Device::CUDA);

    strategy.remove_gaussians(remove_mask);

    const std::vector<float> expected_means{10.0f, 12.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f};
    const std::vector<float> expected_shN{20.0f, 22.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f};

    auto expect_scale_rows = [](const AdamParamState* state,
                                const std::vector<float>& expected,
                                const size_t expected_capacity) {
        ASSERT_NE(state, nullptr);
        ASSERT_TRUE(state->exp_avg_scale.is_valid());
        ASSERT_TRUE(state->exp_avg_sq_scale.is_valid());
        EXPECT_EQ(state->exp_avg_scale.numel(), expected.size());
        EXPECT_EQ(state->exp_avg_sq_scale.numel(), expected.size());
        EXPECT_EQ(state->exp_avg_scale.capacity(), expected_capacity);
        EXPECT_EQ(state->exp_avg_sq_scale.capacity(), expected_capacity);

        const auto exp_avg_scale_cpu = state->exp_avg_scale.cpu();
        const auto exp_avg_sq_scale_cpu = state->exp_avg_sq_scale.cpu();
        const float* exp_avg_scale = exp_avg_scale_cpu.ptr<float>();
        const float* exp_avg_sq_scale = exp_avg_sq_scale_cpu.ptr<float>();
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_FLOAT_EQ(exp_avg_scale[i], expected[i]);
            EXPECT_FLOAT_EQ(exp_avg_sq_scale[i], expected[i] + 100.0f);
        }
    };

    const auto* means_state = strategy.get_optimizer().get_state(ParamType::Means);
    const auto* shN_state = strategy.get_optimizer().get_state(ParamType::ShN);
    expect_scale_rows(means_state, expected_means, 32);
    expect_scale_rows(shN_state, expected_shN, 32);

    ASSERT_NE(shN_state, nullptr);
    ASSERT_TRUE(shN_state->exp_avg.is_valid());
    ASSERT_TRUE(shN_state->exp_avg_sq.is_valid());
    EXPECT_EQ(shN_state->exp_avg.dtype(), DataType::UInt8);
    EXPECT_EQ(shN_state->exp_avg_sq.dtype(), DataType::UInt8);
    EXPECT_EQ(shN_state->size,
              sh_swizzled_float_count(expected_shN.size(), static_cast<uint32_t>(splat_data.max_sh_coeffs_rest())));
    EXPECT_EQ(shN_state->capacity,
              sh_swizzled_float_count(32, static_cast<uint32_t>(splat_data.max_sh_coeffs_rest())));

    const auto exp_avg_cpu = shN_state->exp_avg.cpu();
    const auto* exp_avg = exp_avg_cpu.ptr<std::uint8_t>();
    for (size_t i = 0; i < exp_avg_cpu.numel(); ++i) {
        EXPECT_EQ(exp_avg[i], static_cast<std::uint8_t>(128));
    }
}

TEST(MRNFStrategyTest, GrowAndSplitResetsOptimizerStateForParents) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);

    auto opt_params = param::OptimizationParameters::mrnf_defaults();
    opt_params.iterations = 10'000;
    opt_params.sh_degree_interval = 10'000;
    opt_params.max_cap = 32;
    opt_params.growth_grad_threshold = 0.5f;
    opt_params.grow_fraction = 1.0f;
    opt_params.grow_until_iter = 10'000;

    strategy.initialize(opt_params);

    auto* means_state = strategy.get_optimizer().get_state_mutable(ParamType::Means);
    ASSERT_NE(means_state, nullptr);
    // grad is allocated lazily via get_grad(); force allocation before fill.
    strategy.get_optimizer().get_grad(ParamType::Means);
    means_state->exp_avg_scale.fill_(5.0f);
    means_state->exp_avg_sq_scale.fill_(6.0f);
    means_state->grad.fill_(7.0f);

    strategy._refine_weight_max = Tensor::zeros({static_cast<size_t>(splat_data.size())}, Device::CUDA);
    strategy._vis_count = Tensor::zeros({static_cast<size_t>(splat_data.size())}, Device::CUDA);

    const auto split_idx = Tensor::from_vector(std::vector<int>{0}, TensorShape({1}), Device::CUDA).to(DataType::Int64);
    strategy._refine_weight_max.index_put_(split_idx, Tensor::full({1}, 1.0f, Device::CUDA));
    strategy._vis_count.index_put_(split_idx, Tensor::full({1}, 1.0f, Device::CUDA));

    const size_t initial_size = splat_data.size();
    strategy.grow_and_split(1, 0);

    ASSERT_EQ(splat_data.size(), initial_size + 1);
    ASSERT_EQ(means_state->size, initial_size + 1);

    const auto exp_avg_cpu = means_state->exp_avg.cpu();
    const auto exp_avg_sq_cpu = means_state->exp_avg_sq.cpu();
    const auto exp_avg_scale_cpu = means_state->exp_avg_scale.cpu();
    const auto exp_avg_sq_scale_cpu = means_state->exp_avg_sq_scale.cpu();
    const auto grad_cpu = means_state->grad.cpu();

    const auto* exp_avg_ptr = exp_avg_cpu.ptr<std::uint8_t>();
    const auto* exp_avg_sq_ptr = exp_avg_sq_cpu.ptr<std::uint8_t>();
    const float* exp_avg_scale_ptr = exp_avg_scale_cpu.ptr<float>();
    const float* exp_avg_sq_scale_ptr = exp_avg_sq_scale_cpu.ptr<float>();
    const float* grad_ptr = grad_cpu.ptr<float>();

    for (int c = 0; c < 3; ++c) {
        EXPECT_EQ(exp_avg_ptr[c], static_cast<std::uint8_t>(128));
        EXPECT_EQ(exp_avg_sq_ptr[c], static_cast<std::uint8_t>(0));
        EXPECT_FLOAT_EQ(grad_ptr[c], 0.0f);
    }
    EXPECT_FLOAT_EQ(exp_avg_scale_ptr[0], 0.0f);
    EXPECT_FLOAT_EQ(exp_avg_sq_scale_ptr[0], 0.0f);

    for (int c = 0; c < 3; ++c) {
        EXPECT_FLOAT_EQ(grad_ptr[3 + c], 7.0f);
    }
    EXPECT_FLOAT_EQ(exp_avg_scale_ptr[1], 5.0f);
    EXPECT_FLOAT_EQ(exp_avg_sq_scale_ptr[1], 6.0f);

    const size_t child_offset = initial_size * 3;
    for (int c = 0; c < 3; ++c) {
        EXPECT_EQ(exp_avg_ptr[child_offset + c], static_cast<std::uint8_t>(128));
        EXPECT_EQ(exp_avg_sq_ptr[child_offset + c], static_cast<std::uint8_t>(0));
        EXPECT_FLOAT_EQ(grad_ptr[child_offset + c], 0.0f);
    }
    EXPECT_FLOAT_EQ(exp_avg_scale_ptr[initial_size], 0.0f);
    EXPECT_FLOAT_EQ(exp_avg_sq_scale_ptr[initial_size], 0.0f);
}

TEST(MRNFStrategyTest, SHDegree0KeepsShNEmptyAndFusedAdamUsableAfterGrowth) {
    auto splat_data = create_mrnf_test_splat_data(10, 0);
    MRNF strategy(splat_data);

    auto opt_params = param::OptimizationParameters::mrnf_defaults();
    opt_params.iterations = 10'000;
    opt_params.sh_degree_interval = 10'000;
    opt_params.max_cap = 32;
    opt_params.growth_grad_threshold = 0.5f;
    opt_params.grow_fraction = 1.0f;
    opt_params.grow_until_iter = 10'000;

    strategy.initialize(opt_params);

    ASSERT_TRUE(splat_data.shN().is_valid());
    EXPECT_EQ(splat_data.shN().numel(), 0u);
    auto* shN_state = strategy.get_optimizer().get_state_mutable(ParamType::ShN);
    ASSERT_NE(shN_state, nullptr);
    EXPECT_EQ(shN_state->size, 0u);

    EXPECT_NO_THROW({
        const auto fused = strategy.get_optimizer().prepare_fastgs_fused_adam(457);
        EXPECT_TRUE(fused.means.enabled);
        EXPECT_FALSE(fused.shN.enabled);
    });

    strategy._refine_weight_max = Tensor::zeros({static_cast<size_t>(splat_data.size())}, Device::CUDA);
    strategy._vis_count = Tensor::zeros({static_cast<size_t>(splat_data.size())}, Device::CUDA);

    const auto split_idx = Tensor::from_vector(std::vector<int>{0}, TensorShape({1}), Device::CUDA).to(DataType::Int64);
    strategy._refine_weight_max.index_put_(split_idx, Tensor::full({1}, 1.0f, Device::CUDA));
    strategy._vis_count.index_put_(split_idx, Tensor::full({1}, 1.0f, Device::CUDA));

    const size_t initial_size = splat_data.size();
    strategy.grow_and_split(1, 0);

    EXPECT_EQ(splat_data.size(), initial_size + 1);
    EXPECT_EQ(splat_data.shN().numel(), 0u);
    EXPECT_EQ(shN_state->size, 0u);
    EXPECT_NO_THROW({
        const auto fused = strategy.get_optimizer().prepare_fastgs_fused_adam(457);
        EXPECT_TRUE(fused.means.enabled);
        EXPECT_FALSE(fused.shN.enabled);
    });
}

TEST(MRNFStrategyTest, ShNReservationTracksMaxDegreeAndMaxCap) {
    constexpr int n_gaussians = 10;
    constexpr size_t max_cap = 70;

    const auto make_params = [] {
        auto opt_params = param::OptimizationParameters::mrnf_defaults();
        opt_params.iterations = 10'000;
        opt_params.sh_degree_interval = 10'000;
        opt_params.max_cap = static_cast<int>(max_cap);
        return opt_params;
    };

    const auto expect_shN_capacity = [](const SplatData& splat_data,
                                        const AdamOptimizer& optimizer,
                                        const int max_degree) {
        const auto layout_rest = static_cast<uint32_t>(sh_rest_coefficients_for_degree(max_degree));
        const size_t expected_logical =
            sh_swizzled_float_count(static_cast<size_t>(n_gaussians), layout_rest);
        const size_t expected_capacity = sh_swizzled_float_count(max_cap, layout_rest);

        ASSERT_TRUE(splat_data.shN().is_valid());
        EXPECT_EQ(splat_data.shN().numel(), expected_logical);
        EXPECT_EQ(splat_data.shN().capacity(), expected_capacity);

        const auto* state = optimizer.get_state(ParamType::ShN);
        ASSERT_NE(state, nullptr);
        EXPECT_EQ(state->size, expected_logical);
        EXPECT_EQ(state->capacity, expected_capacity);
        if (layout_rest == 0) {
            EXPECT_FALSE(state->exp_avg.is_valid());
            EXPECT_FALSE(state->exp_avg_sq.is_valid());
        } else {
            ASSERT_TRUE(state->exp_avg.is_valid());
            ASSERT_TRUE(state->exp_avg_sq.is_valid());
            EXPECT_EQ(state->exp_avg.numel(), expected_logical);
            EXPECT_EQ(state->exp_avg.capacity(), expected_capacity);
            EXPECT_EQ(state->exp_avg_sq.numel(), expected_logical);
            EXPECT_EQ(state->exp_avg_sq.capacity(), expected_capacity);
        }
    };

    for (const int sh_degree : {0, 1, 2, 3}) {
        auto splat_data = create_mrnf_test_splat_data(n_gaussians, sh_degree);
        MRNF strategy(splat_data);

        strategy.initialize(make_params());

        expect_shN_capacity(splat_data, strategy.get_optimizer(), sh_degree);
    }

    auto scheduled_splat = create_mrnf_test_splat_data(n_gaussians, 1);
    scheduled_splat.set_active_sh_degree(0);
    MRNF scheduled_strategy(scheduled_splat);

    scheduled_strategy.initialize(make_params());
    expect_shN_capacity(scheduled_splat, scheduled_strategy.get_optimizer(), 1);
    EXPECT_FALSE(scheduled_strategy.get_optimizer().prepare_fastgs_fused_adam(1001).shN.enabled);

    scheduled_splat.increment_sh_degree();
    const auto fused = scheduled_strategy.get_optimizer().prepare_fastgs_fused_adam(1001);
    EXPECT_TRUE(fused.shN.enabled);
    expect_shN_capacity(scheduled_splat, scheduled_strategy.get_optimizer(), 1);
}

TEST(MRNFStrategyTest, GrowAndSplitUsesIgsPlusSplitRule) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);

    auto opt_params = param::OptimizationParameters::mrnf_defaults();
    opt_params.iterations = 10'000;
    opt_params.sh_degree_interval = 10'000;
    opt_params.max_cap = 32;
    opt_params.growth_grad_threshold = 0.5f;
    opt_params.grow_fraction = 1.0f;
    opt_params.grow_until_iter = 10'000;
    strategy.initialize(opt_params);

    strategy._refine_weight_max = Tensor::zeros({static_cast<size_t>(splat_data.size())}, Device::CUDA);
    strategy._vis_count = Tensor::zeros({static_cast<size_t>(splat_data.size())}, Device::CUDA);

    const auto split_idx = Tensor::from_vector(std::vector<int>{0}, TensorShape({1}), Device::CUDA).to(DataType::Int64);
    strategy._refine_weight_max.index_put_(split_idx, Tensor::full({1}, 1.0f, Device::CUDA));
    strategy._vis_count.index_put_(split_idx, Tensor::full({1}, 1.0f, Device::CUDA));

    const size_t initial_size = splat_data.size();
    strategy.grow_and_split(1, 0);

    ASSERT_EQ(splat_data.size(), initial_size + 1);

    const auto means_cpu = splat_data.means().cpu();
    const auto scales_cpu = splat_data.scaling_raw().cpu();
    const auto opacities_cpu = splat_data.opacity_raw().cpu();

    const float* means_ptr = means_cpu.ptr<float>();
    const float* scales_ptr = scales_cpu.ptr<float>();
    const float* opacities_ptr = opacities_cpu.ptr<float>();

    EXPECT_NEAR(means_ptr[0], 0.5f, 1e-5f);
    EXPECT_NEAR(means_ptr[1], 0.0f, 1e-5f);
    EXPECT_NEAR(means_ptr[2], 0.0f, 1e-5f);

    const size_t child_base = initial_size * 3;
    EXPECT_NEAR(means_ptr[child_base + 0], -0.5f, 1e-5f);
    EXPECT_NEAR(means_ptr[child_base + 1], 0.0f, 1e-5f);
    EXPECT_NEAR(means_ptr[child_base + 2], 0.0f, 1e-5f);

    EXPECT_NEAR(scales_ptr[0], std::log(0.5f), 1e-5f);
    EXPECT_NEAR(scales_ptr[1], std::log(0.85f), 1e-5f);
    EXPECT_NEAR(scales_ptr[2], std::log(0.85f), 1e-5f);

    const size_t child_scale_base = initial_size * 3;
    EXPECT_NEAR(scales_ptr[child_scale_base + 0], std::log(0.5f), 1e-5f);
    EXPECT_NEAR(scales_ptr[child_scale_base + 1], std::log(0.85f), 1e-5f);
    EXPECT_NEAR(scales_ptr[child_scale_base + 2], std::log(0.85f), 1e-5f);

    EXPECT_NEAR(opacities_ptr[0], std::log(0.3f / 0.7f), 1e-5f);
    EXPECT_NEAR(opacities_ptr[initial_size], std::log(0.3f / 0.7f), 1e-5f);
}

TEST(MRNFStrategyTest, StepScalingDoesNotScaleSparsifySteps) {
    auto params = param::OptimizationParameters::mrnf_defaults();
    params.grow_until_iter = 15000;
    params.sparsify_steps = 15000;
    params.steps_scaler = 0.5f;

    params.apply_step_scaling();

    EXPECT_EQ(params.grow_until_iter, 7500u);
    EXPECT_EQ(params.sparsify_steps, 15000);
    EXPECT_EQ(params.refine_every, 100u);
    EXPECT_EQ(params.stop_refine, 14250u);
}

TEST(MRNFStrategyTest, GrowAndSplitWithoutMaxCapExtendsBookkeepingMasks) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);

    auto opt_params = param::OptimizationParameters::mrnf_defaults();
    opt_params.iterations = 10'000;
    opt_params.sh_degree_interval = 10'000;
    opt_params.max_cap = 0;
    opt_params.growth_grad_threshold = 0.5f;
    opt_params.grow_fraction = 1.0f;
    opt_params.grow_until_iter = 10'000;
    strategy.initialize(opt_params);

    strategy._refine_weight_max = Tensor::zeros({static_cast<size_t>(splat_data.size())}, Device::CUDA);
    strategy._vis_count = Tensor::zeros({static_cast<size_t>(splat_data.size())}, Device::CUDA);

    const auto split_idx = Tensor::from_vector(std::vector<int>{0}, TensorShape({1}), Device::CUDA).to(DataType::Int64);
    strategy._refine_weight_max.index_put_(split_idx, Tensor::full({1}, 1.0f, Device::CUDA));
    strategy._vis_count.index_put_(split_idx, Tensor::full({1}, 1.0f, Device::CUDA));

    const size_t initial_size = splat_data.size();
    ASSERT_NO_THROW(strategy.grow_and_split(1, 0));

    EXPECT_EQ(splat_data.size(), initial_size + 1);
    ASSERT_TRUE(splat_data.has_deleted_mask());
    EXPECT_EQ(splat_data.deleted().shape()[0], splat_data.size());
    EXPECT_EQ(strategy.free_count(), 0u);
}

TEST(MRNFStrategyTest, GrowAndSplitReplacementSkipsZeroWeightCandidates) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);

    auto opt_params = param::OptimizationParameters::mrnf_defaults();
    opt_params.iterations = 10'000;
    opt_params.sh_degree_interval = 10'000;
    opt_params.max_cap = 32;
    opt_params.growth_grad_threshold = 0.5f;
    opt_params.grow_fraction = 0.0f;
    opt_params.grow_until_iter = 0;
    strategy.initialize(opt_params);

    const auto free_indices = Tensor::from_vector(std::vector<int>{8, 9}, TensorShape({2}), Device::CUDA).to(DataType::Int64);
    strategy.mark_as_free(free_indices);
    auto true_vals = Tensor::ones_bool({2}, Device::CUDA);
    strategy._splat_data->deleted().index_put_(free_indices, true_vals);

    strategy._refine_weight_max = Tensor::zeros({static_cast<size_t>(splat_data.size())}, Device::CUDA);
    strategy._vis_count = Tensor::zeros({static_cast<size_t>(splat_data.size())}, Device::CUDA);

    const auto visible_parent = Tensor::from_vector(std::vector<int>{0}, TensorShape({1}), Device::CUDA).to(DataType::Int64);
    strategy._vis_count.index_put_(visible_parent, Tensor::full({1}, 1.0f, Device::CUDA));

    const size_t initial_size = splat_data.size();
    strategy.grow_and_split(10'001, 2);

    EXPECT_EQ(splat_data.size(), initial_size);
    EXPECT_EQ(strategy.free_count(), 1u);
    EXPECT_EQ(strategy.active_count(), initial_size - 1);
}

TEST(MRNFStrategyTest, GrowAndSplitReusesFreeSlotsBeforeAppending) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);

    auto opt_params = param::OptimizationParameters::mrnf_defaults();
    opt_params.iterations = 10'000;
    opt_params.sh_degree_interval = 10'000;
    opt_params.max_cap = 32;
    opt_params.growth_grad_threshold = 0.5f;
    opt_params.grow_fraction = 1.0f;
    opt_params.grow_until_iter = 10'000;
    strategy.initialize(opt_params);

    const auto free_indices = Tensor::from_vector(std::vector<int>{8, 9}, TensorShape({2}), Device::CUDA).to(DataType::Int64);
    strategy.mark_as_free(free_indices);
    auto true_vals = Tensor::ones_bool({2}, Device::CUDA);
    strategy._splat_data->deleted().index_put_(free_indices, true_vals);

    strategy._refine_weight_max = Tensor::zeros({static_cast<size_t>(splat_data.size())}, Device::CUDA);
    strategy._vis_count = Tensor::zeros({static_cast<size_t>(splat_data.size())}, Device::CUDA);

    const auto split_idx = Tensor::from_vector(std::vector<int>{0}, TensorShape({1}), Device::CUDA).to(DataType::Int64);
    strategy._refine_weight_max.index_put_(split_idx, Tensor::full({1}, 1.0f, Device::CUDA));
    strategy._vis_count.index_put_(split_idx, Tensor::full({1}, 1.0f, Device::CUDA));

    const size_t initial_size = splat_data.size();
    strategy.grow_and_split(1, 0);

    EXPECT_EQ(splat_data.size(), initial_size);
    EXPECT_EQ(strategy.active_count(), initial_size - 1);
    EXPECT_EQ(strategy.free_count(), 1u);
}

TEST(MRNFStrategyTest, SerializeRoundTripPreservesFreeMask) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);

    auto opt_params = param::OptimizationParameters::mrnf_defaults();
    opt_params.iterations = 10'000;
    opt_params.sh_degree_interval = 10'000;
    opt_params.max_cap = 32;
    strategy.initialize(opt_params);

    const auto free_indices = Tensor::from_vector(std::vector<int>{1, 3}, TensorShape({2}), Device::CUDA).to(DataType::Int64);
    strategy.mark_as_free(free_indices);
    auto true_vals = Tensor::ones_bool({2}, Device::CUDA);
    strategy._splat_data->deleted().index_put_(free_indices, true_vals);

    std::stringstream ss;
    strategy.serialize(ss);

    auto splat_data_copy = create_mrnf_test_splat_data();
    MRNF restored(splat_data_copy);
    restored.initialize(opt_params);
    restored.deserialize(ss);

    const auto free_mask_cpu = restored._free_mask.cpu();
    const bool* free_mask_ptr = free_mask_cpu.ptr<bool>();

    EXPECT_TRUE(free_mask_ptr[1]);
    EXPECT_TRUE(free_mask_ptr[3]);
    EXPECT_FALSE(free_mask_ptr[0]);
    EXPECT_EQ(restored.free_count(), 2u);
}

TEST(MRNFStrategyTest, SerializeRoundTripPreservesLrScheduleState) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);

    auto opt_params = param::OptimizationParameters::mrnf_defaults();
    opt_params.iterations = 10'000;
    opt_params.sh_degree_interval = 10'000;
    opt_params.max_cap = 32;
    strategy.initialize(opt_params);

    strategy._mean_lr_unscaled = 1.5e-6;
    strategy._scale_lr_current = 2.5e-4;
    strategy.refresh_decay_schedule_from_current_state();

    std::stringstream ss;
    strategy.serialize(ss);

    auto splat_data_copy = create_mrnf_test_splat_data();
    MRNF restored(splat_data_copy);
    restored.initialize(opt_params);
    restored.deserialize(ss);

    EXPECT_DOUBLE_EQ(restored._mean_lr_unscaled, strategy._mean_lr_unscaled);
    EXPECT_DOUBLE_EQ(restored._scale_lr_current, strategy._scale_lr_current);
    EXPECT_NEAR(restored.get_optimizer().get_param_lr(ParamType::Scaling), strategy._scale_lr_current, 1e-12);
    EXPECT_NEAR(restored.get_optimizer().get_param_lr(ParamType::Means),
                strategy._mean_lr_unscaled * restored._bounds.median_size,
                1e-12);
}

TEST(MRNFStrategyTest, DeserializeResizesTransientBuffersToLoadedModel) {
    auto splat_data = create_mrnf_test_splat_data(12);
    MRNF strategy(splat_data);

    auto opt_params = param::OptimizationParameters::mrnf_defaults();
    opt_params.iterations = 10'000;
    opt_params.sh_degree_interval = 10'000;
    opt_params.max_cap = 32;
    strategy.initialize(opt_params);

    std::stringstream ss;
    splat_data.serialize(ss);
    strategy.serialize(ss);
    ss.seekg(0);

    auto smaller_splat_data = create_mrnf_test_splat_data(5);
    MRNF restored(smaller_splat_data);
    restored.initialize(opt_params);
    smaller_splat_data.deserialize(ss);
    restored.deserialize(ss);

    EXPECT_EQ(restored._refine_weight_max.numel(), 12u);
    EXPECT_EQ(restored._vis_count.numel(), 12u);
    ASSERT_TRUE(restored.get_model()._densification_info.is_valid());
    EXPECT_EQ(restored.get_model()._densification_info.shape()[1], 12u);
    EXPECT_FALSE(restored._edge_precompute_valid);
}

TEST(MRNFStrategyTest, SetOptimizationParamsRecomputesDecayFromCurrentState) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);

    auto opt_params = param::OptimizationParameters::mrnf_defaults();
    opt_params.iterations = 10'000;
    opt_params.sh_degree_interval = 10'000;
    opt_params.max_cap = 32;
    strategy.initialize(opt_params);

    auto* means_state = strategy.get_optimizer().get_state_mutable(ParamType::Means);
    auto* scaling_state = strategy.get_optimizer().get_state_mutable(ParamType::Scaling);
    ASSERT_NE(means_state, nullptr);
    ASSERT_NE(scaling_state, nullptr);
    means_state->step_count = 1200;
    scaling_state->step_count = 1200;

    strategy._mean_lr_unscaled = 2.0e-6;
    strategy._scale_lr_current = 4.0e-4;

    auto resumed_params = opt_params;
    resumed_params.iterations = 3000;
    resumed_params.means_lr_end = 1.0e-7f;
    resumed_params.scaling_lr_end = 1.0e-4f;

    strategy.set_optimization_params(resumed_params);

    const double expected_mean_gamma =
        std::pow(resumed_params.means_lr_end / strategy._mean_lr_unscaled,
                 1.0 / static_cast<double>(resumed_params.iterations - means_state->step_count));
    const double expected_scale_gamma =
        std::pow(resumed_params.scaling_lr_end / strategy._scale_lr_current,
                 1.0 / static_cast<double>(resumed_params.iterations - scaling_state->step_count));

    EXPECT_NEAR(strategy._mean_lr_gamma, expected_mean_gamma, 1e-12);
    EXPECT_NEAR(strategy._scale_lr_gamma, expected_scale_gamma, 1e-12);
    EXPECT_NEAR(strategy.get_optimizer().get_param_lr(ParamType::Scaling), strategy._scale_lr_current, 1e-12);
}
