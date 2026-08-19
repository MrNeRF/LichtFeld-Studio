/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

class MRNFStrategyTest_EdgeGuidanceFactorPrefersHigherPrecomputedEdgeScores_Test;
class MRNFStrategyTest_GrowAndSplitResetsOptimizerStateForParents_Test;
class MRNFStrategyTest_SHDegree0KeepsShNEmptyAndFusedAdamUsableAfterGrowth_Test;
class MRNFStrategyTest_GrowAndSplitUsesIgsPlusSplitRule_Test;
class MRNFStrategyTest_GrowAndSplitWithoutMaxCapExtendsBookkeepingMasks_Test;
class MRNFStrategyTest_DeletedMaskCapacityGrowthPreservesExistingRows_Test;
class MRNFStrategyTest_GrowAndSplitReplacementSkipsZeroWeightCandidates_Test;
class MRNFStrategyTest_GrowAndSplitReusesFreeSlotsBeforeAppending_Test;
class MRNFStrategyTest_SerializeRoundTripPreservesFreeMask_Test;
class MRNFStrategyTest_SerializeRoundTripPreservesLrScheduleState_Test;
class MRNFStrategyTest_DeserializeResizesTransientBuffersToLoadedModel_Test;
class MRNFStrategyTest_SetOptimizationParamsRecomputesDecayFromCurrentState_Test;
class MRNFStrategyTest_DegenerateBoundsStayInvalidAndKeepFiniteMeanLearningRate_Test;
class MRNFStrategyTest_LineBoundsUseFiniteSceneScaleForMeanLearningRate_Test;
class CropDampingStrategyTest_MrnfRejectedRowsAreNotRefineCandidatesAtZeroScale_Test;
class MRNFStrategyTest_FarFieldClassificationStableAfterFarInserts_Test;
class MRNFStrategyTest_FarDecayScaleAppliesOnlyToFarUnfrozenRows_Test;
class MRNFStrategyTest_MaxModeFoldUsesViewMaxRow_Test;
class MRNFStrategyTest_SumModeFoldIsBitIdentical_Test;
class MRNFStrategyTest_FarUnseenCullIncrementsOnlyFarInvisible_Test;
class MRNFStrategyTest_AlwaysModeGrowthAddsTargetAndIgnoresGates_Test;
class MRNFStrategyTest_ThresholdModeMatchesCurrentSelection_Test;
class MRNFStrategyTest_ScreenClipScalesOnlyOverLimitRows_Test;
class MRNFStrategyTest_CadenceScaledMatchesRefineEvery_Test;
class MRNFStrategyTest_FarStarvationFactorFromSyntheticPopulations_Test;

#include "core/camera.hpp"
#include "core/cuda/sh_layout.cuh"
#include "core/parameters.hpp"
#include "core/sh_value_quant.hpp"
#include "core/splat_data.hpp"
#include "lfs/training/joint_adam_codec.hpp"
#include "lfs/training/sh_value_codec.hpp"
#include "training/dataset.hpp"
#include "training/kernels/mrnf_kernels.hpp"
#include "training/optimizer/render_output.hpp"
#include "training/strategies/mrnf.hpp"
#include "training/strategies/strategy_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
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

    void disable_default_far_field(param::OptimizationParameters& p) {
        p.explore_splits = 0;
        p.explore_seeds = 0;
        p.far_growth_cap = 1.0f;
        p.far_decay_scale = 1.0f;
        p.mean_step_mode = param::MeanStepMode::Global;
    }

    param::OptimizationParameters vanilla_mrnf_params() {
        auto p = param::OptimizationParameters::mrnf_defaults();
        disable_default_far_field(p);
        return p;
    }

    void decode_joint_g1g2(const uint8_t* packed, const int bits, const size_t cell,
                           const float umin, const float umax, const float smin, const float smax,
                           float& m, float& v) {
        if (bits == 16) {
            joint_adam::Codec16::decode_g1g2(packed, cell, umin, umax, smin, smax, m, v);
        } else {
            joint_adam::Codec8::decode_g1g2(packed, cell, umin, umax, smin, smax, m, v);
        }
    }

    // 1 LSB of (u, log_s) at the live block bounds, mapped through us_to_g1g2 at (0,0)+1LSB.
    void joint_zero_decode_tol(const float* bb, const int bits, float& m_tol, float& v_tol) {
        const float qmax = bits == 16 ? joint_adam::Codec16::kQMax : joint_adam::Codec8::kQMax;
        const float du = (bb[1] - bb[0]) / qmax;
        const float ds = (bb[3] - bb[2]) / qmax;
        float m = 0.0f;
        float v = 0.0f;
        joint_adam::Codec16::us_to_g1g2(du, ds, m, v);
        m_tol = std::max(std::abs(m) * 2.0f, 1e-6f);
        v_tol = std::max(std::abs(v) * 2.0f, 1e-12f);
    }

    void expect_contiguous_joint_row_zero(const AdamParamState& state, const size_t prim,
                                          const char* label) {
        ASSERT_TRUE(state.is_joint());
        ASSERT_TRUE(state.exp_avg.is_valid());
        ASSERT_TRUE(state.joint_bounds.is_valid());
        ASSERT_GE(state.exp_avg.shape().rank(), 2u);
        const auto packed = state.exp_avg.cpu();
        const auto bounds = state.joint_bounds.cpu();
        const int bpc = joint_adam::bytes_per_cell(state.joint_bits);
        ASSERT_GT(bpc, 0);
        ASSERT_EQ(packed.shape()[1] % static_cast<size_t>(bpc), 0u);
        const size_t n_attr = packed.shape()[1] / static_cast<size_t>(bpc);
        const size_t bidx = prim / static_cast<size_t>(joint_adam::kBlockSize);
        ASSERT_LT(bidx, bounds.shape()[0]);
        const float* bb = bounds.ptr<float>() + bidx * 4;
        float m_tol = 0.0f;
        float v_tol = 0.0f;
        joint_zero_decode_tol(bb, state.joint_bits, m_tol, v_tol);
        const auto* bytes = packed.ptr<uint8_t>();
        for (size_t a = 0; a < n_attr; ++a) {
            float m = 0.0f;
            float v = 0.0f;
            decode_joint_g1g2(bytes, state.joint_bits, prim * n_attr + a,
                              bb[0], bb[1], bb[2], bb[3], m, v);
            EXPECT_NEAR(m, 0.0f, m_tol) << label << " prim=" << prim << " attr=" << a;
            EXPECT_NEAR(v, 0.0f, v_tol) << label << " prim=" << prim << " attr=" << a;
        }
    }

    bool contiguous_joint_row_has_nonzero(const AdamParamState& state, const size_t prim) {
        if (!state.is_joint() || !state.exp_avg.is_valid() || !state.joint_bounds.is_valid() ||
            state.exp_avg.shape().rank() < 2) {
            return false;
        }
        const auto packed = state.exp_avg.cpu();
        const auto bounds = state.joint_bounds.cpu();
        const int bpc = joint_adam::bytes_per_cell(state.joint_bits);
        if (bpc <= 0 || packed.shape()[1] % static_cast<size_t>(bpc) != 0) {
            return false;
        }
        const size_t n_attr = packed.shape()[1] / static_cast<size_t>(bpc);
        const size_t bidx = prim / static_cast<size_t>(joint_adam::kBlockSize);
        if (bidx >= bounds.shape()[0]) {
            return false;
        }
        const float* bb = bounds.ptr<float>() + bidx * 4;
        const auto* bytes = packed.ptr<uint8_t>();
        for (size_t a = 0; a < n_attr; ++a) {
            float m = 0.0f;
            float v = 0.0f;
            decode_joint_g1g2(bytes, state.joint_bits, prim * n_attr + a,
                              bb[0], bb[1], bb[2], bb[3], m, v);
            if (std::abs(m) > 1e-12f || std::abs(v) > 1e-20f) {
                return true;
            }
        }
        return false;
    }

    void seed_shN_joint_nonzero(AdamParamState& state, const uint32_t layout_rest) {
        auto packed = state.exp_avg.cpu();
        auto bounds = state.joint_bounds.cpu();
        auto* bytes = packed.ptr<uint8_t>();
        auto* bb = bounds.ptr<float>();
        bb[0] = 0.5f;
        bb[1] = 1.5f;
        bb[2] = 0.25f;
        bb[3] = 1.0f;
        const uint32_t slots = sh_float4_slots_for_rest(layout_rest);
        for (const uint32_t prim : {0u, 1u}) {
            for (uint32_t k = 0; k < slots; ++k) {
                const size_t base =
                    static_cast<size_t>(sh_swizzled_index(prim, k, layout_rest)) * 4u;
                for (int c = 0; c < 4; ++c) {
                    const size_t cell = base + static_cast<size_t>(c);
                    bytes[cell * 2 + 0] = 200;
                    bytes[cell * 2 + 1] = 200;
                }
            }
        }
        state.exp_avg = packed.cuda();
        state.joint_bounds = bounds.cuda();
    }

    void expect_shN_joint_row_zero(const AdamParamState& state, const size_t prim,
                                   const uint32_t layout_rest, const char* label) {
        ASSERT_TRUE(state.is_joint());
        ASSERT_TRUE(state.exp_avg.is_valid());
        ASSERT_TRUE(state.joint_bounds.is_valid());
        const auto packed = state.exp_avg.cpu();
        const auto bounds = state.joint_bounds.cpu();
        const size_t bidx = prim / static_cast<size_t>(joint_adam::kBlockSize);
        ASSERT_LT(bidx, bounds.shape()[0]);
        const float* bb = bounds.ptr<float>() + bidx * 4;
        float m_tol = 0.0f;
        float v_tol = 0.0f;
        joint_zero_decode_tol(bb, state.joint_bits, m_tol, v_tol);
        const auto* bytes = packed.ptr<uint8_t>();
        const uint32_t slots = sh_float4_slots_for_rest(layout_rest);
        for (uint32_t k = 0; k < slots; ++k) {
            const size_t base =
                static_cast<size_t>(sh_swizzled_index(static_cast<uint32_t>(prim), k, layout_rest)) *
                4u;
            for (int c = 0; c < 4; ++c) {
                float m = 0.0f;
                float v = 0.0f;
                decode_joint_g1g2(bytes, state.joint_bits, base + static_cast<size_t>(c),
                                  bb[0], bb[1], bb[2], bb[3], m, v);
                EXPECT_NEAR(m, 0.0f, m_tol)
                    << label << " prim=" << prim << " slot=" << k << " c=" << c;
                EXPECT_NEAR(v, 0.0f, v_tol)
                    << label << " prim=" << prim << " slot=" << k << " c=" << c;
            }
        }
    }

} // namespace

TEST(MRNFStrategyTest, EdgeGuidanceFactorPrefersHigherPrecomputedEdgeScores) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);

    param::OptimizationParameters opt_params;
    disable_default_far_field(opt_params);
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

TEST(CropDampingStrategyTest, MrnfRejectedRowsAreNotRefineCandidatesAtZeroScale) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);

    auto opt_params = vanilla_mrnf_params();
    opt_params.iterations = 100;
    opt_params.max_cap = 32;
    opt_params.growth_grad_threshold = 0.5f;
    strategy.initialize(opt_params);
    strategy._refine_weight_max = Tensor::ones({10}, Device::CUDA);
    strategy._vis_count = Tensor::ones({10}, Device::CUDA);

    auto crop_mask = Tensor::zeros_bool({10}, Device::CPU);
    crop_mask.ptr<unsigned char>()[0] = 1;
    strategy.get_optimizer().set_crop_damping_mask(crop_mask);
    strategy.get_optimizer().set_cropbox_lr_scale(0.0f);

    const auto damped_candidates =
        strategy.compute_refine_candidates().to(DataType::Int32).to_vector_int();
    ASSERT_EQ(damped_candidates.size(), 10u);
    EXPECT_EQ(damped_candidates[0], 0);
    for (size_t i = 1; i < damped_candidates.size(); ++i) {
        EXPECT_EQ(damped_candidates[i], 1);
    }

    strategy.get_optimizer().set_cropbox_lr_scale(1.0f);
    const auto unit_scale_candidates =
        strategy.compute_refine_candidates().to(DataType::Int32).to_vector_int();
    strategy.get_optimizer().set_crop_damping_mask({});
    const auto unmasked_candidates =
        strategy.compute_refine_candidates().to(DataType::Int32).to_vector_int();
    EXPECT_EQ(unit_scale_candidates, unmasked_candidates);
}

TEST(MRNFStrategyTest, DegenerateBoundsStayInvalidAndKeepFiniteMeanLearningRate) {
    auto splat_data = create_mrnf_test_splat_data(1);
    MRNF strategy(splat_data);

    auto opt_params = vanilla_mrnf_params();
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

    auto opt_params = vanilla_mrnf_params();
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

    auto opt_params = vanilla_mrnf_params();
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

    auto opt_params = vanilla_mrnf_params();
    opt_params.iterations = 10'000;
    opt_params.sh_degree_interval = 10'000;
    opt_params.max_cap = 32;

    strategy.initialize(opt_params);

    const auto* shN_state = strategy.get_optimizer().get_state(ParamType::ShN);
    ASSERT_NE(shN_state, nullptr);
    ASSERT_TRUE(shN_state->is_joint());
    ASSERT_TRUE(shN_state->exp_avg.is_valid());
    ASSERT_TRUE(shN_state->joint_bounds.is_valid());
    ASSERT_EQ(shN_state->exp_avg.dtype(), DataType::UInt8);
    // Joint: all-zero packed + bounds decode to (m,v)=(0,0) — free zero moments.
    const auto packed = shN_state->exp_avg.cpu();
    const auto* bytes = packed.ptr<std::uint8_t>();
    for (size_t i = 0; i < packed.numel(); ++i) {
        EXPECT_EQ(bytes[i], static_cast<std::uint8_t>(0));
    }
}

TEST(MRNFStrategyTest, RemoveGaussiansCompactsQuantizedAdamScalesAndPreservesShNDtype) {
    // Legacy per-primitive scale compaction removed with the legacy Adam codec.
    // Joint path: compact preserves joint packed moments + dtype.
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);

    auto opt_params = vanilla_mrnf_params();
    opt_params.iterations = 10'000;
    opt_params.sh_degree_interval = 10'000;
    opt_params.max_cap = 32;

    strategy.initialize(opt_params);

    constexpr size_t initial_rows = 10;
    const auto remove_mask = Tensor::from_vector(
        std::vector<bool>{false, true, false, true, false, false, false, false, false, false},
        TensorShape({initial_rows}),
        Device::CUDA);

    strategy.remove_gaussians(remove_mask);

    const auto* means_state = strategy.get_optimizer().get_state(ParamType::Means);
    const auto* shN_state = strategy.get_optimizer().get_state(ParamType::ShN);
    ASSERT_NE(means_state, nullptr);
    ASSERT_NE(shN_state, nullptr);
    EXPECT_TRUE(means_state->is_joint());
    EXPECT_TRUE(shN_state->is_joint());
    EXPECT_TRUE(means_state->exp_avg.is_valid());
    EXPECT_TRUE(shN_state->exp_avg.is_valid());
    EXPECT_EQ(shN_state->exp_avg.dtype(), DataType::UInt8);
    constexpr size_t expected_rows = 8;
    EXPECT_EQ(shN_state->size,
              sh_swizzled_float_count(expected_rows, static_cast<uint32_t>(splat_data.max_sh_coeffs_rest())));
}

TEST(MRNFStrategyTest, GrowAndSplitResetsOptimizerStateForParents) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);

    auto opt_params = vanilla_mrnf_params();
    opt_params.iterations = 10'000;
    opt_params.sh_degree_interval = 10'000;
    opt_params.max_cap = 32;
    opt_params.growth_grad_threshold = 0.5f;
    opt_params.grow_fraction = 1.0f;
    opt_params.grow_until_iter = 10'000;

    strategy.initialize(opt_params);

    auto& optimizer = strategy.get_optimizer();
    auto* means_state = optimizer.get_state_mutable(ParamType::Means);
    ASSERT_NE(means_state, nullptr);
    ASSERT_TRUE(means_state->is_joint());

    const ParamType contiguous_types[] = {
        ParamType::Means, ParamType::Sh0, ParamType::Scaling, ParamType::Rotation, ParamType::Opacity};
    for (const auto type : contiguous_types) {
        optimizer.get_grad(type).fill_(7.0f);
    }
    auto* shN_state = optimizer.get_state_mutable(ParamType::ShN);
    ASSERT_NE(shN_state, nullptr);
    if (shN_state->is_joint() && shN_state->exp_avg.is_valid() && shN_state->size > 0) {
        optimizer.get_grad(ParamType::ShN);
    }
    optimizer.step(1);

    ASSERT_TRUE(contiguous_joint_row_has_nonzero(*means_state, 1))
        << "seeding failed: sibling row 1 Means (m,v) still decode to zero";

    const auto layout_rest = static_cast<uint32_t>(splat_data.max_sh_coeffs_rest());
    const bool check_shN = shN_state->is_joint() && shN_state->exp_avg.is_valid() &&
                           shN_state->joint_bounds.is_valid() && shN_state->size > 0 &&
                           layout_rest > 0;
    if (check_shN) {
        seed_shN_joint_nonzero(*shN_state, layout_rest);
    }

    optimizer.get_grad(ParamType::Means).fill_(7.0f);

    strategy._refine_weight_max = Tensor::zeros({static_cast<size_t>(splat_data.size())}, Device::CUDA);
    strategy._vis_count = Tensor::zeros({static_cast<size_t>(splat_data.size())}, Device::CUDA);

    const auto split_idx = Tensor::from_vector(std::vector<int>{0}, TensorShape({1}), Device::CUDA).to(DataType::Int64);
    strategy._refine_weight_max.index_put_(split_idx, Tensor::full({1}, 1.0f, Device::CUDA));
    strategy._vis_count.index_put_(split_idx, Tensor::full({1}, 1.0f, Device::CUDA));

    const size_t initial_size = splat_data.size();
    strategy.grow_and_split(1, 0);

    ASSERT_EQ(splat_data.size(), initial_size + 1);
    ASSERT_EQ(means_state->size, initial_size + 1);
    EXPECT_TRUE(means_state->is_joint());
    EXPECT_TRUE(means_state->exp_avg.is_valid());
    EXPECT_TRUE(means_state->joint_bounds.is_valid());

    // fails when grow_and_split only zeroes grads and leaves joint m/v on split parents
    expect_contiguous_joint_row_zero(*optimizer.get_state(ParamType::Means), 0, "Means");
    expect_contiguous_joint_row_zero(*optimizer.get_state(ParamType::Scaling), 0, "Scaling");
    expect_contiguous_joint_row_zero(*optimizer.get_state(ParamType::Rotation), 0, "Rotation");
    expect_contiguous_joint_row_zero(*optimizer.get_state(ParamType::Opacity), 0, "Opacity");
    expect_contiguous_joint_row_zero(*optimizer.get_state(ParamType::Sh0), 0, "Sh0");
    if (check_shN) {
        expect_shN_joint_row_zero(*optimizer.get_state(ParamType::ShN), 0, layout_rest, "ShN");
    }

    const auto grad_cpu = means_state->grad.cpu();
    const float* grad_ptr = grad_cpu.ptr<float>();
    // Parent row 0 and new child have zeroed grads; sibling row 1 keeps prior fill.
    for (int c = 0; c < 3; ++c) {
        EXPECT_FLOAT_EQ(grad_ptr[c], 0.0f);
    }
    for (int c = 0; c < 3; ++c) {
        EXPECT_FLOAT_EQ(grad_ptr[3 + c], 7.0f);
    }
    const size_t child_offset = initial_size * 3;
    for (int c = 0; c < 3; ++c) {
        EXPECT_FLOAT_EQ(grad_ptr[child_offset + c], 0.0f);
    }
}

TEST(MRNFStrategyTest, SHDegree0KeepsShNEmptyAndFusedAdamUsableAfterGrowth) {
    auto splat_data = create_mrnf_test_splat_data(10, 0);
    MRNF strategy(splat_data);

    auto opt_params = vanilla_mrnf_params();
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
        auto opt_params = vanilla_mrnf_params();
        opt_params.iterations = 10'000;
        opt_params.sh_degree_interval = 10'000;
        opt_params.max_cap = static_cast<int>(max_cap);
        return opt_params;
    };

    const auto expect_shN_capacity = [](const SplatData& splat_data,
                                        const AdamOptimizer& optimizer,
                                        const int max_degree) {
        const auto layout_rest = static_cast<uint32_t>(sh_rest_coefficients_for_degree(max_degree));
        // Adam moments always track float4-swizzle cell count (joint packed).
        const size_t expected_moment_logical =
            sh_swizzled_float_count(static_cast<size_t>(n_gaussians), layout_rest);
        const size_t expected_moment_capacity = sh_swizzled_float_count(max_cap, layout_rest);

        ASSERT_TRUE(splat_data.shN().is_valid());
        // Param storage may be q16 (production) or float4-swizzle (if quant forced off).
        if (splat_data.shN_value_quantized()) {
            const size_t expected_u16 =
                lfs::core::sh_value_quant::sh_value_u16_count(
                    static_cast<size_t>(n_gaussians), layout_rest);
            const size_t expected_u16_cap =
                lfs::core::sh_value_quant::sh_value_u16_count(max_cap, layout_rest);
            EXPECT_EQ(splat_data.shN().numel(), expected_u16);
            EXPECT_GE(splat_data.shN().capacity(), expected_u16_cap);
        } else {
            EXPECT_EQ(splat_data.shN().numel(), expected_moment_logical);
            EXPECT_EQ(splat_data.shN().capacity(), expected_moment_capacity);
        }

        const auto* state = optimizer.get_state(ParamType::ShN);
        ASSERT_NE(state, nullptr);
        EXPECT_EQ(state->size, expected_moment_logical);
        EXPECT_EQ(state->capacity, expected_moment_capacity);
        if (layout_rest == 0) {
            EXPECT_FALSE(state->exp_avg.is_valid());
        } else {
            ASSERT_TRUE(state->is_joint());
            ASSERT_TRUE(state->exp_avg.is_valid());
            ASSERT_TRUE(state->joint_bounds.is_valid());
            // Joint packed: 2 bytes/cell for 8-bit SH moments.
            const int bpc = joint_adam::bytes_per_cell(state->joint_bits);
            EXPECT_EQ(bpc, 2);
            EXPECT_EQ(state->exp_avg.numel(), expected_moment_logical * static_cast<size_t>(bpc));
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

    auto opt_params = vanilla_mrnf_params();
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
    auto params = vanilla_mrnf_params();
    params.grow_until_iter = 15000;
    params.sparsify_steps = 15000;
    params.steps_scaler = 0.5f;

    params.apply_step_scaling();

    EXPECT_EQ(params.grow_until_iter, 7500u);
    EXPECT_EQ(params.sparsify_steps, 15000);
    EXPECT_EQ(params.refine_every, 100u);
    EXPECT_EQ(params.stop_refine, 14250u);
}

TEST(MRNFStrategyTest, StopRefineBoundaryRequestsExclusiveMutation) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);

    auto params = vanilla_mrnf_params();
    params.start_refine = 10;
    params.refine_every = 100;
    params.stop_refine = 150; // deliberately off the regular refine cadence
    params.max_cap = 32;
    strategy.initialize(params);

    EXPECT_TRUE(strategy.is_refining(100));
    EXPECT_FALSE(strategy.is_refining(149));
    EXPECT_TRUE(strategy.is_refining(150));
    EXPECT_FALSE(strategy.is_refining(151));
}

TEST(MRNFStrategyTest, GrowAndSplitWithoutMaxCapExtendsBookkeepingMasks) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);

    auto opt_params = vanilla_mrnf_params();
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

TEST(MRNFStrategyTest, DeletedMaskCapacityGrowthPreservesExistingRows) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);

    auto opt_params = vanilla_mrnf_params();
    opt_params.iterations = 10'000;
    opt_params.sh_degree_interval = 10'000;
    opt_params.max_cap = 0;
    opt_params.growth_grad_threshold = 0.5f;
    opt_params.grow_fraction = 1.0f;
    opt_params.grow_until_iter = 10'000;
    strategy.initialize(opt_params);

    const size_t initial_size = splat_data.size();
    splat_data.deleted() = Tensor::zeros_direct(
        TensorShape({initial_size}), initial_size, Device::CUDA, DataType::Bool);
    const auto deleted_index =
        Tensor::from_vector(std::vector<int>{3}, TensorShape({1}), Device::CUDA)
            .to(DataType::Int64);
    splat_data.deleted().index_put_(deleted_index, Tensor::ones_bool({1}, Device::CUDA));

    strategy._refine_weight_max = Tensor::zeros({initial_size}, Device::CUDA);
    strategy._vis_count = Tensor::zeros({initial_size}, Device::CUDA);
    const auto split_index =
        Tensor::from_vector(std::vector<int>{0}, TensorShape({1}), Device::CUDA)
            .to(DataType::Int64);
    strategy._refine_weight_max.index_put_(split_index, Tensor::full({1}, 1.0f, Device::CUDA));
    strategy._vis_count.index_put_(split_index, Tensor::full({1}, 1.0f, Device::CUDA));

    strategy.grow_and_split(1, 0);

    ASSERT_EQ(splat_data.size(), initial_size + 1);
    const auto deleted_cpu = splat_data.deleted().cpu();
    const bool* values = deleted_cpu.ptr<bool>();
    EXPECT_TRUE(values[3]);
    EXPECT_FALSE(values[initial_size]);
}

TEST(MRNFStrategyTest, GrowAndSplitReplacementSkipsZeroWeightCandidates) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);

    auto opt_params = vanilla_mrnf_params();
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

    auto opt_params = vanilla_mrnf_params();
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

    auto opt_params = vanilla_mrnf_params();
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

    auto opt_params = vanilla_mrnf_params();
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

    auto opt_params = vanilla_mrnf_params();
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

    auto opt_params = vanilla_mrnf_params();
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

namespace {
    Camera make_explore_camera(int width, int height) {
        std::vector<float> R_data = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        std::vector<float> T_data = {0, 0, 4};
        auto R = Tensor::from_vector(R_data, TensorShape({3, 3}), Device::CPU).cuda();
        auto T = Tensor::from_vector(T_data, TensorShape({3}), Device::CPU).cuda();
        return Camera(R, T, 100.f, 100.f, width * 0.5f, height * 0.5f,
                      Tensor(), Tensor(), CameraModelType::PINHOLE, "test", "",
                      std::filesystem::path{}, width, height, 0);
    }

    std::shared_ptr<Camera> make_hull_camera(float x, float y, float z, int uid) {
        std::vector<float> R_data = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        std::vector<float> T_data = {-x, -y, -z};
        auto R = Tensor::from_vector(R_data, TensorShape({3, 3}), Device::CPU).cuda();
        auto T = Tensor::from_vector(T_data, TensorShape({3}), Device::CPU).cuda();
        return std::make_shared<Camera>(
            R, T, 100.f, 100.f, 4.f, 4.f, Tensor(), Tensor(),
            CameraModelType::PINHOLE, "hull", "", std::filesystem::path{}, 8, 8, uid);
    }

    std::shared_ptr<CameraDataset> make_hull_dataset() {
        std::vector<std::shared_ptr<Camera>> cams;
        cams.push_back(make_hull_camera(-1.0f, 0.0f, 0.0f, 0));
        cams.push_back(make_hull_camera(1.0f, 0.0f, 0.0f, 1));
        DatasetConfig cfg;
        cfg.test_every = 1000;
        return std::make_shared<CameraDataset>(std::move(cams), cfg, CameraDataset::Split::ALL);
    }

    // Place a row beyond the 8x-orbit census radius so the scene-level far
    // gate stays active; the row is still far for the 2x runtime mask.
    void place_deep_far_probe_at(SplatData& splat_data, const int row) {
        const int n = static_cast<int>(splat_data.size());
        std::vector<float> means(static_cast<size_t>(n) * 3, 0.0f);
        for (int i = 0; i < n; ++i) {
            means[i * 3 + 0] = static_cast<float>(i);
        }
        means[row * 3 + 0] = 8.5f;
        auto fixed = Tensor::from_vector(means, TensorShape({static_cast<size_t>(n), 3}), Device::CUDA);
        splat_data.means().copy_(fixed);
    }

    void place_deep_far_probe(SplatData& splat_data) {
        std::vector<float> means(8 * 3, 0.0f);
        for (int i = 0; i < 8; ++i) {
            means[i * 3 + 0] = static_cast<float>(i);
        }
        means[7 * 3 + 0] = 8.5f;
        auto fixed = Tensor::from_vector(means, TensorShape({8, 3}), Device::CUDA);
        splat_data.means().copy_(fixed);
    }

    void install_test_camera_hull(MRNF& strategy) {
        strategy.set_training_dataset(make_hull_dataset());
    }

    bool is_far_of_test_hull(float x, float y, float z) {
        const float dist = std::sqrt(x * x + y * y + z * z);
        return dist > 2.0f;
    }

    RenderOutput make_explore_render(Camera& camera, int width, int height, float target_value = 1.0f) {
        RenderOutput out;
        out.camera = &camera;
        out.width = width;
        out.height = height;
        out.image = Tensor::zeros({3, static_cast<size_t>(height), static_cast<size_t>(width)}, Device::CUDA);
        out.target_image = Tensor::full(
            {3, static_cast<size_t>(height), static_cast<size_t>(width)}, target_value, Device::CUDA);
        out.alpha = Tensor::full({1, static_cast<size_t>(height), static_cast<size_t>(width)}, 0.2f, Device::CUDA);
        out.depth = Tensor::full({1, static_cast<size_t>(height), static_cast<size_t>(width)}, 2.0f, Device::CUDA);
        return out;
    }

} // namespace

TEST(MRNFStrategyTest, ExploreSplitsAreDisjointAndRespectMaxCap) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);

    auto opt_params = vanilla_mrnf_params();
    opt_params.iterations = 10'000;
    opt_params.sh_degree_interval = 10'000;
    opt_params.max_cap = 12;
    opt_params.growth_grad_threshold = 0.5f;
    opt_params.grow_fraction = 0.0f;
    opt_params.grow_until_iter = 10'000;
    opt_params.refine_every = 100;
    opt_params.explore_splits = 2;
    opt_params.far_growth_cap = 1.0f;
    strategy.initialize(opt_params);
    strategy._far_starvation = 1.0f;

    const auto free_indices =
        Tensor::from_vector(std::vector<int>{8, 9}, TensorShape({2}), Device::CUDA).to(DataType::Int64);
    strategy.mark_as_free(free_indices);
    splat_data.deleted().index_put_(free_indices, Tensor::ones_bool({2}, Device::CUDA));

    const size_t n = splat_data.size();
    strategy._refine_weight_max = Tensor::zeros({n}, Device::CUDA);
    strategy._vis_count = Tensor::zeros({n}, Device::CUDA);
    strategy._vis_count.index_put_(
        Tensor::from_vector(std::vector<int>{0}, TensorShape({1}), Device::CUDA).to(DataType::Int64),
        Tensor::full({1}, 1.0f, Device::CUDA));

    strategy._explore_score_sum = Tensor::zeros({n}, Device::CUDA);
    strategy._explore_score_sum.index_put_(
        Tensor::from_vector(std::vector<int>{0, 2, 3}, TensorShape({3}), Device::CUDA).to(DataType::Int64),
        Tensor::full({3}, 4.0f, Device::CUDA));
    strategy._explore_sample_count = 1;

    const auto scales_before = splat_data.scaling_raw().cpu();
    const size_t active_before = strategy.active_count();
    strategy.grow_and_split(100, 2);

    EXPECT_LE(strategy.active_count(), static_cast<size_t>(opt_params.max_cap));
    EXPECT_GE(strategy.active_count(), active_before);

    const auto scales_after = splat_data.scaling_raw().cpu();
    const float* before = scales_before.ptr<float>();
    const float* after = scales_after.ptr<float>();
    const bool row0_split = std::abs(after[0] - before[0]) > 1e-4f;
    const bool row2_split = std::abs(after[6] - before[6]) > 1e-4f;
    const bool row3_split = std::abs(after[9] - before[9]) > 1e-4f;
    EXPECT_TRUE(row0_split);
    EXPECT_TRUE(row2_split);
    EXPECT_TRUE(row3_split);
}

TEST(MRNFStrategyTest, FarGrowthCapConstrainsOutsideAllocations) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);

    auto opt_params = vanilla_mrnf_params();
    opt_params.iterations = 10'000;
    opt_params.sh_degree_interval = 10'000;
    opt_params.max_cap = 32;
    opt_params.growth_grad_threshold = 0.5f;
    opt_params.grow_fraction = 0.0f;
    opt_params.grow_until_iter = 10'000;
    opt_params.refine_every = 100;
    opt_params.explore_splits = 5;
    opt_params.far_growth_cap = 0.0f;
    strategy.initialize(opt_params);
    install_test_camera_hull(strategy);
    ASSERT_TRUE(strategy._camera_hull_valid);
    strategy._far_starvation = 1.0f;
    strategy.refresh_far_field_mask(static_cast<size_t>(splat_data.size()));

    const size_t n = splat_data.size();
    strategy._refine_weight_max = Tensor::zeros({n}, Device::CUDA);
    strategy._vis_count = Tensor::zeros({n}, Device::CUDA);
    strategy._explore_score_sum = Tensor::ones({n}, Device::CUDA);
    strategy._explore_sample_count = 1;

    strategy.grow_and_split(100, 0);

    const auto means = splat_data.means().cpu();
    const float* m = means.ptr<float>();
    int outside_children = 0;
    for (size_t i = n; i < splat_data.size(); ++i) {
        if (is_far_of_test_hull(m[i * 3], m[i * 3 + 1], m[i * 3 + 2])) {
            ++outside_children;
        }
    }
    EXPECT_EQ(outside_children, 0);
    EXPECT_LE(splat_data.size(), n + 3u);

    auto splat_unconstrained = create_mrnf_test_splat_data();
    MRNF unconstrained(splat_unconstrained);
    opt_params.far_growth_cap = 1.0f;
    unconstrained.initialize(opt_params);
    install_test_camera_hull(unconstrained);
    unconstrained._far_starvation = 1.0f;
    unconstrained.refresh_far_field_mask(n);
    unconstrained._refine_weight_max = Tensor::zeros({n}, Device::CUDA);
    unconstrained._vis_count = Tensor::zeros({n}, Device::CUDA);
    unconstrained._explore_score_sum = Tensor::ones({n}, Device::CUDA);
    unconstrained._explore_sample_count = 1;
    unconstrained.grow_and_split(100, 0);
    EXPECT_EQ(splat_unconstrained.size(), n + 5u);
}

TEST(MRNFStrategyTest, SeedFromViewInsertsRequestedRows) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);

    auto opt_params = vanilla_mrnf_params();
    opt_params.iterations = 10'000;
    opt_params.sh_degree_interval = 10'000;
    opt_params.max_cap = 32;
    opt_params.grow_until_iter = 10'000;
    opt_params.refine_every = 100;
    opt_params.explore_seeds = 2;
    opt_params.seed_opacity = 0.03f;
    strategy.initialize(opt_params);
    install_test_camera_hull(strategy);
    strategy._bounds.center[0] = 0.0f;
    strategy._bounds.center[1] = 0.0f;
    strategy._bounds.center[2] = 0.0f;
    strategy._bounds.extent[0] = 2.5f;
    strategy._bounds.extent[1] = 2.5f;
    strategy._bounds.extent[2] = 2.5f;
    strategy._bounds.median_size = 5.0f;
    strategy._bounds.max_extent = 2.5f;
    strategy._bounds_valid = true;
    ASSERT_TRUE(strategy._camera_hull_valid);
    strategy._far_starvation = 1.0f;
    strategy.refresh_far_field_mask(static_cast<size_t>(splat_data.size()));

    const auto free_indices =
        Tensor::from_vector(std::vector<int>{8, 9}, TensorShape({2}), Device::CUDA).to(DataType::Int64);
    strategy.mark_as_free(free_indices);
    splat_data.deleted().index_put_(free_indices, Tensor::ones_bool({2}, Device::CUDA));

    const size_t active_before = strategy.active_count();
    RenderOutput invalid;
    strategy.seed_from_view(100, invalid);
    EXPECT_EQ(strategy.active_count(), active_before);

    auto camera = make_explore_camera(8, 8);
    auto render = make_explore_render(camera, 8, 8, 0.8f);
    strategy.seed_from_view(100, render);
    EXPECT_EQ(strategy.active_count(), active_before + 2);

    const auto opac = splat_data.opacity_raw().cpu();
    const auto scales = splat_data.scaling_raw().cpu();
    const auto sh0 = splat_data.sh0().cpu();
    const float expected_logit = std::log(0.03f / 0.97f);
    bool found_seed = false;
    const float* o = opac.ptr<float>();
    const float* s = scales.ptr<float>();
    const float* c = sh0.ptr<float>();
    const size_t rows = splat_data.size();
    for (size_t i = 0; i < rows; ++i) {
        if (std::abs(o[i] - expected_logit) < 1e-4f) {
            found_seed = true;
            EXPECT_NEAR(s[i * 3 + 0], s[i * 3 + 1], 1e-5f);
            EXPECT_NEAR(s[i * 3 + 1], s[i * 3 + 2], 1e-5f);
            EXPECT_NEAR(c[i * 3 + 0], (0.8f - 0.5f) / 0.28209479177387814f, 1e-4f);
        }
    }
    EXPECT_TRUE(found_seed);
}

TEST(MRNFStrategyTest, ExploreScoreIgnoresNonPositiveRadii) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);

    auto opt_params = vanilla_mrnf_params();
    opt_params.iterations = 10'000;
    opt_params.sh_degree_interval = 10'000;
    opt_params.max_cap = 32;
    opt_params.start_refine = 0;
    opt_params.stop_refine = 10'000;
    opt_params.refine_every = 100;
    opt_params.explore_splits = 4;
    strategy.initialize(opt_params);

    auto camera = make_explore_camera(4, 4);
    auto render = make_explore_render(camera, 4, 4, 1.0f);
    const size_t n = splat_data.size();
    std::vector<float> means2d(n * 2, 1.5f);
    std::vector<float> radii(n, 2.0f);
    radii[1] = 0.0f;
    radii[4] = -1.0f;
    render.means2d = Tensor::from_vector(means2d, TensorShape({n, 2}), Device::CUDA);
    render.radii = Tensor::from_vector(radii, TensorShape({n}), Device::CUDA);

    strategy.accumulate_explore_sample(100, render);
    ASSERT_TRUE(strategy._explore_score_sum.is_valid());
    ASSERT_EQ(strategy._explore_score_sum.numel(), n);
    const auto scores = strategy._explore_score_sum.cpu();
    const float* p = scores.ptr<float>();
    EXPECT_FLOAT_EQ(p[1], 0.0f);
    EXPECT_FLOAT_EQ(p[4], 0.0f);
    EXPECT_GT(p[0], 0.0f);
}

TEST(MRNFStrategyTest, FarFieldClassificationStableAfterFarInserts) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);

    auto opt_params = vanilla_mrnf_params();
    opt_params.iterations = 10'000;
    opt_params.sh_degree_interval = 10'000;
    opt_params.max_cap = 32;
    opt_params.grow_until_iter = 10'000;
    opt_params.refine_every = 100;
    opt_params.explore_seeds = 2;
    opt_params.seed_opacity = 0.03f;
    opt_params.far_growth_cap = 1.0f;
    strategy.initialize(opt_params);
    install_test_camera_hull(strategy);
    strategy._bounds.center[0] = 0.0f;
    strategy._bounds.center[1] = 0.0f;
    strategy._bounds.center[2] = 0.0f;
    strategy._bounds.extent[0] = 2.5f;
    strategy._bounds.extent[1] = 2.5f;
    strategy._bounds.extent[2] = 2.5f;
    strategy._bounds.median_size = 5.0f;
    strategy._bounds.max_extent = 2.5f;
    strategy._bounds_valid = true;
    ASSERT_TRUE(strategy._camera_hull_valid);
    strategy._far_starvation = 1.0f;
    strategy.refresh_far_field_mask(static_cast<size_t>(splat_data.size()));

    const size_t n = splat_data.size();
    strategy.begin_far_growth_window(n, 0);
    ASSERT_TRUE(strategy._far_growth.outside_mask.is_valid());
    ASSERT_EQ(strategy._far_growth.outside_mask.numel(), n);
    const auto before = strategy._far_growth.outside_mask.cpu();

    auto camera = make_explore_camera(8, 8);
    auto render = make_explore_render(camera, 8, 8, 0.8f);
    strategy.seed_from_view(100, render);
    ASSERT_GT(splat_data.size(), n);

    strategy.begin_far_growth_window(splat_data.size(), 0);
    ASSERT_TRUE(strategy._far_growth.outside_mask.is_valid());
    ASSERT_GE(strategy._far_growth.outside_mask.numel(), n);
    const auto after = strategy._far_growth.outside_mask.cpu();
    const auto* b = before.ptr<bool>();
    const auto* a = after.ptr<bool>();
    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(a[i], b[i]) << "row " << i;
    }
}

TEST(MRNFStrategyTest, FarDecayScaleAppliesOnlyToFarUnfrozenRows) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);

    auto opt_params = vanilla_mrnf_params();
    opt_params.iterations = 1'000;
    opt_params.sh_degree_interval = 10'000;
    opt_params.max_cap = 32;
    opt_params.far_decay_scale = 1.0f;
    strategy.initialize(opt_params);
    install_test_camera_hull(strategy);
    ASSERT_TRUE(strategy._camera_hull_valid);
    splat_data.set_frozen_ranges({SplatData::FrozenRange{.start = 0, .count = 1}});

    const auto opac_orig = splat_data.opacity_raw().clone();
    const auto scale_orig = splat_data.scaling_raw().clone();
    const auto means = splat_data.means().cpu();
    const float* m = means.ptr<float>();

    const auto expected_raw = [](const float raw, const float decay, const float train_t) {
        const float opac = 1.0f / (1.0f + std::exp(-raw));
        float next = opac - decay * (1.0f - train_t);
        next = std::min(std::max(next, 1e-12f), 1.0f - 1e-12f);
        return std::log(next / (1.0f - next));
    };
    const auto expected_log_s = [](const float log_s, const float decay, const float train_t) {
        const float scale = std::exp(log_s) * (1.0f - decay * (1.0f - train_t));
        return std::log(std::max(scale, 1e-12f));
    };

    strategy.apply_decay(500);
    {
        const auto opac = splat_data.opacity_raw().cpu();
        const auto scales = splat_data.scaling_raw().cpu();
        const float* o = opac.ptr<float>();
        const float* s = scales.ptr<float>();
        const float* o0 = opac_orig.cpu().ptr<float>();
        const float* s0 = scale_orig.cpu().ptr<float>();
        EXPECT_NEAR(o[0], o0[0], 1e-6f);
        EXPECT_NEAR(s[0], s0[0], 1e-6f);
        for (size_t i = 1; i < splat_data.size(); ++i) {
            EXPECT_NEAR(o[i], expected_raw(o0[i], opt_params.opacity_decay, 0.5f), 1e-5f) << i;
            EXPECT_NEAR(s[i * 3], expected_log_s(s0[i * 3], opt_params.scale_decay, 0.5f), 1e-5f) << i;
        }
    }

    splat_data.opacity_raw().copy_(opac_orig);
    splat_data.scaling_raw().copy_(scale_orig);
    opt_params.far_decay_scale = 0.25f;
    strategy.set_optimization_params(opt_params);
    strategy._far_starvation = 1.0f;
    strategy.refresh_far_field_mask(static_cast<size_t>(splat_data.size()));
    strategy.apply_decay(500);

    const auto opac = splat_data.opacity_raw().cpu();
    const auto scales = splat_data.scaling_raw().cpu();
    const float* o = opac.ptr<float>();
    const float* s = scales.ptr<float>();
    const float* o0 = opac_orig.cpu().ptr<float>();
    const float* s0 = scale_orig.cpu().ptr<float>();
    EXPECT_NEAR(o[0], o0[0], 1e-6f);
    EXPECT_NEAR(s[0], s0[0], 1e-6f);
    for (size_t i = 1; i < splat_data.size(); ++i) {
        const bool far = is_far_of_test_hull(m[i * 3], m[i * 3 + 1], m[i * 3 + 2]);
        const float opac_decay = opt_params.opacity_decay * (far ? 0.25f : 1.0f);
        const float scale_decay = opt_params.scale_decay * (far ? 0.25f : 1.0f);
        EXPECT_NEAR(o[i], expected_raw(o0[i], opac_decay, 0.5f), 1e-5f) << "row " << i;
        EXPECT_NEAR(s[i * 3], expected_log_s(s0[i * 3], scale_decay, 0.5f), 1e-5f) << "row " << i;
        if (far) {
            EXPECT_GT(std::abs(expected_raw(o0[i], opt_params.opacity_decay, 0.5f) - o[i]), 1e-6f) << i;
        }
    }
}

TEST(MRNFStrategyTest, SumModeFoldIsBitIdentical) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);
    auto opt_params = vanilla_mrnf_params();
    opt_params.iterations = 1'000;
    opt_params.max_cap = 32;
    strategy.initialize(opt_params);
    strategy.ensure_densification_info_shape();
    ASSERT_TRUE(splat_data._densification_info.is_valid());
    EXPECT_EQ(splat_data._densification_info.shape()[0], 2u);
    EXPECT_EQ(splat_data._densification_info.shape()[1], splat_data.size());
    EXPECT_FALSE(strategy.uses_max_growth_score());
    EXPECT_FALSE(strategy.uses_always_growth());
}

TEST(MRNFStrategyTest, MaxModeFoldUsesViewMaxRow) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);
    auto opt_params = vanilla_mrnf_params();
    opt_params.iterations = 1'000;
    opt_params.max_cap = 32;
    opt_params.growth_score_mode = param::GrowthScoreMode::Max;
    strategy.initialize(opt_params);
    ASSERT_TRUE(splat_data._densification_info.is_valid());
    EXPECT_EQ(splat_data._densification_info.shape()[0], 3u);
    EXPECT_TRUE(strategy.uses_max_growth_score());

    const size_t n = splat_data.size();
    strategy._refine_weight_max = Tensor::zeros({n}, Device::CUDA);
    strategy._vis_count = Tensor::zeros({n}, Device::CUDA);
    std::vector<float> flat(n * 3, 0.f);
    flat[0] = 2.0f;
    flat[n] = 0.2f;
    flat[2 * n] = 1.7f;
    splat_data._densification_info = Tensor::from_vector(flat, {size_t{3}, n}, Device::CUDA);
    mrnf_strategy::launch_fold_densification_and_zero(
        strategy._vis_count.ptr<float>(),
        strategy._refine_weight_max.ptr<float>(),
        splat_data._densification_info.ptr<float>(),
        n,
        nullptr,
        3,
        true);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_FLOAT_EQ(strategy._vis_count.cpu().ptr<float>()[0], 2.0f);
    EXPECT_FLOAT_EQ(strategy._refine_weight_max.cpu().ptr<float>()[0], 1.7f);
}

TEST(MRNFStrategyTest, FarUnseenCullIncrementsOnlyFarInvisible) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);
    auto opt_params = vanilla_mrnf_params();
    opt_params.iterations = 1'000;
    opt_params.max_cap = 32;
    opt_params.far_unseen_cull_windows = 2;
    strategy.initialize(opt_params);
    install_test_camera_hull(strategy);
    ASSERT_TRUE(strategy._camera_hull_valid);
    splat_data.set_frozen_ranges({SplatData::FrozenRange{.start = 9, .count = 1}});

    const size_t n = splat_data.size();
    strategy._vis_count = Tensor::zeros({n}, Device::CUDA);
    std::vector<float> vis(n, 0.f);
    vis[3] = 1.0f;
    vis[1] = 0.0f;
    strategy._vis_count = Tensor::from_vector(vis, {n}, Device::CUDA);

    auto prune = Tensor::zeros_bool({n}, Device::CUDA);
    strategy.apply_far_unseen_cull(prune);
    ASSERT_TRUE(strategy._far_unseen_windows.is_valid());
    const auto c1 = strategy._far_unseen_windows.cpu().to_vector_uint8();
    const auto p1 = prune.cpu();
    const auto means = splat_data.means().cpu();
    const float* m = means.ptr<float>();
    const bool* pruned = p1.ptr<bool>();
    for (size_t i = 0; i < n; ++i) {
        const bool far = is_far_of_test_hull(m[i * 3], m[i * 3 + 1], m[i * 3 + 2]);
        if (i == 9) {
            EXPECT_EQ(c1[i], 0) << "frozen";
            EXPECT_FALSE(pruned[i]);
            continue;
        }
        if (far && vis[i] == 0.f) {
            EXPECT_EQ(c1[i], 1) << i;
            EXPECT_FALSE(pruned[i]);
        } else {
            EXPECT_EQ(c1[i], 0) << i;
            EXPECT_FALSE(pruned[i]);
        }
    }

    strategy.apply_far_unseen_cull(prune);
    const auto c2 = strategy._far_unseen_windows.cpu().to_vector_uint8();
    const auto p2 = prune.cpu();
    const bool* pruned2 = p2.ptr<bool>();
    for (size_t i = 0; i < n; ++i) {
        const bool far = is_far_of_test_hull(m[i * 3], m[i * 3 + 1], m[i * 3 + 2]);
        if (i == 9) {
            EXPECT_FALSE(pruned2[i]);
            continue;
        }
        if (far && vis[i] == 0.f) {
            EXPECT_EQ(c2[i], 2) << i;
            EXPECT_TRUE(pruned2[i]) << i;
        } else {
            EXPECT_EQ(c2[i], 0) << i;
            EXPECT_FALSE(pruned2[i]) << i;
        }
    }

    vis[5] = 1.0f;
    strategy._vis_count = Tensor::from_vector(vis, {n}, Device::CUDA);
    prune = Tensor::zeros_bool({n}, Device::CUDA);
    strategy.apply_far_unseen_cull(prune);
    const auto c3 = strategy._far_unseen_windows.cpu().to_vector_uint8();
    EXPECT_EQ(c3[5], 0);
}

TEST(MRNFStrategyTest, AlwaysModeGrowthAddsTargetAndIgnoresGates) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);
    auto opt_params = vanilla_mrnf_params();
    opt_params.iterations = 10'000;
    opt_params.sh_degree_interval = 10'000;
    opt_params.max_cap = 32;
    opt_params.growth_grad_threshold = 10.0f;
    opt_params.grow_fraction = 0.0f;
    opt_params.grow_until_iter = 10'000;
    opt_params.growth_mode = param::GrowthMode::Always;
    opt_params.growth_target_factor = 1.5f;
    opt_params.far_growth_cap = 1.0f;
    opt_params.use_edge_map = false;
    strategy.initialize(opt_params);

    const size_t n = splat_data.size();
    const size_t active = strategy.active_count();
    strategy._refine_weight_max = Tensor::zeros({n}, Device::CUDA);
    strategy._vis_count = Tensor::zeros({n}, Device::CUDA);
    strategy.grow_and_split(100, 0);
    const int expected = static_cast<int>(std::round(0.5f * static_cast<float>(active)));
    EXPECT_EQ(static_cast<int>(strategy.active_count()), static_cast<int>(active) + expected);
}

TEST(MRNFStrategyTest, ThresholdModeMatchesCurrentSelection) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);
    auto opt_params = vanilla_mrnf_params();
    opt_params.iterations = 10'000;
    opt_params.sh_degree_interval = 10'000;
    opt_params.max_cap = 32;
    opt_params.growth_grad_threshold = 10.0f;
    opt_params.grow_fraction = 1.0f;
    opt_params.grow_until_iter = 10'000;
    opt_params.growth_mode = param::GrowthMode::Threshold;
    opt_params.far_growth_cap = 1.0f;
    strategy.initialize(opt_params);

    const size_t n = splat_data.size();
    const size_t active = strategy.active_count();
    strategy._refine_weight_max = Tensor::ones({n}, Device::CUDA);
    strategy._vis_count = Tensor::zeros({n}, Device::CUDA);
    strategy.grow_and_split(100, 0);
    EXPECT_EQ(strategy.active_count(), active);
}

TEST(MRNFStrategyTest, ScreenClipScalesOnlyOverLimitRows) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);
    auto opt_params = vanilla_mrnf_params();
    opt_params.iterations = 1'000;
    opt_params.max_cap = 32;
    opt_params.max_screen_clip_frac = 0.3f;
    strategy.initialize(opt_params);
    splat_data.set_frozen_ranges({SplatData::FrozenRange{.start = 0, .count = 1}});

    const size_t n = splat_data.size();
    splat_data.scaling_raw().zero_();
    const auto scales_before = splat_data.scaling_raw().clone();

    RenderOutput empty;
    opt_params.max_screen_clip_frac = 0.0f;
    strategy.set_optimization_params(opt_params);
    strategy.apply_screen_size_clip(empty);
    {
        const auto after = splat_data.scaling_raw().cpu();
        const auto before = scales_before.cpu();
        for (size_t i = 0; i < after.numel(); ++i) {
            EXPECT_FLOAT_EQ(after.ptr<float>()[i], before.ptr<float>()[i]);
        }
    }

    opt_params.max_screen_clip_frac = 0.3f;
    strategy.set_optimization_params(opt_params);
    auto camera = make_explore_camera(100, 100);
    RenderOutput render;
    render.camera = &camera;
    render.width = 100;
    render.height = 100;
    std::vector<float> radii(n, 10.0f);
    radii[0] = 90.0f;
    radii[2] = 60.0f;
    render.radii = Tensor::from_vector(radii, {n}, Device::CUDA);
    strategy.apply_screen_size_clip(render);

    const auto after = splat_data.scaling_raw().cpu();
    const float* s = after.ptr<float>();
    EXPECT_NEAR(s[0], 0.0f, 1e-6f);
    EXPECT_NEAR(s[1 * 3], 0.0f, 1e-6f);
    EXPECT_NEAR(s[2 * 3], std::log(30.0f / 60.0f), 1e-5f);
    EXPECT_NEAR(s[2 * 3 + 1], std::log(30.0f / 60.0f), 1e-5f);
    EXPECT_NEAR(s[2 * 3 + 2], std::log(30.0f / 60.0f), 1e-5f);
}

namespace {
    void write_isotropic_log_scales(SplatData& splat, const std::vector<float>& log_s) {
        const size_t n = static_cast<size_t>(splat.size());
        ASSERT_EQ(log_s.size(), n);
        std::vector<float> packed(n * 3);
        for (size_t i = 0; i < n; ++i) {
            packed[i * 3 + 0] = log_s[i];
            packed[i * 3 + 1] = log_s[i];
            packed[i * 3 + 2] = log_s[i];
        }
        splat.scaling_raw() = Tensor::from_vector(packed, TensorShape({n, 3}), Device::CUDA);
    }

    std::vector<float> means_xyz(const SplatData& splat) {
        const auto cpu = splat.means().cpu();
        const float* p = cpu.ptr<float>();
        return {p, p + cpu.numel()};
    }

    param::OptimizationParameters mean_step_test_params(const param::MeanStepMode mode) {
        auto opt = vanilla_mrnf_params();
        opt.iterations = 10'000;
        opt.sh_degree_interval = 10'000;
        opt.max_cap = 32;
        opt.means_lr = 0.1f;
        opt.means_lr_end = 0.1f;
        opt.mean_step_mode = mode;
        return opt;
    }
} // namespace

TEST(MRNFStrategyTest, GlobalModeMeanStepIsBitIdenticalAcrossCodepaths) {
    auto splat_g = create_mrnf_test_splat_data(8);
    place_deep_far_probe(splat_g);
    auto splat_p = create_mrnf_test_splat_data(8);
    place_deep_far_probe(splat_p);
    auto splat_g2 = create_mrnf_test_splat_data(8);
    place_deep_far_probe(splat_g2);

    MRNF global_a(splat_g);
    MRNF per_splat(splat_p);
    MRNF global_b(splat_g2);
    global_a.initialize(mean_step_test_params(param::MeanStepMode::Global));
    per_splat.initialize(mean_step_test_params(param::MeanStepMode::PerSplat));
    global_b.initialize(mean_step_test_params(param::MeanStepMode::Global));
    install_test_camera_hull(per_splat);
    per_splat._far_starvation = 1.0f;
    per_splat.refresh_far_field_mask(static_cast<size_t>(splat_p.size()));
    per_splat.sync_mean_learning_rate();
    ASSERT_NE(per_splat.get_optimizer().mean_step_far_mask(), nullptr);

    ASSERT_TRUE(global_a._bounds_valid);
    ASSERT_TRUE(per_splat._bounds_valid);
    ASSERT_TRUE(per_splat._median_splat_extent_valid);
    const float scene_median = global_a._bounds.median_size;
    const float splat_median = per_splat._median_splat_extent;
    ASSERT_GT(scene_median, 0.0f);
    ASSERT_GT(splat_median, 0.0f);

    write_isotropic_log_scales(splat_g, {std::log(0.1f * scene_median), std::log(2.0f * scene_median),
                                         std::log(0.5f * scene_median), std::log(1.5f * scene_median),
                                         std::log(0.2f * scene_median), std::log(3.0f * scene_median),
                                         std::log(0.8f * scene_median), std::log(1.1f * scene_median)});
    write_isotropic_log_scales(splat_g2, {std::log(3.0f * scene_median), std::log(0.05f * scene_median),
                                          std::log(2.5f * scene_median), std::log(0.2f * scene_median),
                                          std::log(1.8f * scene_median), std::log(0.4f * scene_median),
                                          std::log(2.2f * scene_median), std::log(0.7f * scene_median)});
    write_isotropic_log_scales(splat_p, {std::log(0.01f * splat_median), std::log(0.1f * splat_median),
                                         std::log(0.5f * splat_median), std::log(splat_median),
                                         std::log(0.25f * splat_median), std::log(0.75f * splat_median),
                                         std::log(splat_median), std::log(1.0e-3f * splat_median)});

    const auto means0 = means_xyz(splat_g);
    ASSERT_EQ(means0, means_xyz(splat_p));
    ASSERT_EQ(means0, means_xyz(splat_g2));

    global_a.get_optimizer().get_grad(ParamType::Means).fill_(0.25f);
    per_splat.get_optimizer().get_grad(ParamType::Means).fill_(0.25f);
    global_b.get_optimizer().get_grad(ParamType::Means).fill_(0.25f);
    global_a.get_optimizer().step(1);
    per_splat.get_optimizer().step(1);
    global_b.get_optimizer().step(1);

    const auto after_g = means_xyz(splat_g);
    const auto after_p = means_xyz(splat_p);
    const auto after_g2 = means_xyz(splat_g2);
    ASSERT_EQ(after_g.size(), after_p.size());
    for (size_t i = 0; i < after_g.size(); ++i) {
        EXPECT_FLOAT_EQ(after_g[i], after_p[i]) << "sub-median/median identity axis " << i;
        EXPECT_FLOAT_EQ(after_g[i], after_g2[i]) << "global scale-independence axis " << i;
    }
    EXPECT_FALSE(global_a.get_optimizer().per_splat_mean_step());
    EXPECT_TRUE(per_splat.get_optimizer().per_splat_mean_step());
}

TEST(MRNFStrategyTest, PerSplatMeanStepScalesWithExtentAndClamps) {
    auto splat_p = create_mrnf_test_splat_data(8);
    place_deep_far_probe(splat_p);
    auto splat_g = create_mrnf_test_splat_data(8);
    place_deep_far_probe(splat_g);
    MRNF per_splat(splat_p);
    MRNF global(splat_g);
    per_splat.initialize(mean_step_test_params(param::MeanStepMode::PerSplat));
    global.initialize(mean_step_test_params(param::MeanStepMode::Global));
    install_test_camera_hull(per_splat);
    per_splat._far_starvation = 1.0f;
    per_splat.refresh_far_field_mask(static_cast<size_t>(splat_p.size()));
    per_splat.sync_mean_learning_rate();
    ASSERT_TRUE(per_splat._bounds_valid);
    ASSERT_TRUE(per_splat._median_splat_extent_valid);
    const float med_e = per_splat._median_splat_extent;
    ASSERT_GT(med_e, 0.0f);
    const float r_min = 1.0f;
    const float r_max = 300.0f;
    ASSERT_NE(per_splat.get_optimizer().mean_step_far_mask(), nullptr);
    ASSERT_EQ(per_splat.get_optimizer().mean_step_far_mask_n(), 8);
    ASSERT_TRUE(per_splat._far_field_mask.is_valid());
    const auto mask_i = per_splat._far_field_mask.to(DataType::Int32).to_vector_int();
    ASSERT_EQ(mask_i.size(), 8u);
    EXPECT_EQ(mask_i[1], 0);
    EXPECT_EQ(mask_i[3], 1);

    splat_p.set_frozen_ranges({SplatData::FrozenRange{.start = 0, .count = 1}});
    splat_g.set_frozen_ranges({SplatData::FrozenRange{.start = 0, .count = 1}});
    apply_frozen_ranges_to_optimizer(splat_p, per_splat.get_optimizer());
    apply_frozen_ranges_to_optimizer(splat_g, global.get_optimizer());

    const float s_small = 0.5f * med_e;
    const float s_large = 2.0f * med_e;
    const std::vector<float> logs{
        std::log(s_large),
        std::log(s_large),
        std::log(s_small),
        std::log(s_large),
        std::log(1.0e-8f * med_e),
        std::log(1.0e4f * med_e),
        std::log(r_min * med_e),
        std::log(r_max * med_e),
    };
    write_isotropic_log_scales(splat_p, logs);
    write_isotropic_log_scales(splat_g, logs);

    const auto before_p = means_xyz(splat_p);
    const auto before_g = means_xyz(splat_g);
    ASSERT_EQ(before_p, before_g);
    per_splat.get_optimizer().get_grad(ParamType::Means).fill_(0.2f);
    global.get_optimizer().get_grad(ParamType::Means).fill_(0.2f);
    per_splat.get_optimizer().step(1);
    global.get_optimizer().step(1);
    const auto after_p = means_xyz(splat_p);
    const auto after_g = means_xyz(splat_g);

    auto dx = [](const std::vector<float>& after, const std::vector<float>& before, const size_t row) {
        return after[row * 3] - before[row * 3];
    };
    EXPECT_FLOAT_EQ(dx(after_p, before_p, 0), 0.0f);
    EXPECT_FLOAT_EQ(dx(after_g, before_g, 0), 0.0f);
    for (const size_t row : {size_t{1}, size_t{2}, size_t{4}, size_t{6}}) {
        for (size_t ax = 0; ax < 3; ++ax) {
            EXPECT_FLOAT_EQ(after_p[row * 3 + ax], after_g[row * 3 + ax])
                << "unmasked-or-sub-median row " << row << " axis " << ax;
        }
    }

    const float dx_near_large = dx(after_p, before_p, 1);
    const float dx_small = dx(after_p, before_p, 2);
    const float dx_far_large = dx(after_p, before_p, 3);
    const float dx_tiny = dx(after_p, before_p, 4);
    const float dx_huge = dx(after_p, before_p, 5);
    const float dx_med = dx(after_p, before_p, 6);
    const float dx_cap = dx(after_p, before_p, 7);
    EXPECT_FLOAT_EQ(dx_near_large, dx_med);
    EXPECT_FLOAT_EQ(dx_small, dx_med);
    EXPECT_FLOAT_EQ(dx_tiny, dx_med);
    EXPECT_GT(std::abs(dx_far_large), std::abs(dx_med));
    EXPECT_NEAR(dx_far_large / dx_med, s_large / med_e, 1.0e-3);
    EXPECT_NEAR(dx_huge, dx_cap, 1.0e-6);
    EXPECT_NEAR(dx_huge / dx_med, r_max / r_min, 0.02f * (r_max / r_min));
}

TEST(MRNFStrategyTest, PerSplatMeanStepDecaysIdentically) {
    auto splat_g = create_mrnf_test_splat_data(6);
    auto splat_p = create_mrnf_test_splat_data(6);
    MRNF global(splat_g);
    MRNF per_splat(splat_p);
    global.initialize(mean_step_test_params(param::MeanStepMode::Global));
    per_splat.initialize(mean_step_test_params(param::MeanStepMode::PerSplat));

    ASSERT_DOUBLE_EQ(global._mean_lr_unscaled, per_splat._mean_lr_unscaled);
    ASSERT_DOUBLE_EQ(global._mean_lr_gamma, per_splat._mean_lr_gamma);

    global.step(1);
    per_splat.step(1);
    EXPECT_DOUBLE_EQ(global._mean_lr_unscaled, per_splat._mean_lr_unscaled);
    EXPECT_NEAR(global.get_optimizer().get_param_lr(ParamType::Means),
                global._mean_lr_unscaled * global._bounds.median_size,
                1e-12);
    EXPECT_NEAR(per_splat.get_optimizer().get_param_lr(ParamType::Means),
                per_splat._mean_lr_unscaled * per_splat._bounds.median_size,
                1e-12);
}

TEST(MRNFStrategyTest, PerSplatMeanStepFusedStateUsesUnscaledLr) {
    auto splat_g = create_mrnf_test_splat_data(4);
    auto splat_p = create_mrnf_test_splat_data(4);
    place_deep_far_probe_at(splat_g, 3);
    place_deep_far_probe_at(splat_p, 3);
    MRNF global(splat_g);
    MRNF per_splat(splat_p);
    global.initialize(mean_step_test_params(param::MeanStepMode::Global));
    per_splat.initialize(mean_step_test_params(param::MeanStepMode::PerSplat));
    EXPECT_EQ(per_splat.get_optimizer().mean_step_far_mask(), nullptr);

    install_test_camera_hull(per_splat);
    per_splat._far_starvation = 1.0f;
    per_splat.refresh_far_field_mask(static_cast<size_t>(splat_p.size()));
    per_splat.sync_mean_learning_rate();
    const auto fused_g = global.get_optimizer().prepare_fastgs_fused_adam(1);
    const auto fused_p = per_splat.get_optimizer().prepare_fastgs_fused_adam(1);
    EXPECT_FALSE(fused_g.per_splat_mean_step);
    EXPECT_TRUE(fused_p.per_splat_mean_step);
    EXPECT_NEAR(fused_p.mean_step_median_extent, per_splat._median_splat_extent, 1e-6f);
    EXPECT_NEAR(fused_p.mean_step_r_min, 1.0f, 1e-6f);
    EXPECT_NEAR(fused_p.mean_step_r_max, 300.0f, 1e-4f);
    ASSERT_NE(fused_p.mean_step_far_mask, nullptr);
    EXPECT_EQ(fused_p.mean_step_far_mask_n, 4);
    EXPECT_EQ(fused_g.mean_step_far_mask, nullptr);

    const double bc1_rcp = 1.0 / (1.0 - std::pow(0.9, 1.0));
    EXPECT_NEAR(fused_p.means.step_size,
                static_cast<float>(per_splat._mean_lr_unscaled * per_splat._bounds.median_size * bc1_rcp),
                1e-8);
    EXPECT_NEAR(fused_g.means.step_size,
                static_cast<float>(global._mean_lr_unscaled * global._bounds.median_size * bc1_rcp),
                1e-8);
    EXPECT_NEAR(fused_p.means.step_size, fused_g.means.step_size, 1e-8);
}

TEST(MRNFStrategyTest, MedianSplatExtentMatchesActiveGeomean) {
    auto splat_data = create_mrnf_test_splat_data(7);
    MRNF strategy(splat_data);
    strategy.initialize(mean_step_test_params(param::MeanStepMode::PerSplat));
    write_isotropic_log_scales(splat_data, {
                                               std::log(1.0f),
                                               std::log(2.0f),
                                               std::log(4.0f),
                                               std::log(8.0f),
                                               std::log(16.0f),
                                               std::log(32.0f),
                                               std::log(64.0f),
                                           });
    strategy.compute_bounds();
    ASSERT_TRUE(strategy._median_splat_extent_valid);
    EXPECT_NEAR(strategy._median_splat_extent, 8.0f, 1.0e-4f);

    splat_data.set_frozen_ranges({SplatData::FrozenRange{.start = 0, .count = 1}});
    strategy.compute_bounds();
    ASSERT_TRUE(strategy._median_splat_extent_valid);
    EXPECT_NEAR(strategy._median_splat_extent, 16.0f, 1.0e-4f);
}

TEST(MRNFStrategyTest, PerSplatInvalidMedianFallsBackToGlobal) {
    auto splat_g = create_mrnf_test_splat_data(6);
    auto splat_p = create_mrnf_test_splat_data(6);
    const std::vector<float> dead(6, -std::numeric_limits<float>::infinity());
    write_isotropic_log_scales(splat_g, dead);
    write_isotropic_log_scales(splat_p, dead);

    MRNF global(splat_g);
    MRNF per_splat(splat_p);
    global.initialize(mean_step_test_params(param::MeanStepMode::Global));
    per_splat.initialize(mean_step_test_params(param::MeanStepMode::PerSplat));

    EXPECT_FALSE(global.get_optimizer().per_splat_mean_step());
    EXPECT_FALSE(per_splat.get_optimizer().per_splat_mean_step());
    EXPECT_FALSE(per_splat._median_splat_extent_valid);
    EXPECT_NEAR(global.get_optimizer().get_param_lr(ParamType::Means),
                per_splat.get_optimizer().get_param_lr(ParamType::Means),
                1e-12);

    const auto means0 = means_xyz(splat_g);
    ASSERT_EQ(means0, means_xyz(splat_p));
    global.get_optimizer().get_grad(ParamType::Means).fill_(0.25f);
    per_splat.get_optimizer().get_grad(ParamType::Means).fill_(0.25f);
    global.get_optimizer().step(1);
    per_splat.get_optimizer().step(1);
    const auto after_g = means_xyz(splat_g);
    const auto after_p = means_xyz(splat_p);
    ASSERT_EQ(after_g.size(), after_p.size());
    for (size_t i = 0; i < after_g.size(); ++i) {
        EXPECT_FLOAT_EQ(after_g[i], after_p[i]) << "axis " << i;
    }
}

TEST(MRNFStrategyTest, PerSplatMeanStepNullMaskMatchesGlobal) {
    auto run_pair = [](MRNF& global, MRNF& per_splat, SplatData& splat_g, SplatData& splat_p) {
        EXPECT_EQ(per_splat.get_optimizer().mean_step_far_mask(), nullptr);
        EXPECT_TRUE(per_splat.get_optimizer().per_splat_mean_step());
        const float med_e = per_splat.get_optimizer().mean_step_median_extent();
        ASSERT_GT(med_e, 0.0f);
        write_isotropic_log_scales(splat_g, {std::log(2.0f * med_e), std::log(10.0f * med_e),
                                             std::log(50.0f * med_e), std::log(300.0f * med_e),
                                             std::log(4.0f * med_e), std::log(0.25f * med_e)});
        write_isotropic_log_scales(splat_p, {std::log(2.0f * med_e), std::log(10.0f * med_e),
                                             std::log(50.0f * med_e), std::log(300.0f * med_e),
                                             std::log(4.0f * med_e), std::log(0.25f * med_e)});
        const auto means0 = means_xyz(splat_g);
        ASSERT_EQ(means0, means_xyz(splat_p));
        global.get_optimizer().get_grad(ParamType::Means).fill_(0.25f);
        per_splat.get_optimizer().get_grad(ParamType::Means).fill_(0.25f);
        global.get_optimizer().step(1);
        per_splat.get_optimizer().step(1);
        const auto after_g = means_xyz(splat_g);
        const auto after_p = means_xyz(splat_p);
        ASSERT_EQ(after_g.size(), after_p.size());
        for (size_t i = 0; i < after_g.size(); ++i) {
            EXPECT_FLOAT_EQ(after_g[i], after_p[i]) << "axis " << i;
        }
    };

    {
        auto splat_g = create_mrnf_test_splat_data(6);
        auto splat_p = create_mrnf_test_splat_data(6);
        MRNF global(splat_g);
        MRNF per_splat(splat_p);
        global.initialize(mean_step_test_params(param::MeanStepMode::Global));
        per_splat.initialize(mean_step_test_params(param::MeanStepMode::PerSplat));
        run_pair(global, per_splat, splat_g, splat_p);
    }

    {
        auto splat_g = create_mrnf_test_splat_data(6);
        auto splat_p = create_mrnf_test_splat_data(6);
        MRNF global(splat_g);
        MRNF per_splat(splat_p);
        global.initialize(mean_step_test_params(param::MeanStepMode::Global));
        per_splat.initialize(mean_step_test_params(param::MeanStepMode::PerSplat));
        std::vector<std::shared_ptr<Camera>> cams;
        cams.push_back(make_hull_camera(0.0f, 0.0f, 0.0f, 0));
        DatasetConfig cfg;
        cfg.test_every = 1000;
        per_splat.set_training_dataset(
            std::make_shared<CameraDataset>(std::move(cams), cfg, CameraDataset::Split::ALL));
        run_pair(global, per_splat, splat_g, splat_p);
    }
}

TEST(MRNFStrategyTest, CadenceScaledMatchesRefineEvery) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);

    auto opt_params = vanilla_mrnf_params();
    opt_params.iterations = 1'000;
    opt_params.max_cap = 32;
    opt_params.refine_every = 100;
    strategy.initialize(opt_params);
    EXPECT_EQ(strategy.cadence_scaled(20), 20);
    EXPECT_EQ(strategy.cadence_scaled(0), 0);

    opt_params.refine_every = 200;
    strategy.set_optimization_params(opt_params);
    EXPECT_EQ(strategy.cadence_scaled(20), 40);

    opt_params.refine_every = 50;
    strategy.set_optimization_params(opt_params);
    EXPECT_EQ(strategy.cadence_scaled(20), 10);
}

TEST(MRNFStrategyTest, FarStarvationFactorFromSyntheticPopulations) {
    constexpr float kFull = 2.0f;
    constexpr float kRich = 3.5f;
    EXPECT_FLOAT_EQ(MRNF::far_starvation_factor(1.0f, kFull, kRich), 1.0f);
    EXPECT_FLOAT_EQ(MRNF::far_starvation_factor(2.0f, kFull, kRich), 1.0f);
    EXPECT_FLOAT_EQ(MRNF::far_starvation_factor(2.75f, kFull, kRich), 0.5f);
    EXPECT_FLOAT_EQ(MRNF::far_starvation_factor(3.5f, kFull, kRich), 0.0f);
    EXPECT_FLOAT_EQ(MRNF::far_starvation_factor(6.0f, kFull, kRich), 0.0f);
    EXPECT_FLOAT_EQ(MRNF::far_starvation_factor(0.0f, 0.0f, kRich), 0.0f);

    auto starvation_params = [](const int max_cap) {
        auto opt = vanilla_mrnf_params();
        opt.iterations = 1'000;
        opt.max_cap = max_cap;
        opt.refine_every = 100;
        opt.far_scene_min_fraction = 0.0f;
        opt.far_mask_orbits = 2.0f;
        return opt;
    };

    {
        auto splat = create_mrnf_test_splat_data(10);
        MRNF strategy(splat);
        strategy.initialize(starvation_params(20));
        install_test_camera_hull(strategy);
        EXPECT_EQ(strategy._initial_sfm_point_count, 10u);
        EXPECT_FLOAT_EQ(strategy._far_starvation, 1.0f);
    }

    {
        auto splat = create_mrnf_test_splat_data(10);
        MRNF strategy(splat);
        strategy.initialize(starvation_params(35));
        install_test_camera_hull(strategy);
        EXPECT_EQ(strategy._initial_sfm_point_count, 10u);
        EXPECT_FLOAT_EQ(strategy._far_starvation, 0.0f);
    }

    {
        auto splat = create_mrnf_test_splat_data(10);
        MRNF strategy(splat);
        strategy.initialize(starvation_params(10));
        EXPECT_EQ(strategy._initial_sfm_point_count, 10u);
        EXPECT_FLOAT_EQ(strategy._far_starvation, 1.0f);
        strategy._initial_sfm_point_count = 0;
        strategy.update_far_starvation();
        EXPECT_FLOAT_EQ(strategy._far_starvation, 1.0f);
    }

    {
        auto splat = create_mrnf_test_splat_data(10);
        MRNF strategy(splat);
        strategy.initialize(starvation_params(0));
        EXPECT_EQ(strategy._initial_sfm_point_count, 10u);
        EXPECT_FLOAT_EQ(strategy._far_starvation, 0.0f);
    }
}

TEST(MRNFStrategyTest, OptimizationParametersDefaultsAreFarFieldOn) {
    const param::OptimizationParameters defaults{};
    EXPECT_EQ(defaults.explore_splits, 20);
    EXPECT_EQ(defaults.explore_seeds, 20);
    EXPECT_FLOAT_EQ(defaults.far_growth_cap, 0.3f);
    EXPECT_FLOAT_EQ(defaults.far_decay_scale, 0.25f);
    EXPECT_EQ(defaults.mean_step_mode, param::MeanStepMode::PerSplat);
    EXPECT_FLOAT_EQ(defaults.far_cap_ratio_full, 2.0f);
    EXPECT_FLOAT_EQ(defaults.far_cap_ratio_rich, 3.5f);
}
