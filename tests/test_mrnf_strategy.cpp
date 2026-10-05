/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

class MRNFStrategyTest_EdgeGuidanceFactorPrefersHigherPrecomputedEdgeScores_Test;
class MRNFStrategyTest_GrowAndSplitResetsOptimizerStateForParents_Test;
class MRNFStrategyTest_SHDegree0KeepsShNEmptyAndFusedAdamUsableAfterGrowth_Test;
class MRNFStrategyTest_GrowAndSplitUsesIgsPlusSplitRule_Test;
class MRNFStrategyTest_GrowAndSplitOversizeChannelPrefersOversizedError_Test;
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
class MRNFStrategyTest_ApplyDecaySkipsFrozenRows_Test;
class MRNFStrategyTest_DensificationInfoShapeIsTwoRows_Test;
class MRNFStrategyTest_ZeroVisibilityProducesNoGrowth_Test;
class MRNFStrategyTest_DirectAuxiliaryGrowthPreservesPrefix_Test;
class MRNFStrategyTest_EdgeWindowNormalizesViewsAndClosesBeforeRefineBackward_Test;

#include "core/cuda/memory_arena.hpp"
#include "core/cuda/sh_layout.cuh"
#include "core/parameters.hpp"
#include "core/scene.hpp"
#include "core/sh_value_quant.hpp"
#include "core/splat_data.hpp"
#include "core/tensor_cuda_interop.hpp"
#include "cuda_backend_test.hpp"
#include "io/formats/ply.hpp"
#include "io/project_document.hpp"
#include "lfs/training/joint_adam_codec.hpp"
#include "lfs/training/live_model_mutation_guard.hpp"
#include "lfs/training/ops/registry.hpp"
#include "lfs/training/sh_value_codec.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include "training/kernels/densification_kernels.hpp"
#include "training/kernels/mrnf_kernels.hpp"
#include "training/optimizer/render_output.hpp"
#include "training/strategies/mrnf.hpp"
#include "training/strategies/strategy_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <sstream>
#include <vector>

using namespace lfs::core;
using namespace lfs::training;

class MRNFStrategyTest : public lfs::test::CudaBackendTest {};

TEST_F(MRNFStrategyTest, PruneBoundsOrMatchesTensorChain) {
    constexpr size_t n = 513;
    std::vector<float> means_values(n * 3);
    std::vector<float> scale_values(n);
    for (size_t i = 0; i < n; ++i) {
        means_values[3 * i] = static_cast<float>(static_cast<int>(i % 13) - 6);
        means_values[3 * i + 1] = static_cast<float>(static_cast<int>(i % 7) - 3);
        means_values[3 * i + 2] = static_cast<float>(static_cast<int>(i % 5) - 2);
        scale_values[i] = static_cast<float>(static_cast<int>(i % 11) - 5) * 0.4f;
    }
    means_values[3] = std::numeric_limits<float>::quiet_NaN();
    means_values[4] = 100.0f;
    means_values[6] = std::numeric_limits<float>::infinity();
    scale_values[7] = std::numeric_limits<float>::quiet_NaN();

    const auto means = Tensor::from_vector(means_values, TensorShape({n, 3}), Device::CUDA);
    const auto scale_max = Tensor::from_vector(scale_values, TensorShape({n}), Device::CUDA);
    const auto initial = scale_max < -1.0f;
    auto actual = initial.clone();
    constexpr float center_values[3] = {0.25f, -0.5f, 0.75f};
    const auto center = Tensor::from_vector(
        std::vector<float>{center_values[0], center_values[1], center_values[2]},
        TensorShape({1, 3}), Device::CUDA);
    const auto expected = initial | (scale_max > 1.25f) |
                          ((means - center).abs().max(1) > 3.5f);

    mrnf_strategy::launch_prune_bounds_or(
        means.ptr<float>(), scale_max.ptr<float>(), actual.ptr<bool>(),
        n, center_values, 3.5f, 1.25f);
    const auto actual_host = actual.cpu();
    const auto expected_host = expected.cpu();
    EXPECT_EQ(std::memcmp(actual_host.ptr<bool>(), expected_host.ptr<bool>(), n * sizeof(bool)), 0);
}

TEST_F(MRNFStrategyTest, ReplaceParentWeightsMatchesTensorProducts) {
    constexpr size_t n = 513;
    std::vector<float> opacity_values(n);
    std::vector<float> visibility_values(n);
    std::vector<float> edge_values(n);
    std::vector<bool> active_values(n);
    std::vector<bool> trainable_values(n);
    for (size_t i = 0; i < n; ++i) {
        opacity_values[i] = static_cast<float>(i % 17) / 16.0f;
        visibility_values[i] = i % 3 == 0 ? 0.0f : static_cast<float>(i % 5);
        edge_values[i] = 0.5f + static_cast<float>(i % 11) / 10.0f;
        active_values[i] = i % 7 != 0;
        trainable_values[i] = i % 13 != 0;
    }
    opacity_values[1] = std::numeric_limits<float>::quiet_NaN();
    opacity_values[2] = std::numeric_limits<float>::infinity();
    opacity_values[3] = -0.0f;
    edge_values[4] = std::numeric_limits<float>::quiet_NaN();
    edge_values[5] = -0.0f;

    const auto opacities = Tensor::from_vector(opacity_values, TensorShape({n}), Device::CUDA);
    const auto visibility = Tensor::from_vector(visibility_values, TensorShape({n}), Device::CUDA);
    const auto active = Tensor::from_vector(active_values, TensorShape({n}), Device::CUDA);
    const auto trainable = Tensor::from_vector(trainable_values, TensorShape({n}), Device::CUDA);
    const auto edge = Tensor::from_vector(edge_values, TensorShape({n}), Device::CUDA);

    for (int options = 0; options < 8; ++options) {
        auto expected = opacities * (visibility > 0.0f);
        if (options & 1)
            expected = expected * active;
        if (options & 2)
            expected = expected * trainable;
        if (options & 4)
            expected = expected * edge;

        auto actual = Tensor::empty({n}, Device::CUDA);
        mrnf_strategy::launch_replace_parent_weights(
            opacities.ptr<float>(), visibility.ptr<float>(),
            options & 1 ? active.ptr<bool>() : nullptr,
            options & 2 ? trainable.ptr<bool>() : nullptr,
            options & 4 ? edge.ptr<float>() : nullptr,
            actual.ptr<float>(), n);
        const auto actual_host = actual.cpu();
        const auto expected_host = expected.cpu();
        EXPECT_EQ(std::memcmp(actual_host.ptr<float>(), expected_host.ptr<float>(), n * sizeof(float)), 0)
            << "options=" << options;
    }
}

namespace {
    // The largest trained splat model in the test data folder.
    std::filesystem::path largest_scene_ply() {
        std::filesystem::path best;
        std::uintmax_t best_size = 0;
        std::error_code ec;
        for (const auto& entry : std::filesystem::directory_iterator(TEST_DATA_DIR, ec)) {
            if (entry.path().extension() != ".ply" || !entry.is_regular_file(ec))
                continue;
            const auto size = entry.file_size(ec);
            if (!ec && size > best_size) {
                best = entry.path();
                best_size = size;
            }
        }
        return best;
    }
} // namespace

TEST_F(MRNFStrategyTest, RefinePredicatesAndIndicesMatchTensorReferenceOnRealScene) {
    const auto scene = largest_scene_ply();
    if (scene.empty())
        GTEST_SKIP() << "no splat model in " << TEST_DATA_DIR;
    const auto loaded = lfs::io::load_ply(scene);
    ASSERT_TRUE(loaded.has_value()) << lfs::format_for_developer(loaded.error());
    const SplatData& splat = loaded->value;
    const size_t n = static_cast<size_t>(splat.size());
    ASSERT_GT(n, 100000u);
    const auto means = splat.means();
    const auto scale_max = splat.scaling_raw().max(1);
    const auto initial = splat.opacity_raw().squeeze(-1) < -2.0f;
    const auto center_host = means.slice(0, 0, 1).contiguous().cpu();
    const float center_values[3] = {center_host.ptr<float>()[0], center_host.ptr<float>()[1],
                                    center_host.ptr<float>()[2]};
    const auto center = Tensor::from_vector(
        std::vector<float>{center_values[0], center_values[1], center_values[2]},
        TensorShape({1, 3}), Device::CUDA);
    auto opacities = splat.get_opacity().squeeze(-1);
    const auto visibility = (scale_max > -10.0f).to(DataType::Float32);
    const auto active = means.slice(1, 0, 1).squeeze(-1) > center_values[0];
    const auto trainable = means.slice(1, 1, 2).squeeze(-1) > center_values[1];
    const auto edge = scale_max.abs() + 0.25f;

    for (const float max_allowed : {1.0f, 5.0f, 20.0f, 100.0f}) {
        const float log_max_allowed = std::log(max_allowed);
        auto actual_mask = initial.clone();
        const auto reference_mask = initial | (scale_max > log_max_allowed) |
                                    ((means - center).abs().max(1) > max_allowed);
        mrnf_strategy::launch_prune_bounds_or(
            means.ptr<float>(), scale_max.ptr<float>(), actual_mask.ptr<bool>(),
            n, center_values, max_allowed, log_max_allowed);
        const auto actual_mask_host = actual_mask.cpu();
        const auto reference_mask_host = reference_mask.cpu();
        EXPECT_EQ(std::memcmp(actual_mask_host.ptr<bool>(), reference_mask_host.ptr<bool>(), n), 0);

        const auto expected_weights = opacities * (visibility > 0.0f) * active * trainable * edge;
        auto actual_weights = Tensor::empty({n}, Device::CUDA);
        mrnf_strategy::launch_replace_parent_weights(
            opacities.ptr<float>(), visibility.ptr<float>(), active.ptr<bool>(),
            trainable.ptr<bool>(), edge.ptr<float>(), actual_weights.ptr<float>(), n);
        const auto actual_weights_host = actual_weights.cpu();
        const auto expected_weights_host = expected_weights.cpu();
        EXPECT_EQ(std::memcmp(actual_weights_host.ptr<float>(), expected_weights_host.ptr<float>(),
                              n * sizeof(float)),
                  0);
    }
}

TEST_F(MRNFStrategyTest, ShNBatchArenaMatchesCatFallbackOnRealScene) {
    struct QuantReset {
        ~QuantReset() { sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt); }
    } reset;
    sh_value::set_sh_value_quant_enabled_for_testing(true);
    const auto scene = largest_scene_ply();
    if (scene.empty())
        GTEST_SKIP() << "no splat model in " << TEST_DATA_DIR;
    auto loaded = lfs::io::load_ply(scene);
    ASSERT_TRUE(loaded.has_value()) << lfs::format_for_developer(loaded.error());
    SplatData base = std::move(loaded->value);
    ASSERT_TRUE(sh_value::apply_shN_value_quant(base));
    const size_t n = static_cast<size_t>(base.size());
    constexpr size_t kScatter = 3000;
    constexpr size_t kZero = 700;
    ASSERT_GT(n, kScatter);
    std::vector<int> src_host(kScatter), dest_host(kScatter), zero_host(kZero);
    for (size_t i = 0; i < kScatter; ++i) {
        src_host[i] = static_cast<int>((113 * i + 17) % n);
        dest_host[i] = static_cast<int>((67 * i + 5) % n);
    }
    for (size_t i = 0; i < kZero; ++i)
        zero_host[i] = static_cast<int>((67 * i + 6) % n);
    const auto src_indices = Tensor::from_vector(src_host, TensorShape({kScatter}), Device::CUDA)
                                 .to(DataType::Int64);
    const auto dest_indices = Tensor::from_vector(dest_host, TensorShape({kScatter}), Device::CUDA)
                                  .to(DataType::Int64);
    const auto zero_indices = Tensor::from_vector(zero_host, TensorShape({kZero}), Device::CUDA)
                                  .to(DataType::Int64);
    Tensor canonical;
    sh_value::gather_shN_to_canonical(base, src_indices, canonical);
    ASSERT_EQ(canonical.shape()[0], kScatter);

    auto arena_splat = base.clone();
    auto fallback_splat = base.clone();
    auto& arena = GlobalArenaManager::instance().get_arena();
    const cudaStream_t stream = getCurrentCUDAStream();
    const size_t rows = kScatter + kZero;
    const size_t bytes = ((rows * sizeof(int64_t) + 255) & ~size_t{255}) +
                         rows * static_cast<size_t>(base.max_sh_coeffs_rest()) * 3 * sizeof(float);
    {
        const auto frame = arena.begin_frame(stream);
        ASSERT_NE(arena.get_allocator(frame, "test.shN_batch")(bytes + 4096), nullptr);
        arena.end_frame(frame, stream);
    }
    ASSERT_GE(arena.get_memory_info().arena_capacity, bytes);

    const auto flush = [&](SplatData& splat) {
        LiveModelMutationGuard guard("test.shN_batch");
        sh_value::ShNMutationBatch batch(splat);
        batch.scatter(dest_indices, canonical);
        batch.zero(zero_indices);
        batch.flush();
    };
    flush(arena_splat);
    EXPECT_GT(arena.get_memory_info().current_usage, 0u);
    const auto held = arena.begin_frame(stream);
    flush(fallback_splat);
    arena.end_frame(held, stream);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto arena_codes = arena_splat.shN().cpu().contiguous();
    const auto fallback_codes = fallback_splat.shN().cpu().contiguous();
    const auto arena_bounds = arena_splat.shN_value_bounds().cpu().contiguous();
    const auto fallback_bounds = fallback_splat.shN_value_bounds().cpu().contiguous();
    ASSERT_EQ(arena_codes.bytes(), fallback_codes.bytes());
    ASSERT_EQ(arena_bounds.bytes(), fallback_bounds.bytes());
    EXPECT_EQ(std::memcmp(arena_codes.data_ptr(), fallback_codes.data_ptr(), arena_codes.bytes()), 0);
    EXPECT_EQ(std::memcmp(arena_bounds.data_ptr(), fallback_bounds.data_ptr(), arena_bounds.bytes()), 0);
}

namespace {

    const lfs::gpu_ops::BackwardAdamParam& adam_group(
        const lfs::gpu_ops::BackwardAdam& adam, const lfs::gpu_ops::AdamSlot slot) {
        return adam.groups[static_cast<std::size_t>(slot)];
    }

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

        auto means = Tensor::from_vector(means_data, TensorShape({n, 3}), Device::GPU);
        auto sh0 = Tensor::from_vector(sh0_data, TensorShape({n, 1, 3}), Device::GPU);
        auto shN = Tensor::zeros(TensorShape({n, sh_rest, 3}), Device::GPU);
        auto scaling = Tensor::from_vector(scaling_data, TensorShape({n, 3}), Device::GPU);
        auto rotation = Tensor::from_vector(rotation_data, TensorShape({n, 4}), Device::GPU);
        auto opacity = Tensor::from_vector(opacity_data, TensorShape({n, 1}), Device::GPU);

        return SplatData(sh_degree, means, sh0, shN, scaling, rotation, opacity, 1.0f);
    }

    param::OptimizationParameters vanilla_mrnf_params() {
        return param::OptimizationParameters::mrnf_defaults();
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
        state.exp_avg = packed.gpu();
        state.joint_bounds = bounds.gpu();
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

TEST_F(MRNFStrategyTest, EdgeGuidanceFactorPrefersHigherPrecomputedEdgeScores) {
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
        Tensor::from_vector(edge_scores_data, TensorShape({10}), Device::GPU);
    strategy._edge_precompute_valid = true;

    const auto guidance = strategy.edge_guidance_factor().cpu();
    const float* guidance_ptr = guidance.ptr<float>();

    EXPECT_NEAR(guidance_ptr[2], 1.0f, 1e-5f);
    EXPECT_GT(guidance_ptr[0], 1.0f);
    EXPECT_GT(guidance_ptr[1], guidance_ptr[0]);
}

TEST_F(MRNFStrategyTest, EdgeWindowNormalizesViewsAndClosesBeforeRefineBackward) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);

    auto opt_params = vanilla_mrnf_params();
    opt_params.iterations = 1'000;
    opt_params.start_refine = 0;
    opt_params.stop_refine = 800;
    opt_params.refine_every = 100;
    opt_params.max_cap = 32;
    opt_params.use_edge_map = true;
    strategy.initialize(opt_params);

    std::vector<float> raw_values(10);
    for (size_t i = 0; i < raw_values.size(); ++i) {
        raw_values[i] = static_cast<float>(i + 1);
    }
    const auto raw = Tensor::from_vector(
        raw_values, TensorShape({raw_values.size()}), Device::GPU);

    const auto add_view = [&](const int iter, const float scale) {
        auto scratch = strategy.edge_score_scratch(iter);
        ASSERT_TRUE(scratch.is_valid());
        scratch.copy_(raw * scale);
        strategy.on_edge_score_accumulated(iter);
    };

    // Equal per-view weighting must survive an arbitrary raw score scale.
    add_view(1, 1.0f);
    add_view(99, 17.0f);
    RenderOutput unused;
    strategy.pre_step(100, unused);
    ASSERT_TRUE(strategy._edge_precompute_valid);
    EXPECT_FALSE(strategy._edge_score_sum.is_valid());
    EXPECT_EQ(strategy._edge_sample_count, 0);
    auto first_window = strategy._precomputed_edge_scores.cpu();
    const auto* first_ptr = first_window.ptr<float>();
    for (size_t i = 0; i < raw_values.size(); ++i) {
        EXPECT_NEAR(first_ptr[i], raw_values[i] / 6.0f, 1.0e-6f) << "index=" << i;
    }

    // Refinement at 100 closes before iteration 100's main backward. Its view
    // therefore starts the [100,200) window, and a mask active only for the
    // later sample affects only that sample.
    add_view(100, 3.0f);
    splat_data.set_frozen_ranges({SplatData::FrozenRange{.start = 0, .count = 1}});
    add_view(199, 11.0f);
    splat_data.clear_frozen_ranges();
    strategy.pre_step(200, unused);
    ASSERT_TRUE(strategy._edge_precompute_valid);
    auto second_window = strategy._precomputed_edge_scores.cpu();
    const auto* second_ptr = second_window.ptr<float>();
    EXPECT_NEAR(second_ptr[0], (raw_values[0] / 6.0f) * 0.5f, 1.0e-6f);
    for (size_t i = 1; i < raw_values.size(); ++i) {
        EXPECT_NEAR(second_ptr[i], raw_values[i] / 6.0f, 1.0e-6f) << "index=" << i;
    }
}

class CropDampingStrategyTest : public lfs::test::CudaBackendTest {};

TEST_F(CropDampingStrategyTest, MrnfRejectedRowsAreNotRefineCandidatesAtZeroScale) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);

    auto opt_params = vanilla_mrnf_params();
    opt_params.iterations = 100;
    opt_params.max_cap = 32;
    opt_params.growth_grad_threshold = 0.5f;
    strategy.initialize(opt_params);
    strategy._refine_weight_max = Tensor::ones({10}, Device::GPU);
    strategy.visibility_accumulator().fill_(1.0f);

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

TEST_F(MRNFStrategyTest, DegenerateBoundsStayInvalidAndKeepFiniteMeanLearningRate) {
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

TEST_F(MRNFStrategyTest, RefinementPreservesThinSurfacesAndPrunesCollapsedSplats) {
    auto splat_data = create_mrnf_test_splat_data(8);
    MRNF strategy(splat_data);
    auto opt_params = vanilla_mrnf_params();
    opt_params.iterations = 1'000;
    opt_params.start_refine = 0;
    opt_params.stop_refine = 900;
    opt_params.refine_every = 10;
    opt_params.grow_until_iter = 0;
    opt_params.grow_fraction = 0.0f;
    opt_params.max_cap = 32;
    strategy.initialize(opt_params);

    // Flattening may shrink any normal axis below the minimum extent while
    // leaving a useful surface. Only a splat tiny in every axis is collapsed.
    splat_data.scaling_raw().copy_(Tensor::from_vector(
        std::vector<float>{-30, 0, 0, 0, -30, 0, 0, 0, -30,
                           -30, -30, -30, 0, 0, 0, 0, 0, 0,
                           20, 20, 20, 0, 0, 0},
        TensorShape({8, 3}), Device::CUDA));
    splat_data.opacity_raw().copy_(Tensor::from_vector(
        std::vector<float>{0, 0, 0, 0, -20, 0, 0, 0},
        TensorShape({8, 1}), Device::CUDA));
    splat_data.rotation_raw().index_put_(
        Tensor::from_vector(std::vector<int>{5}, TensorShape({1}), Device::CUDA).to(DataType::Int64),
        Tensor::zeros({1, 4}, Device::CUDA));
    splat_data._densification_info.zero_();

    RenderOutput render_output;
    strategy.post_backward(10, render_output);

    ASSERT_EQ(splat_data.size(), 8u);
    ASSERT_TRUE(splat_data.deleted().is_valid());
    const auto deleted_cpu = splat_data.deleted().cpu();
    const bool* deleted = deleted_cpu.ptr<bool>();
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(deleted[i], i >= 3 && i <= 6) << "splat " << i;
    }
}

TEST_F(MRNFStrategyTest, LineBoundsUseFiniteSceneScaleForMeanLearningRate) {
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

TEST_F(MRNFStrategyTest, RemoveGaussiansKeepsOptimizerStateUsable) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);

    auto opt_params = vanilla_mrnf_params();
    opt_params.iterations = 10'000;
    opt_params.sh_degree_interval = 10'000;
    opt_params.max_cap = 32;

    strategy.initialize(opt_params);
    splat_data._densification_info = Tensor::ones({2, static_cast<size_t>(splat_data.size())}, Device::GPU);

    const auto mask = Tensor::from_vector(
        std::vector<bool>{false, true, false, true, false, false, false, false, false, false},
        TensorShape({10}),
        Device::GPU);

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

TEST_F(MRNFStrategyTest, QuantizedShNFirstMomentStartsAtSignedZeroPoint) {
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

TEST_F(MRNFStrategyTest, RemoveGaussiansCompactsQuantizedAdamScalesAndPreservesShNDtype) {
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
        Device::GPU);

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

TEST_F(MRNFStrategyTest, GrowAndSplitResetsOptimizerStateForParents) {
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

    strategy._refine_weight_max = Tensor::zeros({static_cast<size_t>(splat_data.size())}, Device::GPU);
    auto visibility = strategy.visibility_accumulator();
    visibility.zero_();

    const auto split_idx = Tensor::from_vector(std::vector<int>{0}, TensorShape({1}), Device::GPU).to(DataType::Int64);
    strategy._refine_weight_max.index_put_(split_idx, Tensor::full({1}, 1.0f, Device::GPU));
    visibility.index_put_(split_idx, Tensor::full({1}, 1.0f, Device::GPU));

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

TEST_F(MRNFStrategyTest, SHDegree0KeepsShNEmptyAndFusedAdamUsableAfterGrowth) {
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
        EXPECT_TRUE(adam_group(fused, lfs::gpu_ops::AdamSlot::Means).enabled);
        EXPECT_FALSE(adam_group(fused, lfs::gpu_ops::AdamSlot::ShN).enabled);
    });

    strategy._refine_weight_max = Tensor::zeros({static_cast<size_t>(splat_data.size())}, Device::GPU);
    auto visibility = strategy.visibility_accumulator();
    visibility.zero_();

    const auto split_idx = Tensor::from_vector(std::vector<int>{0}, TensorShape({1}), Device::GPU).to(DataType::Int64);
    strategy._refine_weight_max.index_put_(split_idx, Tensor::full({1}, 1.0f, Device::GPU));
    visibility.index_put_(split_idx, Tensor::full({1}, 1.0f, Device::GPU));

    const size_t initial_size = splat_data.size();
    strategy.grow_and_split(1, 0);

    EXPECT_EQ(splat_data.size(), initial_size + 1);
    EXPECT_EQ(splat_data.shN().numel(), 0u);
    EXPECT_EQ(shN_state->size, 0u);
    EXPECT_NO_THROW({
        const auto fused = strategy.get_optimizer().prepare_fastgs_fused_adam(457);
        EXPECT_TRUE(adam_group(fused, lfs::gpu_ops::AdamSlot::Means).enabled);
        EXPECT_FALSE(adam_group(fused, lfs::gpu_ops::AdamSlot::ShN).enabled);
    });
}

TEST_F(MRNFStrategyTest, ShNReservationTracksMaxDegreeAndMaxCap) {

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
    EXPECT_FALSE(adam_group(scheduled_strategy.get_optimizer().prepare_fastgs_fused_adam(1001),
                            lfs::gpu_ops::AdamSlot::ShN)
                     .enabled);

    scheduled_splat.increment_sh_degree();
    const auto fused = scheduled_strategy.get_optimizer().prepare_fastgs_fused_adam(1001);
    EXPECT_TRUE(adam_group(fused, lfs::gpu_ops::AdamSlot::ShN).enabled);
    expect_shN_capacity(scheduled_splat, scheduled_strategy.get_optimizer(), 1);
}

TEST_F(MRNFStrategyTest, DirectAuxiliaryGrowthPreservesPrefix) {
    constexpr size_t sfm_points = 8;
    constexpr size_t grown_points = sfm_points * 3;

    auto splat_data = create_mrnf_test_splat_data(static_cast<int>(sfm_points));
    MRNF strategy(splat_data);
    auto params = vanilla_mrnf_params();
    params.max_cap = 0;
    strategy.initialize(params);

    // Model the post-SfM state immediately before a reservation boundary:
    // every MRNF auxiliary is a direct-storage tensor with only SfM rows.
    strategy._free_mask = Tensor::zeros_direct(
        {sfm_points}, sfm_points, Device::GPU, DataType::Bool);
    const auto direct_float = [](const size_t n, const float value) {
        auto tensor = Tensor::zeros_direct({n}, n, Device::GPU, DataType::Float32);
        tensor.copy_from(Tensor::full({n}, value, Device::GPU));
        return tensor;
    };
    strategy._refine_weight_max = direct_float(sfm_points, 1.25f);

    EXPECT_TRUE(strategy.get_optimizer().preflight_grow_capacity(grown_points - sfm_points));
    EXPECT_NO_THROW(strategy.ensure_auxiliary_capacity_for_growth(grown_points));

    const auto expect_grown = [grown_points](const Tensor& tensor) {
        EXPECT_EQ(tensor.numel(), grown_points);
        EXPECT_GE(tensor.capacity(), grown_points);
    };
    expect_grown(strategy._free_mask);
    expect_grown(strategy._refine_weight_max);

    const auto expect_prefix = [sfm_points](const Tensor& tensor, const float expected) {
        const auto prefix = tensor.slice(0, 0, sfm_points).cpu();
        const auto* values = prefix.ptr<float>();
        for (size_t i = 0; i < sfm_points; ++i) {
            EXPECT_FLOAT_EQ(values[i], expected);
        }
    };
    expect_prefix(strategy._refine_weight_max, 1.25f);
}

TEST_F(MRNFStrategyTest, GrowAndSplitUsesIgsPlusSplitRule) {
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

    strategy._refine_weight_max = Tensor::zeros({static_cast<size_t>(splat_data.size())}, Device::GPU);
    auto visibility = strategy.visibility_accumulator();
    visibility.zero_();

    const auto split_idx = Tensor::from_vector(std::vector<int>{0}, TensorShape({1}), Device::GPU).to(DataType::Int64);
    strategy._refine_weight_max.index_put_(split_idx, Tensor::full({1}, 1.0f, Device::GPU));
    visibility.index_put_(split_idx, Tensor::full({1}, 1.0f, Device::GPU));

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

TEST_F(MRNFStrategyTest, GrowAndSplitOversizeChannelPrefersOversizedError) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);

    auto opt_params = vanilla_mrnf_params();
    opt_params.iterations = 10'000;
    opt_params.sh_degree_interval = 10'000;
    opt_params.max_cap = 32;
    opt_params.growth_grad_threshold = 0.5f;
    opt_params.grow_fraction = 0.5f;
    opt_params.grow_until_iter = 10'000;
    opt_params.max_screen_share = 0.3f;
    opt_params.oversize_split_fraction = 1.0f;
    strategy.initialize(opt_params);

    const size_t n = static_cast<size_t>(splat_data.size());
    strategy._refine_weight_max = Tensor::zeros({n}, Device::GPU);
    auto visibility = strategy.visibility_accumulator();
    visibility.zero_();

    const auto idx0 = Tensor::from_vector(std::vector<int>{0}, TensorShape({1}), Device::GPU).to(DataType::Int64);
    const auto idx1 = Tensor::from_vector(std::vector<int>{1}, TensorShape({1}), Device::GPU).to(DataType::Int64);
    strategy._refine_weight_max.index_put_(idx0, Tensor::full({1}, 1.0f, Device::GPU));
    strategy._refine_weight_max.index_put_(idx1, Tensor::full({1}, 10.0f, Device::GPU));
    visibility.index_put_(idx0, Tensor::full({1}, 1.0f, Device::GPU));
    visibility.index_put_(idx1, Tensor::full({1}, 1.0f, Device::GPU));

    ASSERT_TRUE(splat_data._max_screen_share.is_valid());
    auto share_cpu = Tensor::zeros({n}, Device::CPU);
    share_cpu.ptr<float>()[0] = 0.9f;
    share_cpu.ptr<float>()[1] = 0.05f;
    splat_data._max_screen_share = share_cpu.gpu();

    const auto means_before = splat_data.means().cpu();
    const float* mb = means_before.ptr<float>();
    const float splat1_x = mb[3];

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
    EXPECT_NEAR(means_ptr[3], splat1_x, 1e-5f);

    const size_t child_base = initial_size * 3;
    EXPECT_NEAR(means_ptr[child_base + 0], -0.5f, 1e-5f);
    EXPECT_NEAR(means_ptr[child_base + 1], 0.0f, 1e-5f);
    EXPECT_NEAR(means_ptr[child_base + 2], 0.0f, 1e-5f);

    EXPECT_NEAR(scales_ptr[0], std::log(0.5f), 1e-5f);
    EXPECT_NEAR(scales_ptr[1], std::log(0.85f), 1e-5f);
    EXPECT_NEAR(scales_ptr[2], std::log(0.85f), 1e-5f);
    EXPECT_NEAR(opacities_ptr[0], std::log(0.3f / 0.7f), 1e-5f);
    EXPECT_NEAR(opacities_ptr[initial_size], std::log(0.3f / 0.7f), 1e-5f);
}

TEST_F(MRNFStrategyTest, StepScalingDoesNotScaleSparsifySteps) {
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

TEST_F(MRNFStrategyTest, StopRefineBoundaryRequestsExclusiveMutation) {
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

TEST_F(MRNFStrategyTest, GrowAndSplitWithoutMaxCapExtendsBookkeepingMasks) {
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

    strategy._refine_weight_max = Tensor::zeros({static_cast<size_t>(splat_data.size())}, Device::GPU);
    auto visibility = strategy.visibility_accumulator();
    visibility.zero_();

    const auto split_idx = Tensor::from_vector(std::vector<int>{0}, TensorShape({1}), Device::GPU).to(DataType::Int64);
    strategy._refine_weight_max.index_put_(split_idx, Tensor::full({1}, 1.0f, Device::GPU));
    visibility.index_put_(split_idx, Tensor::full({1}, 1.0f, Device::GPU));

    const size_t initial_size = splat_data.size();
    ASSERT_NO_THROW(strategy.grow_and_split(1, 0));

    EXPECT_EQ(splat_data.size(), initial_size + 1);
    ASSERT_TRUE(splat_data.has_deleted_mask());
    EXPECT_EQ(splat_data.deleted().shape()[0], splat_data.size());
    EXPECT_EQ(strategy.free_count(), 0u);
}

TEST_F(MRNFStrategyTest, DeletedMaskCapacityGrowthPreservesExistingRows) {
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
        TensorShape({initial_size}), initial_size, Device::GPU, DataType::Bool);
    const auto deleted_index =
        Tensor::from_vector(std::vector<int>{3}, TensorShape({1}), Device::GPU)
            .to(DataType::Int64);
    splat_data.deleted().index_put_(deleted_index, Tensor::ones_bool({1}, Device::GPU));

    strategy._refine_weight_max = Tensor::zeros({initial_size}, Device::GPU);
    auto visibility = strategy.visibility_accumulator();
    visibility.zero_();
    const auto split_index =
        Tensor::from_vector(std::vector<int>{0}, TensorShape({1}), Device::GPU)
            .to(DataType::Int64);
    strategy._refine_weight_max.index_put_(split_index, Tensor::full({1}, 1.0f, Device::GPU));
    visibility.index_put_(split_index, Tensor::full({1}, 1.0f, Device::GPU));

    strategy.grow_and_split(1, 0);

    ASSERT_EQ(splat_data.size(), initial_size + 1);
    const auto deleted_cpu = splat_data.deleted().cpu();
    const bool* values = deleted_cpu.ptr<bool>();
    EXPECT_TRUE(values[3]);
    EXPECT_FALSE(values[initial_size]);
}

TEST_F(MRNFStrategyTest, GrowAndSplitReplacementSkipsZeroWeightCandidates) {
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

    const auto free_indices = Tensor::from_vector(std::vector<int>{8, 9}, TensorShape({2}), Device::GPU).to(DataType::Int64);
    strategy.mark_as_free(free_indices);
    auto true_vals = Tensor::ones_bool({2}, Device::GPU);
    strategy._splat_data->deleted().index_put_(free_indices, true_vals);

    strategy._refine_weight_max = Tensor::zeros({static_cast<size_t>(splat_data.size())}, Device::GPU);
    auto visibility = strategy.visibility_accumulator();
    visibility.zero_();

    const auto visible_parent = Tensor::from_vector(std::vector<int>{0}, TensorShape({1}), Device::GPU).to(DataType::Int64);
    visibility.index_put_(visible_parent, Tensor::full({1}, 1.0f, Device::GPU));

    const size_t initial_size = splat_data.size();
    strategy.grow_and_split(10'001, 2);

    EXPECT_EQ(splat_data.size(), initial_size);
    EXPECT_EQ(strategy.free_count(), 1u);
    EXPECT_EQ(strategy.active_count(), initial_size - 1);
}

TEST_F(MRNFStrategyTest, GrowAndSplitReusesFreeSlotsBeforeAppending) {
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

    const auto free_indices = Tensor::from_vector(std::vector<int>{8, 9}, TensorShape({2}), Device::GPU).to(DataType::Int64);
    strategy.mark_as_free(free_indices);
    auto true_vals = Tensor::ones_bool({2}, Device::GPU);
    strategy._splat_data->deleted().index_put_(free_indices, true_vals);

    strategy._refine_weight_max = Tensor::zeros({static_cast<size_t>(splat_data.size())}, Device::GPU);
    auto visibility = strategy.visibility_accumulator();
    visibility.zero_();

    const auto split_idx = Tensor::from_vector(std::vector<int>{0}, TensorShape({1}), Device::GPU).to(DataType::Int64);
    strategy._refine_weight_max.index_put_(split_idx, Tensor::full({1}, 1.0f, Device::GPU));
    visibility.index_put_(split_idx, Tensor::full({1}, 1.0f, Device::GPU));

    const size_t initial_size = splat_data.size();
    strategy.grow_and_split(1, 0);

    EXPECT_EQ(splat_data.size(), initial_size);
    EXPECT_EQ(strategy.active_count(), initial_size - 1);
    EXPECT_EQ(strategy.free_count(), 1u);
}

TEST_F(MRNFStrategyTest, SerializeRoundTripPreservesFreeMask) {

    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);

    auto opt_params = vanilla_mrnf_params();
    opt_params.iterations = 10'000;
    opt_params.sh_degree_interval = 10'000;
    opt_params.max_cap = 32;
    strategy.initialize(opt_params);

    const auto free_indices = Tensor::from_vector(std::vector<int>{1, 3}, TensorShape({2}), Device::GPU).to(DataType::Int64);
    strategy.mark_as_free(free_indices);
    auto true_vals = Tensor::ones_bool({2}, Device::GPU);
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

TEST_F(MRNFStrategyTest, SerializeRoundTripPreservesLrScheduleState) {

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

TEST_F(MRNFStrategyTest, DeserializeResizesTransientBuffersToLoadedModel) {

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
    EXPECT_EQ(restored.visibility_accumulator().numel(), 12u);
    ASSERT_TRUE(restored.get_model()._densification_info.is_valid());
    EXPECT_EQ(restored.get_model()._densification_info.shape()[1], 12u);
    EXPECT_FALSE(restored._edge_precompute_valid);
}

TEST_F(MRNFStrategyTest, SetOptimizationParamsRecomputesDecayFromCurrentState) {
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

TEST_F(MRNFStrategyTest, ApplyDecaySkipsFrozenRows) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);

    auto opt_params = vanilla_mrnf_params();
    opt_params.iterations = 1'000;
    opt_params.sh_degree_interval = 10'000;
    opt_params.max_cap = 32;
    strategy.initialize(opt_params);
    splat_data.set_frozen_ranges({SplatData::FrozenRange{.start = 0, .count = 1}});

    const auto opac_orig = splat_data.opacity_raw().clone();
    const auto scale_orig = splat_data.scaling_raw().clone();

    const auto expected_raw = [](const float raw, const float decay, const float train_t) {
        const float opac = 1.0f / (1.0f + std::exp(-raw));
        float next = opac - decay * (1.0f - train_t);
        next = std::min(std::max(next, 1e-12f), std::nextafter(1.0f, 0.0f));
        return std::log(next / (1.0f - next));
    };
    const auto expected_log_s = [](const float log_s, const float decay, const float train_t) {
        const float scale = std::exp(log_s) * (1.0f - decay * (1.0f - train_t));
        return std::log(std::max(scale, 1e-12f));
    };

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
        EXPECT_NEAR(o[i], expected_raw(o0[i], opt_params.opacity_decay, 0.5f), 1e-5f) << i;
        EXPECT_NEAR(s[i * 3], expected_log_s(s0[i * 3], opt_params.scale_decay, 0.5f), 1e-5f) << i;
    }
}

TEST_F(MRNFStrategyTest, DensificationInfoShapeIsTwoRows) {
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
}

TEST_F(MRNFStrategyTest, ZeroVisibilityProducesNoGrowth) {
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

    const size_t n = splat_data.size();
    const size_t active = strategy.active_count();
    strategy._refine_weight_max = Tensor::ones({n}, Device::GPU);
    strategy.visibility_accumulator().zero_();
    strategy.grow_and_split(100, 0);
    EXPECT_EQ(strategy.active_count(), active);

    strategy._refine_weight_max = Tensor::ones({n}, Device::GPU);
    strategy.visibility_accumulator().fill_(1.0f);
    strategy.grow_and_split(100, 0);
    EXPECT_GT(strategy.active_count(), active);
}

class MRNFDecayTest : public ::testing::Test {};

TEST_F(MRNFDecayTest, ZeroDecayPreservesFiniteLogitsAndStillDecaysScales) {
    const std::vector<float> original{-80.0f, -20.0f, 0.0f, 16.85f, 20.0f, 80.0f};
    for (const auto [decay, train_t] : {std::pair{0.0f, 0.5f}, std::pair{0.004f, 1.0f}}) {
        auto opacity = Tensor::from_vector(original, {original.size()}, Device::GPU);
        auto scales = Tensor::zeros({original.size(), 3}, Device::GPU);
        for (int step = 0; step < 100; ++step) {
            training_ops(default_gpu_backend()).mrnf->decay(opacity, scales, {}, {.opacity_decay = decay, .scale_decay = 0.01f, .train_t = train_t});
        }
        const auto actual = opacity.cpu().to_vector();
        for (size_t i = 0; i < original.size(); ++i)
            EXPECT_FLOAT_EQ(actual[i], original[i]) << "row " << i;
        const auto actual_scales = scales.cpu().to_vector();
        EXPECT_NEAR(actual_scales[0], 100.0f * std::log(1.0f - 0.01f * (1.0f - train_t)), 1e-5f);
    }
}

TEST_F(MRNFDecayTest, SaturatedAndLegacyInfiniteLogitsStayFinite) {
    const float inf = std::numeric_limits<float>::infinity();
    const std::vector<float> original{16.85f, 20.0f, 80.0f, inf, -inf};
    for (const float decay : {0.0f, 1e-12f, 0.004f}) {
        auto opacity = Tensor::from_vector(original, {original.size()}, Device::GPU);
        auto scales = Tensor::zeros({original.size(), 3}, Device::GPU);
        training_ops(default_gpu_backend()).mrnf->decay(opacity, scales, {}, {.opacity_decay = decay, .scale_decay = 0.0f, .train_t = 0.5f});
        const auto actual = opacity.cpu().to_vector();
        for (size_t i = 0; i < original.size(); ++i) {
            EXPECT_TRUE(std::isfinite(actual[i])) << "row " << i << ", decay " << decay;
            if (decay == 1e-12f && i + 1 < original.size()) {
                const float upper = std::nextafter(1.0f, 0.0f);
                EXPECT_NEAR(actual[i], std::log(upper / (1.0f - upper)), 1e-5f);
            }
            const double expected = i + 1 == original.size() ? 0.0 : 1.0 - decay * 0.5;
            EXPECT_NEAR(1.0 / (1.0 + std::exp(-double(actual[i]))), expected, 2e-7);
        }
    }
}

TEST_F(MRNFDecayTest, FrozenRowsRemainUnchangedWhileNaNsStayVisible) {
    const float inf = std::numeric_limits<float>::infinity();
    auto opacity = Tensor::from_vector(std::vector<float>{inf, std::nanf(""), 20.0f}, {3}, Device::GPU);
    auto scales = Tensor::zeros({3, 3}, Device::GPU);
    const auto frozen = Tensor::from_vector(std::vector<bool>{true, false, false}, {3}, Device::GPU);
    training_ops(default_gpu_backend()).mrnf->decay(opacity, scales, frozen, {.opacity_decay = 0.004f, .scale_decay = 0.01f, .train_t = 0.5f});
    const auto values = opacity.cpu().to_vector();
    EXPECT_EQ(values[0], inf);
    EXPECT_TRUE(std::isnan(values[1]));
    EXPECT_NEAR(1.0 / (1.0 + std::exp(-double(values[2]))), 0.998, 2e-7);
    const auto actual_scales = scales.cpu().to_vector();
    EXPECT_FLOAT_EQ(actual_scales[0], 0.0f);
    EXPECT_LT(actual_scales[6], 0.0f);
}

namespace {
    SplatData create_distinct_mrnf_splat_data(const size_t n) {
        const size_t rest = sh_rest_coefficients_for_degree(3);
        std::vector<float> means(n * 3), sh0(n * 3), scaling(n * 3), rotation(n * 4, 0.0f), opacity(n);
        std::vector<float> shN(n * rest * 3);
        for (size_t i = 0; i < n; ++i) {
            for (size_t c = 0; c < 3; ++c) {
                means[i * 3 + c] = 0.1f * static_cast<float>(i) + 0.01f * static_cast<float>(c);
                sh0[i * 3 + c] = 0.02f * static_cast<float>(i % 17) - 0.1f * static_cast<float>(c);
                scaling[i * 3 + c] = -2.0f + 0.03f * static_cast<float>((i + c) % 11);
            }
            rotation[i * 4 + 0] = 1.0f;
            rotation[i * 4 + 1] = 0.01f * static_cast<float>(i % 7);
            opacity[i] = -1.0f + 0.02f * static_cast<float>(i % 13);
            for (size_t k = 0; k < rest * 3; ++k) {
                shN[i * rest * 3 + k] = 0.001f * static_cast<float>((i * 31 + k * 7) % 97) - 0.05f;
            }
        }
        return SplatData(3,
                         Tensor::from_vector(means, TensorShape({n, 3}), Device::CUDA),
                         Tensor::from_vector(sh0, TensorShape({n, 1, 3}), Device::CUDA),
                         Tensor::from_vector(shN, TensorShape({n, rest, 3}), Device::CUDA),
                         Tensor::from_vector(scaling, TensorShape({n, 3}), Device::CUDA),
                         Tensor::from_vector(rotation, TensorShape({n, 4}), Device::CUDA),
                         Tensor::from_vector(opacity, TensorShape({n, 1}), Device::CUDA),
                         1.0f);
    }
} // namespace

// Large growth events place children chunk by chunk. Fails if a chunk splits
// the wrong parents, a child lands in the wrong free slot or appended row, a
// chunk is skipped, or free-slot reuse does not continue across chunks.
TEST_F(MRNFStrategyTest, ChunkedChildPlacementMatchesSingleChunk) {
    constexpr size_t n = 64;
    auto single_data = create_distinct_mrnf_splat_data(n);
    auto chunked_data = create_distinct_mrnf_splat_data(n);
    MRNF single(single_data);
    MRNF chunked(chunked_data);
    auto opt_params = vanilla_mrnf_params();
    opt_params.iterations = 10'000;
    opt_params.sh_degree_interval = 10'000;
    opt_params.max_cap = 128;
    single.initialize(opt_params);
    chunked.initialize(opt_params);

    const auto free_rows =
        Tensor::from_vector(std::vector<int>{5, 17, 40}, TensorShape({3}), Device::CUDA).to(DataType::Int64);
    for (MRNF* strategy : {&single, &chunked}) {
        strategy->mark_as_free(free_rows);
        strategy->_splat_data->deleted().index_put_(free_rows, Tensor::ones_bool({3}, Device::CUDA));
    }

    const auto parents = Tensor::from_vector(std::vector<int>{0, 2, 3, 7, 9, 11, 20, 21, 30, 33, 50, 60},
                                             TensorShape({12}), Device::CUDA)
                             .to(DataType::Int64);
    const auto [single_reused, single_appended] = single.split_parents_into_children(parents, 12);
    const auto [chunked_reused, chunked_appended] = chunked.split_parents_into_children(parents, 5);

    EXPECT_EQ(single_reused, 3u);
    EXPECT_EQ(single_appended, 9u);
    EXPECT_EQ(chunked_reused, single_reused);
    EXPECT_EQ(chunked_appended, single_appended);
    ASSERT_EQ(chunked_data.size(), single_data.size());
    EXPECT_EQ(chunked.free_count(), single.free_count());

    EXPECT_EQ(chunked_data.means().cpu().to_vector(), single_data.means().cpu().to_vector());
    EXPECT_EQ(chunked_data.sh0().cpu().to_vector(), single_data.sh0().cpu().to_vector());
    EXPECT_EQ(chunked_data.scaling_raw().cpu().to_vector(), single_data.scaling_raw().cpu().to_vector());
    EXPECT_EQ(chunked_data.rotation_raw().cpu().to_vector(), single_data.rotation_raw().cpu().to_vector());
    EXPECT_EQ(chunked_data.opacity_raw().cpu().to_vector(), single_data.opacity_raw().cpu().to_vector());
    EXPECT_EQ(chunked_data.deleted().to(DataType::Float32).cpu().to_vector(),
              single_data.deleted().to(DataType::Float32).cpu().to_vector());

    const auto single_sh = single_data.shN_canonical().cpu().to_vector();
    const auto chunked_sh = chunked_data.shN_canonical().cpu().to_vector();
    ASSERT_EQ(chunked_sh.size(), single_sh.size());
    for (size_t i = 0; i < single_sh.size(); ++i) {
        ASSERT_NEAR(chunked_sh[i], single_sh[i], 1e-4f) << "SH value " << i;
    }
}

TEST_F(MRNFStrategyTest, LongAxisSplitPlacementIgnoresQuaternionMagnitudeOnRealRows) {
    const auto* path = std::getenv("LFS_SPLIT_TEST_MODEL");
    if (!path)
        GTEST_SKIP() << "set LFS_SPLIT_TEST_MODEL to a trained splat model";
    auto document = lfs::io::project::ProjectDocument::open(path);
    ASSERT_TRUE(document.has_value()) << lfs::format_for_developer(document.error());
    Scene scene;
    const auto hydration = document->hydrate(scene);
    ASSERT_TRUE(hydration.has_value()) << lfs::format_for_developer(hydration.error());
    const auto* model_ptr = scene.getCombinedModel();
    ASSERT_NE(model_ptr, nullptr);
    const auto& model = *model_ptr;
    ASSERT_EQ(model.rotation_raw().ndim(), 2);
    ASSERT_EQ(model.rotation_raw().shape()[1], 4);
    ASSERT_EQ(model.scaling_raw().shape(), TensorShape({model.rotation_raw().shape()[0], 3}));
    const auto source_rotations = model.rotation_raw().cpu().to_vector();
    const auto source_scales = model.scaling_raw().cpu().to_vector();
    std::vector<float> rotations, scales, expected_offsets;
    std::vector<float> expected_lengths;
    for (size_t row = 0; row < model.rotation_raw().shape()[0] && expected_lengths.size() < 192; ++row) {
        double squared_norm = 0.0;
        for (int c = 0; c < 4; ++c)
            squared_norm += double(source_rotations[4 * row + c]) * source_rotations[4 * row + c];
        const double norm = std::sqrt(squared_norm);
        if (!std::isfinite(norm) || norm < 1e-6 || std::abs(norm - 1.0) < 0.01)
            continue;
        double q[4];
        for (int c = 0; c < 4; ++c)
            q[c] = source_rotations[4 * row + c] / norm;
        const auto* scale = source_scales.data() + 3 * row;
        const int axis = static_cast<int>(std::max_element(scale, scale + 3) - scale);
        const float length = 0.5f * std::exp(scale[axis]);
        ASSERT_TRUE(std::isfinite(length));
        ASSERT_GT(length, 0.0f);
        for (const float multiplier : {1.0f, static_cast<float>(1.0 / norm), 2.0f}) {
            for (int c = 0; c < 4; ++c)
                rotations.push_back(source_rotations[4 * row + c] * multiplier);
            scales.insert(scales.end(), scale, scale + 3);
            for (int c = 0; c < 3; ++c) {
                const double cross = c == (axis + 1) % 3 ? q[1 + (axis + 2) % 3] : c == (axis + 2) % 3 ? -q[1 + (axis + 1) % 3]
                                                                                                       : 0.0;
                const double direction = (c == axis ? 2.0 * q[0] * q[0] - 1.0 : 0.0) +
                                         2.0 * q[1 + axis] * q[1 + c] + 2.0 * q[0] * cross;
                expected_offsets.push_back(static_cast<float>(length * direction));
            }
            expected_lengths.push_back(length);
        }
    }
    ASSERT_FALSE(expected_lengths.empty()) << "trained model must contain non-unit quaternion rows";
    const size_t n = expected_lengths.size();
    auto positions = Tensor::zeros({n, 3}, Device::CUDA);
    auto quaternions = Tensor::from_vector(rotations, {n, 4}, Device::CUDA);
    auto log_scales = Tensor::from_vector(scales, {n, 3}, Device::CUDA);
    auto sh0 = Tensor::zeros({n, 3}, Device::CUDA);
    auto opacities = Tensor::zeros({n}, Device::CUDA);
    auto children_positions = Tensor::empty({n, 3}, Device::CUDA);
    auto children_rotations = Tensor::empty({n, 4}, Device::CUDA);
    auto children_scales = Tensor::empty({n, 3}, Device::CUDA);
    auto children_sh0 = Tensor::empty({n, 3}, Device::CUDA);
    auto children_opacities = Tensor::empty({n}, Device::CUDA);
    std::vector<int> indices(n);
    for (size_t i = 0; i < n; ++i)
        indices[i] = static_cast<int>(i);
    const auto split_indices = Tensor::from_vector(indices, {n}, Device::CUDA).to(DataType::Int64);
    kernels::launch_long_axis_split_gaussians_inplace(
        positions.ptr<float>(), quaternions.ptr<float>(), log_scales.ptr<float>(),
        sh0.ptr<float>(), nullptr, opacities.ptr<float>(), children_positions.ptr<float>(),
        children_rotations.ptr<float>(), children_scales.ptr<float>(), children_sh0.ptr<float>(),
        nullptr, children_opacities.ptr<float>(), split_indices.ptr<int64_t>(), static_cast<int>(n), 0);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto first = positions.cpu().to_vector();
    const auto second = children_positions.cpu().to_vector();
    EXPECT_EQ(quaternions.cpu().to_vector(), rotations);
    EXPECT_EQ(children_rotations.cpu().to_vector(), rotations);
    for (size_t row = 0; row < n; ++row) {
        SCOPED_TRACE(row);
        const float tolerance = std::max(1e-7f, expected_lengths[row] * 1e-4f);
        float squared_length = 0.0f;
        for (int c = 0; c < 3; ++c) {
            EXPECT_NEAR(first[3 * row + c], expected_offsets[3 * row + c], tolerance);
            EXPECT_NEAR(second[3 * row + c], -expected_offsets[3 * row + c], tolerance);
            squared_length += first[3 * row + c] * first[3 * row + c];
            if (row % 3 != 0)
                EXPECT_NEAR(first[3 * row + c], first[3 * (row - row % 3) + c], tolerance);
        }
        EXPECT_NEAR(std::sqrt(squared_length), expected_lengths[row], tolerance);
    }
}
