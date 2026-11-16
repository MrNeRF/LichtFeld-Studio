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
class MRNFStrategyTest_ApplyDecaySkipsFrozenRows_Test;
class MRNFStrategyTest_DensificationInfoShapeIsTwoRows_Test;
class MRNFStrategyTest_ZeroVisibilityProducesNoGrowth_Test;
class MRNFStrategyTest_DirectAuxiliaryGrowthPreservesPrefix_Test;
class MRNFStrategyTest_EdgeWindowNormalizesViewsAndClosesBeforeRefineBackward_Test;
class MRNFStrategyTest_GrowthExchangeUsesMaxOfImpactRanks_Test;
class MRNFStrategyTest_GrowthExchangeSkipsNewbornsUntilScored_Test;
class MRNFStrategyTest_GrowthExchangeMarksSplitParentsAndChildrenNewborn_Test;
class MRNFStrategyTest_RemoveGaussiansKeepsGrowthExchangeStateAligned_Test;
class MRNFStrategyTest_GrowthExchangeFallsBackToCoverageWithoutImpact_Test;
class MRNFStrategyTest_GrowthExchangeMatchesDemandWithoutDeletingParents_Test;
class MRNFStrategyTest_GrowthExchangeCosineQuotaAtStartMiddleAndEnd_Test;
class MRNFStrategyTest_RayDepthStateFollowsItsSplats_Test;

#include "core/camera.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/cuda/sh_layout.cuh"
#include "core/logger.hpp"
#include "core/memory_pressure.hpp"
#include "core/parameters.hpp"
#include "core/scene.hpp"
#include "core/sh_value_quant.hpp"
#include "core/splat_data.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor/internal/tensor_ops.hpp"
#include "io/formats/ply.hpp"
#include "io/project_document.hpp"
#include "lfs/training/joint_adam_codec.hpp"
#include "lfs/training/live_model_mutation_guard.hpp"
#include "lfs/training/mean_step_scale.cuh"
#include "lfs/training/morton_reorder.hpp"
#include "lfs/training/ray_depth_step.cuh"
#include "lfs/training/row_bias_correction.cuh"
#include "lfs/training/sh_value_codec.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include "training/checkpoint.hpp"
#include "training/dataset.hpp"
#include "training/kernels/densification_kernels.hpp"
#include "training/kernels/image_kernels.hpp"
#include "training/kernels/match_kernels.hpp"
#include "training/kernels/mrnf_kernels.hpp"
#include "training/optimizer/render_output.hpp"
#include "training/rasterization/fast_rasterizer.hpp"
#include "training/strategies/match.hpp"
#include "training/strategies/mrnf.hpp"
#include "training/strategies/strategy_utils.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <tuple>
#include <vector>

using namespace lfs::core;
using namespace lfs::training;

TEST(MRNFStrategyTest, ReplaceParentWeightsMatchesTensorProducts) {
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
    edge_values[6] = -2.0f;

    const auto raw_opacities = Tensor::from_vector(opacity_values, TensorShape({n}), Device::CUDA);
    const auto visibility = Tensor::from_vector(visibility_values, TensorShape({n}), Device::CUDA);
    const auto active = Tensor::from_vector(active_values, TensorShape({n}), Device::CUDA);
    const auto trainable = Tensor::from_vector(trainable_values, TensorShape({n}), Device::CUDA);
    const auto edge = Tensor::from_vector(edge_values, TensorShape({n}), Device::CUDA);
    // Reference: the factor built from a normalized copy of the scores.
    auto edge_factor = edge.clone();
    kernels::launch_normalize_by_positive_median(edge_factor.ptr<float>(), n);
    edge_factor.mul_(0.25f).add_(1.0f);
    const mrnf_strategy::EdgeGuidance guidance{
        .scores = edge.ptr<float>(),
        .median = kernels::launch_select_positive_median(edge.ptr<float>(), n, nullptr),
        .weight = 0.25f,
    };

    for (int options = 0; options < 8; ++options) {
        auto expected = raw_opacities.sigmoid() * (visibility > 0.0f);
        if (options & 1)
            expected = expected * active;
        if (options & 2)
            expected = expected * trainable;
        if (options & 4)
            expected = expected * edge_factor;

        auto actual = Tensor::empty({n}, Device::CUDA);
        mrnf_strategy::launch_replace_parent_weights(
            raw_opacities.ptr<float>(), visibility.ptr<float>(),
            options & 1 ? active.ptr<bool>() : nullptr,
            options & 2 ? trainable.ptr<bool>() : nullptr,
            options & 4 ? guidance : mrnf_strategy::EdgeGuidance{},
            actual.ptr<float>(), n);
        const auto actual_host = actual.cpu();
        const auto expected_host = expected.cpu();
        EXPECT_EQ(std::memcmp(actual_host.ptr<float>(), expected_host.ptr<float>(), n * sizeof(float)), 0)
            << "options=" << options;
    }

    auto scaled = Tensor::full({n}, 3.0f, Device::CUDA);
    mrnf_strategy::launch_apply_edge_guidance(scaled.ptr<float>(), guidance, n);
    const auto scaled_host = scaled.cpu();
    const auto expected_scaled = (Tensor::full({n}, 3.0f, Device::CUDA) * edge_factor).cpu();
    EXPECT_EQ(std::memcmp(scaled_host.ptr<float>(), expected_scaled.ptr<float>(), n * sizeof(float)), 0);
}

// The growth exchange ranks rows by the later of their positions in the two
// importance orders. The radix ranking must give up exactly the rows the
// comparator sorts (Tensor::sort) did, with ties, NaN, -0 and +inf present.
TEST(MRNFStrategyTest, ExchangeRanksMatchComparatorSorts) {
    constexpr size_t n = 300007;
    constexpr size_t count = n / 7;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const std::vector<float> palette{0.0f, -0.0f, 0.25f, 0.5f, 1.0f, 3.0f, inf, nan, 1e-30f};
    std::vector<float> primary(n), secondary(n), eligible(n);
    uint32_t state = 4321u;
    const auto next = [&state] {
        state = state * 1664525u + 1013904223u;
        return state >> 8;
    };
    for (size_t i = 0; i < n; ++i) {
        const uint32_t r = next();
        primary[i] = r % 3 == 0 ? palette[r % palette.size()] : static_cast<float>(r % 997) * 0.01f;
        secondary[i] = palette[next() % palette.size()];
        eligible[i] = next() % 5 != 0 ? 1.0f : 0.0f;
    }
    const auto primary_d = Tensor::from_vector(primary, TensorShape({n}), Device::CUDA);
    const auto secondary_d = Tensor::from_vector(secondary, TensorShape({n}), Device::CUDA);
    const auto eligible_d = Tensor::from_vector(eligible, TensorShape({n}), Device::CUDA) > 0.5f;

    for (const bool with_secondary : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "secondary=" << with_secondary);
        const auto order_of = [&](const Tensor& importance) {
            return importance.masked_fill(eligible_d.logical_not(), inf).sort(0, false).second.to_vector_int64();
        };
        std::vector<int64_t> ranks(n, 0);
        for (const auto* importance : {&primary_d, with_secondary ? &secondary_d : nullptr}) {
            if (importance == nullptr)
                continue;
            const auto order = order_of(*importance);
            for (size_t position = 0; position < n; ++position)
                ranks[order[position]] = std::max<int64_t>(ranks[order[position]], static_cast<int64_t>(position));
        }
        std::vector<int64_t> expected(n);
        for (size_t i = 0; i < n; ++i)
            expected[i] = static_cast<int64_t>(i);
        std::stable_sort(expected.begin(), expected.end(), [&](const int64_t a, const int64_t b) {
            const int64_t rank_a = eligible[a] > 0.5f ? ranks[a] : INT_MAX;
            const int64_t rank_b = eligible[b] > 0.5f ? ranks[b] : INT_MAX;
            return rank_a < rank_b;
        });
        expected.resize(count);

        auto rows = Tensor::empty({count}, Device::CUDA, DataType::Int64);
        mrnf_strategy::launch_rank_exchange_rows(
            primary_d.ptr<float>(), with_secondary ? secondary_d.ptr<float>() : nullptr,
            eligible_d.ptr<bool>(), n, count, rows.ptr<int64_t>());
        EXPECT_EQ(rows.to_vector_int64(), expected);
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

TEST(MRNFStrategyTest, ShNBatchArenaMatchesCatFallbackOnRealScene) {
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

    param::OptimizationParameters vanilla_mrnf_params() {
        auto p = param::OptimizationParameters::mrnf_defaults();
        p.opacity_decay_rendered_only = false;
        p.densify_structure_weight = 0.0f;
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

    auto factors = Tensor::ones({10}, Device::CUDA);
    mrnf_strategy::launch_apply_edge_guidance(factors.ptr<float>(), strategy.edge_guidance(), 10);
    const auto guidance = factors.cpu();
    const float* guidance_ptr = guidance.ptr<float>();

    EXPECT_NEAR(guidance_ptr[2], 1.0f, 1e-5f);
    EXPECT_GT(guidance_ptr[0], 1.0f);
    EXPECT_GT(guidance_ptr[1], guidance_ptr[0]);
}

TEST(MRNFStrategyTest, EdgeWindowNormalizesViewsAndClosesBeforeRefineBackward) {
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
        raw_values, TensorShape({raw_values.size()}), Device::CUDA);

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

TEST(CropDampingStrategyTest, MrnfRejectedRowsAreNotRefineCandidatesAtZeroScale) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);

    auto opt_params = vanilla_mrnf_params();
    opt_params.iterations = 100;
    opt_params.max_cap = 32;
    opt_params.growth_grad_threshold = 0.5f;
    strategy.initialize(opt_params);
    strategy._refine_weight_max = Tensor::ones({10}, Device::CUDA);
    strategy.visibility_accumulator().fill_(1.0f);

    auto crop_mask = Tensor::zeros_bool({10}, Device::CPU);
    crop_mask.ptr<unsigned char>()[0] = 1;
    strategy.get_optimizer().set_crop_score_mask(crop_mask);
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
    strategy.get_optimizer().set_crop_score_mask({});
    const auto unmasked_candidates =
        strategy.compute_refine_candidates().to(DataType::Int32).to_vector_int();
    EXPECT_EQ(unit_scale_candidates, unmasked_candidates);
}

namespace {
    Tensor roi_rows(const std::vector<int>& inside, const size_t n) {
        auto rows = Tensor::zeros_bool({n}, Device::CPU);
        for (const int i : inside)
            rows.ptr<unsigned char>()[i] = 1;
        return rows.cuda();
    }
} // namespace

// Catches growth that ignores the crop box share, or a strict 0 % / 100 % share that still grows on
// the excluded side when its own side runs out of candidates.
TEST(CropDampingStrategyTest, RoiGrowthShareSplitsPicksAndSpillsLeftovers) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);
    auto opt_params = vanilla_mrnf_params();
    opt_params.max_cap = 32;
    strategy.initialize(opt_params);
    auto& optimizer = strategy.get_optimizer();
    const auto weights = Tensor::ones({10}, Device::CUDA);

    EXPECT_FALSE(split_growth_by_roi(optimizer, weights, 10).has_value()) << "no crop box, no split";

    const auto split = [&](const float share, const int64_t k) {
        optimizer.set_crop_roi_rows(roi_rows({0, 1, 2, 3}, 10), share);
        const auto result = split_growth_by_roi(optimizer, weights, k);
        EXPECT_TRUE(result.has_value());
        return std::pair{result->roi_count, result->rest_count};
    };
    EXPECT_EQ(split(0.5f, 4), (std::pair<int64_t, int64_t>{2, 2}));
    // Only four ROI rows: the other four ROI picks spill outside.
    EXPECT_EQ(split(0.8f, 10), (std::pair<int64_t, int64_t>{4, 6}));
    // Strict shares never spill.
    EXPECT_EQ(split(1.0f, 10), (std::pair<int64_t, int64_t>{4, 0}));
    EXPECT_EQ(split(0.0f, 10), (std::pair<int64_t, int64_t>{0, 6}));

    optimizer.set_crop_roi_rows(roi_rows({0, 1, 2, 3}, 10), 1.0f);
    const auto picks = strategy.sample_with_roi_share(weights, 4, 7).cpu().to_vector_int64();
    ASSERT_EQ(picks.size(), 4u);
    for (const auto index : picks)
        EXPECT_LT(index, 4) << "a 100 % share picked a row outside the box";
}

// Catches the crop box split counting NaN and negative weights as candidates (#3806): the sampler then
// read Gumbel scratch indices that compaction never wrote, which picked stale rows or faulted.
TEST(CropDampingStrategyTest, RoiGrowthSplitCountsOnlySelectableWeights) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);
    auto opt_params = vanilla_mrnf_params();
    opt_params.max_cap = 32;
    strategy.initialize(opt_params);
    auto& optimizer = strategy.get_optimizer();
    optimizer.set_crop_roi_rows(roi_rows({0, 1, 2, 3}, 10), 1.0f);

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const auto weights = Tensor::from_vector(
        std::vector<float>{nan, 1.0f, 2.0f, -1.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f}, TensorShape({10}), Device::CUDA);
    const auto split = split_growth_by_roi(optimizer, weights, 8);
    ASSERT_TRUE(split.has_value());
    EXPECT_EQ(split->roi_nnz, 2);
    EXPECT_EQ(split->rest_nnz, 6);
    EXPECT_EQ(split->roi_count, 2);

    auto picks = strategy.sample_with_roi_share(weights, 8, 11).cpu().to_vector_int64();
    std::sort(picks.begin(), picks.end());
    EXPECT_EQ(picks, (std::vector<int64_t>{1, 2}));
}

// Catches MCMC sampling weights that do not give the ROI its share of the probability mass.
TEST(CropDampingStrategyTest, RoiShareSamplingWeightsHoldTheShareOfTheMass) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);
    auto opt_params = vanilla_mrnf_params();
    opt_params.max_cap = 32;
    strategy.initialize(opt_params);
    auto& optimizer = strategy.get_optimizer();
    const auto weights = Tensor::ones({10}, Device::CUDA);

    const auto masses = [&](const std::vector<int>& inside, const float share) {
        optimizer.set_crop_roi_rows(roi_rows(inside, 10), share);
        const auto values = roi_share_sampling_weights(optimizer, weights).cpu().to_vector();
        float roi = 0.0f, rest = 0.0f;
        for (size_t i = 0; i < values.size(); ++i)
            (std::find(inside.begin(), inside.end(), static_cast<int>(i)) != inside.end() ? roi : rest) += values[i];
        return std::pair{roi, rest};
    };
    const auto [roi, rest] = masses({0, 1, 2, 3}, 0.8f);
    EXPECT_NEAR(roi, 0.8f, 1e-5f);
    EXPECT_NEAR(rest, 0.2f, 1e-5f);
    EXPECT_EQ(masses({0, 1, 2, 3}, 1.0f).second, 0.0f);
    EXPECT_EQ(masses({}, 1.0f), (std::pair{0.0f, 0.0f})) << "a strict share with an empty ROI draws nothing";
    EXPECT_GT(masses({}, 0.8f).second, 0.0f) << "an empty ROI hands a partial share to the rest";
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

TEST(MRNFStrategyTest, RefinementPreservesThinSurfacesAndPrunesCollapsedSplats) {
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
        EXPECT_EQ(deleted[i], i >= 3 && i <= 5) << "splat " << i;
    }
}

TEST(MRNFStrategyTest, DistantBackgroundSurvivesRefinement) {
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

    // Bounds for the initial scene are intentionally far behind this row.
    splat_data.means().slice(0, 0, 1).copy_(Tensor::from_vector(
        std::vector<float>{1.0e6f, 1.0e6f, 1.0e6f}, TensorShape({1, 3}), Device::CUDA));
    splat_data._densification_info.zero_();

    RenderOutput render_output;
    strategy.post_backward(10, render_output);

    ASSERT_EQ(splat_data.size(), 8u);
    ASSERT_TRUE(splat_data.deleted().is_valid());
    const auto deleted_cpu = splat_data.deleted().cpu();
    EXPECT_FALSE(deleted_cpu.ptr<bool>()[0]);
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
    auto visibility = strategy.visibility_accumulator();
    visibility.zero_();

    const auto split_idx = Tensor::from_vector(std::vector<int>{0}, TensorShape({1}), Device::CUDA).to(DataType::Int64);
    strategy._refine_weight_max.index_put_(split_idx, Tensor::full({1}, 1.0f, Device::CUDA));
    visibility.index_put_(split_idx, Tensor::full({1}, 1.0f, Device::CUDA));

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
    auto visibility = strategy.visibility_accumulator();
    visibility.zero_();

    const auto split_idx = Tensor::from_vector(std::vector<int>{0}, TensorShape({1}), Device::CUDA).to(DataType::Int64);
    strategy._refine_weight_max.index_put_(split_idx, Tensor::full({1}, 1.0f, Device::CUDA));
    visibility.index_put_(split_idx, Tensor::full({1}, 1.0f, Device::CUDA));

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

TEST(MRNFStrategyTest, DirectAuxiliaryGrowthPreservesPrefix) {
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
        {sfm_points}, sfm_points, Device::CUDA, DataType::Bool);
    const auto direct_float = [](const size_t n, const float value) {
        auto tensor = Tensor::zeros_direct({n}, n, Device::CUDA, DataType::Float32);
        tensor.copy_from(Tensor::full({n}, value, Device::CUDA));
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
    auto visibility = strategy.visibility_accumulator();
    visibility.zero_();

    const auto split_idx = Tensor::from_vector(std::vector<int>{0}, TensorShape({1}), Device::CUDA).to(DataType::Int64);
    strategy._refine_weight_max.index_put_(split_idx, Tensor::full({1}, 1.0f, Device::CUDA));
    visibility.index_put_(split_idx, Tensor::full({1}, 1.0f, Device::CUDA));

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

TEST(MRNFStrategyTest, StepScalingDoesNotScalePOPSpaSteps) {
    auto params = vanilla_mrnf_params();
    params.grow_until_iter = 15000;
    params.popspa_sparsify_steps = 15000;
    params.steps_scaler = 0.5f;

    params.apply_step_scaling();

    EXPECT_EQ(params.grow_until_iter, 7500u);
    EXPECT_EQ(params.popspa_sparsify_steps, 15000);
    EXPECT_EQ(params.refine_every, 82u);
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
    auto visibility = strategy.visibility_accumulator();
    visibility.zero_();

    const auto split_idx = Tensor::from_vector(std::vector<int>{0}, TensorShape({1}), Device::CUDA).to(DataType::Int64);
    strategy._refine_weight_max.index_put_(split_idx, Tensor::full({1}, 1.0f, Device::CUDA));
    visibility.index_put_(split_idx, Tensor::full({1}, 1.0f, Device::CUDA));

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
    auto visibility = strategy.visibility_accumulator();
    visibility.zero_();
    const auto split_index =
        Tensor::from_vector(std::vector<int>{0}, TensorShape({1}), Device::CUDA)
            .to(DataType::Int64);
    strategy._refine_weight_max.index_put_(split_index, Tensor::full({1}, 1.0f, Device::CUDA));
    visibility.index_put_(split_index, Tensor::full({1}, 1.0f, Device::CUDA));

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
    auto visibility = strategy.visibility_accumulator();
    visibility.zero_();

    const auto visible_parent = Tensor::from_vector(std::vector<int>{0}, TensorShape({1}), Device::CUDA).to(DataType::Int64);
    visibility.index_put_(visible_parent, Tensor::full({1}, 1.0f, Device::CUDA));

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
    auto visibility = strategy.visibility_accumulator();
    visibility.zero_();

    const auto split_idx = Tensor::from_vector(std::vector<int>{0}, TensorShape({1}), Device::CUDA).to(DataType::Int64);
    strategy._refine_weight_max.index_put_(split_idx, Tensor::full({1}, 1.0f, Device::CUDA));
    visibility.index_put_(split_idx, Tensor::full({1}, 1.0f, Device::CUDA));

    const size_t initial_size = splat_data.size();
    strategy.grow_and_split(1, 0);

    EXPECT_EQ(splat_data.size(), initial_size);
    EXPECT_EQ(strategy.active_count(), initial_size - 1);
    EXPECT_EQ(strategy.free_count(), 1u);
}

// With per-splat bias correction on, a row's step counter restarts wherever MRNF restarts its
// moments: split parents and appended children, children placed in free slots, and every row
// after a compaction. Other rows keep their counters.
TEST(MRNFStrategyTest, PerSplatBiasCountersRestartWhereMomentsRestart) {
    const auto counters = [](const AdamOptimizer& optimizer) {
        const auto host = optimizer.row_steps().cpu().contiguous();
        const auto* p = host.ptr<std::int32_t>();
        return std::vector<std::int32_t>(p, p + host.numel());
    };
    const auto split_row_zero = [](MRNF& strategy, SplatData& splat_data) {
        strategy._refine_weight_max = Tensor::zeros({static_cast<size_t>(splat_data.size())}, Device::CUDA);
        auto visibility = strategy.visibility_accumulator();
        visibility.zero_();
        const auto split_idx = Tensor::from_vector(std::vector<int>{0}, TensorShape({1}), Device::CUDA).to(DataType::Int64);
        strategy._refine_weight_max.index_put_(split_idx, Tensor::full({1}, 1.0f, Device::CUDA));
        visibility.index_put_(split_idx, Tensor::full({1}, 1.0f, Device::CUDA));
        strategy.grow_and_split(1, 0);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    };
    auto opt_params = vanilla_mrnf_params();
    opt_params.iterations = 10'000;
    opt_params.sh_degree_interval = 10'000;
    opt_params.max_cap = 32;
    opt_params.growth_grad_threshold = 0.5f;
    opt_params.grow_fraction = 1.0f;
    opt_params.grow_until_iter = 10'000;
    opt_params.per_splat_bias_correction = true;

    // Split with an appended child.
    {
        auto splat_data = create_mrnf_test_splat_data();
        MRNF strategy(splat_data);
        strategy.initialize(opt_params);
        auto& optimizer = strategy.get_optimizer();
        ASSERT_TRUE(optimizer.per_splat_bias_correction());
        const size_t initial_size = splat_data.size();
        for (const auto steps : counters(optimizer))
            EXPECT_EQ(steps, kRowStepSaturation);
        split_row_zero(strategy, splat_data);
        ASSERT_EQ(splat_data.size(), initial_size + 1);
        const auto steps = counters(optimizer);
        ASSERT_EQ(steps.size(), initial_size + 1);
        for (size_t row = 0; row < steps.size(); ++row)
            EXPECT_EQ(steps[row], row == 0 || row == initial_size ? 0 : kRowStepSaturation) << row;
    }

    // Split with the child in a free slot, then a compaction.
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);
    strategy.initialize(opt_params);
    auto& optimizer = strategy.get_optimizer();
    const auto free_indices = Tensor::from_vector(std::vector<int>{8, 9}, TensorShape({2}), Device::CUDA).to(DataType::Int64);
    strategy.mark_as_free(free_indices);
    strategy._splat_data->deleted().index_put_(free_indices, Tensor::ones_bool({2}, Device::CUDA));
    const size_t initial_size = splat_data.size();
    split_row_zero(strategy, splat_data);
    ASSERT_EQ(splat_data.size(), initial_size);
    ASSERT_EQ(strategy.free_count(), 1u);
    const auto deleted = strategy._splat_data->deleted().cpu().contiguous();
    const size_t filled = deleted.ptr<bool>()[8] ? 9 : 8;
    auto steps = counters(optimizer);
    ASSERT_EQ(steps.size(), initial_size);
    for (size_t row = 0; row < 8; ++row)
        EXPECT_EQ(steps[row], row == 0 ? 0 : kRowStepSaturation) << row;
    EXPECT_EQ(steps[filled], 0);

    splat_data._densification_info = Tensor::ones({2, static_cast<size_t>(splat_data.size())}, Device::CUDA);
    const auto mask = Tensor::from_vector(
        std::vector<bool>{false, true, false, true, false, false, false, false, false, false},
        TensorShape({10}), Device::CUDA);
    strategy.remove_gaussians(mask);
    ASSERT_LE(splat_data.size(), 8u);
    steps = counters(optimizer);
    ASSERT_EQ(steps.size(), splat_data.size());
    for (const auto s : steps)
        EXPECT_EQ(s, 0);
}

// A refine that runs out of VRAM at any allocation must be undone to the rows
// it started from, and the next refine must grow that model cleanly.
TEST(MRNFStrategyTest, RefineOutOfVramIsUndoneToTheRowsItStartedFrom) {
    auto& pressure = lfs::core::MemoryPressureCoordinator::instance();
    const auto prepare = [](MRNF& strategy, SplatData& splat) {
        strategy.ensure_densification_info_shape();
        strategy._refine_weight_max = Tensor::full({static_cast<size_t>(splat.size())}, 1.0f, Device::CUDA);
        auto visibility = strategy.visibility_accumulator();
        visibility.fill_(1.0f);
    };
    const auto expect_whole = [](MRNF& strategy, SplatData& splat) {
        const size_t n = splat.size();
        for (const auto* rows : {&splat.sh0(), &splat.scaling_raw(), &splat.rotation_raw(), &splat.opacity_raw()})
            EXPECT_EQ(rows->size(0), n);
        EXPECT_TRUE(!splat.has_deleted_mask() || splat.deleted_mask_matches_size());
        // The free mask spans the reserved capacity, not just the live rows.
        EXPECT_GE(strategy._free_mask.numel(), n);
        for (const auto type : {ParamType::Means, ParamType::Sh0, ParamType::Scaling, ParamType::Rotation,
                                ParamType::Opacity}) {
            const auto* state = strategy.get_optimizer().get_state(type);
            ASSERT_NE(state, nullptr);
            EXPECT_EQ(state->size, n) << static_cast<int>(type);
        }
    };

    bool completed = false;
    int undone = 0;
    for (int allowed = 0; !completed && allowed < 256; ++allowed) {
        SCOPED_TRACE(::testing::Message() << "allocations allowed: " << allowed);
        auto splat = create_mrnf_test_splat_data(300);
        MRNF strategy(splat);
        auto params = vanilla_mrnf_params();
        params.iterations = 10'000;
        params.sh_degree_interval = 10'000;
        params.max_cap = 1000;
        params.growth_grad_threshold = 0.5f;
        params.grow_fraction = 1.0f;
        params.grow_until_iter = 10'000;
        strategy.initialize(params);
        prepare(strategy, splat);
        const size_t before = splat.size();
        const auto shN_before = splat.shN().numel();
        const RefineRowSnapshot rows_before(
            splat, strategy.get_optimizer(),
            [&](const std::function<void(Tensor&)>& visit) { strategy.for_each_gaussian_row(visit); });

        int calls = 0;
        pressure.set_allocation_probe([&](const lfs::core::MemoryDomain domain, size_t) {
            return domain == lfs::core::MemoryDomain::CudaDevice && ++calls > allowed;
        });
        bool failed = false;
        try {
            strategy.grow_and_split(1, 0);
        } catch (const lfs::core::MemoryAllocationError&) {
            failed = true;
        }
        pressure.set_allocation_probe(nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        if (failed) {
            ++undone;
            ASSERT_TRUE(rows_before.restore());
            EXPECT_EQ(splat.size(), before);
            EXPECT_EQ(splat.shN().numel(), shN_before);
        } else {
            completed = true;
            EXPECT_GT(splat.size(), before);
        }
        expect_whole(strategy, splat);

        const size_t grown_from = splat.size();
        prepare(strategy, splat);
        ASSERT_NO_THROW(strategy.grow_and_split(2, 0));
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_GT(splat.size(), grown_from);
        expect_whole(strategy, splat);
    }
    EXPECT_TRUE(completed);
    EXPECT_GT(undone, 0);
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
    EXPECT_EQ(restored.visibility_accumulator().numel(), 12u);
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
    strategy._exchange_importance_sum = Tensor::ones({splat_data.size()}, Device::CUDA);
    strategy._exchange_importance_peak = Tensor::ones({splat_data.size()}, Device::CUDA);
    strategy._exchange_birth = Tensor::zeros({splat_data.size()}, Device::CUDA);
    strategy._exchange_windows = 3;
    strategy._exchange_first_iter = 100;

    auto resumed_params = opt_params;
    resumed_params.iterations = 3000;
    resumed_params.means_lr_end = 1.0e-7f;
    resumed_params.scaling_lr_end = 1.0e-4f;
    resumed_params.gut = !opt_params.gut;

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
    EXPECT_FALSE(strategy._exchange_importance_sum.is_valid());
    EXPECT_FALSE(strategy._exchange_importance_peak.is_valid());
    EXPECT_FALSE(strategy._exchange_birth.is_valid());
    EXPECT_EQ(strategy._exchange_windows, 0);
    EXPECT_EQ(strategy._exchange_first_iter, -1);
}

namespace {
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
} // namespace

TEST(MRNFStrategyTest, ApplyDecaySkipsFrozenRows) {
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

TEST(MRNFStrategyTest, DensificationInfoShapeIsTwoRows) {
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

TEST(MRNFStrategyTest, LateLrAnnealDecaysOpacityAndColorAfterGrowth) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);
    auto opt_params = vanilla_mrnf_params();
    opt_params.iterations = 200;
    opt_params.grow_until_iter = 100;
    opt_params.late_lr_anneal = 0.25f;
    strategy.initialize(opt_params);

    auto& optimizer = *strategy._optimizer;
    const double opacity = optimizer.get_param_lr(ParamType::Opacity);
    const double sh0 = optimizer.get_param_lr(ParamType::Sh0);
    const double shn = optimizer.get_param_lr(ParamType::ShN);
    const double rotation = optimizer.get_param_lr(ParamType::Rotation);

    strategy.apply_late_lr_anneal(99);
    EXPECT_EQ(optimizer.get_param_lr(ParamType::Opacity), opacity);

    strategy.apply_late_lr_anneal(150);
    EXPECT_NEAR(optimizer.get_param_lr(ParamType::Opacity), opacity * 0.5, opacity * 1e-12);
    EXPECT_NEAR(optimizer.get_param_lr(ParamType::Sh0), sh0 * 0.5, sh0 * 1e-12);
    EXPECT_NEAR(optimizer.get_param_lr(ParamType::ShN), shn * 0.5, shn * 1e-12);
    EXPECT_EQ(optimizer.get_param_lr(ParamType::Rotation), rotation);

    strategy.apply_late_lr_anneal(200);
    EXPECT_NEAR(optimizer.get_param_lr(ParamType::Opacity), opacity * 0.25, opacity * 1e-12);

    opt_params.late_lr_anneal = 1.0f;
    strategy.set_optimization_params(opt_params);
    strategy.apply_late_lr_anneal(150);
    EXPECT_EQ(optimizer.get_param_lr(ParamType::Opacity), opacity);
    EXPECT_EQ(optimizer.get_param_lr(ParamType::Sh0), sh0);
    EXPECT_EQ(optimizer.get_param_lr(ParamType::ShN), shn);
}

TEST(MRNFStrategyTest, ZeroVisibilityProducesNoGrowth) {
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
    strategy._refine_weight_max = Tensor::ones({n}, Device::CUDA);
    strategy.visibility_accumulator().zero_();
    strategy.grow_and_split(100, 0);
    EXPECT_EQ(strategy.active_count(), active);

    strategy._refine_weight_max = Tensor::ones({n}, Device::CUDA);
    strategy.visibility_accumulator().fill_(1.0f);
    strategy.grow_and_split(100, 0);
    EXPECT_GT(strategy.active_count(), active);
}

namespace {
    param::OptimizationParameters growth_exchange_test_params() {
        auto opt = vanilla_mrnf_params();
        opt.iterations = 10'000;
        opt.sh_degree_interval = 10'000;
        opt.max_cap = 32;
        opt.grow_fraction = 1.0f;
        opt.grow_until_iter = 10'000;
        opt.growth_exchange = true;
        return opt;
    }

    const std::vector<float> kExchangeSum{1000.0f, 0.1f, 0.5f, 0.6f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f};
    const std::vector<float> kExchangePeak{0.01f, 0.95f, 0.02f, 0.03f, 0.40f, 0.41f, 0.42f, 0.43f, 0.44f, 0.45f};

    std::vector<int64_t> selected_rows(const Tensor& indices) {
        if (!indices.is_valid())
            return {};
        const auto cpu = indices.cpu();
        const auto* rows = cpu.ptr<int64_t>();
        std::vector<int64_t> result(rows, rows + cpu.numel());
        std::sort(result.begin(), result.end());
        return result;
    }
} // namespace

TEST(MRNFStrategyTest, GrowthExchangeUsesMaxOfImpactRanks) {
    const auto run = [](const std::vector<float>& primary, const std::vector<float>* secondary) {
        auto splat_data = create_mrnf_test_splat_data();
        MRNF strategy(splat_data);
        strategy.initialize(growth_exchange_test_params());
        const size_t n = splat_data.size();
        strategy._exchange_importance_sum = Tensor::from_vector(primary, TensorShape({n}), Device::CUDA);
        if (secondary)
            strategy._exchange_importance_peak = Tensor::from_vector(*secondary, TensorShape({n}), Device::CUDA);
        strategy._exchange_birth = Tensor::zeros({n}, Device::CUDA);
        strategy._exchange_windows = strategy.exchange_warmup_windows() + 1;
        return selected_rows(strategy.select_exchange_victims(2));
    };

    EXPECT_EQ(run(kExchangeSum, &kExchangePeak), (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(run(kExchangeSum, nullptr), (std::vector<int64_t>{1, 2}));
    EXPECT_EQ(run(kExchangePeak, nullptr), (std::vector<int64_t>{0, 2}));
}

TEST(MRNFStrategyTest, GrowthExchangeSkipsNewbornsUntilScored) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);
    strategy.initialize(growth_exchange_test_params());

    const size_t n = splat_data.size();
    strategy._exchange_importance_sum = Tensor::from_vector(kExchangeSum, TensorShape({n}), Device::CUDA);
    strategy._exchange_importance_peak = Tensor::from_vector(kExchangePeak, TensorShape({n}), Device::CUDA);
    strategy._exchange_birth = Tensor::zeros({n}, Device::CUDA);
    strategy._exchange_windows = strategy.exchange_warmup_windows() + 1;

    const auto filled = Tensor::from_vector(std::vector<int>{2, 3}, TensorShape({2}), Device::CUDA).to(DataType::Int64);
    strategy._exchange_free_before_growth = Tensor::zeros_bool({n}, Device::CUDA);
    strategy._exchange_free_before_growth.index_put_(filled, Tensor::ones_bool({2}, Device::CUDA));
    strategy.record_exchange_births();

    EXPECT_EQ(selected_rows(strategy.select_exchange_victims(2)),
              (std::vector<int64_t>{4, 5}));

    strategy._exchange_windows += strategy.exchange_warmup_windows() + 1;
    EXPECT_EQ(selected_rows(strategy.select_exchange_victims(2)),
              (std::vector<int64_t>{2, 3}));
}

TEST(MRNFStrategyTest, GrowthExchangeMarksSplitParentsAndChildrenNewborn) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);
    strategy.initialize(growth_exchange_test_params());

    const size_t n = splat_data.size();
    strategy._exchange_importance_sum = Tensor::from_vector(kExchangeSum, TensorShape({n}), Device::CUDA);
    strategy._exchange_importance_peak = Tensor::from_vector(kExchangePeak, TensorShape({n}), Device::CUDA);
    strategy._exchange_birth = Tensor::zeros({n}, Device::CUDA);
    strategy._exchange_windows = strategy.exchange_warmup_windows() + 3;
    strategy._exchange_free_before_growth = Tensor::zeros_bool({n}, Device::CUDA);
    const auto children = Tensor::from_vector(std::vector<int>{2, 3}, TensorShape({2}), Device::CUDA).to(DataType::Int64);
    strategy._exchange_free_before_growth.index_put_(children, Tensor::ones_bool({2}, Device::CUDA));
    const auto parents = Tensor::from_vector(std::vector<int>{0}, TensorShape({1}), Device::CUDA).to(DataType::Int64);

    strategy.record_exchange_births(parents);

    const auto sum = strategy._exchange_importance_sum.cpu().to_vector();
    const auto peak = strategy._exchange_importance_peak.cpu().to_vector();
    const auto birth = strategy._exchange_birth.cpu().to_vector();
    for (const size_t row : {0u, 2u, 3u}) {
        EXPECT_EQ(sum[row], 0.0f);
        EXPECT_EQ(peak[row], 0.0f);
        EXPECT_EQ(birth[row], static_cast<float>(strategy._exchange_windows));
    }
    EXPECT_EQ(sum[1], kExchangeSum[1]);
}

TEST(MRNFStrategyTest, RemoveGaussiansKeepsGrowthExchangeStateAligned) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);
    strategy.initialize(growth_exchange_test_params());
    splat_data._densification_info = Tensor::ones({2, static_cast<size_t>(splat_data.size())}, Device::CUDA);

    const size_t n = splat_data.size();
    strategy._exchange_importance_sum = Tensor::from_vector(kExchangeSum, TensorShape({n}), Device::CUDA);
    strategy._exchange_importance_peak = Tensor::from_vector(kExchangePeak, TensorShape({n}), Device::CUDA);
    strategy._exchange_birth = Tensor::zeros({n}, Device::CUDA);
    std::vector<float> impact_sum(n);
    std::vector<float> impact_peak(n);
    std::iota(impact_sum.begin(), impact_sum.end(), 20.0f);
    std::iota(impact_peak.begin(), impact_peak.end(), 40.0f);
    strategy._exchange_window_impact_sum = Tensor::from_vector(impact_sum, TensorShape({n}), Device::CUDA);
    strategy._exchange_window_impact_peak = Tensor::from_vector(impact_peak, TensorShape({n}), Device::CUDA);
    strategy._exchange_impact_written = true;
    strategy._exchange_windows = strategy.exchange_warmup_windows() + 1;

    const auto mask = Tensor::from_vector(
        std::vector<bool>{true, false, false, false, false, false, false, false, false, true},
        TensorShape({10}),
        Device::CUDA);
    strategy.remove_gaussians(mask);

    ASSERT_EQ(splat_data.size(), 8u);
    ASSERT_EQ(strategy._exchange_importance_sum.numel(), 8u);
    ASSERT_EQ(strategy._exchange_importance_peak.numel(), 8u);
    ASSERT_EQ(strategy._exchange_birth.numel(), 8u);
    ASSERT_EQ(strategy._exchange_window_impact_sum.numel(), 8u);
    ASSERT_EQ(strategy._exchange_window_impact_peak.numel(), 8u);
    EXPECT_TRUE(strategy._exchange_impact_written);
    EXPECT_EQ(strategy._exchange_importance_sum.cpu().to_vector(),
              std::vector<float>(kExchangeSum.begin() + 1, kExchangeSum.end() - 1));
    EXPECT_EQ(strategy._exchange_window_impact_sum.cpu().to_vector(),
              std::vector<float>(impact_sum.begin() + 1, impact_sum.end() - 1));
    EXPECT_EQ(strategy._exchange_window_impact_peak.cpu().to_vector(),
              std::vector<float>(impact_peak.begin() + 1, impact_peak.end() - 1));
    EXPECT_EQ(selected_rows(strategy.select_exchange_victims(2)),
              (std::vector<int64_t>{1, 2}));
}

TEST(MRNFStrategyTest, GrowthExchangeFallsBackToCoverageWithoutImpact) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);
    strategy.initialize(growth_exchange_test_params());

    const size_t n = splat_data.size();
    strategy.visibility_accumulator().copy_(
        Tensor::from_vector(kExchangeSum, TensorShape({n}), Device::CUDA));
    strategy._exchange_window_impact_sum = Tensor::zeros({n}, Device::CUDA);
    strategy._exchange_window_impact_peak = Tensor::zeros({n}, Device::CUDA);
    strategy._exchange_impact_written = false;
    strategy.update_exchange_importance();

    EXPECT_EQ(strategy._exchange_importance_sum.cpu().to_vector(), kExchangeSum);
    EXPECT_FALSE(strategy._exchange_importance_peak.is_valid());
    EXPECT_FALSE(strategy._exchange_window_impact_sum.is_valid());
    EXPECT_FALSE(strategy._exchange_window_impact_peak.is_valid());
    strategy._exchange_windows = strategy.exchange_warmup_windows() + 1;
    EXPECT_EQ(selected_rows(strategy.select_exchange_victims(2)),
              (std::vector<int64_t>{1, 2}));
}

TEST(MRNFStrategyTest, GrowthExchangeMatchesDemandWithoutDeletingParents) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);
    auto params = growth_exchange_test_params();
    params.max_cap = static_cast<int>(splat_data.size());
    params.grow_fraction = 0.3f;
    strategy.initialize(params);

    const size_t n = splat_data.size();
    strategy._exchange_importance_sum = Tensor::from_vector(kExchangeSum, TensorShape({n}), Device::CUDA);
    strategy._exchange_importance_peak = Tensor::from_vector(kExchangePeak, TensorShape({n}), Device::CUDA);
    strategy._exchange_birth = Tensor::zeros({n}, Device::CUDA);
    strategy._exchange_windows = strategy.exchange_warmup_windows() + 1;
    strategy._refine_weight_max = Tensor::ones({n}, Device::CUDA);
    strategy.visibility_accumulator().fill_(1.0f);

    const auto pruned = Tensor::from_vector(std::vector<int>{9}, TensorShape({1}), Device::CUDA).to(DataType::Int64);
    strategy.soft_delete_rows(pruned);
    const auto result = strategy.grow_and_split(100, 1, Tensor::ones_bool({n}, Device::CUDA), 2);

    ASSERT_EQ(result.replacement_count, 1);
    ASSERT_EQ(result.growth_count, 2);
    ASSERT_EQ(result.exchange_victims, 2u);
    ASSERT_TRUE(result.split_indices.is_valid());
    ASSERT_TRUE(result.exchange_victim_indices.is_valid());
    EXPECT_LE(result.exchange_victims, static_cast<size_t>(result.growth_count));
    const auto parents = selected_rows(result.split_indices);
    const auto victims = selected_rows(result.exchange_victim_indices);
    std::vector<int64_t> overlap;
    std::set_intersection(parents.begin(), parents.end(), victims.begin(), victims.end(),
                          std::back_inserter(overlap));
    EXPECT_TRUE(overlap.empty());
    EXPECT_EQ(strategy.active_count(), n);
}

TEST(MRNFStrategyTest, GrowthExchangeCosineQuotaAtStartMiddleAndEnd) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);
    auto params = growth_exchange_test_params();
    params.grow_until_iter = 1000;
    strategy.initialize(params);
    strategy._exchange_first_iter = 200;

    // 4% of 1010 active splats at the first exchange, 2.25% halfway, 0.5% at grow_until_iter.
    EXPECT_EQ(strategy.exchange_quota(200, 1010), 40u);
    EXPECT_EQ(strategy.exchange_quota(600, 1010), 22u);
    EXPECT_EQ(strategy.exchange_quota(1000, 1010), 5u);
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

    param::OptimizationParameters mean_step_test_params() {
        auto opt = vanilla_mrnf_params();
        opt.iterations = 10'000;
        opt.sh_degree_interval = 10'000;
        opt.max_cap = 32;
        opt.means_lr = 0.1f;
        opt.means_lr_end = 0.1f;
        return opt;
    }
} // namespace

TEST(MRNFStrategyTest, PerSplatMeanStepScalesWithExtentAndClamps) {
    auto splat_p = create_mrnf_test_splat_data(8);
    place_deep_far_probe(splat_p);
    auto splat_g = create_mrnf_test_splat_data(8);
    place_deep_far_probe(splat_g);
    MRNF per_splat(splat_p);
    MRNF global(splat_g);
    per_splat.initialize(mean_step_test_params());
    global.initialize(mean_step_test_params());
    global.get_optimizer().set_per_splat_mean_step(false, 0.0f);
    install_test_camera_hull(per_splat);
    per_splat.refresh_far_field_mask(static_cast<size_t>(splat_p.size()));
    per_splat.sync_mean_learning_rate();
    ASSERT_TRUE(per_splat._bounds_valid);
    ASSERT_TRUE(per_splat._median_splat_extent_valid);
    const float med_e = per_splat._median_splat_extent;
    ASSERT_GT(med_e, 0.0f);
    const float r_min = kPerSplatMeanStepRatioMin;
    const float r_max = kPerSplatMeanStepRatioMax;
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

TEST(MRNFStrategyTest, MortonReorderKeepsFarMaskPublished) {
    auto splat = create_mrnf_test_splat_data(4, 0);
    auto params = vanilla_mrnf_params();
    params.max_cap = 8;
    MRNF strategy(splat);
    strategy.initialize(params);
    strategy._camera_hull_valid = true;
    strategy._far_field_mask = Tensor::from_vector(
                                   std::vector<int>{0, 1, 0, 1}, TensorShape({4}), Device::CUDA)
                                   .to(DataType::Bool);
    strategy.publish_mean_step_far_mask();
    auto& optimizer = strategy.get_optimizer();
    ASSERT_EQ(optimizer.mean_step_far_mask(), strategy._far_field_mask.ptr<bool>());
    ASSERT_EQ(optimizer.mean_step_far_mask_n(), 4);
    const auto before = strategy._far_field_mask.cpu();

    const auto result = morton::apply_morton_reorder(
        splat, &optimizer, nullptr, [&](const morton::RowVisitor& visit) { strategy.for_each_gaussian_row(visit); });
    ASSERT_TRUE(result.applied);

    EXPECT_EQ(optimizer.mean_step_far_mask(), strategy._far_field_mask.ptr<bool>());
    EXPECT_EQ(optimizer.mean_step_far_mask_n(), strategy._far_field_mask.numel());
    const auto permutation = result.permutation.cpu();
    const auto reordered = strategy._far_field_mask.cpu();
    for (size_t i = 0; i < 4; ++i)
        EXPECT_EQ(reordered.ptr<bool>()[i], before.ptr<bool>()[permutation.ptr<int64_t>()[i]]) << i;

    // A hull refresh with no cameras must clear the borrowed pointer immediately.
    strategy.refresh_camera_hull();
    EXPECT_EQ(optimizer.mean_step_far_mask(), nullptr);
    EXPECT_EQ(optimizer.mean_step_far_mask_n(), 0);
}

// Row r of the ray-depth state is tagged with the x coordinate of splat r; a reorder, a prune and a split must keep
// the tag on its splat, and children take their parent's anchor and spread with fresh moments.
TEST(MRNFStrategyTest, RayDepthStateFollowsItsSplats) {
    constexpr int n = 10;
    constexpr int stride = ray_depth::kStateStride;
    auto splat = create_mrnf_test_splat_data(n, 0);
    {
        std::vector<float> reversed(3 * n, 0.0f);
        for (int i = 0; i < n; ++i)
            reversed[3 * i] = static_cast<float>(n - 1 - i);
        splat.means().copy_from(Tensor::from_vector(reversed, TensorShape({size_t{n}, size_t{3}}), Device::CUDA));
    }
    auto params = vanilla_mrnf_params();
    params.iterations = 10'000;
    params.sh_degree_interval = 10'000;
    params.max_cap = 32;
    params.growth_grad_threshold = 0.5f;
    params.grow_fraction = 1.0f;
    params.grow_until_iter = 10'000;
    MRNF strategy(splat);
    strategy.initialize(params);
    auto& optimizer = strategy.get_optimizer();
    const auto expect_bound = [&] {
        ASSERT_EQ(strategy._ray_depth_state.shape()[0], static_cast<size_t>(splat.size()));
        EXPECT_EQ(optimizer.ray_depth_state().ptr<float>(), strategy._ray_depth_state.ptr<float>());
        EXPECT_EQ(optimizer.ray_depth_state().shape()[0], static_cast<size_t>(splat.size()));
    };
    ASSERT_TRUE(strategy._ray_depth_state.is_valid());
    expect_bound();
    for (const float value : strategy._ray_depth_state.cpu().to_vector())
        ASSERT_EQ(value, ray_depth::kUnobserved);

    const auto tag = [](const float x, const int k) { return x + 100.0f * static_cast<float>(k) + 1.0f; };
    {
        const auto means = splat.means().cpu().to_vector();
        std::vector<float> rows(n * stride);
        for (int r = 0; r < n; ++r)
            for (int k = 0; k < stride; ++k)
                rows[r * stride + k] = tag(means[3 * r], k);
        strategy._ray_depth_state.copy_from(
            Tensor::from_vector(rows, TensorShape({size_t{n}, size_t{stride}}), Device::CUDA));
    }
    const auto expect_tags = [&](const char* stage) {
        SCOPED_TRACE(stage);
        expect_bound();
        const auto means = splat.means().cpu().to_vector();
        const auto rows = strategy._ray_depth_state.cpu().to_vector();
        for (size_t r = 0; r < static_cast<size_t>(splat.size()); ++r)
            for (int k = 0; k < stride; ++k)
                ASSERT_EQ(rows[r * stride + k], tag(means[3 * r], k)) << "row " << r << ", column " << k;
    };

    const auto reorder = morton::apply_morton_reorder(
        splat, &optimizer, nullptr, [&](const morton::RowVisitor& visit) { strategy.for_each_gaussian_row(visit); });
    ASSERT_TRUE(reorder.applied);
    ASSERT_NE(reorder.permutation.cpu().ptr<int64_t>()[0], 0) << "the reorder must move rows";
    expect_tags("reorder");

    const auto keep = Tensor::from_vector(std::vector<int>{1, 1, 0, 1, 1, 0, 1, 1, 1, 1}, TensorShape({size_t{n}}),
                                          Device::CUDA)
                          .to(DataType::Bool);
    strategy.compact_splats(keep);
    ASSERT_EQ(splat.size(), n - 2);
    expect_tags("prune");

    // Split rows 0 and 1: one child takes the freed row 7, the other is appended.
    const auto rows_before = strategy._ray_depth_state.cpu().to_vector();
    const auto freed = Tensor::from_vector(std::vector<int>{7}, TensorShape({1}), Device::CUDA).to(DataType::Int64);
    strategy.mark_as_free(freed);
    splat.deleted().index_put_(freed, Tensor::ones_bool({1}, Device::CUDA));
    strategy._refine_weight_max = Tensor::zeros({static_cast<size_t>(splat.size())}, Device::CUDA);
    auto visibility = strategy.visibility_accumulator();
    visibility.zero_();
    const auto parents = Tensor::from_vector(std::vector<int>{0, 1}, TensorShape({2}), Device::CUDA).to(DataType::Int64);
    strategy._refine_weight_max.index_put_(parents, Tensor::full({2}, 1.0f, Device::CUDA));
    visibility.index_put_(parents, Tensor::full({2}, 1.0f, Device::CUDA));
    const size_t size_before = splat.size();
    strategy.grow_and_split(1, 0);
    ASSERT_EQ(splat.size(), size_before + 1);
    strategy.sync_ray_depth();
    expect_bound();
    const auto rows = strategy._ray_depth_state.cpu().to_vector();
    const auto row = [&](const std::vector<float>& v, const size_t r, const int k) { return v[r * stride + k]; };
    for (const size_t parent : {size_t{0}, size_t{1}}) {
        for (int k = 0; k < 4; ++k)
            EXPECT_EQ(row(rows, parent, k), row(rows_before, parent, k)) << "parent " << parent;
        EXPECT_EQ(row(rows, parent, 4), 0.0f);
        EXPECT_EQ(row(rows, parent, 5), 0.0f);
    }
    for (const size_t child : {size_t{7}, size_before}) {
        const size_t parent = row(rows, child, 0) == row(rows_before, 0, 0) ? 0 : 1;
        for (int k = 0; k < 4; ++k)
            EXPECT_EQ(row(rows, child, k), row(rows_before, parent, k)) << "child " << child;
        EXPECT_EQ(row(rows, child, 4), 0.0f);
        EXPECT_EQ(row(rows, child, 5), 0.0f);
    }
    EXPECT_NE(row(rows, 7, 0), row(rows, size_before, 0)) << "each parent has one child";
    for (size_t r = 2; r < size_before; ++r) {
        if (r == 7)
            continue;
        for (int k = 0; k < stride; ++k)
            EXPECT_EQ(row(rows, r, k), row(rows_before, r, k)) << "untouched row " << r;
    }

    auto off = vanilla_mrnf_params();
    off.ray_depth_lr = 0.0f;
    auto other = create_mrnf_test_splat_data(4, 0);
    MRNF disabled(other);
    disabled.initialize(off);
    EXPECT_FALSE(disabled._ray_depth_state.is_valid());
    EXPECT_FALSE(disabled.get_optimizer().ray_depth_state().is_valid());
}

TEST(MRNFStrategyTest, SetTrainingDatasetBeforeInitialize) {
    auto splat_data = create_mrnf_test_splat_data(0);
    MRNF strategy(splat_data);

    // Trainer wires the dataset before initialize() binds params and creates the optimizer.
    EXPECT_NO_THROW(strategy.set_training_dataset(make_hull_dataset()));
}

TEST(MRNFStrategyTest, HardRemovalRepublishesFarMaskForDegenerateModel) {
    auto splat = create_mrnf_test_splat_data(4, 0);
    auto params = vanilla_mrnf_params();
    params.max_cap = 8;
    MRNF strategy(splat);
    strategy.initialize(params);
    install_test_camera_hull(strategy);
    auto& optimizer = strategy.get_optimizer();
    ASSERT_TRUE(optimizer.per_splat_mean_step());
    ASSERT_NE(optimizer.mean_step_far_mask(), nullptr);
    ASSERT_EQ(optimizer.mean_step_far_mask_n(), 4);

    // Keep the far row, whose label differs from the old first row.
    const auto remove = Tensor::from_vector(
                            std::vector<int>{1, 1, 1, 0}, TensorShape({4}), Device::CUDA)
                            .to(DataType::Bool);
    strategy.remove_gaussians(remove);
    ASSERT_EQ(splat.size(), 1);
    ASSERT_NE(optimizer.mean_step_far_mask(), nullptr);
    ASSERT_EQ(optimizer.mean_step_far_mask_n(), splat.means().shape()[0]);
    bool is_far = false;
    ASSERT_EQ(cudaMemcpy(&is_far, optimizer.mean_step_far_mask(), sizeof(is_far),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_TRUE(is_far);

    optimizer.get_grad(ParamType::Means).fill_(0.2f);
    EXPECT_NO_THROW(optimizer.step(1));
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(optimizer.mean_step_far_mask_n(), splat.means().shape()[0]);
    EXPECT_LT(splat.means().cpu().ptr<float>()[0], 3.0f);
    const auto fused = optimizer.prepare_fastgs_fused_adam(2, nullptr);
    EXPECT_EQ(fused.mean_step_far_mask, optimizer.mean_step_far_mask());
    EXPECT_EQ(fused.mean_step_far_mask_n, 1);
}

TEST(MRNFStrategyTest, DeserializeRepublishesFarMaskWithDegenerateBounds) {
    auto params = vanilla_mrnf_params();
    params.max_cap = 8;
    auto source_splat = create_mrnf_test_splat_data(1, 0);
    source_splat.means().fill_(3.0f);
    MRNF source(source_splat);
    source.initialize(params);
    install_test_camera_hull(source);
    std::stringstream checkpoint;
    source_splat.serialize(checkpoint);
    source.serialize(checkpoint);
    checkpoint.seekg(0);

    auto splat = create_mrnf_test_splat_data(1, 0);
    MRNF restored(splat);
    restored.initialize(params);
    install_test_camera_hull(restored);
    auto& optimizer = restored.get_optimizer();
    ASSERT_NE(optimizer.mean_step_far_mask(), nullptr);
    bool is_far = true;
    ASSERT_EQ(cudaMemcpy(&is_far, optimizer.mean_step_far_mask(), sizeof(is_far),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_FALSE(is_far);

    splat.deserialize(checkpoint);
    restored.deserialize(checkpoint);
    ASSERT_NE(optimizer.mean_step_far_mask(), nullptr);
    ASSERT_EQ(optimizer.mean_step_far_mask_n(), splat.means().shape()[0]);
    ASSERT_EQ(cudaMemcpy(&is_far, optimizer.mean_step_far_mask(), sizeof(is_far),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_TRUE(is_far);
    optimizer.get_grad(ParamType::Means).fill_(0.2f);
    EXPECT_NO_THROW(optimizer.step(1));
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

namespace {
    class FarMaskWarningCapture {
    public:
        FarMaskWarningCapture() : previous_level_(Logger::get().level()) {
            Logger::get().set_level(LogLevel::Warn);
            token_ = Logger::get().add_log_handler(
                [this](LogLevel level, const SourceSite&, std::string_view message) {
                    if (level == LogLevel::Warn &&
                        message.find("mean_step_far_mask row-count mismatch") != std::string_view::npos) {
                        messages.emplace_back(message);
                    }
                });
        }
        ~FarMaskWarningCapture() {
            Logger::get().remove_log_handler(token_);
            Logger::get().set_level(previous_level_);
        }
        std::vector<std::string> messages;

    private:
        LogLevel previous_level_;
        LogHandlerToken token_;
    };
} // namespace

TEST(MRNFStrategyTest, MeanStepFarMaskMismatchIsIgnoredByExplicitAdam) {
    for (const int mask_n : {1, 4}) {
        SCOPED_TRACE(mask_n);
        auto splat = create_mrnf_test_splat_data(2, 0);
        auto control_splat = create_mrnf_test_splat_data(2, 0);
        auto params = vanilla_mrnf_params();
        params.max_cap = 8;
        MRNF strategy(splat);
        MRNF control(control_splat);
        strategy.initialize(params);
        control.initialize(params);
        auto& optimizer = strategy.get_optimizer();
        auto& control_optimizer = control.get_optimizer();
        optimizer.set_per_splat_mean_step(true, 0.5f);
        control_optimizer.set_per_splat_mean_step(true, 0.5f);
        const auto mask = Tensor::ones({static_cast<size_t>(mask_n)}, Device::CUDA).to(DataType::Bool);
        optimizer.set_mean_step_far_mask(mask);
        optimizer.get_grad(ParamType::Means).fill_(0.2f);
        control_optimizer.get_grad(ParamType::Means).fill_(0.2f);
        FarMaskWarningCapture warnings;

        EXPECT_NO_THROW(optimizer.step(1));
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_EQ(optimizer.mean_step_far_mask(), nullptr);
        EXPECT_EQ(optimizer.mean_step_far_mask_n(), 0);
        control_optimizer.step(1);
        EXPECT_EQ(means_xyz(splat), means_xyz(control_splat));
        ASSERT_EQ(warnings.messages.size(), 1u);
        EXPECT_NE(warnings.messages[0].find("mask=" + std::to_string(mask_n) + ", means=2"), std::string::npos);
        optimizer.step(2);
        EXPECT_EQ(warnings.messages.size(), 1u);
    }
}

TEST(MRNFStrategyTest, MeanStepFarMaskMismatchIsIgnoredByFusedAdam) {
    for (const int mask_n : {1, 4}) {
        SCOPED_TRACE(mask_n);
        auto splat = create_mrnf_test_splat_data(2, 0);
        auto params = vanilla_mrnf_params();
        params.max_cap = 8;
        MRNF strategy(splat);
        strategy.initialize(params);
        auto& optimizer = strategy.get_optimizer();
        optimizer.set_per_splat_mean_step(true, 0.5f);
        const auto mask = Tensor::ones({static_cast<size_t>(mask_n)}, Device::CUDA).to(DataType::Bool);
        optimizer.set_mean_step_far_mask(mask);
        FarMaskWarningCapture warnings;

        const auto fused = optimizer.prepare_fastgs_fused_adam(1, nullptr);
        EXPECT_TRUE(fused.enabled);
        EXPECT_TRUE(fused.means.enabled);
        EXPECT_TRUE(fused.per_splat_mean_step);
        EXPECT_EQ(fused.mean_step_far_mask, nullptr);
        EXPECT_EQ(fused.mean_step_far_mask_n, 0);
        EXPECT_EQ(optimizer.mean_step_far_mask(), nullptr);
        EXPECT_EQ(optimizer.mean_step_far_mask_n(), 0);
        ASSERT_EQ(warnings.messages.size(), 1u);
        EXPECT_NE(warnings.messages[0].find("mask=" + std::to_string(mask_n) + ", means=2"), std::string::npos);
        optimizer.prepare_fastgs_fused_adam(2, nullptr);
        EXPECT_EQ(warnings.messages.size(), 1u);

        const auto current_mask = Tensor::zeros_bool({2}, Device::CUDA);
        optimizer.set_mean_step_far_mask(current_mask);
        const auto republished = optimizer.prepare_fastgs_fused_adam(3, nullptr);
        EXPECT_EQ(republished.mean_step_far_mask, current_mask.ptr<bool>());
        EXPECT_EQ(republished.mean_step_far_mask_n, 2);
        EXPECT_EQ(warnings.messages.size(), 1u);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }
}

TEST(MRNFStrategyTest, MeanStepFarMaskUploadsHostStorageBeforeAdam) {
    for (const bool pinned : {false, true}) {
        SCOPED_TRACE(pinned);
        auto splat = create_mrnf_test_splat_data(2, 0);
        auto control_splat = create_mrnf_test_splat_data(2, 0);
        const auto params = vanilla_mrnf_params();
        MRNF strategy(splat);
        MRNF control(control_splat);
        strategy.initialize(params);
        control.initialize(params);
        auto& optimizer = strategy.get_optimizer();
        auto& control_optimizer = control.get_optimizer();
        optimizer.set_per_splat_mean_step(true, 0.5f);
        control_optimizer.set_per_splat_mean_step(true, 0.5f);
        {
            // A strided CPU view exercises both pageable/pinned upload and packing.
            auto host = Tensor::empty({2, 2}, Device::CPU, DataType::Bool, pinned);
            const bool values[] = {true, false, false, true};
            std::memcpy(host.ptr<bool>(), values, sizeof(values));
            auto mask = host.slice(1, 0, 1).squeeze(1);
            ASSERT_FALSE(mask.is_contiguous());
            optimizer.set_mean_step_far_mask(mask);
            control_optimizer.set_mean_step_far_mask(mask.cuda().contiguous());
            EXPECT_NE(optimizer.mean_step_far_mask(), mask.ptr<bool>());
        }
        cudaPointerAttributes attributes{};
        ASSERT_EQ(cudaPointerGetAttributes(&attributes, optimizer.mean_step_far_mask()), cudaSuccess);
        EXPECT_EQ(attributes.type, cudaMemoryTypeDevice);
        bool values[2] = {};
        ASSERT_EQ(cudaMemcpy(values, optimizer.mean_step_far_mask(), sizeof(values),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_TRUE(values[0]);
        EXPECT_FALSE(values[1]);
        const auto fused = optimizer.prepare_fastgs_fused_adam(1, nullptr);
        EXPECT_EQ(fused.mean_step_far_mask, optimizer.mean_step_far_mask());
        EXPECT_EQ(fused.mean_step_far_mask_n, 2);
        optimizer.get_grad(ParamType::Means).fill_(0.2f);
        control_optimizer.get_grad(ParamType::Means).fill_(0.2f);
        optimizer.step(1);
        control_optimizer.step(1);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_EQ(means_xyz(splat), means_xyz(control_splat));
    }
}

TEST(MRNFStrategyTest, MeanStepFarMaskRetainsAllocationUntilBindingIsCleared) {
    auto splat = create_mrnf_test_splat_data(2, 0);
    MRNF strategy(splat);
    strategy.initialize(vanilla_mrnf_params());
    auto& optimizer = strategy.get_optimizer();
    optimizer.set_per_splat_mean_step(true, 0.5f);
    std::weak_ptr<Tensor> allocation;
    {
        auto owner = std::make_shared<Tensor>(Tensor::ones_bool({2}, Device::CUDA));
        allocation = owner;
        auto mask = Tensor::from_external_owner(
            owner->ptr<bool>(), TensorShape({2}), Device::CUDA, DataType::Bool, owner);
        optimizer.set_mean_step_far_mask(mask);
        EXPECT_EQ(optimizer.mean_step_far_mask(), owner->ptr<bool>());
    }
    // This observes ownership directly, independent of allocator address reuse.
    ASSERT_FALSE(allocation.expired());
    optimizer.get_grad(ParamType::Means).fill_(0.2f);
    optimizer.step(1);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    optimizer.set_per_splat_mean_step(false, 0.0f);
    EXPECT_TRUE(allocation.expired());
    EXPECT_EQ(optimizer.mean_step_far_mask(), nullptr);
    EXPECT_EQ(optimizer.mean_step_far_mask_n(), 0);

    // A from_blob view has no allocation owner to retain, so the binding must copy it.
    for (const bool strided : {false, true}) {
        SCOPED_TRACE(strided);
        {
            auto source = Tensor::ones_bool({2, 2}, Device::CUDA);
            auto borrowed = strided
                                ? source.slice(1, 0, 1).squeeze(1)
                                : Tensor::from_blob(source.ptr<bool>(), TensorShape({2}), Device::CUDA, DataType::Bool);
            ASSERT_FALSE(borrowed.owns_memory());
            optimizer.set_mean_step_far_mask(borrowed);
            EXPECT_NE(optimizer.mean_step_far_mask(), source.ptr<bool>());
            // Also prove independence while the source is still allocated.
            source.zero_();
            EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        }
        bool values[2] = {};
        ASSERT_EQ(cudaMemcpy(values, optimizer.mean_step_far_mask(), sizeof(values),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_TRUE(values[0]);
        EXPECT_TRUE(values[1]);
    }
    optimizer.set_mean_step_far_mask({});
}

TEST(MRNFStrategyTest, MeanStepFarMaskEmptyBindingsClearExplicitAndFusedAdam) {
    auto splat = create_mrnf_test_splat_data(2, 0);
    MRNF strategy(splat);
    strategy.initialize(vanilla_mrnf_params());
    auto& optimizer = strategy.get_optimizer();
    optimizer.set_per_splat_mean_step(true, 0.5f);
    for (const auto device : {Device::CPU, Device::CUDA}) {
        optimizer.set_mean_step_far_mask(Tensor::ones_bool({2}, Device::CUDA));
        optimizer.set_mean_step_far_mask(Tensor::empty({0}, device, DataType::Bool));
        EXPECT_EQ(optimizer.mean_step_far_mask(), nullptr);
        EXPECT_EQ(optimizer.mean_step_far_mask_n(), 0);
        const auto fused = optimizer.prepare_fastgs_fused_adam(1, nullptr);
        EXPECT_EQ(fused.mean_step_far_mask, nullptr);
        EXPECT_EQ(fused.mean_step_far_mask_n, 0);
        optimizer.get_grad(ParamType::Means).fill_(0.2f);
        optimizer.step(1);
    }
    // External zero-size views may have non-null host storage. Do not query or upload it.
    auto owner = std::make_shared<bool>(true);
    auto empty = Tensor::from_external_owner(
        owner.get(), TensorShape({0}), Device::CPU, DataType::Bool, owner);
    ASSERT_NE(empty.data_ptr(), nullptr);
    optimizer.set_mean_step_far_mask(Tensor::ones_bool({2}, Device::CUDA));
    optimizer.set_mean_step_far_mask(empty);
    EXPECT_EQ(optimizer.mean_step_far_mask(), nullptr);
    EXPECT_EQ(optimizer.mean_step_far_mask_n(), 0);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

TEST(MRNFStrategyTest, CheckpointLoadPreservesDatasetFarFieldProtection) {
    auto original_model = create_mrnf_test_splat_data(8);
    place_deep_far_probe(original_model);
    MRNF original(original_model);
    param::TrainingParameters params;
    params.optimization = mean_step_test_params();
    const auto dataset = make_hull_dataset();
    original.set_training_dataset(dataset);
    original.initialize(params.optimization);
    ASSERT_NE(original.get_optimizer().mean_step_far_mask(), nullptr);

    std::vector<uint8_t> expected_mask(8);
    ASSERT_EQ(cudaMemcpy(expected_mask.data(), original.get_optimizer().mean_step_far_mask(),
                         expected_mask.size(), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(expected_mask[1], 0);
    ASSERT_EQ(expected_mask[7], 1);

    std::stringstream checkpoint(std::ios::in | std::ios::out | std::ios::binary);
    const auto saved = serialize_checkpoint(checkpoint, 100, original, params,
                                            nullptr, nullptr, nullptr);
    ASSERT_TRUE(saved);

    auto resumed_model = create_mrnf_test_splat_data(8);
    MRNF resumed(resumed_model);
    resumed.set_training_dataset(dataset);
    resumed.initialize(params.optimization);
    checkpoint.seekg(0);
    const auto loaded = load_checkpoint(checkpoint, saved->bytes, resumed, params,
                                        nullptr, nullptr, nullptr);
    ASSERT_TRUE(loaded) << loaded.error();
    ASSERT_EQ(*loaded, 100);
    ASSERT_NE(resumed.get_optimizer().mean_step_far_mask(), nullptr);
    ASSERT_EQ(resumed.get_optimizer().mean_step_far_mask_n(), 8);
    std::vector<uint8_t> actual_mask(8);
    ASSERT_EQ(cudaMemcpy(actual_mask.data(), resumed.get_optimizer().mean_step_far_mask(),
                         actual_mask.size(), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(actual_mask, expected_mask);
}

TEST(MRNFDecayTest, ZeroDecayPreservesFiniteLogitsAndStillDecaysScales) {
    const std::vector<float> original{-80.0f, -20.0f, 0.0f, 16.85f, 20.0f, 80.0f};
    for (const auto [decay, train_t] : {std::pair{0.0f, 0.5f}, std::pair{0.004f, 1.0f}}) {
        auto opacity = Tensor::from_vector(original, {original.size()}, Device::CUDA);
        auto scales = Tensor::zeros({original.size(), 3}, Device::CUDA);
        for (int step = 0; step < 100; ++step) {
            mrnf_strategy::launch_mrnf_decay(opacity.ptr<float>(), scales.ptr<float>(),
                                             nullptr, 0, decay, 0.01f, train_t, original.size());
        }
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const auto actual = opacity.cpu().to_vector();
        for (size_t i = 0; i < original.size(); ++i)
            EXPECT_FLOAT_EQ(actual[i], original[i]) << "row " << i;
        const auto actual_scales = scales.cpu().to_vector();
        EXPECT_NEAR(actual_scales[0], 100.0f * std::log(1.0f - 0.01f * (1.0f - train_t)), 1e-5f);
    }
}

TEST(MRNFDecayTest, SaturatedAndLegacyInfiniteLogitsStayFinite) {
    const float inf = std::numeric_limits<float>::infinity();
    const std::vector<float> original{16.85f, 20.0f, 80.0f, inf, -inf};
    for (const float decay : {0.0f, 1e-12f, 0.004f}) {
        auto opacity = Tensor::from_vector(original, {original.size()}, Device::CUDA);
        auto scales = Tensor::zeros({original.size(), 3}, Device::CUDA);
        mrnf_strategy::launch_mrnf_decay(opacity.ptr<float>(), scales.ptr<float>(),
                                         nullptr, 0, decay, 0.0f, 0.5f, original.size());
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
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

TEST(MRNFDecayTest, FrozenRowsRemainUnchangedWhileNaNsStayVisible) {
    const float inf = std::numeric_limits<float>::infinity();
    auto opacity = Tensor::from_vector(std::vector<float>{inf, std::nanf(""), 20.0f}, {3}, Device::CUDA);
    auto scales = Tensor::zeros({3, 3}, Device::CUDA);
    const auto frozen = Tensor::from_vector(std::vector<bool>{true, false, false}, {3}, Device::CUDA);
    mrnf_strategy::launch_mrnf_decay(opacity.ptr<float>(), scales.ptr<float>(),
                                     frozen.ptr<bool>(), 3, 0.004f, 0.01f, 0.5f, 3);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
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
TEST(MRNFStrategyTest, ChunkedChildPlacementMatchesSingleChunk) {
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

TEST(MRNFStrategyTest, LongAxisSplitPlacementIgnoresQuaternionMagnitudeOnRealRows) {
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
        nullptr, children_opacities.ptr<float>(), split_indices.ptr<int64_t>(), static_cast<int>(n), 0,
        kernels::LONG_AXIS_SPLIT_CROSS_AXIS_SCALE, kernels::LONG_AXIS_SPLIT_OPACITY_SCALE);
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

namespace {
    // Random splats with one distinct long axis each, split once with the given factors.
    struct SplitRun {
        std::vector<float> means, quats, scales, opacities;               // parents before the split
        std::vector<float> first_means, first_scales, first_opacities;    // parent rows after
        std::vector<float> second_means, second_scales, second_opacities; // children
    };

    SplitRun run_long_axis_split(const float cross_axis_scale, const float opacity_scale,
                                 const std::vector<bool>& across = {}) {
        constexpr size_t n = 64;
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
        SplitRun run;
        for (size_t i = 0; i < n; ++i) {
            for (int c = 0; c < 3; ++c)
                run.means.push_back(3.0f * unit(rng));
            float q[4], norm = 0.0f;
            for (float& v : q) {
                v = unit(rng);
                norm += v * v;
            }
            for (const float v : q)
                run.quats.push_back(v / std::sqrt(norm));
            const int axis = static_cast<int>(i % 3);
            for (int c = 0; c < 3; ++c)
                run.scales.push_back(c == axis ? -1.0f + 0.5f * unit(rng) : -3.0f + 0.5f * unit(rng));
            run.opacities.push_back(3.0f * unit(rng));
        }
        auto means = Tensor::from_vector(run.means, {n, 3}, Device::CUDA);
        auto quats = Tensor::from_vector(run.quats, {n, 4}, Device::CUDA);
        auto scales = Tensor::from_vector(run.scales, {n, 3}, Device::CUDA);
        auto sh0 = Tensor::zeros({n, 3}, Device::CUDA);
        auto opacities = Tensor::from_vector(run.opacities, {n}, Device::CUDA);
        auto second_means = Tensor::empty({n, 3}, Device::CUDA);
        auto second_quats = Tensor::empty({n, 4}, Device::CUDA);
        auto second_scales = Tensor::empty({n, 3}, Device::CUDA);
        auto second_sh0 = Tensor::empty({n, 3}, Device::CUDA);
        auto second_opacities = Tensor::empty({n}, Device::CUDA);
        std::vector<int> rows(n);
        for (size_t i = 0; i < n; ++i)
            rows[i] = static_cast<int>(i);
        const auto split_indices = Tensor::from_vector(rows, {n}, Device::CUDA).to(DataType::Int64);
        const auto across_rows = across.empty() ? Tensor() : Tensor::from_vector(across, {n}, Device::CUDA);
        kernels::launch_long_axis_split_gaussians_inplace(
            means.ptr<float>(), quats.ptr<float>(), scales.ptr<float>(), sh0.ptr<float>(), nullptr,
            opacities.ptr<float>(), second_means.ptr<float>(), second_quats.ptr<float>(), second_scales.ptr<float>(),
            second_sh0.ptr<float>(), nullptr, second_opacities.ptr<float>(), split_indices.ptr<int64_t>(),
            static_cast<int>(n), 0, cross_axis_scale, opacity_scale, nullptr,
            across_rows.is_valid() ? across_rows.ptr<bool>() : nullptr);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        run.first_means = means.cpu().to_vector();
        run.first_scales = scales.cpu().to_vector();
        run.first_opacities = opacities.cpu().to_vector();
        run.second_means = second_means.cpu().to_vector();
        run.second_scales = second_scales.cpu().to_vector();
        run.second_opacities = second_opacities.cpu().to_vector();
        return run;
    }

    // Offset of the first child: +0.5 sigma_long along the parent's long axis.
    std::array<float, 3> long_axis_offset(const SplitRun& run, const size_t row) {
        const float* q = run.quats.data() + 4 * row;
        const float w = q[0], x = q[1], y = q[2], z = q[3];
        const float R[9] = {1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)};
        const float* scale = run.scales.data() + 3 * row;
        const int axis = static_cast<int>(std::max_element(scale, scale + 3) - scale);
        const float length = 0.5f * std::exp(scale[axis]);
        return {R[axis] * length, R[axis + 3] * length, R[axis + 6] * length};
    }

    void expect_long_axis_split(const SplitRun& run, const float cross_axis_scale) {
        for (size_t row = 0; row < run.opacities.size(); ++row) {
            SCOPED_TRACE(row);
            const auto offset = long_axis_offset(run, row);
            const float* scale = run.scales.data() + 3 * row;
            const int axis = static_cast<int>(std::max_element(scale, scale + 3) - scale);
            for (int c = 0; c < 3; ++c) {
                EXPECT_NEAR(run.first_means[3 * row + c], run.means[3 * row + c] + offset[c], 1e-5f);
                EXPECT_NEAR(run.second_means[3 * row + c], run.means[3 * row + c] - offset[c], 1e-5f);
                const float expected = scale[c] + std::log(c == axis ? 0.5f : cross_axis_scale);
                EXPECT_NEAR(run.first_scales[3 * row + c], expected, 1e-5f);
                EXPECT_EQ(run.second_scales[3 * row + c], run.first_scales[3 * row + c]);
            }
        }
    }
} // namespace

// Today's call passes 0.85 / 0.6: long axis halved, the others x0.85, both children at 0.6x opacity.
TEST(MRNFStrategyTest, LongAxisSplitDefaultFactorsKeepTodaysRule) {
    const auto run = run_long_axis_split(kernels::LONG_AXIS_SPLIT_CROSS_AXIS_SCALE,
                                         kernels::LONG_AXIS_SPLIT_OPACITY_SCALE);
    expect_long_axis_split(run, 0.85f);
    for (size_t row = 0; row < run.opacities.size(); ++row) {
        const float alpha = 0.6f / (1.0f + std::exp(-run.opacities[row]));
        const float expected = std::log(alpha / (1.0f - alpha));
        EXPECT_NEAR(run.first_opacities[row], expected, 1e-5f) << row;
        EXPECT_EQ(run.second_opacities[row], run.first_opacities[row]) << row;
    }
}

// Receiver rule: both children keep the parent's opacity and are smaller on every axis (long x0.5,
// the others x0.7); they sit at the same +-0.5 sigma_long offsets as with today's rule.
TEST(MRNFStrategyTest, ReceiverSplitKeepsOpacityAndShrinksEveryAxis) {
    const auto receiver = run_long_axis_split(0.7f, 1.0f);
    expect_long_axis_split(receiver, 0.7f);
    EXPECT_EQ(receiver.first_opacities, receiver.opacities);
    EXPECT_EQ(receiver.second_opacities, receiver.opacities);
    for (size_t i = 0; i < receiver.scales.size(); ++i)
        EXPECT_LT(receiver.second_scales[i], receiver.scales[i]) << i;

    const auto today = run_long_axis_split(kernels::LONG_AXIS_SPLIT_CROSS_AXIS_SCALE,
                                           kernels::LONG_AXIS_SPLIT_OPACITY_SCALE);
    EXPECT_EQ(receiver.first_means, today.first_means);
    EXPECT_EQ(receiver.second_means, today.second_means);
}

namespace {
    param::OptimizationParameters need_growth_test_params(const int max_cap) {
        auto opt = vanilla_mrnf_params();
        opt.iterations = 10'000;
        opt.sh_degree_interval = 10'000;
        opt.max_cap = max_cap;
        opt.growth_grad_threshold = 0.5f;
        opt.grow_fraction = 0.0f;
        opt.grow_until_iter = 10'000;
        opt.need_guided_growth = true;
        return opt;
    }

    // Window error and footprint per row, all rows seen.
    void set_need_window(SplatData& splat, Tensor& refine_weight_max, Tensor visibility,
                         const std::vector<float>& error, const std::vector<float>& footprint) {
        const size_t n = splat.size();
        refine_weight_max = Tensor::from_vector(error, TensorShape({n}), Device::CUDA);
        splat._max_screen_share.copy_(Tensor::from_vector(footprint, TensorShape({n}), Device::CUDA));
        visibility.fill_(1.0f);
    }

    Tensor row_range(const size_t from, const size_t to) {
        std::vector<int> rows;
        for (size_t i = from; i < to; ++i)
            rows.push_back(static_cast<int>(i));
        return Tensor::from_vector(rows, TensorShape({rows.size()}), Device::CUDA).to(DataType::Int64);
    }

    size_t rows_below(const Tensor& indices, const int64_t limit) {
        size_t count = 0;
        for (const int64_t row : selected_rows(indices))
            count += row < limit ? 1 : 0;
        return count;
    }

    double mass_share_below(const Tensor& weights, const size_t limit) {
        const auto host = weights.cpu().to_vector();
        double below = 0.0, total = 0.0;
        for (size_t i = 0; i < host.size(); ++i) {
            total += host[i];
            below += i < limit ? host[i] : 0.0f;
        }
        return total > 0.0 ? below / total : 0.0;
    }
} // namespace

// Option off: the error is not turned into NEED and the refill draws by opacity x visibility only.
TEST(MRNFStrategyTest, NeedGrowthOffLeavesErrorAndRefillWeightsUnchanged) {
    constexpr size_t n = 40;
    auto splat_data = create_mrnf_test_splat_data(static_cast<int>(n));
    MRNF strategy(splat_data);
    auto params = need_growth_test_params(static_cast<int>(n));
    params.need_guided_growth = false;
    strategy.initialize(params);

    std::vector<float> error(n), footprint(n);
    for (size_t i = 0; i < n; ++i) {
        error[i] = i < 36 ? 0.01f * static_cast<float>(i + 1) : 0.0f;
        footprint[i] = 0.001f * static_cast<float>(i % 7 + 1);
    }
    set_need_window(splat_data, strategy._refine_weight_max, strategy.visibility_accumulator(), error, footprint);
    strategy.soft_delete_rows(row_range(36, splat_data.size()));
    const auto active = strategy._free_mask.slice(0, 0, n).logical_not().contiguous();

    strategy._densify_n_scratch.ensure_n(n, Device::CUDA);
    const auto weights = strategy.build_replace_parent_weights(n, active, {}, {}).clone();
    auto plain = Tensor::empty({n}, Device::CUDA);
    auto raw = splat_data.opacity_raw().squeeze(-1).contiguous();
    mrnf_strategy::launch_replace_parent_weights(raw.ptr<float>(), strategy.visibility_accumulator().ptr<float>(),
                                                 active.ptr<bool>(), nullptr, {}, plain.ptr<float>(), n);
    EXPECT_EQ(weights.cpu().to_vector(), plain.cpu().to_vector());

    const auto result = strategy.grow_and_split(100, 4, Tensor::zeros_bool({n}, Device::CUDA));
    EXPECT_EQ(result.replacement_count, 4);
    EXPECT_EQ(strategy._refine_weight_max.cpu().to_vector(), error);
}

// Two groups of equal opacity and footprint: the high-error group gets (almost) all refills.
TEST(MRNFStrategyTest, NeedGrowthRefillsFromHighErrorGroup) {
    constexpr size_t n = 220;
    auto splat_data = create_mrnf_test_splat_data(static_cast<int>(n));
    MRNF strategy(splat_data);
    auto params = need_growth_test_params(static_cast<int>(n));
    params.grow_until_iter = 0;
    strategy.initialize(params);

    std::vector<float> error(n, 0.0f), footprint(n, 0.01f);
    for (size_t i = 0; i < 200; ++i)
        error[i] = i < 100 ? 1.0f : 0.01f;
    set_need_window(splat_data, strategy._refine_weight_max, strategy.visibility_accumulator(), error, footprint);
    strategy.soft_delete_rows(row_range(200, splat_data.size()));

    const auto result = strategy.grow_and_split(100, 20, Tensor::zeros_bool({n}, Device::CUDA));
    ASSERT_EQ(result.replacement_count, 20);
    EXPECT_EQ(result.growth_count, 0);
    EXPECT_GE(rows_below(result.split_indices, 100), 16u);

    const auto active = strategy._free_mask.slice(0, 0, n).logical_not().contiguous();
    const auto weights = strategy.build_replace_parent_weights(n, active, {}, {}, strategy._refine_weight_max);
    EXPECT_GT(mass_share_below(weights, 100), 0.98);
}

// Equal error and opacity: the group of large splats (few splats cover the region) gets almost
// all growth and refills.
TEST(MRNFStrategyTest, NeedGrowthPrefersLargeSplatsAtEqualError) {
    constexpr size_t live = 120;
    constexpr size_t large = 49;
    constexpr size_t n = live + 20;
    auto splat_data = create_mrnf_test_splat_data(static_cast<int>(n));
    MRNF strategy(splat_data);
    auto params = need_growth_test_params(1000);
    strategy.initialize(params);

    std::vector<float> error(n, 0.0f), footprint(n, 0.0f);
    for (size_t i = 0; i < live; ++i) {
        error[i] = 1.0f;
        footprint[i] = i < large ? 0.1f : 0.001f;
    }
    set_need_window(splat_data, strategy._refine_weight_max, strategy.visibility_accumulator(), error, footprint);
    strategy.soft_delete_rows(row_range(live, splat_data.size()));

    const auto candidates = strategy._free_mask.slice(0, 0, n).logical_not().contiguous();
    const auto result = strategy.grow_and_split(100, 20, candidates, 10);
    ASSERT_EQ(result.replacement_count, 20);
    ASSERT_EQ(result.growth_count, 10);
    EXPECT_GE(rows_below(result.split_indices, large), 18u);

    // NEED: large splats boosted to the clamp (16x the median footprint), small ones at 1.
    const auto need = strategy._refine_weight_max.cpu().to_vector();
    EXPECT_FLOAT_EQ(need[0], 16.0f);
    EXPECT_FLOAT_EQ(need[large], 1.0f);
    EXPECT_GT(mass_share_below(strategy._refine_weight_max.slice(0, 0, live).contiguous(), large), 0.9);
}

// Donors rank by the blend weight summed over about one pass: windows add, older ones fade.
TEST(MRNFStrategyTest, NeedGrowthDonorsAreLowestSummedContribution) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);
    auto params = need_growth_test_params(32);
    params.refine_every = 1;
    strategy.initialize(params);
    strategy.set_training_dataset(make_hull_dataset()); // two views: older windows keep half
    const size_t n = splat_data.size();
    EXPECT_FALSE(strategy.impact_scratch(100).first.is_valid());

    const std::vector<float> first{1.0f, 4.0f, 0.2f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f};
    const std::vector<float> second{1.0f, 0.0f, 0.2f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 0.5f};
    strategy.visibility_accumulator().copy_(Tensor::from_vector(first, TensorShape({n}), Device::CUDA));
    strategy.update_exchange_importance();
    strategy.visibility_accumulator().copy_(Tensor::from_vector(second, TensorShape({n}), Device::CUDA));
    strategy.update_exchange_importance();

    const auto sum = strategy._exchange_importance_sum.cpu().to_vector();
    for (size_t i = 0; i < n; ++i)
        EXPECT_FLOAT_EQ(sum[i], 0.5f * first[i] + second[i]) << i;
    EXPECT_FALSE(strategy._exchange_importance_peak.is_valid());

    strategy._exchange_windows = strategy.exchange_warmup_windows() + 1;
    // Sums: 1.5, 2.0, 0.3, ..., 5.0 -> row 2 then row 0; row 1 was seen once but brightly.
    EXPECT_EQ(selected_rows(strategy.select_exchange_victims(2)), (std::vector<int64_t>{0, 2}));
}

TEST(MRNFStrategyTest, NeedGrowthProtectsNewbornsUntilAFullPass) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);
    strategy.initialize(need_growth_test_params(32));
    const size_t n = splat_data.size();
    strategy._exchange_importance_sum = Tensor::from_vector(kExchangeSum, TensorShape({n}), Device::CUDA);
    strategy._exchange_birth = Tensor::zeros({n}, Device::CUDA);
    strategy._exchange_windows = strategy.exchange_warmup_windows() + 1;

    const auto filled = Tensor::from_vector(std::vector<int>{1, 2}, TensorShape({2}), Device::CUDA).to(DataType::Int64);
    strategy._exchange_free_before_growth = Tensor::zeros_bool({n}, Device::CUDA);
    strategy._exchange_free_before_growth.index_put_(filled, Tensor::ones_bool({2}, Device::CUDA));
    strategy.record_exchange_births();
    EXPECT_EQ(selected_rows(strategy.select_exchange_victims(2)), (std::vector<int64_t>{3, 4}));

    strategy._exchange_windows += strategy.exchange_warmup_windows() + 1;
    EXPECT_EQ(selected_rows(strategy.select_exchange_victims(2)), (std::vector<int64_t>{1, 2}));
}

// 5 % at the start, a cosine to 0 at grow_until_iter, nothing freed after it.
TEST(MRNFStrategyTest, NeedGrowthDonorShareTapersToZero) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);
    auto params = need_growth_test_params(32);
    params.grow_until_iter = 1000;
    strategy.initialize(params);

    EXPECT_EQ(strategy.exchange_quota(0, 1000), 50u);
    EXPECT_EQ(strategy.exchange_quota(250, 1000), 42u);
    EXPECT_EQ(strategy.exchange_quota(500, 1000), 25u);
    EXPECT_EQ(strategy.exchange_quota(999, 1000), 0u);
    EXPECT_EQ(strategy.exchange_quota(1000, 1000), 0u);
    EXPECT_EQ(strategy.exchange_quota(5000, 1000), 0u);
    EXPECT_TRUE(strategy.growth_exchange_active(999));
    EXPECT_FALSE(strategy.growth_exchange_active(1000));
}

// At the cap, every donor slot goes to growth drawn by NEED, even without error-driven demand,
// and the donors are the least contributing splats, never the parents.
TEST(MRNFStrategyTest, NeedGrowthFreedSlotsGoToTheNeedDraw) {
    constexpr size_t n = 40;
    auto splat_data = create_mrnf_test_splat_data(static_cast<int>(n));
    MRNF strategy(splat_data);
    strategy.initialize(need_growth_test_params(static_cast<int>(n)));

    std::vector<float> error(n), footprint(n, 0.01f), contribution(n, 10.0f);
    for (size_t i = 0; i < n; ++i)
        error[i] = i < 20 ? 1.0f : 0.01f;
    for (size_t i = 20; i < 24; ++i)
        contribution[i] = 0.1f;
    set_need_window(splat_data, strategy._refine_weight_max, strategy.visibility_accumulator(), error, footprint);
    strategy._exchange_importance_sum = Tensor::from_vector(contribution, TensorShape({n}), Device::CUDA);
    strategy._exchange_birth = Tensor::zeros({n}, Device::CUDA);
    strategy._exchange_windows = strategy.exchange_warmup_windows() + 1;

    const auto result = strategy.grow_and_split(100, 0, strategy.compute_refine_candidates(), 4);
    EXPECT_EQ(result.replacement_count, 0);
    ASSERT_EQ(result.growth_count, 4);
    ASSERT_EQ(result.exchange_victims, 4u);
    EXPECT_EQ(selected_rows(result.exchange_victim_indices), (std::vector<int64_t>{20, 21, 22, 23}));
    EXPECT_EQ(rows_below(result.split_indices, 20), 4u);
    EXPECT_EQ(strategy.active_count(), n);
    EXPECT_EQ(splat_data.size(), n);
}

// A resume keeps the donor schedule (derived from the iteration) and rebuilds the contribution
// memory, which warms up for a full pass before donors are taken again.
TEST(MRNFStrategyTest, NeedGrowthResumeRebuildsDonorMemory) {
    constexpr size_t n = 40;
    auto splat_data = create_mrnf_test_splat_data(static_cast<int>(n));
    MRNF strategy(splat_data);
    const auto params = need_growth_test_params(static_cast<int>(n));
    strategy.initialize(params);
    strategy._exchange_importance_sum = Tensor::ones({n}, Device::CUDA);
    strategy._exchange_birth = Tensor::zeros({n}, Device::CUDA);
    strategy._exchange_windows = strategy.exchange_warmup_windows() + 4;
    const auto quota = strategy.exchange_quota(3000, n * 100);

    std::stringstream ss;
    strategy.serialize(ss);
    auto splat_copy = create_mrnf_test_splat_data(static_cast<int>(n));
    MRNF restored(splat_copy);
    restored.initialize(params);
    restored.deserialize(ss);

    EXPECT_EQ(restored.exchange_quota(3000, n * 100), quota);
    EXPECT_EQ(restored._exchange_windows, 0);
    EXPECT_FALSE(restored._exchange_importance_sum.is_valid());

    std::vector<float> error(n, 1.0f), footprint(n, 0.01f);
    set_need_window(splat_copy, restored._refine_weight_max, restored.visibility_accumulator(), error, footprint);
    restored.soft_delete_rows(row_range(n - 4, splat_copy.size()));
    for (int window = 0; window <= restored.exchange_warmup_windows(); ++window)
        restored.update_exchange_importance();
    EXPECT_GT(restored._exchange_windows, restored.exchange_warmup_windows());
    const auto result = restored.grow_and_split(3000, 4, Tensor::zeros_bool({n}, Device::CUDA));
    EXPECT_EQ(result.replacement_count, 4);
    EXPECT_EQ(restored.active_count(), n);
    restored.record_exchange_births(result.split_indices);
    EXPECT_EQ(restored._exchange_birth.numel(), n);
}

// The receiver rule is one choice per refine: off without the option and before the cap; at the
// cap every split of the refine (refills and donor-funded growth) uses it.
TEST(MRNFStrategyTest, NeedGrowthReceiverSplitsOnlyAtTheCap) {
    constexpr size_t n = 200;
    const float today = std::log(0.3f / 0.7f); // 0.6 x sigmoid(0)
    struct Case {
        bool need;
        int max_cap;
        size_t receivers;
        float split_opacity;
        float cross_scale;
    };
    for (const auto& c : {Case{false, static_cast<int>(n), 0, today, 0.85f},
                          Case{true, 1000, 0, today, 0.85f},
                          Case{true, static_cast<int>(n), 5, 0.0f, 0.7f}}) {
        SCOPED_TRACE(testing::Message() << "need " << c.need << ", cap " << c.max_cap);
        auto splat_data = create_mrnf_test_splat_data(static_cast<int>(n));
        MRNF strategy(splat_data);
        auto params = need_growth_test_params(c.max_cap);
        params.need_guided_growth = c.need;
        strategy.initialize(params);
        std::vector<float> error(n, 1.0f), footprint(n, 0.01f), contribution(n, 10.0f);
        for (size_t i = 100; i < 103; ++i)
            contribution[i] = 0.1f;
        set_need_window(splat_data, strategy._refine_weight_max, strategy.visibility_accumulator(), error, footprint);
        strategy._exchange_importance_sum = Tensor::from_vector(contribution, TensorShape({n}), Device::CUDA);
        strategy._exchange_birth = Tensor::zeros({n}, Device::CUDA);
        strategy._exchange_windows = strategy.exchange_warmup_windows() + 1;
        strategy.soft_delete_rows(row_range(n - 2, n)); // 198 of 200 active: at the cap

        const auto result = strategy.grow_and_split(100, 2, strategy.compute_refine_candidates(), 3);
        EXPECT_EQ(result.replacement_count, 2);
        EXPECT_EQ(result.growth_count, c.need ? 3 : 0);
        EXPECT_EQ(static_cast<size_t>(result.receiver_splits), c.receivers);

        const auto opacity = splat_data.opacity_raw().cpu().to_vector();
        const auto scales = splat_data.scaling_raw().cpu().to_vector();
        for (const int64_t row : selected_rows(result.split_indices)) {
            EXPECT_NEAR(opacity[row], c.split_opacity, 1e-6f) << row;
            EXPECT_NEAR(scales[3 * row + 0], std::log(0.5f), 1e-6f) << row;
            EXPECT_NEAR(scales[3 * row + 1], std::log(c.cross_scale), 1e-6f) << row;
            EXPECT_NEAR(scales[3 * row + 2], std::log(c.cross_scale), 1e-6f) << row;
        }
        if (c.receivers > 0) {
            EXPECT_EQ(static_cast<size_t>(result.receiver_splits), result.split_indices.numel());
            const auto free_cpu = strategy._free_mask.cpu();
            const bool* free = free_cpu.ptr<bool>();
            for (size_t row = 0; row < n; ++row)
                if (!free[row])
                    EXPECT_EQ(opacity[row], 0.0f) << row;
        }
    }
}

namespace {
    // Four views over six splats: coverage (row 0) and error (row 1) as the backward leaves them.
    // 0: seen in every view; 1: one view; 2: two views with gaps; 3: never seen;
    // 4: seen in every view, error in one; 5: two equal views.
    constexpr size_t kViewMeanSplats = 6;
    const std::vector<std::vector<float>> kViewCoverage{
        {2.0f, 5.0f, 0.0f, 0.0f, 1.0f, 3.0f},
        {2.0f, 0.0f, 0.5f, 0.0f, 1.0f, 3.0f},
        {2.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f},
        {2.0f, 0.0f, 0.25f, 0.0f, 1.0f, 0.0f}};
    const std::vector<std::vector<float>> kViewError{
        {0.1f, 0.8f, 0.0f, 0.0f, 0.0f, 0.2f},
        {0.2f, 0.0f, 0.5f, 0.0f, 0.0f, 0.2f},
        {0.3f, 0.0f, 0.0f, 0.0f, 0.6f, 0.0f},
        {0.4f, 0.0f, 0.1f, 0.0f, 0.0f, 0.0f}};

    param::OptimizationParameters view_mean_test_params(const bool view_mean) {
        auto p = vanilla_mrnf_params();
        p.iterations = 10'000;
        p.sh_degree_interval = 10'000;
        p.start_refine = 0;
        p.refine_every = 100;
        p.stop_refine = 5'000;
        p.grow_until_iter = 5'000;
        p.grow_fraction = 1.0f;
        p.max_cap = 32;
        // Between the window means and maxes: the mean keeps splats 1 and 2, the max 0, 1, 2 and 4.
        p.growth_grad_threshold = 0.28f;
        p.refine_error_view_mean = view_mean;
        return p;
    }

    void set_info_row(SplatData& splat_data, const size_t row, const std::vector<float>& values) {
        splat_data._densification_info.slice(0, row, row + 1).copy_(Tensor::from_vector(values, TensorShape({1, values.size()}), Device::CUDA));
    }

    std::vector<float> info_row(const SplatData& splat_data, const size_t row) {
        return splat_data._densification_info.slice(0, row, row + 1).contiguous().cpu().to_vector();
    }

    // Adds each view to rows 0 and 1 as the backward does, then runs post_backward on a
    // non-refining step.
    void fold_views(MRNF& strategy, SplatData& splat_data, const size_t views, const int first_iter) {
        RenderOutput render_output;
        for (size_t v = 0; v < views; ++v) {
            for (const size_t row : {size_t{0}, size_t{1}}) {
                auto values = info_row(splat_data, row);
                const auto& view = row == 0 ? kViewCoverage[v] : kViewError[v];
                for (size_t i = 0; i < values.size(); ++i)
                    values[i] += view[i];
                set_info_row(splat_data, row, values);
            }
            strategy.post_backward(first_iter + static_cast<int>(v), render_output);
        }
    }
} // namespace

// Off, the window error is today's max over the views, and row 0 keeps summing the coverage.
TEST(MRNFStrategyTest, RefineErrorViewMeanOffKeepsTheMaxOverViews) {
    auto splat_data = create_mrnf_test_splat_data(static_cast<int>(kViewMeanSplats));
    MRNF strategy(splat_data);
    strategy.initialize(view_mean_test_params(false));
    ASSERT_EQ(splat_data._densification_info.shape()[0], 2u);
    fold_views(strategy, splat_data, kViewCoverage.size(), 96);

    std::vector<float> max(kViewMeanSplats, 0.0f), coverage(kViewMeanSplats, 0.0f);
    for (size_t v = 0; v < kViewCoverage.size(); ++v)
        for (size_t i = 0; i < kViewMeanSplats; ++i) {
            max[i] = std::fmax(max[i], kViewError[v][i]);
            coverage[i] += kViewCoverage[v][i];
        }
    EXPECT_EQ(strategy._refine_weight_max.cpu().to_vector(), max);
    EXPECT_EQ(info_row(splat_data, 0), coverage);
    EXPECT_EQ(info_row(splat_data, 1), std::vector<float>(kViewMeanSplats, 0.0f));
    EXPECT_EQ(strategy.visibility_accumulator().cpu().to_vector(), coverage);

    // The refine grows every splat whose max clears the threshold: 0, 1, 2 and 4.
    const size_t active = strategy.active_count();
    RenderOutput refine_output;
    strategy.post_backward(100, refine_output);
    EXPECT_EQ(strategy.active_count(), active + 4);
}

// On, the window error is the sum over views divided by the views that saw the splat (coverage
// > 0, also with zero error); unseen views do not count. Everything resets at the refine.
TEST(MRNFStrategyTest, RefineErrorViewMeanAveragesVisibleViewsAndResetsAtRefine) {
    auto splat_data = create_mrnf_test_splat_data(static_cast<int>(kViewMeanSplats));
    MRNF strategy(splat_data);
    strategy.initialize(view_mean_test_params(true));
    ASSERT_EQ(splat_data._densification_info.shape()[0], 4u);
    fold_views(strategy, splat_data, kViewCoverage.size(), 96);

    std::vector<float> sum(kViewMeanSplats, 0.0f), coverage(kViewMeanSplats, 0.0f), count(kViewMeanSplats, 0.0f);
    for (size_t v = 0; v < kViewCoverage.size(); ++v)
        for (size_t i = 0; i < kViewMeanSplats; ++i) {
            sum[i] += kViewError[v][i];
            coverage[i] += kViewCoverage[v][i];
            count[i] += kViewCoverage[v][i] > 0.0f ? 1.0f : 0.0f;
        }
    EXPECT_EQ(count, (std::vector<float>{4.0f, 1.0f, 2.0f, 0.0f, 4.0f, 2.0f}));
    EXPECT_EQ(strategy._refine_weight_max.cpu().to_vector(), sum);
    EXPECT_EQ(info_row(splat_data, 0), std::vector<float>(kViewMeanSplats, 0.0f));
    EXPECT_EQ(info_row(splat_data, 1), std::vector<float>(kViewMeanSplats, 0.0f));
    EXPECT_EQ(info_row(splat_data, 2), coverage);
    EXPECT_EQ(info_row(splat_data, 3), count);
    EXPECT_EQ(strategy.visibility_accumulator().cpu().to_vector(), coverage);

    // The mean as refine() takes it, on a copy so the window stays for the refine below.
    auto mean = strategy._refine_weight_max.clone();
    mrnf_strategy::launch_refine_error_view_mean(
        mean.ptr<float>(), splat_data._densification_info.ptr<float>(), kViewMeanSplats);
    const auto mean_host = mean.cpu().to_vector();
    for (size_t i = 0; i < kViewMeanSplats; ++i)
        EXPECT_FLOAT_EQ(mean_host[i], sum[i] / std::fmax(count[i], 1.0f)) << i;
    EXPECT_NEAR(mean_host[0], 0.25f, 1e-6f);
    EXPECT_NEAR(mean_host[2], 0.3f, 1e-6f);
    EXPECT_NEAR(mean_host[4], 0.15f, 1e-6f);
    EXPECT_EQ(mean_host[3], 0.0f);

    // The refine draws from the mean: only splats 1 and 2 clear the threshold.
    const size_t active = strategy.active_count();
    RenderOutput refine_output;
    strategy.post_backward(100, refine_output);
    EXPECT_EQ(strategy.active_count(), active + 2);
    const size_t grown = splat_data.size();
    ASSERT_EQ(splat_data._densification_info.shape()[0], 4u);
    ASSERT_EQ(splat_data._densification_info.shape()[1], grown);
    for (size_t row = 0; row < 4; ++row)
        EXPECT_EQ(info_row(splat_data, row), std::vector<float>(grown, 0.0f)) << row;
    EXPECT_EQ(strategy._refine_weight_max.cpu().to_vector(), std::vector<float>(grown, 0.0f));
}

// A resume restarts the error sum, so it restarts the view count with it; the coverage stays.
TEST(MRNFStrategyTest, RefineErrorViewMeanRestartsTheCountOnResume) {
    auto splat_data = create_mrnf_test_splat_data(static_cast<int>(kViewMeanSplats));
    MRNF strategy(splat_data);
    const auto params = view_mean_test_params(true);
    strategy.initialize(params);
    fold_views(strategy, splat_data, 2, 96);
    std::stringstream ss;
    strategy.serialize(ss);

    auto splat_copy = create_mrnf_test_splat_data(static_cast<int>(kViewMeanSplats));
    MRNF restored(splat_copy);
    restored.initialize(params);
    splat_copy._densification_info = splat_data._densification_info.clone(); // as the checkpoint holds it
    restored.deserialize(ss);

    EXPECT_EQ(info_row(splat_copy, 2), info_row(splat_data, 2));
    EXPECT_EQ(info_row(splat_copy, 3), std::vector<float>(kViewMeanSplats, 0.0f));
    EXPECT_EQ(restored._refine_weight_max.cpu().to_vector(), std::vector<float>(kViewMeanSplats, 0.0f));
}

namespace {
    param::OptimizationParameters compress_test_params() {
        auto p = view_mean_test_params(false);
        p.refine_error_compress = true;
        return p;
    }

    std::vector<float> as_floats(const Tensor& mask) {
        return mask.to(DataType::Float32).cpu().to_vector();
    }
} // namespace

// The densification copy of the error map is squared, with negative values floored to 0.
TEST(MRNFStrategyTest, RefineErrorCompressSquaresTheMapFlooredAtZero) {
    const std::vector<float> map{-0.5f, 0.0f, 0.3f, 2.0f, 1.0f};
    auto device_map = Tensor::from_vector(map, TensorShape({map.size()}), Device::CUDA);
    kernels::launch_square_floored_inplace(device_map.ptr<float>(), map.size());
    const auto squared = device_map.cpu().to_vector();
    EXPECT_EQ(squared[0], 0.0f);
    EXPECT_EQ(squared[1], 0.0f);
    EXPECT_FLOAT_EQ(squared[2], 0.09f);
    EXPECT_EQ(squared[3], 4.0f);
    EXPECT_EQ(squared[4], 1.0f);
}

// Each view's error sum enters the window as s^0.2; the mean is over the views where s > 0, so a
// view that saw the splat without error does not dilute it. Everything resets at the refine, and
// every splat with a score is a candidate.
TEST(MRNFStrategyTest, RefineErrorCompressFoldsThePowerOfEachViewOverViewsWithError) {
    auto splat_data = create_mrnf_test_splat_data(static_cast<int>(kViewMeanSplats));
    MRNF strategy(splat_data);
    strategy.initialize(compress_test_params());
    ASSERT_TRUE(strategy.refine_error_view_mean_on());
    ASSERT_EQ(splat_data._densification_info.shape()[0], 4u);
    fold_views(strategy, splat_data, kViewCoverage.size(), 96);

    std::vector<float> sum(kViewMeanSplats, 0.0f), coverage(kViewMeanSplats, 0.0f), count(kViewMeanSplats, 0.0f);
    for (size_t v = 0; v < kViewCoverage.size(); ++v)
        for (size_t i = 0; i < kViewMeanSplats; ++i) {
            const float s = kViewError[v][i] > 0.0f ? std::pow(kViewError[v][i], 0.2f) : 0.0f;
            sum[i] += s;
            coverage[i] += kViewCoverage[v][i];
            count[i] += s > 0.0f ? 1.0f : 0.0f;
        }
    // Splat 4 is seen in four views with error in one: it counts one view, not four.
    EXPECT_EQ(count, (std::vector<float>{4.0f, 1.0f, 2.0f, 0.0f, 1.0f, 2.0f}));
    const auto weight = strategy._refine_weight_max.cpu().to_vector();
    for (size_t i = 0; i < kViewMeanSplats; ++i)
        EXPECT_NEAR(weight[i], sum[i], 1e-5f) << i;
    EXPECT_EQ(info_row(splat_data, 0), std::vector<float>(kViewMeanSplats, 0.0f));
    EXPECT_EQ(info_row(splat_data, 1), std::vector<float>(kViewMeanSplats, 0.0f));
    EXPECT_EQ(info_row(splat_data, 2), coverage);
    EXPECT_EQ(info_row(splat_data, 3), count);
    EXPECT_EQ(strategy.visibility_accumulator().cpu().to_vector(), coverage);

    auto mean = strategy._refine_weight_max.clone();
    mrnf_strategy::launch_refine_error_view_mean(
        mean.ptr<float>(), splat_data._densification_info.ptr<float>(), kViewMeanSplats);
    const auto mean_host = mean.cpu().to_vector();
    for (size_t i = 0; i < kViewMeanSplats; ++i)
        EXPECT_NEAR(mean_host[i], sum[i] / std::fmax(count[i], 1.0f), 1e-5f) << i;
    EXPECT_NEAR(mean_host[4], std::pow(0.6f, 0.2f), 1e-5f); // the view mean gives 0.15
    EXPECT_EQ(mean_host[3], 0.0f);

    // A 10^4 spread between two splats' view sums becomes 10^0.8, about 6.3x.
    {
        auto info = Tensor::zeros({4, 2}, Device::CUDA);
        info.slice(0, 0, 1).fill_(1.0f);
        info.slice(0, 1, 2).copy_(Tensor::from_vector(std::vector<float>{0.01f, 100.0f}, TensorShape({1, 2}), Device::CUDA));
        auto w = Tensor::zeros({2}, Device::CUDA);
        mrnf_strategy::launch_fold_densification_error_view_sum_and_zero(w.ptr<float>(), info.ptr<float>(), 2, true);
        const auto folded = w.cpu().to_vector();
        EXPECT_NEAR(folded[1] / folded[0], std::pow(10.0f, 0.8f), 1e-3f);
    }

    // The 0.28 threshold does not apply: splats 0, 1, 2, 4 and 5 all grow, against 2 with the view mean.
    const size_t active = strategy.active_count();
    RenderOutput refine_output;
    strategy.post_backward(100, refine_output);
    EXPECT_EQ(strategy.active_count(), active + 5);
    const size_t grown = splat_data.size();
    ASSERT_EQ(splat_data._densification_info.shape()[0], 4u);
    for (size_t row = 0; row < 4; ++row)
        EXPECT_EQ(info_row(splat_data, row), std::vector<float>(grown, 0.0f)) << row;
    EXPECT_EQ(strategy._refine_weight_max.cpu().to_vector(), std::vector<float>(grown, 0.0f));
}

// Candidates: score > threshold today, score > 0 with compression; both need visibility.
TEST(MRNFStrategyTest, RefineErrorCompressCandidatesAreEveryScoredSplat) {
    auto splat_data = create_mrnf_test_splat_data(static_cast<int>(kViewMeanSplats));
    MRNF strategy(splat_data);
    auto params = view_mean_test_params(true);
    params.growth_grad_threshold = 0.003f;
    strategy.initialize(params);
    const std::vector<float> score{0.0f, 1e-6f, 0.002f, 0.5f, 0.7f, 3.0f};
    const auto set_window = [&] {
        strategy._refine_weight_max = Tensor::from_vector(score, TensorShape({kViewMeanSplats}), Device::CUDA);
        set_info_row(splat_data, 2, {1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f});
    };
    set_window();
    EXPECT_EQ(as_floats(strategy.compute_refine_candidates()),
              (std::vector<float>{0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f}));

    params.refine_error_compress = true;
    strategy.set_optimization_params(params);
    set_window();
    EXPECT_EQ(as_floats(strategy.compute_refine_candidates()),
              (std::vector<float>{0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f}));
}

// With compression NEED is the score over its mean among the active splats, without the
// footprint (sparsity) factor; without it the large splats are boosted to the clamp.
TEST(MRNFStrategyTest, RefineErrorCompressNeedHasNoSparsityFactor) {
    constexpr size_t live = 120;
    constexpr size_t large = 49;
    constexpr size_t n = live + 20;
    const auto need_of = [&](const bool compress) {
        auto splat_data = create_mrnf_test_splat_data(static_cast<int>(n));
        MRNF strategy(splat_data);
        auto params = need_growth_test_params(1000);
        params.refine_error_compress = compress;
        strategy.initialize(params);
        std::vector<float> error(n, 0.0f), footprint(n, 0.0f);
        for (size_t i = 0; i < live; ++i) {
            error[i] = i % 2 == 0 ? 1.0f : 0.0f;
            footprint[i] = i < large ? 0.1f : 0.001f;
        }
        set_need_window(splat_data, strategy._refine_weight_max, strategy.visibility_accumulator(), error, footprint);
        strategy.soft_delete_rows(row_range(live, splat_data.size()));
        strategy.fold_need_into_refine_weights();
        return strategy._refine_weight_max.cpu().to_vector();
    };

    // Half of the 120 active splats have score 1: the active mean is 0.5, NEED 2 for large and small.
    const auto compressed = need_of(true);
    EXPECT_FLOAT_EQ(compressed[0], 2.0f);
    EXPECT_FLOAT_EQ(compressed[large + 1], 2.0f);
    EXPECT_EQ(compressed[1], 0.0f);
    for (size_t i = live; i < n; ++i)
        EXPECT_EQ(compressed[i], 0.0f) << i;

    const auto today = need_of(false);
    EXPECT_FLOAT_EQ(today[0], 16.0f);
    EXPECT_FLOAT_EQ(today[large + 1], 1.0f);
}

// One view: a large splat covers 400 pixels at error 0.05, a small one 4 pixels at 0.9 (blend
// weight 1). Today (view mean + NEED with sparsity) the large one ranks higher (sum 20 vs 3.6);
// with compression the small one does (squared sums 1 vs 3.24, then ^0.2, no sparsity).
TEST(MRNFStrategyTest, RefineErrorCompressRanksSmallIntenseOverLargeFaint) {
    constexpr size_t large_px = 400;
    constexpr size_t small_px = 4;
    std::vector<float> map(large_px + small_px);
    for (size_t p = 0; p < map.size(); ++p)
        map[p] = p < large_px ? 0.05f : 0.9f;

    const auto score = [&](const bool compress) {
        auto splat_data = create_mrnf_test_splat_data(static_cast<int>(kViewMeanSplats));
        MRNF strategy(splat_data);
        auto params = view_mean_test_params(true);
        params.need_guided_growth = true;
        params.refine_error_compress = compress;
        strategy.initialize(params);

        auto device_map = Tensor::from_vector(map, TensorShape({map.size()}), Device::CUDA);
        if (compress)
            kernels::launch_square_floored_inplace(device_map.ptr<float>(), map.size());
        const auto e = device_map.cpu().to_vector();
        std::vector<float> coverage(kViewMeanSplats, 0.0f), error(kViewMeanSplats, 0.0f);
        for (size_t p = 0; p < e.size(); ++p) {
            const size_t splat = p < large_px ? 0 : 1;
            coverage[splat] += 1.0f;
            error[splat] += 1.0f * e[p];
        }
        set_info_row(splat_data, 0, coverage);
        set_info_row(splat_data, 1, error);
        RenderOutput render_output;
        strategy.post_backward(96, render_output);

        std::vector<float> footprint(kViewMeanSplats, 0.0f);
        footprint[0] = 0.2f;
        footprint[1] = 0.01f;
        splat_data._max_screen_share.copy_(Tensor::from_vector(footprint, TensorShape({kViewMeanSplats}), Device::CUDA));
        mrnf_strategy::launch_refine_error_view_mean(
            strategy._refine_weight_max.ptr<float>(), splat_data._densification_info.ptr<float>(), kViewMeanSplats);
        strategy.fold_need_into_refine_weights();
        return strategy._refine_weight_max.cpu().to_vector();
    };

    const auto today = score(false);
    EXPECT_GT(today[0], today[1]);
    const auto compressed = score(true);
    EXPECT_GT(compressed[1], compressed[0]);
    EXPECT_NEAR(compressed[1] / compressed[0], std::pow(3.24f, 0.2f), 1e-4f);
}

// Switching between the view mean and compression starts a fresh window: sum and view count.
TEST(MRNFStrategyTest, RefineErrorCompressSwitchStartsAFreshWindow) {
    auto splat_data = create_mrnf_test_splat_data(static_cast<int>(kViewMeanSplats));
    MRNF strategy(splat_data);
    auto params = view_mean_test_params(true);
    strategy.initialize(params);
    fold_views(strategy, splat_data, 2, 96);
    ASSERT_NE(info_row(splat_data, 3), std::vector<float>(kViewMeanSplats, 0.0f));

    params.refine_error_compress = true;
    strategy.set_optimization_params(params);
    EXPECT_EQ(info_row(splat_data, 3), std::vector<float>(kViewMeanSplats, 0.0f));
    EXPECT_EQ(strategy._refine_weight_max.cpu().to_vector(), std::vector<float>(kViewMeanSplats, 0.0f));

    // A resume restarts the count in this mode too.
    fold_views(strategy, splat_data, 2, 98);
    std::stringstream ss;
    strategy.serialize(ss);
    auto splat_copy = create_mrnf_test_splat_data(static_cast<int>(kViewMeanSplats));
    MRNF restored(splat_copy);
    restored.initialize(params);
    splat_copy._densification_info = splat_data._densification_info.clone();
    restored.deserialize(ss);
    EXPECT_EQ(info_row(splat_copy, 3), std::vector<float>(kViewMeanSplats, 0.0f));
}

// --refine-need-power: NEED^p after the fold, in both fold paths. p = 1 is today's NEED; with p = 3
// every active NEED is cubed, zero rows stay zero and free rows (with a stale error) are not touched.
TEST(MRNFStrategyTest, RefineNeedPowerRaisesTheFoldedNeed) {
    constexpr size_t live = 120;
    constexpr size_t n = live + 20;
    std::vector<float> error(n, 0.0f), footprint(n, 0.01f);
    for (size_t i = 0; i < live; ++i)
        error[i] = i % 3 == 0 ? 0.0f : 0.05f * static_cast<float>(i % 17 + 1);
    for (size_t i = live; i < n; ++i)
        error[i] = 0.3f;
    const auto need_of = [&](const bool compress, const std::optional<float> power) {
        auto splat_data = create_mrnf_test_splat_data(static_cast<int>(n));
        MRNF strategy(splat_data);
        auto params = need_growth_test_params(1000);
        params.refine_error_compress = compress;
        if (power)
            params.refine_need_power = *power;
        strategy.initialize(params);
        set_need_window(splat_data, strategy._refine_weight_max, strategy.visibility_accumulator(), error, footprint);
        strategy.soft_delete_rows(row_range(live, splat_data.size()));
        strategy.fold_need_into_refine_weights();
        return strategy._refine_weight_max.cpu().to_vector();
    };

    for (const bool compress : {false, true}) {
        SCOPED_TRACE(compress ? "compressed fold" : "footprint fold");
        // Today: error over its mean (compressed: among the active rows; otherwise among the rows
        // with error); the uniform footprint gives sparsity 1.
        double active_sum = 0.0, error_sum = 0.0;
        size_t with_error = 0;
        for (size_t i = 0; i < n; ++i) {
            error_sum += error[i];
            with_error += error[i] > 0.0f ? 1 : 0;
            active_sum += i < live ? error[i] : 0.0f;
        }
        const double scale = compress ? active_sum / live : error_sum / static_cast<double>(with_error);

        const auto today = need_of(compress, std::nullopt);
        const auto linear = need_of(compress, 1.0f);
        EXPECT_EQ(linear, today);
        for (size_t i = 0; i < live; ++i)
            EXPECT_NEAR(linear[i], error[i] / scale, 1e-5 * (1.0 + error[i] / scale)) << i;

        const auto cubed = need_of(compress, 3.0f);
        size_t zeros = 0;
        for (size_t i = 0; i < live; ++i) {
            if (error[i] == 0.0f) {
                EXPECT_EQ(cubed[i], 0.0f) << i;
                ++zeros;
                continue;
            }
            const double expected = std::pow(static_cast<double>(linear[i]), 3.0);
            EXPECT_NEAR(cubed[i], expected, 1e-5 * expected) << i;
        }
        EXPECT_EQ(zeros, live / 3);
        for (size_t i = live; i < n; ++i) {
            EXPECT_GT(linear[i], 0.0f) << i;
            EXPECT_EQ(cubed[i], linear[i]) << i;
        }
    }
}

// The refill weights are opacity x visibility x NEED and the growth weights candidates x NEED, so
// both draws see the power: two equal groups at error 1 and 0.1 hold 10/11 of the receiver mass
// at p = 1 and 1000/1001 at p = 3, and the refills then come (almost) all from the first group.
TEST(MRNFStrategyTest, RefineNeedPowerSharpensTheReceiverDraws) {
    constexpr size_t n = 220;
    constexpr size_t high = 100;
    constexpr size_t live = 200;
    const auto run = [&](const float power, MRNF::GrowthResult* result) {
        auto splat_data = create_mrnf_test_splat_data(static_cast<int>(n));
        MRNF strategy(splat_data);
        auto params = need_growth_test_params(static_cast<int>(n));
        params.grow_until_iter = 0;
        params.refine_need_power = power;
        strategy.initialize(params);
        std::vector<float> error(n, 0.0f), footprint(n, 0.01f);
        for (size_t i = 0; i < live; ++i)
            error[i] = i < high ? 1.0f : 0.1f;
        set_need_window(splat_data, strategy._refine_weight_max, strategy.visibility_accumulator(), error, footprint);
        strategy.soft_delete_rows(row_range(live, splat_data.size()));

        // The weights grow_and_split draws from, built the same way after the fold.
        const auto active = strategy._free_mask.slice(0, 0, n).logical_not().contiguous();
        strategy.fold_need_into_refine_weights();
        strategy._densify_n_scratch.ensure_n(n, Device::CUDA);
        const auto refill = strategy.build_replace_parent_weights(n, active, {}, {}, strategy._refine_weight_max)
                                .cpu()
                                .to_vector();
        const auto growth = (active * strategy._refine_weight_max).cpu().to_vector();
        if (result) {
            // A fresh window: grow_and_split folds it itself.
            strategy._refine_weight_max = Tensor::from_vector(error, TensorShape({n}), Device::CUDA);
            *result = strategy.grow_and_split(100, 20, Tensor::zeros_bool({n}, Device::CUDA));
        }
        const auto share = [&](const std::vector<float>& w) {
            double below = 0.0, total = 0.0;
            for (size_t i = 0; i < live; ++i) {
                total += w[i];
                below += i < high ? w[i] : 0.0f;
            }
            return below / total;
        };
        return std::pair{share(refill), share(growth)};
    };

    const auto [refill_linear, growth_linear] = run(1.0f, nullptr);
    EXPECT_NEAR(refill_linear, 10.0 / 11.0, 1e-4);
    EXPECT_NEAR(growth_linear, 10.0 / 11.0, 1e-4);

    MRNF::GrowthResult result;
    const auto [refill_cubed, growth_cubed] = run(3.0f, &result);
    EXPECT_NEAR(refill_cubed, 1000.0 / 1001.0, 1e-5);
    EXPECT_NEAR(growth_cubed, 1000.0 / 1001.0, 1e-5);
    ASSERT_EQ(result.replacement_count, 20);
    EXPECT_GE(rows_below(result.split_indices, high), 18u);
}

namespace {
    param::OptimizationParameters donor_per_view_fold_params(const int fold, const bool per_view) {
        auto p = view_mean_test_params(fold == 1);
        p.refine_error_compress = fold == 2;
        p.need_guided_growth = true;
        p.donor_rank_per_view = per_view;
        return p;
    }

    std::shared_ptr<CameraDataset> make_view_dataset(const int views) {
        std::vector<std::shared_ptr<Camera>> cams;
        for (int i = 0; i < views; ++i)
            cams.push_back(make_hull_camera(static_cast<float>(i % 2 == 0 ? -1 : 1), 0.0f, 0.1f * static_cast<float>(i), i));
        DatasetConfig cfg;
        cfg.test_every = 100'000;
        return std::make_shared<CameraDataset>(std::move(cams), cfg, CameraDataset::Split::ALL);
    }

    // Views that covered each splat (coverage > 0) in kViewCoverage, whatever the error did.
    const std::vector<float> kCoveringViews{4.0f, 1.0f, 2.0f, 0.0f, 4.0f, 2.0f};
} // namespace

// The window counts the views with coverage > 0 in all three error folds: the max fold (from the
// growth of the summed coverage in row 0), the view mean (its row 3) and compression (a count of
// its own, since its row 3 counts views with error). The error folds themselves are unchanged.
TEST(MRNFStrategyTest, DonorRankPerViewCountsCoveringViewsInEveryFold) {
    for (const int fold : {0, 1, 2}) {
        SCOPED_TRACE(testing::Message() << "fold " << fold);
        auto splat_data = create_mrnf_test_splat_data(static_cast<int>(kViewMeanSplats));
        MRNF strategy(splat_data);
        strategy.initialize(donor_per_view_fold_params(fold, true));
        ASSERT_TRUE(strategy.donor_rank_per_view_on());
        ASSERT_TRUE(strategy.growth_exchange_active(99));
        fold_views(strategy, splat_data, kViewCoverage.size(), 96);

        EXPECT_EQ(strategy.donor_window_views().cpu().to_vector(), kCoveringViews);
        EXPECT_EQ(strategy._exchange_window_views.is_valid(), fold != 1);
        EXPECT_EQ(strategy._exchange_window_coverage.is_valid(), fold == 0);
        if (fold == 0) {
            std::vector<float> max(kViewMeanSplats, 0.0f);
            for (size_t v = 0; v < kViewCoverage.size(); ++v)
                for (size_t i = 0; i < kViewMeanSplats; ++i)
                    max[i] = std::fmax(max[i], kViewError[v][i]);
            EXPECT_EQ(strategy._refine_weight_max.cpu().to_vector(), max);
        } else {
            EXPECT_EQ(info_row(splat_data, 3), fold == 1 ? kCoveringViews
                                                         : (std::vector<float>{4.0f, 1.0f, 2.0f, 0.0f, 1.0f, 2.0f}));
        }

        strategy.update_exchange_importance();
        EXPECT_EQ(strategy._exchange_view_count.cpu().to_vector(), kCoveringViews);
        EXPECT_FALSE(strategy._exchange_window_views.is_valid());
        EXPECT_FALSE(strategy._exchange_window_coverage.is_valid());
    }

    // Off: no count is kept anywhere.
    auto splat_data = create_mrnf_test_splat_data(static_cast<int>(kViewMeanSplats));
    MRNF strategy(splat_data);
    strategy.initialize(donor_per_view_fold_params(2, false));
    fold_views(strategy, splat_data, kViewCoverage.size(), 96);
    EXPECT_FALSE(strategy._exchange_window_views.is_valid());
    strategy.update_exchange_importance();
    EXPECT_TRUE(strategy._exchange_importance_sum.is_valid());
    EXPECT_FALSE(strategy._exchange_view_count.is_valid());
}

// The view count fades with the same keep as the summed contribution (two views, refine every
// step: older windows keep half), so both cover the same pass.
TEST(MRNFStrategyTest, DonorRankPerViewFadesTheCountWithTheSum) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);
    auto params = need_growth_test_params(32);
    params.refine_every = 1;
    params.donor_rank_per_view = true;
    strategy.initialize(params);
    strategy.set_training_dataset(make_hull_dataset());
    const size_t n = splat_data.size();

    const std::vector<float> first{1.0f, 4.0f, 0.2f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f};
    const std::vector<float> second{1.0f, 0.0f, 0.2f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 0.5f};
    const std::vector<float> first_views{1.0f, 2.0f, 1.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f};
    const std::vector<float> second_views{1.0f, 0.0f, 1.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 1.0f};
    for (const auto* window : {&first, &second}) {
        strategy.visibility_accumulator().copy_(Tensor::from_vector(*window, TensorShape({n}), Device::CUDA));
        strategy._exchange_window_views =
            Tensor::from_vector(window == &first ? first_views : second_views, TensorShape({n}), Device::CUDA);
        strategy.update_exchange_importance();
    }

    const auto sum = strategy._exchange_importance_sum.cpu().to_vector();
    const auto views = strategy._exchange_view_count.cpu().to_vector();
    for (size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(sum[i], 0.5f * first[i] + second[i]) << i;
        EXPECT_FLOAT_EQ(views[i], 0.5f * first_views[i] + second_views[i]) << i;
    }
}

// Off, the donors are exactly the rows the summed contribution ranks lowest, as before, even
// with a view count present.
TEST(MRNFStrategyTest, DonorRankPerViewOffKeepsTheSummedRanking) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);
    strategy.initialize(need_growth_test_params(32));
    strategy.set_training_dataset(make_view_dataset(100));
    ASSERT_FALSE(strategy.donor_rank_per_view_on());
    const size_t n = splat_data.size();
    strategy._exchange_importance_sum = Tensor::from_vector(kExchangeSum, TensorShape({n}), Device::CUDA);
    strategy._exchange_view_count = Tensor::from_vector(
        std::vector<float>{1000.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, TensorShape({n}), Device::CUDA);
    strategy._exchange_birth = Tensor::zeros({n}, Device::CUDA);
    strategy._exchange_windows = strategy.exchange_warmup_windows() + 1;

    const auto chosen = strategy.select_exchange_victims(4);
    auto reference = Tensor::empty({4}, Device::CUDA, DataType::Int64);
    const auto eligible = Tensor::ones_bool({n}, Device::CUDA);
    mrnf_strategy::launch_rank_exchange_rows(strategy._exchange_importance_sum.ptr<float>(), nullptr,
                                             eligible.ptr<bool>(), n, 4, reference.ptr<int64_t>());
    EXPECT_EQ(chosen.cpu().to_vector_int64(), reference.cpu().to_vector_int64());
    EXPECT_EQ(selected_rows(chosen), (std::vector<int64_t>{1, 2, 3, 4}));
}

// On, a splat ranks by sum / max(views, floor), floor = max(1, round(5 % of the training views)).
// Sum 10 over 40 views (0.25) is kept over sum 20 over 200 views (0.1); sum 1 seen by one view
// ranks as 1 / floor (0.2), not 1.
TEST(MRNFStrategyTest, DonorRankPerViewRanksBySumPerViewWithAFloor) {
    const std::vector<float> sum{10.0f, 20.0f, 1.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f};
    const std::vector<float> views{40.0f, 200.0f, 1.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f};
    const auto run = [&](const bool per_view, const size_t count) {
        auto splat_data = create_mrnf_test_splat_data();
        MRNF strategy(splat_data);
        auto params = need_growth_test_params(32);
        params.donor_rank_per_view = per_view;
        strategy.initialize(params);
        strategy.set_training_dataset(make_view_dataset(100));
        EXPECT_EQ(strategy.donor_view_floor(), 5.0f);
        const size_t n = splat_data.size();
        strategy._exchange_importance_sum = Tensor::from_vector(sum, TensorShape({n}), Device::CUDA);
        strategy._exchange_view_count = Tensor::from_vector(views, TensorShape({n}), Device::CUDA);
        strategy._exchange_birth = Tensor::zeros({n}, Device::CUDA);
        strategy._exchange_windows = strategy.exchange_warmup_windows() + 1;
        return selected_rows(strategy.select_exchange_victims(count));
    };

    EXPECT_EQ(run(false, 1), (std::vector<int64_t>{2}));
    EXPECT_EQ(run(false, 2), (std::vector<int64_t>{0, 2}));
    EXPECT_EQ(run(true, 1), (std::vector<int64_t>{1}));
    // Without the floor splat 2 would rank at 1.0, above splat 0 (0.25): the pair would be {0, 1}.
    EXPECT_EQ(run(true, 2), (std::vector<int64_t>{1, 2}));

    // The floor: 5 % of the training views, rounded, at least 1.
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);
    auto params = need_growth_test_params(32);
    params.donor_rank_per_view = true;
    strategy.initialize(params);
    EXPECT_EQ(strategy.donor_view_floor(), 1.0f);
    strategy.set_training_dataset(make_hull_dataset());
    EXPECT_EQ(strategy.donor_view_floor(), 1.0f);
    strategy.set_training_dataset(make_view_dataset(315));
    EXPECT_EQ(strategy.donor_view_floor(), 16.0f);
}

// The view count stays with its splat: newborns (filled slots, split parents, appended rows)
// restart at 0, and compaction and a Morton reorder carry the count and the window rows.
TEST(MRNFStrategyTest, DonorRankPerViewCountFollowsItsSplats) {
    std::vector<float> iota10(10);
    std::iota(iota10.begin(), iota10.end(), 1.0f);
    {
        auto splat_data = create_mrnf_test_splat_data();
        MRNF strategy(splat_data);
        auto params = need_growth_test_params(32);
        params.donor_rank_per_view = true;
        strategy.initialize(params);
        const size_t n = splat_data.size();
        strategy._exchange_importance_sum = Tensor::from_vector(kExchangeSum, TensorShape({n}), Device::CUDA);
        strategy._exchange_view_count = Tensor::from_vector(iota10, TensorShape({n}), Device::CUDA);
        strategy._exchange_birth = Tensor::zeros({n}, Device::CUDA);
        strategy._exchange_windows = strategy.exchange_warmup_windows() + 3;
        strategy._exchange_free_before_growth = Tensor::zeros_bool({n}, Device::CUDA);
        const auto children = Tensor::from_vector(std::vector<int>{2, 3}, TensorShape({2}), Device::CUDA).to(DataType::Int64);
        strategy._exchange_free_before_growth.index_put_(children, Tensor::ones_bool({2}, Device::CUDA));
        const auto parents = Tensor::from_vector(std::vector<int>{0}, TensorShape({1}), Device::CUDA).to(DataType::Int64);
        strategy.record_exchange_births(parents);
        const auto views = strategy._exchange_view_count.cpu().to_vector();
        ASSERT_EQ(views.size(), n);
        for (size_t row = 0; row < n; ++row)
            EXPECT_EQ(views[row], row == 0 || row == 2 || row == 3 ? 0.0f : iota10[row]) << row;
    }
    {
        auto splat_data = create_mrnf_test_splat_data();
        MRNF strategy(splat_data);
        auto params = need_growth_test_params(32);
        params.donor_rank_per_view = true;
        strategy.initialize(params);
        splat_data._densification_info = Tensor::ones({2, static_cast<size_t>(splat_data.size())}, Device::CUDA);
        const size_t n = splat_data.size();
        strategy._exchange_importance_sum = Tensor::from_vector(kExchangeSum, TensorShape({n}), Device::CUDA);
        strategy._exchange_view_count = Tensor::from_vector(iota10, TensorShape({n}), Device::CUDA);
        strategy._exchange_window_views = Tensor::from_vector(iota10, TensorShape({n}), Device::CUDA).mul(2.0f);
        strategy._exchange_window_coverage = Tensor::from_vector(iota10, TensorShape({n}), Device::CUDA).mul(3.0f);
        strategy._exchange_birth = Tensor::zeros({n}, Device::CUDA);
        strategy._exchange_windows = strategy.exchange_warmup_windows() + 1;
        const auto mask = Tensor::from_vector(
            std::vector<bool>{true, false, false, false, false, false, false, false, false, true},
            TensorShape({10}), Device::CUDA);
        strategy.remove_gaussians(mask);
        ASSERT_EQ(splat_data.size(), 8u);
        std::vector<float> kept(iota10.begin() + 1, iota10.end() - 1);
        EXPECT_EQ(strategy._exchange_view_count.cpu().to_vector(), kept);
        for (auto& v : kept)
            v *= 2.0f;
        EXPECT_EQ(strategy._exchange_window_views.cpu().to_vector(), kept);
        for (auto& v : kept)
            v *= 1.5f;
        EXPECT_EQ(strategy._exchange_window_coverage.cpu().to_vector(), kept);
    }
    {
        auto splat = create_mrnf_test_splat_data(4, 0);
        {
            const std::vector<float> means{3.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
            splat.means().copy_from(Tensor::from_vector(means, TensorShape({4, 3}), Device::CUDA));
        }
        auto params = vanilla_mrnf_params();
        params.max_cap = 8;
        MRNF strategy(splat);
        strategy.initialize(params);
        const std::vector<float> count{10.0f, 20.0f, 30.0f, 40.0f};
        strategy._exchange_view_count = Tensor::from_vector(count, TensorShape({4}), Device::CUDA);
        strategy._exchange_window_views = Tensor::from_vector(count, TensorShape({4}), Device::CUDA).mul(2.0f);
        strategy._exchange_window_coverage = Tensor::from_vector(count, TensorShape({4}), Device::CUDA).mul(3.0f);
        const auto result = morton::apply_morton_reorder(
            splat, &strategy.get_optimizer(), nullptr,
            [&](const morton::RowVisitor& visit) { strategy.for_each_gaussian_row(visit); });
        ASSERT_TRUE(result.applied);
        const auto permutation = result.permutation.cpu();
        const auto views = strategy._exchange_view_count.cpu().to_vector();
        const auto window = strategy._exchange_window_views.cpu().to_vector();
        const auto coverage = strategy._exchange_window_coverage.cpu().to_vector();
        bool moved = false;
        for (size_t i = 0; i < 4; ++i) {
            const auto from = static_cast<size_t>(permutation.ptr<int64_t>()[i]);
            moved = moved || from != i;
            EXPECT_EQ(views[i], count[from]) << i;
            EXPECT_EQ(window[i], 2.0f * count[from]) << i;
            EXPECT_EQ(coverage[i], 3.0f * count[from]) << i;
        }
        EXPECT_TRUE(moved);
    }
}

// The view count is transient like the summed contribution: a resume drops it and the next
// windows rebuild it from the folds.
TEST(MRNFStrategyTest, DonorRankPerViewResumeRebuildsTheCount) {
    auto splat_data = create_mrnf_test_splat_data(static_cast<int>(kViewMeanSplats));
    MRNF strategy(splat_data);
    const auto params = donor_per_view_fold_params(2, true);
    strategy.initialize(params);
    fold_views(strategy, splat_data, kViewCoverage.size(), 96);
    strategy.update_exchange_importance();
    ASSERT_TRUE(strategy._exchange_view_count.is_valid());
    fold_views(strategy, splat_data, 2, 96);
    ASSERT_TRUE(strategy._exchange_window_views.is_valid());
    std::stringstream ss;
    strategy.serialize(ss);

    auto splat_copy = create_mrnf_test_splat_data(static_cast<int>(kViewMeanSplats));
    MRNF restored(splat_copy);
    restored.initialize(params);
    restored.deserialize(ss);
    EXPECT_FALSE(restored._exchange_view_count.is_valid());
    EXPECT_FALSE(restored._exchange_window_views.is_valid());
    EXPECT_FALSE(restored._exchange_importance_sum.is_valid());

    fold_views(restored, splat_copy, kViewCoverage.size(), 96);
    restored.update_exchange_importance();
    EXPECT_EQ(restored._exchange_view_count.cpu().to_vector(), kCoveringViews);
}

namespace {
    param::OptimizationParameters donor_impact_test_params(const bool impact) {
        auto p = need_growth_test_params(32);
        p.refine_every = 1;
        p.donor_rank_impact = impact;
        return p;
    }

    // Writes one window of removal impact into the buffers the backward would fill.
    void write_impact_window(MRNF& strategy, const std::vector<float>& sum, const std::vector<float>& peak) {
        auto [sum_out, peak_out] = strategy.impact_scratch(100);
        ASSERT_TRUE(sum_out.is_valid());
        ASSERT_TRUE(peak_out.is_valid());
        sum_out.copy_(Tensor::from_vector(sum, TensorShape({sum.size()}), Device::CUDA));
        peak_out.copy_(Tensor::from_vector(peak, TensorShape({peak.size()}), Device::CUDA));
        strategy.on_impact_accumulated(100);
    }
} // namespace

// Off, need-guided growth asks the backward for no impact and its donors are exactly the rows the
// summed blend weight ranks lowest, as before. The impact option also takes precedence over the
// per-view rank.
TEST(MRNFStrategyTest, DonorRankImpactOffKeepsTheBlendWeightDonors) {
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);
    strategy.initialize(donor_impact_test_params(false));
    strategy.set_training_dataset(make_hull_dataset());
    ASSERT_FALSE(strategy.donor_rank_impact_on());
    EXPECT_FALSE(strategy.impact_scratch(100).first.is_valid());
    EXPECT_FALSE(strategy.impact_scratch(100).second.is_valid());
    const size_t n = splat_data.size();

    const std::vector<float> first{1.0f, 4.0f, 0.2f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f};
    const std::vector<float> second{1.0f, 0.0f, 0.2f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 0.5f};
    for (const auto* window : {&first, &second}) {
        strategy.visibility_accumulator().copy_(Tensor::from_vector(*window, TensorShape({n}), Device::CUDA));
        strategy.update_exchange_importance();
    }
    const auto sum = strategy._exchange_importance_sum.cpu().to_vector();
    for (size_t i = 0; i < n; ++i)
        EXPECT_FLOAT_EQ(sum[i], 0.5f * first[i] + second[i]) << i;
    EXPECT_FALSE(strategy._exchange_importance_peak.is_valid());

    strategy._exchange_windows = strategy.exchange_warmup_windows() + 1;
    const auto chosen = strategy.select_exchange_victims(3);
    auto reference = Tensor::empty({3}, Device::CUDA, DataType::Int64);
    const auto eligible = Tensor::ones_bool({n}, Device::CUDA);
    mrnf_strategy::launch_rank_exchange_rows(strategy._exchange_importance_sum.ptr<float>(), nullptr,
                                             eligible.ptr<bool>(), n, 3, reference.ptr<int64_t>());
    EXPECT_EQ(chosen.cpu().to_vector_int64(), reference.cpu().to_vector_int64());
    EXPECT_EQ(selected_rows(chosen), (std::vector<int64_t>{0, 1, 2}));

    // Both donor options on: impact wins, no view count is kept.
    auto both = donor_impact_test_params(true);
    both.donor_rank_per_view = true;
    strategy.set_optimization_params(both);
    EXPECT_TRUE(strategy.donor_rank_impact_on());
    EXPECT_FALSE(strategy.donor_rank_per_view_on());
    both.donor_rank_impact = false;
    strategy.set_optimization_params(both);
    EXPECT_TRUE(strategy.donor_rank_per_view_on());
    // Without need-guided growth the option does nothing (the plain exchange already ranks by impact).
    auto plain = growth_exchange_test_params();
    plain.donor_rank_impact = true;
    strategy.set_optimization_params(plain);
    EXPECT_FALSE(strategy.donor_rank_impact_on());
}

// On, the real backward pass writes the per-splat removal impact into the buffers need-guided
// growth hands out, and the exchange memory takes them as its first window.
TEST(MRNFStrategyTest, DonorRankImpactBackwardWritesImpactUnderNeedGrowth) {
    constexpr int width = 64, height = 48;
    constexpr float sh_c0 = 0.28209479177387814f;
    auto R = Tensor::eye(3, Device::CUDA);
    auto T = Tensor::from_vector(std::vector<float>{0.0f, 0.0f, 4.0f}, TensorShape({3}), Device::CUDA);
    Camera camera(R, T, 60.0f, 60.0f, width / 2.0f, height / 2.0f, Tensor(), Tensor(), CameraModelType::PINHOLE,
                  "impact", "", std::filesystem::path{}, width, height, 0);
    auto bg = Tensor::full({3}, 0.5f, Device::CUDA);

    // A coloured splat in front of a grey one that matches the grey background.
    const std::vector<float> means{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f};
    const std::vector<float> rotations{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    const std::vector<float> opacities{std::log(0.95f / 0.05f), std::log(0.8f / 0.2f)};
    const std::vector<float> sh0{0.4f / sh_c0, -0.4f / sh_c0, 0.0f, 0.0f, 0.0f, 0.0f};
    SplatData splat_data(0, Tensor::from_vector(means, TensorShape({2, 3}), Device::CUDA),
                         Tensor::from_vector(sh0, TensorShape({2, 1, 3}), Device::CUDA),
                         Tensor::zeros({2, 0, 3}, Device::CUDA),
                         Tensor::full({2, 3}, -1.6f, Device::CUDA),
                         Tensor::from_vector(rotations, TensorShape({2, 4}), Device::CUDA),
                         Tensor::from_vector(opacities, TensorShape({2, 1}), Device::CUDA), 1.0f);

    {
        MRNF off(splat_data);
        off.initialize(donor_impact_test_params(false));
        EXPECT_FALSE(off.impact_scratch(100).first.is_valid());
    }

    MRNF strategy(splat_data);
    strategy.initialize(donor_impact_test_params(true));
    ASSERT_TRUE(strategy.donor_rank_impact_on());
    ASSERT_TRUE(strategy.growth_exchange_active(100));
    auto [impact_sum, impact_peak] = strategy.impact_scratch(100);
    ASSERT_TRUE(impact_sum.is_valid());
    ASSERT_TRUE(impact_peak.is_valid());
    ASSERT_EQ(impact_sum.numel(), 2u);

    auto forward = fast_rasterize_forward(camera, splat_data, bg, 0, 0, 0, 0, false);
    ASSERT_TRUE(forward.has_value()) << lfs::format_for_developer(forward.error());
    strategy.get_optimizer().zero_grad(0);
    FastGSFusedExtraGradients fused;
    fused.impact_sum_out = impact_sum.ptr<float>();
    fused.impact_peak_out = impact_peak.ptr<float>();
    fast_rasterize_backward(forward->second, Tensor::zeros_like(forward->first.image), splat_data,
                            strategy.get_optimizer(), {}, {}, DensificationType::None, 1, fused);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    strategy.on_impact_accumulated(100);

    const auto sum = impact_sum.cpu().to_vector();
    const auto peak = impact_peak.cpu().to_vector();
    // Removing the front splat changes the image wherever it covers; the peak is one pixel's change,
    // and the grey splat behind it changes nothing.
    EXPECT_GT(sum[0], 1.0f);
    EXPECT_GT(peak[0], 0.1f);
    EXPECT_LT(peak[0], sum[0]);
    EXPECT_NEAR(sum[1], 0.0f, 1.0e-4f);

    strategy.visibility_accumulator().fill_(1.0f);
    strategy.update_exchange_importance();
    EXPECT_EQ(strategy._exchange_importance_sum.cpu().to_vector(), sum);
    ASSERT_TRUE(strategy._exchange_importance_peak.is_valid());
    EXPECT_EQ(strategy._exchange_importance_peak.cpu().to_vector(), peak);
    EXPECT_FALSE(strategy._exchange_window_impact_sum.is_valid());
    EXPECT_FALSE(strategy._exchange_impact_written);
}

// On, need-guided growth keeps the plain exchange's memory: faded maxima of the window impact
// sum and peak with a half-life of one pass, the same warmup. Its own cosine quota stays.
TEST(MRNFStrategyTest, DonorRankImpactFadesTheMaximaLikeThePlainExchange) {
    const std::vector<float> sum1{4.0f, 1.0f, 2.0f, 8.0f, 0.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f};
    const std::vector<float> peak1{0.4f, 0.9f, 0.1f, 0.2f, 0.0f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f};
    const std::vector<float> sum2{1.0f, 2.0f, 2.0f, 0.0f, 5.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f};
    const std::vector<float> peak2{0.1f, 0.2f, 0.5f, 0.0f, 0.6f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f};
    const auto run = [&](const param::OptimizationParameters& params) {
        auto splat_data = create_mrnf_test_splat_data();
        MRNF strategy(splat_data);
        strategy.initialize(params);
        strategy.set_training_dataset(make_hull_dataset());
        const size_t n = splat_data.size();
        for (const bool second : {false, true}) {
            write_impact_window(strategy, second ? sum2 : sum1, second ? peak2 : peak1);
            strategy.visibility_accumulator().fill_(1.0f);
            strategy.update_exchange_importance();
        }
        EXPECT_EQ(strategy._exchange_windows, 2);
        return std::tuple{strategy._exchange_importance_sum.cpu().to_vector(),
                          strategy._exchange_importance_peak.cpu().to_vector(),
                          strategy.exchange_warmup_windows(), strategy.exchange_quota(2500, n)};
    };

    auto plain_params = growth_exchange_test_params();
    plain_params.refine_every = 1;
    const auto [sum, peak, warmup, quota] = run(donor_impact_test_params(true));
    const auto [plain_sum, plain_peak, plain_warmup, plain_quota] = run(plain_params);
    const float decay = std::pow(0.5f, 1.0f / 2.0f); // refine every step, two views
    for (size_t i = 0; i < sum.size(); ++i) {
        EXPECT_FLOAT_EQ(sum[i], std::fmax(sum2[i], decay * sum1[i])) << i;
        EXPECT_FLOAT_EQ(peak[i], std::fmax(peak2[i], decay * peak1[i])) << i;
    }
    EXPECT_EQ(sum, plain_sum);
    EXPECT_EQ(peak, plain_peak);
    EXPECT_EQ(warmup, plain_warmup);

    // The quota is need-guided growth's cosine taper from the iteration, with or without the option.
    auto splat_data = create_mrnf_test_splat_data();
    MRNF off(splat_data);
    off.initialize(donor_impact_test_params(false));
    EXPECT_EQ(quota, off.exchange_quota(2500, splat_data.size()));
}

// A small splat that is right where it is (little blend weight, a sharp pixel peak) is kept over a
// large low-impact one; the blend-weight ranking frees the small one.
TEST(MRNFStrategyTest, DonorRankImpactKeepsASmallHighPeakSplat) {
    // Row 0: large and faint in impact (blend weight 50, impact sum 0.5, peak 0.01).
    // Row 1: small twig (blend weight 0.3, impact sum 0.4, peak 0.9). The rest are solid.
    const std::vector<float> blend{50.0f, 0.3f, 80.0f, 80.0f, 80.0f, 80.0f, 80.0f, 80.0f, 80.0f, 80.0f};
    const std::vector<float> sum{0.5f, 0.4f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f};
    const std::vector<float> peak{0.01f, 0.9f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    const auto donors = [&](const bool impact, const size_t count) {
        auto splat_data = create_mrnf_test_splat_data();
        MRNF strategy(splat_data);
        strategy.initialize(donor_impact_test_params(impact));
        strategy.set_training_dataset(make_hull_dataset());
        const size_t n = splat_data.size();
        if (impact)
            write_impact_window(strategy, sum, peak);
        strategy.visibility_accumulator().copy_(Tensor::from_vector(blend, TensorShape({n}), Device::CUDA));
        strategy.update_exchange_importance();
        strategy._exchange_windows = strategy.exchange_warmup_windows() + 1;
        return selected_rows(strategy.select_exchange_victims(count));
    };

    EXPECT_EQ(donors(false, 1), (std::vector<int64_t>{1}));
    EXPECT_EQ(donors(true, 1), (std::vector<int64_t>{0}));
    // By summed impact alone the twig would go first (0.4 < 0.5); its peak ranks it last but one.
    EXPECT_EQ(donors(true, 2), (std::vector<int64_t>{0, 2}));
}

// The impact memory follows its splats as the plain exchange's does: appended rows, filled slots
// and split parents restart at 0 and wait a pass; compaction and a Morton reorder carry the
// memory and the window buffers.
TEST(MRNFStrategyTest, DonorRankImpactStateFollowsItsSplats) {
    std::vector<float> iota10(10);
    std::iota(iota10.begin(), iota10.end(), 1.0f);
    {
        // Two rows appended by growth, slot 3 refilled, row 0 split.
        auto splat_data = create_mrnf_test_splat_data();
        MRNF strategy(splat_data);
        strategy.initialize(donor_impact_test_params(true));
        constexpr size_t before = 8;
        const std::vector<float> old_sum(iota10.begin(), iota10.begin() + before);
        strategy._exchange_importance_sum = Tensor::from_vector(old_sum, TensorShape({before}), Device::CUDA);
        strategy._exchange_importance_peak = strategy._exchange_importance_sum.mul(0.1f);
        strategy._exchange_birth = Tensor::zeros({before}, Device::CUDA);
        strategy._exchange_windows = strategy.exchange_warmup_windows() + 3;
        strategy._exchange_free_before_growth = Tensor::zeros_bool({before}, Device::CUDA);
        const auto filled = Tensor::from_vector(std::vector<int>{3}, TensorShape({1}), Device::CUDA).to(DataType::Int64);
        strategy._exchange_free_before_growth.index_put_(filled, Tensor::ones_bool({1}, Device::CUDA));
        const auto parents = Tensor::from_vector(std::vector<int>{0}, TensorShape({1}), Device::CUDA).to(DataType::Int64);
        strategy.record_exchange_births(parents);

        const auto sum = strategy._exchange_importance_sum.cpu().to_vector();
        const auto peak = strategy._exchange_importance_peak.cpu().to_vector();
        const auto birth = strategy._exchange_birth.cpu().to_vector();
        ASSERT_EQ(sum.size(), 10u);
        ASSERT_EQ(peak.size(), 10u);
        for (size_t row = 0; row < 10; ++row) {
            const bool born = row == 0 || row == 3 || row >= before;
            EXPECT_EQ(sum[row], born ? 0.0f : iota10[row]) << row;
            EXPECT_FLOAT_EQ(peak[row], born ? 0.0f : 0.1f * iota10[row]) << row;
            EXPECT_EQ(birth[row], born ? static_cast<float>(strategy._exchange_windows) : 0.0f) << row;
        }
        // Newborns wait a full pass: the two lowest eligible rows are 1 and 2.
        EXPECT_EQ(selected_rows(strategy.select_exchange_victims(2)), (std::vector<int64_t>{1, 2}));
    }
    {
        auto splat_data = create_mrnf_test_splat_data();
        MRNF strategy(splat_data);
        strategy.initialize(donor_impact_test_params(true));
        splat_data._densification_info = Tensor::ones({2, static_cast<size_t>(splat_data.size())}, Device::CUDA);
        const size_t n = splat_data.size();
        strategy._exchange_importance_sum = Tensor::from_vector(kExchangeSum, TensorShape({n}), Device::CUDA);
        strategy._exchange_importance_peak = Tensor::from_vector(kExchangePeak, TensorShape({n}), Device::CUDA);
        strategy._exchange_birth = Tensor::zeros({n}, Device::CUDA);
        strategy._exchange_windows = strategy.exchange_warmup_windows() + 1;
        write_impact_window(strategy, iota10, std::vector<float>(n, 0.5f));
        const auto mask = Tensor::from_vector(
            std::vector<bool>{true, false, false, false, false, false, false, false, false, true},
            TensorShape({10}), Device::CUDA);
        strategy.remove_gaussians(mask);
        ASSERT_EQ(splat_data.size(), 8u);
        EXPECT_EQ(strategy._exchange_importance_sum.cpu().to_vector(),
                  std::vector<float>(kExchangeSum.begin() + 1, kExchangeSum.end() - 1));
        EXPECT_EQ(strategy._exchange_importance_peak.cpu().to_vector(),
                  std::vector<float>(kExchangePeak.begin() + 1, kExchangePeak.end() - 1));
        EXPECT_EQ(strategy._exchange_window_impact_sum.cpu().to_vector(),
                  std::vector<float>(iota10.begin() + 1, iota10.end() - 1));
        EXPECT_TRUE(strategy._exchange_impact_written);
        // Same rows as the plain exchange after this compaction (worse of the sum and peak ranks).
        EXPECT_EQ(selected_rows(strategy.select_exchange_victims(2)), (std::vector<int64_t>{1, 2}));
    }
    {
        auto splat = create_mrnf_test_splat_data(4, 0);
        {
            const std::vector<float> means{3.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
            splat.means().copy_from(Tensor::from_vector(means, TensorShape({4, 3}), Device::CUDA));
        }
        auto params = donor_impact_test_params(true);
        params.max_cap = 8;
        MRNF strategy(splat);
        strategy.initialize(params);
        const std::vector<float> values{10.0f, 20.0f, 30.0f, 40.0f};
        const auto row = [&](const float scale) {
            return Tensor::from_vector(values, TensorShape({4}), Device::CUDA).mul(scale);
        };
        strategy._exchange_importance_sum = row(1.0f);
        strategy._exchange_importance_peak = row(2.0f);
        strategy._exchange_birth = row(3.0f);
        strategy._exchange_window_impact_sum = row(4.0f);
        strategy._exchange_window_impact_peak = row(5.0f);
        const auto result = morton::apply_morton_reorder(
            splat, &strategy.get_optimizer(), nullptr,
            [&](const morton::RowVisitor& visit) { strategy.for_each_gaussian_row(visit); });
        ASSERT_TRUE(result.applied);
        const auto permutation = result.permutation.cpu();
        const std::array<std::pair<const Tensor*, float>, 5> rows{{
            {&strategy._exchange_importance_sum, 1.0f},
            {&strategy._exchange_importance_peak, 2.0f},
            {&strategy._exchange_birth, 3.0f},
            {&strategy._exchange_window_impact_sum, 4.0f},
            {&strategy._exchange_window_impact_peak, 5.0f},
        }};
        bool moved = false;
        for (const auto& [tensor, scale] : rows) {
            const auto got = tensor->cpu().to_vector();
            for (size_t i = 0; i < 4; ++i) {
                const auto from = static_cast<size_t>(permutation.ptr<int64_t>()[i]);
                moved = moved || from != i;
                EXPECT_EQ(got[i], scale * values[from]) << i;
            }
        }
        EXPECT_TRUE(moved);
    }
}

// The impact memory is transient like the plain exchange's: a resume drops it and the next
// windows rebuild it. Switching the option mid-run starts the memory afresh too, since a summed
// blend weight and a faded impact maximum do not mix; the same parameters keep it.
TEST(MRNFStrategyTest, DonorRankImpactResumeAndSwitchStartAfresh) {
    const std::vector<float> sum{4.0f, 1.0f, 2.0f, 8.0f, 0.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f};
    const std::vector<float> peak{0.4f, 0.9f, 0.1f, 0.2f, 0.0f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f};
    const auto params = donor_impact_test_params(true);
    auto splat_data = create_mrnf_test_splat_data();
    MRNF strategy(splat_data);
    strategy.initialize(params);
    strategy.set_training_dataset(make_hull_dataset());
    write_impact_window(strategy, sum, peak);
    strategy.visibility_accumulator().fill_(1.0f);
    strategy.update_exchange_importance();
    write_impact_window(strategy, sum, peak);
    ASSERT_TRUE(strategy._exchange_importance_peak.is_valid());
    std::stringstream ss;
    strategy.serialize(ss);

    auto splat_copy = create_mrnf_test_splat_data();
    MRNF restored(splat_copy);
    restored.initialize(params);
    restored.set_training_dataset(make_hull_dataset());
    restored.deserialize(ss);
    EXPECT_FALSE(restored._exchange_importance_sum.is_valid());
    EXPECT_FALSE(restored._exchange_importance_peak.is_valid());
    EXPECT_FALSE(restored._exchange_window_impact_sum.is_valid());
    EXPECT_FALSE(restored._exchange_impact_written);
    EXPECT_EQ(restored._exchange_windows, 0);
    write_impact_window(restored, sum, peak);
    restored.visibility_accumulator().fill_(1.0f);
    restored.update_exchange_importance();
    EXPECT_EQ(restored._exchange_importance_sum.cpu().to_vector(), sum);
    EXPECT_EQ(restored._exchange_importance_peak.cpu().to_vector(), peak);

    // Same parameters: the memory stays. Switching the option off (or on) drops it.
    strategy.set_optimization_params(params);
    EXPECT_TRUE(strategy._exchange_importance_sum.is_valid());
    EXPECT_TRUE(strategy._exchange_impact_written);
    strategy.set_optimization_params(donor_impact_test_params(false));
    EXPECT_FALSE(strategy._exchange_importance_sum.is_valid());
    EXPECT_FALSE(strategy._exchange_importance_peak.is_valid());
    EXPECT_FALSE(strategy._exchange_window_impact_sum.is_valid());
    EXPECT_EQ(strategy._exchange_windows, 0);
}

namespace {
    param::OptimizationParameters explore_test_params(const bool walk, const bool gate, const bool scouts) {
        auto p = need_growth_test_params(32);
        p.grow_until_iter = 5'000;
        p.explore_walk = walk;
        p.explore_gate_peak = gate;
        p.explore_scouts = scouts;
        return p;
    }

    float raw_of(const float opacity) { return std::log(opacity / (1.0f - opacity)); }

    std::array<float, 9> host_rotation(float w, float x, float y, float z) {
        const float inv = 1.0f / std::sqrt(w * w + x * x + y * y + z * z);
        w *= inv, x *= inv, y *= inv, z *= inv;
        return {1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y - w * z), 2.0f * (x * z + w * y),
                2.0f * (x * y + w * z), 1.0f - 2.0f * (x * x + z * z), 2.0f * (y * z - w * x),
                2.0f * (x * z - w * y), 2.0f * (y * z + w * x), 1.0f - 2.0f * (x * x + y * y)};
    }

    // n identical splats at the origin; returns the displacements of one walk launch.
    struct WalkRun {
        std::vector<float> step; // [n, 3]
        std::vector<float> rendered_after;
        std::vector<float> peak_after;
    };
    WalkRun run_walk(const size_t n, const float opacity, const std::array<float, 4>& quat,
                     const std::array<float, 3>& log_scales, const float amplitude,
                     const std::vector<float>& rendered, const std::vector<float>* peak, const uint64_t seed) {
        std::vector<float> q(4 * n), s(3 * n), o(n, raw_of(opacity));
        for (size_t i = 0; i < n; ++i) {
            std::copy(quat.begin(), quat.end(), q.begin() + 4 * i);
            std::copy(log_scales.begin(), log_scales.end(), s.begin() + 3 * i);
        }
        auto means = Tensor::zeros({n, 3}, Device::CUDA);
        auto rot = Tensor::from_vector(q, TensorShape({n, 4}), Device::CUDA);
        auto scl = Tensor::from_vector(s, TensorShape({n, 3}), Device::CUDA);
        auto raw = Tensor::from_vector(o, TensorShape({n}), Device::CUDA);
        auto marks = Tensor::from_vector(rendered, TensorShape({n}), Device::CUDA);
        Tensor peaks = peak ? Tensor::from_vector(*peak, TensorShape({n}), Device::CUDA) : Tensor();
        mrnf_strategy::ExploreWalk walk;
        walk.rotations = rot.ptr<float>();
        walk.log_scales = scl.ptr<float>();
        walk.rendered = marks.ptr<float>();
        walk.blend_peak = peak ? peaks.ptr<float>() : nullptr;
        walk.amplitude = amplitude;
        walk.peak_decay = 0.5f;
        mrnf_strategy::launch_explore_walk(means.ptr<float>(), raw.ptr<float>(), nullptr, 0, walk, n, seed);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        return {means.cpu().to_vector(), marks.cpu().to_vector(), peak ? peaks.cpu().to_vector() : std::vector<float>{}};
    }

    // Sample covariance of the [n, 3] steps (mean zero by construction).
    std::array<double, 9> step_covariance(const std::vector<float>& step) {
        std::array<double, 9> c{};
        const size_t n = step.size() / 3;
        for (size_t i = 0; i < n; ++i)
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    c[3 * a + b] += static_cast<double>(step[3 * i + a]) * step[3 * i + b];
        for (auto& v : c)
            v /= static_cast<double>(n);
        return c;
    }

    double moved_rows(const std::vector<float>& step) {
        size_t moved = 0;
        for (size_t i = 0; i < step.size() / 3; ++i)
            moved += (step[3 * i] != 0.0f || step[3 * i + 1] != 0.0f || step[3 * i + 2] != 0.0f) ? 1 : 0;
        return static_cast<double>(moved);
    }
} // namespace

// Off (the default), MRNF asks the backward for no exploration rows, keeps the position noise
// (isotropic in world units whatever the splat's shape, opaque splats still) and the receiver
// children keep the parent's opacity.
TEST(MRNFStrategyTest, ExploreOffKeepsTheNoiseAndTheChildren) {
    constexpr size_t n = 4000;
    auto splat_data = create_mrnf_test_splat_data(static_cast<int>(n), 0);
    {
        std::vector<float> scales(3 * n), opacity(n);
        for (size_t i = 0; i < n; ++i) {
            scales[3 * i] = std::log(0.3f), scales[3 * i + 1] = std::log(0.01f), scales[3 * i + 2] = std::log(0.002f);
            opacity[i] = raw_of(i % 2 == 0 ? 0.001f : 0.9f);
        }
        splat_data.scaling_raw().copy_from(Tensor::from_vector(scales, TensorShape({n, 3}), Device::CUDA));
        splat_data.opacity_raw().copy_from(Tensor::from_vector(opacity, TensorShape({n, 1}), Device::CUDA));
    }
    MRNF strategy(splat_data);
    auto params = explore_test_params(false, false, false);
    params.max_cap = 0;
    strategy.initialize(params);
    EXPECT_FALSE(strategy.explore_walk_on());
    const auto scratch = strategy.explore_scratch(100);
    EXPECT_FALSE(scratch.first.is_valid());
    EXPECT_FALSE(scratch.second.is_valid());
    ASSERT_TRUE(strategy._bounds_valid);

    strategy.visibility_accumulator().fill_(1.0f);
    const auto before = splat_data.means().cpu().to_vector();
    strategy.inject_noise(100);
    const auto after = splat_data.means().cpu().to_vector();
    std::array<double, 3> var{};
    for (size_t i = 0; i < n; ++i) {
        for (int d = 0; d < 3; ++d) {
            const double step = after[3 * i + d] - before[3 * i + d];
            if (i % 2 == 1)
                EXPECT_EQ(step, 0.0) << i; // opaque
            else
                var[d] += step * step;
        }
    }
    ASSERT_GT(var[0], 0.0);
    EXPECT_NEAR(var[1] / var[0], 1.0, 0.15);
    EXPECT_NEAR(var[2] / var[0], 1.0, 0.15);
    EXPECT_FALSE(strategy._explore_rendered.is_valid());

    // Receiver children at the cap keep the parent's opacity and no scout rows exist.
    constexpr size_t m = 200;
    auto cap_data = create_mrnf_test_splat_data(static_cast<int>(m));
    MRNF capped(cap_data);
    capped.initialize(need_growth_test_params(static_cast<int>(m)));
    std::vector<float> error(m, 1.0f), footprint(m, 0.01f);
    set_need_window(cap_data, capped._refine_weight_max, capped.visibility_accumulator(), error, footprint);
    capped._exchange_importance_sum = Tensor::full({m}, 10.0f, Device::CUDA);
    capped._exchange_birth = Tensor::zeros({m}, Device::CUDA);
    capped._exchange_windows = capped.exchange_warmup_windows() + 1;
    capped.soft_delete_rows(row_range(m - 2, m));
    const auto result = capped.grow_and_split(0, 2, capped.compute_refine_candidates(), 3);
    ASSERT_GT(result.receiver_splits, 0);
    const auto opacity = cap_data.opacity_raw().cpu().to_vector();
    const auto free_cpu = capped._free_mask.cpu();
    for (size_t row = 0; row < m; ++row)
        if (!free_cpu.ptr<bool>()[row])
            EXPECT_EQ(opacity[row], 0.0f) << row;
    EXPECT_FALSE(capped._explore_scout_renders.is_valid());
}

// The walk is a step in units of the splat's own sigma along its own axes: over many draws the
// displacement covariance is k^2 R diag(s^2) R^T with k = min(A E(o) (1 - o)^150, 1).
TEST(MRNFStrategyTest, ExploreWalkCovarianceFollowsTheSplatShape) {
    constexpr size_t n = 40'000;
    const std::array<float, 4> quat{0.9f, 0.3f, -0.2f, 0.25f};
    const std::array<float, 3> log_s{std::log(0.1f), std::log(0.02f), std::log(0.005f)};
    const auto R = host_rotation(quat[0], quat[1], quat[2], quat[3]);
    const std::vector<float> rendered(n, 1.0f);
    const float o = 0.01f;
    const float lift = std::sqrt(2.0f * std::log(255.0f * o));
    const float gate = std::pow(1.0f - o, 150.0f);
    for (const float amplitude : {1.0f, 8.0f}) {
        SCOPED_TRACE(amplitude);
        const double k = std::min(amplitude * lift * gate, 1.0f);
        if (amplitude == 8.0f)
            ASSERT_EQ(k, 1.0); // capped at one sigma per step
        const auto run = run_walk(n, o, quat, log_s, amplitude, rendered, nullptr, 1234);
        const auto cov = step_covariance(run.step);
        double scale = 0.0;
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                double expected = 0.0;
                for (int j = 0; j < 3; ++j)
                    expected += R[3 * a + j] * R[3 * b + j] * std::exp(2.0 * log_s[j]);
                expected *= k * k;
                scale = std::max(scale, std::abs(expected));
                EXPECT_NEAR(cov[3 * a + b], expected, 0.03 * k * k * 0.01) << a << "," << b;
            }
        }
        EXPECT_GT(scale, 0.0);
    }
    // At or below 1/255 opacity nothing moves.
    const auto faint = run_walk(64, 1.0f / 300.0f, quat, log_s, 8.0f, std::vector<float>(64, 1.0f), nullptr, 7);
    EXPECT_EQ(moved_rows(faint.step), 0.0);
}

// A(t) = 8 x 0.01^(t / iterations), not stopping at grow_until_iter; only the splats the backward
// marked as rendered in this step walk, and the step clears the marks.
TEST(MRNFStrategyTest, ExploreWalkAmplitudeFollowsTheRunAndOnlyRenderedSplatsWalk) {
    constexpr size_t n = 10;
    auto splat_data = create_mrnf_test_splat_data(static_cast<int>(n), 0);
    splat_data.opacity_raw().fill_(raw_of(0.01f));
    splat_data.scaling_raw().fill_(std::log(0.05f));
    MRNF strategy(splat_data);
    auto params = explore_test_params(true, false, false);
    strategy.initialize(params);
    ASSERT_TRUE(strategy.explore_walk_on());
    EXPECT_NEAR(strategy.explore_walk_amplitude(0), 8.0f, 1e-5f);
    EXPECT_NEAR(strategy.explore_walk_amplitude(5'000), 0.8f, 1e-5f);
    EXPECT_NEAR(strategy.explore_walk_amplitude(10'000), 0.08f, 1e-6f);
    params.explore_walk_scale = 0.5f;
    strategy.set_optimization_params(params);
    EXPECT_NEAR(strategy.explore_walk_amplitude(0), 4.0f, 1e-5f);

    auto [rendered, peak] = strategy.explore_scratch(7'000);
    ASSERT_TRUE(rendered.is_valid());
    EXPECT_FALSE(peak.is_valid()); // no peak row without explore_gate_peak
    ASSERT_EQ(rendered.numel(), n);
    std::vector<float> marks(n, 0.0f);
    marks[2] = marks[5] = 1.0f;
    rendered.copy_(Tensor::from_vector(marks, TensorShape({n}), Device::CUDA));
    const auto before = splat_data.means().cpu().to_vector();
    strategy.explore_step(7'000); // past grow_until_iter: the walk goes on
    const auto after = splat_data.means().cpu().to_vector();
    for (size_t i = 0; i < n; ++i) {
        const bool moved = after[3 * i] != before[3 * i] || after[3 * i + 1] != before[3 * i + 1] ||
                           after[3 * i + 2] != before[3 * i + 2];
        EXPECT_EQ(moved, marks[i] > 0.0f) << i;
    }
    EXPECT_EQ(strategy._explore_rendered.cpu().to_vector(), std::vector<float>(n, 0.0f));

    // Opaque splats stay put.
    splat_data.opacity_raw().fill_(raw_of(0.9f));
    strategy.explore_scratch(7'001).first.fill_(1.0f);
    const auto opaque_before = splat_data.means().cpu().to_vector();
    strategy.explore_step(7'001);
    EXPECT_EQ(splat_data.means().cpu().to_vector(), opaque_before);
}

// With the peak gate an opaque splat hidden behind others (low peak blend weight) walks and a
// visible opaque one does not; without it neither walks. The peak fades by the given factor.
TEST(MRNFStrategyTest, ExploreGatePeakWalksHiddenOpaqueSplatsOnly) {
    constexpr size_t n = 400;
    const std::array<float, 4> quat{1.0f, 0.0f, 0.0f, 0.0f};
    const std::array<float, 3> log_s{std::log(0.05f), std::log(0.05f), std::log(0.05f)};
    std::vector<float> peak(n);
    for (size_t i = 0; i < n; ++i)
        peak[i] = i < n / 2 ? 0.001f : 0.85f; // hidden, then visible
    const std::vector<float> rendered(n, 1.0f);
    const auto gated = run_walk(n, 0.9f, quat, log_s, 8.0f, rendered, &peak, 99);
    for (size_t i = 0; i < n; ++i) {
        const bool moved = gated.step[3 * i] != 0.0f || gated.step[3 * i + 1] != 0.0f || gated.step[3 * i + 2] != 0.0f;
        EXPECT_EQ(moved, i < n / 2) << i;
        EXPECT_FLOAT_EQ(gated.peak_after[i], 0.5f * peak[i]) << i;
    }
    const auto plain = run_walk(n, 0.9f, quat, log_s, 8.0f, rendered, nullptr, 99);
    EXPECT_EQ(moved_rows(plain.step), 0.0);

    // The strategy keeps the peak row only with the gate on, and fades it to half in 800 steps.
    auto splat_data = create_mrnf_test_splat_data(4, 0);
    MRNF strategy(splat_data);
    strategy.initialize(explore_test_params(true, true, false));
    ASSERT_TRUE(strategy.explore_gate_peak_on());
    auto [marks, peaks] = strategy.explore_scratch(100);
    ASSERT_TRUE(peaks.is_valid());
    peaks.fill_(1.0f);
    for (int step = 0; step < 800; ++step)
        strategy.explore_step(100 + step);
    for (const float v : strategy._explore_peak.cpu().to_vector())
        EXPECT_NEAR(v, 0.5f, 1e-3f);
}

// The real backward marks every splat visible in the view and writes each splat's peak blend
// weight: high for the front splat, low for the one mostly hidden behind it.
TEST(MRNFStrategyTest, ExploreBackwardWritesVisibilityAndPeakBlendWeight) {
    constexpr int width = 64, height = 48;
    constexpr float sh_c0 = 0.28209479177387814f;
    auto R = Tensor::eye(3, Device::CUDA);
    auto T = Tensor::from_vector(std::vector<float>{0.0f, 0.0f, 4.0f}, TensorShape({3}), Device::CUDA);
    Camera camera(R, T, 60.0f, 60.0f, width / 2.0f, height / 2.0f, Tensor(), Tensor(), CameraModelType::PINHOLE,
                  "explore", "", std::filesystem::path{}, width, height, 0);
    auto bg = Tensor::full({3}, 0.5f, Device::CUDA);
    const std::vector<float> means{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f};
    const std::vector<float> rotations{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    const std::vector<float> opacities{raw_of(0.95f), raw_of(0.8f)};
    const std::vector<float> sh0{0.4f / sh_c0, -0.4f / sh_c0, 0.0f, 0.0f, 0.0f, 0.0f};
    SplatData splat_data(0, Tensor::from_vector(means, TensorShape({2, 3}), Device::CUDA),
                         Tensor::from_vector(sh0, TensorShape({2, 1, 3}), Device::CUDA),
                         Tensor::zeros({2, 0, 3}, Device::CUDA),
                         Tensor::full({2, 3}, -1.6f, Device::CUDA),
                         Tensor::from_vector(rotations, TensorShape({2, 4}), Device::CUDA),
                         Tensor::from_vector(opacities, TensorShape({2, 1}), Device::CUDA), 1.0f);

    MRNF strategy(splat_data);
    strategy.initialize(explore_test_params(true, true, false));
    auto [rendered, peak] = strategy.explore_scratch(100);
    ASSERT_TRUE(rendered.is_valid());
    ASSERT_TRUE(peak.is_valid());

    auto forward = fast_rasterize_forward(camera, splat_data, bg, 0, 0, 0, 0, false);
    ASSERT_TRUE(forward.has_value()) << lfs::format_for_developer(forward.error());
    strategy.get_optimizer().zero_grad(0);
    FastGSFusedExtraGradients fused;
    fused.explore_rendered = rendered.ptr<float>();
    fused.explore_blend_peak = peak.ptr<float>();
    fast_rasterize_backward(forward->second, Tensor::zeros_like(forward->first.image), splat_data,
                            strategy.get_optimizer(), {}, {}, DensificationType::None, 1, fused);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EXPECT_EQ(rendered.cpu().to_vector(), (std::vector<float>{1.0f, 1.0f}));
    const auto w = peak.cpu().to_vector();
    EXPECT_GT(w[0], 0.8f);
    EXPECT_LE(w[0], 1.0f);
    EXPECT_GT(w[1], 0.0f);
    EXPECT_LT(w[1], 0.5f * w[0]);

    // Without the options the backward gets no rows.
    MRNF off(splat_data);
    off.initialize(explore_test_params(false, false, false));
    EXPECT_FALSE(off.explore_scratch(100).first.is_valid());
}

// Scouts: a seeded share of the children (each child of a split draws) gets the scout opacity and a
// render count; the same seed gives the same children, the share tapers to 0 by grow_until_iter, and
// at the cap every other property is the receiver rule's.
TEST(MRNFStrategyTest, ExploreScoutsShareOpacityTaperAndSeed) {
    constexpr size_t pairs = 20'000;
    const auto draw = [&](const uint64_t seed, const float share) {
        auto opacity = Tensor::zeros({2 * pairs}, Device::CUDA);
        auto renders = Tensor::zeros({2 * pairs}, Device::CUDA);
        const auto first = row_range(0, pairs);
        const auto second = row_range(pairs, 2 * pairs);
        mrnf_strategy::launch_explore_scouts(opacity.ptr<float>(), renders.ptr<float>(), first.ptr<int64_t>(),
                                             second.ptr<int64_t>(), pairs, seed, 0, share, raw_of(0.012f), 20.0f);
        return std::pair{opacity.cpu().to_vector(), renders.cpu().to_vector()};
    };
    const auto [opacity, renders] = draw(42, 0.3f);
    size_t scouts = 0;
    for (size_t i = 0; i < 2 * pairs; ++i) {
        const bool scout = opacity[i] != 0.0f;
        scouts += scout ? 1 : 0;
        if (scout) {
            EXPECT_FLOAT_EQ(opacity[i], raw_of(0.012f));
            EXPECT_EQ(renders[i], 20.0f);
        } else {
            EXPECT_EQ(renders[i], 0.0f);
        }
    }
    EXPECT_NEAR(static_cast<double>(scouts) / (2.0 * pairs), 0.3, 0.01);
    EXPECT_EQ(draw(42, 0.3f).first, opacity);
    EXPECT_NE(draw(43, 0.3f).first, opacity);

    constexpr size_t m = 200;
    auto splat_data = create_mrnf_test_splat_data(static_cast<int>(m));
    MRNF strategy(splat_data);
    auto params = explore_test_params(false, false, true);
    params.max_cap = static_cast<int>(m);
    params.explore_scout_share = 1.0f;
    strategy.initialize(params);
    EXPECT_FLOAT_EQ(strategy.explore_scout_share_at(0), 1.0f);
    EXPECT_NEAR(strategy.explore_scout_share_at(2'500), 0.5f, 1e-6f);
    EXPECT_FLOAT_EQ(strategy.explore_scout_share_at(5'000), 0.0f);
    EXPECT_FLOAT_EQ(strategy.explore_scout_share_at(7'000), 0.0f);

    std::vector<float> error(m, 1.0f), footprint(m, 0.01f);
    set_need_window(splat_data, strategy._refine_weight_max, strategy.visibility_accumulator(), error, footprint);
    strategy._exchange_importance_sum = Tensor::full({m}, 10.0f, Device::CUDA);
    strategy._exchange_birth = Tensor::zeros({m}, Device::CUDA);
    strategy._exchange_windows = strategy.exchange_warmup_windows() + 1;
    strategy.soft_delete_rows(row_range(m - 2, m));
    const auto result = strategy.grow_and_split(0, 2, strategy.compute_refine_candidates(), 3);
    ASSERT_EQ(result.receiver_splits, 5);
    const auto split_opacity = splat_data.opacity_raw().cpu().to_vector();
    const auto scales = splat_data.scaling_raw().cpu().to_vector();
    ASSERT_TRUE(strategy._explore_scout_renders.is_valid());
    const auto held = strategy._explore_scout_renders.cpu().to_vector();
    size_t scout_rows = 0;
    for (size_t row = 0; row < splat_data.size(); ++row) {
        if (held[row] > 0.0f) {
            ++scout_rows;
            EXPECT_FLOAT_EQ(split_opacity[row], raw_of(0.012f)) << row;
            EXPECT_NEAR(scales[3 * row + 0], std::log(0.5f), 1e-6f) << row;
            EXPECT_NEAR(scales[3 * row + 1], std::log(0.7f), 1e-6f) << row;
        } else {
            EXPECT_EQ(split_opacity[row], 0.0f) << row;
        }
    }
    EXPECT_EQ(scout_rows, 10u); // both children of each of the 5 splits at share 1

    // At grow_until_iter the share is 0: the refills are plain receivers.
    auto late_data = create_mrnf_test_splat_data(static_cast<int>(m));
    MRNF late(late_data);
    late.initialize(params);
    set_need_window(late_data, late._refine_weight_max, late.visibility_accumulator(), error, footprint);
    late.soft_delete_rows(row_range(m - 2, m));
    const auto late_result = late.grow_and_split(5'000, 2, late.compute_refine_candidates(), 0);
    ASSERT_GT(late_result.receiver_splits, 0);
    for (const float v : late_data.opacity_raw().cpu().to_vector())
        EXPECT_EQ(v, 0.0f);
}

// A scout skips the opacity decay until the backward has rendered it 20 times; then the normal
// rules apply. Other splats decay as before.
TEST(MRNFStrategyTest, ExploreScoutsSkipOpacityDecayUntilRendered) {
    auto splat_data = create_mrnf_test_splat_data(4, 0);
    splat_data.opacity_raw().fill_(raw_of(0.012f));
    MRNF strategy(splat_data);
    auto params = explore_test_params(false, false, true);
    params.opacity_decay = 0.004f;
    strategy.initialize(params);
    auto [rendered, peak] = strategy.explore_scratch(100);
    ASSERT_TRUE(rendered.is_valid());
    EXPECT_FALSE(peak.is_valid());
    strategy._explore_scout_renders.copy_(Tensor::from_vector(std::vector<float>{20.0f, 0.0f, 20.0f, 0.0f},
                                                              TensorShape({4}), Device::CUDA));
    strategy.apply_decay(100);
    auto opacity = splat_data.opacity_raw().cpu().to_vector();
    EXPECT_FLOAT_EQ(opacity[0], raw_of(0.012f));
    EXPECT_FLOAT_EQ(opacity[2], raw_of(0.012f));
    EXPECT_LT(opacity[1], raw_of(0.012f) - 0.1f);

    // Row 0 is rendered every step, row 2 never: after 19 renders row 0 is still held, after 20 not.
    for (int step = 0; step < 20; ++step) {
        auto marks = strategy.explore_scratch(100 + step).first;
        marks.copy_(Tensor::from_vector(std::vector<float>{1.0f, 1.0f, 0.0f, 0.0f}, TensorShape({4}), Device::CUDA));
        strategy.explore_step(100 + step);
        if (step == 18)
            EXPECT_EQ(strategy._explore_scout_renders.cpu().to_vector()[0], 1.0f);
    }
    EXPECT_EQ(strategy._explore_scout_renders.cpu().to_vector(), (std::vector<float>{0.0f, 0.0f, 20.0f, 0.0f}));
    strategy.apply_decay(200);
    opacity = splat_data.opacity_raw().cpu().to_vector();
    EXPECT_LT(opacity[0], raw_of(0.012f) - 0.1f);
    EXPECT_FLOAT_EQ(opacity[2], raw_of(0.012f));
}

// The exploration rows follow their splats: freed slots start afresh, compaction and the Morton
// reorder carry them, appended rows start at zero.
TEST(MRNFStrategyTest, ExploreRowsFollowTheirSplats) {
    const std::vector<float> values{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    auto splat_data = create_mrnf_test_splat_data(10, 0);
    MRNF strategy(splat_data);
    strategy.initialize(explore_test_params(true, true, true));
    strategy.explore_scratch(100);
    const auto fill = [&](const size_t n) {
        const std::vector<float> v(values.begin(), values.begin() + n);
        strategy._explore_rendered.copy_(Tensor::from_vector(v, TensorShape({n}), Device::CUDA));
        strategy._explore_peak.copy_(Tensor::from_vector(v, TensorShape({n}), Device::CUDA).mul(2.0f));
        strategy._explore_scout_renders.copy_(Tensor::from_vector(v, TensorShape({n}), Device::CUDA).mul(3.0f));
    };
    fill(10);
    strategy.soft_delete_rows(row_range(4, 5));
    for (const auto* row : {&strategy._explore_rendered, &strategy._explore_peak, &strategy._explore_scout_renders})
        EXPECT_EQ(row->cpu().to_vector()[4], 0.0f);

    fill(10);
    const auto mask = Tensor::from_vector(
        std::vector<bool>{true, false, false, false, false, false, false, false, false, true}, TensorShape({10}),
        Device::CUDA);
    strategy.remove_gaussians(mask);
    ASSERT_EQ(splat_data.size(), 8u);
    const std::vector<float> kept(values.begin() + 1, values.end() - 1);
    EXPECT_EQ(strategy._explore_rendered.cpu().to_vector(), kept);
    EXPECT_EQ(strategy._explore_peak.cpu().to_vector().front(), 2.0f * kept.front());
    EXPECT_EQ(strategy._explore_scout_renders.cpu().to_vector().back(), 3.0f * kept.back());

    // Two appended rows start at zero; the prefix is kept.
    auto bigger = create_mrnf_test_splat_data(4, 0);
    {
        const std::vector<float> means{3.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
        bigger.means().copy_from(Tensor::from_vector(means, TensorShape({4, 3}), Device::CUDA));
    }
    auto params = explore_test_params(true, true, true);
    params.max_cap = 8;
    MRNF reorder(bigger);
    reorder.initialize(params);
    reorder.explore_scratch(100);
    reorder._explore_rendered.copy_(Tensor::from_vector(std::vector<float>{10.0f, 20.0f, 30.0f, 40.0f},
                                                        TensorShape({4}), Device::CUDA));
    const auto result = morton::apply_morton_reorder(
        bigger, &reorder.get_optimizer(), nullptr,
        [&](const morton::RowVisitor& visit) { reorder.for_each_gaussian_row(visit); });
    ASSERT_TRUE(result.applied);
    const auto permutation = result.permutation.cpu();
    const auto got = reorder._explore_rendered.cpu().to_vector();
    for (size_t i = 0; i < 4; ++i)
        EXPECT_EQ(got[i], 10.0f * static_cast<float>(permutation.ptr<int64_t>()[i] + 1)) << i;

    reorder._explore_rendered = Tensor::from_vector(std::vector<float>{1.0f, 2.0f}, TensorShape({2}), Device::CUDA);
    reorder.ensure_explore_rows();
    EXPECT_EQ(reorder._explore_rendered.cpu().to_vector(), (std::vector<float>{1.0f, 2.0f, 0.0f, 0.0f}));
}

// Both halves of a receiver split get receiver_opacity_scale x the parent's opacity and
// receiver_cross_scale on the two short axes; the defaults (1, 0.7) keep the raw opacity bit for bit.
// With receiver_opacity_scale_end > 0 the factor ramps linearly to it over [0, grow_until_iter] and
// then stays; 0 keeps the start factor.
TEST(MRNFStrategyTest, ReceiverSplitTakesTheOpacityAndCrossScale) {
    struct ReceiverRun {
        std::vector<float> opacity;
        std::vector<float> scales;
        std::vector<bool> free;
        int receivers = 0;
    };
    // Two refills at the cap with need-guided growth: the receiver rule splits two parents.
    const auto run_refill = [](const param::OptimizationParameters& params, const int iter) {
        constexpr size_t m = 200;
        auto data = create_mrnf_test_splat_data(static_cast<int>(m));
        MRNF strategy(data);
        strategy.initialize(params);
        std::vector<float> error(m, 1.0f), footprint(m, 0.01f);
        set_need_window(data, strategy._refine_weight_max, strategy.visibility_accumulator(), error, footprint);
        strategy.soft_delete_rows(row_range(m - 2, m));
        const auto result = strategy.grow_and_split(iter, 2, strategy.compute_refine_candidates(), 0);
        ReceiverRun run{data.opacity_raw().cpu().to_vector(), data.scaling_raw().cpu().to_vector(), {}, result.receiver_splits};
        const auto free_cpu = strategy._free_mask.cpu();
        for (size_t row = 0; row < run.opacity.size(); ++row)
            run.free.push_back(free_cpu.ptr<bool>()[row]);
        return run;
    };
    const auto check = [](const ReceiverRun& run, const float opacity_scale, const float cross_scale) {
        ASSERT_GT(run.receivers, 0);
        size_t children = 0;
        for (size_t row = 0; row < run.opacity.size(); ++row) {
            SCOPED_TRACE(row);
            if (run.free[row])
                continue;
            if (run.scales[3 * row] == 0.0f) { // untouched: parents start at log scale 0, raw opacity 0
                EXPECT_EQ(run.opacity[row], 0.0f);
                continue;
            }
            ++children;
            EXPECT_NEAR(run.scales[3 * row + 0], std::log(0.5f), 1e-6f);
            EXPECT_NEAR(run.scales[3 * row + 1], std::log(cross_scale), 1e-6f);
            EXPECT_NEAR(run.scales[3 * row + 2], std::log(cross_scale), 1e-6f);
            if (opacity_scale == 1.0f)
                EXPECT_EQ(run.opacity[row], 0.0f); // the parent's raw value, exactly
            else
                EXPECT_NEAR(run.opacity[row], raw_of(0.5f * opacity_scale), 1e-5f);
        }
        EXPECT_EQ(children, 2u * static_cast<size_t>(run.receivers));
    };

    auto params = need_growth_test_params(200); // grow_until_iter 10'000
    EXPECT_EQ(params.receiver_opacity_scale, 1.0f);
    EXPECT_EQ(params.receiver_opacity_scale_end, 0.0f);
    EXPECT_EQ(params.receiver_cross_scale, 0.7f);
    check(run_refill(params, 0), 1.0f, 0.7f);

    params.receiver_opacity_scale = 0.5f;
    params.receiver_cross_scale = 0.85f;
    check(run_refill(params, 0), 0.5f, 0.85f);

    {
        auto data = create_mrnf_test_splat_data(4, 0);
        MRNF strategy(data);
        strategy.initialize(params);
        EXPECT_EQ(strategy.receiver_opacity_scale_at(0), 0.5f);
        EXPECT_EQ(strategy.receiver_opacity_scale_at(20'000), 0.5f);
        params.receiver_opacity_scale_end = 0.55f;
        strategy.set_optimization_params(params);
        EXPECT_FLOAT_EQ(strategy.receiver_opacity_scale_at(0), 0.5f);
        EXPECT_FLOAT_EQ(strategy.receiver_opacity_scale_at(5'000), 0.525f);
        EXPECT_FLOAT_EQ(strategy.receiver_opacity_scale_at(10'000), 0.55f);
        EXPECT_FLOAT_EQ(strategy.receiver_opacity_scale_at(15'000), 0.55f);
    }
    check(run_refill(params, 5'000), 0.525f, 0.85f);
}

namespace {
    param::OptimizationParameters retire_test_params(const size_t cap, const float share) {
        auto p = view_mean_test_params(false);
        p.max_cap = static_cast<int>(cap);
        p.grow_until_iter = 5'000;
        p.mrnf_retire_share = share;
        return p;
    }
} // namespace

// The loss error map implies the compressed view mean (and with it threshold-free candidates); the
// option needs a loss that writes the map, and the defaults leave everything off.
TEST(MRNFStrategyTest, LossErrorMapImpliesTheCompressedViewMean) {
    auto defaults = param::OptimizationParameters::mrnf_defaults();
    EXPECT_FALSE(defaults.mrnf_loss_error_map);
    EXPECT_EQ(defaults.mrnf_retire_share, 0.0f);
    EXPECT_EQ(defaults.mrnf_retire_grace, 30);
    EXPECT_FALSE(defaults.refine_error_compress_active());

    defaults.mrnf_loss_error_map = true;
    EXPECT_TRUE(defaults.refine_error_compress_active());
    EXPECT_NE(defaults.validate().find("mrnf_loss_error_map"), std::string::npos);
    defaults.ssim_blur_tolerant = true;
    EXPECT_EQ(defaults.validate().find("mrnf_loss_error_map"), std::string::npos);
    defaults.ssim_blur_tolerant = false;
    defaults.loss_pyramid = true;
    EXPECT_EQ(defaults.validate().find("mrnf_loss_error_map"), std::string::npos);

    auto params = view_mean_test_params(false);
    params.mrnf_loss_error_map = true;
    params.loss_pyramid = true;

    auto splat_data = create_mrnf_test_splat_data(static_cast<int>(kViewMeanSplats));
    MRNF strategy(splat_data);
    strategy.initialize(params);
    EXPECT_TRUE(strategy.refine_error_compress_on());
    EXPECT_TRUE(strategy.refine_error_view_mean_on());
    EXPECT_EQ(splat_data._densification_info.shape()[0], 4u);

    // Switching the map on mid-run starts a fresh window, as switching the compression does.
    auto plain = view_mean_test_params(false);
    MRNF switched(splat_data);
    switched.initialize(plain);
    EXPECT_FALSE(switched.refine_error_compress_on());
    switched._refine_weight_max.fill_(1.0f);
    switched.set_optimization_params(params);
    EXPECT_TRUE(switched.refine_error_compress_on());
    EXPECT_EQ(switched._refine_weight_max.cpu().to_vector(), std::vector<float>(kViewMeanSplats, 0.0f));
}

// With the loss error map, the score a refine ranks by is the match strategy's: per view the splat's
// error sum ^0.2, the running mean over the views where it had error. Fed the same per-view sums, the
// MRNF window mean and the match step update agree.
TEST(MRNFStrategyTest, LossErrorMapScoreEqualsTheMatchScore) {
    auto splat_data = create_mrnf_test_splat_data(static_cast<int>(kViewMeanSplats));
    MRNF strategy(splat_data);
    auto params = view_mean_test_params(false);
    params.mrnf_loss_error_map = true;
    params.ssim_blur_tolerant = true;
    strategy.initialize(params);
    fold_views(strategy, splat_data, kViewCoverage.size(), 96);
    auto mrnf_score = strategy._refine_weight_max.clone();
    mrnf_strategy::launch_refine_error_view_mean(mrnf_score.ptr<float>(), splat_data._densification_info.ptr<float>(),
                                                 kViewMeanSplats);

    constexpr size_t n = kViewMeanSplats;
    const auto zeros = [] { return Tensor::zeros({n}, Device::CUDA); };
    auto error = zeros(), step_peak = zeros(), visible = zeros(), score = zeros(), score_count = zeros(),
         oversize = zeros(), peak = zeros(), streak = zeros();
    for (size_t v = 0; v < kViewCoverage.size(); ++v) {
        error.copy_(Tensor::from_vector(kViewError[v], TensorShape({n}), Device::CUDA));
        visible.copy_(Tensor::from_vector(kViewCoverage[v], TensorShape({n}), Device::CUDA));
        step_peak.copy_(visible);
        match::StepRows rows;
        rows.error = error.ptr<float>();
        rows.step_peak = step_peak.ptr<float>();
        rows.visible = visible.ptr<float>();
        rows.score = score.ptr<float>();
        rows.score_count = score_count.ptr<float>();
        rows.oversize = oversize.ptr<float>();
        rows.peak = peak.ptr<float>();
        rows.streak = streak.ptr<float>();
        match::StepSettings settings;
        settings.score_power = 0.2f;
        match::launch_step_update(rows, settings, n);
    }
    const auto expected = score.cpu().to_vector();
    const auto got = mrnf_score.cpu().to_vector();
    for (size_t i = 0; i < n; ++i)
        EXPECT_NEAR(got[i], expected[i], 1e-6f * std::max(1.0f, expected[i])) << i;
    EXPECT_GT(expected[4], 0.0f); // error in one of four views: that view alone
    EXPECT_EQ(expected[3], 0.0f); // never seen
}

// Off (the default share 0): no retirement rows, the backward gets the exploration rows it got before
// (none, or the gate's own running peak), and a refine retires nothing.
TEST(MRNFStrategyTest, RetireOffKeepsTheRefineBitIdentical) {
    constexpr size_t n = 32;
    for (const bool gate : {false, true}) {
        SCOPED_TRACE(gate);
        auto splat_data = create_mrnf_test_splat_data(static_cast<int>(n), 0);
        MRNF strategy(splat_data);
        auto params = retire_test_params(n, 0.0f);
        params.need_guided_growth = true;
        params.explore_walk = gate;
        params.explore_gate_peak = gate;
        strategy.initialize(params);
        EXPECT_FALSE(strategy.retire_on());
        const auto [rendered, peak] = strategy.explore_scratch(100);
        EXPECT_EQ(rendered.is_valid(), gate);
        EXPECT_EQ(peak.is_valid(), gate);
        if (gate)
            EXPECT_EQ(peak.ptr<float>(), strategy._explore_peak.ptr<float>());
        EXPECT_FALSE(strategy._retire_step_peak.is_valid());
        EXPECT_FALSE(strategy._retire_renders.is_valid());
        EXPECT_EQ(MRNF::retire_count(params, 100, 100, n), 0u);
        strategy.visibility_accumulator().fill_(1.0f);
        RenderOutput output;
        strategy.post_backward(100, output);
        EXPECT_EQ(strategy._last_retired, 0u);
        EXPECT_EQ(strategy._retire_first_iter, -1);
        EXPECT_EQ(strategy.active_count(), n);
    }
}

// The quota is share x live from the first refine at the cap, falling linearly to 0 at grow_until_iter;
// nothing before the cap. The strategy records the first refine at the cap once.
TEST(MRNFStrategyTest, RetireQuotaStartsAtTheCapAndTapersToGrowthEnd) {
    auto params = retire_test_params(1000, 0.0625f);
    params.grow_until_iter = 10'000;
    EXPECT_EQ(MRNF::retire_count(params, 3'000, -1, 100'000), 0u);
    EXPECT_EQ(MRNF::retire_count(params, 1'000, 2'000, 100'000), 0u);
    EXPECT_EQ(MRNF::retire_count(params, 2'000, 2'000, 100'000), 6250u);
    EXPECT_EQ(MRNF::retire_count(params, 6'000, 2'000, 100'000), 3125u);
    EXPECT_EQ(MRNF::retire_count(params, 9'999, 2'000, 100'000), 0u); // (size_t)(0.0625 x 1/8000 x 1e5)
    EXPECT_EQ(MRNF::retire_count(params, 10'000, 2'000, 100'000), 0u);
    // The match strategy's extra deaths share the taper, ending at stop_refine.
    params.match_extra_death_share = 0.0625f;
    params.stop_refine = 10'000;
    EXPECT_EQ(Match::extra_deaths(params, 6'000, 2'000, 100'000), 3125u);

    constexpr size_t n = 100;
    auto splat_data = create_mrnf_test_splat_data(static_cast<int>(n), 0);
    MRNF strategy(splat_data);
    auto below = retire_test_params(2 * n, 0.05f);
    strategy.initialize(below);
    ASSERT_TRUE(strategy.retire_on());
    auto prune = Tensor::zeros_bool({n}, Device::CUDA);
    EXPECT_EQ(strategy.mark_retirees(200, prune), 0u); // half the cap
    EXPECT_EQ(strategy._retire_first_iter, -1);
    strategy.set_optimization_params(retire_test_params(n, 0.05f));
    strategy.explore_scratch(300);
    strategy._retire_renders.fill_(100.0f);
    EXPECT_EQ(strategy.mark_retirees(300, prune), 5u);
    EXPECT_EQ(strategy._retire_first_iter, 300);
    prune = Tensor::zeros_bool({n}, Device::CUDA);
    EXPECT_EQ(strategy.mark_retirees(2'650, prune), 2u); // (size_t)(0.05 x 0.5 x 100)
    EXPECT_EQ(strategy._retire_first_iter, 300);
}

// The retirees are the live, trainable splats with the lowest peak blend weight among those rendered at
// least mrnf_retire_grace times; pruned and free rows are not counted or taken. The same refine refills
// their slots and their rows start afresh.
TEST(MRNFStrategyTest, RetireTakesTheLowestPeakPastTheGraceAndRefillsTheSlots) {
    constexpr size_t n = 200;
    auto splat_data = create_mrnf_test_splat_data(static_cast<int>(n), 0);
    MRNF strategy(splat_data);
    auto params = retire_test_params(n, 0.05f);
    params.mrnf_retire_grace = 30;
    strategy.initialize(params);
    strategy.explore_scratch(100);
    std::vector<float> peak(n, 0.5f), renders(n, 100.0f);
    for (size_t i = 50; i < 80; ++i)
        peak[i] = 0.001f * static_cast<float>(i - 49); // rising: 50 is the faintest
    for (size_t i = 50; i < 55; ++i)
        renders[i] = 29.0f; // within the grace
    peak[198] = peak[199] = 0.0f;
    strategy._explore_peak.copy_(Tensor::from_vector(peak, TensorShape({n}), Device::CUDA));
    strategy._retire_renders.copy_(Tensor::from_vector(renders, TensorShape({n}), Device::CUDA));
    strategy.soft_delete_rows(row_range(198, 200)); // free: 198 of 200 active, still at the cap

    std::vector<bool> pruned_rows(n, false);
    pruned_rows[55] = true; // pruned already: neither counted as live nor taken again
    auto prune = Tensor::from_vector(pruned_rows, TensorShape({n}), Device::CUDA);
    const size_t marked = strategy.mark_retirees(100, prune);
    ASSERT_EQ(marked, 9u); // (size_t)(0.05 x 197)
    const auto mask = prune.to(DataType::Float32).cpu().to_vector();
    for (size_t i = 0; i < n; ++i) {
        const bool expected = i >= 55 && i <= 64; // 55 (pruned before) and the nine faintest past the grace
        EXPECT_EQ(mask[i] != 0.0f, expected) << i;
    }

    // A whole refine: the retirees are freed and refilled at once, so the model stays at the cap.
    auto data = create_mrnf_test_splat_data(static_cast<int>(n), 0);
    MRNF refining(data);
    refining.initialize(params);
    refining.explore_scratch(100);
    refining._explore_peak.copy_(Tensor::from_vector(std::vector<float>(n, 0.5f), TensorShape({n}), Device::CUDA));
    refining._explore_peak.slice(0, 0, 10).fill_(0.01f);
    refining._retire_renders.fill_(100.0f);
    refining.visibility_accumulator().fill_(1.0f);
    RenderOutput output;
    refining.post_backward(100, output);
    EXPECT_EQ(refining._last_retired, 10u);
    EXPECT_EQ(refining._retire_first_iter, 100);
    EXPECT_EQ(refining.active_count(), n);
    EXPECT_EQ(data.size(), n);
    const auto after_renders = refining._retire_renders.cpu().to_vector();
    size_t fresh = 0;
    for (size_t i = 0; i < n; ++i)
        fresh += after_renders[i] == 0.0f ? 1 : 0;
    EXPECT_EQ(fresh, 10u); // the refilled slots
    for (size_t i = 0; i < 10; ++i)
        EXPECT_EQ(after_renders[i], 0.0f) << i;
}

// Retirement keeps its own per-step peak for the backward and folds it into the running peak (half-life
// 800 steps) and the render count with the match kernels; the walk gate does not read it unless the gate
// is on. The rows follow their splats and are dropped when the topology freezes.
TEST(MRNFStrategyTest, RetireRowsTrackPeakAndRendersAndFollowTheirSplats) {
    constexpr size_t n = 4;
    auto splat_data = create_mrnf_test_splat_data(static_cast<int>(n), 0);
    MRNF strategy(splat_data);
    auto params = retire_test_params(32, 0.01f);
    strategy.initialize(params);
    ASSERT_TRUE(strategy.retire_on());
    auto [rendered, step_peak] = strategy.explore_scratch(100);
    ASSERT_TRUE(rendered.is_valid());
    ASSERT_TRUE(step_peak.is_valid());
    EXPECT_EQ(step_peak.ptr<float>(), strategy._retire_step_peak.ptr<float>());
    ASSERT_TRUE(strategy._explore_peak.is_valid());

    const float decay = std::pow(0.5f, 1.0f / 800.0f);
    step_peak.copy_(Tensor::from_vector(std::vector<float>{0.3f, 0.0f, 0.8f, 0.1f}, TensorShape({n}), Device::CUDA));
    const auto means = splat_data.means().cpu().to_vector();
    strategy.explore_step(100);
    EXPECT_EQ(splat_data.means().cpu().to_vector(), means); // no walk
    EXPECT_EQ(strategy._explore_peak.cpu().to_vector(), (std::vector<float>{0.3f, 0.0f, 0.8f, 0.1f}));
    EXPECT_EQ(strategy._retire_renders.cpu().to_vector(), (std::vector<float>{1.0f, 0.0f, 1.0f, 1.0f}));
    EXPECT_EQ(strategy._retire_step_peak.cpu().to_vector(), std::vector<float>(n, 0.0f));
    strategy.explore_scratch(101).second.copy_(
        Tensor::from_vector(std::vector<float>{0.0f, 0.2f, 0.9f, 0.0f}, TensorShape({n}), Device::CUDA));
    strategy.explore_step(101);
    const auto folded = strategy._explore_peak.cpu().to_vector();
    EXPECT_FLOAT_EQ(folded[0], 0.3f * decay);
    EXPECT_FLOAT_EQ(folded[1], 0.2f);
    EXPECT_FLOAT_EQ(folded[2], 0.9f);
    EXPECT_FLOAT_EQ(folded[3], 0.1f * decay);
    EXPECT_EQ(strategy._retire_renders.cpu().to_vector(), (std::vector<float>{1.0f, 1.0f, 2.0f, 1.0f}));

    strategy.soft_delete_rows(row_range(2, 3));
    EXPECT_EQ(strategy._retire_renders.cpu().to_vector()[2], 0.0f);
    EXPECT_EQ(strategy._explore_peak.cpu().to_vector()[2], 0.0f);

    const auto mask = Tensor::from_vector(std::vector<bool>{true, false, false, false}, TensorShape({n}), Device::CUDA);
    strategy.remove_gaussians(mask);
    ASSERT_EQ(splat_data.size(), 3u);
    EXPECT_EQ(strategy._retire_renders.cpu().to_vector(), (std::vector<float>{1.0f, 0.0f, 1.0f}));

    // The topology freeze drops the retirement rows.
    RenderOutput output;
    strategy.post_backward(static_cast<int>(params.stop_refine), output);
    EXPECT_FALSE(strategy.retire_on());
    strategy.explore_scratch(static_cast<int>(params.stop_refine) + 1);
    EXPECT_FALSE(strategy._retire_step_peak.is_valid());
    EXPECT_FALSE(strategy._retire_renders.is_valid());
}

namespace {
    constexpr float kInf = std::numeric_limits<float>::infinity();

    std::vector<float> spare_test_peaks(const size_t n) {
        std::vector<float> peak(n, 0.5f);
        for (size_t i = 50; i < 80; ++i)
            peak[i] = 0.001f * static_cast<float>(i - 49); // rising: 50 is the faintest
        return peak;
    }

    // A strategy at the cap with retirement (share 0.05, grace 30) and the given spare threshold; the tests set
    // every splat past the grace and the peaks of spare_test_peaks (SPARE_TEST_PRIME).
    std::unique_ptr<MRNF> spare_test_strategy(SplatData& data, const size_t n, const float spare_px) {
        auto strategy = std::make_unique<MRNF>(data);
        auto params = retire_test_params(n, 0.05f);
        params.mrnf_retire_spare_px = spare_px;
        strategy->initialize(params);
        strategy->explore_scratch(100);
        return strategy;
    }

    std::vector<size_t> marked_rows(const Tensor& prune) {
        const auto mask = prune.to(DataType::Float32).cpu().to_vector();
        std::vector<size_t> rows;
        for (size_t i = 0; i < mask.size(); ++i)
            if (mask[i] != 0.0f)
                rows.push_back(i);
        return rows;
    }
} // namespace

// The tests are friends of MRNF, the helpers above are not.
#define SPARE_TEST_PRIME(s)                                                                                 \
    do {                                                                                                    \
        (s)->_explore_peak.copy_(Tensor::from_vector(spare_test_peaks(n), TensorShape({n}), Device::CUDA)); \
        (s)->_retire_renders.fill_(100.0f);                                                                 \
    } while (false)

// Off (the default 0) keeps no minor sigma rows, gives the backward none and retires exactly the rows it
// retired before; on with no thin splat retires the same rows.
TEST(MRNFStrategyTest, RetireSpareOffKeepsTheRetireesBitIdentical) {
    EXPECT_EQ(param::OptimizationParameters::mrnf_defaults().mrnf_retire_spare_px, 0.0f);
    constexpr size_t n = 200;
    std::vector<size_t> expected;
    for (size_t i = 50; i < 60; ++i)
        expected.push_back(i); // (size_t)(0.05 x 200) faintest
    {
        auto data = create_mrnf_test_splat_data(static_cast<int>(n), 0);
        auto strategy = spare_test_strategy(data, n, 0.0f);
        SPARE_TEST_PRIME(strategy);
        EXPECT_TRUE(strategy->retire_on());
        EXPECT_FALSE(strategy->spare_on());
        EXPECT_FALSE(strategy->minor_sigma_scratch(100).is_valid());
        EXPECT_FALSE(strategy->_spare_window_sigma.is_valid());
        EXPECT_FALSE(strategy->_spare_sigma.is_valid());
        auto prune = Tensor::zeros_bool({n}, Device::CUDA);
        EXPECT_EQ(strategy->mark_retirees(100, prune), 10u);
        EXPECT_EQ(marked_rows(prune), expected);
        EXPECT_EQ(strategy->_last_spared, 0u);
    }
    {
        auto data = create_mrnf_test_splat_data(static_cast<int>(n), 0);
        auto strategy = spare_test_strategy(data, n, 1.25f);
        SPARE_TEST_PRIME(strategy);
        ASSERT_TRUE(strategy->spare_on());
        strategy->minor_sigma_scratch(100).fill_(1.3f); // every splat drawn, none thin
        auto prune = Tensor::zeros_bool({n}, Device::CUDA);
        EXPECT_EQ(strategy->mark_retirees(100, prune), 10u);
        EXPECT_EQ(marked_rows(prune), expected);
        EXPECT_EQ(strategy->_last_spared, 0u);
    }
    // Retirement off: no rows even with a threshold.
    auto data = create_mrnf_test_splat_data(static_cast<int>(n), 0);
    MRNF off(data);
    auto params = retire_test_params(n, 0.0f);
    params.mrnf_retire_spare_px = 1.25f;
    off.initialize(params);
    EXPECT_FALSE(off.spare_on());
    EXPECT_FALSE(off.minor_sigma_scratch(100).is_valid());
}

// Splats thinner than the threshold in the last window that drew them are never retired, and the quota
// (share x live, counted on all live splats) is taken from the other splats; with too few others it
// retires what there is. The census count is the live thin splats.
TEST(MRNFStrategyTest, RetireSpareExemptsThinSplatsAndKeepsTheQuota) {
    constexpr size_t n = 200;
    auto data = create_mrnf_test_splat_data(static_cast<int>(n), 0);
    auto strategy = spare_test_strategy(data, n, 1.25f);
    SPARE_TEST_PRIME(strategy);
    std::vector<float> sigma(n, 3.0f);
    for (size_t i = 50; i < 55; ++i)
        sigma[i] = 1.0f; // the five faintest are thin
    sigma[57] = 1.2499f; // thin, just below the threshold
    sigma[58] = 1.25f;   // not below: not exempt
    sigma[150] = 0.2f;   // thin but bright: counted, not a candidate anyway
    sigma[199] = kInf;   // not drawn this window: not thin
    strategy->minor_sigma_scratch(100).copy_(Tensor::from_vector(sigma, TensorShape({n}), Device::CUDA));

    auto prune = Tensor::zeros_bool({n}, Device::CUDA);
    ASSERT_EQ(strategy->mark_retirees(100, prune), 10u); // the quota is unchanged
    EXPECT_EQ(marked_rows(prune), (std::vector<size_t>{55, 56, 58, 59, 60, 61, 62, 63, 64, 65}));
    EXPECT_EQ(strategy->_last_spared, 7u);

    // Too few splats left that are not thin: the refine retires only those.
    auto few_data = create_mrnf_test_splat_data(static_cast<int>(n), 0);
    auto few = spare_test_strategy(few_data, n, 1.25f);
    SPARE_TEST_PRIME(few);
    std::vector<float> mostly_thin(n, 0.6f);
    mostly_thin[3] = mostly_thin[90] = mostly_thin[120] = 2.0f;
    few->minor_sigma_scratch(100).copy_(Tensor::from_vector(mostly_thin, TensorShape({n}), Device::CUDA));
    prune = Tensor::zeros_bool({n}, Device::CUDA);
    EXPECT_EQ(few->mark_retirees(100, prune), 3u);
    EXPECT_EQ(marked_rows(prune), (std::vector<size_t>{3, 90, 120}));
    EXPECT_EQ(few->_last_spared, n - 3);

    // A whole refine: thin faint splats survive, the model stays at the cap, the census counts them.
    auto refine_data = create_mrnf_test_splat_data(static_cast<int>(n), 0);
    auto refining = spare_test_strategy(refine_data, n, 1.25f);
    SPARE_TEST_PRIME(refining);
    refining->_explore_peak.slice(0, 0, 20).fill_(0.0001f);
    std::vector<float> thin_first(n, 3.0f);
    for (size_t i = 0; i < 10; ++i)
        thin_first[i] = 0.9f;
    refining->minor_sigma_scratch(100).copy_(Tensor::from_vector(thin_first, TensorShape({n}), Device::CUDA));
    refining->visibility_accumulator().fill_(1.0f);
    RenderOutput output;
    refining->post_backward(100, output);
    EXPECT_EQ(refining->_last_retired, 10u);
    EXPECT_EQ(refining->_last_spared, 10u);
    EXPECT_EQ(refining->active_count(), n);
    const auto renders = refining->_retire_renders.cpu().to_vector();
    for (size_t i = 0; i < 10; ++i)
        EXPECT_EQ(renders[i], 100.0f) << i; // spared: not freed and refilled
    for (size_t i = 10; i < 20; ++i)
        EXPECT_EQ(renders[i], 0.0f) << i; // retired and refilled
}

// The window row starts at +inf and is what the backward lowers; each refine closes it (drawn rows take its
// minimum, undrawn rows keep the last one) and restarts it. Freed slots and new rows start at +inf; the
// rows follow their splats and are dropped at the topology freeze.
TEST(MRNFStrategyTest, RetireSpareWindowFollowsRendersAndSplats) {
    constexpr size_t n = 4;
    auto splat_data = create_mrnf_test_splat_data(static_cast<int>(n), 0);
    MRNF strategy(splat_data);
    auto params = retire_test_params(32, 0.01f);
    params.mrnf_retire_spare_px = 1.0f;
    strategy.initialize(params);
    auto window = strategy.minor_sigma_scratch(100);
    ASSERT_TRUE(window.is_valid());
    EXPECT_EQ(window.ptr<float>(), strategy._spare_window_sigma.ptr<float>());
    EXPECT_EQ(window.cpu().to_vector(), std::vector<float>(n, kInf));
    EXPECT_EQ(strategy._spare_sigma.cpu().to_vector(), std::vector<float>(n, kInf));

    window.copy_(Tensor::from_vector(std::vector<float>{0.5f, kInf, 2.0f, 0.7f}, TensorShape({n}), Device::CUDA));
    strategy.close_spare_window();
    EXPECT_EQ(strategy._spare_sigma.cpu().to_vector(), (std::vector<float>{0.5f, kInf, 2.0f, 0.7f}));
    EXPECT_EQ(strategy._spare_window_sigma.cpu().to_vector(), std::vector<float>(n, kInf));
    strategy.minor_sigma_scratch(200).copy_(
        Tensor::from_vector(std::vector<float>{kInf, 0.9f, 3.0f, kInf}, TensorShape({n}), Device::CUDA));
    strategy.close_spare_window();
    EXPECT_EQ(strategy._spare_sigma.cpu().to_vector(), (std::vector<float>{0.5f, 0.9f, 3.0f, 0.7f}));

    strategy.soft_delete_rows(row_range(2, 3));
    EXPECT_EQ(strategy._spare_sigma.cpu().to_vector()[2], kInf);
    const auto mask = Tensor::from_vector(std::vector<bool>{true, false, false, false}, TensorShape({n}), Device::CUDA);
    strategy.remove_gaussians(mask);
    ASSERT_EQ(splat_data.size(), 3u);
    strategy.ensure_explore_rows();
    EXPECT_EQ(strategy._spare_sigma.cpu().to_vector(), (std::vector<float>{0.9f, kInf, 0.7f}));
    EXPECT_EQ(strategy._spare_window_sigma.numel(), 3u);

    // New rows (growth) start undrawn.
    strategy._spare_sigma = strategy._spare_sigma.slice(0, 0, 2).contiguous();
    strategy.ensure_explore_rows();
    EXPECT_EQ(strategy._spare_sigma.cpu().to_vector(), (std::vector<float>{0.9f, kInf, kInf}));

    RenderOutput output;
    strategy.post_backward(static_cast<int>(params.stop_refine), output);
    EXPECT_FALSE(strategy.spare_on());
    EXPECT_FALSE(strategy.minor_sigma_scratch(static_cast<int>(params.stop_refine) + 1).is_valid());
    strategy.explore_scratch(static_cast<int>(params.stop_refine) + 1);
    EXPECT_FALSE(strategy._spare_window_sigma.is_valid());
    EXPECT_FALSE(strategy._spare_sigma.is_valid());
}

// The real backward lowers the row to each contributing splat's projected minor-axis sigma before the 2D
// dilation: focal x scale / depth for an axis-aligned splat on the optical axis.
TEST(MRNFStrategyTest, RetireSpareBackwardWritesTheRawMinorSigma) {
    constexpr int width = 64, height = 48;
    constexpr float focal = 60.0f;
    constexpr float sh_c0 = 0.28209479177387814f;
    auto R = Tensor::eye(3, Device::CUDA);
    auto T = Tensor::from_vector(std::vector<float>{0.0f, 0.0f, 4.0f}, TensorShape({3}), Device::CUDA);
    Camera camera(R, T, focal, focal, width / 2.0f, height / 2.0f, Tensor(), Tensor(), CameraModelType::PINHOLE,
                  "spare", "", std::filesystem::path{}, width, height, 0);
    auto bg = Tensor::full({3}, 0.5f, Device::CUDA);
    const std::vector<float> means{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 3.0f, 0.0f, 0.0f};
    const std::vector<float> rotations{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    // 0: thin in y (minor 0.75 px), 1: round behind it (2.69 px), 2: outside the view.
    const std::vector<float> log_scales{-1.6f, -3.0f, -1.6f, -1.6f, -1.6f, -1.6f, -1.6f, -1.6f, -1.6f};
    const std::vector<float> opacities{raw_of(0.95f), raw_of(0.8f), raw_of(0.8f)};
    const std::vector<float> sh0{0.4f / sh_c0, -0.4f / sh_c0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    SplatData splat_data(0, Tensor::from_vector(means, TensorShape({3, 3}), Device::CUDA),
                         Tensor::from_vector(sh0, TensorShape({3, 1, 3}), Device::CUDA),
                         Tensor::zeros({3, 0, 3}, Device::CUDA),
                         Tensor::from_vector(log_scales, TensorShape({3, 3}), Device::CUDA),
                         Tensor::from_vector(rotations, TensorShape({3, 4}), Device::CUDA),
                         Tensor::from_vector(opacities, TensorShape({3, 1}), Device::CUDA), 1.0f);

    MRNF strategy(splat_data);
    auto params = retire_test_params(3, 0.01f);
    params.mrnf_retire_spare_px = 1.25f;
    strategy.initialize(params);
    auto window = strategy.minor_sigma_scratch(100);
    ASSERT_TRUE(window.is_valid());

    const auto backward = [&] {
        auto forward = fast_rasterize_forward(camera, splat_data, bg, 0, 0, 0, 0, false);
        ASSERT_TRUE(forward.has_value()) << lfs::format_for_developer(forward.error());
        strategy.get_optimizer().zero_grad(0);
        FastGSFusedExtraGradients fused;
        fused.explore_minor_sigma = window.ptr<float>();
        fast_rasterize_backward(forward->second, Tensor::full_like(forward->first.image, 0.01f), splat_data,
                                strategy.get_optimizer(), {}, {}, DensificationType::None, 1, fused);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    };
    const auto start_scales = splat_data.scaling_raw().cpu().to_vector();
    backward();
    const auto sigma = window.cpu().to_vector();
    EXPECT_NEAR(sigma[0], focal * std::exp(start_scales[1]) / 4.0f, 1e-3f);
    EXPECT_NEAR(sigma[1], focal * std::exp(start_scales[3]) / 4.5f, 1e-3f);
    EXPECT_EQ(sigma[2], kInf); // not drawn

    // It keeps the minimum over the views of the window.
    window.copy_(Tensor::from_vector(std::vector<float>{0.3f, 9.0f, kInf}, TensorShape({3}), Device::CUDA));
    const auto scales_before = splat_data.scaling_raw().cpu().to_vector();
    backward();
    const auto lowered = window.cpu().to_vector();
    EXPECT_EQ(lowered[0], 0.3f);
    EXPECT_NEAR(lowered[1], focal * std::exp(scales_before[3]) / 4.5f, 1e-3f);
}

namespace {
    // Retirement at the cap (share 0.05, grace 30) with the given rank; rows primed past the grace.
    std::unique_ptr<MRNF> rank_test_strategy(SplatData& data, const size_t n, const param::MrnfRetireRank rank) {
        auto strategy = std::make_unique<MRNF>(data);
        auto params = retire_test_params(n, 0.05f);
        params.mrnf_retire_rank = rank;
        strategy->initialize(params);
        strategy->explore_scratch(100);
        return strategy;
    }

    // Peak blend weight, summed and pixel-peak removal impact per row: rows 100..129 are low in all three (rising),
    // row 0 is the faintest but has the largest impact (a twig against the sky), row 1 the most opaque with no
    // impact (a twig in front of a similar background); the rest are high in all three.
    struct RankRows {
        std::vector<float> peak, sum, impact_peak;
    };
    RankRows rank_test_rows(const size_t n) {
        RankRows rows{std::vector<float>(n, 0.5f), std::vector<float>(n, 9.0f), std::vector<float>(n, 0.5f)};
        for (size_t i = 100; i < 130; ++i) {
            const float k = static_cast<float>(i - 100);
            rows.peak[i] = 0.01f + 0.001f * k;
            rows.sum[i] = 0.1f + 0.01f * k;
            rows.impact_peak[i] = 0.01f + 0.001f * k;
        }
        rows.peak[0] = 0.0001f;
        rows.sum[0] = 50.0f;
        rows.impact_peak[0] = 0.9f;
        rows.peak[1] = 0.99f;
        rows.sum[1] = 0.0001f;
        rows.impact_peak[1] = 0.0001f;
        return rows;
    }
} // namespace

#define RANK_TEST_PRIME(s, rows)                                                                    \
    do {                                                                                            \
        (s)->_explore_peak.copy_(Tensor::from_vector((rows).peak, TensorShape({n}), Device::CUDA)); \
        (s)->_retire_renders.fill_(100.0f);                                                         \
    } while (false)

// Off: the default rank is the peak blend weight, nothing else changes: no impact is asked of the backward, no
// impact rows are kept and the retirees are the lowest peaks as before. The contrast normalisation and the
// keep-long split default off as well.
TEST(MRNFStrategyTest, RetireRankBothOffKeepsTheRetireesBitIdentical) {
    const auto defaults = param::OptimizationParameters::mrnf_defaults();
    EXPECT_EQ(defaults.mrnf_retire_rank, param::MrnfRetireRank::Peak);
    EXPECT_FALSE(defaults.impact_contrast_norm);
    EXPECT_FLOAT_EQ(defaults.impact_contrast_eps, 0.01f);
    EXPECT_EQ(defaults.receiver_keep_long, 0.0f);
    EXPECT_EQ(param::mrnf_retire_rank_name(param::MrnfRetireRank::Both), "both");
    EXPECT_EQ(param::mrnf_retire_rank_from_string("peak"), param::MrnfRetireRank::Peak);
    EXPECT_FALSE(param::mrnf_retire_rank_from_string("impact").has_value());

    constexpr size_t n = 200;
    const auto rows = rank_test_rows(n);
    auto data = create_mrnf_test_splat_data(static_cast<int>(n), 0);
    auto strategy = rank_test_strategy(data, n, param::MrnfRetireRank::Peak);
    RANK_TEST_PRIME(strategy, rows);
    EXPECT_FALSE(strategy->retire_rank_both_on());
    EXPECT_FALSE(strategy->impact_scratch(100).first.is_valid());
    EXPECT_FALSE(strategy->_retire_impact_sum.is_valid());
    EXPECT_FALSE(strategy->_retire_impact_peak.is_valid());
    auto prune = Tensor::zeros_bool({n}, Device::CUDA);
    ASSERT_EQ(strategy->mark_retirees(100, prune), 10u);
    // The lowest peaks: row 0 and the nine faintest of 100..129.
    std::vector<size_t> expected{0};
    for (size_t i = 100; i < 109; ++i)
        expected.push_back(i);
    EXPECT_EQ(marked_rows(prune), expected);

    // Need-guided donors ranked by blend weight still ask for no impact; the rank needs retirement on.
    auto need = need_growth_test_params(32);
    need.mrnf_retire_rank = param::MrnfRetireRank::Both;
    auto need_data = create_mrnf_test_splat_data(8, 0);
    MRNF no_retire(need_data);
    no_retire.initialize(need);
    EXPECT_FALSE(no_retire.retire_rank_both_on());
    EXPECT_FALSE(no_retire.impact_scratch(100).first.is_valid());
    no_retire.ensure_explore_rows();
    EXPECT_FALSE(no_retire._retire_impact_sum.is_valid());
}

// Rank Both retires the rows low in peak blend weight AND in removal impact: the faint twig with a large impact
// and the opaque twig with none both stay, the quota is unchanged and the grace still holds. A whole refine
// frees and refills the same count.
TEST(MRNFStrategyTest, RetireRankBothKeepsHighImpactOrHighPeakSplats) {
    constexpr size_t n = 200;
    const auto rows = rank_test_rows(n);
    auto data = create_mrnf_test_splat_data(static_cast<int>(n), 0);
    auto strategy = rank_test_strategy(data, n, param::MrnfRetireRank::Both);
    RANK_TEST_PRIME(strategy, rows);
    ASSERT_TRUE(strategy->retire_rank_both_on());
    // Before any impact: the peak ranking, with a warning.
    {
        auto prune = Tensor::zeros_bool({n}, Device::CUDA);
        ASSERT_EQ(strategy->mark_retirees(100, prune), 10u);
        EXPECT_EQ(marked_rows(prune).front(), 0u);
        EXPECT_EQ(strategy->_retire_impact_windows, 0);
    }
    write_impact_window(*strategy, rows.sum, rows.impact_peak);
    std::vector<float> renders(n, 100.0f);
    renders[102] = 29.0f; // within the grace
    strategy->_retire_renders.copy_(Tensor::from_vector(renders, TensorShape({n}), Device::CUDA));
    auto prune = Tensor::zeros_bool({n}, Device::CUDA);
    ASSERT_EQ(strategy->mark_retirees(100, prune), 10u);
    EXPECT_EQ(strategy->_retire_impact_windows, 1);
    EXPECT_EQ(strategy->_retire_impact_sum.cpu().to_vector(), rows.sum);
    EXPECT_EQ(strategy->_retire_impact_peak.cpu().to_vector(), rows.impact_peak);
    EXPECT_EQ(marked_rows(prune), (std::vector<size_t>{100, 101, 103, 104, 105, 106, 107, 108, 109, 110}));

    // The rank is the latest of the three positions: high in any one key keeps a row.
    {
        auto keys = rows;
        keys.sum[105] = 99.0f;         // only the summed impact is high
        keys.impact_peak[106] = 0.99f; // only the pixel peak is high
        keys.peak[107] = 0.9f;         // only the peak blend weight is high
        auto d = create_mrnf_test_splat_data(static_cast<int>(n), 0);
        auto s = rank_test_strategy(d, n, param::MrnfRetireRank::Both);
        RANK_TEST_PRIME(s, keys);
        write_impact_window(*s, keys.sum, keys.impact_peak);
        auto p = Tensor::zeros_bool({n}, Device::CUDA);
        ASSERT_EQ(s->mark_retirees(100, p), 10u);
        EXPECT_EQ(marked_rows(p), (std::vector<size_t>{100, 101, 102, 103, 104, 108, 109, 110, 111, 112}));
    }

    // A whole refine: ten retired, freed and refilled, the model stays at the cap.
    auto refine_data = create_mrnf_test_splat_data(static_cast<int>(n), 0);
    auto refining = rank_test_strategy(refine_data, n, param::MrnfRetireRank::Both);
    RANK_TEST_PRIME(refining, rows);
    write_impact_window(*refining, rows.sum, rows.impact_peak);
    refining->visibility_accumulator().fill_(1.0f);
    RenderOutput output;
    refining->post_backward(100, output);
    EXPECT_EQ(refining->_last_retired, 10u);
    EXPECT_EQ(refining->active_count(), n);
    const auto after = refining->_retire_renders.cpu().to_vector();
    EXPECT_EQ(after[0], 100.0f); // the faint high-impact splat stayed
    EXPECT_EQ(after[1], 100.0f); // the opaque no-impact splat stayed
    for (size_t i = 100; i < 110; ++i)
        EXPECT_EQ(after[i], 0.0f) << i; // retired and refilled
    // The window outputs were consumed by the refine.
    EXPECT_FALSE(refining->_exchange_window_impact_sum.is_valid());
    EXPECT_FALSE(refining->_exchange_impact_written);
}

// The retire impact memory folds each window as a faded maximum (half-life one pass, the donor impact memory's),
// also before the cap; it follows its splats and is dropped at the topology freeze. Need-guided donors ranked
// by blend weight read the same window buffers but keep their blend-weight memory exactly.
TEST(MRNFStrategyTest, RetireRankBothFoldsTheWindowImpactApartFromTheDonors) {
    const std::vector<float> sum1{4.0f, 1.0f, 2.0f, 8.0f, 0.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f};
    const std::vector<float> peak1{0.4f, 0.9f, 0.1f, 0.2f, 0.0f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f};
    const std::vector<float> sum2{1.0f, 2.0f, 2.0f, 0.0f, 5.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f};
    const std::vector<float> peak2{0.1f, 0.2f, 0.5f, 0.0f, 0.6f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f};
    const std::vector<float> blend{5.0f, 1.0f, 2.0f, 7.0f, 3.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f};
    constexpr size_t n = 10;

    const auto run = [&](const bool both) {
        auto data = create_mrnf_test_splat_data(static_cast<int>(n), 0);
        auto strategy = std::make_unique<MRNF>(data);
        auto params = need_growth_test_params(32); // below the cap: no quota, the memory still folds
        params.refine_every = 1;
        params.mrnf_retire_share = 0.05f;
        params.mrnf_retire_rank = both ? param::MrnfRetireRank::Both : param::MrnfRetireRank::Peak;
        strategy->initialize(params);
        strategy->set_training_dataset(make_hull_dataset());
        EXPECT_FALSE(strategy->exchange_ranks_by_impact());
        EXPECT_EQ(strategy->impact_scratch(100).first.is_valid(), both);
        for (const bool second : {false, true}) {
            if (both)
                write_impact_window(*strategy, second ? sum2 : sum1, second ? peak2 : peak1);
            auto prune = Tensor::zeros_bool({n}, Device::CUDA);
            EXPECT_EQ(strategy->mark_retirees(100, prune), 0u);
            strategy->visibility_accumulator().copy_(Tensor::from_vector(blend, TensorShape({n}), Device::CUDA));
            strategy->update_exchange_importance();
            EXPECT_FALSE(strategy->_exchange_window_impact_sum.is_valid());
            EXPECT_FALSE(strategy->_exchange_impact_written);
        }
        return strategy;
    };
    const auto both = run(true);
    const auto peak_only = run(false);
    // The donors' memory is the blend weight in both runs.
    EXPECT_EQ(both->_exchange_importance_sum.cpu().to_vector(), peak_only->_exchange_importance_sum.cpu().to_vector());
    EXPECT_FALSE(both->_exchange_importance_peak.is_valid());
    EXPECT_EQ(both->_retire_impact_windows, 2);
    const float decay = std::pow(0.5f, 1.0f / 2.0f); // refine every step, two views
    const auto sum = both->_retire_impact_sum.cpu().to_vector();
    const auto peak = both->_retire_impact_peak.cpu().to_vector();
    for (size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(sum[i], std::fmax(sum2[i], decay * sum1[i])) << i;
        EXPECT_FLOAT_EQ(peak[i], std::fmax(peak2[i], decay * peak1[i])) << i;
    }

    // Rows follow their splats: a freed slot restarts at 0, removal compacts, the freeze drops them.
    both->soft_delete_rows(row_range(2, 3));
    EXPECT_EQ(both->_retire_impact_sum.cpu().to_vector()[2], 0.0f);
    EXPECT_EQ(both->_retire_impact_peak.cpu().to_vector()[2], 0.0f);
    std::vector<bool> drop(n, false);
    drop[0] = true;
    both->remove_gaussians(Tensor::from_vector(drop, TensorShape({n}), Device::CUDA));
    ASSERT_EQ(both->_retire_impact_sum.numel(), n - 1);
    EXPECT_FLOAT_EQ(both->_retire_impact_sum.cpu().to_vector()[0], sum[1]);
    RenderOutput output;
    both->post_backward(static_cast<int>(both->_params->stop_refine), output);
    EXPECT_FALSE(both->retire_rank_both_on());
    both->explore_scratch(static_cast<int>(both->_params->stop_refine) + 1);
    EXPECT_FALSE(both->_retire_impact_sum.is_valid());
    EXPECT_FALSE(both->impact_scratch(static_cast<int>(both->_params->stop_refine) + 1).first.is_valid());
}

// The real backward divides each pixel's removal impact by (photo deviation + eps): a zero map with eps 1 leaves
// the impact bit for bit, a constant deviation scales it by 1 / (sigma + eps), and a map that is flat where the
// front splat lies keeps its impact while a textured one shrinks it.
TEST(MRNFStrategyTest, ImpactContrastNormDividesByThePhotoDeviation) {
    constexpr int width = 64, height = 48;
    constexpr float sh_c0 = 0.28209479177387814f;
    const auto run = [&](const std::vector<float>& sigma, const float eps) {
        auto R = Tensor::eye(3, Device::CUDA);
        auto T = Tensor::from_vector(std::vector<float>{0.0f, 0.0f, 4.0f}, TensorShape({3}), Device::CUDA);
        Camera camera(R, T, 60.0f, 60.0f, width / 2.0f, height / 2.0f, Tensor(), Tensor(), CameraModelType::PINHOLE,
                      "impact", "", std::filesystem::path{}, width, height, 0);
        auto bg = Tensor::full({3}, 0.5f, Device::CUDA);
        const std::vector<float> means{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f};
        const std::vector<float> rotations{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f};
        const std::vector<float> opacities{std::log(0.95f / 0.05f), std::log(0.8f / 0.2f)};
        const std::vector<float> sh0{0.4f / sh_c0, -0.4f / sh_c0, 0.0f, 0.0f, 0.0f, 0.0f};
        SplatData splat_data(0, Tensor::from_vector(means, TensorShape({2, 3}), Device::CUDA),
                             Tensor::from_vector(sh0, TensorShape({2, 1, 3}), Device::CUDA),
                             Tensor::zeros({2, 0, 3}, Device::CUDA), Tensor::full({2, 3}, -1.6f, Device::CUDA),
                             Tensor::from_vector(rotations, TensorShape({2, 4}), Device::CUDA),
                             Tensor::from_vector(opacities, TensorShape({2, 1}), Device::CUDA), 1.0f);
        MRNF strategy(splat_data);
        strategy.initialize(donor_impact_test_params(true));
        auto [impact_sum, impact_peak] = strategy.impact_scratch(100);
        EXPECT_TRUE(impact_sum.is_valid());
        auto forward = fast_rasterize_forward(camera, splat_data, bg, 0, 0, 0, 0, false);
        EXPECT_TRUE(forward.has_value());
        strategy.get_optimizer().zero_grad(0);
        FastGSFusedExtraGradients fused;
        fused.impact_sum_out = impact_sum.ptr<float>();
        fused.impact_peak_out = impact_peak.ptr<float>();
        Tensor sigma_map;
        if (!sigma.empty()) {
            sigma_map = Tensor::from_vector(sigma, TensorShape({static_cast<size_t>(height), static_cast<size_t>(width)}),
                                            Device::CUDA);
            fused.impact_photo_sigma = sigma_map.ptr<float>();
            fused.impact_contrast_eps = eps;
        }
        fast_rasterize_backward(forward->second, Tensor::zeros_like(forward->first.image), splat_data,
                                strategy.get_optimizer(), {}, {}, DensificationType::None, 1, fused);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        return std::pair{impact_sum.cpu().to_vector(), impact_peak.cpu().to_vector()};
    };
    const size_t pixels = static_cast<size_t>(width) * height;
    const auto [sum, peak] = run({}, 0.01f);
    ASSERT_GT(sum[0], 1.0f);

    const auto [unit_sum, unit_peak] = run(std::vector<float>(pixels, 0.0f), 1.0f);
    EXPECT_EQ(unit_sum, sum);
    EXPECT_EQ(unit_peak, peak);

    const auto [scaled_sum, scaled_peak] = run(std::vector<float>(pixels, 0.24f), 0.01f);
    EXPECT_NEAR(scaled_sum[0], sum[0] / 0.25f, 1e-4f * sum[0] / 0.25f);
    EXPECT_NEAR(scaled_peak[0], peak[0] / 0.25f, 1e-4f * peak[0] / 0.25f);
    EXPECT_NEAR(scaled_sum[1], 0.0f, 1e-3f);

    // Flat (sigma 0) on the left half, textured (sigma 0.49) on the right: with eps 0.01 the left pixels count
    // 100x, the right ones 2x, so the sum lies between and the pixel peak is the flat side's.
    std::vector<float> halves(pixels, 0.49f);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width / 2; ++x)
            halves[static_cast<size_t>(y) * width + x] = 0.0f;
    const auto [half_sum, half_peak] = run(halves, 0.01f);
    EXPECT_GT(half_sum[0], 2.0f * sum[0] * 1.01f);
    EXPECT_LT(half_sum[0], 100.0f * sum[0] * 0.99f);
    EXPECT_NEAR(half_peak[0], 100.0f * peak[0], 1e-3f * 100.0f * peak[0]);
}

// The receiver rule splits a receiver whose projected long axis was short in its last window along its second
// axis: the second axis halves and carries the offset, the long and third axes take the cross scale. Off (0),
// above the threshold, or below the cap, the split is today's.
TEST(MRNFStrategyTest, ReceiverKeepLongSplitsShortReceiversAcross) {
    // Kernel: flagged rows split across, the rest as before; no flags is today's split exactly.
    {
        const auto today = run_long_axis_split(0.7f, 1.0f);
        const auto none = run_long_axis_split(0.7f, 1.0f, std::vector<bool>(64, false));
        EXPECT_EQ(none.first_scales, today.first_scales);
        EXPECT_EQ(none.second_means, today.second_means);
        std::vector<bool> flags(64);
        for (size_t i = 0; i < flags.size(); ++i)
            flags[i] = i % 2 == 0;
        const auto across = run_long_axis_split(0.7f, 1.0f, flags);
        for (size_t row = 0; row < 64; ++row) {
            SCOPED_TRACE(row);
            const float* scale = across.scales.data() + 3 * row;
            if (!flags[row]) {
                for (int c = 0; c < 3; ++c)
                    EXPECT_EQ(across.first_scales[3 * row + c], today.first_scales[3 * row + c]);
                continue;
            }
            const int axis = static_cast<int>(row % 3); // the long axis of the fixture
            int second = -1;
            for (int c = 0; c < 3; ++c)
                if (c != axis && (second < 0 || scale[c] > scale[second]))
                    second = c;
            const float* q = across.quats.data() + 4 * row;
            const float w = q[0], x = q[1], y = q[2], z = q[3];
            const float R[9] = {1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                                2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                                2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)};
            const float length = 0.5f * std::exp(scale[second]);
            for (int c = 0; c < 3; ++c) {
                EXPECT_NEAR(across.first_scales[3 * row + c], scale[c] + std::log(c == second ? 0.5f : 0.7f), 1e-5f);
                EXPECT_EQ(across.second_scales[3 * row + c], across.first_scales[3 * row + c]);
                const float offset = R[3 * c + second] * length;
                EXPECT_NEAR(across.first_means[3 * row + c], across.means[3 * row + c] + offset, 1e-5f);
                EXPECT_NEAR(across.second_means[3 * row + c], across.means[3 * row + c] - offset, 1e-5f);
            }
        }
    }

    // Strategy: two refills at the cap split two receivers; every splat's last major sigma is set first.
    struct Run {
        std::vector<float> scales;
        std::vector<bool> free;
        size_t across = 0;
        bool rows = false;
    };
    const auto run_refill = [](const float keep_long, const float major, const size_t cap) {
        constexpr size_t m = 200;
        auto data = create_mrnf_test_splat_data(static_cast<int>(m));
        MRNF strategy(data);
        auto params = need_growth_test_params(static_cast<int>(cap));
        params.receiver_keep_long = keep_long;
        strategy.initialize(params);
        Run run;
        if (auto window = strategy.major_sigma_scratch(0); window.is_valid()) {
            run.rows = true;
            window.fill_(major);
            strategy.close_keep_long_window();
        }
        std::vector<float> error(m, 1.0f), footprint(m, 0.01f);
        set_need_window(data, strategy._refine_weight_max, strategy.visibility_accumulator(), error, footprint);
        strategy.soft_delete_rows(row_range(m - 2, m));
        const auto result = strategy.grow_and_split(0, 2, strategy.compute_refine_candidates(), 0);
        EXPECT_EQ(result.receiver_splits > 0, cap == m);
        run.scales = data.scaling_raw().cpu().to_vector();
        const auto free_cpu = strategy._free_mask.cpu();
        for (size_t row = 0; row < m; ++row)
            run.free.push_back(free_cpu.ptr<bool>()[row]);
        run.across = strategy._last_split_across;
        return run;
    };
    // The fixture's splats are round (all log scales 0), so the long axis is x and the second y.
    const auto children_split_on = [](const Run& run, const int axis, const float cross) {
        size_t children = 0;
        for (size_t row = 0; row < run.free.size(); ++row) {
            if (run.free[row] || (run.scales[3 * row] == 0.0f && run.scales[3 * row + 1] == 0.0f))
                continue;
            ++children;
            for (int c = 0; c < 3; ++c)
                EXPECT_NEAR(run.scales[3 * row + c], std::log(c == axis ? 0.5f : cross), 1e-6f) << row << " " << c;
        }
        return children;
    };
    const auto off = run_refill(0.0f, 1.0f, 200);
    EXPECT_FALSE(off.rows);
    EXPECT_EQ(children_split_on(off, 0, 0.7f), 4u);
    EXPECT_EQ(off.across, 0u);

    const auto short_major = run_refill(3.0f, 1.0f, 200);
    EXPECT_TRUE(short_major.rows);
    EXPECT_EQ(children_split_on(short_major, 1, 0.7f), 4u);
    EXPECT_EQ(short_major.across, 2u);

    const auto long_major = run_refill(3.0f, 3.0f, 200); // not below the threshold
    EXPECT_EQ(children_split_on(long_major, 0, 0.7f), 4u);
    EXPECT_EQ(long_major.across, 0u);

    const auto below_cap = run_refill(3.0f, 1.0f, 400); // no receivers: today's refill split
    EXPECT_EQ(below_cap.across, 0u);
    for (size_t row = 0; row < below_cap.free.size(); ++row)
        if (!below_cap.free[row] && below_cap.scales[3 * row] != 0.0f)
            EXPECT_NEAR(below_cap.scales[3 * row], std::log(0.5f), 1e-6f) << row;
}

// The major sigma window starts at 0 (undrawn) and is what the backward raises; a refine closes it (drawn rows
// take its maximum, the others keep their last value, +inf before any) and restarts it. Freed slots and new
// rows start undrawn; the rows follow their splats and are dropped at the topology freeze. The real backward
// writes the raw projected major sigma and keeps the maximum.
TEST(MRNFStrategyTest, ReceiverKeepLongWindowFollowsRendersAndSplats) {
    constexpr size_t n = 4;
    {
        auto splat_data = create_mrnf_test_splat_data(static_cast<int>(n), 0);
        MRNF strategy(splat_data);
        auto params = need_growth_test_params(32);
        params.receiver_keep_long = 3.0f;
        strategy.initialize(params);
        ASSERT_TRUE(strategy.keep_long_on());
        auto window = strategy.major_sigma_scratch(100);
        ASSERT_TRUE(window.is_valid());
        EXPECT_EQ(window.ptr<float>(), strategy._keep_long_window_major.ptr<float>());
        EXPECT_EQ(window.cpu().to_vector(), std::vector<float>(n, 0.0f));
        EXPECT_EQ(strategy._keep_long_major.cpu().to_vector(), std::vector<float>(n, kInf));

        window.copy_(Tensor::from_vector(std::vector<float>{2.0f, 0.0f, 5.0f, 1.0f}, TensorShape({n}), Device::CUDA));
        strategy.close_keep_long_window();
        EXPECT_EQ(strategy._keep_long_major.cpu().to_vector(), (std::vector<float>{2.0f, kInf, 5.0f, 1.0f}));
        EXPECT_EQ(strategy._keep_long_window_major.cpu().to_vector(), std::vector<float>(n, 0.0f));
        EXPECT_EQ(strategy.keep_long_split_across().to(DataType::Float32).cpu().to_vector(),
                  (std::vector<float>{1.0f, 0.0f, 0.0f, 1.0f}));
        strategy.major_sigma_scratch(200).copy_(
            Tensor::from_vector(std::vector<float>{0.0f, 4.0f, 2.5f, 0.0f}, TensorShape({n}), Device::CUDA));
        strategy.close_keep_long_window();
        EXPECT_EQ(strategy._keep_long_major.cpu().to_vector(), (std::vector<float>{2.0f, 4.0f, 2.5f, 1.0f}));

        strategy.soft_delete_rows(row_range(2, 3));
        EXPECT_EQ(strategy._keep_long_major.cpu().to_vector()[2], kInf);
        const auto mask = Tensor::from_vector(std::vector<bool>{true, false, false, false}, TensorShape({n}), Device::CUDA);
        strategy.remove_gaussians(mask);
        ASSERT_EQ(splat_data.size(), 3u);
        strategy.ensure_explore_rows();
        EXPECT_EQ(strategy._keep_long_major.cpu().to_vector(), (std::vector<float>{4.0f, kInf, 1.0f}));
        strategy._keep_long_major = strategy._keep_long_major.slice(0, 0, 2).contiguous();
        strategy.ensure_explore_rows();
        EXPECT_EQ(strategy._keep_long_major.cpu().to_vector(), (std::vector<float>{4.0f, kInf, kInf}));

        RenderOutput output;
        strategy.post_backward(static_cast<int>(params.stop_refine), output);
        EXPECT_FALSE(strategy.keep_long_on());
        EXPECT_FALSE(strategy.major_sigma_scratch(static_cast<int>(params.stop_refine) + 1).is_valid());
        strategy.explore_scratch(static_cast<int>(params.stop_refine) + 1);
        EXPECT_FALSE(strategy._keep_long_window_major.is_valid());
        EXPECT_FALSE(strategy._keep_long_major.is_valid());
    }

    // Real backward: the larger eigenvalue of the undilated 2D covariance, focal x scale / depth on the axis.
    constexpr int width = 64, height = 48;
    constexpr float focal = 60.0f;
    constexpr float sh_c0 = 0.28209479177387814f;
    auto R = Tensor::eye(3, Device::CUDA);
    auto T = Tensor::from_vector(std::vector<float>{0.0f, 0.0f, 4.0f}, TensorShape({3}), Device::CUDA);
    Camera camera(R, T, focal, focal, width / 2.0f, height / 2.0f, Tensor(), Tensor(), CameraModelType::PINHOLE,
                  "keep_long", "", std::filesystem::path{}, width, height, 0);
    auto bg = Tensor::full({3}, 0.5f, Device::CUDA);
    const std::vector<float> means{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 3.0f, 0.0f, 0.0f};
    const std::vector<float> rotations{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    // 0: long in x (major 3.70 px), thin in y; 1: round behind it (2.69 px); 2: outside the view.
    const std::vector<float> log_scales{-1.4f, -3.0f, -1.6f, -1.6f, -1.6f, -1.6f, -1.6f, -1.6f, -1.6f};
    const std::vector<float> opacities{raw_of(0.95f), raw_of(0.8f), raw_of(0.8f)};
    const std::vector<float> sh0{0.4f / sh_c0, -0.4f / sh_c0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    SplatData splat_data(0, Tensor::from_vector(means, TensorShape({3, 3}), Device::CUDA),
                         Tensor::from_vector(sh0, TensorShape({3, 1, 3}), Device::CUDA),
                         Tensor::zeros({3, 0, 3}, Device::CUDA),
                         Tensor::from_vector(log_scales, TensorShape({3, 3}), Device::CUDA),
                         Tensor::from_vector(rotations, TensorShape({3, 4}), Device::CUDA),
                         Tensor::from_vector(opacities, TensorShape({3, 1}), Device::CUDA), 1.0f);
    MRNF strategy(splat_data);
    auto params = need_growth_test_params(3);
    params.receiver_keep_long = 3.0f;
    strategy.initialize(params);
    auto window = strategy.major_sigma_scratch(100);
    ASSERT_TRUE(window.is_valid());
    const auto backward = [&] {
        auto forward = fast_rasterize_forward(camera, splat_data, bg, 0, 0, 0, 0, false);
        ASSERT_TRUE(forward.has_value()) << lfs::format_for_developer(forward.error());
        strategy.get_optimizer().zero_grad(0);
        FastGSFusedExtraGradients fused;
        fused.explore_major_sigma = window.ptr<float>();
        fast_rasterize_backward(forward->second, Tensor::full_like(forward->first.image, 0.01f), splat_data,
                                strategy.get_optimizer(), {}, {}, DensificationType::None, 1, fused);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    };
    const auto start_scales = splat_data.scaling_raw().cpu().to_vector();
    backward();
    const auto major = window.cpu().to_vector();
    EXPECT_NEAR(major[0], focal * std::exp(start_scales[0]) / 4.0f, 1e-3f);
    EXPECT_NEAR(major[1], focal * std::exp(start_scales[3]) / 4.5f, 1e-3f);
    EXPECT_EQ(major[2], 0.0f); // not drawn
    window.copy_(Tensor::from_vector(std::vector<float>{9.0f, 0.1f, 0.0f}, TensorShape({3}), Device::CUDA));
    const auto scales_before = splat_data.scaling_raw().cpu().to_vector();
    backward();
    const auto raised = window.cpu().to_vector();
    EXPECT_EQ(raised[0], 9.0f);
    EXPECT_NEAR(raised[1], focal * std::exp(scales_before[3]) / 4.5f, 1e-3f);
}
