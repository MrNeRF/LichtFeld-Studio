/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "istrategy.hpp"
#include "kernels/mrnf_kernels.hpp"
#include "lfs/training/refine_scratch.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "optimizer/scheduler.hpp"
#include "strategy_utils.hpp"
#include <cassert>
#include <memory>

namespace lfs::training::sh_value {
    class ShNMutationBatch;
}

class MRNFStrategyTest_EdgeGuidanceFactorPrefersHigherPrecomputedEdgeScores_Test;
class MRNFStrategyTest_GrowAndSplitResetsOptimizerStateForParents_Test;
class MRNFStrategyTest_SHDegree0KeepsShNEmptyAndFusedAdamUsableAfterGrowth_Test;
class MRNFStrategyTest_GrowAndSplitUsesIgsPlusSplitRule_Test;
class MRNFStrategyTest_GrowAndSplitOversizeChannelPrefersOversizedError_Test;
class MRNFStrategyTest_GrowAndSplitWithoutMaxCapExtendsBookkeepingMasks_Test;
class MRNFStrategyTest_DeletedMaskCapacityGrowthPreservesExistingRows_Test;
class MRNFStrategyTest_GrowAndSplitReplacementSkipsZeroWeightCandidates_Test;
class MRNFStrategyTest_GrowAndSplitReusesFreeSlotsBeforeAppending_Test;
class MRNFStrategyTest_ChunkedChildPlacementMatchesSingleChunk_Test;
class MRNFStrategyTest_SerializeRoundTripPreservesFreeMask_Test;
class MRNFStrategyTest_SerializeRoundTripPreservesLrScheduleState_Test;
class MRNFStrategyTest_DeserializeResizesTransientBuffersToLoadedModel_Test;
class MRNFStrategyTest_SetOptimizationParamsRecomputesDecayFromCurrentState_Test;
class MRNFStrategyTest_DegenerateBoundsStayInvalidAndKeepFiniteMeanLearningRate_Test;
class MRNFStrategyTest_LineBoundsUseFiniteSceneScaleForMeanLearningRate_Test;
class CropDampingStrategyTest_MrnfRejectedRowsAreNotRefineCandidatesAtZeroScale_Test;
class MRNFStrategyTest_CompactSplatsCorrectAndPeakBelowThreeX_Test;
class MRNFStrategyTest_CompactSplatsFusedPathLeavesGradsEmpty_Test;
class MRNFStrategyTest_ApplyDecaySkipsFrozenRows_Test;
class MRNFStrategyTest_DensificationInfoShapeIsTwoRows_Test;
class MRNFStrategyTest_ZeroVisibilityProducesNoGrowth_Test;
class MRNFStrategyTest_DirectAuxiliaryGrowthPreservesPrefix_Test;
class MRNFStrategyTest_EdgeWindowNormalizesViewsAndClosesBeforeRefineBackward_Test;

namespace lfs::training {

    class MRNF : public IStrategy, public ICheckpointStateAdopter {
    public:
        MRNF() = delete;
        explicit MRNF(lfs::core::SplatData& splat_data);

        MRNF(const MRNF&) = delete;
        MRNF& operator=(const MRNF&) = delete;
        MRNF(MRNF&&) = delete;
        MRNF& operator=(MRNF&&) = delete;

        void initialize(const lfs::core::param::OptimizationParameters& optimParams) override;
        void pre_step(int iter, RenderOutput& render_output) override;
        void post_backward(int iter, RenderOutput& render_output) override;
        bool is_refining(int iter) const override;
        void step(int iter) override;
        void permute_gaussian_rows(const lfs::core::Tensor& perm) override;

        lfs::core::SplatData& get_model() override { return *_splat_data; }
        const lfs::core::SplatData& get_model() const override { return *_splat_data; }

        void remove_gaussians(const lfs::core::Tensor& mask) override;

        AdamOptimizer& get_optimizer() override {
            assert(_optimizer);
            return *_optimizer;
        }
        const AdamOptimizer& get_optimizer() const override {
            assert(_optimizer);
            return *_optimizer;
        }

        void serialize(std::ostream& os) const override;
        void deserialize(std::istream& is) override;
        bool has_checkpoint_runtime_state() const noexcept override { return static_cast<bool>(_optimizer); }
        bool can_adopt_checkpoint_state(const IStrategy& loaded) const noexcept override;
        void adopt_checkpoint_state(IStrategy& loaded) noexcept override;
        const char* strategy_type() const override { return "mrnf"; }

        void reserve_optimizer_capacity(size_t capacity) override;
        void set_optimization_params(const lfs::core::param::OptimizationParameters& params) override;
        lfs::core::Tensor edge_score_scratch(int iter) override;
        void on_edge_score_accumulated(int iter) override;

    private:
        friend class ::MRNFStrategyTest_EdgeWindowNormalizesViewsAndClosesBeforeRefineBackward_Test;
        friend class ::MRNFStrategyTest_EdgeGuidanceFactorPrefersHigherPrecomputedEdgeScores_Test;
        friend class ::MRNFStrategyTest_GrowAndSplitResetsOptimizerStateForParents_Test;
        friend class ::MRNFStrategyTest_SHDegree0KeepsShNEmptyAndFusedAdamUsableAfterGrowth_Test;
        friend class ::MRNFStrategyTest_GrowAndSplitUsesIgsPlusSplitRule_Test;
        friend class ::MRNFStrategyTest_GrowAndSplitOversizeChannelPrefersOversizedError_Test;
        friend class ::MRNFStrategyTest_GrowAndSplitWithoutMaxCapExtendsBookkeepingMasks_Test;
        friend class ::MRNFStrategyTest_DeletedMaskCapacityGrowthPreservesExistingRows_Test;
        friend class ::MRNFStrategyTest_GrowAndSplitReplacementSkipsZeroWeightCandidates_Test;
        friend class ::MRNFStrategyTest_GrowAndSplitReusesFreeSlotsBeforeAppending_Test;
        friend class ::MRNFStrategyTest_ChunkedChildPlacementMatchesSingleChunk_Test;
        friend class ::MRNFStrategyTest_SerializeRoundTripPreservesFreeMask_Test;
        friend class ::MRNFStrategyTest_SerializeRoundTripPreservesLrScheduleState_Test;
        friend class ::MRNFStrategyTest_DeserializeResizesTransientBuffersToLoadedModel_Test;
        friend class ::MRNFStrategyTest_SetOptimizationParamsRecomputesDecayFromCurrentState_Test;
        friend class ::MRNFStrategyTest_DegenerateBoundsStayInvalidAndKeepFiniteMeanLearningRate_Test;
        friend class ::MRNFStrategyTest_LineBoundsUseFiniteSceneScaleForMeanLearningRate_Test;
        friend class ::CropDampingStrategyTest_MrnfRejectedRowsAreNotRefineCandidatesAtZeroScale_Test;
        friend class ::MRNFStrategyTest_CompactSplatsCorrectAndPeakBelowThreeX_Test;
        friend class ::MRNFStrategyTest_CompactSplatsFusedPathLeavesGradsEmpty_Test;
        friend class ::MRNFStrategyTest_ApplyDecaySkipsFrozenRows_Test;
        friend class ::MRNFStrategyTest_DensificationInfoShapeIsTwoRows_Test;
        friend class ::MRNFStrategyTest_ZeroVisibilityProducesNoGrowth_Test;
        friend class ::MRNFStrategyTest_DirectAuxiliaryGrowthPreservesPrefix_Test;

        void refine(int iter);
        void grow_and_split(int iter, int pruned_count);
        // Splits the given parents and places their children (free slots first,
        // then appended rows) in chunks of at most chunk_rows children.
        // Returns {children placed in free slots, children appended}.
        std::pair<size_t, size_t> split_parents_into_children(const lfs::core::Tensor& split_indices,
                                                              size_t chunk_rows);
        [[nodiscard]] bool screen_share_shrink_active(int iter) const;
        [[nodiscard]] lfs::core::Tensor compute_refine_candidates() const;
        void apply_decay(int iter);
        void inject_noise(int iter);
        void compact_splats(const lfs::core::Tensor& keep_mask);
        void compute_bounds();
        void sync_mean_learning_rate();
        void ensure_densification_info_shape();
        void enforce_max_cap();
        void refresh_decay_schedule_from_current_state();
        void reset_edge_accumulator();
        [[nodiscard]] lfs::core::Tensor visibility_accumulator() const;
        [[nodiscard]] size_t densification_row_count() const;
        [[nodiscard]] lfs::core::Tensor sample_gumbel_topk(
            const lfs::core::Tensor& weights,
            int k,
            uint64_t seed,
            size_t known_nnz = 0);
        size_t append_child_rows(
            const lfs::core::Tensor& child_means,
            const lfs::core::Tensor& child_rotations,
            const lfs::core::Tensor& child_log_scales,
            const lfs::core::Tensor& child_sh0,
            const lfs::core::Tensor& child_shN,
            const lfs::core::Tensor& child_raw_opacities,
            size_t append_start,
            size_t K,
            lfs::training::sh_value::ShNMutationBatch* shn_batch = nullptr);
        void publish_vram_attribution() noexcept;
        size_t active_count() const;
        size_t free_count() const;
        [[nodiscard]] lfs::core::Tensor get_active_indices() const;
        void mark_as_free(const lfs::core::Tensor& indices);
        // Writes child shN linear rows directly into resident swizzled splat_data.shN().
        std::pair<lfs::core::Tensor, int64_t> fill_free_slots_with_data(
            const lfs::core::Tensor& positions,
            const lfs::core::Tensor& rotations,
            const lfs::core::Tensor& scales,
            const lfs::core::Tensor& sh0,
            const lfs::core::Tensor& shN,
            const lfs::core::Tensor& opacities,
            int64_t count,
            lfs::training::sh_value::ShNMutationBatch* shn_batch = nullptr);
        [[nodiscard]] lfs::core::Tensor edge_guidance_factor();

        std::unique_ptr<AdamOptimizer> _optimizer;
        std::unique_ptr<ExponentialLR> _scheduler;
        lfs::core::SplatData* _splat_data = nullptr;
        std::unique_ptr<const lfs::core::param::OptimizationParameters> _params;

        lfs::core::Tensor _refine_weight_max;
        lfs::core::Tensor _precomputed_edge_scores;
        bool _edge_precompute_valid = false;
        lfs::core::Tensor _edge_score_sum;
        lfs::core::Tensor _edge_view_scores;
        int _edge_sample_count = 0;
        lfs::core::Tensor _free_mask;
        bool _topology_frozen = false;

        DensifyNScratch _densify_n_scratch;
        GumbelTopKScratch _gumbel_scratch;
        lfs::core::Tensor _refine_counts_dev;

        std::size_t _strategy_required_peak_bytes = 0;
        std::size_t _strategy_allocated_peak_bytes = 0;
        std::size_t _densify_n_required_peak_bytes = 0;
        std::size_t _densify_n_allocated_peak_bytes = 0;
        std::size_t _densify_child_required_peak_bytes = 0;
        std::size_t _densify_child_allocated_peak_bytes = 0;

        mrnf_strategy::MRNFBounds _bounds = {};
        bool _bounds_valid = false;
        int _refine_windows_since_bounds = 0;

        [[nodiscard]] lfs::core::Tensor build_replace_parent_weights(
            size_t n,
            const lfs::core::Tensor& active_mask,
            const lfs::core::Tensor& trainable_mask,
            const lfs::core::Tensor& edge_guidance) const;

        // Grow MRNF bookkeeping after parameter preflight and before any
        // parameter append. This is intentionally grow-only: refine resets
        // the buffers after the append, while this step preserves their live
        // prefix across a reservation boundary.
        void ensure_auxiliary_capacity_for_growth(size_t target_size);

        // MRNF uses independent exponential schedules for mean and scale learning rates.
        double _mean_lr_unscaled = 0.0;
        double _scale_lr_current = 0.0;
        double _mean_lr_gamma = 1.0;
        double _scale_lr_gamma = 1.0;
    };

} // namespace lfs::training
