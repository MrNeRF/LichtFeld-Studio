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
class MRNFStrategyTest_CompactSplatsCorrectAndPeakBelowThreeX_Test;
class MRNFStrategyTest_ExploreSplitsAreDisjointAndRespectMaxCap_Test;
class MRNFStrategyTest_FarGrowthCapConstrainsOutsideAllocations_Test;
class MRNFStrategyTest_FarDecayScaleAppliesOnlyToFarUnfrozenRows_Test;
class MRNFStrategyTest_SeedFromViewInsertsRequestedRows_Test;
class MRNFStrategyTest_DensificationInfoShapeIsTwoRows_Test;
class MRNFStrategyTest_ZeroVisibilityProducesNoGrowth_Test;
class MRNFStrategyTest_PerSplatMeanStepScalesWithExtentAndClamps_Test;
class MRNFStrategyTest_CadenceScaledMatchesRefineEvery_Test;
class MRNFStrategyTest_FarStarvationFactorFromSyntheticPopulations_Test;
class MRNFStrategyTest_CensusGateActivatesAndSuppressesFarFeatures_Test;
class MRNFStrategyTest_ExploreStarvationWeights_Test;
class MRNFStrategyTest_BackgroundImprovementsOffDisablesEveryProfileMechanism_Test;
class MRNFStrategyTest_BackgroundImprovementsOnKeepsProfileMechanisms_Test;

namespace lfs::training {

    inline constexpr int kExploreSplits = 20;
    inline constexpr int kExploreSeeds = 20;
    inline constexpr float kSeedOpacity = 0.03f;
    inline constexpr float kFarGrowthCap = 0.3f;
    inline constexpr float kFarDecayScale = 0.25f;
    inline constexpr float kFarMaskOrbits = 2.0f;
    inline constexpr float kSeedDepthOrbits = 32.0f;
    inline constexpr float kFarCapRatioFull = 2.0f;
    inline constexpr float kFarCapRatioRich = 3.5f;
    inline constexpr float kStarvEps = mrnf_strategy::kStarvEps;
    inline constexpr float kStarvGamma = mrnf_strategy::kStarvGamma;
    inline constexpr float kExploreStarvDose = mrnf_strategy::kExploreStarvDose;

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
        void post_render(int iter, RenderOutput& render_output) override;
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
        void set_training_dataset(std::shared_ptr<CameraDataset> views) override;

    private:
        friend class ::MRNFStrategyTest_EdgeGuidanceFactorPrefersHigherPrecomputedEdgeScores_Test;
        friend class ::MRNFStrategyTest_GrowAndSplitResetsOptimizerStateForParents_Test;
        friend class ::MRNFStrategyTest_SHDegree0KeepsShNEmptyAndFusedAdamUsableAfterGrowth_Test;
        friend class ::MRNFStrategyTest_GrowAndSplitUsesIgsPlusSplitRule_Test;
        friend class ::MRNFStrategyTest_GrowAndSplitWithoutMaxCapExtendsBookkeepingMasks_Test;
        friend class ::MRNFStrategyTest_DeletedMaskCapacityGrowthPreservesExistingRows_Test;
        friend class ::MRNFStrategyTest_GrowAndSplitReplacementSkipsZeroWeightCandidates_Test;
        friend class ::MRNFStrategyTest_GrowAndSplitReusesFreeSlotsBeforeAppending_Test;
        friend class ::MRNFStrategyTest_SerializeRoundTripPreservesFreeMask_Test;
        friend class ::MRNFStrategyTest_SerializeRoundTripPreservesLrScheduleState_Test;
        friend class ::MRNFStrategyTest_DeserializeResizesTransientBuffersToLoadedModel_Test;
        friend class ::MRNFStrategyTest_SetOptimizationParamsRecomputesDecayFromCurrentState_Test;
        friend class ::MRNFStrategyTest_DegenerateBoundsStayInvalidAndKeepFiniteMeanLearningRate_Test;
        friend class ::MRNFStrategyTest_LineBoundsUseFiniteSceneScaleForMeanLearningRate_Test;
        friend class ::CropDampingStrategyTest_MrnfRejectedRowsAreNotRefineCandidatesAtZeroScale_Test;
        friend class ::MRNFStrategyTest_CompactSplatsCorrectAndPeakBelowThreeX_Test;
        friend class ::MRNFStrategyTest_ExploreSplitsAreDisjointAndRespectMaxCap_Test;
        friend class ::MRNFStrategyTest_FarGrowthCapConstrainsOutsideAllocations_Test;
        friend class ::MRNFStrategyTest_FarDecayScaleAppliesOnlyToFarUnfrozenRows_Test;
        friend class ::MRNFStrategyTest_SeedFromViewInsertsRequestedRows_Test;
        friend class ::MRNFStrategyTest_DensificationInfoShapeIsTwoRows_Test;
        friend class ::MRNFStrategyTest_ZeroVisibilityProducesNoGrowth_Test;
        friend class ::MRNFStrategyTest_PerSplatMeanStepScalesWithExtentAndClamps_Test;
        friend class ::MRNFStrategyTest_CadenceScaledMatchesRefineEvery_Test;
        friend class ::MRNFStrategyTest_FarStarvationFactorFromSyntheticPopulations_Test;
        friend class ::MRNFStrategyTest_CensusGateActivatesAndSuppressesFarFeatures_Test;
        friend class ::MRNFStrategyTest_ExploreStarvationWeights_Test;
        friend class ::MRNFStrategyTest_BackgroundImprovementsOffDisablesEveryProfileMechanism_Test;
        friend class ::MRNFStrategyTest_BackgroundImprovementsOnKeepsProfileMechanisms_Test;

        struct FarGrowthState {
            bool active = false;
            lfs::core::Tensor outside_mask;
            int outside_used = 0;
            int allocated = 0;
            int reserved_for_seeds = 0;
            float cap = 1.0f;
        };

        void refine(int iter, RenderOutput& render_output);
        void grow_and_split(int iter, int pruned_count);
        [[nodiscard]] int effective_grow_until_iter() const;
        [[nodiscard]] lfs::core::Tensor compute_refine_candidates() const;
        void apply_decay(int iter);
        void inject_noise(int iter);
        void compact_splats(const lfs::core::Tensor& keep_mask);
        void compute_bounds();
        void sync_mean_learning_rate();
        void ensure_densification_info_shape();
        void enforce_max_cap();
        void refresh_decay_schedule_from_current_state();
        void accumulate_edge_sample(int iter, const RenderOutput& render_output);
        [[nodiscard]] bool should_accumulate_edge_sample(int iter) const;
        [[nodiscard]] bool should_accumulate_view_sample(int iter) const;
        [[nodiscard]] bool should_accumulate_explore_sample(int iter) const;
        [[nodiscard]] int edge_target_samples_per_refine_window() const;
        void reset_edge_accumulator();
        void reset_explore_accumulator();
        void accumulate_explore_sample(int iter, const RenderOutput& render_output);
        void cache_seed_view(int iter, const RenderOutput& render_output);
        [[nodiscard]] bool should_cache_seed_view(int iter) const;
        void seed_from_view(int iter, const RenderOutput& render_output);
        [[nodiscard]] bool cfg_ratio_rank_on() const;
        [[nodiscard]] float cfg_ratio_pow() const;
        [[nodiscard]] int cfg_fill_target_iter() const;
        [[nodiscard]] int cfg_seed_dose() const;
        [[nodiscard]] bool background_improvements_enabled() const;
        [[nodiscard]] bool far_operators_active() const;
        void refresh_camera_hull();
        void refresh_far_field_mask(size_t n);
        void publish_mean_step_far_mask();
        void ensure_mean_step_far_mask();
        [[nodiscard]] int cadence_scaled(int count) const;
        [[nodiscard]] int starved_cadence_count(int count) const;
        [[nodiscard]] float effective_far_growth_cap() const;
        [[nodiscard]] float effective_far_decay_scale() const;
        [[nodiscard]] float effective_mean_step_ratio_max() const;
        [[nodiscard]] static float far_starvation_factor(float ratio, float full, float rich);
        [[nodiscard]] static float explore_starvation_multiplier(float vis_i, float median_vis);
        [[nodiscard]] bool explore_starvation_weighting_enabled() const;
        lfs::core::Tensor build_explore_split_weights(
            size_t n,
            const lfs::core::Tensor& active_mask,
            const lfs::core::Tensor& trainable_mask,
            const lfs::core::Tensor& replace_mask,
            const lfs::core::Tensor& growth_inds);
        void apply_explore_starvation_weights(lfs::core::Tensor& weights, size_t n);
        void update_far_starvation();
        void begin_far_growth_window(size_t n, int reserved_seeds);
        [[nodiscard]] size_t densification_row_count() const;
        [[nodiscard]] lfs::core::Tensor sample_gumbel_with_far_guard(
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
            size_t K);
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
            int64_t count);
        [[nodiscard]] lfs::core::Tensor edge_guidance_factor();

        std::unique_ptr<AdamOptimizer> _optimizer;
        std::unique_ptr<ExponentialLR> _scheduler;
        lfs::core::SplatData* _splat_data = nullptr;
        std::unique_ptr<const lfs::core::param::OptimizationParameters> _params;

        std::shared_ptr<CameraDataset> _views;

        lfs::core::Tensor _refine_weight_max;
        lfs::core::Tensor _refine_ratio_max;
        lfs::core::Tensor _vis_count;
        lfs::core::Tensor _precomputed_edge_scores;
        bool _edge_precompute_valid = false;
        lfs::core::Tensor _edge_score_sum;
        lfs::core::Tensor _edge_canny_nms_output;
        int _edge_sample_count = 0;
        int _edge_last_sample_iter = -1;
        lfs::core::Tensor _explore_score_sum;
        lfs::core::Tensor _explore_error_hw;
        lfs::core::Tensor _explore_view_scores;
        lfs::core::Tensor _explore_means2d;
        lfs::core::Tensor _explore_radii;
        int _explore_sample_count = 0;
        int _explore_last_sample_iter = -1;
        lfs::core::Tensor _cached_seed_image;
        lfs::core::Tensor _cached_seed_target;
        lfs::core::Tensor _cached_seed_alpha;
        lfs::core::Tensor _cached_seed_depth;
        lfs::core::Camera* _cached_seed_camera = nullptr;
        int _cached_seed_width = 0;
        int _cached_seed_height = 0;
        bool _cached_seed_valid = false;
        FarGrowthState _far_growth;
        lfs::core::Tensor _far_field_mask;
        float _cam_centroid[3] = {0.0f, 0.0f, 0.0f};
        float _orbit_radius = 0.0f;
        bool _camera_hull_valid = false;
        bool _scene_has_far_field = true;
        bool _logged_degenerate_hull = false;
        size_t _initial_sfm_point_count = 0;
        float _far_starvation = 1.0f;
        float _logged_far_starvation = -1.0f;
        lfs::core::Tensor _free_mask;

        DensifyChildWorkspace _densify_ws;
        DensifyNScratch _densify_n_scratch;
        GumbelTopKScratch _gumbel_scratch;
        PositiveMedianScratch _median_scratch;
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
        int _growth_window_count = 0;
        float _median_splat_extent = 0.0f;
        bool _median_splat_extent_valid = false;

        [[nodiscard]] lfs::core::Tensor build_replace_parent_weights(
            size_t n,
            const lfs::core::Tensor& active_mask,
            const lfs::core::Tensor& trainable_mask,
            const lfs::core::Tensor& edge_guidance) const;

        // MRNF uses independent exponential schedules for mean and scale learning rates.
        double _mean_lr_unscaled = 0.0;
        double _scale_lr_current = 0.0;
        double _mean_lr_gamma = 1.0;
        double _scale_lr_gamma = 1.0;
    };

} // namespace lfs::training
