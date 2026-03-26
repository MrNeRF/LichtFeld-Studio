/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "istrategy.hpp"
#include "kernels/lfs_kernels.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "optimizer/scheduler.hpp"
#include <memory>

class LFSStrategyTest_EdgeGuidanceFactorPrefersHigherPrecomputedEdgeScores_Test;
class LFSStrategyTest_GrowAndSplitResetsOptimizerStateForParents_Test;
class LFSStrategyTest_GrowAndSplitUsesIgsPlusSplitRule_Test;
class LFSStrategyTest_GrowAndSplitWithoutMaxCapExtendsBookkeepingMasks_Test;
class LFSStrategyTest_GrowAndSplitReplacementSkipsZeroWeightCandidates_Test;
class LFSStrategyTest_GrowAndSplitReusesFreeSlotsBeforeAppending_Test;
class LFSStrategyTest_SerializeRoundTripPreservesFreeMask_Test;
class LFSStrategyTest_SerializeRoundTripPreservesLrScheduleState_Test;
class LFSStrategyTest_DeserializeResizesTransientBuffersToLoadedModel_Test;
class LFSStrategyTest_SetOptimizationParamsRecomputesDecayFromCurrentState_Test;

namespace lfs::training {

    class LFS : public IStrategy {
    public:
        LFS() = delete;
        explicit LFS(lfs::core::SplatData& splat_data);

        LFS(const LFS&) = delete;
        LFS& operator=(const LFS&) = delete;
        LFS(LFS&&) = delete;
        LFS& operator=(LFS&&) = delete;

        void initialize(const lfs::core::param::OptimizationParameters& optimParams) override;
        void pre_step(int iter, RenderOutput& render_output) override;
        void post_backward(int iter, RenderOutput& render_output) override;
        bool is_refining(int iter) const override;
        void step(int iter) override;

        lfs::core::SplatData& get_model() override { return *_splat_data; }
        const lfs::core::SplatData& get_model() const override { return *_splat_data; }

        void remove_gaussians(const lfs::core::Tensor& mask) override;

        AdamOptimizer& get_optimizer() override { return *_optimizer; }
        const AdamOptimizer& get_optimizer() const override { return *_optimizer; }

        void serialize(std::ostream& os) const override;
        void deserialize(std::istream& is) override;
        const char* strategy_type() const override { return "lfs"; }

        void reserve_optimizer_capacity(size_t capacity) override;
        void set_optimization_params(const lfs::core::param::OptimizationParameters& params) override;
        void set_training_dataset(std::shared_ptr<CameraDataset> views) override { _views = std::move(views); }
        void set_image_loader(lfs::io::PipelinedImageLoader* loader) override { _image_loader = loader; }

    private:
        friend class ::LFSStrategyTest_EdgeGuidanceFactorPrefersHigherPrecomputedEdgeScores_Test;
        friend class ::LFSStrategyTest_GrowAndSplitResetsOptimizerStateForParents_Test;
        friend class ::LFSStrategyTest_GrowAndSplitUsesIgsPlusSplitRule_Test;
        friend class ::LFSStrategyTest_GrowAndSplitWithoutMaxCapExtendsBookkeepingMasks_Test;
        friend class ::LFSStrategyTest_GrowAndSplitReplacementSkipsZeroWeightCandidates_Test;
        friend class ::LFSStrategyTest_GrowAndSplitReusesFreeSlotsBeforeAppending_Test;
        friend class ::LFSStrategyTest_SerializeRoundTripPreservesFreeMask_Test;
        friend class ::LFSStrategyTest_SerializeRoundTripPreservesLrScheduleState_Test;
        friend class ::LFSStrategyTest_DeserializeResizesTransientBuffersToLoadedModel_Test;
        friend class ::LFSStrategyTest_SetOptimizationParamsRecomputesDecayFromCurrentState_Test;

        void refine(int iter);
        void grow_and_split(int iter, int pruned_count);
        void apply_decay(int iter);
        void inject_noise(int iter);
        void compact_splats(const lfs::core::Tensor& keep_mask);
        void compute_bounds();
        void ensure_densification_info_shape();
        void enforce_max_cap();
        void refresh_decay_schedule_from_current_state();
        size_t active_count() const;
        size_t free_count() const;
        [[nodiscard]] lfs::core::Tensor get_active_indices() const;
        void mark_as_free(const lfs::core::Tensor& indices);
        std::pair<lfs::core::Tensor, int64_t> fill_free_slots_with_data(
            const lfs::core::Tensor& positions,
            const lfs::core::Tensor& rotations,
            const lfs::core::Tensor& scales,
            const lfs::core::Tensor& sh0,
            const lfs::core::Tensor& shN,
            const lfs::core::Tensor& opacities,
            int64_t count);
        [[nodiscard]] lfs::core::Tensor compute_edge_scores(int iter);
        [[nodiscard]] lfs::core::Tensor edge_guidance_factor() const;

        std::unique_ptr<AdamOptimizer> _optimizer;
        std::unique_ptr<ExponentialLR> _scheduler;
        lfs::core::SplatData* _splat_data = nullptr;
        std::unique_ptr<const lfs::core::param::OptimizationParameters> _params;

        std::shared_ptr<CameraDataset> _views;
        lfs::io::PipelinedImageLoader* _image_loader = nullptr;

        lfs::core::Tensor _refine_weight_max;
        lfs::core::Tensor _vis_count;
        lfs::core::Tensor _precomputed_edge_scores;
        bool _edge_precompute_valid = false;
        lfs::core::Tensor _free_mask;

        lfs_strategy::LFSBounds _bounds = {};
        bool _bounds_valid = false;
        int _refine_windows_since_bounds = 0;

        // LFS uses independent exponential schedules for mean and scale learning rates.
        double _mean_lr_unscaled = 0.0;
        double _scale_lr_current = 0.0;
        double _mean_lr_gamma = 1.0;
        double _scale_lr_gamma = 1.0;
    };

} // namespace lfs::training
