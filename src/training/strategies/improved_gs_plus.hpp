/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "istrategy.hpp"

#include "optimizer/adam_optimizer.hpp"
#include "optimizer/scheduler.hpp"
#include "trainer.hpp"

#include <vector>

namespace lfs::training {

    class ImprovedGSPlus : public IStrategy {
    public:
        ImprovedGSPlus() = delete;

        explicit ImprovedGSPlus(lfs::core::SplatData& splat_data);

        // Preventing Move & copy operators
        ImprovedGSPlus(const ImprovedGSPlus&) = delete;
        ImprovedGSPlus& operator=(const ImprovedGSPlus&) = delete;
        ImprovedGSPlus(ImprovedGSPlus&&) = delete;
        ImprovedGSPlus& operator=(ImprovedGSPlus&&) = delete;

        // IStrategy interface implementation

        void initialize(const lfs::core::param::OptimizationParameters& optimParams) override;

        void pre_step(int iter, RenderOutput& render_output) override;

        void post_backward(int iter, RenderOutput& render_output) override;

        void step(int iter) override;

        bool is_refining(int iter) const override;

        lfs::core::SplatData& get_model() override { return *_splat_data; }
        const lfs::core::SplatData& get_model() const override { return *_splat_data; }

        void remove_gaussians(const lfs::core::Tensor& mask) override;

        // IStrategy interface - optimizer access
        AdamOptimizer& get_optimizer() override { return *_optimizer; }
        const AdamOptimizer& get_optimizer() const override { return *_optimizer; }
        ExponentialLR* get_scheduler() { return _scheduler.get(); }
        const ExponentialLR* get_scheduler() const { return _scheduler.get(); }

        // Serialization for checkpoints
        void serialize(std::ostream& os) const override;
        void deserialize(std::istream& is) override;
        const char* strategy_type() const override { return "igs+"; }

        // Reserve optimizer capacity for future growth (e.g., after checkpoint load)
        void reserve_optimizer_capacity(size_t capacity) override;
        void set_optimization_params(const lfs::core::param::OptimizationParameters& params) override {
            _params = std::make_unique<const lfs::core::param::OptimizationParameters>(params);
        }

        void set_training_dataset(std::shared_ptr<CameraDataset> views) override { _views = std::move(views); }

        void set_image_loader(lfs::io::PipelinedImageLoader* loader) override { _image_loader = loader; }

        // Get count of active (non-free) Gaussians
        size_t active_count() const;

        // Get count of free slots available for reuse
        size_t free_count() const;

        // Get indices of active (non-free) Gaussians for export
        lfs::core::Tensor get_active_indices() const;

    private:
        // Helper Functions
        inline const int64_t get_current_budget() const noexcept { return _budget_schedule[_current_step + 1]; }
        inline const unsigned global_seed() const noexcept { return _current_step; } // for camera sampling
        std::vector<int> random_cam_indices(const int N = 10) const;                 // N minimum

        std::vector<int64_t> get_count_array();

        const lfs::core::Tensor compute_gaussian_score();
        void ensure_error_score_shape();
        void densify_with_score(const lfs::core::Tensor& edge_scores, const lfs::core::Tensor& error_scores, const int64_t budget);
        void LAS_densify(const lfs::core::Tensor& scores, const int64_t allocation_budget);

        void reset_opacity();
        void prune_post_reset();
        void opacity_prune(const int iter);
        void remove(const lfs::core::Tensor& is_prune);
        void mark_as_free(const lfs::core::Tensor& indices);

        std::pair<lfs::core::Tensor, int64_t> fill_free_slots_with_data(
            const lfs::core::Tensor& positions,
            const lfs::core::Tensor& rotations,
            const lfs::core::Tensor& scales,
            const lfs::core::Tensor& sh0,
            const lfs::core::Tensor& shN,
            const lfs::core::Tensor& opacities,
            int64_t count);

        // Auxiliary variables
        int64_t _initial_points;
        int _current_step;
        int _total_steps;

        std::vector<int64_t> _budget_schedule;

        // Pointers to external data
        std::shared_ptr<CameraDataset> _views;
        lfs::io::PipelinedImageLoader* _image_loader = nullptr;

        // Member variables
        std::unique_ptr<AdamOptimizer> _optimizer;
        std::unique_ptr<ExponentialLR> _scheduler;
        lfs::core::SplatData* _splat_data;
        std::unique_ptr<const lfs::core::param::OptimizationParameters> _params;

        // Pre-computed edge scores for non-blocking densification
        lfs::core::Tensor _precomputed_scores;
        lfs::core::Tensor _error_score_max;
        bool _precompute_valid = false;

        // Free slot tracking - bool tensor [capacity], true = slot is free for reuse
        lfs::core::Tensor _free_mask;
    };
} // namespace lfs::training
