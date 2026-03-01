#pragma once

#include "istrategy.hpp"
#include "trainer.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "optimizer/scheduler.hpp"

#include <vector>

namespace lfs::training {

    class ImprovedGsPlus : public IStrategy {
    public:
        ImprovedGsPlus() = delete;

        explicit ImprovedGsPlus(lfs::core::SplatData& splat_data);

        // Preventing Move & copy operators
        ImprovedGsPlus(const ImprovedGsPlus&) = delete;
        ImprovedGsPlus& operator=(const ImprovedGsPlus&) = delete;
        ImprovedGsPlus(ImprovedGsPlus&&) = delete;
        ImprovedGsPlus& operator=(ImprovedGsPlus&&) = delete;

        // IStrategy interface implementation

        void initialize(const lfs::core::param::OptimizationParameters& optimParams) override;

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
        const char* strategy_type() const override { return "I-GS+"; }

        // Reserve optimizer capacity for future growth (e.g., after checkpoint load)
        void reserve_optimizer_capacity(size_t capacity) override;

        // Get count of active (non-free) Gaussians
        size_t active_count() const;

        // Get count of free slots available for reuse
        size_t free_count() const;

        // Get indices of active (non-free) Gaussians for export
        lfs::core::Tensor get_active_indices() const;

        // Setters
        void set_views(std::shared_ptr<CameraDataset> views) noexcept { this->_views = views; }

        // weighting used to compute gaussian score
        struct ScoreCoefficients {
            float accum_importance; // 10 --> 30
            float blend_importance; // 50 --> 30

            float mse_importance;  // 50
            float edge_importance; // 50
        };

    private:
        // Helper Functions
        inline const int64_t get_current_budget() const noexcept { return _budget_schedule[_current_step + 1]; }
        inline const unsigned global_seed() const noexcept { return _current_step; } // for camera sampling
        const std::vector<CameraExample> random_cam_sample(const int N = 10) const;  // N minimum

        const lfs::core::Tensor get_loss_map(const lfs::core::Tensor reconstructed_img, const lfs::core::Tensor original_img, const lfs::core::Tensor edge_loss_norm) const;

        std::vector<int64_t> get_count_array();
        void get_all_edges();

        const lfs::core::Tensor compute_gaussian_score(const lfs::core::Tensor& gradients);
        void densify_with_score(const lfs::core::Tensor& scores, const lfs::core::Tensor& grads, const int64_t budget);
        void LAS_densify(const lfs::core::Tensor& scores, const int64_t allocation_budget, const lfs::core::Tensor& grad_mask, const lfs::core::Tensor& grads);

        void reset_opacity();
        void prune_post_reset();
        void prune_with_score(const int iter, const lfs::core::Tensor& scores);
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
        lfs::core::Tensor _all_edges;

        // Pointers to external data
        std::shared_ptr<CameraDataset> _views;

        // Member variables
        std::unique_ptr<AdamOptimizer> _optimizer;
        std::unique_ptr<ExponentialLR> _scheduler;
        std::unique_ptr<ExponentialLR> _fast_scheduler;
        lfs::core::SplatData* _splat_data;
        std::unique_ptr<const lfs::core::param::OptimizationParameters> _params;

        static constexpr ScoreCoefficients _score_coefficients = {10, 50, 50, 50};

        // Free slot tracking - bool tensor [capacity], true = slot is free for reuse
        lfs::core::Tensor _free_mask;
    };
} // namespace lfs::training