/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "istrategy.hpp"
#include "optimizer/adam_optimizer.hpp"
#include <cassert>
#include <cstdint>
#include <memory>

namespace lfs::training {

    // Match strategy (--strategy match): error-driven growth by long-axis splits with a dead pool that is
    // recycled in place. FastGS only.
    //
    // Every step, after the backward of step t:
    //   - every 100 steps, splats wider on screen than match_screen_limit (angular radius) are shrunk;
    //   - while t < stop_refine, each splat visible in the view with a non-zero error sum a (error map x blend
    //     weight over its pixels) folds a^match_score_power into its running mean score; oversize splats add
    //     log2(radius / limit) to an oversize sum; the peak blend weight is a running maximum with a
    //     half-life of match_peak_half_life steps;
    //   - on a refine step (t > start_refine, t % refine_every == 0, t < stop_refine): dead splats are revived
    //     as split partners of live splats drawn by score, then the model grows by match_growth (to max_cap),
    //     a match_oversize_share of the births drawn by oversize x score; one budget of match_split_budget x N
    //     covers both; scores reset;
    //   - visible faint splats take a random walk step.
    // The fused backward takes the camera-distance position step, the on-screen size hinge and rendered-only
    // shape regularizers (fused_geometry_hooks).
    class Match : public IStrategy, public ICheckpointStateAdopter {
    public:
        Match() = delete;
        explicit Match(lfs::core::SplatData& splat_data);

        Match(const Match&) = delete;
        Match& operator=(const Match&) = delete;
        Match(Match&&) = delete;
        Match& operator=(Match&&) = delete;

        void initialize(const lfs::core::param::OptimizationParameters& optimParams) override;
        void post_backward(int iter, RenderOutput& render_output) override;
        bool is_refining(int iter) const override;
        void step(int iter) override;

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
        const char* strategy_type() const override { return "match"; }

        void reserve_optimizer_capacity(size_t capacity) override;
        void set_optimization_params(const lfs::core::param::OptimizationParameters& params) override;
        void set_training_dataset(std::shared_ptr<CameraDataset> views) override;

        std::pair<lfs::core::Tensor, lfs::core::Tensor> explore_scratch(int iter) override;
        FusedGeometryHooks fused_geometry_hooks(int iter) override;

        // Iterations count from 1 and iteration i renders step i - 1 (0-based). The hooks run at the start of
        // an iteration, before its render, so they see the backward of the previous iteration: step i - 2.
        [[nodiscard]] static int completed_step(const int iter) { return iter - 2; }
        [[nodiscard]] bool refine_step(int step) const;
        [[nodiscard]] float split_opacity_share(int step) const;
        [[nodiscard]] float walk_amplitude(int step) const;
        [[nodiscard]] int steps_per_epoch() const;

        struct RefineCounts {
            size_t dead = 0;
            size_t revived = 0;
            size_t oversize_born = 0;
            size_t born = 0;
        };
        // Refine counts: the split budget max(1, (int)(share x N)) of revivals and births together; the births
        // min(target - N, budget - revived), target = max(N, min(cap, (int)(growth x N))); the oversize share
        // of the births round(share x births).
        [[nodiscard]] static size_t split_budget(const lfs::core::param::OptimizationParameters& params, size_t n);
        [[nodiscard]] static size_t growth_births(const lfs::core::param::OptimizationParameters& params, size_t n,
                                                  size_t budget, size_t revived);
        [[nodiscard]] static size_t oversize_births(const lfs::core::param::OptimizationParameters& params,
                                                    size_t added);

        // One refine at step `step` (exposed for tests).
        RefineCounts refine(int step);
        [[nodiscard]] const RefineCounts& last_refine() const noexcept { return _last_refine; }

        // [N] views of the strategy rows (tests and diagnostics).
        [[nodiscard]] lfs::core::Tensor score_rows() const { return rows_view(_score); }
        [[nodiscard]] lfs::core::Tensor score_count_rows() const { return rows_view(_score_count); }
        [[nodiscard]] lfs::core::Tensor oversize_rows() const { return rows_view(_oversize); }
        [[nodiscard]] lfs::core::Tensor peak_rows() const { return rows_view(_peak); }
        [[nodiscard]] lfs::core::Tensor streak_rows() const { return rows_view(_streak); }
        [[nodiscard]] lfs::core::Tensor visible_rows() const { return rows_view(_visible); }
        [[nodiscard]] lfs::core::Tensor step_peak_rows() const { return rows_view(_step_peak); }
        [[nodiscard]] lfs::core::Tensor radius_rows() const { return rows_view(_radius); }

    private:
        [[nodiscard]] lfs::core::Tensor rows_view(const lfs::core::Tensor& row) const;
        void ensure_rows();
        void ensure_error_row();
        size_t append_split_children(const lfs::core::Tensor& donors, float share);
        void copy_shN_rows(const lfs::core::Tensor& src, const lfs::core::Tensor& dst);
        void split(const lfs::core::Tensor& src, const lfs::core::Tensor& dst, float share);
        void apply_learning_rates(int step);
        [[nodiscard]] uint64_t seed(int step, uint64_t stream) const;

        std::unique_ptr<AdamOptimizer> _optimizer;
        lfs::core::SplatData* _splat_data = nullptr; // Scene-owned
        std::unique_ptr<const lfs::core::param::OptimizationParameters> _params;
        std::shared_ptr<CameraDataset> _views;

        // Rows of capacity >= N (first N used).
        lfs::core::Tensor _score;
        lfs::core::Tensor _score_count;
        lfs::core::Tensor _oversize;
        lfs::core::Tensor _peak;
        lfs::core::Tensor _streak;
        lfs::core::Tensor _visible;
        lfs::core::Tensor _step_peak;
        lfs::core::Tensor _radius;
        RefineCounts _last_refine;
    };

} // namespace lfs::training
