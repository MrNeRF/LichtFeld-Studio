/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "optimizer/render_output.hpp"
#include <exception>
#include <functional>
#include <istream>
#include <memory>
#include <ostream>
#include <utility>

namespace lfs::io {
    class PipelinedImageLoader;
}

namespace lfs::training {

    struct FastGSFusedMrnfHooks;

    class CameraDataset;

    /**
     * @brief Strategy interface for Gaussian splatting optimization.
     *
     * Strategies operate on a SplatData reference owned by the Scene.
     * This allows the same model to be used for both training and visualization.
     */
    class IStrategy {
    public:
        virtual ~IStrategy() = default;

        virtual void initialize(const lfs::core::param::OptimizationParameters& optimParams) = 0;

        virtual void pre_step(int /*iter*/, RenderOutput& /*render_output*/) {}

        virtual void post_render(int /*iter*/, RenderOutput& /*render_output*/) {}

        virtual void post_backward(int iter, RenderOutput& render_output) = 0;

        virtual void step(int iter) = 0;

        /// Visits row-indexed strategy aux tensors (vis counts, error maxima, free
        /// masks) so a Morton reorder permutes them with the model.
        virtual void for_each_gaussian_row(const std::function<void(lfs::core::Tensor&)>&) {}

        virtual bool is_refining(int iter) const = 0;
        [[nodiscard]] virtual lfs::core::Tensor rendered_support_counts() const { return {}; }

        // Get the underlying Gaussian model (reference to Scene-owned data)
        virtual lfs::core::SplatData& get_model() = 0;
        virtual const lfs::core::SplatData& get_model() const = 0;

        // Get the optimizer (for gradient access during backward pass)
        virtual class AdamOptimizer& get_optimizer() = 0;
        virtual const class AdamOptimizer& get_optimizer() const = 0;

        // Remove Gaussians based on mask
        virtual void remove_gaussians(const lfs::core::Tensor& mask) = 0;

        // Serialization for checkpoints
        virtual void serialize(std::ostream& os) const = 0;
        virtual void deserialize(std::istream& is) = 0;

        // Strategy type identifier for checkpoint compatibility
        virtual const char* strategy_type() const = 0;

        // Reserve optimizer capacity for future growth (e.g., after checkpoint load)
        virtual void reserve_optimizer_capacity(size_t capacity) = 0;

        // Update the strategy's cached optimization parameters after checkpoint params are resolved.
        virtual void set_optimization_params(const lfs::core::param::OptimizationParameters&) {}

        // Restarts the decay of the means learning rate from its initial value, over the remaining steps.
        virtual void restart_mean_lr() {}

        // Optional hook for strategies that need the training dataset (e.g., for view-based scoring)
        virtual void set_training_dataset(std::shared_ptr<CameraDataset>) {}
        virtual std::shared_ptr<CameraDataset> get_training_dataset() const { return {}; }

        virtual void set_image_loader(lfs::io::PipelinedImageLoader*) {}

        // Optional FastGS edge-scoring contract. The main backward writes one
        // view into a zeroed scratch vector. The completion callback runs only
        // after a successful backward and lets the strategy normalize/mask the
        // view before adding it to its refine window.
        virtual lfs::core::Tensor edge_score_scratch(int /*iter*/) { return {}; }
        virtual void on_edge_score_accumulated(int /*iter*/) {}
        // Optional FastGS sparse variant: only rows in visible_indices[0..n_visible) can be non-zero.
        virtual void on_edge_score_accumulated_sparse(
            int iter, const uint32_t* /*visible_indices*/, size_t /*n_visible*/, void* /*stream*/) {
            on_edge_score_accumulated(iter);
        }

        // Optional FastGS single-tile fast path: folds per-step densification error and blend peak
        // directly over visible_indices while the forward context is live. Returns true when handled.
        virtual bool fold_step_stats_sparse(
            int /*backward_iter*/, const uint32_t* /*visible_indices*/, size_t /*n_visible*/, void* /*stream*/) {
            return false;
        }

        // Optional FastGS fused variant of fold_step_stats_sparse (+ the next post_backward's mean
        // noise when allow_noise): fills `out` for the sparse backward without changing strategy
        // state. Returns true when the fold was prepared.
        virtual bool prepare_fastgs_fused_mrnf_hooks(
            int /*backward_iter*/, bool /*allow_noise*/, void* /*stream*/, FastGSFusedMrnfHooks& /*out*/) {
            return false;
        }
        // Records that the sparse backward consumed the prepared fold (and noise).
        virtual void commit_fastgs_fused_mrnf_hooks(const FastGSFusedMrnfHooks& /*prepared*/, void* /*stream*/) {}

        // Optional FastGS [N] row the backward raises to each splat's largest blend weight (alpha x T)
        // over the pixels of the view. Empty when not in use.
        virtual lfs::core::Tensor blend_peak_scratch(int /*iter*/) { return {}; }
        // FastGS: the scale, rank and flatten regularizers skip splats that did not contribute to the view.
        virtual bool regularize_rendered_only() const { return false; }
    };

    class ICheckpointStateAdopter {
    public:
        virtual ~ICheckpointStateAdopter() = default;
        virtual bool has_checkpoint_runtime_state() const noexcept = 0;
        virtual bool can_adopt_checkpoint_state(const IStrategy& source) const noexcept = 0;
        virtual void adopt_checkpoint_state(IStrategy& source) noexcept = 0;

    protected:
        template <typename Strategy>
        static Strategy& checked_checkpoint_source(IStrategy& source) noexcept {
            if (auto* typed_source = dynamic_cast<Strategy*>(&source))
                return *typed_source;
            std::terminate();
        }
    };
} // namespace lfs::training
