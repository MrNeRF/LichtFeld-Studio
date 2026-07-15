/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "optimizer/render_output.hpp"
#include <exception>
#include <istream>
#include <memory>
#include <ostream>

namespace lfs::io {
    class PipelinedImageLoader;
}

namespace lfs::training {

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

        virtual void post_backward(int iter, RenderOutput& render_output) = 0;

        virtual void step(int iter) = 0;

        virtual bool is_refining(int iter) const = 0;

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

        // Optional hook for strategies that need the training dataset (e.g., for view-based scoring)
        virtual void set_training_dataset(std::shared_ptr<CameraDataset>) {}

        virtual void set_image_loader(lfs::io::PipelinedImageLoader*) {}
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
