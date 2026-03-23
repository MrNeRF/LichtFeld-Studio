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

    private:
        void refine(int iter);
        void grow_and_split(int iter, int pruned_count);
        void apply_decay(int iter);
        void inject_noise(int iter);
        void compact_splats(const lfs::core::Tensor& keep_mask);
        void compute_bounds();
        void ensure_densification_info_shape();
        void enforce_max_cap();

        std::unique_ptr<AdamOptimizer> _optimizer;
        std::unique_ptr<ExponentialLR> _scheduler;
        lfs::core::SplatData* _splat_data = nullptr;
        std::unique_ptr<const lfs::core::param::OptimizationParameters> _params;

        lfs::core::Tensor _refine_weight_max;
        lfs::core::Tensor _vis_count;

        lfs_strategy::LFSBounds _bounds = {};
        bool _bounds_valid = false;

        // LFS uses independent exponential schedules for mean and scale learning rates.
        double _mean_lr_unscaled = 0.0;
        double _scale_lr_current = 0.0;
        double _mean_lr_gamma = 1.0;
        double _scale_lr_gamma = 1.0;
    };

} // namespace lfs::training
