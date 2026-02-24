/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "istrategy.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "optimizer/scheduler.hpp"
#include <memory>

namespace lfs::training {

    /// MCMC-based optimization strategy. SplatData owned by Scene.
    class MCMC : public IStrategy {
    public:
        MCMC() = delete;
        /// SplatData must be owned by Scene
        explicit MCMC(lfs::core::SplatData& splat_data);

        // Reference member prevents copy/move
        MCMC(const MCMC&) = delete;
        MCMC& operator=(const MCMC&) = delete;
        MCMC(MCMC&&) = delete;
        MCMC& operator=(MCMC&&) = delete;

        // IStrategy interface implementation
        void initialize(const lfs::core::param::OptimizationParameters& optimParams) override;
        void post_backward(int iter, RenderOutput& render_output) override;
        bool is_refining(int iter) const override;
        void step(int iter) override;

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
        const char* strategy_type() const override { return "mcmc"; }

        // Reserve optimizer capacity for future growth (e.g., after checkpoint load)
        void reserve_optimizer_capacity(size_t capacity) override;

        // Exposed for testing
        int add_new_gs_test() { return add_new_gs(); }
        int add_new_gs_with_indices_test(const lfs::core::Tensor& sampled_idxs);
        int relocate_gs_test() { return relocate_gs(); }

    private:
        // Helper functions
        lfs::core::Tensor multinomial_sample(const lfs::core::Tensor& weights, int n, bool replacement = true);
        int relocate_gs();
        int add_new_gs();
        void inject_noise();
        void update_optimizer_for_relocate(const lfs::core::Tensor& sampled_indices,
                                           const lfs::core::Tensor& dead_indices,
                                           ParamType param_type);
        lfs::core::Tensor get_sampling_weights() const;
        void ensure_densification_info_shape();

        // Member variables
        std::unique_ptr<AdamOptimizer> _optimizer;
        std::unique_ptr<ExponentialLR> _scheduler;
        lfs::core::SplatData* _splat_data = nullptr; // Scene-owned
        std::unique_ptr<const lfs::core::param::OptimizationParameters> _params;

        static constexpr float NOISE_LR = 5e5f;

        // State variables
        lfs::core::Tensor _binoms;       // [n_max, n_max] binomial coefficients
        lfs::core::Tensor _noise_buffer; // Reusable buffer for noise injection
        lfs::core::Tensor _ones_int32;   // [max_cap] cached ones for ratio counting
        lfs::core::Tensor _error_score_max;
        int _error_score_windows = 0;
    };

} // namespace lfs::training
