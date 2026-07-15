/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <istream>
#include <ostream>
#include <vector>

namespace lfs::training {

    class AdamOptimizer;  // Forward declaration
    enum class ParamType; // Forward declaration

    /**
     * Simple Exponential Learning Rate Scheduler
     *
     * Multiplies the learning rate by gamma at each step:
     *   lr_new = lr_current * gamma
     *
     * Controls which learning rates to update:
     * - Empty vector (default): Updates ONLY global LR (for means in MCMC)
     * - Specific params: Updates only those parameter LRs
     * - All params: Pass all_param_types() to update everything
     *
     * Example (MCMC - only global/means LR decays):
     *   AdamOptimizer optimizer(...);
     *   ExponentialLR scheduler(optimizer, 0.99);  // Only global LR
     *
     * Example (update specific params):
     *   ExponentialLR scheduler(optimizer, 0.99, {ParamType::Means, ParamType::Sh0});
     *
     * Example (update all params):
     *   ExponentialLR scheduler(optimizer, 0.99, AdamOptimizer::all_param_types());
     *
     *   for (int iter = 0; iter < 1000; iter++) {
     *       optimizer.step(iter);
     *       scheduler.step();  // Update learning rate
     *   }
     */
    class ExponentialLR {
    public:
        ExponentialLR(AdamOptimizer& optimizer, double gamma,
                      std::vector<ParamType> params_to_update = {})
            : optimizer_(optimizer),
              gamma_(gamma),
              params_to_update_(params_to_update) {
        }

        void step();

        // Serialization for checkpoints
        void serialize(std::ostream& os) const;
        void deserialize(std::istream& is);
        void adopt_checkpoint_state(ExponentialLR& loaded) noexcept;

        // Accessors for state
        double get_gamma() const { return gamma_; }

    private:
        AdamOptimizer& optimizer_;
        double gamma_;
        std::vector<ParamType> params_to_update_; // Empty = only global LR
    };

    /**
     * Exponential Learning Rate Scheduler with Linear Warmup
     *
     * Phase 1 (Warmup): Linearly increase LR from (initial_lr * warmup_start_factor) to initial_lr
     *   lr = initial_lr * (warmup_start_factor + (1 - warmup_start_factor) * progress)
     *   where progress = current_step / warmup_steps
     *
     * Phase 2 (Decay): Exponentially decay LR
     *   lr = initial_lr * gamma^(current_step - warmup_steps)
     *
     * Controls which learning rates to update (same as ExponentialLR):
     * - Empty vector (default): Updates ONLY global LR
     * - Specific params: Updates only those parameter LRs
     * - All params: Pass all_param_types() to update everything
     *
     * Example:
     *   AdamOptimizer optimizer(...);
     *   WarmupExponentialLR scheduler(optimizer,
     *                                  gamma=0.995,           // Exponential decay rate
     *                                  warmup_steps=100,      // 100 steps warmup
     *                                  warmup_start_factor=0.1,  // Start at 10% of initial LR
     *                                  params_to_update={});  // Only global LR
     *
     *   for (int iter = 0; iter < 1000; iter++) {
     *       optimizer.step(iter);
     *       scheduler.step();  // Update learning rate
     *   }
     */
    class WarmupExponentialLR {
    public:
        WarmupExponentialLR(
            AdamOptimizer& optimizer,
            double gamma,
            int warmup_steps = 0,
            double warmup_start_factor = 1.0,
            std::vector<ParamType> params_to_update = {});

        void step();

        // Get current step count
        int get_step() const { return current_step_; }

        // Serialization for checkpoints
        void serialize(std::ostream& os) const;
        void deserialize(std::istream& is);
        void adopt_checkpoint_state(WarmupExponentialLR& loaded) noexcept;

        // Accessors for state
        double get_gamma() const { return gamma_; }
        int get_warmup_steps() const { return warmup_steps_; }
        double get_warmup_start_factor() const { return warmup_start_factor_; }
        double get_initial_lr() const { return initial_lr_; }

    private:
        AdamOptimizer& optimizer_;
        double gamma_;
        int warmup_steps_;
        double warmup_start_factor_;
        int current_step_;
        double initial_lr_;
        std::vector<ParamType> params_to_update_; // Empty = only global LR
    };

} // namespace lfs::training
