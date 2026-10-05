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

    private:
        AdamOptimizer& optimizer_;
        double gamma_;
        std::vector<ParamType> params_to_update_; // Empty = only global LR
    };

} // namespace lfs::training
