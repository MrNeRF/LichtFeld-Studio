/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/vram_ledger.hpp"

#include "core/splat_data.hpp"
#include "training/optimizer/adam_optimizer.hpp"

namespace lfs::training {
    namespace {

        [[nodiscard]] std::size_t tensor_logical_bytes(const lfs::core::Tensor& tensor) {
            if (!tensor.is_valid() || tensor.numel() == 0) {
                return 0;
            }
            return tensor.bytes();
        }

    } // namespace

    diagnostics::TrainingStateLedger
    compute_training_state_ledger(const core::SplatData& splat,
                                  const AdamOptimizer* optimizer) {
        diagnostics::TrainingStateLedger ledger;
        ledger.live_splats = static_cast<std::size_t>(splat.size());

        // --- params (geometry + SH) ---
        ledger.params_bytes += tensor_logical_bytes(splat.means());
        ledger.params_bytes += tensor_logical_bytes(splat.sh0());
        ledger.params_bytes += tensor_logical_bytes(splat.shN());
        ledger.params_bytes += tensor_logical_bytes(splat.scaling_raw());
        ledger.params_bytes += tensor_logical_bytes(splat.rotation_raw());
        ledger.params_bytes += tensor_logical_bytes(splat.opacity_raw());

        // --- densify aux ---
        ledger.densify_aux_bytes += tensor_logical_bytes(splat._densification_info);
        // Soft-delete mask is optional; count when present (≈1 B/splat).
        ledger.densify_aux_bytes += tensor_logical_bytes(splat.deleted());

        // --- optimizer moments + any materialised gradients ---
        if (optimizer != nullptr) {
            for (const auto type : AdamOptimizer::all_param_types()) {
                const auto* state = optimizer->get_state(type);
                if (state == nullptr) {
                    continue;
                }
                ledger.optimizer_bytes += tensor_logical_bytes(state->exp_avg);
                ledger.optimizer_bytes += tensor_logical_bytes(state->exp_avg_sq);
                ledger.optimizer_bytes += tensor_logical_bytes(state->exp_avg_scale);
                ledger.optimizer_bytes += tensor_logical_bytes(state->exp_avg_sq_scale);
                // Transient world grads (non-fused paths only). Fused FastGS keeps
                // these empty — matches the "0 persistent world grads" claim.
                ledger.gradients_or_helpers_bytes += tensor_logical_bytes(state->grad);
            }
        }

        ledger.total_bytes = ledger.params_bytes + ledger.optimizer_bytes +
                             ledger.gradients_or_helpers_bytes + ledger.densify_aux_bytes;
        if (ledger.live_splats > 0) {
            ledger.bytes_per_splat =
                static_cast<double>(ledger.total_bytes) /
                static_cast<double>(ledger.live_splats);
        }
        return ledger;
    }

    void publish_training_state_ledger(const core::SplatData& splat,
                                       const AdamOptimizer* optimizer) {
        auto& profiler = diagnostics::VramProfiler::instance();
        if (!profiler.enabled()) {
            return;
        }
        profiler.setTrainingStateLedger(compute_training_state_ledger(splat, optimizer));
    }

} // namespace lfs::training
