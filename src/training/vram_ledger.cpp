/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/vram_ledger.hpp"

#include "core/cuda/memory_arena.hpp"
#include "core/splat_data.hpp"
#include "training/optimizer/adam_optimizer.hpp"

namespace lfs::training {
    /// Capacity-backed footprint (row capacity × trailing dims × dtype).
    [[nodiscard]] std::size_t tensor_reserved_bytes(const lfs::core::Tensor& tensor) {
        if (!tensor.is_valid()) {
            return 0;
        }
        if (tensor.capacity() == 0 || tensor.ndim() == 0) {
            return tensor.bytes();
        }
        std::size_t row_elems = 1;
        if (tensor.ndim() > 1) {
            for (std::size_t dim = 1; dim < tensor.ndim(); ++dim) {
                row_elems *= tensor.shape()[dim];
            }
        }
        return tensor.capacity() * row_elems * lfs::core::dtype_size(tensor.dtype());
    }

    void record_vram_current(std::string_view scope,
                             std::string_view label,
                             const size_t bytes,
                             const bool publish_zero,
                             const lfs::diagnostics::VramAllocationMethod method) {
        if (bytes == 0 && !publish_zero) {
            return;
        }
        lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(scope, label, bytes, method);
    }

    void record_vram_tensor(std::string_view scope,
                            std::string_view label,
                            const lfs::core::Tensor& tensor) {
        // The zeros_direct storage total backs Direct tensor disclosures.
        // Other external tensors retain their external provenance.
        const auto method = tensor.external_storage_kind() == "cuda.direct"
                                ? lfs::diagnostics::VramAllocationMethod::Direct
                            : tensor.is_external_storage()
                                ? lfs::diagnostics::VramAllocationMethod::External
                                : lfs::diagnostics::VramAllocationMethod::Unknown;
        record_vram_current(scope, label, tensor_reserved_bytes(tensor), false, method);
    }

    void record_rasterizer_arena_disclosure(std::string_view scope) {
        auto* arena = lfs::core::GlobalArenaManager::instance().try_get_arena();
        if (!arena) {
            return;
        }
        const auto info = arena->get_memory_info();
        record_vram_current(scope, "arena.capacity", info.arena_capacity);
        record_vram_current(scope, "arena.current_usage", info.current_usage);
        record_vram_current(scope, "arena.peak_usage", info.peak_usage);
        auto& profiler = lfs::diagnostics::VramProfiler::instance();
        profiler.setGauge("vram.audit.rasterizer_arena.required_bytes",
                          static_cast<double>(info.required_bytes));
        profiler.setGauge("vram.audit.rasterizer_arena.allocated_bytes",
                          static_cast<double>(info.arena_capacity));
        profiler.setGauge("vram.audit.rasterizer_arena.current_usage_bytes",
                          static_cast<double>(info.current_usage));
    }

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
        // float2 bounds per 256-splat block (≪1 B/splat large-N).
        ledger.params_bytes += tensor_logical_bytes(splat.shN_value_bounds());
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
                ledger.optimizer_bytes += tensor_logical_bytes(state->joint_bounds);
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

    std::size_t compute_training_state_reserved_bytes(const core::SplatData& splat,
                                                      const AdamOptimizer* optimizer) {
        std::size_t bytes = 0;
        bytes += tensor_reserved_bytes(splat.means());
        bytes += tensor_reserved_bytes(splat.sh0());
        bytes += tensor_reserved_bytes(splat.shN());
        bytes += tensor_reserved_bytes(splat.shN_value_bounds());
        bytes += tensor_reserved_bytes(splat.scaling_raw());
        bytes += tensor_reserved_bytes(splat.rotation_raw());
        bytes += tensor_reserved_bytes(splat.opacity_raw());
        bytes += tensor_reserved_bytes(splat._densification_info);
        bytes += tensor_reserved_bytes(splat.deleted());
        if (optimizer != nullptr) {
            for (const auto type : AdamOptimizer::all_param_types()) {
                const auto* state = optimizer->get_state(type);
                if (state == nullptr) {
                    continue;
                }
                bytes += tensor_reserved_bytes(state->exp_avg);
                bytes += tensor_reserved_bytes(state->joint_bounds);
                bytes += tensor_reserved_bytes(state->grad);
            }
        }
        return bytes;
    }

    void publish_training_state_ledger(const core::SplatData& splat,
                                       const AdamOptimizer* optimizer) {
        auto& profiler = diagnostics::VramProfiler::instance();
        if (!profiler.enabled()) {
            return;
        }
        const auto ledger = compute_training_state_ledger(splat, optimizer);
        const auto reserved = compute_training_state_reserved_bytes(splat, optimizer);
        profiler.setTrainingStateLedger(ledger);
        profiler.setGauge("vram.audit.training_state.required_bytes",
                          static_cast<double>(ledger.total_bytes));
        profiler.setGauge("vram.audit.training_state.allocated_bytes",
                          static_cast<double>(reserved));
    }

} // namespace lfs::training
