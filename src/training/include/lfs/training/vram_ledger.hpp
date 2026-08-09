/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

/**
 * @file vram_ledger.hpp
 * @brief Training-state bytes-per-splat ledger.
 *
 * Computes the persistent training footprint buckets from SplatData + Adam
 * state. For SH degree 3:
 *   params ≈ 248 B/splat (SH3, fp32, swizzled shN)
 *   optimizer ≈ 152 B/splat (joint moment codec at large N)
 *   densify-aux ≈ 8 B/splat (densification_info [2,N])
 *   grads-or-helpers = 0 on the fused FastGS path (no persistent world grads)
 */

#include "diagnostics/vram_profiler.hpp"

#include <cstddef>

namespace lfs::core {
    class SplatData;
}

namespace lfs::training {

    class AdamOptimizer;

    /// Sum logical device bytes for params / Adam / densify aux / live grads.
    [[nodiscard]] diagnostics::TrainingStateLedger
    compute_training_state_ledger(const core::SplatData& splat,
                                  const AdamOptimizer* optimizer = nullptr);

    /// Capacity-backed footprint (row capacity × trailing dims) for peak attribution.
    [[nodiscard]] std::size_t
    compute_training_state_reserved_bytes(const core::SplatData& splat,
                                          const AdamOptimizer* optimizer = nullptr);

    /// Convenience: compute + publish into the process-wide VramProfiler.
    void publish_training_state_ledger(const core::SplatData& splat,
                                       const AdamOptimizer* optimizer = nullptr);

} // namespace lfs::training
