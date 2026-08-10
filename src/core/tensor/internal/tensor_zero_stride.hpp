/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

/**
 * @file tensor_zero_stride.hpp
 * @brief Correctness firewall for zero-stride expand / broadcast views.
 *
 * Expand / broadcast_to may return views with stride-0 on broadcast dims.
 * Only consumers on the allowlist may touch such views without an explicit
 * materialization barrier. Every other op MUST call contiguous_read /
 * contiguous() at its boundary so strided kernels never
 * see stride-0 storage.
 */

#include <cstddef>

namespace lfs::core::zero_stride {

    /// Per-op allowlist: only these kinds may consume zero-stride views without
    /// an extra materialize step *beyond* the existing TensorLeaf/contiguous_read
    /// firewall (elementwise goes through that firewall and is therefore safe).
    enum class ConsumerKind : int {
        Contiguous = 0,      // contiguous() materializes — safe
        Clone,               // clone of non-contig materializes — safe
        ElementwiseFirewall, // contiguous_read / TensorLeaf — safe
        BroadcastBinary,     // shape-indexed broadcast kernels — safe
        Reduce,              // reduce forces contiguous at entry — safe
        // --- non-allowlisted (must materialize at boundary) ---
        Cat,
        MaskedSelect,
        InPlaceMutate, // zero-stride in-place is rejected (throws)
    };

    /// True iff @p kind is verified safe to receive a zero-stride view
    /// (either because it materializes via the firewall, or is shape-indexed).
    [[nodiscard]] inline bool is_allowlisted(ConsumerKind kind) noexcept {
        switch (kind) {
        case ConsumerKind::Contiguous:
        case ConsumerKind::Clone:
        case ConsumerKind::ElementwiseFirewall:
        case ConsumerKind::BroadcastBinary:
        case ConsumerKind::Reduce:
            return true;
        default:
            return false;
        }
    }

} // namespace lfs::core::zero_stride
