/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lfs/training/ops/types.hpp"

#include <cstdint>

namespace lfs::gpu_ops {

    enum class SampleDomain { AliveIndices,
                              All };

    // SH-rest rows use their own packed-storage path.
    struct McmcRows {
        Out means, sh0, raw_scales, raw_quats, raw_opacity;
    };

    struct McmcOps {
        void (*initialize)(int n_max);
        void (*relocate)(In opacity, In scales, In ratios,
                         Out new_opacity, Out new_scales, float min_opacity);
        void (*noise)(In raw_opacity, In raw_scales, In raw_quats, In frozen,
                      Out means, uint64_t seed, float lr);
        void (*copy_rows)(In source_indices, In destination_indices, const McmcRows&);
        // Both input and output scales are logarithmic, as in the live caller.
        void (*update_rows)(In indices, In raw_scales, In raw_opacity,
                            Out raw_scales_out, Out raw_opacity_out);
        void (*sample)(In weights, In opacity, In raw_scales, In alive_indices,
                       Out indices, Out sampled_opacity, Out sampled_scales,
                       SampleDomain, uint64_t seed);
        // Fold row 1 into the max and zero both densification rows.
        void (*fold_error)(Out error_max, Out densification);
    };

} // namespace lfs::gpu_ops
