/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lfs/training/ops/types.hpp"

namespace lfs::gpu_ops {

    // SH-rest rows use their own packed-storage path.
    struct RefineInputs {
        In means, rotations, scales, sh0, opacity;
    };

    struct RefineOutputs {
        Out means, rotations, scales, sh0, opacity;
    };

    struct RefineOps {
        void (*split)(const RefineOutputs& parents, const RefineOutputs& children, In indices);
        void (*fill_slots)(In indices, const RefineInputs& source, const RefineOutputs& destination, Out free_mask);
        void (*counts)(In bool0, In bool1, In float0, In float1, Out counts4);
        void (*normalize_positive_median)(Out values);
        void (*clip_scales)(Out log_scales, In shares, In frozen, float limit);
        void (*oversize_scores)(In error, In shares, In frozen, Out scores, float limit);
        void (*dead_mask)(In opacity, In rotations, Out mask, float minimum_opacity);
        void (*rotation_mask)(In rotations, Out mask);
    };

} // namespace lfs::gpu_ops
