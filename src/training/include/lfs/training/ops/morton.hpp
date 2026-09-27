/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lfs/training/ops/adam.hpp"

namespace lfs::gpu_ops {

    struct MortonOps {
        Tensor (*permutation)(In means);
        void (*permute_joint)(In packed, In bounds, In permutation,
                              Out new_packed, Out new_bounds, const JointCodecParams&);
        // Re-encode swizzled SH moments in place through caller-owned scratch.
        // The caller installs new_bounds after all groups have been copied back.
        void (*permute_joint_grouped)(Out packed, In bounds, In permutation,
                                      Out new_bounds, Out scratch, const JointCodecParams&);
    };

} // namespace lfs::gpu_ops
