/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lfs/training/ops/popspa.hpp"

namespace lfs::training {
    // Single-source kernels for Metal and Vulkan, with Float32Pair scores.
    const gpu_ops::PopspaOps& portable_popspa_ops();
    // Adds one view's Float32 [N] removal energy into Float32Pair scores [N, 8].
    void accumulate_pop_view_scores(gpu_ops::Out scores, gpu_ops::In view);
} // namespace lfs::training
