/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"
#include "core/tensor.hpp"

#include <optional>

namespace lfs::core {
    // Rows of splat simplification: Float32 means [N,3], activated scales [N,3], unit rotations [N,4]
    // (w, x, y, z), opacities as alpha [N] and appearance [N,A].
    struct SimplifyRows {
        Tensor means;
        Tensor scales;
        Tensor rotation;
        Tensor opacity;
        Tensor appearance;
    };

    // One row per group, merged by moment matching as splat simplification merges a voxel group: members
    // are Int32 row indices, ascending within each group, and Int32 [G+1] offsets delimit the groups. A
    // group of one is copied. Rows and groups live on one GPU backend; nullopt when that backend has no
    // kernel for it.
    LFS_CORE_API std::optional<SimplifyRows> simplify_merge_groups(const SimplifyRows& rows, const Tensor& offsets,
                                                                   const Tensor& members);
} // namespace lfs::core
