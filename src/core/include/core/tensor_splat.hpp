/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/export.hpp"
#include "core/splat_transform_math.hpp"
#include "core/tensor_fwd.hpp"
namespace lfs::core {
    // Transform covariance axes by a row-major affine linear part. Inputs are
    // Float32 [N,3] log scales and [N,4] wxyz quaternions. Outputs must be
    // contiguous and distinct; input/output aliasing is supported.
    LFS_CORE_API void affine_splat_geometry(const splat_transform::LinearTransform& linear,
                                            const Tensor& scales, const Tensor& rotations,
                                            Tensor& output_scales, Tensor& output_rotations);
} // namespace lfs::core
