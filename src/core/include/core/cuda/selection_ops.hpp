/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/tensor.hpp"

namespace lfs::core::cuda {

    /// Grow selection by radius using spatial hashing. O(N).
    /// @param mask     [N] UInt8 selection mask (0 = unselected, >0 = group ID)
    /// @param means    [N, 3] Float32 gaussian positions
    /// @param radius   expansion radius in scene units
    /// @param group_id group ID to assign to newly selected gaussians
    /// @return new mask with expanded selection
    LFS_CUDA_API Tensor selection_grow(const Tensor& mask, const Tensor& means, float radius, uint8_t group_id);

    /// Shrink selection by radius using spatial hashing. O(N).
    /// @param mask     [N] UInt8 selection mask
    /// @param means    [N, 3] Float32 gaussian positions
    /// @param radius   erosion radius in scene units
    /// @return new mask with contracted selection
    LFS_CUDA_API Tensor selection_shrink(const Tensor& mask, const Tensor& means, float radius);

    /// Select gaussians by activated opacity range.
    /// @param opacity_raw  [N] Float32 raw opacity (pre-sigmoid)
    /// @param min_opacity  minimum activated opacity (inclusive)
    /// @param max_opacity  maximum activated opacity (inclusive)
    /// @param group_id     group ID to assign
    /// @return [N] UInt8 mask
    LFS_CUDA_API Tensor select_by_opacity(const Tensor& opacity_raw, float min_opacity, float max_opacity, uint8_t group_id);

    /// Select gaussians by max activated scale.
    /// @param scale_raw  [N, 3] Float32 raw scale (pre-exp)
    /// @param max_scale  maximum activated scale threshold
    /// @param group_id   group ID to assign
    /// @return [N] UInt8 mask
    LFS_CUDA_API Tensor select_by_scale(const Tensor& scale_raw, float max_scale, uint8_t group_id);

} // namespace lfs::core::cuda
