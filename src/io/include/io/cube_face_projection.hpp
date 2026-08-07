/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/cube_face_projection.hpp"
#include "core/export.hpp"

#include <cstdint>
#include <vector>

namespace lfs::io {

    [[nodiscard]] LFS_IO_API int projected_face_output_size(
        const lfs::core::CubeFaceProjection& projection,
        int resize_factor,
        int max_width);

    [[nodiscard]] LFS_IO_API std::vector<uint8_t> project_cube_face_hwc(
        const uint8_t* source,
        int width,
        int height,
        int channels,
        const lfs::core::CubeFaceProjection& projection,
        int output_size);

    // Projects an equirectangular depth panorama onto one cube face. Returns
    // normalised samples in [0, 1], matching the 1/255 and 1/65535 scaling the
    // grayscale upload kernels apply.
    //
    // Two things differ from the colour path, both of which matter for
    // supervision:
    //
    //  - Nearest sampling. Averaging across a depth discontinuity invents a
    //    surface lying in neither the foreground nor the background, and the
    //    depth loss would then be supervised towards that invented surface.
    //
    //  - Radial-to-z conversion. Equirectangular depth stores distance along the
    //    ray from the camera centre, but LFS supervises camera-space z: both the
    //    fastgs rasterizer and depth_anchor_collect_kernel take the third row of
    //    world-to-camera. For a face pixel at plane coordinates (u, v) the ray is
    //    (u, v, 1) normalised, so z = r / sqrt(1 + u^2 + v^2). Without this the
    //    corner of a 96 degree face is off by ~1.9x, and because the error varies
    //    across the image the depth loss's per-camera affine fit cannot absorb it.
    //
    // Assumes the panorama stores radial distance, the usual convention for
    // equirectangular depth. A prior storing disparity would need the reciprocal
    // correction instead.
    [[nodiscard]] LFS_IO_API std::vector<float> project_cube_face_depth(
        const uint8_t* source,
        bool source_is_16bit,
        int width,
        int height,
        const lfs::core::CubeFaceProjection& projection,
        int output_size);

} // namespace lfs::io
