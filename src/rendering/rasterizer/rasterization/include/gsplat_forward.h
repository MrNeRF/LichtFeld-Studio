/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::rendering {

    enum class GutCameraModel : int {
        PINHOLE = 0,
        ORTHO = 1,
        FISHEYE = 2,
        EQUIRECTANGULAR = 3
    };

    enum class GutRenderMode : int {
        RGB = 0,
        DEPTH = 1,
        EXPECTED_DEPTH = 2,
        RGB_DEPTH = 3,
        RGB_EXPECTED_DEPTH = 4
    };

    // Forward-only GUT rasterization (no backward pass, uses memory arena)
    void gsplat_forward_gut(
        const float* means,     // [N, 3]
        const float* quats,     // [N, 4] normalized
        const float* scales,    // [N, 3] activated
        const float* opacities, // [N] activated
        const float* sh_coeffs, // [N, K, 3]
        uint32_t sh_degree,
        uint32_t N,
        uint32_t K,
        uint32_t image_width,
        uint32_t image_height,
        const float* viewmat,      // [4, 4]
        const float* K_intrinsics, // [3, 3]
        GutCameraModel camera_model,
        const float* radial_coeffs,
        const float* tangential_coeffs,
        const float* background, // [3] or nullptr
        GutRenderMode render_mode,
        float scaling_modifier,
        float* render_colors_out, // [H, W, C]
        float* render_alphas_out, // [H, W]
        float* render_depth_out,  // [H, W] or nullptr
        cudaStream_t stream = nullptr);

} // namespace lfs::rendering
