/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "Common.h"
#include <cstdint>
#include <cuda_runtime.h>

namespace gsplat_lfs {

    /////////////////////////////////////////////////
    // rasterize_to_pixels_from_world_3dgs - Forward
    /////////////////////////////////////////////////

    template <uint32_t CDIM>
    void launch_rasterize_to_pixels_from_world_3dgs_fwd_kernel(
        // Gaussian parameters
        const float* means,       // [N, 3]
        const float* quats,       // [N, 4]
        const float* scales,      // [N, 3]
        const float* colors,      // [C, N, CDIM]
        const float* opacities,   // [C, N]
        const float* backgrounds, // [C, CDIM] optional (can be nullptr)
        const bool* masks,        // [C, tile_height, tile_width] optional
        // dimensions
        uint32_t C,
        uint32_t N,
        uint32_t n_isects,
        uint32_t image_width,
        uint32_t image_height,
        uint32_t tile_size,
        // camera
        const float* viewmats0, // [C, 4, 4]
        const float* viewmats1, // [C, 4, 4] optional
        const float* Ks,        // [C, 3, 3]
        CameraModelType camera_model,
        const UnscentedTransformParameters& ut_params,
        ShutterType rs_type,
        const float* radial_coeffs,     // optional
        const float* tangential_coeffs, // optional
        const float* thin_prism_coeffs, // optional
        // intersections
        const int32_t* tile_offsets, // [C, tile_height, tile_width]
        const int32_t* flatten_ids,  // [n_isects]
        // outputs (pre-allocated)
        float* renders,    // [C, image_height, image_width, CDIM]
        float* alphas,     // [C, image_height, image_width, 1]
        int32_t* last_ids, // [C, image_height, image_width]
        cudaStream_t stream = nullptr);

    /////////////////////////////////////////////////
    // rasterize_to_pixels_from_world_3dgs - Backward
    /////////////////////////////////////////////////

    template <uint32_t CDIM>
    void launch_rasterize_to_pixels_from_world_3dgs_bwd_kernel(
        // Gaussian parameters
        const float* means,       // [N, 3]
        const float* quats,       // [N, 4]
        const float* scales,      // [N, 3]
        const float* colors,      // [C, N, CDIM]
        const float* opacities,   // [C, N]
        const float* backgrounds, // [C, CDIM] optional
        const bool* masks,        // optional
        // dimensions
        uint32_t C,
        uint32_t N,
        uint32_t n_isects,
        uint32_t image_width,
        uint32_t image_height,
        uint32_t tile_size,
        // camera
        const float* viewmats0, // [C, 4, 4]
        const float* viewmats1, // [C, 4, 4] optional
        const float* Ks,        // [C, 3, 3]
        CameraModelType camera_model,
        const UnscentedTransformParameters& ut_params,
        ShutterType rs_type,
        const float* radial_coeffs,
        const float* tangential_coeffs,
        const float* thin_prism_coeffs,
        // intersections
        const int32_t* tile_offsets, // [C, tile_height, tile_width]
        const int32_t* flatten_ids,  // [n_isects]
        // forward outputs
        const float* render_alphas, // [C, image_height, image_width, 1]
        const int32_t* last_ids,    // [C, image_height, image_width]
        // gradients of outputs
        const float* v_render_colors, // [C, image_height, image_width, CDIM]
        const float* v_render_alphas, // [C, image_height, image_width, 1]
        // gradient outputs (pre-allocated, atomic accumulation)
        float* v_means,     // [N, 3]
        float* v_quats,     // [N, 4]
        float* v_scales,    // [N, 3]
        float* v_colors,    // [C, N, CDIM]
        float* v_opacities, // [C, N]
        cudaStream_t stream = nullptr);

    /////////////////////////////////////////////////
    // rasterize_to_indices_3dgs
    /////////////////////////////////////////////////

    void launch_rasterize_to_indices_3dgs_kernel(
        uint32_t range_start,
        uint32_t range_end,
        const float* transmittances, // [C, image_height, image_width]
        // Gaussian parameters
        const float* means2d,   // [C, N, 2]
        const float* conics,    // [C, N, 3]
        const float* opacities, // [C, N]
        // dimensions
        uint32_t C,
        uint32_t N,
        uint32_t image_width,
        uint32_t image_height,
        uint32_t tile_size,
        // intersections
        const int32_t* tile_offsets, // [C, tile_height, tile_width]
        const int32_t* flatten_ids,  // [n_isects]
        // helper for double pass
        const int32_t* chunk_starts, // [C, image_height, image_width] optional
        // outputs
        int32_t* chunk_cnts,   // [C, image_height, image_width] optional
        int32_t* gaussian_ids, // [n_elems] optional
        int32_t* pixel_ids,    // [n_elems] optional
        cudaStream_t stream = nullptr);

} // namespace gsplat_lfs
