/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Forward-only gsplat operations for viewer (no backward pass)

#pragma once

#include "Common.h"
#include <cstdint>
#include <cuda_runtime.h>

namespace gsplat_fwd {

    //=========================================================================
    // Spherical Harmonics - Forward Only
    //=========================================================================

    void spherical_harmonics_fwd(
        uint32_t degrees_to_use,
        const float* dirs,              // [M, 3] viewing directions
        const float* coeffs,            // [N_total, K, 3] SH coefficients (N-sized when using visible_indices)
        const bool* masks,              // [M] optional (can be nullptr)
        const int32_t* visible_indices, // [M] maps elem_id -> global_idx, nullptr = direct
        int64_t total_elements,         // M (visible gaussians)
        int32_t K,                      // number of SH coefficients
        float* colors,                  // [M, 3] output (pre-allocated)
        cudaStream_t stream = nullptr);

    //=========================================================================
    // Tile Intersection
    //=========================================================================

    struct IntersectTileResult {
        int32_t* tiles_per_gauss; // [C, N] - output buffer provided by caller
        int64_t* isect_ids;       // [n_isects] - allocated internally
        int32_t* flatten_ids;     // [n_isects] - allocated internally
        int32_t n_isects;         // Total number of intersections
    };

    // Note: isect_ids and flatten_ids are allocated internally
    // Caller must free them with cudaFree when done
    IntersectTileResult intersect_tile(
        const float* means2d,        // [C, N, 2]
        const int32_t* radii,        // [C, N, 2]
        const float* depths,         // [C, N]
        const int32_t* camera_ids,   // [nnz] optional (nullptr for dense)
        const int32_t* gaussian_ids, // [nnz] optional (nullptr for dense)
        uint32_t C,
        uint32_t N,
        uint32_t tile_size,
        uint32_t tile_width,
        uint32_t tile_height,
        bool sort,
        int32_t* tiles_per_gauss_out, // [C, N] pre-allocated output
        cudaStream_t stream = nullptr);

    void intersect_offset(
        const int64_t* isect_ids, // [n_isects]
        int32_t n_isects,
        uint32_t C,
        uint32_t tile_width,
        uint32_t tile_height,
        int32_t* isect_offsets, // [C, tile_height, tile_width] output
        cudaStream_t stream = nullptr);

    //=========================================================================
    // Quaternion to Rotation Matrix
    //=========================================================================

    void quats_to_rotmats(
        const float* quats, // [N, 4]
        int64_t N,
        float* rotmats, // [N, 3, 3] output
        cudaStream_t stream = nullptr);

    //=========================================================================
    // View Direction Computation for SH
    //=========================================================================

    void compute_view_dirs(
        const float* means,    // [N_total, 3]
        const float* viewmats, // [C, 4, 4]
        uint32_t C,
        uint32_t N_total,           // Total gaussians in input arrays
        uint32_t M,                 // Visible gaussians to process
        const int* visible_indices, // [M] maps output idx → global gaussian idx, nullptr = all visible
        float* dirs,                // [C, M, 3] output
        cudaStream_t stream = nullptr);

    //=========================================================================
    // Projection - Unscented Transform for 3DGS
    //=========================================================================

    void projection_ut_3dgs_fused(
        // inputs
        const float* means,     // [N_total, 3]
        const float* quats,     // [N_total, 4]
        const float* scales,    // [N_total, 3]
        const float* opacities, // [N_total] optional (can be nullptr)
        const float* viewmats0, // [C, 4, 4]
        const float* viewmats1, // [C, 4, 4] optional for rolling shutter
        const float* Ks,        // [C, 3, 3]
        uint32_t N_total,       // Total gaussians in input arrays
        uint32_t M,             // Visible gaussians to process (M <= N_total)
        uint32_t C,
        uint32_t image_width,
        uint32_t image_height,
        float eps2d,
        float near_plane,
        float far_plane,
        float radius_clip,
        bool calc_compensations,
        CameraModelType camera_model,
        const UnscentedTransformParameters& ut_params,
        ShutterType rs_type,
        const float* radial_coeffs,       // [C, 6/4] optional
        const float* tangential_coeffs,   // [C, 2] optional
        const float* thin_prism_coeffs,   // [C, 2] optional
        const int* transform_indices,     // [N_total] optional
        const bool* node_visibility_mask, // optional
        int num_visibility_nodes,
        const int* visible_indices, // [M] maps output idx → global gaussian idx, nullptr = all visible
        // outputs (sized to [C, M, ...])
        int32_t* radii,       // [C, M, 2]
        float* means2d,       // [C, M, 2]
        float* depths,        // [C, M]
        float* conics,        // [C, M, 3]
        float* compensations, // [C, M] optional
        cudaStream_t stream = nullptr);

    //=========================================================================
    // Rasterization - Forward Only
    //=========================================================================

    void rasterize_to_pixels_from_world_3dgs_fwd(
        // Gaussian parameters (N_total-sized, use visible_indices for access)
        const float* means,       // [N_total, 3]
        const float* quats,       // [N_total, 4]
        const float* scales,      // [N_total, 3]
        const float* colors,      // [C, M, channels] (M-sized from SH)
        const float* opacities,   // [N_total]
        const float* backgrounds, // [C, channels] (can be nullptr)
        const bool* masks,        // [C, tile_height, tile_width] (can be nullptr)
        const float* depths,      // [C, M] per-gaussian depths (M-sized from projection)
        // dimensions
        uint32_t C,
        uint32_t N,
        uint32_t n_isects,
        uint32_t channels,
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
        // indirect indexing for visibility filtering
        const int32_t* visible_indices, // [M] maps g -> global gaussian idx, nullptr = direct
        // outputs (pre-allocated)
        float* renders,       // [C, image_height, image_width, channels]
        float* alphas,        // [C, image_height, image_width, 1]
        int32_t* last_ids,    // [C, image_height, image_width]
        float* median_depths, // [C, image_height, image_width] (can be nullptr)
        cudaStream_t stream = nullptr);

    //=========================================================================
    // High-level API: Fully fused rasterization with SH evaluation
    //=========================================================================

    struct RasterizeWithSHResult {
        // Caller must pre-allocate these buffers:
        float* render_colors;     // [C, H, W, channels]
        float* render_alphas;     // [C, H, W, 1]
        int32_t* radii;           // [C, N, 2]
        float* means2d;           // [C, N, 2]
        float* depths;            // [C, N]
        float* colors;            // [C, N, channels]
        float* dirs;              // [C, N, 3] viewing directions for SH
        float* conics;            // [C, N, 3] covariance matrices
        int32_t* tiles_per_gauss; // [C, N]
        int32_t* tile_offsets;    // [C, tile_height, tile_width]
        int32_t* last_ids;        // [C, H, W]
        float* median_depths;     // [C, H, W] optional (can be nullptr)
        float* compensations;     // [C, N] optional (can be nullptr)
        // These are allocated internally - caller must free with cudaFree:
        int64_t* isect_ids;   // [n_isects]
        int32_t* flatten_ids; // [n_isects]
        int32_t n_isects;
    };

    void rasterize_from_world_with_sh_fwd(
        // Gaussian parameters
        const float* means,     // [N_total, 3]
        const float* quats,     // [N_total, 4]
        const float* scales,    // [N_total, 3]
        const float* opacities, // [N_total]
        const float* sh_coeffs, // [N_total, K, 3]
        uint32_t sh_degree,
        const float* backgrounds, // [C, channels] optional
        const bool* masks,        // optional
        // dimensions
        uint32_t N_total, // Total gaussians in input arrays
        uint32_t M,       // Visible gaussians to process (M <= N_total)
        uint32_t C,
        uint32_t K, // number of SH coefficients
        uint32_t image_width,
        uint32_t image_height,
        uint32_t tile_size,
        // camera
        const float* viewmats0, // [C, 4, 4]
        const float* viewmats1, // [C, 4, 4] optional
        const float* Ks,        // [C, 3, 3]
        CameraModelType camera_model,
        // settings
        float eps2d,
        float near_plane,
        float far_plane,
        float radius_clip,
        float scaling_modifier,
        bool calc_compensations,
        int render_mode, // 0=RGB, 1=D, 2=ED, 3=RGB_D, 4=RGB_ED
        const UnscentedTransformParameters& ut_params,
        ShutterType rs_type,
        const float* radial_coeffs,     // optional
        const float* tangential_coeffs, // optional
        const float* thin_prism_coeffs, // optional
        // node visibility culling
        const int* transform_indices,     // [N_total] optional (can be nullptr)
        const bool* node_visibility_mask, // [num_visibility_nodes] optional (can be nullptr)
        int num_visibility_nodes,
        const int* visible_indices, // [M] maps output idx → global gaussian idx, nullptr = all visible
        // outputs (result struct with pre-allocated buffers)
        RasterizeWithSHResult& result,
        cudaStream_t stream = nullptr);

} // namespace gsplat_fwd
