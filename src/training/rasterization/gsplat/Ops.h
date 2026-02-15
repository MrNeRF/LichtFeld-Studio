/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "Common.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <tuple>

namespace gsplat_lfs {

    //=========================================================================
    // Spherical Harmonics
    //=========================================================================

    void spherical_harmonics_fwd(
        uint32_t degrees_to_use,
        const float* dirs,      // [..., 3] flattened
        const float* coeffs,    // [..., K, 3] flattened
        const bool* masks,      // [...] optional (can be nullptr)
        int64_t total_elements, // total batch size
        int32_t K,              // number of SH coefficients
        float* colors,          // [..., 3] output (pre-allocated)
        cudaStream_t stream = nullptr);

    void spherical_harmonics_bwd(
        uint32_t K,
        uint32_t degrees_to_use,
        const float* dirs,     // [..., 3]
        const float* coeffs,   // [..., K, 3]
        const bool* masks,     // [...] optional
        const float* v_colors, // [..., 3] gradient
        int64_t total_elements,
        bool compute_v_dirs,
        float* v_coeffs, // [..., K, 3] output
        float* v_dirs,   // [..., 3] optional output
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
    // Relocation (MCMC densification)
    //=========================================================================

    void relocation(
        float* opacities,    // [N] - modified in-place
        float* scales,       // [N, 3] - modified in-place
        const float* ratios, // [N]
        const float* binoms, // [n_max, n_max]
        int64_t N,
        int32_t n_max,
        cudaStream_t stream = nullptr);

    void add_noise(
        float* raw_opacities, // [N] - modified in-place
        float* raw_scales,    // [N, 3] - modified in-place
        float* raw_quats,     // [N, 4] - modified in-place
        const float* noise,   // [N, 3]
        float* means,         // [N, 3] - modified in-place
        int64_t N,
        float current_lr,
        cudaStream_t stream = nullptr);

    //=========================================================================
    // View Direction Computation for SH
    //=========================================================================

    void compute_view_dirs(
        const float* means,    // [N, 3]
        const float* viewmats, // [C, 4, 4]
        uint32_t C,
        uint32_t N,
        float* dirs, // [C, N, 3] output
        cudaStream_t stream = nullptr);

    //=========================================================================
    // Projection - Unscented Transform for 3DGS
    //=========================================================================

    void projection_ut_3dgs_fused(
        // inputs
        const float* means,     // [N, 3]
        const float* quats,     // [N, 4]
        const float* scales,    // [N, 3]
        const float* opacities, // [N] optional (can be nullptr)
        const float* viewmats0, // [C, 4, 4]
        const float* viewmats1, // [C, 4, 4] optional for rolling shutter
        const float* Ks,        // [C, 3, 3]
        uint32_t N,
        uint32_t C,
        uint32_t image_width,
        uint32_t image_height,
        float eps2d,
        float near_plane,
        float far_plane,
        float radius_clip,
        bool calc_compensations,
        ::CameraModelType camera_model,
        const ::UnscentedTransformParameters& ut_params,
        ::ShutterType rs_type,
        const float* radial_coeffs,     // [C, 6/4] optional
        const float* tangential_coeffs, // [C, 2] optional
        const float* thin_prism_coeffs, // [C, 2] optional
        // outputs (pre-allocated)
        int32_t* radii,       // [C, N, 2]
        float* means2d,       // [C, N, 2]
        float* depths,        // [C, N]
        float* conics,        // [C, N, 3]
        float* compensations, // [C, N] optional
        cudaStream_t stream = nullptr);

    //=========================================================================
    // Rasterization - Forward
    //=========================================================================

    void rasterize_to_pixels_from_world_3dgs_fwd(
        // Gaussian parameters
        const float* means,       // [N, 3]
        const float* quats,       // [N, 4]
        const float* scales,      // [N, 3]
        const float* colors,      // [C, N, channels]
        const float* opacities,   // [C, N]
        const float* backgrounds, // [C, channels] (can be nullptr) - solid color
        const float* bg_images,   // [C, channels, H, W] (can be nullptr) - per-pixel background
        const bool* masks,        // [C, tile_height, tile_width] (can be nullptr)
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
        ::CameraModelType camera_model,
        const ::UnscentedTransformParameters& ut_params,
        ::ShutterType rs_type,
        const float* radial_coeffs,     // optional
        const float* tangential_coeffs, // optional
        const float* thin_prism_coeffs, // optional
        // intersections
        const int32_t* tile_offsets, // [C, tile_height, tile_width]
        const int32_t* flatten_ids,  // [n_isects]
        // outputs (pre-allocated)
        float* renders,    // [C, image_height, image_width, channels]
        float* alphas,     // [C, image_height, image_width, 1]
        int32_t* last_ids, // [C, image_height, image_width]
        cudaStream_t stream = nullptr);

    //=========================================================================
    // Rasterization - Backward
    //=========================================================================

    void rasterize_to_pixels_from_world_3dgs_bwd(
        // Gaussian parameters
        const float* means,       // [N, 3]
        const float* quats,       // [N, 4]
        const float* scales,      // [N, 3]
        const float* colors,      // [C, N, channels]
        const float* opacities,   // [C, N]
        const float* backgrounds, // [C, channels] optional - solid color
        const float* bg_images,   // [C, channels, H, W] optional - per-pixel background
        const bool* masks,        // optional
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
        ::CameraModelType camera_model,
        const ::UnscentedTransformParameters& ut_params,
        ::ShutterType rs_type,
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
        const float* v_render_colors, // [C, image_height, image_width, channels]
        const float* v_render_alphas, // [C, image_height, image_width, 1]
        // gradient outputs (pre-allocated, accumulated with atomics)
        float* v_means,     // [N, 3]
        float* v_quats,     // [N, 4]
        float* v_scales,    // [N, 3]
        float* v_colors,    // [C, N, channels]
        float* v_opacities, // [C, N]
        float* densification_info,         // [2, N] flattened or nullptr
        const float* densification_error_map, // [H, W] or nullptr
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
        float* compensations;     // [C, N] optional (can be nullptr)
        // These are allocated internally - caller must free with cudaFree:
        int64_t* isect_ids;   // [n_isects]
        int32_t* flatten_ids; // [n_isects]
        int32_t n_isects;
    };

    void rasterize_from_world_with_sh_fwd(
        // Gaussian parameters
        const float* means,     // [N, 3]
        const float* quats,     // [N, 4]
        const float* scales,    // [N, 3]
        const float* opacities, // [N]
        const float* sh_coeffs, // [N, K, 3]
        uint32_t sh_degree,
        const float* backgrounds, // [C, channels] optional - solid color
        const float* bg_images,   // [C, channels, H, W] optional - per-pixel background
        const bool* masks,        // optional
        // dimensions
        uint32_t N,
        uint32_t C,
        uint32_t K, // number of SH coefficients
        uint32_t image_width,
        uint32_t image_height,
        uint32_t tile_size,
        // camera
        const float* viewmats0, // [C, 4, 4]
        const float* viewmats1, // [C, 4, 4] optional
        const float* Ks,        // [C, 3, 3]
        ::CameraModelType camera_model,
        // settings
        float eps2d,
        float near_plane,
        float far_plane,
        float radius_clip,
        float scaling_modifier,
        bool calc_compensations,
        int render_mode, // 0=RGB, 1=D, 2=ED, 3=RGB_D, 4=RGB_ED
        const ::UnscentedTransformParameters& ut_params,
        ::ShutterType rs_type,
        const float* radial_coeffs,     // optional
        const float* tangential_coeffs, // optional
        const float* thin_prism_coeffs, // optional
        // outputs (result struct with pre-allocated buffers)
        RasterizeWithSHResult& result,
        cudaStream_t stream = nullptr);

    void rasterize_from_world_with_sh_bwd(
        // Gaussian parameters
        const float* means,     // [N, 3]
        const float* quats,     // [N, 4]
        const float* scales,    // [N, 3]
        const float* opacities, // [N]
        const float* sh_coeffs, // [N, K, 3]
        uint32_t sh_degree,
        const float* backgrounds, // [C, channels] optional - solid color
        const float* bg_images,   // [C, channels, H, W] optional - per-pixel background
        const bool* masks,        // optional
        // dimensions
        uint32_t N,
        uint32_t C,
        uint32_t K,
        uint32_t image_width,
        uint32_t image_height,
        uint32_t tile_size,
        // camera
        const float* viewmats0, // [C, 4, 4]
        const float* viewmats1, // optional
        const float* Ks,        // [C, 3, 3]
        ::CameraModelType camera_model,
        // settings
        float eps2d,
        float near_plane,
        float far_plane,
        float radius_clip,
        float scaling_modifier,
        bool calc_compensations,
        int render_mode,
        const ::UnscentedTransformParameters& ut_params,
        ::ShutterType rs_type,
        const float* radial_coeffs,
        const float* tangential_coeffs,
        const float* thin_prism_coeffs,
        // saved from forward
        const float* render_alphas,  // [C, H, W, 1]
        const int32_t* last_ids,     // [C, H, W]
        const int32_t* tile_offsets, // [C, tile_height, tile_width]
        const int32_t* flatten_ids,  // [n_isects]
        uint32_t n_isects,
        const float* colors,        // [C, N, channels]
        const int32_t* radii,       // [C, N, 2]
        const float* means2d,       // [C, N, 2]
        const float* depths,        // [C, N]
        const float* compensations, // [C, N]
        // gradients of outputs
        const float* v_render_colors, // [C, H, W, channels]
        const float* v_render_alphas, // [C, H, W, 1]
        // gradient outputs (pre-allocated, accumulated)
        float* v_means,     // [N, 3]
        float* v_quats,     // [N, 4]
        float* v_scales,    // [N, 3]
        float* v_opacities, // [N]
        float* v_sh_coeffs, // [N, K, 3]
        float* densification_info,          // [2, N] flattened or nullptr
        const float* densification_error_map, // [H, W] or nullptr
        cudaStream_t stream = nullptr);

} // namespace gsplat_lfs
