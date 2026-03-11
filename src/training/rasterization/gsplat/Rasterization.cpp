/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "Rasterization.h"
#include "Common.h"
#include "Ops.h"

#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>

namespace gsplat_lfs {

    //=========================================================================
    // Forward rasterization dispatcher
    //=========================================================================

    void rasterize_to_pixels_from_world_3dgs_fwd(
        const float* means,
        const float* quats,
        const float* scales,
        const float* colors,
        const float* opacities,
        const float* backgrounds,
        const float* bg_images,
        const bool* masks,
        uint32_t C,
        uint32_t N,
        uint32_t n_isects,
        uint32_t channels,
        uint32_t image_width,
        uint32_t image_height,
        uint32_t tile_size,
        const float* viewmats0,
        const float* viewmats1,
        const float* Ks,
        CameraModelType camera_model,
        const UnscentedTransformParameters& ut_params,
        ShutterType rs_type,
        const float* radial_coeffs,
        const float* tangential_coeffs,
        const float* thin_prism_coeffs,
        const int32_t* tile_offsets,
        const int32_t* flatten_ids,
        float* renders,
        float* alphas,
        int32_t* last_ids,
        cudaStream_t stream) {
        GSPLAT_CHECK_CUDA_PTR(means, "means");
        GSPLAT_CHECK_CUDA_PTR(quats, "quats");
        GSPLAT_CHECK_CUDA_PTR(scales, "scales");
        GSPLAT_CHECK_CUDA_PTR(colors, "colors");
        GSPLAT_CHECK_CUDA_PTR(opacities, "opacities");
        GSPLAT_CHECK_CUDA_PTR(renders, "renders");
        GSPLAT_CHECK_CUDA_PTR(alphas, "alphas");
        GSPLAT_CHECK_CUDA_PTR(last_ids, "last_ids");

#define __LAUNCH_KERNEL__(CDIM)                                      \
    case CDIM:                                                       \
        launch_rasterize_to_pixels_from_world_3dgs_fwd_kernel<CDIM>( \
            means, quats, scales, colors, opacities,                 \
            backgrounds, bg_images, masks, C, N, n_isects,           \
            image_width, image_height, tile_size,                    \
            viewmats0, viewmats1, Ks, camera_model,                  \
            ut_params, rs_type,                                      \
            radial_coeffs, tangential_coeffs, thin_prism_coeffs,     \
            tile_offsets, flatten_ids,                               \
            renders, alphas, last_ids, stream);                      \
        break;

        switch (channels) {
            __LAUNCH_KERNEL__(1)
            __LAUNCH_KERNEL__(2)
            __LAUNCH_KERNEL__(3)
            __LAUNCH_KERNEL__(4)
            __LAUNCH_KERNEL__(5)
            __LAUNCH_KERNEL__(8)
            __LAUNCH_KERNEL__(9)
            __LAUNCH_KERNEL__(16)
            __LAUNCH_KERNEL__(17)
            __LAUNCH_KERNEL__(32)
            __LAUNCH_KERNEL__(33)
            __LAUNCH_KERNEL__(64)
            __LAUNCH_KERNEL__(65)
            __LAUNCH_KERNEL__(128)
            __LAUNCH_KERNEL__(129)
            __LAUNCH_KERNEL__(256)
            __LAUNCH_KERNEL__(257)
            __LAUNCH_KERNEL__(512)
            __LAUNCH_KERNEL__(513)
        default:
            fprintf(stderr, "GSPLAT ERROR: Unsupported number of channels: %u\n", channels);
            assert(false && "Unsupported number of channels");
        }
#undef __LAUNCH_KERNEL__
    }

    //=========================================================================
    // Backward rasterization dispatcher
    //=========================================================================

    void rasterize_to_pixels_from_world_3dgs_bwd(
        const float* means,
        const float* quats,
        const float* scales,
        const float* colors,
        const float* opacities,
        const float* backgrounds,
        const float* bg_images,
        const bool* masks,
        uint32_t C,
        uint32_t N,
        uint32_t n_isects,
        uint32_t channels,
        uint32_t image_width,
        uint32_t image_height,
        uint32_t tile_size,
        const float* viewmats0,
        const float* viewmats1,
        const float* Ks,
        CameraModelType camera_model,
        const UnscentedTransformParameters& ut_params,
        ShutterType rs_type,
        const float* radial_coeffs,
        const float* tangential_coeffs,
        const float* thin_prism_coeffs,
        const int32_t* tile_offsets,
        const int32_t* flatten_ids,
        const float* render_alphas,
        const int32_t* last_ids,
        const float* v_render_colors,
        const float* v_render_alphas,
        float* v_means,
        float* v_quats,
        float* v_scales,
        float* v_colors,
        float* v_opacities,
        float* densification_info,
        const float* densification_error_map,
        cudaStream_t stream) {
        GSPLAT_CHECK_CUDA_PTR(means, "means");
        GSPLAT_CHECK_CUDA_PTR(quats, "quats");
        GSPLAT_CHECK_CUDA_PTR(scales, "scales");
        GSPLAT_CHECK_CUDA_PTR(colors, "colors");
        GSPLAT_CHECK_CUDA_PTR(opacities, "opacities");
        GSPLAT_CHECK_CUDA_PTR(v_means, "v_means");
        GSPLAT_CHECK_CUDA_PTR(v_quats, "v_quats");
        GSPLAT_CHECK_CUDA_PTR(v_scales, "v_scales");
        GSPLAT_CHECK_CUDA_PTR(v_colors, "v_colors");
        GSPLAT_CHECK_CUDA_PTR(v_opacities, "v_opacities");

        if (n_isects == 0) {
            // Skip kernel launch if no intersections
            return;
        }

#define __LAUNCH_KERNEL__(CDIM)                                      \
    case CDIM:                                                       \
        launch_rasterize_to_pixels_from_world_3dgs_bwd_kernel<CDIM>( \
            means, quats, scales, colors, opacities,                 \
            backgrounds, bg_images, masks, C, N, n_isects,           \
            image_width, image_height, tile_size,                    \
            viewmats0, viewmats1, Ks, camera_model,                  \
            ut_params, rs_type,                                      \
            radial_coeffs, tangential_coeffs, thin_prism_coeffs,     \
            tile_offsets, flatten_ids,                               \
            render_alphas, last_ids,                                 \
            v_render_colors, v_render_alphas,                        \
            v_means, v_quats, v_scales, v_colors, v_opacities,       \
            densification_info, densification_error_map, stream);    \
        break;

        switch (channels) {
            __LAUNCH_KERNEL__(1)
            __LAUNCH_KERNEL__(2)
            __LAUNCH_KERNEL__(3)
            __LAUNCH_KERNEL__(4)
            __LAUNCH_KERNEL__(5)
            __LAUNCH_KERNEL__(8)
            __LAUNCH_KERNEL__(9)
            __LAUNCH_KERNEL__(16)
            __LAUNCH_KERNEL__(17)
            __LAUNCH_KERNEL__(32)
            __LAUNCH_KERNEL__(33)
            __LAUNCH_KERNEL__(64)
            __LAUNCH_KERNEL__(65)
            __LAUNCH_KERNEL__(128)
            __LAUNCH_KERNEL__(129)
            __LAUNCH_KERNEL__(256)
            __LAUNCH_KERNEL__(257)
            __LAUNCH_KERNEL__(512)
            __LAUNCH_KERNEL__(513)
        default:
            fprintf(stderr, "GSPLAT ERROR: Unsupported number of channels: %u\n", channels);
            assert(false && "Unsupported number of channels");
        }
#undef __LAUNCH_KERNEL__
    }

    //=========================================================================
    // High-level fused forward with SH evaluation
    //=========================================================================

    void rasterize_from_world_with_sh_fwd(
        const float* means,
        const float* quats,
        const float* scales,
        const float* opacities,
        const float* sh_coeffs,
        uint32_t sh_degree,
        const float* backgrounds,
        const float* bg_images,
        const bool* masks,
        uint32_t N,
        uint32_t C,
        uint32_t K,
        uint32_t image_width,
        uint32_t image_height,
        uint32_t tile_size,
        const float* viewmats0,
        const float* viewmats1,
        const float* Ks,
        CameraModelType camera_model,
        float eps2d,
        float near_plane,
        float far_plane,
        float radius_clip,
        float scaling_modifier,
        bool calc_compensations,
        int render_mode,
        const UnscentedTransformParameters& ut_params,
        ShutterType rs_type,
        const float* radial_coeffs,
        const float* tangential_coeffs,
        const float* thin_prism_coeffs,
        RasterizeWithSHResult& result,
        cudaStream_t stream) {
        GSPLAT_CHECK_CUDA_PTR(means, "means");
        GSPLAT_CHECK_CUDA_PTR(quats, "quats");
        GSPLAT_CHECK_CUDA_PTR(scales, "scales");
        GSPLAT_CHECK_CUDA_PTR(opacities, "opacities");
        GSPLAT_CHECK_CUDA_PTR(sh_coeffs, "sh_coeffs");

        const uint32_t tile_width = (image_width + tile_size - 1) / tile_size;
        const uint32_t tile_height = (image_height + tile_size - 1) / tile_size;

        // Determine output channels based on render mode
        // render_mode: 0=RGB, 1=D, 2=ED, 3=RGB_D, 4=RGB_ED
        uint32_t channels = 3; // Default RGB
        if (render_mode == 1 || render_mode == 2) {
            channels = 1; // Depth only
        } else if (render_mode == 3 || render_mode == 4) {
            channels = 4; // RGB + Depth
        }

        // Use scales directly (scaling_modifier should be applied by caller if needed)
        const float* scaled_scales = scales;

        // Step 1: Projection
        projection_ut_3dgs_fused(
            means, quats, scaled_scales, opacities,
            viewmats0, viewmats1, Ks,
            N, C, image_width, image_height,
            eps2d, near_plane, far_plane, radius_clip,
            calc_compensations, camera_model,
            ut_params, rs_type,
            radial_coeffs, tangential_coeffs, thin_prism_coeffs,
            result.radii, result.means2d, result.depths, result.conics,
            result.compensations, stream);

        // Step 2: Tile intersection
        auto isect_result = intersect_tile(
            result.means2d, result.radii, result.depths,
            nullptr, nullptr,
            C, N, tile_size, tile_width, tile_height,
            true,
            result.tiles_per_gauss, stream);

        result.n_isects = isect_result.n_isects;
        result.isect_ids = isect_result.isect_ids;
        result.flatten_ids = isect_result.flatten_ids;

        intersect_offset(
            result.isect_ids, result.n_isects,
            C, tile_width, tile_height,
            result.tile_offsets, stream);

        // Step 3: Compute viewing directions and evaluate SH
        if (render_mode == 0 || render_mode == 3 || render_mode == 4) {
            compute_view_dirs(means, viewmats0, C, N, result.dirs, stream);

            spherical_harmonics_fwd(
                sh_degree, result.dirs, sh_coeffs, nullptr,
                static_cast<int64_t>(C) * N, K,
                result.colors, stream);
        }

        // Step 4: Rasterize to pixels
        rasterize_to_pixels_from_world_3dgs_fwd(
            means, quats, scaled_scales, result.colors, opacities,
            backgrounds, bg_images, masks,
            C, N, result.n_isects, channels,
            image_width, image_height, tile_size,
            viewmats0, viewmats1, Ks, camera_model,
            ut_params, rs_type,
            radial_coeffs, tangential_coeffs, thin_prism_coeffs,
            result.tile_offsets, result.flatten_ids,
            result.render_colors, result.render_alphas, result.last_ids,
            stream);
    }

    //=========================================================================
    // High-level fused backward with SH evaluation
    //=========================================================================

    void rasterize_from_world_with_sh_bwd(
        const float* means,
        const float* quats,
        const float* scales,
        const float* opacities,
        const float* sh_coeffs,
        uint32_t sh_degree,
        const float* backgrounds,
        const float* bg_images,
        const bool* masks,
        uint32_t N,
        uint32_t C,
        uint32_t K,
        uint32_t image_width,
        uint32_t image_height,
        uint32_t tile_size,
        const float* viewmats0,
        const float* viewmats1,
        const float* Ks,
        CameraModelType camera_model,
        float eps2d,
        float near_plane,
        float far_plane,
        float radius_clip,
        float scaling_modifier,
        bool calc_compensations,
        int render_mode,
        const UnscentedTransformParameters& ut_params,
        ShutterType rs_type,
        const float* radial_coeffs,
        const float* tangential_coeffs,
        const float* thin_prism_coeffs,
        const float* render_alphas,
        const int32_t* last_ids,
        const int32_t* tile_offsets,
        const int32_t* flatten_ids,
        uint32_t n_isects,
        const float* colors,
        const int32_t* radii,
        const float* means2d,
        const float* depths,
        const float* compensations,
        const float* v_render_colors,
        const float* v_render_alphas,
        float* v_means,
        float* v_quats,
        float* v_scales,
        float* v_opacities,
        float* v_sh_coeffs,
        float* densification_info,
        const float* densification_error_map,
        cudaStream_t stream) {
        // Determine output channels
        uint32_t channels = 3;
        if (render_mode == 1 || render_mode == 2) {
            channels = 1;
        } else if (render_mode == 3 || render_mode == 4) {
            channels = 4;
        }

        // Temporary buffers for gradients
        float* v_colors = nullptr;
        cudaMalloc(&v_colors, C * N * channels * sizeof(float));
        cudaMemsetAsync(v_colors, 0, C * N * channels * sizeof(float), stream);

        // Backward through rasterization
        rasterize_to_pixels_from_world_3dgs_bwd(
            means, quats, scales, colors, opacities,
            backgrounds, bg_images, masks,
            C, N, n_isects, channels,
            image_width, image_height, tile_size,
            viewmats0, viewmats1, Ks, camera_model,
            ut_params, rs_type,
            radial_coeffs, tangential_coeffs, thin_prism_coeffs,
            tile_offsets, flatten_ids,
            render_alphas, last_ids,
            v_render_colors, v_render_alphas,
            v_means, v_quats, v_scales, v_colors, v_opacities,
            densification_info, densification_error_map,
            stream);

        // Backward through SH
        if (render_mode == 0 || render_mode == 3 || render_mode == 4) {
            spherical_harmonics_bwd(
                K, sh_degree,
                nullptr, // dirs
                sh_coeffs,
                nullptr, // masks
                v_colors,
                static_cast<int64_t>(C) * N,
                false, // compute_v_dirs
                v_sh_coeffs,
                nullptr, // v_dirs
                stream);
        }

        // Apply scaling modifier to scale gradients
        if (scaling_modifier != 1.0f) {
            // TODO: Scale v_scales by scaling_modifier
        }

        cudaFree(v_colors);
    }

} // namespace gsplat_lfs
