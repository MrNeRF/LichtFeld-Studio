// A collection of operators for gsplat
#pragma once

#include <ATen/core/Tensor.h>

#include "Cameras.h"
#include "Common.h"

namespace gsplat {

    // Sphereical harmonics
    at::Tensor spherical_harmonics_fwd(
        const uint32_t degrees_to_use,
        const at::Tensor dirs,               // [..., 3]
        const at::Tensor coeffs,             // [..., K, 3]
        const at::optional<at::Tensor> masks // [...]
    );
    std::tuple<at::Tensor, at::Tensor> spherical_harmonics_bwd(
        const uint32_t K,
        const uint32_t degrees_to_use,
        const at::Tensor dirs,                // [..., 3]
        const at::Tensor coeffs,              // [..., K, 3]
        const at::optional<at::Tensor> masks, // [...]
        const at::Tensor v_colors,            // [..., 3]
        bool compute_v_dirs);

    // GS Tile Intersection
    std::tuple<at::Tensor, at::Tensor, at::Tensor> intersect_tile(
        const at::Tensor means2d,                    // [C, N, 2] or [nnz, 2]
        const at::Tensor radii,                      // [C, N, 2] or [nnz, 2]
        const at::Tensor depths,                     // [C, N] or [nnz]
        const at::optional<at::Tensor> camera_ids,   // [nnz]
        const at::optional<at::Tensor> gaussian_ids, // [nnz]
        const uint32_t C,
        const uint32_t tile_size,
        const uint32_t tile_width,
        const uint32_t tile_height,
        const bool sort);
    at::Tensor intersect_offset(
        const at::Tensor isect_ids, // [n_isects]
        const uint32_t C,
        const uint32_t tile_width,
        const uint32_t tile_height);

    // Compute Rotation Matrix from Quaternion
    at::Tensor quats_to_rotmats(
        const at::Tensor quats // [N, 4]
    );

    // Relocate some Gaussians in the Densification Process.
    // Equation (9) in "3D Gaussian Splatting as Markov Chain Monte Carlo"
    std::tuple<at::Tensor, at::Tensor> relocation(
        at::Tensor opacities, // [N]
        at::Tensor scales,    // [N, 3]
        at::Tensor ratios,    // [N]
        at::Tensor binoms,    // [n_max, n_max]
        const int n_max);

    void add_noise(
        at::Tensor raw_opacities, // [N]
        at::Tensor raw_scales,    // [N, 3]
        at::Tensor raw_quats,     // [N, 4]
        at::Tensor noise,         // [N, 3]
        at::Tensor means,         // [N, 3]
        const float current_lr);

    // Use uncented transform to project 3D gaussians to 2D. (none differentiable)
    // https://arxiv.org/abs/2412.12507
    std::tuple<
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor>
    projection_ut_3dgs_fused(
        const at::Tensor means,                   // [N, 3]
        const at::Tensor quats,                   // [N, 4]
        const at::Tensor scales,                  // [N, 3]
        const at::optional<at::Tensor> opacities, // [N] optional
        const at::Tensor viewmats0,               // [C, 4, 4]
        const at::optional<at::Tensor>
            viewmats1,       // [C, 4, 4] optional for rolling shutter
        const at::Tensor Ks, // [C, 3, 3]
        const uint32_t image_width,
        const uint32_t image_height,
        const float eps2d,
        const float near_plane,
        const float far_plane,
        const float radius_clip,
        const bool calc_compensations,
        const CameraModelType camera_model,
        // uncented transform
        const UnscentedTransformParameters ut_params,
        ShutterType rs_type,
        const at::optional<at::Tensor> radial_coeffs,     // [C, 6] or [C, 4] optional
        const at::optional<at::Tensor> tangential_coeffs, // [C, 2] optional
        const at::optional<at::Tensor> thin_prism_coeffs  // [C, 2] optional
    );

    std::tuple<at::Tensor, at::Tensor, at::Tensor>
    rasterize_to_pixels_from_world_3dgs_fwd(
        // Gaussian parameters
        const at::Tensor means,                     // [N, 3]
        const at::Tensor quats,                     // [N, 4]
        const at::Tensor scales,                    // [N, 3]
        const at::Tensor colors,                    // [C, N, channels] or [nnz, channels]
        const at::Tensor opacities,                 // [C, N]  or [nnz]
        const at::optional<at::Tensor> backgrounds, // [C, channels]
        const at::optional<at::Tensor> masks,       // [C, tile_height, tile_width]
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        // camera
        const at::Tensor viewmats0, // [C, 4, 4]
        const at::optional<at::Tensor>
            viewmats1,       // [C, 4, 4] optional for rolling shutter
        const at::Tensor Ks, // [C, 3, 3]
        const CameraModelType camera_model,
        // uncented transform
        const UnscentedTransformParameters ut_params,
        ShutterType rs_type,
        const at::optional<at::Tensor> radial_coeffs,     // [C, 6] or [C, 4] optional
        const at::optional<at::Tensor> tangential_coeffs, // [C, 2] optional
        const at::optional<at::Tensor> thin_prism_coeffs, // [C, 2] optional
        // intersections
        const at::Tensor tile_offsets, // [C, tile_height, tile_width]
        const at::Tensor flatten_ids   // [n_isects]
    );

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    rasterize_to_pixels_from_world_3dgs_bwd(
        // Gaussian parameters
        const at::Tensor means,                     // [N, 3]
        const at::Tensor quats,                     // [N, 4]
        const at::Tensor scales,                    // [N, 3]
        const at::Tensor colors,                    // [C, N, 3] or [nnz, 3]
        const at::Tensor opacities,                 // [C, N] or [nnz]
        const at::optional<at::Tensor> backgrounds, // [C, 3]
        const at::optional<at::Tensor> masks,       // [C, tile_height, tile_width]
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        // camera
        const at::Tensor viewmats0, // [C, 4, 4]
        const at::optional<at::Tensor>
            viewmats1,       // [C, 4, 4] optional for rolling shutter
        const at::Tensor Ks, // [C, 3, 3]
        const CameraModelType camera_model,
        // uncented transform
        const UnscentedTransformParameters ut_params,
        ShutterType rs_type,
        const at::optional<at::Tensor> radial_coeffs,     // [C, 6] or [C, 4] optional
        const at::optional<at::Tensor> tangential_coeffs, // [C, 2] optional
        const at::optional<at::Tensor> thin_prism_coeffs, // [C, 2] optional
        // intersections
        const at::Tensor tile_offsets, // [C, tile_height, tile_width]
        const at::Tensor flatten_ids,  // [n_isects]
        // forward outputs
        const at::Tensor render_alphas, // [C, image_height, image_width, 1]
        const at::Tensor last_ids,      // [C, image_height, image_width]
        // gradients of outputs
        const at::Tensor v_render_colors, // [C, image_height, image_width, 3]
        const at::Tensor v_render_alphas  // [C, image_height, image_width, 1]
    );

    // Fully fused rasterization with SH evaluation
    // This function combines projection, SH evaluation, tile intersection, and rasterization
    // into a single call, similar to fastgs implementation
    // Returns: (render_colors, render_alphas, radii, means2d, depths, colors,
    //           tile_offsets, flatten_ids, last_ids, compensations)
    // render_mode: 0=RGB, 1=D, 2=ED, 3=RGB_D, 4=RGB_ED
    std::tuple<
        at::Tensor,  // render_colors [C, H, W, channels] where channels depends on render_mode
        at::Tensor,  // render_alphas [C, H, W, 1]
        at::Tensor,  // radii [C, N, 2]
        at::Tensor,  // means2d [C, N, 2]
        at::Tensor,  // depths [C, N]
        at::Tensor,  // colors [C, N, channels] where channels depends on render_mode
        at::Tensor,  // tile_offsets [C, tile_height, tile_width]
        at::Tensor,  // flatten_ids [n_isects]
        at::Tensor,  // last_ids [C, H, W]
        at::Tensor   // compensations [C, N] or empty
    >
    rasterize_from_world_with_sh_fwd(
        // Gaussian parameters
        const at::Tensor means,                     // [N, 3]
        const at::Tensor quats,                     // [N, 4]
        const at::Tensor scales,                    // [N, 3]
        const at::Tensor opacities,                 // [N] or [N, 1]
        const at::Tensor sh_coeffs,                 // [N, K, 3] - full SH coefficients
        const uint32_t sh_degree,                   // active SH degree to use
        const at::optional<at::Tensor> backgrounds, // [C, channels] optional, channels depends on render_mode
        const at::optional<at::Tensor> masks,       // [C, tile_height, tile_width]
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        // camera
        const at::Tensor viewmats0, // [C, 4, 4]
        const at::optional<at::Tensor>
            viewmats1,       // [C, 4, 4] optional for rolling shutter
        const at::Tensor Ks, // [C, 3, 3]
        const CameraModelType camera_model,
        // settings
        const float eps2d,
        const float near_plane,
        const float far_plane,
        const float radius_clip,
        const float scaling_modifier,
        const bool calc_compensations,
        const int render_mode, // 0=RGB, 1=D, 2=ED, 3=RGB_D, 4=RGB_ED
        // uncented transform
        const UnscentedTransformParameters ut_params,
        ShutterType rs_type,
        const at::optional<at::Tensor> radial_coeffs,     // [C, 6] or [C, 4] optional
        const at::optional<at::Tensor> tangential_coeffs, // [C, 2] optional
        const at::optional<at::Tensor> thin_prism_coeffs  // [C, 2] optional
    );

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    rasterize_from_world_with_sh_bwd(
        // Gaussian parameters
        const at::Tensor means,                     // [N, 3]
        const at::Tensor quats,                     // [N, 4]
        const at::Tensor scales,                    // [N, 3]
        const at::Tensor opacities,                 // [N] or [N, 1]
        const at::Tensor sh_coeffs,                 // [N, K, 3]
        const uint32_t sh_degree,                   // active SH degree
        const at::optional<at::Tensor> backgrounds, // [C, channels]
        const at::optional<at::Tensor> masks,       // [C, tile_height, tile_width]
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        // camera
        const at::Tensor viewmats0, // [C, 4, 4]
        const at::optional<at::Tensor>
            viewmats1,       // [C, 4, 4] optional for rolling shutter
        const at::Tensor Ks, // [C, 3, 3]
        const CameraModelType camera_model,
        // settings
        const float eps2d,
        const float near_plane,
        const float far_plane,
        const float radius_clip,
        const float scaling_modifier,
        const bool calc_compensations,
        const int render_mode, // 0=RGB, 1=D, 2=ED, 3=RGB_D, 4=RGB_ED
        // uncented transform
        const UnscentedTransformParameters ut_params,
        ShutterType rs_type,
        const at::optional<at::Tensor> radial_coeffs,     // [C, 6] or [C, 4] optional
        const at::optional<at::Tensor> tangential_coeffs, // [C, 2] optional
        const at::optional<at::Tensor> thin_prism_coeffs, // [C, 2] optional
        // saved from forward
        const at::Tensor render_alphas, // [C, image_height, image_width, 1]
        const at::Tensor last_ids,      // [C, image_height, image_width]
        const at::Tensor tile_offsets,  // [C, tile_height, tile_width]
        const at::Tensor flatten_ids,   // [n_isects]
        const at::Tensor colors,        // [C, N, channels] - computed colors (RGB and/or depth)
        const at::Tensor radii,         // [C, N, 2] - projected radii
        const at::Tensor means2d,       // [C, N, 2] - projected 2D positions
        const at::Tensor depths,        // [C, N] - depths
        const at::Tensor compensations, // [C, N] - opacity compensations
        // gradients of outputs
        const at::Tensor v_render_colors, // [C, image_height, image_width, channels]
        const at::Tensor v_render_alphas  // [C, image_height, image_width, 1]
    );

} // namespace gsplat
