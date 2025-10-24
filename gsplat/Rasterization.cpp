#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h> // for DEVICE_GUARD
#include <tuple>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

#include "Cameras.h"
#include "Common.h"
#include "Ops.h"
#include "Rasterization.h"

namespace gsplat {

    ////////////////////////////////////////////////////
    // 3DGS (from world)
    ////////////////////////////////////////////////////

    std::tuple<at::Tensor, at::Tensor, at::Tensor> rasterize_to_pixels_from_world_3dgs_fwd(
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
        const at::Tensor viewmats0,               // [C, 4, 4]
        const at::optional<at::Tensor> viewmats1, // [C, 4, 4] optional for rolling shutter
        const at::Tensor Ks,                      // [C, 3, 3]
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
    ) {
        DEVICE_GUARD(means);
        CHECK_INPUT(means);
        CHECK_INPUT(quats);
        CHECK_INPUT(colors);
        CHECK_INPUT(colors);
        CHECK_INPUT(opacities);
        CHECK_INPUT(tile_offsets);
        CHECK_INPUT(flatten_ids);
        if (backgrounds.has_value()) {
            CHECK_INPUT(backgrounds.value());
        }
        if (masks.has_value()) {
            CHECK_INPUT(masks.value());
        }

        uint32_t C = tile_offsets.size(0); // number of cameras
        uint32_t channels = colors.size(-1);
        assert(channels == 3); // only support RGB for now

        at::Tensor renders =
            at::empty({C, image_height, image_width, channels}, means.options());
        at::Tensor alphas =
            at::empty({C, image_height, image_width, 1}, means.options());
        at::Tensor last_ids = at::empty(
            {C, image_height, image_width}, means.options().dtype(at::kInt));

#define __LAUNCH_KERNEL__(N)                                      \
    case N:                                                       \
        launch_rasterize_to_pixels_from_world_3dgs_fwd_kernel<N>( \
            means,                                                \
            quats,                                                \
            scales,                                               \
            colors,                                               \
            opacities,                                            \
            backgrounds,                                          \
            masks,                                                \
            image_width,                                          \
            image_height,                                         \
            tile_size,                                            \
            viewmats0,                                            \
            viewmats1,                                            \
            Ks,                                                   \
            camera_model,                                         \
            ut_params,                                            \
            rs_type,                                              \
            radial_coeffs,                                        \
            tangential_coeffs,                                    \
            thin_prism_coeffs,                                    \
            tile_offsets,                                         \
            flatten_ids,                                          \
            renders,                                              \
            alphas,                                               \
            last_ids);                                            \
        break;

        // TODO: an optimization can be done by passing the actual number of
        // channels into the kernel functions and avoid necessary global memory
        // writes. This requires moving the channel padding from python to C side.
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
            AT_ERROR("Unsupported number of channels: ", channels);
        }
#undef __LAUNCH_KERNEL__

        return std::make_tuple(renders, alphas, last_ids);
    };

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
        const at::Tensor viewmats0,               // [C, 4, 4]
        const at::optional<at::Tensor> viewmats1, // [C, 4, 4] optional for rolling shutter
        const at::Tensor Ks,                      // [C, 3, 3]
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
    ) {
        DEVICE_GUARD(means);
        CHECK_INPUT(means);
        CHECK_INPUT(quats);
        CHECK_INPUT(scales);
        CHECK_INPUT(colors);
        CHECK_INPUT(opacities);
        CHECK_INPUT(tile_offsets);
        CHECK_INPUT(flatten_ids);
        CHECK_INPUT(render_alphas);
        CHECK_INPUT(last_ids);
        CHECK_INPUT(v_render_colors);
        CHECK_INPUT(v_render_alphas);
        if (backgrounds.has_value()) {
            CHECK_INPUT(backgrounds.value());
        }
        if (masks.has_value()) {
            CHECK_INPUT(masks.value());
        }

        uint32_t channels = colors.size(-1);

        at::Tensor v_means = at::zeros_like(means);
        at::Tensor v_quats = at::zeros_like(quats);
        at::Tensor v_scales = at::zeros_like(scales);
        at::Tensor v_colors = at::zeros_like(colors);
        at::Tensor v_opacities = at::zeros_like(opacities);

#define __LAUNCH_KERNEL__(N)                                      \
    case N:                                                       \
        launch_rasterize_to_pixels_from_world_3dgs_bwd_kernel<N>( \
            means,                                                \
            quats,                                                \
            scales,                                               \
            colors,                                               \
            opacities,                                            \
            backgrounds,                                          \
            masks,                                                \
            image_width,                                          \
            image_height,                                         \
            tile_size,                                            \
            viewmats0,                                            \
            viewmats1,                                            \
            Ks,                                                   \
            camera_model,                                         \
            ut_params,                                            \
            rs_type,                                              \
            radial_coeffs,                                        \
            tangential_coeffs,                                    \
            thin_prism_coeffs,                                    \
            tile_offsets,                                         \
            flatten_ids,                                          \
            render_alphas,                                        \
            last_ids,                                             \
            v_render_colors,                                      \
            v_render_alphas,                                      \
            v_means,                                              \
            v_quats,                                              \
            v_scales,                                             \
            v_colors,                                             \
            v_opacities);                                         \
        break;

        // TODO: an optimization can be done by passing the actual number of
        // channels into the kernel functions and avoid necessary global memory
        // writes. This requires moving the channel padding from python to C side.
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
            AT_ERROR("Unsupported number of channels: ", channels);
        }
#undef __LAUNCH_KERNEL__

        return std::make_tuple(
            v_means, v_quats, v_scales, v_colors, v_opacities);
    }

    ////////////////////////////////////////////////////
    // Fully Fused Rasterization with SH
    ////////////////////////////////////////////////////

    std::tuple<
        at::Tensor,  // render_colors
        at::Tensor,  // render_alphas
        at::Tensor,  // radii
        at::Tensor,  // means2d
        at::Tensor,  // depths
        at::Tensor,  // colors
        at::Tensor,  // tile_offsets
        at::Tensor,  // flatten_ids
        at::Tensor,  // last_ids
        at::Tensor   // compensations
    >
    rasterize_from_world_with_sh_fwd(
        // Gaussian parameters
        const at::Tensor means,                     // [N, 3]
        const at::Tensor quats,                     // [N, 4]
        const at::Tensor scales,                    // [N, 3]
        const at::Tensor opacities,                 // [N] or [N, 1]
        const at::Tensor sh_coeffs,                 // [N, K, 3] - full SH coefficients
        const uint32_t sh_degree,                   // active SH degree to use
        const at::optional<at::Tensor> backgrounds, // [C, channels] optional
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
    ) {
        DEVICE_GUARD(means);
        CHECK_INPUT(means);
        CHECK_INPUT(quats);
        CHECK_INPUT(scales);
        CHECK_INPUT(opacities);
        CHECK_INPUT(sh_coeffs);
        CHECK_INPUT(viewmats0);
        CHECK_INPUT(Ks);

        const uint32_t N = means.size(0);
        const uint32_t C = viewmats0.size(0);

        // Step 1: Project 3D Gaussians to 2D
        auto scaled_scales = scales * scaling_modifier;
        auto proj_results = projection_ut_3dgs_fused(
            means,
            quats,
            scaled_scales,
            opacities,
            viewmats0,
            viewmats1,
            Ks,
            image_width,
            image_height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            calc_compensations,
            camera_model,
            ut_params,
            rs_type,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs);

        auto radii = std::get<0>(proj_results);           // [C, N, 2]
        auto means2d = std::get<1>(proj_results);         // [C, N, 2]
        auto depths = std::get<2>(proj_results);          // [C, N]
        auto conics = std::get<3>(proj_results);          // [C, N, 3]
        auto compensations = std::get<4>(proj_results);   // [C, N] or empty

        // Step 2: Compute colors from Spherical Harmonics
        // Compute camera positions from inverse viewmats
        auto viewmat_inv = at::inverse(viewmats0);  // [C, 4, 4]
        auto campos = viewmat_inv.index({"...", at::indexing::Slice(at::indexing::None, 3), 3}); // [C, 3]

        // Compute directions from camera to each Gaussian
        auto dirs = means.unsqueeze(0) - campos.unsqueeze(1); // [C, N, 3]

        // Create masks based on radii
        auto radii_masks = (radii > 0).all(-1); // [C, N]

        // Expand SH coefficients for multiple cameras
        auto shs = sh_coeffs.unsqueeze(0); // [1, N, K, 3]

        // Construct colors based on render mode
        at::Tensor colors;

        // render_mode: 0=RGB, 1=D, 2=ED, 3=RGB_D, 4=RGB_ED
        if (render_mode == 0 || render_mode == 3 || render_mode == 4) {
            // RGB modes: compute RGB colors using spherical harmonics
            auto rgb_colors = spherical_harmonics_fwd(sh_degree, dirs, shs, radii_masks); // [C, N, 3]
            // Apply SH offset and clamping
            rgb_colors = at::clamp_min(rgb_colors + 0.5f, 0.0f);

            if (render_mode == 0) {
                // RGB only
                colors = rgb_colors;
            } else {
                // RGB + depth: concatenate RGB and depth
                // depths is [C, N], need to add a channel dimension
                auto depth_channel = depths.unsqueeze(-1); // [C, N, 1]
                colors = at::cat({rgb_colors, depth_channel}, -1); // [C, N, 4]
            }
        } else {
            // Depth only modes (D or ED)
            // depths is [C, N], need to add a channel dimension
            colors = depths.unsqueeze(-1); // [C, N, 1]
        }

        // Step 3: Apply opacity with compensations
        auto final_opacities = opacities.unsqueeze(0); // [C, N]
        if (calc_compensations && compensations.defined() && compensations.numel() > 0) {
            final_opacities = final_opacities * compensations;
        }

        // Prepare background
        at::Tensor final_bg;
        if (backgrounds.has_value() && backgrounds->defined() && backgrounds->numel() > 0) {
            final_bg = backgrounds.value();
        } else {
            final_bg = at::empty({0}, means.options().dtype(at::kFloat));
        }

        // Step 4: Tile intersection
        const uint32_t tile_width = (image_width + tile_size - 1) / tile_size;
        const uint32_t tile_height = (image_height + tile_size - 1) / tile_size;

        auto isect_results = intersect_tile(
            means2d, radii, depths,
            at::nullopt, at::nullopt,  // camera_ids, gaussian_ids
            C, tile_size, tile_width, tile_height,
            true);  // sort

        auto tiles_per_gauss = std::get<0>(isect_results);
        auto isect_ids = std::get<1>(isect_results);
        auto flatten_ids = std::get<2>(isect_results);

        auto isect_offsets = intersect_offset(isect_ids, C, tile_width, tile_height);
        isect_offsets = isect_offsets.reshape({C, tile_height, tile_width});

        // Step 5: Rasterize to pixels
        auto raster_results = rasterize_to_pixels_from_world_3dgs_fwd(
            means,
            quats,
            scaled_scales,
            colors,
            final_opacities,
            final_bg,
            masks,
            image_width,
            image_height,
            tile_size,
            viewmats0,
            viewmats1,
            Ks,
            camera_model,
            ut_params,
            rs_type,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
            isect_offsets,
            flatten_ids);

        auto render_colors = std::get<0>(raster_results); // [C, H, W, 3]
        auto render_alphas = std::get<1>(raster_results); // [C, H, W, 1]
        auto last_ids = std::get<2>(raster_results);      // [C, H, W]

        // Return all intermediate values needed for backward
        return std::make_tuple(
            render_colors,
            render_alphas,
            radii,
            means2d,
            depths,
            colors,
            isect_offsets,
            flatten_ids,
            last_ids,
            compensations
        );
    }

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
    ) {
        DEVICE_GUARD(means);
        CHECK_INPUT(means);
        CHECK_INPUT(quats);
        CHECK_INPUT(scales);
        CHECK_INPUT(opacities);
        CHECK_INPUT(sh_coeffs);
        CHECK_INPUT(v_render_colors);
        CHECK_INPUT(v_render_alphas);

        // Prepare background
        at::Tensor final_bg;
        if (backgrounds.has_value() && backgrounds->defined() && backgrounds->numel() > 0) {
            final_bg = backgrounds.value();
        } else {
            final_bg = at::empty({0}, means.options().dtype(at::kFloat));
        }

        // Prepare opacities
        auto final_opacities = opacities.unsqueeze(0); // [C, N]
        if (calc_compensations && compensations.defined() && compensations.numel() > 0) {
            final_opacities = final_opacities * compensations;
        }

        auto scaled_scales = scales * scaling_modifier;

        // Backward through rasterization
        auto raster_grads = rasterize_to_pixels_from_world_3dgs_bwd(
            means,
            quats,
            scaled_scales,
            colors,
            final_opacities,
            final_bg,
            masks,
            image_width,
            image_height,
            tile_size,
            viewmats0,
            viewmats1,
            Ks,
            camera_model,
            ut_params,
            rs_type,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
            tile_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
            v_render_colors,
            v_render_alphas);

        auto v_means_raster = std::get<0>(raster_grads);
        auto v_quats_raster = std::get<1>(raster_grads);
        auto v_scales_raster = std::get<2>(raster_grads);
        auto v_colors = std::get<3>(raster_grads);
        auto v_opacities_raster = std::get<4>(raster_grads);

        // Backward through SH computation (only for RGB modes)
        at::Tensor v_sh_coeffs;

        // render_mode: 0=RGB, 1=D, 2=ED, 3=RGB_D, 4=RGB_ED
        if (render_mode == 0 || render_mode == 3 || render_mode == 4) {
            // RGB modes: need to backward through SH

            // Extract RGB gradients (first 3 channels)
            at::Tensor v_rgb_colors;
            if (render_mode == 0) {
                v_rgb_colors = v_colors;  // All channels are RGB
            } else {
                // RGB + depth: extract RGB channels [C, N, 3]
                v_rgb_colors = v_colors.index({"...", at::indexing::Slice(at::indexing::None, 3)});
            }

            // Recompute camera positions and directions
            auto viewmat_inv = at::inverse(viewmats0);
            auto campos = viewmat_inv.index({"...", at::indexing::Slice(at::indexing::None, 3), 3}); // [C, 3]
            auto dirs = means.unsqueeze(0) - campos.unsqueeze(1); // [C, N, 3]
            auto radii_masks = (radii > 0).all(-1); // [C, N]
            auto shs = sh_coeffs.unsqueeze(0); // [1, N, K, 3]

            // Get RGB colors from saved colors
            at::Tensor rgb_colors;
            if (render_mode == 0) {
                rgb_colors = colors;
            } else {
                rgb_colors = colors.index({"...", at::indexing::Slice(at::indexing::None, 3)});
            }

            // Backward through SH clamping and offset
            auto v_colors_pre_clamp = v_rgb_colors.clone();
            v_colors_pre_clamp.masked_fill_((rgb_colors <= 0.0f), 0.0f);

            // Backward through SH evaluation
            const uint32_t K = sh_coeffs.size(1);
            auto sh_grads = spherical_harmonics_bwd(
                K, sh_degree,
                dirs.reshape({-1, 3}),
                shs.reshape({-1, K, 3}),
                radii_masks.reshape({-1}),
                v_colors_pre_clamp.reshape({-1, 3}),
                false  // don't compute v_dirs for now
            );

            v_sh_coeffs = std::get<0>(sh_grads);  // Reshape back
            v_sh_coeffs = v_sh_coeffs.reshape(shs.sizes()).sum(0);  // Sum over cameras
        } else {
            // Depth only modes: no SH gradients
            const uint32_t N = means.size(0);
            const uint32_t K = sh_coeffs.size(1);
            v_sh_coeffs = at::zeros({N, K, 3}, sh_coeffs.options());
        }

        // Scale gradients are affected by scaling_modifier
        v_scales_raster = v_scales_raster * scaling_modifier;

        return std::make_tuple(
            v_means_raster,
            v_quats_raster,
            v_scales_raster,
            v_opacities_raster,
            v_sh_coeffs
        );
    }

} // namespace gsplat
