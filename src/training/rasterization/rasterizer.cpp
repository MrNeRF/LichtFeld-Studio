/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rasterizer.hpp"
#include "Ops.h"
#include "rasterizer_autograd.hpp"
#include <torch/torch.h>

namespace gs::training {
    using torch::indexing::None;
    using torch::indexing::Slice;

    inline torch::Tensor spherical_harmonics(
        int sh_degree,
        const torch::Tensor& dirs,
        const torch::Tensor& coeffs,
        const torch::Tensor& masks = {}) {
        // Validate inputs
        TORCH_CHECK((sh_degree + 1) * (sh_degree + 1) <= coeffs.size(-2),
                    "coeffs K dimension must be at least ", (sh_degree + 1) * (sh_degree + 1),
                    ", got ", coeffs.size(-2));
        TORCH_CHECK(dirs.sizes().slice(0, dirs.dim() - 1) == coeffs.sizes().slice(0, coeffs.dim() - 2),
                    "dirs and coeffs batch dimensions must match");
        TORCH_CHECK(dirs.size(-1) == 3, "dirs last dimension must be 3, got ", dirs.size(-1));
        TORCH_CHECK(coeffs.size(-1) == 3, "coeffs last dimension must be 3, got ", coeffs.size(-1));

        if (masks.defined()) {
            TORCH_CHECK(masks.sizes() == dirs.sizes().slice(0, dirs.dim() - 1),
                        "masks shape must match dirs shape without last dimension");
        }

        // Create sh_degree tensor
        auto sh_degree_tensor = torch::tensor({sh_degree},
                                              torch::TensorOptions().dtype(torch::kInt32).device(dirs.device()));

        // Call the autograd function
        return SphericalHarmonicsFunction::apply(
            sh_degree_tensor,
            dirs.contiguous(),
            coeffs.contiguous(),
            masks.defined() ? masks.contiguous() : masks)[0];
    }

    // Main render function
    RenderOutput rasterize(
        Camera& viewpoint_camera,
        const SplatData& gaussian_model,
        torch::Tensor& bg_color,
        float scaling_modifier,
        bool packed,
        bool antialiased,
        RenderMode render_mode,
        const gs::geometry::BoundingBox* bounding_box) {
        // Ensure we don't use packed mode (not supported in this implementation)
        TORCH_CHECK(!packed, "Packed mode is not supported in this implementation");

        // Get camera parameters
        const int image_height = static_cast<int>(viewpoint_camera.image_height());
        const int image_width = static_cast<int>(viewpoint_camera.image_width());

        // Prepare viewmat and K
        auto viewmat = viewpoint_camera.world_view_transform().to(torch::kCUDA);
        TORCH_CHECK(viewmat.dim() == 3 && viewmat.size(0) == 1 && viewmat.size(1) == 4 && viewmat.size(2) == 4,
                    "viewmat must be [1, 4, 4] after transpose and unsqueeze, got ", viewmat.sizes());
        TORCH_CHECK(viewmat.is_cuda(), "viewmat must be on CUDA");

        const auto K = viewpoint_camera.K().to(torch::kCUDA);
        TORCH_CHECK(K.is_cuda(), "K must be on CUDA");

        // Get Gaussian parameters
        auto means3D = gaussian_model.get_means();

        auto opacities = gaussian_model.get_opacity();
        if (opacities.dim() == 2 && opacities.size(1) == 1) {
            opacities = opacities.squeeze(-1);
        }
        auto scales = gaussian_model.get_scaling();
        auto rotations = gaussian_model.get_rotation();
        auto sh_coeffs = gaussian_model.get_shs();
        const int sh_degree = gaussian_model.get_active_sh_degree();

        // Validate Gaussian parameters
        const int N = static_cast<int>(means3D.size(0));
        TORCH_CHECK(means3D.dim() == 2 && means3D.size(1) == 3,
                    "means3D must be [N, 3], got ", means3D.sizes());
        TORCH_CHECK(opacities.dim() == 1 && opacities.size(0) == N,
                    "opacities must be [N], got ", opacities.sizes());
        TORCH_CHECK(scales.dim() == 2 && scales.size(0) == N && scales.size(1) == 3,
                    "scales must be [N, 3], got ", scales.sizes());
        TORCH_CHECK(rotations.dim() == 2 && rotations.size(0) == N && rotations.size(1) == 4,
                    "rotations must be [N, 4], got ", rotations.sizes());
        TORCH_CHECK(sh_coeffs.dim() == 3 && sh_coeffs.size(0) == N && sh_coeffs.size(2) == 3,
                    "sh_coeffs must be [N, K, 3], got ", sh_coeffs.sizes());

        // Check if we have enough SH coefficients for the requested degree
        const int required_sh_coeffs = (sh_degree + 1) * (sh_degree + 1);
        TORCH_CHECK(sh_coeffs.size(1) >= required_sh_coeffs,
                    "Not enough SH coefficients. Expected at least ", required_sh_coeffs,
                    " but got ", sh_coeffs.size(1));

        // Device checks for Gaussian parameters
        TORCH_CHECK(means3D.is_cuda(), "means3D must be on CUDA");
        TORCH_CHECK(opacities.is_cuda(), "opacities must be on CUDA");
        TORCH_CHECK(scales.is_cuda(), "scales must be on CUDA");
        TORCH_CHECK(rotations.is_cuda(), "rotations must be on CUDA");
        TORCH_CHECK(sh_coeffs.is_cuda(), "sh_coeffs must be on CUDA");

        // Handle background color - can be undefined
        torch::Tensor prepared_bg_color;
        if (!bg_color.defined() || bg_color.numel() == 0) {
            // Keep it undefined
            prepared_bg_color = torch::Tensor();
        } else {
            prepared_bg_color = bg_color.view({1, -1}).to(torch::kCUDA);
            TORCH_CHECK(prepared_bg_color.size(0) == 1 && prepared_bg_color.size(1) == 3,
                        "bg_color must be reshapeable to [1, 3], got ", prepared_bg_color.sizes());
            TORCH_CHECK(prepared_bg_color.is_cuda(), "bg_color must be on CUDA");
        }

        const float eps2d = 0.3f;
        const float near_plane = 0.01f;
        const float far_plane = 10000.0f;
        const float radius_clip = 0.0f;
        const int tile_size = 16;
        const bool calc_compensations = antialiased;

        std::optional<torch::Tensor> radial_distortion;
        if (viewpoint_camera.radial_distortion().numel() > 0) {
            auto radial_distortion_val = viewpoint_camera.radial_distortion().to(torch::kCUDA);
            TORCH_CHECK(radial_distortion_val.dim() == 1, "radial_distortion must be 1D, got ", radial_distortion_val.sizes());
            if (radial_distortion_val.size(-1) < 4) {
                // Pad to 4 coefficients if less are provided
                radial_distortion_val = torch::nn::functional::pad(
                    radial_distortion_val,
                    torch::nn::functional::PadFuncOptions({0, 4 - radial_distortion_val.size(-1)}).mode(torch::kConstant).value(0));
            }
            radial_distortion = radial_distortion_val;
        }
        std::optional<torch::Tensor> tangential_distortion;
        if (viewpoint_camera.tangential_distortion().numel() > 0) {
            auto tangential_distortion_val = viewpoint_camera.tangential_distortion().to(torch::kCUDA);
            TORCH_CHECK(tangential_distortion_val.dim() == 1, "tangential_distortion must be 1D, got ", tangential_distortion_val.sizes());
            if (tangential_distortion_val.size(-1) < 2) {
                // Pad to 2 coefficients if less are provided
                tangential_distortion_val = torch::nn::functional::pad(
                    tangential_distortion_val,
                    torch::nn::functional::PadFuncOptions({0, 2 - tangential_distortion_val.size(-1)}).mode(torch::kConstant).value(0));
            }
            tangential_distortion = tangential_distortion_val;
        }

        // Step 1: Projection
        torch::Tensor radii;
        torch::Tensor means2d;
        torch::Tensor depths;
        torch::Tensor conics;
        torch::Tensor compensations;

        auto proj_settings = GUTProjectionSettings{
            image_width,
            image_height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            scaling_modifier,
            viewpoint_camera.camera_model_type()};
        auto proj_outputs = fully_fused_projection_with_ut(
            means3D,
            rotations,
            scales,
            opacities,
            viewmat,
            K,
            radial_distortion,
            tangential_distortion,
            std::nullopt,
            proj_settings,
            UnscentedTransformParameters());

        radii = proj_outputs[0];
        means2d = proj_outputs[1];
        depths = proj_outputs[2];
        conics = proj_outputs[3];
        compensations = proj_outputs[4];

        // Create means2d with gradient tracking for backward compatibility
        auto means2d_with_grad = means2d.contiguous();
        means2d_with_grad.set_requires_grad(true);
        means2d_with_grad.retain_grad();

        // Step 2: Compute colors from SH
        // First, compute camera position from inverse viewmat
        auto viewmat_inv = torch::inverse(viewmat);
        auto campos = viewmat_inv.index({Slice(), Slice(None, 3), 3}); // [C, 3]

        // Compute directions from camera to each Gaussian
        auto dirs = means3D.unsqueeze(0) - campos.unsqueeze(1); // [C, N, 3]

        // Create masks based on radii
        auto masks = (radii > 0).all(-1); // [C, N]

        // The Python code broadcasts colors from [N, K, 3] to [C, N, K, 3] if needed
        auto shs = sh_coeffs.unsqueeze(0); // [1, N, K, 3]

        // Now call spherical harmonics with proper directions
        auto colors = spherical_harmonics(sh_degree, dirs, shs, masks); // [C, N, 3]

        // Apply the SH offset and clamping for rendering (shift from [-0.5, 0.5] to [0, 1])
        colors = torch::clamp_min(colors + 0.5f, 0.0f);

        // Step 3: Handle depth based on render mode
        torch::Tensor render_colors;
        torch::Tensor final_bg;

        switch (render_mode) {
        case RenderMode::RGB:
            render_colors = colors;
            final_bg = prepared_bg_color;
            break;

        case RenderMode::D:
        case RenderMode::ED:
            render_colors = depths.unsqueeze(-1); // [C, N, 1]
            if (prepared_bg_color.defined()) {
                final_bg = torch::zeros({1, 1}, prepared_bg_color.options());
            } else {
                final_bg = torch::Tensor(); // Keep undefined
            }
            break;

        case RenderMode::RGB_D:
        case RenderMode::RGB_ED:
            // Concatenate colors and depths
            render_colors = torch::cat({colors, depths.unsqueeze(-1)}, -1); // [C, N, 4]
            if (prepared_bg_color.defined()) {
                final_bg = torch::cat({prepared_bg_color, torch::zeros({1, 1}, prepared_bg_color.options())}, -1);
            } else {
                final_bg = torch::Tensor(); // Keep undefined
            }
            break;
        }

        if (!final_bg.defined()) {
            // Create empty tensor on CUDA - same pattern as compensations in projection
            final_bg = at::empty({0}, colors.options().dtype(torch::kFloat32));
        }

        // Step 4: Apply opacity with compensations
        torch::Tensor final_opacities;
        if (calc_compensations && compensations.defined() && compensations.numel() > 0) {
            final_opacities = opacities.unsqueeze(0) * compensations;
        } else {
            final_opacities = opacities.unsqueeze(0);
        }
        TORCH_CHECK(final_opacities.is_cuda(), "final_opacities must be on CUDA");

        // Step 5: Tile intersection
        const int tile_width = (image_width + tile_size - 1) / tile_size;
        const int tile_height = (image_height + tile_size - 1) / tile_size;

        const auto isect_results = gsplat::intersect_tile(
            means2d_with_grad, radii, depths, {}, {},
            1, tile_size, tile_width, tile_height,
            true);

        const auto tiles_per_gauss = std::get<0>(isect_results);
        const auto isect_ids = std::get<1>(isect_results);
        const auto flatten_ids = std::get<2>(isect_results);

        auto isect_offsets = gsplat::intersect_offset(
            isect_ids, 1, tile_width, tile_height);
        isect_offsets = isect_offsets.reshape({1, tile_height, tile_width});

        TORCH_CHECK(tiles_per_gauss.is_cuda(), "tiles_per_gauss must be on CUDA");
        TORCH_CHECK(isect_ids.is_cuda(), "isect_ids must be on CUDA");
        TORCH_CHECK(flatten_ids.is_cuda(), "flatten_ids must be on CUDA");
        TORCH_CHECK(isect_offsets.is_cuda(), "isect_offsets must be on CUDA");

        // Step 6: Rasterization
        auto raster_settings = GUTRasterizationSettings{
            image_width,
            image_height,
            tile_size,
            scaling_modifier,
            viewpoint_camera.camera_model_type()};
        auto ut_params = UnscentedTransformParameters{};
        auto raster_outputs = GUTRasterizationFunction::apply(
            means3D,
            rotations,
            scales,
            render_colors,
            final_opacities,
            final_bg,
            std::nullopt,
            viewmat,
            K,
            radial_distortion,
            tangential_distortion,
            std::nullopt, // thin_prism_coeffs
            isect_offsets,
            flatten_ids,
            raster_settings,
            ut_params);
        auto rendered_image = raster_outputs[0];
        auto rendered_alpha = raster_outputs[1];

        // Step 7: Post-process based on render mode
        torch::Tensor final_image, final_depth;

        switch (render_mode) {
        case RenderMode::RGB:
            final_image = rendered_image;
            final_depth = torch::Tensor(); // Empty
            break;

        case RenderMode::D:
            final_depth = rendered_image;  // It's actually depth
            final_image = torch::Tensor(); // Empty
            break;

        case RenderMode::ED:
            // Normalize accumulated depth by alpha to get expected depth
            final_depth = rendered_image / rendered_alpha.clamp_min(1e-10);
            final_image = torch::Tensor(); // Empty
            break;

        case RenderMode::RGB_D:
            final_image = rendered_image.index({Slice(), Slice(), Slice(), Slice(None, -1)});
            final_depth = rendered_image.index({Slice(), Slice(), Slice(), Slice(-1, None)});
            break;

        case RenderMode::RGB_ED:
            final_image = rendered_image.index({Slice(), Slice(), Slice(), Slice(None, -1)});
            auto accum_depth = rendered_image.index({Slice(), Slice(), Slice(), Slice(-1, None)});
            final_depth = accum_depth / rendered_alpha.clamp_min(1e-10);
            break;
        }

        // Prepare output
        RenderOutput result;

        // Handle image output
        if (final_image.defined() && final_image.numel() > 0) {
            result.image = torch::clamp(final_image.squeeze(0).permute({2, 0, 1}), 0.0f, 1.0f);
        } else {
            result.image = torch::Tensor();
        }

        // Handle alpha output - always present
        result.alpha = rendered_alpha.squeeze(0).permute({2, 0, 1});

        // Handle depth output
        if (final_depth.defined() && final_depth.numel() > 0) {
            result.depth = final_depth.squeeze(0).permute({2, 0, 1});
        } else {
            result.depth = torch::Tensor();
        }

        result.means2d = means2d_with_grad;
        result.depths = depths.squeeze(0);
        result.radii = std::get<0>(radii.squeeze(0).max(-1));
        result.visibility = (result.radii > 0);
        result.width = image_width;
        result.height = image_height;

        // Final device checks for outputs
        if (result.image.defined() && result.image.numel() > 0) {
            TORCH_CHECK(result.image.is_cuda(), "result.image must be on CUDA");
        }
        TORCH_CHECK(result.alpha.is_cuda(), "result.alpha must be on CUDA");
        if (result.depth.defined() && result.depth.numel() > 0) {
            TORCH_CHECK(result.depth.is_cuda(), "result.depth must be on CUDA");
        }
        TORCH_CHECK(result.means2d.is_cuda(), "result.means2d must be on CUDA");
        TORCH_CHECK(result.depths.is_cuda(), "result.depths must be on CUDA");
        TORCH_CHECK(result.radii.is_cuda(), "result.radii must be on CUDA");
        TORCH_CHECK(result.visibility.is_cuda(), "result.visibility must be on CUDA");

        return result;
    }
} // namespace gs::training
