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

        // Use fully fused rasterization with SH evaluation
        // This combines: projection + SH + tile intersection + rasterization in one call
        // Currently only supports RGB rendering; depth modes to be added
        auto fused_settings = FusedRasterizationSettings{
            image_width,
            image_height,
            tile_size,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            scaling_modifier,
            calc_compensations,
            viewpoint_camera.camera_model_type(),
            render_mode};

        auto ut_params = UnscentedTransformParameters{};

        auto fused_outputs = FusedRasterizationWithSHFunction::apply(
            means3D,
            rotations,
            scales,
            opacities,
            sh_coeffs,
            sh_degree,
            prepared_bg_color.defined() ? prepared_bg_color : torch::tensor({0.0f, 0.0f, 0.0f}, means3D.options()),
            std::nullopt,  // masks
            viewmat,
            K,
            radial_distortion,
            tangential_distortion,
            std::nullopt,  // thin_prism_coeffs
            fused_settings,
            ut_params);

        auto rendered_image = fused_outputs[0];  // [1, H, W, channels]
        auto rendered_alpha = fused_outputs[1];  // [1, H, W, 1]
        auto radii = fused_outputs[2];           // [C, N, 2]
        auto means2d_with_grad = fused_outputs[3]; // [C, N, 2]
        auto depths = fused_outputs[4];          // [C, N]

        // The fused function already computed all intermediate values - no need to recompute!
        means2d_with_grad = means2d_with_grad.contiguous();
        means2d_with_grad.set_requires_grad(true);
        means2d_with_grad.retain_grad();

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
