/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rasterization/rasterizer_autograd.hpp"
#include "Projection.h"

namespace gs::training {
    using namespace torch::indexing;

    torch::autograd::tensor_list FusedRasterizationWithSHFunction::forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor means,                            // [N, 3]
        torch::Tensor quats,                            // [N, 4]
        torch::Tensor scales,                           // [N, 3]
        torch::Tensor opacities,                        // [N] or [N, 1]
        torch::Tensor sh_coeffs,                        // [N, K, 3]
        int sh_degree,                                  // active SH degree
        torch::Tensor bg_color,                         // [3] or [1, 3]
        std::optional<torch::Tensor> masks,             // [C, tile_height, tile_width]
        torch::Tensor viewmat,                          // [C, 4, 4]
        torch::Tensor K,                                // [C, 3, 3]
        std::optional<torch::Tensor> radial_coeffs,     // [C, 6] or [C, 4]
        std::optional<torch::Tensor> tangential_coeffs, // [C, 2]
        std::optional<torch::Tensor> thin_prism_coeffs, // [C, 4]
        FusedRasterizationSettings settings,
        UnscentedTransformParameters ut_params) {

        // Prepare background color
        std::optional<torch::Tensor> bg_opt;
        if (bg_color.defined() && bg_color.numel() > 0) {
            bg_opt = bg_color.view({1, -1}).contiguous();
        }

        // Call the fused gsplat function
        auto results = gsplat::rasterize_from_world_with_sh_fwd(
            means.contiguous(),
            quats.contiguous(),
            scales.contiguous(),
            opacities.contiguous(),
            sh_coeffs.contiguous(),
            static_cast<uint32_t>(sh_degree),
            bg_opt,
            masks,
            settings.width,
            settings.height,
            settings.tile_size,
            viewmat.contiguous(),
            std::nullopt,  // viewmats1 (rolling shutter)
            K.contiguous(),
            settings.camera_model,
            settings.eps2d,
            settings.near_plane,
            settings.far_plane,
            settings.radius_clip,
            settings.scaling_modifier,
            settings.calc_compensations,
            static_cast<int>(settings.render_mode),  // Pass render_mode as int
            ut_params,
            ShutterType::GLOBAL,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs);

        auto render_colors = std::get<0>(results).contiguous();
        auto render_alphas = std::get<1>(results).contiguous();
        auto radii = std::get<2>(results).contiguous();
        auto means2d = std::get<3>(results).contiguous();
        auto depths = std::get<4>(results).contiguous();
        auto colors = std::get<5>(results).contiguous();
        auto tile_offsets = std::get<6>(results).contiguous();
        auto flatten_ids = std::get<7>(results).contiguous();
        auto last_ids = std::get<8>(results).contiguous();
        auto compensations = std::get<9>(results);  // May be empty

        // Save for backward - save all intermediate values
        ctx->save_for_backward({means, quats, scales, opacities, sh_coeffs,
                                bg_color, viewmat, K,
                                radial_coeffs.has_value() ? *radial_coeffs : torch::Tensor(),
                                tangential_coeffs.has_value() ? *tangential_coeffs : torch::Tensor(),
                                thin_prism_coeffs.has_value() ? *thin_prism_coeffs : torch::Tensor(),
                                masks.has_value() ? *masks : torch::Tensor(),
                                render_alphas, radii, means2d, depths, colors,
                                tile_offsets, flatten_ids, last_ids, compensations});

        ctx->saved_data["sh_degree"] = sh_degree;
        ctx->saved_data["width"] = settings.width;
        ctx->saved_data["height"] = settings.height;
        ctx->saved_data["tile_size"] = settings.tile_size;
        ctx->saved_data["eps2d"] = settings.eps2d;
        ctx->saved_data["near_plane"] = settings.near_plane;
        ctx->saved_data["far_plane"] = settings.far_plane;
        ctx->saved_data["radius_clip"] = settings.radius_clip;
        ctx->saved_data["scaling_modifier"] = settings.scaling_modifier;
        ctx->saved_data["calc_compensations"] = settings.calc_compensations;
        ctx->saved_data["camera_model"] = static_cast<int>(settings.camera_model);
        ctx->saved_data["render_mode"] = static_cast<int>(settings.render_mode);
        ctx->saved_data["ut_params"] = ut_params.to_tensor();

        // Return all useful outputs: render_colors, render_alphas, radii, means2d, depths
        // This avoids recomputing projection in rasterizer.cpp
        return {render_colors, render_alphas, radii, means2d, depths};
    }

    torch::autograd::tensor_list FusedRasterizationWithSHFunction::backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs) {

        auto v_render_colors = grad_outputs[0].contiguous();
        auto v_render_alphas = grad_outputs[1].contiguous();

        auto saved = ctx->get_saved_variables();
        const auto& means = saved[0];
        const auto& quats = saved[1];
        const auto& scales = saved[2];
        const auto& opacities = saved[3];
        const auto& sh_coeffs = saved[4];
        const auto& bg_color = saved[5];
        const auto& viewmat = saved[6];
        const auto& K = saved[7];
        const std::optional<torch::Tensor> radial_coeffs = saved[8].numel() > 0 ? std::optional(saved[8]) : std::nullopt;
        const std::optional<torch::Tensor> tangential_coeffs = saved[9].numel() > 0 ? std::optional(saved[9]) : std::nullopt;
        const std::optional<torch::Tensor> thin_prism_coeffs = saved[10].numel() > 0 ? std::optional(saved[10]) : std::nullopt;
        const std::optional<torch::Tensor> masks = saved[11].numel() > 0 ? std::optional(saved[11]) : std::nullopt;
        const auto& render_alphas = saved[12];
        const auto& radii = saved[13];
        const auto& means2d = saved[14];
        const auto& depths = saved[15];
        const auto& colors = saved[16];
        const auto& tile_offsets = saved[17];
        const auto& flatten_ids = saved[18];
        const auto& last_ids = saved[19];
        const auto& compensations = saved[20];

        // Extract settings
        const int sh_degree = ctx->saved_data["sh_degree"].toInt();
        const int width = ctx->saved_data["width"].toInt();
        const int height = ctx->saved_data["height"].toInt();
        const int tile_size = ctx->saved_data["tile_size"].toInt();
        const float eps2d = static_cast<float>(ctx->saved_data["eps2d"].toDouble());
        const float near_plane = static_cast<float>(ctx->saved_data["near_plane"].toDouble());
        const float far_plane = static_cast<float>(ctx->saved_data["far_plane"].toDouble());
        const float radius_clip = static_cast<float>(ctx->saved_data["radius_clip"].toDouble());
        const float scaling_modifier = static_cast<float>(ctx->saved_data["scaling_modifier"].toDouble());
        const bool calc_compensations = ctx->saved_data["calc_compensations"].toBool();
        const gsplat::CameraModelType camera_model =
            static_cast<gsplat::CameraModelType>(ctx->saved_data["camera_model"].toInt());
        const RenderMode render_mode = static_cast<RenderMode>(ctx->saved_data["render_mode"].toInt());
        auto ut_params = UnscentedTransformParameters::from_tensor(ctx->saved_data["ut_params"].toTensor());

        // Prepare background
        std::optional<torch::Tensor> bg_opt;
        if (bg_color.defined() && bg_color.numel() > 0) {
            bg_opt = bg_color.view({1, -1});
        }

        // Call the backward function with saved intermediate values
        auto grads = gsplat::rasterize_from_world_with_sh_bwd(
            means,
            quats,
            scales,
            opacities,
            sh_coeffs,
            static_cast<uint32_t>(sh_degree),
            bg_opt,
            masks,
            width,
            height,
            tile_size,
            viewmat,
            std::nullopt,  // viewmats1
            K,
            camera_model,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            scaling_modifier,
            calc_compensations,
            static_cast<int>(render_mode),  // Pass render_mode as int
            ut_params,
            ShutterType::GLOBAL,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
            // Saved forward outputs
            render_alphas,
            last_ids,
            tile_offsets,
            flatten_ids,
            colors,
            radii,
            means2d,
            depths,
            compensations,
            // Gradients of outputs
            v_render_colors,
            v_render_alphas);

        auto v_means = std::get<0>(grads);
        auto v_quats = std::get<1>(grads);
        auto v_scales = std::get<2>(grads);
        auto v_opacities = std::get<3>(grads);
        auto v_sh_coeffs = std::get<4>(grads);

        auto v_bg_color = torch::Tensor();
        if (ctx->needs_input_grad(6)) {
            v_bg_color = (v_render_colors * (1.0f - render_alphas)).toType(torch::kFloat32).sum({-3, -2});
        }

        return {
            v_means, v_quats, v_scales, v_opacities, v_sh_coeffs,
            torch::Tensor(), // sh_degree
            v_bg_color,
            torch::Tensor(), // masks
            torch::Tensor(), // viewmat
            torch::Tensor(), // K
            torch::Tensor(), // radial_coeffs
            torch::Tensor(), // tangential_coeffs
            torch::Tensor(), // thin_prism_coeffs
            torch::Tensor(), // settings
            torch::Tensor()  // ut_params
        };
    }
} // namespace gs::training
