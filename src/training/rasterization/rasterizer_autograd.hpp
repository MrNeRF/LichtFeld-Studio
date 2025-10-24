/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "Ops.h"
#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include <torch/torch.h>

namespace gs::training {

    // Forward declare RenderMode
    enum class RenderMode;

    // Fully fused rasterization settings
    struct FusedRasterizationSettings {
        int width;
        int height;
        int tile_size;
        float eps2d;
        float near_plane;
        float far_plane;
        float radius_clip;
        float scaling_modifier;
        bool calc_compensations;
        gsplat::CameraModelType camera_model;
        RenderMode render_mode;
    };

    // Fully fused autograd function: projection + SH + tile intersection + rasterization
    class FusedRasterizationWithSHFunction : public torch::autograd::Function<FusedRasterizationWithSHFunction> {
    public:
        static torch::autograd::tensor_list forward(
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
            UnscentedTransformParameters ut_params);

        static torch::autograd::tensor_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::tensor_list grad_outputs);
    };
} // namespace gs::training
