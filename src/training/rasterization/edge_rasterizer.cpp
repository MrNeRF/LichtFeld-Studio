/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "edge_rasterizer.hpp"
#include "core/cuda/memory_arena.hpp"
#include <edge_rasterization_api.h>
#include <cassert>

namespace lfs::training {

    std::expected<RenderOutput, std::string> edge_rasterize_forward(
        core::Camera& viewpoint_camera,
        core::SplatData& gaussian_model,
        const lfs::core::Tensor& pixel_weights,
        bool mip_filter) {
        // Get camera parameters
        const int width = viewpoint_camera.image_width();
        const int height = viewpoint_camera.image_height();

        auto [fx, fy, cx, cy] = viewpoint_camera.get_intrinsics();

        assert(!pixel_weights.is_valid() ||
               pixel_weights.numel() == static_cast<size_t>(width) * static_cast<size_t>(height));

        // Get Gaussian parameters
        auto& means = gaussian_model.means();
        auto& raw_opacities = gaussian_model.opacity_raw();
        auto& raw_scales = gaussian_model.scaling_raw();
        auto& raw_rotations = gaussian_model.rotation_raw();

        constexpr float near_plane = 0.01f;
        constexpr float far_plane = 1e10f;

        // Get direct GPU pointers (tensors are already contiguous on CUDA)
        const float* w2c_ptr = viewpoint_camera.world_view_transform_ptr();

        const int n_primitives = static_cast<int>(means.shape()[0]);

        if (n_primitives == 0) {
            return std::unexpected("n_primitives is 0 - model has no gaussians");
        }

        // Input pixel_weights pointer and output accum_weights
        auto pixel_weights_contig = pixel_weights.contiguous();
        const float* pixel_weights_ptr = pixel_weights_contig.ptr<float>();

        auto accum_weights = core::Tensor::zeros(
            {static_cast<size_t>(n_primitives)}, core::Device::CUDA, core::DataType::Float32);
        float* accum_weights_out = accum_weights.ptr<float>();

        // Call forward_raw with raw pointers (no PyTorch wrappers)
        // Use adjusted cx/cy for tile rendering
        auto forward_ctx = edge_compute::rasterization::edge_forward_raw(
            means.ptr<float>(),
            raw_scales.ptr<float>(),
            raw_rotations.ptr<float>(),
            raw_opacities.ptr<float>(),
            w2c_ptr,
            n_primitives,
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            near_plane,
            far_plane,
            mip_filter,
            pixel_weights_ptr,
            accum_weights_out);

        if (!forward_ctx.success) {
            return std::unexpected(std::string(forward_ctx.error_message));
        }

        // Release arena frame — edge rasterization has no backward pass
        auto& arena = core::GlobalArenaManager::instance().get_arena();
        arena.end_frame(forward_ctx.frame_id);

        // Prepare render output
        RenderOutput render_output;

        render_output.edges_score = std::move(accum_weights);
        render_output.image = core::Tensor();
        render_output.alpha = core::Tensor();
        render_output.width = width;
        render_output.height = height;

        return render_output;
    }
} // namespace lfs::training
