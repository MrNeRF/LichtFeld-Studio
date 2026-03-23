/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "edge_rasterizer.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/tensor/internal/tensor_serialization.hpp"
#include "training/kernels/grad_alpha.hpp"
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>

namespace lfs::training {

    std::expected<std::pair<RenderOutput, FastRasterizeContext>, std::string> edge_rasterize_forward(
        core::Camera& viewpoint_camera,
        core::SplatData& gaussian_model,
        core::Tensor& bg_color,
        const lfs::core::Tensor& pixel_weights,
        int tile_x_offset,
        int tile_y_offset,
        int tile_width,
        int tile_height,
        bool mip_filter,
        const core::Tensor& bg_image) {
        // Get camera parameters
        const int full_width = viewpoint_camera.image_width();
        const int full_height = viewpoint_camera.image_height();

        // Determine tile dimensions (tile_width/height=0 means render full image)
        const int width = (tile_width > 0) ? tile_width : full_width;
        const int height = (tile_height > 0) ? tile_height : full_height;

        auto [fx, fy, cx, cy] = viewpoint_camera.get_intrinsics();

        // Adjust camera center point for tile rendering
        // When rendering a tile at offset, the principal point shifts
        const float cx_adjusted = cx - static_cast<float>(tile_x_offset);
        const float cy_adjusted = cy - static_cast<float>(tile_y_offset);

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
        const float* cam_position_ptr = viewpoint_camera.cam_position_ptr();

        const int n_primitives = static_cast<int>(means.shape()[0]);

        if (n_primitives == 0) {
            return std::unexpected("n_primitives is 0 - model has no gaussians");
        }

        // Pre-allocate output tensors (reused across iterations)
        thread_local core::Tensor alpha;
        thread_local int last_width = -1;
        thread_local int last_height = -1;

        // Only reallocate if dimensions changed
        if (last_width != width || last_height != height) {
            alpha = core::Tensor::empty({1, static_cast<size_t>(height), static_cast<size_t>(width)});
            last_width = width;
            last_height = height;
        }

        // Input pixel_weights pointer and output accum_weights
        auto pixel_weights_contig = pixel_weights.contiguous();
        const float* pixel_weights_ptr = pixel_weights_contig.ptr<float>();

        auto accum_weights = core::Tensor::zeros(
            {static_cast<size_t>(n_primitives)}, core::Device::CUDA, core::DataType::Float32);
        float* accum_weights_out = accum_weights.ptr<float>();

        // Call forward_raw with raw pointers (no PyTorch wrappers)
        // Use adjusted cx/cy for tile rendering
        edge_compute::rasterization::ForwardContext forward_ctx;
        try {
            forward_ctx = edge_compute::rasterization::edge_forward_raw(
                means.ptr<float>(),
                raw_scales.ptr<float>(),
                raw_rotations.ptr<float>(),
                raw_opacities.ptr<float>(),
                w2c_ptr,
                cam_position_ptr,
                alpha.ptr<float>(),
                n_primitives,
                width,
                height,
                fx,
                fy,
                cx_adjusted, // Use adjusted cx for tile offset
                cy_adjusted, // Use adjusted cy for tile offset
                near_plane,
                far_plane,
                pixel_weights_ptr,
                accum_weights_out);
        } catch (const std::exception& e) {
        }

        // Check if forward failed due to OOM
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
        render_output.alpha = alpha;
        render_output.width = width;
        render_output.height = height;

        return std::pair{render_output, FastRasterizeContext{}};
    }
} // namespace lfs::training
