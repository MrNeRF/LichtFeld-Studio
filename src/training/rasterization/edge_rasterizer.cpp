/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "edge_rasterizer.hpp"
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
        printf("Edge rasterizer.cpp\n");
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

        // Get Gaussian parameters
        auto& means = gaussian_model.means();
        auto& raw_opacities = gaussian_model.opacity_raw();
        auto& raw_scales = gaussian_model.scaling_raw();
        auto& raw_rotations = gaussian_model.rotation_raw();
        auto& sh0 = gaussian_model.sh0();
        auto& shN = gaussian_model.shN();

        const int sh_degree = gaussian_model.get_active_sh_degree();
        const int active_sh_bases = (sh_degree + 1) * (sh_degree + 1);

        constexpr float near_plane = 0.01f;
        constexpr float far_plane = 1e10f;

        // Get direct GPU pointers (tensors are already contiguous on CUDA)
        const float* w2c_ptr = viewpoint_camera.world_view_transform_ptr();
        const float* cam_position_ptr = viewpoint_camera.cam_position_ptr();

        const int n_primitives = static_cast<int>(means.shape()[0]);
        const int total_bases_sh_rest = (shN.is_valid() && shN.ndim() >= 2)
                                            ? static_cast<int>(shN.shape()[1])
                                            : 0;

        if (n_primitives == 0) {
            return std::unexpected("n_primitives is 0 - model has no gaussians");
        }

        // Pre-allocate output tensors (reused across iterations)
        thread_local core::Tensor image;
        thread_local core::Tensor alpha;
        thread_local core::Tensor output_image;
        thread_local int last_width = -1;
        thread_local int last_height = -1;

        // Only reallocate if dimensions changed
        if (last_width != width || last_height != height) {
            image = core::Tensor::empty({3, static_cast<size_t>(height), static_cast<size_t>(width)});
            alpha = core::Tensor::empty({1, static_cast<size_t>(height), static_cast<size_t>(width)});
            output_image = core::Tensor::empty({3, static_cast<size_t>(height), static_cast<size_t>(width)}, core::Device::CUDA);
            last_width = width;
            last_height = height;
        }

        // Input pixel_weights pointer and output accum_weights
        const float* pixel_weights_ptr = pixel_weights.ptr<float>();

        float* accum_weights_out;
        const size_t acumm_weights_size = sizeof(float) * n_primitives;

        cudaMalloc(&accum_weights_out, acumm_weights_size);
        cudaMemsetAsync(accum_weights_out, 0, acumm_weights_size, nullptr);



        // Call forward_raw with raw pointers (no PyTorch wrappers)
        // Use adjusted cx/cy for tile rendering
        edge_compute::rasterization::ForwardContext forward_ctx;
        try {
            forward_ctx = edge_compute::rasterization::edge_forward_raw(
                means.ptr<float>(),
                raw_scales.ptr<float>(),
                raw_rotations.ptr<float>(),
                raw_opacities.ptr<float>(),
                sh0.ptr<float>(),
                shN.ptr<float>(),
                w2c_ptr,
                cam_position_ptr,
                image.ptr<float>(),
                alpha.ptr<float>(),
                n_primitives,
                active_sh_bases,
                total_bases_sh_rest,
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

        // Prepare render output
        RenderOutput render_output;
        // output = image + (1 - alpha) * bg_color (or bg_image)
        // (output_image is pre-allocated above)

        const cudaStream_t stream = output_image.stream();

        render_output.image = output_image;
        render_output.alpha = alpha;
        render_output.width = width;
        render_output.height = height;

        return std::pair{render_output, FastRasterizeContext{}};
    }
} // namespace lfs::training
