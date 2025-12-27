/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "optimizer/render_output.hpp"
#include <expected>
#include <rasterization_api.h>
#include <string>

namespace lfs::training {
    // Forward pass context - holds intermediate buffers needed for backward
    struct FastRasterizeContext {
        lfs::core::Tensor image;
        lfs::core::Tensor alpha;
        lfs::core::Tensor bg_color; // Saved for alpha gradient computation

        // Gaussian parameters (saved to avoid re-fetching in backward)
        lfs::core::Tensor means;
        lfs::core::Tensor raw_scales;
        lfs::core::Tensor raw_rotations;
        lfs::core::Tensor raw_opacities;
        lfs::core::Tensor shN;

        const float* w2c_ptr = nullptr;
        const float* cam_position_ptr = nullptr;

        // Forward context (contains buffer pointers, frame_id, etc.)
        fast_lfs::rasterization::ForwardContext forward_ctx;

        int active_sh_bases;
        int total_bases_sh_rest;
        int width;
        int height;
        float focal_x;
        float focal_y;
        float center_x;
        float center_y;
        float near_plane;
        float far_plane;
        bool mip_filter = false;

        // Tile information (for tile-based training)
        int tile_x_offset = 0; // Horizontal offset of this tile
        int tile_y_offset = 0; // Vertical offset of this tile
        int tile_width = 0;    // Width of this tile (0 = full image)
        int tile_height = 0;   // Height of this tile (0 = full image)
    };

    // Explicit forward pass - returns render output and context for backward
    // Optional tile parameters for memory-efficient training (tile_width/height=0 means full image)
    std::expected<std::pair<RenderOutput, FastRasterizeContext>, std::string> fast_rasterize_forward(
        lfs::core::Camera& viewpoint_camera,
        lfs::core::SplatData& gaussian_model,
        lfs::core::Tensor& bg_color,
        int tile_x_offset = 0,
        int tile_y_offset = 0,
        int tile_width = 0,
        int tile_height = 0,
        bool mip_filter = false);

    // Backward pass with optional extra alpha gradient for masked training
    void fast_rasterize_backward(
        const FastRasterizeContext& ctx,
        const lfs::core::Tensor& grad_image,
        lfs::core::SplatData& gaussian_model,
        AdamOptimizer& optimizer,
        const lfs::core::Tensor& grad_alpha_extra = {});

    // Convenience wrapper for inference (no backward needed)
    inline RenderOutput fast_rasterize(
        lfs::core::Camera& viewpoint_camera,
        lfs::core::SplatData& gaussian_model,
        lfs::core::Tensor& bg_color) {
        auto result = fast_rasterize_forward(viewpoint_camera, gaussian_model, bg_color);
        if (!result) {
            throw std::runtime_error(result.error());
        }
        return result->first;
    }
} // namespace lfs::training
