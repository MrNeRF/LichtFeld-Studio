/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/splat_data.hpp"
#include "gsplat/Common.h"
#include "optimizer/adam_optimizer.hpp"
#include "optimizer/render_output.hpp"
#include <cstdint>
#include <cuda_runtime.h>
#include <expected>
#include <string>

namespace lfs::training {

    // Render modes for gsplat rasterizer
    enum class GsplatRenderMode {
        RGB = 0,   // RGB only
        D = 1,     // Depth only
        ED = 2,    // Expected depth
        RGB_D = 3, // RGB + depth
        RGB_ED = 4 // RGB + expected depth
    };

    // Forward pass context - holds raw pointers needed for backward (arena allocated)
    struct GsplatRasterizeContext {
        // Raw pointers to arena-allocated intermediate buffers
        float* render_colors_ptr = nullptr;  // [C, H, W, channels]
        float* render_alphas_ptr = nullptr;  // [C, H, W, 1]
        int32_t* radii_ptr = nullptr;        // [C, N, 2]
        float* means2d_ptr = nullptr;        // [C, N, 2]
        float* depths_ptr = nullptr;         // [C, N]
        float* colors_ptr = nullptr;         // [C, N, channels]
        int32_t* tile_offsets_ptr = nullptr; // [C, tile_height, tile_width]
        int32_t* last_ids_ptr = nullptr;     // [C, H, W]
        float* compensations_ptr = nullptr;  // [C, N] or nullptr

        // Internally allocated by gsplat (must cudaFree in backward)
        int64_t* isect_ids_ptr = nullptr;
        int32_t* flatten_ids_ptr = nullptr;
        int32_t n_isects = 0;

        // Saved input tensors (references, not copies)
        lfs::core::Tensor means;     // [N, 3]
        lfs::core::Tensor quats;     // [N, 4]
        lfs::core::Tensor scales;    // [N, 3]
        lfs::core::Tensor opacities; // [N]
        lfs::core::Tensor sh_coeffs; // [N, K, 3]
        lfs::core::Tensor bg_color;  // [3] or [C, 3]

        // Camera pointers (kept alive by K_tensor)
        const float* viewmat_ptr = nullptr; // [C, 4, 4]
        const float* K_ptr = nullptr;       // [C, 3, 3]
        lfs::core::Tensor K_tensor;         // Keeps K_ptr alive

        // Distortion coefficients
        const float* radial_ptr = nullptr;
        const float* tangential_ptr = nullptr;
        const float* thin_prism_ptr = nullptr;
        lfs::core::Tensor radial_cuda;
        lfs::core::Tensor tangential_cuda;
        lfs::core::Tensor thin_prism_cuda;

        // Dimensions
        uint32_t N = 0;
        uint32_t K_sh = 0;
        uint32_t channels = 0;
        uint32_t tile_width = 0;
        uint32_t tile_height = 0;

        // Settings
        uint32_t sh_degree = 0;
        uint32_t image_width = 0;
        uint32_t image_height = 0;
        uint32_t tile_size = 0;
        float eps2d = 0.0f;
        float near_plane = 0.0f;
        float far_plane = 0.0f;
        float radius_clip = 0.0f;
        float scaling_modifier = 1.0f;
        bool calc_compensations = false;
        GsplatRenderMode render_mode = GsplatRenderMode::RGB;
        CameraModelType camera_model = PINHOLE;

        // Memory arena frame ID (for releasing arena memory in backward)
        uint64_t frame_id = 0;

        // Tile-based training (0 = full image)
        int render_tile_x_offset = 0;
        int render_tile_y_offset = 0;
        int render_tile_width = 0;
        int render_tile_height = 0;
    };

    // Forward pass with optional tiling (tile_width/height=0 = full image)
    std::expected<std::pair<RenderOutput, GsplatRasterizeContext>, std::string> gsplat_rasterize_forward(
        lfs::core::Camera& viewpoint_camera,
        lfs::core::SplatData& gaussian_model,
        lfs::core::Tensor& bg_color,
        int tile_x_offset = 0,
        int tile_y_offset = 0,
        int tile_width = 0,
        int tile_height = 0,
        float scaling_modifier = 1.0f,
        bool antialiased = false,
        GsplatRenderMode render_mode = GsplatRenderMode::RGB,
        bool use_gut = false);

    // Explicit backward pass - computes gradients and accumulates into optimizer
    void gsplat_rasterize_backward(
        const GsplatRasterizeContext& ctx,
        const lfs::core::Tensor& grad_image,
        const lfs::core::Tensor& grad_alpha,
        lfs::core::SplatData& gaussian_model,
        AdamOptimizer& optimizer);

    // Convenience wrapper for inference (no backward needed)
    inline RenderOutput gsplat_rasterize(
        lfs::core::Camera& viewpoint_camera,
        lfs::core::SplatData& gaussian_model,
        lfs::core::Tensor& bg_color,
        float scaling_modifier = 1.0f,
        bool antialiased = false,
        GsplatRenderMode render_mode = GsplatRenderMode::RGB,
        bool use_gut = false) {
        auto result = gsplat_rasterize_forward(
            viewpoint_camera, gaussian_model, bg_color, 0, 0, 0, 0,
            scaling_modifier, antialiased, render_mode, use_gut);
        if (!result) {
            throw std::runtime_error(result.error());
        }
        // Free internally allocated buffers since backward won't be called
        if (result->second.isect_ids_ptr != nullptr) {
            cudaFree(result->second.isect_ids_ptr);
        }
        if (result->second.flatten_ids_ptr != nullptr) {
            cudaFree(result->second.flatten_ids_ptr);
        }
        // Release arena frame since no backward will be called
        auto& arena = core::GlobalArenaManager::instance().get_arena();
        arena.end_frame(result->second.frame_id);
        return result->first;
    }

} // namespace lfs::training
