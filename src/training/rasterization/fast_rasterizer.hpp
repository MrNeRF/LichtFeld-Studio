/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/splat_data.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "optimizer/render_output.hpp"
#include <expected>
#include <rasterization_api.h>
#include <string>
#include <utility>

namespace lfs::training {
    // Forward pass context - holds intermediate buffers needed for backward
    struct FastRasterizeContext {
        FastRasterizeContext() = default;
        ~FastRasterizeContext() {
            release_forward_context();
        }

        FastRasterizeContext(const FastRasterizeContext&) = delete;
        FastRasterizeContext& operator=(const FastRasterizeContext&) = delete;

        FastRasterizeContext(FastRasterizeContext&& other) noexcept {
            move_from(std::move(other));
        }

        FastRasterizeContext& operator=(FastRasterizeContext&& other) noexcept {
            if (this != &other) {
                release_forward_context();
                move_from(std::move(other));
            }
            return *this;
        }

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
        fast_lfs::rasterization::ForwardContext forward_ctx = {};

        int active_sh_bases = 0;
        int total_bases_sh_rest = 0;
        int width = 0;
        int height = 0;
        float focal_x = 0.0f;
        float focal_y = 0.0f;
        float center_x = 0.0f;
        float center_y = 0.0f;
        float near_plane = 0.0f;
        float far_plane = 0.0f;
        bool mip_filter = false;

        // Tile information (for tile-based training)
        int tile_x_offset = 0; // Horizontal offset of this tile
        int tile_y_offset = 0; // Vertical offset of this tile
        int tile_width = 0;    // Width of this tile (0 = full image)
        int tile_height = 0;   // Height of this tile (0 = full image)

        // Background image for per-pixel blending (optional, empty = use bg_color)
        lfs::core::Tensor bg_image;

        void set_forward_context(fast_lfs::rasterization::ForwardContext ctx) noexcept {
            release_forward_context();
            forward_ctx = ctx;
            owns_forward_context_ = ctx.success;
        }

        void release_forward_context() noexcept {
            if (!owns_forward_context_) {
                return;
            }
            owns_forward_context_ = false;
            fast_lfs::rasterization::release_forward_context(forward_ctx);
            forward_ctx = {};
        }

        void mark_forward_context_released() noexcept {
            owns_forward_context_ = false;
            forward_ctx = {};
        }

    private:
        bool owns_forward_context_ = false;

        void move_from(FastRasterizeContext&& other) noexcept {
            image = std::move(other.image);
            alpha = std::move(other.alpha);
            bg_color = std::move(other.bg_color);
            means = std::move(other.means);
            raw_scales = std::move(other.raw_scales);
            raw_rotations = std::move(other.raw_rotations);
            raw_opacities = std::move(other.raw_opacities);
            shN = std::move(other.shN);
            w2c_ptr = std::exchange(other.w2c_ptr, nullptr);
            cam_position_ptr = std::exchange(other.cam_position_ptr, nullptr);
            forward_ctx = std::exchange(other.forward_ctx, {});
            active_sh_bases = std::exchange(other.active_sh_bases, 0);
            total_bases_sh_rest = std::exchange(other.total_bases_sh_rest, 0);
            width = std::exchange(other.width, 0);
            height = std::exchange(other.height, 0);
            focal_x = std::exchange(other.focal_x, 0.0f);
            focal_y = std::exchange(other.focal_y, 0.0f);
            center_x = std::exchange(other.center_x, 0.0f);
            center_y = std::exchange(other.center_y, 0.0f);
            near_plane = std::exchange(other.near_plane, 0.0f);
            far_plane = std::exchange(other.far_plane, 0.0f);
            mip_filter = std::exchange(other.mip_filter, false);
            tile_x_offset = std::exchange(other.tile_x_offset, 0);
            tile_y_offset = std::exchange(other.tile_y_offset, 0);
            tile_width = std::exchange(other.tile_width, 0);
            tile_height = std::exchange(other.tile_height, 0);
            bg_image = std::move(other.bg_image);
            owns_forward_context_ = std::exchange(other.owns_forward_context_, false);
        }
    };

    struct FastGSFusedExtraGradients {
        float scale_reg_weight = 0.0f;
        float opacity_reg_weight = 0.0f;
        const float* sparsity_opa_sigmoid = nullptr;
        const float* sparsity_z = nullptr;
        const float* sparsity_u = nullptr;
        int sparsity_n = 0;
        float sparsity_rho = 0.0f;
        float sparsity_grad_loss = 0.0f;
    };

    // Explicit forward pass - returns render output and context for backward
    // Optional tile parameters for memory-efficient training (tile_width/height=0 means full image)
    // bg_image is optional - if provided, uses per-pixel background blending instead of solid color
    std::expected<std::pair<RenderOutput, FastRasterizeContext>, std::string> fast_rasterize_forward(
        lfs::core::Camera& viewpoint_camera,
        lfs::core::SplatData& gaussian_model,
        lfs::core::Tensor& bg_color,
        int tile_x_offset = 0,
        int tile_y_offset = 0,
        int tile_width = 0,
        int tile_height = 0,
        bool mip_filter = false,
        const lfs::core::Tensor& bg_image = {});

    // Backward pass with optional extra alpha gradient for masked training
    void fast_rasterize_backward(
        FastRasterizeContext& ctx,
        const lfs::core::Tensor& grad_image,
        lfs::core::SplatData& gaussian_model,
        AdamOptimizer& optimizer,
        const lfs::core::Tensor& grad_alpha_extra = {},
        const lfs::core::Tensor& pixel_error_map = {},
        DensificationType densification_type = DensificationType::None,
        int iteration = 0,
        const FastGSFusedExtraGradients& fused_extra_gradients = {});

    // Convenience wrapper for inference (no backward needed)
    inline RenderOutput fast_rasterize(
        lfs::core::Camera& viewpoint_camera,
        lfs::core::SplatData& gaussian_model,
        lfs::core::Tensor& bg_color,
        bool mip_filter = false,
        const lfs::core::Tensor& bg_image = {}) {
        auto result = fast_rasterize_forward(viewpoint_camera, gaussian_model, bg_color, 0, 0, 0, 0, mip_filter, bg_image);
        if (!result) {
            throw std::runtime_error(result.error());
        }
        RenderOutput output = std::move(result->first);
        result->second.release_forward_context();
        return output;
    }

    // Inference-only rasterization does not mutate the camera; this overload avoids
    // forcing callers with const camera handles to cast away constness at the call site.
    inline RenderOutput fast_rasterize(
        const lfs::core::Camera& viewpoint_camera,
        lfs::core::SplatData& gaussian_model,
        lfs::core::Tensor& bg_color,
        bool mip_filter = false,
        const lfs::core::Tensor& bg_image = {}) {
        return fast_rasterize(
            const_cast<lfs::core::Camera&>(viewpoint_camera),
            gaussian_model,
            bg_color,
            mip_filter,
            bg_image);
    }
} // namespace lfs::training
