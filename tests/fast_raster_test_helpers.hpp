/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/error.hpp"
#include "core/splat_data.hpp"
#include "lfs/training/ops/fast_cuda.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "optimizer/render_output.hpp"
#include "training/rasterization/fast_rasterizer.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <expected>
#include <utility>

namespace lfs::training {

    // Camera-facing test context. The saved state is owned here, and forward_ctx
    // copies the scalar fields tests read off the CUDA frame.
    struct FastRasterizeContext {
        lfs::gpu_ops::FastSaved saved;
        CudaFastFrameView forward_ctx;
        cudaStream_t completion_stream = nullptr;
        int width = 0;
        int height = 0;

        void release_forward_context() noexcept {
            if (saved.backend) {
                cuda_fast_ops().release(saved);
            }
            forward_ctx = {};
            completion_stream = nullptr;
        }
    };

    [[nodiscard]] std::expected<std::pair<RenderOutput, FastRasterizeContext>, lfs::Error> fast_rasterize_forward(
        lfs::core::Camera& viewpoint_camera,
        lfs::core::SplatData& gaussian_model,
        lfs::core::Tensor& bg_color,
        int tile_x_offset = 0,
        int tile_y_offset = 0,
        int tile_width = 0,
        int tile_height = 0,
        bool mip_filter = false,
        const lfs::core::Tensor& bg_image = {},
        bool render_normal = false);

    void fast_rasterize_backward(
        FastRasterizeContext& ctx,
        const lfs::core::Tensor& grad_image,
        lfs::core::SplatData& gaussian_model,
        AdamOptimizer& optimizer,
        const lfs::core::Tensor& grad_alpha_extra = {},
        const lfs::core::Tensor& pixel_error_map = {},
        DensificationType densification_type = DensificationType::None,
        int iteration = 0,
        const FastGSFusedExtraGradients& fused_extra_gradients = {},
        const lfs::core::Tensor& grad_depth = {},
        const lfs::core::Tensor& grad_normal = {});

    inline RenderOutput fast_rasterize(
        lfs::core::Camera& camera, lfs::core::SplatData& model, lfs::core::Tensor& bg_color) {
        const auto& ops = cuda_fast_ops();
        lfs::gpu_ops::FastSaved saved{.backend = ops.create()};
        return fast_infer(ops, saved, camera, model, bg_color, false);
    }

} // namespace lfs::training
