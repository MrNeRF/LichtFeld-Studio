/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "fast_raster_test_helpers.hpp"

#include "core/tensor.hpp"
#include "core/tensor_cuda_interop.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace lfs::training {
    namespace {

        void publish_frame(FastRasterizeContext& ctx) {
            ctx.forward_ctx = cuda_fast_frame_view(ctx.saved);
            ctx.completion_stream = ctx.forward_ctx.completion_stream;
        }

        [[nodiscard]] lfs::core::Tensor device_f32(void* data, lfs::core::TensorShape shape) {
            if (data == nullptr || shape.elements() == 0) {
                return {};
            }
            return lfs::core::Tensor::from_blob(
                data, std::move(shape), lfs::core::Device::GPU, lfs::core::DataType::Float32);
        }

    } // namespace

    std::expected<std::pair<RenderOutput, FastRasterizeContext>, lfs::Error> fast_rasterize_forward(
        lfs::core::Camera& viewpoint_camera,
        lfs::core::SplatData& gaussian_model,
        lfs::core::Tensor& bg_color,
        const int tile_x_offset,
        const int tile_y_offset,
        const int tile_width,
        const int tile_height,
        const bool mip_filter,
        const lfs::core::Tensor& bg_image,
        const bool render_normal) {
        const auto& ops = cuda_fast_ops();
        FastRasterizeContext ctx;
        ctx.saved.backend = ops.create();
        RenderOutput output;
        const auto result = fast_render(
            ops, ctx.saved, viewpoint_camera, gaussian_model, bg_color, tile_x_offset, tile_y_offset,
            tile_width, tile_height, mip_filter, bg_image, render_normal, true, output);
        if (result.code != lfs::gpu_ops::RasterResult::Code::Success) {
            return std::unexpected(fast_raster_error(result));
        }
        ctx.width = output.width;
        ctx.height = output.height;
        publish_frame(ctx);
        return std::pair{std::move(output), std::move(ctx)};
    }

    void fast_rasterize_backward(
        FastRasterizeContext& ctx,
        const lfs::core::Tensor& grad_image,
        lfs::core::SplatData& gaussian_model,
        AdamOptimizer& optimizer,
        const lfs::core::Tensor& grad_alpha_extra,
        const lfs::core::Tensor& pixel_error_map,
        const DensificationType densification_type,
        const int iteration,
        const FastGSFusedExtraGradients& fused_extra_gradients,
        const lfs::core::Tensor& grad_depth,
        const lfs::core::Tensor& grad_normal) {
        const auto& ops = cuda_fast_ops();
        const cudaStream_t stream = lfs::core::getCurrentCUDAStream();
        const auto prepared = optimizer.prepare_fastgs_fused_adam(iteration, stream);
        lfs::core::Tensor scale_loss = device_f32(fused_extra_gradients.scale_reg_loss_out, {1});
        lfs::core::Tensor opacity_loss = device_f32(fused_extra_gradients.opacity_reg_loss_out, {1});
        const auto sparsity_n = static_cast<std::size_t>(std::max(fused_extra_gradients.sparsity_n, 0));
        lfs::core::Tensor sparsity_sigmoid = device_f32(
            const_cast<float*>(fused_extra_gradients.sparsity_opa_sigmoid), {sparsity_n});
        lfs::core::Tensor sparsity_z = device_f32(
            const_cast<float*>(fused_extra_gradients.sparsity_z), {sparsity_n});
        lfs::core::Tensor sparsity_u = device_f32(
            const_cast<float*>(fused_extra_gradients.sparsity_u), {sparsity_n});
        const auto edge_n = ctx.width > 0 && ctx.height > 0
                                ? static_cast<std::size_t>(ctx.height) * static_cast<std::size_t>(ctx.width)
                                : 0;
        lfs::core::Tensor edge_map = device_f32(
            const_cast<float*>(fused_extra_gradients.edge_weight_map),
            {static_cast<std::size_t>(std::max(ctx.height, 0)), static_cast<std::size_t>(std::max(ctx.width, 0))});
        if (edge_n == 0) {
            edge_map = {};
        }
        lfs::core::Tensor edge_scores = device_f32(
            fused_extra_gradients.edge_score_out, {static_cast<std::size_t>(gaussian_model.size())});
        try {
            ops.backward(
                ctx.saved,
                {.image = grad_image, .alpha = grad_alpha_extra, .depth = grad_depth, .normal = grad_normal},
                gaussian_model._densification_info,
                pixel_error_map,
                edge_map,
                edge_scores,
                {.groups = prepared.groups,
                 .scale_reg_loss = scale_loss,
                 .opacity_reg_loss = opacity_loss,
                 .sparsity_sigmoid = sparsity_sigmoid,
                 .sparsity_z = sparsity_z,
                 .sparsity_u = sparsity_u,
                 .far_mask = prepared.far_mask,
                 .beta1 = prepared.beta1,
                 .beta2 = prepared.beta2,
                 .eps = prepared.eps,
                 .scale_reg_weight = fused_extra_gradients.scale_reg_weight,
                 .flatten_reg_weight = fused_extra_gradients.flatten_reg_weight,
                 .opacity_reg_weight = fused_extra_gradients.opacity_reg_weight,
                 .sparsity_rho = fused_extra_gradients.sparsity_rho,
                 .sparsity_grad_loss = fused_extra_gradients.sparsity_grad_loss,
                 .median_extent = prepared.median_extent,
                 .r_min = prepared.r_min,
                 .r_max = prepared.r_max,
                 .per_splat_mean_step = prepared.per_splat_mean_step},
                densification_type);
            if (fastgs_adam_enabled(prepared)) {
                optimizer.commit_fastgs_fused_adam(iteration);
            }
            publish_frame(ctx);
        } catch (...) {
            publish_frame(ctx);
            throw;
        }
    }

} // namespace lfs::training
