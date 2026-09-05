/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "fastgs/rasterization/include/pop_scores.h"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "fast_rasterizer.hpp"
#include "gsplat/PopScores.h"
#include "gsplat_rasterizer.hpp"
#include <format>
#include <limits>

namespace lfs::training {
    namespace {
        lfs::Status invalid_pop(std::string detail) {
            return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::InvalidArgument,
                .domain = lfs::ErrorDomain::Rendering,
                .user_message = "GaussianPOP scoring inputs do not match the live forward context.",
                .detail = std::move(detail),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            }));
        }
        lfs::Status pop_runtime_error(const std::exception& error) {
            return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::Internal,
                .domain = lfs::ErrorDomain::Rendering,
                .user_message = "GaussianPOP score accumulation failed.",
                .detail = error.what(),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            }));
        }
        lfs::Status validate_pop(const core::Tensor& scores, size_t n, uint32_t width, uint32_t height,
                                 const core::Tensor& background, const core::Tensor& image) {
            if (!scores.is_valid() || scores.ndim() != 2 || scores.shape()[0] != n || scores.shape()[1] != sizeof(double) ||
                scores.dtype() != core::DataType::UInt8 || scores.device() != core::Device::CUDA || !scores.is_contiguous() ||
                reinterpret_cast<uintptr_t>(scores.data_ptr()) % alignof(double) != 0)
                return invalid_pop(std::format("scores must be aligned contiguous CUDA UInt8 [{},8] containing FP64 values (valid={}, numel={}, ndim={}, dtype={}, device={}, contiguous={})", n,
                                               scores.is_valid(), scores.is_valid() ? scores.numel() : 0, scores.is_valid() ? scores.ndim() : 0,
                                               scores.is_valid() ? core::dtype_name(scores.dtype()) : "invalid", scores.is_valid() ? int(scores.device()) : -1,
                                               scores.is_valid() && scores.is_contiguous()));
            const uint64_t pixels = uint64_t(width) * height;
            if (pixels == 0 || pixels > std::numeric_limits<int>::max())
                return invalid_pop(std::format("image pixel count must fit positive int (width={}, height={})", width, height));
            for (const auto* tensor : {&background, &image}) {
                if (!tensor->is_valid() || tensor->is_empty())
                    continue;
                const size_t expected = tensor == &image ? 3 * pixels : 3;
                if (tensor->device() != core::Device::CUDA || tensor->dtype() != core::DataType::Float32 ||
                    !tensor->is_contiguous() || tensor->numel() != expected ||
                    (tensor == &image && (tensor->ndim() != 3 || tensor->shape()[0] != 3 ||
                                          tensor->shape()[1] != height || tensor->shape()[2] != width)))
                    return invalid_pop(std::format("background must be contiguous CUDA Float32 with {} elements and image layout [3,{},{}] (numel={}, ndim={}, dtype={}, device={})",
                                                   expected, height, width, tensor->numel(), tensor->ndim(), core::dtype_name(tensor->dtype()), int(tensor->device())));
            }
            return {};
        }
        const float* optional_float(const core::Tensor& tensor) {
            return tensor.is_valid() && !tensor.is_empty() ? tensor.ptr<float>() : nullptr;
        }
        void prepare_background(const core::Tensor& background, const core::Tensor& image, cudaStream_t stream) {
            for (const auto* tensor : {&background, &image}) {
                if (tensor->is_valid() && !tensor->is_empty())
                    tensor->sync_to_stream(stream);
            }
        }
    } // namespace

    lfs::Status fast_accumulate_pop_scores(const FastRasterizeContext& ctx, core::Tensor& scores) try {
        const size_t n = ctx.means.is_valid() && ctx.means.ndim() > 0 ? ctx.means.shape()[0] : 0;
        if (auto status = validate_pop(scores, n, ctx.width, ctx.height, ctx.bg_color, ctx.bg_image); !status)
            return status;
        if (!ctx.forward_ctx.success || ctx.forward_ctx.n_instances < 0 || ctx.forward_ctx.n_visible < 0 ||
            static_cast<size_t>(ctx.forward_ctx.n_visible) > n || (ctx.forward_ctx.n_instances > 0 && (!ctx.forward_ctx.per_primitive_buffers || !ctx.forward_ctx.per_tile_buffers || !ctx.forward_ctx.sorted_primitive_indices || !ctx.forward_ctx.primitive_work_indices)))
            return invalid_pop(std::format("FastGS context must be live (success={}, instances={}, visible={})",
                                           ctx.forward_ctx.success, ctx.forward_ctx.n_instances, ctx.forward_ctx.n_visible));
        const auto stream = core::prepare_inputs_for_stream({&scores}, ctx.forward_ctx.stream ? ctx.forward_ctx.stream : core::getCurrentCUDAStream());
        prepare_background(ctx.bg_color, ctx.bg_image, stream);
        fast_lfs::rasterization::accumulate_pop_scores(ctx.forward_ctx, ctx.width, ctx.height,
                                                       optional_float(ctx.bg_color), optional_float(ctx.bg_image), reinterpret_cast<double*>(scores.data_ptr()), stream);
        scores.set_stream(stream);
        return {};
    } catch (const lfs::Exception& e) {
        return lfs::Status::failure(e.error());
    } catch (const std::exception& e) {
        // LFS-CENSUS-OK(empty-catch): pop_runtime_error() returns a typed rendering error with the exception detail.
        return pop_runtime_error(e);
    }

    lfs::Status gsplat_accumulate_pop_scores(const GsplatRasterizeContext& ctx, core::Tensor& scores) try {
        if (auto status = validate_pop(scores, ctx.N, ctx.image_width, ctx.image_height, ctx.bg_color, ctx.bg_image); !status)
            return status;
        if (ctx.channels < 3 || ctx.channels > 4 || ctx.tile_size == 0 || ctx.n_isects < 0 || !ctx.viewmat_ptr || !ctx.K_ptr ||
            (ctx.n_isects > 0 && (!ctx.colors_ptr || !ctx.tile_offsets_ptr || !ctx.flatten_ids_ptr)))
            return invalid_pop(std::format("gsplat RGB context must be live (channels={}, tile_size={}, intersections={})", ctx.channels, ctx.tile_size, ctx.n_isects));
        if (ctx.n_isects == 0)
            return {};
        for (const auto* tensor : {&ctx.means, &ctx.quats, &ctx.scales, &ctx.opacities}) {
            const size_t expected = tensor == &ctx.quats ? size_t(ctx.N) * 4 : tensor == &ctx.opacities ? ctx.N
                                                                                                        : size_t(ctx.N) * 3;
            if (!tensor->is_valid() || tensor->numel() != expected || tensor->device() != core::Device::CUDA ||
                tensor->dtype() != core::DataType::Float32 || !tensor->is_contiguous())
                return invalid_pop(std::format("gsplat input must be contiguous CUDA Float32 with {} elements (valid={}, numel={})", expected,
                                               tensor->is_valid(), tensor->is_valid() ? tensor->numel() : 0));
        }
        const gsplat_lfs::PopScoreInputs inputs{
            ctx.means.ptr<float>(), ctx.quats.ptr<float>(), ctx.scales.ptr<float>(), ctx.opacities.ptr<float>(), ctx.colors_ptr,
            optional_float(ctx.bg_color), optional_float(ctx.bg_image), ctx.viewmat_ptr, ctx.K_ptr,
            ctx.radial_ptr, ctx.tangential_ptr, ctx.thin_prism_ptr, ctx.tile_offsets_ptr, ctx.flatten_ids_ptr,
            ctx.image_width, ctx.image_height, ctx.tile_size, ctx.channels, ctx.camera_model};
        const auto stream = core::prepare_inputs_for_stream({&scores, &ctx.means, &ctx.quats, &ctx.scales, &ctx.opacities}, ctx.stream ? ctx.stream : core::getCurrentCUDAStream());
        prepare_background(ctx.bg_color, ctx.bg_image, stream);
        gsplat_lfs::launch_pop_scores(inputs, reinterpret_cast<double*>(scores.data_ptr()), stream);
        scores.set_stream(stream);
        return {};
    } catch (const lfs::Exception& e) {
        return lfs::Status::failure(e.error());
    } catch (const std::exception& e) {
        // LFS-CENSUS-OK(empty-catch): pop_runtime_error() returns a typed rendering error with the exception detail.
        return pop_runtime_error(e);
    }
} // namespace lfs::training
