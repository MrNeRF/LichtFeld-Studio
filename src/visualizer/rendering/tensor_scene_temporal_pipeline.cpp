/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "tensor_scene_temporal_pipeline.hpp"

#include "core/tensor.hpp"
#include "rendering/scene_depth_contract.hpp"
#include "tensor_scene_temporal.hpp"

#include <algorithm>
#include <array>
#include <cstring>

namespace lfs::vis {
    namespace {
        template <class T>
        std::array<T, 16> matrixArray(const glm::mat4& matrix) {
            std::array<T, 16> result{};
            std::memcpy(result.data(), &matrix, sizeof(matrix));
            return result;
        }

        lfs::Error temporalError(std::string detail) {
            return lfs::make_error({
                .code = lfs::ErrorCode::InvalidArgument,
                .domain = lfs::ErrorDomain::Rendering,
                .detail = std::move(detail),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }
    } // namespace

    struct TensorSceneTemporalPipeline::Impl {
        explicit Impl(const lfs::core::GpuBackend backend) : kernels(backend) {}
        lfs::rendering::TensorSceneTemporalKernels kernels;
        SceneTemporalCoordinator coordinator;
        std::array<std::shared_ptr<lfs::core::Tensor>,
                   static_cast<std::size_t>(TemporalViewId::Count)>
            color_history, depth_history;
        std::array<std::array<std::shared_ptr<core::Tensor>, 2>, size_t(TemporalViewId::Count)> depth_buffers;
        std::array<size_t, size_t(TemporalViewId::Count)> depth_cursor{};

        static std::size_t index(const TemporalViewId view) {
            return static_cast<std::size_t>(view);
        }
    };

    TensorSceneTemporalPipeline::TensorSceneTemporalPipeline(
        const lfs::core::GpuBackend backend)
        : impl_(std::make_unique<Impl>(backend)) {}
    TensorSceneTemporalPipeline::~TensorSceneTemporalPipeline() = default;

    lfs::Result<std::shared_ptr<lfs::core::Tensor>> TensorSceneTemporalPipeline::spatial(
        const std::shared_ptr<lfs::core::Tensor>& color,
        const glm::ivec2 render_extent, const glm::ivec2 output_extent) {
        if (!color || !color->is_valid() || color->ndim() != 3 || color->size(2) < 4 ||
            render_extent.x <= 0 || render_extent.y <= 0 ||
            output_extent.x <= 0 || output_extent.y <= 0) {
            return lfs::Result<std::shared_ptr<lfs::core::Tensor>>(
                temporalError("Tensor spatial reconstruction inputs are incomplete"));
        }
        const auto allocation_width = static_cast<std::uint32_t>(color->size(1));
        const auto allocation_height = static_cast<std::uint32_t>(color->size(0));
        const glm::vec2 current_scale{
            static_cast<float>(render_extent.x) / allocation_width,
            static_cast<float>(render_extent.y) / allocation_height};
        const glm::vec2 current_clamp{
            (static_cast<float>(render_extent.x) - 0.5f) / allocation_width,
            (static_cast<float>(render_extent.y) - 0.5f) / allocation_height};
        lfs::rendering::TensorSceneResolveParameters parameters{
            .extents = {static_cast<std::uint32_t>(render_extent.x),
                        static_cast<std::uint32_t>(render_extent.y),
                        static_cast<std::uint32_t>(output_extent.x),
                        static_cast<std::uint32_t>(output_extent.y)},
            .current_layout = {allocation_width, allocation_height,
                               static_cast<std::uint32_t>(color->size(2)),
                               color->dtype() == lfs::core::DataType::Float32 ? 1u : 0u},
            .current_uv = {current_scale.x, current_scale.y,
                           current_clamp.x, current_clamp.y},
        };
        lfs::core::Tensor output;
        if (auto result = impl_->kernels.spatial(*color, parameters, output); !result)
            return lfs::Result<std::shared_ptr<lfs::core::Tensor>>(std::move(result).error());
        return std::make_shared<lfs::core::Tensor>(std::move(output));
    }

    lfs::Result<TensorSceneTemporalResult> TensorSceneTemporalPipeline::resolve(
        const TensorSceneTemporalRequest& request) {
        if (static_cast<size_t>(request.view) >= static_cast<size_t>(TemporalViewId::Count) ||
            !request.color || !request.depth ||
            request.frame.view.size != request.render_extent ||
            request.frame.output_extent != request.output_extent ||
            request.render_extent.x <= 0 || request.render_extent.y <= 0 ||
            request.output_extent.x <= 0 || request.output_extent.y <= 0) {
            return lfs::Result<TensorSceneTemporalResult>(
                temporalError("Tensor temporal reconstruction inputs are incomplete"));
        }
        const SceneTemporalRequest temporal{
            .view = request.view,
            .requirements = {
                .depth = true,
                .motion = true,
                .jitter = !request.frame.view.orthographic,
                .history_color = true,
                .history_depth = true,
            },
            .frame = request.frame,
            .render_extent = request.render_extent,
            .output_extent = request.output_extent,
        };
        const auto prepared = impl_->coordinator.prepare(temporal);
        if (!prepared.active()) {
            return lfs::Result<TensorSceneTemporalResult>(
                temporalError("Tensor temporal reconstruction request is invalid"));
        }
        const auto projections = makeTemporalMotionViewProjectionPair(prepared.frame);
        if (!projections) {
            impl_->coordinator.discard(prepared, TemporalResetReason::Projection);
            return lfs::Result<TensorSceneTemporalResult>(
                temporalError("Tensor temporal reconstruction projection is unavailable"));
        }

        lfs::core::Tensor motion;
        lfs::rendering::TensorSceneMotionParameters motion_params{
            .inverse_current_view_projection = matrixArray<float>(glm::inverse(projections->current)),
            .previous_view_projection = matrixArray<float>(projections->previous),
            .render_info = {static_cast<std::uint32_t>(request.render_extent.x),
                            static_cast<std::uint32_t>(request.render_extent.y),
                            request.flip_y ? 1u : 0u,
                            request.frame.view.orthographic ? 2u : 1u},
            .depth_info = {request.frame.view.near_plane, request.frame.view.far_plane,
                           request.flip_y ? 1.0f : 0.0f, float(request.depth->size(1))},
        };
        if (auto result = impl_->kernels.motion(*request.depth, motion_params, motion); !result) {
            impl_->coordinator.discard(prepared, TemporalResetReason::ResolveFailure);
            return lfs::Result<TensorSceneTemporalResult>(std::move(result).error());
        }

        const auto slot = Impl::index(request.view);
        const bool history_valid = prepared.history.matches(prepared.plan) &&
                                   impl_->color_history[slot] && impl_->depth_history[slot];
        const auto allocation_width = static_cast<std::uint32_t>(request.color->size(1));
        const auto allocation_height = static_cast<std::uint32_t>(request.color->size(0));
        const glm::vec2 current_scale{
            static_cast<float>(request.render_extent.x) / allocation_width,
            static_cast<float>(request.render_extent.y) / allocation_height};
        const glm::vec2 current_clamp{
            (static_cast<float>(request.render_extent.x) - 0.5f) / allocation_width,
            (static_cast<float>(request.render_extent.y) - 0.5f) / allocation_height};
        const glm::vec2 current_jitter = sceneTemporalJitterPixels(
            prepared.frame.current_jitter, request.render_extent, request.flip_y);
        const glm::vec2 previous_jitter = sceneTemporalJitterPixels(
            prepared.frame.previous_jitter, request.render_extent, request.flip_y);
        lfs::rendering::TensorSceneResolveParameters resolve_params{
            .extents = {static_cast<std::uint32_t>(request.render_extent.x),
                        static_cast<std::uint32_t>(request.render_extent.y),
                        static_cast<std::uint32_t>(request.output_extent.x),
                        static_cast<std::uint32_t>(request.output_extent.y)},
            .current_layout = {allocation_width, allocation_height,
                               static_cast<std::uint32_t>(request.color->size(2)),
                               request.color->dtype() == lfs::core::DataType::Float32 ? 1u : 0u},
            .control = {history_valid ? 1.0f : 0.0f,
                        sceneTemporalHistoryWeight(request.settings.history_weight,
                                                   prepared.frame.sequence),
                        std::max(0.0f, request.settings.motion_rejection_pixels), 0.0f},
            .current_uv = {current_scale.x, current_scale.y,
                           current_clamp.x, current_clamp.y},
            .depth_control = {history_valid ? 1.0f : 0.0f,
                              std::max(0.0f, request.settings.depth_relative_threshold),
                              std::max(0.0f, request.settings.depth_absolute_threshold),
                              request.frame.view.far_plane},
            .jitter_pixels = {current_jitter.x, current_jitter.y,
                              previous_jitter.x, previous_jitter.y},
            .reconstruction = {std::clamp(request.settings.current_sharpness, 0.0f, 0.25f),
                               std::max(0.0f, request.settings.motion_confidence_pixels), 0.0f, 0.0f},
        };
        lfs::core::Tensor resolved;
        if (auto result = impl_->kernels.resolve(
                *request.color, history_valid ? impl_->color_history[slot].get() : nullptr,
                motion, history_valid ? request.depth.get() : nullptr,
                history_valid ? impl_->depth_history[slot].get() : nullptr,
                resolve_params, resolved);
            !result) {
            impl_->coordinator.discard(prepared, TemporalResetReason::ResolveFailure);
            return lfs::Result<TensorSceneTemporalResult>(std::move(result).error());
        }
        impl_->color_history[slot] = std::make_shared<lfs::core::Tensor>(resolved);
        // A public caller may mutate or recycle its input before the next frame.
        auto& depth_buffer = impl_->depth_buffers[slot][impl_->depth_cursor[slot]];
        if (!depth_buffer || depth_buffer->shape() != request.depth->shape())
            depth_buffer = std::make_shared<core::Tensor>(request.depth->clone());
        else
            depth_buffer->copy_from(*request.depth);
        impl_->depth_history[slot] = depth_buffer;
        impl_->depth_cursor[slot] ^= 1;
        if (!impl_->coordinator.commit(prepared, SceneHistoryStorage::Tensor,
                                       SceneHistoryStorage::Tensor)) {
            reset(request.view, TemporalResetReason::ResolveFailure);
            return lfs::Result<TensorSceneTemporalResult>(
                temporalError("Tensor temporal reconstruction history commit failed"));
        }
        return TensorSceneTemporalResult{
            .color = impl_->color_history[slot],
            .sequence = prepared.frame.sequence + 1,
            .reset_reasons = prepared.frame.reset_reasons,
        };
    }

    void TensorSceneTemporalPipeline::reset(const TemporalViewId view,
                                            const TemporalResetReason reason) {
        if (Impl::index(view) >= impl_->color_history.size()) return;
        impl_->coordinator.reset(view, reason);
        impl_->color_history[Impl::index(view)].reset();
        impl_->depth_history[Impl::index(view)].reset();
    }

    void TensorSceneTemporalPipeline::resetAll(const TemporalResetReason reason) {
        impl_->coordinator.resetAll(reason);
        impl_->color_history.fill({});
        impl_->depth_history.fill({});
    }
} // namespace lfs::vis
