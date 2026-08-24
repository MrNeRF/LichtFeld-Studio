/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_scene_temporal_pipeline.hpp"

#include "window/vulkan_context.hpp"

namespace lfs::vis {
    namespace {
        [[nodiscard]] constexpr bool sameExtents(
            const VulkanSceneTemporalPipelineRequest& request) noexcept {
            return request.motion.render_extent == request.temporal.render_extent &&
                   request.resolve.render_extent == request.temporal.render_extent &&
                   request.resolve.output_extent == request.temporal.output_extent;
        }

        [[nodiscard]] constexpr bool sameView(
            const VulkanSceneTemporalPipelineRequest& request) noexcept {
            return request.resolve.view == request.temporal.view &&
                   (!request.resolve.current_depth.enabled ||
                    request.resolve.current_depth.view == request.temporal.view);
        }

        [[nodiscard]] constexpr bool sameDepthSource(
            const VulkanSceneTemporalPipelineRequest& request) noexcept {
            if (!request.resolve.current_depth.enabled)
                return true;
            const auto& motion = request.motion.depth;
            const auto& history = request.resolve.current_depth.depth;
            return request.resolve.current_depth.current_depth_view == request.motion.depth_view &&
                   request.resolve.current_depth.current_depth_layout ==
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL &&
                   motion.encoding == history.encoding && motion.storage == history.storage &&
                   motion.width == history.width && motion.height == history.height &&
                   motion.near_plane == history.near_plane &&
                   motion.far_plane == history.far_plane &&
                   motion.orthographic == history.orthographic &&
                   motion.flip_y == history.flip_y;
        }
    } // namespace

    bool validVulkanSceneTemporalPipelineRequest(
        const VulkanSceneTemporalPipelineRequest& request) noexcept {
        const auto plan = makeSceneTemporalPlan(request.temporal.requirements,
                                                request.temporal.render_extent,
                                                request.temporal.output_extent);
        if (!plan.active())
            return plan.valid() && request.temporal.render_extent == glm::ivec2(0) &&
                   request.temporal.output_extent == glm::ivec2(0) &&
                   !request.motion.enabled && !request.resolve.enabled;
        if (!plan.valid() || !validTemporalViewId(request.temporal.view) ||
            request.temporal.frame.view.size != request.temporal.render_extent ||
            request.temporal.frame.output_extent != request.temporal.output_extent ||
            !sameExtents(request) || !sameView(request) || !sameDepthSource(request))
            return false;
        if (!request.temporal.requirements.depth || !request.temporal.requirements.motion ||
            request.temporal.requirements.jitter != request.motion.includes_jitter ||
            !request.temporal.requirements.history_color || !request.motion.enabled ||
            !canRecordVulkanSceneMotion(request.motion) ||
            request.resolve.current_color_view == VK_NULL_HANDLE ||
            request.resolve.motion_view != VK_NULL_HANDLE || !request.resolve.enabled ||
            !validTemporalDepthInputs(request.resolve))
            return false;
        return request.temporal.requirements.history_depth ==
               request.resolve.current_depth.enabled;
    }

    struct VulkanSceneTemporalPipeline::Impl {
        VulkanContext* context = nullptr;
        SceneTemporalCoordinator coordinator;
        VulkanSceneMotionPass motion;
        VulkanSceneTemporalResolvePass resolve;
        bool motion_initialized = false;
        bool resolve_initialized = false;

        [[nodiscard]] bool init(VulkanContext& ctx) {
            context = &ctx;
            return ctx.device() != VK_NULL_HANDLE && ctx.allocator() != VK_NULL_HANDLE;
        }

        void reset(const TemporalViewId view, const TemporalResetReason reason) {
            coordinator.reset(view, reason);
            resolve.reset(view);
        }

        void resetAll(const TemporalResetReason reason) {
            coordinator.resetAll(reason);
            resolve.resetAll();
        }

        [[nodiscard]] VulkanSceneTemporalPipelineResult fail(
            const PreparedSceneTemporalFrame& prepared,
            const VulkanSceneTemporalPipelineStatus status,
            const TemporalResetReason reason) {
            coordinator.discard(prepared, reason);
            resolve.reset(prepared.view);
            return {.status = status, .view = prepared.view};
        }

        [[nodiscard]] VulkanSceneTemporalPipelineResult record(
            const VkCommandBuffer command_buffer,
            const VulkanSceneTemporalPipelineRequest& request) {
            if (!validVulkanSceneTemporalPipelineRequest(request))
                return {.status = VulkanSceneTemporalPipelineStatus::InvalidRequest,
                        .view = request.temporal.view};

            if (!request.temporal.requirements.any()) {
                reset(request.temporal.view, TemporalResetReason::HistoryDisabled);
                return {.status = VulkanSceneTemporalPipelineStatus::Inactive,
                        .view = request.temporal.view};
            }
            if (command_buffer == VK_NULL_HANDLE)
                return {.status = VulkanSceneTemporalPipelineStatus::InvalidRequest,
                        .view = request.temporal.view};

            const auto prepared = coordinator.prepare(request.temporal);
            if (!prepared.active()) {
                resolve.reset(request.temporal.view);
                return {.status = VulkanSceneTemporalPipelineStatus::Inactive,
                        .view = request.temporal.view};
            }
            const auto view_projections = makeTemporalViewProjectionPair(prepared.frame);
            if (!view_projections) {
                return fail(prepared,
                            VulkanSceneTemporalPipelineStatus::InvalidRequest,
                            TemporalResetReason::Projection);
            }

            if (!motion_initialized)
                motion_initialized = motion.init(*context);
            if (!motion_initialized)
                return fail(prepared,
                            VulkanSceneTemporalPipelineStatus::MotionUnavailable,
                            TemporalResetReason::RuntimeUnavailable);
            const auto motion_slot =
                temporalMotionResourceSlot(request.frame_slot, request.temporal.view);
            if (!motion_slot)
                return fail(prepared,
                            VulkanSceneTemporalPipelineStatus::InvalidRequest,
                            TemporalResetReason::InvalidInput);
            auto motion_params = request.motion;
            motion_params.inverse_current_view_projection =
                glm::inverse(view_projections->current);
            motion_params.previous_view_projection = view_projections->previous;
            if (!motion.record(command_buffer, motion_params, *motion_slot))
                return fail(prepared,
                            VulkanSceneTemporalPipelineStatus::MotionFailure,
                            TemporalResetReason::ResolveFailure);

            if (!resolve_initialized)
                resolve_initialized = resolve.init(*context);
            if (!resolve_initialized)
                return fail(prepared,
                            VulkanSceneTemporalPipelineStatus::ResolveUnavailable,
                            TemporalResetReason::RuntimeUnavailable);
            auto resolve_params = request.resolve;
            resolve_params.motion_view = motion.motionView(*motion_slot);
            resolve_params.motion_layout = VK_IMAGE_LAYOUT_GENERAL;
            resolve_params.current_jitter_ndc = prepared.frame.current_jitter;
            resolve_params.previous_jitter_ndc = prepared.frame.previous_jitter;
            resolve_params.jitter_flip_y = request.motion.flip_y;
            resolve_params.sequence = prepared.frame.sequence;
            resolve_params.history_valid = prepared.history.matches(prepared.plan);
            if (!resolve.record(command_buffer, resolve_params))
                return fail(prepared,
                            VulkanSceneTemporalPipelineStatus::ResolveFailure,
                            TemporalResetReason::ResolveFailure);

            const auto history = resolve.contract(prepared.view);
            const auto output = resolve.outputView(prepared.view);
            if (!history.matches(prepared.plan) || output == VK_NULL_HANDLE)
                return fail(prepared,
                            VulkanSceneTemporalPipelineStatus::CommitFailure,
                            TemporalResetReason::ResolveFailure);
            if (!coordinator.commit(prepared, history.color_storage, history.depth_storage))
                return fail(prepared,
                            VulkanSceneTemporalPipelineStatus::CommitFailure,
                            TemporalResetReason::ResolveFailure);
            return {.status = VulkanSceneTemporalPipelineStatus::Resolved,
                    .view = prepared.view,
                    .sequence = history.sequence,
                    .output_view = output,
                    .history = history};
        }
    };

    VulkanSceneTemporalPipeline::VulkanSceneTemporalPipeline() = default;
    VulkanSceneTemporalPipeline::~VulkanSceneTemporalPipeline() = default;
    VulkanSceneTemporalPipeline::VulkanSceneTemporalPipeline(
        VulkanSceneTemporalPipeline&&) noexcept = default;
    VulkanSceneTemporalPipeline& VulkanSceneTemporalPipeline::operator=(
        VulkanSceneTemporalPipeline&&) noexcept = default;

    bool VulkanSceneTemporalPipeline::init(VulkanContext& context) {
        if (!impl_)
            impl_ = std::make_unique<Impl>();
        return impl_->init(context);
    }

    VulkanSceneTemporalPipelineResult VulkanSceneTemporalPipeline::record(
        const VkCommandBuffer command_buffer,
        const VulkanSceneTemporalPipelineRequest& request) {
        return impl_ ? impl_->record(command_buffer, request)
                     : VulkanSceneTemporalPipelineResult{
                           .status = VulkanSceneTemporalPipelineStatus::InvalidRequest,
                           .view = request.temporal.view};
    }

    void VulkanSceneTemporalPipeline::reset(const TemporalViewId view,
                                            const TemporalResetReason reason) {
        if (impl_)
            impl_->reset(view, reason);
    }

    void VulkanSceneTemporalPipeline::resetAll(const TemporalResetReason reason) {
        if (impl_)
            impl_->resetAll(reason);
    }

    void VulkanSceneTemporalPipeline::shutdown() { impl_.reset(); }

    VkImageView VulkanSceneTemporalPipeline::outputView(const TemporalViewId view) const {
        return impl_ ? impl_->resolve.outputView(view) : VK_NULL_HANDLE;
    }

    SceneHistoryContract VulkanSceneTemporalPipeline::contract(const TemporalViewId view) const {
        return impl_ ? impl_->resolve.contract(view) : SceneHistoryContract{};
    }

    VulkanSceneTemporalResourceStats VulkanSceneTemporalPipeline::resourceStats() const {
        return impl_ ? impl_->resolve.resourceStats() : VulkanSceneTemporalResourceStats{};
    }
} // namespace lfs::vis
