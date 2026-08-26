/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_scene_temporal_pipeline.hpp"

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
            !request.temporal.requirements.history_color || !request.motion.enabled ||
            !canRecordVulkanSceneMotion(request.motion) ||
            request.resolve.current_color_view == VK_NULL_HANDLE ||
            request.resolve.motion_view != VK_NULL_HANDLE || !request.resolve.enabled ||
            !validTemporalDepthInputs(request.resolve))
            return false;
        return request.temporal.requirements.history_depth ==
               request.resolve.current_depth.enabled;
    }
} // namespace lfs::vis
