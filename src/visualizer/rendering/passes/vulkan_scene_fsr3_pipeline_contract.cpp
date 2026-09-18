/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_scene_fsr3_pipeline.hpp"

#include <cmath>
#include <numbers>

namespace lfs::vis {
    bool validVulkanSceneFsr3PipelineRequest(
        const VulkanSceneFsr3PipelineRequest& request) noexcept {
        const auto& camera = request.temporal.temporal.frame.view;
        const float vertical_fov = amdFsr3CameraVerticalFovRadians(camera);
        const VulkanSceneDlssDepthParams depth{
            .enabled = true,
            .current_depth_view = request.temporal.motion.depth_view,
            .current_depth_layout =
                request.temporal.resolve.current_depth.current_depth_layout,
            .depth = request.temporal.motion.depth,
            .allocation_extent =
                request.temporal.resolve.current_depth.allocation_extent,
        };
        return !camera.orthographic && std::isfinite(camera.near_plane) &&
               std::isfinite(camera.far_plane) && camera.near_plane > 0.0f &&
               camera.far_plane > camera.near_plane && std::isfinite(vertical_fov) &&
               vertical_fov > 0.0f &&
               vertical_fov <= std::numbers::pi_v<float> &&
               amdFsr3SupportsOutputExtent(request.temporal.temporal.output_extent) &&
               validVulkanSceneTemporalPipelineRequest(request.temporal) &&
               request.color_image != VK_NULL_HANDLE &&
               request.color_format == VK_FORMAT_R8G8B8A8_UNORM &&
               request.depth_image != VK_NULL_HANDLE &&
               request.depth_format == VK_FORMAT_R32_SFLOAT &&
               canRecordVulkanSceneDlssDepth(depth) &&
               request.temporal.resolve.current_allocation_extent.x >=
                   request.temporal.temporal.render_extent.x &&
               request.temporal.resolve.current_allocation_extent.y >=
                   request.temporal.temporal.render_extent.y &&
               request.temporal.resolve.current_depth.allocation_extent.x >=
                   request.temporal.temporal.render_extent.x &&
               request.temporal.resolve.current_depth.allocation_extent.y >=
                   request.temporal.temporal.render_extent.y &&
               request.temporal.resolve.current_color_layout ==
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL &&
               request.temporal.resolve.current_depth.current_depth_layout ==
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    bool reusableVulkanSceneFsr3PipelineInput(
        const VulkanSceneFsr3PipelineRequest& current,
        const VulkanSceneFsr3PipelineRequest& previous) noexcept {
        if (current.color_generation == 0 || current.depth_generation == 0 ||
            current.temporal.temporal.frame.camera_cut) {
            return false;
        }

        const auto& current_temporal = current.temporal.temporal;
        const auto& previous_temporal = previous.temporal.temporal;
        return current.color_image == previous.color_image &&
               current.color_format == previous.color_format &&
               current.color_generation == previous.color_generation &&
               current.depth_image == previous.depth_image &&
               current.depth_format == previous.depth_format &&
               current.depth_generation == previous.depth_generation &&
               current.quality == previous.quality &&
               current_temporal.view == previous_temporal.view &&
               current_temporal.render_extent.x == previous_temporal.render_extent.x &&
               current_temporal.render_extent.y == previous_temporal.render_extent.y &&
               current_temporal.output_extent.x == previous_temporal.output_extent.x &&
               current_temporal.output_extent.y == previous_temporal.output_extent.y &&
               current_temporal.frame.scene_generation ==
                   previous_temporal.frame.scene_generation &&
               current_temporal.frame.backend_key ==
                   previous_temporal.frame.backend_key;
    }
} // namespace lfs::vis
