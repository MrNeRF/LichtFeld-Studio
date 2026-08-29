/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_scene_dlss_pipeline.hpp"

namespace lfs::vis {
    bool validVulkanSceneDlssPipelineRequest(
        const VulkanSceneDlssPipelineRequest& request) noexcept {
        const VulkanSceneDlssDepthParams depth{
            .enabled = true,
            .current_depth_view = request.temporal.motion.depth_view,
            .current_depth_layout =
                request.temporal.resolve.current_depth.current_depth_layout,
            .depth = request.temporal.motion.depth,
            .allocation_extent =
                request.temporal.resolve.current_depth.allocation_extent,
        };
        return nvidiaDlssSupportsOutputExtent(request.temporal.temporal.output_extent) &&
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
               request.temporal.resolve.current_color_layout != VK_IMAGE_LAYOUT_UNDEFINED &&
               request.temporal.resolve.current_depth.current_depth_layout !=
                   VK_IMAGE_LAYOUT_UNDEFINED;
    }
} // namespace lfs::vis
