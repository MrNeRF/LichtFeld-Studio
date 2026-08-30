/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_scene_dlss_pipeline.hpp"

#include "rendering/scene_upscaler_plugin_api.h"

namespace lfs::vis {
    std::uint32_t pluginResetFlags(const TemporalResetReason reasons) noexcept {
        std::uint32_t flags = LFS_SCENE_UPSCALER_PLUGIN_RESET_NONE;
        if (hasTemporalResetReason(reasons, TemporalResetReason::CameraCut) ||
            hasTemporalResetReason(reasons, TemporalResetReason::Projection))
            flags |= LFS_SCENE_UPSCALER_PLUGIN_RESET_CAMERA_CUT;
        if (hasTemporalResetReason(reasons, TemporalResetReason::RenderSize) ||
            hasTemporalResetReason(reasons, TemporalResetReason::RenderScale))
            flags |= LFS_SCENE_UPSCALER_PLUGIN_RESET_RENDER_SIZE;
        if (hasTemporalResetReason(reasons, TemporalResetReason::OutputExtent))
            flags |= LFS_SCENE_UPSCALER_PLUGIN_RESET_OUTPUT_SIZE;
        if (hasTemporalResetReason(reasons, TemporalResetReason::Scene) ||
            hasTemporalResetReason(reasons, TemporalResetReason::Backend))
            flags |= LFS_SCENE_UPSCALER_PLUGIN_RESET_SCENE;
        if (hasTemporalResetReason(reasons, TemporalResetReason::Quality))
            flags |= LFS_SCENE_UPSCALER_PLUGIN_RESET_QUALITY;
        if (hasTemporalResetReason(reasons, TemporalResetReason::FirstFrame) ||
            hasTemporalResetReason(reasons, TemporalResetReason::Requested) ||
            hasTemporalResetReason(reasons, TemporalResetReason::HistoryDisabled))
            flags |= LFS_SCENE_UPSCALER_PLUGIN_RESET_REQUESTED;
        if (hasTemporalResetReason(reasons, TemporalResetReason::RuntimeUnavailable) ||
            hasTemporalResetReason(reasons, TemporalResetReason::ResolveFailure) ||
            hasTemporalResetReason(reasons, TemporalResetReason::InvalidInput))
            flags |= LFS_SCENE_UPSCALER_PLUGIN_RESET_RUNTIME;
        return flags;
    }

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
               request.temporal.resolve.current_color_layout ==
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL &&
               request.temporal.resolve.current_depth.current_depth_layout ==
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
} // namespace lfs::vis
