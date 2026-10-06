/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_scene_plugin_pipeline.hpp"

#include "rendering/render_constants.hpp"
#include "rendering/scene_upscaler_plugin_api.h"

#include <cmath>

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

    float sceneUpscalerCameraVerticalFovRadians(const lfs::rendering::FrameView& view) noexcept {
        if (view.orthographic)
            return 0.0f;
        if (view.intrinsics_override && view.size.y > 0 &&
            std::isfinite(view.intrinsics_override->focal_y) &&
            view.intrinsics_override->focal_y > 0.0f) {
            return 2.0f * std::atan(0.5f * static_cast<float>(view.size.y) /
                                    view.intrinsics_override->focal_y);
        }
        return lfs::rendering::focalLengthToVFovRad(view.focal_length_mm);
    }

    bool validVulkanScenePluginPipelineRequest(
        const VulkanScenePluginPipelineRequest& request) noexcept {
        const VulkanScenePluginDepthParams depth{
            .enabled = true,
            .current_depth_view = request.temporal.motion.depth_view,
            .current_depth_layout =
                request.temporal.resolve.current_depth.current_depth_layout,
            .depth = request.temporal.motion.depth,
            .allocation_extent =
                request.temporal.resolve.current_depth.allocation_extent,
        };
        return sceneUpscalerPluginSupportsOutputExtent(request.temporal.temporal.output_extent) &&
               validVulkanSceneTemporalPipelineRequest(request.temporal) &&
               request.color_image != VK_NULL_HANDLE &&
               request.color_format == VK_FORMAT_R8G8B8A8_UNORM &&
               request.depth_image != VK_NULL_HANDLE &&
               request.depth_format == VK_FORMAT_R32_SFLOAT &&
               canRecordVulkanScenePluginDepth(depth) &&
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

    bool reusableVulkanScenePluginPipelineInput(
        const VulkanScenePluginPipelineRequest& current,
        const VulkanScenePluginPipelineRequest& previous) noexcept {
        const auto& now = current.temporal.temporal;
        const auto& before = previous.temporal.temporal;
        return current.color_generation != 0 && current.depth_generation != 0 &&
               !now.frame.camera_cut &&
               current.color_image == previous.color_image &&
               current.color_format == previous.color_format &&
               current.color_generation == previous.color_generation &&
               current.depth_image == previous.depth_image &&
               current.depth_format == previous.depth_format &&
               current.depth_generation == previous.depth_generation &&
               current.quality == previous.quality && now.view == before.view &&
               now.render_extent == before.render_extent &&
               now.output_extent == before.output_extent &&
               now.frame.scene_generation == before.frame.scene_generation &&
               now.frame.backend_key == before.frame.backend_key;
    }
} // namespace lfs::vis
