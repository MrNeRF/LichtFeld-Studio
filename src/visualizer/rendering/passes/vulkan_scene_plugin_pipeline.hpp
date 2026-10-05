/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "rendering/scene_temporal_resolve.hpp"
#include "vulkan_scene_plugin_depth_pass.hpp"
#include "vulkan_scene_temporal_pipeline.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vulkan/vulkan.h>

namespace lfs::vis {
    class SceneUpscalerPlugin;
    class VulkanContext;

    inline constexpr int SCENE_UPSCALER_PLUGIN_MIN_OUTPUT_EXTENT = 32;

    [[nodiscard]] constexpr bool sceneUpscalerPluginSupportsOutputExtent(
        const glm::ivec2 extent) noexcept {
        return extent.x >= SCENE_UPSCALER_PLUGIN_MIN_OUTPUT_EXTENT &&
               extent.y >= SCENE_UPSCALER_PLUGIN_MIN_OUTPUT_EXTENT;
    }

    [[nodiscard]] LFS_VIS_API std::uint32_t pluginResetFlags(
        TemporalResetReason reasons) noexcept;

    [[nodiscard]] LFS_VIS_API float sceneUpscalerCameraVerticalFovRadians(
        const lfs::rendering::FrameView& view) noexcept;

    struct VulkanScenePluginPipelineRequest {
        VulkanSceneTemporalPipelineRequest temporal;
        VkImage color_image = VK_NULL_HANDLE;
        VkFormat color_format = VK_FORMAT_UNDEFINED;
        // Publication generations of color and depth; 0 means unknown. Equal
        // non-zero generations let the pipeline skip re-evaluating the same frame.
        std::uint64_t color_generation = 0;
        VkImage depth_image = VK_NULL_HANDLE;
        VkFormat depth_format = VK_FORMAT_UNDEFINED;
        std::uint64_t depth_generation = 0;
        SceneTemporalQuality quality = SceneTemporalQuality::Balanced;
    };

    enum class VulkanScenePluginPipelineStatus : std::uint8_t {
        Inactive = 0,
        Resolved,
        InvalidRequest,
        RuntimeUnavailable,
        MotionUnavailable,
        MotionFailure,
        DepthUnavailable,
        DepthFailure,
        OutputFailure,
        FeatureFailure,
        EvaluateFailure,
        CommitFailure,
    };

    struct VulkanScenePluginPipelineResult {
        VulkanScenePluginPipelineStatus status = VulkanScenePluginPipelineStatus::Inactive;
        TemporalViewId view = TemporalViewId::Main;
        std::uint64_t sequence = 0;
        VkImageView output_view = VK_NULL_HANDLE;
        VkImageLayout output_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        SceneHistoryContract history;

        [[nodiscard]] constexpr bool resolved() const noexcept {
            return status == VulkanScenePluginPipelineStatus::Resolved &&
                   output_view != VK_NULL_HANDLE &&
                   output_layout != VK_IMAGE_LAYOUT_UNDEFINED && history.valid();
        }
    };

    [[nodiscard]] LFS_VIS_API bool validVulkanScenePluginPipelineRequest(
        const VulkanScenePluginPipelineRequest& request) noexcept;
    // True when `current` republishes exactly the frame `previous` evaluated.
    [[nodiscard]] LFS_VIS_API bool reusableVulkanScenePluginPipelineInput(
        const VulkanScenePluginPipelineRequest& current,
        const VulkanScenePluginPipelineRequest& previous) noexcept;

    class VulkanScenePluginPipeline {
    public:
        VulkanScenePluginPipeline();
        ~VulkanScenePluginPipeline();
        VulkanScenePluginPipeline(const VulkanScenePluginPipeline&) = delete;
        VulkanScenePluginPipeline& operator=(const VulkanScenePluginPipeline&) = delete;
        VulkanScenePluginPipeline(VulkanScenePluginPipeline&&) noexcept;
        VulkanScenePluginPipeline& operator=(VulkanScenePluginPipeline&&) noexcept;

        [[nodiscard]] bool init(VulkanContext& context, SceneUpscalerPlugin& plugin);
        [[nodiscard]] SceneUpscalerPlugin* plugin() const noexcept;
        [[nodiscard]] VulkanScenePluginPipelineResult record(
            VkCommandBuffer command_buffer,
            const VulkanScenePluginPipelineRequest& request);
        void reset(TemporalViewId view,
                   TemporalResetReason reason = TemporalResetReason::HistoryDisabled);
        void resetAll(TemporalResetReason reason = TemporalResetReason::HistoryDisabled);
        void releaseResources(
            TemporalResetReason reason = TemporalResetReason::HistoryDisabled);
        void shutdown();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::vis
