/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "rendering/scene_temporal_resolve.hpp"
#include "vulkan_scene_dlss_depth_pass.hpp"
#include "vulkan_scene_temporal_pipeline.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vulkan/vulkan.h>

namespace lfs::vis {
    class VulkanContext;

    inline constexpr int NVIDIA_DLSS_MIN_OUTPUT_EXTENT = 32;

    [[nodiscard]] constexpr bool nvidiaDlssSupportsOutputExtent(
        const glm::ivec2 extent) noexcept {
        return extent.x >= NVIDIA_DLSS_MIN_OUTPUT_EXTENT &&
               extent.y >= NVIDIA_DLSS_MIN_OUTPUT_EXTENT;
    }

    struct VulkanSceneDlssPipelineRequest {
        VulkanSceneTemporalPipelineRequest temporal;
        VkImage color_image = VK_NULL_HANDLE;
        VkFormat color_format = VK_FORMAT_UNDEFINED;
        VkImage depth_image = VK_NULL_HANDLE;
        VkFormat depth_format = VK_FORMAT_UNDEFINED;
        SceneTemporalQuality quality = SceneTemporalQuality::Balanced;
    };

    enum class VulkanSceneDlssPipelineStatus : std::uint8_t {
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

    struct VulkanSceneDlssPipelineResult {
        VulkanSceneDlssPipelineStatus status = VulkanSceneDlssPipelineStatus::Inactive;
        TemporalViewId view = TemporalViewId::Main;
        std::uint64_t sequence = 0;
        VkImageView output_view = VK_NULL_HANDLE;
        SceneHistoryContract history;

        [[nodiscard]] constexpr bool resolved() const noexcept {
            return status == VulkanSceneDlssPipelineStatus::Resolved &&
                   output_view != VK_NULL_HANDLE && history.valid();
        }
    };

    [[nodiscard]] LFS_VIS_API bool validVulkanSceneDlssPipelineRequest(
        const VulkanSceneDlssPipelineRequest& request) noexcept;

    class VulkanSceneDlssPipeline {
    public:
        VulkanSceneDlssPipeline();
        ~VulkanSceneDlssPipeline();
        VulkanSceneDlssPipeline(const VulkanSceneDlssPipeline&) = delete;
        VulkanSceneDlssPipeline& operator=(const VulkanSceneDlssPipeline&) = delete;
        VulkanSceneDlssPipeline(VulkanSceneDlssPipeline&&) noexcept;
        VulkanSceneDlssPipeline& operator=(VulkanSceneDlssPipeline&&) noexcept;

        [[nodiscard]] bool init(VulkanContext& context);
        [[nodiscard]] VulkanSceneDlssPipelineResult record(
            VkCommandBuffer command_buffer,
            const VulkanSceneDlssPipelineRequest& request);
        void reset(TemporalViewId view,
                   TemporalResetReason reason = TemporalResetReason::HistoryDisabled);
        void resetAll(TemporalResetReason reason = TemporalResetReason::HistoryDisabled);
        void releaseResources(
            TemporalResetReason reason = TemporalResetReason::HistoryDisabled);
        void shutdown();

        [[nodiscard]] std::size_t residentOutputCount() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::vis
