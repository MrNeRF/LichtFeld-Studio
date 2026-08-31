/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "rendering/render_constants.hpp"
#include "rendering/scene_temporal_resolve.hpp"
#include "vulkan_scene_dlss_depth_pass.hpp"
#include "vulkan_scene_temporal_pipeline.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vulkan/vulkan.h>

namespace lfs::vis {
    class VulkanContext;

    inline constexpr int AMD_FSR3_MIN_OUTPUT_EXTENT = 32;

    [[nodiscard]] constexpr bool amdFsr3SupportsOutputExtent(
        const glm::ivec2 extent) noexcept {
        return extent.x >= AMD_FSR3_MIN_OUTPUT_EXTENT &&
               extent.y >= AMD_FSR3_MIN_OUTPUT_EXTENT;
    }

    [[nodiscard]] inline float amdFsr3CameraVerticalFovRadians(
        const lfs::rendering::FrameView& view) noexcept {
        if (view.intrinsics_override && view.size.y > 0 &&
            std::isfinite(view.intrinsics_override->focal_y) &&
            view.intrinsics_override->focal_y > 0.0f) {
            return 2.0f * std::atan(
                              0.5f * static_cast<float>(view.size.y) /
                              view.intrinsics_override->focal_y);
        }
        return lfs::rendering::focalLengthToVFovRad(view.focal_length_mm);
    }

    [[nodiscard]] LFS_VIS_API std::uint32_t pluginResetFlags(
        TemporalResetReason reasons) noexcept;

    struct VulkanSceneFsr3PipelineRequest {
        VulkanSceneTemporalPipelineRequest temporal;
        VkImage color_image = VK_NULL_HANDLE;
        VkFormat color_format = VK_FORMAT_UNDEFINED;
        VkImage depth_image = VK_NULL_HANDLE;
        VkFormat depth_format = VK_FORMAT_UNDEFINED;
        SceneTemporalQuality quality = SceneTemporalQuality::Balanced;
    };

    enum class VulkanSceneFsr3PipelineStatus : std::uint8_t {
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

    struct VulkanSceneFsr3PipelineResult {
        VulkanSceneFsr3PipelineStatus status = VulkanSceneFsr3PipelineStatus::Inactive;
        TemporalViewId view = TemporalViewId::Main;
        std::uint64_t sequence = 0;
        VkImageView output_view = VK_NULL_HANDLE;
        SceneHistoryContract history;

        [[nodiscard]] constexpr bool resolved() const noexcept {
            return status == VulkanSceneFsr3PipelineStatus::Resolved &&
                   output_view != VK_NULL_HANDLE && history.valid();
        }
    };

    [[nodiscard]] LFS_VIS_API bool validVulkanSceneFsr3PipelineRequest(
        const VulkanSceneFsr3PipelineRequest& request) noexcept;

    class VulkanSceneFsr3Pipeline {
    public:
        VulkanSceneFsr3Pipeline();
        ~VulkanSceneFsr3Pipeline();
        VulkanSceneFsr3Pipeline(const VulkanSceneFsr3Pipeline&) = delete;
        VulkanSceneFsr3Pipeline& operator=(const VulkanSceneFsr3Pipeline&) = delete;
        VulkanSceneFsr3Pipeline(VulkanSceneFsr3Pipeline&&) noexcept;
        VulkanSceneFsr3Pipeline& operator=(VulkanSceneFsr3Pipeline&&) noexcept;

        [[nodiscard]] bool init(VulkanContext& context);
        [[nodiscard]] VulkanSceneFsr3PipelineResult record(
            VkCommandBuffer command_buffer,
            const VulkanSceneFsr3PipelineRequest& request);
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
