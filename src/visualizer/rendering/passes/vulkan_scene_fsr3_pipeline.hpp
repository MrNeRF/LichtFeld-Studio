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
#include <optional>
#include <vulkan/vulkan.h>

namespace lfs::vis {
    class VulkanContext;

    inline constexpr int AMD_FSR3_MIN_OUTPUT_EXTENT = 32;
    inline constexpr std::uint32_t AMD_FSR3_MOTION_VECTORS_INCLUDE_JITTER = 1;

    [[nodiscard]] constexpr bool amdFsr3SupportsOutputExtent(
        const glm::ivec2 extent) noexcept {
        return extent.x >= AMD_FSR3_MIN_OUTPUT_EXTENT &&
               extent.y >= AMD_FSR3_MIN_OUTPUT_EXTENT;
    }

    // Mirrors ffxFsr3UpscalerGetJitterPhaseCount without making the core
    // viewport depend on an optional plugin SDK. Keep the float arithmetic and
    // pow call identical to SDK 1.1.4: the final conversion truncates.
    [[nodiscard]] inline std::uint32_t amdFsr3JitterPhaseCount(
        const int render_width, const int display_width) noexcept {
        if (render_width <= 0 || display_width <= 0)
            return 1;
        const float ratio = static_cast<float>(display_width) /
                            static_cast<float>(render_width);
        const float phase = 8.0f * std::pow(ratio, 2.0f);
        if (!std::isfinite(phase) || phase < 1.0f)
            return 1;
        return static_cast<std::uint32_t>(phase);
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

    // The depth buffer was rasterized with the jittered scene projection, so
    // reconstruct motion from that same projection pair. The plugin advertises
    // this fact to FidelityFX and enables its jitter-cancellation contract.
    [[nodiscard]] inline std::optional<TemporalProjectionPair>
    makeAmdFsr3MotionViewProjectionPair(const TemporalFrameState& state) {
        return makeTemporalViewProjectionPair(state);
    }

    // FidelityFX expects the pixel offset applied to the camera projection.
    // Texture storage/presentation orientation is unrelated and must not flip
    // the reported Y offset: doing so makes FSR accumulate history against a
    // different subpixel sample than the one VkSplat actually rendered.
    [[nodiscard]] inline glm::vec2 amdFsr3DispatchJitterPixels(
        const TemporalFrameState& state, const glm::ivec2 render_extent) noexcept {
        return sceneTemporalJitterPixels(state.current_jitter, render_extent, false);
    }

    [[nodiscard]] LFS_VIS_API std::uint32_t pluginResetFlags(
        TemporalResetReason reasons) noexcept;

    struct VulkanSceneFsr3PipelineRequest {
        VulkanSceneTemporalPipelineRequest temporal;
        VkImage color_image = VK_NULL_HANDLE;
        VkFormat color_format = VK_FORMAT_UNDEFINED;
        std::uint64_t color_generation = 0;
        VkImage depth_image = VK_NULL_HANDLE;
        VkFormat depth_format = VK_FORMAT_UNDEFINED;
        std::uint64_t depth_generation = 0;
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
        VkImageLayout output_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        SceneHistoryContract history;

        [[nodiscard]] constexpr bool resolved() const noexcept {
            return status == VulkanSceneFsr3PipelineStatus::Resolved &&
                   output_view != VK_NULL_HANDLE &&
                   output_layout != VK_IMAGE_LAYOUT_UNDEFINED && history.valid();
        }
    };

    [[nodiscard]] LFS_VIS_API bool validVulkanSceneFsr3PipelineRequest(
        const VulkanSceneFsr3PipelineRequest& request) noexcept;
    [[nodiscard]] LFS_VIS_API bool reusableVulkanSceneFsr3PipelineInput(
        const VulkanSceneFsr3PipelineRequest& current,
        const VulkanSceneFsr3PipelineRequest& previous) noexcept;

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
