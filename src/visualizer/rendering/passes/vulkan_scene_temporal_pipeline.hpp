/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "rendering/scene_temporal_coordinator.hpp"
#include "vulkan_scene_motion_pass.hpp"
#include "vulkan_scene_temporal_resolve_pass.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vulkan/vulkan.h>

namespace lfs::vis {
    class VulkanContext;

    struct VulkanSceneTemporalPipelineRequest {
        SceneTemporalRequest temporal;
        VulkanSceneMotionParams motion;
        VulkanSceneTemporalResolveParams resolve;
        std::size_t frame_slot = 0;
    };

    enum class VulkanSceneTemporalPipelineStatus : std::uint8_t {
        Inactive = 0,
        Resolved,
        InvalidRequest,
        MotionUnavailable,
        MotionFailure,
        ResolveUnavailable,
        ResolveFailure,
        CommitFailure,
    };

    struct VulkanSceneTemporalPipelineResult {
        VulkanSceneTemporalPipelineStatus status =
            VulkanSceneTemporalPipelineStatus::Inactive;
        TemporalViewId view = TemporalViewId::Main;
        std::uint64_t sequence = 0;
        VkImageView output_view = VK_NULL_HANDLE;
        SceneHistoryContract history;

        [[nodiscard]] constexpr bool resolved() const noexcept {
            return status == VulkanSceneTemporalPipelineStatus::Resolved &&
                   output_view != VK_NULL_HANDLE && history.valid();
        }
    };

    [[nodiscard]] LFS_VIS_API bool validVulkanSceneTemporalPipelineRequest(
        const VulkanSceneTemporalPipelineRequest& request) noexcept;

    class VulkanSceneTemporalPipeline {
    public:
        VulkanSceneTemporalPipeline();
        ~VulkanSceneTemporalPipeline();
        VulkanSceneTemporalPipeline(const VulkanSceneTemporalPipeline&) = delete;
        VulkanSceneTemporalPipeline& operator=(const VulkanSceneTemporalPipeline&) = delete;
        VulkanSceneTemporalPipeline(VulkanSceneTemporalPipeline&&) noexcept;
        VulkanSceneTemporalPipeline& operator=(VulkanSceneTemporalPipeline&&) noexcept;

        [[nodiscard]] bool init(VulkanContext& context);
        [[nodiscard]] VulkanSceneTemporalPipelineResult record(
            VkCommandBuffer command_buffer,
            const VulkanSceneTemporalPipelineRequest& request);
        void reset(TemporalViewId view,
                   TemporalResetReason reason = TemporalResetReason::HistoryDisabled);
        void resetAll(TemporalResetReason reason = TemporalResetReason::HistoryDisabled);
        void releaseHistory(
            TemporalResetReason reason = TemporalResetReason::HistoryDisabled);
        void shutdown();

        [[nodiscard]] VkImageView outputView(TemporalViewId view) const;
        [[nodiscard]] SceneHistoryContract contract(TemporalViewId view) const;
        [[nodiscard]] VulkanSceneTemporalResourceStats resourceStats() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::vis
