/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "rendering/scene_temporal_plan.hpp"
#include "rendering/temporal_frame_tracker.hpp"

#include <algorithm>
#include <cstddef>
#include <glm/glm.hpp>
#include <memory>
#include <vulkan/vulkan.h>

namespace lfs::vis {
    class VulkanContext;

    [[nodiscard]] constexpr std::size_t nextTemporalHistoryWriteIndex(
        const bool has_history, const std::size_t read_index) {
        return has_history ? 1u - std::min<std::size_t>(read_index, 1u) : 0u;
    }

    [[nodiscard]] constexpr glm::vec4 temporalCurrentUvTransform(
        const glm::ivec2 render_extent, const glm::ivec2 allocation_extent) {
        if (render_extent.x <= 0 || render_extent.y <= 0 || allocation_extent.x <= 0 ||
            allocation_extent.y <= 0 || render_extent.x > allocation_extent.x ||
            render_extent.y > allocation_extent.y) {
            return {};
        }
        return {
            static_cast<float>(render_extent.x) / allocation_extent.x,
            static_cast<float>(render_extent.y) / allocation_extent.y,
            (static_cast<float>(render_extent.x) - 0.5f) / allocation_extent.x,
            (static_cast<float>(render_extent.y) - 0.5f) / allocation_extent.y,
        };
    }

    struct VulkanSceneTemporalResolveParams {
        bool enabled = false;
        TemporalViewId view = TemporalViewId::Main;
        VkImageView current_color_view = VK_NULL_HANDLE;
        VkImageView motion_view = VK_NULL_HANDLE;
        VkImageLayout current_color_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkImageLayout motion_layout = VK_IMAGE_LAYOUT_GENERAL;
        glm::ivec2 render_extent{0, 0};
        glm::ivec2 output_extent{0, 0};
        glm::ivec2 current_allocation_extent{0, 0};
        std::uint64_t sequence = 0;
        bool history_valid = false;
        float history_weight = 0.9f;
        float motion_rejection_pixels = 128.0f;
    };

    class VulkanSceneTemporalResolvePass {
    public:
        VulkanSceneTemporalResolvePass();
        ~VulkanSceneTemporalResolvePass();
        VulkanSceneTemporalResolvePass(const VulkanSceneTemporalResolvePass&) = delete;
        VulkanSceneTemporalResolvePass& operator=(const VulkanSceneTemporalResolvePass&) = delete;
        VulkanSceneTemporalResolvePass(VulkanSceneTemporalResolvePass&&) noexcept;
        VulkanSceneTemporalResolvePass& operator=(VulkanSceneTemporalResolvePass&&) noexcept;

        [[nodiscard]] bool init(VulkanContext& context);
        [[nodiscard]] bool record(VkCommandBuffer command_buffer,
                                  const VulkanSceneTemporalResolveParams& params);
        void reset(TemporalViewId view);
        void resetAll();
        void shutdown();

        [[nodiscard]] VkImageView outputView(TemporalViewId view) const;
        [[nodiscard]] SceneHistoryContract contract(TemporalViewId view) const;
        [[nodiscard]] bool initialized() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::vis
