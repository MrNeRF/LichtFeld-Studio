/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "rendering/scene_depth_contract.hpp"
#include "rendering/temporal_frame_tracker.hpp"

#include <glm/glm.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vulkan/vulkan.h>

namespace lfs::vis {

    struct VulkanSceneDepthHistoryParams {
        bool enabled = false;
        TemporalViewId view = TemporalViewId::Main;
        VkImageView current_depth_view = VK_NULL_HANDLE;
        VkImageLayout current_depth_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        SceneDepthContract depth;
        glm::ivec2 allocation_extent{0, 0};
    };

    [[nodiscard]] constexpr bool validSceneDepthHistoryView(
        const TemporalViewId view) noexcept {
        return static_cast<std::size_t>(view) <
               static_cast<std::size_t>(TemporalViewId::Count);
    }

    [[nodiscard]] inline bool canRecordVulkanSceneDepthHistory(
        const VulkanSceneDepthHistoryParams& params) noexcept {
        return params.enabled && validSceneDepthHistoryView(params.view) &&
               params.current_depth_view != VK_NULL_HANDLE && params.depth.available() &&
               params.depth.valid() && params.depth.storage == SceneDepthStorage::VulkanImage &&
               (params.depth.encoding == SceneDepthEncoding::LinearView ||
                params.depth.encoding == SceneDepthEncoding::VulkanNdc) &&
               (params.current_depth_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL ||
                params.current_depth_layout == VK_IMAGE_LAYOUT_GENERAL);
    }

    [[nodiscard]] constexpr std::uint32_t sceneDepthHistoryEncodingCode(
        const SceneDepthContract& depth) noexcept {
        if (depth.encoding == SceneDepthEncoding::LinearView) {
            return 1;
        }
        if (depth.encoding == SceneDepthEncoding::VulkanNdc) {
            return depth.orthographic ? 3 : 2;
        }
        return 0;
    }

    [[nodiscard]] constexpr std::optional<glm::vec4> sceneDepthHistoryUvTransform(
        const glm::ivec2 valid_extent, const glm::ivec2 allocation_extent) noexcept {
        if (valid_extent.x <= 0 || valid_extent.y <= 0 || allocation_extent.x <= 0 ||
            allocation_extent.y <= 0 || valid_extent.x > allocation_extent.x ||
            valid_extent.y > allocation_extent.y) {
            return std::nullopt;
        }
        return glm::vec4{
            static_cast<float>(valid_extent.x) / allocation_extent.x,
            static_cast<float>(valid_extent.y) / allocation_extent.y,
            (static_cast<float>(valid_extent.x) - 0.5f) / allocation_extent.x,
            (static_cast<float>(valid_extent.y) - 0.5f) / allocation_extent.y,
        };
    }

} // namespace lfs::vis
