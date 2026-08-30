/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "rendering/scene_depth_contract.hpp"

#include <cstddef>
#include <memory>
#include <vulkan/vulkan.h>

namespace lfs::vis {

    class VulkanContext;

    struct VulkanSceneDlssDepthParams {
        bool enabled = false;
        VkImageView current_depth_view = VK_NULL_HANDLE;
        VkImageLayout current_depth_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        SceneDepthContract depth;
        glm::ivec2 allocation_extent{0, 0};
    };

    [[nodiscard]] inline bool canRecordVulkanSceneDlssDepth(
        const VulkanSceneDlssDepthParams& params) noexcept {
        return params.enabled && params.current_depth_view != VK_NULL_HANDLE &&
               params.depth.available() && params.depth.valid() &&
               params.depth.storage == SceneDepthStorage::VulkanImage &&
               (params.depth.encoding == SceneDepthEncoding::LinearView ||
                params.depth.encoding == SceneDepthEncoding::VulkanNdc) &&
               params.allocation_extent.x >= params.depth.width &&
               params.allocation_extent.y >= params.depth.height &&
               (params.current_depth_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL ||
                params.current_depth_layout == VK_IMAGE_LAYOUT_GENERAL);
    }

    [[nodiscard]] constexpr std::uint32_t sceneDlssDepthEncodingCode(
        const SceneDepthContract& depth) noexcept {
        if (depth.encoding == SceneDepthEncoding::VulkanNdc)
            return 1;
        if (depth.encoding == SceneDepthEncoding::LinearView)
            return depth.orthographic ? 3 : 2;
        return 0;
    }

    class VulkanSceneDlssDepthPass {
    public:
        VulkanSceneDlssDepthPass();
        ~VulkanSceneDlssDepthPass();

        VulkanSceneDlssDepthPass(const VulkanSceneDlssDepthPass&) = delete;
        VulkanSceneDlssDepthPass& operator=(const VulkanSceneDlssDepthPass&) = delete;
        VulkanSceneDlssDepthPass(VulkanSceneDlssDepthPass&&) noexcept;
        VulkanSceneDlssDepthPass& operator=(VulkanSceneDlssDepthPass&&) noexcept;

        [[nodiscard]] bool init(VulkanContext& context);
        [[nodiscard]] bool record(VkCommandBuffer command_buffer,
                                  const VulkanSceneDlssDepthParams& params,
                                  std::size_t resource_slot);
        void shutdown();

        [[nodiscard]] VkImageView depthView(std::size_t resource_slot) const;
        [[nodiscard]] VkImage depthImage(std::size_t resource_slot) const;
        [[nodiscard]] bool initialized() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::vis
