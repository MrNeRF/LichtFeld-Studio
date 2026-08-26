/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "vulkan_scene_depth_history_contract.hpp"

#include <cstddef>
#include <memory>
#include <vulkan/vulkan.h>

namespace lfs::vis {

    class VulkanContext;

    class VulkanSceneDepthHistoryPass {
    public:
        VulkanSceneDepthHistoryPass();
        ~VulkanSceneDepthHistoryPass();

        VulkanSceneDepthHistoryPass(const VulkanSceneDepthHistoryPass&) = delete;
        VulkanSceneDepthHistoryPass& operator=(const VulkanSceneDepthHistoryPass&) = delete;
        VulkanSceneDepthHistoryPass(VulkanSceneDepthHistoryPass&&) noexcept;
        VulkanSceneDepthHistoryPass& operator=(VulkanSceneDepthHistoryPass&&) noexcept;

        [[nodiscard]] bool init(VulkanContext& context);
        [[nodiscard]] bool record(VkCommandBuffer command_buffer,
                                  const VulkanSceneDepthHistoryParams& params,
                                  std::size_t resource_slot);
        void invalidate(std::size_t resource_slot);
        void invalidateAll();
        void shutdown();

        [[nodiscard]] VkImageView depthView(std::size_t resource_slot) const;
        [[nodiscard]] VkImage depthImage(std::size_t resource_slot) const;
        [[nodiscard]] SceneDepthContract contract(std::size_t resource_slot) const;
        [[nodiscard]] bool initialized() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::vis
