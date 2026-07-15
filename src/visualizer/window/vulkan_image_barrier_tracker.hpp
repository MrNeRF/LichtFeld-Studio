/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vulkan/vulkan.h>

namespace lfs::vis {

    class VulkanImageBarrierTracker {
    public:
        struct AccessScope {
            VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_NONE;
            VkAccessFlags2 access = VK_ACCESS_2_NONE;
        };

        enum class AccessDirection {
            Source,
            Destination,
        };

        struct ImageState {
            VkImageAspectFlags aspect_mask = VK_IMAGE_ASPECT_COLOR_BIT;
            VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
            VkPipelineStageFlags2 last_stage = VK_PIPELINE_STAGE_2_NONE;
            VkAccessFlags2 last_access = VK_ACCESS_2_NONE;
        };

        void reset();
        void clearSwapchainOnly();
        void forgetImage(VkImage image);
        void registerImage(VkImage image, VkImageAspectFlags aspect_mask, VkImageLayout layout,
                           bool external = false);

        [[nodiscard]] VkImageLayout imageLayout(VkImage image,
                                                VkImageLayout fallback = VK_IMAGE_LAYOUT_UNDEFINED) const;

        [[nodiscard]] static AccessScope layoutAccess(VkImageLayout layout,
                                                      AccessDirection direction) noexcept;

        void transitionImage(VkCommandBuffer command_buffer,
                             VkImage image,
                             VkImageAspectFlags aspect_mask,
                             VkImageLayout new_layout);

        // Layouts do not identify the queue or shader stage that produced/consumes
        // an image. Cross-queue users must provide the scopes represented by the
        // submission's semaphore edges instead of inheriting a graphics-only scope.
        void transitionImage(VkCommandBuffer command_buffer,
                             VkImage image,
                             VkImageAspectFlags aspect_mask,
                             VkImageLayout new_layout,
                             AccessScope source,
                             AccessScope destination);

    private:
        std::unordered_map<VkImage, ImageState> images_;
        std::unordered_set<VkImage> external_images_;
    };

} // namespace lfs::vis
