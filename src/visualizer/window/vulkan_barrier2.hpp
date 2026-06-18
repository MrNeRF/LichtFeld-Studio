/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <vulkan/vulkan.h>

namespace lfs::vis {

    // Stateless synchronization2 image-layout transition for one-shot upload/transfer command
    // buffers that do not participate in VulkanImageBarrierTracker. Records a single
    // VkImageMemoryBarrier2 over mip 0 / layer 0 of the given aspect.
    inline void cmdImageBarrier2(const VkCommandBuffer cmd,
                                 const VkImage image,
                                 const VkImageAspectFlags aspect,
                                 const VkImageLayout old_layout,
                                 const VkImageLayout new_layout,
                                 const VkPipelineStageFlags2 src_stage,
                                 const VkAccessFlags2 src_access,
                                 const VkPipelineStageFlags2 dst_stage,
                                 const VkAccessFlags2 dst_access) {
        VkImageMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.srcStageMask = src_stage;
        barrier.srcAccessMask = src_access;
        barrier.dstStageMask = dst_stage;
        barrier.dstAccessMask = dst_access;
        barrier.oldLayout = old_layout;
        barrier.newLayout = new_layout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = aspect;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = 1;

        VkDependencyInfo dependency{};
        dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependency.imageMemoryBarrierCount = 1;
        dependency.pImageMemoryBarriers = &barrier;

        vkCmdPipelineBarrier2(cmd, &dependency);
    }

} // namespace lfs::vis
