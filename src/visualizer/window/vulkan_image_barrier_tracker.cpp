/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_image_barrier_tracker.hpp"

#ifdef LFS_VULKAN_VIEWER_ENABLED
namespace lfs::vis {

    namespace {
        struct LayoutAccess {
            VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_NONE;
            VkAccessFlags2 access = VK_ACCESS_2_NONE;
        };

        [[nodiscard]] LayoutAccess layoutAccess(const VkImageLayout layout, const bool source) {
            switch (layout) {
            case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
                return {
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    source ? VkAccessFlags2(VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT)
                           : VkAccessFlags2(VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                                            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT),
                };
            case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
                return {
                    VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                        VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                    source ? VkAccessFlags2(VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)
                           : VkAccessFlags2(VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT),
                };
            case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
                return {VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT};
            case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
                return {VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT};
            case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
                return {VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT};
            case VK_IMAGE_LAYOUT_GENERAL:
                return {
                    VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                    VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                };
            case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
            case VK_IMAGE_LAYOUT_UNDEFINED:
            default:
                return {};
            }
        }
    } // namespace

    void VulkanImageBarrierTracker::reset() {
        images_.clear();
        external_images_.clear();
    }

    void VulkanImageBarrierTracker::clearSwapchainOnly() {
        for (auto it = images_.begin(); it != images_.end();) {
            if (external_images_.contains(it->first)) {
                ++it;
            } else {
                it = images_.erase(it);
            }
        }
    }

    void VulkanImageBarrierTracker::forgetImage(const VkImage image) {
        if (image != VK_NULL_HANDLE) {
            images_.erase(image);
            external_images_.erase(image);
        }
    }

    void VulkanImageBarrierTracker::registerImage(const VkImage image,
                                                  const VkImageAspectFlags aspect_mask,
                                                  const VkImageLayout layout,
                                                  const bool external) {
        if (image == VK_NULL_HANDLE) {
            return;
        }
        const LayoutAccess access = layoutAccess(layout, true);
        images_[image] = ImageState{
            .aspect_mask = aspect_mask,
            .layout = layout,
            .last_stage = access.stage,
            .last_access = access.access,
        };
        if (external) {
            external_images_.insert(image);
        } else {
            external_images_.erase(image);
        }
    }

    VkImageLayout VulkanImageBarrierTracker::imageLayout(const VkImage image, const VkImageLayout fallback) const {
        const auto it = images_.find(image);
        return it != images_.end() ? it->second.layout : fallback;
    }

    void VulkanImageBarrierTracker::transitionImage(const VkCommandBuffer command_buffer,
                                                    const VkImage image,
                                                    const VkImageAspectFlags aspect_mask,
                                                    const VkImageLayout new_layout) {
        if (command_buffer == VK_NULL_HANDLE || image == VK_NULL_HANDLE) {
            return;
        }

        auto& state = images_[image];
        if (state.aspect_mask == 0) {
            state.aspect_mask = aspect_mask;
        }
        if (state.layout == new_layout) {
            return;
        }

        const LayoutAccess src =
            state.last_stage != VK_PIPELINE_STAGE_2_NONE || state.last_access != VK_ACCESS_2_NONE
                ? LayoutAccess{state.last_stage, state.last_access}
                : layoutAccess(state.layout, true);
        const LayoutAccess dst = layoutAccess(new_layout, false);

        VkImageMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.srcStageMask = src.stage;
        barrier.srcAccessMask = src.access;
        barrier.dstStageMask = dst.stage;
        barrier.dstAccessMask = dst.access;
        barrier.oldLayout = state.layout;
        barrier.newLayout = new_layout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = aspect_mask;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        VkDependencyInfo dependency{};
        dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependency.imageMemoryBarrierCount = 1;
        dependency.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(command_buffer, &dependency);

        state.aspect_mask = aspect_mask;
        state.layout = new_layout;
        state.last_stage = dst.stage;
        state.last_access = dst.access;
    }

} // namespace lfs::vis
#endif
