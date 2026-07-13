/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_image_barrier_tracker.hpp"
#include "vulkan_result.hpp"

namespace lfs::vis {

    VulkanImageBarrierTracker::AccessScope
    VulkanImageBarrierTracker::layoutAccess(const VkImageLayout layout,
                                            const AccessDirection direction) noexcept {
        const bool source = direction == AccessDirection::Source;
        switch (layout) {
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            return {
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                source ? VkAccessFlags2(VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT)
                       : VkAccessFlags2(VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                                        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT),
            };
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
        case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL:
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
            // The acquire semaphore wait and the first-use layout transition form a
            // COLOR_ATTACHMENT_OUTPUT -> COLOR_ATTACHMENT_OUTPUT dependency chain. There
            // is no source access to make available, but the source stage must match the
            // wait stage so the transition itself cannot run ahead of image acquisition.
            // A transition *to* PRESENT needs no destination scope; presentation supplies
            // its own visibility operation after waiting on render_finished_.
            return source
                       ? AccessScope{VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                     VK_ACCESS_2_NONE}
                       : AccessScope{};
        case VK_IMAGE_LAYOUT_UNDEFINED:
            // Contents are discarded, so there is no prior access to make available.
            return {};
        default:
            return {
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            };
        }
    }

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
        const AccessScope access = layoutAccess(layout, AccessDirection::Source);
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
            throw std::logic_error(std::format(
                "Image barrier transition requires non-null handles (command_buffer={:#x}, image={:#x}, aspect_mask={:#x}, requested_layout={}({})) ({}:{})",
                vkHandleValue(command_buffer),
                vkHandleValue(image),
                static_cast<std::uint32_t>(aspect_mask),
                vkImageLayoutToString(new_layout),
                static_cast<int>(new_layout),
                __FILE__,
                __LINE__));
        }

        [[maybe_unused]] const auto tracked = images_.find(image);
        LFS_VK_DEBUG_ASSERT(
            tracked != images_.end(),
            "Image barrier tracker does not know the transitioned image (image={:#x}, requested_layout={}({}), aspect_mask={:#x}, tracked_images={})",
            vkHandleValue(image),
            vkImageLayoutToString(new_layout),
            static_cast<int>(new_layout),
            static_cast<std::uint32_t>(aspect_mask),
            images_.size());
        auto& state = images_[image];
        if (state.aspect_mask == 0) {
            state.aspect_mask = aspect_mask;
        }
        if (state.layout == new_layout) {
            return;
        }

        const AccessScope src =
            state.last_stage != VK_PIPELINE_STAGE_2_NONE || state.last_access != VK_ACCESS_2_NONE
                ? AccessScope{state.last_stage, state.last_access}
                : layoutAccess(state.layout, AccessDirection::Source);
        const AccessScope dst = layoutAccess(new_layout, AccessDirection::Destination);

        transitionImage(command_buffer,
                        image,
                        aspect_mask,
                        new_layout,
                        AccessScope{src.stage, src.access},
                        AccessScope{dst.stage, dst.access});
    }

    void VulkanImageBarrierTracker::transitionImage(const VkCommandBuffer command_buffer,
                                                    const VkImage image,
                                                    const VkImageAspectFlags aspect_mask,
                                                    const VkImageLayout new_layout,
                                                    const AccessScope source,
                                                    const AccessScope destination) {
        if (command_buffer == VK_NULL_HANDLE || image == VK_NULL_HANDLE) {
            throw std::logic_error(std::format(
                "Image barrier transition requires non-null handles (command_buffer={:#x}, image={:#x}, aspect_mask={:#x}, requested_layout={}({})) ({}:{})",
                vkHandleValue(command_buffer),
                vkHandleValue(image),
                static_cast<std::uint32_t>(aspect_mask),
                vkImageLayoutToString(new_layout),
                static_cast<int>(new_layout),
                __FILE__,
                __LINE__));
        }

        [[maybe_unused]] const auto tracked = images_.find(image);
        LFS_VK_DEBUG_ASSERT(
            tracked != images_.end(),
            "Image barrier tracker does not know the explicitly transitioned image (image={:#x}, requested_layout={}({}), aspect_mask={:#x}, tracked_images={})",
            vkHandleValue(image),
            vkImageLayoutToString(new_layout),
            static_cast<int>(new_layout),
            static_cast<std::uint32_t>(aspect_mask),
            images_.size());
        auto& state = images_[image];
        if (state.aspect_mask == 0) {
            state.aspect_mask = aspect_mask;
        }
        if (state.layout == new_layout) {
            return;
        }

        VkImageMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.srcStageMask = source.stage;
        barrier.srcAccessMask = source.access;
        barrier.dstStageMask = destination.stage;
        barrier.dstAccessMask = destination.access;
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
        state.last_stage = destination.stage;
        state.last_access = destination.access;
    }

} // namespace lfs::vis
