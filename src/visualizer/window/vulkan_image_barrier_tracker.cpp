/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_image_barrier_tracker.hpp"
#include "vulkan_result.hpp"

#include <stdexcept>

namespace lfs::vis {
    namespace {

        constexpr VkAccessFlags2 kWriteAccessMask =
            VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT |
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT |
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_2_HOST_WRITE_BIT | VK_ACCESS_2_TRANSFORM_FEEDBACK_WRITE_BIT_EXT |
            VK_ACCESS_2_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT |
            VK_ACCESS_2_COMMAND_PREPROCESS_WRITE_BIT_EXT | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;

    } // namespace

    bool VulkanImageBarrierTracker::isWriteScope(const AccessScope scope) noexcept {
        return (scope.access & kWriteAccessMask) != 0;
    }

    void VulkanImageBarrierTracker::accumulateReader(ImageState& state,
                                                     const AccessScope destination) noexcept {
        state.reader_stages |= destination.stage;
        state.reader_access |= destination.access;
    }

    void VulkanImageBarrierTracker::applyWriteDestination(ImageState& state,
                                                          const AccessScope destination) noexcept {
        state.last_stage = destination.stage;
        state.last_access = destination.access;
        state.reader_stages = VK_PIPELINE_STAGE_2_NONE;
        state.reader_access = VK_ACCESS_2_NONE;
    }

    void VulkanImageBarrierTracker::emitBarrier(const VkCommandBuffer command_buffer,
                                                const VkImage image,
                                                const VkImageAspectFlags aspect_mask,
                                                const VkImageLayout old_layout,
                                                const VkImageLayout new_layout,
                                                const AccessScope source,
                                                const AccessScope destination) const {
        VkImageMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.srcStageMask = source.stage;
        barrier.srcAccessMask = source.access;
        barrier.dstStageMask = destination.stage;
        barrier.dstAccessMask = destination.access;
        barrier.oldLayout = old_layout;
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
        if (cmd_pipeline_barrier2_ == nullptr) {
            throw std::logic_error(
                "VulkanImageBarrierTracker pipeline barrier emitter is null");
        }
        cmd_pipeline_barrier2_(command_buffer, &dependency);
    }

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

    void VulkanImageBarrierTracker::clearSwapchainOnly() {
        for (auto it = images_.begin(); it != images_.end();) {
            if (it->second.external) {
                ++it;
            } else {
                it = images_.erase(it);
            }
        }
    }

    void VulkanImageBarrierTracker::forgetImage(const VkImage image,
                                                const std::uint64_t generation) {
        if (image == VK_NULL_HANDLE) {
            return;
        }
        const auto it = images_.find(image);
        if (it != images_.end() && it->second.generation == generation) {
            images_.erase(it);
        }
    }

    void VulkanImageBarrierTracker::registerImage(const VkImage image,
                                                  const std::uint64_t generation,
                                                  const VkImageAspectFlags aspect_mask,
                                                  const VkImageLayout layout,
                                                  const bool external) {
        if (image == VK_NULL_HANDLE) {
            return;
        }
        const AccessScope access = layoutAccess(layout, AccessDirection::Source);
        images_[image] = Entry{
            .generation = generation,
            .state =
                ImageState{
                    .aspect_mask = aspect_mask,
                    .layout = layout,
                    .last_stage = access.stage,
                    .last_access = access.access,
                    .reader_stages = VK_PIPELINE_STAGE_2_NONE,
                    .reader_access = VK_ACCESS_2_NONE,
                },
            .external = external,
        };
    }

    VkImageLayout VulkanImageBarrierTracker::imageLayout(const VkImage image,
                                                         const std::uint64_t generation,
                                                         const VkImageLayout fallback) const {
        const auto it = images_.find(image);
        if (it == images_.end() || it->second.generation != generation) {
            return fallback;
        }
        return it->second.state.layout;
    }

    void VulkanImageBarrierTracker::transitionImage(const VkCommandBuffer command_buffer,
                                                    const VkImage image,
                                                    const std::uint64_t generation,
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

        const auto tracked = images_.find(image);
        const bool generation_matches =
            tracked != images_.end() && tracked->second.generation == generation;
        LFS_VK_DEBUG_ASSERT(
            generation_matches,
            "Image barrier tracker does not know the transitioned image (image={:#x}, generation={}, requested_layout={}({}), aspect_mask={:#x}, tracked_images={})",
            vkHandleValue(image),
            generation,
            vkImageLayoutToString(new_layout),
            static_cast<int>(new_layout),
            static_cast<std::uint32_t>(aspect_mask),
            images_.size());

        if (!generation_matches) {
            // Untracked / stale generation: emit a conservative barrier and leave map state alone.
            const AccessScope src = layoutAccess(VK_IMAGE_LAYOUT_UNDEFINED, AccessDirection::Source);
            const AccessScope dst = layoutAccess(new_layout, AccessDirection::Destination);
            if (VK_IMAGE_LAYOUT_UNDEFINED != new_layout || src.stage != VK_PIPELINE_STAGE_2_NONE ||
                src.access != VK_ACCESS_2_NONE || dst.stage != VK_PIPELINE_STAGE_2_NONE ||
                dst.access != VK_ACCESS_2_NONE) {
                emitBarrier(command_buffer,
                            image,
                            aspect_mask,
                            VK_IMAGE_LAYOUT_UNDEFINED,
                            new_layout,
                            src.stage != VK_PIPELINE_STAGE_2_NONE || src.access != VK_ACCESS_2_NONE
                                ? src
                                : layoutAccess(VK_IMAGE_LAYOUT_GENERAL, AccessDirection::Source),
                            dst);
            }
            return;
        }

        auto& state = tracked->second.state;
        if (state.aspect_mask == 0) {
            state.aspect_mask = aspect_mask;
        }

        const AccessScope dst = layoutAccess(new_layout, AccessDirection::Destination);
        if (state.layout == new_layout) {
            if (!isWriteScope(dst)) {
                accumulateReader(state, dst);
            } else {
                applyWriteDestination(state, dst);
            }
            return;
        }

        // Any layout-changing transition (read or write destination) must WAR on
        // accumulated readers; the transition rewrites the image in place.
        AccessScope src{
            state.last_stage | state.reader_stages,
            state.last_access | state.reader_access,
        };
        if (src.stage == VK_PIPELINE_STAGE_2_NONE && src.access == VK_ACCESS_2_NONE) {
            src = layoutAccess(state.layout, AccessDirection::Source);
        }

        transitionImage(command_buffer,
                        image,
                        generation,
                        aspect_mask,
                        new_layout,
                        src,
                        dst);
    }

    void VulkanImageBarrierTracker::transitionImage(const VkCommandBuffer command_buffer,
                                                    const VkImage image,
                                                    const std::uint64_t generation,
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

        const auto tracked = images_.find(image);
        const bool generation_matches =
            tracked != images_.end() && tracked->second.generation == generation;
        LFS_VK_DEBUG_ASSERT(
            generation_matches,
            "Image barrier tracker does not know the explicitly transitioned image (image={:#x}, generation={}, requested_layout={}({}), aspect_mask={:#x}, tracked_images={})",
            vkHandleValue(image),
            generation,
            vkImageLayoutToString(new_layout),
            static_cast<int>(new_layout),
            static_cast<std::uint32_t>(aspect_mask),
            images_.size());

        if (!generation_matches) {
            // Caller scopes are used verbatim even on the untracked path; map state is untouched.
            emitBarrier(command_buffer,
                        image,
                        aspect_mask,
                        VK_IMAGE_LAYOUT_UNDEFINED,
                        new_layout,
                        source,
                        destination);
            return;
        }

        auto& state = tracked->second.state;
        if (state.aspect_mask == 0) {
            state.aspect_mask = aspect_mask;
        }

        if (state.layout == new_layout) {
            // Same-layout early return still accumulates the destination scope for readers.
            if (!isWriteScope(destination)) {
                accumulateReader(state, destination);
            } else {
                applyWriteDestination(state, destination);
            }
            return;
        }

        // Explicit-scope: caller source/destination are used VERBATIM (never OR readers into src).
        emitBarrier(command_buffer,
                    image,
                    aspect_mask,
                    state.layout,
                    new_layout,
                    source,
                    destination);

        // After any layout-changing transition the barrier has consumed prior
        // readers/writer; destination becomes the new last access and readers reset.
        state.aspect_mask = aspect_mask;
        state.layout = new_layout;
        applyWriteDestination(state, destination);
    }

} // namespace lfs::vis
