/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <cstdint>
#include <unordered_map>
#include <vulkan/vulkan.h>

namespace lfs::vis {

    class LFS_VIS_API VulkanImageBarrierTracker {
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
            // Readers since the last write (WAR sources for the next writer).
            VkPipelineStageFlags2 reader_stages = VK_PIPELINE_STAGE_2_NONE;
            VkAccessFlags2 reader_access = VK_ACCESS_2_NONE;
        };

        struct Entry {
            std::uint64_t generation = 0;
            ImageState state{};
            bool external = false;
        };

        void clearSwapchainOnly();
        void forgetImage(VkImage image, std::uint64_t generation);
        void registerImage(VkImage image,
                           std::uint64_t generation,
                           VkImageAspectFlags aspect_mask,
                           VkImageLayout layout,
                           bool external = false);

        [[nodiscard]] VkImageLayout imageLayout(VkImage image,
                                                std::uint64_t generation,
                                                VkImageLayout fallback = VK_IMAGE_LAYOUT_UNDEFINED) const;

        [[nodiscard]] static AccessScope layoutAccess(VkImageLayout layout,
                                                      AccessDirection direction) noexcept;

        void transitionImage(VkCommandBuffer command_buffer,
                             VkImage image,
                             std::uint64_t generation,
                             VkImageAspectFlags aspect_mask,
                             VkImageLayout new_layout);

        // Layouts do not identify the queue or shader stage that produced/consumes
        // an image. Cross-queue users must provide the scopes represented by the
        // submission's semaphore edges instead of inheriting a graphics-only scope.
        void transitionImage(VkCommandBuffer command_buffer,
                             VkImage image,
                             std::uint64_t generation,
                             VkImageAspectFlags aspect_mask,
                             VkImageLayout new_layout,
                             AccessScope source,
                             AccessScope destination);

        // Injectable emission seam (tests capture barriers without a device).
        void setPipelineBarrierEmitter(PFN_vkCmdPipelineBarrier2 emitter) noexcept {
            cmd_pipeline_barrier2_ = emitter != nullptr ? emitter : vkCmdPipelineBarrier2;
        }

    private:
        [[nodiscard]] static bool isWriteScope(AccessScope scope) noexcept;
        void emitBarrier(VkCommandBuffer command_buffer,
                         VkImage image,
                         VkImageAspectFlags aspect_mask,
                         VkImageLayout old_layout,
                         VkImageLayout new_layout,
                         AccessScope source,
                         AccessScope destination) const;
        void accumulateReader(ImageState& state, AccessScope destination) noexcept;
        void applyWriteDestination(ImageState& state, AccessScope destination) noexcept;

        std::unordered_map<VkImage, Entry> images_;
        PFN_vkCmdPipelineBarrier2 cmd_pipeline_barrier2_ = vkCmdPipelineBarrier2;
    };

} // namespace lfs::vis
