/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "viewport_frame_result.hpp"
#include "viewport_interop_service.hpp"

#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>

namespace lfs::vis {
    struct VulkanViewportFrameResources {
        VkImage external_image = VK_NULL_HANDLE;
        VkImageView external_image_view = VK_NULL_HANDLE;
        VkImageLayout external_image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        std::uint64_t external_image_generation = 0;
        VkSemaphore completion_semaphore = VK_NULL_HANDLE;
        std::uint64_t completion_value = 0;
        std::vector<ViewportInteropService::FrameCompletion> additional_completions;
    };
} // namespace lfs::vis
