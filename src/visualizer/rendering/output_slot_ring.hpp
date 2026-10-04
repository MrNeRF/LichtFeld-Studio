/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include "generic_output_slot_ring.hpp"
#include "render_target_id.hpp"
#include "window/vulkan_context.hpp"
#include <unordered_map>
#include <vector>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <glm/glm.hpp>
#include <string_view>
#include <vulkan/vulkan.h>

namespace lfs::vis {

    // Per (logical × ring) image payload + bookkeeping. Slot images are
    // non-owning copies; OutputImagePool owns destruction via the serials.
    struct OutputImageSlot {
        VulkanContext::ExternalImage image{};
        VulkanContext::ExternalImage depth_image{};
        glm::ivec2 size{0, 0};       // valid/logical extent
        glm::ivec2 alloc_size{0, 0}; // allocated (ceil64) extent
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImageLayout depth_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        // Content identity for consumers (compose may overwrite). Not a tracker key.
        std::uint64_t generation = 0;
        // Resource identity for VulkanImageBarrierTracker. Set from the color
        // pool acquisition serial when the VkImages are (re)acquired.
        std::uint64_t image_generation = 0;
        // Pool acquisition serials (0 = none).
        std::uint64_t color_pool_serial = 0;
        std::uint64_t depth_pool_serial = 0;
        // Timeline value signalled by the compute submission that produced
        // this exact ring image. Graphics-queue readbacks wait this value.
        std::uint64_t completion_value = 0;
    };

    using OutputSlotRing = BasicOutputSlotRing<OutputImageSlot>;

} // namespace lfs::vis
