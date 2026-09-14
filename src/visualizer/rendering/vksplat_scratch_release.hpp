/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "rendering/rasterizer/vulkan/src/buffer.h"

namespace lfs::vis::detail {

    // Invalidate every view immediately; only owned allocations enter the
    // caller's timeline retirement queue. Aliases must not retain capacity
    // that would allow a later resize to reuse the retired owner's handle.
    template <typename T, typename Retire>
    std::size_t releaseScratchBuffer(Buffer<T>& buffer, Retire&& retire) {
        auto& dev = buffer.deviceBuffer;
        if (dev.buffer == VK_NULL_HANDLE) {
            return 0;
        }
        _VulkanBuffer old = dev;
        dev = {};
        dev.label = old.label;
        dev.extra_usage = old.extra_usage;
        buffer.clear();
        buffer.shrink_to_fit();
        if (old.allocation == VK_NULL_HANDLE) {
            return 0;
        }
        const auto bytes = old.allocSize;
        retire(old);
        return bytes;
    }

} // namespace lfs::vis::detail
