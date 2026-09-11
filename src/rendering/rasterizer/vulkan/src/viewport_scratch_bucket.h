/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// Viewport scratch capacity is bucketed to a 64-px grid (issue #1565), matching
// visualizer output_image_pool::ceil64. Lives in the rasterizer so this lib does
// not include visualizer headers. Allocation uses alloc_*; indexing/uniforms
// stay on the caller's logical extents.

#include "config.h"

#include <cstddef>
#include <cstdint>

namespace lfs::rendering::vulkan {

    // 16-px tiles must divide the 64-px bucket so alloc tile counts are exact.
    static_assert(TILE_WIDTH > 0 && 64 % TILE_WIDTH == 0,
                  "TILE_WIDTH must divide the 64-px viewport scratch bucket");
    static_assert(TILE_HEIGHT > 0 && 64 % TILE_HEIGHT == 0,
                  "TILE_HEIGHT must divide the 64-px viewport scratch bucket");

    [[nodiscard]] constexpr std::uint32_t ceil64(const std::uint32_t v) noexcept {
        return ((v + 63u) / 64u) * 64u;
    }

    struct ViewportScratchBucket {
        std::uint32_t alloc_w = 0;
        std::uint32_t alloc_h = 0;
        std::size_t alloc_pixels = 0;
        std::size_t alloc_tiles = 0;
    };

    // Capacity extents for pixel/tile scratch: ceil64 per axis, size_t products.
    [[nodiscard]] constexpr ViewportScratchBucket viewportScratchBucket(
        const std::uint32_t logical_w,
        const std::uint32_t logical_h) noexcept {
        ViewportScratchBucket b;
        b.alloc_w = ceil64(logical_w);
        b.alloc_h = ceil64(logical_h);
        b.alloc_pixels =
            static_cast<std::size_t>(b.alloc_w) * static_cast<std::size_t>(b.alloc_h);
        b.alloc_tiles =
            static_cast<std::size_t>(_CEIL_DIV(b.alloc_w, static_cast<std::uint32_t>(TILE_WIDTH))) *
            static_cast<std::size_t>(_CEIL_DIV(b.alloc_h, static_cast<std::uint32_t>(TILE_HEIGHT)));
        return b;
    }

} // namespace lfs::rendering::vulkan
