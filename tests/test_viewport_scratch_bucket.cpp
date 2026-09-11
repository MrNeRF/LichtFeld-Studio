/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/rasterizer/vulkan/src/viewport_scratch_bucket.h"

#include <gtest/gtest.h>

using lfs::rendering::vulkan::ceil64;
using lfs::rendering::vulkan::viewportScratchBucket;

TEST(ViewportScratchBucket, Ceil64Boundaries) {
    EXPECT_EQ(ceil64(0), 0u);
    EXPECT_EQ(ceil64(1), 64u);
    EXPECT_EQ(ceil64(63), 64u);
    EXPECT_EQ(ceil64(64), 64u);
    EXPECT_EQ(ceil64(65), 128u);
    EXPECT_EQ(ceil64(128), 128u);
    EXPECT_EQ(ceil64(129), 192u);
}

TEST(ViewportScratchBucket, TileDivisibilityAssumptions) {
    // Scratch tile counts use TILE_WIDTH/HEIGHT over ceil64 extents; 16 | 64.
    EXPECT_EQ(64 % TILE_WIDTH, 0);
    EXPECT_EQ(64 % TILE_HEIGHT, 0);
    EXPECT_EQ(TILE_WIDTH, 16);
    EXPECT_EQ(TILE_HEIGHT, 16);
}

TEST(ViewportScratchBucket, AllocPixelsAndTiles) {
    {
        const auto b = viewportScratchBucket(1, 1);
        EXPECT_EQ(b.alloc_w, 64u);
        EXPECT_EQ(b.alloc_h, 64u);
        EXPECT_EQ(b.alloc_pixels, 64u * 64u);
        EXPECT_EQ(b.alloc_tiles, 4u * 4u); // 64/16
    }
    {
        const auto b = viewportScratchBucket(63, 64);
        EXPECT_EQ(b.alloc_w, 64u);
        EXPECT_EQ(b.alloc_h, 64u);
        EXPECT_EQ(b.alloc_pixels, 64u * 64u);
        EXPECT_EQ(b.alloc_tiles, 4u * 4u);
    }
    {
        const auto b = viewportScratchBucket(65, 129);
        EXPECT_EQ(b.alloc_w, 128u);
        EXPECT_EQ(b.alloc_h, 192u);
        EXPECT_EQ(b.alloc_pixels, static_cast<std::size_t>(128) * 192);
        EXPECT_EQ(b.alloc_tiles, static_cast<std::size_t>(8) * 12);
    }
    {
        // Exact bucket: no padding.
        const auto b = viewportScratchBucket(128, 128);
        EXPECT_EQ(b.alloc_w, 128u);
        EXPECT_EQ(b.alloc_h, 128u);
        EXPECT_EQ(b.alloc_pixels, 128u * 128u);
        EXPECT_EQ(b.alloc_tiles, 8u * 8u);
    }
}
