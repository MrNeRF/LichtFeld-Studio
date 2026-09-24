/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Quantized LOD pool layout for RAD pages and resident tensors.

#pragma once

#include <cstddef>
#include <cstdint>

namespace lfs::core::radq {

    // Per-splat bytes in each pool region. xyz stays f32x3: position drives
    // culling/selection shaders that require float positions.
    inline constexpr std::size_t kXyzBytes = 12;     // f32x3
    inline constexpr std::size_t kSh0Bytes = 8;      // f16x3 + 16-bit pad (uint2)
    inline constexpr std::size_t kShNSlotBytes = 4;  // s8x4 per float4 slot (uchar4)
    inline constexpr std::size_t kRotationBytes = 8; // f16x4, pool order (w,x,y,z)
    inline constexpr std::size_t kScalingBytes = 8;  // log-domain f16x3 + pad
    inline constexpr std::size_t kOpacityBytes = 2;  // f16, post lod/logit transform

    // Node metadata stays in the sidecar's quantized records; the selector
    // dequantizes against the page frame. Logical indices derive from
    // page_to_chunk.
    inline constexpr std::size_t kMetaBoundsBytes = 8; // RadMetaBoundsQ (u16 x4)
    inline constexpr std::size_t kMetaLinksBytes = 12; // RadMetaLinksQ (u32 x3)
    inline constexpr std::size_t kMetaLinksWords = 3;

    // Per-page dequant frame in the InputPageFrames region: float4[4].
    //   [0] = (sh1_max_abs, sh2_max_abs, sh3_max_abs, unused)
    //   [1] = (bbox_min.xyz, log_size_min)
    //   [2] = (bbox_extent.xyz, log_size_range)
    //   [3] reserved
    // The from-tensors payload path writes only [0]; the render thread owns
    // [1..2] for sync-published pages (two writers, disjoint slots).
    inline constexpr std::size_t kPageFrameBytes = 64;
    inline constexpr std::size_t kPageFrameFloat4s = kPageFrameBytes / 16;
    inline constexpr std::size_t kPageFrameBoundsOffset = 16;

    struct PageFrame {
        float sh_max[3] = {0.0f, 0.0f, 0.0f};
        float reserved0 = 0.0f;
        float bbox_min[3] = {0.0f, 0.0f, 0.0f};
        float log_size_min = 0.0f;
        float bbox_extent[3] = {0.0f, 0.0f, 0.0f};
        float log_size_range = 0.0f;
        float reserved[4] = {};
    };
    static_assert(sizeof(PageFrame) == kPageFrameBytes);

} // namespace lfs::core::radq
