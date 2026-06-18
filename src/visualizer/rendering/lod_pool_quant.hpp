/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Canonical quantized layout of the LOD page pool. Locked against the
// converter's streaming profile (means f32_lebytes, rgb r8_delta, scales
// ln_f16, orientation oct88r8, SH bands s8, alpha r8/f16) so the pool never
// adds loss over the file: s8 SH and log-f16 scales pass through bit-exact,
// f16 elsewhere is >=8x finer than the file's own 8-bit quantization.
// Shared by the dequant kernels (.cu), the upload engine, and the renderer's
// region layout; nvcc-safe.

#pragma once

#include <cstddef>
#include <cstdint>

namespace lfs::vis::lodq {

    // Per-splat bytes in each pool region. xyz stays f32x3: position drives
    // culling/selection shaders that are not quant-aware, and the file stores
    // f32 there anyway.
    inline constexpr std::size_t kXyzBytes = 12;     // f32x3
    inline constexpr std::size_t kSh0Bytes = 8;      // f16x3 + 16-bit pad (uint2)
    inline constexpr std::size_t kShNSlotBytes = 4;  // s8x4 per float4 slot (uchar4)
    inline constexpr std::size_t kRotationBytes = 8; // f16x4, pool order (w,x,y,z)
    inline constexpr std::size_t kScalingBytes = 8;  // log-domain f16x3 + pad
    inline constexpr std::size_t kOpacityBytes = 2;  // f16, post lod/logit transform

    // Node metadata stays in the sidecar's quantized records; the selector
    // dequantizes against the page frame. logical is derived from
    // page_to_chunk, so the expanded 16 B records are gone (32 -> 20 B/node).
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

} // namespace lfs::vis::lodq
