// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
// Layouts shared by the tensor-program splat rasterizer passes. They match
// the native Metal rasterizer (src/rendering/rasterizer/metal) byte for byte.
#pragma once

struct ProjectedSplat {
    float4 mean_depth;
    float4 conic_opacity;
    float4 color; // w holds the positive depth key bits
    uint4 bounds; // pixel rectangle [x0, y0, x1, y1)
};

struct RasterParameters {
    uint count, width, height, columns;
    uint tiles, capacity, mode, unused;
    float4 background;
    float4 render_origin;
    float4 intrinsics, clip;
    uint4 camera;
    float4 panorama;
    uint4 mask_limits;
};

struct RasterStatus {
    uint64_t required;
    uint error, blend_threads, maximum_tile_instances, padding;
};

// Indirect dispatch layout in the shared argument tensor.
static const uint kSortArgs = 0;   // threadgroups of 2048 sort entries
static const uint kRangeArgs = 3;  // threadgroups of 256 instances
static const uint kWaveWidth = 32; // Apple and the Vulkan reference run 32-wide subgroups

uint live_sort_blocks(RasterStatus status) {
    return max(1u, uint(((status.error != 0 ? 0ul : status.required) + 2047ul) / 2048ul));
}
