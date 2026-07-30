// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "config.slang"
#include "indirect_layout.inc"

import utils;

#define _ENTRY_generate_keys 1
#define _ENTRY_compute_tile_ranges 2
#define _ENTRY_compute_tile_ranges_and_batch_counts 3

#ifndef ENTRY
#define ENTRY 0
#endif

// LFS_VISIBLE_BOUNDED=1 (HiGS chain): the primitive domain is the compact
// visible set, bounded by a GPU-resident count instead of uniforms.num_splats,
// and the cumsum tail lives at [visible_count - 1].
#ifndef LFS_VISIBLE_BOUNDED
#define LFS_VISIBLE_BOUNDED 0
#endif

static const uint32_t FLOAT32_FRACTION_BITS = 23;


#if ENTRY == _ENTRY_generate_keys

// Two-stage sort: this dispatch walks primitives in depth-rank order
// (output of stage-1 sort over per-primitive depth keys). For each depth_rank
// we look up the original primitive index via `primitive_sort_indices`, read
// per-primitive metadata at that index, and emit one key per touched tile.
//
// Keys are tile-id only (no depth bits packed in). The stage-2 radix sort is
// stable per digit, so the write order produced by this dispatch — depth
// ascending across primitives, then binning order within a primitive — is
// preserved within each tile bucket after the sort.
//
// Matches gsplat_fwd/kernels_forward.cuh::create_instances_cu.

layout(binding=0) StructuredBuffer<float> xy_vs;
layout(binding=1) StructuredBuffer<float> inv_cov_vs_opacity;
#if USE_EMULATED_INT64
layout(binding=2) StructuredBuffer<int32_t> rect_tile_space;
#else
layout(binding=2) StructuredBuffer<int64_t> rect_tile_space;
#endif
layout(binding=3) StructuredBuffer<int32_t> index_buffer_offset;
layout(binding=4) StructuredBuffer<int32_t> primitive_sort_indices;

layout(binding=5) RWStructuredBuffer<sortingKey_t> out_unsorted_keys;
layout(binding=6) RWStructuredBuffer<int32_t> out_unsorted_gauss_idx;
#if LFS_VISIBLE_BOUNDED
layout(binding=7) StructuredBuffer<uint32_t> visible_count;
#else
layout(binding=7) StructuredBuffer<uint32_t> depth_wave_dispatch;
#endif

#define BLOCK_SIZE 64
#define NUM_SUBGROUPS (BLOCK_SIZE / SUBGROUP_SIZE)

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uint3 groupThreadID : SV_GroupThreadID,
    uniform Uniforms uniforms
) {
    const uint image_height = uniforms.image_height;
    const uint image_width = uniforms.image_width;
    const uint grid_height = uniforms.grid_height;
    const uint grid_width = uniforms.grid_width;
    const bool wrap_x = base_camera_model(uniforms.camera_model) == uint(CameraModel::EQUIRECTANGULAR);

#if LFS_VISIBLE_BOUNDED
    int32_t depth_rank = dispatchThreadID.x;
#else
    const uint record = (1u + uniforms.depth_wave) * LFS_VK_DEPTH_WAVE_RECORD_STRIDE_WORDS;
    const uint local_rank = dispatchThreadID.x;
    const uint rank_count = depth_wave_dispatch[record + LFS_VK_DEPTH_WAVE_RANK_COUNT_WORD];
    if (local_rank >= rank_count)
        return;
    int32_t depth_rank = int32_t(
        depth_wave_dispatch[record + LFS_VK_DEPTH_WAVE_RANK_BASE_WORD] + local_rank);
#endif

#if LFS_VISIBLE_BOUNDED
    // The cumsum only covers [0, visible_count) entries; offsets past it are
    // uninitialized, so this bound is a correctness gate, not an optimization.
    if (uint32_t(depth_rank) >= min(visible_count[0], uint32_t(uniforms.num_splats)))
        return;
#else
    if (depth_rank >= uniforms.num_splats)
        return;
#endif

    // index_buffer_offset is cumsum'd over the depth-ordered tiles_touched
    // array, so the offset interval for this depth_rank is contiguous.
#if LFS_VISIBLE_BOUNDED
    const int instance_base = 0;
#else
    const int instance_base =
        int(depth_wave_dispatch[record + LFS_VK_DEPTH_WAVE_INSTANCE_BASE_WORD]);
#endif
    int offset = (depth_rank == 0 ? 0 : index_buffer_offset[depth_rank-1]) - instance_base;
    int offset_max = index_buffer_offset[depth_rank] - instance_base;
    const int sort_capacity = int(uniforms.sort_capacity);
    if (sort_capacity <= 0 || offset < 0 || offset_max < 0 ||
        offset >= sort_capacity || offset >= offset_max || offset_max > sort_capacity)
        return;

    int32_t primitive_idx = primitive_sort_indices[depth_rank];

#if TILE_CULLING_MODE > 0
    float2 xy = read_t2_float2(primitive_idx, xy_vs);
    float4 inv_cov_opac = read_t4_float4(primitive_idx, inv_cov_vs_opacity);

    float extend = extend_from_opacity(inv_cov_opac.w);
    float3 inv_cov = inv_cov_opac.xyz / (extend*extend);
#endif  // TILE_CULLING_MODE > 0

  #if USE_EMULATED_INT64
    rectangle rect;
    rect.min_x = (uint16_t)rect_tile_space[2*primitive_idx+0];
    rect.min_y = (uint16_t)(rect_tile_space[2*primitive_idx+0] >> 16);
    rect.max_x = (uint16_t)rect_tile_space[2*primitive_idx+1];
    rect.max_y = (uint16_t)(rect_tile_space[2*primitive_idx+1] >> 16);
  #else
    rectangle rect = reinterpret<rectangle>(rect_tile_space[primitive_idx]);
  #endif

#if TILE_CULLING_MODE == 0 || TILE_CULLING_MODE == 1

    for (uint16_t y = rect.min_y; y < rect.max_y; y++) {
        for (uint16_t x = rect.min_x; x < rect.max_x; x++) {
            if (offset >= offset_max)
                return;
            uint32_t key_x = wrap_x ? uint32_t(x) % grid_width : uint32_t(x);
            uint32_t key = uint32_t(y) * grid_width + key_x;
            out_unsorted_keys[offset] = sortingKey_t(key);
            out_unsorted_gauss_idx[offset] = primitive_idx;
            offset++;
        }
    }

#endif  // TILE_CULLING_MODE == 0 || TILE_CULLING_MODE == 1

#if TILE_CULLING_MODE == 2

    if (wrap_x) {
        for (uint16_t y = rect.min_y; y < rect.max_y; y++) {
            for (uint16_t x = rect.min_x; x < rect.max_x && offset < offset_max; x++) {
                uint32_t key = uint32_t(y) * grid_width + (uint32_t(x) % grid_width);
                out_unsorted_keys[offset] = sortingKey_t(key);
                out_unsorted_gauss_idx[offset] = primitive_idx;
                offset++;
            }
        }
        return;
    }

    if (rect.max_y-rect.min_y <= rect.max_x-rect.min_x) {
        for (uint16_t y = rect.min_y; y < rect.max_y; y++) {
            float y0 = y * TILE_HEIGHT - xy.y;
            float y1 = y0 + TILE_HEIGHT;

            float2 bound = ellipse_range_bound(inv_cov, y0, y1);
            uint16_t min_x = uint16_t(clamp(
                floor((bound.x + xy.x) / TILE_WIDTH),
                rect.min_x, rect.max_x
            ));
            uint16_t max_x = uint16_t(clamp(
                ceil((bound.y + xy.x) / TILE_WIDTH),
                rect.min_x, rect.max_x
            ));
            for (uint16_t x = min_x; x < max_x && offset < offset_max; x++) {
                uint32_t key = uint32_t(y) * grid_width + uint32_t(x);
                out_unsorted_keys[offset] = sortingKey_t(key);
                out_unsorted_gauss_idx[offset] = primitive_idx;
                offset++;
            }
        }
    } else {
        inv_cov = inv_cov.zyx;
        for (uint16_t x = rect.min_x; x < rect.max_x; x++) {
            float x0 = x * TILE_WIDTH - xy.x;
            float x1 = x0 + TILE_WIDTH;

            float2 bound = ellipse_range_bound(inv_cov, x0, x1);
            uint16_t min_y = uint16_t(clamp(
                floor((bound.x + xy.y) / TILE_HEIGHT),
                rect.min_y, rect.max_y
            ));
            uint16_t max_y = uint16_t(clamp(
                ceil((bound.y + xy.y) / TILE_HEIGHT),
                rect.min_y, rect.max_y
            ));
            for (uint16_t y = min_y; y < max_y && offset < offset_max; y++) {
                uint32_t key = uint32_t(y) * grid_width + uint32_t(x);
                out_unsorted_keys[offset] = sortingKey_t(key);
                out_unsorted_gauss_idx[offset] = primitive_idx;
                offset++;
            }
        }
    }

    // Pad over-estimated tail with max-tile sentinel so excess slots sort to
    // the end (where they fall outside the per-tile range scan).
    for (; offset < offset_max; offset++) {
        out_unsorted_keys[offset] = sortingKey_t(grid_width * grid_height);
        out_unsorted_gauss_idx[offset] = primitive_idx;
    }

#endif  // TILE_CULLING_MODE == 2
}

#endif  // ENTRY == _ENTRY_generate_keys



#if ENTRY == _ENTRY_compute_tile_ranges

layout(binding=0) StructuredBuffer<sortingKey_t> sorted_keys;
layout(binding=1) RWStructuredBuffer<int32_t> out_tile_ranges;
layout(binding=2) StructuredBuffer<uint32_t> tile_instance_count;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    uint32_t globalIdx = dispatchThreadID.x;

    const uint grid_height = uniforms.grid_height;
    const uint grid_width = uniforms.grid_width;
    const uint32_t num_tiles = grid_height*grid_width;

    const uint32_t num_isects = tile_instance_count[0];
    if (globalIdx > num_isects)
        return;

    uint32_t currtile = num_tiles;
    if (globalIdx < num_isects) {
        currtile = min((uint32_t)sorted_keys[globalIdx], num_tiles);
    }
    uint32_t prevtile = globalIdx == 0 ? (uint32_t)-1 :
        min((uint32_t)sorted_keys[globalIdx-1], num_tiles);
    for (uint32_t i = prevtile+1; i <= currtile; ++i) {
        out_tile_ranges[i] = globalIdx;
    }
}

#endif  // ENTRY == _ENTRY_compute_tile_ranges


#if ENTRY == _ENTRY_compute_tile_ranges_and_batch_counts

layout(binding=0) StructuredBuffer<sortingKey_t> sorted_keys;
layout(binding=1) RWStructuredBuffer<int32_t> out_tile_ranges;
layout(binding=2) StructuredBuffer<uint32_t> tile_instance_count;
layout(binding=3) RWStructuredBuffer<int32_t> out_tile_batch_counts;

// One thread owns both boundaries and the derived count for a tile. Binary
// searches avoid a cross-workgroup ranges->counts dependency and its barrier.
[ForceInline]
uint lower_bound_tile(uint target, uint count) {
    uint low = 0u;
    uint high = count;
    while (low < high) {
        const uint mid = low + (high - low) / 2u;
        if (uint(sorted_keys[mid]) < target)
            low = mid + 1u;
        else
            high = mid;
    }
    return low;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    const uint tile_idx = dispatchThreadID.x;
    const uint num_tiles = uniforms.grid_height * uniforms.grid_width;
    if (tile_idx >= num_tiles)
        return;

    const uint num_isects = tile_instance_count[0];
    const uint range_start = lower_bound_tile(tile_idx, num_isects);
    const uint range_end = lower_bound_tile(tile_idx + 1u, num_isects);
    out_tile_ranges[tile_idx] = int32_t(range_start);
    if (tile_idx + 1u == num_tiles)
        out_tile_ranges[num_tiles] = int32_t(range_end);

    const uint tile_count = range_end - range_start;
    out_tile_batch_counts[tile_idx] =
        tile_count > RASTER_DENSE_TILE_THRESHOLD
            ? int32_t((tile_count + RASTER_BATCH_SIZE - 1u) / RASTER_BATCH_SIZE)
            : 0;
}

#endif  // ENTRY == _ENTRY_compute_tile_ranges_and_batch_counts
