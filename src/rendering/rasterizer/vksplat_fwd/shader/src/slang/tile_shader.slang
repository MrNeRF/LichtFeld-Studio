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

import utils;

#define _ENTRY_generate_keys 1
#define _ENTRY_compute_tile_ranges 2

#ifndef ENTRY
#define ENTRY 0
#endif

static const uint32_t FLOAT32_FRACTION_BITS = 23;


#if ENTRY == _ENTRY_generate_keys

layout(binding=0) StructuredBuffer<float> xy_vs;
layout(binding=1) StructuredBuffer<float> inv_cov_vs_opacity;
layout(binding=2) StructuredBuffer<float> depths;
#if USE_EMULATED_INT64
layout(binding=3) StructuredBuffer<int32_t> rect_tile_space;
#else
layout(binding=3) StructuredBuffer<int64_t> rect_tile_space;
#endif
layout(binding=4) StructuredBuffer<int32_t> index_buffer_offset;

layout(binding=5) RWStructuredBuffer<sortingKey_t> out_unsorted_keys;
layout(binding=6) RWStructuredBuffer<int32_t> out_unsorted_gauss_idx;

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

    int32_t globalIdx = dispatchThreadID.x;
    
    if (globalIdx >= uniforms.num_splats)
        return;

    uint32_t DEPTH_BITS = min(uint32_t(31.99-log2(grid_height*grid_width)), FLOAT32_FRACTION_BITS);

#if TILE_CULLING_MODE > 0
    float2 xy = read_t2_float2(globalIdx, xy_vs);
    float4 inv_cov_opac = read_t4_float4(globalIdx, inv_cov_vs_opacity);

    float extend = extend_from_opacity(inv_cov_opac.w);
    float3 inv_cov = inv_cov_opac.xyz / (extend*extend);
#endif  // TILE_CULLING_MODE > 0

    float zf = read_t1_float(globalIdx, depths);
    zf = (2.0*zf+1.0)/(zf+1.0);  // between 1 and 2, make float bits easier
    uint32_t z = (reinterpret<uint32_t>(zf) & ((1u<<FLOAT32_FRACTION_BITS)-1u)) >> (FLOAT32_FRACTION_BITS-DEPTH_BITS);

  #if USE_EMULATED_INT64
    rectangle rect;
    rect.min_x = (uint16_t)rect_tile_space[2*globalIdx+0];
    rect.min_y = (uint16_t)(rect_tile_space[2*globalIdx+0] >> 16);
    rect.max_x = (uint16_t)rect_tile_space[2*globalIdx+1];
    rect.max_y = (uint16_t)(rect_tile_space[2*globalIdx+1] >> 16);
  #else
    rectangle rect = reinterpret<rectangle>(rect_tile_space[globalIdx]);
  #endif

    int offset = globalIdx == 0 ? 0 : index_buffer_offset[globalIdx-1];
    int offset_max = index_buffer_offset[globalIdx];

#if TILE_CULLING_MODE == 0 || TILE_CULLING_MODE == 1

    for (uint16_t y = rect.min_y; y < rect.max_y; y++) {
        for (uint16_t x = rect.min_x; x < rect.max_x; x++) {
            uint32_t key = uint32_t(y) * grid_width + uint32_t(x);
            out_unsorted_keys[offset] = sortingKey_t((key << DEPTH_BITS) | z);
            out_unsorted_gauss_idx[offset] = globalIdx;
            offset++;
        }
    }

#endif  // TILE_CULLING_MODE == 0 || TILE_CULLING_MODE == 1

#if TILE_CULLING_MODE == 2

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
                out_unsorted_keys[offset] = sortingKey_t((key << DEPTH_BITS) | z);
                out_unsorted_gauss_idx[offset] = globalIdx;
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
                out_unsorted_keys[offset] = sortingKey_t((key << DEPTH_BITS) | z);
                out_unsorted_gauss_idx[offset] = globalIdx;
                offset++;
            }
        }
    }

    for (; offset < offset_max; offset++) {
        // out_unsorted_keys[offset] = sortingKey_t(-1);
        out_unsorted_keys[offset] = sortingKey_t(
            (uint32_t(offset%(grid_width*grid_height)) << DEPTH_BITS) | ((1u << DEPTH_BITS) - 1u));
        out_unsorted_gauss_idx[offset] = globalIdx;
    }

#endif  // TILE_CULLING_MODE == 2
}

#endif  // ENTRY == _ENTRY_generate_keys



#if ENTRY == _ENTRY_compute_tile_ranges

layout(binding=0) StructuredBuffer<sortingKey_t> sorted_keys;
layout(binding=1) RWStructuredBuffer<int32_t> out_tile_ranges;
layout(binding=2) StructuredBuffer<int32_t> index_buffer_offset_in;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    int32_t globalIdx = dispatchThreadID.x;

    const uint grid_height = uniforms.grid_height;
    const uint grid_width = uniforms.grid_width;
    uint32_t DEPTH_BITS = min(uint32_t(31.99-log2(grid_height*grid_width)), FLOAT32_FRACTION_BITS);

    // num_isects (= num_indices) is read directly from the GPU-resident cumsum
    // tail to keep this dispatch CPU-readback-free.
    uint32_t num_isects =
        uniforms.num_splats == 0u
            ? 0u
            : (uint32_t)index_buffer_offset_in[uniforms.num_splats - 1u];
    if (globalIdx > num_isects)
        return;

    // Read tile ID from key. Update start/end of tile range if at limit.
    sortingKey_t key = sorted_keys[globalIdx];
    uint32_t currtile = globalIdx == num_isects ?
        grid_height*grid_width : (uint32_t)(key) >> DEPTH_BITS;
    uint32_t prevtile = globalIdx == 0 ? (uint32_t)-1 :
        (uint32_t)sorted_keys[globalIdx-1] >> DEPTH_BITS;
    for (uint32_t i = prevtile+1; i <= currtile; ++i) {
        out_tile_ranges[i] = globalIdx;
    }
}

#endif  // ENTRY == _ENTRY_compute_tile_ranges
