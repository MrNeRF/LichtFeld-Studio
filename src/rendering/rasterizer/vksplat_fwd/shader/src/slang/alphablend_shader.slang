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

#define SHARED_STRUCT 1

#if SUBTILE_SIZE > 0
#define SUBTILE_PER_SUBGROUP max(SUBGROUP_SIZE / SUBTILE_SIZE, 1)
#define SUBGRID_HEIGHT (TILE_HEIGHT/SUBTILE_HEIGHT)
#define SUBGRID_WIDTH (TILE_WIDTH/SUBTILE_WIDTH)
#define SUBGRID_SIZE (SUBGRID_HEIGHT*SUBGRID_WIDTH)
#endif


[ForceInline]
[Differentiable]
float4 update_pixel_state(float4 pixel_state_t_nm1, float4 gauss_rgba_t_n)
{
    float3 color_t_n = pixel_state_t_nm1.rgb + gauss_rgba_t_n.rgb * pixel_state_t_nm1.a;
    float transmittance_t_n = pixel_state_t_nm1.a * (float(1) - gauss_rgba_t_n.a);
    return float4(color_t_n, transmittance_t_n);
}

[ForceInline]
float4 undo_pixel_state(float4 pixel_state_t_n, float4 gauss_rgba_t_n)
{
    float transmittance_t_nm1 = pixel_state_t_n.a / (float(1) - gauss_rgba_t_n.a);
    float3 color_t_nm1 = pixel_state_t_n.rgb - gauss_rgba_t_n.rgb * transmittance_t_nm1;
    return float4(color_t_nm1, transmittance_t_nm1);
}

#if SUBTILE_SIZE > 0

[ForceInline]
void decodeThreadToSubtiles(uint thread_rank, inout uint2 pix_coord, out uint out_subgrid_idx, out uint out_pixel_rank) {
  #if 1  // align subtiles with subgroups
    uint subgrid_idx = thread_rank / SUBTILE_SIZE;
    uint subtile_offset = thread_rank % SUBTILE_SIZE;
    uint2 subgrid_idx2 = uint2(subgrid_idx % SUBGRID_WIDTH, subgrid_idx / SUBGRID_WIDTH);
    uint2 subtile_offset2 = uint2(subtile_offset % SUBTILE_WIDTH, subtile_offset / SUBTILE_WIDTH);
    uint2 tile_idx2 = pix_coord / uint2(TILE_WIDTH, TILE_HEIGHT);  // groupID.xy
    uint2 tile_offset2 = subgrid_idx2 * uint2(SUBTILE_WIDTH, SUBTILE_HEIGHT) + subtile_offset2;
    pix_coord = uint2(tile_idx2) * uint2(TILE_WIDTH, TILE_HEIGHT) + tile_offset2;
  #else  // reference (slower)
    uint2 tile_idx2 = pix_coord / uint2(TILE_WIDTH, TILE_HEIGHT);  // groupID.xy
    uint2 tile_offset2 = pix_coord % uint2(TILE_WIDTH, TILE_HEIGHT);  // groupThreadID.xy
    uint2 subgrid_idx2 = tile_offset2 / uint2(SUBTILE_WIDTH, SUBTILE_HEIGHT);
    uint2 subtile_offset2 = tile_offset2 % uint2(SUBTILE_WIDTH, SUBTILE_HEIGHT);
    uint subgrid_idx = subgrid_idx2.y * SUBGRID_WIDTH + subgrid_idx2.x;
    uint subtile_offset = subtile_offset2.y * SUBTILE_WIDTH + subtile_offset2.x;
  #endif
    out_subgrid_idx = subgrid_idx;
    out_pixel_rank = tile_offset2.y * TILE_WIDTH + tile_offset2.x;
}

[ForceInline]
uint32_t computeTileOverlapBitmask(bool exact, float2 g_xy, float4 g_inv_cov_opac) {
    uint32_t bitmask = 0;
    float extend = extend_from_opacity(g_inv_cov_opac.w);
    float3 g_inv_cov = g_inv_cov_opac.xyz;
    g_inv_cov /= (extend*extend);
    for (int yi = 0; yi < SUBGRID_HEIGHT; yi++) {
        float y0 = float(yi * SUBTILE_HEIGHT) - g_xy.y;
        float y1 = y0 + SUBTILE_HEIGHT;
        for (int xi = 0; xi < SUBGRID_WIDTH; xi++) {
            float x0 = float(xi * SUBTILE_WIDTH) - g_xy.x;
            float x1 = x0 + SUBTILE_WIDTH;
            bool overlap = exact ?
                ellipse_box_overlap_test(g_inv_cov, x0, x1, y0, y1) :
                ellipse_box_overlap_test_fast(g_inv_cov, x0, x1, y0, y1);
            bitmask |= (uint32_t(overlap) << (yi * SUBGRID_WIDTH + xi));
        }
    }
    return bitmask;
}

#endif


// forward
#if EXPORT_MODE == EXPORT_MODE_VULKAN_FORWARD

#if SHARED_STRUCT
groupshared Splat_2D_AlphaBlend collected_splats[TILE_SIZE];
#else
groupshared float2 collected_xy_vs[TILE_SIZE];
groupshared float4 collected_inv_cov_vs_opacity[TILE_SIZE];
#endif

groupshared uint32_t collected_idx[TILE_SIZE];

#if SUBTILE_SIZE > 0
groupshared uint32_t subtile_overlap_bitmasks[TILE_SIZE];
#endif


layout(binding=0) StructuredBuffer<int32_t> sorted_gauss_idx;
layout(binding=1) StructuredBuffer<int32_t> tile_ranges;
layout(binding=2) StructuredBuffer<float2> xy_vs;
layout(binding=3) StructuredBuffer<float4> inv_cov_vs_opacity;
layout(binding=4) StructuredBuffer<float> rgb;

layout(binding=5) RWStructuredBuffer<float4> final_pixel_state;
layout(binding=6) RWStructuredBuffer<int32_t> n_contributors;

[shader("compute")]
[numthreads(TILE_WIDTH, TILE_HEIGHT, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uint3 groupThreadID : SV_GroupThreadID,
    uint3 groupID : SV_GroupID,
    uniform Uniforms uniforms
) {
    const uint32_t H = uniforms.image_height;
    const uint32_t W = uniforms.image_width;
    const uint grid_width = uniforms.grid_width;

    uint2 pix_coord = dispatchThreadID.xy;

    uint tile_idx = groupID.y * grid_width + groupID.x;
    uint32_t tile_idx_start = tile_ranges[tile_idx];
    uint32_t tile_idx_end = tile_ranges[tile_idx+1];

    uint thread_rank = groupThreadID.y * TILE_WIDTH + groupThreadID.x;

    #if SUBTILE_SIZE > 0
        uint2 pix_coord_base = groupID.xy * uint2(TILE_WIDTH, TILE_HEIGHT);
        uint subgrid_idx, pixel_rank;
        decodeThreadToSubtiles(thread_rank, pix_coord, subgrid_idx, pixel_rank);
    #endif
    float4 curr_pixel_state = float4(0, 0, 0, 1);
    bool is_inside = (pix_coord.x < W && pix_coord.y < H);
    bool thread_active = is_inside;

    float2 center_pix_coord = pix_coord;

    const int shared_memory_rounds = ((tile_idx_end - tile_idx_start + TILE_SIZE - 1) / TILE_SIZE);

    int32_t num_cull = 0;  // for visualization
    int32_t local_n_contrib = 0;
    int splats_left_to_process = tile_idx_end - tile_idx_start;
    for (int i = 0; i < shared_memory_rounds; i++)
    {
        // break if thread_active is false for all, doesn't seem to speed up
        // equivalent to CUDA `if (__syncthreads_count(thread_active) == 0) break;`
        uint subgroup_idx = thread_rank / SUBGROUP_SIZE;
        uint32_t thread_active_u = uint32_t(WaveActiveAnyTrue(thread_active));
        if (WaveIsFirstLane())
            collected_idx[subgroup_idx] = thread_active_u;
        GroupMemoryBarrierWithGroupSync();
        for (int s = 1; s < TILE_SIZE/SUBGROUP_SIZE; s <<= 1) {
            if (thread_rank < TILE_SIZE/SUBGROUP_SIZE) {
                thread_active_u += collected_idx[(thread_rank+s)%TILE_SIZE];
                collected_idx[thread_rank] = thread_active_u;
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if (SUBGROUP_SIZE*SUBGROUP_SIZE >= TILE_SIZE)
            GroupMemoryBarrierWithGroupSync();
        if (collected_idx[0] == 0)
            break;

        // Collectively fetch per-Gaussian data from global to shared
        int splat_pointer_offset = i * TILE_SIZE + thread_rank;
        if (tile_idx_start + splat_pointer_offset < tile_idx_end) {
            uint32_t coll_id = uint32_t(sorted_gauss_idx[tile_idx_start + splat_pointer_offset]);
            #if SHARED_STRUCT
            collected_splats[thread_rank] = load_splat_alphablend(coll_id, xy_vs, inv_cov_vs_opacity, rgb);
            #else
            collected_xy_vs[thread_rank] = read_t2_float2(coll_id, xy_vs);
            collected_inv_cov_vs_opacity[thread_rank] = read_t4_float4(coll_id, inv_cov_vs_opacity);
            collected_idx[thread_rank] = coll_id;
            #endif
        }
        GroupMemoryBarrierWithGroupSync();

        // check if each splat overlaps with each subtile
        #if SUBTILE_SIZE > 0
        if (thread_rank < min(TILE_SIZE, splats_left_to_process)) {
            #if SHARED_STRUCT
                Splat_2D_AlphaBlend g = collected_splats[thread_rank];
                float4 g_inv_cov_opac = g.inv_cov_vs_opacity;
                float2 g_xy = g.xy_vs;
            #else
                float4 g_inv_cov_opac = collected_inv_cov_vs_opacity[thread_rank];
                float2 g_xy = collected_xy_vs[thread_rank];
            #endif
            subtile_overlap_bitmasks[thread_rank] =
                computeTileOverlapBitmask(false, g_xy - pix_coord_base, g_inv_cov_opac);
        }
        GroupMemoryBarrierWithGroupSync();
        #endif

        if (thread_active) {
            for (int j = 0; j < min(TILE_SIZE, splats_left_to_process); j++) {
                local_n_contrib++;

                // subtile culling
            #if SUBTILE_SIZE > 0
                int overlap = (subtile_overlap_bitmasks[j] >> subgrid_idx) & 1;
                if (overlap == 0) {
                    num_cull++;
                    continue;
                }
            #endif

            #if SHARED_STRUCT
 
                Splat_2D_AlphaBlend g = collected_splats[j];
                float4 gauss_rgba = evaluate_splat(g, center_pix_coord);
                if (gauss_rgba.a < ALPHA_THRESHOLD)
                    continue;

            #else

                float2 g_xy = collected_xy_vs[j];
                float4 g_inv_cov_opacity = collected_inv_cov_vs_opacity[j];
                float3 g_inv_cov = g_inv_cov_opacity.xyz;
                float g_opacity = g_inv_cov_opacity.w;

                float alpha = evaluate_splat_alpha(g_xy, g_opacity, g_inv_cov, center_pix_coord);
                if (alpha < ALPHA_THRESHOLD)
                    continue;

                uint32_t id = collected_idx[j];
                float3 rgb = read_t3_float3(id, rgb);
                float4 gauss_rgba = float4(rgb*alpha, alpha);

            #endif

                float4 new_pixel_state = update_pixel_state(curr_pixel_state, gauss_rgba);
                if (new_pixel_state.a < 0.0001f) {
                    // This Splat never registred so we subtract it before we break.
                    local_n_contrib--;
                    thread_active = false;
                    break;
                }
                curr_pixel_state = new_pixel_state;
            }
        }
        splats_left_to_process -= TILE_SIZE;
    }
    GroupMemoryBarrierWithGroupSync();

    if (is_inside) {
        int pix_id = pix_coord.y*W+pix_coord.x;
        n_contributors[pix_id] = local_n_contrib;
        final_pixel_state[pix_id] = curr_pixel_state;
    }
}

#endif // #if EXPORT_MODE == EXPORT_MODE_VULKAN_FORWARD


// backward
#if EXPORT_MODE != EXPORT_MODE_VULKAN_FORWARD

#define BACKWARD_MODE_PER_PIXEL 0
#define BACKWARD_MODE_PER_SPLAT 1
#define BACKWARD_MODE_TENSOR 2

#ifndef BACKWARD_MODE
    // #define BACKWARD_MODE BACKWARD_MODE_TENSOR
    #define BACKWARD_MODE BACKWARD_MODE_PER_SPLAT
    // #define BACKWARD_MODE BACKWARD_MODE_PER_PIXEL
#endif // #ifndef BACKWARD_MODE

#if BACKWARD_MODE == BACKWARD_MODE_PER_PIXEL
    #include "alphablend_shader_bwd_per_pixel.slang"
#elif BACKWARD_MODE == BACKWARD_MODE_PER_SPLAT
    #include "alphablend_shader_bwd_per_splat.slang"
#elif BACKWARD_MODE == BACKWARD_MODE_TENSOR
    #include "alphablend_shader_bwd_tensor.slang"
#endif

#endif

