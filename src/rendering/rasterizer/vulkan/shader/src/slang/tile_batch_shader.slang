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

#ifndef TILE_BATCH_ENTRY
#define TILE_BATCH_ENTRY 0
#endif

#ifndef LFS_RASTER_OVERLAYS
#define LFS_RASTER_OVERLAYS 0
#endif

#define TILE_BATCH_ENTRY_COUNTS 1
#define TILE_BATCH_ENTRY_DESCRIPTORS 2
#define TILE_BATCH_ENTRY_RASTERIZE 3
#define TILE_BATCH_ENTRY_COMPOSE 4

static const float FAR_DEPTH = 1.0e10f;

[ForceInline]
float4 update_pixel_state(float4 pixel_state_t_nm1, float4 gauss_rgba_t_n)
{
    float3 color_t_n = pixel_state_t_nm1.rgb + gauss_rgba_t_n.rgb * pixel_state_t_nm1.a;
    float transmittance_t_n = pixel_state_t_nm1.a * (float(1) - gauss_rgba_t_n.a);
    return float4(color_t_n, transmittance_t_n);
}

#if TILE_BATCH_ENTRY == TILE_BATCH_ENTRY_COUNTS

layout(binding=0) StructuredBuffer<int32_t> tile_ranges;
layout(binding=1) RWStructuredBuffer<int32_t> out_tile_batch_counts;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID,
          uniform Uniforms uniforms)
{
    uint tile_idx = dispatchThreadID.x;
    uint num_tiles = uniforms.grid_height * uniforms.grid_width;
    if (tile_idx >= num_tiles)
        return;

    int32_t range_start = tile_ranges[tile_idx];
    int32_t range_end = tile_ranges[tile_idx + 1u];
    uint tile_count = uint(max(range_end - range_start, 0));
    out_tile_batch_counts[tile_idx] =
        tile_count > RASTER_DENSE_TILE_THRESHOLD
            ? int32_t((tile_count + RASTER_BATCH_SIZE - 1u) / RASTER_BATCH_SIZE)
            : 0;
}

#elif TILE_BATCH_ENTRY == TILE_BATCH_ENTRY_DESCRIPTORS

layout(binding=0) StructuredBuffer<int32_t> tile_ranges;
layout(binding=1) StructuredBuffer<int32_t> tile_batch_offsets;
layout(binding=2) RWStructuredBuffer<uint4> out_tile_batch_descriptors;
layout(binding=3) RWStructuredBuffer<uint32_t> out_dispatch_args;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID,
          uniform Uniforms uniforms)
{
    uint tile_idx = dispatchThreadID.x;
    uint num_tiles = uniforms.grid_height * uniforms.grid_width;

    if (tile_idx == 0u) {
        uint total_batches = num_tiles > 0u
            ? uint(max(tile_batch_offsets[num_tiles - 1u], 0))
            : 0u;
        const uint raster = LFS_VK_TILE_BATCH_DISPATCH_RASTER_WORD_OFFSET;
        out_dispatch_args[raster + LFS_VK_DISPATCH_X_WORD_OFFSET] = total_batches;
        out_dispatch_args[raster + LFS_VK_DISPATCH_Y_WORD_OFFSET] = 1u;
        out_dispatch_args[raster + LFS_VK_DISPATCH_Z_WORD_OFFSET] = 1u;
    }

    if (tile_idx >= num_tiles)
        return;

    uint desc_begin = tile_idx == 0u
        ? 0u
        : uint(max(tile_batch_offsets[tile_idx - 1u], 0));
    uint desc_end = uint(max(tile_batch_offsets[tile_idx], 0));
    if (desc_end <= desc_begin)
        return;

    uint range_start = uint(max(tile_ranges[tile_idx], 0));
    uint range_end = uint(max(tile_ranges[tile_idx + 1u], int32_t(range_start)));
    uint batch_count = desc_end - desc_begin;

    for (uint batch = 0u; batch < batch_count; ++batch) {
        uint batch_start = range_start + batch * RASTER_BATCH_SIZE;
        uint batch_end = min(batch_start + RASTER_BATCH_SIZE, range_end);
        out_tile_batch_descriptors[desc_begin + batch] =
            uint4(tile_idx, batch_start, batch_end, 0u);
    }
}

#elif TILE_BATCH_ENTRY == TILE_BATCH_ENTRY_RASTERIZE

groupshared Splat_2D_AlphaBlend collected_splats[TILE_SIZE];
groupshared uint32_t active_subgroups[TILE_SIZE / SUBGROUP_SIZE];
#if LFS_RASTER_OVERLAYS
groupshared uint32_t collected_gauss_idx[TILE_SIZE];
#endif

layout(binding=0) StructuredBuffer<int32_t> sorted_gauss_idx;
layout(binding=1) StructuredBuffer<uint4> tile_batch_descriptors;
layout(binding=2) StructuredBuffer<float2> xy_vs;
layout(binding=3) StructuredBuffer<float4> inv_cov_vs_opacity;
layout(binding=4) StructuredBuffer<float> rgb;
layout(binding=5) RWStructuredBuffer<float4> tile_batch_pixel_state;
layout(binding=6) RWStructuredBuffer<int32_t> tile_batch_n_contributors;
#if LFS_RASTER_OVERLAYS
layout(binding=7) ByteAddressBuffer selection_mask;
layout(binding=8) ByteAddressBuffer preview_mask;
layout(binding=9) StructuredBuffer<float4> selection_colors;
layout(binding=10) StructuredBuffer<uint> gaussian_overlay_flags;
layout(binding=11) StructuredBuffer<float4> overlay_params;
#endif

[shader("compute")]
[numthreads(TILE_WIDTH, TILE_HEIGHT, 1)]
void main(uint3 groupThreadID : SV_GroupThreadID,
          uint3 groupID : SV_GroupID,
          uniform Uniforms uniforms)
{
    uint batch_idx = groupID.x;
    uint4 descriptor = tile_batch_descriptors[batch_idx];
    uint tile_idx = descriptor.x;
    uint tile_x = tile_idx % uniforms.grid_width;
    uint tile_y = tile_idx / uniforms.grid_width;
    uint thread_rank = groupThreadID.y * TILE_WIDTH + groupThreadID.x;
    uint2 pix_coord = uint2(tile_x * TILE_WIDTH + groupThreadID.x,
                            tile_y * TILE_HEIGHT + groupThreadID.y);
    bool is_inside = pix_coord.x < uniforms.image_width &&
                     pix_coord.y < uniforms.image_height;
    bool thread_active = is_inside;
    float2 center_pix_coord = float2(pix_coord);
    float4 curr_pixel_state = float4(0.0f, 0.0f, 0.0f, 1.0f);
    int32_t local_n_contrib = 0;

    uint range_start = descriptor.y;
    uint range_end = descriptor.z;
    uint splats_to_process = range_end - range_start;
    uint shared_memory_rounds =
        (splats_to_process + TILE_SIZE - 1u) / TILE_SIZE;

    for (uint round = 0u; round < shared_memory_rounds; ++round) {
        const uint subgroup_idx = thread_rank / SUBGROUP_SIZE;
        const uint32_t subgroup_has_active_thread =
            uint32_t(WaveActiveAnyTrue(thread_active));
        if (WaveIsFirstLane())
            active_subgroups[subgroup_idx] = subgroup_has_active_thread;
        GroupMemoryBarrierWithGroupSync();
        if (thread_rank == 0u) {
            uint32_t any_active = 0u;
            for (uint s = 0u; s < TILE_SIZE / SUBGROUP_SIZE; ++s)
                any_active |= active_subgroups[s];
            active_subgroups[0] = any_active;
        }
        GroupMemoryBarrierWithGroupSync();
        if (active_subgroups[0] == 0u)
            break;

        uint splat_pointer_offset = round * TILE_SIZE + thread_rank;
        if (splat_pointer_offset < splats_to_process) {
            uint32_t splat_id =
                uint32_t(sorted_gauss_idx[range_start + splat_pointer_offset]);
            collected_splats[thread_rank] =
                load_splat_alphablend(int32_t(splat_id), xy_vs, inv_cov_vs_opacity, rgb);
            #if LFS_RASTER_OVERLAYS
            collected_gauss_idx[thread_rank] = splat_id;
            #endif
        }
        GroupMemoryBarrierWithGroupSync();

        if (thread_active) {
            uint round_begin = round * TILE_SIZE;
            uint splats_in_round = min(uint(TILE_SIZE), splats_to_process - round_begin);
            for (uint j = 0u; j < splats_in_round; ++j) {
                ++local_n_contrib;
                Splat_2D_AlphaBlend g = collected_splats[j];
                float4 gauss_rgba = evaluate_splat(g, center_pix_coord);
                if (gauss_rgba.a < ALPHA_THRESHOLD)
                    continue;

                #if LFS_RASTER_OVERLAYS
                uint32_t splat_id = collected_gauss_idx[j];
                uint flags = gaussian_overlay_flags[splat_id];
                uint sel_status = selection_status(g.xy_vs,
                                                   splat_id,
                                                   flags,
                                                   selection_mask,
                                                   preview_mask,
                                                   overlay_params);
                bool selectable = (flags & OVERLAY_FLAG_NONSELECTABLE) == 0u;
                gauss_rgba = apply_projection_overlay_color(gauss_rgba,
                                                            g.xy_vs,
                                                            flags,
                                                            overlay_params);
                gauss_rgba = apply_ring_overlay(gauss_rgba,
                                                g,
                                                center_pix_coord,
                                                sel_status,
                                                flags,
                                                overlay_params,
                                                selection_colors);

                if (overlay_enabled(overlay_params[OVERLAY_PARAM_MARKER_FLAGS].y) && selectable) {
                    const float INNER_RADIUS_SQ = 2.25f;
                    const float OUTER_RADIUS_SQ = 6.25f;
                    const float OUTLINE_DARKEN = 0.4f;
                    float2 marker_delta = g.xy_vs - center_pix_coord;
                    float dist_sq = dot(marker_delta, marker_delta);
                    if (dist_sq > OUTER_RADIUS_SQ)
                        continue;

                    float3 marker_color = selection_overlay_target_color(sel_status, selection_colors);
                    if (dist_sq > INNER_RADIUS_SQ)
                        marker_color *= OUTLINE_DARKEN;

                    curr_pixel_state = float4(marker_color, 0.0f);
                    thread_active = false;
                    break;
                }

                gauss_rgba = apply_selection_overlay(gauss_rgba, sel_status, selection_colors);
                gauss_rgba = apply_flash_overlay(gauss_rgba, flags, overlay_params);
                #endif

                float4 new_pixel_state = update_pixel_state(curr_pixel_state, gauss_rgba);
                if (new_pixel_state.a < 0.0001f) {
                    --local_n_contrib;
                    thread_active = false;
                    break;
                }
                curr_pixel_state = new_pixel_state;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    uint partial_idx = batch_idx * TILE_SIZE + thread_rank;
    tile_batch_pixel_state[partial_idx] =
        is_inside ? curr_pixel_state : float4(0.0f, 0.0f, 0.0f, 1.0f);
    tile_batch_n_contributors[partial_idx] =
        is_inside ? local_n_contrib : 0;
}

#elif TILE_BATCH_ENTRY == TILE_BATCH_ENTRY_COMPOSE

layout(binding=0) StructuredBuffer<int32_t> sorted_gauss_idx;
layout(binding=1) StructuredBuffer<uint4> tile_batch_descriptors;
layout(binding=2) StructuredBuffer<int32_t> tile_batch_offsets;
layout(binding=3) StructuredBuffer<float2> xy_vs;
layout(binding=4) StructuredBuffer<float4> inv_cov_vs_opacity;
layout(binding=5) StructuredBuffer<float> rgb;
layout(binding=6) StructuredBuffer<float> depths;
layout(binding=7) StructuredBuffer<float4> tile_batch_pixel_state;
layout(binding=8) StructuredBuffer<int32_t> tile_batch_n_contributors;
layout(binding=9) RWStructuredBuffer<float4> final_pixel_state;
layout(binding=10) RWStructuredBuffer<float> final_pixel_depth;
layout(binding=11) RWStructuredBuffer<int32_t> final_n_contributors;
#if LFS_RASTER_OVERLAYS
layout(binding=12) ByteAddressBuffer selection_mask;
layout(binding=13) ByteAddressBuffer preview_mask;
layout(binding=14) StructuredBuffer<float4> selection_colors;
layout(binding=15) StructuredBuffer<uint> gaussian_overlay_flags;
layout(binding=16) StructuredBuffer<float4> overlay_params;
#endif

[ForceInline]
void replay_batch(uint range_start,
                  uint range_end,
                  float2 center_pix_coord,
                  inout float4 curr_pixel_state,
                  inout float median_depth,
                  inout int32_t local_n_contrib,
                  inout bool thread_active)
{
    for (uint idx = range_start; idx < range_end && thread_active; ++idx) {
        uint32_t splat_id = uint32_t(sorted_gauss_idx[idx]);
        Splat_2D_AlphaBlend g =
            load_splat_alphablend(int32_t(splat_id), xy_vs, inv_cov_vs_opacity, rgb);
        ++local_n_contrib;

        float4 gauss_rgba = evaluate_splat(g, center_pix_coord);
        if (gauss_rgba.a < ALPHA_THRESHOLD)
            continue;
        float splat_depth = read_t1_float(splat_id, depths);

        #if LFS_RASTER_OVERLAYS
        uint flags = gaussian_overlay_flags[splat_id];
        uint sel_status = selection_status(g.xy_vs,
                                           splat_id,
                                           flags,
                                           selection_mask,
                                           preview_mask,
                                           overlay_params);
        bool selectable = (flags & OVERLAY_FLAG_NONSELECTABLE) == 0u;
        gauss_rgba = apply_projection_overlay_color(gauss_rgba,
                                                    g.xy_vs,
                                                    flags,
                                                    overlay_params);
        gauss_rgba = apply_ring_overlay(gauss_rgba,
                                        g,
                                        center_pix_coord,
                                        sel_status,
                                        flags,
                                        overlay_params,
                                        selection_colors);

        if (overlay_enabled(overlay_params[OVERLAY_PARAM_MARKER_FLAGS].y) && selectable) {
            const float INNER_RADIUS_SQ = 2.25f;
            const float OUTER_RADIUS_SQ = 6.25f;
            const float OUTLINE_DARKEN = 0.4f;
            float2 marker_delta = g.xy_vs - center_pix_coord;
            float dist_sq = dot(marker_delta, marker_delta);
            if (dist_sq > OUTER_RADIUS_SQ)
                continue;

            float3 marker_color = selection_overlay_target_color(sel_status, selection_colors);
            if (dist_sq > INNER_RADIUS_SQ)
                marker_color *= OUTLINE_DARKEN;

            if (curr_pixel_state.a > 0.5f) {
                median_depth = splat_depth;
            }
            curr_pixel_state = float4(marker_color, 0.0f);
            thread_active = false;
            break;
        }

        gauss_rgba = apply_selection_overlay(gauss_rgba, sel_status, selection_colors);
        gauss_rgba = apply_flash_overlay(gauss_rgba, flags, overlay_params);
        #endif

        float4 new_pixel_state = update_pixel_state(curr_pixel_state, gauss_rgba);
        if (median_depth >= FAR_DEPTH &&
            curr_pixel_state.a > 0.5f &&
            new_pixel_state.a <= 0.5f) {
            median_depth = splat_depth;
        }
        if (new_pixel_state.a < 0.0001f) {
            --local_n_contrib;
            thread_active = false;
            break;
        }
        curr_pixel_state = new_pixel_state;
    }
}

[shader("compute")]
[numthreads(TILE_WIDTH, TILE_HEIGHT, 1)]
void main(uint3 groupThreadID : SV_GroupThreadID,
          uint3 groupID : SV_GroupID,
          uniform Uniforms uniforms)
{
    uint tile_idx = groupID.y * uniforms.grid_width + groupID.x;
    uint thread_rank = groupThreadID.y * TILE_WIDTH + groupThreadID.x;
    uint2 pix_coord = uint2(groupID.x * TILE_WIDTH + groupThreadID.x,
                            groupID.y * TILE_HEIGHT + groupThreadID.y);
    bool is_inside = pix_coord.x < uniforms.image_width &&
                     pix_coord.y < uniforms.image_height;
    if (!is_inside)
        return;

    uint desc_begin = tile_idx == 0u
        ? 0u
        : uint(max(tile_batch_offsets[tile_idx - 1u], 0));
    uint desc_end = uint(max(tile_batch_offsets[tile_idx], 0));
    if (desc_end <= desc_begin)
        return;

    float2 center_pix_coord = float2(pix_coord);
    float4 curr_pixel_state = float4(0.0f, 0.0f, 0.0f, 1.0f);
    float median_depth = FAR_DEPTH;
    int32_t local_n_contrib = 0;
    bool thread_active = true;

    for (uint desc_idx = desc_begin; desc_idx < desc_end && thread_active; ++desc_idx) {
        uint partial_idx = desc_idx * TILE_SIZE + thread_rank;
        float4 partial_state = tile_batch_pixel_state[partial_idx];
        int32_t partial_n_contrib = tile_batch_n_contributors[partial_idx];

        bool crosses_median =
            median_depth >= FAR_DEPTH &&
            curr_pixel_state.a > 0.5f &&
            curr_pixel_state.a * partial_state.a <= 0.5f;
        bool crosses_opacity =
            curr_pixel_state.a * partial_state.a < 0.0001f;

        uint4 descriptor = tile_batch_descriptors[desc_idx];
        if (crosses_opacity) {
            replay_batch(descriptor.y,
                         descriptor.z,
                         center_pix_coord,
                         curr_pixel_state,
                         median_depth,
                         local_n_contrib,
                         thread_active);
            break;
        }

        if (crosses_median) {
            float4 replay_state = curr_pixel_state;
            float replay_median_depth = median_depth;
            int32_t replay_n_contrib = 0;
            bool replay_active = true;
            replay_batch(descriptor.y,
                         descriptor.z,
                         center_pix_coord,
                         replay_state,
                         replay_median_depth,
                         replay_n_contrib,
                         replay_active);
            median_depth = replay_median_depth;
        }

        curr_pixel_state =
            float4(curr_pixel_state.rgb + partial_state.rgb * curr_pixel_state.a,
                   curr_pixel_state.a * partial_state.a);
        local_n_contrib += partial_n_contrib;
    }

    uint pix_id = pix_coord.y * uniforms.image_width + pix_coord.x;
    final_n_contributors[pix_id] = local_n_contrib;
    final_pixel_state[pix_id] = curr_pixel_state;
    final_pixel_depth[pix_id] = median_depth;
}

#endif
