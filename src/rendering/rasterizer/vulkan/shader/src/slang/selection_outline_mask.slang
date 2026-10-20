// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "config.slang"
import utils;

layout(binding=0) StructuredBuffer<int> sorted_ids;
layout(binding=1) StructuredBuffer<int> macro_ranges;
layout(binding=2) StructuredBuffer<float2> positions;
layout(binding=3) StructuredBuffer<float4> conics;
layout(binding=4) StructuredBuffer<float> depths;
layout(binding=5) StructuredBuffer<int> logical_ids;
layout(binding=6) ByteAddressBuffer flags;
layout(binding=7) ByteAddressBuffer transform_indices;
layout(binding=8) StructuredBuffer<float4> overlay_params;
layout(binding=9) RWStructuredBuffer<uint> outline_ids;
layout(binding=10) RWStructuredBuffer<float4> scene_depths;
layout(binding=11) StructuredBuffer<uint> outline_tiles;
groupshared uint any_active;

groupshared float2 batch_positions[256];
groupshared float4 batch_conics[256];
groupshared float batch_depths[256];
groupshared uint batch_nodes[256];
groupshared uint batch_selected[256];

[numthreads(16, 16, 1)]
void main(uint3 pixel : SV_DispatchThreadID, uint3 lane : SV_GroupThreadID,
          uint3 group : SV_GroupID, uniform Uniforms uniforms)
{
    uint rank = lane.y * 16u + lane.x;
    bool inside = pixel.x < uniforms.image_width && pixel.y < uniforms.image_height;
    uint index = pixel.y * uniforms.image_width + pixel.x;
    if (outline_tiles[group.y * uniforms.grid_width + group.x] == 0u)
        return;
    uint id = inside ? outline_ids[index] & 0x3fffffffu : 0u;
    float4 previous_depths = inside && uniforms.depth_wave > 0u
                                 ? scene_depths[index] : float4(1.0e10f, 1.0e10f, 0.0f, 0.0f);
    float scene_depth = previous_depths.x;
    float selected_depth = previous_depths.y;
    float scene_coverage = previous_depths.z;
    float selected_coverage = previous_depths.w;
    uint macro = (group.y / HIGS_MACRO_T16_H) * higs_macro_grid_width(uniforms.grid_width)
                 + group.x / HIGS_MACRO_T16_W;
    uint begin = uint(macro_ranges[macro]);
    uint end = uint(macro_ranges[macro + 1u]);
    bool has_indices = overlay_enabled(overlay_params[OVERLAY_PARAM_EMPHASIS_FLAGS].y);
    float2 coordinate = float2(pixel.xy) + (full_frame_binning(uniforms)
                              ? float2(uniforms.render_origin_x, uniforms.render_origin_y) : float2(0.0f));
    const float threshold = overlay_params[OVERLAY_PARAM_MARKER_FLAGS].w;
    bool active = inside && (scene_coverage < threshold || selected_coverage < threshold);
    for (uint base = begin; base < end; base += 256u) {
        if (rank == 0u) any_active = 0u;
        GroupMemoryBarrierWithGroupSync();
        bool wave_active = WaveActiveAnyTrue(active);
        if (WaveIsFirstLane() && wave_active)
            InterlockedOr(any_active, 1u);
        GroupMemoryBarrierWithGroupSync();
        if (any_active == 0u) break;
        if (base + rank < end) {
            uint slot = uint(sorted_ids[base + rank]);
            batch_positions[rank] = positions[slot];
            batch_conics[rank] = conics[slot];
            batch_depths[rank] = depths[slot];
            batch_selected[rank] = read_byte(flags, slot) & OVERLAY_FLAG_OUTLINE_SELECTED;
            batch_nodes[rank] = has_indices && batch_selected[rank] != 0u
                                    ? uint(max(read_int32(transform_indices, uint(logical_ids[slot])), 0)) : 0u;
        }
        GroupMemoryBarrierWithGroupSync();
        if (active) {
            for (uint j = 0u; j < min(256u, end - base); ++j) {
                uint node_id = batch_nodes[j] + 2u;
                Splat_2D_AlphaBlend splat;
                splat.xy_vs = batch_positions[j];
                splat.inv_cov_vs_opacity = batch_conics[j];
                splat.rgb = float3(0.0f);
                float alpha = (uniforms.lod_enabled & 4u) != 0u
                                  ? evaluate_splat_spark_rad(splat, coordinate).a
                                  : evaluate_splat_for_profile(splat, coordinate, uniforms.splat_render_profile).a;
                if (alpha < ALPHA_THRESHOLD)
                    continue;
                if (scene_coverage < threshold) {
                    scene_coverage += (1.0f - scene_coverage) * alpha;
                    if (scene_coverage >= threshold)
                        scene_depth = batch_depths[j];
                }
                if (batch_selected[j] != 0u && selected_coverage < threshold) {
                    selected_coverage += (1.0f - selected_coverage) * alpha;
                    if (selected_coverage >= threshold) {
                        id = node_id;
                        selected_depth = batch_depths[j];
                    }
                }
                active = scene_coverage < threshold || selected_coverage < threshold;
                if (!active) break;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (inside) {
        float epsilon = 1.0e-4f * max(1.0f, abs(scene_depth));
        if (id > 1u && selected_depth > scene_depth + epsilon)
            id |= 0x80000000u;
        if (selected_coverage < threshold)
            id |= 0x40000000u;
        outline_ids[index] = id;
        scene_depths[index] = float4(scene_depth, selected_depth, scene_coverage, selected_coverage);
    }
}
