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
#if USE_EMULATED_INT64
layout(binding=12) StructuredBuffer<int> tile_rects;
#else
layout(binding=12) StructuredBuffer<int64_t> tile_rects;
#endif

bool overlaps_render_tile(uint slot, uint2 tile, uniform Uniforms uniforms) {
    // Keep wrapped panoramas on their existing conservative macro-list path.
    if (base_camera_model(uniforms.camera_model) == uint(CameraModel::EQUIRECTANGULAR)) return true;
#if USE_EMULATED_INT64
    rectangle rect;
    rect.min_x = (uint16_t)tile_rects[2u * slot];
    rect.min_y = (uint16_t)(tile_rects[2u * slot] >> 16);
    rect.max_x = (uint16_t)tile_rects[2u * slot + 1u];
    rect.max_y = (uint16_t)(tile_rects[2u * slot + 1u] >> 16);
#else
    rectangle rect = reinterpret<rectangle>(tile_rects[slot]);
#endif
    if (tile.y < uint(rect.min_y) || tile.y >= uint(rect.max_y)) return false;
    return tile.x >= uint(rect.min_x) && tile.x < uint(rect.max_x);
}

// Same alpha arithmetic as utils.evaluate_splat_alpha. The mask reuses a
// per-Gaussian cutoff, avoiding a logarithm per pixel without changing the
// shared color/training helpers or their default calling convention.
[ForceInline]
float outline_splat_alpha(Splat_2D_AlphaBlend splat, float2 coordinate, float maximum_power) {
    float2 d = coordinate - splat.xy_vs;
    float3 conic = splat.inv_cov_vs_opacity.xyz;
    float sigma_over_2 = 0.5f * (conic.x * d.x * d.x + conic.z * d.y * d.y) + conic.y * d.x * d.y;
    float value = sigma_over_2 < 0.0f || sigma_over_2 > maximum_power ? 0.0f : exp(-sigma_over_2);
    return float(clipOpacity(splat.inv_cov_vs_opacity.w * value));
}

groupshared uint any_active;
groupshared uint scene_open;
// One bit per batch entry, so compaction keeps the raster order.
groupshared uint batch_keep[8];

groupshared float2 batch_positions[256];
groupshared float4 batch_conics[256];
groupshared float batch_depths[256];
groupshared uint batch_nodes[256];
groupshared uint batch_selected[256];
groupshared float batch_maximum_power[256];

[numthreads(16, 16, 1)]
void main(uint3 pixel : SV_DispatchThreadID, uint3 lane : SV_GroupThreadID,
          uint3 group : SV_GroupID, uniform Uniforms uniforms)
{
    uint rank = lane.y * 16u + lane.x;
    bool inside = pixel.x < uniforms.image_width && pixel.y < uniforms.image_height;
    uint index = pixel.y * uniforms.image_width + pixel.x;
    uint tile_bound = outline_tiles[group.y * uniforms.grid_width + group.x];
    if (tile_bound == 0u)
        return;
    uint rank_offset = uniforms.grid_width * uniforms.grid_height;
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
        // Macro lists preserve the global depth-rank order. This tile cannot
        // gain selected coverage after its final selected rank, even when
        // individual pixels remain below threshold. All lanes exit together.
        if (outline_tiles[rank_offset + uint(sorted_ids[base])] > tile_bound)
            break;
        if (rank == 0u) {
            any_active = 0u;
            scene_open = 0u;
        }
        if (rank < 8u) batch_keep[rank] = 0u;
        GroupMemoryBarrierWithGroupSync();
        bool wave_active = WaveActiveAnyTrue(active);
        bool wave_scene_open = WaveActiveAnyTrue(active && scene_coverage < threshold);
        if (WaveIsFirstLane()) {
            if (wave_active) InterlockedOr(any_active, 1u);
            if (wave_scene_open) InterlockedOr(scene_open, 1u);
        }
        GroupMemoryBarrierWithGroupSync();
        if (any_active == 0u) break;
        uint slot = 0u;
        uint selected = 0u;
        bool keep = false;
        if (base + rank < end) {
            slot = uint(sorted_ids[base + rank]);
            selected = read_byte(flags, slot) & OVERLAY_FLAG_OUTLINE_SELECTED;
            // Once every pixel has settled its scene coverage, only selected
            // splats can change the mask. Macro lists cover several render
            // tiles; reuse the projection's conservative tile rectangle.
            keep = (scene_open != 0u || selected != 0u) && overlaps_render_tile(slot, group.xy, uniforms);
        }
        if (keep) InterlockedOr(batch_keep[rank / 32u], 1u << (rank % 32u));
        GroupMemoryBarrierWithGroupSync();
        uint batch_count = 0u;
        uint destination = 0u;
        for (uint word = 0u; word < 8u; ++word) {
            uint bits = batch_keep[word];
            if (word == rank / 32u)
                destination = batch_count + countbits(bits & ((1u << (rank % 32u)) - 1u));
            batch_count += countbits(bits);
        }
        if (keep) {
            batch_selected[destination] = selected;
            batch_positions[destination] = positions[slot];
            float4 conic = conics[slot];
            batch_conics[destination] = conic;
            // The cutoff depends on the splat, not the pixel. Avoid
            // repeating its logarithm in every lane of the mask batch.
            batch_maximum_power[destination] = power_threshold_for_opacity(conic.w);
            batch_depths[destination] = depths[slot];
            batch_nodes[destination] = has_indices && selected != 0u
                                           ? uint(max(read_int32(transform_indices, uint(logical_ids[slot])), 0)) : 0u;
        }
        GroupMemoryBarrierWithGroupSync();
        if (active) {
            for (uint j = 0u; j < batch_count; ++j) {
                uint candidate = batch_selected[j];
                if (scene_coverage >= threshold && candidate == 0u)
                    continue;
                uint node_id = batch_nodes[j] + 2u;
                Splat_2D_AlphaBlend splat;
                splat.xy_vs = batch_positions[j];
                splat.inv_cov_vs_opacity = batch_conics[j];
                splat.rgb = float3(0.0f);
                float alpha;
                if ((uniforms.lod_enabled & 4u) != 0u)
                    alpha = evaluate_splat_spark_rad(splat, coordinate).a;
                else if (uniforms.splat_render_profile == 1u)
                    alpha = evaluate_splat_for_profile(splat, coordinate, uniforms.splat_render_profile).a;
                else
                    alpha = outline_splat_alpha(splat, coordinate, batch_maximum_power[j]);
                if (alpha < ALPHA_THRESHOLD)
                    continue;
                if (scene_coverage < threshold) {
                    scene_coverage += (1.0f - scene_coverage) * alpha;
                    if (scene_coverage >= threshold)
                        scene_depth = batch_depths[j];
                }
                if ((candidate & OVERLAY_FLAG_OUTLINE_SELECTED) != 0u && selected_coverage < threshold) {
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
