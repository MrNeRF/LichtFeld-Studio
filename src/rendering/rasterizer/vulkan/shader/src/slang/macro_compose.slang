// HiGS compose: per pixel, walk the parent macro tile's batches inside the
// current wave window and composite the half4 partials. Front-to-back
// blending is linear in incoming transmittance, so compositing per-batch
// partials is exact for color (selection markers included — the raster pass
// encoded their absorb-all write into the partial). Only the median depth
// needs a per-pixel replay, and only for the single batch where
// transmittance crosses 0.5:
//   mip_filter bit 1 set  -> exact replay (mesh/point-cloud z-compositing or
//                            depth view consume the value every frame)
//   bit 1 clear           -> leading depth of the crossing batch; off by at
//                            most the batch's depth span, only consumed by
//                            sporadic focus picks.

#include "config.slang"

import utils;

#ifndef LFS_RASTER_OVERLAYS
#define LFS_RASTER_OVERLAYS 0
#endif

static const float FAR_DEPTH = 1.0e10f;

layout(binding=0) StructuredBuffer<int32_t> macro_sorted_ids;
layout(binding=1) StructuredBuffer<int32_t> macro_ranges;
layout(binding=2) StructuredBuffer<int32_t> macro_batch_offsets;
layout(binding=3) StructuredBuffer<float2> xy_vs;
layout(binding=4) StructuredBuffer<float4> inv_cov_vs_opacity;
layout(binding=5) StructuredBuffer<float> rgb;
layout(binding=6) StructuredBuffer<float> depths;
layout(binding=7) StructuredBuffer<half4> macro_batch_pixel_state;
layout(binding=8) StructuredBuffer<uint32_t> macro_batch_active_mask;
layout(binding=9) RWStructuredBuffer<float4> final_pixel_state;
layout(binding=10) RWStructuredBuffer<float> final_pixel_depth;
layout(binding=11) RWStructuredBuffer<int32_t> final_n_contributors;
#if LFS_RASTER_OVERLAYS
layout(binding=12) ByteAddressBuffer selection_mask;
layout(binding=13) ByteAddressBuffer preview_mask;
layout(binding=14) StructuredBuffer<float4> selection_colors;
layout(binding=15) StructuredBuffer<uint> gaussian_overlay_flags;
layout(binding=16) StructuredBuffer<float4> overlay_params;
layout(binding=17) StructuredBuffer<int32_t> orig_ids;
#endif

[ForceInline]
bool splat_overlaps_render_tile(Splat_2D_AlphaBlend g, uint2 tile_origin) {
    const float2 tile_xy = g.xy_vs - float2(float(tile_origin.x), float(tile_origin.y));
    const float extend = max(extend_from_opacity(g.inv_cov_vs_opacity.w), 1.0e-6f);
    const float3 inv_cov = g.inv_cov_vs_opacity.xyz / (extend * extend);
    return ellipse_box_overlap_test_fast(inv_cov,
                                         -tile_xy.x,
                                         float(HIGS_TILE_WIDTH) - tile_xy.x,
                                         -tile_xy.y,
                                         float(HIGS_TILE_HEIGHT) - tile_xy.y);
}

[ForceInline]
void replay_batch(uint range_start,
                  uint range_end,
                  uint2 tile_origin,
                  float2 center_pix_coord,
                  bool spark_rad_opacity_path,
#if LFS_RASTER_OVERLAYS
                  bool markers_enabled,
#endif
                  inout float4 curr_pixel_state,
                  inout float median_depth,
                  inout bool thread_active)
{
    for (uint idx = range_start; idx < range_end && thread_active; ++idx) {
        const uint32_t slot = uint32_t(macro_sorted_ids[idx]);
        const Splat_2D_AlphaBlend g =
            load_splat_alphablend(int32_t(slot), xy_vs, inv_cov_vs_opacity, rgb);
        if (!splat_overlaps_render_tile(g, tile_origin))
            continue;

        float4 gauss_rgba = spark_rad_opacity_path
                                ? evaluate_splat_spark_rad(g, center_pix_coord)
                                : evaluate_splat(g, center_pix_coord);
        if (gauss_rgba.a < ALPHA_THRESHOLD)
            continue;
        const float splat_depth = read_t1_float(slot, depths);

      #if LFS_RASTER_OVERLAYS
        const uint flags = gaussian_overlay_flags[slot];
        const uint sel_splat_id = uint(orig_ids[slot]);
        const uint sel_status = selection_status(g.xy_vs,
                                                 sel_splat_id,
                                                 flags,
                                                 selection_mask,
                                                 preview_mask,
                                                 overlay_params);
        const bool selectable = (flags & OVERLAY_FLAG_NONSELECTABLE) == 0u;
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

        if (markers_enabled && selectable) {
            const float INNER_RADIUS_SQ = 2.25f;
            const float OUTER_RADIUS_SQ = 6.25f;
            const float OUTLINE_DARKEN = 0.4f;
            const float2 marker_delta = g.xy_vs - center_pix_coord;
            const float dist_sq = dot(marker_delta, marker_delta);
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

        const float3 color = curr_pixel_state.rgb + gauss_rgba.rgb * curr_pixel_state.a;
        const float transmittance = curr_pixel_state.a * (1.0f - gauss_rgba.a);
        if (median_depth >= FAR_DEPTH &&
            curr_pixel_state.a > 0.5f &&
            transmittance <= 0.5f) {
            median_depth = splat_depth;
        }
        curr_pixel_state = float4(color, transmittance);
        if (transmittance < 0.0001f) {
            thread_active = false;
            break;
        }
    }
}

[shader("compute")]
[numthreads(HIGS_TILE_WIDTH, HIGS_TILE_HEIGHT, 1)]
void main(uint3 groupThreadID : SV_GroupThreadID,
          uint3 groupID : SV_GroupID,
          uniform Uniforms uniforms)
{
    const uint2 pix_coord = groupID.xy * uint2(HIGS_TILE_WIDTH, HIGS_TILE_HEIGHT) + groupThreadID.xy;
    const bool is_inside = pix_coord.x < uniforms.image_width && pix_coord.y < uniforms.image_height;
    if (!is_inside)
        return;

    const uint macro_tile_idx =
        (groupID.y / HIGS_MACRO_TILE_HEIGHT_TILES) * higs_macro_grid_width(uniforms.grid_width) +
        (groupID.x / HIGS_MACRO_TILE_WIDTH_TILES);
    const uint local_tile_idx =
        (groupID.y % HIGS_MACRO_TILE_HEIGHT_TILES) * HIGS_MACRO_TILE_WIDTH_TILES +
        (groupID.x % HIGS_MACRO_TILE_WIDTH_TILES);
    const uint thread_rank = groupThreadID.y * HIGS_TILE_WIDTH + groupThreadID.x;
    const uint2 tile_origin = groupID.xy * uint2(HIGS_TILE_WIDTH, HIGS_TILE_HEIGHT);

    const uint batch_begin = uint(max(macro_tile_idx == 0u ? 0 : macro_batch_offsets[macro_tile_idx - 1u], 0));
    const uint batch_end = uint(max(macro_batch_offsets[macro_tile_idx], int32_t(batch_begin)));

    const uint pix_id = pix_coord.y * uniforms.image_width + pix_coord.x;
    const float2 center_pix_coord = float2(pix_coord);
    const bool spark_rad_opacity_path = (uniforms.lod_enabled & 4u) != 0u;
  #if LFS_RASTER_OVERLAYS
    const bool markers_enabled = overlay_enabled(overlay_params[OVERLAY_PARAM_MARKER_FLAGS].y);
  #endif

    // The per-pixel blend state persists in the output buffers between waves;
    // wave 0 initializes it.
    float4 curr_pixel_state;
    float median_depth;
    if (uniforms.depth_wave == 0u && uniforms.wave_base == 0u) {
        curr_pixel_state = float4(0.0f, 0.0f, 0.0f, 1.0f);
        median_depth = FAR_DEPTH;
    } else {
        curr_pixel_state = final_pixel_state[pix_id];
        median_depth = final_pixel_depth[pix_id];
    }
    bool thread_active = curr_pixel_state.a >= 0.0001f;
    const uint32_t local_tile_mask = 1u << local_tile_idx;

    const uint wave_walk_begin = max(batch_begin, uniforms.wave_base);
    const uint wave_walk_end = min(batch_end, uniforms.wave_base + HIGS_RASTER_WAVE_BATCHES);

    for (uint batch_idx = wave_walk_begin; batch_idx < wave_walk_end && thread_active; ++batch_idx) {
        if ((macro_batch_active_mask[batch_idx] & local_tile_mask) == 0u)
            continue;

        const uint partial_idx =
            ((batch_idx - uniforms.wave_base) * HIGS_MACRO_TILE_SIZE_TILES + local_tile_idx) * HIGS_TILE_SIZE +
            thread_rank;
        const float4 partial_state = float4(macro_batch_pixel_state[partial_idx]);

        const bool crosses_median =
            median_depth >= FAR_DEPTH &&
            curr_pixel_state.a > 0.5f &&
            curr_pixel_state.a * partial_state.a <= 0.5f;

        if (crosses_median) {
            const uint batch_in_macro = batch_idx - batch_begin;
            const uint range_start =
                uint(max(macro_ranges[macro_tile_idx], 0)) + batch_in_macro * RASTER_BATCH_SIZE;
            uint range_end = uint(max(macro_ranges[macro_tile_idx + 1u], int32_t(range_start)));
            range_end = min(range_start + RASTER_BATCH_SIZE, range_end);

            if ((uniforms.mip_filter & 2u) != 0u) {
                float4 replay_state = curr_pixel_state;
                float replay_median_depth = median_depth;
                bool replay_active = true;
                replay_batch(range_start,
                             range_end,
                             tile_origin,
                             center_pix_coord,
                             spark_rad_opacity_path,
                          #if LFS_RASTER_OVERLAYS
                             markers_enabled,
                          #endif
                             replay_state,
                             replay_median_depth,
                             replay_active);
                median_depth = replay_median_depth;
            } else {
                median_depth = read_t1_float(uint(macro_sorted_ids[range_start]), depths);
            }
        }

        curr_pixel_state =
            float4(curr_pixel_state.rgb + partial_state.rgb * curr_pixel_state.a,
                   curr_pixel_state.a * partial_state.a);
        if (curr_pixel_state.a < 0.0001f)
            thread_active = false;
    }

    final_n_contributors[pix_id] = 0;
    final_pixel_state[pix_id] = curr_pixel_state;
    final_pixel_depth[pix_id] = median_depth;
}
