// HiGS composes FP16 color partials in FP32. Exact median depth keeps
// independent FP32 transmittance across batches and waves: a rounded color
// partial cannot decide which batch crosses 0.5. Replay stops at the first
// exact crossing. The ordinary focus-pick path keeps its batch approximation.

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
// HiGS does not produce contributor counts (unlike the legacy training
// rasterizer). Reuse that 32-bit-per-pixel scratch for exact depth state.
// The binding and host allocation stay unchanged; no new GPU buffer is needed.
layout(binding = 11) RWStructuredBuffer<float> exact_depth_transmittance;
#if LFS_RASTER_OVERLAYS
layout(binding=12) ByteAddressBuffer selection_mask;
layout(binding=13) ByteAddressBuffer preview_mask;
layout(binding=14) StructuredBuffer<float4> selection_colors;
layout(binding=15) ByteAddressBuffer gaussian_overlay_flags;
layout(binding=16) StructuredBuffer<float4> overlay_params;
layout(binding=17) StructuredBuffer<int32_t> orig_ids;
layout(binding=18) StructuredBuffer<uint32_t> exact_depth_sample_mask;
#else
layout(binding=12) StructuredBuffer<uint32_t> exact_depth_sample_mask;
#endif

[ForceInline] bool splat_overlaps_render_tile(Splat_2D_AlphaBlend g, uint2 tile_origin) {
    const float2 tile_xy = g.xy_vs - float2(float(tile_origin.x), float(tile_origin.y));
    const float extend = max(extend_from_opacity(g.inv_cov_vs_opacity.w), 1.0e-6f);
    const float3 inv_cov = g.inv_cov_vs_opacity.xyz / (extend * extend);
    return ellipse_box_overlap_test_fast(inv_cov,
                                         -tile_xy.x,
                                         float(HIGS_TILE_WIDTH) - tile_xy.x,
                                         -tile_xy.y,
                                         float(HIGS_TILE_HEIGHT) - tile_xy.y);
}

// Conservative minimum of a positive quadratic over live pixel bounds.
// A roundoff margin keeps support-boundary contributors in the replay.
float depth_lower_power(float2 d, float3 c) {
    const float3 terms = float3(c.x * d.x * d.x, 2.0f * c.y * d.x * d.y, c.z * d.y * d.y);
    return 0.5f * (terms.x + terms.y + terms.z) - 1.0e-4f * (1.0f + dot(abs(terms), float3(1.0f)));
}
bool depth_intersects_pixels(Splat_2D_AlphaBlend g, uint2 origin, uint2 maximum, float support) {
    const float3 c = g.inv_cov_vs_opacity.xyz;
    if (c.x <= 0.0f || c.z <= 0.0f || c.x * c.z <= c.y * c.y)
        return true;
    const float2 lo = float2(origin) - g.xy_vs, hi = float2(maximum) - g.xy_vs;
    if (all(lo <= 0.0f) && all(hi >= 0.0f))
        return true;
    const float2 y = clamp(-c.y * float2(lo.x, hi.x) / c.z, lo.y, hi.y);
    const float2 x = clamp(-c.y * float2(lo.y, hi.y) / c.x, lo.x, hi.x);
    const float lower = min(min(depth_lower_power(float2(lo.x, y.x), c), depth_lower_power(float2(hi.x, y.y), c)),
                            min(depth_lower_power(float2(x.x, lo.y), c), depth_lower_power(float2(x.y, hi.y), c)));
    return !isfinite(lower) || lower <= support + 1.0e-4f;
}

// Each SIMD wave shares source values through lane broadcasts. Exact depth
// retires at its median, independently of the FP16 color saturation point.
[ForceInline] void replay_batch(uint range_start,
                                uint range_end,
                                uint2 tile_origin,
                                float2 center_pix_coord,
                                uint splat_render_profile,
                                bool spark_rad_opacity_path,
#if LFS_RASTER_OVERLAYS
                                OverlayUniformState overlay_state,
#endif
                                inout float4 curr_pixel_state,
                                inout float median_depth,
                                inout bool thread_active) {
    const uint lane = WaveGetLaneIndex();
    const uint width = WaveGetLaneCount();
    for (uint block = range_start; block < range_end && WaveActiveAnyTrue(thread_active); block += width) {
        const uint2 live_min = WaveActiveMin(thread_active ? uint2(center_pix_coord) : uint2(0xffffffffu));
        const uint2 live_max = WaveActiveMax(thread_active ? uint2(center_pix_coord) : uint2(0u));
        const uint count = min(width, range_end - block);
        const uint source_slot = lane < count ? uint(macro_sorted_ids[block + lane]) : 0u;
        Splat_2D_AlphaBlend source;
        source.xy_vs = float2(0.0f);
        source.inv_cov_vs_opacity = float4(0.0f);
        source.rgb = float3(0.0f);
        float source_depth = 0.0f, source_support = 0.0f;
        bool source_overlaps = false;
        if (lane < count) {
            source.xy_vs = xy_vs[source_slot];
            source.inv_cov_vs_opacity = inv_cov_vs_opacity[source_slot];
            source_depth = read_t1_float(source_slot, depths);
            source_support = spark_rad_opacity_path ? power_threshold_for_spark_rad_opacity(source.inv_cov_vs_opacity.w) : power_threshold_for_opacity(source.inv_cov_vs_opacity.w);
#if LFS_RASTER_OVERLAYS
            source_overlaps = splat_overlaps_render_tile(source, tile_origin);
#else
            source_overlaps = depth_intersects_pixels(source, live_min, live_max, source_support);
#endif
        }
        const uint4 candidates = WaveActiveBallot(source_overlaps);
        for (uint index = 0u; index < count; ++index) {
            if ((candidates[index / 32u] & (1u << (index % 32u))) == 0u)
                continue;
            // All lanes broadcast before inactive pixels skip evaluation. No
            // threadgroup storage, atomics or cross-wave barriers are required.
            const uint slot = WaveReadLaneAt(source_slot, index);
            Splat_2D_AlphaBlend g;
            g.xy_vs = WaveReadLaneAt(source.xy_vs, index);
            g.inv_cov_vs_opacity = WaveReadLaneAt(source.inv_cov_vs_opacity, index);
            g.rgb = WaveReadLaneAt(source.rgb, index);
            const float splat_depth = WaveReadLaneAt(source_depth, index);
            const float support = WaveReadLaneAt(source_support, index);
            const bool overlaps = WaveReadLaneAt(source_overlaps, index);
            if (!thread_active || !overlaps)
                continue;

            float4 gauss_rgba;
            if (!spark_rad_opacity_path && splat_render_profile == 0u) {
                const float2 delta = center_pix_coord - g.xy_vs;
                const float3 conic = g.inv_cov_vs_opacity.xyz;
                const float power = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
                const float alpha = power < 0.0f || power > support ? 0.0f
                                                                    : min(g.inv_cov_vs_opacity.w * exp(-power), ALPHA_BLEND_MAX_OPACITY);
                gauss_rgba = float4(0.0f, 0.0f, 0.0f, alpha);
            } else {
                gauss_rgba = spark_rad_opacity_path
                                 ? evaluate_splat_spark_rad(g, center_pix_coord)
                                 : evaluate_splat_for_profile(g, center_pix_coord, splat_render_profile);
            }
            if (gauss_rgba.a < ALPHA_THRESHOLD)
                continue;

#if LFS_RASTER_OVERLAYS
            // Overlay attributes remain keyed by source slot. Hoisted uniforms
            // avoid per-splat overlay_params traffic during the exact replay.
            // Flags are keyed by macro-sorted slot; selection masks by original id.
            const uint flags = read_byte(gaussian_overlay_flags, slot);
            const uint sel_splat_id = uint(orig_ids[slot]);
            const uint selection_byte = overlay_state.selection_mask_enabled
                                            ? read_byte(selection_mask, sel_splat_id)
                                            : 0u;
            const uint preview_byte = overlay_state.preview_mask_enabled
                                          ? read_byte(preview_mask, sel_splat_id)
                                          : 0u;
            const uint sel_status = selection_status(g.xy_vs,
                                                     sel_splat_id,
                                                     flags,
                                                     selection_byte,
                                                     preview_byte,
                                                     overlay_state);
            const bool selectable = (flags & OVERLAY_FLAG_NONSELECTABLE) == 0u;
            gauss_rgba = apply_projection_overlay_color(gauss_rgba,
                                                        g.xy_vs,
                                                        flags,
                                                        overlay_state);
            gauss_rgba = apply_ring_overlay(gauss_rgba,
                                            g,
                                            center_pix_coord,
                                            sel_status,
                                            flags,
                                            overlay_state.show_rings,
                                            overlay_state.ring_width,
                                            selection_colors);

            if (overlay_state.show_center_markers && selectable) {
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
                continue;
            }

            gauss_rgba = apply_selection_overlay(gauss_rgba, sel_status, selection_colors);
#endif

            const float3 color = curr_pixel_state.rgb + gauss_rgba.rgb * curr_pixel_state.a;
            const float transmittance = curr_pixel_state.a * (1.0f - gauss_rgba.a);
            if (median_depth >= FAR_DEPTH &&
                curr_pixel_state.a > 0.5f &&
                transmittance <= 0.5f) {
                median_depth = splat_depth;
                thread_active = false;
            }
            curr_pixel_state = float4(color, transmittance);
            if (transmittance < 0.0001f) {
                thread_active = false;
                continue;
            }
        }
    }
}

    [shader("compute")]
    [numthreads(HIGS_TILE_WIDTH, HIGS_TILE_HEIGHT, 1)] void main(uint3 groupThreadID : SV_GroupThreadID,
                                                                 uint3 groupID : SV_GroupID,
                                                                 uniform Uniforms uniforms) {
    const uint2 tile_size = uint2(HIGS_TILE_WIDTH, HIGS_TILE_HEIGHT);
    const uint2 render_origin = full_frame_binning(uniforms)
        ? uint2(uniforms.render_origin_x, uniforms.render_origin_y) : uint2(0u);
    // Keep each wave on one full-image tile, even for an unaligned crop.
    // Leading pixels underflow and fail the output bounds check; the indirect
    // dispatch includes the trailing partial tile as well.
    const uint2 tile_coord = groupID.xy + render_origin / tile_size;
    const uint2 pix_coord = tile_coord * tile_size + groupThreadID.xy;
    const uint2 output_pixel = pix_coord - render_origin;
    const bool is_inside = output_pixel.x < uniforms.image_width && output_pixel.y < uniforms.image_height;
    const uint sample_tile = (output_pixel.y / 4u) * ((uniforms.image_width + 3u) / 4u) + output_pixel.x / 4u;
    const bool exact_depth_enabled = (uniforms.mip_filter & 2u) != 0u;
    bool exact_depth = exact_depth_enabled;
    if (is_inside && (uniforms.mip_filter & 8u) != 0u)
        exact_depth = exact_depth && (exact_depth_sample_mask[sample_tile / 32u] & (1u << (sample_tile % 32u))) != 0u;
    if (!is_inside && !exact_depth)
        return;

    const uint macro_tile_idx =
        (tile_coord.y / HIGS_MACRO_TILE_HEIGHT_TILES) * higs_macro_grid_width(uniforms.grid_width) +
        (tile_coord.x / HIGS_MACRO_TILE_WIDTH_TILES);
    const uint local_tile_idx =
        (tile_coord.y % HIGS_MACRO_TILE_HEIGHT_TILES) * HIGS_MACRO_TILE_WIDTH_TILES +
        (tile_coord.x % HIGS_MACRO_TILE_WIDTH_TILES);
    const uint thread_rank = groupThreadID.y * HIGS_TILE_WIDTH + groupThreadID.x;
    const uint2 tile_origin = tile_coord * uint2(HIGS_TILE_WIDTH, HIGS_TILE_HEIGHT);

    const uint batch_begin = uint(max(macro_tile_idx == 0u ? 0 : macro_batch_offsets[macro_tile_idx - 1u], 0));
    const uint batch_end = uint(max(macro_batch_offsets[macro_tile_idx], int32_t(batch_begin)));

    const uint pix_id = output_pixel.y * uniforms.image_width + output_pixel.x;
    const float2 center_pix_coord = float2(pix_coord);
    const bool spark_rad_opacity_path = (uniforms.lod_enabled & 4u) != 0u;
  #if LFS_RASTER_OVERLAYS
    const OverlayUniformState overlay_state = load_overlay_uniform_state(overlay_params);
  #endif

    // The per-pixel blend state persists in the output buffers between waves;
    // wave 0 initializes it.
    float4 curr_pixel_state;
    float median_depth;
    float depth_transmittance;
    if (!is_inside || (uniforms.depth_wave == 0u && uniforms.wave_base == 0u)) {
        curr_pixel_state = float4(0.0f, 0.0f, 0.0f, 1.0f);
        median_depth = FAR_DEPTH;
        depth_transmittance = 1.0f;
    } else {
        curr_pixel_state = final_pixel_state[pix_id];
        median_depth = final_pixel_depth[pix_id];
        depth_transmittance = exact_depth ? exact_depth_transmittance[pix_id] : 1.0f;
    }
    bool thread_active = is_inside && curr_pixel_state.a >= 0.0001f;
    const uint32_t local_tile_mask = 1u << local_tile_idx;

    const uint wave_walk_begin = max(batch_begin, uniforms.wave_base);
    const uint wave_walk_end = min(batch_end, uniforms.wave_base + HIGS_RASTER_WAVE_BATCHES);

    if (exact_depth_enabled && wave_walk_begin < wave_walk_end) {
        const uint macro_begin = uint(max(macro_ranges[macro_tile_idx], 0));
        const uint macro_end = uint(max(macro_ranges[macro_tile_idx + 1u], int32_t(macro_begin)));

        float4 replay_state = float4(0.0f, 0.0f, 0.0f, depth_transmittance);
        bool replay_active = is_inside && exact_depth && median_depth >= FAR_DEPTH;
        for (uint depth_batch = wave_walk_begin; depth_batch < wave_walk_end; ++depth_batch) {
            if ((macro_batch_active_mask[depth_batch] & local_tile_mask) == 0u)
                continue;
            const uint range_start = macro_begin + (depth_batch - batch_begin) * RASTER_BATCH_SIZE;
            const uint range_end = min(macro_end, range_start + RASTER_BATCH_SIZE);
            replay_batch(range_start,
                         range_end,
                         tile_origin,
                         center_pix_coord,
                         uniforms.splat_render_profile,
                         spark_rad_opacity_path,
#if LFS_RASTER_OVERLAYS
                         overlay_state,
#endif
                         replay_state,
                         median_depth,
                         replay_active);
        }
        depth_transmittance = replay_state.a;
    }
    // Partial edge pixels remained available as cooperative source lanes.
    if (!is_inside)
        return;

    for (uint batch_idx = wave_walk_begin; batch_idx < wave_walk_end && thread_active; ++batch_idx) {
        if ((macro_batch_active_mask[batch_idx] & local_tile_mask) == 0u)
            continue;

        const uint partial_idx =
            ((batch_idx - uniforms.wave_base) * HIGS_MACRO_TILE_SIZE_TILES + local_tile_idx) * HIGS_TILE_SIZE +
            thread_rank;
        float4 partial_state = float4(macro_batch_pixel_state[partial_idx]);
#if defined(LFS_VULKAN_MACOS_REFERENCE)
        if ((uniforms.mip_filter & 16u) != 0u && partial_state.a < 0.0f)
            partial_state.a = clamp(1.0f + partial_state.a + 0.00006103515625f, 0.0f, 1.0f);
#endif

        if (!exact_depth && median_depth >= FAR_DEPTH &&
            curr_pixel_state.a > 0.5f && curr_pixel_state.a * partial_state.a <= 0.5f) {
            const uint batch_in_macro = batch_idx - batch_begin;
            const uint range_start =
                uint(max(macro_ranges[macro_tile_idx], 0)) + batch_in_macro * RASTER_BATCH_SIZE;
            median_depth = read_t1_float(uint(macro_sorted_ids[range_start]), depths);
        }

        // Color retains the original FP16-partial composition and cutoff,
        // even if exact depth needs to inspect another batch.
        if (thread_active) {
            curr_pixel_state =
                float4(curr_pixel_state.rgb + partial_state.rgb * curr_pixel_state.a,
                       curr_pixel_state.a * partial_state.a);
            if (curr_pixel_state.a < 0.0001f)
                thread_active = false;
        }
    }

    // Out-of-output lanes still participate in depth wave broadcasts, but
    // crop-leading unsigned coordinates must never alias a valid output pixel.
    if (is_inside) {
        exact_depth_transmittance[pix_id] = exact_depth ? depth_transmittance : 0.0f;
        final_pixel_state[pix_id] = curr_pixel_state;
        final_pixel_depth[pix_id] = median_depth;
    }
}
