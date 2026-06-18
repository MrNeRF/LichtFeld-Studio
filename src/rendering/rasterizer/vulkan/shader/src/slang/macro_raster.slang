// HiGS macro-tile raster: one 320-thread CTA per (macro tile, 1024-splat
// batch). Loads the batch once into shared memory (positions, conics, colors),
// builds per-splat render-tile masks in registers and transposes them to
// per-tile splat masks with one ballot per tile, then 10 warps claim render
// tiles from a shared work queue and blend their 32 splat-mask words with a
// unanimous saturation vote per tile. Partials are written as half4 per
// (batch, tile, pixel) plus a per-batch active-tile mask for the compose pass.
//
// LFS_HIGS_FP32=0 (default): half staging. Positions are macro-tile-relative
// in render-tile units (bounded, so fp16-safe), conics are Cholesky factors
// prescaled by tile size and 1/sqrt(2 ln 2) — the quadratic evaluates without
// the catastrophic cancellation a half-precision inverse covariance would
// have, and alpha = opacity * exp2(-q).
//
// LFS_HIGS_FP32=1: float staging with the exact legacy evaluate_splat math;
// used for validation and as the overlay/editor variant.

#include "config.slang"

import utils;

#ifndef LFS_HIGS_FP32
#define LFS_HIGS_FP32 0
#endif

#ifndef LFS_RASTER_OVERLAYS
#define LFS_RASTER_OVERLAYS 0
#endif

#if LFS_RASTER_OVERLAYS && !LFS_HIGS_FP32
#error "overlay variant requires fp32 staging"
#endif

#define RASTER_THREAD_COUNT 320u
#define MINI_BATCHES (RASTER_BATCH_SIZE / SUBGROUP_SIZE)
static const uint PIXELS_PER_THREAD = HIGS_TILE_SIZE / SUBGROUP_SIZE;
static const uint ROWS_PER_P = SUBGROUP_SIZE / HIGS_TILE_WIDTH;
static const uint DRAIN_CAPACITY = HIGS_DRAIN_BATCH + SUBGROUP_SIZE;
static const float INV_SQRT_2LN2 = 0.84932180028801904272f;
static const float INV_LN2 = 1.44269504088896340736f;

#if LFS_HIGS_FP32
groupshared float2 shared_xy[RASTER_BATCH_SIZE];
groupshared float4 shared_inv_cov_opac[RASTER_BATCH_SIZE];
#else
groupshared half2 shared_xy[RASTER_BATCH_SIZE];          // tile units, macro-relative
groupshared half4 shared_chol_thr[RASTER_BATCH_SIZE];    // {l00', l01', l11', thr'}
#endif
groupshared half4 shared_rgb_opac[RASTER_BATCH_SIZE];    // {r, g, b, opacity}
#if LFS_RASTER_OVERLAYS
groupshared uint32_t shared_slot[RASTER_BATCH_SIZE];
#endif
groupshared uint32_t shared_tile_masks[MINI_BATCHES * HIGS_MACRO_TILE_SIZE_TILES];
#if !LFS_HIGS_FP32
// Per-warp drain ring: compacted splat indices, blended in dense batches so
// the hot loop carries no serial bit-scan dependency.
groupshared uint16_t shared_drain_queue[(RASTER_THREAD_COUNT / SUBGROUP_SIZE) * DRAIN_CAPACITY];
#endif
groupshared uint32_t shared_tile_queue[HIGS_MACRO_TILE_SIZE_TILES];
groupshared uint32_t shared_tile_queue_count;
groupshared uint32_t shared_tile_cursor;
groupshared uint32_t shared_active_mask;
groupshared uint32_t shared_macro_tile_idx;

layout(binding=0) StructuredBuffer<int32_t> macro_sorted_ids;
layout(binding=1) StructuredBuffer<int32_t> macro_ranges;
layout(binding=2) StructuredBuffer<int32_t> macro_batch_offsets;
layout(binding=3) StructuredBuffer<float2> xy_vs;
layout(binding=4) StructuredBuffer<float4> inv_cov_vs_opacity;
layout(binding=5) StructuredBuffer<float> rgb;
layout(binding=6) RWStructuredBuffer<half4> macro_batch_pixel_state;
layout(binding=7) RWStructuredBuffer<uint32_t> macro_batch_active_mask;
#if LFS_RASTER_OVERLAYS
layout(binding=8) ByteAddressBuffer selection_mask;
layout(binding=9) ByteAddressBuffer preview_mask;
layout(binding=10) StructuredBuffer<float4> selection_colors;
layout(binding=11) StructuredBuffer<uint> gaussian_overlay_flags;
layout(binding=12) StructuredBuffer<float4> overlay_params;
layout(binding=13) StructuredBuffer<int32_t> orig_ids;
#endif

[ForceInline]
uint find_macro_tile_for_batch(uint batch_idx, uint num_macro) {
    uint lo = 0u;
    uint hi = num_macro;
    while (lo < hi) {
        const uint mid = (lo + hi) >> 1;
        if (uint(max(macro_batch_offsets[mid], 0)) <= batch_idx)
            lo = mid + 1u;
        else
            hi = mid;
    }
    return lo;
}

// Full 32-bit render-tile mask for one splat against the 8x4 macro-tile
// layout, from its opacity-aware bounding box. The extent math runs once per
// splat instead of once per splat-tile pair.
[ForceInline]
uint32_t splat_macro_tile_mask(float2 xy, float4 inv_cov_opac, float2 macro_origin) {
    const float extend = max(extend_from_opacity(inv_cov_opac.w), 1.0e-6f);
    const float det = inv_cov_opac.x * inv_cov_opac.z - inv_cov_opac.y * inv_cov_opac.y;
    if (det <= 0.0f)
        return 0u;
    const float2 wh = extend * sqrt(float2(inv_cov_opac.z, inv_cov_opac.x) / det);
    const float2 rel = xy - macro_origin;

    const int x0 = clamp(int(floor((rel.x - wh.x) / float(HIGS_TILE_WIDTH))), 0, int(HIGS_MACRO_TILE_WIDTH_TILES));
    const int x1 = clamp(int(ceil((rel.x + wh.x) / float(HIGS_TILE_WIDTH))), x0, int(HIGS_MACRO_TILE_WIDTH_TILES));
    const int y0 = clamp(int(floor((rel.y - wh.y) / float(HIGS_TILE_HEIGHT))), 0, int(HIGS_MACRO_TILE_HEIGHT_TILES));
    const int y1 = clamp(int(ceil((rel.y + wh.y) / float(HIGS_TILE_HEIGHT))), y0, int(HIGS_MACRO_TILE_HEIGHT_TILES));
    if (x0 >= x1 || y0 >= y1)
        return 0u;

    const uint32_t row_bits = ((1u << uint(x1 - x0)) - 1u) << uint(x0);
    uint32_t mask = 0u;
    for (int y = y0; y < y1; ++y)
        mask |= row_bits << (uint(y) * HIGS_MACRO_TILE_WIDTH_TILES);
    return mask;
}

[shader("compute")]
[numthreads(RASTER_THREAD_COUNT, 1, 1)]
void main(uint3 groupThreadID : SV_GroupThreadID,
          uint3 groupID : SV_GroupID,
          uniform Uniforms uniforms)
{
    const uint macro_batch_idx = uniforms.wave_base + groupID.x;
    const uint pool_batch_idx = groupID.x;
    const uint thread_rank = groupThreadID.x;
    const uint lane_id = thread_rank % SUBGROUP_SIZE;
    const uint warp_id = thread_rank / SUBGROUP_SIZE;
    const uint warp_count = RASTER_THREAD_COUNT / SUBGROUP_SIZE;
    const uint mgw = higs_macro_grid_width(uniforms.grid_width);
    const uint num_macro = mgw * higs_macro_grid_height(uniforms.grid_height);

    if (thread_rank == 0u) {
        shared_active_mask = 0u;
        shared_tile_queue_count = 0u;
        shared_tile_cursor = 0u;
        shared_macro_tile_idx = find_macro_tile_for_batch(macro_batch_idx, num_macro);
    }
    GroupMemoryBarrierWithGroupSync();

    const uint macro_tile_idx = shared_macro_tile_idx;
    if (macro_tile_idx >= num_macro)
        return;

    const uint macro_tile_x = macro_tile_idx % mgw;
    const uint macro_tile_y = macro_tile_idx / mgw;
    const float2 macro_origin = float2(float(macro_tile_x * HIGS_MACRO_TILE_WIDTH_TILES * HIGS_TILE_WIDTH),
                                       float(macro_tile_y * HIGS_MACRO_TILE_HEIGHT_TILES * HIGS_TILE_HEIGHT));

    const uint macro_batch_begin = uint(max(macro_tile_idx == 0u ? 0 : macro_batch_offsets[macro_tile_idx - 1u], 0));
    const uint batch_in_macro = macro_batch_idx - macro_batch_begin;
    const uint range_start = uint(max(macro_ranges[macro_tile_idx], 0)) + batch_in_macro * RASTER_BATCH_SIZE;
    uint range_end = uint(max(macro_ranges[macro_tile_idx + 1u], int32_t(range_start)));
    range_end = min(range_start + RASTER_BATCH_SIZE, range_end);

    const uint splats_to_process = range_end - range_start;
    const uint num_mini_batches = (splats_to_process + SUBGROUP_SIZE - 1u) / SUBGROUP_SIZE;
    const bool spark_rad_opacity_path = (uniforms.lod_enabled & 4u) != 0u;

    // Phase 1: cooperative load + per-splat tile mask + ballot transpose. The
    // whole batch (positions, conics, colors) is staged once; render tiles
    // never re-read splat data from global memory.
    for (uint mini_batch = warp_id; mini_batch < num_mini_batches; mini_batch += warp_count) {
        const uint i = mini_batch * SUBGROUP_SIZE + lane_id;
        uint32_t tile_mask = 0u;
        if (i < splats_to_process) {
            const uint32_t slot = uint32_t(macro_sorted_ids[range_start + i]);
            const float2 g_xy = xy_vs[slot];
            const float4 g_ico = inv_cov_vs_opacity[slot];
            const float3 g_rgb = clamp(read_t3_float3(int32_t(slot), rgb), float3(0.0f), float3(MAX_BLEND_COLOR));
            tile_mask = splat_macro_tile_mask(g_xy, g_ico, macro_origin);

          #if LFS_HIGS_FP32
            shared_xy[i] = g_xy;
            shared_inv_cov_opac[i] = g_ico;
          #else
            const float2 rel_tiles = (g_xy - macro_origin) * float2(1.0f / HIGS_TILE_WIDTH, 1.0f / HIGS_TILE_HEIGHT);
            const float a = max(g_ico.x, 1.0e-12f);
            const float l00 = sqrt(a);
            const float l01 = g_ico.y / l00;
            const float l11 = sqrt(max(g_ico.z - l01 * l01, 0.0f));
            // Spark-rad opacity scenes are routed to the fp32 variant by the
            // host, so the fp16 path only needs the standard threshold.
            const float thr = power_threshold_for_opacity(g_ico.w);
            shared_xy[i] = half2(rel_tiles);
            shared_chol_thr[i] = half4(float4(l00 * float(HIGS_TILE_WIDTH) * INV_SQRT_2LN2,
                                              l01 * float(HIGS_TILE_HEIGHT) * INV_SQRT_2LN2,
                                              l11 * float(HIGS_TILE_HEIGHT) * INV_SQRT_2LN2,
                                              thr * INV_LN2));
          #endif
            shared_rgb_opac[i] = half4(float4(g_rgb, g_ico.w));
          #if LFS_RASTER_OVERLAYS
            shared_slot[i] = slot;
          #endif
        }
        // One ballot per render tile transposes the per-splat tile masks into
        // a per-tile splat mask, lane t collecting tile t.
        uint32_t tile_gauss_mask = 0u;
        [unroll]
        for (uint t = 0u; t < HIGS_MACRO_TILE_SIZE_TILES; ++t) {
            const uint4 ballot = WaveActiveBallot(((tile_mask >> t) & 1u) != 0u);
            if (lane_id == t)
                tile_gauss_mask = ballot.x;
        }
        shared_tile_masks[mini_batch * HIGS_MACRO_TILE_SIZE_TILES + lane_id] = tile_gauss_mask;
    }
    GroupMemoryBarrierWithGroupSync();

    // Active-tile mask + work queue.
    if (thread_rank < HIGS_MACRO_TILE_SIZE_TILES) {
        uint32_t any = 0u;
        for (uint mini_batch = 0u; mini_batch < num_mini_batches; ++mini_batch)
            any |= shared_tile_masks[mini_batch * HIGS_MACRO_TILE_SIZE_TILES + thread_rank];
        const uint tile_x = macro_tile_x * HIGS_MACRO_TILE_WIDTH_TILES + (thread_rank % HIGS_MACRO_TILE_WIDTH_TILES);
        const uint tile_y = macro_tile_y * HIGS_MACRO_TILE_HEIGHT_TILES + (thread_rank / HIGS_MACRO_TILE_WIDTH_TILES);
        const bool tile_in_image = tile_x * HIGS_TILE_WIDTH < uniforms.image_width &&
                                   tile_y * HIGS_TILE_HEIGHT < uniforms.image_height;
        if (any != 0u && tile_in_image) {
            uint q;
            InterlockedAdd(shared_tile_queue_count, 1u, q);
            shared_tile_queue[q] = thread_rank;
            InterlockedOr(shared_active_mask, 1u << thread_rank);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (thread_rank == 0u) {
        macro_batch_active_mask[macro_batch_idx] = shared_active_mask;
    }

    // Phase 2: warps steal render tiles from the queue until it runs dry, so
    // no warp idles while early-saturating tiles finish ahead of dense ones.
    for (;;) {
        uint queue_idx = 0u;
        if (WaveIsFirstLane()) {
            InterlockedAdd(shared_tile_cursor, 1u, queue_idx);
        }
        queue_idx = WaveReadLaneFirst(queue_idx);
        if (queue_idx >= shared_tile_queue_count)
            break;

        const uint local_tile_idx = shared_tile_queue[queue_idx];
        const uint tile_x = macro_tile_x * HIGS_MACRO_TILE_WIDTH_TILES + (local_tile_idx % HIGS_MACRO_TILE_WIDTH_TILES);
        const uint tile_y = macro_tile_y * HIGS_MACRO_TILE_HEIGHT_TILES + (local_tile_idx / HIGS_MACRO_TILE_WIDTH_TILES);
        const uint2 tile_origin = uint2(tile_x * HIGS_TILE_WIDTH, tile_y * HIGS_TILE_HEIGHT);

      #if LFS_HIGS_FP32
        float4 pixel_state[PIXELS_PER_THREAD];
        bool pixel_active[PIXELS_PER_THREAD];
        float2 pixel_coord[PIXELS_PER_THREAD];

        [unroll]
        for (uint p = 0u; p < PIXELS_PER_THREAD; ++p) {
            const uint pixel_linear = lane_id + p * SUBGROUP_SIZE;
            const uint2 pix = tile_origin + uint2(pixel_linear % HIGS_TILE_WIDTH, pixel_linear / HIGS_TILE_WIDTH);
            pixel_state[p] = float4(0.0f, 0.0f, 0.0f, 1.0f);
            pixel_active[p] = pix.x < uniforms.image_width && pix.y < uniforms.image_height;
            pixel_coord[p] = float2(pix);
        }

        bool tile_active = true;
        for (uint mini_batch = 0u; mini_batch < num_mini_batches && tile_active; ++mini_batch) {
            uint32_t gauss_mask = shared_tile_masks[mini_batch * HIGS_MACRO_TILE_SIZE_TILES + local_tile_idx];
            while (gauss_mask != 0u) {
                const uint bit_idx = firstbitlow(gauss_mask);
                gauss_mask &= gauss_mask - 1u;
                const uint i = mini_batch * SUBGROUP_SIZE + bit_idx;
                if (i >= splats_to_process)
                    continue;

                const float4 g_rgbo = float4(shared_rgb_opac[i]);
                const float2 g_xy = shared_xy[i];
                const float4 g_ico = shared_inv_cov_opac[i];

              #if LFS_RASTER_OVERLAYS
                const uint32_t splat_slot = shared_slot[i];
                const uint flags = gaussian_overlay_flags[splat_slot];
                const uint sel_splat_id = uint(orig_ids[splat_slot]);
                const uint sel_status = selection_status(g_xy,
                                                         sel_splat_id,
                                                         flags,
                                                         selection_mask,
                                                         preview_mask,
                                                         overlay_params);
                const bool selectable = (flags & OVERLAY_FLAG_NONSELECTABLE) == 0u;
                const Splat_2D_AlphaBlend overlay_splat = { g_ico, g_xy, g_rgbo.rgb };
              #endif

                [unroll]
                for (uint p = 0u; p < PIXELS_PER_THREAD; ++p) {
                    if (!pixel_active[p])
                        continue;

                    const float2 d = pixel_coord[p] - g_xy;
                    const float sigma_over_2 =
                        0.5f * (g_ico.x * d.x * d.x + g_ico.z * d.y * d.y) + g_ico.y * d.x * d.y;
                    const float thr = spark_rad_opacity_path
                                          ? power_threshold_for_spark_rad_opacity(g_rgbo.w)
                                          : power_threshold_for_opacity(g_rgbo.w);
                    if (sigma_over_2 < 0.0f || sigma_over_2 > thr)
                        continue;
                    const float val = exp(-sigma_over_2);

                    const float alpha = spark_rad_opacity_path
                                            ? spark_rad_alpha_from_value(g_rgbo.w, val)
                                            : float(clipOpacity(g_rgbo.w * val));
                    if (alpha < ALPHA_THRESHOLD)
                        continue;
                    float4 gauss_rgba = float4(g_rgbo.rgb * alpha, alpha);

                  #if LFS_RASTER_OVERLAYS
                    gauss_rgba = apply_projection_overlay_color(gauss_rgba,
                                                                overlay_splat.xy_vs,
                                                                flags,
                                                                overlay_params);
                    gauss_rgba = apply_ring_overlay(gauss_rgba, overlay_splat, pixel_coord[p], flags, overlay_params);

                    if (overlay_enabled(overlay_params[OVERLAY_PARAM_MARKER_FLAGS].y) && selectable) {
                        const float INNER_RADIUS_SQ = 2.25f;
                        const float OUTER_RADIUS_SQ = 6.25f;
                        const float OUTLINE_DARKEN = 0.4f;
                        const float2 marker_delta = overlay_splat.xy_vs - pixel_coord[p];
                        const float dist_sq = dot(marker_delta, marker_delta);
                        if (dist_sq > OUTER_RADIUS_SQ)
                            continue;

                        const uint group_id = sel_status & SELECTION_GROUP_MASK;
                        const bool is_preview = (sel_status & SELECTION_PREVIEW_FLAG) != 0u;
                        float3 marker_color = is_preview ? selection_colors[SELECTION_PREVIEW_COLOR_INDEX].xyz
                                                         : selection_colors[group_id].xyz;
                        if (dist_sq > INNER_RADIUS_SQ)
                            marker_color *= OUTLINE_DARKEN;

                        // Absorb-all write: the partial encodes the marker
                        // exactly, so the compose pass stays a pure composite.
                        pixel_state[p] = float4(marker_color, 0.0f);
                        pixel_active[p] = false;
                        continue;
                    }

                    gauss_rgba = apply_selection_overlay(gauss_rgba, sel_status, selection_colors);
                    gauss_rgba = apply_flash_overlay(gauss_rgba, flags, overlay_params);
                  #endif

                    const float3 color = pixel_state[p].rgb + gauss_rgba.rgb * pixel_state[p].a;
                    const float transmittance = pixel_state[p].a * (1.0f - gauss_rgba.a);
                    if (transmittance < 0.0001f) {
                        pixel_active[p] = false;
                    }
                    pixel_state[p] = float4(color, transmittance);
                }
            }

            // Unanimous saturation vote once per 32-splat mask word (the
            // reference votes per drain batch, not per splat): the ballot
            // cost amortizes and at most one word of saturated work leaks.
            bool thread_active = false;
            [unroll]
            for (uint p = 0u; p < PIXELS_PER_THREAD; ++p)
                thread_active = thread_active || pixel_active[p];
            if (!WaveActiveAnyTrue(thread_active)) {
                tile_active = false;
            }
        }

        [unroll]
        for (uint p = 0u; p < PIXELS_PER_THREAD; ++p) {
            const uint pixel_linear = lane_id + p * SUBGROUP_SIZE;
            const uint partial_idx =
                (pool_batch_idx * HIGS_MACRO_TILE_SIZE_TILES + local_tile_idx) * HIGS_TILE_SIZE + pixel_linear;
            macro_batch_pixel_state[partial_idx] = half4(pixel_state[p]);
        }

      #else // !LFS_HIGS_FP32: half2-packed blend core

        // Lane pixel layout: pixel_linear = lane + p*SUBGROUP_SIZE, so
        // x = lane % tile_width is constant per lane and y steps by
        // ROWS_PER_P px per p. Pixels are packed in pairs (2k, 2k+1) and
        // evaluated as half2 — dx is computed once per splat, and the
        // quadratic, exp2 and blend all run two pixels per op.
        static const uint PIXEL_PAIRS = PIXELS_PER_THREAD / 2u;
        static const float PIX_STEP_T = float(ROWS_PER_P) / float(HIGS_TILE_HEIGHT);

        const float fx_t = (float(tile_origin.x + (lane_id % HIGS_TILE_WIDTH)) - macro_origin.x) * (1.0f / HIGS_TILE_WIDTH);
        const float fy0_t = (float(tile_origin.y + (lane_id / HIGS_TILE_WIDTH)) - macro_origin.y) * (1.0f / HIGS_TILE_HEIGHT);
        const half px_x = half(fx_t);
        const bool col_inside = (tile_origin.x + (lane_id % HIGS_TILE_WIDTH)) < uniforms.image_width;

        half2 pix_y2[PIXEL_PAIRS];
        half2 acc_r[PIXEL_PAIRS];
        half2 acc_g[PIXEL_PAIRS];
        half2 acc_b[PIXEL_PAIRS];
        half2 acc_t[PIXEL_PAIRS];
        [unroll]
        for (uint k = 0u; k < PIXEL_PAIRS; ++k) {
            pix_y2[k] = half2(half(fy0_t + float(2u * k) * PIX_STEP_T),
                              half(fy0_t + float(2u * k + 1u) * PIX_STEP_T));
            const uint y0 = tile_origin.y + (lane_id / HIGS_TILE_WIDTH) + 2u * ROWS_PER_P * k;
            const bool in0 = col_inside && y0 < uniforms.image_height;
            const bool in1 = col_inside && (y0 + ROWS_PER_P) < uniforms.image_height;
            acc_r[k] = half2(0.0h, 0.0h);
            acc_g[k] = half2(0.0h, 0.0h);
            acc_b[k] = half2(0.0h, 0.0h);
            acc_t[k] = half2(in0 ? 1.0h : 0.0h, in1 ? 1.0h : 0.0h);
        }

        const half2 eps2 = half2(half(0.0001f), half(0.0001f));
        const half2 alpha_min2 = half2(half(ALPHA_THRESHOLD), half(ALPHA_THRESHOLD));
        const half2 alpha_cap2 = half2(half(ALPHA_BLEND_MAX_OPACITY), half(ALPHA_BLEND_MAX_OPACITY));
        const half2 zero2 = half2(0.0h, 0.0h);
        const half2 one2 = half2(1.0h, 1.0h);

        // Drain ring: the warp stream-compacts each mask word into a per-warp
        // shared queue (every lane writes at most one index per word), then
        // blends the dense list — the hot loop carries no serial bit-scan
        // dependency and software-pipelines the shared-memory loads. The
        // unanimous saturation vote runs once per drained batch (Bd = 64).
        const uint qbase = warp_id * DRAIN_CAPACITY;
        uint qcount = 0u;
        bool tile_active = true;
        for (uint mini_batch = 0u; mini_batch < num_mini_batches && tile_active; ++mini_batch) {
            const uint32_t word = shared_tile_masks[mini_batch * HIGS_MACRO_TILE_SIZE_TILES + local_tile_idx];
            if (word != 0u) {
                const uint ofs = countbits(word & ((1u << lane_id) - 1u));
                if (((word >> lane_id) & 1u) != 0u) {
                    shared_drain_queue[qbase + qcount + ofs] = uint16_t(mini_batch * SUBGROUP_SIZE + lane_id);
                }
                qcount += countbits(word);
            }
            if (qcount >= HIGS_DRAIN_BATCH || (qcount > 0u && mini_batch + 1u == num_mini_batches)) {
                GroupMemoryBarrierWithWaveSync();
                for (uint j = 0u; j < qcount; ++j) {
                    const uint i = uint(shared_drain_queue[qbase + j]);
                    const half2 g_xy = shared_xy[i];
                    const half4 g_chol = shared_chol_thr[i];
                    const half4 g_rgbo = shared_rgb_opac[i];
                    const half u_dx = g_chol.x * (px_x - g_xy.x);
                    const half2 u_dx2 = half2(u_dx, u_dx);
                    const half2 thr2 = half2(g_chol.w, g_chol.w);
                    const half2 gy2 = half2(g_xy.y, g_xy.y);

                    [unroll]
                    for (uint k = 0u; k < PIXEL_PAIRS; ++k) {
                        const half2 dy2 = pix_y2[k] - gy2;
                        const half2 u2 = u_dx2 + g_chol.y * dy2;
                        const half2 v2 = g_chol.z * dy2;
                        const half2 q2 = u2 * u2 + v2 * v2;
                        half2 alpha2 = min(g_rgbo.w * exp2(-q2), alpha_cap2);
                        alpha2 = select(q2 <= thr2, alpha2, zero2);
                        alpha2 = select(alpha2 >= alpha_min2, alpha2, zero2);
                        const half2 w2 = alpha2 * acc_t[k];
                        acc_r[k] += g_rgbo.x * w2;
                        acc_g[k] += g_rgbo.y * w2;
                        acc_b[k] += g_rgbo.z * w2;
                        acc_t[k] *= one2 - alpha2;
                    }
                }
                qcount = 0u;
                bool thread_active = false;
                [unroll]
                for (uint k = 0u; k < PIXEL_PAIRS; ++k)
                    thread_active = thread_active || any(acc_t[k] >= eps2);
                if (!WaveActiveAnyTrue(thread_active)) {
                    tile_active = false;
                }
                // Next round's compaction writes must not race this round's
                // dense reads.
                GroupMemoryBarrierWithWaveSync();
            }
        }

        [unroll]
        for (uint k = 0u; k < PIXEL_PAIRS; ++k) {
            [unroll]
            for (uint j = 0u; j < 2u; ++j) {
                const uint p = 2u * k + j;
                const uint pixel_linear = lane_id + p * SUBGROUP_SIZE;
                const uint partial_idx =
                    (pool_batch_idx * HIGS_MACRO_TILE_SIZE_TILES + local_tile_idx) * HIGS_TILE_SIZE + pixel_linear;
                macro_batch_pixel_state[partial_idx] =
                    half4(acc_r[k][j], acc_g[k][j], acc_b[k][j], acc_t[k][j]);
            }
        }
      #endif // LFS_HIGS_FP32
    }
}
