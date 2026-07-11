// HiGS macro-tile binning chain. Replaces the per-render-tile key expansion
// with macro-tile granularity (8x4 render tiles = 128x64 px): instance counts
// drop 60-85%, the stage-2 stable radix shrinks to the macro-id bit width,
// and the raster dispatches one CTA per (macro tile, 1024-splat batch).
//
// Entries:
//   1 coverage      per depth rank: conservative macro-tile count (bbox)
//   2 generate_keys per depth rank: emit (macro_id, slot) at cumsum offsets,
//                   per-macro-row ellipse refinement, sentinel padding
//   3 ranges        per sorted instance: macro segment boundaries
//   4 batch_counts  per macro tile: ceil(len / RASTER_BATCH_SIZE)
//   5 batch_prepare wave-chunked raster/compose indirect args

#include "config.slang"
#include "indirect_layout.inc"

import utils;

#define _MACRO_ENTRY_coverage 1
#define _MACRO_ENTRY_generate_keys 2
#define _MACRO_ENTRY_ranges 3
#define _MACRO_ENTRY_batch_counts 4
#define _MACRO_ENTRY_batch_prepare 5

#ifndef MACRO_ENTRY
#define MACRO_ENTRY 0
#endif

static const float MACRO_PX_W = float(HIGS_MACRO_TILE_WIDTH_TILES * HIGS_TILE_WIDTH);
static const float MACRO_PX_H = float(HIGS_MACRO_TILE_HEIGHT_TILES * HIGS_TILE_HEIGHT);

// Macro-tile bbox from the projection's render-tile rect. Exact for the
// non-wrapped case; the wrapped (equirectangular) case adds one column of
// slack for the unaligned start, covered by sentinel padding at emission.
[ForceInline]
uint4 macro_rect_from_tile_rect(rectangle rect, bool wrap_x, uint mgw, uint mgh) {
    uint my0 = uint(rect.min_y) / HIGS_MACRO_T16_H;
    uint my1 = min((uint(rect.max_y) + HIGS_MACRO_T16_H - 1u) / HIGS_MACRO_T16_H, mgh);
    if (wrap_x) {
        uint span_tiles = uint(rect.max_x) - uint(rect.min_x);
        uint span_macro = min((span_tiles + HIGS_MACRO_T16_W - 1u) / HIGS_MACRO_T16_W + 1u, mgw);
        return uint4(uint(rect.min_x) / HIGS_MACRO_T16_W, my0, span_macro, my1);
    }
    uint mx0 = uint(rect.min_x) / HIGS_MACRO_T16_W;
    uint mx1 = min((uint(rect.max_x) + HIGS_MACRO_T16_W - 1u) / HIGS_MACRO_T16_W, mgw);
    return uint4(mx0, my0, mx1 - mx0, my1);
}

[ForceInline]
rectangle load_rect(int32_t primitive_idx,
#if USE_EMULATED_INT64
                    StructuredBuffer<int32_t> rect_tile_space
#else
                    StructuredBuffer<int64_t> rect_tile_space
#endif
) {
#if USE_EMULATED_INT64
    rectangle rect;
    rect.min_x = (uint16_t)rect_tile_space[2*primitive_idx+0];
    rect.min_y = (uint16_t)(rect_tile_space[2*primitive_idx+0] >> 16);
    rect.max_x = (uint16_t)rect_tile_space[2*primitive_idx+1];
    rect.max_y = (uint16_t)(rect_tile_space[2*primitive_idx+1] >> 16);
    return rect;
#else
    return reinterpret<rectangle>(rect_tile_space[primitive_idx]);
#endif
}

#if MACRO_ENTRY == _MACRO_ENTRY_coverage

// Macro-instance count per depth rank, written in rank order so the
// visible-bounded cumsum yields depth-ordered emission offsets. The non-wrap
// count runs the same per-macro-row ellipse refinement as the emission walk
// (identical float expressions, so the counts match exactly and no sentinel
// padding is emitted); the rare wrapped case stays at the bbox bound.

layout(binding=0) StructuredBuffer<int32_t> primitive_sort_indices;
#if USE_EMULATED_INT64
layout(binding=1) StructuredBuffer<int32_t> rect_tile_space;
#else
layout(binding=1) StructuredBuffer<int64_t> rect_tile_space;
#endif
layout(binding=2) RWStructuredBuffer<int32_t> out_macro_counts;
layout(binding=3) StructuredBuffer<uint32_t> visible_count;
layout(binding=4) StructuredBuffer<float> xy_vs;
layout(binding=5) StructuredBuffer<float> inv_cov_vs_opacity;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uniform Uniforms uniforms) {
    const uint depth_rank = dispatchThreadID.x;
    const uint visible = min(visible_count[0], uniforms.num_splats);
    if (depth_rank >= visible)
        return;

    const bool wrap_x = base_camera_model(uniforms.camera_model) == uint(CameraModel::EQUIRECTANGULAR);
    const uint mgw = higs_macro_grid_width(uniforms.grid_width);
    const uint mgh = higs_macro_grid_height(uniforms.grid_height);

    const int32_t slot = primitive_sort_indices[depth_rank];
    const rectangle rect = load_rect(slot, rect_tile_space);
    const uint4 mrect = macro_rect_from_tile_rect(rect, wrap_x, mgw, mgh);
    const uint rows = mrect.w > mrect.y ? mrect.w - mrect.y : 0u;

    if (wrap_x) {
        out_macro_counts[depth_rank] = int32_t(mrect.z * rows);
        return;
    }

    const float2 xy = read_t2_float2(slot, xy_vs);
    const float4 inv_cov_opac = read_t4_float4(slot, inv_cov_vs_opacity);
    const float extend = extend_from_opacity(inv_cov_opac.w);
    const float3 inv_cov = inv_cov_opac.xyz / (extend * extend);

    const uint mx0 = mrect.x;
    const uint mx1 = mrect.x + mrect.z;
    uint count = 0u;
    for (uint my = mrect.y; my < mrect.w; ++my) {
        const float y0 = float(my) * MACRO_PX_H - xy.y;
        const float y1 = y0 + MACRO_PX_H;
        const float2 bound = ellipse_range_bound(inv_cov, y0, y1);
        const uint rx0 = uint(clamp(floor((bound.x + xy.x) / MACRO_PX_W), float(mx0), float(mx1)));
        const uint rx1 = uint(clamp(ceil((bound.y + xy.x) / MACRO_PX_W), float(mx0), float(mx1)));
        count += rx1 > rx0 ? rx1 - rx0 : 0u;
    }
    out_macro_counts[depth_rank] = int32_t(count);
}

#elif MACRO_ENTRY == _MACRO_ENTRY_generate_keys

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
layout(binding=7) StructuredBuffer<uint32_t> visible_count;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uniform Uniforms uniforms) {
    const uint depth_rank = dispatchThreadID.x;
    const uint visible = min(visible_count[0], uniforms.num_splats);
    if (depth_rank >= visible)
        return;

    int offset = depth_rank == 0u ? 0 : index_buffer_offset[depth_rank - 1u];
    int offset_max = index_buffer_offset[depth_rank];
    const int sort_capacity = int(uniforms.sort_capacity);
    if (sort_capacity <= 0 || offset < 0 || offset_max < 0 ||
        offset >= sort_capacity || offset >= offset_max)
        return;
    offset_max = min(offset_max, sort_capacity);

    const bool wrap_x = base_camera_model(uniforms.camera_model) == uint(CameraModel::EQUIRECTANGULAR);
    const uint mgw = higs_macro_grid_width(uniforms.grid_width);
    const uint mgh = higs_macro_grid_height(uniforms.grid_height);
    const sortingKey_t sentinel = sortingKey_t(mgw * mgh);

    const int32_t slot = primitive_sort_indices[depth_rank];
    const rectangle rect = load_rect(slot, rect_tile_space);
    const uint4 mrect = macro_rect_from_tile_rect(rect, wrap_x, mgw, mgh);

    if (wrap_x) {
        for (uint my = mrect.y; my < mrect.w; ++my) {
            for (uint i = 0u; i < mrect.z && offset < offset_max; ++i) {
                const uint mx = (mrect.x + i) % mgw;
                out_unsorted_keys[offset] = sortingKey_t(my * mgw + mx);
                out_unsorted_gauss_idx[offset] = slot;
                offset++;
            }
        }
    } else {
        const float2 xy = read_t2_float2(slot, xy_vs);
        const float4 inv_cov_opac = read_t4_float4(slot, inv_cov_vs_opacity);
        const float extend = extend_from_opacity(inv_cov_opac.w);
        const float3 inv_cov = inv_cov_opac.xyz / (extend * extend);

        const uint mx0 = mrect.x;
        const uint mx1 = mrect.x + mrect.z;
        for (uint my = mrect.y; my < mrect.w; ++my) {
            // Refine the column range per macro row with the exact ellipse
            // bound, mirroring generate_keys' TILE_CULLING_MODE=2 walk at
            // macro granularity.
            const float y0 = float(my) * MACRO_PX_H - xy.y;
            const float y1 = y0 + MACRO_PX_H;
            const float2 bound = ellipse_range_bound(inv_cov, y0, y1);
            const uint rx0 = uint(clamp(floor((bound.x + xy.x) / MACRO_PX_W), float(mx0), float(mx1)));
            const uint rx1 = uint(clamp(ceil((bound.y + xy.x) / MACRO_PX_W), float(mx0), float(mx1)));
            for (uint mx = rx0; mx < rx1 && offset < offset_max; ++mx) {
                out_unsorted_keys[offset] = sortingKey_t(my * mgw + mx);
                out_unsorted_gauss_idx[offset] = slot;
                offset++;
            }
        }
    }

    for (; offset < offset_max; ++offset) {
        out_unsorted_keys[offset] = sentinel;
        out_unsorted_gauss_idx[offset] = slot;
    }
}

#elif MACRO_ENTRY == _MACRO_ENTRY_ranges

layout(binding=0) StructuredBuffer<sortingKey_t> sorted_keys;
layout(binding=1) RWStructuredBuffer<int32_t> out_macro_ranges;
layout(binding=2) StructuredBuffer<uint32_t> tile_instance_count;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uniform Uniforms uniforms) {
    const uint32_t globalIdx = dispatchThreadID.x;
    const uint32_t num_macro =
        higs_macro_grid_width(uniforms.grid_width) * higs_macro_grid_height(uniforms.grid_height);

    const uint32_t num_isects = min(tile_instance_count[0], uniforms.sort_capacity);
    if (globalIdx > num_isects)
        return;

    uint32_t currtile = num_macro;
    if (globalIdx < num_isects) {
        currtile = min((uint32_t)sorted_keys[globalIdx], num_macro);
    }
    uint32_t prevtile = globalIdx == 0u ? (uint32_t)-1 :
        min((uint32_t)sorted_keys[globalIdx - 1u], num_macro);
    for (uint32_t i = prevtile + 1u; i <= currtile; ++i) {
        out_macro_ranges[i] = globalIdx;
    }
}

#elif MACRO_ENTRY == _MACRO_ENTRY_batch_counts

layout(binding=0) StructuredBuffer<int32_t> macro_ranges;
layout(binding=1) RWStructuredBuffer<int32_t> out_macro_batch_counts;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uniform Uniforms uniforms) {
    const uint mt = dispatchThreadID.x;
    const uint num_macro =
        higs_macro_grid_width(uniforms.grid_width) * higs_macro_grid_height(uniforms.grid_height);
    if (mt >= num_macro)
        return;

    const int32_t range_start = max(macro_ranges[mt], 0);
    const int32_t range_end = max(macro_ranges[mt + 1u], range_start);
    const uint len = uint(range_end - range_start);
    out_macro_batch_counts[mt] = int32_t((len + RASTER_BATCH_SIZE - 1u) / RASTER_BATCH_SIZE);
}

#elif MACRO_ENTRY == _MACRO_ENTRY_batch_prepare

layout(binding=0) StructuredBuffer<int32_t> macro_batch_offsets;
// Per-wave raster/compose bases and stride are defined by
// LFS_VK_MACRO_WAVE_DISPATCH_* in indirect_layout.inc.
layout(binding=1) RWStructuredBuffer<uint32_t> out_wave_args;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uniform Uniforms uniforms) {
    const uint num_macro =
        higs_macro_grid_width(uniforms.grid_width) * higs_macro_grid_height(uniforms.grid_height);
    const uint total_batches = num_macro > 0u
        ? uint(max(macro_batch_offsets[num_macro - 1u], 0))
        : 0u;

    for (uint w = 0u; w < HIGS_RASTER_MAX_WAVES; ++w) {
        const uint wave_begin = w * HIGS_RASTER_WAVE_BATCHES;
        const uint wave_batches = total_batches > wave_begin
            ? min(total_batches - wave_begin, HIGS_RASTER_WAVE_BATCHES)
            : 0u;
        const uint raster = LFS_VK_MACRO_WAVE_DISPATCH_RASTER_BASE_WORD_OFFSET +
                            w * LFS_VK_MACRO_WAVE_DISPATCH_WAVE_STRIDE_WORDS;
        out_wave_args[raster + LFS_VK_DISPATCH_X_WORD_OFFSET] = wave_batches;
        out_wave_args[raster + LFS_VK_DISPATCH_Y_WORD_OFFSET] = wave_batches > 0u ? 1u : 0u;
        out_wave_args[raster + LFS_VK_DISPATCH_Z_WORD_OFFSET] = wave_batches > 0u ? 1u : 0u;
        // Wave 0's compose always runs so an empty frame still initializes
        // the pixel state instead of presenting stale content.
        const bool compose_runs = wave_batches > 0u || w == 0u;
        const uint higs_grid_w = (uniforms.image_width + HIGS_TILE_WIDTH - 1u) / HIGS_TILE_WIDTH;
        const uint higs_grid_h = (uniforms.image_height + HIGS_TILE_HEIGHT - 1u) / HIGS_TILE_HEIGHT;
        const uint compose = LFS_VK_MACRO_WAVE_DISPATCH_COMPOSE_BASE_WORD_OFFSET +
                             w * LFS_VK_MACRO_WAVE_DISPATCH_WAVE_STRIDE_WORDS;
        out_wave_args[compose + LFS_VK_DISPATCH_X_WORD_OFFSET] = compose_runs ? higs_grid_w : 0u;
        out_wave_args[compose + LFS_VK_DISPATCH_Y_WORD_OFFSET] = compose_runs ? higs_grid_h : 0u;
        out_wave_args[compose + LFS_VK_DISPATCH_Z_WORD_OFFSET] = compose_runs ? 1u : 0u;
    }
}

#endif
