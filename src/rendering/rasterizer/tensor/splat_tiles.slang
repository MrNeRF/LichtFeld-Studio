// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
// Tile binning for 16x16 pixel tiles: per-splat tile counts, the instance
// total and indirect sort arguments, one (tile, depth) key per covered tile,
// and each tile's range in the sorted keys. Ported from the native Metal
// viewer rasterizer; keys and their order are identical.
#include "splat_types.slang"

static const uint kDepthChunkSize = 576u;

struct Parameters {
    Ptr<ProjectedSplat, Access.Read> splats;
    Ptr<uint64_t, Access.ReadWrite> counts;
    Ptr<uint64_t, Access.ReadWrite> offsets;
    Ptr<RasterStatus, Access.ReadWrite> status;
    Ptr<uint, Access.ReadWrite> dispatch_args;
    Ptr<uint64_t, Access.ReadWrite> keys; // uint32 tile keys when source-sorted
    Ptr<uint, Access.ReadWrite> indices;
    Ptr<uint, Access.ReadWrite> ranges;   // [tiles][begin, end)
    Ptr<uint, Access.Read> source_order;      // optional: source-sorted frames
    Ptr<uint64_t, Access.Read> source_counts; // optional: source-sorted frames
    Ptr<RasterParameters, Access.Read> raster;
};
[[vk::push_constant]] ConstantBuffer<Parameters> p;

uint4 clipped_bounds(ProjectedSplat s, RasterParameters r) {
    if (!all(isfinite(s.mean_depth)) || s.mean_depth.z <= 0 ||
        !all(isfinite(s.conic_opacity)) || !all(isfinite(s.color)))
        return uint4(0);
    if (r.camera.z == 2u) {
        const uint span = r.columns * 16u;
        if (s.bounds.x >= span || s.bounds.z < s.bounds.x || s.bounds.z - s.bounds.x > span ||
            (s.bounds.x % 16u) != 0u || (s.bounds.z % 16u) != 0u)
            return uint4(0);
        return uint4(s.bounds.x, min(s.bounds.y, r.height), s.bounds.z, min(s.bounds.w, r.height));
    }
    return min(s.bounds, uint4(r.width, r.height, r.width, r.height));
}

bool source_sorted(RasterParameters r) { return (r.unused & 256u) != 0u; }

[shader("compute")]
[numthreads(256, 1, 1)]
void tile_counts(uint3 tid : SV_DispatchThreadID) {
    const uint i = tid.x;
    const RasterParameters r = p.raster[0];
    if (i >= r.count)
        return;
    if (source_sorted(r)) {
        const uint source = p.source_order[i];
        p.counts[i] = source < r.count ? p.source_counts[source] : 0ul;
    } else {
        const uint4 b = clipped_bounds(p.splats[i], r);
        p.counts[i] = b.z > b.x && b.w > b.y
                          ? uint64_t((b.z + 15u) / 16u - b.x / 16u) * uint64_t((b.w + 15u) / 16u - b.y / 16u)
                          : 0ul;
    }
}

void write_sort_dispatch(RasterStatus status) {
    const uint64_t live = status.error != 0 ? 0ul : status.required;
    p.dispatch_args[kSortArgs] = max(1u, uint((live + 2047ul) / 2048ul));
    p.dispatch_args[kSortArgs + 1] = 1u;
    p.dispatch_args[kSortArgs + 2] = 1u;
    p.dispatch_args[kRangeArgs] = max(1u, uint((live + 255ul) / 256ul));
    p.dispatch_args[kRangeArgs + 1] = 1u;
    p.dispatch_args[kRangeArgs + 2] = 1u;
}

// One thread: the instance total, overflow against the sort capacity, and
// the indirect arguments every later pass dispatches with.
[shader("compute")]
[numthreads(1, 1, 1)]
void tile_status(uint3 tid : SV_DispatchThreadID) {
    const RasterParameters r = p.raster[0];
    RasterStatus status;
    status.required = r.count != 0u ? p.offsets[r.count - 1u] + p.counts[r.count - 1u] : 0ul;
    status.error = status.required > uint64_t(r.capacity) ? 1u : 0u;
    status.blend_threads = (r.unused & 128u) != 0u ? 32u : 64u;
    status.maximum_tile_instances = 0u;
    status.padding = 0u;
    p.status[0] = status;
    write_sort_dispatch(status);
    const bool parallel = status.required <= uint64_t(r.mask_limits.z);
    p.dispatch_args[18] = parallel
                              ? (uint(((status.error != 0 ? 0ul : status.required) + uint64_t(kDepthChunkSize - 1u)) / uint64_t(kDepthChunkSize)) + r.tiles) * 8u
                              : 1u;
    p.dispatch_args[19] = 1u;
    p.dispatch_args[20] = 1u;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void tile_instances(uint3 tid : SV_DispatchThreadID) {
    const uint i = tid.x;
    const RasterParameters r = p.raster[0];
    if (i >= r.count || p.status[0].error != 0u)
        return;
    const bool sorted = source_sorted(r);
    const uint source = sorted ? p.source_order[i] : i;
    if (source >= r.count)
        return;
    const ProjectedSplat s = p.splats[source];
    const uint4 bounds = clipped_bounds(s, r);
    if (bounds.z <= bounds.x || bounds.w <= bounds.y)
        return;
    uint64_t at = p.offsets[i];
    for (uint y = bounds.y / 16u; y < (bounds.w + 15u) / 16u; ++y) {
        for (uint x = bounds.x / 16u; x < (bounds.z + 15u) / 16u; ++x) {
            const uint column = r.camera.z == 2u ? x % r.columns : x;
            const uint tile = y * r.columns + column;
            // Source sorting already ordered the full depth key and its ties;
            // the remaining stable pass needs only the tile.
            if (sorted)
                ((Ptr<uint, Access.ReadWrite>)p.keys)[at] = tile;
            else
                p.keys[at] = (uint64_t(tile) << 32) | uint64_t(asuint(s.color.w));
            p.indices[at] = source;
            ++at;
        }
    }
}

uint tile_of(uint64_t index, bool sorted) {
    return sorted ? ((Ptr<uint, Access.ReadWrite>)p.keys)[index] : uint(p.keys[index] >> 32);
}

// ranges must be zero before this pass; tiles without instances stay [0, 0).
[shader("compute")]
[numthreads(256, 1, 1)]
void tile_ranges(uint3 tid : SV_DispatchThreadID) {
    const uint i = tid.x;
    const RasterStatus status = p.status[0];
    if (status.error != 0u || uint64_t(i) >= status.required)
        return;
    const bool sorted = source_sorted(p.raster[0]);
    const uint tile = tile_of(i, sorted);
    if (i == 0u) {
        p.ranges[tile * 2u] = 0u;
    } else {
        const uint previous = tile_of(i - 1u, sorted);
        if (tile != previous) {
            p.ranges[previous * 2u + 1u] = i;
            p.ranges[tile * 2u] = i;
        }
    }
    if (uint64_t(i) + 1ul == status.required)
        p.ranges[tile * 2u + 1u] = i + 1u;
}
