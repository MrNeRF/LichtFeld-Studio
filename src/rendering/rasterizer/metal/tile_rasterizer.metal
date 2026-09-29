// SPDX-License-Identifier: GPL-3.0-or-later
#include <metal_stdlib>
using namespace metal;

struct ProjectedSplat {
    float4 mean_depth, conic_opacity, color;
    uint4 bounds;
};
struct RasterParameters {
    uint count, width, height, columns;
    uint tiles, capacity, mode, unused;
    float4 background;
};
struct RasterStatus { ulong required; uint error, unused; };
uint4 clipped_bounds(ProjectedSplat s, constant RasterParameters& p) {
    if (!all(isfinite(s.mean_depth)) || s.mean_depth.z <= 0 ||
        !all(isfinite(s.conic_opacity)) || !all(isfinite(s.color))) return uint4(0);
    return min(s.bounds, uint4(p.width,p.height,p.width,p.height));
}

kernel void tile_counts(device const ProjectedSplat* splats [[buffer(0)]],
                        device ulong* counts [[buffer(1)]],
                        constant RasterParameters& p [[buffer(2)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= p.count) return;
    const uint4 b = clipped_bounds(splats[i],p);
    counts[i] = b.z > b.x && b.w > b.y ?
        ulong((b.z + 15) / 16 - b.x / 16) * ((b.w + 15) / 16 - b.y / 16) : 0;
}

// Hierarchical exclusive scan. UInt64 preserves the required size on overflow;
// no 64-bit atomics or CPU roundtrip are needed.
kernel void scan_blocks(device const ulong* input [[buffer(0)]],
                        device ulong* output [[buffer(1)]],
                        device ulong* sums [[buffer(2)]],
                        constant uint& count [[buffer(3)]],
                        uint group [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup ulong group_sums[8];
    const uint i = group * 256 + lane;
    const ulong value = i < count ? input[i] : 0;
    // MSL 2.4 SIMD reductions do not accept ulong. Sum 16-bit limbs with
    // uint32 operations (32 lanes cannot overflow), then carry in ulong.
    ulong prefix = 0, simd_total = 0;
    for (uint shift = 0; shift < 64; shift += 16) {
        const uint limb = uint((value >> shift) & 65535);
        prefix += ulong(simd_prefix_exclusive_sum(limb)) << shift;
        simd_total += ulong(simd_sum(limb)) << shift;
    }
    if ((lane & 31) == 0) group_sums[sg] = simd_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 0; s < sg; ++s) prefix += group_sums[s];
    if (i < count) output[i] = prefix;
    if (lane == 255) sums[group] = prefix + value;
}
kernel void scan_add(device ulong* output [[buffer(0)]],
                     device const ulong* offsets [[buffer(1)]],
                     constant uint& count [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < count) output[i] += offsets[i / 256];
}
kernel void tile_status(device const ulong* counts [[buffer(0)]],
                        device const ulong* offsets [[buffer(1)]],
                        device RasterStatus& status [[buffer(2)]],
                        constant RasterParameters& p [[buffer(3)]]) {
    status.required = p.count ? offsets[p.count - 1] + counts[p.count - 1] : 0;
    status.error = status.required > p.capacity ? 1 : 0;
    status.unused = 0;
}
kernel void tile_instances(device const ProjectedSplat* splats [[buffer(0)]],
                           device const ulong* offsets [[buffer(1)]],
                           device const RasterStatus& status [[buffer(2)]],
                           device ulong* keys [[buffer(3)]],
                           device uint* indices [[buffer(4)]],
                           constant RasterParameters& p [[buffer(5)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= p.count || status.error) return;
    const auto s = splats[i];
    const uint4 bounds = clipped_bounds(s,p);
    if (bounds.z <= bounds.x || bounds.w <= bounds.y) return;
    ulong at = offsets[i];
    for (uint y = bounds.y / 16; y < (bounds.w + 15) / 16; ++y)
        for (uint x = bounds.x / 16; x < (bounds.z + 15) / 16; ++x) {
            keys[at] = (ulong(y * p.columns + x) << 32) | as_type<uint>(s.mean_depth.z);
            indices[at++] = i;
        }
}

// Stable LSD radix sort adapted from training/kernels/metal/fast_raster.metal.
// Full float32 positive depth is retained; equal depths preserve source order.
struct SortParameters { uint blocks, shift; };
kernel void tile_histogram(device const ulong* keys [[buffer(0)]],
                           device ulong* histogram [[buffer(1)]],
                           device const RasterStatus& status [[buffer(2)]],
                           constant SortParameters& p [[buffer(3)]],
                           uint group [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup atomic_uint counts[8 * 256];
    for (uint s = 0; s < 8; ++s)
        atomic_store_explicit(&counts[s * 256 + lane], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ulong n = status.error ? 0 : status.required;
    for (uint j = 0; j < 8; ++j) {
        const ulong i = ulong(group) * 2048 + j * 256 + lane;
        if (i < n)
            atomic_fetch_add_explicit(&counts[sg * 256 + ((keys[i] >> p.shift) & 255)],
                                       1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint total = 0;
    for (uint s = 0; s < 8; ++s)
        total += atomic_load_explicit(&counts[s * 256 + lane], memory_order_relaxed);
    histogram[lane * p.blocks + group] = total;
}
kernel void tile_scatter(device const ulong* keys_in [[buffer(0)]],
                         device const uint* values_in [[buffer(1)]],
                         device ulong* keys_out [[buffer(2)]],
                         device uint* values_out [[buffer(3)]],
                         device const ulong* histogram [[buffer(4)]],
                         device const RasterStatus& status [[buffer(5)]],
                         constant SortParameters& p [[buffer(6)]],
                         uint group [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_threadgroup]],
                         uint sl [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
    const ulong n = status.error ? 0 : status.required;
    const ulong base = ulong(group) * 2048;
    if (base >= n) return; // Uniform for the entire group.
    threadgroup uint digit_base[256];
    threadgroup uint offsets[8 * 256];
    digit_base[lane] = uint(histogram[lane * p.blocks + group]);
    const uint below = (1u << sl) - 1;
    for (ulong chunk = base; chunk < min(base + 2048, n); chunk += 256) {
        for (uint s = 0; s < 8; ++s) offsets[s * 256 + lane] = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ulong i = chunk + lane;
        const bool valid = i < n;
        const ulong key = valid ? keys_in[i] : 0;
        const uint digit = uint((key >> p.shift) & 255);
        uint peers = uint(static_cast<simd_vote::vote_t>(simd_ballot(valid)));
        for (uint b = 0; b < 8; ++b) {
            const bool bit = ((digit >> b) & 1) != 0;
            const uint vote = uint(static_cast<simd_vote::vote_t>(simd_ballot(bit)));
            peers &= bit ? vote : ~vote;
        }
        const uint rank = popcount(peers & below);
        if (valid && rank == 0) offsets[sg * 256 + digit] = popcount(peers);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint running = digit_base[lane];
        for (uint s = 0; s < 8; ++s) {
            const uint count = offsets[s * 256 + lane];
            offsets[s * 256 + lane] = running;
            running += count;
        }
        digit_base[lane] = running;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (valid) {
            const uint destination = offsets[sg * 256 + digit] + rank;
            keys_out[destination] = key;
            values_out[destination] = values_in[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
kernel void tile_ranges(device const ulong* keys [[buffer(0)]],
                        device uint* ranges [[buffer(1)]],
                        device const RasterStatus& status [[buffer(2)]],
                        uint i [[thread_position_in_grid]]) {
    if (status.error || i >= status.required) return;
    const uint tile = uint(keys[i] >> 32);
    if (i == 0) ranges[tile * 2] = 0;
    else {
        const uint previous = uint(keys[i - 1] >> 32);
        if (tile != previous) {
            ranges[previous * 2 + 1] = i;
            ranges[tile * 2] = i;
        }
    }
    if (ulong(i) + 1 == status.required) ranges[tile * 2 + 1] = i + 1;
}

kernel void tile_blend(device const ProjectedSplat* splats [[buffer(0)]],
                       device const uint* indices [[buffer(1)]],
                       device const uint* ranges [[buffer(2)]],
                       device const RasterStatus& status [[buffer(3)]],
                       constant RasterParameters& p [[buffer(4)]],
                       texture2d<float, access::write> color [[texture(0)]],
                       texture2d<float, access::write> depth [[texture(1)]],
                       texture2d<uint, access::write> pick [[texture(2)]],
                       uint tile [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float4 means[256], conics[256], colors[256];
    threadgroup uint ids[256], finished[8];
    const uint2 pixel = uint2((tile % p.columns) * 16 + lane % 16,
                              (tile / p.columns) * 16 + lane / 16);
    const bool valid = pixel.x < p.width && pixel.y < p.height;
    bool done = !valid;
    float transmittance = 1, weighted_depth = 0, nearest = 0, median = 1e10f;
    float3 rgb = 0;
    uint picked = 0xffffffff;
    const uint begin = status.error ? 0 : ranges[2 * tile];
    const uint end = status.error ? 0 : ranges[2 * tile + 1];
    for (ulong batch = begin; batch < end; batch += 256) {
        const bool all_done = simd_all(done);
        if ((lane & 31) == 0) finished[sg] = all_done;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool stop = true;
        for (uint s = 0; s < 8; ++s) stop = stop && finished[s];
        if (stop) break;
        const uint count = uint(min(ulong(256), ulong(end) - batch));
        if (lane < count) {
            const uint id = indices[batch + lane];
            const auto splat = splats[id];
            ids[lane] = id;
            means[lane] = splat.mean_depth;
            conics[lane] = splat.conic_opacity;
            colors[lane] = splat.color;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!done) for (uint j = 0; j < count; ++j) {
            const float2 d = float2(pixel) + .5f - means[j].xy;
            const float4 c = conics[j];
            const float q = c.x*d.x*d.x + 2*c.y*d.x*d.y + c.z*d.y*d.y;
            if (q < 0 || !isfinite(q)) continue;
            float alpha;
            if (p.mode == 1) alpha = dot(d,d) <= means[j].w*means[j].w ? c.w : 0;
            else if (p.mode == 2) alpha = q <= 9 ? c.w : 0;
            else alpha = c.w * exp(-.5f * q);
            alpha = min(alpha, .999f);
            if (alpha < 1.f/255) continue;
            const float weight = alpha * transmittance;
            if (picked == 0xffffffff) { picked = ids[j]; nearest = means[j].z; }
            rgb += colors[j].xyz * weight;
            weighted_depth += means[j].z * weight;
            const float next_transmittance = transmittance * (1 - alpha);
            if (transmittance >= .5f && next_transmittance < .5f) median = means[j].z;
            transmittance = next_transmittance;
            if (transmittance < 1e-4f) { done = true; break; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (valid) {
        const float alpha = 1 - transmittance;
        // Premultiplied RGBA. Background alpha participates in composition.
        color.write(float4(rgb + p.background.rgb*p.background.a*transmittance,
                            alpha + p.background.a*transmittance), pixel);
        // weighted depth, accumulated alpha, first contributor depth.
        depth.write(float4(weighted_depth, alpha, nearest, median), pixel);
        pick.write(uint4(picked), pixel);
    }
}
