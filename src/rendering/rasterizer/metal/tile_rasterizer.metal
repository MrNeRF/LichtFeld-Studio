// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include <metal_stdlib>
using namespace metal;
@LFS_METAL_OVERLAY@

struct ProjectedSplat {
    float4 mean_depth, conic_opacity, color;
    uint4 bounds;
};
struct GutSplat { float4 inverse0, inverse1, inverse2, mean_opacity; };
struct RasterParameters {
    uint count, width, height, columns;
    uint tiles, capacity, mode, unused;
    float4 background;
    float4 render_origin;
    float4 intrinsics, clip;
    uint4 camera;
    float4 panorama;
    uint4 mask_limits;
};
struct RasterStatus { ulong required; uint error, unused; };
uint4 clipped_bounds(ProjectedSplat s, constant RasterParameters& p) {
    if (!all(isfinite(s.mean_depth)) || s.mean_depth.z <= 0 ||
        !all(isfinite(s.conic_opacity)) || !all(isfinite(s.color))) return uint4(0);
    if(p.mode==3u && p.camera.z==2u) {
        const uint span=p.columns*16u;
        if(s.bounds.x>=span || s.bounds.z<s.bounds.x || s.bounds.z-s.bounds.x>span ||
           (s.bounds.x%16u) || (s.bounds.z%16u))return uint4(0);
        return uint4(s.bounds.x,min(s.bounds.y,p.height),s.bounds.z,min(s.bounds.w,p.height));
    }
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
                        constant uint& primitive_counts [[buffer(4)]],
                        uint group [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup ulong group_sums[8];
    const uint i = group * 256 + lane;
    const ulong value = i < count ? input[i] : 0;
    // MSL 2.4 SIMD reductions do not accept ulong. Sum 16-bit limbs with
    // uint32 operations (32 lanes cannot overflow), then carry in ulong.
    ulong prefix = 0, simd_total = 0;
    if (primitive_counts) {
        // Width/height are bounded by 16384: a source covers at most
        // 1024*1024 bins. A 32-lane sum fits uint32 without truncation;
        // group totals and every subsequent hierarchy level stay uint64.
        prefix = ulong(simd_prefix_exclusive_sum(uint(value)));
        simd_total = ulong(simd_sum(uint(value)));
    } else for (uint shift = 0; shift < 64; shift += 16) {
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
// Histogram totals cannot exceed the admitted uint32 instance capacity. Keep
// the pre-admission count scan above in uint64 so overflow reporting stays exact.
// Scratch remains reserved for the largest frame, but both its digit stride and
// recursive scan extents follow the live instance count without a CPU readback.
uint live_sort_blocks(device const RasterStatus& status) {
    return max(1u, uint(((status.error ? 0ul : status.required) + 2047ul) / 2048ul));
}
uint histogram_scan_count(device const RasterStatus& status, uint level) {
    uint count = live_sort_blocks(status) * 256u;
    for (uint i = 0; i < level; ++i) count = (count + 255u) / 256u;
    return count;
}
kernel void scan32_blocks(device const uint* input [[buffer(0)]],
                          device uint* output [[buffer(1)]],
                          device uint* sums [[buffer(2)]],
                          constant uint& level [[buffer(3)]],
                          device const RasterStatus& status [[buffer(4)]],
                          uint group [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint group_sums[8];
    const uint count = histogram_scan_count(status, level);
    const uint i=group*256+lane;
    const uint value=i<count?input[i]:0;
    uint prefix=simd_prefix_exclusive_sum(value);
    const uint total=simd_sum(value);
    if((lane&31)==0)group_sums[sg]=total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=0;s<sg;++s)prefix+=group_sums[s];
    if(i<count)output[i]=prefix;
    if(lane==255)sums[group]=prefix+value;
}
kernel void scan32_add(device uint* output [[buffer(0)]],
                       device const uint* offsets [[buffer(1)]],
                       constant uint& level [[buffer(2)]],
                       device const RasterStatus& status [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    const uint count = histogram_scan_count(status, level);
    if(i<count)output[i]+=offsets[i/256];
}
kernel void tile_status(device const ulong* counts [[buffer(0)]],
                        device const ulong* offsets [[buffer(1)]],
                        device RasterStatus& status [[buffer(2)]],
                        constant RasterParameters& p [[buffer(3)]],
                        device uint* dispatch_args [[buffer(4)]]) {
    status.required = p.count ? offsets[p.count - 1] + counts[p.count - 1] : 0;
    status.error = status.required > p.capacity ? 1 : 0;
    status.unused = 0;
    const ulong live=status.error?0:status.required;
    dispatch_args[0]=max(1u,uint((live+2047)/2048));
    dispatch_args[1]=1; dispatch_args[2]=1;
    dispatch_args[3]=max(1u,uint((live+255)/256));
    dispatch_args[4]=1; dispatch_args[5]=1;
    uint histogram_count = live_sort_blocks(status) * 256u;
    for (uint level = 0; level < 4; ++level) {
        const uint groups = (histogram_count + 255u) / 256u;
        dispatch_args[6 + level * 3] = groups;
        dispatch_args[7 + level * 3] = 1;
        dispatch_args[8 + level * 3] = 1;
        histogram_count = groups;
    }
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
            const uint column=p.mode==3u && p.camera.z==2u?x%p.columns:x;
            keys[at] = (ulong(y * p.columns + column) << 32) | as_type<uint>(s.color.w);
            indices[at++] = i;
        }
}

// Stable LSD radix sort adapted from training/kernels/metal/fast_raster.metal.
// Full float32 positive depth is retained; equal depths preserve source order.
struct SortParameters { uint blocks, shift; };
kernel void tile_histogram(device const ulong* keys [[buffer(0)]],
                           device uint* histogram [[buffer(1)]],
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
        const bool valid = i < n;
        const uint digit = valid ? uint((keys[i] >> p.shift) & 255ul) : 0u;
        uint peers = uint(static_cast<simd_vote::vote_t>(simd_ballot(valid)));
        for (uint bit = 0; bit < 8; ++bit) {
            const bool set = ((digit >> bit) & 1u) != 0;
            const uint vote = uint(static_cast<simd_vote::vote_t>(simd_ballot(set)));
            peers &= set ? vote : ~vote;
        }
        // Coalesce equal radix digits in each SIMD, avoiding 32 serialized
        // atomics for the common high depth/tile bytes. Integer counts are exact.
        if (valid && (peers & ((1u << (lane & 31u)) - 1u)) == 0u)
            atomic_fetch_add_explicit(&counts[sg * 256 + digit], popcount(peers), memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint total = 0;
    for (uint s = 0; s < 8; ++s)
        total += atomic_load_explicit(&counts[s * 256 + lane], memory_order_relaxed);
    histogram[lane * live_sort_blocks(status) + group] = total;
}
kernel void tile_scatter(device const ulong* keys_in [[buffer(0)]],
                         device const uint* values_in [[buffer(1)]],
                         device ulong* keys_out [[buffer(2)]],
                         device uint* values_out [[buffer(3)]],
                         device const uint* histogram [[buffer(4)]],
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
    digit_base[lane] = histogram[lane * live_sort_blocks(status) + group];
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

// Specialize the blend body to retain only the active renderer features.
constant uint kRasterMode [[function_constant(0)]];
constant uint kRasterFlags [[function_constant(1)]];
kernel void tile_blend(device const ProjectedSplat* splats [[buffer(0)]],
                       device const uint* indices [[buffer(1)]],
                       device const uint* ranges [[buffer(2)]],
                       device const RasterStatus& status [[buffer(3)]],
                       constant RasterParameters& p [[buffer(4)]],
                       device const float4* overlay_params [[buffer(5)]],
                       device const uint* overlay_flags [[buffer(6)]],
                       device const uchar* selection [[buffer(7)]],
                       device const uchar* preview [[buffer(8)]],
                       device const float4* selection_colors [[buffer(9)]],
                       device const GutSplat* gut [[buffer(10)]],
                       device const uint* logical_ids [[buffer(11)]],
                       constant uint& logical_count [[buffer(12)]],
                       texture2d<float, access::write> color [[texture(0)]],
                       texture2d<float, access::write> depth [[texture(1)]],
                       texture2d<uint, access::write> pick [[texture(2)]],
                       uint group [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
    // Keep the stable 16x16 bin/sort contract. Four independent 8x8 blend
    // groups traverse the same ordered list, each with its own saturation vote.
    // This reduces shared storage and avoids waiting for unrelated pixels.
    // Ordinary GS uses one SIMD32 group per 8x4 pixel region. Retain
    // the same 64-source batch boundary so half composition stays unchanged.
    const bool single_simd=(kRasterFlags&128u)!=0;
    const uint subtiles=single_simd?8u:4u, pixel_height=single_simd?4u:8u;
    const uint tile=group/subtiles, subtile=group%subtiles;
    threadgroup float4 means[64], conics[64], colors[64];
    threadgroup GutSplat geometry[64];
    threadgroup half4 macro_chol[64];
    threadgroup half2 macro_center[64];
    threadgroup uint ids[64], finished[2], active_counts[2];
    const uint2 tile_origin = uint2((tile % p.columns)*16+(subtile%2)*8,
                                    (tile / p.columns)*16+(subtile/2)*pixel_height);
    const uint2 pixel = tile_origin+uint2(lane%8,lane/8);
    // Ordinary GS bounds are suitable only without extended overlays. GUT
    // uses a separate conservative 3D-support sphere; portal and panorama
    // retain their full parent list.
    const bool compact_gs=kRasterMode==0u && p.camera.z!=2u && !(kRasterFlags&16u) &&
        (!(kRasterFlags&1u) || (!overlay_enabled(overlay_params[22].x) && !overlay_enabled(overlay_params[22].y)));
    const bool compact_gut=kRasterMode==3u && p.camera.z!=2u && !(kRasterFlags&4u);
    const bool compact_candidates=compact_gs || compact_gut;
    const float2 ray_min=(float2(tile_origin)+.5f-p.intrinsics.zw)/p.intrinsics.xy;
    const float2 ray_max=(float2(tile_origin)+7.5f-p.intrinsics.zw)/p.intrinsics.xy;
    const float4 plane_lengths=p.camera.z==1u?float4(1):sqrt(1.f+float4(ray_min,ray_max)*float4(ray_min,ray_max));
    const bool batch_half=(kRasterMode==0u && (kRasterFlags&(4u|64u))) ||
        (kRasterMode!=3u && !(kRasterFlags&16u) && (kRasterFlags&1u) && overlay_enabled(overlay_params[22].y));
    const float2 first_macro=floor((float2(tile_origin)+p.render_origin.xy)/overlay_macro_extent);
    const bool uniform_macro=all(first_macro==floor((float2(tile_origin)+7.f+p.render_origin.xy)/overlay_macro_extent));
    const float2 batch_macro_origin=first_macro*overlay_macro_extent;
    // Pixel coordinates and crop-relative macro identity are invariant across
    // every batch. Keep the same half rounding outside the contributor loop.
    const float2 pixel_macro_origin=uniform_macro?batch_macro_origin:floor((float2(pixel)+p.render_origin.xy)/overlay_macro_extent)*overlay_macro_extent;
    const half2 pixel_macro_coord=half2((float2(pixel)+p.render_origin.xy-pixel_macro_origin)/overlay_tile_extent);
    const bool valid = pixel.x < p.width && pixel.y < p.height;
    bool done = !valid;
    float transmittance = 1, weighted_depth = 0, nearest = 0, median = 1e10f;
    float valid_depth_weight=0;
    float3 rgb = 0;
    // Compress each cooperative blend batch, then compose in FP32. A single
    // half accumulator for the whole depth stream loses weak contributions
    // behind a foreground layer once its RGB exceeds their half-ULP weight.
    float3 composed_rgb=0;
    float composed_transmittance=1;
    uint picked = 0xffffffff;
    const uint begin = status.error ? 0 : ranges[2 * tile];
    const uint end = status.error ? 0 : ranges[2 * tile + 1];
    // A camera ray is invariant across every Gaussian and tile batch. Compute
    // spherical trigonometry once per live pixel, never inside the hot loop.
    float3 gut_origin=0,gut_direction=float3(0,0,1);
    if(kRasterMode==3u && valid && end>begin) {
        const bool orthographic=p.camera.z==1u;
        if(p.camera.z==2u) {
            const float2 angle=((float2(pixel)+.5f+p.panorama.zw)/p.panorama.xy-.5f)*float2(2.f*M_PI_F,M_PI_F);
            const float elevation_cos=cos(angle.y);
            gut_direction=float3(sin(angle.x)*elevation_cos,sin(angle.y),cos(angle.x)*elevation_cos);
        } else {
            const float2 xy=(float2(pixel)+.5f-p.intrinsics.zw)/p.intrinsics.xy;
            gut_origin=orthographic?float3(xy,0):float3(0);
            gut_direction=orthographic?float3(0,0,1):float3(xy,1);
        }
    }
    for (ulong batch = begin; batch < end; batch += 64) {
        const bool all_done = simd_all(done);
        if(single_simd) {
            if(all_done)break;
        } else {
            if ((lane & 31) == 0) finished[sg] = all_done;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(finished[0] && finished[1])break;
        }
        const uint source_count=uint(min(ulong(64),ulong(end)-batch));
        uint count=single_simd && compact_candidates?0u:source_count;
        const uint loaders=single_simd?32u:64u;
        for(uint source_offset=0;source_offset<64u;source_offset+=loaders) {
        const uint source_lane=source_offset+lane;
        uint destination=source_lane, id=0;
        ProjectedSplat splat{};
        GutSplat g{};
        bool active=source_lane<source_count;
        if (active) {
            id=indices[batch+source_lane];
            splat=splats[id];
            if(kRasterMode==3u)g=gut[id];
        }
        if (compact_candidates) {
            // Integer support with a one-pixel margin includes rounding of the
            // full-float projection and the optional half portal footprint.
            const int2 minimum=int2(tile_origin)-1, maximum=int2(tile_origin)+int2(9,int(pixel_height)+1);
            if(compact_gs)active=active && all(int2(splat.bounds.xy)<maximum) && all(int2(splat.bounds.zw)>minimum);
            if(compact_gut && active && g.inverse0.w>0.f) {
                // Reject only if the complete alpha-support sphere is outside
                // one of the four pixel-ray frustum planes. Unlike a projected
                // UT rectangle, this remains conservative for the actual 3D ray.
                const float3 center=g.mean_opacity.xyz;
                const float z=p.camera.z==1u?1.f:center.z;
                const float4 edges=float4(ray_min*z,ray_max*z);
                const float4 distances=float4(center.xy-edges.xy,edges.zw-center.xy);
                const float4 rounding=1e-6f*(float4(abs(center.xy),abs(center.xy))+abs(edges));
                active=all(distances>=-g.inverse0.w*plane_lengths-rounding);
            }
            const uint live=active?1u:0u;
            destination=simd_prefix_exclusive_sum(live);
            const uint total=simd_sum(live);
            if(single_simd) {
                destination+=count;
                count+=total;
            } else {
                if ((lane&31)==0) active_counts[sg]=total;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (sg) destination+=active_counts[0];
                count=active_counts[0]+active_counts[1];
            }
        }
        if (active) {
            ids[destination] = id;
            if (kRasterMode==3u) {
                if (p.camera.z!=1u) {
                    // Pinhole/spherical rays start at camera origin. The
                    // inverse-Gaussian origin is invariant across all pixels:
                    // cache it here, reusing xyz of the shared mean payload.
                    const float3 delta=-g.mean_opacity.xyz;
                    g.mean_opacity.xyz=float3(dot(g.inverse0.xyz,delta),dot(g.inverse1.xyz,delta),dot(g.inverse2.xyz,delta));
                }
                geometry[destination]=g;
            }
            means[destination] = splat.mean_depth;
            conics[destination] = splat.conic_opacity;
            colors[destination] = splat.color;
            if(batch_half) {
                const float4 c=splat.conic_opacity;
                const float l00=sqrt(max(c.x,1e-12f)),l01=c.y/l00,l11=sqrt(max(c.z-l01*l01,0.f));
                macro_chol[destination]=half4(float4(l00*overlay_tile_extent.x*.849321800288019f,l01*overlay_tile_extent.y*.849321800288019f,
                    l11*overlay_tile_extent.y*.849321800288019f,max(4.f,log(c.w*510.f))*1.4426950408889634f));
                macro_center[destination]=half2((splat.mean_depth.xy+p.render_origin.xy-batch_macro_origin)/overlay_tile_extent);
            }
            colors[destination].xyz = clamp(colors[destination].xyz, 0.f, 4.f);
            if(kRasterFlags&64u)colors[destination].xyz=float3(half3(colors[destination].xyz));
            if (!(kRasterFlags&16u) && kRasterMode!=1u) {
                const float opacity=kRasterMode==3u?geometry[destination].mean_opacity.w:splat.conic_opacity.w;
                // The radial key is no longer needed after sorting. Cache the
                // alpha support once per Gaussian/tile, then reject before exp.
                // The small margin retains FP32 threshold-boundary contributors.
                colors[destination].w=max(4.f,log(max(opacity,.5f/255.f)*510.f))+1e-4f;
            }
            if((kRasterFlags&16u) && kRasterMode!=1u) {
                const float opacity=kRasterMode==3u?geometry[destination].mean_opacity.w:splat.conic_opacity.w;
                const float adjusted=sqrt(8.f)+.7f*(min(opacity,5.f)-1.f);
                const float power=opacity>1.f?.5f*adjusted*adjusted:max(4.f,log(max(opacity,.5f/255.f)*510.f));
                // Neither radius nor radial sort metric is used by this hot
                // blend path. Reuse shared slots for cutoff and density, with
                // one exp per Gaussian/tile and no new buffers or shared memory.
                means[destination].w=exp(-power);
                colors[destination].w=opacity>1.f?exp((opacity*opacity-1.f)/2.718281828459045f):0.f;
            }
        }
        }
        if(single_simd)simdgroup_barrier(mem_flags::mem_threadgroup);
        else threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!done) for (uint j = 0; j < count; ++j) {
            // Studio's CUDA/Vulkan convention samples at integer pixel coordinates.
            const uint logical=(kRasterFlags&8u)?logical_ids[ids[j]]:ids[j];
            if((kRasterFlags&8u) && logical>=logical_count)continue;
            const float2 d = float2(pixel) - means[j].xy;
            const float4 c = conics[j];
            const bool half_footprint=(kRasterMode==0u && (kRasterFlags&(4u|64u))) || (kRasterMode!=3u && !(kRasterFlags&16u) && (kRasterFlags&1u) && overlay_enabled(overlay_params[22].y) && !(overlay_flags[ids[j]]&2u));
            // The HiGS profile evaluates the half Cholesky footprint below.
            // Its reference does not evaluate a second full-float conic first.
            const float q = half_footprint ? 0.f : c.x*d.x*d.x + 2*c.y*d.x*d.y + c.z*d.y*d.y;
            if (q < 0 || !isfinite(q)) continue;
            if (!(kRasterFlags&16u) && kRasterMode==0u && !half_footprint && .5f*q>colors[j].w) continue;
            float alpha;
            float splat_depth=means[j].z;
            if (kRasterMode == 1) alpha = dot(d,d) <= means[j].w*means[j].w ? c.w : 0;
            else if (kRasterMode == 2) alpha = q <= 9 ? c.w : 0;
            else if(kRasterMode==3u) {
                const auto g=geometry[j];
                if(kRasterFlags&4u) {
                    const float2 axis=float2(g.inverse0.w,g.inverse1.w);
                    if(abs(dot(d,axis))>means[j].w || abs(dot(d,float2(axis.y,-axis.x)))>g.inverse2.w)continue;
                }
                float3 local_origin=g.mean_opacity.xyz;
                if (p.camera.z==1u) {
                    const float3 delta=gut_origin-g.mean_opacity.xyz;
                    local_origin=float3(dot(g.inverse0.xyz,delta),dot(g.inverse1.xyz,delta),dot(g.inverse2.xyz,delta));
                }
                const float3 local_direction=float3(dot(g.inverse0.xyz,gut_direction),dot(g.inverse1.xyz,gut_direction),dot(g.inverse2.xyz,gut_direction));
                const float denom=dot(local_direction,local_direction);
                if(!isfinite(denom)||denom<=1e-12f)continue;
                const float3 distance=cross(local_direction*rsqrt(denom),local_origin);
                const float power=.5f*dot(distance,distance);
                if (!(kRasterFlags&16u) && power>colors[j].w) continue;
                alpha=g.mean_opacity.w*exp(-power);
                const float t=-dot(local_direction,local_origin)/denom;
                const float z=gut_origin.z+t*gut_direction.z;
                splat_depth=t>0 && z>p.clip.x && isfinite(z)?z:1e10f;
            } else if(half_footprint) {
                // The macro-relative FP16 conic can admit a boundary point
                // that the full-float conic rejects. Evaluate its support below
                // directly instead of computing and then overwriting a float exp.
                alpha=0.f;
            } else if(kRasterFlags&4u) {
                const float edge=.01831563888873418f;
                const float value=q>8.f?0.f:exp(-.5f*q);
                alpha=c.w*max(0.f,(value-edge)/(1.f-edge));
            } else alpha = c.w * exp(-.5f * q);
            if((kRasterFlags&16u) && kRasterMode!=1u) {
                const float opacity=kRasterMode==3u?geometry[j].mean_opacity.w:c.w;
                const float value=clamp(alpha/max(opacity,1e-8f),0.f,1.f);
                if(value<means[j].w)continue;
                if(colors[j].w>0)alpha=1.f-pow(max(0.f,1.f-value),colors[j].w);
            }
            alpha = min(alpha, .999f);
            if(half_footprint){
                const half2 center=uniform_macro?macro_center[j]:half2((means[j].xy+p.render_origin.xy-pixel_macro_origin)/overlay_tile_extent);
                const half4 chol=macro_chol[j];
                const half2 delta=pixel_macro_coord-center;
                // The macro reference rounds the X product before adding Y.
                const half dx=half(chol.x*delta.x);
                const half u=half(dx+chol.y*delta.y),v=half(chol.z*delta.y);
                const half power=u*u+v*v;
                if(!isfinite(power) || power<0 || power>chol.w)continue;
                // Match the portal's macro-relative FP16 footprint. The
                // final native composition remains FP32; this is display-profile math.
                half value=exp2(-power);
                if(kRasterFlags&4u) {
                    const half edge=half(.01831563888873418f);
                    value=max(half(0),(value-edge)/(half(1)-edge));
                }
                alpha=float(min(half(c.w)*value,half(.999f)));
            }
            const float minimum_alpha=(kRasterFlags&4u)?1.f/255.f:.5f/255.f;
            if (alpha < (half_footprint?float(half(minimum_alpha)):minimum_alpha)) continue;
            // The reference gates overlays again in FP32 after its half
            // body threshold. A half value equal to the rounded threshold
            // can lie below .5/255 and must not become an opaque ring.
            if ((kRasterFlags&1u) && alpha < .5f/255.f) continue;
            float3 radiance=colors[j].xyz;
            if(kRasterFlags&1u){
                const uint flags=overlay_flags[ids[j]];
                // Pixel-sized overlays use the same macro-relative half position
                // as the desktop reference. Keep Gaussian blending in FP32.
                // KEEP IN SYNC with Vulkan config.slang: tile 8x8, macro 8x4 tiles.
                const float2 macro_origin=floor((float2(pixel)+p.render_origin.xy)/overlay_macro_extent)*overlay_macro_extent;
                // Vulkan's 3DGUT shared-struct path retains full float centers;
                // only the 3DGS macro-relative path compresses them to half.
                const float2 overlay_center=(kRasterMode==3u || (kRasterFlags&16u))?means[j].xy:
                    float2(half2((means[j].xy+p.render_origin.xy-macro_origin)/overlay_tile_extent))*overlay_tile_extent+macro_origin-p.render_origin.xy;
                const uint status=overlay_selection(overlay_params,logical,flags,overlay_center+(p.camera.z==2u?p.panorama.zw:float2(0)),selection,preview,p.mask_limits.xy);
                const bool selectable=(flags&2u)==0;
                if(overlay_enabled(overlay_params[22].x)&&selectable){
                    // Rings use the full conic at the same reconstructed
                    // center as the reference, independently of the body
                    // footprint. A half body deliberately leaves q unused.
                    const float2 ring_delta=float2(pixel)-overlay_center;
                    const float sigma_over_2=.5f*(c.x*ring_delta.x*ring_delta.x+c.z*ring_delta.y*ring_delta.y)+c.y*ring_delta.x*ring_delta.y;
                    const float gaussian=exp(-sigma_over_2);
                    const float boundary=(.5f/255.f)/max(c.w,1e-8f);
                    const float width=overlay_params[21].w*10;
                    if(gaussian<boundary*(1+width)&&gaussian>boundary*(1-width)){
                        if(status)radiance=overlay_target(status,selection_colors);
                        alpha=.8f;
                    }
                }
                if(overlay_enabled(overlay_params[22].y)&&selectable){
                    // Desktop marker mode contributes dots only, without the
                    // Gaussian body outside the marker's circular support.
                    const float2 marker_delta=float2(pixel)-overlay_center;
                    // Match the desktop's rounded distance comparisons at
                    // the hard marker boundaries, rather than squared radii.
                    const float marker_distance=length(marker_delta);
                    if(marker_distance>2.5f)continue;
                    rgb=overlay_target(status,selection_colors)*(marker_distance>1.5f?.4f:1.f);
                    if(((kRasterFlags&64u)?composed_transmittance*transmittance:transmittance)>.5f)median=splat_depth;
                    transmittance=0; done=true; break;
                }
                if(status&128u)radiance=mix(radiance,overlay_target(status,selection_colors),.9f);
                else if(status&127u)radiance=mix(radiance,selection_colors[status&127u].xyz,.8f);
                if((flags&4u)&&overlay_params[20].w>0)radiance=mix(radiance,float3(1,.95f,.6f),overlay_params[20].w*.5f);
            }
            const float total_transmittance=(kRasterFlags&64u)?composed_transmittance*transmittance:transmittance;
            const float weight = alpha * total_transmittance;
            if (picked == 0xffffffff) { picked = logical; nearest = splat_depth; }
            // Match expected_far in the reference: invalid/too-distant
            // GUT depths affect transmittance but never the depth average.
            if(kRasterFlags&2u) {
                if(splat_depth<=p.clip.y) {
                    weighted_depth += splat_depth * weight;
                    valid_depth_weight += weight;
                }
            } else weighted_depth += splat_depth * weight;
            const float next_transmittance = transmittance * (1 - alpha);
            const float next_total_transmittance=(kRasterFlags&64u)?composed_transmittance*next_transmittance:next_transmittance;
            if (total_transmittance > .5f && next_total_transmittance <= .5f) median = splat_depth;
            // The existing desktop GUT legacy chain records depth before
            // dropping a saturating splat's color and transmittance update.
            // Keep this explicit: native analytic and Spark math includes it.
            if ((kRasterFlags & 32u) && next_transmittance < 1e-4f) {
                done = true;
                break;
            }
            if(kRasterFlags&64u) {
                if(kRasterFlags&1u) {
                    rgb=float3(half3(rgb+radiance*(alpha*transmittance)));
                    transmittance=float(half(next_transmittance));
                } else {
                    const half half_weight=half(half(alpha)*half(transmittance));
                    rgb=float3(half3(half3(rgb)+half3(radiance)*half_weight));
                    const half remaining=half(half(1)-half(alpha));
                    transmittance=float(half(half(transmittance)*remaining));
                }
            } else {
                rgb += radiance * weight;
                transmittance = next_transmittance;
            }
            if (((kRasterFlags&64u)?composed_transmittance*transmittance:transmittance) < 1e-4f) { done = true; break; }
        }
        if(kRasterFlags&64u) {
            composed_rgb+=rgb*composed_transmittance;
            composed_transmittance*=transmittance;
            rgb=0;transmittance=1;
        }
        if(single_simd)simdgroup_barrier(mem_flags::mem_threadgroup);
        else threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (valid) {
        if(kRasterFlags&64u){rgb=composed_rgb;transmittance=composed_transmittance;}
        const float alpha = 1 - transmittance;
        // Premultiplied RGBA. Background alpha participates in composition.
        color.write(float4(rgb + p.background.rgb*p.background.a*transmittance,
                            alpha + p.background.a*transmittance), pixel);
        // weighted depth, accumulated alpha, first contributor depth.
        depth.write(float4(weighted_depth, alpha, (kRasterFlags&2u)?valid_depth_weight:nearest, median), pixel);
        pick.write(uint4(picked), pixel);
    }
}
