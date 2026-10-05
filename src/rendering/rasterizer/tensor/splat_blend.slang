// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
// Front-to-back tile blending of the sorted splat lists into premultiplied
// color, depth statistics and the first contributor. Ported from the native
// Metal viewer's tile_blend and tile_depth_compose. The raster mode and
// feature flags are compile-time, like the native function constants:
//   LFS_BLEND_MODE   0 Gaussian, 1 points, 2 discs, 3 3DGUT
//   LFS_BLEND_FLAGS  (optional; otherwise RasterParameters.unused) 1 overlay, 2 expected depth, 4 portal edge, 8 logical IDs,
//                    16 Spark, 32 GUT saturation stop, 64 half composition,
//                    128 one 32-wide subgroup per 8x4 pixels, 512 depth batches,
//                    1024 depth-batch prefix, 2048 separate median,
//                    8192 panorama wrap, 16384 float footprint
#include "../vulkan/shader/src/slang/config.slang"
#include "splat_overlay.slang"
#include "splat_types.slang"

#ifndef LFS_BLEND_MODE
#define LFS_BLEND_MODE 0
#endif
static const uint kRasterMode = LFS_BLEND_MODE;
#ifdef LFS_BLEND_FLAGS
// Fully specialized: every feature branch folds away, as with the native
// function constants. Generic variants read the flags of the dispatch.
static const uint kRasterFlags = LFS_BLEND_FLAGS;
#define LFS_BLEND_SINGLE_SIMD ((LFS_BLEND_FLAGS & 128) != 0)
#else
static uint kRasterFlags;
#ifndef LFS_BLEND_SINGLE_SIMD
#define LFS_BLEND_SINGLE_SIMD 0
#endif
#endif
static const uint kDepthChunkSize = 576u;
static const float2 overlay_macro_extent = float2(HIGS_TILE_WIDTH * HIGS_MACRO_TILE_WIDTH_TILES, HIGS_TILE_HEIGHT * HIGS_MACRO_TILE_HEIGHT_TILES);
static const float2 overlay_tile_extent = float2(HIGS_TILE_WIDTH, HIGS_TILE_HEIGHT);

struct GutSplat {
    float4 inverse0, inverse1, inverse2, mean_opacity;
};

struct Parameters {
    Ptr<ProjectedSplat, Access.Read> splats;
    Ptr<uint, Access.Read> indices;
    Ptr<uint, Access.Read> ranges;
    Ptr<RasterStatus, Access.ReadWrite> status;
    Ptr<RasterParameters, Access.Read> raster;
    Ptr<float4, Access.Read> overlay_params;
    Ptr<uint, Access.Read> overlay_flags;
    Ptr<uint8_t, Access.Read> selection;
    Ptr<uint8_t, Access.Read> preview;
    Ptr<float4, Access.Read> selection_colors;
    Ptr<GutSplat, Access.Read> gut;
    Ptr<uint, Access.Read> logical_ids;
    Ptr<uint2, Access.Read> depth_jobs;
    Ptr<float4, Access.ReadWrite> partial_color;
    Ptr<float4, Access.ReadWrite> partial_depth;
    Ptr<uint, Access.ReadWrite> partial_pick;
    Ptr<half, Access.ReadWrite> color; // [height][width][4] premultiplied RGBA
    Ptr<float4, Access.ReadWrite> depth; // weighted depth, alpha, first or valid weight, median
    Ptr<uint, Access.ReadWrite> pick;
    uint logical_count;
    uint dispatch_flags; // OR'd into the raster flags: 1024 for the depth-batch prefix pass
};
[[vk::push_constant]] ConstantBuffer<Parameters> p;

uint depth_split_threshold(RasterStatus status, RasterParameters r) {
    return status.required > uint64_t(r.tiles) * 896ul ? 2048u : 16384u;
}

// A positive conic reaches its minimum over a rectangle at its center or at
// the constrained minimum of one edge; subtract a roundoff bound.
float conic_lower_power(float2 d, float3 c) {
    const float3 terms = float3(c.x * d.x * d.x, 2.f * c.y * d.x * d.y, c.z * d.y * d.y);
    return .5f * (terms.x + terms.y + terms.z) - 1e-4f * (1.f + dot(abs(terms), float3(1)));
}
bool conic_intersects_pixels(ProjectedSplat s, uint2 origin, uint2 maximum) {
    const float3 c = s.conic_opacity.xyz;
    if (c.x <= 0.f || c.z <= 0.f || c.x * c.z <= c.y * c.y)
        return true;
    const float2 lo = float2(origin) - s.mean_depth.xy;
    const float2 hi = float2(maximum) - s.mean_depth.xy;
    if (all(lo <= 0.f) && all(hi >= 0.f))
        return true;
    const float2 y = clamp(-c.y * float2(lo.x, hi.x) / c.z, lo.y, hi.y);
    const float2 x = clamp(-c.y * float2(lo.y, hi.y) / c.x, lo.x, hi.x);
    const float lower = min(min(conic_lower_power(float2(lo.x, y.x), c), conic_lower_power(float2(hi.x, y.y), c)),
                            min(conic_lower_power(float2(x.x, lo.y), c), conic_lower_power(float2(x.y, hi.y), c)));
    const float support = max(4.f, log(max(s.conic_opacity.w, .5f / 255.f) * 510.f)) + 1e-4f;
    return !isfinite(lower) || lower <= support;
}

void write_pixel(uint2 pixel, RasterParameters r, float3 rgb, float transmittance, float weighted_depth,
                 float third, float median, uint picked) {
    const uint at = pixel.y * r.width + pixel.x;
    const float alpha = 1 - transmittance;
    // Premultiplied RGBA; the background alpha participates in composition.
    const float4 c = float4(rgb + r.background.rgb * r.background.a * transmittance, alpha + r.background.a * transmittance);
    p.color[at * 4u] = half(c.x);
    p.color[at * 4u + 1u] = half(c.y);
    p.color[at * 4u + 2u] = half(c.z);
    p.color[at * 4u + 3u] = half(c.w);
    p.depth[at] = float4(weighted_depth, alpha, third, median);
    p.pick[at] = picked;
}

#if LFS_BLEND_SINGLE_SIMD
#define LFS_BLEND_THREADS 32
#else
#define LFS_BLEND_THREADS 64
#endif

// Generic variants take their flags from the dispatch; the group width is fixed.
void load_flags(RasterParameters r) {
#ifndef LFS_BLEND_FLAGS
    kRasterFlags = (r.unused & ~128u) | p.dispatch_flags | (LFS_BLEND_SINGLE_SIMD ? 128u : 0u);
#endif
}

groupshared float4 means[64], conics[64], colors[64];
groupshared GutSplat geometry[64];
groupshared half4 macro_chol[64];
groupshared half2 macro_center[64];
groupshared uint ids[64], finished[2], active_counts[2];

// Single-subgroup groups need only a subgroup barrier; Slang has none for
// Metal, and a threadgroup barrier over one subgroup is equivalent.
void blend_barrier() { GroupMemoryBarrierWithGroupSync(); }

[shader("compute")]
[numthreads(LFS_BLEND_THREADS, 1, 1)]
void tile_blend(uint3 group_id : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint group = group_id.x, lane = thread.x, sg = lane / 32u;
    const RasterParameters r = p.raster[0];
    load_flags(r);
    const RasterStatus status = p.status[0];
    const bool single_simd = (kRasterFlags & 128u) != 0u;
    const uint subtiles = single_simd ? 8u : 4u, pixel_height = single_simd ? 4u : 8u;
    const bool prefix = (kRasterFlags & 1024u) != 0u;
    const bool depth_dispatch = (kRasterFlags & 512u) != 0u && !prefix;
    const uint job_slot = group / subtiles;
    if (depth_dispatch && p.depth_jobs[job_slot].x == 0xffffffffu)
        return;
    const uint tile = depth_dispatch ? p.depth_jobs[job_slot].x : job_slot, subtile = group % subtiles;
    const uint slot = prefix ? p.ranges[tile * 2u] / kDepthChunkSize + tile : job_slot;
    const bool depth_batch = (kRasterFlags & 512u) != 0u && status.required <= uint64_t(r.mask_limits.z) &&
                             p.ranges[tile * 2u + 1u] - p.ranges[tile * 2u] > depth_split_threshold(status, r);
    if (lane == 0u && subtile == 0u && !depth_dispatch)
        InterlockedMax(((Ptr<uint, Access.ReadWrite>)p.status)[4], p.ranges[tile * 2u + 1u] - p.ranges[tile * 2u]);
    if (depth_dispatch && (!depth_batch || p.depth_jobs[slot].y == p.ranges[tile * 2u]))
        return;
    const uint2 tile_origin = uint2((tile % r.columns) * 16u + (subtile % 2u) * 8u, (tile / r.columns) * 16u + (subtile / 2u) * pixel_height);
    const uint2 pixel = tile_origin + uint2(lane % 8u, lane / 8u);
    const bool overlays = (kRasterFlags & 1u) != 0u;
    const bool compact_gs = kRasterMode == 0u && r.camera.z != 2u && (kRasterFlags & 16u) == 0u &&
                            (!overlays || (!overlay_enabled(p.overlay_params[22].x) && !overlay_enabled(p.overlay_params[22].y)));
    const bool compact_gut = kRasterMode == 3u && r.camera.z != 2u && (kRasterFlags & 4u) == 0u;
    const bool compact_candidates = compact_gs || compact_gut;
    const float2 ray_min = (float2(tile_origin) + .5f - r.intrinsics.zw) / r.intrinsics.xy;
    const float2 ray_max = (float2(tile_origin) + float2(7.5f, float(pixel_height) - .5f) - r.intrinsics.zw) / r.intrinsics.xy;
    const float4 plane_lengths = r.camera.z == 1u ? float4(1) : sqrt(1.f + float4(ray_min, ray_max) * float4(ray_min, ray_max));
    const bool separate_median = kRasterMode == 0u && (kRasterFlags & (4u | 2048u)) == (4u | 2048u);
    const bool batch_half = ((kRasterFlags & 2048u) == 0u || separate_median) &&
                            ((kRasterMode == 0u && (kRasterFlags & (4u | 64u)) != 0u) ||
                             (kRasterMode != 3u && (kRasterFlags & 16u) == 0u && overlays && overlay_enabled(p.overlay_params[22].y)));
    const float2 first_macro = floor((float2(tile_origin) + r.render_origin.xy) / overlay_macro_extent);
    const bool uniform_macro = all(first_macro == floor((float2(tile_origin) + 7.f + r.render_origin.xy) / overlay_macro_extent));
    const float2 batch_macro_origin = first_macro * overlay_macro_extent;
    const float2 pixel_macro_origin = uniform_macro ? batch_macro_origin : floor((float2(pixel) + r.render_origin.xy) / overlay_macro_extent) * overlay_macro_extent;
    const half2 pixel_macro_coord = half2((float2(pixel) + r.render_origin.xy - pixel_macro_origin) / overlay_tile_extent);
    const bool valid = pixel.x < r.width && pixel.y < r.height;
    bool done = !valid;
    float local_cutoff = 1e-4f;
    if (depth_dispatch) {
        const uint first = p.ranges[tile * 2u] / kDepthChunkSize + tile;
        const float prefix_transmittance = valid ? p.partial_color[first * 256u + subtile * 32u + lane].w : 0.f;
        done = done || prefix_transmittance < 1e-4f;
        // Later chunks only lower incoming T: skip what lies behind saturation.
        local_cutoff = 1e-4f * .99999f / max(prefix_transmittance, 1e-4f);
        if (WaveActiveAllTrue(done))
            return;
    }
    float median_transmittance = 1;
    float transmittance = 1, weighted_depth = 0, nearest = 0, median = 1e10f;
    float valid_depth_weight = 0;
    float3 rgb = 0;
    // Compose each half-precision batch in FP32.
    float3 composed_rgb = 0;
    float composed_transmittance = 1;
    uint picked = 0xffffffffu;
    const uint begin = status.error != 0u ? 0u : depth_batch && !prefix ? p.depth_jobs[slot].y : p.ranges[2u * tile];
    const uint end = status.error != 0u ? 0u : depth_batch ? begin + min(kDepthChunkSize, p.ranges[2u * tile + 1u] - begin) : p.ranges[2u * tile + 1u];
    // The camera ray is invariant across every Gaussian of the pixel.
    float3 gut_origin = 0, gut_direction = float3(0, 0, 1);
    if (kRasterMode == 3u && valid && end > begin) {
        const bool orthographic = r.camera.z == 1u;
        if (r.camera.z == 2u) {
            const float2 angle = ((float2(pixel) + .5f + r.panorama.zw) / r.panorama.xy - .5f) * float2(2.f * M_PI, M_PI);
            const float elevation_cos = cos(angle.y);
            gut_direction = float3(sin(angle.x) * elevation_cos, sin(angle.y), cos(angle.x) * elevation_cos);
        } else {
            const float2 xy = (float2(pixel) + .5f - r.intrinsics.zw) / r.intrinsics.xy;
            gut_origin = orthographic ? float3(xy, 0) : float3(0);
            gut_direction = orthographic ? float3(0, 0, 1) : float3(xy, 1);
        }
    }
    for (uint batch = begin; batch < end; batch += 64u) {
        const bool all_done = WaveActiveAllTrue(done);
        if (single_simd) {
            if (all_done)
                break;
        } else {
            if ((lane & 31u) == 0u)
                finished[sg] = all_done ? 1u : 0u;
            GroupMemoryBarrierWithGroupSync();
            if (finished[0] != 0u && finished[1] != 0u)
                break;
        }
        uint2 live_min = tile_origin, live_max = tile_origin + uint2(7, pixel_height - 1u);
        if (single_simd && compact_gs && WaveActiveAnyTrue(done)) {
            live_min = WaveActiveMin(done ? uint2(0xffffffffu) : pixel);
            live_max = WaveActiveMax(done ? uint2(0) : pixel);
        }
        const uint source_count = min(64u, end - batch);
        uint count = single_simd && compact_candidates ? 0u : source_count;
        const uint loaders = single_simd ? 32u : 64u;
        for (uint source_offset = 0; source_offset < 64u; source_offset += loaders) {
            const uint source_lane = source_offset + lane;
            uint destination = source_lane, id = 0;
            ProjectedSplat splat = (ProjectedSplat)0;
            GutSplat g = (GutSplat)0;
            bool active = source_lane < source_count;
            if (active) {
                id = p.indices[batch + source_lane];
                splat = p.splats[id];
                if (kRasterMode == 3u)
                    g = p.gut[id];
            }
            if (compact_candidates) {
                // Integer support with a one-pixel rounding margin.
                const int2 minimum = int2(live_min) - 1, maximum = int2(live_max) + 2;
                if (compact_gs)
                    active = active && all(int2(splat.bounds.xy) < maximum) && all(int2(splat.bounds.zw) > minimum);
                if (compact_gs && active && (kRasterFlags & (4u | 64u)) == 0u)
                    active = conic_intersects_pixels(splat, live_min, live_max);
                if (compact_gut && active && g.inverse0.w > 0.f) {
                    // Reject only when the alpha-support sphere misses a ray plane.
                    const float3 center = g.mean_opacity.xyz;
                    const float z = r.camera.z == 1u ? 1.f : center.z;
                    const float4 edges = float4(ray_min * z, ray_max * z);
                    const float4 distances = float4(center.xy - edges.xy, edges.zw - center.xy);
                    const float4 rounding = 1e-6f * (float4(abs(center.xy), abs(center.xy)) + abs(edges));
                    active = all(distances >= -g.inverse0.w * plane_lengths - rounding);
                }
                const uint live = active ? 1u : 0u;
                destination = WavePrefixSum(live);
                const uint total = WaveActiveSum(live);
                if (single_simd) {
                    destination += count;
                    count += total;
                } else {
                    if ((lane & 31u) == 0u)
                        active_counts[sg] = total;
                    GroupMemoryBarrierWithGroupSync();
                    if (sg != 0u)
                        destination += active_counts[0];
                    count = active_counts[0] + active_counts[1];
                }
            }
            if (active) {
                ids[destination] = id;
                if (kRasterMode == 3u) {
                    if (r.camera.z != 1u) {
                        // Pinhole and spherical rays start at the camera:
                        // cache the inverse-Gaussian origin once per batch.
                        const float3 delta = -g.mean_opacity.xyz;
                        g.mean_opacity.xyz = float3(dot(g.inverse0.xyz, delta), dot(g.inverse1.xyz, delta), dot(g.inverse2.xyz, delta));
                    }
                    geometry[destination] = g;
                }
                means[destination] = splat.mean_depth;
                conics[destination] = splat.conic_opacity;
                colors[destination] = splat.color;
                if (batch_half) {
                    const float4 c = splat.conic_opacity;
                    const float l00 = sqrt(max(c.x, 1e-12f)), l01 = c.y / l00, l11 = sqrt(max(c.z - l01 * l01, 0.f));
                    macro_chol[destination] = half4(float4(l00 * overlay_tile_extent.x * .849321800288019f, l01 * overlay_tile_extent.y * .849321800288019f,
                                                           l11 * overlay_tile_extent.y * .849321800288019f, max(4.f, log(c.w * 510.f)) * 1.4426950408889634f));
                    macro_center[destination] = half2((splat.mean_depth.xy + r.render_origin.xy - batch_macro_origin) / overlay_tile_extent);
                }
                colors[destination].xyz = clamp(colors[destination].xyz, 0.f, 4.f);
                if ((kRasterFlags & 64u) != 0u)
                    colors[destination].xyz = float3(half3(colors[destination].xyz));
                if ((kRasterFlags & 16u) == 0u && kRasterMode != 1u) {
                    const float opacity = kRasterMode == 3u ? geometry[destination].mean_opacity.w : splat.conic_opacity.w;
                    // Cache the alpha support and reject before exp.
                    colors[destination].w = max(4.f, log(max(opacity, .5f / 255.f) * 510.f)) + 1e-4f;
                }
                if ((kRasterFlags & 16u) != 0u && kRasterMode != 1u) {
                    const float opacity = kRasterMode == 3u ? geometry[destination].mean_opacity.w : splat.conic_opacity.w;
                    const float adjusted = sqrt(8.f) + .7f * (min(opacity, 5.f) - 1.f);
                    const float power = opacity > 1.f ? .5f * adjusted * adjusted : max(4.f, log(max(opacity, .5f / 255.f) * 510.f));
                    means[destination].w = exp(-power);
                    colors[destination].w = opacity > 1.f ? exp((opacity * opacity - 1.f) / 2.718281828459045f) : 0.f;
                }
            }
        }
        blend_barrier();
        if (!done) {
            for (uint j = 0; j < count; ++j) {
                // Integer pixel coordinates, as the CUDA/Vulkan reference samples.
                const uint logical = (kRasterFlags & 8u) != 0u ? p.logical_ids[ids[j]] : ids[j];
                if ((kRasterFlags & 8u) != 0u && logical >= p.logical_count)
                    continue;
                float2 d = float2(pixel) - means[j].xy;
                if ((kRasterFlags & 8192u) != 0u)
                    d.x -= r.panorama.x * round(d.x / r.panorama.x);
                const float4 c = conics[j];
                if (separate_median && median_transmittance > .5f) {
                    // Full-precision median, independent of the display footprint.
                    const float power = .5f * (c.x * d.x * d.x + c.z * d.y * d.y) + c.y * d.x * d.y;
                    const float edge = .01831563888873418f;
                    const float value = power < 0.f || power > 4.f ? 0.f : exp(-power);
                    float exact_alpha = min(c.w * max(0.f, (value - edge) / (1.f - edge)), .999f);
                    if (c.w <= 1.f / 255.f || exact_alpha < 1.f / 255.f)
                        exact_alpha = 0.f;
                    if (overlays && exact_alpha > 0.f) {
                        const uint flags = p.overlay_flags[ids[j]];
                        if (overlay_enabled(p.overlay_params[22].x)) {
                            const float gaussian = exp(-power), boundary = (.5f / 255.f) / max(c.w, 1e-8f);
                            const float width = p.overlay_params[21].w * 10.f;
                            if (gaussian < boundary * (1 + width) && gaussian > boundary * (1 - width))
                                exact_alpha = .8f;
                        }
                        if (overlay_enabled(p.overlay_params[22].y) && (flags & 2u) == 0u)
                            exact_alpha = dot(d, d) <= 6.25f ? 1.f : 0.f;
                    }
                    const float remaining = median_transmittance * (1.f - exact_alpha);
                    if (remaining <= .5f)
                        median = means[j].z;
                    median_transmittance = remaining;
                }
                const bool half_footprint = (kRasterFlags & 16384u) == 0u && ((kRasterFlags & 2048u) == 0u || separate_median) &&
                                            ((kRasterMode == 0u && (kRasterFlags & (4u | 64u)) != 0u) ||
                                             (kRasterMode != 3u && (kRasterFlags & 16u) == 0u && overlays && overlay_enabled(p.overlay_params[22].y) &&
                                              (p.overlay_flags[ids[j]] & 2u) == 0u));
                const float q = half_footprint ? 0.f : c.x * d.x * d.x + 2 * c.y * d.x * d.y + c.z * d.y * d.y;
                if (q < 0 || !isfinite(q))
                    continue;
                if ((kRasterFlags & 16u) == 0u && kRasterMode == 0u && !half_footprint && .5f * q > colors[j].w)
                    continue;
                float alpha;
                float splat_depth = means[j].z;
                if (kRasterMode == 1u) {
                    alpha = dot(d, d) <= means[j].w * means[j].w ? c.w : 0;
                } else if (kRasterMode == 2u) {
                    alpha = q <= 9 ? c.w : 0;
                } else if (kRasterMode == 3u) {
                    const GutSplat g = geometry[j];
                    if ((kRasterFlags & 4u) != 0u) {
                        const float2 axis = float2(g.inverse0.w, g.inverse1.w);
                        if (abs(dot(d, axis)) > means[j].w || abs(dot(d, float2(axis.y, -axis.x))) > g.inverse2.w)
                            continue;
                    }
                    float3 local_origin = g.mean_opacity.xyz;
                    if (r.camera.z == 1u) {
                        const float3 delta = gut_origin - g.mean_opacity.xyz;
                        local_origin = float3(dot(g.inverse0.xyz, delta), dot(g.inverse1.xyz, delta), dot(g.inverse2.xyz, delta));
                    }
                    const float3 local_direction = float3(dot(g.inverse0.xyz, gut_direction), dot(g.inverse1.xyz, gut_direction), dot(g.inverse2.xyz, gut_direction));
                    const float denom = dot(local_direction, local_direction);
                    if (!isfinite(denom) || denom <= 1e-12f)
                        continue;
                    const float3 distance = cross(local_direction * rsqrt(denom), local_origin);
                    const float power = .5f * dot(distance, distance);
                    if ((kRasterFlags & 16u) == 0u && power > colors[j].w)
                        continue;
                    alpha = g.mean_opacity.w * exp(-power);
                    const float t = -dot(local_direction, local_origin) / denom;
                    const float z = gut_origin.z + t * gut_direction.z;
                    splat_depth = t > 0 && z > r.clip.x && isfinite(z) ? z : 1e10f;
                } else if (half_footprint) {
                    alpha = 0.f; // evaluated on the half footprint below
                } else if ((kRasterFlags & 4u) != 0u) {
                    const float edge = .01831563888873418f;
                    const float value = q > 8.f ? 0.f : exp(-.5f * q);
                    alpha = c.w * max(0.f, (value - edge) / (1.f - edge));
                } else {
                    alpha = c.w * exp(-.5f * q);
                }
                if ((kRasterFlags & 16u) != 0u && kRasterMode != 1u) {
                    const float opacity = kRasterMode == 3u ? geometry[j].mean_opacity.w : c.w;
                    const float value = clamp(alpha / max(opacity, 1e-8f), 0.f, 1.f);
                    if (value < means[j].w)
                        continue;
                    if (colors[j].w > 0)
                        alpha = 1.f - pow(max(0.f, 1.f - value), colors[j].w);
                }
                alpha = min(alpha, .999f);
                if (half_footprint) {
                    const half2 center = uniform_macro ? macro_center[j] : half2((means[j].xy + r.render_origin.xy - pixel_macro_origin) / overlay_tile_extent);
                    const half4 chol = macro_chol[j];
                    half2 delta = pixel_macro_coord - center;
                    if ((kRasterFlags & 8192u) != 0u)
                        delta.x -= half(r.panorama.x / overlay_tile_extent.x) * round(delta.x / half(r.panorama.x / overlay_tile_extent.x));
                    // The macro reference rounds the X product before adding Y.
                    const half dx = half(chol.x * delta.x);
                    const half u = half(dx + chol.y * delta.y), v = half(chol.z * delta.y);
                    const half power = u * u + v * v;
                    if (!isfinite(power) || power < half(0) || power > chol.w)
                        continue;
                    half value = exp2(-power);
                    if ((kRasterFlags & 4u) != 0u) {
                        const half edge = half(.01831563888873418f);
                        value = max(half(0), (value - edge) / (half(1) - edge));
                    }
                    // half(.999f) is exactly .9990234375; a float min of two
                    // half values equals their half min. (Slang folds a half
                    // constant to an unsuffixed MSL literal, which is ambiguous.)
                    alpha = min(float(half(c.w) * value), .9990234375f);
                }
                const float minimum_alpha = (kRasterFlags & 4u) != 0u ? 1.f / 255.f : .5f / 255.f;
                if (alpha < (half_footprint ? float(half(minimum_alpha)) : minimum_alpha))
                    continue;
                if (overlays && alpha < .5f / 255.f)
                    continue;
                float3 radiance = colors[j].xyz;
                if (overlays) {
                    const uint flags = p.overlay_flags[ids[j]];
                    // Pixel-sized overlays use the macro-relative half center.
                    const float2 macro_origin = floor((float2(pixel) + r.render_origin.xy) / overlay_macro_extent) * overlay_macro_extent;
                    const float2 overlay_center = (kRasterMode == 3u || (kRasterFlags & (16u | 2048u | 16384u)) != 0u)
                                                      ? means[j].xy
                                                      : float2(half2((means[j].xy + r.render_origin.xy - macro_origin) / overlay_tile_extent)) * overlay_tile_extent + macro_origin - r.render_origin.xy;
                    const uint status_bits = overlay_selection(p.overlay_params, logical, flags, overlay_center + (r.camera.z == 2u ? r.panorama.zw : float2(0)),
                                                               p.selection, p.preview, r.mask_limits.xy);
                    const bool selectable = (flags & 2u) == 0u;
                    if (overlay_enabled(p.overlay_params[22].x) && selectable) {
                        float2 ring_delta = float2(pixel) - overlay_center;
                        if ((kRasterFlags & 8192u) != 0u)
                            ring_delta.x -= r.panorama.x * round(ring_delta.x / r.panorama.x);
                        const float sigma_over_2 = .5f * (c.x * ring_delta.x * ring_delta.x + c.z * ring_delta.y * ring_delta.y) + c.y * ring_delta.x * ring_delta.y;
                        const float gaussian = exp(-sigma_over_2);
                        const float boundary = (.5f / 255.f) / max(c.w, 1e-8f);
                        const float width = p.overlay_params[21].w * 10;
                        if (gaussian < boundary * (1 + width) && gaussian > boundary * (1 - width)) {
                            if (status_bits != 0u)
                                radiance = overlay_target(status_bits, p.selection_colors);
                            alpha = .8f;
                        }
                    }
                    if (overlay_enabled(p.overlay_params[22].y) && selectable) {
                        // Marker mode contributes dots only.
                        float2 marker_delta = float2(pixel) - overlay_center;
                        if ((kRasterFlags & 8192u) != 0u)
                            marker_delta.x -= r.panorama.x * round(marker_delta.x / r.panorama.x);
                        const float marker_distance = length(marker_delta);
                        if (marker_distance > 2.5f)
                            continue;
                        rgb = overlay_target(status_bits, p.selection_colors) * (marker_distance > 1.5f ? .4f : 1.f);
                        if (!separate_median && ((kRasterFlags & 64u) != 0u ? composed_transmittance * transmittance : transmittance) > .5f)
                            median = splat_depth;
                        transmittance = 0;
                        done = true;
                        break;
                    }
                    if ((status_bits & 128u) != 0u)
                        radiance = lerp(radiance, overlay_target(status_bits, p.selection_colors), .9f);
                    else if ((status_bits & 127u) != 0u)
                        radiance = lerp(radiance, p.selection_colors[status_bits & 127u].xyz, .8f);
                    if ((flags & 4u) != 0u && p.overlay_params[20].w > 0)
                        radiance = lerp(radiance, float3(1, .95f, .6f), p.overlay_params[20].w * .5f);
                }
                const float total_transmittance = (kRasterFlags & 64u) != 0u ? composed_transmittance * transmittance : transmittance;
                const float weight = alpha * total_transmittance;
                if (picked == 0xffffffffu) {
                    picked = logical;
                    nearest = splat_depth;
                }
                // Invalid or too-distant GUT depths never enter the average.
                if ((kRasterFlags & 2u) != 0u) {
                    if (splat_depth <= r.clip.y) {
                        weighted_depth += splat_depth * weight;
                        valid_depth_weight += weight;
                    }
                } else {
                    weighted_depth += splat_depth * weight;
                }
                const float next_transmittance = transmittance * (1 - alpha);
                const float next_total_transmittance = (kRasterFlags & 64u) != 0u ? composed_transmittance * next_transmittance : next_transmittance;
                if (!separate_median && total_transmittance > .5f && next_total_transmittance <= .5f)
                    median = splat_depth;
                // The GUT reference records depth before dropping a saturating splat.
                if ((kRasterFlags & 32u) != 0u && next_transmittance < 1e-4f) {
                    done = true;
                    break;
                }
                if ((kRasterFlags & 64u) != 0u) {
                    if (overlays) {
                        rgb = float3(half3(rgb + radiance * (alpha * transmittance)));
                        transmittance = float(half(next_transmittance));
                    } else {
                        const half half_weight = half(half(alpha) * half(transmittance));
                        rgb = float3(half3(half3(rgb) + half3(radiance) * half_weight));
                        const half remaining = half(half(1) - half(alpha));
                        transmittance = float(half(half(transmittance) * remaining));
                    }
                } else {
                    rgb += radiance * weight;
                    transmittance = next_transmittance;
                }
                if (((kRasterFlags & 64u) != 0u ? composed_transmittance * transmittance : transmittance) < local_cutoff) {
                    done = true;
                    break;
                }
            }
        }
        if ((kRasterFlags & 64u) != 0u) {
            composed_rgb += rgb * composed_transmittance;
            composed_transmittance *= transmittance;
            rgb = 0;
            transmittance = 1;
        }
        blend_barrier();
    }
    if (valid) {
        if ((kRasterFlags & 64u) != 0u) {
            rgb = composed_rgb;
            transmittance = composed_transmittance;
        }
        if (depth_batch) {
            const uint at = slot * 256u + subtile * 32u + lane;
            p.partial_color[at] = float4(rgb, transmittance);
            p.partial_depth[at] = float4(weighted_depth, valid_depth_weight, nearest, median);
            p.partial_pick[at] = picked;
            return;
        }
        write_pixel(pixel, r, rgb, transmittance, weighted_depth, (kRasterFlags & 2u) != 0u ? valid_depth_weight : nearest, median, picked);
    }
}

groupshared float4 replay_means[64], replay_conics[64], replay_colors[64];
groupshared uint replay_ids[64];

// Combines the parallel depth chunks of long tiles. Partial colors and
// weighted depths are linear in incoming transmittance; chunks that cross the
// median or saturate replay their contributors.
[shader("compute")]
[numthreads(32, 1, 1)]
void tile_depth_compose(uint3 group_id : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const RasterParameters r = p.raster[0];
    const RasterStatus status = p.status[0];
    if (status.required > uint64_t(r.mask_limits.z))
        return;
    const uint group = group_id.x, lane = thread.x;
    const uint tile = group / 8u, subtile = group % 8u;
    const uint2 pixel = uint2(tile % r.columns * 16u + subtile % 2u * 8u + lane % 8u, tile / r.columns * 16u + subtile / 2u * 4u + lane / 8u);
    const bool valid = pixel.x < r.width && pixel.y < r.height;
    const uint begin = status.error != 0u ? 0u : p.ranges[tile * 2u], end = status.error != 0u ? 0u : p.ranges[tile * 2u + 1u];
    // The prefix pass wrote short, empty and overflow tiles directly.
    if (end - begin <= depth_split_threshold(status, r))
        return;
    float3 rgb = 0;
    float transmittance = 1, weighted_depth = 0, valid_depth_weight = 0, nearest = 0, median = 1e10f;
    uint picked = 0xffffffffu;
    const uint chunks = (end - begin) / kDepthChunkSize + ((end - begin) % kDepthChunkSize != 0u ? 1u : 0u);
    for (uint chunk = 0; chunk < chunks; ++chunk) {
        const uint batch = begin + chunk * kDepthChunkSize;
        const bool active = valid && transmittance >= 1e-4f;
        if (!WaveActiveAnyTrue(active))
            break;
        const uint slot = begin / kDepthChunkSize + tile + (batch - begin) / kDepthChunkSize;
        const uint at = slot * 256u + subtile * 32u + lane;
        const float4 rgba = active ? p.partial_color[at] : float4(0, 0, 0, 1);
        const float4 z = active ? p.partial_depth[at] : float4(0);
        const float incoming = transmittance, next = incoming * rgba.w;
        const bool crossing = active && incoming > .5f && next <= .5f;
        const bool saturating = active && next < 1e-4f;
        bool replay = active && incoming != 1.f && (crossing || saturating);
        const bool replay_color = replay && saturating;
        if (active && !replay_color) {
            rgb += rgba.xyz * incoming;
            weighted_depth += z.x * incoming;
            valid_depth_weight += z.y * incoming;
            if (picked == 0xffffffffu && p.partial_pick[at] != 0xffffffffu) {
                picked = p.partial_pick[at];
                nearest = z.z;
            }
            if (crossing && incoming == 1.f)
                median = z.w;
            transmittance = next;
        }
        float replay_transmittance = incoming;
        const uint chunk_size = min(kDepthChunkSize, end - batch);
        for (uint block = 0; block < chunk_size; block += 64u) {
            const uint offset = batch + block;
            if (!WaveActiveAnyTrue(replay))
                break;
            const uint available = min(64u, chunk_size - block);
            const uint2 live_min = WaveActiveMin(replay ? pixel : uint2(0xffffffffu));
            const uint2 live_max = WaveActiveMax(replay ? pixel : uint2(0));
            uint candidates = 0;
            for (uint source_offset = 0; source_offset < 64u; source_offset += 32u) {
                const uint source = source_offset + lane;
                const uint id = source < available ? p.indices[offset + source] : 0u;
                ProjectedSplat v = (ProjectedSplat)0;
                if (source < available)
                    v = p.splats[id];
                const uint live = source < available && conic_intersects_pixels(v, live_min, live_max) ? 1u : 0u;
                const uint destination = candidates + WavePrefixSum(live);
                candidates += WaveActiveSum(live);
                if (live != 0u) {
                    replay_means[destination] = v.mean_depth;
                    replay_means[destination].w = max(8.f, 2.f * log(max(v.conic_opacity.w, .5f / 255.f) * 510.f)) + 1e-3f;
                    replay_conics[destination] = v.conic_opacity;
                    replay_colors[destination] = v.color;
                    replay_ids[destination] = id;
                }
            }
            GroupMemoryBarrierWithGroupSync(); // one subgroup per group
            for (uint j = 0; j < candidates && replay; ++j) {
                const float2 d = float2(pixel) - replay_means[j].xy;
                const float4 c = replay_conics[j];
                const float q = c.x * d.x * d.x + 2 * c.y * d.x * d.y + c.z * d.y * d.y;
                if (q < 0.f || !isfinite(q) || q > replay_means[j].w)
                    continue;
                const float alpha = min(c.w * exp(-.5f * q), .999f);
                if (alpha < .5f / 255.f)
                    continue;
                const float splat_depth = replay_means[j].z;
                if (replay_color) {
                    const float weight = alpha * replay_transmittance;
                    if (picked == 0xffffffffu) {
                        picked = replay_ids[j];
                        nearest = splat_depth;
                    }
                    if ((r.unused & 2u) != 0u) {
                        if (splat_depth <= r.clip.y) {
                            weighted_depth += splat_depth * weight;
                            valid_depth_weight += weight;
                        }
                    } else {
                        weighted_depth += splat_depth * weight;
                    }
                    rgb += clamp(replay_colors[j].xyz, 0.f, 4.f) * weight;
                }
                const float remaining = replay_transmittance * (1 - alpha);
                if (replay_transmittance > .5f && remaining <= .5f)
                    median = splat_depth;
                replay_transmittance = remaining;
                replay = remaining >= 1e-4f && (replay_color || remaining > .5f);
            }
            GroupMemoryBarrierWithGroupSync(); // one subgroup per group
        }
        if (replay_color)
            transmittance = replay_transmittance;
    }
    if (!valid)
        return;
    write_pixel(pixel, r, rgb, transmittance, weighted_depth, (r.unused & 2u) != 0u ? valid_depth_weight : nearest, median, picked);
}
