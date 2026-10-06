// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
// Turns the blended color and depth statistics into the compositor's RGBA8
// image and linear view depth: tone curves and exposure, transparent
// unpremultiplication, the depth view, expected-depth capture, and the last
// frame on overflow. Ported from the native Metal viewer's present_viewer.
#include "splat_types.slang"

#define LFS_COLOR_INLINE
#define LFS_COLOR_VEC3 float3
#define LFS_COLOR_UINT uint
#define LFS_COLOR_MIX lerp
#include "../slang/display_color.inc"

struct PresentParameters {
    float exposure;
    uint tone, transparent, has_previous;
    float depth_min, depth_max;
    uint depth_view, depth_mode;
    float4 background;
    uint4 capture;      // x: expected depth
    uint4 extent;       // width, height, previous width, previous height
};

struct Parameters {
    Ptr<half, Access.Read> color;        // [height][width][4] premultiplied
    Ptr<float4, Access.Read> depth;      // weighted depth, alpha, first or valid weight, median
    Ptr<RasterStatus, Access.Read> status;
    Ptr<PresentParameters, Access.Read> present;
    Ptr<uint, Access.Read> previous_rgba;     // optional
    Ptr<float, Access.Read> previous_depth;   // optional
    Ptr<uint, Access.ReadWrite> rgba;         // packed RGBA8
    Ptr<float, Access.ReadWrite> linear_depth;
};
[[vk::push_constant]] ConstantBuffer<Parameters> p;

float normalized_depth(float depth, float lo, float hi) {
    lo = max(lo, 1e-4f);
    hi = max(hi, lo + 1e-4f);
    depth = clamp(depth, lo, hi);
    const float linear = clamp((depth - lo) / max(hi - lo, 1e-5f), 0.0f, 1.0f);
    const float logarithmic = clamp(log2(depth / lo) / max(log2(hi / lo), 1e-4f), 0.0f, 1.0f);
    return smoothstep(0.0f, 1.0f, lerp(linear, logarithmic, smoothstep(1.75f, 24.0f, hi / lo)));
}

float3 depth_palette(float t) {
    t = clamp(t, 0.0f, 1.0f);
    const float3 far0 = {.050f, .040f, .150f}, far1 = {.060f, .195f, .500f};
    const float3 mid0 = {0, .500f, .650f}, mid1 = {.360f, .735f, .410f};
    const float3 near0 = {.965f, .820f, .300f}, near1 = {.985f, .430f, .125f};
    if (t < .20f)
        return lerp(far0, far1, smoothstep(0.0f, .20f, t));
    if (t < .43f)
        return lerp(far1, mid0, smoothstep(.20f, .43f, t));
    if (t < .67f)
        return lerp(mid0, mid1, smoothstep(.43f, .67f, t));
    if (t < .86f)
        return lerp(mid1, near0, smoothstep(.67f, .86f, t));
    return lerp(near0, near1, smoothstep(.86f, 1.0f, t));
}

// Unorm8 conversion of a render-target write: clamp, scale, round to nearest.
uint pack_rgba8(float4 c) {
    const uint4 v = uint4(round(saturate(c) * 255.0f));
    return v.x | (v.y << 8) | (v.z << 16) | (v.w << 24);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void present_viewer(uint3 tid : SV_DispatchThreadID) {
    const PresentParameters q = p.present[0];
    const uint2 pixel = tid.xy;
    if (pixel.x >= q.extent.x || pixel.y >= q.extent.y)
        return;
    const uint at = pixel.y * q.extent.x + pixel.x;
    if (p.status[0].error != 0u && q.has_previous != 0u) {
        const uint2 previous = uint2(uint64_t(pixel.x) * q.extent.z / q.extent.x, uint64_t(pixel.y) * q.extent.w / q.extent.y);
        const uint from = previous.y * q.extent.z + previous.x;
        p.rgba[at] = p.previous_rgba[from];
        p.linear_depth[at] = p.previous_depth[from];
        return;
    }
    float4 c = float4(float(p.color[at * 4u]), float(p.color[at * 4u + 1u]), float(p.color[at * 4u + 2u]), float(p.color[at * 4u + 3u]));
    const float4 d = p.depth[at];
    // Expected capture divides by the valid-depth weight; display uses the median.
    const float output_depth = q.capture.x != 0u ? (d.z > 1e-4f ? d.x / d.z : 1e10f) : d.w;
    p.linear_depth[at] = output_depth;
    if (q.depth_view != 0u) {
        const bool empty = d.y < .02f || d.w >= 1e9f || d.w <= 0;
        const float hi = q.depth_max <= q.depth_min + 1e-5f ? q.depth_min + 1 : q.depth_max;
        const float near_t = empty ? 0 : 1 - normalized_depth(d.w, q.depth_min, hi);
        const float3 rgb = q.depth_mode == 1u ? float3(near_t) : depth_palette(near_t);
        const float coverage = empty ? 0 : smoothstep(.02f, .72f, d.y);
        p.rgba[at] = pack_rgba8(q.transparent != 0u ? float4(rgb, coverage) : float4(lerp(q.background.rgb, rgb, coverage), 1));
        return;
    }
    if (q.transparent != 0u) {
        // Unpremultiply only published coverage, using the FP32 alpha.
        const float coverage = d.y;
        if (coverage <= .5f / 255.f) {
            p.rgba[at] = 0u;
            return;
        }
        c.rgb /= coverage;
        c.a = coverage;
    }
    c.rgb = lfsDisplayTone(max(c.rgb, 0.0f), q.tone, q.exposure);
    p.rgba[at] = pack_rgba8(c);
}
