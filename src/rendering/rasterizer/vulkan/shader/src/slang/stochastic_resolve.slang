/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// SteadySplats Eq. 9 and Algorithm 3, with the web viewer's running average
// at rest. History is radiance plus transmittance, before display conversion.
struct ResolveUniforms {
    uint width;
    uint height;
    uint mode; // 1: basic stochastic CMA, 2: SteadySplats spatial + temporal
    uint history_valid;
    uint sample_count; // Consecutive samples with an unchanged camera.
    uint orthographic;
    uint previous_orthographic;
    uint temporal_projection_valid;
    float4 intrinsics;
    float4 previous_intrinsics;
    float4 view_to_previous[3];
    float4 origins;
    float4 background;
};

layout(binding=0) StructuredBuffer<float4> samples;
layout(binding=1) StructuredBuffer<float> depths;
layout(binding=2) StructuredBuffer<float4> outgoing_weights;
layout(binding=3) StructuredBuffer<float4> previous_color;
layout(binding=4) RWStructuredBuffer<float4> resolved_color;
layout(binding=5) RWStructuredBuffer<float4> rest_average;

static const int2 OFFSETS[4] = {int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1)};
static const uint REST_SPATIAL_FRAMES = 8u;
static const float REST_BLEND = 8.0f;
static const uint GROUP_WIDTH = 16u;
static const uint CACHE_WIDTH = GROUP_WIDTH + 2u;
static const uint CACHE_PIXELS = CACHE_WIDTH * CACHE_WIDTH;

// Stage the one-pixel apron once for spatial reuse (Eq. 9) and the 3x3
// neighborhood moments (Appendix 8). Values retain their full precision.
groupshared float4 cached_samples[CACHE_PIXELS];
groupshared float4 cached_weights[CACHE_PIXELS];

uint cache_index(int2 pixel) {
    return uint(pixel.y) * CACHE_WIDTH + uint(pixel.x);
}

uint pixel_index(int2 pixel, ResolveUniforms u) {
    return uint(pixel.y) * u.width + uint(pixel.x);
}

float3 ycocg(float3 rgb) {
    return float3(dot(rgb, float3(0.25f, 0.5f, 0.25f)),
                  dot(rgb, float3(0.5f, 0.0f, -0.5f)),
                  dot(rgb, float3(-0.25f, 0.5f, -0.25f)));
}

float3 radiance(float4 state, ResolveUniforms u) {
    return state.rgb + state.a * u.background.rgb;
}

float4 catmull_weights(float f) {
    float f2 = f * f;
    float f3 = f2 * f;
    return float4(-0.5f * f3 + f2 - 0.5f * f,
                  1.5f * f3 - 2.5f * f2 + 1.0f,
                 -1.5f * f3 + 2.0f * f2 + 0.5f * f,
                  0.5f * f3 - 0.5f * f2);
}

float4 fetch_history(float2 position, ResolveUniforms u) {
    int2 base = int2(floor(position));
    float2 fraction = position - float2(base);
    float4 wx = catmull_weights(fraction.x);
    float4 wy = catmull_weights(fraction.y);
    float4 result = float4(0.0f);
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x)
            result += previous_color[pixel_index(base + int2(x - 1, y - 1), u)] * wx[x] * wy[y];
    // The filter undershoots next to steps; keep radiance non-negative.
    return max(result, float4(0.0f));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID,
          uint3 groupThreadID : SV_GroupThreadID, uint3 groupID : SV_GroupID,
          uniform ResolveUniforms u) {
    int2 p = int2(dispatchThreadID.xy);
    bool inside = p.x < int(u.width) && p.y < int(u.height);
    if (u.mode == 1u) {
        if (!inside)
            return;
        uint index = pixel_index(p, u);
        float4 current = samples[index];
        // Reset on camera or content changes. Each submitted frame is one new
        // sample; the viewport never waits for a batch before displaying it.
        if (u.history_valid != 0u && u.sample_count > 0u)
            current = lerp(previous_color[index], current, 1.0f / float(u.sample_count + 1u));
        resolved_color[index] = current;
        return;
    }

    int2 origin = int2(groupID.xy) * int(GROUP_WIDTH) - int2(1);
    uint rank = groupThreadID.y * GROUP_WIDTH + groupThreadID.x;
    for (uint i = rank; i < CACHE_PIXELS; i += GROUP_WIDTH * GROUP_WIDTH) {
        int2 q = clamp(origin + int2(i % CACHE_WIDTH, i / CACHE_WIDTH),
                       int2(0), int2(u.width, u.height) - 1);
        uint source = pixel_index(q, u);
        cached_samples[i] = samples[source];
        cached_weights[i] = outgoing_weights[source];
    }
    // Inactive lanes in partial edge groups also load the apron and reach the
    // barrier. Returning before it would leave their neighbors unsynchronized.
    GroupMemoryBarrierWithGroupSync();
    if (!inside)
        return;
    uint index = pixel_index(p, u);
    int2 local = int2(groupThreadID.xy) + int2(1);
    float4 raw = cached_samples[cache_index(local)];
    float4 current = raw;
    float weight_sum = 1.0f;
    for (uint d = 0; d < 4; ++d) {
        int2 neighbor = p + OFFSETS[d];
        if (any(neighbor < int2(0)) || neighbor.x >= int(u.width) || neighbor.y >= int(u.height))
            continue;
        uint n = cache_index(local + OFFSETS[d]);
        float weight = cached_weights[n][d ^ 1u];
        current += weight * cached_samples[n];
        weight_sum += weight;
    }
    current /= weight_sum;

    if (u.history_valid == 0u || u.temporal_projection_valid == 0u) {
        resolved_color[index] = current;
        return;
    }

    float2 previous_pixel = float2(p);
    float4 history;
    if (u.sample_count > 0u) {
        // Identical rays need no depth or interpolation. In particular, a
        // background draw must not discard valid stationary color history.
        history = previous_color[index];
    } else {
        float depth = depths[index];
        bool background = raw.a > 0.5f;
        if ((!background && (!isfinite(depth) || depth >= 1.0e9f)) ||
            (background && (u.orthographic != 0u || u.previous_orthographic != 0u))) {
            resolved_color[index] = current;
            return;
        }
        float2 camera_pixel = float2(p) + u.origins.xy;
        float2 xy = (camera_pixel - u.intrinsics.zw) / u.intrinsics.xy;
        // A perspective background sample is a homogeneous direction at
        // infinity (w=0): rotation reprojects it, translation drops out.
        float4 position;
        if (background)
            position = float4(xy, 1.0f, 0.0f);
        else
            position = float4(u.orthographic != 0u ? xy : xy * depth, depth, 1.0f);
        float3 previous_view = float3(dot(u.view_to_previous[0], position),
                                      dot(u.view_to_previous[1], position),
                                      dot(u.view_to_previous[2], position));
        if (u.previous_orthographic == 0u && previous_view.z <= 0.0f) {
            resolved_color[index] = current;
            return;
        }
        float2 previous_xy = u.previous_orthographic != 0u
            ? previous_view.xy : previous_view.xy / previous_view.z;
        previous_pixel = previous_xy * u.previous_intrinsics.xy + u.previous_intrinsics.zw - u.origins.zw;
        if (!all(isfinite(previous_pixel)) || any(previous_pixel < float2(2.0f)) ||
            any(previous_pixel >= float2(u.width, u.height) - 2.0f)) {
            resolved_color[index] = current;
            return;
        }
        history = fetch_history(previous_pixel, u);
    }
    float3 mean = float3(0.0f);
    float3 second = float3(0.0f);
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x) {
            float3 color = ycocg(radiance(cached_samples[cache_index(local + int2(x, y))], u));
            mean += color;
            second += color * color;
        }
    mean /= 9.0f;
    float3 deviation = sqrt(max(float3(0.0f), second / 9.0f - mean * mean));
    float velocity = length(float2(p) - previous_pixel);
    float gamma = lerp(4.0f, 1.0f, min(1.0f, velocity / 4.0f));
    float3 extent = max(gamma * deviation, float3(0.01f, 0.005f, 0.005f));
    float3 distance = abs(ycocg(radiance(history, u)) - mean) / extent;
    float m = max(distance.x, max(distance.y, distance.z));
    float retention = 0.70f / max(1.0f, m);
    // Soft rejection changes the weight. It never clips historical radiance.
    float4 result = lerp(current, history, retention);
    if (u.sample_count > 0u) {
        // Start a separate average after motion. Spatial estimates settle the
        // first frames; raw visibility samples then make their influence fade.
        float n = float(u.sample_count);
        float4 sample = u.sample_count <= REST_SPATIAL_FRAMES ? current : raw;
        float4 average = sample;
        if (u.sample_count > 1u) {
            float4 previous = rest_average[index];
            average = previous + (sample - previous) / n;
        }
        rest_average[index] = average;
        result = lerp(result, average, n / (n + REST_BLEND));
    }
    resolved_color[index] = result;
}
