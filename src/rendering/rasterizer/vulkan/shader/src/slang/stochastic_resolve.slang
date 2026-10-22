/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// SteadySplats Eq. 9 and Algorithm 3. History is radiance plus transmittance,
// before exposure, tone mapping, quantization, or background composition.
struct ResolveUniforms {
    uint width;
    uint height;
    uint mode; // 1: basic stochastic CMA, 2: SteadySplats spatial + temporal
    uint history_valid;
    uint sample_count;
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

static const int2 OFFSETS[4] = {int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1)};

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
    return result;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uniform ResolveUniforms u) {
    int2 p = int2(dispatchThreadID.xy);
    if (p.x >= int(u.width) || p.y >= int(u.height))
        return;
    uint index = pixel_index(p, u);
    float4 current = samples[index];
    if (u.mode == 1u) {
        // Reset on camera or content changes. Each submitted frame is one new
        // sample; the viewport never waits for a batch before displaying it.
        if (u.history_valid != 0u && u.sample_count > 0u)
            current = lerp(previous_color[index], current, 1.0f / float(u.sample_count + 1u));
        resolved_color[index] = current;
        return;
    }

    float weight_sum = 1.0f;
    for (uint d = 0; d < 4; ++d) {
        int2 neighbor = p + OFFSETS[d];
        if (any(neighbor < int2(0)) || neighbor.x >= int(u.width) || neighbor.y >= int(u.height))
            continue;
        uint n = pixel_index(neighbor, u);
        float weight = outgoing_weights[n][d ^ 1u];
        current += weight * samples[n];
        weight_sum += weight;
    }
    current /= weight_sum;

    float depth = depths[index];
    if (u.history_valid == 0u || u.temporal_projection_valid == 0u ||
        !isfinite(depth) || depth >= 1.0e9f) {
        resolved_color[index] = current;
        return;
    }
    float2 camera_pixel = float2(p) + u.origins.xy;
    float2 xy = (camera_pixel - u.intrinsics.zw) / u.intrinsics.xy;
    float3 view = float3(u.orthographic != 0u ? xy : xy * depth, depth);
    float4 position = float4(view, 1.0f);
    float3 previous_view = float3(dot(u.view_to_previous[0], position),
                                  dot(u.view_to_previous[1], position),
                                  dot(u.view_to_previous[2], position));
    if (u.previous_orthographic == 0u && previous_view.z <= 0.0f) {
        resolved_color[index] = current;
        return;
    }
    float2 previous_xy = u.previous_orthographic != 0u
        ? previous_view.xy : previous_view.xy / previous_view.z;
    float2 previous_pixel = previous_xy * u.previous_intrinsics.xy + u.previous_intrinsics.zw - u.origins.zw;
    if (!all(isfinite(previous_pixel)) || any(previous_pixel < float2(2.0f)) ||
        any(previous_pixel >= float2(u.width, u.height) - 2.0f)) {
        resolved_color[index] = current;
        return;
    }

    float4 history = fetch_history(previous_pixel, u);
    float3 mean = float3(0.0f);
    float3 second = float3(0.0f);
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x) {
            int2 q = clamp(p + int2(x, y), int2(0), int2(u.width, u.height) - 1);
            float3 color = ycocg(radiance(samples[pixel_index(q, u)], u));
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
    resolved_color[index] = lerp(current, history, retention);
}
