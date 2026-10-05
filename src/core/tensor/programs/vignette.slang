// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
// The reference viewport/vignette.frag formula, evaluated at pixel centers.
struct Parameters {
    Ptr<float4, Access.ReadWrite> output;
    uint width;
    uint height;
    float intensity;
    float radius;
    float softness;
    uint padding;
};
[[vk::push_constant]] ConstantBuffer<Parameters> p;

[shader("compute")]
[numthreads(64, 1, 1)]
void vignette(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= p.width || tid.y >= p.height) return;
    float2 viewport = float2(p.width, p.height);
    float2 uv = (float2(tid.xy) + 0.5) / viewport;
    float fadeWidth = (1.0 - clamp(p.radius, 0.0, 1.0)) * 0.5 * min(viewport.x, viewport.y);
    float alpha = 0;
    if (fadeWidth > 0) {
        float2 halfExtent = 0.5 * viewport;
        float2 innerHalf = max(halfExtent - fadeWidth, float2(0));
        float2 point = abs(uv * viewport - halfExtent) - innerHalf;
        float dist = length(max(point, float2(0)));
        float visible = clamp(1.0 - dist / fadeWidth, 0.0, 1.0);
        visible = lerp(visible, smoothstep(0.0, 1.0, visible), clamp(p.softness, 0.0, 1.0));
        alpha = clamp(p.intensity, 0.0, 1.0) * (1.0 - visible);
    }
    p.output[tid.y * p.width + tid.x] = float4(0, 0, 0, alpha);
}
