// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
struct Parameters {
    Ptr<uint8_t, Access.ReadWrite> destination;
    uint width;
    uint height;
    int viewportX;
    int viewportY;
    uint viewportWidth;
    uint viewportHeight;
    float intensity;
    float radius;
    float softness;
    uint padding;
};
[[vk::push_constant]] ConstantBuffer<Parameters> p;
[shader("compute")]
[numthreads(64, 1, 1)]
void applyVignette(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= p.viewportWidth || tid.y >= p.viewportHeight) return;
    uint2 pixel = uint2(p.viewportX, p.viewportY) + tid.xy;
    if (pixel.x >= p.width || pixel.y >= p.height) return;
    float2 viewport = float2(p.viewportWidth, p.viewportHeight);
    float2 uv = (float2(tid.xy) + 0.5) / max(viewport, 1.0);
    float fadeWidth = (1.0 - saturate(p.radius)) * 0.5 * min(viewport.x, viewport.y);
    float alpha = 0.0;
    if (fadeWidth > 0.0) {
        float2 halfExtent = 0.5 * viewport;
        float2 point = abs(uv * viewport - halfExtent) - max(halfExtent - fadeWidth, 0.0);
        float visible = saturate(1.0 - length(max(point, 0.0)) / fadeWidth);
        visible = lerp(visible, smoothstep(0.0, 1.0, visible), saturate(p.softness));
        alpha = saturate(p.intensity) * (1.0 - visible);
    }
    uint i = (pixel.y * p.width + pixel.x) * 4u;
    float scale = 1.0 - alpha;
    p.destination[i] = uint8_t(float(p.destination[i]) * scale + 0.5);
    p.destination[i + 1] = uint8_t(float(p.destination[i + 1]) * scale + 0.5);
    p.destination[i + 2] = uint8_t(float(p.destination[i + 2]) * scale + 0.5);
}
