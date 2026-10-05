// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

struct Parameters {
    Ptr<uint8_t, Access.Read> source;
    Ptr<uint8_t, Access.ReadWrite> destination;
    uint sourceWidth;
    uint sourceHeight;
    uint destinationWidth;
    uint destinationHeight;
};
[[vk::push_constant]] ConstantBuffer<Parameters> p;

float4 texel(int2 q) {
    q = clamp(q, int2(0), int2(p.sourceWidth, p.sourceHeight) - 1);
    uint offset = (uint(q.y) * p.sourceWidth + uint(q.x)) * 4u;
    return float4(p.source[offset], p.source[offset + 1],
                  p.source[offset + 2], p.source[offset + 3]);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void resampleRgba8(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= p.destinationWidth || tid.y >= p.destinationHeight)
        return;
    float2 position = (float2(tid.xy) + 0.5) *
                      float2(p.sourceWidth, p.sourceHeight) /
                      float2(p.destinationWidth, p.destinationHeight) - 0.5;
    int2 base = int2(floor(position));
    float2 f = position - floor(position);
    float4 top = lerp(texel(base), texel(base + int2(1, 0)), f.x);
    float4 bottom = lerp(texel(base + int2(0, 1)), texel(base + int2(1, 1)), f.x);
    float4 value = lerp(top, bottom, f.y);
    uint offset = (tid.y * p.destinationWidth + tid.x) * 4u;
    p.destination[offset] = uint8_t(clamp(value.r, 0.0, 255.0) + 0.5);
    p.destination[offset + 1] = uint8_t(clamp(value.g, 0.0, 255.0) + 0.5);
    p.destination[offset + 2] = uint8_t(clamp(value.b, 0.0, 255.0) + 0.5);
    p.destination[offset + 3] = uint8_t(clamp(value.a, 0.0, 255.0) + 0.5);
}
