// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
struct CompositeParameters {
    Ptr<uint8_t, Access.Read> source;
    Ptr<uint8_t, Access.ReadWrite> destination;
    uint width;
    uint height;
    uint replace;
    uint padding;
};
[[vk::push_constant]] ConstantBuffer<CompositeParameters> c;

[shader("compute")]
[numthreads(64, 1, 1)]
void composite(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= c.width || tid.y >= c.height) return;
    uint i = (tid.y * c.width + tid.x) * 4;
    float4 source = float4(c.source[i], c.source[i+1], c.source[i+2], c.source[i+3]) / 255.0;
    float4 destination = float4(c.destination[i], c.destination[i+1], c.destination[i+2], c.destination[i+3]) / 255.0;
    float4 result = c.replace != 0 ? source : source + destination * (1.0 - source.a);
    c.destination[i] = uint8_t(clamp(result.r, 0.0, 1.0) * 255.0 + 0.5);
    c.destination[i+1] = uint8_t(clamp(result.g, 0.0, 1.0) * 255.0 + 0.5);
    c.destination[i+2] = uint8_t(clamp(result.b, 0.0, 1.0) * 255.0 + 0.5);
    c.destination[i+3] = uint8_t(clamp(result.a, 0.0, 1.0) * 255.0 + 0.5);
}
