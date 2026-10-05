// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
struct Parameters {
    Ptr<uint8_t, Access.Read> color;
    Ptr<float, Access.Read> depth;
    Ptr<float2, Access.Read> motion;
    Ptr<uint, Access.ReadWrite> packedColor;
    Ptr<float, Access.ReadWrite> packedDepth;
    Ptr<float2, Access.ReadWrite> packedMotion;
    Ptr<float4, Access.ReadWrite> output;
    Ptr<uint8_t, Access.Read> unused;
    uint4 extents;
    uint4 layout; // source width, float color, flip Y, temporal
    uint4 pitches; // color/depth/motion/output byte strides
    float4 depthInfo; // near, far, orthographic, depth source stride
    float4 jitter;
};
[[vk::push_constant]] ConstantBuffer<Parameters> p;
float4 colorAt(uint2 xy) {
    if (p.layout.z != 0) xy.y = p.extents.y - 1 - xy.y;
    uint index = xy.y * p.layout.x + xy.x;
    if (p.layout.y != 0) return ((Ptr<float4, Access.Read>)p.color)[index];
    index *= 4;
    return float4(p.color[index], p.color[index + 1], p.color[index + 2], p.color[index + 3]) / 255.0f;
}
[shader("compute")]
[numthreads(8, 8, 1)]
void scene_texture_pack(uint3 tid : SV_DispatchThreadID) {
    uint2 xy = tid.xy;
    if (any(xy >= p.extents.xy)) return;
    float4 c = colorAt(xy);
    uint index = xy.y * (p.pitches.x / 4) + xy.x * 2;
    p.packedColor[index] = f32tof16(c.x) | (f32tof16(c.y) << 16);
    p.packedColor[index + 1] = f32tof16(c.z) | (f32tof16(c.w) << 16);
    if (p.layout.w != 0) {
        uint sourceY = p.layout.z != 0 ? p.extents.y - 1 - xy.y : xy.y;
        float z = p.depth[sourceY * uint(p.depthInfo.w) + xy.x];
        float n = p.depthInfo.x, f = p.depthInfo.y;
        float raster = (!isfinite(z) || z <= 0) ? 1.0f :
            (p.depthInfo.z != 0 ? (z - n) / (f - n) : f / (f - n) * (1.0f - n / max(z, n)));
        p.packedDepth[xy.y * (p.pitches.y / 4) + xy.x] = clamp(raster, 0.0f, 1.0f);
        p.packedMotion[xy.y * (p.pitches.z / 8) + xy.x] = p.motion[xy.y * p.extents.x + xy.x];
    }
}
[shader("compute")]
[numthreads(8, 8, 1)]
void scene_texture_unpack(uint3 tid : SV_DispatchThreadID) {
    uint2 xy = tid.xy;
    if (any(xy >= p.extents.zw)) return;
    uint index = xy.y * (p.pitches.w / 4) + xy.x * 2;
    uint lo = p.packedColor[index], hi = p.packedColor[index + 1];
    float4 value = float4(f16tof32(lo & 65535u), f16tof32(lo >> 16), f16tof32(hi & 65535u), f16tof32(hi >> 16));
    float2 pos = (float2(xy) + 0.5f) * float2(p.extents.xy) / float2(p.extents.zw) + p.jitter.xy - 0.5f;
    int2 base = int2(floor(pos));
    float2 t = frac(pos);
    uint2 a = uint2(clamp(base, int2(0), int2(p.extents.xy) - 1));
    uint2 b = uint2(clamp(base + 1, int2(0), int2(p.extents.xy) - 1));
    value.a = lerp(lerp(colorAt(a).a, colorAt(uint2(b.x, a.y)).a, t.x),
                 lerp(colorAt(uint2(a.x, b.y)).a, colorAt(b).a, t.x), t.y);
    p.output[xy.y * p.extents.z + xy.x] = value;
}
