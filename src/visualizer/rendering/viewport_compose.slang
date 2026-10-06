// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
struct ComposeParameters {
    Ptr<uint8_t, Access.ReadWrite> destination;
    Ptr<uint8_t, Access.Read> scene;
    uint destinationWidth;
    uint destinationHeight;
    int viewportX;
    int viewportY;
    uint viewportWidth;
    uint viewportHeight;
    uint sourceWidth;
    uint sourceHeight;
    uint sourceAllocationWidth;
    uint sourceAllocationHeight;
    uint sourceChannels;
    uint sourceFloat;
    uint sourceChw;
    uint flipY;
    uint hasScene;
    uint overDestination; // blend over the frame (environment) instead of the background
    float4 background;
};
[[vk::push_constant]] ConstantBuffer<ComposeParameters> p;

float readSceneChannel(uint x, uint y, uint channel) {
    x = min(x, max(p.sourceWidth, 1u) - 1u);
    y = min(y, max(p.sourceHeight, 1u) - 1u);
    uint index = p.sourceChw != 0
        ? channel * p.sourceAllocationWidth * p.sourceAllocationHeight + y * p.sourceAllocationWidth + x
        : (y * p.sourceAllocationWidth + x) * p.sourceChannels + channel;
    return p.sourceFloat != 0 ? ((Ptr<float, Access.Read>)p.scene)[index]
                              : float(p.scene[index]) * (1.0 / 255.0);
}

float4 sampleScene(float2 uv) {
    float2 extent = float2(max(p.sourceWidth, 1u), max(p.sourceHeight, 1u));
    // Texel centers sit at half-integers, as with a hardware linear sampler,
    // so a 1:1 scene image is copied without resampling.
    float2 pixel = clamp(uv * extent - 0.5, 0.0, extent - 1.0);
    uint2 a = uint2(floor(pixel));
    uint2 b = min(a + 1u, uint2(p.sourceWidth - 1u, p.sourceHeight - 1u));
    float2 f = frac(pixel);
    uint c1 = min(1u, p.sourceChannels - 1u), c2 = min(2u, p.sourceChannels - 1u);
    float4 c00 = float4(readSceneChannel(a.x, a.y, 0), readSceneChannel(a.x, a.y, c1), readSceneChannel(a.x, a.y, c2), p.sourceChannels == 4 ? readSceneChannel(a.x, a.y, 3) : 1.0);
    float4 c10 = float4(readSceneChannel(b.x, a.y, 0), readSceneChannel(b.x, a.y, c1), readSceneChannel(b.x, a.y, c2), p.sourceChannels == 4 ? readSceneChannel(b.x, a.y, 3) : 1.0);
    float4 c01 = float4(readSceneChannel(a.x, b.y, 0), readSceneChannel(a.x, b.y, c1), readSceneChannel(a.x, b.y, c2), p.sourceChannels == 4 ? readSceneChannel(a.x, b.y, 3) : 1.0);
    float4 c11 = float4(readSceneChannel(b.x, b.y, 0), readSceneChannel(b.x, b.y, c1), readSceneChannel(b.x, b.y, c2), p.sourceChannels == 4 ? readSceneChannel(b.x, b.y, 3) : 1.0);
    return lerp(lerp(c00, c10, f.x), lerp(c01, c11, f.x), f.y);
}

void writeDestination(uint2 pixel, float4 color) {
    uint i = (pixel.y * p.destinationWidth + pixel.x) * 4u;
    p.destination[i] = uint8_t(saturate(color.r) * 255.0 + 0.5);
    p.destination[i + 1] = uint8_t(saturate(color.g) * 255.0 + 0.5);
    p.destination[i + 2] = uint8_t(saturate(color.b) * 255.0 + 0.5);
    p.destination[i + 3] = uint8_t(saturate(color.a) * 255.0 + 0.5);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void composeViewport(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= p.destinationWidth || tid.y >= p.destinationHeight) return;
    int2 pixel = int2(tid.xy);
    bool inside = pixel.x >= p.viewportX && pixel.y >= p.viewportY &&
                  pixel.x < p.viewportX + int(p.viewportWidth) && pixel.y < p.viewportY + int(p.viewportHeight);
    // The graphics context cleared the frame; pixels outside the panel belong to the UI.
    if (!inside) return;
    float4 color = p.background;
    if (p.overDestination != 0) {
        uint i = (uint(pixel.y) * p.destinationWidth + uint(pixel.x)) * 4u;
        color = float4(p.destination[i], p.destination[i + 1], p.destination[i + 2], 255) / 255.0;
    }
    if (p.hasScene != 0) {
        float2 uv = (float2(pixel - int2(p.viewportX, p.viewportY)) + 0.5) /
                    float2(max(p.viewportWidth, 1u), max(p.viewportHeight, 1u));
        if (p.flipY != 0) uv.y = 1.0 - uv.y;
        float4 scene = sampleScene(uv);
        color.rgb = scene.rgb * scene.a + color.rgb * (1.0 - scene.a);
        color.a = 1.0;
    }
    writeDestination(tid.xy, color);
}
