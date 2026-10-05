// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
struct Parameters {
    Ptr<float2, Access.Read> positions;
    Ptr<float4, Access.Read> colors;
    Ptr<float2, Access.Read> texcoords;
    Ptr<uint8_t, Access.Read> texture;
    Ptr<uint8_t, Access.Read> mask;
    uint2 pointerPadding;
    column_major float4x4 transform;
    float2 translation;
    uint2 textureSize;
    uint2 targetSize;
    uint textured;
    uint maskEnabled;
    float maskValue;
    uint linearFilter;
};
[[vk::push_constant]] ConstantBuffer<Parameters> p;

struct VertexOutput {
    float4 position : SV_Position;
    float4 color : COLOR0;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VertexOutput uiVertex(uint vertexId : SV_VertexID) {
    VertexOutput result;
    result.position = mul(p.transform, float4(p.positions[vertexId] + p.translation, 0, 1));
    result.color = p.colors[vertexId];
    result.uv = p.texcoords[vertexId];
    return result;
}

float4 texel(int2 q) {
    q = clamp(q, int2(0), int2(max(p.textureSize, uint2(1u))) - 1);
    uint offset = (uint(q.y) * p.textureSize.x + uint(q.x)) * 4;
    return float4(p.texture[offset], p.texture[offset + 1],
                  p.texture[offset + 2], p.texture[offset + 3]) / 255.0;
}

// Clamp-to-edge like the Vulkan renderer's samplers: nearest for generated
// textures (glyph atlases), bilinear for loaded images and saved layers.
float4 sampleTexture(float2 uv) {
    if (p.linearFilter == 0)
        return texel(int2(floor(uv * float2(max(p.textureSize, uint2(1u))))));
    float2 position = uv * float2(max(p.textureSize, uint2(1u))) - 0.5;
    float2 base = floor(position);
    float2 f = position - base;
    int2 q = int2(base);
    float4 top = lerp(texel(q), texel(q + int2(1, 0)), f.x);
    float4 bottom = lerp(texel(q + int2(0, 1)), texel(q + int2(1, 1)), f.x);
    return lerp(top, bottom, f.y);
}

[shader("fragment")]
float4 uiFragment(VertexOutput input) : SV_Target {
    float4 color = input.color;
    if (p.textured != 0) color *= sampleTexture(input.uv);
    if (p.maskEnabled != 0) {
        uint2 q = uint2(input.position.xy);
        if (q.x >= p.targetSize.x || q.y >= p.targetSize.y) discard;
        color *= float(p.mask[(q.y * p.targetSize.x + q.x) * 4 + 3]) / 255.0;
    }
    return color;
}

[shader("fragment")]
float4 maskFragment(VertexOutput) : SV_Target {
    return float4(p.maskValue, p.maskValue, p.maskValue, p.maskValue);
}
