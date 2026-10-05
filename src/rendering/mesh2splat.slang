// Derived from Mesh2Splat by Electronic Arts Inc.
// Original: Copyright (c) 2025 Electronic Arts Inc. All rights reserved.
// Licensed under BSD 3-Clause (see THIRD_PARTY_LICENSES.md)
// Modifications: Copyright (c) 2025-2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// Port of the Vulkan converter's GLSL pipeline (mesh2splat.cpp). Metal has no
// geometry stage, so the vertex stage pulls its whole triangle and recomputes
// the per-triangle values the geometry shader emitted; every vertex of a
// triangle computes them from the same inputs, so the flat outputs agree.

struct Parameters {
    Ptr<float, Access.Read> vertices; // PerVertexData, 21 floats per vertex
    Ptr<float4, Access.ReadWrite> gaussians; // GaussianVertex, 6 float4 each
    Ptr<uint, Access.ReadWrite> counter;
    Ptr<uint8_t, Access.Read> albedoTexture; // RGBA8 texels
    Ptr<uint8_t, Access.Read> normalTexture;
    Ptr<uint8_t, Access.Read> metallicRoughnessTexture;
    float4 materialFactor;
    float4 bboxMinMetallic;
    float4 bboxMaxRoughness;
    // x: albedo texture (1 linear, 2 sRGB), y: normal texture,
    // z: metallic-roughness texture, w: vertex colors.
    uint4 flags;
    uint2 albedoSize;
    uint2 normalSize;
    uint2 metallicRoughnessSize;
    uint capacity;
    uint padding;
};
[[vk::push_constant]] ConstantBuffer<Parameters> p;

struct VertexOutput {
    float4 clipPosition : SV_Position;
    float3 position : TEXCOORD0;
    nointerpolation float3 scale : TEXCOORD1;
    float2 uv : TEXCOORD2;
    float4 tangent : TEXCOORD3;
    float4 vertexColor : COLOR0;
    float3 normal : TEXCOORD4;
    nointerpolation float4 quaternion : TEXCOORD5;
};

float vertexAttribute(uint index, uint component) {
    return p.vertices[index * 21u + component];
}

float3 vertexPosition(uint index) {
    return float3(vertexAttribute(index, 0u), vertexAttribute(index, 1u), vertexAttribute(index, 2u));
}

// glm::quat_cast of the column-major matrix (xAxis, yAxis, zAxis); returns (x, y, z, w).
float4 quatCast(float3 xAxis, float3 yAxis, float3 zAxis) {
    float fourXSquaredMinus1 = xAxis.x - yAxis.y - zAxis.z;
    float fourYSquaredMinus1 = yAxis.y - xAxis.x - zAxis.z;
    float fourZSquaredMinus1 = zAxis.z - xAxis.x - yAxis.y;
    float fourWSquaredMinus1 = xAxis.x + yAxis.y + zAxis.z;

    int biggestIndex = 0;
    float fourBiggestSquaredMinus1 = fourWSquaredMinus1;
    if (fourXSquaredMinus1 > fourBiggestSquaredMinus1) {
        fourBiggestSquaredMinus1 = fourXSquaredMinus1;
        biggestIndex = 1;
    }
    if (fourYSquaredMinus1 > fourBiggestSquaredMinus1) {
        fourBiggestSquaredMinus1 = fourYSquaredMinus1;
        biggestIndex = 2;
    }
    if (fourZSquaredMinus1 > fourBiggestSquaredMinus1) {
        fourBiggestSquaredMinus1 = fourZSquaredMinus1;
        biggestIndex = 3;
    }

    float biggestVal = sqrt(fourBiggestSquaredMinus1 + 1.0f) * 0.5f;
    float mult = 0.25f / biggestVal;
    float4 q;
    if (biggestIndex == 0) {
        q.w = biggestVal;
        q.x = (yAxis.z - zAxis.y) * mult;
        q.y = (zAxis.x - xAxis.z) * mult;
        q.z = (xAxis.y - yAxis.x) * mult;
    } else if (biggestIndex == 1) {
        q.w = (yAxis.z - zAxis.y) * mult;
        q.x = biggestVal;
        q.y = (xAxis.y + yAxis.x) * mult;
        q.z = (zAxis.x + xAxis.z) * mult;
    } else if (biggestIndex == 2) {
        q.w = (zAxis.x - xAxis.z) * mult;
        q.x = (xAxis.y + yAxis.x) * mult;
        q.y = biggestVal;
        q.z = (yAxis.z + zAxis.y) * mult;
    } else {
        q.w = (xAxis.y - yAxis.x) * mult;
        q.x = (zAxis.x + xAxis.z) * mult;
        q.y = (yAxis.z + zAxis.y) * mult;
        q.z = biggestVal;
    }
    return q;
}

[shader("vertex")]
VertexOutput mesh2splatVertex(uint vertexId : SV_VertexID) {
    uint first = vertexId / 3u * 3u;
    uint corner = vertexId - first;
    float3 vPosition[3] = { vertexPosition(first), vertexPosition(first + 1u), vertexPosition(first + 2u) };

    float3 edge1 = vPosition[1] - vPosition[0];
    float3 edge2 = vPosition[2] - vPosition[0];
    float3 edge3 = vPosition[2] - vPosition[1];

    float3 bboxMin = p.bboxMinMetallic.xyz;
    float3 bboxMax = p.bboxMaxRoughness.xyz;

    if (length(edge2) > length(edge1) && length(edge2) > length(edge3)) {
        float3 temp = edge1;
        edge1 = edge2;
        edge2 = temp;
    } else if (length(edge3) > length(edge1) && length(edge3) > length(edge2)) {
        float3 temp = edge1;
        edge1 = edge3;
        edge3 = temp;
    }

    edge1 = normalize(edge1);
    float3 normal = normalize(cross(edge1, edge2));

    float absX = abs(normal.x);
    float absY = abs(normal.y);
    float absZ = abs(normal.z);

    float2 orthogonalUvs[3];
    for (int i = 0; i < 3; i++) {
        float u, v;
        float3 pos = vPosition[i];
        if (absX > absY && absX > absZ) {
            float range = max(bboxMax.y - bboxMin.y, bboxMax.z - bboxMin.z);
            u = (pos.y - bboxMin.y) / range;
            v = (pos.z - bboxMin.z) / range;
        } else if (absY > absZ) {
            float range = max(bboxMax.x - bboxMin.x, bboxMax.z - bboxMin.z);
            u = (pos.x - bboxMin.x) / range;
            v = (pos.z - bboxMin.z) / range;
        } else {
            float range = max(bboxMax.x - bboxMin.x, bboxMax.y - bboxMin.y);
            u = (pos.x - bboxMin.x) / range;
            v = (pos.y - bboxMin.y) / range;
        }
        orthogonalUvs[i] = float2(u, v);
    }

    float3 xAxis = edge1;
    float3 yAxis = normalize(cross(normal, xAxis));
    float3 zAxis = normal;
    float4 q = quatCast(xAxis, yAxis, zAxis);

    // J = V * inverse(UV) as in computeUv3DJacobian: V's columns are the
    // triangle edges, UV's columns the projected UV edges.
    float2 uvEdge1 = orthogonalUvs[1] - orthogonalUvs[0];
    float2 uvEdge2 = orthogonalUvs[2] - orthogonalUvs[0];
    float determinant = uvEdge1.x * uvEdge2.y - uvEdge1.y * uvEdge2.x;
    // inverse2x2(UVMatrix), indexed [column][row] like the GLSL matrix.
    float inv00 = 0.0, inv01 = 0.0, inv10 = 0.0, inv11 = 0.0;
    if (determinant != 0.0) {
        float invDet = 1.0 / determinant;
        inv00 = uvEdge2.y * invDet;
        inv10 = -uvEdge2.x * invDet;
        inv01 = -uvEdge1.y * invDet;
        inv11 = uvEdge1.x * invDet;
    }
    float3 positionEdge1 = vPosition[1] - vPosition[0];
    float3 positionEdge2 = vPosition[2] - vPosition[0];
    float3 Ju = positionEdge1 * inv00 + positionEdge2 * inv01;
    float3 Jv = positionEdge1 * inv10 + positionEdge2 * inv11;

    VertexOutput result;
    result.position = vPosition[corner];
    result.normal = float3(vertexAttribute(vertexId, 3u), vertexAttribute(vertexId, 4u), vertexAttribute(vertexId, 5u));
    result.tangent = float4(vertexAttribute(vertexId, 6u), vertexAttribute(vertexId, 7u),
                            vertexAttribute(vertexId, 8u), vertexAttribute(vertexId, 9u));
    result.uv = float2(vertexAttribute(vertexId, 10u), vertexAttribute(vertexId, 11u));
    result.vertexColor = float4(vertexAttribute(vertexId, 17u), vertexAttribute(vertexId, 18u),
                                vertexAttribute(vertexId, 19u), vertexAttribute(vertexId, 20u));
    result.scale = float3(length(Ju), length(Jv), 1e-7);
    result.quaternion = float4(q.w, q.x, q.y, q.z);
    // Vulkan clip space points +Y down, tensor raster clip space +Y up: negate
    // y so each triangle covers the Vulkan framebuffer's pixels and edge ties
    // resolve alike.
    float2 clip = orthogonalUvs[corner] * 2.0 - 1.0;
    result.clipPosition = float4(clip.x, -clip.y, 0.0, 1.0);
    return result;
}

float srgbToLinear(float value) {
    return value <= 0.04045 ? value / 12.92 : pow((value + 0.055) / 1.055, 2.4);
}

float4 texel(Ptr<uint8_t, Access.Read> texture, int2 size, int2 q, bool srgb) {
    q = ((q % size) + size) % size;
    uint at = (uint(q.y) * uint(size.x) + uint(q.x)) * 4u;
    float4 value = float4(texture[at], texture[at + 1u], texture[at + 2u], texture[at + 3u]) / 255.0;
    if (srgb)
        value.rgb = float3(srgbToLinear(value.r), srgbToLinear(value.g), srgbToLinear(value.b));
    return value;
}

// The Vulkan sampler: bilinear, repeat, one mip level; sRGB textures decode
// per texel before filtering.
float4 sampleTexture(Ptr<uint8_t, Access.Read> texture, uint2 dimensions, float2 uv, bool srgb) {
    int2 size = int2(dimensions);
    float2 position = uv * float2(size) - 0.5;
    float2 base = floor(position);
    float2 f = position - base;
    int2 q = int2(base);
    float4 top = lerp(texel(texture, size, q, srgb), texel(texture, size, q + int2(1, 0), srgb), f.x);
    float4 bottom = lerp(texel(texture, size, q + int2(0, 1), srgb), texel(texture, size, q + int2(1, 1), srgb), f.x);
    return lerp(top, bottom, f.y);
}

[shader("fragment")]
float4 mesh2splatFragment(VertexOutput input) : SV_Target {
    uint index;
    InterlockedAdd(p.counter[0], 1u, index);
    if (index >= p.capacity)
        return float4(0.0);

    float4 outColor = float4(1.0);
    if (p.flags.x != 0u) {
        outColor = sampleTexture(p.albedoTexture, p.albedoSize, input.uv, p.flags.x == 2u);
    } else if (p.flags.w == 1u) {
        outColor = input.vertexColor;
    }

    float3 outNormal;
    if (p.flags.y == 1u) {
        float3 normalMapNormal = sampleTexture(p.normalTexture, p.normalSize, input.uv, false).xyz;
        float3 retrievedNormal = normalize(normalMapNormal * 2.0 - 1.0);
        float3 bitangent = normalize(cross(input.normal, input.tangent.xyz)) * input.tangent.w;
        // TBN * retrievedNormal with TBN = mat3(tangent, bitangent, normal) as columns.
        outNormal = normalize(input.tangent.xyz * retrievedNormal.x + bitangent * retrievedNormal.y +
                              normalize(input.normal) * retrievedNormal.z);
    } else {
        outNormal = normalize(input.normal);
    }

    float metallic = p.bboxMinMetallic.w;
    float roughness = p.bboxMaxRoughness.w;
    if (p.flags.z == 1u) {
        float3 orm = sampleTexture(p.metallicRoughnessTexture, p.metallicRoughnessSize, input.uv, false).rgb;
        roughness *= orm.g;
        metallic *= orm.b;
    }
    roughness = max(roughness, 0.04);

    float3 albedo = (outColor * p.materialFactor).rgb;
    float3 color = pow(clamp(albedo, 0.0, 1.0), float3(1.0 / 2.2));

    uint at = index * 6u;
    p.gaussians[at] = float4(input.position, 1.0);
    p.gaussians[at + 1u] = float4(color, outColor.a);
    p.gaussians[at + 2u] = float4(input.scale, 0.0);
    p.gaussians[at + 3u] = float4(outNormal, 0.0);
    p.gaussians[at + 4u] = input.quaternion;
    p.gaussians[at + 5u] = float4(metallic, roughness, 0.0, 1.0);
    return float4(0.0);
}
