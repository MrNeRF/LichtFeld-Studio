// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// Viewport meshes, ported from resources/viewport/mesh*.{vert,frag}. Vertices
// are pulled through the index buffer, so a draw's vertex count is its index
// count. Per-draw state lives in a small uniform tensor (see MeshUniform in
// viewport_tensor_meshes.cpp) to keep push constants within every backend's
// limit.
struct Parameters {
    // Four records per vertex: (position, u), (normal, v), tangent, color.
    Ptr<float4, Access.Read> vertices;
    Ptr<int, Access.Read> indices;
    Ptr<float4, Access.Read> uniforms;
    Ptr<uint8_t, Access.Read> albedo;
    Ptr<uint8_t, Access.Read> normalMap;
    Ptr<uint8_t, Access.Read> metallicRoughness;
    // Shadow-map depth, zero-to-one, row 0 at the top.
    Ptr<float, Access.Read> shadow;
    uint firstIndex;
    uint padding;
};
[[vk::push_constant]] ConstantBuffer<Parameters> p;

static const float PI = 3.14159265359;

// Uniform record indices.
static const uint U_MVP = 0;
static const uint U_MODEL = 4;
static const uint U_LIGHT_VP = 8;
static const uint U_CAMERA = 12;
static const uint U_LIGHT_DIR = 13;
static const uint U_LIGHT = 14;      // intensity, ambient, shadow enabled, shadow size
static const uint U_SELECTION = 15;  // emphasized, dim others, flash, wire width
static const uint U_BASE_COLOR = 16;
static const uint U_EMISSIVE_METALLIC = 17;
static const uint U_ROUGHNESS_FLAGS = 18; // roughness, has albedo, has normal, has metallic-roughness
static const uint U_VERTEX_COLORS = 19;
static const uint U_TEXTURE_SIZES = 20;   // albedo w,h, normal w,h
static const uint U_TEXTURE_SIZES2 = 21;  // metallic-roughness w,h
static const uint U_WIRE_COLOR = 22;

float4 uniformRecord(uint index) { return p.uniforms[index]; }

// Records hold matrix columns.
float4x4 uniformMatrix(uint index) {
    return transpose(float4x4(uniformRecord(index), uniformRecord(index + 1),
                              uniformRecord(index + 2), uniformRecord(index + 3)));
}

uint vertexIndex(uint vertexId) {
    return uint(p.indices[p.firstIndex + vertexId]);
}

float3x3 inverse3(float3x3 m) {
    float3 c0 = cross(m[1], m[2]), c1 = cross(m[2], m[0]), c2 = cross(m[0], m[1]);
    float determinant = dot(m[0], c0);
    return transpose(float3x3(c0, c1, c2)) / determinant;
}

struct MeshOutput {
    float4 position : SV_Position;
    float3 world : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float2 texcoord : TEXCOORD2;
    float4 color : COLOR0;
    float3 tangent : TEXCOORD3;
    float3 bitangent : TEXCOORD4;
};

[shader("vertex")]
MeshOutput meshVertex(uint vertexId : SV_VertexID) {
    uint v = vertexIndex(vertexId) * 4u;
    float4 a = p.vertices[v], b = p.vertices[v + 1], tangent = p.vertices[v + 2];
    float4x4 model = uniformMatrix(U_MODEL);
    MeshOutput result;
    result.world = mul(model, float4(a.xyz, 1.0)).xyz;
    // transpose(inverse(mat3(model))) handles non-uniform scale.
    float3x3 normalMatrix = transpose(inverse3((float3x3)model));
    result.normal = normalize(mul(normalMatrix, b.xyz));
    result.texcoord = float2(a.w, b.w);
    result.color = p.vertices[v + 3];
    if (length(tangent.xyz) > 0.0) {
        float3 t = normalize(mul(normalMatrix, tangent.xyz));
        t = normalize(t - dot(t, result.normal) * result.normal);
        result.tangent = t;
        result.bitangent = cross(result.normal, t) * tangent.w;
    } else {
        result.tangent = float3(1, 0, 0);
        result.bitangent = float3(0, 1, 0);
    }
    result.position = mul(uniformMatrix(U_MVP), float4(a.xyz, 1.0));
    return result;
}

// Bilinear, repeat wrapping, RGBA8.
float4 texel(Ptr<uint8_t, Access.Read> texture, int2 size, int2 q) {
    q = ((q % size) + size) % size;
    uint offset = (uint(q.y) * uint(size.x) + uint(q.x)) * 4u;
    return float4(texture[offset], texture[offset + 1], texture[offset + 2], texture[offset + 3]) / 255.0;
}

float4 sampleTexture(Ptr<uint8_t, Access.Read> texture, float2 size, float2 uv) {
    int2 extent = int2(max(size, float2(1.0)));
    float2 position = uv * float2(extent) - 0.5;
    float2 base = floor(position);
    float2 f = position - base;
    int2 q = int2(base);
    float4 top = lerp(texel(texture, extent, q), texel(texture, extent, q + int2(1, 0)), f.x);
    float4 bottom = lerp(texel(texture, extent, q + int2(0, 1)), texel(texture, extent, q + int2(1, 1)), f.x);
    return lerp(top, bottom, f.y);
}

float distributionGgx(float3 n, float3 h, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float nDotH = max(dot(n, h), 0.0);
    float denominator = nDotH * nDotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denominator * denominator);
}

float geometrySchlickGgx(float nDotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return nDotV / (nDotV * (1.0 - k) + k);
}

float geometrySmith(float3 n, float3 v, float3 l, float roughness) {
    return geometrySchlickGgx(max(dot(n, v), 0.0), roughness) * geometrySchlickGgx(max(dot(n, l), 0.0), roughness);
}

float3 fresnelSchlick(float cosTheta, float3 f0) {
    return f0 + (1.0 - f0) * pow(saturate(1.0 - cosTheta), 5.0);
}

// 3x3 percentage-closer filtering of the light-space depth map.
float shadowFactor(float3 world) {
    float4 light = mul(uniformMatrix(U_LIGHT_VP), float4(world, 1.0));
    float3 projected = light.xyz / light.w;
    float2 uv = float2(projected.x * 0.5 + 0.5, 0.5 - projected.y * 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || projected.z > 1.0)
        return 1.0;
    int size = int(uniformRecord(U_LIGHT).w);
    float lit = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            int2 q = clamp(int2(uv * float(size)) + int2(x, y), int2(0), int2(size - 1));
            float occluder = p.shadow[uint(q.y) * uint(size) + uint(q.x)];
            lit += projected.z <= occluder + 0.002 ? 1.0 : 0.0;
        }
    }
    return lit / 9.0;
}

[shader("fragment")]
float4 meshFragment(MeshOutput input, bool front : SV_IsFrontFace) : SV_Target {
    float4 flags = uniformRecord(U_ROUGHNESS_FLAGS);
    float4 sizes = uniformRecord(U_TEXTURE_SIZES);
    float4 albedo = uniformRecord(U_BASE_COLOR);
    if (flags.y > 0.5)
        albedo *= sampleTexture(p.albedo, sizes.xy, input.texcoord);
    if (uniformRecord(U_VERTEX_COLORS).x > 0.5)
        albedo *= input.color;

    float4 emissiveMetallic = uniformRecord(U_EMISSIVE_METALLIC);
    float metallic = emissiveMetallic.w;
    float roughness = flags.x;
    float ao = 1.0;
    if (flags.w > 0.5) {
        float3 orm = sampleTexture(p.metallicRoughness, uniformRecord(U_TEXTURE_SIZES2).xy, input.texcoord).rgb;
        ao = orm.r;
        roughness *= orm.g;
        metallic *= orm.b;
    }
    roughness = max(roughness, 0.04);

    float3 n = normalize(input.normal);
    if (flags.z > 0.5) {
        float3 sampled = sampleTexture(p.normalMap, sizes.zw, input.texcoord).rgb * 2.0 - 1.0;
        n = normalize(sampled.x * input.tangent + sampled.y * input.bitangent + sampled.z * normalize(input.normal));
    }

    float4 lightParams = uniformRecord(U_LIGHT);
    float3 v = normalize(uniformRecord(U_CAMERA).xyz - input.world);
    float3 l = normalize(uniformRecord(U_LIGHT_DIR).xyz);
    float3 h = normalize(v + l);
    float nDotL = max(dot(n, l), 0.0);
    float shadow = lightParams.z > 0.5 ? shadowFactor(input.world) : 1.0;

    float3 f0 = lerp(float3(0.04), albedo.rgb, metallic);
    float3 f = fresnelSchlick(max(dot(h, v), 0.0), f0);
    float3 diffuse = (1.0 - f) * (1.0 - metallic) * albedo.rgb / PI;
    float3 specular = (distributionGgx(n, h, roughness) * geometrySmith(n, v, l, roughness) * f) /
                      (4.0 * max(dot(n, v), 0.0) * nDotL + 0.0001);
    float3 color = albedo.rgb * lightParams.y * ao + (diffuse + specular) * nDotL * lightParams.x * shadow +
                   emissiveMetallic.xyz;
    color = color / (color + 1.0);
    color = pow(color, float3(1.0 / 2.2));

    float4 selection = uniformRecord(U_SELECTION);
    bool emphasized = selection.x > 0.5;
    if (selection.y > 0.5 && !emphasized)
        color = lerp(color, float3(dot(color, float3(0.2126, 0.7152, 0.0722))), 0.6);
    if (emphasized && selection.z > 0.0)
        color = lerp(color, float3(1.0, 0.95, 0.6), selection.z * 0.5);
    return float4(color, albedo.a);
}

// --- Wireframe overlay: triangle edges from per-vertex barycentrics -------------

struct WireOutput {
    float4 position : SV_Position;
    float3 barycentric : TEXCOORD0;
};

[shader("vertex")]
WireOutput wireVertex(uint vertexId : SV_VertexID) {
    uint v = vertexIndex(vertexId) * 4u;
    WireOutput result;
    result.position = mul(uniformMatrix(U_MVP), float4(p.vertices[v].xyz, 1.0));
    uint corner = vertexId % 3u;
    result.barycentric = float3(corner == 0u, corner == 1u, corner == 2u);
    return result;
}

[shader("fragment")]
float4 wireFragment(WireOutput input) : SV_Target {
    // fwidth, spelled out: Slang's Metal target lacks it.
    float3 width = abs(ddx(input.barycentric)) + abs(ddy(input.barycentric));
    float3 distance = input.barycentric / max(width, float3(1e-6));
    float edge = min(distance.x, min(distance.y, distance.z));
    float halfWidth = max(uniformRecord(U_SELECTION).w, 1.0) * 0.5;
    float alpha = 1.0 - smoothstep(halfWidth - 0.5, halfWidth + 0.5, edge);
    if (alpha <= 0.001)
        discard;
    return float4(uniformRecord(U_WIRE_COLOR).rgb, alpha);
}

// --- Shadow map: light-space depth only ------------------------------------------

struct ShadowOutput {
    float4 position : SV_Position;
};

[shader("vertex")]
ShadowOutput shadowVertex(uint vertexId : SV_VertexID) {
    uint v = vertexIndex(vertexId) * 4u;
    ShadowOutput result;
    result.position = mul(uniformMatrix(U_MVP), float4(p.vertices[v].xyz, 1.0));
    return result;
}

[shader("fragment")]
float4 shadowFragment(ShadowOutput input) : SV_Target {
    return float4(0.0);
}
