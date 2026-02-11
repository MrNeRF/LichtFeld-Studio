/* Derived from Mesh2Splat by Electronic Arts Inc.
 * Original: Copyright (c) 2025 Electronic Arts Inc. All rights reserved.
 * Licensed under BSD 3-Clause (see THIRD_PARTY_LICENSES.md)
 *
 * Modifications: Copyright (c) 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#version 430 core

uniform sampler2D albedoTexture;
uniform sampler2D normalTexture;
uniform sampler2D metallicRoughnessTexture;
uniform sampler2D occlusionTexture;
uniform sampler2D emissiveTexture;

uniform int hasAlbedoMap;
uniform int hasNormalMap;
uniform int hasMetallicRoughnessMap;
uniform int hasVertexColors;
uniform vec4 u_materialFactor;
uniform float u_metallicFactor;
uniform float u_roughnessFactor;
uniform vec3 u_lightDir;
uniform float u_lightIntensity;
uniform float u_ambient;

struct GaussianVertex {
    vec4 position;
    vec4 color;
    vec4 scale;
    vec4 normal;
    vec4 rotation;
    vec4 pbr;
};

layout(std430, binding = 0) buffer GaussianBuffer {
    GaussianVertex vertices[];
} gaussianBuffer;

layout(binding = 1) uniform atomic_uint g_validCounter;

// Inputs from the geometry shader
in vec3 Position;
flat in vec3 Scale;
in vec2 UV;
in vec4 Tangent;
in vec3 Normal;
in vec4 VertexColor;
flat in vec4 Quaternion;

const float PI = 3.14159265359;

float distribution_ggx(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float denom = NdotH2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

float geometry_schlick_ggx(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float geometry_smith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    return geometry_schlick_ggx(NdotV, roughness) * geometry_schlick_ggx(NdotL, roughness);
}

vec3 fresnel_schlick(float cos_theta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

void main() {
    uint index = atomicCounterIncrement(g_validCounter);

    vec4 out_Color = vec4(1);
    if (hasAlbedoMap == 1)
        out_Color = texture(albedoTexture, UV);
    else if (hasVertexColors == 1)
        out_Color = VertexColor;

    vec3 out_Normal;
    if (hasNormalMap == 1) {
        vec3 normalMap_normal = texture(normalTexture, UV).xyz;
        vec3 retrievedNormal = normalize(normalMap_normal * 2.0 - 1.0);
        vec3 bitangent = normalize(cross(Normal, Tangent.xyz)) * Tangent.w;
        mat3 TBN = mat3(Tangent.xyz, bitangent, normalize(Normal));
        out_Normal = normalize(TBN * retrievedNormal);
    } else {
        out_Normal = normalize(Normal);
    }

    float metallic = u_metallicFactor;
    float roughness = u_roughnessFactor;
    float ao = 1.0;
    if (hasMetallicRoughnessMap == 1) {
        vec3 orm = texture(metallicRoughnessTexture, UV).rgb;
        ao = orm.r;
        roughness *= orm.g;
        metallic *= orm.b;
    }
    roughness = max(roughness, 0.04);

    vec3 albedo = (out_Color * u_materialFactor).rgb;
    vec3 N = out_Normal;

    vec3 V = u_lightDir;
    vec3 L = u_lightDir;

    if (dot(N, L) < 0.0)
        N = -N;

    vec3 H = normalize(V + L);

    float NdotL = max(dot(N, L), 0.0);

    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    float NDF = distribution_ggx(N, H, roughness);
    float G = geometry_smith(N, V, L, roughness);
    vec3 F = fresnel_schlick(max(dot(H, V), 0.0), F0);

    vec3 kS = F;
    vec3 kD = (1.0 - kS) * (1.0 - metallic);

    vec3 diffuse = kD * albedo / PI;
    vec3 spec = (NDF * G * F) / (4.0 * max(dot(N, V), 0.0) * NdotL + 0.0001);

    vec3 Lo = (diffuse + spec) * NdotL * u_lightIntensity;
    vec3 ambient_color = albedo * u_ambient * ao;
    vec3 color = ambient_color + Lo;

    color = pow(clamp(color, 0.0, 1.0), vec3(1.0 / 2.2));

    gaussianBuffer.vertices[index].position = vec4(Position.xyz, 1);
    gaussianBuffer.vertices[index].color = vec4(color, out_Color.a);
    gaussianBuffer.vertices[index].scale = vec4(Scale, 0.0);
    gaussianBuffer.vertices[index].normal = vec4(out_Normal, 0.0);
    gaussianBuffer.vertices[index].rotation = Quaternion;
    gaussianBuffer.vertices[index].pbr = vec4(metallic, roughness, 0, 1);
}
