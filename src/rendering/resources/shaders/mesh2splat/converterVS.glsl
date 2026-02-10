///////////////////////////////////////////////////////////////////////////////
//         Mesh2Splat: fast mesh to 3D gaussian splat conversion             //
//        Copyright (c) 2025 Electronic Arts Inc. All rights reserved.       //
///////////////////////////////////////////////////////////////////////////////

#version 430 core

// Input attributes
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 tangent;
layout(location = 3) in vec2 uv;
layout(location = 4) in vec2 normalizedUv;
layout(location = 5) in vec3 scale;


// Output struct definition
out VS_OUT{
    vec3 position;
    vec3 normal;
    vec4 tangent;
    vec2 uv;
    vec2 normalizedUv;
    vec3 scale;
} vs_out;

void main() {
    vs_out.position = position;
    vs_out.normal = normal;
    vs_out.tangent = tangent;
    vs_out.uv = uv;
    vs_out.normalizedUv = normalizedUv;
    vs_out.scale = scale;
}
