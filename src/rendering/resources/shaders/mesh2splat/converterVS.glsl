/* Derived from Mesh2Splat by Electronic Arts Inc.
 * Original: Copyright (c) 2025 Electronic Arts Inc. All rights reserved.
 * Licensed under BSD 3-Clause (see THIRD_PARTY_LICENSES.md)
 *
 * Modifications: Copyright (c) 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#version 430 core

// Input attributes
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 tangent;
layout(location = 3) in vec2 uv;
layout(location = 4) in vec2 normalizedUv;
layout(location = 5) in vec3 scale;
layout(location = 6) in vec4 vertexColor;

// Output struct definition
out VS_OUT{
    vec3 position;
    vec3 normal;
    vec4 tangent;
    vec2 uv;
    vec2 normalizedUv;
    vec3 scale;
    vec4 vertexColor;
} vs_out;

void main() {
    vs_out.position = position;
    vs_out.normal = normal;
    vs_out.tangent = tangent;
    vs_out.uv = uv;
    vs_out.normalizedUv = normalizedUv;
    vs_out.scale = scale;
    vs_out.vertexColor = vertexColor;
    gl_Position = vec4(position, 1.0);
}
