/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#version 450

layout(location = 0) in vec2 TexCoord;
layout(set = 0, binding = 0) uniform sampler2D u_depth;

layout(push_constant) uniform Push {
    vec4 params; // x = near, y = far, z = is_view_depth (1) vs ndc (0), w = flip_y
} pc;

float view_depth_to_ndc(float z) {
    float n = pc.params.x;
    float f = pc.params.y;
    return f / (f - n) * (1.0 - n / max(z, n));
}

void main() {
    vec2 uv = TexCoord;
    if (pc.params.w > 0.5) {
        uv.y = 1.0 - uv.y;
    }
    float d = texture(u_depth, uv).r;
    float ndc;
    if (pc.params.z > 0.5) {
        // view-space depth (CUDA rasterizer default): convert to NDC
        if (d <= 0.0 || d >= 1e9) {
            // sentinel: leave depth at far (no occlusion contribution)
            discard;
        }
        ndc = view_depth_to_ndc(d);
    } else {
        if (d >= 1.0) {
            discard;
        }
        ndc = d;
    }
    gl_FragDepth = clamp(ndc, 0.0, 1.0);
}
