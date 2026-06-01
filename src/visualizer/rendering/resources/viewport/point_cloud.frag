/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#version 450

layout(location = 0) in vec3 v_color;
layout(location = 1) in float v_view_depth;

layout(location = 0) out vec4 frag_color;

layout(push_constant) uniform PushConstants {
    mat4 view_proj;
    mat4 view;
    mat4 crop_to_local;
    vec4 crop_min;             // w = depth_view_min
    vec4 crop_max;             // w = depth_view_max
    vec4 voxel_focal_ortho;    // w = depth_view enabled
    ivec4 counts;
} pc;

float normalizedDepth(float depth, float lo, float hi) {
    lo = max(lo, 1.0e-4);
    hi = max(hi, lo + 1.0e-4);
    depth = clamp(depth, lo, hi);

    const float linear_t = clamp((depth - lo) / max(hi - lo, 1.0e-5), 0.0, 1.0);
    const float log_span = max(log2(hi / lo), 1.0e-4);
    const float log_t = clamp(log2(depth / lo) / log_span, 0.0, 1.0);
    const float log_weight = smoothstep(1.75, 24.0, hi / lo);

    return smoothstep(0.0, 1.0, mix(linear_t, log_t, log_weight));
}

vec3 depthPalette(float near_t) {
    near_t = clamp(near_t, 0.0, 1.0);
    const vec3 far_0 = vec3(0.050, 0.040, 0.150);
    const vec3 far_1 = vec3(0.060, 0.195, 0.500);
    const vec3 mid_0 = vec3(0.000, 0.500, 0.650);
    const vec3 mid_1 = vec3(0.360, 0.735, 0.410);
    const vec3 near_0 = vec3(0.965, 0.820, 0.300);
    const vec3 near_1 = vec3(0.985, 0.430, 0.125);

    if (near_t < 0.20) {
        return mix(far_0, far_1, smoothstep(0.00, 0.20, near_t));
    }
    if (near_t < 0.43) {
        return mix(far_1, mid_0, smoothstep(0.20, 0.43, near_t));
    }
    if (near_t < 0.67) {
        return mix(mid_0, mid_1, smoothstep(0.43, 0.67, near_t));
    }
    if (near_t < 0.86) {
        return mix(mid_1, near_0, smoothstep(0.67, 0.86, near_t));
    }
    return mix(near_0, near_1, smoothstep(0.86, 1.00, near_t));
}

void main() {
    vec2 d = gl_PointCoord * 2.0 - 1.0;
    if (dot(d, d) > 1.0) {
        discard;
    }
    if (pc.voxel_focal_ortho.w != 0.0) {
        float lo = pc.crop_min.w;
        float hi = pc.crop_max.w;
        if (hi <= lo + 1.0e-5) {
            hi = lo + 1.0;
        }
        const float depth_t = normalizedDepth(v_view_depth, lo, hi);
        const float near_t = 1.0 - depth_t;
        vec3 depth_color = depthPalette(near_t);

        const float band = abs(fract(depth_t * 24.0) - 0.5);
        const float contour = smoothstep(0.475, 0.5, band);
        depth_color *= 1.0 - contour * 0.07;

        frag_color = vec4(depth_color, 1.0);
        return;
    }
    frag_color = vec4(v_color, 1.0);
}
