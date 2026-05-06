/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#version 450

layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 frag_color;
layout(set = 0, binding = 0) uniform sampler2D u_left;
layout(set = 0, binding = 1) uniform sampler2D u_right;

layout(push_constant) uniform Push {
    // x = split_position (0..1 in viewport space)
    // y = left_flip_y (0/1)
    // z = right_flip_y (0/1)
    // w = divider_color_rgb_packed (unused; split into two vec4 below)
    vec4 split;

    // Viewport pixel rect (x, y, width, height) — letterboxed content extent.
    vec4 rect;

    // Panel x-range in viewport-uv space: (left_start, left_end, right_start, right_end).
    // When normalize_x_to_panel is true on either side, sampling uses (u - start) / (end - start).
    vec4 panel_norm;
    // Per-panel "normalize_x_to_panel" flags packed as (left, right, _, _).
    vec4 panel_flags;

    // Background color for letterboxed regions.
    vec4 background;

    // Divider visual constants: bar half-width, handle half-width, handle half-height,
    // corner radius (px).
    vec4 divider;
    // Grip line config: spacing, half-width, half-length, line count (rounded up).
    vec4 grip;
} pc;

vec3 sample_panel(sampler2D tex, vec2 uv, float start, float end, float normalize, float flip_y) {
    float u = uv.x;
    if (normalize > 0.5) {
        float span = max(end - start, 1e-6);
        u = (uv.x - start) / span;
    }
    float v = flip_y > 0.5 ? 1.0 - uv.y : uv.y;
    return texture(tex, vec2(u, v)).rgb;
}

void main() {
    // Pixel-space coordinate (gl_FragCoord origin top-left in Vulkan).
    vec2 px = gl_FragCoord.xy;

    // Letterbox: outside content rect → background.
    if (px.x < pc.rect.x || px.x >= pc.rect.x + pc.rect.z ||
        px.y < pc.rect.y || px.y >= pc.rect.y + pc.rect.w) {
        frag_color = vec4(pc.background.rgb, 1.0);
        return;
    }

    // UV inside the content rect (0..1).
    vec2 content_uv = vec2(
        pc.rect.z > 1.0 ? (px.x - pc.rect.x) / (pc.rect.z - 1.0) : 0.0,
        pc.rect.w > 1.0 ? (px.y - pc.rect.y) / (pc.rect.w - 1.0) : 0.0);

    float split_x = pc.rect.x + clamp(pc.split.x, 0.0, 1.0) * pc.rect.z;
    float divider_pixel = pc.rect.x + floor(pc.split.x * pc.rect.z + 0.5);
    bool use_left = px.x < divider_pixel;

    vec3 color = use_left
        ? sample_panel(u_left, content_uv, pc.panel_norm.x, pc.panel_norm.y,
                       pc.panel_flags.x, pc.split.y)
        : sample_panel(u_right, content_uv, pc.panel_norm.z, pc.panel_norm.w,
                       pc.panel_flags.y, pc.split.z);

    // Divider/handle/grip overlay. Mirrors compositeSplitImages CPU geometry
    // pixel-for-pixel: vertical bar + rounded handle + horizontal grip lines.
    float dist_from_split = abs(px.x - split_x);
    if (dist_from_split < pc.divider.x) {
        vec3 divider_color = vec3(0.29, 0.33, 0.42);
        vec3 out_color = divider_color;

        float center_y = pc.rect.y + pc.rect.w * 0.5;
        float dist_from_center = abs(px.y - center_y);
        float handle_w = min(pc.divider.y * 2.0, pc.rect.z) * 0.5;
        float handle_h = min(pc.divider.z * 2.0, pc.rect.w) * 0.5;

        if (dist_from_center < handle_h && dist_from_split < handle_w) {
            vec2 local = vec2(dist_from_split, dist_from_center);
            float corner_radius = min(pc.divider.w, min(handle_w, handle_h));
            vec2 corner_dist = local - (vec2(handle_w, handle_h) - vec2(corner_radius));
            if (corner_dist.x <= 0.0 || corner_dist.y <= 0.0 ||
                length(corner_dist) <= corner_radius) {
                out_color = divider_color * 0.8;

                float local_y = px.y - center_y;
                float spacing = pc.grip.x;
                float half_w = pc.grip.y;
                float half_l = pc.grip.z;
                int line_count = int(pc.grip.w);
                for (int i = -line_count; i <= line_count; ++i) {
                    float line_y = float(i) * spacing;
                    if (abs(local_y - line_y) < half_w &&
                        dist_from_split < half_l) {
                        out_color = vec3(0.9);
                        break;
                    }
                }
            }
        }
        frag_color = vec4(out_color, 1.0);
        return;
    }

    frag_color = vec4(color, 1.0);
}
