/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#version 450

layout(set = 0, binding = 0) uniform sampler2D sceneTexture;
layout(location = 0) out vec4 FragColor;

layout(push_constant) uniform ReprojectPush {
    // Upper-left 3x3: current NDC to source NDC, see sceneReprojectionHomography.
    mat4 current_to_source;
    vec4 viewport_rect;
    vec4 color_uv_region; // xy = uv scale, zw = uv clamp max
    vec4 flip_y;          // x: color rows are flipped
} pc;

void main() {
    vec2 frag_uv = (gl_FragCoord.xy - pc.viewport_rect.xy) / pc.viewport_rect.zw;
    vec2 target = vec2(frag_uv.x, 1.0 - frag_uv.y) * 2.0 - 1.0;
    vec3 source = mat3(pc.current_to_source) * vec3(target, 1.0);
    // The last render never saw this pixel: leave the viewport background rather
    // than stretching the image border across it.
    if (source.z <= 0.0) {
        discard;
    }
    vec2 ndc = source.xy / source.z;
    if (any(greaterThan(abs(ndc), vec2(1.0)))) {
        discard;
    }
    vec2 uv = vec2(ndc.x, -ndc.y) * 0.5 + 0.5;
    if (pc.flip_y.x > 0.5) {
        uv.y = 1.0 - uv.y;
    }
    FragColor = texture(sceneTexture, min(uv * pc.color_uv_region.xy, pc.color_uv_region.zw));
}
