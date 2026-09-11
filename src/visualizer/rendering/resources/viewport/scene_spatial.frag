/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#version 450
layout(set = 0, binding = 0) uniform sampler2D sceneTexture;
layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 FragColor;

layout(push_constant) uniform ScenePush {
    vec2 uv_scale;
    vec2 uv_clamp_max;
} pc;

vec4 sampleScene(vec2 uv) {
    return texture(sceneTexture, clamp(uv, vec2(0.0), pc.uv_clamp_max));
}

void main() {
    const float strength = 0.18;
    vec2 uv = min(TexCoord * pc.uv_scale, pc.uv_clamp_max);
    vec2 texel = 1.0 / vec2(textureSize(sceneTexture, 0));
    vec4 center = sampleScene(uv);
    vec4 left = sampleScene(uv - vec2(texel.x, 0.0));
    vec4 right = sampleScene(uv + vec2(texel.x, 0.0));
    vec4 up = sampleScene(uv - vec2(0.0, texel.y));
    vec4 down = sampleScene(uv + vec2(0.0, texel.y));
    vec3 sharpened = center.rgb * (1.0 + 4.0 * strength) -
                     (left.rgb + right.rgb + up.rgb + down.rgb) * strength;
    vec3 neighborhood_min = min(center.rgb, min(min(left.rgb, right.rgb), min(up.rgb, down.rgb)));
    vec3 neighborhood_max = max(center.rgb, max(max(left.rgb, right.rgb), max(up.rgb, down.rgb)));
    FragColor = vec4(clamp(sharpened, neighborhood_min, neighborhood_max), center.a);
}
