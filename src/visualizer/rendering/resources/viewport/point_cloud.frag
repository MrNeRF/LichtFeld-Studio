/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#version 450

layout(location = 0) in vec3 v_color;

layout(location = 0) out vec4 frag_color;

void main() {
    vec2 d = gl_PointCoord * 2.0 - 1.0;
    if (dot(d, d) > 1.0) {
        discard;
    }
    frag_color = vec4(v_color, 1.0);
}
