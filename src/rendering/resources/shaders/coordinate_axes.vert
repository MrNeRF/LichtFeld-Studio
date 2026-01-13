#version 330 core

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_color;

uniform mat4 u_mvp;
uniform mat4 u_view;
uniform bool u_equirectangular;

#define PI 3.14159265359

out vec3 v_color;
out float v_ndc_x;
out int v_equi;

void main() {
    v_equi = u_equirectangular ? 1 : 0;
    v_ndc_x = 0.0;

    if (u_equirectangular) {
        vec4 view_pos = u_view * vec4(a_position, 1.0);
        vec3 dir = normalize(view_pos.xyz);
        float theta = atan(dir.x, -dir.z);
        float phi = asin(clamp(dir.y, -1.0, 1.0));
        float depth = length(view_pos.xyz);
        v_ndc_x = theta / PI;
        gl_Position = vec4(v_ndc_x, -phi / (PI * 0.5), -1.0 / depth, 1.0);
    } else {
        gl_Position = u_mvp * vec4(a_position, 1.0);
    }
    v_color = a_color;
}
