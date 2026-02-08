#version 430 core

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec4 a_tangent;
layout(location = 3) in vec2 a_texcoord;
layout(location = 4) in vec4 a_color;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform mat3 u_normal_matrix;

out vec3 v_world_pos;
out vec3 v_normal;
out vec2 v_texcoord;
out vec4 v_color;
out mat3 v_tbn;

void main() {
    vec4 world_pos = u_model * vec4(a_position, 1.0);
    v_world_pos = world_pos.xyz;

    v_normal = normalize(u_normal_matrix * a_normal);
    v_texcoord = a_texcoord;
    v_color = a_color;

    if (length(a_tangent.xyz) > 0.0) {
        vec3 T = normalize(u_normal_matrix * a_tangent.xyz);
        vec3 N = v_normal;
        T = normalize(T - dot(T, N) * N);
        vec3 B = cross(N, T) * a_tangent.w;
        v_tbn = mat3(T, B, N);
    } else {
        v_tbn = mat3(1.0);
    }

    gl_Position = u_projection * u_view * world_pos;
}
