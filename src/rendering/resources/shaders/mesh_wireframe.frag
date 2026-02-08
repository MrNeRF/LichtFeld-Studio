#version 430 core

uniform vec3 u_color;

layout(location = 0) out vec4 frag_color;
layout(location = 1) out float frag_depth;

uniform vec3 u_camera_pos;

in vec3 v_world_pos;

void main() {
    frag_color = vec4(u_color, 1.0);
    frag_depth = length(u_camera_pos - v_world_pos);
}
