#version 430 core

layout(location = 0) in vec3 a_position;

uniform mat4 u_mvp;
uniform mat4 u_model;

out vec3 v_world_pos;

void main() {
    v_world_pos = vec3(u_model * vec4(a_position, 1.0));
    gl_Position = u_mvp * vec4(a_position, 1.0);
}
