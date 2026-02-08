#version 430 core

uniform vec3 u_color;

layout(location = 0) out vec4 frag_color;

void main() {
    frag_color = vec4(u_color, 1.0);
}
