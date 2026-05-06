#version 450
layout(location = 0) in vec3 inPosition;

layout(push_constant) uniform WirePush {
    mat4 mvp;
    vec4 color;
} push;

void main() {
    gl_Position = push.mvp * vec4(inPosition, 1.0);
}
