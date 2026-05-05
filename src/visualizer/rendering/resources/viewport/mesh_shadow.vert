#version 450
layout(location = 0) in vec3 inPosition;

layout(push_constant) uniform ShadowPush {
    mat4 light_mvp;
} push;

void main() {
    gl_Position = push.light_mvp * vec4(inPosition, 1.0);
}
