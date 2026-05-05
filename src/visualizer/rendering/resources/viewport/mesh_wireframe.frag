#version 450
layout(push_constant) uniform WirePush {
    layout(offset = 64) vec4 color;
} push;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(push.color.rgb, 1.0);
}
