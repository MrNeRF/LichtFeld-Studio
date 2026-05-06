#version 450
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec4 aColor;
layout(location = 0) out vec4 Color;
void main() {
    gl_Position = vec4(aPos.xy, 0.0, 1.0);
    Color = aColor;
}
