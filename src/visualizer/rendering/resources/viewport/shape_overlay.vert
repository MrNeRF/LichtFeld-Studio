#version 450
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aScreenPos;
layout(location = 2) in vec2 aP0;
layout(location = 3) in vec2 aP1;
layout(location = 4) in vec4 aColor;
layout(location = 5) in vec4 aParams;
layout(location = 0) out vec2 ScreenPos;
layout(location = 1) out vec2 P0;
layout(location = 2) out vec2 P1;
layout(location = 3) out vec4 Color;
layout(location = 4) out vec4 Params;
void main() {
    gl_Position = vec4(aPos.xy, 0.0, 1.0);
    ScreenPos = aScreenPos;
    P0 = aP0;
    P1 = aP1;
    Color = aColor;
    Params = aParams;
}
