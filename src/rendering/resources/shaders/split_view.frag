#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D leftTexture;
uniform sampler2D rightTexture;
uniform float splitPosition;
uniform bool showDivider;
uniform vec4 dividerColor;
uniform float dividerWidth;
uniform vec2 leftTexcoordScale;
uniform vec2 rightTexcoordScale;

void main() {
    vec2 uv = TexCoord;

    vec4 leftColor = texture(leftTexture, uv * leftTexcoordScale);
    vec4 rightColor = texture(rightTexture, uv * rightTexcoordScale);

    vec4 color = (uv.x < splitPosition) ? leftColor : rightColor;

    if (showDivider && abs(uv.x - splitPosition) < dividerWidth * 0.5) {
        color = dividerColor;
    }

    FragColor = color;
}
