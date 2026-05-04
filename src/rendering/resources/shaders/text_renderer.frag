#version 330 core
in vec2 TexCoords;
out vec4 color;

uniform sampler2D text;
uniform vec3 textColor;
uniform float textAlpha = 1.0;

void main()
{
    float a = texture(text, TexCoords).r * textAlpha;
    color = vec4(textColor, 1.0) * vec4(1.0, 1.0, 1.0, a);
}
