#version 450
layout(set = 0, binding = 0) uniform sampler2D sceneTexture;
layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 FragColor;
void main() {
    FragColor = texture(sceneTexture, TexCoord);
}
