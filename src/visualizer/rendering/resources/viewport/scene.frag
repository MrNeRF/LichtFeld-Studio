#version 450
layout(set = 0, binding = 0) uniform sampler2D sceneTexture;
layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 FragColor;

layout(push_constant) uniform ScenePush {
    vec2 uv_scale;
    vec2 uv_clamp_max;
} pc;

void main() {
    vec2 uv = min(TexCoord * pc.uv_scale, pc.uv_clamp_max);
    FragColor = texture(sceneTexture, uv);
}
