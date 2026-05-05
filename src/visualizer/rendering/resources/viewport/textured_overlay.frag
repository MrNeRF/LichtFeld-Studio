#version 450
layout(set = 0, binding = 0) uniform sampler2D overlayTexture;
layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 FragColor;
layout(push_constant) uniform TexturedOverlayPush {
    vec4 tint_opacity;
    vec4 effects;
} u;

void main() {
    vec4 sampled = texture(overlayTexture, TexCoord);
    vec3 rgb = mix(sampled.rgb, u.tint_opacity.rgb, clamp(u.effects.x, 0.0, 1.0));
    rgb = mix(rgb, vec3(0.5), clamp(u.effects.y, 0.0, 1.0));
    FragColor = vec4(rgb, sampled.a * clamp(u.tint_opacity.a, 0.0, 1.0));
}
