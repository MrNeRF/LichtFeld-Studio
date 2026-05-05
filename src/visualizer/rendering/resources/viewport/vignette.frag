#version 450
layout(location = 0) in vec2 TexCoord;
layout(location = 0) out vec4 FragColor;
layout(push_constant) uniform VignettePush {
    vec4 viewport_intensity_radius;
    vec4 softness_padding;
} u;

float vignette_alpha(vec2 screen_uv) {
    vec2 viewport = max(u.viewport_intensity_radius.xy, vec2(1.0, 1.0));
    float intensity = u.viewport_intensity_radius.z;
    float radius = u.viewport_intensity_radius.w;
    float softness = u.softness_padding.x;
    float min_dim = min(viewport.x, viewport.y);
    float fade_width = (1.0 - clamp(radius, 0.0, 1.0)) * 0.5 * min_dim;
    if (fade_width <= 0.0) {
        return 0.0;
    }

    vec2 half_extent = 0.5 * viewport;
    vec2 inner_half = max(half_extent - vec2(fade_width), vec2(0.0, 0.0));
    vec2 p = abs(screen_uv * viewport - half_extent) - inner_half;
    float dist = length(max(p, vec2(0.0, 0.0)));
    float visible = clamp(1.0 - dist / fade_width, 0.0, 1.0);
    visible = mix(visible, smoothstep(0.0, 1.0, visible), clamp(softness, 0.0, 1.0));
    return clamp(intensity, 0.0, 1.0) * (1.0 - visible);
}

void main() {
    FragColor = vec4(0.0, 0.0, 0.0, vignette_alpha(TexCoord));
}
