#version 430 core

in vec2 TexCoord;

uniform vec2 u_viewport_size;
uniform float u_vignette_intensity;
uniform float u_vignette_radius;
uniform float u_vignette_softness;

out vec4 FragColor;

float vignette_alpha(vec2 screen_uv) {
    vec2 viewport = max(u_viewport_size, vec2(1.0, 1.0));
    float min_dim = min(viewport.x, viewport.y);
    float fade_width = (1.0 - clamp(u_vignette_radius, 0.0, 1.0)) * 0.5 * min_dim;
    if (fade_width <= 0.0) {
        return 0.0;
    }

    vec2 half_extent = 0.5 * viewport;
    vec2 inner_half = max(half_extent - vec2(fade_width), vec2(0.0, 0.0));
    vec2 p = abs(screen_uv * viewport - half_extent) - inner_half;
    float dist = length(max(p, vec2(0.0, 0.0)));
    float visible = clamp(1.0 - dist / fade_width, 0.0, 1.0);
    visible = mix(visible, smoothstep(0.0, 1.0, visible), clamp(u_vignette_softness, 0.0, 1.0));
    return clamp(u_vignette_intensity, 0.0, 1.0) * (1.0 - visible);
}

void main() {
    FragColor = vec4(0.0, 0.0, 0.0, vignette_alpha(TexCoord));
}
