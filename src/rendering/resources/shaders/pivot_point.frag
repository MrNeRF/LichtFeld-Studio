#version 330 core

uniform vec3 u_color;
uniform float u_opacity;

in vec2 v_uv;
out vec4 FragColor;

const float DOT_RADIUS = 0.06;
const float RING_WIDTH = 0.045;
const float MAX_RADIUS = 0.92;
const float EDGE_AA = 0.015;

float ring(float dist, float radius, float width) {
    float inner = radius - width * 0.5;
    float outer = radius + width * 0.5;
    return smoothstep(inner - EDGE_AA, inner + EDGE_AA, dist) *
           (1.0 - smoothstep(outer - EDGE_AA, outer + EDGE_AA, dist));
}

void main() {
    float dist = length(v_uv);
    float progress = 1.0 - u_opacity;

    // Flash burst
    float flash_intensity = pow(max(0.0, 1.0 - progress * 4.0), 2.0);
    float flash = (1.0 - smoothstep(0.0, 0.25, dist)) * flash_intensity;

    // Center dot with pulse
    float dot_scale = 1.0 + 0.3 * sin(progress * 6.28) * (1.0 - progress);
    float dot_radius = DOT_RADIUS * dot_scale;
    float dot_alpha = (1.0 - smoothstep(dot_radius - EDGE_AA, dot_radius + EDGE_AA, dist));
    dot_alpha *= pow(u_opacity, 0.5);

    // Expanding rings
    float r1_prog = clamp(progress * 1.5, 0.0, 1.0);
    float r1_radius = DOT_RADIUS + r1_prog * (MAX_RADIUS - DOT_RADIUS);
    float r1_alpha = ring(dist, r1_radius, RING_WIDTH) * pow(1.0 - r1_prog, 1.5);

    float r2_prog = clamp((progress - 0.15) * 1.5, 0.0, 1.0);
    float r2_radius = DOT_RADIUS + r2_prog * (MAX_RADIUS * 0.85 - DOT_RADIUS);
    float r2_alpha = ring(dist, r2_radius, RING_WIDTH * 0.7) * pow(1.0 - r2_prog, 1.5) * 0.6;

    // Glow
    float glow = exp(-pow((dist - r1_radius * 0.95) * 4.0, 2.0)) * (1.0 - r1_prog) * 0.3;

    float alpha = max(max(dot_alpha, flash), max(r1_alpha + r2_alpha, glow));
    if (alpha < 0.01) discard;

    vec3 color = u_color + vec3(0.4) * (flash + dot_alpha * 0.3);
    FragColor = vec4(color, alpha);
}
