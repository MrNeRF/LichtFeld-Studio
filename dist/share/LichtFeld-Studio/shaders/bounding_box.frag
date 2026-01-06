#version 430 core

uniform vec3 u_color;

in float v_backface_factor;

out vec4 frag_color;

void main() {
    // Back-facing edges are more transparent
    float alpha = mix(1.0, 0.2, v_backface_factor);
    frag_color = vec4(u_color, alpha);
}