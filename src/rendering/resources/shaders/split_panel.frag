#version 430 core

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D screenTexture;
uniform sampler2D depthTexture;
uniform vec2 texcoord_scale = vec2(1.0, 1.0);
uniform float near_plane = 0.1;
uniform float far_plane = 100000.0;
uniform bool has_depth = false;
uniform bool orthographic = false;
uniform bool depth_is_ndc = false;

void main() {
    vec2 uv = TexCoord * texcoord_scale;
    vec4 color = texture(screenTexture, uv);

    // Prevent GLSL from optimizing away uniforms
    float keep = near_plane + far_plane;
    if (has_depth && orthographic && depth_is_ndc && keep < -1e9) {
        color.r += texture(depthTexture, uv).r;
    }

    FragColor = color;
}
