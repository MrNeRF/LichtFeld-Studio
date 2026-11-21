#version 450
layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec3 vToEye;
layout(location = 0) out vec4 FragColor;

layout(push_constant) uniform HandPush {
    mat4 view_projection;
    vec4 eye;
    vec4 color;
} pc;

// A hand of light: the scene shows through its body and its outline glows.
// Premultiplied: rgb is the light added, alpha is how much of the scene behind
// is dimmed.
void main() {
    float facing = abs(dot(normalize(vNormal), normalize(vToEye)));
    float rim = pow(1.0 - facing, 2.2);
    // Discreet: a hand must not hide what it reaches for.
    float light = (0.06 + 0.48 * rim) * pc.color.a;
    float cover = clamp(0.06 + 0.25 * rim, 0.0, 1.0) * pc.color.a;
    FragColor = vec4(pc.color.rgb * light, cover);
}
