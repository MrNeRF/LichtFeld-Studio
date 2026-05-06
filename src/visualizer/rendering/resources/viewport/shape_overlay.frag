#version 450
layout(location = 0) in vec2 ScreenPos;
layout(location = 1) in vec2 P0;
layout(location = 2) in vec2 P1;
layout(location = 3) in vec4 Color;
layout(location = 4) in vec4 Params;
layout(location = 0) out vec4 FragColor;

float sdSegment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h);
}

void main() {
    float shape = Params.x;
    float thickness = max(Params.y, 1.0);
    float radius = max(Params.z, 0.0);
    float aa = max(Params.w, 0.75);
    float signed_dist = 1e6;

    if (shape < 0.5) {
        signed_dist = sdSegment(ScreenPos, P0, P1) - thickness * 0.5;
    } else if (shape < 1.5) {
        signed_dist = length(ScreenPos - P0) - radius;
    } else {
        signed_dist = abs(length(ScreenPos - P0) - radius) - thickness * 0.5;
    }

    float alpha = Color.a * smoothstep(aa, -aa, signed_dist);
    if (alpha <= 0.001) {
        discard;
    }
    FragColor = vec4(Color.rgb, alpha);
}
