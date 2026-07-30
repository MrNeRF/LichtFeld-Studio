#version 450
layout(location = 0) in vec2 ScreenPos;
layout(location = 1) in vec2 P0;
layout(location = 2) in vec2 P1;
layout(location = 3) in vec4 Color;
layout(location = 4) in vec4 Params;
layout(location = 5) in float ViewDepth;
layout(location = 0) out vec4 FragColor;

layout(set = 0, binding = 0) uniform sampler2D u_splat_depth;

layout(push_constant) uniform ShapeOverlayPush {
    // x,y: viewport origin in framebuffer pixels.
    // z,w: viewport size in framebuffer pixels.
    vec4 viewport_rect;
    // x: depth available (1.0 = sample splat depth, 0.0 = no fade),
    // y: flip-y for depth UV (1.0 = flip),
    // z,w: unused.
    vec4 params;
} pc;

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
    } else if (shape < 2.5) {
        signed_dist = abs(length(ScreenPos - P0) - radius) - thickness * 0.5;
    } else {
        signed_dist = -aa;
    }

    float alpha = Color.a * smoothstep(aa, -aa, signed_dist);

    // Soft passthrough: lines in front of splats render at full opacity,
    // lines behind dense splats fade to ~0.25 so the wireframe outline stays
    // readable through the scene. ViewDepth = 0 means "UI overlay, always in
    // front" (gizmos / pivots) so they skip the comparison entirely.
    if (pc.params.x > 0.5 && ViewDepth > 0.0) {
        vec2 vp = max(pc.viewport_rect.zw, vec2(1.0));
        vec2 uv = (gl_FragCoord.xy - pc.viewport_rect.xy) / vp;
        if (pc.params.y > 0.5) {
            uv.y = 1.0 - uv.y;
        }
        float splat_depth = texture(u_splat_depth, uv).r;
        if (splat_depth > 0.0 && splat_depth < 1.0e9) {
            float fade = smoothstep(0.01, 0.15, ViewDepth - splat_depth);
            alpha *= mix(1.0, 0.25, fade);
        }
    }

    if (alpha <= 0.001) {
        discard;
    }
    FragColor = vec4(Color.rgb, alpha);
}
