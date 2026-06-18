#version 450
layout(set = 0, binding = 0) uniform sampler2D overlayTexture;
layout(set = 1, binding = 0) uniform sampler2D u_splat_depth;
layout(location = 0) in vec2 TexCoord;
layout(location = 1) in float ViewDepth;
layout(location = 0) out vec4 FragColor;
layout(push_constant) uniform TexturedOverlayPush {
    vec4 tint_opacity;
    vec4 effects;
    // x,y: viewport origin in framebuffer pixels.
    // z,w: viewport size in framebuffer pixels.
    vec4 viewport_rect;
    // x: depth_available (1.0 = sample splat depth, 0.0 = skip occlusion),
    // y: flip-y for depth UV (1.0 = flip),
    // z,w: unused.
    vec4 depth_params;
} u;

void main() {
    // Hard depth occlusion against the splat surface so the frustum thumbnail
    // hides behind dense splats. (The wireframe in shape_overlay soft-fades
    // instead, to keep the outline readable through the scene.)
    if (u.depth_params.x > 0.5 && ViewDepth > 0.0) {
        vec2 vp = max(u.viewport_rect.zw, vec2(1.0));
        vec2 uv = (gl_FragCoord.xy - u.viewport_rect.xy) / vp;
        if (u.depth_params.y > 0.5) {
            uv.y = 1.0 - uv.y;
        }
        float splat_depth = texture(u_splat_depth, uv).r;
        if (splat_depth > 0.0 && splat_depth < 1.0e9 && ViewDepth > splat_depth + 0.01) {
            discard;
        }
    }

    vec4 sampled = texture(overlayTexture, TexCoord);
    vec3 rgb = mix(sampled.rgb, u.tint_opacity.rgb, clamp(u.effects.x, 0.0, 1.0));
    rgb = mix(rgb, vec3(0.5), clamp(u.effects.y, 0.0, 1.0));
    FragColor = vec4(rgb, sampled.a * clamp(u.tint_opacity.a, 0.0, 1.0));
}
