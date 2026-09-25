/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#version 450

layout(set = 0, binding = 0) uniform sampler2D sceneTexture;
layout(set = 1, binding = 0) uniform sampler2D splatDepth;
layout(location = 0) out vec4 FragColor;

layout(push_constant) uniform ReprojectPush {
    // (source_ndc * depth, depth, 1) -> current clip, see sceneReprojectionMatrix.
    mat4 source_to_current;
    vec4 viewport_rect;
    vec4 color_uv_region; // xy = uv scale, zw = uv clamp max
    vec4 depth_uv_region;
    vec4 flip_y; // x: color rows, y: depth rows
} pc;

const int kIterations = 8;
// Uncovered pixels hold the renderer's far sentinel; move them like distant background.
const float kBackgroundDepth = 1.0e6;
// Converged within a tenth of a pixel of a 2000 pixel viewport, in NDC.
const float kConvergedResidual = 1.0e-4;

vec2 imageUv(vec2 ndc, float flip) {
    vec2 uv = vec2(ndc.x, -ndc.y) * 0.5 + 0.5;
    return flip > 0.5 ? vec2(uv.x, 1.0 - uv.y) : uv;
}

float surfaceDepth(float depth) {
    return depth > 0.0 && depth < 1.0e9 ? depth : kBackgroundDepth;
}

// Nearest of the 2x2 texels around the sample: thin foreground (spokes, twigs)
// wins over the background behind it instead of flickering between the two.
float sourceDepth(vec2 ndc) {
    vec2 uv = min(imageUv(ndc, pc.flip_y.y) * pc.depth_uv_region.xy, pc.depth_uv_region.zw);
    vec4 depths = textureGather(splatDepth, uv, 0);
    return min(min(surfaceDepth(depths.x), surfaceDepth(depths.y)),
               min(surfaceDepth(depths.z), surfaceDepth(depths.w)));
}

vec2 warpToCurrent(vec2 source, float depth) {
    vec4 clip = pc.source_to_current * vec4(source * depth, depth, 1.0);
    return clip.xy / max(clip.w, 1.0e-6);
}

struct Candidate {
    vec2 source;
    float depth;
    float residual;
};

// Inverse warp by fixed-point iteration: find the source pixel whose depth lands
// on the target in the current view.
Candidate solve(vec2 target, vec2 seed) {
    Candidate c = Candidate(seed, kBackgroundDepth, 1.0e9);
    for (int i = 0; i < kIterations; ++i) {
        c.depth = sourceDepth(c.source);
        vec2 error = target - warpToCurrent(c.source, c.depth);
        c.residual = dot(error, error);
        c.source += error;
    }
    return c;
}

void main() {
    vec2 frag_uv = (gl_FragCoord.xy - pc.viewport_rect.xy) / pc.viewport_rect.zw;
    vec2 target = vec2(frag_uv.x, 1.0 - frag_uv.y) * 2.0 - 1.0;
    // Near and far surfaces can both claim a pixel; seed from the unmoved pixel
    // and from where distant background lands, then keep the converged nearer one.
    vec2 far_seed = target;
    for (int i = 0; i < 3; ++i) {
        far_seed += target - warpToCurrent(far_seed, kBackgroundDepth);
    }
    Candidate a = solve(target, target);
    Candidate b = solve(target, far_seed);
    bool a_converged = a.residual < kConvergedResidual * kConvergedResidual;
    bool b_converged = b.residual < kConvergedResidual * kConvergedResidual;
    Candidate best = a_converged && b_converged ? (a.depth <= b.depth ? a : b)
                   : a_converged                ? a
                   : b_converged                ? b
                   : (a.residual <= b.residual ? a : b);
    vec2 source = clamp(best.source, vec2(-1.0), vec2(1.0));
    vec2 uv = min(imageUv(source, pc.flip_y.x) * pc.color_uv_region.xy, pc.color_uv_region.zw);
    FragColor = texture(sceneTexture, uv);
}
