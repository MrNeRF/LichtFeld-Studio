#ifndef LFS_VK_REVEAL_EFFECTS_INC
#define LFS_VK_REVEAL_EFFECTS_INC

// Built-in reveal equations. utils.slang includes them for the legacy reveal
// shaders and lfs_reveal.slang for the custom effect hosts.

// Adapted from https://github.com/sparkjsdev/spark/blob/main/examples/splat-reveal-effects/index.html
// Spark Spread's radial expansion, seed/full scale, and color stages. Map the
// sequencer interval to the time needed for the whole scene to settle.
float4 lfs_reveal_spread(float3 position, float4 reveal, float4 bounds) {
    float progress = saturate(reveal.w);
    // Spark's examples operate on models a few units wide. Normalize by the
    // scene's typical radius so large reconstructions animate at the same pace.
    float unit = max(bounds.w * 0.5f, 0.0001f);
    float radius = length(position.xz - bounds.yz) / unit;
    float end_time = 14.0f;
    float tt = 0.5f + progress * progress * (end_time - 0.5f);
    float expansion = min(1.0f, 0.3f + max(0.0f, tt * 0.05f));
    float scale = max(0.0001f, max(min(tt - 7.0f - radius * 2.5f, 1.0f),
                                  0.2f * min(tt - 1.0f - radius * 2.0f, 1.0f)));
    float color = saturate(tt - radius * 2.5f - 3.0f);
    return float4(scale, color, expansion, 0.0f);
}

// Effect equations adapted from Spark's splat-reveal-effects example.
struct LfsRevealEffect {
    float3 position;
    float scale_mix;
    float seed_scale;
    float opacity_multiplier;
    float color_mix;
    float color_add;
    float spin;
    float3 edge_rgb;
    float grayscale;
};

#include "reveal_noise.inc"

float2 lfs_reveal_rotate(float2 p, float a) {
    float s = sin(a), c = cos(a);
    return float2(c*p.x + s*p.y, -s*p.x + c*p.y);
}

LfsRevealEffect lfs_reveal_finish(LfsRevealEffect e, float3 original, float progress) {
    // Blend the final part of the clip to the original splat, including rare
    // points outside the robust scene bounds, instead of popping at the end.
    float settle = smoothstep(0.9f, 1.0f, progress);
    e.position = lerp(e.position, original, settle);
    e.scale_mix = lerp(e.scale_mix, 1.0f, settle);
    e.seed_scale *= 1.0f - settle;
    e.opacity_multiplier = lerp(e.opacity_multiplier, 1.0f, settle);
    e.color_mix = lerp(e.color_mix, 1.0f, settle);
    e.color_add *= 1.0f - settle;
    e.spin *= 1.0f - settle;
    e.edge_rgb *= 1.0f - settle;
    e.grayscale *= 1.0f - settle;
    return e;
}

LfsRevealEffect lfs_reveal_effect(float3 position, float3 scales, float4 reveal, float4 bounds) {
    LfsRevealEffect e = {position, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, float3(0.0f), 0.0f};
    float progress = saturate(reveal.w);
    if (progress >= 1.0f) return e;
    int type = int(round(bounds.x));
    if (type >= 6 && type <= 10) {
        // Unveil is a spherical wave from the scene origin. Use the
        // robust scene extent and the clip's softness in place of fixed units.
        float range = max(length(float2(bounds.w, max(abs(reveal.y), abs(reveal.z)))), 0.0001f);
        float blend_range = max(range * (reveal.x - 1.0f), range * 0.001f);
        float distance = length(position);
        float blend = 1.0f - saturate((distance - progress * (range + blend_range) + blend_range) / blend_range);
        if (type != 9) e.scale_mix = max(blend, 0.000001f);
        if (type == 9) e.grayscale = 1.0f - blend;
        float edge_delta = distance / range - progress;
        if (type == 7 || type == 8) {
            float glow = exp(-pow(abs(edge_delta) / 0.01f, 2.0f));
            if (type == 7) e.edge_rgb = float3(glow);
            else {
                float h = frac(distance / range * 10.0f);
                float3 rgb = saturate(float3(abs(h*6.0f-3.0f)-1.0f, 2.0f-abs(h*6.0f-2.0f), 2.0f-abs(h*6.0f-4.0f)));
                e.edge_rgb = 1.5f * glow * rgb;
            }
        }
        if (type == 10) {
            float glow = exp(-pow(abs(edge_delta) / 0.04f, 2.0f));
            float3 direction = position.zxy;
            e.position -= 0.04f * range * glow * direction / max(length(direction), 0.000001f);
        }
        return lfs_reveal_finish(e, position, progress);
    }
    if (type == 11) return e; // Camera light is applied after color projection.
    float unit = max(bounds.w * 0.5f, 0.0001f);
    float3 p = (position - float3(bounds.y, 0.0f, bounds.z)) / unit;
    float l = length(p.xz);
    if (type == 2) {
        float4 spread = lfs_reveal_spread(position, reveal, bounds);
        e.position.xz = bounds.yz + p.xz * unit * spread.z;
        e.scale_mix = spread.x;
        e.color_mix = spread.y;
        return lfs_reveal_finish(e, position, progress);
    }
    // Spark evaluates its modifier before a 180-degree X rotation. Convert
    // both Y and Z, so its vertical direction and handedness match in world
    // space. Spread above is radial and can stay directly in world space.
    p.yz = -p.yz;
    float t = progress * (type == 3 ? max(15.0f, 2.7f + reveal.z / unit) : 15.0f);
    if (type == 1) {
        float s = smoothstep(0.0f, 10.0f, t - 4.5f) * 10.0f;
        float border = abs(s - l - 0.5f);
        p *= 1.0f - 0.2f * exp(-20.0f * border);
        float hidden = smoothstep(s - 0.5f, s, l + 0.5f);
        float3 animated = p + 0.1f*lfs_reveal_noise(p*2.0f+t*0.5f)*hidden;
        animated.yz = -animated.yz;
        e.position = animated * unit + float3(bounds.y,0,bounds.z);
        e.scale_mix = 1.0f - hidden;
        e.seed_scale = 0.002f * unit * hidden;
        float angle = atan2(p.x,p.z)/3.1416f;
        e.opacity_multiplier = step(angle, t-3.1416f);
        e.color_add = exp(-20.0f*border) + exp(-50.0f*abs(t-angle-3.1416f))*0.5f;
        return lfs_reveal_finish(e, position, progress);
    }
    if (type == 3) {
        p.xz = lfs_reveal_rotate(p.xz, (p.y*50.0f-20.0f)*exp(-t));
        float3 animated = p;
        animated.yz = -animated.yz;
        e.position = animated * unit * (1.0f - exp(-t)*2.0f) + float3(bounds.y,0,bounds.z);
        e.scale_mix = smoothstep(0.3f,0.7f,t+p.y-2.0f);
        e.seed_scale = 0.002f*unit*(1.0f-e.scale_mix);
        e.opacity_multiplier = step(0.0f,t*0.5f+p.y-0.5f);
        return lfs_reveal_finish(e, position, progress);
    }
    float3 h = lfs_reveal_hash(p);
    if (type == 4) {
        float s = smoothstep(0.0f,8.0f,t*t*0.1f-l*2.0f+2.0f);
        if (length(scales)/unit<0.05f) p.y = lerp(min(-10.0f,-reveal.z/unit-1.0f),p.y,pow(s,2.0f*h.x));
        p.xz = lerp(p.xz*0.5f,p.xz,pow(s,2.0f*h.x));
        p.xz = lfs_reveal_rotate(p.xz,t*(1.0f-s)*0.2f+p.y*20.0f*(1.0f-s)*exp(-length(p.xz)));
        float strength = s*s*s*s;
        e.scale_mix = pow(strength,12.0f);
        e.seed_scale = 0.002f*unit*(1.0f-e.scale_mix);
        e.spin = -t*0.3f*(1.0f-strength);
    } else if (type == 5) {
        float s = pow(smoothstep(0.0f,5.0f,t*t*0.1f-l*2.0f+1.0f),0.5f+h.x);
        float y = p.y;
        float bottom = min(-10.0f,-reveal.z/unit-1.0f);
        float top = max(5.0f,-reveal.y/unit+1.0f);
        p.y = min(lerp(bottom,top,s),p.y);
        p.xz = lfs_reveal_rotate(lerp(p.xz*0.3f,p.xz,s),t*0.3f*(1.0f-s));
        float strength = y <= bottom + 1e-6f ? step(y,p.y) : smoothstep(bottom,y,p.y);
        e.scale_mix = pow(strength,30.0f);
        e.seed_scale = 0.005f*unit*(1.0f-e.scale_mix);
        e.spin = -t*0.3f*(1.0f-s);
    }
    p.yz = -p.yz;
    // The model rotation reverses the direction of rotation about world Y.
    e.spin = -e.spin;
    e.position = p * unit + float3(bounds.y,0,bounds.z);
    return lfs_reveal_finish(e, position, progress);
}

#endif
