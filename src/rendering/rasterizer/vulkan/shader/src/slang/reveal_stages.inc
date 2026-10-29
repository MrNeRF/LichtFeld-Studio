#ifndef LFS_VK_REVEAL_STAGES_INC
#define LFS_VK_REVEAL_STAGES_INC

// How the projection and 3DGUT raster hosts apply a built-in reveal effect.
// Macros rather than functions, so the legacy shaders compile to exactly the
// code they had before the custom effect hosts shared these stages.

// Color stage on the view-dependent SH color `rgb`; declares `luminance`.
#define LFS_REVEAL_SHADE(rgb, color_mix, multiplier, add, grayscale, edge_rgb) \
    rgb = lerp(float3(0.3f), rgb, color_mix) * multiplier + add; \
    float luminance = dot(rgb, float3(0.299f, 0.587f, 0.114f)); \
    rgb = lerp(rgb, float3(0.3f * luminance), grayscale) + edge_rgb;

// 3DGUT stage on a world-space center, the source-space inverse scale/rotation
// rows and the raw opacity. source_scales are the inverse row lengths before
// any change. Declares `effect`, `ratio`, `c` and `s`.
#define LFS_REVEAL_GUT(xyz, iscl, opac, source_scales, reveal, bounds) \
    LfsRevealEffect effect = lfs_reveal_effect(xyz, source_scales, reveal, bounds); \
    xyz = effect.position; \
    float3 ratio = source_scales / max(source_scales * effect.scale_mix + effect.seed_scale, 1e-8f); \
    iscl[0] *= ratio.x; \
    iscl[1] *= ratio.y; \
    iscl[2] *= ratio.z; \
    float c = cos(effect.spin), s = sin(effect.spin); \
    iscl = mul(iscl, float3x3(c,0,-s, 0,1,0, s,0,c)); \
    opac = lerp(0.3f, opac, effect.color_mix) * effect.opacity_multiplier + effect.color_add;

#endif
