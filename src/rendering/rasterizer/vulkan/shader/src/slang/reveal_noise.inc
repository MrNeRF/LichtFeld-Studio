#ifndef LFS_VK_REVEAL_NOISE_INC
#define LFS_VK_REVEAL_NOISE_INC

// Reveal hash and value noise. Shared verbatim by the built-in reveal effects
// in utils.slang and the public helpers in lfs_helpers.slang.

float3 lfs_reveal_hash(float3 p) {
    p = frac(p * 0.3183099f + 0.1f) * 17.0f;
    return frac(float3(p.x*p.y*p.z, p.x+p.y*p.z, p.x*p.y+p.z));
}

float3 lfs_reveal_noise(float3 p) {
    float3 i = floor(p), f = frac(p);
    f = f*f*(3.0f-2.0f*f);
    float3 a = lerp(lfs_reveal_hash(i), lfs_reveal_hash(i+float3(1,0,0)), f.x);
    float3 b = lerp(lfs_reveal_hash(i+float3(0,1,0)), lfs_reveal_hash(i+float3(1,1,0)), f.x);
    float3 c = lerp(lfs_reveal_hash(i+float3(0,0,1)), lfs_reveal_hash(i+float3(1,0,1)), f.x);
    float3 d = lerp(lfs_reveal_hash(i+float3(0,1,1)), lfs_reveal_hash(i+float3(1,1,1)), f.x);
    return lerp(lerp(a,b,f.y), lerp(c,d,f.y), f.z);
}

#endif
