// Shared by Slang rasterizer shaders and GLSL panorama shaders. The camera
// frame is right-handed with forward along -Z and up along +Y.
#ifdef LFS_EQUIRECT_GLSL
#define LFS_EQUIRECT_FLOAT2 vec2
#define LFS_EQUIRECT_FLOAT3 vec3
#define LFS_EQUIRECT_ATAN2(y, x) atan(y, x)
#define LFS_EQUIRECT_DIFFERENTIABLE
#else
#define LFS_EQUIRECT_FLOAT2 float2
#define LFS_EQUIRECT_FLOAT3 float3
#define LFS_EQUIRECT_ATAN2(y, x) atan2(y, x)
#define LFS_EQUIRECT_DIFFERENTIABLE [Differentiable]
#endif

#define LFS_EQUIRECT_PI 3.14159265358979323846

LFS_EQUIRECT_DIFFERENTIABLE
LFS_EQUIRECT_FLOAT2 lfs_equirect_uv(LFS_EQUIRECT_FLOAT3 direction) {
    direction = normalize(direction);
    const float longitude = (direction.x == 0.0 && direction.z == 0.0)
                                ? 0.0
                                : LFS_EQUIRECT_ATAN2(direction.x, -direction.z);
    const float latitude = asin(clamp(direction.y, -1.0, 1.0));
    return LFS_EQUIRECT_FLOAT2(longitude / (2.0 * LFS_EQUIRECT_PI) + 0.5,
                               0.5 - latitude / LFS_EQUIRECT_PI);
}

LFS_EQUIRECT_FLOAT3 lfs_equirect_direction(LFS_EQUIRECT_FLOAT2 uv) {
    const float longitude = (uv.x - 0.5) * (2.0 * LFS_EQUIRECT_PI);
    const float latitude = (0.5 - uv.y) * LFS_EQUIRECT_PI;
    const float cos_latitude = cos(latitude);
    return normalize(LFS_EQUIRECT_FLOAT3(sin(longitude) * cos_latitude,
                                         sin(latitude),
                                         -cos(longitude) * cos_latitude));
}

#undef LFS_EQUIRECT_FLOAT2
#undef LFS_EQUIRECT_FLOAT3
#undef LFS_EQUIRECT_ATAN2
#undef LFS_EQUIRECT_DIFFERENTIABLE
#undef LFS_EQUIRECT_PI
