#include "config.slang"

#define _FOR_LOOP(varname, numiter) \
    [ForceUnroll] for (int varname = 0; varname < numiter; ++varname)


[ForceInline]
float3 read_float3(uint32_t idx, StructuredBuffer<float> t3) {
    return float3(t3[3*idx+0], t3[3*idx+1], t3[3*idx+2]);
}

[ForceInline]
float3 read_float3(uint32_t idx, RWStructuredBuffer<float> t3) {
    return float3(t3[3*idx+0], t3[3*idx+1], t3[3*idx+2]);
}

[ForceInline]
void write_float3(float3 val, uint32_t idx, RWStructuredBuffer<float> t3) {
    t3[3*idx+0] = val.x;
    t3[3*idx+1] = val.y;
    t3[3*idx+2] = val.z;
}

[ForceInline]
float3 read_mean(uint32_t idx, StructuredBuffer<float> t3, uint s=1, uint o=0) {
    return read_float3(idx*s+o, t3);
}

[ForceInline]
float3 read_mean(uint32_t idx, RWStructuredBuffer<float> t3, uint s=1, uint o=0) {
    return read_float3(idx*s+o, t3);
}

[ForceInline]
void write_mean(float3 val, uint32_t idx, RWStructuredBuffer<float> t3, uint s=1, uint o=0) {
    idx = idx*s+o;
    t3[3*idx+0] = val.x;
    t3[3*idx+1] = val.y;
    t3[3*idx+2] = val.z;
}


#if SH_REORDER_SIZE <= 1
[ForceInline]
uint32_t shAt(uint32_t gid, uint sid) {
    return 12*gid + sid;
}
#else
[ForceInline]
uint32_t shAt(uint32_t gid, uint sid) {
    uint32_t base = (gid/SH_REORDER_SIZE) * (12*SH_REORDER_SIZE);
    uint32_t offset = gid%SH_REORDER_SIZE;
    return base + SH_REORDER_SIZE*sid + offset;
}
#endif


uint32_t hash_u32_u32(uint32_t x) {
    x ^= x >> 16;
    x *= uint32_t(0x7feb352d);
    x ^= x >> 15;
    x *= uint32_t(0x846ca68b);
    x ^= x >> 16;
    return x;
}

float hash_u32_float(uint32_t x) {
    return hash_u32_u32(x) / 4294967296.0;
}

uint64_t hash_u32_u64(uint32_t a, uint32_t b) {
    uint64_t x = hash_u32_u32(a ^ ((b + 1u) * uint32_t(0x9e3779b9)));
    uint64_t y = hash_u32_u32(b ^ ((a + 1u) * uint32_t(0x9e3779b9)));
    return (x << 32) | y;
}

#if USE_EMULATED_INT64

uint32_t2 hash_u32_u322(uint32_t a, uint32_t b) {
    uint32_t x = hash_u32_u32(a ^ ((b + 1u) * uint32_t(0x9e3779b9)));
    uint32_t y = hash_u32_u32(b ^ ((a + 1u) * uint32_t(0x9e3779b9)));
    return uint32_t2(y, x);
}

uint32_t mul_hi(uint32_t a, uint32_t b) {
    uint32_t a_hi = a >> 16;
    uint32_t a_lo = a & 0xFFFFU;
    uint32_t b_hi = b >> 16;
    uint32_t b_lo = b & 0xFFFFU;
    uint32_t mid = a_lo * b_hi + ( (a_hi * b_lo) & 0xFFFFU );
    return a_hi * b_hi + ( (a_hi * b_lo) >> 16 ) + ( mid >> 16 );
}

uint32_t mod64(uint32_t2 val, uint32_t m) {
    uint32_t lo = val.x, hi = val.y;
    uint32_t k = (((uint32_t)0xFFFFFFFFu % m) + 1u) % m;
    uint32_t r = hi % m;
    uint32_t loP = r * k;
    uint32_t hiP = mul_hi(r, k);
    r = (loP % m + hiP % m * (k % m)) % m;
    r = (r + (lo % m)) % m;
    return r;
}

#endif

float3 randn3(uint32_t a, uint32_t b) {
    float r0 = hash_u32_float(a ^ ((b + 1) * uint32_t(0x9e3779b9)));
    float r1 = hash_u32_float(b ^ ((a + 1) * uint32_t(0x9e3779b9)));
    float r2 = hash_u32_float(a ^ ((b + 2) * uint32_t(0x9e3779b9)));
    float r3 = hash_u32_float(b ^ ((a + 2) * uint32_t(0x9e3779b9)));
    float z0 = sqrt(-2.0*log(1.0-r0)) * cos(2.0*M_PI*r1);
    float z1 = sqrt(-2.0*log(1.0-r0)) * sin(2.0*M_PI*r1);
    float z2 = sqrt(-2.0*log(1.0-r2)) * cos(2.0*M_PI*r3);
    return float3(z0, z1, z2);
}


float3x3 sqrt_cov_from_quat_scales(float4 q, float3 s) {
    q = normalize(q);
    float r = q[0], x = q[1], y = q[2], z = q[3];

    float3x3 rotation_matrix = float3x3(
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y));

    float3x3 scales_matrix = float3x3(s[0], 0, 0,
                                      0, s[1], 0,
                                      0, 0, s[2]);

    return mul(rotation_matrix, scales_matrix);
}
