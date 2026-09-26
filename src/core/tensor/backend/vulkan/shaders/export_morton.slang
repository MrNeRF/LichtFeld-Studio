static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 64;

[SpecializationConstant]
const uint kPhase = 0;

struct Parameters {
    uint64_t positions;
    uint64_t partials;
    uint64_t keys;
    uint n;
    uint blocks;
    uint padding;
    float minX; float minY; float minZ;
    float mulX; float mulY; float mulZ;
    uint tail;
};

uint64_t mask64(uint hi, uint lo) {
    return (uint64_t(hi) << 32) | uint64_t(lo);
}

// Masks match morton_spread. Slang's `u` suffix is 32-bit, so the 64-bit
// constants are split; a truncated mask would drop the high bits of the key.
uint64_t spread(uint x) {
    uint64_t v = uint64_t(x) & 0x1fffffu;
    v = (v | (v << 32)) & mask64(0x001f0000u, 0x0000ffffu);
    v = (v | (v << 16)) & mask64(0x001f0000u, 0xff0000ffu);
    v = (v | (v << 8)) & mask64(0x100f00f0u, 0x0f00f00fu);
    v = (v | (v << 4)) & mask64(0x10c30c30u, 0xc30c30c3u);
    return (v | (v << 2)) & mask64(0x12492492u, 0x49249249u);
}

groupshared float red[256 * 6];

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform Parameters p) {
    const uint3 groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    Ptr<float, Access.Read> pos = (Ptr<float, Access.Read>)p.positions;
    if (kPhase == 0) {
        const float inf = asfloat(0x7f800000u);
        float3 lo = float3(inf, inf, inf);
        float3 hi = float3(-inf, -inf, -inf);
        for (uint i = gid.x * 256u + tid.x; i < p.n; i += groups.x * 256u) {
            const float3 v = float3(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
            lo = min(lo, v);
            hi = max(hi, v);
        }
        red[tid.x] = lo.x; red[256 + tid.x] = lo.y; red[512 + tid.x] = lo.z;
        red[768 + tid.x] = hi.x; red[1024 + tid.x] = hi.y; red[1280 + tid.x] = hi.z;
        GroupMemoryBarrierWithGroupSync();
        for (uint step = 128u; step > 0u; step >>= 1) {
            if (tid.x < step) {
                [ForceUnroll] for (uint c = 0; c < 3; ++c) {
                    red[c * 256 + tid.x] = min(red[c * 256 + tid.x], red[c * 256 + tid.x + step]);
                    red[(c + 3) * 256 + tid.x] = max(red[(c + 3) * 256 + tid.x], red[(c + 3) * 256 + tid.x + step]);
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if (tid.x == 0) {
            Ptr<float, Access.ReadWrite> outp = (Ptr<float, Access.ReadWrite>)p.partials;
            [ForceUnroll] for (uint c = 0; c < 6; ++c) outp[gid.x * 6 + c] = red[c * 256];
        }
    } else if (kPhase == 1) {
        if (gid.x != 0) return;
        const float inf = asfloat(0x7f800000u);
        float3 lo = float3(inf, inf, inf);
        float3 hi = float3(-inf, -inf, -inf);
        Ptr<float, Access.Read> part = (Ptr<float, Access.Read>)p.partials;
        for (uint b = tid.x; b < p.blocks; b += 256u) {
            lo = min(lo, float3(part[b * 6], part[b * 6 + 1], part[b * 6 + 2]));
            hi = max(hi, float3(part[b * 6 + 3], part[b * 6 + 4], part[b * 6 + 5]));
        }
        red[tid.x] = lo.x; red[256 + tid.x] = lo.y; red[512 + tid.x] = lo.z;
        red[768 + tid.x] = hi.x; red[1024 + tid.x] = hi.y; red[1280 + tid.x] = hi.z;
        GroupMemoryBarrierWithGroupSync();
        for (uint step = 128u; step > 0u; step >>= 1) {
            if (tid.x < step) {
                [ForceUnroll] for (uint c = 0; c < 3; ++c) {
                    red[c * 256 + tid.x] = min(red[c * 256 + tid.x], red[c * 256 + tid.x + step]);
                    red[(c + 3) * 256 + tid.x] = max(red[(c + 3) * 256 + tid.x], red[(c + 3) * 256 + tid.x + step]);
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if (tid.x == 0) {
            Ptr<float, Access.ReadWrite> outp = (Ptr<float, Access.ReadWrite>)p.keys;
            [ForceUnroll] for (uint c = 0; c < 6; ++c) outp[c] = red[c * 256];
        }
    } else {
        Ptr<uint64_t, Access.ReadWrite> keys = (Ptr<uint64_t, Access.ReadWrite>)p.keys;
        const float axisMax = float((1u << 21) - 1u);
        for (uint i = gid.x * 256u + tid.x; i < p.n; i += groups.x * 256u) {
            uint64_t code = 0;
            [ForceUnroll] for (uint a = 0; a < 3; ++a) {
                const float low = a == 0 ? p.minX : a == 1 ? p.minY : p.minZ;
                const float mul = a == 0 ? p.mulX : a == 1 ? p.mulY : p.mulZ;
                const float nrm = (pos[i * 3 + a] - low) * mul;
                const uint q = nrm <= 0 ? 0u : nrm >= axisMax ? uint(axisMax) : uint(nrm);
                code |= spread(q) << a;
            }
            keys[i] = code;
        }
    }
}
