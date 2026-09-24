// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 104;
[SpecializationConstant] const uint kFlags = 0;
[SpecializationConstant] const uint kModel = 0;
static const float kPi = 3.14159265358979323846f;

struct Parameters
{
    uint64_t mask, points, transforms, indices, allowed, boxTransform, boxMin, boxMax, ellipsoidTransform, radii,
        window;
    uint count, transformCount, allowedCount, flags;
};

#include "../../../../include/core/rounded_math.slangh"
#define LFS_POINT_FILTER_FLAGS_ONLY
#include "../../../internal/point_filter.hpp"

#include "byte_read.slangh"
float3 loadPoint(uint64_t a, uint i)
{
    Ptr<float, Access.Read> f = (Ptr<float, Access.Read>)a;
    const uint64_t b = uint64_t(i) * 3;
    return float3(f[b], f[b + 1], f[b + 2]);
}
float3 localPoint(float3 pos, uint64_t matrix)
{
    Ptr<float, Access.Read> m = (Ptr<float, Access.Read>)matrix;
    return float3(roundedAdd(dotRounded(pos, float3(m[0], m[4], m[8])), m[12]),
                  roundedAdd(dotRounded(pos, float3(m[1], m[5], m[9])), m[13]),
                  roundedAdd(dotRounded(pos, float3(m[2], m[6], m[10])), m[14]));
}
bool keep(Parameters p, uint i)
{
    int node = 0;
    if (p.indices != 0)
        node = ((Ptr<int, Access.Read>)p.indices)[i];
    if ((kFlags & LFS_FILTER_NODES) != 0u)
    {
        if (node < 0 || uint(node) >= p.allowedCount)
            return false;
        if (loadByte(p.allowed, uint(node)) == 0)
            return false;
    }
    if ((kFlags & LFS_FILTER_GEOMETRY) == 0u)
        return true;
    float3 pos = loadPoint(p.points, i);
    if (p.transforms != 0 && p.transformCount > 0)
    {
        node = clamp(node, 0, int(p.transformCount - 1));
        Ptr<float, Access.Read> m = (Ptr<float, Access.Read>)(p.transforms + uint64_t(node) * 64);
        pos = float3(roundedAdd(dotRounded(pos, float3(m[0], m[1], m[2])), m[3]),
                     roundedAdd(dotRounded(pos, float3(m[4], m[5], m[6])), m[7]),
                     roundedAdd(dotRounded(pos, float3(m[8], m[9], m[10])), m[11]));
    }
    if ((kFlags & LFS_FILTER_BOX) != 0u)
    {
        const float3 local = localPoint(pos, p.boxTransform);
        const bool inside = all(local >= loadPoint(p.boxMin, 0)) && all(local <= loadPoint(p.boxMax, 0));
        if (inside == ((kFlags & LFS_FILTER_INVERSE_BOX) != 0u))
            return false;
    }
    if ((kFlags & LFS_FILTER_ELLIPSOID) != 0u)
    {
        const float3 local = localPoint(pos, p.ellipsoidTransform), r = loadPoint(p.radii, 0);
        const float norm = roundedAdd(roundedAdd(roundedDiv(roundedMul(local.x, local.x), roundedMul(r.x, r.x)),
                                                 roundedDiv(roundedMul(local.y, local.y), roundedMul(r.y, r.y))),
                                      roundedDiv(roundedMul(local.z, local.z), roundedMul(r.z, r.z)));
        if ((norm <= 1.f) == ((kFlags & LFS_FILTER_INVERSE_ELLIPSOID) != 0u))
            return false;
    }
    if ((kFlags & LFS_FILTER_WINDOW) == 0u)
        return true;
    Ptr<float, Access.Read> w = (Ptr<float, Access.Read>)p.window;
    const float3 d = float3(roundedSub(pos.x, w[9]), roundedSub(pos.y, w[10]), roundedSub(pos.z, w[11]));
    const float vx = dotRounded(d, float3(w[0], w[1], w[2]));
    const float vy = -dotRounded(d, float3(w[3], w[4], w[5]));
    const float vz = -dotRounded(d, float3(w[6], w[7], w[8]));
    float px = 0, py = 0, depth = vz;
    if (kModel == 0)
    {
        px = roundedAdd(roundedDiv(roundedMul(vx, w[12]), vz), w[14]);
        py = roundedAdd(roundedDiv(roundedMul(vy, w[13]), vz), w[15]);
    }
    else if (kModel == 1)
    {
        px = roundedAdd(roundedMul(vx, w[16]), .5f * w[17]);
        py = roundedAdd(roundedMul(vy, w[16]), .5f * w[18]);
    }
    else
    {
        const float len = sqrt(dotRounded(float3(vx, vy, vz), float3(vx, vy, vz)));
        if (len <= 1.e-6f || !isfinite(len))
            return false;
        const float dirx = roundedDiv(vx, len), diry = clamp(roundedDiv(vy, len), -1.f, 1.f),
                    dirz = roundedDiv(vz, len);
        const float latitude = asin(diry);
        px = roundedMul(roundedAdd(roundedDiv(atan2(dirx, dirz), 2.f * kPi), .5f), w[17]);
        py = roundedMul(roundedAdd(roundedDiv(latitude, kPi), .5f), w[18]);
        depth = len;
    }
    return abs(roundedSub(px, w[21])) <= w[19] && abs(roundedSub(py, w[22])) <= w[20] && depth >= w[23] &&
           depth <= w[24] && depth > 0.f;
}
[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uniform Parameters p)
{
    const uint3 groups = spirv_asm
    {
    result:
        $$uint3 = OpLoad builtin(NumWorkgroups : uint3);
    };
    const uint offset = uint(p.mask & uint64_t(3)), words = (p.count + offset + 3) / 4;
    Ptr<uint, Access.ReadWrite> mask = (Ptr<uint, Access.ReadWrite>)(p.mask & ~uint64_t(3));
    for (uint word = id.x; word < words; word += groups.x * LFS_LOCAL_SIZE_X)
    {
        uint value = mask[word];
        [unroll]
        for (uint byte = 0; byte < 4; ++byte)
        {
            const uint slot = word * 4 + byte;
            if (slot < offset || slot - offset >= p.count)
                continue;
            const uint old = (value >> (byte * 8)) & 255u;
            if (old != 0 && !keep(p, slot - offset))
                value &= ~(255u << (byte * 8));
        }
        mask[word] = value;
    }
}
