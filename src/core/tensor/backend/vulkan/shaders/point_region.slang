// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 56;

[SpecializationConstant]
const uint kKind = 0; // disk, rectangle, polygon, disks

struct Parameters
{
    uint64_t mask;
    uint64_t points;
    uint64_t geometry;
    uint count;
    uint geometryCount;
    float4 bounds;
    float radiusSq;
    float minimumCoordinate;
};

#include "../../../../include/core/rounded_math.slangh"

bool inDisk(float2 point, float2 center, float radiusSq)
{
    const float dx = roundedSub(point.x, center.x), dy = roundedSub(point.y, center.y);
    return roundedAdd(roundedMul(dx, dx), roundedMul(dy, dy)) <= radiusSq;
}
bool insideRegion(Parameters p, uint index)
{
    Ptr<float, Access.Read> xy = (Ptr<float, Access.Read>)p.points;
    const uint64_t offset = uint64_t(index) * 2;
    const float2 point = float2(xy[offset], xy[offset + 1]);
    if (!(point.x >= p.minimumCoordinate && point.y >= p.minimumCoordinate)) return false;
    if (kKind == 0) return inDisk(point, p.bounds.xy, p.radiusSq);
    if (kKind == 1) return point.x >= p.bounds.x && point.y >= p.bounds.y && point.x <= p.bounds.z && point.y <= p.bounds.w;
    Ptr<float, Access.Read> vertices = (Ptr<float, Access.Read>)p.geometry;
    if (kKind == 3)
    {
        for (uint k = 0; k < p.geometryCount; ++k)
        {
            const uint64_t base = uint64_t(k) * 2;
            if (inDisk(point, float2(vertices[base], vertices[base + 1]), p.radiusSq)) return true;
        }
        return false;
    }
    bool inside = false;
    for (uint k = 0, j = p.geometryCount - 1; k < p.geometryCount; j = k++)
    {
        const uint64_t ki = uint64_t(k) * 2, ji = uint64_t(j) * 2;
        const float xi = vertices[ki], yi = vertices[ki + 1];
        const float xj = vertices[ji], yj = vertices[ji + 1];
        if ((yi > point.y) != (yj > point.y))
        {
            const float hitX = roundedAdd(roundedDiv(roundedMul(roundedSub(xj, xi), roundedSub(point.y, yi)), roundedSub(yj, yi)), xi);
            if (point.x < hitX) inside = !inside;
        }
    }
    return inside;
}

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID, uniform Parameters p)
{
    const uint3 groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    const uint stride = groups.x * LFS_LOCAL_SIZE_X;
    const uint offset = uint(p.mask & uint64_t(3));
    const uint wordCount = (p.count + offset + 3) / 4;
    Ptr<uint, Access.ReadWrite> output = (Ptr<uint, Access.ReadWrite>)(p.mask & ~uint64_t(3));
    // One writer owns each word, including the neighboring bytes of a slice.
    for (uint word = dispatchThreadId.x; word < wordCount; word += stride)
    {
        const uint original = output[word];
        uint value = original;
        [unroll]
        for (uint lane = 0; lane < 4; ++lane)
        {
            const uint slot = word * 4 + lane;
            if (slot < offset || slot - offset >= p.count) continue;
            if (insideRegion(p, slot - offset))
                value = (value & ~(255u << (lane * 8))) | (1u << (lane * 8));
        }
        if (value != original) output[word] = value;
    }
}
