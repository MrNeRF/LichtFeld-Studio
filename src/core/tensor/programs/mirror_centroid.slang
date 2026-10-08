// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
// Paired float partials retain low bits without requiring native GPU doubles.
struct Parameters {
    Ptr<float> means;
    Ptr<uint> selected;
    Ptr<float, Access.ReadWrite> partials;
    uint count;
    uint blocks;
};
[[vk::push_constant]] ConstantBuffer<Parameters> p;
groupshared float hi[3][256];
groupshared float lo[3][256];
groupshared uint counts[256];

#ifdef LFS_GPU_VULKAN
#include "../../include/core/rounded_math.slangh"
#else
float roundedAdd(float a, float b) { return a + b; }
float roundedSub(float a, float b) { return a - b; }
#endif

float2 addPair(float2 a, float2 b) {
    float s = roundedAdd(a.x, b.x);
    float v = roundedSub(s, a.x);
    float e = roundedAdd(roundedSub(a.x, roundedSub(s, v)), roundedSub(b.x, v));
    float tail = roundedAdd(roundedAdd(a.y, b.y), e);
    float h = roundedAdd(s, tail);
    float l = roundedSub(tail, roundedSub(h, s));
    return float2(h, l);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mirrorCentroid(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    uint t = thread.x;
    float2 sum[3];
    for (uint axis = 0; axis < 3; ++axis) sum[axis] = float2(0);
    uint count = 0;
    for (uint64_t i = uint64_t(group.x) * 256 + t; i < p.count; i += uint64_t(p.blocks) * 256) {
        if (p.selected[i] == 0) continue;
        ++count;
        for (uint axis = 0; axis < 3; ++axis)
            sum[axis] = addPair(sum[axis], float2(p.means[i * 3 + axis], 0));
    }
    for (uint axis = 0; axis < 3; ++axis) {
        hi[axis][t] = sum[axis].x;
        lo[axis][t] = sum[axis].y;
    }
    counts[t] = count;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (t < stride) {
            for (uint axis = 0; axis < 3; ++axis) {
                float2 s = addPair(float2(hi[axis][t], lo[axis][t]), float2(hi[axis][t + stride], lo[axis][t + stride]));
                hi[axis][t] = s.x;
                lo[axis][t] = s.y;
            }
            counts[t] += counts[t + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (t == 0) {
        for (uint axis = 0; axis < 3; ++axis) {
            p.partials[group.x * 7 + axis * 2] = hi[axis][0];
            p.partials[group.x * 7 + axis * 2 + 1] = lo[axis][0];
        }
        p.partials[group.x * 7 + 6] = float(counts[0]);
    }
}
