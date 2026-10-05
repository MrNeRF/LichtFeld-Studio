// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
// Features a compute-heavy rasterizer needs, compiled for every backend:
// subgroup operations, threadgroup memory, atomics and 16/64-bit storage.
struct Parameters {
    Ptr<uint, Access.ReadWrite> values;
    Ptr<uint, Access.ReadWrite> output;
    Ptr<uint, Access.ReadWrite> counters;
    Ptr<half, Access.ReadWrite> halves;
    Ptr<uint16_t, Access.ReadWrite> shorts;
    Ptr<int64_t, Access.ReadWrite> longs;
    uint count;
    uint padding;
};
[[vk::push_constant]] ConstantBuffer<Parameters> p;

// Per lane: inclusive prefix sum, subgroup sum, odd-lane count, the next
// lane's value, and the subgroup size.
[shader("compute")]
[numthreads(64, 1, 1)]
void waveOps(uint3 tid : SV_DispatchThreadID) {
    const uint v = tid.x < p.count ? p.values[tid.x] : 0u;
    const uint lane = WaveGetLaneIndex();
    const uint size = WaveGetLaneCount();
    const uint prefix = WavePrefixSum(v) + v;
    const uint sum = WaveActiveSum(v);
    const uint odd = WaveActiveCountBits((v & 1u) != 0u);
    const uint next = WaveReadLaneAt(v, (lane + 1u) % size);
    const uint4 ballot = WaveActiveBallot(v > 8u);
    if (tid.x < p.count) {
        const uint base = tid.x * 6u;
        p.output[base] = prefix;
        p.output[base + 1] = sum;
        p.output[base + 2] = odd;
        p.output[base + 3] = next;
        p.output[base + 4] = size;
        p.output[base + 5] = ballot.x;
    }
}

groupshared uint shared_sum[256];

// One sum per 256-thread group through threadgroup memory and barriers.
[shader("compute")]
[numthreads(256, 1, 1)]
void sharedReduce(uint3 tid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    shared_sum[gtid.x] = tid.x < p.count ? p.values[tid.x] : 0u;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (gtid.x < stride)
            shared_sum[gtid.x] += shared_sum[gtid.x + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (gtid.x == 0u)
        p.output[gid.x] = shared_sum[0];
}

// counters: [add, min, max, or] across every thread.
[shader("compute")]
[numthreads(64, 1, 1)]
void atomics(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= p.count)
        return;
    const uint v = p.values[tid.x];
    uint previous;
    InterlockedAdd(p.counters[0], v, previous);
    InterlockedMin(p.counters[1], v);
    InterlockedMax(p.counters[2], v);
    InterlockedOr(p.counters[3], 1u << (v & 31u));
}

// halves[i] *= 2, shorts[i] += 1, longs[i] = longs[i] * 3 - (1 << 40).
[shader("compute")]
[numthreads(64, 1, 1)]
void narrowAndWide(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= p.count)
        return;
    p.halves[tid.x] = p.halves[tid.x] * half(2.0);
    p.shorts[tid.x] = uint16_t(p.shorts[tid.x] + uint16_t(1));
    p.longs[tid.x] = p.longs[tid.x] * 3 - (int64_t(1) << 40);
}

// counters[0..2] = threadgroups covering `count` (or none when count is 0).
[shader("compute")]
[numthreads(1, 1, 1)]
void prepareIndirect(uint3 tid : SV_DispatchThreadID) {
    p.counters[0] = (p.count + 63u) / 64u;
    p.counters[1] = 1u;
    p.counters[2] = 1u;
}

// output[i] = values[i] + 1 for the threads an indirect dispatch launches.
[shader("compute")]
[numthreads(64, 1, 1)]
void markIndirect(uint3 tid : SV_DispatchThreadID) {
    if (tid.x < p.count)
        p.output[tid.x] = p.values[tid.x] + 1u;
}

// A null pointer is an absent optional input.
[shader("compute")]
[numthreads(64, 1, 1)]
void optionalInput(uint3 tid : SV_DispatchThreadID) {
    if (tid.x < p.count)
        p.output[tid.x] = p.longs != nullptr ? uint(p.longs[tid.x]) : 7u;
}

#ifndef LFS_FEATURE_VARIANT
#define LFS_FEATURE_VARIANT 0
#endif
// Writes the compile-time variant this module was built with.
[shader("compute")]
[numthreads(1, 1, 1)]
void variant(uint3 tid : SV_DispatchThreadID) {
    p.output[0] = LFS_FEATURE_VARIANT;
}
