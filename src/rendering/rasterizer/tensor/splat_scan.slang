// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
// Hierarchical exclusive scan of uint64 counts: scan_blocks per 256 values,
// then scan_add folds in the scanned block sums. UInt64 keeps the required
// size exact even when it overflows the sort capacity.
struct Parameters {
    Ptr<uint64_t, Access.Read> input;
    Ptr<uint64_t, Access.ReadWrite> output;
    Ptr<uint64_t, Access.ReadWrite> sums;
    Ptr<uint64_t, Access.Read> offsets;
    uint count;
    uint primitive_counts; // level 0: each value fits 32 bits per subgroup
};
[[vk::push_constant]] ConstantBuffer<Parameters> p;

groupshared uint64_t group_sums[8];

[shader("compute")]
[numthreads(256, 1, 1)]
void scan_blocks(uint3 group_id : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint lane = thread.x, sg = lane / 32u;
    const uint i = group_id.x * 256u + lane;
    const uint64_t value = i < p.count ? p.input[i] : 0;
    // Subgroup reductions are 32-bit: sum 16-bit limbs (32 lanes cannot
    // overflow) and carry in uint64.
    uint64_t prefix = 0, simd_total = 0;
    if (p.primitive_counts != 0u) {
        prefix = uint64_t(WavePrefixSum(uint(value)));
        simd_total = uint64_t(WaveActiveSum(uint(value)));
    } else {
        for (uint shift = 0; shift < 64; shift += 16) {
            const uint limb = uint((value >> shift) & 65535ul);
            prefix += uint64_t(WavePrefixSum(limb)) << shift;
            simd_total += uint64_t(WaveActiveSum(limb)) << shift;
        }
    }
    if ((lane & 31u) == 0u)
        group_sums[sg] = simd_total;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 0; s < sg; ++s)
        prefix += group_sums[s];
    if (i < p.count)
        p.output[i] = prefix;
    if (lane == 255u)
        p.sums[group_id.x] = prefix + value;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void scan_add(uint3 tid : SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i < p.count)
        p.output[i] += p.offsets[i / 256u];
}
