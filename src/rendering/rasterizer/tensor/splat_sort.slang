// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
// Stable LSD radix sort, 8 bits per pass, of uint64 keys (or uint32 keys when
// built with LFS_SORT_KEY32) carrying uint32 values. Blocks of 2048 entries;
// the entry count is status.required and dispatches are indirect, so nothing
// waits on the host. Ported from the native Metal viewer rasterizer.
#include "splat_types.slang"

#ifndef LFS_SORT_KEY32
#define LFS_SORT_KEY32 0
#endif

struct Parameters {
    Ptr<uint64_t, Access.Read> keys_in;    // uint32 keys when LFS_SORT_KEY32
    Ptr<uint, Access.Read> values_in;
    Ptr<uint64_t, Access.ReadWrite> keys_out;
    Ptr<uint, Access.ReadWrite> values_out;
    Ptr<uint, Access.ReadWrite> histogram;         // [256][blocks] digit counts
    Ptr<uint, Access.ReadWrite> histogram_offsets; // [256][blocks] exclusive per digit
    Ptr<uint, Access.ReadWrite> digit_offsets;     // [257] exclusive over digits
    Ptr<RasterStatus, Access.Read> status;
    uint shift;
    uint padding;
};
[[vk::push_constant]] ConstantBuffer<Parameters> p;

uint64_t key_at(Ptr<uint64_t, Access.Read> keys, uint64_t index) {
#if LFS_SORT_KEY32
    return uint64_t(((Ptr<uint, Access.Read>)keys)[index]);
#else
    return keys[index];
#endif
}
void store_key(uint64_t index, uint64_t key) {
#if LFS_SORT_KEY32
    ((Ptr<uint, Access.ReadWrite>)p.keys_out)[index] = uint(key);
#else
    p.keys_out[index] = key;
#endif
}
// Lanes of this subgroup whose `valid` digit equals this lane's digit.
uint digit_peers(bool valid, uint digit) {
    uint peers = WaveActiveBallot(valid).x;
    for (uint bit = 0; bit < 8; ++bit) {
        const bool set = ((digit >> bit) & 1u) != 0u;
        const uint vote = WaveActiveBallot(set).x;
        peers &= set ? vote : ~vote;
    }
    return peers;
}

groupshared uint counts[8 * 256];

[shader("compute")]
[numthreads(256, 1, 1)]
void sort_histogram(uint3 group_id : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint lane = thread.x, sg = lane / 32u, group = group_id.x;
    for (uint s = 0; s < 8; ++s)
        counts[s * 256u + lane] = 0u;
    GroupMemoryBarrierWithGroupSync();
    const RasterStatus status = p.status[0];
    const uint64_t n = status.error != 0 ? 0ul : status.required;
    for (uint j = 0; j < 8; ++j) {
        const uint64_t i = uint64_t(group) * 2048ul + j * 256u + lane;
        const bool valid = i < n;
        const uint digit = valid ? uint((key_at(p.keys_in, i) >> p.shift) & 255ul) : 0u;
        const uint peers = digit_peers(valid, digit);
        // One atomic per distinct digit and subgroup; integer counts are exact.
        if (valid && (peers & ((1u << (lane & 31u)) - 1u)) == 0u)
            InterlockedAdd(counts[sg * 256u + digit], countbits(peers));
    }
    GroupMemoryBarrierWithGroupSync();
    uint total = 0;
    for (uint s = 0; s < 8; ++s)
        total += counts[s * 256u + lane];
    p.histogram[lane * live_sort_blocks(status) + group] = total;
}

groupshared uint scan_sums[8];

// One workgroup per digit scans that digit's counts across all blocks.
[shader("compute")]
[numthreads(256, 1, 1)]
void sort_digit_scan(uint3 group_id : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint lane = thread.x, sg = lane / 32u, digit = group_id.x;
    const uint blocks = live_sort_blocks(p.status[0]);
    uint carry = 0;
    for (uint block = 0; block < blocks; block += 256u) {
        const uint i = block + lane;
        const uint value = i < blocks ? p.histogram[digit * blocks + i] : 0u;
        uint prefix = WavePrefixSum(value);
        const uint total = WaveActiveSum(value);
        if ((lane & 31u) == 0u)
            scan_sums[sg] = total;
        GroupMemoryBarrierWithGroupSync();
        for (uint j = 0; j < sg; ++j)
            prefix += scan_sums[j];
        if (i < blocks)
            p.histogram_offsets[digit * blocks + i] = carry + prefix;
        for (uint j = 0; j < 8u; ++j)
            carry += scan_sums[j];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u)
        p.digit_offsets[digit] = carry;
}

// Exclusive scan of the 256 digit totals; digit_offsets[256] is the total.
[shader("compute")]
[numthreads(256, 1, 1)]
void sort_digit_offsets(uint3 thread : SV_GroupThreadID) {
    const uint lane = thread.x, sg = lane / 32u;
    const uint value = p.digit_offsets[lane];
    uint prefix = WavePrefixSum(value);
    const uint total = WaveActiveSum(value);
    if ((lane & 31u) == 0u)
        scan_sums[sg] = total;
    GroupMemoryBarrierWithGroupSync();
    for (uint j = 0; j < sg; ++j)
        prefix += scan_sums[j];
    p.digit_offsets[lane] = prefix;
    if (lane == 255u)
        p.digit_offsets[256] = prefix + value;
}

groupshared uint digit_base[256];
groupshared uint offsets[8 * 256];

[shader("compute")]
[numthreads(256, 1, 1)]
void sort_scatter(uint3 group_id : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    const uint lane = thread.x, sg = lane / 32u, sl = lane & 31u, group = group_id.x;
    const RasterStatus status = p.status[0];
    const uint64_t n = status.error != 0 ? 0ul : status.required;
    const uint64_t base = uint64_t(group) * 2048ul;
    if (base >= n)
        return; // uniform for the whole group
    const uint blocks = live_sort_blocks(status);
    // A block whose entries share one digit is already stable: copy it.
    const uint homogeneous_digit = uint((key_at(p.keys_in, base) >> p.shift) & 255ul);
    const uint histogram_index = homogeneous_digit * blocks + group;
    const uint block_destination = p.histogram_offsets[histogram_index] + p.digit_offsets[homogeneous_digit];
    const uint next_destination = group + 1u < blocks
                                      ? p.histogram_offsets[histogram_index + 1u] + p.digit_offsets[homogeneous_digit]
                                      : p.digit_offsets[homogeneous_digit + 1u];
    const uint block_count = uint(min(2048ul, n - base));
    if (next_destination - block_destination == block_count) {
        for (uint offset = lane; offset < block_count; offset += 256u) {
            const uint64_t source = base + offset;
            store_key(block_destination + offset, key_at(p.keys_in, source));
            p.values_out[block_destination + offset] = p.values_in[source];
        }
        return;
    }
    digit_base[lane] = p.histogram_offsets[lane * blocks + group] + p.digit_offsets[lane];
    const uint below = (1u << sl) - 1u;
    for (uint64_t chunk = base; chunk < min(base + 2048ul, n); chunk += 256ul) {
        for (uint s = 0; s < 8; ++s)
            offsets[s * 256u + lane] = 0u;
        GroupMemoryBarrierWithGroupSync();
        const uint64_t i = chunk + lane;
        const bool valid = i < n;
        const uint64_t key = valid ? key_at(p.keys_in, i) : 0ul;
        const uint digit = uint((key >> p.shift) & 255ul);
        const uint peers = digit_peers(valid, digit);
        const uint rank = countbits(peers & below);
        if (valid && rank == 0u)
            offsets[sg * 256u + digit] = countbits(peers);
        GroupMemoryBarrierWithGroupSync();
        uint running = digit_base[lane];
        for (uint s = 0; s < 8; ++s) {
            const uint count = offsets[s * 256u + lane];
            offsets[s * 256u + lane] = running;
            running += count;
        }
        digit_base[lane] = running;
        GroupMemoryBarrierWithGroupSync();
        if (valid) {
            const uint destination = offsets[sg * 256u + digit] + rank;
            store_key(destination, key);
            p.values_out[destination] = p.values_in[i];
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
