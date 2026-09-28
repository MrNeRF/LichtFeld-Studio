// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// MortonOps: ports of morton_reorder_kernels.cu (codes, joint moment
// permutation) and the gathered Q16 kernels of sh_value_quant_kernels.cu.
// Safe math mode, as joint_adam.metal explains.
#pragma METAL fp math_mode(safe)

constant constexpr float kMortonInf = 1e30f;
constant constexpr uint kMortonSimdGroups = kThreadgroupWidth / 32;

static uint morton_part1by2(uint x) {
    x &= 0x000003ffu;
    x = (x ^ (x << 16)) & 0xff0000ffu;
    x = (x ^ (x << 8)) & 0x0300f00fu;
    x = (x ^ (x << 4)) & 0x030c30c3u;
    x = (x ^ (x << 2)) & 0x09249249u;
    return x;
}

static uint morton_axis(const float v, const float lo, const float hi) {
    const float length = hi - lo;
    // The CUDA side divides on the host.
    const float mul = length == 0.0f ? 0.0f : precise::divide(1024.0f, length);
    const float t = (v - lo) * mul;
    if (!(t > 0.0f))
        return 0u;
    return min(1023u, uint(t));
}

struct MortonCodeParams {
    device const float* means;
    device const float* lo;
    device const float* hi;
    device uint* codes;
    uint count;
};

kernel void morton_codes(constant MortonCodeParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count)
        return;
    const uint x = morton_axis(p.means[i * 3 + 0], p.lo[0], p.hi[0]);
    const uint y = morton_axis(p.means[i * 3 + 1], p.lo[1], p.hi[1]);
    const uint z = morton_axis(p.means[i * 3 + 2], p.lo[2], p.hi[2]);
    p.codes[i] = (morton_part1by2(z) << 2) + (morton_part1by2(y) << 1) + morton_part1by2(x);
}

// `width` is attributes per row, or float4 slots per primitive when swizzled.
// Encode writes slots [slot_begin, slot_end) as a slot_end - slot_begin wide
// swizzled array.
struct MortonJointParams {
    device const uchar* source;
    device const float4* source_bounds;
    device uchar* destination;
    device float4* destination_bounds;
    device const long* permutation;
    int primitives;
    uint width;
    uint swizzled;
    int bits;
    uint slot_begin;
    uint slot_end;
};

static long morton_source_cell(constant MortonJointParams& p, const uint prim, const uint i) {
    return p.swizzled != 0u ? long(sh_slot_index(prim, i / 4u, p.width)) * 4 + i % 4u : long(prim) * p.width + i;
}

template <int BITS>
static float4 morton_joint_range(constant MortonJointParams& p, const uint source) {
    using C = JointCodec<BITS>;
    const float4 mm = p.source_bounds[source / uint(kJointBlock)];
    const uint cells = p.swizzled != 0u ? p.width * 4u : p.width;
    float4 local = float4(kMortonInf, -kMortonInf, kMortonInf, -kMortonInf);
    for (uint i = 0; i < cells; ++i) {
        const float2 us = C::decode_us(p.source, morton_source_cell(p, source, i), mm);
        local = float4(fmin(local.x, us.x), fmax(local.y, us.x), fmin(local.z, us.y), fmax(local.w, us.y));
    }
    return local;
}

kernel void morton_joint_bounds(constant MortonJointParams& p [[buffer(0)]],
                                uint group [[threadgroup_position_in_grid]],
                                uint lane [[thread_position_in_threadgroup]],
                                uint simd_lane [[thread_index_in_simdgroup]],
                                uint simd_group [[simdgroup_index_in_threadgroup]],
                                uint simd_groups [[simdgroups_per_threadgroup]]) {
    threadgroup float4 scratch[kMortonSimdGroups];
    const int prim = int(group) * kJointBlock + int(lane);
    float4 local = float4(kMortonInf, -kMortonInf, kMortonInf, -kMortonInf);
    if (prim < p.primitives && p.width > 0u) {
        const long source = p.permutation[prim];
        if (source >= 0 && source < p.primitives)
            local = p.bits == 16 ? morton_joint_range<16>(p, uint(source)) : morton_joint_range<8>(p, uint(source));
    }
    const float4 mm = threadgroup_joint_bounds(local, scratch, lane, simd_lane, simd_group, simd_groups);
    if (lane == 0)
        p.destination_bounds[group] = mm;
}

template <int BITS>
static void morton_joint_row(constant MortonJointParams& p, const uint prim, const uint source) {
    using C = JointCodec<BITS>;
    const float4 source_mm = p.source_bounds[source / uint(kJointBlock)];
    const float4 mm = p.destination_bounds[prim / uint(kJointBlock)];
    if (p.swizzled == 0u) {
        for (uint a = 0; a < p.width; ++a) {
            const float2 us = C::decode_us(p.source, morton_source_cell(p, source, a), source_mm);
            C::encode_us(p.destination, long(prim) * p.width + a, us.x, us.y, mm);
        }
        return;
    }
    const uint group_slots = p.slot_end - p.slot_begin;
    for (uint k = p.slot_begin; k < p.slot_end; ++k) {
        for (uint c = 0; c < 4u; ++c) {
            const float2 us = C::decode_us(p.source, morton_source_cell(p, source, k * 4u + c), source_mm);
            C::encode_us(p.destination, long(sh_slot_index(prim, k - p.slot_begin, group_slots)) * 4 + c, us.x, us.y,
                         mm);
        }
    }
}

kernel void morton_joint_encode(constant MortonJointParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (int(i) >= p.primitives || p.width == 0u)
        return;
    const long source = p.permutation[i];
    if (source < 0 || source >= p.primitives)
        return;
    if (p.bits == 16)
        morton_joint_row<16>(p, i, uint(source));
    else
        morton_joint_row<8>(p, i, uint(source));
}

// Encode writes cells [cell_begin, cell_end) as a cell_end - cell_begin wide
// swizzled array.
struct MortonQ16Params {
    device const ushort* source;
    device const float2* source_bounds;
    device const long* permutation;
    device float2* destination_bounds;
    device ushort* destination;
    uint count;
    uint cells;
    uint cell_begin;
    uint cell_end;
};

static bool morton_q16_source(constant MortonQ16Params& p, const uint prim, thread uint& source) {
    const long index = p.permutation[prim];
    source = uint(index);
    return index >= 0 && index < long(p.count);
}

// Block bounds of the gathered values; a source out of range reads as zeros.
kernel void morton_q16_bounds(constant MortonQ16Params& p [[buffer(0)]], uint group [[threadgroup_position_in_grid]],
                              uint lane [[thread_position_in_threadgroup]],
                              uint simd_lane [[thread_index_in_simdgroup]],
                              uint simd_group [[simdgroup_index_in_threadgroup]],
                              uint simd_groups [[simdgroups_per_threadgroup]]) {
    threadgroup float4 scratch[kMortonSimdGroups];
    const uint prim = group * 256u + lane;
    const uint cells = min(p.cells, 48u);
    float lo = kMortonInf;
    float hi = -kMortonInf;
    if (prim < p.count) {
        uint source;
        const bool valid = morton_q16_source(p, prim, source);
        const float2 mm = valid ? p.source_bounds[source / 256u] : float2(0.0f);
        for (uint c = 0; c < cells; ++c) {
            const float v = valid ? sh_q16_decode(p.source[sh_q16_index(source, c, p.cells)], mm.x, mm.y) : 0.0f;
            lo = fmin(lo, v);
            hi = fmax(hi, v);
        }
    }
    const float2 mm = threadgroup_q16_bounds(lo, hi, scratch, lane, simd_lane, simd_group, simd_groups);
    if (lane == 0)
        p.destination_bounds[group] = mm;
}

kernel void morton_q16_encode(constant MortonQ16Params& p [[buffer(0)]], uint prim [[thread_position_in_grid]]) {
    if (prim >= p.count)
        return;
    uint source;
    const bool valid = morton_q16_source(p, prim, source);
    const float2 source_mm = valid ? p.source_bounds[source / 256u] : float2(0.0f);
    const float2 mm = p.destination_bounds[prim / 256u];
    const uint group_cells = p.cell_end - p.cell_begin;
    for (uint c = p.cell_begin; c < p.cell_end; ++c) {
        const float v =
            valid ? sh_q16_decode(p.source[sh_q16_index(source, c, p.cells)], source_mm.x, source_mm.y) : 0.0f;
        p.destination[sh_q16_index(prim, c - p.cell_begin, group_cells)] = sh_q16_encode(v, mm.x, mm.y);
    }
}

#pragma METAL fp math_mode(fast)
