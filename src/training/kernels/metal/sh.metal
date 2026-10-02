// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// ShOps: ports of core/cuda/sh_value_quant_kernels.cu (Q16 encode, decode,
// touched-block reencode, block runs) and core/cuda/kernels/sh_layout.cu
// (swizzled row gathers and scatters, ranged deswizzle).
// Safe math mode, as joint_adam.metal explains.
#pragma METAL fp math_mode(safe)

constant constexpr uint kShMaxCells = 48u;
constant constexpr float kShInf = 1e30f;
constant constexpr uint kShSimdGroups = kThreadgroupWidth / 32;

struct ShEncodeParams {
    device const float4* source;
    device ushort* codes;
    device float2* bounds;
    uint primitives;
    uint slots;
    uint cells;
};

// One threadgroup per 256-primitive quant block.
kernel void sh_encode_q16(constant ShEncodeParams& p [[buffer(0)]], uint group [[threadgroup_position_in_grid]],
                          uint lane [[thread_position_in_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]],
                          uint simd_group [[simdgroup_index_in_threadgroup]],
                          uint simd_groups [[simdgroups_per_threadgroup]]) {
    threadgroup float4 scratch[kShSimdGroups];
    const uint prim = group * 256u + lane;
    const bool in_range = prim < p.primitives;
    const uint cells = min(p.cells, kShMaxCells);
    float values[kShMaxCells];
    float lo = kShInf;
    float hi = -kShInf;
    if (in_range) {
        for (uint c = 0; c < cells; ++c) {
            const float v = c / 4u < p.slots ? p.source[sh_slot_index(prim, c / 4u, p.slots)][c % 4u] : 0.0f;
            values[c] = v;
            lo = fmin(lo, v);
            hi = fmax(hi, v);
        }
    }
    const float2 mm = threadgroup_q16_bounds(lo, hi, scratch, lane, simd_lane, simd_group, simd_groups);
    if (lane == 0)
        p.bounds[group] = mm;
    if (!in_range)
        return;
    for (uint c = 0; c < cells; ++c)
        p.codes[sh_q16_index(prim, c, p.cells)] = sh_q16_encode(values[c], mm.x, mm.y);
}

struct ShDecodeParams {
    device const ushort* codes;
    device const float2* bounds;
    device float4* destination;
    uint primitives;
    uint slots;
    uint cells;
};

kernel void sh_decode_q16(constant ShDecodeParams& p [[buffer(0)]], uint prim [[thread_position_in_grid]]) {
    if (prim >= p.primitives)
        return;
    const float2 mm = p.bounds[prim / 256u];
    for (uint k = 0; k < p.slots; ++k) {
        float4 value = float4(0.0f);
        for (uint c = 0; c < 4u; ++c) {
            if (k * 4u + c < p.cells)
                value[c] = sh_q16_decode(p.codes[sh_q16_index(prim, k * 4u + c, p.cells)], mm.x, mm.y);
        }
        p.destination[sh_slot_index(prim, k, p.slots)] = value;
    }
}

struct ShBlockIdParams {
    device const long* indices;
    device float* ids;
    uint count;
};

kernel void sh_block_ids(constant ShBlockIdParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count)
        return;
    const long destination = p.indices[i];
    p.ids[i] = destination < 0 ? -1.0f : float(destination / 256);
}

struct ShRunParams {
    device const float* keys;
    device int* starts;
    device const int* inclusive;
    device int* unique;
    device int* offsets;
    device int* run_count;
    uint count;
};

kernel void sh_run_starts(constant ShRunParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count)
        return;
    p.starts[i] = i == 0 || p.keys[i] != p.keys[i - 1] ? 1 : 0;
}

// inclusive is the inclusive scan of starts: run r begins where it reaches r + 1.
kernel void sh_run_compact(constant ShRunParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count)
        return;
    if (p.starts[i] != 0) {
        const int run = p.inclusive[i] - 1;
        p.unique[run] = int(p.keys[i]);
        p.offsets[run] = int(i);
    }
    if (i == p.count - 1)
        *p.run_count = p.inclusive[i];
}

struct ShReencodeParams {
    device ushort* codes;
    device float2* bounds;
    device const float* canonical;
    device const long* canonical_rows;
    device const long* destinations;
    device const int* unique;
    device const int* offsets;
    device const int* run_count;
    uint sorted_count;
    uint primitives;
    uint decode_rows;
    uint cells;
};

// One threadgroup per touched quant block: decodes it, overwrites the touched
// rows (the last matching sorted row wins) and reencodes under new bounds.
kernel void sh_reencode_touched(constant ShReencodeParams& p [[buffer(0)]], uint group [[threadgroup_position_in_grid]],
                                uint lane [[thread_position_in_threadgroup]],
                                uint simd_lane [[thread_index_in_simdgroup]],
                                uint simd_group [[simdgroup_index_in_threadgroup]],
                                uint simd_groups [[simdgroups_per_threadgroup]]) {
    threadgroup float4 scratch[kShSimdGroups];
    const int runs = *p.run_count;
    if (int(group) >= runs)
        return;
    const int block = p.unique[group];
    if (block < 0)
        return;
    const uint block_start = uint(block) * 256u;
    if (block_start >= p.primitives)
        return;
    const uint n_in = min(p.primitives - block_start, 256u);
    const uint prim = block_start + lane;
    const bool in_range = lane < n_in;
    const uint n_decode = p.decode_rows > block_start ? min(p.decode_rows - block_start, n_in) : 0u;
    const uint cells = min(p.cells, kShMaxCells);

    const float2 old_mm = p.bounds[block];
    float values[kShMaxCells];
    if (in_range) {
        for (uint c = 0; c < cells; ++c)
            values[c] = lane < n_decode ? sh_q16_decode(p.codes[sh_q16_index(prim, c, p.cells)], old_mm.x, old_mm.y)
                                        : 0.0f;
        const int run_end = int(group) + 1 == runs ? int(p.sorted_count) : p.offsets[group + 1];
        for (int i = p.offsets[group]; i < run_end; ++i) {
            if (p.destinations[i] != long(prim))
                continue;
            const long row = p.canonical_rows != nullptr ? p.canonical_rows[i] : long(i);
            device const float* source = p.canonical + row * p.cells;
            for (uint c = 0; c < cells; ++c)
                values[c] = source[c];
        }
    }

    float lo = kShInf;
    float hi = -kShInf;
    if (in_range) {
        for (uint c = 0; c < cells; ++c) {
            lo = fmin(lo, values[c]);
            hi = fmax(hi, values[c]);
        }
    }
    const float2 mm = threadgroup_q16_bounds(lo, hi, scratch, lane, simd_lane, simd_group, simd_groups);
    if (lane == 0)
        p.bounds[block] = mm;
    if (!in_range)
        return;
    for (uint c = 0; c < cells; ++c)
        p.codes[sh_q16_index(prim, c, p.cells)] = sh_q16_encode(values[c], mm.x, mm.y);
}

constant constexpr uint kShRangeFloat32 = 0u;
constant constexpr uint kShRangeHalf = 1u;
constant constexpr uint kShRangeQ16 = 2u;

// `width` is Q16 cells per primitive, or float4 slots for float and half storage.
struct ShRangeParams {
    device const void* values;
    device const float2* bounds;
    device float* destination;
    ulong offset;
    ulong count;
    uint floats_per_primitive;
    uint width;
    uint storage;
};

kernel void sh_decode_range(constant ShRangeParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (ulong(i) >= p.count)
        return;
    const ulong canonical = p.offset + i;
    const uint prim = uint(canonical / p.floats_per_primitive);
    const uint cell = uint(canonical % p.floats_per_primitive);
    if (p.storage == kShRangeQ16) {
        const float2 mm = p.bounds[prim / 256u];
        p.destination[i] =
            sh_q16_decode(static_cast<device const ushort*>(p.values)[sh_q16_index(prim, cell, p.width)], mm.x, mm.y);
        return;
    }
    const ulong index = ulong(sh_slot_index(prim, cell / 4u, p.width)) * 4u + cell % 4u;
    p.destination[i] = p.storage == kShRangeHalf ? float(static_cast<device const half*>(p.values)[index])
                                                 : static_cast<device const float*>(p.values)[index];
}

struct ShZeroParams {
    device float4* values;
    device const int* indices;
    uint count;
    uint slots;
};

kernel void sh_zero_rows(constant ShZeroParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count)
        return;
    const uint prim = uint(p.indices[i]);
    for (uint k = 0; k < p.slots; ++k)
        p.values[sh_slot_index(prim, k, p.slots)] = float4(0.0f);
}

// Slots are float4 (16 bytes) or uchar4 moment cells (4 bytes).
struct ShGatherParams {
    device const void* source;
    device void* destination;
    device const void* indices;
    uint count;
    uint destination_offset;
    uint slots;
    uint wide_indices;
    uint slot_bytes;
};

kernel void sh_gather_swizzled(constant ShGatherParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count)
        return;
    const uint source = p.wide_indices != 0u ? uint(static_cast<device const long*>(p.indices)[i])
                                             : uint(static_cast<device const int*>(p.indices)[i]);
    const uint destination = p.destination_offset + i;
    for (uint k = 0; k < p.slots; ++k) {
        const uint from = sh_slot_index(source, k, p.slots);
        const uint to = sh_slot_index(destination, k, p.slots);
        if (p.slot_bytes == 16u)
            static_cast<device float4*>(p.destination)[to] = static_cast<device const float4*>(p.source)[from];
        else
            static_cast<device uint*>(p.destination)[to] = static_cast<device const uint*>(p.source)[from];
    }
}

struct ShCanonicalParams {
    device float4* swizzled;
    device float* canonical;
    device const void* indices;
    uint count;
    uint destination_offset;
    uint active_floats;
    uint slots;
    uint wide_indices;
};

static uint sh_canonical_primitive(constant ShCanonicalParams& p, const uint i) {
    if (p.indices == nullptr)
        return p.destination_offset + i;
    return p.wide_indices != 0u ? uint(static_cast<device const long*>(p.indices)[i])
                                : uint(static_cast<device const int*>(p.indices)[i]);
}

// Swizzled rows to canonical [count, rest, 3]; floats past the layout stay untouched.
kernel void sh_gather_canonical(constant ShCanonicalParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count)
        return;
    const uint prim = sh_canonical_primitive(p, i);
    device float* row = p.canonical + ulong(i) * p.active_floats;
    for (uint k = 0; k < p.slots && k * 4u < p.active_floats; ++k) {
        const float4 value = p.swizzled[sh_slot_index(prim, k, p.slots)];
        for (uint c = 0; c < 4u; ++c) {
            if (k * 4u + c < p.active_floats)
                row[k * 4u + c] = value[c];
        }
    }
}

// Canonical rows into swizzled slots at indices (scatter) or from
// destination_offset on (append); floats past the canonical row write zero.
kernel void sh_write_canonical(constant ShCanonicalParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count)
        return;
    const uint prim = sh_canonical_primitive(p, i);
    device const float* row = p.canonical + ulong(i) * p.active_floats;
    for (uint k = 0; k < p.slots; ++k) {
        float4 value = float4(0.0f);
        for (uint c = 0; c < 4u; ++c) {
            if (k * 4u + c < p.active_floats)
                value[c] = row[k * 4u + c];
        }
        p.swizzled[sh_slot_index(prim, k, p.slots)] = value;
    }
}

struct ShFillParams {
    device uchar* destination;
    ulong count;
    uint value;
};

kernel void sh_fill_bytes(constant ShFillParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (ulong(i) < p.count)
        p.destination[i] = uchar(p.value);
}

// Copies `rows` rows of `width` bytes between pitched buffers.
struct ShCopyParams {
    device const uchar* source;
    device uchar* destination;
    ulong width;
    ulong rows;
    ulong source_pitch;
    ulong destination_pitch;
};

kernel void sh_copy_rows(constant ShCopyParams& p [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (ulong(i) >= p.width * p.rows)
        return;
    const ulong row = ulong(i) / p.width;
    const ulong column = ulong(i) % p.width;
    p.destination[row * p.destination_pitch + column] = p.source[row * p.source_pitch + column];
}

#pragma METAL fp math_mode(fast)
