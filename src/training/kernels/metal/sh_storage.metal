// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// SH-rest storage shared by the rasterizers, Sh, Morton and Adam: ports of
// core/cuda/sh_layout.cuh and core/sh_value_codec.cuh. Primitives are
// swizzled in groups of 32; float storage holds float4 slots per primitive,
// Q16 storage holds pad-dropped uint16 cells with bounds per 256 primitives.
// Safe math mode, as joint_adam.metal explains.
#pragma METAL fp math_mode(safe)

constant constexpr uint kShReorder = 32u;
constant constexpr uint kShMaxRest = 15u;
constant constexpr uint kShMaxSlots = 12u;

static uint sh_float4_slots(const uint coeffs_rest) {
    return coeffs_rest == 0u ? 0u : min(kShMaxSlots, (min(coeffs_rest, kShMaxRest) * 3u + 3u) / 4u);
}

// Index of a primitive's float4 slot in swizzled storage with `slots` per primitive.
static uint sh_slot_index(const uint primitive, const uint slot, const uint slots) {
    return (primitive / kShReorder) * (slots * kShReorder) + slot * kShReorder + primitive % kShReorder;
}

// Index of a primitive's float4 slot in swizzled float storage.
static uint sh_swizzled_index(const uint primitive, const uint slot, const uint layout_rest) {
    return sh_slot_index(primitive, slot, sh_float4_slots(layout_rest));
}

// Index of a primitive's cell in swizzled Q16 storage with `cells` per primitive.
static uint sh_q16_index(const uint primitive, const uint cell, const uint cells) {
    return sh_slot_index(primitive, cell, cells);
}

static float sh_q16_decode(const ushort q, const float lo, const float hi) {
    // nvcc contracts the CUDA codec's sum into an fma.
    return fma(hi - lo, float(q) * (1.0f / 65535.0f), lo);
}

static ushort sh_q16_encode(const float v, const float lo, const float hi) {
    const float range = fmax(hi - lo, 1e-20f);
    return ushort(fmin(fmax(round(65535.0f * (v - lo) / range), 0.0f), 65535.0f));
}

// Q16 block bounds from each lane's (lo, hi): zero when no lane held a value
// (lanes without values pass +-1e30).
static float2 threadgroup_q16_bounds(const float lo, const float hi, threadgroup float4* scratch, const uint lane,
                                     const uint simd_lane, const uint simd_group, const uint simd_groups) {
    const float4 red =
        threadgroup_min4(float4(lo, -hi, 0.0f, 0.0f), scratch, lane, simd_lane, simd_group, simd_groups);
    return red.x > -red.y ? float2(0.0f) : float2(red.x, -red.y);
}

#pragma METAL fp math_mode(fast)
