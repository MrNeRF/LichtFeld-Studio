// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// SH-rest storage shared by the rasterizers, Sh, Morton and Adam: ports of
// core/cuda/sh_layout.cuh and core/sh_value_codec.cuh. Primitives are
// swizzled in groups of 32; float storage holds float4 slots per primitive,
// Q16 storage holds pad-dropped uint16 cells with bounds per 256 primitives.

constant constexpr uint kShReorder = 32u;
constant constexpr uint kShMaxRest = 15u;
constant constexpr uint kShMaxSlots = 12u;

static uint sh_float4_slots(const uint coeffs_rest) {
    return coeffs_rest == 0u ? 0u : min(kShMaxSlots, (min(coeffs_rest, kShMaxRest) * 3u + 3u) / 4u);
}

// Index of a primitive's float4 slot in swizzled float storage.
static uint sh_swizzled_index(const uint primitive, const uint slot, const uint layout_rest) {
    return (primitive / kShReorder) * (sh_float4_slots(layout_rest) * kShReorder) + slot * kShReorder +
           primitive % kShReorder;
}

// Index of a primitive's cell in swizzled Q16 storage with `cells` per primitive.
static uint sh_q16_index(const uint primitive, const uint cell, const uint cells) {
    return (primitive / kShReorder) * (cells * kShReorder) + cell * kShReorder + primitive % kShReorder;
}

static float sh_q16_decode(const ushort q, const float lo, const float hi) {
    return lo + (hi - lo) * (float(q) * (1.0f / 65535.0f));
}

static ushort sh_q16_encode(const float v, const float lo, const float hi) {
    const float range = fmax(hi - lo, 1e-20f);
    return ushort(fmin(fmax(round(65535.0f * (v - lo) / range), 0.0f), 65535.0f));
}
