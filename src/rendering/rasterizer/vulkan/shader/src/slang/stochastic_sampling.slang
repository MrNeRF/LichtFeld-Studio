/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Never share one threshold across overlapping primitives (Algorithm 1).
// Basic stochastic mode is the independent white-noise reference.
layout(binding=16) ByteAddressBuffer stochastic_noise;
[ForceInline]
uint stochastic_hash(uint value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

[ForceInline]
float stochastic_threshold(uint pixel_seed, uint primitive_seed, uint frame_seed) {
    uint seed = stochastic_hash(pixel_seed ^ primitive_seed);
    seed = stochastic_hash(seed ^ frame_seed);
    return float(seed >> 8) * (1.0f / 16777216.0f);
}

// SteadySplats section 4.3: one shared scalar STBN volume, with per-primitive
// R2 offsets and a golden-ratio Cranley-Patterson rotation between 64-frame
// periods. Fixed-point arithmetic preserves the offsets for large IDs and
// frame serials. Prepare once per collected primitive, not once per fragment.
[ForceInline]
uint2 stochastic_primitive_noise(uint primitive) {
    uint x = (primitive * 0xc13fa9a9u) >> 25;
    uint y = (primitive * 0x91e10da5u) >> 25;
    return uint2(x | (y << 7), stochastic_hash(primitive + 0x9e3779b9u));
}

[ForceInline]
float stochastic_blue_threshold(uint2 pixel, uint2 primitive_noise, uint frame) {
    uint2 offset = uint2(primitive_noise.x, primitive_noise.x >> 7) & 127u;
    uint2 xy = (pixel + offset) & 127u;
    uint index = ((frame & 63u) * 128u + xy.y) * 128u + xy.x;
    uint value = (stochastic_noise.Load(index & ~3u) >> ((index & 3u) * 8u)) & 255u;
    // A full-precision primitive phase also avoids 8-bit threshold bias. Rotate
    // complete periods so consecutive frames retain the volume's temporal order.
    uint rotated = (value << 24) + primitive_noise.y + (frame >> 6) * 0x9e3779b9u;
    return float(rotated >> 8) * (1.0f / 16777216.0f);
}
