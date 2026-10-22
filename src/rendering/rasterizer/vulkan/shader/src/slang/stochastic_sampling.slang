/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// A separate random variable for each (pixel, primitive, frame). In particular,
// do not reuse a pixel threshold for overlapping primitives: that changes the
// visibility distribution from alpha_i * T_i (SteadySplats, Algorithm 1).
[ForceInline]
uint stochastic_hash(uint value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

[ForceInline]
float stochastic_threshold(uint2 pixel, uint primitive, uint frame) {
    uint seed = stochastic_hash(pixel.x ^ stochastic_hash(pixel.y));
    seed = stochastic_hash(seed ^ stochastic_hash(primitive + 0x9e3779b9u));
    seed = stochastic_hash(seed ^ stochastic_hash(frame + 0x85ebca6bu));
    return float(seed >> 8) * (1.0f / 16777216.0f);
}
