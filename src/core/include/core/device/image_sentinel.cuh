/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

__device__ __forceinline__ uint8_t sentinel_byte(const size_t index, const uint32_t seed) {
    uint32_t value = static_cast<uint32_t>(index) ^ seed;
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return static_cast<uint8_t>(value);
}
