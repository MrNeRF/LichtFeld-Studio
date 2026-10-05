/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>

namespace lfs::rendering::metal {
    // A conservative admission estimate, not a measurement of driver allocation.
    // Inputs are 32-bit, while every product and scan total uses 64-bit arithmetic.
    constexpr uint64_t scanReservationBytes(uint64_t count, uint32_t element_bytes = 8) {
        uint64_t bytes = 0;
        do {
            count = (count + 255) / 256;
            bytes += count * (2 * element_bytes) + 2 * 65536;
        } while (count > 1);
        return bytes;
    }

    constexpr uint64_t frameReservationBytes(uint32_t width, uint32_t height,
                                             uint32_t splats, uint32_t instances, bool points) {
        const uint64_t pixels = ((uint64_t(width) + 63) / 64 * 64) *
                                ((uint64_t(height) + 63) / 64 * 64);
        if (points)
            return pixels * 12 + 8 * 65536;
        const uint64_t tiles = ((uint64_t(width) + 15) / 16) * ((uint64_t(height) + 15) / 16);
        const uint64_t histogram = ((uint64_t(instances) + 2047) / 2048) * 256;
        return pixels * 36 + uint64_t(splats) * 84 + uint64_t(instances) * 24 +
               histogram * 8 + tiles * 8 + scanReservationBytes(splats) +
               scanReservationBytes(histogram, 4) + 24 * 65536;
    }

    constexpr uint64_t viewportOutputReservationBytes(uint32_t width, uint32_t height, bool points) {
        const uint64_t pixels = ((uint64_t(width) + 63) / 64 * 64) *
                                ((uint64_t(height) + 63) / 64 * 64);
        return pixels * (points ? 12 : 36) + (points ? 8 : 24) * 65536;
    }

    constexpr bool frameFitsWorkingSet(uint64_t allocated, uint64_t reservation, uint64_t recommended) {
        if (!recommended)
            return true; // Driver did not expose a usable recommendation.
        const uint64_t budget = recommended - recommended / 5;
        return allocated <= budget && reservation <= budget - allocated;
    }
} // namespace lfs::rendering::metal
