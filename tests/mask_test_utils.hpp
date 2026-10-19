/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace lfs::test::mask {
    inline float exact_ratio(uint32_t numerator, uint32_t denominator) {
        return static_cast<float>(static_cast<double>(numerator) / denominator);
    }

    inline std::vector<uint8_t> byte_ramp(std::size_t count) {
        std::vector<uint8_t> values(count);
        for (std::size_t i = 0; i < count; ++i)
            values[i] = static_cast<uint8_t>(i % 256);
        return values;
    }

    inline bool same_float(float a, float b) {
        return std::bit_cast<uint32_t>(a) == std::bit_cast<uint32_t>(b);
    }
} // namespace lfs::test::mask
