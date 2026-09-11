/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <cstring>

namespace lfs::io::usd_flat {

    inline float half_to_float(const std::uint16_t bits) {
        const std::uint32_t sign = (static_cast<std::uint32_t>(bits & 0x8000u)) << 16u;
        const std::uint32_t exponent = (bits >> 10u) & 0x1fu;
        const std::uint32_t fraction = bits & 0x03ffu;
        std::uint32_t result = sign;

        if (exponent == 0) {
            if (fraction != 0) {
                std::uint32_t normalized = fraction;
                std::uint32_t shift = 0;
                while ((normalized & 0x0400u) == 0) {
                    normalized <<= 1u;
                    ++shift;
                }
                const std::uint32_t float_exponent = 127u - 15u - shift + 1u;
                result |= float_exponent << 23u;
                result |= (normalized & 0x03ffu) << 13u;
            }
        } else if (exponent == 0x1fu) {
            result |= 0x7f800000u | (fraction << 13u);
        } else {
            result |= (exponent + (127u - 15u)) << 23u;
            result |= fraction << 13u;
        }

        float value;
        static_assert(sizeof(value) == sizeof(result));
        std::memcpy(&value, &result, sizeof(value));
        return value;
    }

    inline std::uint16_t float_to_half(const float value) {
        std::uint32_t bits;
        static_assert(sizeof(value) == sizeof(bits));
        std::memcpy(&bits, &value, sizeof(bits));

        const std::uint32_t sign = (bits >> 16u) & 0x8000u;
        const std::uint32_t exponent = (bits >> 23u) & 0xffu;
        const std::uint32_t fraction = bits & 0x7fffffu;
        if (exponent == 0xffu) {
            return static_cast<std::uint16_t>(sign | 0x7c00u | (fraction ? 0x0200u : 0));
        }

        if (exponent < 102u) {
            return static_cast<std::uint16_t>(sign);
        }

        const auto round_to_nearest_even = [](const std::uint32_t mantissa, const unsigned shift) {
            const std::uint32_t truncated = mantissa >> shift;
            const std::uint32_t remainder = mantissa & ((1u << shift) - 1u);
            const std::uint32_t halfway = 1u << (shift - 1u);
            return truncated + (remainder > halfway || (remainder == halfway && (truncated & 1u) != 0u) ? 1u : 0u);
        };

        if (exponent < 113u) {
            const std::uint32_t mantissa = fraction | 0x800000u;
            const auto subnormal = round_to_nearest_even(mantissa, 126u - exponent);
            if (subnormal >= 0x400u) {
                return static_cast<std::uint16_t>(sign | 0x0400u);
            }
            return static_cast<std::uint16_t>(sign | subnormal);
        }
        if (exponent > 142u) {
            return static_cast<std::uint16_t>(sign | 0x7c00u);
        }

        const auto rounded = round_to_nearest_even(fraction, 13u);
        if (rounded == 0x400u) {
            const auto half_exponent = exponent - 111u;
            if (half_exponent == 31u) {
                return static_cast<std::uint16_t>(sign | 0x7c00u);
            }
            return static_cast<std::uint16_t>(sign | (half_exponent << 10u));
        }
        return static_cast<std::uint16_t>(sign | ((exponent - 112u) << 10u) | rounded);
    }

} // namespace lfs::io::usd_flat
