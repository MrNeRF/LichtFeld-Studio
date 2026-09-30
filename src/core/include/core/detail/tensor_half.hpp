/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#ifndef LFS_HAS_CUDA
#define LFS_HAS_CUDA 1
#endif

#if LFS_HAS_CUDA
#include <cuda_fp16.h>
#else
#include "core/rad_dequant_math.hpp"
#include <cstdint>
#endif

namespace lfs::core::detail {

#if LFS_HAS_CUDA
#if defined(__CUDACC__)
#define LFS_TENSOR_HALF_HD __host__ __device__
#else
#define LFS_TENSOR_HALF_HD
#endif
    using tensor_half_t = __half;

    LFS_TENSOR_HALF_HD inline float tensor_half_to_float(const tensor_half_t value) {
        return __half2float(value);
    }

    LFS_TENSOR_HALF_HD inline tensor_half_t tensor_float_to_half(const float value) {
        return __float2half(value);
    }
#undef LFS_TENSOR_HALF_HD
#else
    // IEEE binary32 -> binary16 with round-to-nearest-even and subnormals, like
    // __float2half. radmath::floatToHalf flushes subnormals for the RAD format.
    inline std::uint16_t tensor_float_to_half_bits(const float value) {
        const std::uint32_t bits = radmath::floatToBits(value);
        const auto sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000u);
        const std::uint32_t magnitude = bits & 0x7fffffffu;
        if (magnitude >= 0x7f800000u) // inf, or NaN kept quiet
            return sign | 0x7c00u | (magnitude > 0x7f800000u ? 0x200u | ((magnitude >> 13) & 0x3ffu) : 0u);
        if (magnitude >= 0x477ff000u) // rounds past 65504
            return sign | 0x7c00u;
        if (magnitude >= 0x38800000u) { // normal half
            std::uint32_t rebased = magnitude - 0x38000000u;
            rebased += 0x0fffu + ((rebased >> 13) & 1u);
            return static_cast<std::uint16_t>(sign | (rebased >> 13));
        }
        if (magnitude <= 0x33000000u) // at most half the smallest subnormal
            return sign;
        const std::uint32_t mantissa = (magnitude & 0x7fffffu) | 0x800000u;
        const std::uint32_t shift = 126u - (magnitude >> 23);
        std::uint32_t half = mantissa >> shift;
        const std::uint32_t rest = mantissa & ((1u << shift) - 1u), halfway = 1u << (shift - 1u);
        if (rest > halfway || (rest == halfway && (half & 1u) != 0u))
            ++half;
        return static_cast<std::uint16_t>(sign | half);
    }

    struct tensor_half_t {
        std::uint16_t bits{};

        tensor_half_t() = default;
        tensor_half_t(const float value) : bits(tensor_float_to_half_bits(value)) {}

        operator float() const {
            return radmath::halfToFloat(bits);
        }
    };

    static_assert(sizeof(tensor_half_t) == sizeof(std::uint16_t));

    inline float tensor_half_to_float(const tensor_half_t value) {
        return static_cast<float>(value);
    }

    inline tensor_half_t tensor_float_to_half(const float value) {
        return tensor_half_t(value);
    }
#endif

} // namespace lfs::core::detail
