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
    struct tensor_half_t {
        std::uint16_t bits{};

        tensor_half_t() = default;
        tensor_half_t(const float value) : bits(radmath::floatToHalf(value)) {}

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
