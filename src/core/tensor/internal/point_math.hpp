/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/internal/private_access.hpp"
#include <cmath>

#ifdef __CUDACC__
#define LFS_POINT_HD __host__ __device__
#else
#define LFS_POINT_HD
#endif

namespace lfs::core::internal {
    LFS_POINT_HD inline float roundedAdd(float a, float b) {
#ifdef __CUDA_ARCH__
        return __fadd_rn(a, b);
#else
        volatile float result = a + b;
        return result;
#endif
    }
    LFS_POINT_HD inline float roundedSub(float a, float b) {
        return roundedAdd(a, -b);
    }
    LFS_POINT_HD inline float roundedMul(float a, float b) {
#ifdef __CUDA_ARCH__
        return __fmul_rn(a, b);
#else
        volatile float result = a * b;
        return result;
#endif
    }
    LFS_POINT_HD inline float roundedDiv(float a, float b) {
#ifdef __CUDA_ARCH__
        return __fdiv_rn(a, b);
#else
        return a / b;
#endif
    }
    LFS_POINT_HD inline float roundedSqrt(float a) {
#ifdef __CUDA_ARCH__
        return __fsqrt_rn(a);
#else
        return std::sqrt(a);
#endif
    }
    LFS_POINT_HD inline bool pointFinite(float value) {
#ifdef __CUDA_ARCH__
        return isfinite(value);
#else
        return std::isfinite(value);
#endif
    }
    LFS_POINT_HD inline float pointDot3(float a, float x, float b, float y, float c, float z) {
        return roundedAdd(roundedAdd(roundedMul(a, x), roundedMul(b, y)), roundedMul(c, z));
    }
} // namespace lfs::core::internal
