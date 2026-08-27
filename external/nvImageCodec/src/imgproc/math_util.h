/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#ifdef __SSE__
#include <xmmintrin.h>
#endif
#ifdef __CUDA_ARCH__
#include <cuda_runtime.h>
#else
#include <algorithm>
#include <cmath>
#endif
#include "force_inline.h"
#include "host_dev.h"
#include <cstdint>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace nvimgcodec {

    /// @brief Round down and convert to integer
    inline NVIMGCODEC_HOST_DEV int floor_int(float x) {
#ifdef __CUDA_ARCH__
        return __float2int_rd(x);
#else
        return static_cast<int>(std::floor(x));
#endif
    }

    /// @brief Round up and convert to integer
    inline NVIMGCODEC_HOST_DEV int ceil_int(float x) {
#ifdef __CUDA_ARCH__
        return __float2int_ru(x);
#else
        return static_cast<int>(std::ceil(x));
#endif
    }

    /// @brief Round to nearest integer and convert
    inline NVIMGCODEC_HOST_DEV int round_int(float x) {
#ifdef __CUDA_ARCH__
        return __float2int_rn(x);
#else
        return static_cast<int>(std::roundf(x));
#endif
    }

    template <typename T>
    NVIMGCODEC_HOST_DEV constexpr T clamp(const T& value, const T& lo, const T& hi) {
// Going through intermediate value saves us a jump
#ifdef __CUDA_ARCH__
        return ::min(hi, ::max(value, lo));
#else
        T x = value < lo ? lo : value;
        return x > hi ? hi : x;
#endif
    }

    /// @brief Calculate square root reciprocal.
    NVIMGCODEC_HOST_DEV inline float rsqrt(float x) {
#ifdef __CUDA_ARCH__
        return __frsqrt_rn(x);
#elif defined __SSE__
        // Use SSE intrinsic and one Newton-Raphson refinement step
        // - faster and less hacky than the hack below.
        __m128 X = _mm_set_ss(x);
        __m128 tmp = _mm_rsqrt_ss(X);
        float y = _mm_cvtss_f32(tmp);
        return y * (1.5f - x * 0.5f * y * y);
#else
        // Fallback to bit-level hacking.
        // https://en.wikipedia.org/wiki/Fast_inverse_square_root
        int32_t i;
        float x2, y;
        x2 = x * 0.5f;
        y = x;
        i = *(const int32_t*)&y;
        i = 0x5F375A86 - (i >> 1);
        y = *(const float*)&i;
        // Two Newton-Raphson steps gives 5-6 significant digits
        y = y * (1.5f - (x2 * y * y));
        y = y * (1.5f - (x2 * y * y));
        return y;
#endif
    }

    /// @brief Calculate square root reciprocal.
    NVIMGCODEC_HOST_DEV inline double rsqrt(double x) {
        return 1.0 / sqrt(x);
    }

} // namespace nvimgcodec