/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::core::cuda {
    __host__ __device__ __forceinline__ float normalize_mask_byte(const std::uint8_t value) {
#if defined(__CUDA_ARCH__)
        return __fdiv_rn(static_cast<float>(value), 255.0f);
#else
        return static_cast<float>(value) / 255.0f;
#endif
    }

    __host__ __device__ __forceinline__ float normalize_mask_u16_value(const std::uint16_t value) {
#if defined(__CUDA_ARCH__)
        return __fdiv_rn(static_cast<float>(value), 65535.0f);
#else
        return static_cast<float>(value) / 65535.0f;
#endif
    }

#if defined(__CUDACC__)
    // Explicit non-FTZ comparison: a positive subnormal cutoff excludes zero.
    __device__ __forceinline__ bool mask_value_at_least(const float value, const float threshold) {
        unsigned result;
        asm("{ .reg .pred p; setp.ge.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }"
            : "=r"(result) : "f"(value), "f"(threshold));
        return result != 0;
    }
#endif
} // namespace lfs::core::cuda
