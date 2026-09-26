/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/internal/private_access.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#ifdef __CUDACC__
#define LFS_SH_INLINE __host__ __device__ inline
#else
#define LFS_SH_INLINE inline
#endif
namespace lfs::core::internal::sh {
    LFS_SH_INLINE size_t offset(size_t row, uint32_t cell, uint32_t rest, bool quantized) {
        if (quantized)
            return (row / 32 * (rest * 3) + cell) * 32 + row % 32;
        const uint32_t slots = (rest * 3 + 3) / 4;
        return ((row / 32 * slots + cell / 4) * 32 + row % 32) * 4 + cell % 4;
    }
    LFS_SH_INLINE float decode(uint16_t code, float lo, float hi) {
        return fmaf(hi - lo, float(code) * (1.0f / 65535.0f), lo);
    }
    LFS_SH_INLINE uint16_t encode(float value, float lo, float hi) {
        const float range = fmaxf(hi - lo, 1e-20f);
        const float scaled = (65535.0f * (value - lo)) * (1.0f / range);
        return static_cast<uint16_t>(fminf(fmaxf(roundf(scaled), 0.0f), 65535.0f));
    }
} // namespace lfs::core::internal::sh
#undef LFS_SH_INLINE
