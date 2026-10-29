/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "rasterization_config.h"
#include <cuda_runtime.h>

namespace fast_lfs::rasterization::kernels {

    // Persistent conics retain their covariance-inverse convention. Only the
    // raster staging copy uses a base-2 exponent, so tile geometry and its
    // derivatives never need to undo a rounded scale conversion.
    __device__ __forceinline__ float3 base2_conic(const float3 conic) {
        constexpr float diagonal = -0.7213475204444817f;
        constexpr float cross = -1.4426950408889634f;
        return make_float3(__fmul_rn(diagonal, conic.x),
                           __fmul_rn(cross, conic.y),
                           __fmul_rn(diagonal, conic.z));
    }

    __device__ __forceinline__ float exp2_approx(const float power) {
        float result;
        asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(power));
        return result;
    }

    __device__ __forceinline__ float rcp_approx(const float value) {
        float result;
        asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
        return result;
    }

    // Terms of one pixel column at dx = mean.x - pixel.x: both rows of a thread
    // share them, leaving two FMAs, one exp2 and one multiply per row. Forward,
    // backward and scoring evaluate rows only through this type, with explicit
    // rounding, so their acceptance decisions agree.
    struct AlphaColumn {
        float constant;
        float slope;
        float opacity;

        __device__ __forceinline__ static AlphaColumn at(const float3 conic_base2, const float opacity, const float dx) {
            return {__fmul_rn(__fmul_rn(conic_base2.x, dx), dx), __fmul_rn(conic_base2.y, dx), opacity};
        }

        __device__ __forceinline__ float alpha(const float c, const float dy) const {
            return __fmul_rn(opacity, exp2_approx(__fmaf_rn(__fmaf_rn(c, dy, slope), dy, constant)));
        }
    };

    __device__ __forceinline__ bool alpha_accepted(const float unclamped_alpha) {
        return unclamped_alpha >= config::min_alpha_threshold;
    }

    __device__ __forceinline__ float clamped_alpha(const float unclamped_alpha) {
        return fminf(unclamped_alpha, config::max_fragment_alpha);
    }

} // namespace fast_lfs::rasterization::kernels
