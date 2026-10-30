/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <cuda_runtime.h>

namespace fast_lfs::rasterization::kernels {

    __device__ __forceinline__ bool transmittance_complete(float value) {
        return (__float_as_uint(value) & 0x80000000u) != 0;
    }

    __device__ __forceinline__ float transmittance_magnitude(float value) {
        return __uint_as_float(__float_as_uint(value) & 0x7fffffffu);
    }

    __device__ __forceinline__ float signed_transmittance(float value, bool complete) {
        return __uint_as_float((__float_as_uint(value) & 0x7fffffffu) | (complete ? 0x80000000u : 0u));
    }

} // namespace fast_lfs::rasterization::kernels
