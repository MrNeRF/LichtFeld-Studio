/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include <cuda_runtime.h>

namespace fast_lfs::rasterization::kernels {

    // One reduce-scatter exchange: lanes with `Offset` clear keep the low half
    // of the array and send the high half, the others do the opposite.
    template <int N, int Offset>
    __device__ __forceinline__ void halving_step(float* values, const unsigned lane) {
        constexpr int half = (N + 1) / 2;
        const bool upper = (lane & Offset) != 0;
#pragma unroll
        for (int i = 0; i < half; ++i) {
            const float low = values[i];
            const float high = i + half < N ? values[i + half] : 0.0f;
            values[i] = (upper ? high : low) + __shfl_xor_sync(0xffffffffu, upper ? low : high, Offset);
        }
    }

    // Element of the original `N`-element array that a lane holds at position
    // `element` after `Steps` halving steps at lane bits FirstBit,
    // FirstBit / 2, ..., or -1 when that position holds padding.
    template <int N, unsigned FirstBit, int Steps>
    __host__ __device__ constexpr int halving_index(const unsigned lane, const int element = 0) {
        if constexpr (Steps == 0) {
            return element < N ? element : -1;
        } else {
            constexpr int half = (N + 1) / 2;
            const int inner = halving_index<half, FirstBit / 2, Steps - 1>(lane, element);
            if (inner < 0)
                return -1;
            const int index = inner + ((lane & FirstBit) ? half : 0);
            return index < N ? index : -1;
        }
    }

    template <int Offset>
    __device__ __forceinline__ float reduce_max(float value) {
#pragma unroll
        for (int offset = Offset; offset > 0; offset >>= 1)
            value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
        return value;
    }

} // namespace fast_lfs::rasterization::kernels
