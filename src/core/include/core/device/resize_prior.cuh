/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

template <int Channels>
__device__ bool prior_valid(const float* src, size_t i, size_t plane) {
    if constexpr (Channels == 1) {
        return isfinite(src[i]) && src[i] > 0.0f;
    } else {
        const float x = src[i], y = src[plane + i], z = src[2 * plane + i];
        // Matches kNormalLossMinPriorNorm, including quantized neutral PNGs.
        return isfinite(x) && isfinite(y) && isfinite(z) && x * x + y * y + z * z >= 0.25f;
    }
}

template <int Channels>
__global__ void resize_prior_kernel(const float* src, float* dst, int sw, int sh, int dw, int dh) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh)
        return;
    const size_t sp = static_cast<size_t>(sw) * sh, dp = static_cast<size_t>(dw) * dh;
    const size_t out = static_cast<size_t>(y) * dw + x;
    const float sx = fmaxf(0.0f, fminf(sw - 1.0f, (x + 0.5f) * sw / dw - 0.5f));
    const float sy = fmaxf(0.0f, fminf(sh - 1.0f, (y + 0.5f) * sh / dh - 0.5f));
    const int x0 = static_cast<int>(sx), y0 = static_cast<int>(sy);
    const size_t nearest = static_cast<size_t>(static_cast<int>(sy + 0.5f)) * sw + static_cast<int>(sx + 0.5f);
    float value[Channels] = {}, weight = 0.0f;
    if (prior_valid<Channels>(src, nearest, sp)) {
        for (int j = 0; j < 2; ++j) {
            for (int i = 0; i < 2; ++i) {
                const size_t index = static_cast<size_t>(min(y0 + j, sh - 1)) * sw + min(x0 + i, sw - 1);
                if (!prior_valid<Channels>(src, index, sp))
                    continue;
                const float w = (i ? sx - x0 : 1.0f - (sx - x0)) * (j ? sy - y0 : 1.0f - (sy - y0));
                weight += w;
                for (int c = 0; c < Channels; ++c)
                    value[c] += w * src[c * sp + index];
            }
        }
    }
    if constexpr (Channels == 3) {
        const float norm = sqrtf(value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
        weight = norm > 1e-8f ? norm : 0.0f;
    }
    for (int c = 0; c < Channels; ++c)
        dst[c * dp + out] = weight > 1e-8f ? value[c] / weight : 0.0f;
}
