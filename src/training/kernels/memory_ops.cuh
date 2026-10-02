/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>

namespace lfs::core {

    /**
     * @brief Load float with streaming cache hint (bypass L1)
     *
     * Use for data that will only be accessed once (e.g., activations in forward pass).
     * This keeps L1 cache free for data that will be reused.
     *
     * Expected: 1.5-2× memory bandwidth improvement when L1 is under pressure
     */
    __device__ __forceinline__ float load_cs(const float* ptr) {
#if __CUDA_ARCH__ >= 350
        return __ldcs(ptr);
#else
        return __ldg(ptr);
#endif
    }

    /**
     * @brief Load float with read-only cache hint
     *
     * Use for read-only data that may be accessed multiple times within a kernel.
     */
    __device__ __forceinline__ float load_ro(const float* ptr) {
        return __ldg(ptr);
    }

    /**
     * @brief RGB triplet for vectorized loads
     */
    struct RGB {
        float r, g, b;

        __device__ __forceinline__ RGB() : r(0),
                                           g(0),
                                           b(0) {}
        __device__ __forceinline__ RGB(float r_, float g_, float b_) : r(r_),
                                                                       g(g_),
                                                                       b(b_) {}
    };

    /**
     * @brief Load RGB triplet with streaming cache hint
     *
     * Loads 3 consecutive floats with cache bypass.
     * More efficient than 3 separate loads due to coalescing.
     */
    __device__ __forceinline__ RGB load_rgb_cs(const float* ptr) {
        RGB result;
        result.r = load_cs(ptr + 0);
        result.g = load_cs(ptr + 1);
        result.b = load_cs(ptr + 2);
        return result;
    }

    /**
     * @brief Load RGB triplet with read-only cache hint
     */
    __device__ __forceinline__ RGB load_rgb_ro(const float* ptr) {
        RGB result;
        result.r = load_ro(ptr + 0);
        result.g = load_ro(ptr + 1);
        result.b = load_ro(ptr + 2);
        return result;
    }

    // RGB to grayscale conversion constants
    constexpr float kC2G_r = 0.299f;
    constexpr float kC2G_g = 0.587f;
    constexpr float kC2G_b = 0.114f;

} // namespace lfs::core
