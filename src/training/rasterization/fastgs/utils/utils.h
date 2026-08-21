/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/cuda_error.hpp"

#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>

#define LFS_FASTGS_CUDA_CALL(call, name)                          \
    do {                                                          \
        LFS_CUDA_CHECK_MSG((call), "FastGS operation: {}", name); \
    } while (0)

#define LFS_FASTGS_PHASE_CHECK(name)                                             \
    do {                                                                         \
        LFS_CUDA_CHECK_MSG(cudaGetLastError(), "FastGS phase launch: {}", name); \
        if (::lfs::core::cuda_sync_debug_enabled()) {                            \
            LFS_CUDA_CHECK_MSG(cudaDeviceSynchronize(),                          \
                               "FastGS phase synchronization: {}", name);        \
        }                                                                        \
    } while (0)

template <typename T>
inline __host__ __device__ T div_round_up(T value, T divisor) {
    return (value + divisor - 1) / divisor;
}

inline int checked_to_int(uint64_t value, const char* message) {
    if (value > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(message);
    }
    return static_cast<int>(value);
}

// Host-side EWA clip box (same IEEE expression the kernels used to evaluate per thread).
inline void ewa_clip_bounds(
    const float w,
    const float h,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    float& clip_left,
    float& clip_right,
    float& clip_top,
    float& clip_bottom) {
    clip_left = (-0.15f * w - cx) / fx;
    clip_right = (1.15f * w - cx) / fx;
    clip_top = (-0.15f * h - cy) / fy;
    clip_bottom = (1.15f * h - cy) / fy;
}

inline int checked_fastgs_instance_count(uint64_t value, uint64_t n_primitives, uint64_t n_tiles) {
    if (value > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "FastGS instance count exceeds 32-bit range: " + std::to_string(value) +
            " instances from " + std::to_string(n_primitives) +
            " primitives across " + std::to_string(n_tiles) + " tiles");
    }
    return static_cast<int>(value);
}
