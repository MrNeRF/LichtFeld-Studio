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

inline int checked_fastgs_instance_count(uint64_t value, uint64_t n_primitives, uint64_t n_tiles) {
    if (value > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "FastGS instance count exceeds 32-bit range: " + std::to_string(value) +
            " instances from " + std::to_string(n_primitives) +
            " primitives across " + std::to_string(n_tiles) + " tiles");
    }
    return static_cast<int>(value);
}
