/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/cuda_error.hpp"

#include <cstdint>
#include <cuda_runtime.h>

#define LFS_EDGE_PHASE_CHECK(name)                                              \
    do {                                                                        \
        if (::lfs::core::cuda_sync_debug_enabled()) {                           \
            LFS_CUDA_CHECK_MSG(cudaDeviceSynchronize(),                         \
                               "edge-compute phase synchronization: {}", name); \
        }                                                                       \
    } while (false)

template <typename T>
inline __host__ __device__ T div_round_up(T value, T divisor) {
    return (value + divisor - 1) / divisor;
}
