/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <stdexcept>

#define CHECK_CUDA(debug, name)                                                                                                       \
    if (debug) {                                                                                                                      \
        auto ret = cudaDeviceSynchronize();                                                                                           \
        if (ret != cudaSuccess) {                                                                                                     \
            std::cerr << "\n[CUDA ERROR] in " << name << " " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
            throw std::runtime_error(cudaGetErrorString(ret));                                                                        \
        }                                                                                                                             \
    }

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
