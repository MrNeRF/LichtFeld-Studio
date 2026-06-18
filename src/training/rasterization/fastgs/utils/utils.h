/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

inline void check_cuda_result(cudaError_t ret, const char* name, const char* file, int line) {
    if (ret != cudaSuccess) {
        const std::string message = std::string("CUDA error in ") + name + " at " + file + ":" + std::to_string(line) +
                                    " - " + cudaGetErrorName(ret) + ": " + cudaGetErrorString(ret);
        std::cerr << "\n[CUDA ERROR] " << message;
        throw std::runtime_error(message);
    }
}

#define CUDA_CHECK(call, name)                               \
    do {                                                     \
        check_cuda_result((call), name, __FILE__, __LINE__); \
    } while (0)

#define CHECK_CUDA(debug, name)                                                   \
    do {                                                                          \
        check_cuda_result(cudaGetLastError(), name, __FILE__, __LINE__);          \
        if constexpr (debug) {                                                    \
            check_cuda_result(cudaDeviceSynchronize(), name, __FILE__, __LINE__); \
        }                                                                         \
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
