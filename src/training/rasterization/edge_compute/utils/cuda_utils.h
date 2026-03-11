/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define CHECK_CUDA_PTR(ptr, name)                                                         \
    if (!ptr) {                                                                           \
        throw std::runtime_error("Null pointer for " + std::string(name));                \
    }                                                                                     \
    {                                                                                     \
        cudaPointerAttributes attrs;                                                      \
        cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);                          \
        if (err != cudaSuccess) {                                                         \
            cudaGetLastError(); /* Clear error */                                         \
            throw std::runtime_error(std::string(name) + " is not a valid CUDA pointer"); \
        }                                                                                 \
        if (attrs.type != cudaMemoryTypeDevice) {                                         \
            throw std::runtime_error(std::string(name) + " is not a device pointer");     \
        }                                                                                 \
    }

#define CHECK_CUDA_PTR_OPTIONAL(ptr, name) \
    if (ptr) {                             \
        CHECK_CUDA_PTR(ptr, name)          \
    }

// Simple validation for CUDA pointers without throwing
inline bool is_valid_cuda_ptr(const void* ptr) {
    if (!ptr)
        return false;

    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    if (err != cudaSuccess) {
        cudaGetLastError(); // Clear error
        return false;
    }

    return attrs.type == cudaMemoryTypeDevice;
}

// Validate that a pointer is accessible from device (more lenient check)
inline bool is_device_accessible(const void* ptr) {
    if (!ptr)
        return false;

    // Try a simple test read
    float test_value;
    cudaError_t err = cudaMemcpy(&test_value, ptr, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaGetLastError(); // Clear error
        return false;
    }

    return true;
}
