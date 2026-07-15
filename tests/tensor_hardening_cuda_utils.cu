/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cstdint>

#include <cuda_runtime.h>

namespace tensor_hardening {
    namespace {
        __global__ void delay_kernel(const uint64_t cycles) {
            const uint64_t start = clock64();
            while (clock64() - start < cycles) {
                __nanosleep(100);
            }
        }
    } // namespace

    cudaError_t launch_delay_kernel(const cudaStream_t stream, const uint64_t cycles) {
        delay_kernel<<<1, 1, 0, stream>>>(cycles);
        return cudaGetLastError();
    }
} // namespace tensor_hardening
