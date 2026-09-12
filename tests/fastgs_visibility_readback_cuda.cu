/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cuda_runtime.h>

namespace {
    __global__ void delay_visibility(const unsigned long long cycles) {
        const unsigned long long start = clock64();
        while (clock64() - start < cycles) {
            __nanosleep(1000);
        }
    }
} // namespace

cudaError_t fastgs_visibility_readback_delay(cudaStream_t stream, unsigned long long cycles) {
    delay_visibility<<<1, 1, 0, stream>>>(cycles);
    return cudaGetLastError();
}
