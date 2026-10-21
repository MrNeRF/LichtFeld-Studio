/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "probe_api.h"
#include <cstdio>
#include <hip/hip_runtime.h>

namespace {
    __global__ void affine(const float* input, float* output, size_t count) {
        const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < count)
            output[i] = input[i] * 1.25f + 0.125f;
    }
} // namespace

#if defined(_WIN32) && !defined(__HIP_DEVICE_COMPILE__)
#define LFS_HIP_PROBE_EXPORT __declspec(dllexport)
#else
#define LFS_HIP_PROBE_EXPORT __attribute__((visibility("default")))
#endif

extern "C" LFS_HIP_PROBE_EXPORT int lfs_hip_probe_affine(
    const float* input, float* output, const size_t count,
    char* error, const size_t error_capacity) {
    if (!error || !error_capacity)
        return 1;
    error[0] = '\0';
    if (!input || !output || count > SIZE_MAX / sizeof(float)) {
        std::snprintf(error, error_capacity, "Invalid probe input");
        return 1;
    }
    if (!count)
        return 0;
    const size_t bytes = count * sizeof(float);
    float* device_input = nullptr;
    float* device_output = nullptr;
    auto status = hipSetDevice(0);
    if (status == hipSuccess)
        status = hipMalloc(&device_input, bytes);
    if (status == hipSuccess)
        status = hipMalloc(&device_output, bytes);
    if (status == hipSuccess)
        status = hipMemcpy(device_input, input, bytes, hipMemcpyHostToDevice);
    if (status == hipSuccess) {
        hipLaunchKernelGGL(affine, dim3(static_cast<unsigned>((count + 255) / 256)),
                           dim3(256), 0, 0, device_input, device_output, count);
        status = hipGetLastError();
    }
    if (status == hipSuccess)
        status = hipMemcpy(output, device_output, bytes, hipMemcpyDeviceToHost);
    const auto release_output = device_output ? hipFree(device_output) : hipSuccess;
    const auto release_input = device_input ? hipFree(device_input) : hipSuccess;
    if (status == hipSuccess)
        status = release_output;
    if (status == hipSuccess)
        status = release_input;
    if (status != hipSuccess) {
        std::snprintf(error, error_capacity, "%s", hipGetErrorString(status));
        return 1;
    }
    return 0;
}
