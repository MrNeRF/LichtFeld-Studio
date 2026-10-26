/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "pointwise_runtime.hpp"
namespace {
    __global__ void arange_kernel(float* out, size_t n, float start, float step) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
            out[i] = start + static_cast<float>(i) * step;
    }
} // namespace
extern "C" {
int lfs_hip_pointwise_f32(LfsHipContext* context, uint32_t op, LfsHipView lhs, LfsHipView rhs, LfsHipView output, size_t count, float scalar, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && op < LFS_HIP_OP_COUNT, "Unknown HIP pointwise operation");
        DeviceScope device(context->device);
        using namespace lfs::core;
        switch (op) {
#include "hip_pointwise_f32.inc"
        default: throw std::invalid_argument("Unknown HIP pointwise operation");
        }
    });
}

int lfs_hip_pointwise_f16(LfsHipContext* context, uint32_t op, LfsHipView lhs, LfsHipView rhs, LfsHipView output, size_t count, float scalar, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && op < LFS_HIP_OP_COUNT, "Unknown HIP pointwise operation");
        DeviceScope device(context->device);
        using namespace lfs::core;
        switch (op) {
#include "hip_pointwise_f16.inc"
        default: throw std::invalid_argument("Unknown HIP pointwise operation");
        }
    });
}

int lfs_hip_arange_f32(LfsHipContext* context, LfsHipView dst, size_t count, float start, float step, LfsHipError* error) {
    return boundary(error, [&] {
        validate(context, dst.buffer, dst.offset, float_bytes(count));
        require(dst.offset % 4 == 0, "Unaligned arange output");
        DeviceScope device(context->device);
        if (!count)
            return;
        hipLaunchKernelGGL(arange_kernel, dim3(blocks(count)), dim3(256), 0, context->stream,
                           reinterpret_cast<float*>(static_cast<char*>(dst.buffer->data) + dst.offset), count, start, step);
        check(hipGetLastError());
    });
}
}
