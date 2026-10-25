/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "nn_api.h"
namespace hip_nn {
    __device__ inline float row_sum(float x) {
        for (unsigned step = 16; step; step /= 2)
            x += __shfl_down(x, step, 32);
        return __shfl(x, 0, 32);
    }
    template <class T>
    __global__ void norm(const T* input, const T* weight, const T* bias, T* output, size_t rows, size_t cols, float eps) {
        const unsigned lane = threadIdx.x % 32;
        for (size_t row = size_t(blockIdx.x) * 4 + threadIdx.x / 32; row < rows; row += size_t(gridDim.x) * 4) {
            const auto base = row * cols;
            float mean = 0;
            if (bias) {
                float sum = 0;
                for (size_t col = lane; col < cols; col += 32)
                    sum += float(input[base + col]);
                mean = row_sum(sum) / float(cols);
            }
            float squares = 0;
            for (size_t col = lane; col < cols; col += 32) {
                const float x = float(input[base + col]) - mean;
                squares = fmaf(x, x, squares);
            }
            const float deviation = sqrtf(row_sum(squares) / float(cols) + eps);
            for (size_t col = lane; col < cols; col += 32) {
                float y = (float(input[base + col]) - mean) / deviation * float(weight[col]);
                if (bias)
                    y += float(bias[col]);
                if constexpr (std::is_same_v<T, __half>)
                    output[base + col] = __float2half_rn(y);
                else
                    output[base + col] = y;
            }
        }
    }
} // namespace hip_nn
int lfs_hip_nn_norm(LfsHipContext* context, LfsHipView input, LfsHipView weight, LfsHipView bias, LfsHipView output, const LfsHipNorm* p, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && p && p->cols && p->rows <= size_t(INT32_MAX) / p->cols && std::isfinite(p->eps) && p->eps >= 0 &&
                    (p->dtype == LFS_HIP_F32 || p->dtype == LFS_HIP_F16),
                "Invalid HIP NN norm descriptor");
        if (!p->rows)
            return;
        const size_t width = p->dtype == LFS_HIP_F32 ? 4 : 2, bytes = p->rows * p->cols * width, weights = p->cols * width;
        validate(context, input.buffer, input.offset, bytes);
        validate(context, output.buffer, output.offset, bytes);
        validate(context, weight.buffer, weight.offset, weights);
        if (bias.buffer)
            validate(context, bias.buffer, bias.offset, weights);
        require(input.offset % width == 0 && output.offset % width == 0 && weight.offset % width == 0 && (!bias.buffer || bias.offset % width == 0), "Misaligned HIP NN norm storage");
        const auto disjoint = [&](LfsHipView view, size_t size) { return view.buffer != output.buffer || view.offset >= output.offset + bytes || output.offset >= view.offset + size; };
        require(disjoint(input, bytes) || input.offset == output.offset, "Partially overlapping HIP NN norm input/output");
        require(disjoint(weight, weights) && (!bias.buffer || disjoint(bias, weights)), "HIP NN norm output aliases affine weights");
        DeviceScope device(context->device);
        const auto ptr = [](LfsHipView view) -> void* { return view.buffer ? static_cast<char*>(view.buffer->data) + view.offset : nullptr; };
        const size_t groups = std::min<size_t>((p->rows + 3) / 4, 65535);
        const auto launch = [&]<class T>() {
            hip_nn::norm<T><<<groups, 128, 0, context->stream>>>(static_cast<const T*>(ptr(input)), static_cast<const T*>(ptr(weight)),
                                                                 static_cast<const T*>(ptr(bias)), static_cast<T*>(ptr(output)), p->rows, p->cols, p->eps);
            check(hipGetLastError());
        };
        if (p->dtype == LFS_HIP_F32)
            launch.template operator()<float>();
        else
            launch.template operator()<__half>();
        check(hipGetLastError());
    });
}
