/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "matrix_product_runtime.hpp"
#include "nn_api.h"
namespace hip_nn {
    __device__ inline float activate(float x, uint32_t mode) {
#pragma clang fp contract(off)
        if (mode == 1)
            return fmaxf(x, 0.f);
        if (mode == 2)
            return .5f * x * (1.f + tanhf(.7978845608028654f * (x + .044715f * x * x * x)));
        if (mode == 3) {
            const float a = fabsf(x * .7071067811865475f), t = 1.f / (1.f + .3275911f * a);
            const float r = 1.f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - .284496736f) * t + .254829592f) * t * expf(-a * a);
            return .5f * x * (1.f + (x < 0 ? -r : r));
        }
        if (mode == 4)
            return x / (1.f + expf(-x));
        return x;
    }
    template <class Out, class In>
    __device__ void linear_store(Out* output, size_t index, size_t column, float value, const In* bias, const In* scale, const Out* residual, uint32_t activation) {
#pragma clang fp contract(off)
        if (bias)
            value += float(bias[column]);
        value = activate(value, activation);
        if (scale)
            value *= float(scale[column]);
        if (residual)
            value += float(residual[index]);
        if constexpr (std::is_same_v<Out, __half>)
            output[index] = __float2half_rn(value);
        else
            output[index] = value;
    }
    template <class T, bool Transpose, class Weight = T, class Out = T, bool Masked = false>
    __global__ void linear_skinny(const T* a, const Weight* w, const T* bias, const T* scale, const Out* residual, Out* output,
                                  size_t m, size_t n, size_t k, size_t weight_stride, uint32_t activation, size_t output_pitch, bool row_affine, size_t a_stride, size_t output_stride, const uint8_t* validity = nullptr) {
        const size_t column = size_t(blockIdx.x) * 256 + threadIdx.x, row = blockIdx.y, batch = blockIdx.z;
        if (column >= n)
            return;
        const size_t ai = batch * a_stride + row * k, wi = batch * weight_stride;
        const float value = hip_matrix::dot<Transpose, Masked>(k ? a + ai : a, k ? w + wi : w, column, n, k, validity);
        linear_store(output, batch * output_stride + row * output_pitch + column, row_affine ? row : column, value, bias, scale, residual, activation);
    }
    template <class T, bool Transpose, class Weight = T, class Out = T, bool Masked = false>
    __global__ void linear_tiled(const T* a, const Weight* w, const T* bias, const T* scale, const Out* residual, Out* output,
                                 size_t m, size_t n, size_t k, size_t weight_stride, uint32_t activation, size_t output_pitch, bool row_affine, size_t a_stride, size_t output_stride, const uint8_t* validity = nullptr) {
        __shared__ float aa[16][64], bb[16][64];
        __shared__ uint8_t valid[Masked ? 16 : 1][Masked ? 64 : 1];
        const unsigned tx = threadIdx.x, ty = threadIdx.y;
        const size_t row = size_t(blockIdx.y) * 64, col = size_t(blockIdx.x) * 64, batch = blockIdx.z;
        if (k) {
            a += batch * a_stride;
            w += batch * weight_stride;
        }
        float sums[4][4]{};
        hip_matrix::tile<Transpose, Masked>(a, w, m, n, k, row, col, aa, bb, sums, &valid[0][0], validity);
        for (unsigned i = 0; i < 4; ++i)
            for (unsigned j = 0; j < 4; ++j) {
                const size_t r = row + ty + i * 16, c = col + tx + j * 16;
                if (r < m && c < n)
                    linear_store(output, batch * output_stride + r * output_pitch + c, row_affine ? r : c, sums[i][j], bias, scale, residual, activation);
            }
    }
} // namespace hip_nn
int lfs_hip_nn_linear(LfsHipContext* context, const LfsHipView* inputs, LfsHipView output, const LfsHipLinear* p, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && inputs && p && p->transpose_b <= 1 && p->batched_b <= 1 && p->activation <= 4 &&
                    (p->dtype == LFS_HIP_F32 || p->dtype == LFS_HIP_F16) && p->m <= 65535ull * 64,
                "Invalid HIP NN linear descriptor");
        if (!p->batch || !p->m || !p->n)
            return;
        const size_t width = p->dtype == LFS_HIP_F32 ? 4 : 2;
        const auto bytes = [&](uint64_t a, uint64_t b, uint64_t c) {
            require(a <= INT32_MAX && b <= INT32_MAX && c <= INT32_MAX && (!b || a <= uint64_t(INT32_MAX) / b) && (!c || a * b <= uint64_t(INT32_MAX) / c), "HIP NN linear shape exceeds indexing range");
            return size_t(a * b * c) * width;
        };
        const size_t out_bytes = bytes(p->batch, p->m, p->n);
        const size_t sizes[]{bytes(p->batch, p->m, p->k), bytes(p->batched_b ? p->batch : 1, p->k, p->n), bytes(1, 1, p->n), bytes(1, 1, p->n), out_bytes};
        validate(context, output.buffer, output.offset, out_bytes);
        require(output.offset % width == 0, "Misaligned HIP NN linear output");
        for (unsigned i = 0; i < 5; ++i) {
            if ((i >= 2 && !inputs[i].buffer) || !sizes[i])
                continue;
            validate(context, inputs[i].buffer, inputs[i].offset, sizes[i]);
            require(inputs[i].offset % width == 0, "Misaligned HIP NN linear input");
            require(inputs[i].buffer != output.buffer || inputs[i].offset >= output.offset + out_bytes || output.offset >= inputs[i].offset + sizes[i], "HIP NN linear output aliases input");
        }
        DeviceScope device(context->device);
        const auto ptr = [](LfsHipView view) -> void* { return view.buffer ? static_cast<char*>(view.buffer->data) + view.offset : nullptr; };
        const auto launch = [&]<class T, bool Transpose>() {
            for (size_t batch = 0; batch < p->batch; batch += 65535) {
                const size_t chunk = std::min<size_t>(65535, p->batch - batch), stride = p->batched_b ? p->k * p->n : 0;
                const auto* a = sizes[0] ? static_cast<const T*>(ptr(inputs[0])) + batch * p->m * p->k : nullptr;
                const auto* w = sizes[1] ? static_cast<const T*>(ptr(inputs[1])) + batch * stride : nullptr;
                const auto* bias = static_cast<const T*>(ptr(inputs[2]));
                const auto* scale = static_cast<const T*>(ptr(inputs[3]));
                const auto* residual = inputs[4].buffer ? static_cast<const T*>(ptr(inputs[4])) + batch * p->m * p->n : nullptr;
                auto* out = static_cast<T*>(ptr(output)) + batch * p->m * p->n;
                if (p->m <= 4)
                    hip_nn::linear_skinny<T, Transpose><<<dim3((p->n + 255) / 256, p->m, chunk), 256, 0, context->stream>>>(a, w, bias, scale, residual, out, p->m, p->n, p->k, stride, p->activation, p->n, false, p->m * p->k, p->m * p->n);
                else
                    hip_nn::linear_tiled<T, Transpose><<<dim3((p->n + 63) / 64, (p->m + 63) / 64, chunk), dim3(16, 16), 0, context->stream>>>(a, w, bias, scale, residual, out, p->m, p->n, p->k, stride, p->activation, p->n, false, p->m * p->k, p->m * p->n);
                check(hipGetLastError());
            }
        };
        if (p->dtype == LFS_HIP_F32) {
            if (p->transpose_b)
                launch.template operator()<float, true>();
            else
                launch.template operator()<float, false>();
        } else {
            if (p->transpose_b)
                launch.template operator()<__half, true>();
            else
                launch.template operator()<__half, false>();
        }
    });
}
