/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "nn_api.h"
#include "nn_linear_runtime.hpp"
namespace hip_nn {
    template <class T>
    __global__ void attention_softmax(float* scores, const T* mask, size_t keys, size_t rows, size_t tile,
                                      size_t group_start, size_t query_start, LfsHipAttention p) {
        __shared__ float reduction[256];
        __shared__ unsigned invalid[256];
        const size_t row = blockIdx.x;
        const unsigned lane = threadIdx.x;
        if (row >= rows)
            return;
        const size_t group = group_start + row / tile, query = query_start + row % tile;
        const size_t mask_base = (group / p.heads) * p.strides[0] + (group % p.heads) * p.strides[1] + query * p.strides[2];
        float maximum = -__FLT_MAX__;
        unsigned bad = 0;
        for (size_t key = lane; key < keys; key += 256) {
            float value = scores[row * keys + key] * p.scale;
            if (mask)
                value += float(mask[mask_base + key * p.strides[3]]);
            scores[row * keys + key] = value;
            bad |= isnan(value) || value == INFINITY;
            maximum = fmaxf(maximum, value);
        }
        reduction[lane] = maximum;
        invalid[lane] = bad;
        __syncthreads();
        for (unsigned step = 128; step; step /= 2) {
            if (lane < step) {
                reduction[lane] = fmaxf(reduction[lane], reduction[lane + step]);
                invalid[lane] |= invalid[lane + step];
            }
            __syncthreads();
        }
        maximum = reduction[0];
        bad = invalid[0];
        float total = 0;
        for (size_t key = lane; key < keys; key += 256) {
            const float value = bad ? NAN : expf(scores[row * keys + key] - maximum);
            scores[row * keys + key] = value;
            total += value;
        }
        reduction[lane] = total;
        __syncthreads();
        for (unsigned step = 128; step; step /= 2) {
            if (lane < step)
                reduction[lane] += reduction[lane + step];
            __syncthreads();
        }
        const float denominator = fmaxf(reduction[0], 1.f);
        for (size_t key = lane; key < keys; key += 256)
            scores[row * keys + key] /= denominator;
    }
} // namespace hip_nn
int lfs_hip_nn_attention(LfsHipContext* context, const LfsHipView* inputs, LfsHipView output, const LfsHipAttention* p, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && inputs && p && p->heads && p->groups % p->heads == 0 && p->dim && p->dim <= INT32_MAX &&
                    (p->dtype == LFS_HIP_F32 || p->dtype == LFS_HIP_F16),
                "Invalid HIP NN attention descriptor");
        if (!p->groups || !p->queries)
            return;
        const size_t width = p->dtype == LFS_HIP_F32 ? 4 : 2;
        const auto count = [](uint64_t a, uint64_t b, uint64_t c) {
            require(a <= INT32_MAX && b <= INT32_MAX && c <= INT32_MAX && (!b || a <= uint64_t(INT32_MAX) / b) && (!c || a * b <= uint64_t(INT32_MAX) / c), "HIP NN attention shape exceeds indexing range");
            return size_t(a * b * c);
        };
        const size_t qbytes = count(p->groups, p->queries, p->dim) * width, kvbytes = count(p->groups, p->keys, p->dim) * width;
        size_t mask_count = 0;
        if (inputs[3].buffer && p->keys) {
            const uint64_t dimensions[]{p->groups / p->heads, p->heads, p->queries, p->keys};
            for (unsigned i = 0; i < 4; ++i) {
                require(p->strides[i] >= 0, "HIP attention mask strides must be nonnegative");
                const uint64_t stride = uint64_t(p->strides[i]), extent = dimensions[i] - 1;
                require(!stride || extent <= (SIZE_MAX / width - 1 - mask_count) / stride, "HIP attention mask span overflow");
                mask_count += extent * stride;
            }
            ++mask_count;
        }
        const size_t sizes[]{qbytes, kvbytes, kvbytes, mask_count * width};
        validate(context, output.buffer, output.offset, qbytes);
        require(output.offset % width == 0, "Misaligned HIP attention output");
        for (unsigned i = 0; i < 4; ++i) {
            if (!sizes[i])
                continue;
            validate(context, inputs[i].buffer, inputs[i].offset, sizes[i]);
            require(inputs[i].offset % width == 0, "Misaligned HIP attention input");
            require(inputs[i].buffer != output.buffer || inputs[i].offset >= output.offset + qbytes || output.offset >= inputs[i].offset + sizes[i], "HIP attention output aliases input");
        }
        DeviceScope device(context->device);
        if (!p->keys) {
            check(hipMemsetAsync(static_cast<char*>(output.buffer->data) + output.offset, 0, qbytes, context->stream));
            return;
        }
        constexpr size_t budget = 8 * 1024 * 1024;
        const size_t group_chunk = std::max<size_t>(1, std::min<size_t>({p->groups, 65535, budget / (p->keys * 4)}));
        const size_t query_chunk = std::max<size_t>(1, std::min<size_t>({p->queries, 64, budget / (group_chunk * p->keys * 4)}));
        const size_t required = count(group_chunk, query_chunk, p->keys) * 4;
        ensure_workspace(context, context->sort_workspace, context->sort_workspace_bytes, required);
        auto* scores = static_cast<float*>(context->sort_workspace);
        const auto ptr = [](LfsHipView view) -> void* { return view.buffer ? static_cast<char*>(view.buffer->data) + view.offset : nullptr; };
        const auto launch = [&]<class T>() {
            for (size_t g = 0; g < p->groups; g += group_chunk)
                for (size_t start = 0; start < p->queries; start += query_chunk) {
                    const size_t groups = std::min<size_t>(group_chunk, p->groups - g), rows = std::min<size_t>(query_chunk, p->queries - start);
                    const auto* q = static_cast<const T*>(ptr(inputs[0])) + (g * p->queries + start) * p->dim;
                    const auto* k = static_cast<const T*>(ptr(inputs[1])) + g * p->keys * p->dim;
                    const auto* v = static_cast<const T*>(ptr(inputs[2])) + g * p->keys * p->dim;
                    auto* out = static_cast<T*>(ptr(output)) + (g * p->queries + start) * p->dim;
                    hip_nn::linear_tiled<T, true, T, float><<<dim3((p->keys + 63) / 64, (rows + 63) / 64, groups), dim3(16, 16), 0, context->stream>>>(
                        q, k, nullptr, nullptr, nullptr, scores, rows, p->keys, p->dim, p->keys * p->dim, 0, p->keys, false, p->queries * p->dim, rows * p->keys);
                    check(hipGetLastError());
                    hip_nn::attention_softmax<T><<<groups * rows, 256, 0, context->stream>>>(scores, static_cast<const T*>(ptr(inputs[3])), p->keys, groups * rows, rows, g, start, *p);
                    check(hipGetLastError());
                    hip_nn::linear_tiled<float, false, T, T><<<dim3((p->dim + 63) / 64, (rows + 63) / 64, groups), dim3(16, 16), 0, context->stream>>>(
                        scores, v, nullptr, nullptr, nullptr, out, rows, p->dim, p->keys, p->keys * p->dim, 0, p->dim, false, rows * p->keys, p->queries * p->dim);
                    check(hipGetLastError());
                }
        };
        if (p->dtype == LFS_HIP_F32)
            launch.template operator()<float>();
        else
            launch.template operator()<__half>();
    });
}
