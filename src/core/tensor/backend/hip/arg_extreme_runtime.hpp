/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "runtime_api.h"

namespace {
    template <bool Maximum>
    __device__ uint64_t indexed_extreme_key(float value, uint32_t position) {
        uint32_t bits = __float_as_uint(value), order;
        if ((bits & 0x7fffffffu) > 0x7f800000u) {
            order = UINT32_MAX; // Every NaN wins over numbers; the first NaN wins ties.
        } else {
            if ((bits & 0x7fffffffu) == 0)
                bits = 0; // Signed zero ties, preserving the selected input's bits.
            order = (bits & 0x80000000u) ? ~bits : bits | 0x80000000u;
            if constexpr (!Maximum)
                order = ~order;
        }
        return (uint64_t(order) << 32) | uint64_t(~position);
    }

    __device__ void finish_indexed_extreme(const float* input, float* values, int64_t* indices,
                                           size_t output, size_t reduced, size_t inner, uint64_t key) {
        const uint32_t position = ~uint32_t(key);
        const size_t offset = (output / inner) * reduced * inner + output % inner;
        values[output] = input[offset + size_t(position) * inner];
        indices[output] = position;
    }

    template <bool Maximum>
    __global__ void indexed_extreme_rows(const float* input, float* values, int64_t* indices, uint64_t* partials,
                                         size_t outputs, size_t reduced, unsigned parts) {
        using Reduction = hipcub::BlockReduce<uint64_t, 256>;
        __shared__ Reduction::TempStorage storage;
        const size_t chunk = (reduced + parts - 1) / parts;
        for (size_t job = blockIdx.x; job < outputs * parts; job += gridDim.x) {
            const size_t row = job / parts, part = job % parts;
            const size_t end = min(reduced, (part + 1) * chunk);
            uint64_t key = 0;
            for (size_t position = part * chunk + threadIdx.x; position < end; position += blockDim.x)
                key = max(key, indexed_extreme_key<Maximum>(input[row * reduced + position], uint32_t(position)));
            key = Reduction(storage).Reduce(key, hipcub::Max{});
            if (threadIdx.x == 0) {
                if (parts == 1)
                    finish_indexed_extreme(input, values, indices, row, reduced, 1, key);
                else
                    partials[part * outputs + row] = key;
            }
            __syncthreads();
        }
    }

    template <bool Maximum>
    __global__ void indexed_extreme_columns(const float* input, float* values, int64_t* indices, uint64_t* partials,
                                            size_t outputs, size_t reduced, size_t inner, unsigned parts) {
        const size_t chunk = (reduced + parts - 1) / parts;
        for (size_t job = size_t(blockIdx.x) * blockDim.x + threadIdx.x; job < outputs * parts; job += size_t(gridDim.x) * blockDim.x) {
            const size_t row = job % outputs, part = job / outputs;
            const size_t base = (row / inner) * reduced * inner + row % inner;
            const size_t end = min(reduced, (part + 1) * chunk);
            uint64_t key = 0;
            for (size_t position = part * chunk; position < end; ++position)
                key = max(key, indexed_extreme_key<Maximum>(input[base + position * inner], uint32_t(position)));
            if (parts == 1)
                finish_indexed_extreme(input, values, indices, row, reduced, inner, key);
            else
                partials[job] = key;
        }
    }

    __global__ void indexed_extreme_fold(const float* input, float* values, int64_t* indices, const uint64_t* partials,
                                         size_t outputs, size_t reduced, size_t inner, unsigned parts) {
        using Reduction = hipcub::BlockReduce<uint64_t, 256>;
        __shared__ Reduction::TempStorage storage;
        for (size_t row = blockIdx.x; row < outputs; row += gridDim.x) {
            uint64_t key = 0;
            for (unsigned part = threadIdx.x; part < parts; part += blockDim.x)
                key = max(key, partials[size_t(part) * outputs + row]);
            key = Reduction(storage).Reduce(key, hipcub::Max{});
            if (threadIdx.x == 0)
                finish_indexed_extreme(input, values, indices, row, reduced, inner, key);
            __syncthreads();
        }
    }

    template <bool Maximum>
    void launch_indexed_extreme(LfsHipContext* context, LfsHipView input, LfsHipView values, LfsHipView indices,
                                size_t outputs, size_t reduced, size_t inner, unsigned parts) {
        const auto* source = reinterpret_cast<const float*>(static_cast<const char*>(input.buffer->data) + input.offset);
        auto* val = reinterpret_cast<float*>(static_cast<char*>(values.buffer->data) + values.offset);
        auto* idx = reinterpret_cast<int64_t*>(static_cast<char*>(indices.buffer->data) + indices.offset);
        auto* partials = static_cast<uint64_t*>(context->sort_workspace);
        if (inner == 1 && reduced > 32) {
            const unsigned groups = unsigned(std::min<size_t>(outputs * parts, 4096));
            hipLaunchKernelGGL((indexed_extreme_rows<Maximum>), dim3(groups), dim3(256), 0, context->stream,
                               source, val, idx, partials, outputs, reduced, parts);
        } else {
            hipLaunchKernelGGL((indexed_extreme_columns<Maximum>), dim3(blocks(outputs * parts)), dim3(256), 0, context->stream,
                               source, val, idx, partials, outputs, reduced, inner, parts);
        }
        check(hipGetLastError());
        if (parts > 1) {
            const unsigned groups = unsigned(std::min<size_t>(outputs, 4096));
            hipLaunchKernelGGL(indexed_extreme_fold, dim3(groups), dim3(256), 0, context->stream,
                               source, val, idx, partials, outputs, reduced, inner, parts);
            check(hipGetLastError());
        }
    }
} // namespace

int lfs_hip_arg_extreme_f32(LfsHipContext* context, LfsHipView input, LfsHipView values, LfsHipView indices,
                            size_t outer, size_t reduced, size_t inner, uint32_t maximum,
                            LfsHipError* error) {
    return boundary(error, [&] {
        require(maximum <= 1 && reduced <= INT32_MAX, "Invalid HIP indexed extrema descriptor");
        require(!inner || outer <= SIZE_MAX / inner, "HIP indexed extrema output shape overflow");
        const size_t outputs = outer * inner;
        require(!outputs || reduced > 0, "HIP indexed extrema cannot reduce an empty axis");
        require(!reduced || outputs <= SIZE_MAX / reduced, "HIP indexed extrema input shape overflow");
        const size_t input_bytes = float_bytes(outputs * reduced), value_bytes = float_bytes(outputs);
        const size_t index_bytes = typed_bytes(LFS_HIP_I64, outputs);
        validate(context, input.buffer, input.offset, input_bytes);
        validate(context, values.buffer, values.offset, value_bytes);
        validate(context, indices.buffer, indices.offset, index_bytes);
        require(input.offset % 4 == 0 && values.offset % 4 == 0 && indices.offset % 8 == 0, "Unaligned HIP indexed extrema views");
        require_disjoint(input, input_bytes, values, value_bytes, "HIP indexed extrema operands overlap");
        require_disjoint(input, input_bytes, indices, index_bytes, "HIP indexed extrema operands overlap");
        require_disjoint(values, value_bytes, indices, index_bytes, "HIP indexed extrema operands overlap");
        if (!outputs)
            return;
        DeviceScope device(context->device);
        const unsigned parts = reduced > 4096 && outputs < 4096
                                   ? unsigned(std::min<size_t>(4096 / outputs, (reduced - 1) / 4096 + 1))
                                   : 1;
        if (parts > 1) {
            const size_t required = outputs * parts * sizeof(uint64_t); // At most 32 KiB.
            ensure_workspace(context, context->sort_workspace, context->sort_workspace_bytes, required);
        }
        if (maximum)
            launch_indexed_extreme<true>(context, input, values, indices, outputs, reduced, inner, parts);
        else
            launch_indexed_extreme<false>(context, input, values, indices, outputs, reduced, inner, parts);
    });
}
