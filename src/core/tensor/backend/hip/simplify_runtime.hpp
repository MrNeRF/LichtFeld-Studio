/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "point_filter_kernels.hpp"
namespace {
    __global__ void simplify_validate(const int32_t* offsets, const int32_t* members, size_t groups, size_t capacity, size_t rows, uint32_t* invalid) {
        for (size_t g = size_t(blockIdx.x) * 256 + threadIdx.x; g < groups; g += size_t(gridDim.x) * 256) {
            const int32_t first = offsets[g], last = offsets[g + 1];
            if (first < 0 || last <= first || size_t(last) > capacity) {
                atomicExch(invalid, 1u);
                continue;
            }
            int32_t previous = -1;
            for (int32_t i = first; i < last; ++i) {
                const int32_t row = members[i];
                if (row < 0 || size_t(row) >= rows || row < previous)
                    atomicExch(invalid, 1u);
                previous = row;
            }
        }
    }
} // namespace
int lfs_hip_simplify_merge(LfsHipContext* context, const LfsHipView* inputs, const LfsHipView* outputs, uint32_t groups, uint32_t appearance, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && inputs && outputs && groups < INT32_MAX && appearance <= INT32_MAX, "Invalid HIP simplify descriptor");
        if (!groups)
            return;
        const size_t widths[]{3, 3, 4, 1, appearance};
        size_t rows = SIZE_MAX;
        size_t input_sizes[7]{}, output_sizes[5]{};
        for (unsigned i = 0; i < 5; ++i) {
            if (!widths[i])
                continue;
            validate(context, inputs[i].buffer, inputs[i].offset, 0);
            require(inputs[i].offset % 4 == 0, "Misaligned HIP simplify input");
            rows = std::min(rows, (inputs[i].buffer->bytes - inputs[i].offset) / (widths[i] * 4));
            require(size_t(groups) <= SIZE_MAX / (widths[i] * 4), "HIP simplify output range overflow");
            output_sizes[i] = size_t(groups) * widths[i] * 4;
            validate(context, outputs[i].buffer, outputs[i].offset, output_sizes[i]);
            require(outputs[i].offset % 4 == 0, "Misaligned HIP simplify output");
        }
        require(rows > 0, "HIP simplify needs source rows");
        for (unsigned i = 0; i < 5; ++i)
            input_sizes[i] = rows * widths[i] * 4;
        input_sizes[5] = (size_t(groups) + 1) * 4;
        validate(context, inputs[5].buffer, inputs[5].offset, input_sizes[5]);
        validate(context, inputs[6].buffer, inputs[6].offset, 0);
        require(inputs[5].offset % 4 == 0 && inputs[6].offset % 4 == 0, "Misaligned HIP simplify grouping");
        const size_t members = (inputs[6].buffer->bytes - inputs[6].offset) / 4;
        input_sizes[6] = members * 4;
        const auto disjoint = [](LfsHipView a, size_t ab, LfsHipView b, size_t bb) {
            require(!ab || !bb || a.buffer != b.buffer || a.offset >= b.offset + bb || b.offset >= a.offset + ab, "HIP simplify output aliases another range");
        };
        for (unsigned i = 0; i < 5; ++i) {
            for (unsigned j = 0; j < 7; ++j)
                disjoint(outputs[i], output_sizes[i], inputs[j], input_sizes[j]);
            for (unsigned j = 0; j < i; ++j)
                disjoint(outputs[i], output_sizes[i], outputs[j], output_sizes[j]);
        }
        DeviceScope device(context->device);
        if (!context->reduction_workspace)
            check(hipMalloc(&context->reduction_workspace, reduction_workspace_floats * sizeof(float)));
        auto* invalid = reinterpret_cast<uint32_t*>(context->reduction_workspace);
        check(hipMemsetAsync(invalid, 0, 4, context->stream));
        const auto ptr = [](LfsHipView v) -> void* { return v.buffer ? static_cast<char*>(v.buffer->data) + v.offset : nullptr; };
        simplify_validate<<<blocks(groups), 256, 0, context->stream>>>(static_cast<const int32_t*>(ptr(inputs[5])), static_cast<const int32_t*>(ptr(inputs[6])), groups, members, rows, invalid);
        check(hipGetLastError());
        uint32_t bad = 0;
        read_device_scalar(context, &bad, invalid, sizeof(bad));
        require(!bad, "Invalid HIP simplify offsets or sorted members");
        hip_simplify::simplify_merge_kernel<<<(groups + 127) / 128, 128, 0, context->stream>>>(
            static_cast<const float*>(ptr(inputs[0])), static_cast<const float*>(ptr(inputs[1])), static_cast<const float*>(ptr(inputs[2])),
            static_cast<const float*>(ptr(inputs[3])), static_cast<const float*>(ptr(inputs[4])), static_cast<const int32_t*>(ptr(inputs[5])), static_cast<const int32_t*>(ptr(inputs[6])),
            static_cast<float*>(ptr(outputs[0])), static_cast<float*>(ptr(outputs[1])), static_cast<float*>(ptr(outputs[2])), static_cast<float*>(ptr(outputs[3])), static_cast<float*>(ptr(outputs[4])), groups, appearance);
        check(hipGetLastError());
    });
}
