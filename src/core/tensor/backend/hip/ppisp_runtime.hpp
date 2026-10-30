/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "point_filter_kernels.hpp"
#include "point_tree_api.h"
namespace {
    __global__ void ppisp_native(const float* input, float* output, int width, int height, hip_ppisp::PpispParams params) {
        for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < int64_t(width) * height; i += int64_t(gridDim.x) * blockDim.x)
            hip_ppisp::ppisp_pixel(input, output, width, height, params, int(i));
    }
} // namespace
int lfs_hip_ppisp(LfsHipContext* context, LfsHipView input, LfsHipView output,
                  int32_t width, int32_t height, const LfsHipPpisp* params, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && params && width > 0 && height > 0 && int64_t(width) * height <= INT32_MAX / 3, "Invalid HIP PPISP shape");
        const int full_width = params->full_width > 0 ? params->full_width : width;
        const int full_height = params->full_height > 0 ? params->full_height : height;
        require(params->x_offset >= 0 && params->y_offset >= 0 && width <= full_width && height <= full_height &&
                    params->x_offset <= full_width - width && params->y_offset <= full_height - height,
                "Invalid HIP PPISP region");
        const size_t bytes = size_t(width) * height * 3 * sizeof(float);
        validate(context, input.buffer, input.offset, bytes);
        validate(context, output.buffer, output.offset, bytes);
        require(input.offset % 4 == 0 && output.offset % 4 == 0, "Misaligned HIP PPISP storage");
        require(input.buffer != output.buffer || input.offset == output.offset || input.offset >= output.offset + bytes ||
                    output.offset >= input.offset + bytes,
                "Overlapping HIP PPISP ranges");
        hip_ppisp::PpispParams native{};
        static_assert(sizeof(native) == sizeof(*params));
        std::memcpy(&native, params, sizeof(native));
        DeviceScope device(context->device);
        const auto* src = reinterpret_cast<const float*>(static_cast<const char*>(input.buffer->data) + input.offset);
        auto* dst = reinterpret_cast<float*>(static_cast<char*>(output.buffer->data) + output.offset);
        const size_t blocks = std::min<size_t>((size_t(width) * height + 255) / 256, 65535);
        ppisp_native<<<blocks, 256, 0, context->stream>>>(src, dst, width, height, native);
        check(hipGetLastError());
    });
}
