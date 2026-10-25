/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "bias_api.h"
namespace {
    template <bool Relu>
    __global__ void tensor_bias(const float* input, const float* bias, float* output, size_t count, unsigned channels, unsigned spatial) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
            const float value = input[i] + bias[(i / spatial) % channels];
            output[i] = Relu ? fmaxf(value, 0.f) : value;
        }
    }
} // namespace
extern "C" int lfs_hip_bias_f32(LfsHipContext* context, LfsHipView input, LfsHipView bias, LfsHipView output, uint32_t count, uint32_t channels, uint32_t spatial, uint32_t relu, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && channels && spatial && relu <= 1, "Invalid HIP bias descriptor");
        const auto bytes = float_bytes(count), bias_bytes = float_bytes(channels);
        validate(context, input.buffer, input.offset, bytes);
        validate(context, bias.buffer, bias.offset, bias_bytes);
        validate(context, output.buffer, output.offset, bytes);
        require(input.offset % 4 == 0 && bias.offset % 4 == 0 && output.offset % 4 == 0, "Unaligned HIP bias binding");
        auto overlap = [&](LfsHipView v, size_t n) { return v.buffer == output.buffer && v.offset < output.offset + bytes && output.offset < v.offset + n; };
        require(!overlap(input, bytes) || input.offset == output.offset, "HIP bias output partially overlaps input");
        require(!overlap(bias, bias_bytes) || (bias.offset == output.offset && channels == count && spatial == 1), "HIP bias output overlaps a broadcast bias");
        if (!count)
            return;
        DeviceScope selected(context->device);
        const auto* a = reinterpret_cast<const float*>(static_cast<char*>(input.buffer->data) + input.offset);
        const auto* b = reinterpret_cast<const float*>(static_cast<char*>(bias.buffer->data) + bias.offset);
        auto* out = reinterpret_cast<float*>(static_cast<char*>(output.buffer->data) + output.offset);
        if (relu) {
            hipLaunchKernelGGL((tensor_bias<true>), dim3(blocks(count)), dim3(256), 0, context->stream, a, b, out, count, channels, spatial);
        } else {
            hipLaunchKernelGGL((tensor_bias<false>), dim3(blocks(count)), dim3(256), 0, context->stream, a, b, out, count, channels, spatial);
        }
        check(hipGetLastError());
    });
}
