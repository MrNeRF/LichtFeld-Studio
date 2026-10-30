/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "nn_api.h"
#include "nn_linear_runtime.hpp"
namespace hip_nn {
    // Transposed weights arrive as [out][ky][kx][in]. Gathering their input
    // taps directly avoids a full column image and a second accumulation pass.
    template <class T, bool Transpose>
    __global__ void conv_columns(const T* input, T* columns, size_t taps, size_t count, size_t offset, LfsHipInferencePush p, uint8_t* validity) {
        const size_t total = taps * count;
        for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < total; i += size_t(gridDim.x) * 256) {
            const size_t column = i % count + offset, tap = i / count;
            int64_t x, y, c;
            if constexpr (Transpose) {
                c = tap % p.channels;
                const size_t k = tap / p.channels;
                x = int64_t(column % p.out_width) + p.pad_w - int64_t(k % p.kernel_w) * p.dilation_w;
                y = int64_t(column / p.out_width) + p.pad_h - int64_t(k / p.kernel_w) * p.dilation_h;
                if (x < 0 || y < 0 || x % p.stride_w || y % p.stride_h) {
                    columns[i] = T(0);
                    validity[i] = 0;
                    continue;
                }
                x /= p.stride_w;
                y /= p.stride_h;
            } else {
                c = tap / (size_t(p.kernel_h) * p.kernel_w);
                x = int64_t(column % p.out_width) * p.stride_w - p.pad_w + int64_t(tap % p.kernel_w) * p.dilation_w;
                y = int64_t(column / p.out_width) * p.stride_h - p.pad_h + int64_t((tap / p.kernel_w) % p.kernel_h) * p.dilation_h;
                if (p.mode == 1) {
                    x = x < 0 ? 0 : x >= p.width ? p.width - 1
                                                 : x;
                    y = y < 0 ? 0 : y >= p.height ? p.height - 1
                                                  : y;
                }
            }
            const bool valid = x >= 0 && x < p.width && y >= 0 && y < p.height;
            if constexpr (Transpose)
                validity[i] = valid;
            columns[i] = valid ? input[(c * p.height + y) * p.width + x] : T(0);
        }
    }
} // namespace hip_nn
int lfs_hip_nn_conv(LfsHipContext* context, const LfsHipView* inputs, LfsHipView output, const LfsHipConv* p, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && inputs && p && p->groups && p->out_channels && p->out_channels % p->groups == 0 && p->transpose <= 1 && p->activation <= 4 &&
                    (p->dtype == LFS_HIP_F32 || p->dtype == LFS_HIP_F16),
                "Invalid HIP NN convolution descriptor");
        const auto& g = p->geometry;
        require(g.channels > 0 && g.height > 0 && g.width > 0 && g.out_height > 0 && g.out_width > 0 && g.kernel_h > 0 && g.kernel_w > 0 &&
                    g.stride_h > 0 && g.stride_w > 0 && g.dilation_h > 0 && g.dilation_w > 0 && g.pad_h >= 0 && g.pad_w >= 0 &&
                    g.mode >= 0 && g.mode <= 1 && (!p->transpose || g.mode == 0),
                "Invalid HIP NN convolution geometry");
        const auto count = [](std::initializer_list<uint64_t> dims) {size_t n=1;for(auto d:dims){require(d<=INT32_MAX && (!d || n<=size_t(INT32_MAX)/d),"HIP NN convolution shape exceeds indexing range");n*=size_t(d);}return n; };
        const size_t input_count = count({p->batch, p->groups, uint64_t(g.channels), uint64_t(g.height), uint64_t(g.width)});
        const size_t output_count = count({p->batch, p->out_channels, uint64_t(g.out_height), uint64_t(g.out_width)});
        const size_t weight_count = count({p->out_channels, uint64_t(g.channels), uint64_t(g.kernel_h), uint64_t(g.kernel_w)});
        const size_t taps = count({uint64_t(g.channels), uint64_t(g.kernel_h), uint64_t(g.kernel_w)}), pixels = count({uint64_t(g.out_height), uint64_t(g.out_width)});
        const size_t channels = p->out_channels / p->groups, width = p->dtype == LFS_HIP_F32 ? 4 : 2;
        require(channels <= 65535ull * 64, "HIP convolution channel grid exceeds range");
        if (!p->batch)
            return;
        const size_t sizes[]{input_count * width, weight_count * width, p->out_channels * width};
        validate(context, output.buffer, output.offset, output_count * width);
        require(output.offset % width == 0, "Misaligned HIP convolution output");
        for (unsigned i = 0; i < 3; ++i) {
            if (i == 2 && !inputs[i].buffer)
                continue;
            validate(context, inputs[i].buffer, inputs[i].offset, sizes[i]);
            require(inputs[i].offset % width == 0, "Misaligned HIP convolution input");
            require(inputs[i].buffer != output.buffer || inputs[i].offset >= output.offset + output_count * width || output.offset >= inputs[i].offset + sizes[i], "HIP convolution output aliases input");
        }
        DeviceScope device(context->device);
        const size_t chunk = std::max<size_t>(1, std::min<size_t>({pixels, 1024, (16 * 1024 * 1024) / (taps * (width + p->transpose))})), required = taps * chunk * (width + p->transpose);
        ensure_workspace(context, context->sort_workspace, context->sort_workspace_bytes, required);
        const auto ptr = [](LfsHipView v) -> void* { return v.buffer ? static_cast<char*>(v.buffer->data) + v.offset : nullptr; };
        const auto launch = [&]<class T, bool Transpose>() {
            auto* columns = static_cast<T*>(context->sort_workspace);
            auto* validity = Transpose ? reinterpret_cast<uint8_t*>(columns + taps * chunk) : nullptr;
            for (size_t batch = 0; batch < p->batch; ++batch)
                for (size_t group = 0; group < p->groups; ++group)
                    for (size_t first = 0; first < pixels; first += chunk) {
                        const size_t used = std::min(chunk, pixels - first);
                        const auto* input = static_cast<const T*>(ptr(inputs[0])) + (batch * p->groups + group) * g.channels * g.height * g.width;
                        const auto* weights = static_cast<const T*>(ptr(inputs[1])) + group * channels * taps;
                        const auto* bias = inputs[2].buffer ? static_cast<const T*>(ptr(inputs[2])) + group * channels : nullptr;
                        auto* out = static_cast<T*>(ptr(output)) + (batch * p->out_channels + group * channels) * pixels + first;
                        hip_nn::conv_columns<T, Transpose><<<blocks(taps * used), 256, 0, context->stream>>>(input, columns, taps, used, first, g, validity);
                        check(hipGetLastError());
                        if (channels <= 4)
                            hip_nn::linear_skinny<T, false, T, T, Transpose><<<dim3((used + 255) / 256, channels, 1), 256, 0, context->stream>>>(weights, columns, bias, nullptr, nullptr, out, channels, used, taps, 0, p->activation, pixels, true, channels * taps, channels * pixels, validity);
                        else
                            hip_nn::linear_tiled<T, false, T, T, Transpose><<<dim3((used + 63) / 64, (channels + 63) / 64, 1), dim3(16, 16), 0, context->stream>>>(weights, columns, bias, nullptr, nullptr, out, channels, used, taps, 0, p->activation, pixels, true, channels * taps, channels * pixels, validity);
                        check(hipGetLastError());
                    }
        };
        if (p->dtype == LFS_HIP_F32) {
            if (p->transpose)
                launch.template operator()<float, true>();
            else
                launch.template operator()<float, false>();
        } else {
            if (p->transpose)
                launch.template operator()<__half, true>();
            else
                launch.template operator()<__half, false>();
        }
    });
}
