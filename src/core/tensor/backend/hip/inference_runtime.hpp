/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "hip_inference.hpp"
#include "inference_api.h"
int lfs_hip_inference_typed(LfsHipContext* context, LfsHipView input, LfsHipView output, const LfsHipInference* call, uint32_t dtype, LfsHipError* error) {
    return boundary(error, [&] {
        require((dtype == LFS_HIP_F32 || dtype == LFS_HIP_F16) && context && call && call->struct_bytes == sizeof(*call) && call->operation <= 5 && call->operation != 3, "Invalid HIP inference descriptor");
        auto p = call->parameters;
        require(p.total <= INT_MAX, "HIP inference output exceeds int32 indexing");
        if (!p.total)
            return;
        size_t source_count = 0;
        const auto op = call->operation;
        auto product = [](std::initializer_list<int32_t> dimensions) {
            size_t result = 1;
            for (int32_t d : dimensions) {
                require(d > 0 && result <= size_t(INT_MAX) / size_t(d), "HIP inference geometry exceeds int32 indexing");
                result *= size_t(d);
            }
            return result;
        };
        if (op == 0 || op == 1) {
            require(p.stride_h > 0 && p.stride_w > 0 && p.dilation_h > 0 && p.dilation_w > 0 && p.pad_h >= 0 && p.pad_w >= 0, "Invalid HIP convolution geometry");
            const auto kernel_area = product({p.kernel_h, p.kernel_w});
            const auto source_pixels = product({p.height, p.width});
            const auto output_pixels = product({p.out_height, p.out_width});
            // The shader computes coordinates in int32, including padded tails.
            require(int64_t(p.out_height) * p.stride_h + int64_t(p.kernel_h) * p.dilation_h + p.pad_h <= INT_MAX &&
                        int64_t(p.out_width) * p.stride_w + int64_t(p.kernel_w) * p.dilation_w + p.pad_w <= INT_MAX,
                    "HIP convolution coordinates exceed int32 indexing");
            if (op == 0) {
                const auto taps = product({p.channels, p.kernel_h, p.kernel_w});
                source_count = product({p.channels, p.height, p.width});
                require(p.columns > 0 && p.offset >= 0 && size_t(p.offset) <= output_pixels && size_t(p.columns) <= output_pixels - size_t(p.offset) &&
                            taps <= size_t(INT_MAX) / size_t(p.columns) && taps * size_t(p.columns) == p.total && (p.mode == 0 || p.mode == 1),
                        "Invalid HIP im2col tile");
            } else {
                // Transposed convolutions can have different input and output
                // channel counts. The fold's planes follow its output, while
                // geometry.channels still describes the original input group.
                require(p.total % output_pixels == 0, "Invalid HIP col2im output planes");
                const size_t planes = p.total / output_pixels;
                require(planes <= size_t(INT_MAX) / kernel_area && planes * kernel_area <= size_t(INT_MAX) / source_pixels, "Invalid HIP col2im geometry");
                source_count = planes * kernel_area * source_pixels;
            }
        } else if (op == 2) {
            const auto source_pixels = product({p.height, p.width}), output_pixels = product({p.out_height, p.out_width});
            require(p.total % output_pixels == 0 && p.mode >= 0 && p.mode <= 2 && p.coord >= 0 && p.coord <= 2, "Invalid HIP resize geometry");
            require(source_pixels <= size_t(INT_MAX) / (p.total / output_pixels), "HIP resize input exceeds int32 indexing");
            source_count = source_pixels * (p.total / output_pixels);
        } else if (op == 4) {
            require(p.mode >= 0 && p.mode <= 4, "Invalid HIP activation");
            source_count = p.total;
        } else {
            const auto pixels = product({p.height, p.width});
            require(pixels <= size_t(INT_MAX) / 2 && p.total == pixels * 2 && std::isfinite(p.u0) && std::isfinite(p.u1) &&
                        std::isfinite(p.v0) && std::isfinite(p.v1),
                    "Invalid HIP normalized grid geometry");
        }
        const size_t width = dtype == LFS_HIP_F32 ? 4 : 2, source_bytes = source_count * width, output_bytes = size_t(p.total) * width;
        if (source_bytes)
            validate(context, input.buffer, input.offset, source_bytes);
        validate(context, output.buffer, output.offset, output_bytes);
        require(output.offset % width == 0 && (!source_bytes || input.offset % width == 0), "Misaligned HIP inference storage");
        const bool overlap = source_bytes && input.buffer == output.buffer && input.offset < output.offset + output_bytes && output.offset < input.offset + source_bytes;
        require(!overlap || (op == 4 && input.offset == output.offset), "HIP inference input/output overlap");
        DeviceScope device(context->device);
        p.input = source_bytes ? reinterpret_cast<uintptr_t>(static_cast<char*>(input.buffer->data) + input.offset) : 0;
        p.output = reinterpret_cast<uintptr_t>(static_cast<char*>(output.buffer->data) + output.offset);
        const unsigned groups = std::min<unsigned>((p.total + 255) / 256, 65535);
        p.step = groups * 256;
        hip_inference::Parameters_0 native{};
        static_assert(sizeof(native) == sizeof(p));
        static_assert(offsetof(hip_inference::Parameters_0, u0_0) == offsetof(LfsHipInferencePush, u0));
        std::memcpy(&native, &p, sizeof(p));
        const auto launch = [&]<class Storage>() {
#define LFS_INFERENCE_CASE(N) \
    case N: hipLaunchKernelGGL((hip_inference::inference_kernel<N, Storage>), dim3(groups), dim3(256), 0, context->stream, native); break
            switch (op) {
                LFS_INFERENCE_CASE(0);
                LFS_INFERENCE_CASE(1);
                LFS_INFERENCE_CASE(2);
                LFS_INFERENCE_CASE(4);
                LFS_INFERENCE_CASE(5);
            }
#undef LFS_INFERENCE_CASE
        };
        if (dtype == LFS_HIP_F32)
            launch.template operator()<float>();
        else
            launch.template operator()<__half>();
        check(hipGetLastError());
    });
}

int lfs_hip_inference_f32(LfsHipContext* context, LfsHipView input, LfsHipView output, const LfsHipInference* call, LfsHipError* error) {
    return lfs_hip_inference_typed(context, input, output, call, LFS_HIP_F32, error);
}
