/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "hip_shared_resize.hpp"
int lfs_hip_shared_resize(LfsHipContext* context, const LfsHipSharedResize* p, LfsHipView source,
                          LfsHipView destination, LfsHipView coefficients_x, LfsHipView coefficients_y, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && p && p->struct_bytes == sizeof(*p) && p->mode <= 4 && p->channels >= 1 && p->channels <= 4 &&
                    (p->dtype == LFS_HIP_U8 || p->dtype == LFS_HIP_F32),
                "Invalid HIP resize descriptor");
        require(p->source_h > 0 && p->source_w > 0 && p->destination_h > 0 && p->destination_w > 0 &&
                    uint64_t(p->source_h) * p->source_w <= INT32_MAX / p->channels &&
                    uint64_t(p->destination_h) * p->destination_w <= INT32_MAX / p->channels,
                "HIP resize extent exceeds indexing");
        require((p->mode != 1 || p->channels == 1) && (p->mode != 2 || (p->channels == 3 && p->dtype == LFS_HIP_F32)) &&
                    (p->mode != 3 || (p->channels == 1 && p->dtype == LFS_HIP_F32)) &&
                    (p->mode != 4 || (p->channels == 3 && p->dtype == LFS_HIP_F32)),
                "HIP resize layout/dtype mismatch");
        const size_t input_bytes = size_t(p->source_h) * p->source_w * p->channels * (p->dtype == LFS_HIP_U8 ? 1 : 4);
        const size_t output_bytes = size_t(p->destination_h) * p->destination_w * p->channels * 4;
        validate(context, source.buffer, source.offset, input_bytes);
        validate(context, destination.buffer, destination.offset, output_bytes);
        require(destination.offset % 4 == 0 && (p->dtype == LFS_HIP_U8 || source.offset % 4 == 0), "Misaligned HIP resize pixels");
        size_t coefficient_bytes_x = 0, coefficient_bytes_y = 0;
        if (p->mode < 3) {
            require(p->kernel_size > 0 && p->kernel_size <= INT32_MAX, "Invalid HIP Lanczos kernel size");
            const uint64_t needed_x = (2ull * p->kernel_size * p->source_w + p->destination_w - 1) / p->destination_w + 1;
            const uint64_t needed_y = (2ull * p->kernel_size * p->source_h + p->destination_h - 1) / p->destination_h + 1;
            require(p->stride_x >= needed_x && p->stride_y >= needed_y &&
                        uint64_t(p->stride_x) * p->destination_w <= UINT32_MAX / 4 &&
                        uint64_t(p->stride_y) * p->destination_h <= UINT32_MAX / 4,
                    "HIP Lanczos coefficient extent exceeds indexing");
            coefficient_bytes_x = size_t(p->stride_x) * p->destination_w * 4;
            coefficient_bytes_y = size_t(p->stride_y) * p->destination_h * 4;
            validate(context, coefficients_x.buffer, coefficients_x.offset, coefficient_bytes_x);
            validate(context, coefficients_y.buffer, coefficients_y.offset, coefficient_bytes_y);
            require(coefficients_x.offset % 4 == 0 && coefficients_y.offset % 4 == 0, "Misaligned HIP Lanczos coefficients");
        }
        const LfsHipView views[]{source, destination, coefficients_x, coefficients_y};
        const size_t sizes[]{input_bytes, output_bytes, coefficient_bytes_x, coefficient_bytes_y};
        for (unsigned i = 0; i < 4; ++i)
            for (unsigned j = i + 1; j < 4; ++j)
                require(!sizes[i] || !sizes[j] || views[i].buffer != views[j].buffer ||
                            views[i].offset >= views[j].offset + sizes[j] || views[j].offset >= views[i].offset + sizes[i],
                        "HIP resize workspaces overlap pixels/each other");
        DeviceScope device(context->device);
        auto stream = context->stream;
        auto address = [](LfsHipView v) { return static_cast<char*>(v.buffer->data) + v.offset; };
        const auto* in8 = reinterpret_cast<const uint8_t*>(address(source));
        const auto* in32 = reinterpret_cast<const float*>(address(source));
        auto* out = reinterpret_cast<float*>(address(destination));
        const dim3 grid((uint32_t(p->destination_w) + 15) / 16, (uint32_t(p->destination_h) + 15) / 16), block(16, 16);
        require(grid.y <= 65535, "HIP resize grid height exceeds device limit");
        using namespace hip_shared_resize;
        if (p->mode >= 3) {
            if (p->mode == 3)
                hipLaunchKernelGGL((resize_prior_kernel<1>), grid, block, 0, stream, in32, out, p->source_w, p->source_h, p->destination_w, p->destination_h);
            else
                hipLaunchKernelGGL((resize_prior_kernel<3>), grid, block, 0, stream, in32, out, p->source_w, p->source_h, p->destination_w, p->destination_h);
        } else {
            auto* cx = reinterpret_cast<float*>(address(coefficients_x));
            auto* cy = reinterpret_cast<float*>(address(coefficients_y));
            hipLaunchKernelGGL(PreComputeCoef, dim3((uint32_t(p->destination_w) + 255) / 256), dim3(256), 0, stream, p->source_w, p->destination_w, int(p->kernel_size), p->stride_x, cx);
            hipLaunchKernelGGL(PreComputeCoef, dim3((uint32_t(p->destination_h) + 255) / 256), dim3(256), 0, stream, p->source_h, p->destination_h, int(p->kernel_size), p->stride_y, cy);
#define RESIZE_ARGS p->source_h, p->source_w, p->destination_h, p->destination_w, int(p->kernel_size), p->stride_x, p->stride_y, cx, cy
            if (p->mode == 0) {
                auto launch = [&]<unsigned C>() {
                    if (p->dtype == LFS_HIP_U8)
                        hipLaunchKernelGGL((LanczosResampleCUDA<C, uint8_t>), grid, block, 0, stream, RESIZE_ARGS, in8, out);
                    else
                        hipLaunchKernelGGL((LanczosResampleCUDA<C, float>), grid, block, 0, stream, RESIZE_ARGS, in32, out);
                };
                switch (p->channels) {
                case 1: launch.template operator()<1>(); break;
                case 2: launch.template operator()<2>(); break;
                case 3: launch.template operator()<3>(); break;
                case 4: launch.template operator()<4>(); break;
                }
            } else if (p->mode == 1) {
                if (p->dtype == LFS_HIP_U8)
                    hipLaunchKernelGGL((LanczosResampleGrayscaleCUDA<uint8_t>), grid, block, 0, stream, RESIZE_ARGS, in8, 1.f / 255.f, out);
                else
                    hipLaunchKernelGGL((LanczosResampleGrayscaleCUDA<float>), grid, block, 0, stream, RESIZE_ARGS, in32, 1.f, out);
            } else
                hipLaunchKernelGGL((LanczosResamplePlanarCUDA<3>), grid, block, 0, stream, RESIZE_ARGS, in32, out);
#undef RESIZE_ARGS
        }
        check(hipGetLastError());
    });
}
