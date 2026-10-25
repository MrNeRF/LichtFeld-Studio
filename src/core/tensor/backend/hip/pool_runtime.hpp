/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
namespace {
    template <class Storage>
    __global__ void inference_pool(const Storage* input, Storage* output, LfsHipPool p) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < p.count; i += size_t(gridDim.x) * blockDim.x) {
            const int64_t x = int64_t(i % p.out_width) * p.stride_w - p.pad_w;
            const int64_t y = int64_t((i / p.out_width) % p.out_height) * p.stride_h - p.pad_h;
            const size_t plane = i / (size_t(p.out_width) * p.out_height);
            if (p.average == 2) {
                const int64_t ox = int64_t(i % p.out_width), oy = int64_t((i / p.out_width) % p.out_height);
                const int64_t x0 = ox * p.width / p.out_width, x1 = ((ox + 1) * p.width + p.out_width - 1) / p.out_width;
                const int64_t y0 = oy * p.height / p.out_height, y1 = ((oy + 1) * p.height + p.out_height - 1) / p.out_height;
                float sum = 0;
                for (int64_t iy = y0; iy < y1; ++iy)
                    for (int64_t ix = x0; ix < x1; ++ix)
                        sum += float(input[(plane * p.height + size_t(iy)) * p.width + size_t(ix)]);
                output[i] = sum / float((x1 - x0) * (y1 - y0));
                continue;
            }
            float value = p.average ? 0.f : -INFINITY;
            int count = 0;
            for (int ky = 0; ky < p.kernel_h; ++ky)
                for (int kx = 0; kx < p.kernel_w; ++kx)
                    if (y + ky >= 0 && y + ky < p.height && x + kx >= 0 && x + kx < p.width) {
                        const float v = input[(plane * p.height + size_t(y + ky)) * p.width + size_t(x + kx)];
                        value = p.average ? value + v : fmaxf(value, v);
                        ++count;
                    }
            if (p.average)
                value /= float(max(1, p.include_pad ? p.kernel_h * p.kernel_w : count));
            output[i] = value;
        }
    }
} // namespace
extern "C" int lfs_hip_pool_typed(LfsHipContext* context, LfsHipView input, LfsHipView output, const LfsHipPool* p, uint32_t dtype, LfsHipError* error) {
    return boundary(error, [&] {
        require((dtype == LFS_HIP_F32 || dtype == LFS_HIP_F16) && context && p && p->height > 0 && p->width > 0 && p->out_height > 0 && p->out_width > 0 &&
                    p->kernel_h > 0 && p->kernel_w > 0 && p->kernel_h <= INT32_MAX / p->kernel_w &&
                    p->stride_h > 0 && p->stride_w > 0 && p->pad_h >= 0 && p->pad_w >= 0 &&
                    p->average <= 2 && p->include_pad <= 1,
                "Invalid HIP pooling geometry");
        const size_t out_plane = size_t(p->out_height) * p->out_width;
        require(p->count % out_plane == 0, "HIP pooling output does not contain whole planes");
        require(p->average == 2 || ((int64_t(p->height) + 2ll * p->pad_h - p->kernel_h) / p->stride_h + 1 == p->out_height &&
                                    (int64_t(p->width) + 2ll * p->pad_w - p->kernel_w) / p->stride_w + 1 == p->out_width),
                "HIP pooling output geometry mismatch");
        if (!p->count)
            return;
        const size_t planes = p->count / out_plane, in_plane = size_t(p->height) * p->width, width = dtype == LFS_HIP_F32 ? 4 : 2;
        require(planes <= SIZE_MAX / width / in_plane && p->count <= SIZE_MAX / width, "HIP pooling extent overflow");
        const size_t in_bytes = planes * in_plane * width, out_bytes = p->count * width;
        validate(context, input.buffer, input.offset, in_bytes);
        validate(context, output.buffer, output.offset, out_bytes);
        require(input.offset % width == 0 && output.offset % width == 0, "Misaligned HIP pooling view");
        require(input.buffer != output.buffer || input.offset >= output.offset + out_bytes || output.offset >= input.offset + in_bytes, "HIP pooling buffers overlap");
        DeviceScope selected(context->device);
        const auto launch = [&]<class Storage>() {
            const auto* src = reinterpret_cast<const Storage*>(static_cast<const char*>(input.buffer->data) + input.offset);
            auto* dst = reinterpret_cast<Storage*>(static_cast<char*>(output.buffer->data) + output.offset);
            hipLaunchKernelGGL((inference_pool<Storage>), dim3(blocks(p->count)), dim3(256), 0, context->stream, src, dst, *p);
        };
        if (dtype == LFS_HIP_F32)
            launch.template operator()<float>();
        else
            launch.template operator()<__half>();
        check(hipGetLastError());
    });
}

extern "C" int lfs_hip_pool_f32(LfsHipContext* context, LfsHipView input, LfsHipView output, const LfsHipPool* p, LfsHipError* error) {
    return lfs_hip_pool_typed(context, input, output, p, LFS_HIP_F32, error);
}
