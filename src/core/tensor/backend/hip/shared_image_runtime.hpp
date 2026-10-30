/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "shared_image_kernels.hpp"
namespace hip_shared_image {
    __global__ void exact_mask(const uint8_t* input, float* output, size_t count) {
        const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < count)
            output[i] = normalize_mask_byte(input[i]);
    }
} // namespace hip_shared_image
int lfs_hip_shared_image(LfsHipContext* context, const LfsHipSharedImage* p, LfsHipView source,
                         LfsHipView destination, LfsHipView auxiliary, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && p && p->struct_bytes == sizeof(*p) && p->operation <= 22, "Invalid HIP shared-image descriptor");
        require(p->height && p->width && p->channels && p->height <= INT32_MAX / p->width &&
                    p->height * p->width <= INT32_MAX / p->channels,
                "HIP shared-image extent exceeds indexing");
        const size_t pixels = p->height * p->width, total = pixels * p->channels;
        const auto op = p->operation;
        size_t src_bytes = 0, dst_bytes = 0, aux_bytes = 0;
        size_t count = total;
        switch (op) {
        case 0:
        case 8:
        case 14:
            src_bytes = total;
            dst_bytes = total * (op == 8 ? 1 : 4);
            break;
        case 1:
        case 9:
        case 15:
            src_bytes = total * 2;
            dst_bytes = total * (op == 9 ? 1 : 4);
            break;
        case 2:
            src_bytes = total * 4;
            dst_bytes = total * 2;
            break;
        case 3:
            src_bytes = total * 2;
            dst_bytes = total * 4;
            break;
        case 4:
        case 5:
        case 6:
        case 7:
        case 12:
            require(p->channels == 3, "HIP normal conversion needs three channels");
            src_bytes = total * (op == 6 ? 1 : op == 7 ? 2
                                                       : 4);
            dst_bytes = total * 4;
            count = pixels;
            break;
        case 10:
            src_bytes = total * 4;
            dst_bytes = total;
            break;
        case 11:
        case 13:
            require(p->channels == 1, "HIP scalar conversion needs one channel");
            src_bytes = total;
            dst_bytes = total * 4;
            break;
        case 16: dst_bytes = total; break;
        case 17:
            src_bytes = total;
            dst_bytes = 4;
            break;
        case 18:
        case 19:
            require(p->channels == 4, "HIP RGBA split needs four channels");
            src_bytes = total;
            dst_bytes = pixels * 3 * (op == 18 ? 4 : 1);
            aux_bytes = pixels * 4;
            count = pixels;
            break;
        case 20:
        case 21:
        case 22:
            require(p->channels == 1 && std::isfinite(p->threshold), "Invalid HIP mask transform");
            dst_bytes = total * 4;
            break;
        }
        if (src_bytes)
            validate(context, source.buffer, source.offset, src_bytes);
        validate(context, destination.buffer, destination.offset, dst_bytes);
        if (aux_bytes)
            validate(context, auxiliary.buffer, auxiliary.offset, aux_bytes);
        auto overlap = [](LfsHipView a, size_t an, LfsHipView b, size_t bn) { return an && bn && a.buffer == b.buffer && a.offset < b.offset + bn && b.offset < a.offset + an; };
        require(!overlap(source, src_bytes, destination, dst_bytes) && !overlap(source, src_bytes, auxiliary, aux_bytes) &&
                    !overlap(destination, dst_bytes, auxiliary, aux_bytes),
                "HIP image conversion buffers overlap");
        const bool src16 = op == 1 || op == 3 || op == 7 || op == 9 || op == 15;
        const bool src32 = op == 2 || op == 4 || op == 5 || op == 10 || op == 12;
        const bool dst16 = op == 2;
        const bool dst32 = !(op == 2 || op == 8 || op == 9 || op == 10 || op == 16 || op == 19);
        require((!src16 || source.offset % 2 == 0) && (!src32 || source.offset % 4 == 0) &&
                    (!dst16 || destination.offset % 2 == 0) && (!dst32 || destination.offset % 4 == 0) &&
                    (!aux_bytes || auxiliary.offset % 4 == 0),
                "Misaligned HIP image view");
        for (float value : p->w2c)
            require(std::isfinite(value), "Nonfinite HIP normal transform");
        DeviceScope device(context->device);
        auto address = [](LfsHipView v) -> char* { return v.buffer ? static_cast<char*>(v.buffer->data) + v.offset : nullptr; };
        const auto* in8 = reinterpret_cast<const uint8_t*>((src_bytes ? address(source) : nullptr));
        const auto* in16 = reinterpret_cast<const uint16_t*>((src_bytes ? address(source) : nullptr));
        const auto* in32 = reinterpret_cast<const float*>((src_bytes ? address(source) : nullptr));
        auto* out8 = reinterpret_cast<uint8_t*>(address(destination));
        auto* out16 = reinterpret_cast<uint16_t*>(address(destination));
        auto* out32 = reinterpret_cast<float*>(address(destination));
        auto* alpha = reinterpret_cast<float*>((aux_bytes ? address(auxiliary) : nullptr));
        const dim3 grid((count + 255) / 256), block(256);
        auto stream = context->stream;
        using namespace hip_shared_image;
        NormalPriorTransform transform{bool(p->srgb), bool(p->flip_yz), bool(p->world_to_camera), {}};
        std::copy(p->w2c, p->w2c + 9, transform.w2c);
#define IMAGE_LAUNCH(kernel, ...) hipLaunchKernelGGL(kernel, grid, block, 0, stream, __VA_ARGS__)
        switch (op) {
        case 0: IMAGE_LAUNCH((uint8_hwc_to_float32_chw_kernel<false>), in8, out32, p->height, p->width, p->channels); break;
        case 1: IMAGE_LAUNCH((uint16_hwc_to_float32_chw_kernel<false>), in16, out32, p->height, p->width, p->channels); break;
        case 2: IMAGE_LAUNCH(float32_hwc_to_uint16_hwc_kernel, in32, out16, total); break;
        case 3: IMAGE_LAUNCH((uint16_hwc_to_float_hwc_kernel<false>), in16, out32, total); break;
        case 4: IMAGE_LAUNCH(normal_chw_to_jpeg2k_hwc_kernel, in32, out32, pixels); break;
        case 5: IMAGE_LAUNCH(jpeg2k_hwc_to_normal_chw_kernel, in32, out32, pixels); break;
        case 6: IMAGE_LAUNCH((normal_prior_hwc_to_float32_chw_kernel<uint8_t>), in8, out32, pixels, transform); break;
        case 7: IMAGE_LAUNCH((normal_prior_hwc_to_float32_chw_kernel<uint16_t>), in16, out32, pixels, transform); break;
        case 8: IMAGE_LAUNCH(uint8_hwc_to_uint8_chw_kernel, in8, out8, p->height, p->width, p->channels); break;
        case 9: IMAGE_LAUNCH(uint16_hwc_to_uint8_chw_kernel, in16, out8, p->height, p->width, p->channels); break;
        case 10: IMAGE_LAUNCH(float32_chw_to_uint8_chw_kernel, in32, out8, total); break;
        case 11: IMAGE_LAUNCH(uint8_hw_to_float32_hw_kernel, in8, out32, total); break;
        case 12: IMAGE_LAUNCH((normal_prior_hwc_to_float32_chw_kernel<float>), in32, out32, pixels, transform); break;
        case 13: IMAGE_LAUNCH(exact_mask, in8, out32, total); break;
        case 14: IMAGE_LAUNCH((uint8_hwc_to_float32_chw_kernel<true>), in8, out32, p->height, p->width, p->channels); break;
        case 15: IMAGE_LAUNCH((uint16_hwc_to_float32_chw_kernel<true>), in16, out32, p->height, p->width, p->channels); break;
        case 16: IMAGE_LAUNCH(fill_u8_sentinel_kernel, out8, total, p->seed); break;
        case 17: IMAGE_LAUNCH(flag_u8_sentinel_unchanged_kernel, in8, total, p->seed, reinterpret_cast<uint32_t*>(out8)); break;
        case 18: IMAGE_LAUNCH(uint8_rgba_split_kernel, in8, out32, alpha, pixels); break;
        case 19: IMAGE_LAUNCH(uint8_rgba_split_uint8_rgb_kernel, in8, out8, alpha, pixels); break;
        case 20: IMAGE_LAUNCH(mask_invert_kernel, out32, total); break;
        case 21: IMAGE_LAUNCH(mask_threshold_kernel, out32, total, p->threshold); break;
        case 22: IMAGE_LAUNCH((quantize_mask_grid_kernel<false>), out32, nullptr, total); break;
        }
#undef IMAGE_LAUNCH
        check(hipGetLastError());
    });
}
