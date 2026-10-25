/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "inverse_warp_api.h"
int lfs_hip_inverse_warp(LfsHipContext* context, LfsHipView input, LfsHipView output,
                         LfsHipView validity, uint32_t channels, uint32_t mode,
                         const LfsHipLens* lens, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && lens && lens->model_type >= 0 && lens->model_type <= 4 && channels >= 1 && channels <= 4 &&
                    mode <= 3 && (mode != 2 || channels == 1) && (mode != 3 || channels == 3),
                "Invalid HIP inverse warp descriptor");
        require(lens->src_width > 0 && lens->src_height > 0 && lens->dst_width > 0 && lens->dst_height > 0 &&
                    uint64_t(lens->src_width) * lens->src_height <= INT32_MAX / channels &&
                    uint64_t(lens->dst_width) * lens->dst_height <= INT32_MAX / channels,
                "HIP inverse warp extent exceeds signed indexing");
        require(std::isfinite(lens->src_fx) && lens->src_fx > 0 && std::isfinite(lens->src_fy) && lens->src_fy > 0 &&
                    std::isfinite(lens->dst_fx) && lens->dst_fx > 0 && std::isfinite(lens->dst_fy) && lens->dst_fy > 0 &&
                    std::isfinite(lens->src_cx) && std::isfinite(lens->src_cy) && std::isfinite(lens->dst_cx) &&
                    std::isfinite(lens->dst_cy) && lens->num_distortion >= 0 && lens->num_distortion <= 12,
                "Invalid HIP inverse camera calibration");
        for (int i = 0; i < lens->num_distortion; ++i)
            require(std::isfinite(lens->distortion[i]), "Nonfinite HIP inverse distortion");
        const size_t src_bytes = size_t(lens->dst_width) * lens->dst_height * channels * 4;
        const size_t dst_bytes = size_t(lens->src_width) * lens->src_height * channels * 4;
        const size_t valid_bytes = mode == 0 ? size_t(lens->src_width) * lens->src_height : 0;
        validate(context, input.buffer, input.offset, src_bytes);
        validate(context, output.buffer, output.offset, dst_bytes);
        if (valid_bytes)
            validate(context, validity.buffer, validity.offset, valid_bytes);
        require(input.offset % 4 == 0 && output.offset % 4 == 0, "Misaligned HIP inverse warp");
        auto overlap = [](LfsHipView a, size_t an, LfsHipView b, size_t bn) { return an && bn && a.buffer == b.buffer && a.offset < b.offset + bn && b.offset < a.offset + an; };
        require(!overlap(input, src_bytes, output, dst_bytes) && !overlap(input, src_bytes, validity, valid_bytes) &&
                    !overlap(output, dst_bytes, validity, valid_bytes),
                "HIP inverse warp storage overlaps");
        DeviceScope device(context->device);
        auto address = [](LfsHipView v) { return static_cast<char*>(v.buffer->data) + v.offset; };
        const auto* src = reinterpret_cast<const float*>(address(input));
        auto* dst = reinterpret_cast<float*>(address(output));
        auto* valid = valid_bytes ? reinterpret_cast<uint8_t*>(address(validity)) : nullptr;
        for (int row = 0; row < lens->src_height; row += 32) {
            const int end = std::min(lens->src_height, row + 32);
            const dim3 block(16, 16), grid((uint32_t(lens->src_width) + 15) / 16, (end - row + 15) / 16);
            if (mode == 0)
                hipLaunchKernelGGL(hip_lens::distort_image_to_source_kernel, grid, block, 0, context->stream, src, dst, valid, int(channels), row, end, *lens);
            else
                hipLaunchKernelGGL(hip_lens::distort_area_to_source_kernel, grid, block, 0, context->stream, src, dst, int(channels), static_cast<hip_lens::AreaFilterMode>(mode - 1), row, end, *lens);
            check(hipGetLastError());
        }
    });
}
