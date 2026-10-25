/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
int lfs_hip_shared_warp(LfsHipContext* context, LfsHipView input, uint32_t dtype, uint32_t channels,
                        uint32_t mode, LfsHipView output, const LfsHipLens* lens, LfsHipError* error) {
    // Preserve the already verified three-channel pinhole implementation exactly.
    if (lens && lens->model_type == 0 && channels == 3 && mode == 0)
        return lfs_hip_warp_rgb(context, input, dtype, output, lens, error);
    return boundary(error, [&] {
        require(context && lens && lens->model_type >= 0 && lens->model_type <= 4 && channels >= 1 && channels <= 4 && mode <= 3,
                "Invalid HIP shared camera descriptor");
        require((dtype == LFS_HIP_F32 || dtype == LFS_HIP_U8) && (!mode || dtype == LFS_HIP_F32) &&
                    (mode != 2 || channels == 1) && (mode != 3 || channels == 3),
                "HIP shared camera layout/dtype mismatch");
        require(lens->src_width > 0 && lens->src_height > 0 && lens->dst_width > 0 && lens->dst_height > 0 &&
                    uint64_t(lens->src_width) * lens->src_height <= INT32_MAX / channels &&
                    uint64_t(lens->dst_width) * lens->dst_height <= INT32_MAX / channels,
                "HIP shared camera extent exceeds indexing");
        require(std::isfinite(lens->src_fx) && lens->src_fx > 0 && std::isfinite(lens->src_fy) && lens->src_fy > 0 &&
                    std::isfinite(lens->dst_fx) && lens->dst_fx > 0 && std::isfinite(lens->dst_fy) && lens->dst_fy > 0 &&
                    std::isfinite(lens->src_cx) && std::isfinite(lens->src_cy) && std::isfinite(lens->dst_cx) && std::isfinite(lens->dst_cy) &&
                    lens->num_distortion >= 0 && lens->num_distortion <= 12,
                "Invalid HIP shared camera calibration");
        bool identity = mode == 0 && lens->model_type == 0 && lens->src_width == lens->dst_width && lens->src_height == lens->dst_height &&
                        lens->src_fx == lens->dst_fx && lens->src_fy == lens->dst_fy && lens->src_cx == lens->dst_cx && lens->src_cy == lens->dst_cy;
        for (int i = 0; i < lens->num_distortion; ++i) {
            require(std::isfinite(lens->distortion[i]), "Nonfinite HIP shared camera distortion");
            identity = identity && lens->distortion[i] == 0;
        }

        const float footprint = std::max(lens->src_fx / lens->dst_fx, lens->src_fy / lens->dst_fy);
        require(std::isfinite(footprint) && footprint <= 64, "HIP camera sampling footprint exceeds bounded work");
        const int quadrature = std::max(8, int(std::ceil(footprint)));
        const uint64_t row_work = uint64_t(lens->dst_width) * quadrature * quadrature;
        require(identity || row_work <= 2 * 1024 * 1024, "HIP camera row exceeds bounded sampling work");
        const size_t input_bytes = size_t(lens->src_width) * lens->src_height * channels * (dtype == LFS_HIP_U8 ? 1 : 4);
        const size_t output_bytes = size_t(lens->dst_width) * lens->dst_height * channels * 4;
        validate(context, input.buffer, input.offset, input_bytes);
        validate(context, output.buffer, output.offset, output_bytes);
        require(output.offset % 4 == 0 && (dtype == LFS_HIP_U8 || input.offset % 4 == 0), "Misaligned HIP shared camera view");
        require(input.buffer != output.buffer || input.offset >= output.offset + output_bytes || output.offset >= input.offset + input_bytes,
                "HIP shared camera buffers overlap");
        DeviceScope device(context->device);
        const auto* source = static_cast<const char*>(input.buffer->data) + input.offset;
        auto* destination = reinterpret_cast<float*>(static_cast<char*>(output.buffer->data) + output.offset);
        if (identity) {
            const size_t count = output_bytes / 4;
            if (dtype == LFS_HIP_U8)
                hipLaunchKernelGGL((identity_rgb<uint8_t>), dim3(blocks(count)), dim3(256), 0, context->stream, reinterpret_cast<const uint8_t*>(source), destination, count);
            else
                hipLaunchKernelGGL((identity_rgb<float>), dim3(blocks(count)), dim3(256), 0, context->stream, reinterpret_cast<const float*>(source), destination, count);
            check(hipGetLastError());
            return;
        }
        const int rows = std::max(1, std::min(32, int((2 * 1024 * 1024) / row_work)));
        for (int y = 0; y < lens->dst_height; y += rows) {
            const int end = std::min(lens->dst_height, y + rows);
            const dim3 block(16, 16), grid((uint32_t(lens->dst_width) + 15) / 16, (end - y + 15) / 16);
            if (mode) {
                const auto filter = static_cast<hip_lens::AreaFilterMode>(mode - 1);
                hipLaunchKernelGGL(hip_lens::undistort_area_kernel, grid, block, 0, context->stream,
                                   reinterpret_cast<const float*>(source), destination, int(channels), filter, y, end, quadrature, *lens);
            } else if (dtype == LFS_HIP_U8)
                hipLaunchKernelGGL((hip_lens::undistort_image_kernel<uint8_t>), grid, block, 0, context->stream,
                                   reinterpret_cast<const uint8_t*>(source), destination, int(channels), y, end, quadrature, *lens);
            else
                hipLaunchKernelGGL((hip_lens::undistort_image_kernel<float>), grid, block, 0, context->stream,
                                   reinterpret_cast<const float*>(source), destination, int(channels), y, end, quadrature, *lens);
            check(hipGetLastError());
        }
    });
}
