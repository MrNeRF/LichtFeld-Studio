/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/mask_normalization.cuh"
#include "core/cuda_error.hpp"
#include "image_format_kernels.cuh"
#include <cassert>

namespace lfs::io::cuda {

    using lfs::gpu_ops::NormalPriorTransform;

    namespace {
        constexpr int BLOCK_SIZE = 256;
        constexpr float NORMALIZE_SCALE_U8 = 1.0f / 255.0f;
        constexpr float NORMALIZE_SCALE_U16 = 1.0f / 65535.f;

#include "core/device/image_sentinel.cuh"
    } // namespace

#include "core/device/image_format.cuh"

    void launch_fill_u8_sentinel(
        uint8_t* output,
        const size_t byte_count,
        const uint32_t seed,
        cudaStream_t stream) {
        assert(output != nullptr);
        assert(byte_count > 0);
        int num_blocks = static_cast<int>((byte_count + BLOCK_SIZE - 1) / BLOCK_SIZE);
        num_blocks = num_blocks > 1024 ? 1024 : num_blocks;
        fill_u8_sentinel_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            output, byte_count, seed);
        LFS_CUDA_LAUNCH_CHECK(stream, "io.image.fill_u8_sentinel");
    }

    void launch_flag_u8_sentinel_unchanged(
        const uint8_t* input,
        const size_t byte_count,
        const uint32_t seed,
        uint32_t* unchanged,
        cudaStream_t stream) {
        assert(input != nullptr);
        assert(unchanged != nullptr);
        assert(byte_count > 0);
        int num_blocks = static_cast<int>((byte_count + BLOCK_SIZE - 1) / BLOCK_SIZE);
        num_blocks = num_blocks > 1024 ? 1024 : num_blocks;
        flag_u8_sentinel_unchanged_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input, byte_count, seed, unchanged);
        LFS_CUDA_LAUNCH_CHECK(stream, "io.image.flag_u8_sentinel_unchanged");
    }

    namespace {

    } // namespace

    void launch_uint8_hwc_to_float32_chw(
        const uint8_t* input,
        float* output,
        const size_t height,
        const size_t width,
        const size_t channels,
        cudaStream_t stream, bool exact_mask_alpha) {

        const size_t total = height * width * channels;
        const int num_blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);

        if (channels == 4 && exact_mask_alpha) {
            uint8_hwc_to_float32_chw_kernel<true><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                input, output, height, width, channels);
            LFS_CUDA_LAUNCH_CHECK(stream, "io.image.u8_hwc_to_f32_chw");
        } else {
            uint8_hwc_to_float32_chw_kernel<false><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                input, output, height, width, channels);
            LFS_CUDA_LAUNCH_CHECK(stream, "io.image.u8_hwc_to_f32_chw");
        }
    }

    void launch_uint16_hwc_to_float32_chw(
        const uint16_t* input,
        float* output,
        size_t height,
        size_t width,
        size_t channels,
        cudaStream_t stream, bool exact_mask_alpha) {

        const size_t total = height * width * channels;
        const int num_blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);

        if (channels == 4 && exact_mask_alpha) {
            uint16_hwc_to_float32_chw_kernel<true><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                input, output, height, width, channels);
            LFS_CUDA_LAUNCH_CHECK(stream, "io.image.u16_hwc_to_f32_chw");
        } else {
            uint16_hwc_to_float32_chw_kernel<false><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                input, output, height, width, channels);
            LFS_CUDA_LAUNCH_CHECK(stream, "io.image.u16_hwc_to_f32_chw");
        }
    }

    void launch_normal_prior_u8_hwc_to_float32_chw(
        const uint8_t* input,
        float* output,
        size_t height,
        size_t width,
        const NormalPriorTransform& transform,
        cudaStream_t stream) {

        const size_t pixel_count = height * width;
        const int num_blocks = static_cast<int>((pixel_count + BLOCK_SIZE - 1) / BLOCK_SIZE);

        normal_prior_hwc_to_float32_chw_kernel<uint8_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input, output, pixel_count, transform);
        LFS_CUDA_LAUNCH_CHECK(stream, "io.image.normal_prior_u8");
    }

    void launch_normal_prior_u16_hwc_to_float32_chw(
        const uint16_t* input,
        float* output,
        size_t height,
        size_t width,
        const NormalPriorTransform& transform,
        cudaStream_t stream) {

        const size_t pixel_count = height * width;
        const int num_blocks = static_cast<int>((pixel_count + BLOCK_SIZE - 1) / BLOCK_SIZE);

        normal_prior_hwc_to_float32_chw_kernel<uint16_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input, output, pixel_count, transform);
        LFS_CUDA_LAUNCH_CHECK(stream, "io.image.normal_prior_u16");
    }

    void launch_normal_prior_f32_hwc_to_float32_chw(
        const float* input,
        float* output,
        size_t height,
        size_t width,
        const NormalPriorTransform& transform,
        cudaStream_t stream) {

        const size_t pixel_count = height * width;
        const int num_blocks = static_cast<int>((pixel_count + BLOCK_SIZE - 1) / BLOCK_SIZE);

        normal_prior_hwc_to_float32_chw_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input, output, pixel_count, transform);
        LFS_CUDA_LAUNCH_CHECK(stream, "io.image.normal_prior_f32");
    }

    void launch_float32_hwc_to_uint16_hwc(
        const float* input,
        uint16_t* output,
        size_t height,
        size_t width,
        size_t channels,
        cudaStream_t stream) {

        const size_t total = height * width * channels;
        const int num_blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);

        float32_hwc_to_uint16_hwc_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input, output, total);
        LFS_CUDA_LAUNCH_CHECK(stream, "io.image.f32_hwc_to_u16_hwc");
    }

    void launch_uint16_hwc_to_float32_hwc(
        const uint16_t* input,
        float* output,
        size_t height,
        size_t width,
        size_t channels,
        cudaStream_t stream, bool exact_mask_alpha) {

        const size_t total = height * width * channels;
        const int num_blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);

        if (channels == 4 && exact_mask_alpha) {
            uint16_hwc_to_float_hwc_kernel<true><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                input, output, total);
            LFS_CUDA_LAUNCH_CHECK(stream, "io.image.u16_hwc_to_f32_hwc");
        } else {
            uint16_hwc_to_float_hwc_kernel<false><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                input, output, total);
            LFS_CUDA_LAUNCH_CHECK(stream, "io.image.u16_hwc_to_f32_hwc");
        }
    }

    void launch_normal_chw_to_jpeg2k_hwc(
        const float* input,
        float* output,
        size_t height,
        size_t width,
        cudaStream_t stream) {

        const size_t pixel_count = height * width;
        const int num_blocks = static_cast<int>((pixel_count + BLOCK_SIZE - 1) / BLOCK_SIZE);

        normal_chw_to_jpeg2k_hwc_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input, output, pixel_count);
        LFS_CUDA_LAUNCH_CHECK(stream, "io.image.normal_to_j2k");
    }

    void launch_jpeg2k_hwc_to_normal_chw(
        const float* input,
        float* output,
        size_t height,
        size_t width,
        cudaStream_t stream) {

        const size_t pixel_count = height * width;
        const int num_blocks = static_cast<int>((pixel_count + BLOCK_SIZE - 1) / BLOCK_SIZE);

        jpeg2k_hwc_to_normal_chw_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input, output, pixel_count);
        LFS_CUDA_LAUNCH_CHECK(stream, "io.image.j2k_to_normal");
    }

    void launch_uint8_hwc_to_uint8_chw(
        const uint8_t* input,
        uint8_t* output,
        const size_t height,
        const size_t width,
        const size_t channels,
        cudaStream_t stream) {

        const size_t total = height * width * channels;
        const int num_blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);

        uint8_hwc_to_uint8_chw_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input, output, height, width, channels);
        LFS_CUDA_LAUNCH_CHECK(stream, "io.image.u8_hwc_to_u8_chw");
    }

    void launch_uint16_hwc_to_uint8_chw(
        const uint16_t* input,
        uint8_t* output,
        const size_t height,
        const size_t width,
        const size_t channels,
        cudaStream_t stream) {

        const size_t total = height * width * channels;
        const int num_blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);

        uint16_hwc_to_uint8_chw_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input, output, height, width, channels);
        LFS_CUDA_LAUNCH_CHECK(stream, "io.image.u16_hwc_to_u8_chw");
    }

    void launch_float32_chw_to_uint8_chw(
        const float* input,
        uint8_t* output,
        const size_t height,
        const size_t width,
        const size_t channels,
        cudaStream_t stream) {

        const size_t total = height * width * channels;
        const int num_blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);

        float32_chw_to_uint8_chw_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input, output, total);
        LFS_CUDA_LAUNCH_CHECK(stream, "io.image.f32_chw_to_u8_chw");
    }

    void launch_uint8_hw_to_float32_hw(
        const uint8_t* input,
        float* output,
        const size_t height,
        const size_t width,
        cudaStream_t stream) {

        const size_t total = height * width;
        const int num_blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);

        uint8_hw_to_float32_hw_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input, output, total);
        LFS_CUDA_LAUNCH_CHECK(stream, "io.image.u8_hw_to_f32_hw");
    }

    void launch_uint8_rgba_split_to_float32_rgb_and_alpha(
        const uint8_t* input,
        float* rgb_output,
        float* alpha_output,
        const size_t height,
        const size_t width,
        cudaStream_t stream) {

        assert(input && "input must not be null");
        assert(rgb_output && "rgb_output must not be null");
        assert(alpha_output && "alpha_output must not be null");
        assert(height > 0 && width > 0);

        const size_t total = height * width;
        const int num_blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);

        uint8_rgba_split_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input, rgb_output, alpha_output, total);
        LFS_CUDA_LAUNCH_CHECK(stream, "io.image.rgba_split_f32");
    }

    void launch_uint8_rgba_split_to_uint8_rgb_and_float32_alpha(
        const uint8_t* input,
        uint8_t* rgb_output,
        float* alpha_output,
        const size_t height,
        const size_t width,
        cudaStream_t stream) {

        assert(input && "input must not be null");
        assert(rgb_output && "rgb_output must not be null");
        assert(alpha_output && "alpha_output must not be null");
        assert(height > 0 && width > 0);

        const size_t total = height * width;
        const int num_blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);

        uint8_rgba_split_uint8_rgb_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input, rgb_output, alpha_output, total);
        LFS_CUDA_LAUNCH_CHECK(stream, "io.image.rgba_split_u8");
    }

    void launch_mask_invert(
        float* data,
        const size_t height,
        const size_t width,
        cudaStream_t stream) {

        const size_t total = height * width;
        const int num_blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);

        mask_invert_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(data, total);
        LFS_CUDA_LAUNCH_CHECK(stream, "io.image.mask_invert");
    }

    void launch_mask_threshold(
        float* data,
        const size_t height,
        const size_t width,
        const float threshold,
        cudaStream_t stream) {

        const size_t total = height * width;
        const int num_blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);

        mask_threshold_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(data, total, threshold);
        LFS_CUDA_LAUNCH_CHECK(stream, "io.image.mask_threshold");
    }

} // namespace lfs::io::cuda
