/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-FileCopyrightText: 2025 Youyu Chen (original Lanczos implementation)
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-License-Identifier: MIT (original Lanczos implementation)
 */

#include "core/logger.hpp"
#include "lanczos_resize.hpp"

#include <cassert>
#include <cmath>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_X      16
#define BLOCK_Y      16
#define NUM_CHANNELS 3

namespace cg = cooperative_groups;

namespace {
    template <typename T>
    struct PixelTraits;

    template <>
    struct PixelTraits<uint8_t> {
        __device__ __forceinline__ static float load_and_normalize(const uint8_t* ptr, int idx) {
            return static_cast<float>(ptr[idx]) / 255.0f;
        }
    };

    template <>
    struct PixelTraits<float> {
        __device__ __forceinline__ static float load_and_normalize(const float* ptr, int idx) {
            return ptr[idx];
        }
    };

} // namespace

namespace lfs::core {
    namespace detail {

        __device__ float sinc(const float x) {
            if (fabsf(x) < 1e-12f)
                return 1.0f;
            return sinf(M_PI * x) / (M_PI * x);
        }

        __device__ float lanczos_kernel(const float x, const float a) {
            if (x <= -a || x >= a)
                return 0.0f;
            return sinc(x) * sinc(x / a);
        }

        __global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
            PreComputeCoef(
                const int input_size,
                const int output_size,
                const int kernel_size,
                float* __restrict__ kernel_values) {
            const auto block = cg::this_thread_block();
            const uint32_t thread_idx = block.thread_index().x;
            const uint32_t output_idx = block.group_index().x * BLOCK_X * BLOCK_Y + thread_idx;
            const float output_ax = (float)output_idx;
            const float scale = 1.0f * input_size / output_size;
            const bool inside = output_idx < output_size;

            if (!inside)
                return;

            const float center = (output_ax + 0.5f) * scale;
            const int2 box = {
                max((int)(center - kernel_size * scale + 0.5f), 0),
                min((int)(center + kernel_size * scale + 0.5f), input_size)};

            const uint32_t offset = output_idx * (uint32_t)(kernel_size * scale * 2 + 1 + 0.5f);
            float norm = 0.0f;
            for (int i = box.x; i < box.y; i++) {
                float value = lanczos_kernel((i + 0.5f - center) / scale, kernel_size);
                kernel_values[offset + i - box.x] = value;
                norm += value;
            }
            for (int i = box.x; i < box.y; i++) {
                kernel_values[offset + i - box.x] /= norm;
            }
        }

        template <uint32_t CHANNELS, typename T>
        __global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
            LanczosResampleCUDA(
                const int input_h, const int input_w,
                const int output_h, const int output_w,
                const int kernel_size,
                const float* __restrict__ pre_coef_x,
                const float* __restrict__ pre_coef_y,
                const T* __restrict__ input, // [H, W, C] T
                float* __restrict__ output   // [C, H, W] float32
            ) {
            const auto block = cg::this_thread_block();
            const uint32_t thread_idx_x = block.thread_index().x;
            const uint32_t thread_idx_y = block.thread_index().y;
            const uint2 pix = {
                block.group_index().x * BLOCK_X + thread_idx_x,
                block.group_index().y * BLOCK_Y + thread_idx_y};
            const float2 pixf = {(float)pix.x, (float)pix.y};
            float scale_h = 1.0f * input_h / output_h, scale_w = 1.0f * input_w / output_w;

            const bool inside = (pix.x < output_w && pix.y < output_h);

            if (!inside)
                return;

            const float2 center = {(pixf.x + 0.5f) * scale_w, (pixf.y + 0.5f) * scale_h};

            const int2 LU = {
                max((int)(center.x - kernel_size * scale_w + 0.5f), 0),
                max((int)(center.y - kernel_size * scale_h + 0.5f), 0)};
            const int2 RD = {
                min((int)(center.x + kernel_size * scale_w + 0.5f), input_w),
                min((int)(center.y + kernel_size * scale_h + 0.5f), input_h)};

            uint32_t coef_offset_step_y = (uint32_t)(kernel_size * scale_h * 2 + 1 + 0.5f);
            uint32_t coef_offset_step_x = (uint32_t)(kernel_size * scale_w * 2 + 1 + 0.5f);

            float accumulator[CHANNELS] = {0.0f};

            for (int y = LU.y; y < RD.y; y++) {
                const float kernel_value_y = pre_coef_y[pix.y * coef_offset_step_y + y - LU.y];
                for (int x = LU.x; x < RD.x; x++) {
                    const uint32_t input_pix_id = input_w * y + x;
                    const float kernel_value_x = pre_coef_x[pix.x * coef_offset_step_x + x - LU.x];
                    const float kernel_value = kernel_value_y * kernel_value_x;

                    for (int ch = 0; ch < CHANNELS; ch++) {
                        accumulator[ch] += PixelTraits<T>::load_and_normalize(input, input_pix_id * 3 + ch) * kernel_value;
                    }
                }
            }

            const int H_out = output_h;
            const int W_out = output_w;
            for (int ch = 0; ch < CHANNELS; ch++) {
                output[ch * (H_out * W_out) + pix.y * W_out + pix.x] = accumulator[ch];
            }
        }

        template <typename TIn>
        __global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
            LanczosResampleGrayscaleCUDA(
                const int input_h, const int input_w,
                const int output_h, const int output_w,
                const int kernel_size,
                const float* __restrict__ pre_coef_x,
                const float* __restrict__ pre_coef_y,
                const TIn* __restrict__ input, // [H, W]
                const float input_scale,       // 1/255 for uint8, 1 for float32
                float* __restrict__ output     // [H, W] float32
            ) {
            const auto block = cg::this_thread_block();
            const uint32_t thread_idx_x = block.thread_index().x;
            const uint32_t thread_idx_y = block.thread_index().y;
            const uint2 pix = {
                block.group_index().x * BLOCK_X + thread_idx_x,
                block.group_index().y * BLOCK_Y + thread_idx_y};
            const float2 pixf = {(float)pix.x, (float)pix.y};
            float scale_h = 1.0f * input_h / output_h, scale_w = 1.0f * input_w / output_w;

            const bool inside = (pix.x < output_w && pix.y < output_h);

            if (!inside)
                return;

            const float2 center = {(pixf.x + 0.5f) * scale_w, (pixf.y + 0.5f) * scale_h};

            const int2 LU = {
                max((int)(center.x - kernel_size * scale_w + 0.5f), 0),
                max((int)(center.y - kernel_size * scale_h + 0.5f), 0)};
            const int2 RD = {
                min((int)(center.x + kernel_size * scale_w + 0.5f), input_w),
                min((int)(center.y + kernel_size * scale_h + 0.5f), input_h)};

            const uint32_t coef_offset_step_y = (uint32_t)(kernel_size * scale_h * 2 + 1 + 0.5f);
            const uint32_t coef_offset_step_x = (uint32_t)(kernel_size * scale_w * 2 + 1 + 0.5f);

            float accumulator = 0.0f;

            for (int y = LU.y; y < RD.y; y++) {
                const float kernel_value_y = pre_coef_y[pix.y * coef_offset_step_y + y - LU.y];
                for (int x = LU.x; x < RD.x; x++) {
                    const uint32_t input_pix_id = input_w * y + x;
                    const float kernel_value_x = pre_coef_x[pix.x * coef_offset_step_x + x - LU.x];
                    const float kernel_value = kernel_value_y * kernel_value_x;
                    const float pixel_value = (float)input[input_pix_id] * input_scale;
                    accumulator += pixel_value * kernel_value;
                }
            }

            output[pix.y * output_w + pix.x] = accumulator;
        }

        template <uint32_t CHANNELS>
        __global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
            LanczosResamplePlanarCUDA(
                const int input_h, const int input_w,
                const int output_h, const int output_w,
                const int kernel_size,
                const float* __restrict__ pre_coef_x,
                const float* __restrict__ pre_coef_y,
                const float* __restrict__ input, // [C, H, W] float32
                float* __restrict__ output       // [C, H, W] float32
            ) {
            const auto block = cg::this_thread_block();
            const uint32_t thread_idx_x = block.thread_index().x;
            const uint32_t thread_idx_y = block.thread_index().y;
            const uint2 pix = {
                block.group_index().x * BLOCK_X + thread_idx_x,
                block.group_index().y * BLOCK_Y + thread_idx_y};
            const float2 pixf = {(float)pix.x, (float)pix.y};
            float scale_h = 1.0f * input_h / output_h, scale_w = 1.0f * input_w / output_w;

            const bool inside = (pix.x < output_w && pix.y < output_h);

            if (!inside)
                return;

            const float2 center = {(pixf.x + 0.5f) * scale_w, (pixf.y + 0.5f) * scale_h};

            const int2 LU = {
                max((int)(center.x - kernel_size * scale_w + 0.5f), 0),
                max((int)(center.y - kernel_size * scale_h + 0.5f), 0)};
            const int2 RD = {
                min((int)(center.x + kernel_size * scale_w + 0.5f), input_w),
                min((int)(center.y + kernel_size * scale_h + 0.5f), input_h)};

            const uint32_t coef_offset_step_y = (uint32_t)(kernel_size * scale_h * 2 + 1 + 0.5f);
            const uint32_t coef_offset_step_x = (uint32_t)(kernel_size * scale_w * 2 + 1 + 0.5f);

            const int input_plane = input_h * input_w;
            const int output_plane = output_h * output_w;

            float accumulator[CHANNELS] = {0.0f};

            for (int y = LU.y; y < RD.y; y++) {
                const float kernel_value_y = pre_coef_y[pix.y * coef_offset_step_y + y - LU.y];
                for (int x = LU.x; x < RD.x; x++) {
                    const uint32_t input_pix_id = input_w * y + x;
                    const float kernel_value_x = pre_coef_x[pix.x * coef_offset_step_x + x - LU.x];
                    const float kernel_value = kernel_value_y * kernel_value_x;

                    for (int ch = 0; ch < CHANNELS; ch++) {
                        accumulator[ch] += input[ch * input_plane + input_pix_id] * kernel_value;
                    }
                }
            }

            for (int ch = 0; ch < CHANNELS; ch++) {
                output[ch * output_plane + pix.y * output_w + pix.x] = accumulator[ch];
            }
        }

        template <typename T>
        static Tensor lanczos_resize_impl(
            const Tensor& input,
            int output_h,
            int output_w,
            int kernel_size,
            cudaStream_t cuda_stream) {
            const int input_h = static_cast<int>(input.size(0));
            const int input_w = static_cast<int>(input.size(1));
            const int channels = static_cast<int>(input.size(2));

            if (channels != 3) {
                LOG_ERROR("lanczos_resize: Only 3-channel (RGB) images supported, got {}", channels);
                return Tensor();
            }

            auto output = Tensor::empty(
                TensorShape({static_cast<size_t>(channels),
                             static_cast<size_t>(output_h),
                             static_cast<size_t>(output_w)}),
                Device::CUDA, DataType::Float32);

            cudaMemsetAsync(output.data_ptr(), 0, output.bytes(), cuda_stream);

            const uint32_t offset_step_x =
                (uint32_t)(kernel_size * (1.0 * input_w / output_w) * 2 + 1 + 0.5f);
            const uint32_t offset_step_y =
                (uint32_t)(kernel_size * (1.0 * input_h / output_h) * 2 + 1 + 0.5f);

            float *coef_x, *coef_y;
            cudaMalloc(&coef_x, sizeof(float) * output_w * offset_step_x);
            cudaMalloc(&coef_y, sizeof(float) * output_h * offset_step_y);

            const int threads = BLOCK_X * BLOCK_Y;
            detail::PreComputeCoef<<<(output_w + threads - 1) / threads, threads, 0, cuda_stream>>>(
                input_w, output_w, kernel_size, coef_x);
            detail::PreComputeCoef<<<(output_h + threads - 1) / threads, threads, 0, cuda_stream>>>(
                input_h, output_h, kernel_size, coef_y);

            const dim3 tile_grid((output_w + BLOCK_X - 1) / BLOCK_X,
                                 (output_h + BLOCK_Y - 1) / BLOCK_Y);
            const dim3 block(BLOCK_X, BLOCK_Y, 1);

            detail::LanczosResampleCUDA<NUM_CHANNELS, T>
                <<<tile_grid, block, 0, cuda_stream>>>(
                    input_h, input_w, output_h, output_w, kernel_size,
                    coef_x, coef_y,
                    input.ptr<T>(), output.ptr<float>());

            cudaFree(coef_x);
            cudaFree(coef_y);
            return output;
        }

    } // namespace detail

    Tensor lanczos_resize(
        const Tensor& input,
        int output_h,
        int output_w,
        int kernel_size,
        cudaStream_t cuda_stream) {

        if (!input.is_valid() || input.device() != Device::CUDA) {
            LOG_ERROR("lanczos_resize: Input must be a valid CUDA tensor");
            return Tensor();
        }
        if (input.ndim() != 3) {
            LOG_ERROR("lanczos_resize: Input must be 3D tensor [H, W, C]");
            return Tensor();
        }

        if (input.dtype() == DataType::UInt8) {
            return detail::lanczos_resize_impl<uint8_t>(
                input, output_h, output_w, kernel_size, cuda_stream);
        } else if (input.dtype() == DataType::Float32) {
            return detail::lanczos_resize_impl<float>(
                input, output_h, output_w, kernel_size, cuda_stream);
        } else {
            LOG_ERROR("lanczos_resize: Unsupported dtype: {}",
                      dtype_name(input.dtype()));
            return Tensor();
        }
    }

    Tensor lanczos_resize_grayscale(
        const Tensor& input,
        int output_h,
        int output_w,
        int kernel_size,
        cudaStream_t cuda_stream) {

        if (!input.is_valid() || input.device() != Device::CUDA) {
            LOG_ERROR("lanczos_resize_grayscale: Input must be a valid CUDA tensor");
            return Tensor();
        }

        if (input.dtype() != DataType::UInt8 && input.dtype() != DataType::Float32) {
            LOG_ERROR("lanczos_resize_grayscale: Input must be UInt8 or Float32 dtype");
            return Tensor();
        }

        if (input.ndim() != 2) {
            LOG_ERROR("lanczos_resize_grayscale: Input must be 2D tensor [H, W]");
            return Tensor();
        }

        const int input_h = static_cast<int>(input.size(0));
        const int input_w = static_cast<int>(input.size(1));

        auto output = Tensor::empty(
            TensorShape({static_cast<size_t>(output_h), static_cast<size_t>(output_w)}),
            Device::CUDA,
            DataType::Float32);

        cudaMemsetAsync(output.data_ptr(), 0, output.bytes(), cuda_stream);

        const uint32_t offset_step_x = (uint32_t)(kernel_size * (1.0 * input_w / output_w) * 2 + 1 + 0.5f);
        const uint32_t offset_step_y = (uint32_t)(kernel_size * (1.0 * input_h / output_h) * 2 + 1 + 0.5f);

        float* coef_x;
        float* coef_y;
        cudaMalloc(&coef_x, sizeof(float) * output_w * offset_step_x);
        cudaMalloc(&coef_y, sizeof(float) * output_h * offset_step_y);

        detail::PreComputeCoef<<<(output_w + BLOCK_X * BLOCK_Y - 1) / (BLOCK_X * BLOCK_Y), BLOCK_X * BLOCK_Y, 0, cuda_stream>>>(
            input_w, output_w, kernel_size, coef_x);

        detail::PreComputeCoef<<<(output_h + BLOCK_X * BLOCK_Y - 1) / (BLOCK_X * BLOCK_Y), BLOCK_X * BLOCK_Y, 0, cuda_stream>>>(
            input_h, output_h, kernel_size, coef_y);

        const dim3 tile_grid((output_w + BLOCK_X - 1) / BLOCK_X, (output_h + BLOCK_Y - 1) / BLOCK_Y);
        const dim3 block(BLOCK_X, BLOCK_Y, 1);

        if (input.dtype() == DataType::UInt8) {
            detail::LanczosResampleGrayscaleCUDA<uint8_t><<<tile_grid, block, 0, cuda_stream>>>(
                input_h, input_w,
                output_h, output_w,
                kernel_size,
                coef_x,
                coef_y,
                input.ptr<uint8_t>(),
                1.0f / 255.0f,
                output.ptr<float>());
        } else {
            detail::LanczosResampleGrayscaleCUDA<float><<<tile_grid, block, 0, cuda_stream>>>(
                input_h, input_w,
                output_h, output_w,
                kernel_size,
                coef_x,
                coef_y,
                input.ptr<float>(),
                1.0f,
                output.ptr<float>());
        }

        cudaFree(coef_x);
        cudaFree(coef_y);

        return output;
    }

    Tensor lanczos_resize_float_chw(
        const Tensor& input,
        int output_h,
        int output_w,
        int kernel_size,
        cudaStream_t cuda_stream) {

        if (!input.is_valid() || input.device() != Device::CUDA) {
            LOG_ERROR("lanczos_resize_float_chw: Input must be a valid CUDA tensor");
            return Tensor();
        }
        if (input.dtype() != DataType::Float32) {
            LOG_ERROR("lanczos_resize_float_chw: Input must be Float32 dtype");
            return Tensor();
        }
        if (input.ndim() != 3) {
            LOG_ERROR("lanczos_resize_float_chw: Input must be 3D tensor [C, H, W]");
            return Tensor();
        }

        const int channels = static_cast<int>(input.size(0));
        const int input_h = static_cast<int>(input.size(1));
        const int input_w = static_cast<int>(input.size(2));
        assert(channels == NUM_CHANNELS && "lanczos_resize_float_chw expects 3 channels");
        if (channels != NUM_CHANNELS) {
            LOG_ERROR("lanczos_resize_float_chw: Only 3-channel images supported, got {}", channels);
            return Tensor();
        }

        auto output = Tensor::empty(
            TensorShape({static_cast<size_t>(channels),
                         static_cast<size_t>(output_h),
                         static_cast<size_t>(output_w)}),
            Device::CUDA, DataType::Float32);

        cudaMemsetAsync(output.data_ptr(), 0, output.bytes(), cuda_stream);

        const uint32_t offset_step_x = (uint32_t)(kernel_size * (1.0 * input_w / output_w) * 2 + 1 + 0.5f);
        const uint32_t offset_step_y = (uint32_t)(kernel_size * (1.0 * input_h / output_h) * 2 + 1 + 0.5f);

        float* coef_x;
        float* coef_y;
        cudaMalloc(&coef_x, sizeof(float) * output_w * offset_step_x);
        cudaMalloc(&coef_y, sizeof(float) * output_h * offset_step_y);

        const int threads = BLOCK_X * BLOCK_Y;
        detail::PreComputeCoef<<<(output_w + threads - 1) / threads, threads, 0, cuda_stream>>>(
            input_w, output_w, kernel_size, coef_x);
        detail::PreComputeCoef<<<(output_h + threads - 1) / threads, threads, 0, cuda_stream>>>(
            input_h, output_h, kernel_size, coef_y);

        const dim3 tile_grid((output_w + BLOCK_X - 1) / BLOCK_X, (output_h + BLOCK_Y - 1) / BLOCK_Y);
        const dim3 block(BLOCK_X, BLOCK_Y, 1);

        detail::LanczosResamplePlanarCUDA<NUM_CHANNELS><<<tile_grid, block, 0, cuda_stream>>>(
            input_h, input_w, output_h, output_w, kernel_size,
            coef_x, coef_y,
            input.ptr<float>(), output.ptr<float>());

        cudaFree(coef_x);
        cudaFree(coef_y);

        return output;
    }

} // namespace lfs::core
