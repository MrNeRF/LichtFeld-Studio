/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/core/warp_reduce.cuh"
#include "lfs/kernels/ssim.cuh"
#include "lfs/kernels/ssim_reduction.cuh"
#include <algorithm>
#include <type_traits>

namespace lfs::training::kernels {

    namespace {
        constexpr int REDUCTION_BLOCK_SIZE = 256;
        constexpr float UINT8_TO_FLOAT = 1.0f / 255.0f;

        template <typename T>
        __device__ __forceinline__ float target_value(const T* data, size_t idx) {
            if constexpr (std::is_same_v<std::remove_cv_t<T>, uint8_t>) {
                return static_cast<float>(data[idx]) * UINT8_TO_FLOAT;
            } else {
                return static_cast<float>(data[idx]);
            }
        }

        template <typename T>
        __device__ __forceinline__ float mask_value(const T* data, size_t idx) {
            if constexpr (std::is_same_v<std::remove_cv_t<T>, uint8_t>) {
                return data[idx] != 0 ? 1.0f : 0.0f;
            } else {
                return static_cast<float>(data[idx]);
            }
        }

        inline int reduction_num_blocks(size_t total_items) {
            return static_cast<int>(std::min<size_t>(
                (total_items + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE,
                size_t{1024}));
        }

        __global__ void fused_ssim_mean_kernel(
            const float* __restrict__ ssim_map,
            float* __restrict__ partial_sums,
            int N, int C, int H, int W,
            bool apply_valid_padding) {

            float local_sum = 0.0f;

            const int h_start = apply_valid_padding && H > 10 ? 5 : 0;
            const int h_end = apply_valid_padding && H > 10 ? H - 5 : H;
            const int w_start = apply_valid_padding && W > 10 ? 5 : 0;
            const int w_end = apply_valid_padding && W > 10 ? W - 5 : W;

            for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                 idx < static_cast<size_t>(N) * C * H * W;
                 idx += blockDim.x * gridDim.x) {
                size_t rem = idx % (static_cast<size_t>(H) * W);
                const int h = static_cast<int>(rem / W);
                const int w = static_cast<int>(rem % W);

                if (h >= h_start && h < h_end && w >= w_start && w < w_end) {
                    local_sum += ssim_map[idx];
                }
            }

            local_sum = lfs::core::warp_ops::block_reduce_sum(local_sum);
            if (threadIdx.x == 0) {
                partial_sums[blockIdx.x] = local_sum;
            }
        }

        template <typename TargetT>
        __global__ void fused_l1_ssim_sum_kernel(
            const float* __restrict__ img1,
            const TargetT* __restrict__ img2,
            const float* __restrict__ ssim_map,
            float* __restrict__ partial_sums,
            float ssim_weight,
            int N, int C, int H, int W,
            bool apply_valid_padding) {

            float local_sum = 0.0f;
            const float l1_weight = 1.0f - ssim_weight;

            const int h_start = apply_valid_padding && H > 10 ? 5 : 0;
            const int h_end = apply_valid_padding && H > 10 ? H - 5 : H;
            const int w_start = apply_valid_padding && W > 10 ? 5 : 0;
            const int w_end = apply_valid_padding && W > 10 ? W - 5 : W;

            for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                 idx < static_cast<size_t>(N) * C * H * W;
                 idx += blockDim.x * gridDim.x) {
                size_t rem = idx % (static_cast<size_t>(H) * W);
                const int h = static_cast<int>(rem / W);
                const int w = static_cast<int>(rem % W);

                if (h >= h_start && h < h_end && w >= w_start && w < w_end) {
                    const float l1 = fabsf(img1[idx] - target_value(img2, idx));
                    const size_t batch_idx = idx / (static_cast<size_t>(C) * H * W);
                    const float ssim = ssim_map[batch_idx * H * W + rem];
                    local_sum += l1_weight * l1 + ssim_weight * (1.0f - ssim);
                }
            }

            local_sum = lfs::core::warp_ops::block_reduce_sum(local_sum);
            if (threadIdx.x == 0) {
                partial_sums[blockIdx.x] = local_sum;
            }
        }

        template <typename TargetT, typename MaskT>
        __global__ void masked_fused_l1_ssim_sum_kernel(
            const float* __restrict__ img1,
            const TargetT* __restrict__ img2,
            const float* __restrict__ ssim_map,
            const MaskT* __restrict__ mask,
            float* __restrict__ partial_sums,
            float ssim_weight,
            int N, int C, int H, int W) {

            float local_sum = 0.0f;
            const float l1_weight = 1.0f - ssim_weight;

            for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                 idx < static_cast<size_t>(N) * C * H * W;
                 idx += blockDim.x * gridDim.x) {
                size_t rem = idx % (static_cast<size_t>(H) * W);
                const int h = static_cast<int>(rem / W);
                const int w = static_cast<int>(rem % W);
                const float mask_weight = mask_value(mask, h * W + w);
                const float l1 = fabsf(img1[idx] - target_value(img2, idx));
                const size_t batch_idx = idx / (static_cast<size_t>(C) * H * W);
                const float ssim = ssim_map[batch_idx * H * W + rem];
                const float loss = l1_weight * l1 + ssim_weight * (1.0f - ssim);
                local_sum += loss * mask_weight;
            }

            local_sum = lfs::core::warp_ops::block_reduce_sum(local_sum);
            if (threadIdx.x == 0) {
                partial_sums[blockIdx.x] = local_sum;
            }
        }

        template <typename MaskT>
        __global__ void mask_sum_kernel(
            const MaskT* __restrict__ mask,
            float* __restrict__ partial_sums,
            int H, int W) {

            float local_sum = 0.0f;
            const size_t total = static_cast<size_t>(H) * W;

            for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                 idx < total;
                 idx += blockDim.x * gridDim.x) {
                local_sum += mask_value(mask, idx);
            }

            local_sum = lfs::core::warp_ops::block_reduce_sum(local_sum);
            if (threadIdx.x == 0) {
                partial_sums[blockIdx.x] = local_sum;
            }
        }

        __global__ void final_mean_reduce_kernel(
            const float* __restrict__ partial_sums,
            float* __restrict__ result,
            int num_blocks,
            size_t total_valid_pixels) {

            float sum = 0.0f;
            for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
                sum += partial_sums[i];
            }

            sum = lfs::core::warp_ops::block_reduce_sum(sum);
            if (threadIdx.x == 0) {
                result[0] = sum / static_cast<float>(total_valid_pixels);
            }
        }

        __global__ void final_masked_mean_reduce_kernel(
            const float* __restrict__ partial_loss_sums,
            const float* __restrict__ partial_mask_sums,
            float* __restrict__ loss_result,
            float* __restrict__ mask_sum_result,
            int loss_num_blocks,
            int mask_num_blocks,
            int C) {

            float loss_sum = 0.0f;
            for (int i = threadIdx.x; i < loss_num_blocks; i += blockDim.x) {
                loss_sum += partial_loss_sums[i];
            }

            float mask_sum = 0.0f;
            for (int i = threadIdx.x; i < mask_num_blocks; i += blockDim.x) {
                mask_sum += partial_mask_sums[i];
            }

            loss_sum = lfs::core::warp_ops::block_reduce_sum(loss_sum);
            mask_sum = lfs::core::warp_ops::block_reduce_sum(mask_sum);
            if (threadIdx.x == 0) {
                const float normalized_mask_sum = mask_sum * static_cast<float>(C) + SSIM_EPSILON;
                loss_result[0] = loss_sum / normalized_mask_sum;
                mask_sum_result[0] = normalized_mask_sum;
            }
        }
    } // namespace

    void launch_fused_ssim_mean_device(
        const float* ssim_map,
        float* temp_buffer,
        float* result_buffer,
        int N, int C, int H, int W,
        bool apply_valid_padding,
        cudaStream_t stream) {

        const size_t total_pixels = static_cast<size_t>(N) * C * H * W;
        const int num_blocks = reduction_num_blocks(total_pixels);

        // Compute number of valid pixels for normalization
        const int h_valid = apply_valid_padding && H > 10 ? H - 10 : H;
        const int w_valid = apply_valid_padding && W > 10 ? W - 10 : W;
        const size_t total_valid_pixels = static_cast<size_t>(N) * C * h_valid * w_valid;

        fused_ssim_mean_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(
            ssim_map, temp_buffer, N, C, H, W, apply_valid_padding);

        final_mean_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(
            temp_buffer, result_buffer, num_blocks, total_valid_pixels);
    }

    template <typename TargetT>
    void launch_fused_l1_ssim_mean_device_impl(
        const float* img1,
        const TargetT* img2,
        const float* ssim_map,
        float ssim_weight,
        float* temp_buffer,
        float* result_buffer,
        int N, int C, int H, int W,
        bool apply_valid_padding,
        cudaStream_t stream) {

        const size_t total_pixels = static_cast<size_t>(N) * C * H * W;
        const int num_blocks = reduction_num_blocks(total_pixels);

        const int h_valid = apply_valid_padding && H > 10 ? H - 10 : H;
        const int w_valid = apply_valid_padding && W > 10 ? W - 10 : W;
        const size_t total_valid_pixels = static_cast<size_t>(N) * C * h_valid * w_valid;

        fused_l1_ssim_sum_kernel<TargetT><<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(
            img1, img2, ssim_map, temp_buffer, ssim_weight, N, C, H, W, apply_valid_padding);

        final_mean_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(
            temp_buffer, result_buffer, num_blocks, total_valid_pixels);
    }

    void launch_fused_l1_ssim_mean_device(
        const float* img1,
        const float* img2,
        const float* ssim_map,
        float ssim_weight,
        float* temp_buffer,
        float* result_buffer,
        int N, int C, int H, int W,
        bool apply_valid_padding,
        cudaStream_t stream) {
        launch_fused_l1_ssim_mean_device_impl(
            img1, img2, ssim_map, ssim_weight, temp_buffer, result_buffer,
            N, C, H, W, apply_valid_padding, stream);
    }

    void launch_fused_l1_ssim_mean_device(
        const float* img1,
        const uint8_t* img2,
        const float* ssim_map,
        float ssim_weight,
        float* temp_buffer,
        float* result_buffer,
        int N, int C, int H, int W,
        bool apply_valid_padding,
        cudaStream_t stream) {
        launch_fused_l1_ssim_mean_device_impl(
            img1, img2, ssim_map, ssim_weight, temp_buffer, result_buffer,
            N, C, H, W, apply_valid_padding, stream);
    }

    template <typename TargetT, typename MaskT>
    void launch_masked_fused_l1_ssim_mean_device_impl(
        const float* img1,
        const TargetT* img2,
        const float* ssim_map,
        const MaskT* mask,
        float ssim_weight,
        float* temp_buffer,
        float* loss_buffer,
        float* mask_sum_buffer,
        int N, int C, int H, int W,
        cudaStream_t stream) {

        constexpr int MAX_REDUCTION_BLOCKS = 1024;
        float* loss_temp_buffer = temp_buffer;
        float* mask_temp_buffer = temp_buffer + MAX_REDUCTION_BLOCKS;

        const size_t total_loss_pixels = static_cast<size_t>(N) * C * H * W;
        const int loss_num_blocks = reduction_num_blocks(total_loss_pixels);
        masked_fused_l1_ssim_sum_kernel<TargetT, MaskT><<<loss_num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(
            img1, img2, ssim_map, mask, loss_temp_buffer, ssim_weight, N, C, H, W);

        const size_t total_mask_pixels = static_cast<size_t>(H) * W;
        const int mask_num_blocks = reduction_num_blocks(total_mask_pixels);
        mask_sum_kernel<MaskT><<<mask_num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(
            mask, mask_temp_buffer, H, W);

        final_masked_mean_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(
            loss_temp_buffer,
            mask_temp_buffer,
            loss_buffer,
            mask_sum_buffer,
            loss_num_blocks,
            mask_num_blocks,
            C);
    }

    void launch_masked_fused_l1_ssim_mean_device(
        const float* img1,
        const float* img2,
        const float* ssim_map,
        const float* mask,
        float ssim_weight,
        float* temp_buffer,
        float* loss_buffer,
        float* mask_sum_buffer,
        int N, int C, int H, int W,
        cudaStream_t stream) {
        launch_masked_fused_l1_ssim_mean_device_impl(
            img1, img2, ssim_map, mask, ssim_weight, temp_buffer, loss_buffer,
            mask_sum_buffer, N, C, H, W, stream);
    }

    void launch_masked_fused_l1_ssim_mean_device(
        const float* img1,
        const float* img2,
        const float* ssim_map,
        const uint8_t* mask,
        float ssim_weight,
        float* temp_buffer,
        float* loss_buffer,
        float* mask_sum_buffer,
        int N, int C, int H, int W,
        cudaStream_t stream) {
        launch_masked_fused_l1_ssim_mean_device_impl(
            img1, img2, ssim_map, mask, ssim_weight, temp_buffer, loss_buffer,
            mask_sum_buffer, N, C, H, W, stream);
    }

    void launch_masked_fused_l1_ssim_mean_device(
        const float* img1,
        const uint8_t* img2,
        const float* ssim_map,
        const float* mask,
        float ssim_weight,
        float* temp_buffer,
        float* loss_buffer,
        float* mask_sum_buffer,
        int N, int C, int H, int W,
        cudaStream_t stream) {
        launch_masked_fused_l1_ssim_mean_device_impl(
            img1, img2, ssim_map, mask, ssim_weight, temp_buffer, loss_buffer,
            mask_sum_buffer, N, C, H, W, stream);
    }

    void launch_masked_fused_l1_ssim_mean_device(
        const float* img1,
        const uint8_t* img2,
        const float* ssim_map,
        const uint8_t* mask,
        float ssim_weight,
        float* temp_buffer,
        float* loss_buffer,
        float* mask_sum_buffer,
        int N, int C, int H, int W,
        cudaStream_t stream) {
        launch_masked_fused_l1_ssim_mean_device_impl(
            img1, img2, ssim_map, mask, ssim_weight, temp_buffer, loss_buffer,
            mask_sum_buffer, N, C, H, W, stream);
    }

} // namespace lfs::training::kernels
