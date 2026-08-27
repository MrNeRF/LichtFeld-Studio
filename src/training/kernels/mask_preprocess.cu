/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mask_preprocess.hpp"

#include "core/cuda_error.hpp"
#include "kernel_stream.hpp"
#include "lfs/core/warp_reduce.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace lfs::training::kernels {
    namespace {

        constexpr int kBlock = 256;
        constexpr int kMaxReduceBlocks = 1024;

        [[nodiscard]] inline int num_blocks_1d(const int total) {
            return (total + kBlock - 1) / kBlock;
        }

        [[nodiscard]] inline int reduce_blocks(const int total) {
            return std::min(num_blocks_1d(total), kMaxReduceBlocks);
        }

        template <typename MaskT>
        __device__ __forceinline__ float load_mask_raw(const MaskT* mask, const int idx) {
            if constexpr (std::is_same_v<std::remove_cv_t<MaskT>, uint8_t>) {
                return static_cast<float>(mask[idx]);
            } else {
                return mask[idx];
            }
        }

        /// mask_as_float from trainer: UInt8/Bool → (v>0)?1:0; Float32 pass-through.
        template <typename MaskT>
        __device__ __forceinline__ float mask_as_float(const MaskT* mask, const int idx) {
            if constexpr (std::is_same_v<std::remove_cv_t<MaskT>, uint8_t>) {
                return mask[idx] != 0 ? 1.0f : 0.0f;
            } else {
                return mask[idx];
            }
        }

        template <typename MaskT>
        __device__ __forceinline__ float photo_weight(
            const MaskT* mask,
            const int idx,
            const MaskPhotoMode mode) {
            const float v = load_mask_raw(mask, idx);
            if (mode == MaskPhotoMode::SegmentAndIgnore) {
                // After trainer remap: >250 kept as 255 → weight 1; else 0.
                return v > 250.0f ? 1.0f : 0.0f;
            }
            // BinaryGt0 matches the reference weight composition:
            //   UInt8/Bool → (v != 0) as float; Float32 → pass-through.
            if constexpr (std::is_same_v<std::remove_cv_t<MaskT>, uint8_t>) {
                return mask[idx] != 0 ? 1.0f : 0.0f;
            } else {
                return v;
            }
        }

        /// Background weight for opacity penalty (pre-power).
        template <typename MaskT>
        __device__ __forceinline__ float opacity_bg(
            const MaskT* mask,
            const int idx,
            const MaskOpacityMode mode) {
            if (mode == MaskOpacityMode::SegmentAndIgnore) {
                const float v = load_mask_raw(mask, idx);
                // After trainer remaps: [128,250] → BG (penalty), else FG (no penalty).
                return (v >= 128.0f && v <= 250.0f) ? 1.0f : 0.0f;
            }
            return 1.0f - mask_as_float(mask, idx);
        }

        template <typename MaskT>
        __global__ void fuse_photometric_mask_weight_kernel(
            const MaskT* __restrict__ mask,
            const float* __restrict__ roi_weight,
            float* __restrict__ out,
            const int n,
            const MaskPhotoMode mode) {
            const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (idx >= n)
                return;
            float w = photo_weight(mask, idx, mode);
            if (roi_weight != nullptr) {
                w *= roi_weight[idx];
            }
            out[idx] = w;
        }

        template <typename MaskT>
        __global__ void fuse_mask_opacity_penalty_kernel(
            const float* __restrict__ alpha,
            const MaskT* __restrict__ mask,
            const float* __restrict__ roi_weight,
            float* __restrict__ grad_alpha,
            float* __restrict__ partial_sums,
            const int n,
            const float power,
            const float grad_scale,
            const MaskOpacityMode mode) {
            float local_sum = 0.0f;
            for (int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
                 idx < n;
                 idx += blockDim.x * gridDim.x) {
                float bg = opacity_bg(mask, idx, mode);
                // pow(0, p)=0, pow(1,p)=1; soft masks use powf.
                float pen = (bg <= 0.0f) ? 0.0f : ((bg >= 1.0f && power > 0.0f) ? 1.0f : powf(bg, power));
                if (roi_weight != nullptr) {
                    pen *= roi_weight[idx];
                }
                grad_alpha[idx] = pen * grad_scale;
                local_sum += alpha[idx] * pen;
            }
            local_sum = lfs::core::warp_ops::block_reduce_sum(local_sum);
            if (threadIdx.x == 0) {
                partial_sums[blockIdx.x] = local_sum;
            }
        }

        __global__ void final_mean_scale_reduce_kernel(
            const float* __restrict__ partial_sums,
            float* __restrict__ loss_out,
            const int num_blocks,
            const float scale,
            const int n) {
            float sum = 0.0f;
            for (int i = static_cast<int>(threadIdx.x); i < num_blocks; i += blockDim.x) {
                sum += partial_sums[i];
            }
            sum = lfs::core::warp_ops::block_reduce_sum(sum);
            if (threadIdx.x == 0) {
                loss_out[0] = scale * (sum / static_cast<float>(n));
            }
        }

        template <typename MaskT>
        __global__ void fuse_alpha_consistent_kernel(
            const float* __restrict__ alpha,
            const MaskT* __restrict__ mask,
            const float* __restrict__ roi_weight,
            float* __restrict__ grad_alpha,
            float* __restrict__ partial_sums,
            const int n,
            const float grad_scale) {
            float local_sum = 0.0f;
            for (int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
                 idx < n;
                 idx += blockDim.x * gridDim.x) {
                const float mf = mask_as_float(mask, idx);
                const float diff = alpha[idx] - mf;
                // Match Tensor::sign: +1 / -1 / 0
                const float s = (diff > 0.0f) ? 1.0f : ((diff < 0.0f) ? -1.0f : 0.0f);
                const float abs_d = fabsf(diff);
                float roi = 1.0f;
                if (roi_weight != nullptr) {
                    roi = roi_weight[idx];
                }
                grad_alpha[idx] = s * roi * grad_scale;
                local_sum += abs_d * roi;
            }
            local_sum = lfs::core::warp_ops::block_reduce_sum(local_sum);
            if (threadIdx.x == 0) {
                partial_sums[blockIdx.x] = local_sum;
            }
        }

        template <typename MaskT>
        void launch_photo(
            const MaskT* mask,
            const float* roi,
            float* out,
            const int H,
            const int W,
            const MaskPhotoMode mode,
            cudaStream_t stream) {
            stream = resolve_stream(stream);
            const int n = H * W;
            if (n <= 0)
                return;
            const int blocks = num_blocks_1d(n);
            fuse_photometric_mask_weight_kernel<MaskT><<<blocks, kBlock, 0, stream>>>(
                mask, roi, out, n, mode);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.mask_preprocess.photo_weight");
        }

        template <typename MaskT>
        void launch_opacity(
            const float* alpha,
            const MaskT* mask,
            const float* roi,
            float* grad_alpha,
            float* reduce_temp,
            float* loss_out,
            const int H,
            const int W,
            const float power,
            const float scale,
            const MaskOpacityMode mode,
            cudaStream_t stream) {
            stream = resolve_stream(stream);
            const int n = H * W;
            if (n <= 0) {
                return;
            }
            if (scale == 0.0f) {
                // Still zero grads for a clean steady state.
                cudaMemsetAsync(grad_alpha, 0, static_cast<size_t>(n) * sizeof(float), stream);
                cudaMemsetAsync(loss_out, 0, sizeof(float), stream);
                return;
            }
            const int blocks = reduce_blocks(n);
            const float grad_scale = scale / static_cast<float>(n);
            fuse_mask_opacity_penalty_kernel<MaskT><<<blocks, kBlock, 0, stream>>>(
                alpha, mask, roi, grad_alpha, reduce_temp, n, power, grad_scale, mode);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.mask_preprocess.opacity_penalty");
            final_mean_scale_reduce_kernel<<<1, kBlock, 0, stream>>>(
                reduce_temp, loss_out, blocks, scale, n);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.mask_preprocess.opacity_reduce");
        }

        template <typename MaskT>
        void launch_alpha_cons(
            const float* alpha,
            const MaskT* mask,
            const float* roi,
            float* grad_alpha,
            float* reduce_temp,
            float* loss_out,
            const int H,
            const int W,
            const float weight,
            cudaStream_t stream) {
            stream = resolve_stream(stream);
            const int n = H * W;
            if (n <= 0)
                return;
            const int blocks = reduce_blocks(n);
            const float grad_scale = weight / static_cast<float>(n);
            fuse_alpha_consistent_kernel<MaskT><<<blocks, kBlock, 0, stream>>>(
                alpha, mask, roi, grad_alpha, reduce_temp, n, grad_scale);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.mask_preprocess.alpha_consistent");
            final_mean_scale_reduce_kernel<<<1, kBlock, 0, stream>>>(
                reduce_temp, loss_out, blocks, weight, n);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.mask_preprocess.alpha_reduce");
        }

    } // namespace

    void launch_fuse_photometric_mask_weight_u8(
        const uint8_t* mask,
        const float* roi_weight,
        float* out,
        const int H,
        const int W,
        const MaskPhotoMode mode,
        cudaStream_t stream) {
        launch_photo(mask, roi_weight, out, H, W, mode, stream);
    }

    void launch_fuse_photometric_mask_weight_f32(
        const float* mask,
        const float* roi_weight,
        float* out,
        const int H,
        const int W,
        const MaskPhotoMode mode,
        cudaStream_t stream) {
        launch_photo(mask, roi_weight, out, H, W, mode, stream);
    }

    void launch_fuse_mask_opacity_penalty_u8(
        const float* alpha,
        const uint8_t* mask,
        const float* roi_weight,
        float* grad_alpha,
        float* reduce_temp,
        float* loss_out,
        const int H,
        const int W,
        const float power,
        const float scale,
        const MaskOpacityMode mode,
        cudaStream_t stream) {
        launch_opacity(alpha, mask, roi_weight, grad_alpha, reduce_temp, loss_out,
                       H, W, power, scale, mode, stream);
    }

    void launch_fuse_mask_opacity_penalty_f32(
        const float* alpha,
        const float* mask,
        const float* roi_weight,
        float* grad_alpha,
        float* reduce_temp,
        float* loss_out,
        const int H,
        const int W,
        const float power,
        const float scale,
        const MaskOpacityMode mode,
        cudaStream_t stream) {
        launch_opacity(alpha, mask, roi_weight, grad_alpha, reduce_temp, loss_out,
                       H, W, power, scale, mode, stream);
    }

    void launch_fuse_alpha_consistent_u8(
        const float* alpha,
        const uint8_t* mask,
        const float* roi_weight,
        float* grad_alpha,
        float* reduce_temp,
        float* loss_out,
        const int H,
        const int W,
        const float weight,
        cudaStream_t stream) {
        launch_alpha_cons(alpha, mask, roi_weight, grad_alpha, reduce_temp, loss_out,
                          H, W, weight, stream);
    }

    void launch_fuse_alpha_consistent_f32(
        const float* alpha,
        const float* mask,
        const float* roi_weight,
        float* grad_alpha,
        float* reduce_temp,
        float* loss_out,
        const int H,
        const int W,
        const float weight,
        cudaStream_t stream) {
        launch_alpha_cons(alpha, mask, roi_weight, grad_alpha, reduce_temp, loss_out,
                          H, W, weight, stream);
    }

} // namespace lfs::training::kernels
