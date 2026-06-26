/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "depth_loss.hpp"
#include "lfs/core/warp_reduce.cuh"

#include <algorithm>
#include <cmath>

#include "kernel_stream.hpp"

namespace lfs::training::kernels {
    namespace {
        constexpr int kThreadsPerBlock = 256;

        constexpr float kMinAlphaForDepthLoss = 1.0e-3f;

        constexpr float kMinDepthVariance = 1.0e-12f;
        constexpr float kMinWarpScale = 1.0e-4f;
        constexpr int kStatsSlots = 6;
        constexpr int kLossScratchOffset = 8;

        [[nodiscard]] size_t depth_loss_block_count(const size_t num_pixels) {
            return std::min((num_pixels + kThreadsPerBlock - 1) / kThreadsPerBlock, size_t(1024));
        }

        __device__ __forceinline__ float clamp_unit_depth(const float value) {
            return fminf(fmaxf(value, 0.0f), 1.0f);
        }

        __device__ __forceinline__ float adaptive_warp_scale_from_means(
            const float mean_pred,
            const float mean_target_oriented) {
            const float target = fminf(fmaxf(mean_target_oriented, 1.0e-3f), 1.0f - 1.0e-3f);
            const float pred = fmaxf(mean_pred, kMinWarpScale);
            if (target < 0.5f) {
                return fmaxf(pred / fmaxf(2.0f * target, 1.0e-3f), kMinWarpScale);
            }
            return fmaxf(2.0f * pred * fmaxf(1.0f - target, 1.0e-3f), kMinWarpScale);
        }

        __device__ __forceinline__ float adaptive_warp_depth(
            const float depth,
            const float scale,
            float& derivative) {
            const float z = fmaxf(depth, 0.0f);
            const float s = fmaxf(scale, kMinWarpScale);
            if (z < s) {
                derivative = 0.5f / s;
                return 0.5f * z / s;
            }
            const float z_safe = fmaxf(z, kMinWarpScale);
            derivative = 0.5f * s / (z_safe * z_safe);
            return 1.0f - 0.5f * s / z_safe;
        }

        __global__ void depth_loss_stats_kernel(
            const float* __restrict__ rendered_depth_accum,
            const float* __restrict__ rendered_alpha_accum,
            const float* __restrict__ target_depth,
            float* __restrict__ partial_sums,
            const size_t num_pixels,
            const int num_blocks) {

            float local_pred_sum = 0.0f;
            float local_target_sum = 0.0f;
            float local_pred_sq_sum = 0.0f;
            float local_target_sq_sum = 0.0f;
            float local_cross_sum = 0.0f;
            float local_count = 0.0f;

            for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                 idx < num_pixels;
                 idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
                const float target_raw = target_depth[idx];
                const float depth_accum_raw = rendered_depth_accum[idx];
                const float alpha_accum = rendered_alpha_accum[idx];
                const float depth_accum = fmaxf(depth_accum_raw, 0.0f);
                const bool active = target_raw > 0.0f &&
                                    alpha_accum > kMinAlphaForDepthLoss &&
                                    isfinite(target_raw) &&
                                    isfinite(depth_accum_raw) &&
                                    isfinite(alpha_accum);

                if (active) {
                    const float pred = depth_accum;
                    const float target = target_raw;
                    local_pred_sum += pred;
                    local_target_sum += target;
                    local_pred_sq_sum += pred * pred;
                    local_target_sq_sum += target * target;
                    local_cross_sum += pred * target;
                    local_count += 1.0f;
                }
            }

            local_pred_sum = lfs::core::warp_ops::block_reduce_sum(local_pred_sum);
            local_target_sum = lfs::core::warp_ops::block_reduce_sum(local_target_sum);
            local_pred_sq_sum = lfs::core::warp_ops::block_reduce_sum(local_pred_sq_sum);
            local_target_sq_sum = lfs::core::warp_ops::block_reduce_sum(local_target_sq_sum);
            local_cross_sum = lfs::core::warp_ops::block_reduce_sum(local_cross_sum);
            local_count = lfs::core::warp_ops::block_reduce_sum(local_count);
            if (threadIdx.x == 0) {
                partial_sums[blockIdx.x] = local_pred_sum;
                partial_sums[num_blocks + blockIdx.x] = local_target_sum;
                partial_sums[2 * num_blocks + blockIdx.x] = local_pred_sq_sum;
                partial_sums[3 * num_blocks + blockIdx.x] = local_target_sq_sum;
                partial_sums[4 * num_blocks + blockIdx.x] = local_cross_sum;
                partial_sums[5 * num_blocks + blockIdx.x] = local_count;
            }
        }

        __global__ void depth_loss_finalize_kernel(
            float* __restrict__ partial_sums,
            float* __restrict__ loss_out,
            const int num_blocks,
            const float weight,
            const DepthLossMode mode) {

            float pred_sum = 0.0f;
            float target_sum = 0.0f;
            float pred_sq_sum = 0.0f;
            float target_sq_sum = 0.0f;
            float cross_sum = 0.0f;
            float count = 0.0f;
            for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
                pred_sum += partial_sums[i];
                target_sum += partial_sums[num_blocks + i];
                pred_sq_sum += partial_sums[2 * num_blocks + i];
                target_sq_sum += partial_sums[3 * num_blocks + i];
                cross_sum += partial_sums[4 * num_blocks + i];
                count += partial_sums[5 * num_blocks + i];
            }

            pred_sum = lfs::core::warp_ops::block_reduce_sum(pred_sum);
            target_sum = lfs::core::warp_ops::block_reduce_sum(target_sum);
            pred_sq_sum = lfs::core::warp_ops::block_reduce_sum(pred_sq_sum);
            target_sq_sum = lfs::core::warp_ops::block_reduce_sum(target_sq_sum);
            cross_sum = lfs::core::warp_ops::block_reduce_sum(cross_sum);
            count = lfs::core::warp_ops::block_reduce_sum(count);
            if (threadIdx.x == 0) {
                float mean_pred = 0.0f;
                float mean_target = 0.0f;
                float corr = 0.0f;
                float inv_norm_prod = 0.0f;
                float inv_pred_var = 0.0f;
                float corr_sign = 0.0f;
                float adaptive_warp_scale = 0.0f;
                if (count > 1.0f) {
                    const float inv_count = 1.0f / count;
                    mean_pred = pred_sum * inv_count;
                    mean_target = target_sum * inv_count;
                    const float pred_var = fmaxf(pred_sq_sum - pred_sum * mean_pred, 0.0f);
                    const float target_var = fmaxf(target_sq_sum - target_sum * mean_target, 0.0f);
                    const float cross = cross_sum - pred_sum * mean_target;
                    if (pred_var > kMinDepthVariance && target_var > kMinDepthVariance) {
                        inv_norm_prod = rsqrtf(pred_var * target_var);
                        inv_pred_var = 1.0f / pred_var;
                        corr = fmaxf(-1.0f, fminf(1.0f, cross * inv_norm_prod));
                        corr_sign = corr >= 0.0f ? 1.0f : -1.0f;
                    }
                }

                if (corr_sign != 0.0f && mode == DepthLossMode::AdaptiveWarpedL1) {
                    const float mean_target_oriented = corr_sign >= 0.0f ? mean_target : 1.0f - mean_target;
                    adaptive_warp_scale = adaptive_warp_scale_from_means(mean_pred, mean_target_oriented);
                }

                loss_out[0] = corr_sign != 0.0f && mode == DepthLossMode::PearsonAbs
                                  ? weight * (1.0f - fabsf(corr))
                                  : 0.0f;
                partial_sums[0] = count;
                partial_sums[1] = mean_pred;
                partial_sums[2] = mean_target;
                partial_sums[3] = mode == DepthLossMode::AdaptiveWarpedL1 ? adaptive_warp_scale : inv_norm_prod;
                partial_sums[4] = corr;
                partial_sums[5] = inv_pred_var;
                partial_sums[6] = corr_sign;
            }
        }

        __global__ void depth_loss_grad_kernel(
            const float* __restrict__ rendered_depth_accum,
            const float* __restrict__ rendered_alpha_accum,
            const float* __restrict__ target_depth,
            float* __restrict__ grad_depth,
            float* __restrict__ partial_sums,
            const size_t num_pixels,
            const float weight,
            const DepthLossMode mode,
            const int num_blocks) {

            const float count = partial_sums[0];
            const float mean_pred = partial_sums[1];
            const float mean_target = partial_sums[2];
            const float inv_norm_prod_or_adaptive_scale = partial_sums[3];
            const float corr = partial_sums[4];
            const float inv_pred_var = partial_sums[5];
            const float corr_sign = partial_sums[6];
            float local_adaptive_l1_loss = 0.0f;
            for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                 idx < num_pixels;
                 idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
                const float target_raw = target_depth[idx];
                const float depth_accum_raw = rendered_depth_accum[idx];
                const float depth_accum = fmaxf(depth_accum_raw, 0.0f);
                const bool active = target_raw > 0.0f &&
                                    rendered_alpha_accum[idx] > kMinAlphaForDepthLoss &&
                                    corr_sign != 0.0f &&
                                    isfinite(target_raw) &&
                                    isfinite(depth_accum_raw) &&
                                    isfinite(rendered_alpha_accum[idx]);

                float depth_grad = 0.0f;
                if (active) {
                    const float pred = depth_accum;
                    if (mode == DepthLossMode::AdaptiveWarpedL1) {
                        float warp_derivative = 0.0f;
                        const float pred_warped = adaptive_warp_depth(pred, inv_norm_prod_or_adaptive_scale, warp_derivative);
                        const float target_oriented = corr_sign >= 0.0f
                                                          ? clamp_unit_depth(target_raw)
                                                          : 1.0f - clamp_unit_depth(target_raw);
                        const float diff = pred_warped - target_oriented;
                        local_adaptive_l1_loss += fabsf(diff);
                        if (depth_accum_raw >= 0.0f && count > 1.0f) {
                            const float sign = diff > 0.0f ? 1.0f : (diff < 0.0f ? -1.0f : 0.0f);
                            depth_grad = weight * sign * warp_derivative / count;
                        }
                    } else {
                        const float centered_pred = pred - mean_pred;
                        const float centered_target = target_raw - mean_target;
                        const float d_corr_d_pred =
                            centered_target * inv_norm_prod_or_adaptive_scale -
                            corr * centered_pred * inv_pred_var;
                        const float d_loss_d_pred = -corr_sign * weight * d_corr_d_pred;
                        depth_grad = depth_accum_raw >= 0.0f ? d_loss_d_pred : 0.0f;
                    }
                }
                grad_depth[idx] = depth_grad;
            }

            if (mode == DepthLossMode::AdaptiveWarpedL1) {
                local_adaptive_l1_loss = lfs::core::warp_ops::block_reduce_sum(local_adaptive_l1_loss);
                if (threadIdx.x == 0) {
                    partial_sums[kLossScratchOffset + blockIdx.x] = local_adaptive_l1_loss;
                }
            } else if (threadIdx.x == 0 && blockIdx.x < num_blocks) {
                partial_sums[kLossScratchOffset + blockIdx.x] = 0.0f;
            }
        }

        __global__ void depth_loss_adaptive_l1_finalize_kernel(
            const float* __restrict__ partial_sums,
            float* __restrict__ loss_out,
            const int num_blocks,
            const float weight) {

            float loss_sum = 0.0f;
            for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
                loss_sum += partial_sums[kLossScratchOffset + i];
            }
            loss_sum = lfs::core::warp_ops::block_reduce_sum(loss_sum);
            if (threadIdx.x == 0) {
                const float count = partial_sums[0];
                loss_out[0] = count > 1.0f ? weight * loss_sum / count : 0.0f;
            }
        }
    } // namespace

    size_t depth_loss_partial_count(const size_t num_pixels) {
        const size_t num_blocks = depth_loss_block_count(num_pixels);
        return std::max(num_blocks * size_t(kStatsSlots), size_t(kLossScratchOffset) + num_blocks);
    }

    void launch_depth_loss(
        const float* rendered_depth_accum,
        const float* rendered_alpha_accum,
        const float* target_depth,
        float* grad_depth,
        float* loss_out,
        float* partial_sums,
        const size_t num_pixels,
        const float weight,
        const DepthLossMode mode,
        cudaStream_t stream) {
        stream = resolve_stream(stream);

        const int num_blocks = static_cast<int>(depth_loss_block_count(num_pixels));

        depth_loss_stats_kernel<<<num_blocks, kThreadsPerBlock, 0, stream>>>(
            rendered_depth_accum,
            rendered_alpha_accum,
            target_depth,
            partial_sums,
            num_pixels,
            num_blocks);
        depth_loss_finalize_kernel<<<1, kThreadsPerBlock, 0, stream>>>(
            partial_sums,
            loss_out,
            num_blocks,
            weight,
            mode);
        depth_loss_grad_kernel<<<num_blocks, kThreadsPerBlock, 0, stream>>>(
            rendered_depth_accum,
            rendered_alpha_accum,
            target_depth,
            grad_depth,
            partial_sums,
            num_pixels,
            weight,
            mode,
            num_blocks);
        if (mode == DepthLossMode::AdaptiveWarpedL1) {
            depth_loss_adaptive_l1_finalize_kernel<<<1, kThreadsPerBlock, 0, stream>>>(
                partial_sums,
                loss_out,
                num_blocks,
                weight);
        }
    }

} // namespace lfs::training::kernels
