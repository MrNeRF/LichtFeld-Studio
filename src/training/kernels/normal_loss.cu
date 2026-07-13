/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/core/warp_reduce.cuh"
#include "normal_loss.hpp"

#include <algorithm>

#include "kernel_stream.hpp"

namespace lfs::training::kernels {
    namespace {
        namespace slots = normal_loss_slots;

        constexpr int kThreadsPerBlock = 256;
        constexpr size_t kMaxBlocks = 1024;

        constexpr int kStatCount = 3;
        constexpr int kLossStatCount = 1;

        [[nodiscard]] size_t normal_loss_block_count(const size_t num_pixels) {
            return std::min((num_pixels + kThreadsPerBlock - 1) / kThreadsPerBlock, kMaxBlocks);
        }

        struct PixelSample {
            bool active;
            float alpha;
            float cos;
            float3 target_hat;
            float3 render_hat;
            float render_norm;
        };

        __device__ __forceinline__ PixelSample load_pixel(
            const float* __restrict__ rendered_normal,
            const float* __restrict__ rendered_alpha,
            const float* __restrict__ target_normal,
            const size_t idx,
            const size_t num_pixels) {
            PixelSample s = {};

            const float a = rendered_alpha[idx];
            const float tx = target_normal[idx];
            const float ty = target_normal[num_pixels + idx];
            const float tz = target_normal[2 * num_pixels + idx];
            const float nx = rendered_normal[idx];
            const float ny = rendered_normal[num_pixels + idx];
            const float nz = rendered_normal[2 * num_pixels + idx];

            if (!isfinite(a) || !isfinite(tx) || !isfinite(ty) || !isfinite(tz) ||
                !isfinite(nx) || !isfinite(ny) || !isfinite(nz) ||
                a <= kNormalLossMinAlpha) {
                return s;
            }

            const float t_norm = sqrtf(tx * tx + ty * ty + tz * tz);
            if (t_norm < kNormalLossMinPriorNorm) {
                return s;
            }
            const float n_norm = sqrtf(nx * nx + ny * ny + nz * nz);
            if (n_norm < kNormalLossMinRenderNorm) {
                return s;
            }

            const float t_rcp = 1.0f / t_norm;
            const float n_rcp = 1.0f / n_norm;
            s.target_hat = make_float3(tx * t_rcp, ty * t_rcp, tz * t_rcp);
            s.render_hat = make_float3(nx * n_rcp, ny * n_rcp, nz * n_rcp);
            s.cos = s.render_hat.x * s.target_hat.x +
                    s.render_hat.y * s.target_hat.y +
                    s.render_hat.z * s.target_hat.z;
            s.alpha = a;
            s.render_norm = n_norm;
            s.active = true;
            return s;
        }

        // Sequential block reductions share warp_reduce's static shared buffer;
        // the leading __syncthreads keeps consecutive reductions from racing.
        template <int N>
        __device__ void reduce_and_store(
            double (&vals)[N],
            double* __restrict__ out,
            const int num_blocks) {
#pragma unroll
            for (int i = 0; i < N; ++i) {
                __syncthreads();
                const double reduced = lfs::core::warp_ops::block_reduce_sum(vals[i]);
                if (threadIdx.x == 0) {
                    out[i * num_blocks + blockIdx.x] = reduced;
                }
            }
        }

        template <int N>
        __device__ void reduce_block_partials(
            const double* __restrict__ block_partials,
            double (&sums)[N],
            const int num_blocks) {
            for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
#pragma unroll
                for (int k = 0; k < N; ++k) {
                    sums[k] += block_partials[k * num_blocks + i];
                }
            }
#pragma unroll
            for (int k = 0; k < N; ++k) {
                __syncthreads();
                sums[k] = lfs::core::warp_ops::block_reduce_sum(sums[k]);
            }
        }

        __global__ void normal_loss_stats_kernel(
            const float* __restrict__ rendered_normal,
            const float* __restrict__ rendered_alpha,
            const float* __restrict__ target_normal,
            double* __restrict__ block_partials,
            const size_t num_pixels,
            const int num_blocks) {

            double sums[kStatCount] = {};
            for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                 idx < num_pixels;
                 idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
                const PixelSample s = load_pixel(rendered_normal, rendered_alpha, target_normal, idx, num_pixels);
                if (!s.active) {
                    continue;
                }
                sums[0] += s.alpha;
                sums[1] += 1.0;
                sums[2] += static_cast<double>(s.alpha) * s.cos;
            }
            reduce_and_store(sums, block_partials, num_blocks);
        }

        __global__ void normal_loss_finalize_kernel(
            const double* __restrict__ block_partials,
            float* __restrict__ finals,
            const int num_blocks,
            const float weight) {

            double sums[kStatCount] = {};
            reduce_block_partials(block_partials, sums, num_blocks);
            if (threadIdx.x != 0) {
                return;
            }

            const double sum_alpha = sums[0];
            const double count = sums[1];
            const bool valid = count >= kNormalLossMinValidCount &&
                               sum_alpha >= kNormalLossMinValidWeight;

            finals[slots::kValid] = valid ? 1.0f : 0.0f;
            finals[slots::kSumAlpha] = static_cast<float>(sum_alpha);
            finals[slots::kCount] = static_cast<float>(count);
            finals[slots::kMeanCos] = sum_alpha > 0.0 ? static_cast<float>(sums[2] / sum_alpha) : 0.0f;
            finals[slots::kInvNorm] = valid ? static_cast<float>(weight / fmax(sum_alpha, 1.0)) : 0.0f;
        }

        __global__ void normal_loss_grad_kernel(
            const float* __restrict__ rendered_normal,
            const float* __restrict__ rendered_alpha,
            const float* __restrict__ target_normal,
            float* __restrict__ grad_normal,
            const float* __restrict__ finals,
            double* __restrict__ block_partials,
            const size_t num_pixels,
            const int num_blocks) {

            const float inv_norm = finals[slots::kInvNorm];

            double sums[kLossStatCount] = {};
            for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                 idx < num_pixels;
                 idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
                const PixelSample s = load_pixel(rendered_normal, rendered_alpha, target_normal, idx, num_pixels);
                if (!s.active || inv_norm == 0.0f) {
                    grad_normal[idx] = 0.0f;
                    grad_normal[num_pixels + idx] = 0.0f;
                    grad_normal[2 * num_pixels + idx] = 0.0f;
                    continue;
                }

                sums[0] += static_cast<double>(s.alpha) * (1.0 - s.cos);

                // d(1 - cos)/dn = -(t_hat - cos * n_hat) / |n|
                const float scale = -inv_norm * s.alpha / s.render_norm;
                grad_normal[idx] = scale * (s.target_hat.x - s.cos * s.render_hat.x);
                grad_normal[num_pixels + idx] = scale * (s.target_hat.y - s.cos * s.render_hat.y);
                grad_normal[2 * num_pixels + idx] = scale * (s.target_hat.z - s.cos * s.render_hat.z);
            }
            reduce_and_store(sums, block_partials, num_blocks);
        }

        __global__ void normal_loss_finalize_loss_kernel(
            const double* __restrict__ block_partials,
            const float* __restrict__ finals,
            float* __restrict__ loss_out,
            const int num_blocks) {

            double sums[kLossStatCount] = {};
            reduce_block_partials(block_partials, sums, num_blocks);
            if (threadIdx.x != 0) {
                return;
            }
            loss_out[0] = finals[slots::kInvNorm] * static_cast<float>(sums[0]);
        }
    } // namespace

    size_t normal_loss_partial_count(const size_t num_pixels) {
        const size_t num_blocks = normal_loss_block_count(num_pixels);
        return static_cast<size_t>(slots::kSlotCount) +
               2 * static_cast<size_t>(kStatCount) * num_blocks;
    }

    void launch_normal_loss(
        const float* rendered_normal,
        const float* rendered_alpha,
        const float* target_normal,
        float* grad_normal,
        float* loss_out,
        float* partial_sums,
        const int width,
        const int height,
        const float weight,
        cudaStream_t stream) {
        stream = resolve_stream(stream);

        const size_t num_pixels = static_cast<size_t>(width) * height;
        const int num_blocks = static_cast<int>(normal_loss_block_count(num_pixels));
        float* finals = partial_sums;
        double* block_partials = reinterpret_cast<double*>(partial_sums + slots::kSlotCount);

        normal_loss_stats_kernel<<<num_blocks, kThreadsPerBlock, 0, stream>>>(
            rendered_normal,
            rendered_alpha,
            target_normal,
            block_partials,
            num_pixels,
            num_blocks);
        normal_loss_finalize_kernel<<<1, kThreadsPerBlock, 0, stream>>>(
            block_partials,
            finals,
            num_blocks,
            weight);
        normal_loss_grad_kernel<<<num_blocks, kThreadsPerBlock, 0, stream>>>(
            rendered_normal,
            rendered_alpha,
            target_normal,
            grad_normal,
            finals,
            block_partials,
            num_pixels,
            num_blocks);
        normal_loss_finalize_loss_kernel<<<1, kThreadsPerBlock, 0, stream>>>(
            block_partials,
            finals,
            loss_out,
            num_blocks);
    }

} // namespace lfs::training::kernels
