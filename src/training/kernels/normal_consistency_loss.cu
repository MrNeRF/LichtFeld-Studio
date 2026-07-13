/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/core/warp_reduce.cuh"
#include "normal_consistency_loss.hpp"
#include "normal_loss.hpp"

#include <algorithm>

#include "kernel_stream.hpp"

namespace lfs::training::kernels {
    namespace {
        namespace slots = normal_consistency_slots;

        constexpr int kThreadsPerBlock = 256;
        constexpr size_t kMaxBlocks = 1024;

        constexpr int kStatCount = 3;
        constexpr int kLossStatCount = 1;

        constexpr float kMinExpectedDepth = 1.0e-6f;
        constexpr float kMinCrossNormSq = 1.0e-24f;

        [[nodiscard]] size_t consistency_block_count(const size_t num_pixels) {
            return std::min((num_pixels + kThreadsPerBlock - 1) / kThreadsPerBlock, kMaxBlocks);
        }

        struct Intrinsics {
            float fx;
            float fy;
            float cx;
            float cy;
        };

        __device__ __forceinline__ float3 pixel_ray(const Intrinsics k, const int x, const int y) {
            return make_float3(
                (static_cast<float>(x) + 0.5f - k.cx) / k.fx,
                (static_cast<float>(y) + 0.5f - k.cy) / k.fy,
                1.0f);
        }

        __device__ __forceinline__ bool expected_depth_at(
            const float* __restrict__ depth_accum,
            const float* __restrict__ alpha_map,
            const size_t idx,
            float& e_out,
            float& alpha_out) {
            const float a = alpha_map[idx];
            const float d = depth_accum[idx];
            if (!isfinite(a) || !isfinite(d) || a < kNormalConsistencyMinAlpha) {
                return false;
            }
            const float e = fmaxf(d, 0.0f) / a;
            if (e < kMinExpectedDepth) {
                return false;
            }
            e_out = e;
            alpha_out = a;
            return true;
        }

        // Neighbor order: +x, -x, +y, -y.
        struct DepthNormalSample {
            bool active;
            float alpha;
            float sign;
            float nraw_norm;
            float3 nd;
            float3 tx;
            float3 ty;
            float neighbor_e[4];
            float neighbor_alpha[4];
        };

        __device__ DepthNormalSample load_depth_normal_sample(
            const float* __restrict__ depth_accum,
            const float* __restrict__ alpha_map,
            const Intrinsics k,
            const int x,
            const int y,
            const int width,
            const int height,
            const size_t num_pixels) {
            DepthNormalSample s = {};

            if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1) {
                return s;
            }
            const size_t idx = static_cast<size_t>(y) * width + x;

            float e_c, a_c;
            if (!expected_depth_at(depth_accum, alpha_map, idx, e_c, a_c)) {
                return s;
            }

            const size_t neighbor_idx[4] = {idx + 1, idx - 1, idx + width, idx - width};
            const float jump_limit = kNormalConsistencyMaxRelDepthJump * e_c;
            for (int i = 0; i < 4; ++i) {
                if (!expected_depth_at(depth_accum, alpha_map, neighbor_idx[i],
                                       s.neighbor_e[i], s.neighbor_alpha[i])) {
                    return s;
                }
                if (fabsf(s.neighbor_e[i] - e_c) > jump_limit) {
                    return s;
                }
            }

            const float3 ray_xp = pixel_ray(k, x + 1, y);
            const float3 ray_xm = pixel_ray(k, x - 1, y);
            const float3 ray_yp = pixel_ray(k, x, y + 1);
            const float3 ray_ym = pixel_ray(k, x, y - 1);
            const float3 tx = make_float3(
                s.neighbor_e[0] * ray_xp.x - s.neighbor_e[1] * ray_xm.x,
                s.neighbor_e[0] * ray_xp.y - s.neighbor_e[1] * ray_xm.y,
                s.neighbor_e[0] * ray_xp.z - s.neighbor_e[1] * ray_xm.z);
            const float3 ty = make_float3(
                s.neighbor_e[2] * ray_yp.x - s.neighbor_e[3] * ray_ym.x,
                s.neighbor_e[2] * ray_yp.y - s.neighbor_e[3] * ray_ym.y,
                s.neighbor_e[2] * ray_yp.z - s.neighbor_e[3] * ray_ym.z);

            const float3 n_raw = make_float3(
                tx.y * ty.z - tx.z * ty.y,
                tx.z * ty.x - tx.x * ty.z,
                tx.x * ty.y - tx.y * ty.x);
            const float nraw_norm_sq = n_raw.x * n_raw.x + n_raw.y * n_raw.y + n_raw.z * n_raw.z;
            if (nraw_norm_sq < kMinCrossNormSq) {
                return s;
            }

            // Camera-facing orientation, detached like the rasterizer's own flip.
            const float3 ray_c = pixel_ray(k, x, y);
            const float facing = n_raw.x * ray_c.x + n_raw.y * ray_c.y + n_raw.z * ray_c.z;
            const float sign = facing > 0.0f ? -1.0f : 1.0f;

            const float nraw_norm = sqrtf(nraw_norm_sq);
            const float nd_rcp = sign / nraw_norm;
            s.nd = make_float3(n_raw.x * nd_rcp, n_raw.y * nd_rcp, n_raw.z * nd_rcp);
            s.sign = sign;
            s.nraw_norm = nraw_norm;
            s.tx = tx;
            s.ty = ty;
            s.alpha = a_c;
            s.active = true;
            return s;
        }

        struct ConsistencySample {
            bool active;
            float alpha;
            float cos;
            float sign;
            float nraw_norm;
            float nr_norm;
            float3 nd;
            float3 nr_hat;
            float3 tx;
            float3 ty;
            float neighbor_e[4];
            float neighbor_alpha[4];
        };

        __device__ ConsistencySample load_sample(
            const float* __restrict__ rendered_normal,
            const float* __restrict__ depth_accum,
            const float* __restrict__ alpha_map,
            const Intrinsics k,
            const int x,
            const int y,
            const int width,
            const int height,
            const size_t num_pixels) {
            ConsistencySample s = {};
            const DepthNormalSample d = load_depth_normal_sample(
                depth_accum, alpha_map, k, x, y, width, height, num_pixels);
            if (!d.active) {
                return s;
            }
            const size_t idx = static_cast<size_t>(y) * width + x;
            const float nx = rendered_normal[idx];
            const float ny = rendered_normal[num_pixels + idx];
            const float nz = rendered_normal[2 * num_pixels + idx];
            if (!isfinite(nx) || !isfinite(ny) || !isfinite(nz)) {
                return s;
            }
            const float nr_norm = sqrtf(nx * nx + ny * ny + nz * nz);
            if (nr_norm < kNormalLossMinRenderNorm) {
                return s;
            }

            const float nr_rcp = 1.0f / nr_norm;
            s.nr_hat = make_float3(nx * nr_rcp, ny * nr_rcp, nz * nr_rcp);
            s.cos = d.nd.x * s.nr_hat.x + d.nd.y * s.nr_hat.y + d.nd.z * s.nr_hat.z;
            s.active = true;
            s.alpha = d.alpha;
            s.sign = d.sign;
            s.nraw_norm = d.nraw_norm;
            s.nr_norm = nr_norm;
            s.nd = d.nd;
            s.tx = d.tx;
            s.ty = d.ty;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                s.neighbor_e[i] = d.neighbor_e[i];
                s.neighbor_alpha[i] = d.neighbor_alpha[i];
            }
            return s;
        }

        template <typename Sample>
        __device__ void accumulate_depth_normal_backward(
            const Sample& s,
            const float3 target_hat,
            const float cos,
            const float g_w,
            const Intrinsics k,
            const int x,
            const int y,
            const int width,
            float* __restrict__ grad_depth_accum,
            float* __restrict__ grad_alpha) {
            // d(1 - cos)/dn_raw = -sign * (target_hat - cos * n_d) / |n_raw|
            const float nraw_scale = -g_w * s.sign / s.nraw_norm;
            const float3 g_raw = make_float3(
                nraw_scale * (target_hat.x - cos * s.nd.x),
                nraw_scale * (target_hat.y - cos * s.nd.y),
                nraw_scale * (target_hat.z - cos * s.nd.z));

            // n_raw = tx x ty: grad_tx = ty x g_raw, grad_ty = g_raw x tx
            const float3 g_tx = make_float3(
                s.ty.y * g_raw.z - s.ty.z * g_raw.y,
                s.ty.z * g_raw.x - s.ty.x * g_raw.z,
                s.ty.x * g_raw.y - s.ty.y * g_raw.x);
            const float3 g_ty = make_float3(
                g_raw.y * s.tx.z - g_raw.z * s.tx.y,
                g_raw.z * s.tx.x - g_raw.x * s.tx.z,
                g_raw.x * s.tx.y - g_raw.y * s.tx.x);

            // tx = E_xp*r_xp - E_xm*r_xm, ty = E_yp*r_yp - E_ym*r_ym
            const float3 ray_xp = pixel_ray(k, x + 1, y);
            const float3 ray_xm = pixel_ray(k, x - 1, y);
            const float3 ray_yp = pixel_ray(k, x, y + 1);
            const float3 ray_ym = pixel_ray(k, x, y - 1);
            const float g_e[4] = {
                g_tx.x * ray_xp.x + g_tx.y * ray_xp.y + g_tx.z * ray_xp.z,
                -(g_tx.x * ray_xm.x + g_tx.y * ray_xm.y + g_tx.z * ray_xm.z),
                g_ty.x * ray_yp.x + g_ty.y * ray_yp.y + g_ty.z * ray_yp.z,
                -(g_ty.x * ray_ym.x + g_ty.y * ray_ym.y + g_ty.z * ray_ym.z)};

            const size_t idx = static_cast<size_t>(y) * width + x;
            const size_t neighbor_idx[4] = {idx + 1, idx - 1, idx + width, idx - width};
            for (int i = 0; i < 4; ++i) {
                // E = max(accum, 0)/alpha: dE/daccum = 1/alpha, dE/dalpha = -E/alpha
                const float inv_a = 1.0f / s.neighbor_alpha[i];
                atomicAdd(&grad_depth_accum[neighbor_idx[i]], g_e[i] * inv_a);
                atomicAdd(&grad_alpha[neighbor_idx[i]], -g_e[i] * s.neighbor_e[i] * inv_a);
            }
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

        __global__ void consistency_stats_kernel(
            const float* __restrict__ rendered_normal,
            const float* __restrict__ depth_accum,
            const float* __restrict__ alpha_map,
            double* __restrict__ block_partials,
            const Intrinsics k,
            const int width,
            const int height,
            const int num_blocks) {

            const size_t num_pixels = static_cast<size_t>(width) * height;
            double sums[kStatCount] = {};
            for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                 idx < num_pixels;
                 idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
                const int x = static_cast<int>(idx % width);
                const int y = static_cast<int>(idx / width);
                const ConsistencySample s = load_sample(
                    rendered_normal, depth_accum, alpha_map, k, x, y, width, height, num_pixels);
                if (!s.active) {
                    continue;
                }
                sums[0] += s.alpha;
                sums[1] += 1.0;
                sums[2] += static_cast<double>(s.alpha) * s.cos;
            }
            reduce_and_store(sums, block_partials, num_blocks);
        }

        __global__ void consistency_finalize_stats_kernel(
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
            const bool valid = count >= kNormalConsistencyMinValidCount &&
                               sum_alpha >= kNormalConsistencyMinValidWeight;

            finals[slots::kValid] = valid ? 1.0f : 0.0f;
            finals[slots::kSumAlpha] = static_cast<float>(sum_alpha);
            finals[slots::kCount] = static_cast<float>(count);
            finals[slots::kMeanCos] = sum_alpha > 0.0 ? static_cast<float>(sums[2] / sum_alpha) : 0.0f;
            finals[slots::kInvNorm] = valid ? static_cast<float>(weight / fmax(sum_alpha, 1.0)) : 0.0f;
        }

        __global__ void consistency_grad_kernel(
            const float* __restrict__ rendered_normal,
            const float* __restrict__ depth_accum,
            const float* __restrict__ alpha_map,
            float* __restrict__ grad_normal,
            float* __restrict__ grad_depth_accum,
            float* __restrict__ grad_alpha,
            const float* __restrict__ finals,
            double* __restrict__ block_partials,
            const Intrinsics k,
            const int width,
            const int height,
            const int num_blocks) {

            const float inv_norm = finals[slots::kInvNorm];
            const size_t num_pixels = static_cast<size_t>(width) * height;

            double sums[kLossStatCount] = {};
            for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                 idx < num_pixels;
                 idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
                const int x = static_cast<int>(idx % width);
                const int y = static_cast<int>(idx / width);
                const ConsistencySample s = load_sample(
                    rendered_normal, depth_accum, alpha_map, k, x, y, width, height, num_pixels);
                if (!s.active || inv_norm == 0.0f) {
                    continue;
                }

                sums[0] += static_cast<double>(s.alpha) * (1.0 - s.cos);

                const float g_w = inv_norm * s.alpha;

                // d(1 - cos)/dn_render = -(n_d - cos * n_hat) / |n_render|
                const float nr_scale = -g_w / s.nr_norm;
                grad_normal[idx] += nr_scale * (s.nd.x - s.cos * s.nr_hat.x);
                grad_normal[num_pixels + idx] += nr_scale * (s.nd.y - s.cos * s.nr_hat.y);
                grad_normal[2 * num_pixels + idx] += nr_scale * (s.nd.z - s.cos * s.nr_hat.z);

                accumulate_depth_normal_backward(
                    s, s.nr_hat, s.cos, g_w, k, x, y, width, grad_depth_accum, grad_alpha);
            }
            reduce_and_store(sums, block_partials, num_blocks);
        }

        __device__ __forceinline__ bool prior_normal_at(
            const float* __restrict__ prior_normal,
            const size_t idx,
            const size_t num_pixels,
            float3& prior_hat) {
            const float nx = prior_normal[idx];
            const float ny = prior_normal[num_pixels + idx];
            const float nz = prior_normal[2 * num_pixels + idx];
            if (!isfinite(nx) || !isfinite(ny) || !isfinite(nz)) {
                return false;
            }
            const float norm = sqrtf(nx * nx + ny * ny + nz * nz);
            if (norm < kNormalLossMinPriorNorm) {
                return false;
            }
            const float inv_norm = 1.0f / norm;
            prior_hat = make_float3(nx * inv_norm, ny * inv_norm, nz * inv_norm);
            return true;
        }

        __global__ void prior_depth_stats_kernel(
            const float* __restrict__ prior_normal,
            const float* __restrict__ depth_accum,
            const float* __restrict__ alpha_map,
            double* __restrict__ block_partials,
            const Intrinsics k,
            const int width,
            const int height,
            const int num_blocks) {

            const size_t num_pixels = static_cast<size_t>(width) * height;
            double sums[kStatCount] = {};
            for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                 idx < num_pixels;
                 idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
                const int x = static_cast<int>(idx % width);
                const int y = static_cast<int>(idx / width);
                const DepthNormalSample s = load_depth_normal_sample(
                    depth_accum, alpha_map, k, x, y, width, height, num_pixels);
                float3 prior_hat;
                if (!s.active || !prior_normal_at(prior_normal, idx, num_pixels, prior_hat)) {
                    continue;
                }
                const float cos = s.nd.x * prior_hat.x + s.nd.y * prior_hat.y + s.nd.z * prior_hat.z;
                sums[0] += s.alpha;
                sums[1] += 1.0;
                sums[2] += static_cast<double>(s.alpha) * cos;
            }
            reduce_and_store(sums, block_partials, num_blocks);
        }

        __global__ void prior_depth_finalize_stats_kernel(
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
            const bool valid = sum_alpha > 0.0;

            finals[slots::kValid] = valid ? 1.0f : 0.0f;
            finals[slots::kSumAlpha] = static_cast<float>(sum_alpha);
            finals[slots::kCount] = static_cast<float>(count);
            finals[slots::kMeanCos] = sum_alpha > 0.0 ? static_cast<float>(sums[2] / sum_alpha) : 0.0f;
            finals[slots::kInvNorm] = valid ? static_cast<float>(weight / fmax(sum_alpha, 1.0)) : 0.0f;
        }

        __global__ void prior_depth_grad_kernel(
            const float* __restrict__ prior_normal,
            const float* __restrict__ depth_accum,
            const float* __restrict__ alpha_map,
            float* __restrict__ grad_depth_accum,
            float* __restrict__ grad_alpha,
            const float* __restrict__ finals,
            double* __restrict__ block_partials,
            const Intrinsics k,
            const int width,
            const int height,
            const int num_blocks) {

            const float inv_norm = finals[slots::kInvNorm];
            const size_t num_pixels = static_cast<size_t>(width) * height;

            double sums[kLossStatCount] = {};
            for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                 idx < num_pixels;
                 idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
                const int x = static_cast<int>(idx % width);
                const int y = static_cast<int>(idx / width);
                const DepthNormalSample s = load_depth_normal_sample(
                    depth_accum, alpha_map, k, x, y, width, height, num_pixels);
                float3 prior_hat;
                const bool active = s.active && prior_normal_at(prior_normal, idx, num_pixels, prior_hat);
                float cos = 0.0f;
                if (active) {
                    cos = s.nd.x * prior_hat.x + s.nd.y * prior_hat.y + s.nd.z * prior_hat.z;
                }
                if (!active || inv_norm == 0.0f) {
                    continue;
                }

                sums[0] += static_cast<double>(s.alpha) * (1.0 - cos);
                const float g_w = inv_norm * s.alpha;
                accumulate_depth_normal_backward(
                    s, prior_hat, cos, g_w, k, x, y, width, grad_depth_accum, grad_alpha);
            }
            reduce_and_store(sums, block_partials, num_blocks);
        }

        __global__ void consistency_finalize_loss_kernel(
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

    size_t normal_consistency_partial_count(const size_t num_pixels) {
        const size_t num_blocks = consistency_block_count(num_pixels);
        return static_cast<size_t>(slots::kSlotCount) +
               2 * static_cast<size_t>(kStatCount) * num_blocks;
    }

    void launch_normal_consistency_loss(
        const float* rendered_normal,
        const float* rendered_depth_accum,
        const float* rendered_alpha,
        float* grad_normal,
        float* grad_depth_accum,
        float* grad_alpha,
        float* loss_out,
        float* partial_sums,
        const int width,
        const int height,
        const float fx,
        const float fy,
        const float cx,
        const float cy,
        const float weight,
        cudaStream_t stream) {
        stream = resolve_stream(stream);

        const size_t num_pixels = static_cast<size_t>(width) * height;
        const int num_blocks = static_cast<int>(consistency_block_count(num_pixels));
        float* finals = partial_sums;
        double* block_partials = reinterpret_cast<double*>(partial_sums + slots::kSlotCount);
        const Intrinsics k{fx, fy, cx, cy};

        consistency_stats_kernel<<<num_blocks, kThreadsPerBlock, 0, stream>>>(
            rendered_normal,
            rendered_depth_accum,
            rendered_alpha,
            block_partials,
            k,
            width,
            height,
            num_blocks);
        consistency_finalize_stats_kernel<<<1, kThreadsPerBlock, 0, stream>>>(
            block_partials,
            finals,
            num_blocks,
            weight);
        consistency_grad_kernel<<<num_blocks, kThreadsPerBlock, 0, stream>>>(
            rendered_normal,
            rendered_depth_accum,
            rendered_alpha,
            grad_normal,
            grad_depth_accum,
            grad_alpha,
            finals,
            block_partials,
            k,
            width,
            height,
            num_blocks);
        consistency_finalize_loss_kernel<<<1, kThreadsPerBlock, 0, stream>>>(
            block_partials,
            finals,
            loss_out,
            num_blocks);
    }

    void launch_normal_prior_depth_loss(
        const float* prior_normal,
        const float* rendered_depth_accum,
        const float* rendered_alpha,
        float* grad_depth_accum,
        float* grad_alpha,
        float* loss_out,
        float* partial_sums,
        const int width,
        const int height,
        const float fx,
        const float fy,
        const float cx,
        const float cy,
        const float weight,
        cudaStream_t stream) {
        stream = resolve_stream(stream);

        const size_t num_pixels = static_cast<size_t>(width) * height;
        const int num_blocks = static_cast<int>(consistency_block_count(num_pixels));
        float* finals = partial_sums;
        double* block_partials = reinterpret_cast<double*>(partial_sums + slots::kSlotCount);
        const Intrinsics k{fx, fy, cx, cy};

        prior_depth_stats_kernel<<<num_blocks, kThreadsPerBlock, 0, stream>>>(
            prior_normal,
            rendered_depth_accum,
            rendered_alpha,
            block_partials,
            k,
            width,
            height,
            num_blocks);
        prior_depth_finalize_stats_kernel<<<1, kThreadsPerBlock, 0, stream>>>(
            block_partials,
            finals,
            num_blocks,
            weight);
        prior_depth_grad_kernel<<<num_blocks, kThreadsPerBlock, 0, stream>>>(
            prior_normal,
            rendered_depth_accum,
            rendered_alpha,
            grad_depth_accum,
            grad_alpha,
            finals,
            block_partials,
            k,
            width,
            height,
            num_blocks);
        consistency_finalize_loss_kernel<<<1, kThreadsPerBlock, 0, stream>>>(
            block_partials,
            finals,
            loss_out,
            num_blocks);
    }

} // namespace lfs::training::kernels
