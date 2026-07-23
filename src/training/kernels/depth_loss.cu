/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include "core/logger.hpp"
#include "depth_loss.hpp"
#include "lfs/core/warp_reduce.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "kernel_stream.hpp"

namespace lfs::training::kernels {
    namespace {
        namespace slots = depth_loss_slots;

        constexpr int kThreadsPerBlock = 256;
        constexpr size_t kMaxBlocks = 1024;

        constexpr float kMinFloorAbs = 1.0e-8f;
        constexpr double kMinVariance = 1.0e-20;

        constexpr int kPrimaryStatCount = 3;
        constexpr int kInverseStatCount = 2;
        constexpr int kLossStatCount = 2;

        [[nodiscard]] size_t depth_loss_block_count(const size_t num_pixels) {
            return std::min((num_pixels + kThreadsPerBlock - 1) / kThreadsPerBlock, kMaxBlocks);
        }

        __device__ __forceinline__ float sign_of(const float v) {
            return v > 0.0f ? 1.0f : (v < 0.0f ? -1.0f : 0.0f);
        }

        __device__ __forceinline__ float robust_rho(const float x) {
            const float x2 = x * x;
            return 0.5f * x2 / (1.0f + x2);
        }

        __device__ __forceinline__ float robust_psi(const float x) {
            const float x2 = x * x;
            const float den = 1.0f + x2;
            return x / (den * den);
        }

        __device__ __forceinline__ float deadband_signed_residual(const float r, const float delta) {
            const float excess = fabsf(r) - delta;
            return excess > 0.0f ? sign_of(r) * excess : 0.0f;
        }

        __device__ __forceinline__ bool pixel_active(
            const float target,
            const float depth_accum,
            const float alpha) {
            return target > 0.0f &&
                   alpha > kDepthLossMinAlpha &&
                   isfinite(target) &&
                   isfinite(depth_accum) &&
                   isfinite(alpha);
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

        __global__ void depth_loss_stats_primary_kernel(
            const float* __restrict__ rendered_depth_accum,
            const float* __restrict__ rendered_alpha_accum,
            const float* __restrict__ target_depth,
            double* __restrict__ block_partials,
            const size_t num_pixels,
            const int num_blocks) {

            double sums[kPrimaryStatCount] = {};
            for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                 idx < num_pixels;
                 idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
                const float t = target_depth[idx];
                const float d_raw = rendered_depth_accum[idx];
                const float a = rendered_alpha_accum[idx];
                if (!pixel_active(t, d_raw, a)) {
                    continue;
                }
                const double aw = a;
                const double e = fmaxf(d_raw, 0.0f) / a;
                sums[0] += aw;
                sums[1] += aw * e;
                sums[2] += 1.0;
            }
            reduce_and_store(sums, block_partials, num_blocks);
        }

        __global__ void depth_loss_finalize_primary_kernel(
            const double* __restrict__ block_partials,
            float* __restrict__ finals,
            const int num_blocks) {

            double sums[kPrimaryStatCount] = {};
            reduce_block_partials(block_partials, sums, num_blocks);
            if (threadIdx.x != 0) {
                return;
            }

            const double sum_alpha = sums[0];
            const double count = sums[2];
            const bool valid = sum_alpha > 0.0;
            const double mean_e = valid ? sums[1] / sum_alpha : 0.0;

            finals[slots::kValid] = valid ? 1.0f : 0.0f;
            finals[slots::kFloor] = fmaxf(kMinFloorAbs,
                                          kDepthLossFloorFraction * static_cast<float>(mean_e));
            finals[slots::kSumAlpha] = static_cast<float>(sum_alpha);
            finals[slots::kCount] = static_cast<float>(count);
            finals[slots::kMeanExpectedDepth] = static_cast<float>(mean_e);
        }

        __global__ void depth_loss_stats_inverse_kernel(
            const float* __restrict__ rendered_depth_accum,
            const float* __restrict__ rendered_alpha_accum,
            const float* __restrict__ target_depth,
            const float* __restrict__ finals,
            double* __restrict__ block_partials,
            const size_t num_pixels,
            const int num_blocks,
            const float floor_override) {

            const float floor_f = floor_override > 0.0f ? floor_override : finals[slots::kFloor];
            const bool valid = finals[slots::kValid] > 0.5f;

            double sums[kInverseStatCount] = {};
            if (valid) {
                for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                     idx < num_pixels;
                     idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
                    const float t = target_depth[idx];
                    const float d_raw = rendered_depth_accum[idx];
                    const float a = rendered_alpha_accum[idx];
                    if (!pixel_active(t, d_raw, a)) {
                        continue;
                    }
                    const float e = fmaxf(d_raw, 0.0f) / a;
                    const double p = 1.0f / (e + floor_f);
                    const double aw = a;
                    sums[0] += aw * p;
                    sums[1] += aw * p * p;
                }
            }
            reduce_and_store(sums, block_partials, num_blocks);
        }

        __global__ void depth_loss_finalize_alignment_kernel(
            const double* __restrict__ block_partials,
            float* __restrict__ finals,
            const int num_blocks,
            const int anchor_model,
            const float anchor_scale,
            const float anchor_shift,
            const float anchor_floor) {

            double sums[kInverseStatCount] = {};
            reduce_block_partials(block_partials, sums, num_blocks);
            if (threadIdx.x != 0) {
                return;
            }

            bool valid = finals[slots::kValid] > 0.5f &&
                         anchor_floor > 0.0f &&
                         isfinite(anchor_scale) &&
                         isfinite(anchor_shift);
            const double sum_alpha = finals[slots::kSumAlpha];

            double mean_p = 0.0;
            double var_p = 0.0;
            double sigma_p = 0.0;
            if (valid) {
                mean_p = sums[0] / sum_alpha;
                var_p = fmax(sums[1] / sum_alpha - mean_p * mean_p, 0.0);
                sigma_p = sqrt(fmax(var_p, kMinVariance));
            }

            if (valid) {
                finals[slots::kFloor] = anchor_floor;
            }

            finals[slots::kValid] = valid ? 1.0f : 0.0f;
            finals[slots::kModel] = static_cast<float>(anchor_model);
            finals[slots::kScale] = anchor_scale;
            finals[slots::kShift] = anchor_shift;
            finals[slots::kSigmaP] = static_cast<float>(sigma_p);
            finals[slots::kInvNorm] = valid ? static_cast<float>(1.0 / sum_alpha) : 0.0f;
        }

        struct PixelSample {
            bool ok;
            float alpha;
            float e;
            float p;
            float d;
            float delta;
        };

        __device__ __forceinline__ PixelSample load_sample(
            const size_t idx,
            const float* __restrict__ rendered_depth_accum,
            const float* __restrict__ rendered_alpha_accum,
            const float* __restrict__ target_depth,
            const int model,
            const float a,
            const float b,
            const float floor_f,
            const float p_max,
            const float half_step) {

            PixelSample s{};
            const float t = target_depth[idx];
            const float d_raw = rendered_depth_accum[idx];
            const float alpha = rendered_alpha_accum[idx];
            if (!pixel_active(t, d_raw, alpha)) {
                return s;
            }
            s.alpha = alpha;
            s.e = fmaxf(d_raw, 0.0f) / alpha;
            s.p = 1.0f / (s.e + floor_f);
            const float fit = a * t + b;
            if (model == 0) {
                if (!(fit > 0.0f)) {
                    return s;
                }
                s.d = fminf(fit, p_max);
                s.delta = half_step;
            } else {
                if (!(fit > 0.0f)) {
                    return s;
                }
                s.d = 1.0f / (fit + floor_f);
                s.delta = half_step * s.d * s.d;
            }
            s.ok = true;
            return s;
        }

        __global__ void depth_loss_grad_kernel(
            const float* __restrict__ rendered_depth_accum,
            const float* __restrict__ rendered_alpha_accum,
            const float* __restrict__ target_depth,
            float* __restrict__ grad_depth,
            float* __restrict__ grad_alpha,
            const float* __restrict__ finals,
            double* __restrict__ block_partials,
            const int width,
            const int height,
            const float weight,
            const float lambda_grad,
            const float quantization_step,
            const int num_blocks) {

            const bool valid = finals[slots::kValid] > 0.5f;
            const int model = static_cast<int>(finals[slots::kModel]);
            const float a = finals[slots::kScale];
            const float b = finals[slots::kShift];
            const float floor_f = finals[slots::kFloor];
            const float inv_norm = finals[slots::kInvNorm];
            const float sigma_p = finals[slots::kSigmaP];
            const float inv_scaled_sigma =
                valid && sigma_p > 0.0f ? 1.0f / (kDepthLossResidualScale * sigma_p) : 0.0f;
            const float p_max = 1.0f / floor_f;
            const float half_step = 0.5f * quantization_step * fabsf(a);
            const size_t num_pixels = static_cast<size_t>(width) * height;

            double sums[kLossStatCount] = {};
            for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                 idx < num_pixels;
                 idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
                if (!valid) {
                    grad_depth[idx] = 0.0f;
                    grad_alpha[idx] = 0.0f;
                    continue;
                }
                const int x = static_cast<int>(idx % width);
                const int y = static_cast<int>(idx / width);
                const PixelSample c = load_sample(
                    idx, rendered_depth_accum, rendered_alpha_accum, target_depth,
                    model, a, b, floor_f, p_max, half_step);

                float gp = 0.0f;
                if (c.ok) {
                    const float x_r =
                        deadband_signed_residual(c.p - c.d, c.delta) * inv_scaled_sigma;
                    if (x_r != 0.0f) {
                        sums[0] += static_cast<double>(c.alpha) * robust_rho(x_r);
                        gp += c.alpha * robust_psi(x_r) * inv_scaled_sigma;
                    }

                    if (x + 1 < width) {
                        const PixelSample n = load_sample(
                            idx + 1, rendered_depth_accum, rendered_alpha_accum, target_depth,
                            model, a, b, floor_f, p_max, half_step);
                        if (n.ok) {
                            const float h = (n.p - c.p) - (n.d - c.d);
                            const float x_h =
                                deadband_signed_residual(h, c.delta + n.delta) * inv_scaled_sigma;
                            if (x_h != 0.0f) {
                                const float w2 = fminf(c.alpha, n.alpha);
                                sums[1] += static_cast<double>(w2) * robust_rho(x_h);
                                gp -= lambda_grad * w2 * robust_psi(x_h) * inv_scaled_sigma;
                            }
                        }
                    }
                    if (y + 1 < height) {
                        const PixelSample n = load_sample(
                            idx + width, rendered_depth_accum, rendered_alpha_accum, target_depth,
                            model, a, b, floor_f, p_max, half_step);
                        if (n.ok) {
                            const float v = (n.p - c.p) - (n.d - c.d);
                            const float x_v =
                                deadband_signed_residual(v, c.delta + n.delta) * inv_scaled_sigma;
                            if (x_v != 0.0f) {
                                const float w2 = fminf(c.alpha, n.alpha);
                                sums[1] += static_cast<double>(w2) * robust_rho(x_v);
                                gp -= lambda_grad * w2 * robust_psi(x_v) * inv_scaled_sigma;
                            }
                        }
                    }
                    if (x > 0) {
                        const PixelSample n = load_sample(
                            idx - 1, rendered_depth_accum, rendered_alpha_accum, target_depth,
                            model, a, b, floor_f, p_max, half_step);
                        if (n.ok) {
                            const float h = (c.p - n.p) - (c.d - n.d);
                            const float x_h =
                                deadband_signed_residual(h, c.delta + n.delta) * inv_scaled_sigma;
                            if (x_h != 0.0f) {
                                const float w2 = fminf(n.alpha, c.alpha);
                                gp += lambda_grad * w2 * robust_psi(x_h) * inv_scaled_sigma;
                            }
                        }
                    }
                    if (y > 0) {
                        const PixelSample n = load_sample(
                            idx - width, rendered_depth_accum, rendered_alpha_accum, target_depth,
                            model, a, b, floor_f, p_max, half_step);
                        if (n.ok) {
                            const float v = (c.p - n.p) - (c.d - n.d);
                            const float x_v =
                                deadband_signed_residual(v, c.delta + n.delta) * inv_scaled_sigma;
                            if (x_v != 0.0f) {
                                const float w2 = fminf(n.alpha, c.alpha);
                                gp += lambda_grad * w2 * robust_psi(x_v) * inv_scaled_sigma;
                            }
                        }
                    }
                }

                if (c.ok && gp != 0.0f) {
                    const float dp_de = -c.p * c.p;
                    const float g = weight * inv_norm * gp * dp_de;
                    grad_depth[idx] = g / c.alpha;
                    grad_alpha[idx] = -g * c.e / c.alpha;
                } else {
                    grad_depth[idx] = 0.0f;
                    grad_alpha[idx] = 0.0f;
                }
            }
            reduce_and_store(sums, block_partials, num_blocks);
        }

        constexpr size_t kMaxAnchorSamples = 262144;
        constexpr size_t kRansacScoreSubset = 16384;
        constexpr int kRansacIterations = 256;
        constexpr int kMinAnchorSamples = 256;
        constexpr float kMinAnchorCorr = 0.35f;
        constexpr float kMinAnchorInlierFraction = 0.3f;

        // Projects every stride-th point, samples the prior at the landing
        // pixel, and appends (prior value, camera-space depth) pairs.
        __global__ void depth_anchor_collect_kernel(
            const float* __restrict__ points_xyz,
            const size_t num_samples,
            const size_t stride,
            const float* __restrict__ w2c,
            const float fx,
            const float fy,
            const float cx,
            const float cy,
            const float* __restrict__ prior,
            const int width,
            const int height,
            const float near_plane,
            const float3 aabb_lo,
            const float3 aabb_hi,
            float2* __restrict__ pairs_out,
            int* __restrict__ pair_count,
            const int pair_capacity) {

            const float4 r1 = make_float4(w2c[0], w2c[1], w2c[2], w2c[3]);
            const float4 r2 = make_float4(w2c[4], w2c[5], w2c[6], w2c[7]);
            const float4 r3 = make_float4(w2c[8], w2c[9], w2c[10], w2c[11]);

            for (size_t s = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                 s < num_samples;
                 s += static_cast<size_t>(blockDim.x) * gridDim.x) {
                const size_t idx = s * stride;
                const float x = points_xyz[idx * 3 + 0];
                const float y = points_xyz[idx * 3 + 1];
                const float zw = points_xyz[idx * 3 + 2];
                if (!isfinite(x) || !isfinite(y) || !isfinite(zw)) {
                    continue;
                }
                if (x < aabb_lo.x || x > aabb_hi.x ||
                    y < aabb_lo.y || y > aabb_hi.y ||
                    zw < aabb_lo.z || zw > aabb_hi.z) {
                    continue;
                }
                const float z = r3.x * x + r3.y * y + r3.z * zw + r3.w;
                if (!(z > near_plane) || !isfinite(z)) {
                    continue;
                }
                const float xc = r1.x * x + r1.y * y + r1.z * zw + r1.w;
                const float yc = r2.x * x + r2.y * y + r2.z * zw + r2.w;
                const int u = static_cast<int>(floorf(fx * xc / z + cx));
                const int v = static_cast<int>(floorf(fy * yc / z + cy));
                if (u < 0 || u >= width || v < 0 || v >= height) {
                    continue;
                }
                const float t = prior[static_cast<size_t>(v) * width + u];
                if (!(t > 0.0f) || !isfinite(t)) {
                    continue;
                }
                const int slot = atomicAdd(pair_count, 1);
                if (slot < pair_capacity) {
                    pairs_out[slot] = make_float2(t, z);
                }
            }
        }

        struct RobustFit {
            bool usable = false;
            double a = 0.0;
            double b = 0.0;
            double corr = 0.0;
            size_t inliers = 0;
        };

        // RANSAC affine fit y = a*t + b over (t, y) pairs, least-squares refit
        // on the consensus set. Survives the heavily contaminated sparse
        // clouds that break a plain least-squares + trim.
        [[nodiscard]] RobustFit ransac_affine_fit(
            const std::vector<float>& ts,
            const std::vector<float>& ys,
            const uint32_t seed) {
            RobustFit out;
            const size_t n = ts.size();
            if (n < static_cast<size_t>(kMinAnchorSamples)) {
                return out;
            }

            const auto ls_fit = [&](const auto& accept) {
                double st = 0.0, sy = 0.0, stt = 0.0, syy = 0.0, sty = 0.0;
                size_t m = 0;
                for (size_t i = 0; i < n; ++i) {
                    if (!accept(i)) {
                        continue;
                    }
                    const double t = ts[i];
                    const double y = ys[i];
                    st += t;
                    sy += y;
                    stt += t * t;
                    syy += y * y;
                    sty += t * y;
                    ++m;
                }
                RobustFit fit;
                if (m < static_cast<size_t>(kMinAnchorSamples)) {
                    return fit;
                }
                const double inv_m = 1.0 / static_cast<double>(m);
                const double mean_t = st * inv_m;
                const double mean_y = sy * inv_m;
                const double var_t = std::max(stt * inv_m - mean_t * mean_t, 0.0);
                const double var_y = std::max(syy * inv_m - mean_y * mean_y, 0.0);
                const double cov = sty * inv_m - mean_t * mean_y;
                fit.a = cov / (var_t + kDepthLossTargetVarRidge);
                fit.b = mean_y - fit.a * mean_t;
                if (var_t > kMinVariance && var_y > kMinVariance) {
                    fit.corr = cov / std::sqrt(var_t * var_y);
                }
                fit.inliers = m;
                fit.usable = true;
                return fit;
            };

            const RobustFit initial = ls_fit([](size_t) { return true; });
            if (!initial.usable) {
                return out;
            }

            // Robust residual scale from the median absolute deviation.
            std::vector<float> abs_res(n);
            for (size_t i = 0; i < n; ++i) {
                abs_res[i] = std::fabs(static_cast<float>(ys[i] - (initial.a * ts[i] + initial.b)));
            }
            std::nth_element(abs_res.begin(), abs_res.begin() + n / 2, abs_res.end());
            const double mad_sigma = 1.4826 * abs_res[n / 2];
            const double eps = std::max(3.0 * mad_sigma, 1.0e-6 + 1.0e-4 * std::fabs(initial.b));

            const size_t score_stride = std::max<size_t>(1, n / kRansacScoreSubset);
            uint32_t state = seed | 1u;
            const auto next_rand = [&state]() {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                return state;
            };

            double best_a = initial.a;
            double best_b = initial.b;
            size_t best_score = 0;
            for (size_t i = 0; i < n; i += score_stride) {
                if (std::fabs(ys[i] - (initial.a * ts[i] + initial.b)) <= eps) {
                    ++best_score;
                }
            }

            for (int it = 0; it < kRansacIterations; ++it) {
                const size_t i0 = next_rand() % n;
                const size_t i1 = next_rand() % n;
                const double dt = static_cast<double>(ts[i1]) - ts[i0];
                if (std::fabs(dt) < 1.0e-12) {
                    continue;
                }
                const double a = (static_cast<double>(ys[i1]) - ys[i0]) / dt;
                const double b = ys[i0] - a * ts[i0];
                size_t score = 0;
                for (size_t i = 0; i < n; i += score_stride) {
                    if (std::fabs(ys[i] - (a * ts[i] + b)) <= eps) {
                        ++score;
                    }
                }
                if (score > best_score) {
                    best_score = score;
                    best_a = a;
                    best_b = b;
                }
            }

            const double consensus_a = best_a;
            const double consensus_b = best_b;
            RobustFit refined = ls_fit([&](size_t i) {
                return std::fabs(ys[i] - (consensus_a * ts[i] + consensus_b)) <= eps;
            });
            if (!refined.usable) {
                return out;
            }
            const double refined_a = refined.a;
            const double refined_b = refined.b;
            refined = ls_fit([&](size_t i) {
                return std::fabs(ys[i] - (refined_a * ts[i] + refined_b)) <= eps;
            });
            return refined.usable ? refined : out;
        }

        __global__ void depth_loss_finalize_loss_kernel(
            const double* __restrict__ block_partials,
            const float* __restrict__ finals,
            float* __restrict__ loss_out,
            const int num_blocks,
            const float weight,
            const float lambda_grad) {

            double sums[kLossStatCount] = {};
            reduce_block_partials(block_partials, sums, num_blocks);
            if (threadIdx.x == 0) {
                const bool valid = finals[slots::kValid] > 0.5f;
                const float inv_norm = finals[slots::kInvNorm];
                loss_out[0] = valid
                                  ? weight * inv_norm * static_cast<float>(sums[0] + lambda_grad * sums[1])
                                  : 0.0f;
            }
        }
    } // namespace

    size_t depth_loss_partial_count(const size_t num_pixels) {
        const size_t num_blocks = depth_loss_block_count(num_pixels);
        return static_cast<size_t>(slots::kSlotCount) +
               2 * static_cast<size_t>(kPrimaryStatCount) * num_blocks;
    }

    std::vector<float2> collect_depth_anchor_samples(
        const float* points_xyz,
        const size_t num_points,
        const float* w2c,
        const float fx,
        const float fy,
        const float cx,
        const float cy,
        const float* prior,
        const int width,
        const int height,
        const float near_plane,
        const float aabb_lo[3],
        const float aabb_hi[3],
        cudaStream_t stream) {
        stream = resolve_stream(stream);

        if (num_points == 0) {
            return {};
        }
        const size_t stride = std::max<size_t>(1, num_points / kMaxAnchorSamples);
        const size_t num_samples = (num_points + stride - 1) / stride;
        const int num_blocks = static_cast<int>(
            std::min((num_samples + kThreadsPerBlock - 1) / kThreadsPerBlock, kMaxBlocks));
        const int pair_capacity = static_cast<int>(kMaxAnchorSamples);

        float2* pairs_dev = nullptr;
        int* count_dev = nullptr;
        if (cudaMallocAsync(&pairs_dev, sizeof(float2) * kMaxAnchorSamples, stream) != cudaSuccess) {
            return {};
        }
        if (cudaMallocAsync(&count_dev, sizeof(int), stream) != cudaSuccess) {
            cudaFreeAsync(pairs_dev, stream);
            return {};
        }
        cudaMemsetAsync(count_dev, 0, sizeof(int), stream);

        const auto anchor_ticket =
            ::lfs::core::cuda_record_range(stream, "training.depth.anchor_collect_pipeline");
        depth_anchor_collect_kernel<<<num_blocks, kThreadsPerBlock, 0, stream>>>(
            points_xyz, num_samples, stride, w2c, fx, fy, cx, cy,
            prior, width, height, near_plane,
            make_float3(aabb_lo[0], aabb_lo[1], aabb_lo[2]),
            make_float3(aabb_hi[0], aabb_hi[1], aabb_hi[2]),
            pairs_dev, count_dev, pair_capacity);
        if (const cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
            LOG_ERROR("depth_anchor_collect_kernel launch failed: {}", cudaGetErrorString(err));
            cudaFreeAsync(pairs_dev, stream);
            cudaFreeAsync(count_dev, stream);
            return {};
        }

        int pair_count = 0;
        cudaMemcpyAsync(&pair_count, count_dev, sizeof(int), cudaMemcpyDeviceToHost, stream);
        try {
            LFS_CUDA_AWAIT(anchor_ticket, cudaStreamSynchronize(stream), "training.depth.anchor_count_readback");
        } catch (const std::exception& e) {
            LOG_ERROR("depth anchor count readback failed: {}", e.what());
            cudaFreeAsync(pairs_dev, stream);
            cudaFreeAsync(count_dev, stream);
            return {};
        }
        pair_count = std::min(pair_count, pair_capacity);
        if (pair_count < kMinAnchorSamples) {
            cudaFreeAsync(pairs_dev, stream);
            cudaFreeAsync(count_dev, stream);
            return {};
        }

        const auto pairs_ticket =
            ::lfs::core::cuda_record_range(stream, "training.depth.anchor_pairs_pipeline");
        std::vector<float2> pairs(static_cast<size_t>(pair_count));
        cudaMemcpyAsync(pairs.data(), pairs_dev, sizeof(float2) * pair_count,
                        cudaMemcpyDeviceToHost, stream);
        try {
            LFS_CUDA_AWAIT(pairs_ticket, cudaStreamSynchronize(stream), "training.depth.anchor_pairs_readback");
        } catch (const std::exception& e) {
            LOG_ERROR("depth anchor pairs readback failed: {}", e.what());
            cudaFreeAsync(pairs_dev, stream);
            cudaFreeAsync(count_dev, stream);
            return {};
        }
        cudaFreeAsync(pairs_dev, stream);
        cudaFreeAsync(count_dev, stream);
        return pairs;
    }

    DepthAnchor fit_depth_anchor_from_samples(const std::vector<float2>& pairs) {
        DepthAnchor anchor;
        const size_t n = pairs.size();
        if (n < static_cast<size_t>(kMinAnchorSamples)) {
            return anchor;
        }
        std::vector<float> ts(n);
        std::vector<float> zs(n);
        for (size_t i = 0; i < n; ++i) {
            ts[i] = pairs[i].x;
            zs[i] = pairs[i].y;
        }

        // Robust floor from the median depth: contaminated sparse clouds have
        // extreme outliers that would wreck a mean-based floor.
        std::vector<float> z_sorted = zs;
        std::nth_element(z_sorted.begin(), z_sorted.begin() + n / 2, z_sorted.end());
        const float median_z = z_sorted[n / 2];
        const float floor_f = std::max(1.0e-8f, kDepthLossFloorFraction * median_z);

        std::vector<float> qs(n);
        for (size_t i = 0; i < n; ++i) {
            qs[i] = 1.0f / (zs[i] + floor_f);
        }

        double mean_t = 0.0;
        for (const float t : ts) {
            mean_t += t;
        }
        mean_t /= static_cast<double>(n);
        double var_t = 0.0;
        for (const float t : ts) {
            var_t += (t - mean_t) * (t - mean_t);
        }
        var_t /= static_cast<double>(n);

        if (var_t <= kDepthLossFlatPriorVar) {
            return anchor;
        }

        const uint32_t seed = 0x9e3779b9u ^ static_cast<uint32_t>(n);
        const RobustFit fit_q = ransac_affine_fit(ts, qs, seed);
        const RobustFit fit_z = ransac_affine_fit(ts, zs, seed * 2654435761u);

        const auto resolvable = [&](const RobustFit& fit) {
            return fit.usable &&
                   fit.inliers >= static_cast<size_t>(kMinAnchorInlierFraction * n) &&
                   std::fabs(fit.corr) >= kMinAnchorCorr;
        };
        const bool q_ok = resolvable(fit_q);
        const bool z_ok = resolvable(fit_z);
        if (q_ok) {
            anchor.disparity.valid = true;
            anchor.disparity.scale = static_cast<float>(fit_q.a);
            anchor.disparity.shift = static_cast<float>(fit_q.b);
            anchor.disparity.corr = static_cast<float>(fit_q.corr);
            anchor.disparity.samples = static_cast<int>(fit_q.inliers);
        }
        if (z_ok) {
            anchor.depth.valid = true;
            anchor.depth.scale = static_cast<float>(fit_z.a);
            anchor.depth.shift = static_cast<float>(fit_z.b);
            anchor.depth.corr = static_cast<float>(fit_z.corr);
            anchor.depth.samples = static_cast<int>(fit_z.inliers);
        }

        if (q_ok || z_ok) {
            const bool use_q = q_ok && (!z_ok || std::fabs(fit_q.corr) >= std::fabs(fit_z.corr));
            const DepthAnchorCandidate& fit = use_q ? anchor.disparity : anchor.depth;
            anchor.valid = true;
            anchor.model = use_q ? 0 : 1;
            anchor.scale = fit.scale;
            anchor.shift = fit.shift;
            anchor.floor = floor_f;
            anchor.corr = fit.corr;
            anchor.samples = fit.samples;
            return anchor;
        }

        return anchor;
    }

    DepthAnchor fit_depth_anchor(
        const float* points_xyz,
        const size_t num_points,
        const float* w2c,
        const float fx,
        const float fy,
        const float cx,
        const float cy,
        const float* prior,
        const int width,
        const int height,
        const float near_plane,
        const float aabb_lo[3],
        const float aabb_hi[3],
        cudaStream_t stream) {
        return fit_depth_anchor_from_samples(collect_depth_anchor_samples(
            points_xyz, num_points, w2c, fx, fy, cx, cy,
            prior, width, height, near_plane, aabb_lo, aabb_hi, stream));
    }

    void launch_depth_loss(
        const float* rendered_depth_accum,
        const float* rendered_alpha_accum,
        const float* target_depth,
        float* grad_depth,
        float* grad_alpha,
        float* loss_out,
        float* partial_sums,
        const int width,
        const int height,
        const float weight,
        const float gradient_term_weight,
        const float prior_quantization_step,
        const DepthAnchor* anchor,
        cudaStream_t stream) {
        stream = resolve_stream(stream);

        const size_t num_pixels = static_cast<size_t>(width) * height;
        const int num_blocks = static_cast<int>(depth_loss_block_count(num_pixels));
        float* finals = partial_sums;
        double* block_partials = reinterpret_cast<double*>(partial_sums + slots::kSlotCount);
        const bool use_anchor = anchor != nullptr && anchor->valid;

        depth_loss_stats_primary_kernel<<<num_blocks, kThreadsPerBlock, 0, stream>>>(
            rendered_depth_accum,
            rendered_alpha_accum,
            target_depth,
            block_partials,
            num_pixels,
            num_blocks);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.depth.stats_primary");
        depth_loss_finalize_primary_kernel<<<1, kThreadsPerBlock, 0, stream>>>(
            block_partials,
            finals,
            num_blocks);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.depth.finalize_primary");
        depth_loss_stats_inverse_kernel<<<num_blocks, kThreadsPerBlock, 0, stream>>>(
            rendered_depth_accum,
            rendered_alpha_accum,
            target_depth,
            finals,
            block_partials,
            num_pixels,
            num_blocks,
            use_anchor ? anchor->floor : 0.0f);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.depth.stats_inverse");
        depth_loss_finalize_alignment_kernel<<<1, kThreadsPerBlock, 0, stream>>>(
            block_partials,
            finals,
            num_blocks,
            use_anchor ? anchor->model : 0,
            use_anchor ? anchor->scale : 0.0f,
            use_anchor ? anchor->shift : 0.0f,
            use_anchor ? anchor->floor : 0.0f);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.depth.finalize_alignment");
        depth_loss_grad_kernel<<<num_blocks, kThreadsPerBlock, 0, stream>>>(
            rendered_depth_accum,
            rendered_alpha_accum,
            target_depth,
            grad_depth,
            grad_alpha,
            finals,
            block_partials,
            width,
            height,
            weight,
            gradient_term_weight,
            fmaxf(prior_quantization_step, 0.0f),
            num_blocks);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.depth.grad");
        depth_loss_finalize_loss_kernel<<<1, kThreadsPerBlock, 0, stream>>>(
            block_partials,
            finals,
            loss_out,
            num_blocks,
            weight,
            gradient_term_weight);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.depth.finalize_loss");
    }

} // namespace lfs::training::kernels
