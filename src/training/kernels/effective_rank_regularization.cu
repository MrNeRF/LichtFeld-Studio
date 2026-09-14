/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/cuda_error.hpp"
#include "kernel_stream.hpp"
#include "lfs/kernels/effective_rank_regularization.cuh"
#include <cmath>

namespace lfs::training::kernels {
    namespace {
        __device__ float effective_rank_row(const float* raw, float* grad, const bool* active,
                                            const bool* frozen, size_t n, size_t i,
                                            float inv_count, float rw, float tw, float eps) {
            if (i >= n || (active && !active[i]))
                return 0;
            // Double intermediates retain the near-rank-one derivative. The parameter
            // and accumulated gradient remain FP32, matching the model's storage.
            double scales[3], eigen[3], p[3], dh[3];
            double sum = 0;
            int minimum = 0;
            for (int d = 0; d < 3; ++d) {
                scales[d] = exp(static_cast<double>(raw[3 * i + d]));
                eigen[d] = scales[d] * scales[d];
                sum += eigen[d];
                if (scales[d] < scales[minimum])
                    minimum = d;
            }
            const double denom = fmax(sum, static_cast<double>(eps));
            double entropy = 0, weighted_dh = 0;
            for (int d = 0; d < 3; ++d) {
                p[d] = eigen[d] / denom;
                const double logp = log(fmax(p[d], static_cast<double>(eps)));
                entropy -= p[d] * logp;
                dh[d] = -logp - (p[d] >= eps ? 1.0 : 0.0);
                weighted_dh += p[d] * dh[d];
            }
            const double rank = exp(entropy);
            const double argument = rank - 1.0 + eps;
            const double needle = fmax(-log(argument), 0.0);
            const float contribution = static_cast<float>((rw * needle + tw * scales[minimum]) * inv_count);
            if (!grad || (frozen && frozen[i]))
                return contribution;
            const double rank_factor = argument <= 1.0 ? -rw * rank / argument : 0.0;
            for (int d = 0; d < 3; ++d) {
                const double dh_dlog = 2.0 * p[d] * (dh[d] - (sum >= eps ? weighted_dh : 0.0));
                const double thin_grad = d == minimum ? tw * scales[d] : 0.0;
                grad[3 * i + d] += static_cast<float>((rank_factor * dh_dlog + thin_grad) * inv_count);
            }
            return contribution;
        }
        __global__ void effective_rank_kernel(const float* raw, float* grad, const bool* active,
                                              const bool* frozen, float* loss, size_t n,
                                              float inv_count, float rw, float tw, float eps) {
            __shared__ float partial[256];
            const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
            partial[threadIdx.x] = effective_rank_row(raw, grad, active, frozen, n, i, inv_count, rw, tw, eps);
            __syncthreads();
            for (unsigned stride = 128; stride; stride >>= 1) {
                if (threadIdx.x < stride)
                    partial[threadIdx.x] += partial[threadIdx.x + stride];
                __syncthreads();
            }
            if (threadIdx.x == 0)
                atomicAdd(loss, partial[0]);
        }
    } // namespace
    void launch_effective_rank_regularization(const float* log_scales, float* grad,
                                              const bool* active, const bool* frozen,
                                              float* loss, size_t n, size_t active_count,
                                              float rw, float tw, float eps, cudaStream_t stream) {
        stream = resolve_stream(stream);
        if (n == 0 || active_count == 0)
            return;
        effective_rank_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
            log_scales, grad, active, frozen, loss, n, 1.0f / active_count, rw, tw, eps);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.popspa.effective_rank");
    }
} // namespace lfs::training::kernels
