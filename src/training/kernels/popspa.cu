/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/cuda_error.hpp"
#include "kernel_stream.hpp"
#include "lfs/kernels/popspa.cuh"
#include <cmath>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

namespace lfs::training::kernels {
    namespace {
        struct Priority {
            const double* values;
            const bool* active;
            const bool* frozen;
            __device__ bool operator()(int64_t a, int64_t b) const {
                if (active[a] != active[b])
                    return active[a];
                if (frozen[a] != frozen[b])
                    return frozen[a];
                if (values[a] != values[b])
                    return values[a] > values[b];
                return a < b;
            }
        };
        __global__ void shifted_kernel(const float* raw, const float* u, double* shifted, size_t n) {
            const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n)
                shifted[i] = static_cast<double>(1.0f / (1.0f + expf(-raw[i])) + u[i]);
        }
        __global__ void scatter_proxy(const double* shifted, const int64_t* keep, float* z, size_t k) {
            const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < k)
                z[keep[i]] = static_cast<float>(shifted[keep[i]]);
        }
        __global__ void dual_kernel(const float* raw, const float* z, float* u,
                                    const bool* active, const bool* frozen, size_t n) {
            const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n && active[i] && !frozen[i])
                u[i] += 1.0f / (1.0f + expf(-raw[i])) - z[i];
        }
        __global__ void penalty_kernel(const float* raw, const float* z, const float* u,
                                       const bool* active, const bool* frozen, float* grad,
                                       float* loss, size_t n, float rho) {
            const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
            __shared__ float partial[256];
            float contribution = 0;
            if (i < n && active[i] && !frozen[i]) {
                const float a = 1.0f / (1.0f + expf(-raw[i]));
                const float d = a - z[i] + u[i];
                grad[i] += rho * d * a * (1.0f - a);
                contribution = 0.5f * rho * d * d;
            }
            partial[threadIdx.x] = contribution;
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
    void launch_popspa_keep_indices(const double* scores, const bool* active, const bool* frozen,
                                    size_t n, size_t k, int64_t* scratch, int64_t* keep,
                                    cudaStream_t stream) {
        stream = resolve_stream(stream);
        auto policy = thrust::cuda::par.on(stream);
        auto begin = thrust::device_pointer_cast(scratch);
        thrust::sequence(policy, begin, begin + n, int64_t{0});
        thrust::sort(policy, begin, begin + n, Priority{scores, active, frozen});
        LFS_CUDA_CHECK(cudaMemcpyAsync(keep, scratch, k * sizeof(int64_t), cudaMemcpyDeviceToDevice, stream));
        auto selected = thrust::device_pointer_cast(keep);
        thrust::sort(policy, selected, selected + k);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.popspa.select");
    }
    void launch_popspa_project(const float* raw, float* z, float* u, const bool* active,
                               const bool* frozen, size_t n, size_t k, double* shifted,
                               int64_t* scratch, int64_t* keep, bool update_dual, cudaStream_t stream) {
        stream = resolve_stream(stream);
        shifted_kernel<<<(n + 255) / 256, 256, 0, stream>>>(raw, u, shifted, n);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.popspa.shifted");
        launch_popspa_keep_indices(shifted, active, frozen, n, k, scratch, keep, stream);
        LFS_CUDA_CHECK(cudaMemsetAsync(z, 0, n * sizeof(float), stream));
        scatter_proxy<<<(k + 255) / 256, 256, 0, stream>>>(shifted, keep, z, k);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.popspa.proxy");
        if (update_dual) {
            dual_kernel<<<(n + 255) / 256, 256, 0, stream>>>(raw, z, u, active, frozen, n);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.popspa.dual");
        }
    }
    void launch_popspa_penalty(const float* raw, const float* z, const float* u,
                               const bool* active, const bool* frozen, float* grad,
                               float* loss, size_t n, float rho, cudaStream_t stream) {
        stream = resolve_stream(stream);
        penalty_kernel<<<(n + 255) / 256, 256, 0, stream>>>(raw, z, u, active, frozen, grad, loss, n, rho);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.popspa.penalty");
    }
} // namespace lfs::training::kernels
