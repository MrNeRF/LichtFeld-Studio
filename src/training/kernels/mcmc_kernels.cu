/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include "core/tensor.hpp"
#include "lfs/cuda_scratch.hpp"
#include "mcmc_kernels.hpp"
#include <cassert>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <limits>
#include <thrust/adjacent_difference.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "kernel_stream.hpp"

namespace lfs::training::mcmc {

    // GLM type aliases for CUDA (matching gsplat)
    using vec2 = glm::vec<2, float>;
    using vec3 = glm::vec<3, float>;
    using vec4 = glm::vec<4, float>;
    using mat3 = glm::mat<3, 3, float>;

    constexpr int RELOCATION_N_MAX = 51;
    __constant__ float d_relocation_coefficients[RELOCATION_N_MAX * RELOCATION_N_MAX];

    void init_relocation_coefficients(int n_max) {
        assert(n_max <= RELOCATION_N_MAX);
        std::vector<float> coeffs(RELOCATION_N_MAX * RELOCATION_N_MAX, 0.0f);
        for (int n = 0; n < n_max; n++) {
            float binom = 1.0f;
            for (int k = 0; k <= n; k++) {
                const float sign = (k % 2 == 0) ? 1.0f : -1.0f;
                coeffs[n * RELOCATION_N_MAX + k] = binom * sign * rsqrtf(static_cast<float>(k + 1));
                if (k < n)
                    binom *= static_cast<float>(n - k) / static_cast<float>(k + 1);
            }
        }
        LFS_CUDA_CHECK_MSG(
            cudaMemcpyToSymbol(d_relocation_coefficients, coeffs.data(),
                               RELOCATION_N_MAX * RELOCATION_N_MAX * sizeof(float)),
            "MCMC relocation coefficient upload (n_max={})", n_max);
    }

    // Equation (9) in "3D Gaussian Splatting as Markov Chain Monte Carlo"
    __global__ void relocation_kernel(
        const float* __restrict__ opacities,
        const float* __restrict__ scales,
        const int32_t* __restrict__ ratios,
        const float min_opacity,
        float* __restrict__ new_opacities,
        float* __restrict__ new_scales,
        const size_t N) {

        const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= N)
            return;

        constexpr float OPACITY_MIN = 1e-6f;
        constexpr float OPACITY_MAX = 1.0f - 1e-6f;
        constexpr float DENOM_MIN = 1e-8f;
        constexpr float COEFF_MAX = 1e6f;
        constexpr float SCALE_MIN = 1e-10f;

        const int n_idx = ratios[idx];
        const float opacity = fminf(fmaxf(opacities[idx], OPACITY_MIN), OPACITY_MAX);

        // new_opacity = 1 - (1 - opacity)^(1/n_idx)
        const float new_opacity = fminf(fmaxf(
                                            1.0f - powf(1.0f - opacity, 1.0f / static_cast<float>(n_idx)),
                                            fmaxf(OPACITY_MIN, min_opacity)),
                                        OPACITY_MAX);
        new_opacities[idx] = new_opacity;

        // Compute denominator sum using pre-computed coefficients from __constant__ memory
        float denom_sum = 0.0f;
        for (int i = 1; i <= n_idx; ++i) {
            for (int k = 0; k <= (i - 1); ++k) {
                denom_sum += d_relocation_coefficients[(i - 1) * RELOCATION_N_MAX + k] *
                             powf(new_opacity, static_cast<float>(k + 1));
            }
        }

        // Safe division with sign preservation
        float safe_denom = fmaxf(fabsf(denom_sum), DENOM_MIN);
        if (denom_sum < 0.0f)
            safe_denom = -safe_denom;

        const float coeff = fminf(fmaxf(opacity / safe_denom, -COEFF_MAX), COEFF_MAX);

        for (int i = 0; i < 3; ++i) {
            new_scales[idx * 3 + i] = fmaxf(fabsf(coeff * scales[idx * 3 + i]), SCALE_MIN);
        }
    }

    void launch_relocation_kernel(
        const float* opacities,
        const float* scales,
        const int32_t* ratios,
        float min_opacity,
        float* new_opacities,
        float* new_scales,
        size_t N,
        void* stream) {

        if (N == 0) {
            return;
        }

        dim3 threads(256);
        dim3 grid((N + threads.x - 1) / threads.x);

        cudaStream_t cuda_stream = resolve_stream(stream);

        relocation_kernel<<<grid, threads, 0, cuda_stream>>>(
            opacities,
            scales,
            ratios,
            min_opacity,
            new_opacities,
            new_scales,
            N);
        LFS_CUDA_LAUNCH_CHECK(cuda_stream, "training.mcmc.relocation");
    }

    // Helper: Quaternion to rotation matrix
    __device__ inline mat3 raw_quat_to_rotmat(const vec4 raw_quat) {
        float w = raw_quat[0], x = raw_quat[1], y = raw_quat[2], z = raw_quat[3];
        // normalize
        float inv_norm = fminf(rsqrt(x * x + y * y + z * z + w * w), 1e+12f); // match torch normalize
        x *= inv_norm;
        y *= inv_norm;
        z *= inv_norm;
        w *= inv_norm;
        float x2 = x * x, y2 = y * y, z2 = z * z;
        float xy = x * y, xz = x * z, yz = y * z;
        float wx = w * x, wy = w * y, wz = w * z;
        return mat3(
            (1.f - 2.f * (y2 + z2)),
            (2.f * (xy + wz)),
            (2.f * (xz - wy)), // 1st col
            (2.f * (xy - wz)),
            (1.f - 2.f * (x2 + z2)),
            (2.f * (yz + wx)), // 2nd col
            (2.f * (xz + wy)),
            (2.f * (yz - wx)),
            (1.f - 2.f * (x2 + y2)) // 3rd col
        );
    }

    __global__ void add_noise_kernel(
        const float* raw_opacities,
        const float* raw_scales,
        const float* raw_quats,
        const float* noise,
        float* means,
        const bool* frozen_mask,
        size_t frozen_mask_size,
        float current_lr,
        size_t N) {

        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= N)
            return;
        if (frozen_mask != nullptr && idx < frozen_mask_size && frozen_mask[idx])
            return;

        size_t idx_3d = 3 * idx;

        // Compute S^2 (diagonal matrix from exp(2 * raw_scale))
        const vec3 raw_scale = glm::make_vec3(raw_scales + idx_3d);
        mat3 S2 = mat3(__expf(2.f * raw_scale[0]), 0.f, 0.f,
                       0.f, __expf(2.f * raw_scale[1]), 0.f,
                       0.f, 0.f, __expf(2.f * raw_scale[2]));

        // Get rotation matrix R from quaternion
        vec4 raw_quat = glm::make_vec4(raw_quats + 4 * idx);
        mat3 R = raw_quat_to_rotmat(raw_quat);

        // Compute covariance = R * S^2 * R^T
        mat3 covariance = R * S2 * glm::transpose(R);

        // Transform noise: transformed_noise = covariance * noise
        vec3 transformed_noise = covariance * glm::make_vec3(noise + idx_3d);

        // Compute opacity-based scaling factor
        float opacity = __frcp_rn(1.f + __expf(-raw_opacities[idx]));
        float op_sigmoid = __frcp_rn(1.f + __expf(100.f * opacity - 0.5f));
        float noise_factor = current_lr * op_sigmoid;

        // Add scaled noise to means
        means[idx_3d] += noise_factor * transformed_noise.x;
        means[idx_3d + 1] += noise_factor * transformed_noise.y;
        means[idx_3d + 2] += noise_factor * transformed_noise.z;
    }

    void launch_add_noise_kernel(
        const float* raw_opacities,
        const float* raw_scales,
        const float* raw_quats,
        const float* noise,
        float* means,
        const bool* frozen_mask,
        size_t frozen_mask_size,
        float current_lr,
        size_t N,
        void* stream) {

        if (N == 0) {
            return;
        }

        dim3 threads(256);
        dim3 grid((N + threads.x - 1) / threads.x);

        cudaStream_t cuda_stream = resolve_stream(stream);

        add_noise_kernel<<<grid, threads, 0, cuda_stream>>>(
            raw_opacities,
            raw_scales,
            raw_quats,
            noise,
            means,
            frozen_mask,
            frozen_mask_size,
            current_lr,
            N);
        LFS_CUDA_LAUNCH_CHECK(cuda_stream, "training.mcmc.add_noise");
    }

    // Kernel to update scaling and opacity at specific indices (avoids index_put_ which loses capacity)
    __global__ void update_scaling_opacity_kernel(
        const int64_t* __restrict__ indices,
        const float* __restrict__ new_scaling,     // [n_indices, 3]
        const float* __restrict__ new_opacity_raw, // [n_indices] or [n_indices, 1]
        float* __restrict__ scaling_raw,
        float* __restrict__ opacity_raw,
        size_t n_indices,
        int opacity_dim,
        size_t N) {

        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= n_indices)
            return;

        int64_t target_idx = indices[idx];

        // Bounds check
        if (target_idx < 0 || target_idx >= static_cast<int64_t>(N)) {
            return;
        }

        // Update scaling [3]
        for (int i = 0; i < 3; ++i) {
            scaling_raw[target_idx * 3 + i] = new_scaling[idx * 3 + i];
        }

        // Update opacity
        opacity_raw[target_idx] = new_opacity_raw[idx];
    }

    void launch_update_scaling_opacity(
        const int64_t* indices,
        const float* new_scaling,
        const float* new_opacity_raw,
        float* scaling_raw,
        float* opacity_raw,
        size_t n_indices,
        int opacity_dim,
        size_t N,
        void* stream) {

        if (n_indices == 0) {
            return;
        }

        dim3 threads(256);
        dim3 grid((n_indices + threads.x - 1) / threads.x);

        cudaStream_t cuda_stream = resolve_stream(stream);

        update_scaling_opacity_kernel<<<grid, threads, 0, cuda_stream>>>(
            indices,
            new_scaling,
            new_opacity_raw,
            scaling_raw,
            opacity_raw,
            n_indices,
            opacity_dim,
            N);
        LFS_CUDA_LAUNCH_CHECK(cuda_stream, "training.mcmc.update_scaling_opacity");
    }

    // Fused copy kernel - copies all parameters from src_indices to dst_indices
    __global__ void copy_gaussian_params_kernel(
        const int64_t* __restrict__ src_indices,
        const int64_t* __restrict__ dst_indices,
        float* __restrict__ means,
        float* __restrict__ sh0,
        float* __restrict__ shN,
        float* __restrict__ scales,
        float* __restrict__ rotations,
        float* __restrict__ opacities,
        size_t n_copy,
        size_t sh_rest,
        int opacity_dim,
        size_t N) { // Add N parameter for bounds checking

        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= n_copy)
            return;

        int64_t src_idx = src_indices[idx];
        int64_t dst_idx = dst_indices[idx];

        // Bounds check - CRITICAL for safety
        if (src_idx < 0 || src_idx >= static_cast<int64_t>(N) ||
            dst_idx < 0 || dst_idx >= static_cast<int64_t>(N)) {
            // Invalid index - skip this copy
            return;
        }

        // Copy means [3]
        for (int i = 0; i < 3; ++i) {
            means[dst_idx * 3 + i] = means[src_idx * 3 + i];
        }

        // Copy sh0 [1, 3] -> [3]
        for (int i = 0; i < 3; ++i) {
            sh0[dst_idx * 3 + i] = sh0[src_idx * 3 + i];
        }

        // Copy shN [sh_rest, 3]
        for (size_t i = 0; i < sh_rest * 3; ++i) {
            shN[dst_idx * sh_rest * 3 + i] = shN[src_idx * sh_rest * 3 + i];
        }

        // Copy scales [3]
        for (int i = 0; i < 3; ++i) {
            scales[dst_idx * 3 + i] = scales[src_idx * 3 + i];
        }

        // Copy rotations [4]
        for (int i = 0; i < 4; ++i) {
            rotations[dst_idx * 4 + i] = rotations[src_idx * 4 + i];
        }

        // Copy opacities [1] or []
        if (opacity_dim == 1) {
            opacities[dst_idx] = opacities[src_idx];
        } else {
            opacities[dst_idx] = opacities[src_idx];
        }
    }

    void launch_copy_gaussian_params(
        const int64_t* src_indices,
        const int64_t* dst_indices,
        float* means,
        float* sh0,
        float* shN,
        float* scales,
        float* rotations,
        float* opacities,
        size_t n_copy,
        size_t sh_rest,
        int opacity_dim,
        size_t N, // Add N parameter
        void* stream) {

        if (n_copy == 0) {
            return;
        }

        dim3 threads(256);
        dim3 grid((n_copy + threads.x - 1) / threads.x);

        cudaStream_t cuda_stream = resolve_stream(stream);

        copy_gaussian_params_kernel<<<grid, threads, 0, cuda_stream>>>(
            src_indices,
            dst_indices,
            means,
            sh0,
            shN,
            scales,
            rotations,
            opacities,
            n_copy,
            sh_rest,
            opacity_dim,
            N);
        LFS_CUDA_LAUNCH_CHECK(cuda_stream, "training.mcmc.copy_params");
    }

    // ============================================================================
    // FUSED: Multinomial Sample + Gather
    // ============================================================================

    /**
     * Fused multinomial sampling + gather kernel
     *
     * Performs multinomial sampling from opacities[alive_indices] and directly
     * gathers the results WITHOUT any intermediate tensor allocations.
     *
     * Each thread:
     * 1. Generates a random sample u ∈ [0, prob_sum]
     * 2. Performs cumulative sum search through opacities[alive_indices[i]]
     * 3. Finds the index where cumsum >= u
     * 4. Outputs the global index and directly gathers opacity/scales
     */
    __global__ void multinomial_sample_and_gather_kernel(
        const float* __restrict__ opacities,
        const float* __restrict__ scaling_raw, // raw scaling, exp() applied inline
        const int64_t* __restrict__ alive_indices,
        const float* __restrict__ cumsum,
        size_t n_alive,
        size_t n_samples,
        uint64_t seed,
        int64_t* __restrict__ sampled_global_indices,
        float* __restrict__ sampled_opacities,
        float* __restrict__ sampled_scales,
        size_t N) {

        const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= n_samples)
            return;

        // The inclusive cumsum's last element is the total probability mass —
        // no separate reduction or host readback needed.
        const float prob_sum = cumsum[n_alive - 1];
        if (prob_sum <= 0.0f) {
            sampled_global_indices[idx] = 0;
            sampled_opacities[idx] = 0.0f;
            sampled_scales[idx * 3 + 0] = 0.0f;
            sampled_scales[idx * 3 + 1] = 0.0f;
            sampled_scales[idx * 3 + 2] = 0.0f;
            return;
        }

        curandState state;
        curand_init(seed, idx, 0, &state);

        const float u = curand_uniform(&state) * prob_sum;

        int64_t left = 0;
        int64_t right = static_cast<int64_t>(n_alive) - 1;
        int64_t selected_idx = static_cast<int64_t>(n_alive) - 1;

        while (left <= right) {
            const int64_t mid = (left + right) / 2;
            if (cumsum[mid] >= u) {
                selected_idx = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        const int64_t selected_global_idx = alive_indices[selected_idx];

        sampled_global_indices[idx] = selected_global_idx;
        sampled_opacities[idx] = opacities[selected_global_idx];
        sampled_scales[idx * 3 + 0] = expf(scaling_raw[selected_global_idx * 3 + 0]);
        sampled_scales[idx * 3 + 1] = expf(scaling_raw[selected_global_idx * 3 + 1]);
        sampled_scales[idx * 3 + 2] = expf(scaling_raw[selected_global_idx * 3 + 2]);
    }

    void launch_multinomial_sample_and_gather(
        const float* sampling_weights,
        const float* opacities,
        const float* scaling_raw,
        const int64_t* alive_indices,
        size_t n_alive,
        size_t n_samples,
        uint64_t seed,
        int64_t* sampled_global_indices,
        float* sampled_opacities,
        float* sampled_scales,
        size_t N,
        void* stream) {

        if (n_samples == 0 || n_alive == 0)
            return;

        LFS_ASSERT_MSG(n_alive <= static_cast<size_t>(std::numeric_limits<int>::max()),
                       "MCMC multinomial input exceeds CUB's int item-count limit");

        const cudaStream_t cuda_stream = resolve_stream(stream);
        // Home the scan/sampling temporaries on the launch stream so the
        // stream-aware pool cannot recycle them before this work completes.
        const lfs::core::CUDAStreamGuard stream_guard(cuda_stream);

        auto alive_probs = lfs::core::Tensor::empty({n_alive}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        auto cumsum_buf = lfs::core::Tensor::empty({n_alive}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);

        thrust::transform(thrust::cuda::par.on(cuda_stream),
                          thrust::counting_iterator<int>(0),
                          thrust::counting_iterator<int>(n_alive),
                          thrust::device_ptr<float>(alive_probs.ptr<float>()),
                          [=] __device__(int i) { return sampling_weights[alive_indices[i]]; });

        cuda_scratch::CubWorkspace cub_workspace(
            "cub::DeviceScan::InclusiveSum", cuda_stream,
            [&](void* workspace, size_t& workspace_bytes) {
                return cub::DeviceScan::InclusiveSum(
                    workspace, workspace_bytes,
                    alive_probs.ptr<float>(), cumsum_buf.ptr<float>(), n_alive, cuda_stream);
            });
        cub_workspace.run([&](void* workspace, size_t& workspace_bytes) {
            return cub::DeviceScan::InclusiveSum(
                workspace, workspace_bytes,
                alive_probs.ptr<float>(), cumsum_buf.ptr<float>(), n_alive, cuda_stream);
        });

        const dim3 sample_threads(256);
        const dim3 sample_grid((n_samples + sample_threads.x - 1) / sample_threads.x);

        multinomial_sample_and_gather_kernel<<<sample_grid, sample_threads, 0, cuda_stream>>>(
            opacities,
            scaling_raw,
            alive_indices,
            cumsum_buf.ptr<float>(),
            n_alive,
            n_samples,
            seed,
            sampled_global_indices,
            sampled_opacities,
            sampled_scales,
            N);
        LFS_CUDA_CHECK_MSG(cudaGetLastError(), "MCMC multinomial gather kernel launch");
    }

    // Multinomial sampling from all N weights (no alive_indices indirection)
    __global__ void multinomial_sample_all_kernel(
        const float* __restrict__ opacities,
        const float* __restrict__ scaling_raw, // raw scaling, exp() applied inline
        const float* __restrict__ cumsum,
        size_t N,
        size_t n_samples,
        uint64_t seed,
        int64_t* __restrict__ sampled_indices,
        float* __restrict__ sampled_opacities,
        float* __restrict__ sampled_scales) {

        const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= n_samples)
            return;

        const float prob_sum = cumsum[N - 1];
        if (prob_sum <= 0.0f) {
            sampled_indices[idx] = 0;
            sampled_opacities[idx] = 0.0f;
            sampled_scales[idx * 3 + 0] = 0.0f;
            sampled_scales[idx * 3 + 1] = 0.0f;
            sampled_scales[idx * 3 + 2] = 0.0f;
            return;
        }

        curandState state;
        curand_init(seed, idx, 0, &state);

        const float u = curand_uniform(&state) * prob_sum;

        int64_t left = 0;
        int64_t right = static_cast<int64_t>(N) - 1;
        int64_t selected_idx = static_cast<int64_t>(N) - 1;

        while (left <= right) {
            const int64_t mid = (left + right) / 2;
            if (cumsum[mid] >= u) {
                selected_idx = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        sampled_indices[idx] = selected_idx;
        sampled_opacities[idx] = opacities[selected_idx];
        sampled_scales[idx * 3 + 0] = expf(scaling_raw[selected_idx * 3 + 0]);
        sampled_scales[idx * 3 + 1] = expf(scaling_raw[selected_idx * 3 + 1]);
        sampled_scales[idx * 3 + 2] = expf(scaling_raw[selected_idx * 3 + 2]);
    }

    void launch_multinomial_sample_all(
        const float* sampling_weights,
        const float* opacities,
        const float* scaling_raw,
        size_t N,
        size_t n_samples,
        uint64_t seed,
        int64_t* sampled_indices,
        float* sampled_opacities,
        float* sampled_scales,
        void* stream) {

        if (n_samples == 0 || N == 0)
            return;

        LFS_ASSERT_MSG(N <= static_cast<size_t>(std::numeric_limits<int>::max()),
                       "MCMC multinomial input exceeds CUB's int item-count limit");

        const cudaStream_t cuda_stream = resolve_stream(stream);
        // Home the scan temporary on the launch stream (see launch_multinomial_sample).
        const lfs::core::CUDAStreamGuard stream_guard(cuda_stream);

        auto cumsum_buf = lfs::core::Tensor::empty({N}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);

        cuda_scratch::CubWorkspace cub_workspace(
            "cub::DeviceScan::InclusiveSum", cuda_stream,
            [&](void* workspace, size_t& workspace_bytes) {
                return cub::DeviceScan::InclusiveSum(
                    workspace, workspace_bytes,
                    sampling_weights, cumsum_buf.ptr<float>(), N, cuda_stream);
            });
        cub_workspace.run([&](void* workspace, size_t& workspace_bytes) {
            return cub::DeviceScan::InclusiveSum(
                workspace, workspace_bytes,
                sampling_weights, cumsum_buf.ptr<float>(), N, cuda_stream);
        });

        const dim3 sample_threads(256);
        const dim3 sample_grid((n_samples + sample_threads.x - 1) / sample_threads.x);

        multinomial_sample_all_kernel<<<sample_grid, sample_threads, 0, cuda_stream>>>(
            opacities,
            scaling_raw,
            cumsum_buf.ptr<float>(),
            N,
            n_samples,
            seed,
            sampled_indices,
            sampled_opacities,
            sampled_scales);
        LFS_CUDA_CHECK_MSG(cudaGetLastError(), "MCMC multinomial kernel launch");
    }

    __global__ void elementwise_max_inplace_kernel(
        float* __restrict__ a,
        const float* __restrict__ b,
        size_t N) {

        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= N)
            return;

        a[idx] = fmaxf(a[idx], b[idx]);
    }

    void launch_elementwise_max_inplace(
        float* a,
        const float* b,
        size_t N,
        void* stream) {

        if (N == 0)
            return;

        dim3 threads(256);
        dim3 grid((N + threads.x - 1) / threads.x);

        cudaStream_t cuda_stream = resolve_stream(stream);

        elementwise_max_inplace_kernel<<<grid, threads, 0, cuda_stream>>>(a, b, N);
        LFS_CUDA_LAUNCH_CHECK(cuda_stream, "training.mcmc.elementwise_max");
    }

} // namespace lfs::training::mcmc
