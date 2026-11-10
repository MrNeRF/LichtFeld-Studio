/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include "core/tensor.hpp"
#include "mcmc_kernels.hpp"
#include "training/kernels/cuda_scratch.hpp"
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
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
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
        const auto stream = lfs::core::getCurrentCUDAStream();
        LFS_CUDA_CHECK_MSG(
            cudaMemcpyToSymbolAsync(d_relocation_coefficients, coeffs.data(),
                                    RELOCATION_N_MAX * RELOCATION_N_MAX * sizeof(float),
                                    0, cudaMemcpyHostToDevice, stream),
            "MCMC relocation coefficient upload (n_max={})", n_max);
        LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Equation (9) of "3D Gaussian Splatting as Markov Chain Monte Carlo": a
    // Gaussian sampled n_idx times becomes n_idx copies that together render
    // like the original.
    __device__ __forceinline__ void relocated_opacity_scale(
        const float source_opacity,
        const float* source_scale,
        const int n_idx,
        const float min_opacity,
        float& new_opacity,
        float* new_scale) {

        constexpr float OPACITY_MIN = 1e-6f;
        constexpr float OPACITY_MAX = 1.0f - 1e-6f;
        constexpr float DENOM_MIN = 1e-8f;
        constexpr float COEFF_MAX = 1e6f;
        constexpr float SCALE_MIN = 1e-10f;

        const float opacity = fminf(fmaxf(source_opacity, OPACITY_MIN), OPACITY_MAX);

        // new_opacity = 1 - (1 - opacity)^(1/n_idx)
        new_opacity = fminf(fmaxf(
                                1.0f - powf(1.0f - opacity, 1.0f / static_cast<float>(n_idx)),
                                fmaxf(OPACITY_MIN, min_opacity)),
                            OPACITY_MAX);

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
            new_scale[i] = fmaxf(fabsf(coeff * source_scale[i]), SCALE_MIN);
        }
    }

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
        relocated_opacity_scale(opacities[idx], scales + idx * 3, ratios[idx], min_opacity,
                                new_opacities[idx], new_scales + idx * 3);
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

    // Each copy row still holds its source's raw values. The relocated values
    // go to the copy and to the source; copies of one source write the same
    // values, and no thread reads a source row.
    __global__ void relocate_copies_kernel(
        const int64_t* __restrict__ src_indices,
        const int64_t* __restrict__ copy_indices,
        const int64_t first_copy,
        const int32_t* __restrict__ ratios,
        const float min_opacity,
        float* __restrict__ scaling_raw,
        float* __restrict__ opacity_raw,
        const size_t n,
        const size_t N) {

        const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= n)
            return;

        const int64_t src = src_indices[idx];
        const int64_t copy = copy_indices ? copy_indices[idx] : first_copy + static_cast<int64_t>(idx);
        if (src < 0 || src >= static_cast<int64_t>(N) || copy < 0 || copy >= static_cast<int64_t>(N)) {
            return;
        }

        // Same expressions as SplatData::get_opacity(), get_scaling(), and
        // Tensor::clamp/logit/log, so results match the tensor path bitwise.
        const float source_opacity = 1.0f / (1.0f + expf(-opacity_raw[copy]));
        float source_scale[3];
        for (int i = 0; i < 3; ++i) {
            source_scale[i] = expf(scaling_raw[copy * 3 + i]);
        }
        float new_opacity;
        float new_scale[3];
        relocated_opacity_scale(source_opacity, source_scale, ratios[idx], min_opacity, new_opacity, new_scale);

        constexpr float LOGIT_EPS = 1e-7f;
        new_opacity = fminf(fmaxf(new_opacity, min_opacity), 1.0f - LOGIT_EPS);
        new_opacity = fminf(fmaxf(new_opacity, LOGIT_EPS), 1.0f - LOGIT_EPS);
        const float raw_opacity = logf(new_opacity / (1.0f - new_opacity));
        opacity_raw[copy] = raw_opacity;
        opacity_raw[src] = raw_opacity;
        for (int i = 0; i < 3; ++i) {
            const float raw_scale = logf(new_scale[i]);
            scaling_raw[copy * 3 + i] = raw_scale;
            scaling_raw[src * 3 + i] = raw_scale;
        }
    }

    void launch_relocate_copies(
        const int64_t* src_indices,
        const int64_t* copy_indices,
        int64_t first_copy,
        const int32_t* ratios,
        float min_opacity,
        float* scaling_raw,
        float* opacity_raw,
        size_t n,
        size_t N,
        void* stream) {

        if (n == 0) {
            return;
        }

        dim3 threads(256);
        dim3 grid((n + threads.x - 1) / threads.x);

        cudaStream_t cuda_stream = resolve_stream(stream);

        relocate_copies_kernel<<<grid, threads, 0, cuda_stream>>>(
            src_indices,
            copy_indices,
            first_copy,
            ratios,
            min_opacity,
            scaling_raw,
            opacity_raw,
            n,
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

    __device__ __forceinline__ void apply_noise_to_mean(
        const float* raw_opacities,
        const float* raw_scales,
        const float* raw_quats,
        float* means,
        size_t idx,
        float current_lr,
        float noise_x,
        float noise_y,
        float noise_z) {
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
        vec3 transformed_noise = covariance * vec3(noise_x, noise_y, noise_z);

        // Compute opacity-based scaling factor
        float opacity = __frcp_rn(1.f + __expf(-raw_opacities[idx]));
        float op_sigmoid = __frcp_rn(1.f + __expf(100.f * opacity - 0.5f));
        float noise_factor = current_lr * op_sigmoid;

        // Add scaled noise to means
        means[idx_3d] += noise_factor * transformed_noise.x;
        means[idx_3d + 1] += noise_factor * transformed_noise.y;
        means[idx_3d + 2] += noise_factor * transformed_noise.z;
    }

    __global__ void inject_noise_kernel(
        const float* raw_opacities,
        const float* raw_scales,
        const float* raw_quats,
        float* means,
        const bool* frozen_mask,
        size_t frozen_mask_size,
        float current_lr,
        size_t N,
        uint64_t seed) {

        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= N)
            return;
        if (frozen_mask != nullptr && idx < frozen_mask_size && frozen_mask[idx])
            return;

        // Philox: free counter init vs XORWOW skip-ahead (1149→6.5µs).
        curandStatePhilox4_32_10_t rng;
        curand_init(seed, idx, 0, &rng);
        const float4 n = curand_normal4(&rng);

        apply_noise_to_mean(
            raw_opacities, raw_scales, raw_quats, means, idx, current_lr,
            n.x, n.y, n.z);
    }

    void launch_inject_noise_kernel(
        const float* raw_opacities,
        const float* raw_scales,
        const float* raw_quats,
        float* means,
        const bool* frozen_mask,
        size_t frozen_mask_size,
        float current_lr,
        size_t N,
        uint64_t seed,
        void* stream) {

        if (N == 0) {
            return;
        }

        dim3 threads(256);
        dim3 grid((N + threads.x - 1) / threads.x);
        cudaStream_t cuda_stream = resolve_stream(stream);

        inject_noise_kernel<<<grid, threads, 0, cuda_stream>>>(
            raw_opacities,
            raw_scales,
            raw_quats,
            means,
            frozen_mask,
            frozen_mask_size,
            current_lr,
            N,
            seed);
        LFS_CUDA_LAUNCH_CHECK(cuda_stream, "training.mcmc.inject_noise");
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

        curandStatePhilox4_32_10_t state;
        curand_init(seed, idx, 0, &state);

        // Rescale small CDFs by an exact power of two so a positive draw
        // stays normal under fast math and cannot select a leading zero weight.
        const float probability_scale = prob_sum < 1.0f ? 0x1p64f : 1.0f;
        const float u = curand_uniform(&state) * (prob_sum * probability_scale);

        int64_t left = 0;
        int64_t right = static_cast<int64_t>(n_alive) - 1;
        int64_t selected_idx = static_cast<int64_t>(n_alive) - 1;

        while (left <= right) {
            const int64_t mid = (left + right) / 2;
            if (cumsum[mid] * probability_scale >= u) {
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

        auto alive_probs = lfs::core::Tensor::empty({n_alive}, lfs::core::Device::GPU, lfs::core::DataType::Float32);
        auto cumsum_buf = lfs::core::Tensor::empty({n_alive}, lfs::core::Device::GPU, lfs::core::DataType::Float32);

        thrust::transform(thrust::cuda::par_nosync.on(cuda_stream),
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

        curandStatePhilox4_32_10_t state;
        curand_init(seed, idx, 0, &state);

        // Rescale small CDFs by an exact power of two so a positive draw
        // stays normal under fast math and cannot select a leading zero weight.
        const float probability_scale = prob_sum < 1.0f ? 0x1p64f : 1.0f;
        const float u = curand_uniform(&state) * (prob_sum * probability_scale);

        int64_t left = 0;
        int64_t right = static_cast<int64_t>(N) - 1;
        int64_t selected_idx = static_cast<int64_t>(N) - 1;

        while (left <= right) {
            const int64_t mid = (left + right) / 2;
            if (cumsum[mid] * probability_scale >= u) {
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

        auto cumsum_buf = lfs::core::Tensor::empty({N}, lfs::core::Device::GPU, lfs::core::DataType::Float32);

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

    // Weight a row contributes to the CDF: excluded rows contribute nothing.
    // Row i's weight floored like Tensor::clamp_min (NaN stays NaN), or 0
    // when the row is excluded.
    struct SampleableWeight {
        const float* weights;
        float min_weight;
        const uint8_t* excluded;
        __host__ __device__ float operator()(const int64_t i) const {
            if (excluded && excluded[i]) {
                return 0.0f;
            }
            const float weight = weights[i];
            return isnan(weight) ? weight : fmaxf(weight, min_weight);
        }
    };

    // Multinomial sampling from all N weights, skipping rows that cannot be sampled
    __global__ void multinomial_sample_rows_kernel(
        const SampleableWeight weight_of,
        const float* __restrict__ cumsum,
        size_t N,
        size_t n_samples,
        uint64_t seed,
        int64_t* __restrict__ sampled_indices) {

        const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= n_samples)
            return;

        const float prob_sum = cumsum[N - 1];
        if (prob_sum <= 0.0f) {
            sampled_indices[idx] = 0;
            return;
        }

        curandStatePhilox4_32_10_t state;
        curand_init(seed, idx, 0, &state);

        // Rescale small CDFs by an exact power of two so a positive draw
        // stays normal under fast math and cannot select a leading zero weight.
        const float probability_scale = prob_sum < 1.0f ? 0x1p64f : 1.0f;
        const float u = curand_uniform(&state) * (prob_sum * probability_scale);

        int64_t left = 0;
        int64_t right = static_cast<int64_t>(N) - 1;
        int64_t selected_idx = static_cast<int64_t>(N) - 1;

        while (left <= right) {
            const int64_t mid = (left + right) / 2;
            if (cumsum[mid] * probability_scale >= u) {
                selected_idx = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        // Tiles of the float scan round differently, so the CDF can step up
        // by an ulp at a zero-weight row. Never return a row that cannot be
        // sampled; the nearest sampleable neighbour owns that step.
        const auto sampleable = [&](const int64_t i) {
            return weight_of(i) > 0.0f;
        };
        if (!sampleable(selected_idx)) {
            constexpr int64_t kMaxWalk = 64;
            int64_t found = -1;
            for (int64_t i = selected_idx + 1; i < static_cast<int64_t>(N) && i <= selected_idx + kMaxWalk; ++i) {
                if (sampleable(i)) {
                    found = i;
                    break;
                }
            }
            for (int64_t i = selected_idx - 1; found < 0 && i >= 0 && i >= selected_idx - kMaxWalk; --i) {
                if (sampleable(i)) {
                    found = i;
                }
            }
            if (found >= 0) {
                selected_idx = found;
            }
        }

        sampled_indices[idx] = selected_idx;
    }

    bool launch_multinomial_sample_all(
        const float* sampling_weights,
        const float min_weight,
        const uint8_t* excluded,
        size_t N,
        size_t n_samples,
        uint64_t seed,
        int64_t* sampled_indices,
        void* stream) {

        if (n_samples == 0 || N == 0)
            return false;

        LFS_ASSERT_MSG(N <= static_cast<size_t>(std::numeric_limits<int>::max()),
                       "MCMC multinomial input exceeds CUB's int item-count limit");

        const cudaStream_t cuda_stream = resolve_stream(stream);
        // Home the scan temporary on the launch stream so the stream-aware
        // pool cannot recycle it before the sampling kernel completes.
        const lfs::core::CUDAStreamGuard stream_guard(cuda_stream);

        auto cumsum_buf = lfs::core::Tensor::empty({N}, lfs::core::Device::GPU, lfs::core::DataType::Float32);

        const auto scan = [&](const auto input) {
            const auto inclusive_sum = [&](void* workspace, size_t& workspace_bytes) {
                return cub::DeviceScan::InclusiveSum(
                    workspace, workspace_bytes, input, cumsum_buf.ptr<float>(), static_cast<int>(N), cuda_stream);
            };
            cuda_scratch::CubWorkspace cub_workspace("cub::DeviceScan::InclusiveSum", cuda_stream, inclusive_sum);
            cub_workspace.run(inclusive_sum);
        };
        const SampleableWeight weight_of{sampling_weights, min_weight, excluded};
        if (excluded || min_weight != -std::numeric_limits<float>::infinity()) {
            scan(thrust::make_transform_iterator(thrust::counting_iterator<int64_t>(0), weight_of));
        } else {
            scan(sampling_weights);
        }

        // Weights are finite and non-negative, so the total is positive
        // exactly when some row can be sampled.
        float total = 0.0f;
        LFS_CUDA_CHECK_MSG(cudaMemcpyAsync(&total, cumsum_buf.ptr<float>() + (N - 1), sizeof(float),
                                           cudaMemcpyDeviceToHost, cuda_stream),
                           "MCMC multinomial total readback");
        LFS_CUDA_CHECK_MSG(cudaStreamSynchronize(cuda_stream), "MCMC multinomial total sync");
        if (!(total > 0.0f)) {
            return false;
        }

        const dim3 sample_threads(256);
        const dim3 sample_grid((n_samples + sample_threads.x - 1) / sample_threads.x);

        multinomial_sample_rows_kernel<<<sample_grid, sample_threads, 0, cuda_stream>>>(
            weight_of,
            cumsum_buf.ptr<float>(),
            N,
            n_samples,
            seed,
            sampled_indices);
        LFS_CUDA_CHECK_MSG(cudaGetLastError(), "MCMC multinomial kernel launch");
        return true;
    }

    __global__ void max_error_and_zero_densification_kernel(
        float* __restrict__ error_max,
        float* __restrict__ error_row,
        float* __restrict__ weight_row,
        size_t N) {

        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= N)
            return;

        const float err = error_row[idx];
        error_max[idx] = fmaxf(error_max[idx], err);
        error_row[idx] = 0.f;
        if (weight_row != nullptr)
            weight_row[idx] = 0.f;
    }

    void launch_max_error_and_zero_densification(
        float* error_max,
        float* error_row,
        float* weight_row,
        size_t N,
        void* stream) {

        if (N == 0)
            return;

        dim3 threads(256);
        dim3 grid((N + threads.x - 1) / threads.x);
        cudaStream_t cuda_stream = resolve_stream(stream);

        max_error_and_zero_densification_kernel<<<grid, threads, 0, cuda_stream>>>(
            error_max, error_row, weight_row, N);
        LFS_CUDA_LAUNCH_CHECK(cuda_stream, "training.mcmc.max_error_and_zero_densif");
    }

    void launch_max_error_and_zero_densification(
        float* error_max,
        float* densification_info,
        size_t N,
        void* stream) {
        // [2, N]: the weight row, then the error row.
        launch_max_error_and_zero_densification(error_max, densification_info + N, densification_info, N, stream);
    }

} // namespace lfs::training::mcmc
