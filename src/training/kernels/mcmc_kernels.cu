/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "mcmc_kernels.hpp"
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <thrust/adjacent_difference.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace lfs::training::mcmc {

    // GLM type aliases for CUDA (matching gsplat)
    using vec2 = glm::vec<2, float>;
    using vec3 = glm::vec<3, float>;
    using vec4 = glm::vec<4, float>;
    using mat3 = glm::mat<3, 3, float>;

    // Equation (9) in "3D Gaussian Splatting as Markov Chain Monte Carlo"
    __global__ void relocation_kernel(
        const float* __restrict__ opacities,
        const float* __restrict__ scales,
        const int32_t* __restrict__ ratios,
        const float* __restrict__ binoms,
        const int n_max,
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
                                            OPACITY_MIN),
                                        OPACITY_MAX);
        new_opacities[idx] = new_opacity;

        // Compute denominator sum for scale calculation
        float denom_sum = 0.0f;
        for (int i = 1; i <= n_idx; ++i) {
            for (int k = 0; k <= (i - 1); ++k) {
                const float sign = (k & 1) ? -1.0f : 1.0f;
                const float term = (sign / sqrtf(static_cast<float>(k + 1))) *
                                   powf(new_opacity, static_cast<float>(k + 1));
                denom_sum += binoms[(i - 1) * n_max + k] * term;
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
        const float* binoms,
        int n_max,
        float* new_opacities,
        float* new_scales,
        size_t N,
        void* stream) {

        if (N == 0) {
            return;
        }

        dim3 threads(256);
        dim3 grid((N + threads.x - 1) / threads.x);

        cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

        relocation_kernel<<<grid, threads, 0, cuda_stream>>>(
            opacities,
            scales,
            ratios,
            binoms,
            n_max,
            new_opacities,
            new_scales,
            N);
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
        float current_lr,
        size_t N) {

        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= N)
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
        float current_lr,
        size_t N,
        void* stream) {

        if (N == 0) {
            return;
        }

        dim3 threads(256);
        dim3 grid((N + threads.x - 1) / threads.x);

        cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

        add_noise_kernel<<<grid, threads, 0, cuda_stream>>>(
            raw_opacities,
            raw_scales,
            raw_quats,
            noise,
            means,
            current_lr,
            N);
    }

    // Fused gather kernel - collects all parameters at once
    __global__ void gather_gaussian_params_kernel(
        const int64_t* __restrict__ indices,
        const float* __restrict__ src_means,
        const float* __restrict__ src_sh0,
        const float* __restrict__ src_shN,
        const float* __restrict__ src_scales,
        const float* __restrict__ src_rotations,
        const float* __restrict__ src_opacities,
        float* __restrict__ dst_means,
        float* __restrict__ dst_sh0,
        float* __restrict__ dst_shN,
        float* __restrict__ dst_scales,
        float* __restrict__ dst_rotations,
        float* __restrict__ dst_opacities,
        size_t n_samples,
        size_t sh_rest,
        int opacity_dim,
        size_t N) { // Add N parameter for bounds checking

        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= n_samples)
            return;

        int64_t src_idx = indices[idx];

        // Bounds check - CRITICAL for safety
        if (src_idx < 0 || src_idx >= static_cast<int64_t>(N)) {
            // Invalid index - skip this gather (leave output uninitialized or zero it)
            return;
        }

        // Gather means [3]
        for (int i = 0; i < 3; ++i) {
            dst_means[idx * 3 + i] = src_means[src_idx * 3 + i];
        }

        // Gather sh0 [N, 1, 3] -> output [n_samples, 1, 3]
        // Memory layout: each Gaussian has 1*3 = 3 floats
        for (int i = 0; i < 3; ++i) {
            dst_sh0[idx * 3 + i] = src_sh0[src_idx * 3 + i];
        }

        // Gather shN [sh_rest, 3]
        for (size_t i = 0; i < sh_rest * 3; ++i) {
            dst_shN[idx * sh_rest * 3 + i] = src_shN[src_idx * sh_rest * 3 + i];
        }

        // Gather scales [3]
        for (int i = 0; i < 3; ++i) {
            dst_scales[idx * 3 + i] = src_scales[src_idx * 3 + i];
        }

        // Gather rotations [4]
        for (int i = 0; i < 4; ++i) {
            dst_rotations[idx * 4 + i] = src_rotations[src_idx * 4 + i];
        }

        // Gather opacities [1] or []
        if (opacity_dim == 1) {
            dst_opacities[idx] = src_opacities[src_idx];
        } else {
            dst_opacities[idx] = src_opacities[src_idx];
        }
    }

    void launch_gather_gaussian_params(
        const int64_t* indices,
        const float* src_means,
        const float* src_sh0,
        const float* src_shN,
        const float* src_scales,
        const float* src_rotations,
        const float* src_opacities,
        float* dst_means,
        float* dst_sh0,
        float* dst_shN,
        float* dst_scales,
        float* dst_rotations,
        float* dst_opacities,
        size_t n_samples,
        size_t sh_rest,
        int opacity_dim,
        size_t N, // Add N parameter
        void* stream) {

        if (n_samples == 0) {
            return;
        }

        dim3 threads(256);
        dim3 grid((n_samples + threads.x - 1) / threads.x);

        cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

        gather_gaussian_params_kernel<<<grid, threads, 0, cuda_stream>>>(
            indices,
            src_means,
            src_sh0,
            src_shN,
            src_scales,
            src_rotations,
            src_opacities,
            dst_means,
            dst_sh0,
            dst_shN,
            dst_scales,
            dst_rotations,
            dst_opacities,
            n_samples,
            sh_rest,
            opacity_dim,
            N);
    }

    // Fused kernel: Compute raw opacity and scaling values (ZERO intermediate allocations)
    // Performs: clamp(opacity) -> inverse_sigmoid -> log -> optional unsqueeze
    //           log(scales)
    __global__ void compute_raw_values_kernel(
        const float* __restrict__ opacities, // [n]
        const float* __restrict__ scales,    // [n, 3]
        float* __restrict__ opacity_raw,     // [n] or [n, 1]
        float* __restrict__ scaling_raw,     // [n, 3]
        size_t n,
        float min_opacity,
        int opacity_dim) {

        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= n)
            return;

        // Clamp opacity
        float opacity_clamped = fminf(fmaxf(opacities[idx], min_opacity), 1.0f - 1e-7f);

        // Inverse sigmoid: log(opacity / (1 - opacity))
        float opacity_raw_val = logf(opacity_clamped / (1.0f - opacity_clamped));

        // Write opacity (handle both [n] and [n, 1] shapes)
        if (opacity_dim == 1) {
            opacity_raw[idx] = opacity_raw_val; // [n, 1] is still indexed linearly
        } else {
            opacity_raw[idx] = opacity_raw_val; // [n]
        }

        // Log of scales (3 values)
        for (int i = 0; i < 3; ++i) {
            scaling_raw[idx * 3 + i] = logf(scales[idx * 3 + i]);
        }
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

    void launch_compute_raw_values(
        const float* opacities,
        const float* scales,
        float* opacity_raw,
        float* scaling_raw,
        size_t n,
        float min_opacity,
        int opacity_dim,
        void* stream) {

        if (n == 0) {
            return;
        }

        dim3 threads(256);
        dim3 grid((n + threads.x - 1) / threads.x);

        cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

        compute_raw_values_kernel<<<grid, threads, 0, cuda_stream>>>(
            opacities,
            scales,
            opacity_raw,
            scaling_raw,
            n,
            min_opacity,
            opacity_dim);
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

        cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

        update_scaling_opacity_kernel<<<grid, threads, 0, cuda_stream>>>(
            indices,
            new_scaling,
            new_opacity_raw,
            scaling_raw,
            opacity_raw,
            n_indices,
            opacity_dim,
            N);
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

        cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

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
    }

    // Histogram kernel using atomics - counts occurrences of each index
    __global__ void histogram_kernel(
        const int64_t* __restrict__ indices,
        int32_t* __restrict__ counts,
        size_t n_samples,
        size_t N) {

        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= n_samples)
            return;

        int64_t index = indices[idx];

        // Bounds check
        if (index < 0 || index >= static_cast<int64_t>(N))
            return;

        // Atomic increment
        atomicAdd(&counts[index], 1);
    }

    void launch_histogram(
        const int64_t* indices,
        int32_t* counts,
        size_t n_samples,
        size_t N,
        void* stream) {

        if (n_samples == 0)
            return;

        dim3 threads(256);
        dim3 grid((n_samples + threads.x - 1) / threads.x);
        cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

        histogram_kernel<<<grid, threads, 0, cuda_stream>>>(
            indices, counts, n_samples, N);
    }

    // Smarter histogram: Use hash map-style approach with sorting
    // This works well when n_samples << N (which is our case)
    __global__ void histogram_gather_sorted_kernel(
        const int64_t* __restrict__ indices,
        int32_t* __restrict__ output_counts,
        size_t n_samples,
        size_t N) {

        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= n_samples)
            return;

        int64_t my_index = indices[idx];

        // Bounds check
        if (my_index < 0 || my_index >= static_cast<int64_t>(N)) {
            output_counts[idx] = 0;
            return;
        }

        // Count occurrences: scan forward until we find a different index
        // This works ONLY if indices are sorted or if we accept O(n) per thread
        // Since indices are NOT sorted, we do linear scan (unavoidable without large temp storage)

        // Optimization: Use warp-level primitives to speed up counting
        int32_t count = 0;

        // Each thread scans the array looking for matches
        // This is O(n) per thread, total O(n*n_samples)
        // BUT: We can optimize using warp primitives

        const int WARP_SIZE = 32;
        int lane_id = threadIdx.x % WARP_SIZE;

        // Each warp cooperatively counts for one index
        for (size_t i = lane_id; i < n_samples; i += WARP_SIZE) {
            if (indices[i] == my_index) {
                count++;
            }
        }

        // Warp-level reduction to sum counts
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            count += __shfl_down_sync(0xffffffff, count, offset);
        }

        // First lane writes the result
        if (lane_id == 0) {
            output_counts[idx] = count;
        }
    }

    void launch_histogram_sort(
        const int64_t* indices,
        int32_t* output_counts,
        size_t n_samples,
        void* stream) {

        if (n_samples == 0)
            return;

        cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

        // Algorithm: Sort indices, then use adjacent_difference to find run boundaries,
        // then use inclusive_scan to count run lengths, then scatter back to original positions

        // Step 1: Create position array and copy indices
        thrust::device_vector<int32_t> orig_positions(n_samples);
        thrust::sequence(thrust::cuda::par.on(cuda_stream), orig_positions.begin(), orig_positions.end());

        thrust::device_vector<int64_t> sorted_indices(indices, indices + n_samples);

        // Step 2: Sort indices while tracking original positions
        thrust::sort_by_key(thrust::cuda::par.on(cuda_stream),
                            sorted_indices.begin(), sorted_indices.end(),
                            orig_positions.begin());

        // Step 3: Mark segment boundaries (1 where index changes, 0 otherwise)
        thrust::device_vector<int32_t> head_flags(n_samples);
        thrust::adjacent_difference(thrust::cuda::par.on(cuda_stream),
                                    sorted_indices.begin(), sorted_indices.end(),
                                    head_flags.begin(),
                                    thrust::not_equal_to<int64_t>());
        // First element is always a segment head
        if (n_samples > 0) {
            thrust::fill_n(thrust::cuda::par.on(cuda_stream), head_flags.begin(), 1, 1);
        }

        // Step 4: Compute run lengths with exclusive_scan_by_key
        // This gives each element its position within its segment
        thrust::device_vector<int32_t> run_positions(n_samples);
        thrust::device_vector<int32_t> ones(n_samples, 1);

        thrust::exclusive_scan_by_key(thrust::cuda::par.on(cuda_stream),
                                      sorted_indices.begin(), sorted_indices.end(),
                                      ones.begin(),
                                      run_positions.begin());

        // Step 5: Find the tail of each segment and compute run length
        // Use a kernel to compute the count for each element
        thrust::device_vector<int32_t> run_counts(n_samples);

        thrust::transform(thrust::cuda::par.on(cuda_stream),
                          thrust::make_counting_iterator<int>(0),
                          thrust::make_counting_iterator<int>(n_samples),
                          run_counts.begin(),
                          [sorted_indices_ptr = thrust::raw_pointer_cast(sorted_indices.data()),
                           run_positions_ptr = thrust::raw_pointer_cast(run_positions.data()),
                           n_samples] __device__(int idx) {
                              int64_t my_index = sorted_indices_ptr[idx];
                              int my_pos = run_positions_ptr[idx];

                              // Find the last occurrence of this index
                              int count = 1;
                              if (idx + 1 < n_samples && sorted_indices_ptr[idx + 1] == my_index) {
                                  // Not the last in segment, find it
                                  for (int i = idx + 1; i < n_samples && sorted_indices_ptr[i] == my_index; ++i) {
                                      count = run_positions_ptr[i] + 1;
                                  }
                              } else {
                                  // Last in segment
                                  count = my_pos + 1;
                              }
                              return count;
                          });

        // Step 6: Scatter counts back to original positions
        thrust::scatter(thrust::cuda::par.on(cuda_stream),
                        run_counts.begin(), run_counts.end(),
                        orig_positions.begin(),
                        output_counts);
    }

    // Fused gather kernel for 2 tensors - replaces 2x index_select
    // OPTIMIZED: Unroll loops for common cases
    __global__ void gather_2tensors_kernel(
        const int64_t* __restrict__ indices,
        const float* __restrict__ src_a,
        const float* __restrict__ src_b,
        float* __restrict__ dst_a,
        float* __restrict__ dst_b,
        size_t n_samples,
        size_t dim_a,
        size_t dim_b,
        size_t N) {

        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= n_samples)
            return;

        int64_t src_idx = indices[idx];

        // Bounds check - CRITICAL for safety
        if (src_idx < 0 || src_idx >= static_cast<int64_t>(N)) {
            // Zero the output for invalid indices
            for (size_t i = 0; i < dim_a; ++i)
                dst_a[idx * dim_a + i] = 0.0f;
            for (size_t i = 0; i < dim_b; ++i)
                dst_b[idx * dim_b + i] = 0.0f;
            return;
        }

        // Fast path for common case: dim_a=1, dim_b=3
        if (dim_a == 1 && dim_b == 3) {
            dst_a[idx] = src_a[src_idx];
            dst_b[idx * 3 + 0] = src_b[src_idx * 3 + 0];
            dst_b[idx * 3 + 1] = src_b[src_idx * 3 + 1];
            dst_b[idx * 3 + 2] = src_b[src_idx * 3 + 2];
            return;
        }

        // General case: Gather first tensor (dim_a elements)
        for (size_t i = 0; i < dim_a; ++i) {
            dst_a[idx * dim_a + i] = src_a[src_idx * dim_a + i];
        }

        // Gather second tensor (dim_b elements)
        for (size_t i = 0; i < dim_b; ++i) {
            dst_b[idx * dim_b + i] = src_b[src_idx * dim_b + i];
        }
    }

    void launch_gather_2tensors(
        const int64_t* indices,
        const float* src_a,
        const float* src_b,
        float* dst_a,
        float* dst_b,
        size_t n_samples,
        size_t dim_a,
        size_t dim_b,
        size_t N,
        void* stream) {

        if (n_samples == 0)
            return;

        dim3 threads(256);
        dim3 grid((n_samples + threads.x - 1) / threads.x);
        cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

        gather_2tensors_kernel<<<grid, threads, 0, cuda_stream>>>(
            indices,
            src_a,
            src_b,
            dst_a,
            dst_b,
            n_samples,
            dim_a,
            dim_b,
            N);
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
    // Block-cooperative reduction kernel (ZERO allocations, uses only shared memory)
    // Computes sum of opacities[alive_indices] using shared memory
    __global__ void reduce_opacities_kernel(
        const float* __restrict__ opacities,
        const int64_t* __restrict__ alive_indices,
        size_t n_alive,
        float* __restrict__ partial_sums,
        size_t N) {

        extern __shared__ float sdata[];

        size_t tid = threadIdx.x;
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        // Load data into shared memory
        float sum = 0.0f;
        if (idx < n_alive) {
            int64_t global_i = alive_indices[idx];
            if (global_i >= 0 && global_i < static_cast<int64_t>(N)) {
                sum = opacities[global_i];
            }
        }
        sdata[tid] = sum;
        __syncthreads();

        // Block-level reduction in shared memory
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        // Write result for this block
        if (tid == 0) {
            partial_sums[blockIdx.x] = sdata[0];
        }
    }

    __global__ void multinomial_sample_and_gather_kernel(
        const float* __restrict__ opacities,
        const float* __restrict__ scaling_raw, // OPTIMIZATION: Takes raw scaling, applies exp() inline
        const int64_t* __restrict__ alive_indices,
        const float* __restrict__ cumsum, // Pre-computed cumulative sum
        size_t n_alive,
        size_t n_samples,
        float prob_sum,
        uint64_t seed,
        int64_t* __restrict__ sampled_global_indices,
        float* __restrict__ sampled_opacities,
        float* __restrict__ sampled_scales,
        size_t N) {

        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= n_samples)
            return;

        // Initialize RNG for this thread
        curandState state;
        curand_init(seed, idx, 0, &state);

        // Generate random sample in [0, prob_sum]
        float u = curand_uniform(&state) * prob_sum;

        // Binary search in pre-computed cumsum (O(log n) instead of O(n)!)
        int left = 0;
        int right = n_alive - 1;
        int selected_idx = n_alive - 1;

        while (left <= right) {
            int mid = (left + right) / 2;
            if (cumsum[mid] >= u) {
                selected_idx = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        int64_t selected_global_idx = alive_indices[selected_idx];

        // Output global index
        sampled_global_indices[idx] = selected_global_idx;

        // Directly gather opacity and scales (fused in same kernel)
        // OPTIMIZATION: Apply exp() inline to avoid intermediate tensor allocation
        sampled_opacities[idx] = opacities[selected_global_idx];
        sampled_scales[idx * 3 + 0] = expf(scaling_raw[selected_global_idx * 3 + 0]);
        sampled_scales[idx * 3 + 1] = expf(scaling_raw[selected_global_idx * 3 + 1]);
        sampled_scales[idx * 3 + 2] = expf(scaling_raw[selected_global_idx * 3 + 2]);
    }

    void launch_multinomial_sample_and_gather(
        const float* opacities,
        const float* scaling_raw, // OPTIMIZATION: Takes raw scaling, kernel applies exp() inline
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

        cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

        // Block-level reduction
        constexpr int THREADS = 256;
        const int blocks = (n_alive + THREADS - 1) / THREADS;
        const size_t shared_mem_size = THREADS * sizeof(float);

        float* d_partial_sums = nullptr;
        cudaMallocAsync(&d_partial_sums, blocks * sizeof(float), cuda_stream);

        reduce_opacities_kernel<<<blocks, THREADS, shared_mem_size, cuda_stream>>>(
            opacities, alive_indices, n_alive, d_partial_sums, N);

        // Final reduction with CUB
        float* d_prob_sum = nullptr;
        cudaMallocAsync(&d_prob_sum, sizeof(float), cuda_stream);

        void* d_reduce_temp = nullptr;
        size_t reduce_temp_bytes = 0;
        cub::DeviceReduce::Sum(d_reduce_temp, reduce_temp_bytes, d_partial_sums, d_prob_sum, blocks, cuda_stream);
        cudaMallocAsync(&d_reduce_temp, reduce_temp_bytes, cuda_stream);
        cub::DeviceReduce::Sum(d_reduce_temp, reduce_temp_bytes, d_partial_sums, d_prob_sum, blocks, cuda_stream);
        cudaFreeAsync(d_reduce_temp, cuda_stream);

        // Copy result to host (required for branching)
        float prob_sum = 0.0f;
        cudaMemcpyAsync(&prob_sum, d_prob_sum, sizeof(float), cudaMemcpyDeviceToHost, cuda_stream);
        cudaStreamSynchronize(cuda_stream);

        cudaFreeAsync(d_partial_sums, cuda_stream);
        cudaFreeAsync(d_prob_sum, cuda_stream);

        if (prob_sum <= 0.0f) {
            cudaMemsetAsync(sampled_global_indices, 0, n_samples * sizeof(int64_t), cuda_stream);
            cudaMemsetAsync(sampled_opacities, 0, n_samples * sizeof(float), cuda_stream);
            cudaMemsetAsync(sampled_scales, 0, n_samples * 3 * sizeof(float), cuda_stream);
            return;
        }

        // Cumulative sum for sampling
        auto alive_probs = lfs::core::Tensor::empty({n_alive}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        auto cumsum_buf = lfs::core::Tensor::empty({n_alive}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);

        thrust::transform(thrust::cuda::par.on(cuda_stream),
                          thrust::counting_iterator<int>(0),
                          thrust::counting_iterator<int>(n_alive),
                          thrust::device_ptr<float>(alive_probs.ptr<float>()),
                          [=] __device__(int i) { return opacities[alive_indices[i]]; });

        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                      alive_probs.ptr<float>(), cumsum_buf.ptr<float>(), n_alive, cuda_stream);
        cudaMallocAsync(&d_temp_storage, temp_storage_bytes, cuda_stream);
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                      alive_probs.ptr<float>(), cumsum_buf.ptr<float>(), n_alive, cuda_stream);

        constexpr int SAMPLE_THREADS = 256;
        const int sample_grid = (n_samples + SAMPLE_THREADS - 1) / SAMPLE_THREADS;

        multinomial_sample_and_gather_kernel<<<sample_grid, SAMPLE_THREADS, 0, cuda_stream>>>(
            opacities,
            scaling_raw,
            alive_indices,
            cumsum_buf.ptr<float>(),
            n_alive,
            n_samples,
            prob_sum,
            seed,
            sampled_global_indices,
            sampled_opacities,
            sampled_scales,
            N);

        cudaFreeAsync(d_temp_storage, cuda_stream);
    }

    // Simplified multinomial kernel that samples from ALL N opacities (no alive_indices)
    __global__ void multinomial_sample_all_kernel(
        const float* __restrict__ opacities,
        const float* __restrict__ scaling_raw, // OPTIMIZATION: Takes raw scaling, applies exp() inline
        const float* __restrict__ cumsum,      // Pre-computed cumulative sum
        size_t N,
        size_t n_samples,
        float prob_sum,
        uint64_t seed,
        int64_t* __restrict__ sampled_indices,
        float* __restrict__ sampled_opacities,
        float* __restrict__ sampled_scales) {

        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= n_samples)
            return;

        // Initialize RNG for this thread
        curandState state;
        curand_init(seed, idx, 0, &state);

        // Generate random sample in [0, prob_sum]
        float u = curand_uniform(&state) * prob_sum;

        // Binary search in pre-computed cumsum (O(log N) instead of O(N)!)
        int left = 0;
        int right = N - 1;
        int64_t selected_idx = N - 1;

        while (left <= right) {
            int mid = (left + right) / 2;
            if (cumsum[mid] >= u) {
                selected_idx = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        // Output index and gather opacity/scales
        // OPTIMIZATION: Apply exp() inline to avoid intermediate tensor allocation
        sampled_indices[idx] = selected_idx;
        sampled_opacities[idx] = opacities[selected_idx];
        sampled_scales[idx * 3 + 0] = expf(scaling_raw[selected_idx * 3 + 0]);
        sampled_scales[idx * 3 + 1] = expf(scaling_raw[selected_idx * 3 + 1]);
        sampled_scales[idx * 3 + 2] = expf(scaling_raw[selected_idx * 3 + 2]);
    }

    // Reduction kernel for summing all opacities (simpler version without indices)
    __global__ void reduce_all_opacities_kernel(
        const float* __restrict__ opacities,
        size_t N,
        float* __restrict__ partial_sums) {

        extern __shared__ float sdata[];

        size_t tid = threadIdx.x;
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        // Load data into shared memory
        float sum = 0.0f;
        if (idx < N) {
            sum = opacities[idx];
        }
        sdata[tid] = sum;
        __syncthreads();

        // Block-level reduction in shared memory
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        // Write result for this block
        if (tid == 0) {
            partial_sums[blockIdx.x] = sdata[0];
        }
    }

    void launch_multinomial_sample_all(
        const float* opacities,
        const float* scaling_raw, // OPTIMIZATION: Takes raw scaling, kernel applies exp() inline
        size_t N,
        size_t n_samples,
        uint64_t seed,
        int64_t* sampled_indices,
        float* sampled_opacities,
        float* sampled_scales,
        void* stream) {

        if (n_samples == 0 || N == 0)
            return;

        cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

        // Block-level reduction
        constexpr int THREADS = 256;
        const int blocks = (N + THREADS - 1) / THREADS;
        const size_t shared_mem_size = THREADS * sizeof(float);

        float* d_partial_sums = nullptr;
        cudaMallocAsync(&d_partial_sums, blocks * sizeof(float), cuda_stream);

        reduce_all_opacities_kernel<<<blocks, THREADS, shared_mem_size, cuda_stream>>>(
            opacities, N, d_partial_sums);

        // Final reduction with CUB
        float* d_prob_sum = nullptr;
        cudaMallocAsync(&d_prob_sum, sizeof(float), cuda_stream);

        void* d_reduce_temp = nullptr;
        size_t reduce_temp_bytes = 0;
        cub::DeviceReduce::Sum(d_reduce_temp, reduce_temp_bytes, d_partial_sums, d_prob_sum, blocks, cuda_stream);
        cudaMallocAsync(&d_reduce_temp, reduce_temp_bytes, cuda_stream);
        cub::DeviceReduce::Sum(d_reduce_temp, reduce_temp_bytes, d_partial_sums, d_prob_sum, blocks, cuda_stream);
        cudaFreeAsync(d_reduce_temp, cuda_stream);

        // Copy result to host (required for branching)
        float prob_sum = 0.0f;
        cudaMemcpyAsync(&prob_sum, d_prob_sum, sizeof(float), cudaMemcpyDeviceToHost, cuda_stream);
        cudaStreamSynchronize(cuda_stream);

        cudaFreeAsync(d_partial_sums, cuda_stream);
        cudaFreeAsync(d_prob_sum, cuda_stream);

        if (prob_sum <= 0.0f) {
            cudaMemsetAsync(sampled_indices, 0, n_samples * sizeof(int64_t), cuda_stream);
            cudaMemsetAsync(sampled_opacities, 0, n_samples * sizeof(float), cuda_stream);
            cudaMemsetAsync(sampled_scales, 0, n_samples * 3 * sizeof(float), cuda_stream);
            return;
        }

        // Cumulative sum for sampling
        auto cumsum_buf = lfs::core::Tensor::empty({N}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);

        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                      opacities, cumsum_buf.ptr<float>(), N, cuda_stream);
        cudaMallocAsync(&d_temp_storage, temp_storage_bytes, cuda_stream);
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                      opacities, cumsum_buf.ptr<float>(), N, cuda_stream);

        constexpr int SAMPLE_THREADS = 256;
        const int sample_grid = (n_samples + SAMPLE_THREADS - 1) / SAMPLE_THREADS;

        multinomial_sample_all_kernel<<<sample_grid, SAMPLE_THREADS, 0, cuda_stream>>>(
            opacities,
            scaling_raw,
            cumsum_buf.ptr<float>(),
            N,
            n_samples,
            prob_sum,
            seed,
            sampled_indices,
            sampled_opacities,
            sampled_scales);

        cudaFreeAsync(d_temp_storage, cuda_stream);
    }

    // Compute rotation magnitude squared kernel (eliminates [N,4] intermediate tensor)
    __global__ void compute_rotation_mag_sq_kernel(
        const float* rotations, // [N, 4]
        float* mag_sq,          // [N]
        size_t N) {

        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= N)
            return;

        // Each rotation is a quaternion [w, x, y, z] or [x, y, z, w]
        // Compute ||q||^2 = q[0]^2 + q[1]^2 + q[2]^2 + q[3]^2
        const float* q = &rotations[idx * 4];
        mag_sq[idx] = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    }

    void launch_compute_rotation_mag_sq(
        const float* rotations,
        float* mag_sq,
        size_t N,
        void* stream) {

        if (N == 0) {
            return;
        }

        dim3 threads(256);
        dim3 grid((N + threads.x - 1) / threads.x);

        cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

        compute_rotation_mag_sq_kernel<<<grid, threads, 0, cuda_stream>>>(
            rotations,
            mag_sq,
            N);
    }

    // Fused dead mask computation kernel (ZERO intermediate allocations)
    __global__ void compute_dead_mask_kernel(
        const float* opacities, // [N]
        const float* rotations, // [N, 4]
        uint8_t* dead_mask,     // [N]
        size_t N,
        float min_opacity) {

        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= N)
            return;

        // Check opacity condition: opacity <= min_opacity
        bool is_dead = opacities[idx] <= min_opacity;

        // Check rotation magnitude condition: ||rotation||^2 < 1e-8
        if (!is_dead) {
            const float* q = &rotations[idx * 4];
            float mag_sq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
            is_dead = mag_sq < 1e-8f;
        }

        dead_mask[idx] = is_dead ? 1 : 0;
    }

    void launch_compute_dead_mask(
        const float* opacities,
        const float* rotations,
        uint8_t* dead_mask,
        size_t N,
        float min_opacity,
        void* stream) {

        if (N == 0) {
            return;
        }

        dim3 threads(256);
        dim3 grid((N + threads.x - 1) / threads.x);

        cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : nullptr;

        compute_dead_mask_kernel<<<grid, threads, 0, cuda_stream>>>(
            opacities,
            rotations,
            dead_mask,
            N,
            min_opacity);
    }

} // namespace lfs::training::mcmc
