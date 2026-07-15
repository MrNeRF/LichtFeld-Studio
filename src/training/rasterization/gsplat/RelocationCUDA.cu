/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cuda_runtime.h>

#include "Common.h"
#include "Relocation.h"

namespace gsplat_lfs {

    // Equation (9) in "3D Gaussian Splatting as Markov Chain Monte Carlo"
    template <typename scalar_t>
    __global__ void relocation_kernel(
        int N,
        scalar_t* opacities,    // [N] - modified in-place
        scalar_t* scales,       // [N, 3] - modified in-place
        const int* ratios,      // [N] - integer split counts
        const scalar_t* binoms, // [n_max, n_max]
        int n_max,
        float min_opacity) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= N)
            return;

        int n_idx = ratios[idx];
        float denom_sum = 0.0f;

        // Read original opacity
        float old_opacity = opacities[idx];

        // Compute new opacity
        float new_opacity = fmaxf(1.0f - powf(1.0f - old_opacity, 1.0f / n_idx), min_opacity);

        // Compute denominator sum for scale
        for (int i = 1; i <= n_idx; ++i) {
            for (int k = 0; k <= (i - 1); ++k) {
                float bin_coeff = binoms[(i - 1) * n_max + k];
                float term = (pow(-1.0f, k) / sqrt(static_cast<float>(k + 1))) *
                             pow(new_opacity, k + 1);
                denom_sum += (bin_coeff * term);
            }
        }

        // Compute coefficient for scale
        float coeff = (old_opacity / denom_sum);

        // Write results back in-place
        opacities[idx] = new_opacity;
        for (int i = 0; i < 3; ++i) {
            scales[idx * 3 + i] = coeff * scales[idx * 3 + i];
        }
    }

    void launch_relocation_kernel(
        float* opacities,      // [N] - modified in-place
        float* scales,         // [N, 3] - modified in-place
        const int32_t* ratios, // [N] - integer split counts
        const float* binoms,   // [n_max, n_max]
        int64_t N,
        int32_t n_max,
        float min_opacity,
        cudaStream_t stream) {
        int64_t n_elements = N;
        dim3 threads(256);
        dim3 grid((n_elements + threads.x - 1) / threads.x);
        int64_t shmem_size = 0; // No shared memory used in this kernel

        if (n_elements == 0) {
            // skip the kernel launch if there are no elements
            return;
        }

        relocation_kernel<float>
            <<<grid, threads, shmem_size, stream>>>(
                static_cast<int>(N),
                opacities,
                scales,
                ratios,
                binoms,
                n_max,
                min_opacity);
    }

    inline __device__ mat3 raw_quat_to_rotmat(const vec4 raw_quat) {
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

    template <typename scalar_t>
    __global__ void add_noise_kernel(
        int N,
        const scalar_t* raw_opacities, // [N] - read only for noise computation
        const scalar_t* raw_scales,    // [N, 3] - read only for noise computation
        const scalar_t* raw_quats,     // [N, 4] - read only for noise computation
        const scalar_t* noise,         // [N, 3]
        scalar_t* means,               // [N, 3] - modified in-place
        float current_lr) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= N)
            return;

        int idx_3d = 3 * idx;

        const vec3 raw_scale = glm::make_vec3(raw_scales + idx_3d);
        mat3 S2 = mat3(__expf(2.f * raw_scale[0]), 0.f, 0.f, 0.f, __expf(2.f * raw_scale[1]), 0.f, 0.f, 0.f, __expf(2.f * raw_scale[2]));

        vec4 raw_quat = glm::make_vec4(raw_quats + 4 * idx);
        mat3 R = raw_quat_to_rotmat(raw_quat);

        mat3 covariance = R * S2 * glm::transpose(R);

        vec3 transformed_noise = covariance * glm::make_vec3(noise + idx_3d);

        float opacity = __frcp_rn(1.f + __expf(-raw_opacities[idx]));
        float op_sigmoid = __frcp_rn(1.f + __expf(100.f * opacity - 0.5f));
        float noise_factor = current_lr * op_sigmoid;

        means[idx_3d] += noise_factor * transformed_noise.x;
        means[idx_3d + 1] += noise_factor * transformed_noise.y;
        means[idx_3d + 2] += noise_factor * transformed_noise.z;
    }

    void launch_add_noise_kernel(
        float* raw_opacities, // [N] - read only for this kernel
        float* raw_scales,    // [N, 3] - read only for this kernel
        float* raw_quats,     // [N, 4] - read only for this kernel
        const float* noise,   // [N, 3]
        float* means,         // [N, 3] - modified in-place
        int64_t N,
        float current_lr,
        cudaStream_t stream) {
        int64_t n_elements = N;
        dim3 threads(256);
        dim3 grid((n_elements + threads.x - 1) / threads.x);
        int64_t shmem_size = 0; // No shared memory used in this kernel

        if (n_elements == 0) {
            // skip the kernel launch if there are no elements
            return;
        }

        add_noise_kernel<float>
            <<<grid, threads, shmem_size, stream>>>(
                static_cast<int>(N),
                raw_opacities,
                raw_scales,
                raw_quats,
                noise,
                means,
                current_lr);
    }

} // namespace gsplat_lfs
