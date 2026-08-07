// densification_kernels.cu
/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include "densification_kernels.hpp"
#include <cub/cub.cuh>

#include "kernel_stream.hpp"

namespace lfs::training::kernels {

    // ============================================================================
    // Helper functions
    // ============================================================================

    __device__ inline float sigmoid(float x) {
        return 1.0f / (1.0f + expf(-x));
    }

    __device__ inline float inverse_sigmoid(float y) {
        // logit(y) = log(y / (1-y))
        // Clamp to avoid infinities
        y = fmaxf(1e-7f, fminf(1.0f - 1e-7f, y));
        return logf(y / (1.0f - y));
    }

    /**
     * @brief Convert quaternion to rotation matrix
     *
     * Quaternion format: [w, x, y, z]
     * Output: 3x3 rotation matrix stored row-major in R[9]
     */
    __device__ inline void quat_to_rotmat(const float* q, float* R) {
        float w = q[0], x = q[1], y = q[2], z = q[3];

        // R = [[1-2(y²+z²), 2(xy-wz), 2(xz+wy)],
        //      [2(xy+wz), 1-2(x²+z²), 2(yz-wx)],
        //      [2(xz-wy), 2(yz+wx), 1-2(x²+y²)]]

        R[0] = 1.0f - 2.0f * (y * y + z * z); // r00
        R[1] = 2.0f * (x * y - w * z);        // r01
        R[2] = 2.0f * (x * z + w * y);        // r02

        R[3] = 2.0f * (x * y + w * z);        // r10
        R[4] = 1.0f - 2.0f * (x * x + z * z); // r11
        R[5] = 2.0f * (y * z - w * x);        // r12

        R[6] = 2.0f * (x * z - w * y);        // r20
        R[7] = 2.0f * (y * z + w * x);        // r21
        R[8] = 1.0f - 2.0f * (x * x + y * y); // r22
    }

    /**
     * @brief Matrix-vector multiply: out = R * v (where R is 3x3, v is 3x1)
     */
    __device__ inline void matvec_3x3(const float* R, const float* v, float* out) {
        out[0] = R[0] * v[0] + R[1] * v[1] + R[2] * v[2];
        out[1] = R[3] * v[0] + R[4] * v[1] + R[5] * v[2];
        out[2] = R[6] * v[0] + R[7] * v[1] + R[8] * v[2];
    }

    // ============================================================================
    // Duplicate Gaussians Kernels (Split into two to avoid warp divergence)
    // ============================================================================

    // ============================================================================
    // Launch functions
    // ============================================================================

    // ============================================================================
    // In-place Long-Axis-Split Kernel
    // ============================================================================

    // Helper function to get the maximum value index in an array of size 3
    __device__ uint3 get_max_value_index(const float* arr) {

        float v0 = arr[0], v1 = arr[1], v2 = arr[2];
        float max_value = fmaxf(v0, fmaxf(v1, v2));

        if (max_value == v0) {
            return make_uint3(0, 1, 2);
        }
        if (max_value == v1) {
            return make_uint3(1, 0, 2);
        }
        return make_uint3(2, 0, 1);
    }

    __global__ void long_axis_split_gaussians_inplace_kernel(
        float* __restrict__ positions,        // [N, 3] - modified in-place
        float* __restrict__ rotations,        // [N, 4] - unchanged
        float* __restrict__ scales,           // [N, 3] - modified in-place
        const float* __restrict__ sh0,        // [N, 3] - read only
        const float* __restrict__ shN,        // [N, shN_dim] - read only
        float* __restrict__ opacities,        // [N, 1] - modified in-place
        float* __restrict__ second_positions, // [num_split, 3]
        float* __restrict__ second_rotations, // [num_split, 4]
        float* __restrict__ second_scales,    // [num_split, 3]
        float* __restrict__ second_sh0,       // [num_split, 3]
        float* __restrict__ second_shN,       // [num_split, shN_dim]
        float* __restrict__ second_opacities, // [num_split, 1]
        const int64_t* __restrict__ split_indices,
        int num_split,
        int shN_dim) {
        int split_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (split_idx >= num_split)
            return;

        int src_idx = split_indices[split_idx];

        // Load original data
        float pos[3], quat[4], scale[3], opacity;
        pos[0] = positions[src_idx * 3 + 0];
        pos[1] = positions[src_idx * 3 + 1];
        pos[2] = positions[src_idx * 3 + 2];

        quat[0] = rotations[src_idx * 4 + 0];
        quat[1] = rotations[src_idx * 4 + 1];
        quat[2] = rotations[src_idx * 4 + 2];
        quat[3] = rotations[src_idx * 4 + 3];

        scale[0] = scales[src_idx * 3 + 0];
        scale[1] = scales[src_idx * 3 + 1];
        scale[2] = scales[src_idx * 3 + 2];

        opacity = opacities[src_idx];

        // Convert quaternion to rotation matrix
        float R[9];
        quat_to_rotmat(quat, R);

        // Identify greater axis (stored at 'x')
        uint3 scale_idxs = get_max_value_index(scale);
        unsigned int longest_idx = scale_idxs.x;
        float offset_magnitude = expf(scale[longest_idx]) * 0.5f;

        // New scale,
        float new_scale[3];
        new_scale[longest_idx] = scale[longest_idx] + logf(0.5f);
        new_scale[scale_idxs.y] = scale[scale_idxs.y] + logf(0.85);
        new_scale[scale_idxs.z] = scale[scale_idxs.z] + logf(0.85);

        // Adjust opacity according to LAS algorithm
        float sig = sigmoid(opacity);
        float raw_sig = sig * 0.6f;
        float new_opacity = inverse_sigmoid(raw_sig);

        // Compute offset for first split (copy 0)
        // Directly in global coordinates
        float global_offset[3];
        global_offset[0] = R[longest_idx] * offset_magnitude;
        global_offset[1] = R[longest_idx + 3] * offset_magnitude;
        global_offset[2] = R[longest_idx + 6] * offset_magnitude;

        // Write first split result back to original position (in-place)
        positions[src_idx * 3 + 0] = pos[0] + global_offset[0];
        positions[src_idx * 3 + 1] = pos[1] + global_offset[1];
        positions[src_idx * 3 + 2] = pos[2] + global_offset[2];

        scales[src_idx * 3 + 0] = new_scale[0];
        scales[src_idx * 3 + 1] = new_scale[1];
        scales[src_idx * 3 + 2] = new_scale[2];

        opacities[src_idx] = new_opacity;

        // Write second split result to output arrays
        second_positions[split_idx * 3 + 0] = pos[0] - global_offset[0];
        second_positions[split_idx * 3 + 1] = pos[1] - global_offset[1];
        second_positions[split_idx * 3 + 2] = pos[2] - global_offset[2];

        second_rotations[split_idx * 4 + 0] = quat[0];
        second_rotations[split_idx * 4 + 1] = quat[1];
        second_rotations[split_idx * 4 + 2] = quat[2];
        second_rotations[split_idx * 4 + 3] = quat[3];

        second_scales[split_idx * 3 + 0] = new_scale[0];
        second_scales[split_idx * 3 + 1] = new_scale[1];
        second_scales[split_idx * 3 + 2] = new_scale[2];

        // Copy SH coefficients
        second_sh0[split_idx * 3 + 0] = sh0[src_idx * 3 + 0];
        second_sh0[split_idx * 3 + 1] = sh0[src_idx * 3 + 1];
        second_sh0[split_idx * 3 + 2] = sh0[src_idx * 3 + 2];

        for (int i = 0; i < shN_dim; ++i) {
            second_shN[split_idx * shN_dim + i] = shN[src_idx * shN_dim + i];
        }

        second_opacities[split_idx] = new_opacity;
    }

    void launch_long_axis_split_gaussians_inplace(
        float* positions,
        float* rotations,
        float* scales,
        const float* sh0,
        const float* shN,
        float* opacities,
        float* second_positions,
        float* second_rotations,
        float* second_scales,
        float* second_sh0,
        float* second_shN,
        float* second_opacities,
        const int64_t* split_indices,
        int num_split,
        int shN_dim,
        cudaStream_t stream) {
        stream = resolve_stream(stream);
        if (num_split == 0)
            return;

        const int block_size = 256;
        const int num_blocks = (num_split + block_size - 1) / block_size;

        long_axis_split_gaussians_inplace_kernel<<<num_blocks, block_size, 0, stream>>>(
            positions, rotations, scales, sh0, shN, opacities,
            second_positions, second_rotations, second_scales,
            second_sh0, second_shN, second_opacities,
            split_indices, num_split, shN_dim);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.densify.long_axis_split_inplace");
    }

} // namespace lfs::training::kernels
