// densification_kernels.cu
/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "densification_kernels.hpp"
#include <cub/cub.cuh>

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

    // Kernel 1: Copy all N Gaussians (fully coalesced, no divergence)
    __global__ void duplicate_copy_kernel(
        const float* __restrict__ positions_in,
        const float* __restrict__ rotations_in,
        const float* __restrict__ scales_in,
        const float* __restrict__ sh0_in,
        const float* __restrict__ shN_in,
        const float* __restrict__ opacities_in,
        float* __restrict__ positions_out,
        float* __restrict__ rotations_out,
        float* __restrict__ scales_out,
        float* __restrict__ sh0_out,
        float* __restrict__ shN_out,
        float* __restrict__ opacities_out,
        int N,
        int shN_dim) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= N)
            return;

        // Sequential copy (src_idx == dst_idx)
        // Use vectorized loads for positions (float3)
        const float3 pos = make_float3(
            positions_in[idx * 3 + 0],
            positions_in[idx * 3 + 1],
            positions_in[idx * 3 + 2]);
        positions_out[idx * 3 + 0] = pos.x;
        positions_out[idx * 3 + 1] = pos.y;
        positions_out[idx * 3 + 2] = pos.z;

        // Use vectorized loads for rotations (float4)
        const float4 rot = make_float4(
            rotations_in[idx * 4 + 0],
            rotations_in[idx * 4 + 1],
            rotations_in[idx * 4 + 2],
            rotations_in[idx * 4 + 3]);
        rotations_out[idx * 4 + 0] = rot.x;
        rotations_out[idx * 4 + 1] = rot.y;
        rotations_out[idx * 4 + 2] = rot.z;
        rotations_out[idx * 4 + 3] = rot.w;

        // Use vectorized loads for scales (float3)
        const float3 scale = make_float3(
            scales_in[idx * 3 + 0],
            scales_in[idx * 3 + 1],
            scales_in[idx * 3 + 2]);
        scales_out[idx * 3 + 0] = scale.x;
        scales_out[idx * 3 + 1] = scale.y;
        scales_out[idx * 3 + 2] = scale.z;

        // Copy sh0 (3 elements) - use float3
        const float3 sh0 = make_float3(
            sh0_in[idx * 3 + 0],
            sh0_in[idx * 3 + 1],
            sh0_in[idx * 3 + 2]);
        sh0_out[idx * 3 + 0] = sh0.x;
        sh0_out[idx * 3 + 1] = sh0.y;
        sh0_out[idx * 3 + 2] = sh0.z;

        // Copy shN coefficients with vectorized loads (float4)
        const float* shN_src = shN_in + idx * shN_dim;
        float* shN_dst = shN_out + idx * shN_dim;

        // Process in chunks of 4 (most efficient for SH degree 3: 45 elements)
        int i = 0;
        for (; i + 4 <= shN_dim; i += 4) {
            float4 sh_val = *reinterpret_cast<const float4*>(shN_src + i);
            *reinterpret_cast<float4*>(shN_dst + i) = sh_val;
        }
        // Handle remainder
        for (; i < shN_dim; ++i) {
            shN_dst[i] = shN_src[i];
        }

        // Copy opacity (1 element)
        opacities_out[idx] = opacities_in[idx];
    }

    // Kernel 2: Gather from selected indices (scattered reads, but no divergence)
    __global__ void duplicate_gather_kernel(
        const float* __restrict__ positions_in,
        const float* __restrict__ rotations_in,
        const float* __restrict__ scales_in,
        const float* __restrict__ sh0_in,
        const float* __restrict__ shN_in,
        const float* __restrict__ opacities_in,
        float* __restrict__ positions_out,
        float* __restrict__ rotations_out,
        float* __restrict__ scales_out,
        float* __restrict__ sh0_out,
        float* __restrict__ shN_out,
        float* __restrict__ opacities_out,
        const int64_t* __restrict__ selected_indices,
        int N,
        int num_selected,
        int shN_dim) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_selected)
            return;

        // Gather from selected indices
        int src_idx = selected_indices[idx];
        int dst_idx = N + idx; // Append after first N elements

        // Use vectorized loads for positions (float3)
        const float3 pos = make_float3(
            positions_in[src_idx * 3 + 0],
            positions_in[src_idx * 3 + 1],
            positions_in[src_idx * 3 + 2]);
        positions_out[dst_idx * 3 + 0] = pos.x;
        positions_out[dst_idx * 3 + 1] = pos.y;
        positions_out[dst_idx * 3 + 2] = pos.z;

        // Use vectorized loads for rotations (float4)
        const float4 rot = make_float4(
            rotations_in[src_idx * 4 + 0],
            rotations_in[src_idx * 4 + 1],
            rotations_in[src_idx * 4 + 2],
            rotations_in[src_idx * 4 + 3]);
        rotations_out[dst_idx * 4 + 0] = rot.x;
        rotations_out[dst_idx * 4 + 1] = rot.y;
        rotations_out[dst_idx * 4 + 2] = rot.z;
        rotations_out[dst_idx * 4 + 3] = rot.w;

        // Use vectorized loads for scales (float3)
        const float3 scale = make_float3(
            scales_in[src_idx * 3 + 0],
            scales_in[src_idx * 3 + 1],
            scales_in[src_idx * 3 + 2]);
        scales_out[dst_idx * 3 + 0] = scale.x;
        scales_out[dst_idx * 3 + 1] = scale.y;
        scales_out[dst_idx * 3 + 2] = scale.z;

        // Copy sh0 (3 elements) - use float3
        const float3 sh0 = make_float3(
            sh0_in[src_idx * 3 + 0],
            sh0_in[src_idx * 3 + 1],
            sh0_in[src_idx * 3 + 2]);
        sh0_out[dst_idx * 3 + 0] = sh0.x;
        sh0_out[dst_idx * 3 + 1] = sh0.y;
        sh0_out[dst_idx * 3 + 2] = sh0.z;

        // Copy shN coefficients with vectorized loads (float4)
        const float* shN_src = shN_in + src_idx * shN_dim;
        float* shN_dst = shN_out + dst_idx * shN_dim;

        // Process in chunks of 4 (most efficient for SH degree 3: 45 elements)
        int i = 0;
        for (; i + 4 <= shN_dim; i += 4) {
            float4 sh_val = *reinterpret_cast<const float4*>(shN_src + i);
            *reinterpret_cast<float4*>(shN_dst + i) = sh_val;
        }
        // Handle remainder
        for (; i < shN_dim; ++i) {
            shN_dst[i] = shN_src[i];
        }

        // Copy opacity (1 element)
        opacities_out[dst_idx] = opacities_in[src_idx];
    }

    // ============================================================================
    // Split Gaussians Kernel
    // ============================================================================

    __global__ void split_gaussians_keep_kernel(
        const float* __restrict__ positions_in,
        const float* __restrict__ rotations_in,
        const float* __restrict__ scales_in,
        const float* __restrict__ sh0_in,
        const float* __restrict__ shN_in,
        const float* __restrict__ opacities_in,
        float* __restrict__ positions_out,
        float* __restrict__ rotations_out,
        float* __restrict__ scales_out,
        float* __restrict__ sh0_out,
        float* __restrict__ shN_out,
        float* __restrict__ opacities_out,
        const int64_t* __restrict__ keep_indices,
        int num_keep,
        int shN_dim) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= num_keep)
            return;

        int src_idx = keep_indices[idx];

        // Copy all parameters for kept Gaussians
        positions_out[idx * 3 + 0] = positions_in[src_idx * 3 + 0];
        positions_out[idx * 3 + 1] = positions_in[src_idx * 3 + 1];
        positions_out[idx * 3 + 2] = positions_in[src_idx * 3 + 2];

        rotations_out[idx * 4 + 0] = rotations_in[src_idx * 4 + 0];
        rotations_out[idx * 4 + 1] = rotations_in[src_idx * 4 + 1];
        rotations_out[idx * 4 + 2] = rotations_in[src_idx * 4 + 2];
        rotations_out[idx * 4 + 3] = rotations_in[src_idx * 4 + 3];

        scales_out[idx * 3 + 0] = scales_in[src_idx * 3 + 0];
        scales_out[idx * 3 + 1] = scales_in[src_idx * 3 + 1];
        scales_out[idx * 3 + 2] = scales_in[src_idx * 3 + 2];

        // Copy sh0 (3 dims) and shN (shN_dim dims) separately
        sh0_out[idx * 3 + 0] = sh0_in[src_idx * 3 + 0];
        sh0_out[idx * 3 + 1] = sh0_in[src_idx * 3 + 1];
        sh0_out[idx * 3 + 2] = sh0_in[src_idx * 3 + 2];

        for (int i = 0; i < shN_dim; ++i) {
            shN_out[idx * shN_dim + i] = shN_in[src_idx * shN_dim + i];
        }

        opacities_out[idx] = opacities_in[src_idx];
    }

    __global__ void split_gaussians_split_kernel(
        const float* __restrict__ positions_in,
        const float* __restrict__ rotations_in,
        const float* __restrict__ scales_in,
        const float* __restrict__ sh0_in,
        const float* __restrict__ shN_in,
        const float* __restrict__ opacities_in,
        float* __restrict__ positions_out,
        float* __restrict__ rotations_out,
        float* __restrict__ scales_out,
        float* __restrict__ sh0_out,
        float* __restrict__ shN_out,
        float* __restrict__ opacities_out,
        const int64_t* __restrict__ split_indices,
        const float* __restrict__ random_noise, // Shape: [2, num_split, 3]
        int num_keep,
        int num_split,
        int shN_dim,
        bool revised_opacity) {
        int split_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (split_idx >= num_split)
            return;

        int src_idx = split_indices[split_idx];

        // Load input data
        float pos[3], quat[4], scale[3], opacity;
        pos[0] = positions_in[src_idx * 3 + 0];
        pos[1] = positions_in[src_idx * 3 + 1];
        pos[2] = positions_in[src_idx * 3 + 2];

        quat[0] = rotations_in[src_idx * 4 + 0];
        quat[1] = rotations_in[src_idx * 4 + 1];
        quat[2] = rotations_in[src_idx * 4 + 2];
        quat[3] = rotations_in[src_idx * 4 + 3];

        scale[0] = scales_in[src_idx * 3 + 0];
        scale[1] = scales_in[src_idx * 3 + 1];
        scale[2] = scales_in[src_idx * 3 + 2];

        opacity = opacities_in[src_idx];

        // Convert quaternion to rotation matrix
        float R[9];
        quat_to_rotmat(quat, R);

        // New scale = log(exp(old_scale) / 1.6)
        float new_scale[3];
        new_scale[0] = scale[0] - logf(1.6f);
        new_scale[1] = scale[1] - logf(1.6f);
        new_scale[2] = scale[2] - logf(1.6f);

        // Adjust opacity if revised formula
        float new_opacity = opacity;
        if (revised_opacity) {
            float sig = sigmoid(opacity);
            float one_minus_sig = 1.0f - sig;
            float adjusted = 1.0f - sqrtf(one_minus_sig);
            new_opacity = inverse_sigmoid(adjusted);
        }

        // Process both copies with DIFFERENT random noise
        // Output ordering: [all_copy0, all_copy1] to match CPU implementation
        for (int split_copy = 0; split_copy < 2; ++split_copy) {
            // Output index: first all copy0, then all copy1
            int out_idx = num_keep + split_copy * num_split + split_idx;

            // Get random noise for this copy: random_noise[split_copy, split_idx, :]
            // Shape is [2, num_split, 3], so index is split_copy * num_split * 3 + split_idx * 3
            float noise[3];
            int noise_offset = split_copy * num_split * 3 + split_idx * 3;
            noise[0] = random_noise[noise_offset + 0];
            noise[1] = random_noise[noise_offset + 1];
            noise[2] = random_noise[noise_offset + 2];

            // Compute offset = R * (exp(S) * noise)
            float scaled_noise[3];
            scaled_noise[0] = expf(scale[0]) * noise[0];
            scaled_noise[1] = expf(scale[1]) * noise[1];
            scaled_noise[2] = expf(scale[2]) * noise[2];

            float offset[3];
            matvec_3x3(R, scaled_noise, offset);

            // Position: original + offset (each copy has different offset due to different noise)
            positions_out[out_idx * 3 + 0] = pos[0] + offset[0];
            positions_out[out_idx * 3 + 1] = pos[1] + offset[1];
            positions_out[out_idx * 3 + 2] = pos[2] + offset[2];

            // Rotation: unchanged
            rotations_out[out_idx * 4 + 0] = quat[0];
            rotations_out[out_idx * 4 + 1] = quat[1];
            rotations_out[out_idx * 4 + 2] = quat[2];
            rotations_out[out_idx * 4 + 3] = quat[3];

            // Scale: divided by 1.6
            scales_out[out_idx * 3 + 0] = new_scale[0];
            scales_out[out_idx * 3 + 1] = new_scale[1];
            scales_out[out_idx * 3 + 2] = new_scale[2];

            // SH: copy sh0 (3 dims) and shN (shN_dim dims) separately
            sh0_out[out_idx * 3 + 0] = sh0_in[src_idx * 3 + 0];
            sh0_out[out_idx * 3 + 1] = sh0_in[src_idx * 3 + 1];
            sh0_out[out_idx * 3 + 2] = sh0_in[src_idx * 3 + 2];

            for (int i = 0; i < shN_dim; ++i) {
                shN_out[out_idx * shN_dim + i] = shN_in[src_idx * shN_dim + i];
            }

            // Opacity: adjusted if revised
            opacities_out[out_idx] = new_opacity;
        }
    }

    // ============================================================================
    // Launch functions
    // ============================================================================

    void launch_duplicate_gaussians(
        const float* positions_in,
        const float* rotations_in,
        const float* scales_in,
        const float* sh0_in,
        const float* shN_in,
        const float* opacities_in,
        float* positions_out,
        float* rotations_out,
        float* scales_out,
        float* sh0_out,
        float* shN_out,
        float* opacities_out,
        const int64_t* selected_indices,
        int N,
        int num_selected,
        int shN_dim,
        cudaStream_t stream) {
        // Step 1: Copy all N Gaussians using cudaMemcpyAsync (DMA-accelerated, like LibTorch's cat)
        if (N > 0) {
            cudaMemcpyAsync(positions_out, positions_in, N * 3 * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(rotations_out, rotations_in, N * 4 * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(scales_out, scales_in, N * 3 * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(sh0_out, sh0_in, N * 3 * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(shN_out, shN_in, N * shN_dim * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(opacities_out, opacities_in, N * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        }

        // Step 2: Gather from selected indices using kernel (scattered reads)
        if (num_selected > 0) {
            const int block_size = 256;
            const int num_blocks_gather = (num_selected + block_size - 1) / block_size;
            duplicate_gather_kernel<<<num_blocks_gather, block_size, 0, stream>>>(
                positions_in, rotations_in, scales_in, sh0_in, shN_in, opacities_in,
                positions_out, rotations_out, scales_out, sh0_out, shN_out, opacities_out,
                selected_indices, N, num_selected, shN_dim);
        }
    }

    void launch_split_gaussians(
        const float* positions_in,
        const float* rotations_in,
        const float* scales_in,
        const float* sh0_in,
        const float* shN_in,
        const float* opacities_in,
        float* positions_out,
        float* rotations_out,
        float* scales_out,
        float* sh0_out,
        float* shN_out,
        float* opacities_out,
        const int64_t* split_indices,
        const int64_t* keep_indices,
        const float* random_noise,
        int N,
        int num_split,
        int num_keep,
        int shN_dim,
        bool revised_opacity,
        cudaStream_t stream) {
        const int block_size = 256;

        // Kernel 1: Copy kept Gaussians
        if (num_keep > 0) {
            const int num_blocks_keep = (num_keep + block_size - 1) / block_size;
            split_gaussians_keep_kernel<<<num_blocks_keep, block_size, 0, stream>>>(
                positions_in, rotations_in, scales_in, sh0_in, shN_in, opacities_in,
                positions_out, rotations_out, scales_out, sh0_out, shN_out, opacities_out,
                keep_indices, num_keep, shN_dim);
        }

        // Kernel 2: Split selected Gaussians
        if (num_split > 0) {
            const int num_blocks_split = (num_split + block_size - 1) / block_size;
            split_gaussians_split_kernel<<<num_blocks_split, block_size, 0, stream>>>(
                positions_in, rotations_in, scales_in, sh0_in, shN_in, opacities_in,
                positions_out, rotations_out, scales_out, sh0_out, shN_out, opacities_out,
                split_indices, random_noise,
                num_keep, num_split, shN_dim, revised_opacity);
        }
    }

    // ============================================================================
    // In-place Split Kernel
    // ============================================================================

    /**
     * In-place split kernel: modifies original positions for first split,
     * writes second split to separate output arrays.
     */
    __global__ void split_gaussians_inplace_kernel(
        float* __restrict__ positions,        // [N, 3] - modified in-place
        float* __restrict__ rotations,        // [N, 4] - unchanged
        float* __restrict__ scales,           // [N, 3] - modified in-place
        const float* __restrict__ sh0,        // [N, 3] - read only
        const float* __restrict__ shN,        // [N, shN_dim] - read only
        float* __restrict__ opacities,        // [N, 1] - modified if revised
        float* __restrict__ second_positions, // [num_split, 3]
        float* __restrict__ second_rotations, // [num_split, 4]
        float* __restrict__ second_scales,    // [num_split, 3]
        float* __restrict__ second_sh0,       // [num_split, 3]
        float* __restrict__ second_shN,       // [num_split, shN_dim]
        float* __restrict__ second_opacities, // [num_split, 1]
        const int64_t* __restrict__ split_indices,
        const float* __restrict__ random_noise, // [2, num_split, 3]
        int num_split,
        int shN_dim,
        bool revised_opacity) {
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

        // New scale = log(exp(old_scale) / 1.6)
        float new_scale[3];
        new_scale[0] = scale[0] - logf(1.6f);
        new_scale[1] = scale[1] - logf(1.6f);
        new_scale[2] = scale[2] - logf(1.6f);

        // Adjust opacity if revised formula
        float new_opacity = opacity;
        if (revised_opacity) {
            float sig = sigmoid(opacity);
            float one_minus_sig = 1.0f - sig;
            float adjusted = 1.0f - sqrtf(one_minus_sig);
            new_opacity = inverse_sigmoid(adjusted);
        }

        // Compute offset for first split (copy 0)
        float noise0[3];
        int noise_offset0 = 0 * num_split * 3 + split_idx * 3; // copy 0
        noise0[0] = random_noise[noise_offset0 + 0];
        noise0[1] = random_noise[noise_offset0 + 1];
        noise0[2] = random_noise[noise_offset0 + 2];

        float scaled_noise0[3];
        scaled_noise0[0] = expf(scale[0]) * noise0[0];
        scaled_noise0[1] = expf(scale[1]) * noise0[1];
        scaled_noise0[2] = expf(scale[2]) * noise0[2];

        float offset0[3];
        matvec_3x3(R, scaled_noise0, offset0);

        // Write first split result back to original position (in-place)
        positions[src_idx * 3 + 0] = pos[0] + offset0[0];
        positions[src_idx * 3 + 1] = pos[1] + offset0[1];
        positions[src_idx * 3 + 2] = pos[2] + offset0[2];

        scales[src_idx * 3 + 0] = new_scale[0];
        scales[src_idx * 3 + 1] = new_scale[1];
        scales[src_idx * 3 + 2] = new_scale[2];

        if (revised_opacity) {
            opacities[src_idx] = new_opacity;
        }

        // Compute offset for second split (copy 1)
        float noise1[3];
        int noise_offset1 = 1 * num_split * 3 + split_idx * 3; // copy 1
        noise1[0] = random_noise[noise_offset1 + 0];
        noise1[1] = random_noise[noise_offset1 + 1];
        noise1[2] = random_noise[noise_offset1 + 2];

        float scaled_noise1[3];
        scaled_noise1[0] = expf(scale[0]) * noise1[0];
        scaled_noise1[1] = expf(scale[1]) * noise1[1];
        scaled_noise1[2] = expf(scale[2]) * noise1[2];

        float offset1[3];
        matvec_3x3(R, scaled_noise1, offset1);

        // Write second split result to output arrays
        second_positions[split_idx * 3 + 0] = pos[0] + offset1[0];
        second_positions[split_idx * 3 + 1] = pos[1] + offset1[1];
        second_positions[split_idx * 3 + 2] = pos[2] + offset1[2];

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

    void launch_split_gaussians_inplace(
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
        const float* random_noise,
        int num_split,
        int shN_dim,
        bool revised_opacity,
        cudaStream_t stream) {
        if (num_split == 0)
            return;

        const int block_size = 256;
        const int num_blocks = (num_split + block_size - 1) / block_size;

        split_gaussians_inplace_kernel<<<num_blocks, block_size, 0, stream>>>(
            positions, rotations, scales, sh0, shN, opacities,
            second_positions, second_rotations, second_scales,
            second_sh0, second_shN, second_opacities,
            split_indices, random_noise,
            num_split, shN_dim, revised_opacity);
    }

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
        float* __restrict__ opacities,        // [N, 1] - modified if revised
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
        if (num_split == 0)
            return;

        const int block_size = 256;
        const int num_blocks = (num_split + block_size - 1) / block_size;

        long_axis_split_gaussians_inplace_kernel<<<num_blocks, block_size, 0, stream>>>(
            positions, rotations, scales, sh0, shN, opacities,
            second_positions, second_rotations, second_scales,
            second_sh0, second_shN, second_opacities,
            split_indices, num_split, shN_dim);
    }

} // namespace lfs::training::kernels
