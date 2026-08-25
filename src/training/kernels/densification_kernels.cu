// densification_kernels.cu
/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include "densification_kernels.hpp"
#include "lfs/training/refine_scratch.hpp"
#include <cub/cub.cuh>
#include <limits>

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

    __global__ void fill_free_slots_fused_kernel(
        const int64_t* __restrict__ target_indices,
        size_t n_fill,
        const float* __restrict__ src_means,
        const float* __restrict__ src_rotations,
        const float* __restrict__ src_scales,
        const float* __restrict__ src_sh0,
        const float* __restrict__ src_opacities,
        float* __restrict__ dst_means,
        float* __restrict__ dst_rotations,
        float* __restrict__ dst_scales,
        float* __restrict__ dst_sh0,
        float* __restrict__ dst_opacities,
        int opacity_dim,
        float* const* __restrict__ adam_scale_ptrs,
        int n_adam_scales,
        bool* __restrict__ free_mask,
        size_t N) {

        const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= n_fill)
            return;

        const int64_t t = target_indices[i];
        if (t < 0 || static_cast<size_t>(t) >= N)
            return;
        const size_t dst = static_cast<size_t>(t);

        dst_means[dst * 3 + 0] = src_means[i * 3 + 0];
        dst_means[dst * 3 + 1] = src_means[i * 3 + 1];
        dst_means[dst * 3 + 2] = src_means[i * 3 + 2];

        dst_rotations[dst * 4 + 0] = src_rotations[i * 4 + 0];
        dst_rotations[dst * 4 + 1] = src_rotations[i * 4 + 1];
        dst_rotations[dst * 4 + 2] = src_rotations[i * 4 + 2];
        dst_rotations[dst * 4 + 3] = src_rotations[i * 4 + 3];

        dst_scales[dst * 3 + 0] = src_scales[i * 3 + 0];
        dst_scales[dst * 3 + 1] = src_scales[i * 3 + 1];
        dst_scales[dst * 3 + 2] = src_scales[i * 3 + 2];

        // sh0 is 3 floats/row whether layout is [N,3] or [N,1,3]
        dst_sh0[dst * 3 + 0] = src_sh0[i * 3 + 0];
        dst_sh0[dst * 3 + 1] = src_sh0[i * 3 + 1];
        dst_sh0[dst * 3 + 2] = src_sh0[i * 3 + 2];

        if (opacity_dim == 1) {
            dst_opacities[dst] = src_opacities[i]; // [N,1] still one float per row when col=1
        } else {
            dst_opacities[dst] = src_opacities[i];
        }

        for (int a = 0; a < n_adam_scales; ++a) {
            float* scales = adam_scale_ptrs[a];
            if (scales != nullptr) {
                scales[dst] = 0.0f;
            }
        }

        if (free_mask != nullptr) {
            free_mask[dst] = false;
        }
    }

    void launch_fill_free_slots_fused(
        const int64_t* target_indices,
        size_t n_fill,
        const float* src_means,
        const float* src_rotations,
        const float* src_scales,
        const float* src_sh0,
        const float* src_opacities,
        float* dst_means,
        float* dst_rotations,
        float* dst_scales,
        float* dst_sh0,
        float* dst_opacities,
        int opacity_dim,
        float* const* adam_scale_ptrs,
        int n_adam_scales,
        bool* free_mask,
        size_t N,
        cudaStream_t stream) {

        stream = resolve_stream(stream);
        if (n_fill == 0)
            return;

        // Copy pointer table to device (tiny; stack H2D once per launch).
        float** d_adam = nullptr;
        if (n_adam_scales > 0 && adam_scale_ptrs != nullptr) {
            LFS_CUDA_CHECK_MSG(
                cudaMallocAsync(reinterpret_cast<void**>(&d_adam),
                                sizeof(float*) * static_cast<size_t>(n_adam_scales), stream),
                "fill_free_slots adam ptr table");
            LFS_CUDA_CHECK_MSG(
                cudaMemcpyAsync(d_adam, adam_scale_ptrs,
                                sizeof(float*) * static_cast<size_t>(n_adam_scales),
                                cudaMemcpyHostToDevice, stream),
                "fill_free_slots adam ptr H2D");
        }

        const int block = 256;
        const int grid = static_cast<int>((n_fill + block - 1) / block);
        fill_free_slots_fused_kernel<<<grid, block, 0, stream>>>(
            target_indices, n_fill,
            src_means, src_rotations, src_scales, src_sh0, src_opacities,
            dst_means, dst_rotations, dst_scales, dst_sh0, dst_opacities,
            opacity_dim, d_adam, n_adam_scales, free_mask, N);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.densify.fill_free_slots_fused");

        if (d_adam != nullptr) {
            LFS_CUDA_CHECK_MSG(cudaFreeAsync(d_adam, stream), "fill_free_slots free adam ptrs");
        }
    }

    __global__ void zero_adam_scales_kernel(
        const int64_t* __restrict__ indices,
        size_t n_indices,
        float* const* __restrict__ adam_scale_ptrs,
        int n_adam_scales,
        size_t N) {

        const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= n_indices)
            return;
        const int64_t t = indices[i];
        if (t < 0 || static_cast<size_t>(t) >= N)
            return;
        const size_t dst = static_cast<size_t>(t);
        for (int a = 0; a < n_adam_scales; ++a) {
            float* scales = adam_scale_ptrs[a];
            if (scales != nullptr) {
                scales[dst] = 0.0f;
            }
        }
    }

    void launch_zero_adam_scales_at_indices(
        const int64_t* indices,
        size_t n_indices,
        float* const* adam_scale_ptrs,
        int n_adam_scales,
        size_t N,
        cudaStream_t stream) {

        stream = resolve_stream(stream);
        if (n_indices == 0 || n_adam_scales <= 0)
            return;

        float** d_adam = nullptr;
        LFS_CUDA_CHECK_MSG(
            cudaMallocAsync(reinterpret_cast<void**>(&d_adam),
                            sizeof(float*) * static_cast<size_t>(n_adam_scales), stream),
            "zero_adam adam ptr table");
        LFS_CUDA_CHECK_MSG(
            cudaMemcpyAsync(d_adam, adam_scale_ptrs,
                            sizeof(float*) * static_cast<size_t>(n_adam_scales),
                            cudaMemcpyHostToDevice, stream),
            "zero_adam adam ptr H2D");

        const int block = 256;
        const int grid = static_cast<int>((n_indices + block - 1) / block);
        zero_adam_scales_kernel<<<grid, block, 0, stream>>>(
            indices, n_indices, d_adam, n_adam_scales, N);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.densify.zero_adam_scales");
        LFS_CUDA_CHECK_MSG(cudaFreeAsync(d_adam, stream), "zero_adam free ptrs");
    }

    __global__ void packed_refine_counts_kernel(
        const bool* __restrict__ bool0,
        size_t n_bool0,
        const bool* __restrict__ bool1,
        size_t n_bool1,
        const float* __restrict__ float0,
        size_t n_float0,
        const float* __restrict__ float1,
        size_t n_float1,
        int64_t* __restrict__ out_counts4) {

        // One block does all four reductions via shared atomics / CUB block reduce.
        typedef cub::BlockReduce<int64_t, 256> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp;

        const int tid = threadIdx.x;
        int64_t local0 = 0, local1 = 0, local2 = 0, local3 = 0;

        if (bool0 != nullptr) {
            for (size_t i = tid; i < n_bool0; i += blockDim.x)
                local0 += bool0[i] ? 1 : 0;
        }
        if (bool1 != nullptr) {
            for (size_t i = tid; i < n_bool1; i += blockDim.x)
                local1 += bool1[i] ? 1 : 0;
        }
        if (float0 != nullptr) {
            for (size_t i = tid; i < n_float0; i += blockDim.x)
                local2 += (float0[i] > 0.0f) ? 1 : 0;
        }
        if (float1 != nullptr) {
            for (size_t i = tid; i < n_float1; i += blockDim.x)
                local3 += (float1[i] > 0.0f) ? 1 : 0;
        }

        const int64_t sum0 = BlockReduce(temp).Sum(local0);
        __syncthreads();
        const int64_t sum1 = BlockReduce(temp).Sum(local1);
        __syncthreads();
        const int64_t sum2 = BlockReduce(temp).Sum(local2);
        __syncthreads();
        const int64_t sum3 = BlockReduce(temp).Sum(local3);

        if (tid == 0) {
            out_counts4[0] = (bool0 != nullptr) ? sum0 : 0;
            out_counts4[1] = (bool1 != nullptr) ? sum1 : 0;
            out_counts4[2] = (float0 != nullptr) ? sum2 : 0;
            out_counts4[3] = (float1 != nullptr) ? sum3 : 0;
        }
    }

    void launch_packed_refine_counts(
        const bool* bool0,
        size_t n_bool0,
        const bool* bool1,
        size_t n_bool1,
        const float* float0,
        size_t n_float0,
        const float* float1,
        size_t n_float1,
        int64_t* out_counts4,
        cudaStream_t stream) {

        stream = resolve_stream(stream);
        packed_refine_counts_kernel<<<1, 256, 0, stream>>>(
            bool0, n_bool0, bool1, n_bool1,
            float0, n_float0, float1, n_float1,
            out_counts4);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.densify.packed_refine_counts");
    }

    __global__ void zero_nan_kernel(float* data, size_t n) {
        const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < n && isnan(data[i]))
            data[i] = 0.0f;
    }

    struct PositivePred {
        __host__ __device__ bool operator()(const float& x) const { return x > 0.0f; }
    };

    __global__ void div_by_device_scalar_kernel(
        float* data, size_t n, const float* scalar, float skip_below) {
        const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= n)
            return;
        const float s = fmaxf(*scalar, 1e-9f);
        if (s <= skip_below)
            return;
        data[i] /= s;
    }

    __global__ void fill_pos_inf_kernel(float* data, size_t n) {
        const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < n)
            data[i] = INFINITY;
    }

    __global__ void div_by_positive_median_or_zero_kernel(
        float* data, size_t n, const float* sorted, const int* count) {
        const int c = *count;
        const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= n)
            return;
        if (c <= 0) {
            data[i] = 0.0f;
            return;
        }
        data[i] /= fmaxf(sorted[c / 2], 1e-9f);
    }

    void launch_normalize_by_positive_median(
        float* data,
        size_t n,
        cudaStream_t stream,
        PositiveMedianScratch* scratch) {

        stream = resolve_stream(stream);
        if (n == 0 || data == nullptr)
            return;

        const int block = 256;
        const int grid = static_cast<int>((n + block - 1) / block);
        zero_nan_kernel<<<grid, block, 0, stream>>>(data, n);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.densify.zero_nan");

        if (scratch) {
            LFS_ASSERT_MSG(n <= static_cast<size_t>(std::numeric_limits<int>::max()),
                           "positive-median input exceeds CUB's int item-count limit");
            scratch->ensure_n(n, lfs::core::Device::CUDA);
            LFS_ASSERT_MSG(scratch->n_capacity >= n &&
                               scratch->selected.is_valid() &&
                               scratch->selected.ptr<float>() != nullptr,
                           lfs::core::detail::format_cuda_safe(
                               "positive-median selected scratch must cover n (cap={}, n={})",
                               scratch->n_capacity, n));
            LFS_ASSERT_MSG(scratch->sorted.is_valid() && scratch->sorted.ptr<float>() != nullptr,
                           "positive-median sorted scratch must be a non-null CUDA f32 tensor");
            LFS_ASSERT_MSG(scratch->count.is_valid() && scratch->count.ptr<int>() != nullptr,
                           "positive-median count scratch must be a non-null CUDA i32 tensor");

            float* d_selected = scratch->selected.ptr<float>();
            float* d_sorted = scratch->sorted.ptr<float>();
            int* d_count = scratch->count.ptr<int>();
            const int n_int = static_cast<int>(n);

            fill_pos_inf_kernel<<<grid, block, 0, stream>>>(d_selected, n);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.densify.positive_median_fill_inf");

            size_t temp_bytes = 0;
            LFS_CUDA_CHECK_MSG(
                cub::DeviceSelect::If(nullptr, temp_bytes, data, d_selected, d_count,
                                      n_int, PositivePred{}, stream),
                "positive_median select size");
            scratch->ensure_temps(temp_bytes, 0, lfs::core::Device::CUDA);
            LFS_ASSERT_MSG(temp_bytes == 0 ||
                               (scratch->select_temp.is_valid() &&
                                scratch->select_temp_bytes >= temp_bytes &&
                                scratch->select_temp.data_ptr() != nullptr),
                           lfs::core::detail::format_cuda_safe(
                               "positive-median select temp must cover queried bytes (have={}, need={})",
                               scratch->select_temp_bytes, temp_bytes));
            LFS_CUDA_CHECK_MSG(
                cub::DeviceSelect::If(
                    temp_bytes == 0 ? nullptr : scratch->select_temp.data_ptr(),
                    temp_bytes, data, d_selected, d_count,
                    n_int, PositivePred{}, stream),
                "positive_median select");

            size_t sort_bytes = 0;
            LFS_CUDA_CHECK_MSG(
                cub::DeviceRadixSort::SortKeys(nullptr, sort_bytes, d_selected, d_sorted,
                                               n_int, 0, sizeof(float) * 8, stream),
                "positive_median sort size");
            scratch->ensure_temps(temp_bytes, sort_bytes, lfs::core::Device::CUDA);
            LFS_ASSERT_MSG(sort_bytes == 0 ||
                               (scratch->sort_temp.is_valid() &&
                                scratch->sort_temp_bytes >= sort_bytes &&
                                scratch->sort_temp.data_ptr() != nullptr),
                           lfs::core::detail::format_cuda_safe(
                               "positive-median sort temp must cover queried bytes (have={}, need={})",
                               scratch->sort_temp_bytes, sort_bytes));
            LFS_CUDA_CHECK_MSG(
                cub::DeviceRadixSort::SortKeys(
                    sort_bytes == 0 ? nullptr : scratch->sort_temp.data_ptr(),
                    sort_bytes, d_selected, d_sorted,
                    n_int, 0, sizeof(float) * 8, stream),
                "positive_median sort");

            div_by_positive_median_or_zero_kernel<<<grid, block, 0, stream>>>(
                data, n, d_sorted, d_count);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.densify.div_by_median");
            return;
        }

        // Compact positives into a scratch buffer, radix-sort that only, pick mid.
        // Falls back to no-op (leave data) when zero positives.
        float* d_selected = nullptr;
        int* d_count = nullptr;
        LFS_CUDA_CHECK_MSG(
            cudaMallocAsync(reinterpret_cast<void**>(&d_selected), n * sizeof(float), stream),
            "positive_median selected");
        LFS_CUDA_CHECK_MSG(
            cudaMallocAsync(reinterpret_cast<void**>(&d_count), sizeof(int), stream),
            "positive_median count");
        LFS_CUDA_CHECK_MSG(cudaMemsetAsync(d_count, 0, sizeof(int), stream), "positive_median count z");

        // CUB DeviceSelect::If
        size_t temp_bytes = 0;
        LFS_CUDA_CHECK_MSG(
            cub::DeviceSelect::If(nullptr, temp_bytes, data, d_selected, d_count,
                                  static_cast<int>(n), PositivePred{}, stream),
            "positive_median select size");
        void* d_temp = nullptr;
        if (temp_bytes > 0) {
            LFS_CUDA_CHECK_MSG(
                cudaMallocAsync(&d_temp, temp_bytes, stream), "positive_median select temp");
        }
        LFS_CUDA_CHECK_MSG(
            cub::DeviceSelect::If(d_temp, temp_bytes, data, d_selected, d_count,
                                  static_cast<int>(n), PositivePred{}, stream),
            "positive_median select");

        int h_count = 0;
        LFS_CUDA_CHECK_MSG(
            cudaMemcpyAsync(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost, stream),
            "positive_median count D2H");
        LFS_CUDA_CHECK_MSG(cudaStreamSynchronize(stream), "positive_median count sync");

        if (h_count <= 0) {
            // No positives → zero the tensor (match prior masked_select empty path).
            LFS_CUDA_CHECK_MSG(cudaMemsetAsync(data, 0, n * sizeof(float), stream),
                               "positive_median zero empty");
            if (d_temp)
                cudaFreeAsync(d_temp, stream);
            cudaFreeAsync(d_selected, stream);
            cudaFreeAsync(d_count, stream);
            return;
        }

        // Radix sort the compacted positives only (O(P log P), P << n for sparse edges).
        float* d_sorted = nullptr;
        LFS_CUDA_CHECK_MSG(
            cudaMallocAsync(reinterpret_cast<void**>(&d_sorted),
                            static_cast<size_t>(h_count) * sizeof(float), stream),
            "positive_median sorted");
        size_t sort_bytes = 0;
        LFS_CUDA_CHECK_MSG(
            cub::DeviceRadixSort::SortKeys(nullptr, sort_bytes, d_selected, d_sorted,
                                           h_count, 0, sizeof(float) * 8, stream),
            "positive_median sort size");
        void* d_sort_temp = nullptr;
        if (sort_bytes > 0) {
            LFS_CUDA_CHECK_MSG(
                cudaMallocAsync(&d_sort_temp, sort_bytes, stream), "positive_median sort temp");
        }
        LFS_CUDA_CHECK_MSG(
            cub::DeviceRadixSort::SortKeys(d_sort_temp, sort_bytes, d_selected, d_sorted,
                                           h_count, 0, sizeof(float) * 8, stream),
            "positive_median sort");

        // Median at count/2 (same index as prior sorted[valid.numel()/2]).
        const float* d_median = d_sorted + (h_count / 2);
        div_by_device_scalar_kernel<<<grid, block, 0, stream>>>(data, n, d_median, 0.0f);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.densify.div_by_median");

        if (d_sort_temp)
            cudaFreeAsync(d_sort_temp, stream);
        if (d_temp)
            cudaFreeAsync(d_temp, stream);
        cudaFreeAsync(d_sorted, stream);
        cudaFreeAsync(d_selected, stream);
        cudaFreeAsync(d_count, stream);
    }

} // namespace lfs::training::kernels
