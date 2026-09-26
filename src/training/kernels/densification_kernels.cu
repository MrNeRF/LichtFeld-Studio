// densification_kernels.cu
/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/assert.hpp"
#include "core/cuda_error.hpp"
#include "densification_kernels.hpp"
#include "lfs/cuda_scratch.hpp"
#include "lfs/training/screen_share.cuh"
#include <algorithm>
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
        bool* free_mask,
        size_t N,
        cudaStream_t stream) {

        stream = resolve_stream(stream);
        if (n_fill == 0)
            return;

        const int block = 256;
        const int grid = static_cast<int>((n_fill + block - 1) / block);
        fill_free_slots_fused_kernel<<<grid, block, 0, stream>>>(
            target_indices, n_fill,
            src_means, src_rotations, src_scales, src_sh0, src_opacities,
            dst_means, dst_rotations, dst_scales, dst_sh0, dst_opacities,
            opacity_dim, free_mask, N);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.densify.fill_free_slots_fused");
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

    // Exact order statistic without sorting: three radix passes (11/11/10 bits)
    // over the float keys, each building a histogram of one digit among the
    // values that share the digits already fixed. Keys use the same bit
    // transform as cub::DeviceRadixSort on floats, so the selected element is
    // the one sorted[k] would return, including NaN and signed-zero ordering.
    namespace radix_select {
        constexpr int kBins = 2048;
        constexpr int kThreads = 256;
        __host__ __device__ constexpr int digit_shift(const int pass) { return pass == 0 ? 21 : (pass == 1 ? 10 : 0); }
        __host__ __device__ constexpr int digit_bits(const int pass) { return pass == 2 ? 10 : 11; }

        struct State {
            unsigned int total;
            unsigned int rank;
            unsigned int prefix;
            float value;
        };

        __device__ __forceinline__ unsigned int ordered_key(const float v) {
            const unsigned int bits = __float_as_uint(v);
            return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
        }

        __device__ __forceinline__ float key_value(const unsigned int key) {
            return __uint_as_float((key & 0x80000000u) ? (key & 0x7fffffffu) : ~key);
        }

        template <bool kPositiveOnly, int kPass>
        __global__ void histogram_kernel(const float* __restrict__ values,
                                         const size_t n,
                                         State* __restrict__ state,
                                         unsigned int* __restrict__ hist) {
            __shared__ unsigned int shared_hist[kBins];
            for (int b = threadIdx.x; b < kBins; b += blockDim.x) {
                shared_hist[b] = 0u;
            }
            __syncthreads();

            constexpr int shift = digit_shift(kPass);
            constexpr unsigned int digit_mask = (1u << digit_bits(kPass)) - 1u;
            constexpr int fixed_shift = shift + digit_bits(kPass);
            const unsigned int fixed = kPass == 0 ? 0u : state->prefix >> fixed_shift;
            unsigned int counted = 0u;
            const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
            for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
                const float v = values[i];
                if (kPositiveOnly && !(v > 0.0f)) {
                    continue;
                }
                const unsigned int key = ordered_key(v);
                if constexpr (kPass > 0) {
                    if ((key >> fixed_shift) != fixed) {
                        continue;
                    }
                }
                ++counted;
                atomicAdd(&shared_hist[(key >> shift) & digit_mask], 1u);
            }
            if constexpr (kPass == 0 && kPositiveOnly) {
                for (int offset = 16; offset > 0; offset >>= 1) {
                    counted += __shfl_down_sync(0xffffffffu, counted, offset);
                }
                if ((threadIdx.x & 31) == 0 && counted != 0u) {
                    atomicAdd(&state->total, counted);
                }
            }
            __syncthreads();
            for (int b = threadIdx.x; b < kBins; b += blockDim.x) {
                if (shared_hist[b] != 0u) {
                    atomicAdd(&hist[b], shared_hist[b]);
                }
            }
        }

        // One thread walks the digit histogram; kBins steps are negligible
        // next to the histogram passes and keep the pick deterministic.
        template <int kPass>
        __global__ void pick_kernel(const bool positive_only,
                                    const size_t n,
                                    State* __restrict__ state,
                                    unsigned int* __restrict__ hist) {
            if (threadIdx.x != 0 || blockIdx.x != 0) {
                return;
            }
            if constexpr (kPass == 0) {
                if (!positive_only) {
                    state->total = static_cast<unsigned int>(n);
                }
                state->rank = state->total / 2u;
                state->prefix = 0u;
            }
            constexpr int bins = 1 << digit_bits(kPass);
            unsigned int below = 0u;
            int digit = bins - 1;
            for (int b = 0; b < bins; ++b) {
                const unsigned int count = hist[b];
                if (state->rank < below + count) {
                    digit = b;
                    break;
                }
                below += count;
            }
            for (int b = 0; b < kBins; ++b) {
                hist[b] = 0u;
            }
            state->prefix |= static_cast<unsigned int>(digit) << digit_shift(kPass);
            state->rank -= below;
            if constexpr (kPass == 2) {
                state->value = state->total == 0u ? 0.0f : key_value(state->prefix);
            }
        }

        template <bool kPositiveOnly>
        void run(const float* values, const size_t n, State* state, unsigned int* hist, cudaStream_t stream) {
            const int blocks = static_cast<int>(std::clamp<size_t>(
                (n + static_cast<size_t>(kThreads) * 16 - 1) / (static_cast<size_t>(kThreads) * 16), 1, 512));
            histogram_kernel<kPositiveOnly, 0><<<blocks, kThreads, 0, stream>>>(values, n, state, hist);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.radix_select.histogram0");
            pick_kernel<0><<<1, 32, 0, stream>>>(kPositiveOnly, n, state, hist);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.radix_select.pick0");
            histogram_kernel<kPositiveOnly, 1><<<blocks, kThreads, 0, stream>>>(values, n, state, hist);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.radix_select.histogram1");
            pick_kernel<1><<<1, 32, 0, stream>>>(kPositiveOnly, n, state, hist);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.radix_select.pick1");
            histogram_kernel<kPositiveOnly, 2><<<blocks, kThreads, 0, stream>>>(values, n, state, hist);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.radix_select.histogram2");
            pick_kernel<2><<<1, 32, 0, stream>>>(kPositiveOnly, n, state, hist);
            LFS_CUDA_LAUNCH_CHECK(stream, "training.radix_select.pick2");
        }

        struct Workspace {
            cuda_scratch::DeviceBuffer buffer;
            State* state = nullptr;
            unsigned int* hist = nullptr;

            explicit Workspace(cudaStream_t stream)
                : buffer(sizeof(State) + sizeof(unsigned int) * kBins, stream,
                         "training.radix_select.workspace") {
                state = buffer.as<State>();
                hist = reinterpret_cast<unsigned int*>(buffer.as<char>() + sizeof(State));
                LFS_CUDA_CHECK_MSG(
                    cudaMemsetAsync(buffer.get(), 0, sizeof(State) + sizeof(unsigned int) * kBins, stream),
                    "radix select workspace clear");
            }
        };
    } // namespace radix_select

    __global__ void div_by_positive_median_or_zero_kernel(
        float* data, size_t n, const radix_select::State* median) {
        const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= n)
            return;
        if (median->total == 0u) {
            data[i] = 0.0f;
            return;
        }
        data[i] /= fmaxf(median->value, 1e-9f);
    }

    float launch_select_median(const float* values, const size_t n, cudaStream_t stream) {
        stream = resolve_stream(stream);
        if (n == 0 || values == nullptr) {
            return 0.0f;
        }
        LFS_ASSERT_MSG(n <= static_cast<size_t>(std::numeric_limits<unsigned int>::max()),
                       "median selection input exceeds the 32-bit count limit");
        radix_select::Workspace workspace(stream);
        radix_select::run<false>(values, n, workspace.state, workspace.hist, stream);
        radix_select::State result{};
        LFS_CUDA_CHECK_MSG(
            cudaMemcpyAsync(&result, workspace.state, sizeof(result), cudaMemcpyDeviceToHost, stream),
            "median selection readback");
        LFS_CUDA_CHECK_MSG(cudaStreamSynchronize(stream), "median selection sync");
        return result.value;
    }

    void launch_normalize_by_positive_median(float* data, size_t n, cudaStream_t stream) {
        stream = resolve_stream(stream);
        if (n == 0 || data == nullptr)
            return;
        LFS_ASSERT_MSG(n <= static_cast<size_t>(std::numeric_limits<unsigned int>::max()),
                       "positive-median input exceeds the 32-bit count limit");

        const int block = 256;
        const int grid = static_cast<int>((n + block - 1) / block);
        zero_nan_kernel<<<grid, block, 0, stream>>>(data, n);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.densify.zero_nan");

        radix_select::Workspace workspace(stream);
        radix_select::run<true>(data, n, workspace.state, workspace.hist, stream);
        div_by_positive_median_or_zero_kernel<<<grid, block, 0, stream>>>(data, n, workspace.state);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.densify.div_by_median");
    }

    namespace {
        __global__ void accumulate_projected_screen_share_kernel(
            const int32_t* __restrict__ radii, const float* __restrict__ means2d,
            float* shares, size_t n, uint32_t width, uint32_t height) {
            const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= n || radii[2 * i] <= 0 || radii[2 * i + 1] <= 0)
                return;
            const float x = means2d[2 * i], y = means2d[2 * i + 1];
            const float rx = radii[2 * i], ry = radii[2 * i + 1];
            const float dx = fmaxf(0.f, fminf(float(width), x + rx) - fmaxf(0.f, x - rx));
            const float dy = fmaxf(0.f, fminf(float(height), y + ry) - fmaxf(0.f, y - ry));
            const float share = (dx / float(width)) * (dy / float(height));
            // One writer per splat, once per frame on the training stream.
            // Retain the maximum across views until the strategy resets it.
            shares[i] = fmaxf(shares[i], share);
        }
    } // namespace

    void launch_accumulate_projected_screen_share(
        const int32_t* radii, const float* means2d,
        float* shares, size_t n, uint32_t width, uint32_t height, cudaStream_t stream) {
        if (n == 0 || width == 0 || height == 0)
            return;
        accumulate_projected_screen_share_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
            radii, means2d, shares, n, width, height);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.densify.accumulate_projected_screen_share");
    }

    namespace {
        __global__ void clip_log_scale_by_screen_share_kernel(
            float* __restrict__ log_scales,
            const float* __restrict__ max_share,
            const bool* __restrict__ frozen_mask,
            size_t frozen_n,
            float limit,
            size_t n) {
            const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= n)
                return;
            if (frozen_mask != nullptr && i < frozen_n && frozen_mask[i])
                return;
            const float share = max_share[i];
            if (!(share > limit))
                return;
            const float delta = fminf(logf(share / limit), logf(1.5f));
            float scale[3] = {
                log_scales[i * 3 + 0],
                log_scales[i * 3 + 1],
                log_scales[i * 3 + 2]};
            const unsigned int axis = get_max_value_index(scale).x;
            log_scales[i * 3 + axis] -= delta;
        }

        __global__ void oversize_split_scores_kernel(
            const float* __restrict__ error_score,
            const float* __restrict__ max_share,
            const bool* __restrict__ frozen_mask,
            size_t frozen_n,
            float* __restrict__ out_scores,
            float limit,
            size_t n) {
            const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= n)
                return;
            if (frozen_mask != nullptr && i < frozen_n && frozen_mask[i]) {
                out_scores[i] = 0.0f;
                return;
            }
            out_scores[i] = lfs::training::oversize_split_score(
                error_score[i], max_share[i], limit);
        }
    } // namespace

    void launch_clip_log_scale_by_screen_share(
        float* log_scales,
        const float* max_share,
        const bool* frozen_mask,
        size_t frozen_n,
        float limit,
        size_t n,
        cudaStream_t stream) {
        LFS_ASSERT_MSG(log_scales != nullptr && max_share != nullptr,
                       "screen-share clip requires log_scales and max_share");
        if (n == 0 || !(limit > 0.0f) || !(limit < 1.0f))
            return;
        stream = lfs::resolve_stream(stream);
        constexpr int kBlock = 256;
        const int blocks = static_cast<int>((n + kBlock - 1) / kBlock);
        clip_log_scale_by_screen_share_kernel<<<blocks, kBlock, 0, stream>>>(
            log_scales, max_share, frozen_mask, frozen_n, limit, n);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.densify.clip_screen_share");
    }

    void launch_oversize_split_scores(
        const float* error_score,
        const float* max_share,
        const bool* frozen_mask,
        size_t frozen_n,
        float* out_scores,
        float limit,
        size_t n,
        cudaStream_t stream) {
        LFS_ASSERT_MSG(error_score != nullptr && max_share != nullptr && out_scores != nullptr,
                       "oversize-split scores require error, max_share, and output");
        if (n == 0)
            return;
        stream = lfs::resolve_stream(stream);
        constexpr int kBlock = 256;
        const int blocks = static_cast<int>((n + kBlock - 1) / kBlock);
        oversize_split_scores_kernel<<<blocks, kBlock, 0, stream>>>(
            error_score, max_share, frozen_mask, frozen_n, out_scores, limit, n);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.densify.oversize_split_scores");
    }

} // namespace lfs::training::kernels
