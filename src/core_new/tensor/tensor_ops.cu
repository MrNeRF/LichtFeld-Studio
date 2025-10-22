/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "internal/memory_pool.hpp"
#include "internal/tensor_functors.hpp"
#include "internal/tensor_impl.hpp"
#include "internal/tensor_ops.hpp"
#include "internal/warp_reduce.cuh"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <limits>

// Thrust headers
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

namespace lfs::core::tensor_ops {

    // ============= GENERIC OPERATIONS - NOW IN HEADER =============
    // Template implementations moved to include/core/tensor_generic_ops.cuh for:
    // - Better inlining and optimization
    // - Support for expression template fusion (composed functors)
    // - No need for explicit instantiations
    // - Faster compilation (no separable compilation needed for these)

    // ============= CLAMP OPERATIONS (USING FUNCTORS) =============

    // Vectorized clamp kernel with float4 loads (2-4x faster!)
    __global__ void clamp_kernel_vectorized(float* __restrict__ data, float min_val, float max_val, size_t n) {
        const size_t vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t idx = vec_idx * 4;

        // Vectorized path: Load 4 floats in one transaction
        if (idx + 3 < n) {
            float4 vals = reinterpret_cast<float4*>(data)[vec_idx];

            // Clamp all 4 values
            vals.x = fminf(fmaxf(vals.x, min_val), max_val);
            vals.y = fminf(fmaxf(vals.y, min_val), max_val);
            vals.z = fminf(fmaxf(vals.z, min_val), max_val);
            vals.w = fminf(fmaxf(vals.w, min_val), max_val);

            // Store 4 floats in one transaction
            reinterpret_cast<float4*>(data)[vec_idx] = vals;
        }
        // Scalar fallback for remainder
        else if (idx < n) {
            for (size_t i = idx; i < n; ++i) {
                float val = data[i];
                val = fmaxf(val, min_val);
                val = fminf(val, max_val);
                data[i] = val;
            }
        }
    }

    // Optimized clamp kernel with perfect memory coalescing (FALLBACK for unaligned data)
    __global__ void clamp_kernel_optimized(float* __restrict__ data, float min_val, float max_val, size_t n) {
        // Sequential access pattern for perfect coalescing within warps
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;

        // Grid-stride loop for any array size
        for (size_t i = idx; i < n; i += stride) {
            // Single precision math, no NaN check (PyTorch doesn't check either)
            float val = data[i];
            val = fmaxf(val, min_val); // max(val, min)
            val = fminf(val, max_val); // min(result, max)
            data[i] = val;
        }
    }

    void launch_clamp_scalar(float* data, float min_val, float max_val, size_t n, cudaStream_t stream) {
        if (n == 0)
            return;

        // Check alignment for float4 vectorization
        bool is_aligned = (reinterpret_cast<uintptr_t>(data) % 16) == 0;

        // Optimized launch configuration for maximum occupancy
        constexpr int BLOCK_SIZE = 256;

        // Use vectorized kernel if aligned and large enough
        if (is_aligned && n > 1024) {
            // IMPORTANT: Each thread processes up to 4 elements, but we need enough threads
            // to cover all elements. Grid size should ensure ALL elements are processed.
            int num_threads_needed = (n + 3) / 4; // Round up
            int grid_size = (num_threads_needed + BLOCK_SIZE - 1) / BLOCK_SIZE;
            // Don't cap grid_size - we need to process ALL elements!

            clamp_kernel_vectorized<<<grid_size, BLOCK_SIZE, 0, stream>>>(data, min_val, max_val, n);
        } else {
            // Fallback to scalar kernel for unaligned data
            int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            grid_size = min(grid_size, 2048);

            clamp_kernel_optimized<<<grid_size, BLOCK_SIZE, 0, stream>>>(data, min_val, max_val, n);
        }
    }

    // Vectorized fused clamp kernel (2-4x faster!)
    __global__ void clamp_kernel_fused_vectorized(const float* __restrict__ src, float* __restrict__ dst,
                                                  float min_val, float max_val, size_t n) {
        const size_t vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t idx = vec_idx * 4;

        if (idx + 3 < n) {
            float4 vals = reinterpret_cast<const float4*>(src)[vec_idx];

            vals.x = fminf(fmaxf(vals.x, min_val), max_val);
            vals.y = fminf(fmaxf(vals.y, min_val), max_val);
            vals.z = fminf(fmaxf(vals.z, min_val), max_val);
            vals.w = fminf(fmaxf(vals.w, min_val), max_val);

            reinterpret_cast<float4*>(dst)[vec_idx] = vals;
        } else if (idx < n) {
            for (size_t i = idx; i < n; ++i) {
                float val = src[i];
                val = fmaxf(val, min_val);
                val = fminf(val, max_val);
                dst[i] = val;
            }
        }
    }

    // Fused clamp kernel - reads from src, writes clamped to dst (non-in-place) - FALLBACK
    __global__ void clamp_kernel_fused(const float* __restrict__ src, float* __restrict__ dst,
                                       float min_val, float max_val, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;

        for (size_t i = idx; i < n; i += stride) {
            float val = src[i];
            val = fmaxf(val, min_val);
            val = fminf(val, max_val);
            dst[i] = val;
        }
    }

    void launch_clamp_fused(const float* src, float* dst, float min_val, float max_val,
                            size_t n, cudaStream_t stream) {
        if (n == 0)
            return;

        bool src_aligned = (reinterpret_cast<uintptr_t>(src) % 16) == 0;
        bool dst_aligned = (reinterpret_cast<uintptr_t>(dst) % 16) == 0;

        constexpr int BLOCK_SIZE = 256;

        if (src_aligned && dst_aligned && n > 1024) {
            int num_threads_needed = (n + 3) / 4;
            int grid_size = (num_threads_needed + BLOCK_SIZE - 1) / BLOCK_SIZE;
            // Don't cap grid_size!

            clamp_kernel_fused_vectorized<<<grid_size, BLOCK_SIZE, 0, stream>>>(src, dst, min_val, max_val, n);
        } else {
            int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            grid_size = min(grid_size, 2048);

            clamp_kernel_fused<<<grid_size, BLOCK_SIZE, 0, stream>>>(src, dst, min_val, max_val, n);
        }
    }

    void launch_clamp_scalar_int(int* data, int min_val, int max_val, size_t n, cudaStream_t stream) {
        if (n == 0)
            return;
        auto data_ptr = thrust::device_pointer_cast(data);

        run_with_thrust_policy(stream, [&](auto policy) {
            thrust::transform(policy, data_ptr, data_ptr + n, data_ptr,
                              ops::clamp_range_op<int>(min_val, max_val));
        });
    }

    // ============= TYPE CONVERSIONS (USING FUNCTORS) =============

    template <typename SrcT, typename DstT>
    struct ConvertFunctor {
        __device__ DstT operator()(SrcT x) const {
            return static_cast<DstT>(x);
        }
    };

    // Specializations for clamping conversions
    template <>
    struct ConvertFunctor<float, uint8_t> {
        __device__ uint8_t operator()(float x) const {
            return static_cast<uint8_t>(fminf(fmaxf(roundf(x), 0.0f), 255.0f));
        }
    };

    template <>
    struct ConvertFunctor<int, uint8_t> {
        __device__ uint8_t operator()(int x) const {
            return static_cast<uint8_t>(max(0, min(255, x)));
        }
    };

    template <>
    struct ConvertFunctor<int64_t, uint8_t> {
        __device__ uint8_t operator()(int64_t x) const {
            return static_cast<uint8_t>(max(static_cast<int64_t>(0),
                                            min(static_cast<int64_t>(255), x)));
        }
    };

    template <typename SrcT, typename DstT>
    void launch_convert_type(const SrcT* src, DstT* dst, size_t n, cudaStream_t stream) {
        if (n == 0)
            return;
        auto src_ptr = thrust::device_pointer_cast(src);
        auto dst_ptr = thrust::device_pointer_cast(dst);
        run_with_thrust_policy(stream, [&](auto policy) {
            thrust::transform(policy, src_ptr, src_ptr + n, dst_ptr, ConvertFunctor<SrcT, DstT>());
        });
    }

    // ============= BROADCASTING BINARY OPERATIONS =============
    // NOTE: launch_broadcast_binary is now defined in tensor_broadcast_ops.cuh
    // CUDA kernels and host function template are inlined for correct instantiation

    // ============= OPTIMIZED SEGMENTED REDUCTION =============

    template <typename T, typename Op>
    void launch_segmented_reduce(
        const T* input, T* output,
        size_t outer_size, size_t reduce_size, size_t inner_size,
        T init_value, Op op, cudaStream_t stream) {
        if (outer_size == 0 || reduce_size == 0 || inner_size == 0)
            return;

        size_t output_size = outer_size * inner_size;

        // Special case: no reduction needed
        if (reduce_size == 1) {
            auto in_ptr = thrust::device_pointer_cast(input);
            auto out_ptr = thrust::device_pointer_cast(output);
            if (stream) {
                thrust::copy(thrust::cuda::par.on(stream),
                             in_ptr, in_ptr + output_size, out_ptr);
            } else {
                thrust::copy(thrust::cuda::par,
                             in_ptr, in_ptr + output_size, out_ptr);
            }
            return;
        }

        // OPTIMIZED PATH: Contiguous segments - use CUB's segmented reduce
        if (inner_size == 1) {
            // begin_offsets: [0, N, 2N, 3N, ...]
            auto begin_offsets = thrust::make_transform_iterator(
                thrust::counting_iterator<int>(0),
                [reduce_size] __host__ __device__(int i) -> int {
                    return i * static_cast<int>(reduce_size);
                });

            // end_offsets: [N, 2N, 3N, 4N, ...]
            auto end_offsets = thrust::make_transform_iterator(
                thrust::counting_iterator<int>(1),
                [reduce_size] __host__ __device__(int i) -> int {
                    return i * static_cast<int>(reduce_size);
                });

            void* d_temp_storage = nullptr;
            size_t temp_storage_bytes = 0;

            // First call to determine temp storage size needed
            cub::DeviceSegmentedReduce::Reduce(
                d_temp_storage,
                temp_storage_bytes,
                input,
                output,
                static_cast<int>(outer_size),
                begin_offsets,
                end_offsets,
                op,
                init_value,
                stream);

            // Allocate temp storage from memory pool (fast!)
            d_temp_storage = CudaMemoryPool::instance().allocate(temp_storage_bytes, stream);
            if (!d_temp_storage) {
                LOG_ERROR("Failed to allocate {} bytes for CUB temp storage from memory pool",
                          temp_storage_bytes);
                return;
            }

            // Actual reduction with temp storage
            cub::DeviceSegmentedReduce::Reduce(
                d_temp_storage,
                temp_storage_bytes,
                input,
                output,
                static_cast<int>(outer_size),
                begin_offsets,
                end_offsets,
                op,
                init_value,
                stream);

            // Return temp storage to memory pool (instant, cached for reuse)
            CudaMemoryPool::instance().deallocate(d_temp_storage, stream);
            return;
        }

        // STRIDED PATH: Non-contiguous segments
        // For strided reductions (e.g., dim0 on [1024, 1024]), we have inner_size > 1
        //
        // The Thrust lambda approach is slow but simple. A better approach would be
        // to use a custom CUDA kernel, but for now we keep the Thrust fallback.
        //
        // TODO: Implement optimized strided reduction kernel or use CUB with proper setup
        if (stream) {
            thrust::for_each(thrust::cuda::par.on(stream),
                             thrust::counting_iterator<size_t>(0),
                             thrust::counting_iterator<size_t>(output_size),
                             [=] __device__(size_t out_idx) {
                                 size_t outer_idx = out_idx / inner_size;
                                 size_t inner_idx = out_idx % inner_size;

                                 T result = init_value;
                                 for (size_t r = 0; r < reduce_size; ++r) {
                                     size_t in_idx = (outer_idx * reduce_size + r) * inner_size + inner_idx;
                                     result = op(result, input[in_idx]);
                                 }
                                 output[out_idx] = result;
                             });
        } else {
            thrust::for_each(thrust::cuda::par,
                             thrust::counting_iterator<size_t>(0),
                             thrust::counting_iterator<size_t>(output_size),
                             [=] __device__(size_t out_idx) {
                                 size_t outer_idx = out_idx / inner_size;
                                 size_t inner_idx = out_idx % inner_size;

                                 T result = init_value;
                                 for (size_t r = 0; r < reduce_size; ++r) {
                                     size_t in_idx = (outer_idx * reduce_size + r) * inner_size + inner_idx;
                                     result = op(result, input[in_idx]);
                                 }
                                 output[out_idx] = result;
                             });
        }
    }

    // ============= MULTI-AXIS REDUCTION (USING FUNCTORS) =============

    template <typename Op>
    __global__ void multi_axis_reduce_kernel(
        const float* input, float* output,
        const size_t* input_shape, const bool* is_reduced_dim,
        size_t input_rank, size_t output_elements, float init_val,
        Op op) {
        size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (out_idx >= output_elements)
            return;

        size_t input_strides[10];
        input_strides[input_rank - 1] = 1;
        for (int i = input_rank - 2; i >= 0; --i) {
            input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
        }

        size_t out_shape[10];
        size_t out_rank = 0;
        for (size_t i = 0; i < input_rank; ++i) {
            if (!is_reduced_dim[i]) {
                out_shape[out_rank++] = input_shape[i];
            }
        }

        size_t output_strides[10];
        if (out_rank > 0) {
            output_strides[out_rank - 1] = 1;
            for (int i = out_rank - 2; i >= 0; --i) {
                output_strides[i] = output_strides[i + 1] * out_shape[i + 1];
            }
        }

        size_t out_coords[10] = {0};
        size_t temp = out_idx;
        for (size_t i = 0; i < out_rank; ++i) {
            out_coords[i] = temp / output_strides[i];
            temp %= output_strides[i];
        }

        size_t base_input_coords[10];
        size_t out_coord_idx = 0;
        for (size_t i = 0; i < input_rank; ++i) {
            base_input_coords[i] = is_reduced_dim[i] ? 0 : out_coords[out_coord_idx++];
        }

        size_t reduce_count = 1;
        size_t reduced_dims[10];
        size_t num_reduced = 0;
        for (size_t i = 0; i < input_rank; ++i) {
            if (is_reduced_dim[i]) {
                reduced_dims[num_reduced++] = i;
                reduce_count *= input_shape[i];
            }
        }

        float result = init_val;
        for (size_t r = 0; r < reduce_count; ++r) {
            size_t temp_r = r;
            size_t full_input_coords[10];

            for (size_t i = 0; i < input_rank; ++i) {
                full_input_coords[i] = base_input_coords[i];
            }

            for (int rd_idx = num_reduced - 1; rd_idx >= 0; --rd_idx) {
                size_t dim = reduced_dims[rd_idx];
                full_input_coords[dim] = temp_r % input_shape[dim];
                temp_r /= input_shape[dim];
            }

            size_t in_idx = 0;
            for (size_t i = 0; i < input_rank; ++i) {
                in_idx += full_input_coords[i] * input_strides[i];
            }

            result = op(result, input[in_idx]);
        }

        output[out_idx] = result;
    }

    void launch_multi_axis_reduce(
        const float* input, float* output,
        const size_t* input_shape, const bool* is_reduced_dim,
        size_t input_rank, size_t output_elements,
        float init_val, ReduceOp op, cudaStream_t stream) {
        int blocks = (output_elements + 255) / 256;

        switch (op) {
        case ReduceOp::Sum:
        case ReduceOp::Mean:
            multi_axis_reduce_kernel<<<blocks, 256, 0, stream>>>(
                input, output, input_shape, is_reduced_dim,
                input_rank, output_elements, init_val, ops::add_op{});
            break;
        case ReduceOp::Max:
            multi_axis_reduce_kernel<<<blocks, 256, 0, stream>>>(
                input, output, input_shape, is_reduced_dim,
                input_rank, output_elements, init_val, ops::maximum_op{});
            break;
        case ReduceOp::Min:
            multi_axis_reduce_kernel<<<blocks, 256, 0, stream>>>(
                input, output, input_shape, is_reduced_dim,
                input_rank, output_elements, init_val, ops::minimum_op{});
            break;
        case ReduceOp::Prod:
            multi_axis_reduce_kernel<<<blocks, 256, 0, stream>>>(
                input, output, input_shape, is_reduced_dim,
                input_rank, output_elements, init_val, ops::mul_op{});
            break;
        }
    }

    struct DivideByFunctor {
        float divisor;
        DivideByFunctor(float d) : divisor(d) {}
        __device__ float operator()(float x) const { return x / divisor; }
    };

    // ============= MAIN REDUCE OPERATION DISPATCH =============

    void launch_reduce_op(const void* input, void* output, const size_t* shape, size_t rank,
                          const int* axes, size_t num_axes, bool keepdim, ReduceOp op,
                          DataType dtype, cudaStream_t stream) {
        if (dtype != DataType::Float32)
            return;

        size_t n = 1;
        for (size_t i = 0; i < rank; ++i)
            n *= shape[i];
        if (n == 0)
            return;

        auto input_ptr = thrust::device_pointer_cast(static_cast<const float*>(input));
        auto output_ptr = thrust::device_pointer_cast(static_cast<float*>(output));

        // Full reduction to scalar
        if (num_axes == 0 || num_axes == rank) {
            const float* d_in = static_cast<const float*>(input);
            float* d_out = static_cast<float*>(output);

            // FAST PATH: Use warp-level reduction for small-medium tensors
            if (should_use_warp_reduce(n, 1)) {
                // Initialize output to appropriate init value
                float init_val = 0.0f;
                switch (op) {
                case ReduceOp::Sum:
                case ReduceOp::Mean:
                    init_val = 0.0f;
                    break;
                case ReduceOp::Max:
                    init_val = -std::numeric_limits<float>::infinity();
                    break;
                case ReduceOp::Min:
                    init_val = std::numeric_limits<float>::infinity();
                    break;
                case ReduceOp::Prod:
                    init_val = 1.0f;
                    break;
                }
                cudaMemcpyAsync(d_out, &init_val, sizeof(float), cudaMemcpyHostToDevice, stream);

                // Launch warp-level reduction (5-10x faster!)
                launch_warp_reduce_full(d_in, d_out, n, op, stream);

                // Handle mean: divide by count
                if (op == ReduceOp::Mean) {
                    auto out_ptr = thrust::device_pointer_cast(d_out);
                    run_with_thrust_policy(stream, [&](auto policy) {
                        thrust::transform(policy, out_ptr, out_ptr + 1, out_ptr,
                                          DivideByFunctor(static_cast<float>(n)));
                    });
                }
                return;
            }

            // SLOW PATH: Use CUB for very large tensors
            // Determine temp storage requirements
            void* d_temp_storage = nullptr;
            size_t temp_storage_bytes = 0;

            switch (op) {
            case ReduceOp::Sum:
                // Two-phase CUB pattern: 1) Query temp storage size, 2) Perform reduction
                cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, n, stream);
                cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream);
                cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, n, stream);
                cudaFreeAsync(d_temp_storage, stream);
                break;
            case ReduceOp::Mean: {
                // Sum then divide by count
                cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, n, stream);
                cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream);
                cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, n, stream);
                cudaFreeAsync(d_temp_storage, stream);
                // Divide result by n
                auto out_ptr = thrust::device_pointer_cast(d_out);
                run_with_thrust_policy(stream, [&](auto policy) {
                    thrust::transform(policy, out_ptr, out_ptr + 1, out_ptr,
                                      DivideByFunctor(static_cast<float>(n)));
                });
                break;
            }
            case ReduceOp::Max:
                cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, n, stream);
                cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream);
                cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, n, stream);
                cudaFreeAsync(d_temp_storage, stream);
                break;
            case ReduceOp::Min:
                cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, d_in, d_out, n, stream);
                cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream);
                cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, d_in, d_out, n, stream);
                cudaFreeAsync(d_temp_storage, stream);
                break;
            case ReduceOp::Prod:
                // CUB doesn't have built-in Prod, use Thrust for this rare operation
                {
                    float result = 0.0f;
                    run_with_thrust_policy(stream, [&](auto policy) {
                        result = thrust::reduce(policy, input_ptr, input_ptr + n, 1.0f, ops::mul_op{});
                    });
                    cudaMemcpyAsync(output, &result, sizeof(float), cudaMemcpyHostToDevice, stream);
                }
                break;
            default: {
                float zero = 0.0f;
                cudaMemcpyAsync(output, &zero, sizeof(float), cudaMemcpyHostToDevice, stream);
            } break;
            }
            return;
        }

        // Single-axis reduction
        if (num_axes == 1) {
            int dim = axes[0];
            size_t outer_size = 1;
            for (int i = 0; i < dim; ++i)
                outer_size *= shape[i];
            size_t reduce_size = shape[dim];
            size_t inner_size = 1;
            for (size_t i = dim + 1; i < rank; ++i)
                inner_size *= shape[i];

            const float* input_f = static_cast<const float*>(input);
            float* output_f = static_cast<float*>(output);

            // FAST PATH: Use warp-level reduction
            size_t output_size = outer_size * inner_size;

            if (inner_size == 1) {
                // Contiguous segments - use vectorized warp reduction
                if (should_use_warp_reduce(n, outer_size)) {
                    launch_warp_segmented_reduce(input_f, output_f, outer_size, reduce_size, op, stream);

                    // Handle mean: divide by reduce_size
                    if (op == ReduceOp::Mean) {
                        auto out_ptr = thrust::device_pointer_cast(output_f);
                        run_with_thrust_policy(stream, [&](auto policy) {
                            thrust::transform(policy, out_ptr, out_ptr + outer_size, out_ptr,
                                              DivideByFunctor(static_cast<float>(reduce_size)));
                        });
                    }
                    return;
                }
            } else {
                // Strided segments - use warp reduction when compute-bound (large reduce_size)
                // CUB has overhead from segmented reduce setup, so warp reduction is often better
                //
                // Memory access pattern: Each thread accesses with stride=inner_size*4 bytes
                // - Small inner_size (≤ 512): Good cache locality (≤ 2KB stride)
                // - Medium inner_size (512-2048): If reduce_size is large (≥ 512), compute-bound!
                // - Large inner_size (> 2048): CUB's optimized segmented reduce is better
                bool good_stride = (inner_size <= 512) ||
                                   (inner_size <= 2048 && reduce_size >= 512);
                bool use_strided_warp = good_stride && should_use_warp_reduce(n, output_size);

                if (use_strided_warp) {
                    launch_warp_strided_reduce(input_f, output_f, outer_size, reduce_size, inner_size, op, stream);

                    // Handle mean: divide by reduce_size
                    if (op == ReduceOp::Mean) {
                        auto out_ptr = thrust::device_pointer_cast(output_f);
                        run_with_thrust_policy(stream, [&](auto policy) {
                            thrust::transform(policy, out_ptr, out_ptr + output_size, out_ptr,
                                              DivideByFunctor(static_cast<float>(reduce_size)));
                        });
                    }
                    return;
                }
            }

            // SLOW PATH: Use CUB/Thrust for very large tensors
            float init_val = 0.0f;

            switch (op) {
            case ReduceOp::Sum:
                init_val = 0.0f;
                launch_segmented_reduce(input_f, output_f, outer_size, reduce_size, inner_size,
                                        init_val, ops::add_op{}, stream);
                break;
            case ReduceOp::Mean:
                init_val = 0.0f;
                launch_segmented_reduce(input_f, output_f, outer_size, reduce_size, inner_size,
                                        init_val, ops::add_op{}, stream);
                {
                    auto out_ptr = thrust::device_pointer_cast(output_f);
                    size_t output_size = outer_size * inner_size;
                    run_with_thrust_policy(stream, [&](auto policy) {
                        thrust::transform(policy, out_ptr, out_ptr + output_size, out_ptr,
                                          DivideByFunctor(static_cast<float>(reduce_size)));
                    });
                }
                break;
            case ReduceOp::Max:
                init_val = -std::numeric_limits<float>::infinity();
                launch_segmented_reduce(input_f, output_f, outer_size, reduce_size, inner_size,
                                        init_val, ops::maximum_op{}, stream);
                break;
            case ReduceOp::Min:
                init_val = std::numeric_limits<float>::infinity();
                launch_segmented_reduce(input_f, output_f, outer_size, reduce_size, inner_size,
                                        init_val, ops::minimum_op{}, stream);
                break;
            case ReduceOp::Prod:
                init_val = 1.0f;
                launch_segmented_reduce(input_f, output_f, outer_size, reduce_size, inner_size,
                                        init_val, ops::mul_op{}, stream);
                break;
            default:
                break;
            }
            return;
        }

        // Multi-axis reduction
        const float* input_f = static_cast<const float*>(input);
        float* output_f = static_cast<float*>(output);

        // Check if we can optimize as contiguous reduction
        // This happens when all reduced axes are contiguous (e.g., {0,1} or {1,2})
        bool axes_contiguous = true;
        if (num_axes > 1) {
            std::vector<int> sorted_axes(axes, axes + num_axes);
            std::sort(sorted_axes.begin(), sorted_axes.end());
            for (size_t i = 1; i < num_axes; ++i) {
                if (sorted_axes[i] != sorted_axes[i - 1] + 1) {
                    axes_contiguous = false;
                    break;
                }
            }
        }

        // FAST PATH: Contiguous multi-axis reduction
        // Example: sum({0, 1}) on [256, 256, 64] reduces to [64]
        // This is much faster than the generic multi-axis kernel!
        if (axes_contiguous && num_axes > 0) {
            // Find first and last reduced axis
            int first_axis = axes[0];
            int last_axis = axes[0];
            for (size_t i = 1; i < num_axes; ++i) {
                first_axis = std::min(first_axis, axes[i]);
                last_axis = std::max(last_axis, axes[i]);
            }

            // Check if axes span a contiguous range
            if (last_axis - first_axis + 1 == static_cast<int>(num_axes)) {
                // Compute output size and reduce count
                size_t outer_size = 1;
                for (int i = 0; i < first_axis; ++i) {
                    outer_size *= shape[i];
                }

                size_t reduce_count = 1;
                for (int i = first_axis; i <= last_axis; ++i) {
                    reduce_count *= shape[i];
                }

                size_t inner_size = 1;
                for (size_t i = last_axis + 1; i < rank; ++i) {
                    inner_size *= shape[i];
                }

                size_t output_size = outer_size * inner_size;

                // If inner_size == 1 and outer_size == 1, it's a full reduction (contiguous segment)
                // Use warp reduction for small-medium tensors
                if (inner_size == 1 && outer_size == 1) {
                    // Full tensor reduction - already handled above
                    // This shouldn't happen, but just in case...
                    if (should_use_warp_reduce(n, 1)) {
                        float init_val = 0.0f;
                        switch (op) {
                        case ReduceOp::Sum:
                        case ReduceOp::Mean:
                            init_val = 0.0f;
                            break;
                        case ReduceOp::Max:
                            init_val = -std::numeric_limits<float>::infinity();
                            break;
                        case ReduceOp::Min:
                            init_val = std::numeric_limits<float>::infinity();
                            break;
                        case ReduceOp::Prod:
                            init_val = 1.0f;
                            break;
                        }
                        cudaMemcpyAsync(output_f, &init_val, sizeof(float), cudaMemcpyHostToDevice, stream);
                        launch_warp_reduce_full(input_f, output_f, n, op, stream);

                        if (op == ReduceOp::Mean) {
                            auto out_ptr = thrust::device_pointer_cast(output_f);
                            run_with_thrust_policy(stream, [&](auto policy) {
                                thrust::transform(policy, out_ptr, out_ptr + 1, out_ptr,
                                                  DivideByFunctor(static_cast<float>(reduce_count)));
                            });
                        }
                        return;
                    }
                }

                // If inner_size == 1, we can treat this as segmented reduction
                if (inner_size == 1 && should_use_warp_reduce(n, outer_size)) {
                    launch_warp_segmented_reduce(input_f, output_f, outer_size, reduce_count, op, stream);

                    if (op == ReduceOp::Mean) {
                        auto out_ptr = thrust::device_pointer_cast(output_f);
                        run_with_thrust_policy(stream, [&](auto policy) {
                            thrust::transform(policy, out_ptr, out_ptr + outer_size, out_ptr,
                                              DivideByFunctor(static_cast<float>(reduce_count)));
                        });
                    }
                    return;
                }

                // Otherwise, use the new multi-axis warp reduction
                // This handles cases like: [256, 256, 64] with axes {0,1} → [64]
                // Each of the 64 output elements sums 256*256=65536 input elements
                //
                // SPECIAL HEURISTIC for multi-axis contiguous reductions:
                // Unlike strided single-axis reductions, multi-axis contiguous reductions
                // have GOOD memory access patterns (sequential chunks). Our warp kernel
                // is competitive even for larger tensors!
                //
                // Conditions:
                // - output_size < 100K (reasonable number of output elements)
                // - reduce_count < 10M (each output reduces < 10M elements - vectorized segment reduce handles this well!)
                // - Total tensor size < 100M (to avoid extreme cases)
                bool use_warp_multi_axis = output_size < 100000 &&
                                           reduce_count < 10000000 &&
                                           n < 100000000;

                if (use_warp_multi_axis) {
                    launch_warp_multi_axis_reduce(input_f, output_f, output_size, reduce_count, op, stream);

                    if (op == ReduceOp::Mean) {
                        auto out_ptr = thrust::device_pointer_cast(output_f);
                        run_with_thrust_policy(stream, [&](auto policy) {
                            thrust::transform(policy, out_ptr, out_ptr + output_size, out_ptr,
                                              DivideByFunctor(static_cast<float>(reduce_count)));
                        });
                    }
                    return;
                }
            }
        }

        // SLOW PATH: Generic multi-axis reduction
        thrust::device_vector<bool> d_is_reduced(rank, false);
        for (size_t i = 0; i < num_axes; ++i) {
            d_is_reduced[axes[i]] = true;
        }

        size_t output_elements = 1;
        for (size_t i = 0; i < rank; ++i) {
            if (!d_is_reduced[i] || keepdim) {
                output_elements *= (d_is_reduced[i] ? 1 : shape[i]);
            }
        }

        thrust::device_vector<size_t> d_input_shape(shape, shape + rank);

        float init_val = 0.0f;
        switch (op) {
        case ReduceOp::Sum:
        case ReduceOp::Mean:
            init_val = 0.0f;
            break;
        case ReduceOp::Max:
            init_val = -std::numeric_limits<float>::infinity();
            break;
        case ReduceOp::Min:
            init_val = std::numeric_limits<float>::infinity();
            break;
        case ReduceOp::Prod:
            init_val = 1.0f;
            break;
        }

        launch_multi_axis_reduce(
            input_f,
            output_f,
            thrust::raw_pointer_cast(d_input_shape.data()),
            thrust::raw_pointer_cast(d_is_reduced.data()),
            rank, output_elements, init_val, op, stream);

        if (op == ReduceOp::Mean) {
            size_t reduce_count = 1;
            for (size_t i = 0; i < num_axes; ++i) {
                reduce_count *= shape[axes[i]];
            }
            float scale = 1.0f / reduce_count;
            run_with_thrust_policy(stream, [&](auto policy) {
                thrust::transform(policy, output_ptr, output_ptr + output_elements,
                                  thrust::make_constant_iterator(scale), output_ptr,
                                  ops::mul_op{});
            });
        }
    }

    // ============= TERNARY OPERATIONS =============

    // ============= LOAD OPERATIONS =============

    void launch_load_op(void* output, const size_t* shape, size_t rank, LoadOp op,
                        const void* args, DataType dtype, cudaStream_t stream) {
        if (dtype != DataType::Float32)
            return;

        size_t n = 1;
        for (size_t i = 0; i < rank; ++i)
            n *= shape[i];
        if (n == 0)
            return;

        if (op == LoadOp::Const && args) {
            float value = *static_cast<const float*>(args);
            auto output_ptr = thrust::device_pointer_cast(static_cast<float*>(output));
            run_with_thrust_policy(stream, [&](auto policy) {
                thrust::fill(policy, output_ptr, output_ptr + n, value);
            });
        }
    }

    // ============= OPTIMIZED CUMULATIVE SUM =============

    template <typename T>
    __global__ void cumsum_noncontiguous_kernel(T* data, size_t outer_size, size_t dim_size, size_t inner_size) {
        size_t scan_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (scan_idx >= outer_size * inner_size)
            return;

        size_t outer_idx = scan_idx / inner_size;
        size_t inner_idx = scan_idx % inner_size;
        size_t base = outer_idx * dim_size * inner_size + inner_idx;

        T accumulator = data[base];
        for (size_t d = 1; d < dim_size; ++d) {
            size_t idx = base + d * inner_size;
            accumulator = accumulator + data[idx];
            data[idx] = accumulator;
        }
    }

    template <typename T>
    void launch_cumsum_optimized(T* data, size_t outer_size, size_t dim_size,
                                 size_t inner_size, cudaStream_t stream) {
        if (outer_size == 0 || dim_size == 0 || inner_size == 0)
            return;

        if (inner_size == 1 && dim_size > 1) {
            // Contiguous segments - use Thrust's optimized segmented scan
            auto data_ptr = thrust::device_pointer_cast(data);
            size_t total_elements = outer_size * dim_size;
            thrust::device_vector<int> keys(total_elements);

            if (stream) {
                thrust::transform(thrust::cuda::par.on(stream),
                                  thrust::counting_iterator<size_t>(0),
                                  thrust::counting_iterator<size_t>(total_elements),
                                  keys.begin(),
                                  [=] __device__(size_t idx) -> int {
                                      return static_cast<int>(idx / dim_size);
                                  });

                thrust::inclusive_scan_by_key(thrust::cuda::par.on(stream),
                                              keys.begin(), keys.end(),
                                              data_ptr, data_ptr);
            } else {
                thrust::transform(thrust::cuda::par,
                                  thrust::counting_iterator<size_t>(0),
                                  thrust::counting_iterator<size_t>(total_elements),
                                  keys.begin(),
                                  [=] __device__(size_t idx) -> int {
                                      return static_cast<int>(idx / dim_size);
                                  });

                thrust::inclusive_scan_by_key(thrust::cuda::par,
                                              keys.begin(), keys.end(),
                                              data_ptr, data_ptr);
            }
        } else {
            // Non-contiguous - use custom kernel
            size_t total_scans = outer_size * inner_size;
            int blocks = (total_scans + 255) / 256;
            cumsum_noncontiguous_kernel<<<blocks, 256, 0, stream>>>(data, outer_size, dim_size, inner_size);
        }
    }

    void launch_cumsum(void* data, const size_t* shape, size_t rank,
                       int dim, DataType dtype, cudaStream_t stream) {
        if (dtype != DataType::Float32 && dtype != DataType::Int32)
            return;

        size_t total = 1;
        for (size_t i = 0; i < rank; ++i)
            total *= shape[i];
        if (total == 0)
            return;

        if (rank == 1) {
            if (dtype == DataType::Float32) {
                auto data_ptr = thrust::device_pointer_cast(static_cast<float*>(data));
                if (stream) {
                    thrust::inclusive_scan(thrust::cuda::par.on(stream), data_ptr, data_ptr + total, data_ptr);
                } else {
                    thrust::inclusive_scan(thrust::cuda::par, data_ptr, data_ptr + total, data_ptr);
                }
            } else if (dtype == DataType::Int32) {
                auto data_ptr = thrust::device_pointer_cast(static_cast<int*>(data));
                if (stream) {
                    thrust::inclusive_scan(thrust::cuda::par.on(stream), data_ptr, data_ptr + total, data_ptr);
                } else {
                    thrust::inclusive_scan(thrust::cuda::par, data_ptr, data_ptr + total, data_ptr);
                }
            }
            return;
        }

        size_t outer_size = 1;
        for (int i = 0; i < dim; ++i)
            outer_size *= shape[i];
        size_t dim_size = shape[dim];
        size_t inner_size = 1;
        for (size_t i = dim + 1; i < rank; ++i)
            inner_size *= shape[i];

        if (dtype == DataType::Float32) {
            launch_cumsum_optimized<float>(static_cast<float*>(data), outer_size, dim_size, inner_size, stream);
        } else if (dtype == DataType::Int32) {
            launch_cumsum_optimized<int>(static_cast<int*>(data), outer_size, dim_size, inner_size, stream);
        }
    }

    // ============= OPTIMIZED PAIRWISE DISTANCE =============

    template <int BLOCK_SIZE = 16>
    __global__ void cdist_l2_optimized_kernel(
        const float* __restrict__ a,
        const float* __restrict__ b,
        float* __restrict__ out,
        size_t N, size_t M, size_t D) {
        __shared__ float tile_a[BLOCK_SIZE][BLOCK_SIZE + 1];
        __shared__ float tile_b[BLOCK_SIZE][BLOCK_SIZE + 1];

        size_t row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
        size_t col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

        float sum = 0.0f;

        size_t num_tiles = (D + BLOCK_SIZE - 1) / BLOCK_SIZE;
        for (size_t tile = 0; tile < num_tiles; ++tile) {
            size_t d_idx = tile * BLOCK_SIZE + threadIdx.x;
            if (row < N && d_idx < D) {
                tile_a[threadIdx.y][threadIdx.x] = a[row * D + d_idx];
            } else {
                tile_a[threadIdx.y][threadIdx.x] = 0.0f;
            }

            d_idx = tile * BLOCK_SIZE + threadIdx.y;
            if (col < M && d_idx < D) {
                tile_b[threadIdx.y][threadIdx.x] = b[col * D + d_idx];
            } else {
                tile_b[threadIdx.y][threadIdx.x] = 0.0f;
            }

            __syncthreads();

            size_t d_start = tile * BLOCK_SIZE;
            size_t d_end = (d_start + BLOCK_SIZE < D) ? (d_start + BLOCK_SIZE) : D;
            size_t tile_size = d_end - d_start;

#pragma unroll
            for (size_t k = 0; k < BLOCK_SIZE; ++k) {
                if (k < tile_size) {
                    float diff = tile_a[threadIdx.y][k] - tile_b[k][threadIdx.x];
                    sum += diff * diff;
                }
            }

            __syncthreads();
        }

        if (row < N && col < M) {
            out[row * M + col] = sqrtf(sum);
        }
    }

    __global__ void cdist_l2_vectorized_kernel(
        const float* __restrict__ a,
        const float* __restrict__ b,
        float* __restrict__ out,
        size_t N, size_t M, size_t D) {
        size_t i = blockIdx.y * blockDim.y + threadIdx.y;
        size_t j = blockIdx.x * blockDim.x + threadIdx.x;

        if (i >= N || j >= M)
            return;

        float sum = 0.0f;
        const float4* a_vec = reinterpret_cast<const float4*>(a + i * D);
        const float4* b_vec = reinterpret_cast<const float4*>(b + j * D);

        size_t vec_d = D / 4;

        for (size_t d = 0; d < vec_d; ++d) {
            float4 va = a_vec[d];
            float4 vb = b_vec[d];

            float diff_x = va.x - vb.x;
            float diff_y = va.y - vb.y;
            float diff_z = va.z - vb.z;
            float diff_w = va.w - vb.w;

            sum += diff_x * diff_x + diff_y * diff_y + diff_z * diff_z + diff_w * diff_w;
        }

        for (size_t d = vec_d * 4; d < D; ++d) {
            float diff = a[i * D + d] - b[j * D + d];
            sum += diff * diff;
        }

        out[i * M + j] = sqrtf(sum);
    }

    template <int BLOCK_SIZE = 16>
    __global__ void cdist_l1_optimized_kernel(
        const float* __restrict__ a,
        const float* __restrict__ b,
        float* __restrict__ out,
        size_t N, size_t M, size_t D) {
        __shared__ float tile_a[BLOCK_SIZE][BLOCK_SIZE + 1];
        __shared__ float tile_b[BLOCK_SIZE][BLOCK_SIZE + 1];

        size_t row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
        size_t col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

        float sum = 0.0f;

        size_t num_tiles = (D + BLOCK_SIZE - 1) / BLOCK_SIZE;
        for (size_t tile = 0; tile < num_tiles; ++tile) {
            size_t d_idx = tile * BLOCK_SIZE + threadIdx.x;
            if (row < N && d_idx < D) {
                tile_a[threadIdx.y][threadIdx.x] = a[row * D + d_idx];
            } else {
                tile_a[threadIdx.y][threadIdx.x] = 0.0f;
            }

            d_idx = tile * BLOCK_SIZE + threadIdx.y;
            if (col < M && d_idx < D) {
                tile_b[threadIdx.y][threadIdx.x] = b[col * D + d_idx];
            } else {
                tile_b[threadIdx.y][threadIdx.x] = 0.0f;
            }

            __syncthreads();

            size_t d_start = tile * BLOCK_SIZE;
            size_t d_end = (d_start + BLOCK_SIZE < D) ? (d_start + BLOCK_SIZE) : D;
            size_t tile_size = d_end - d_start;

#pragma unroll
            for (size_t k = 0; k < BLOCK_SIZE; ++k) {
                if (k < tile_size) {
                    sum += fabsf(tile_a[threadIdx.y][k] - tile_b[k][threadIdx.x]);
                }
            }

            __syncthreads();
        }

        if (row < N && col < M) {
            out[row * M + col] = sum;
        }
    }

    __global__ void cdist_lp_kernel(
        const float* __restrict__ a,
        const float* __restrict__ b,
        float* __restrict__ out,
        size_t N, size_t M, size_t D, float p) {
        size_t i = blockIdx.y * blockDim.y + threadIdx.y;
        size_t j = blockIdx.x * blockDim.x + threadIdx.x;

        if (i >= N || j >= M)
            return;

        float dist = 0.0f;
        for (size_t d = 0; d < D; ++d) {
            float diff = fabsf(a[i * D + d] - b[j * D + d]);
            dist += powf(diff, p);
        }
        out[i * M + j] = powf(dist, 1.0f / p);
    }

    void launch_cdist(const float* a, const float* b, float* out,
                      size_t N, size_t M, size_t D, float p, cudaStream_t stream) {
        if (N == 0 || M == 0)
            return;

        constexpr int BLOCK_SIZE = 16;

        if (p == 2.0f) {
            if (D >= 128 && D % 4 == 0) {
                dim3 block(BLOCK_SIZE, BLOCK_SIZE);
                dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
                cdist_l2_vectorized_kernel<<<grid, block, 0, stream>>>(a, b, out, N, M, D);
            } else {
                dim3 block(BLOCK_SIZE, BLOCK_SIZE);
                dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
                cdist_l2_optimized_kernel<BLOCK_SIZE><<<grid, block, 0, stream>>>(a, b, out, N, M, D);
            }
        } else if (p == 1.0f) {
            dim3 block(BLOCK_SIZE, BLOCK_SIZE);
            dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
            cdist_l1_optimized_kernel<BLOCK_SIZE><<<grid, block, 0, stream>>>(a, b, out, N, M, D);
        } else {
            dim3 block(16, 16);
            dim3 grid((M + 15) / 16, (N + 15) / 16);
            cdist_lp_kernel<<<grid, block, 0, stream>>>(a, b, out, N, M, D, p);
        }
    }

    // ============= SORTING =============

    void launch_sort_1d(float* values, int64_t* indices, size_t n, bool descending, cudaStream_t stream) {
        if (n == 0)
            return;

        auto values_ptr = thrust::device_pointer_cast(values);
        auto indices_ptr = thrust::device_pointer_cast(indices);

        run_with_thrust_policy(stream, [&](auto policy) {
            thrust::sequence(policy, indices_ptr, indices_ptr + n);
        });

        if (stream) {
            if (descending) {
                thrust::sort_by_key(thrust::cuda::par.on(stream), values_ptr, values_ptr + n,
                                    indices_ptr, thrust::greater<float>());
            } else {
                thrust::sort_by_key(thrust::cuda::par.on(stream), values_ptr, values_ptr + n,
                                    indices_ptr, thrust::less<float>());
            }
        } else {
            if (descending) {
                thrust::sort_by_key(thrust::cuda::par, values_ptr, values_ptr + n,
                                    indices_ptr, thrust::greater<float>());
            } else {
                thrust::sort_by_key(thrust::cuda::par, values_ptr, values_ptr + n,
                                    indices_ptr, thrust::less<float>());
            }
        }
    }

    __global__ void extract_slice_kernel(const float* input, float* output,
                                         size_t outer_size, size_t dim_size, size_t inner_size,
                                         size_t outer_idx, size_t inner_idx) {
        size_t d = blockIdx.x * blockDim.x + threadIdx.x;
        if (d < dim_size) {
            size_t src_idx = outer_idx * dim_size * inner_size + d * inner_size + inner_idx;
            output[d] = input[src_idx];
        }
    }

    __global__ void write_slice_kernel(float* output, int64_t* output_idx,
                                       const float* sorted_vals, const int64_t* sorted_idx,
                                       size_t outer_size, size_t dim_size, size_t inner_size,
                                       size_t outer_idx, size_t inner_idx) {
        size_t d = blockIdx.x * blockDim.x + threadIdx.x;
        if (d < dim_size) {
            size_t dst_idx = outer_idx * dim_size * inner_size + d * inner_size + inner_idx;
            output[dst_idx] = sorted_vals[d];
            output_idx[dst_idx] = sorted_idx[d];
        }
    }

    void launch_sort_2d(float* values, int64_t* indices,
                        size_t outer_size, size_t dim_size, size_t inner_size,
                        int dim, bool descending, cudaStream_t stream) {
        if (dim_size == 0 || outer_size == 0 || inner_size == 0)
            return;

        thrust::device_vector<float> temp_vals(dim_size);
        thrust::device_vector<int64_t> temp_idx(dim_size);
        int blocks = (dim_size + 255) / 256;

        for (size_t outer = 0; outer < outer_size; ++outer) {
            for (size_t inner = 0; inner < inner_size; ++inner) {
                extract_slice_kernel<<<blocks, 256, 0, stream>>>(
                    values, thrust::raw_pointer_cast(temp_vals.data()),
                    outer_size, dim_size, inner_size, outer, inner);

                thrust::sequence(thrust::cuda::par.on(stream), temp_idx.begin(), temp_idx.end(), 0LL);

                if (descending) {
                    thrust::sort_by_key(thrust::cuda::par.on(stream),
                                        temp_vals.begin(), temp_vals.end(), temp_idx.begin(),
                                        thrust::greater<float>());
                } else {
                    thrust::sort_by_key(thrust::cuda::par.on(stream),
                                        temp_vals.begin(), temp_vals.end(), temp_idx.begin(),
                                        thrust::less<float>());
                }

                write_slice_kernel<<<blocks, 256, 0, stream>>>(
                    values, indices,
                    thrust::raw_pointer_cast(temp_vals.data()),
                    thrust::raw_pointer_cast(temp_idx.data()),
                    outer_size, dim_size, inner_size, outer, inner);
            }
        }

        if (stream)
            cudaStreamSynchronize(stream);
    }

    // ============= CONCATENATION OPERATIONS =============

    // OPTIMIZED: Vectorized cat kernel with float4 (4× memory bandwidth!)
    // Special case for RGB→RGBA conversion (most common case)
    __global__ void cat_rgb_to_rgba_kernel(
        float* output,
        const float* rgb,
        size_t num_pixels,
        float alpha_value) {
        size_t pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (pixel_idx >= num_pixels)
            return;

        // Each thread processes one pixel: RGB → RGBA
        const size_t rgb_offset = pixel_idx * 3;
        const size_t rgba_offset = pixel_idx * 4;

        float3 rgb_vals = make_float3(
            rgb[rgb_offset + 0],
            rgb[rgb_offset + 1],
            rgb[rgb_offset + 2]);

        // Check alignment for float4 output
        bool out_aligned = (reinterpret_cast<uintptr_t>(&output[rgba_offset]) % 16) == 0;

        if (out_aligned) {
            // Vectorized write: Store all 4 channels in one transaction
            float4 rgba = make_float4(rgb_vals.x, rgb_vals.y, rgb_vals.z, alpha_value);
            reinterpret_cast<float4*>(&output[rgba_offset])[0] = rgba;
        } else {
            // Scalar fallback
            output[rgba_offset + 0] = rgb_vals.x;
            output[rgba_offset + 1] = rgb_vals.y;
            output[rgba_offset + 2] = rgb_vals.z;
            output[rgba_offset + 3] = alpha_value;
        }
    }

    // Generic vectorized cat kernel
    template <typename T>
    __global__ void cat_last_dim_kernel_vectorized(
        T* output,
        const T** input_ptrs,
        const size_t* input_sizes,
        size_t num_tensors,
        size_t num_rows,
        size_t row_size) {
        size_t row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= num_rows)
            return;

        size_t result_offset = 0;
        for (size_t t = 0; t < num_tensors; ++t) {
            size_t tensor_dim_size = input_sizes[t];

            const T* src = input_ptrs[t] + row * tensor_dim_size;
            T* dst = output + row * row_size + result_offset;

            // FAST PATH: Vectorized copy with float4 (4× bandwidth!)
            if constexpr (std::is_same_v<T, float>) {
                bool src_aligned = (reinterpret_cast<uintptr_t>(src) % 16) == 0;
                bool dst_aligned = (reinterpret_cast<uintptr_t>(dst) % 16) == 0;

                if (src_aligned && dst_aligned && tensor_dim_size >= 4) {
                    size_t vec_size = tensor_dim_size / 4;

                    // Vectorized copy: 4 floats per transaction
                    for (size_t v = 0; v < vec_size; ++v) {
                        reinterpret_cast<float4*>(dst)[v] = reinterpret_cast<const float4*>(src)[v];
                    }

                    // Handle remainder
                    for (size_t i = vec_size * 4; i < tensor_dim_size; ++i) {
                        dst[i] = src[i];
                    }
                } else {
                    // Scalar fallback for unaligned data
                    for (size_t i = 0; i < tensor_dim_size; ++i) {
                        dst[i] = src[i];
                    }
                }
            } else {
                // Scalar path for non-float types
                for (size_t i = 0; i < tensor_dim_size; ++i) {
                    dst[i] = src[i];
                }
            }

            result_offset += tensor_dim_size;
        }
    }

    // DEPRECATED: Old scalar kernel (kept for compatibility)
    template <typename T>
    __global__ void cat_last_dim_kernel(
        T* output,
        const T** input_ptrs,
        const size_t* input_sizes,
        size_t num_tensors,
        size_t num_rows,
        size_t row_size) {
        size_t row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= num_rows)
            return;

        size_t result_offset = 0;
        for (size_t t = 0; t < num_tensors; ++t) {
            size_t tensor_dim_size = input_sizes[t];

            const T* src = input_ptrs[t] + row * tensor_dim_size;
            T* dst = output + row * row_size + result_offset;

            for (size_t i = 0; i < tensor_dim_size; ++i) {
                dst[i] = src[i];
            }

            result_offset += tensor_dim_size;
        }
    }

    void launch_cat_last_dim(
        void* output,
        const std::vector<Tensor>& tensors,
        size_t num_rows,
        size_t row_size,
        size_t element_size,
        cudaStream_t stream) {
        size_t num_tensors = tensors.size();

        // FAST PATH: RGB→RGBA conversion (adding alpha channel)
        // This is the most common case in image processing pipelines
        if (num_tensors == 2 &&
            tensors[0].shape()[tensors[0].shape().rank() - 1] == 3 &&
            tensors[1].shape()[tensors[1].shape().rank() - 1] == 1 &&
            element_size == sizeof(float)) {
            // Special case: cat([RGB, alpha]) → RGBA
            size_t num_pixels = num_rows;
            int block_size = 256;
            int grid_size = (num_pixels + block_size - 1) / block_size;

            // Assume alpha is constant (most common case)
            float alpha_value = 1.0f; // Default alpha = 1.0 for opaque

            cat_rgb_to_rgba_kernel<<<grid_size, block_size, 0, stream>>>(
                static_cast<float*>(output),
                static_cast<const float*>(tensors[0].raw_ptr()),
                num_pixels,
                alpha_value);
            return;
        }

        // GENERIC PATH: Use memory pool for metadata (NO thrust::device_vector!)
        // Allocate from memory pool (fast, cached, no synchronization)
        const float** d_input_ptrs = static_cast<const float**>(
            CudaMemoryPool::instance().allocate(num_tensors * sizeof(float*), stream));
        size_t* d_input_sizes = static_cast<size_t*>(
            CudaMemoryPool::instance().allocate(num_tensors * sizeof(size_t), stream));

        if (!d_input_ptrs || !d_input_sizes) {
            LOG_ERROR("Failed to allocate cat metadata from memory pool");
            if (d_input_ptrs)
                CudaMemoryPool::instance().deallocate(const_cast<float**>(d_input_ptrs), stream);
            if (d_input_sizes)
                CudaMemoryPool::instance().deallocate(d_input_sizes, stream);
            return;
        }

        // Copy metadata to device
        std::vector<const float*> h_input_ptrs(num_tensors);
        std::vector<size_t> h_input_sizes(num_tensors);

        for (size_t i = 0; i < num_tensors; ++i) {
            h_input_ptrs[i] = static_cast<const float*>(tensors[i].raw_ptr());
            h_input_sizes[i] = tensors[i].shape()[tensors[i].shape().rank() - 1];
        }

        cudaMemcpyAsync(const_cast<float**>(d_input_ptrs), h_input_ptrs.data(),
                        num_tensors * sizeof(float*), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_input_sizes, h_input_sizes.data(),
                        num_tensors * sizeof(size_t), cudaMemcpyHostToDevice, stream);

        int block_size = 256;
        int grid_size = (num_rows + block_size - 1) / block_size;

        // Use vectorized kernel (4× faster for float data!)
        cat_last_dim_kernel_vectorized<<<grid_size, block_size, 0, stream>>>(
            static_cast<float*>(output),
            d_input_ptrs,
            d_input_sizes,
            num_tensors,
            num_rows,
            row_size);

        // Return metadata arrays to memory pool (instant, cached for reuse)
        CudaMemoryPool::instance().deallocate(const_cast<float**>(d_input_ptrs), stream);
        CudaMemoryPool::instance().deallocate(d_input_sizes, stream);
    }

    template <typename T>
    __global__ void cat_middle_dim_kernel(
        T* output,
        const T** input_ptrs,
        const size_t* input_sizes,
        size_t num_tensors,
        size_t outer_size,
        size_t inner_size,
        size_t total_dim_size) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t total = outer_size * total_dim_size * inner_size;

        if (idx >= total)
            return;

        size_t outer_idx = idx / (total_dim_size * inner_size);
        size_t remainder = idx % (total_dim_size * inner_size);
        size_t dim_idx = remainder / inner_size;
        size_t inner_idx = remainder % inner_size;

        size_t accumulated = 0;
        for (size_t t = 0; t < num_tensors; ++t) {
            if (dim_idx < accumulated + input_sizes[t]) {
                size_t tensor_dim_idx = dim_idx - accumulated;
                size_t src_idx = outer_idx * input_sizes[t] * inner_size +
                                 tensor_dim_idx * inner_size + inner_idx;
                output[idx] = input_ptrs[t][src_idx];
                return;
            }
            accumulated += input_sizes[t];
        }
    }

    void launch_cat_middle_dim(
        void* output,
        const std::vector<Tensor>& tensors,
        size_t outer_size,
        size_t inner_size,
        int resolved_dim,
        size_t element_size,
        cudaStream_t stream) {
        size_t num_tensors = tensors.size();
        size_t total_dim_size = 0;
        for (const auto& t : tensors) {
            total_dim_size += t.shape()[resolved_dim];
        }

        size_t total_elements = outer_size * total_dim_size * inner_size;

        // OPTIMIZED: Use memory pool instead of thrust::device_vector
        const float** d_input_ptrs = static_cast<const float**>(
            CudaMemoryPool::instance().allocate(num_tensors * sizeof(float*), stream));
        size_t* d_input_sizes = static_cast<size_t*>(
            CudaMemoryPool::instance().allocate(num_tensors * sizeof(size_t), stream));

        if (!d_input_ptrs || !d_input_sizes) {
            LOG_ERROR("Failed to allocate cat_middle_dim metadata from memory pool");
            if (d_input_ptrs)
                CudaMemoryPool::instance().deallocate(const_cast<float**>(d_input_ptrs), stream);
            if (d_input_sizes)
                CudaMemoryPool::instance().deallocate(d_input_sizes, stream);
            return;
        }

        // Copy metadata to device
        std::vector<const float*> h_input_ptrs(num_tensors);
        std::vector<size_t> h_input_sizes(num_tensors);

        for (size_t i = 0; i < num_tensors; ++i) {
            h_input_ptrs[i] = static_cast<const float*>(tensors[i].raw_ptr());
            h_input_sizes[i] = tensors[i].shape()[resolved_dim];
        }

        cudaMemcpyAsync(const_cast<float**>(d_input_ptrs), h_input_ptrs.data(),
                        num_tensors * sizeof(float*), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_input_sizes, h_input_sizes.data(),
                        num_tensors * sizeof(size_t), cudaMemcpyHostToDevice, stream);

        int block_size = 256;
        int grid_size = (total_elements + block_size - 1) / block_size;

        cat_middle_dim_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<float*>(output),
            d_input_ptrs,
            d_input_sizes,
            num_tensors,
            outer_size,
            inner_size,
            total_dim_size);

        // Return metadata arrays to memory pool
        CudaMemoryPool::instance().deallocate(const_cast<float**>(d_input_ptrs), stream);
        CudaMemoryPool::instance().deallocate(d_input_sizes, stream);
    }

    // ============= EXPLICIT TEMPLATE INSTANTIATIONS =============

    // Type conversions
    template void launch_convert_type<float, uint8_t>(const float*, uint8_t*, size_t, cudaStream_t);
    template void launch_convert_type<uint8_t, float>(const uint8_t*, float*, size_t, cudaStream_t);
    template void launch_convert_type<int, uint8_t>(const int*, uint8_t*, size_t, cudaStream_t);
    template void launch_convert_type<uint8_t, int>(const uint8_t*, int*, size_t, cudaStream_t);
    template void launch_convert_type<int64_t, float>(const int64_t*, float*, size_t, cudaStream_t);
    template void launch_convert_type<float, int64_t>(const float*, int64_t*, size_t, cudaStream_t);
    template void launch_convert_type<int, int64_t>(const int*, int64_t*, size_t, cudaStream_t);
    template void launch_convert_type<int64_t, int>(const int64_t*, int*, size_t, cudaStream_t);
    template void launch_convert_type<uint8_t, int64_t>(const uint8_t*, int64_t*, size_t, cudaStream_t);
    template void launch_convert_type<int64_t, uint8_t>(const int64_t*, uint8_t*, size_t, cudaStream_t);
    template void launch_convert_type<int, float>(const int*, float*, size_t, cudaStream_t);
    template void launch_convert_type<float, int>(const float*, int*, size_t, cudaStream_t);

    // ============= EXPLICIT INSTANTIATIONS FOR C++ FILES =============
    // C++ files (not CUDA) can't see tensor_generic_ops.cuh (which is #ifdef __CUDACC__),
    // so we need explicit instantiations for functors used by C++ expression templates.

    // Basic unary operations (comprehensive list)
    template void launch_unary_op_generic<float, float, ops::log_op>(
        const float*, float*, size_t, ops::log_op, cudaStream_t);
    template void launch_unary_op_generic<int, int, ops::log_op>(
        const int*, int*, size_t, ops::log_op, cudaStream_t);
    template void launch_unary_op_generic<float, float, ops::exp_op>(
        const float*, float*, size_t, ops::exp_op, cudaStream_t);
    template void launch_unary_op_generic<int, int, ops::exp_op>(
        const int*, int*, size_t, ops::exp_op, cudaStream_t);
    template void launch_unary_op_generic<float, float, ops::abs_op>(
        const float*, float*, size_t, ops::abs_op, cudaStream_t);
    template void launch_unary_op_generic<int, int, ops::abs_op>(
        const int*, int*, size_t, ops::abs_op, cudaStream_t);
    template void launch_unary_op_generic<float, float, ops::sqrt_op>(
        const float*, float*, size_t, ops::sqrt_op, cudaStream_t);
    template void launch_unary_op_generic<int, int, ops::sqrt_op>(
        const int*, int*, size_t, ops::sqrt_op, cudaStream_t);
    template void launch_unary_op_generic<float, float, ops::square_op>(
        const float*, float*, size_t, ops::square_op, cudaStream_t);
    template void launch_unary_op_generic<int, int, ops::square_op>(
        const int*, int*, size_t, ops::square_op, cudaStream_t);
    template void launch_unary_op_generic<float, float, ops::relu_op>(
        const float*, float*, size_t, ops::relu_op, cudaStream_t);
    template void launch_unary_op_generic<int, int, ops::relu_op>(
        const int*, int*, size_t, ops::relu_op, cudaStream_t);
    template void launch_unary_op_generic<float, float, ops::sigmoid_op>(
        const float*, float*, size_t, ops::sigmoid_op, cudaStream_t);
    template void launch_unary_op_generic<int, int, ops::sigmoid_op>(
        const int*, int*, size_t, ops::sigmoid_op, cudaStream_t);
    template void launch_unary_op_generic<float, float, ops::neg_op>(
        const float*, float*, size_t, ops::neg_op, cudaStream_t);
    template void launch_unary_op_generic<int, int, ops::neg_op>(
        const int*, int*, size_t, ops::neg_op, cudaStream_t);
    template void launch_unary_op_generic<float, float, ops::floor_op>(
        const float*, float*, size_t, ops::floor_op, cudaStream_t);
    template void launch_unary_op_generic<int, int, ops::floor_op>(
        const int*, int*, size_t, ops::floor_op, cudaStream_t);
    template void launch_unary_op_generic<float, float, ops::ceil_op>(
        const float*, float*, size_t, ops::ceil_op, cudaStream_t);
    template void launch_unary_op_generic<int, int, ops::ceil_op>(
        const int*, int*, size_t, ops::ceil_op, cudaStream_t);
    template void launch_unary_op_generic<float, float, ops::round_op>(
        const float*, float*, size_t, ops::round_op, cudaStream_t);
    template void launch_unary_op_generic<int, int, ops::round_op>(
        const int*, int*, size_t, ops::round_op, cudaStream_t);
    template void launch_unary_op_generic<float, float, ops::sin_op>(
        const float*, float*, size_t, ops::sin_op, cudaStream_t);
    template void launch_unary_op_generic<int, int, ops::sin_op>(
        const int*, int*, size_t, ops::sin_op, cudaStream_t);
    template void launch_unary_op_generic<float, float, ops::cos_op>(
        const float*, float*, size_t, ops::cos_op, cudaStream_t);
    template void launch_unary_op_generic<int, int, ops::cos_op>(
        const int*, int*, size_t, ops::cos_op, cudaStream_t);
    template void launch_unary_op_generic<float, float, ops::tan_op>(
        const float*, float*, size_t, ops::tan_op, cudaStream_t);
    template void launch_unary_op_generic<int, int, ops::tan_op>(
        const int*, int*, size_t, ops::tan_op, cudaStream_t);
    template void launch_unary_op_generic<float, float, ops::tanh_op>(
        const float*, float*, size_t, ops::tanh_op, cudaStream_t);
    template void launch_unary_op_generic<int, int, ops::tanh_op>(
        const int*, int*, size_t, ops::tanh_op, cudaStream_t);
    template void launch_unary_op_generic<float, float, ops::sign_op>(
        const float*, float*, size_t, ops::sign_op, cudaStream_t);
    template void launch_unary_op_generic<int, int, ops::sign_op>(
        const int*, int*, size_t, ops::sign_op, cudaStream_t);
    template void launch_unary_op_generic<float, float, ops::reciprocal_op>(
        const float*, float*, size_t, ops::reciprocal_op, cudaStream_t);
    template void launch_unary_op_generic<int, int, ops::reciprocal_op>(
        const int*, int*, size_t, ops::reciprocal_op, cudaStream_t);
    template void launch_unary_op_generic<float, unsigned char, ops::logical_not_op>(
        const float*, unsigned char*, size_t, ops::logical_not_op, cudaStream_t);
    template void launch_unary_op_generic<int, unsigned char, ops::logical_not_op>(
        const int*, unsigned char*, size_t, ops::logical_not_op, cudaStream_t);
    template void launch_unary_op_generic<unsigned char, unsigned char, ops::logical_not_op>(
        const unsigned char*, unsigned char*, size_t, ops::logical_not_op, cudaStream_t);

    // Basic binary operations (same input/output type - comprehensive list)
    template void launch_binary_op_generic<float, float, ops::add_op>(
        const float*, const float*, float*, size_t, ops::add_op, cudaStream_t);
    template void launch_binary_op_generic<int, int, ops::add_op>(
        const int*, const int*, int*, size_t, ops::add_op, cudaStream_t);
    template void launch_binary_op_generic<float, float, ops::sub_op>(
        const float*, const float*, float*, size_t, ops::sub_op, cudaStream_t);
    template void launch_binary_op_generic<int, int, ops::sub_op>(
        const int*, const int*, int*, size_t, ops::sub_op, cudaStream_t);
    template void launch_binary_op_generic<float, float, ops::mul_op>(
        const float*, const float*, float*, size_t, ops::mul_op, cudaStream_t);
    template void launch_binary_op_generic<int, int, ops::mul_op>(
        const int*, const int*, int*, size_t, ops::mul_op, cudaStream_t);
    template void launch_binary_op_generic<float, float, ops::div_op>(
        const float*, const float*, float*, size_t, ops::div_op, cudaStream_t);
    template void launch_binary_op_generic<int, int, ops::div_op>(
        const int*, const int*, int*, size_t, ops::div_op, cudaStream_t);
    template void launch_binary_op_generic<float, float, ops::minimum_op>(
        const float*, const float*, float*, size_t, ops::minimum_op, cudaStream_t);
    template void launch_binary_op_generic<int, int, ops::minimum_op>(
        const int*, const int*, int*, size_t, ops::minimum_op, cudaStream_t);
    template void launch_binary_op_generic<float, float, ops::maximum_op>(
        const float*, const float*, float*, size_t, ops::maximum_op, cudaStream_t);
    template void launch_binary_op_generic<int, int, ops::maximum_op>(
        const int*, const int*, int*, size_t, ops::maximum_op, cudaStream_t);
    template void launch_binary_op_generic<float, float, ops::pow_op>(
        const float*, const float*, float*, size_t, ops::pow_op, cudaStream_t);
    template void launch_binary_op_generic<int, int, ops::pow_op>(
        const int*, const int*, int*, size_t, ops::pow_op, cudaStream_t);

    // Comparison operations (input T -> output unsigned char/bool)
    template void launch_binary_op_generic<float, unsigned char, ops::greater_op>(
        const float*, const float*, unsigned char*, size_t, ops::greater_op, cudaStream_t);
    template void launch_binary_op_generic<int, unsigned char, ops::greater_op>(
        const int*, const int*, unsigned char*, size_t, ops::greater_op, cudaStream_t);
    template void launch_binary_op_generic<unsigned char, unsigned char, ops::greater_op>(
        const unsigned char*, const unsigned char*, unsigned char*, size_t, ops::greater_op, cudaStream_t);

    template void launch_binary_op_generic<float, unsigned char, ops::greater_equal_op>(
        const float*, const float*, unsigned char*, size_t, ops::greater_equal_op, cudaStream_t);
    template void launch_binary_op_generic<int, unsigned char, ops::greater_equal_op>(
        const int*, const int*, unsigned char*, size_t, ops::greater_equal_op, cudaStream_t);
    template void launch_binary_op_generic<unsigned char, unsigned char, ops::greater_equal_op>(
        const unsigned char*, const unsigned char*, unsigned char*, size_t, ops::greater_equal_op, cudaStream_t);

    template void launch_binary_op_generic<float, unsigned char, ops::less_equal_op>(
        const float*, const float*, unsigned char*, size_t, ops::less_equal_op, cudaStream_t);
    template void launch_binary_op_generic<int, unsigned char, ops::less_equal_op>(
        const int*, const int*, unsigned char*, size_t, ops::less_equal_op, cudaStream_t);
    template void launch_binary_op_generic<unsigned char, unsigned char, ops::less_equal_op>(
        const unsigned char*, const unsigned char*, unsigned char*, size_t, ops::less_equal_op, cudaStream_t);

    // Logical operations (bool/unsigned char -> unsigned char)
    template void launch_binary_op_generic<float, unsigned char, ops::logical_and_op>(
        const float*, const float*, unsigned char*, size_t, ops::logical_and_op, cudaStream_t);
    template void launch_binary_op_generic<int, unsigned char, ops::logical_and_op>(
        const int*, const int*, unsigned char*, size_t, ops::logical_and_op, cudaStream_t);
    template void launch_binary_op_generic<unsigned char, unsigned char, ops::logical_and_op>(
        const unsigned char*, const unsigned char*, unsigned char*, size_t, ops::logical_and_op, cudaStream_t);

    template void launch_binary_op_generic<float, unsigned char, ops::logical_or_op>(
        const float*, const float*, unsigned char*, size_t, ops::logical_or_op, cudaStream_t);
    template void launch_binary_op_generic<int, unsigned char, ops::logical_or_op>(
        const int*, const int*, unsigned char*, size_t, ops::logical_or_op, cudaStream_t);
    template void launch_binary_op_generic<unsigned char, unsigned char, ops::logical_or_op>(
        const unsigned char*, const unsigned char*, unsigned char*, size_t, ops::logical_or_op, cudaStream_t);

    template void launch_binary_op_generic<float, unsigned char, ops::less_op>(
        const float*, const float*, unsigned char*, size_t, ops::less_op, cudaStream_t);
    template void launch_binary_op_generic<int, unsigned char, ops::less_op>(
        const int*, const int*, unsigned char*, size_t, ops::less_op, cudaStream_t);
    template void launch_binary_op_generic<unsigned char, unsigned char, ops::less_op>(
        const unsigned char*, const unsigned char*, unsigned char*, size_t, ops::less_op, cudaStream_t);

    template void launch_binary_op_generic<float, unsigned char, ops::equal_op>(
        const float*, const float*, unsigned char*, size_t, ops::equal_op, cudaStream_t);
    template void launch_binary_op_generic<int, unsigned char, ops::equal_op>(
        const int*, const int*, unsigned char*, size_t, ops::equal_op, cudaStream_t);
    template void launch_binary_op_generic<unsigned char, unsigned char, ops::equal_op>(
        const unsigned char*, const unsigned char*, unsigned char*, size_t, ops::equal_op, cudaStream_t);

    template void launch_binary_op_generic<float, unsigned char, ops::not_equal_op>(
        const float*, const float*, unsigned char*, size_t, ops::not_equal_op, cudaStream_t);
    template void launch_binary_op_generic<int, unsigned char, ops::not_equal_op>(
        const int*, const int*, unsigned char*, size_t, ops::not_equal_op, cudaStream_t);
    template void launch_binary_op_generic<unsigned char, unsigned char, ops::not_equal_op>(
        const unsigned char*, const unsigned char*, unsigned char*, size_t, ops::not_equal_op, cudaStream_t);

    // Scalar operations (uses constant_iterator, different from scalar_right_op!)
    template void launch_scalar_op_generic<float, float, ops::add_op>(
        const float*, float, float*, size_t, ops::add_op, cudaStream_t);
    template void launch_scalar_op_generic<float, float, ops::sub_op>(
        const float*, float, float*, size_t, ops::sub_op, cudaStream_t);
    template void launch_scalar_op_generic<float, float, ops::mul_op>(
        const float*, float, float*, size_t, ops::mul_op, cudaStream_t);

    // scalar_right_op instantiations for various operations (comprehensive list)
    template void launch_unary_op_generic<float, float, ops::scalar_right_op<ops::add_op, float>>(
        const float*, float*, size_t, ops::scalar_right_op<ops::add_op, float>, cudaStream_t);
    template void launch_unary_op_generic<int, int, ops::scalar_right_op<ops::add_op, float>>(
        const int*, int*, size_t, ops::scalar_right_op<ops::add_op, float>, cudaStream_t);
    template void launch_unary_op_generic<float, float, ops::scalar_right_op<ops::sub_op, float>>(
        const float*, float*, size_t, ops::scalar_right_op<ops::sub_op, float>, cudaStream_t);
    template void launch_unary_op_generic<int, int, ops::scalar_right_op<ops::sub_op, float>>(
        const int*, int*, size_t, ops::scalar_right_op<ops::sub_op, float>, cudaStream_t);
    template void launch_unary_op_generic<float, float, ops::scalar_right_op<ops::mul_op, float>>(
        const float*, float*, size_t, ops::scalar_right_op<ops::mul_op, float>, cudaStream_t);
    template void launch_unary_op_generic<int, int, ops::scalar_right_op<ops::mul_op, float>>(
        const int*, int*, size_t, ops::scalar_right_op<ops::mul_op, float>, cudaStream_t);
    template void launch_unary_op_generic<float, float, ops::scalar_right_op<ops::div_op, float>>(
        const float*, float*, size_t, ops::scalar_right_op<ops::div_op, float>, cudaStream_t);
    template void launch_unary_op_generic<int, int, ops::scalar_right_op<ops::div_op, float>>(
        const int*, int*, size_t, ops::scalar_right_op<ops::div_op, float>, cudaStream_t);
    template void launch_unary_op_generic<float, float, ops::scalar_right_op<ops::pow_op, float>>(
        const float*, float*, size_t, ops::scalar_right_op<ops::pow_op, float>, cudaStream_t);
    template void launch_unary_op_generic<int, int, ops::scalar_right_op<ops::pow_op, float>>(
        const int*, int*, size_t, ops::scalar_right_op<ops::pow_op, float>, cudaStream_t);
    template void launch_unary_op_generic<float, unsigned char, ops::scalar_right_op<ops::not_equal_op, float>>(
        const float*, unsigned char*, size_t, ops::scalar_right_op<ops::not_equal_op, float>, cudaStream_t);
    template void launch_unary_op_generic<int, unsigned char, ops::scalar_right_op<ops::not_equal_op, float>>(
        const int*, unsigned char*, size_t, ops::scalar_right_op<ops::not_equal_op, float>, cudaStream_t);
    template void launch_unary_op_generic<unsigned char, unsigned char, ops::scalar_right_op<ops::not_equal_op, float>>(
        const unsigned char*, unsigned char*, size_t, ops::scalar_right_op<ops::not_equal_op, float>, cudaStream_t);

    template void launch_unary_op_generic<float, unsigned char, ops::scalar_right_op<ops::equal_op, float>>(
        const float*, unsigned char*, size_t, ops::scalar_right_op<ops::equal_op, float>, cudaStream_t);
    template void launch_unary_op_generic<int, unsigned char, ops::scalar_right_op<ops::equal_op, float>>(
        const int*, unsigned char*, size_t, ops::scalar_right_op<ops::equal_op, float>, cudaStream_t);
    template void launch_unary_op_generic<unsigned char, unsigned char, ops::scalar_right_op<ops::equal_op, float>>(
        const unsigned char*, unsigned char*, size_t, ops::scalar_right_op<ops::equal_op, float>, cudaStream_t);

    template void launch_unary_op_generic<float, unsigned char, ops::scalar_right_op<ops::greater_op, float>>(
        const float*, unsigned char*, size_t, ops::scalar_right_op<ops::greater_op, float>, cudaStream_t);
    template void launch_unary_op_generic<int, unsigned char, ops::scalar_right_op<ops::greater_op, float>>(
        const int*, unsigned char*, size_t, ops::scalar_right_op<ops::greater_op, float>, cudaStream_t);
    template void launch_unary_op_generic<unsigned char, unsigned char, ops::scalar_right_op<ops::greater_op, float>>(
        const unsigned char*, unsigned char*, size_t, ops::scalar_right_op<ops::greater_op, float>, cudaStream_t);

    template void launch_unary_op_generic<float, unsigned char, ops::scalar_right_op<ops::less_op, float>>(
        const float*, unsigned char*, size_t, ops::scalar_right_op<ops::less_op, float>, cudaStream_t);
    template void launch_unary_op_generic<int, unsigned char, ops::scalar_right_op<ops::less_op, float>>(
        const int*, unsigned char*, size_t, ops::scalar_right_op<ops::less_op, float>, cudaStream_t);
    template void launch_unary_op_generic<unsigned char, unsigned char, ops::scalar_right_op<ops::less_op, float>>(
        const unsigned char*, unsigned char*, size_t, ops::scalar_right_op<ops::less_op, float>, cudaStream_t);

    template void launch_unary_op_generic<float, unsigned char, ops::scalar_right_op<ops::greater_equal_op, float>>(
        const float*, unsigned char*, size_t, ops::scalar_right_op<ops::greater_equal_op, float>, cudaStream_t);
    template void launch_unary_op_generic<int, unsigned char, ops::scalar_right_op<ops::greater_equal_op, float>>(
        const int*, unsigned char*, size_t, ops::scalar_right_op<ops::greater_equal_op, float>, cudaStream_t);
    template void launch_unary_op_generic<unsigned char, unsigned char, ops::scalar_right_op<ops::greater_equal_op, float>>(
        const unsigned char*, unsigned char*, size_t, ops::scalar_right_op<ops::greater_equal_op, float>, cudaStream_t);

    template void launch_unary_op_generic<float, unsigned char, ops::scalar_right_op<ops::less_equal_op, float>>(
        const float*, unsigned char*, size_t, ops::scalar_right_op<ops::less_equal_op, float>, cudaStream_t);
    template void launch_unary_op_generic<int, unsigned char, ops::scalar_right_op<ops::less_equal_op, float>>(
        const int*, unsigned char*, size_t, ops::scalar_right_op<ops::less_equal_op, float>, cudaStream_t);
    template void launch_unary_op_generic<unsigned char, unsigned char, ops::scalar_right_op<ops::less_equal_op, float>>(
        const unsigned char*, unsigned char*, size_t, ops::scalar_right_op<ops::less_equal_op, float>, cudaStream_t);

    template void launch_unary_op_generic<float, float, ops::scalar_right_op<ops::mod_op, float>>(
        const float*, float*, size_t, ops::scalar_right_op<ops::mod_op, float>, cudaStream_t);
    template void launch_unary_op_generic<int, int, ops::scalar_right_op<ops::mod_op, float>>(
        const int*, int*, size_t, ops::scalar_right_op<ops::mod_op, float>, cudaStream_t);

    // Composed unary operations (expression template fusion) - test-specific
    template void launch_unary_op_generic<float, float, ops::composed_unary_op<ops::exp_op, ops::scalar_right_op<ops::mul_op, float>>>(
        const float*, float*, size_t, ops::composed_unary_op<ops::exp_op, ops::scalar_right_op<ops::mul_op, float>>, cudaStream_t);

    template void launch_unary_op_generic<float, float, ops::composed_unary_op<ops::scalar_right_op<ops::mul_op, float>, ops::abs_op>>(
        const float*, float*, size_t, ops::composed_unary_op<ops::scalar_right_op<ops::mul_op, float>, ops::abs_op>, cudaStream_t);

    template void launch_unary_op_generic<float, float, ops::composed_unary_op<ops::scalar_right_op<ops::mul_op, float>, ops::relu_op>>(
        const float*, float*, size_t, ops::composed_unary_op<ops::scalar_right_op<ops::mul_op, float>, ops::relu_op>, cudaStream_t);

} // namespace lfs::core::tensor_ops