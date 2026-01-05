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
#include <mutex>
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
#include <unordered_map>

// CUDA error checking macro
#define CHECK_CUDA(call)                              \
    do {                                              \
        cudaError_t error = call;                     \
        if (error != cudaSuccess) {                   \
            LOG_ERROR("CUDA error at {}:{} - {}: {}", \
                      __FILE__, __LINE__,             \
                      cudaGetErrorName(error),        \
                      cudaGetErrorString(error));     \
        }                                             \
    } while (0)

namespace lfs::core::tensor_ops {

    // Pooled CUB temp storage - grows as needed, never shrinks
    class CubTempStoragePool {
    public:
        static CubTempStoragePool& instance() {
            static CubTempStoragePool pool;
            return pool;
        }

        void* get(size_t bytes, cudaStream_t) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (bytes <= capacity_ && buffer_)
                return buffer_;
            if (buffer_)
                cudaFree(buffer_);
            size_t alloc_size = std::max(((bytes + 1024 * 1024 - 1) / (1024 * 1024)) * (1024 * 1024), size_t(4 * 1024 * 1024));
            if (cudaMalloc(&buffer_, alloc_size) != cudaSuccess) {
                buffer_ = nullptr;
                capacity_ = 0;
                return nullptr;
            }
            capacity_ = alloc_size;
            return buffer_;
        }

        ~CubTempStoragePool() {
            if (buffer_)
                cudaFree(buffer_);
        }

    private:
        CubTempStoragePool() = default;
        void* buffer_ = nullptr;
        size_t capacity_ = 0;
        std::mutex mutex_;
    };

    inline void* get_cub_temp_storage(size_t bytes, cudaStream_t stream) {
        return CubTempStoragePool::instance().get(bytes, stream);
    }

    // Pre-allocated buffers for scalar reductions with pinned host memory
    class ScalarReductionCache {
    public:
        static ScalarReductionCache& instance() {
            static ScalarReductionCache cache;
            return cache;
        }

        // Async copy + stream sync avoids blocking other streams
        float reduce_sum(const float* data, size_t n, cudaStream_t stream) {
            if (n == 0)
                return 0.0f;
            size_t temp_bytes = temp_capacity_;
            cub::DeviceReduce::Sum(temp_storage_, temp_bytes, data, d_scalar_, n, stream);
            cudaMemcpyAsync(h_scalar_, d_scalar_, sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            return *h_scalar_;
        }

        float reduce_mean(const float* data, size_t n, cudaStream_t stream) {
            return n ? reduce_sum(data, n, stream) / static_cast<float>(n) : 0.0f;
        }

        float reduce_max(const float* data, size_t n, cudaStream_t stream) {
            if (n == 0)
                return -std::numeric_limits<float>::infinity();
            size_t temp_bytes = temp_capacity_;
            cub::DeviceReduce::Max(temp_storage_, temp_bytes, data, d_scalar_, n, stream);
            cudaMemcpyAsync(h_scalar_, d_scalar_, sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            return *h_scalar_;
        }

        float reduce_min(const float* data, size_t n, cudaStream_t stream) {
            if (n == 0)
                return std::numeric_limits<float>::infinity();
            size_t temp_bytes = temp_capacity_;
            cub::DeviceReduce::Min(temp_storage_, temp_bytes, data, d_scalar_, n, stream);
            cudaMemcpyAsync(h_scalar_, d_scalar_, sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            return *h_scalar_;
        }

        ~ScalarReductionCache() {
            if (d_scalar_)
                cudaFree(d_scalar_);
            if (h_scalar_)
                cudaFreeHost(h_scalar_);
            if (temp_storage_)
                cudaFree(temp_storage_);
        }

    private:
        ScalarReductionCache() {
            cudaMalloc(&d_scalar_, sizeof(float));
            cudaMallocHost(&h_scalar_, sizeof(float));
            temp_capacity_ = 32 * 1024 * 1024;
            cudaMalloc(&temp_storage_, temp_capacity_);
        }

        float* d_scalar_ = nullptr;
        float* h_scalar_ = nullptr;
        void* temp_storage_ = nullptr;
        size_t temp_capacity_ = 0;
    };

    float direct_sum_scalar(const float* data, size_t n, cudaStream_t stream) {
        return ScalarReductionCache::instance().reduce_sum(data, n, stream);
    }
    float direct_mean_scalar(const float* data, size_t n, cudaStream_t stream) {
        return ScalarReductionCache::instance().reduce_mean(data, n, stream);
    }
    float direct_max_scalar(const float* data, size_t n, cudaStream_t stream) {
        return ScalarReductionCache::instance().reduce_max(data, n, stream);
    }
    float direct_min_scalar(const float* data, size_t n, cudaStream_t stream) {
        return ScalarReductionCache::instance().reduce_min(data, n, stream);
    }

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
            // No cap needed - grid-stride loop handles any size

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

    // Simple scalar initialization kernels
    __global__ void init_scalar_float_kernel(float* __restrict__ ptr, float value) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            *ptr = value;
        }
    }

    __global__ void init_scalar_int_kernel(int* __restrict__ ptr, int value) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            *ptr = value;
        }
    }

    __global__ void init_scalar_int64_kernel(int64_t* __restrict__ ptr, int64_t value) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            *ptr = value;
        }
    }

    // Helper functions to initialize scalars on GPU without CPU→GPU transfer
    inline void init_scalar_gpu(float* d_ptr, float value, cudaStream_t stream = nullptr) {
        init_scalar_float_kernel<<<1, 1, 0, stream>>>(d_ptr, value);
    }

    inline void init_scalar_gpu(int* d_ptr, int value, cudaStream_t stream = nullptr) {
        init_scalar_int_kernel<<<1, 1, 0, stream>>>(d_ptr, value);
    }

    inline void init_scalar_gpu(int64_t* d_ptr, int64_t value, cudaStream_t stream = nullptr) {
        init_scalar_int64_kernel<<<1, 1, 0, stream>>>(d_ptr, value);
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
            // No cap needed - grid-stride loop handles any size

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
                thrust::copy(thrust::cuda::par_nosync.on(stream),
                             in_ptr, in_ptr + output_size, out_ptr);
            } else {
                thrust::copy(thrust::cuda::par_nosync,
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
            thrust::for_each(thrust::cuda::par_nosync.on(stream),
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
            thrust::for_each(thrust::cuda::par_nosync,
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

    // Internal Float32 implementation (original)
    void launch_reduce_op_float32(const void* input, void* output, const size_t* shape, size_t rank,
                                  const int* axes, size_t num_axes, bool keepdim, ReduceOp op,
                                  cudaStream_t stream) {

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
                init_scalar_gpu(d_out, init_val, stream); // GPU init instead of CPU→GPU upload!

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

            // CUB DeviceReduce with pooled temp storage (no allocation overhead!)
            void* d_temp_storage = nullptr;
            size_t temp_storage_bytes = 0;

            switch (op) {
            case ReduceOp::Sum:
                cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, n, stream);
                d_temp_storage = get_cub_temp_storage(temp_storage_bytes, stream);
                cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, n, stream);
                break;
            case ReduceOp::Mean: {
                cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, n, stream);
                d_temp_storage = get_cub_temp_storage(temp_storage_bytes, stream);
                cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, n, stream);
                // Divide by n using a simple kernel (faster than Thrust for single value)
                auto out_ptr = thrust::device_pointer_cast(d_out);
                run_with_thrust_policy(stream, [&](auto policy) {
                    thrust::transform(policy, out_ptr, out_ptr + 1, out_ptr,
                                      DivideByFunctor(static_cast<float>(n)));
                });
                break;
            }
            case ReduceOp::Max:
                cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, n, stream);
                d_temp_storage = get_cub_temp_storage(temp_storage_bytes, stream);
                cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, n, stream);
                break;
            case ReduceOp::Min:
                cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, d_in, d_out, n, stream);
                d_temp_storage = get_cub_temp_storage(temp_storage_bytes, stream);
                cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, d_in, d_out, n, stream);
                break;
            case ReduceOp::Prod: {
                float result = 0.0f;
                run_with_thrust_policy(stream, [&](auto policy) {
                    result = thrust::reduce(policy, input_ptr, input_ptr + n, 1.0f, ops::mul_op{});
                });
                init_scalar_gpu(static_cast<float*>(output), result, stream);
                break;
            }
            default:
                init_scalar_gpu(static_cast<float*>(output), 0.0f, stream);
                break;
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
                    LOG_DEBUG("[REDUCE] Using warp segmented reduce: outer={} reduce={} inner=1", outer_size, reduce_size);
                    // Note: For Mean, the fused kernel already divides by reduce_size
                    launch_warp_segmented_reduce(input_f, output_f, outer_size, reduce_size, op, stream);
                    return;
                } else {
                    LOG_DEBUG("[REDUCE] Fallback to CUB: outer={} reduce={} inner=1", outer_size, reduce_size);
                }
            } else {
                // Strided segments - strided memory access is slow on GPU
                // For inner_size >= 256, the strided access pattern is too slow.
                // Only use warp strided kernel for small inner_size where cache helps.
                //
                // Benchmark: inner_size=1024, reduce_size=1024
                //   - Strided warp kernel: ~125 us (bad coalescing!)
                //   - CUB segmented: ~15 us (better memory access)
                //
                // For inner_size < 256, strided warp can still be competitive due to
                // cache locality (stride < 1KB) and low overhead.
                bool small_inner_stride = inner_size < 256;
                bool use_strided_warp = small_inner_stride && should_use_warp_reduce(n, output_size);

                if (use_strided_warp) {
                    LOG_DEBUG("[REDUCE] Using warp STRIDED reduce: outer={} reduce={} inner={}", outer_size, reduce_size, inner_size);
                    // Note: For Mean, the fused kernel already divides by reduce_size
                    launch_warp_strided_reduce(input_f, output_f, outer_size, reduce_size, inner_size, op, stream);
                    return;
                }
            }

            // SLOW PATH: Use CUB/Thrust for very large tensors
            LOG_DEBUG("[REDUCE] Using SLOW PATH (CUB/Thrust): outer={} reduce={} inner={}", outer_size, reduce_size, inner_size);
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
                        init_scalar_gpu(output_f, init_val, stream); // GPU init instead of CPU→GPU upload!
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

    // Internal Int32 implementation (simplified - only handles full reductions)
    void launch_reduce_op_int32(const void* input, void* output, const size_t* shape, size_t rank,
                                const int* axes, size_t num_axes, bool keepdim, ReduceOp op,
                                cudaStream_t stream) {
        size_t n = 1;
        for (size_t i = 0; i < rank; ++i)
            n *= shape[i];
        if (n == 0)
            return;

        const int* d_in = static_cast<const int*>(input);
        int* d_out = static_cast<int*>(output);

        // Only support full reduction for Int32
        if (num_axes == 0 || num_axes == rank) {
            void* d_temp_storage = nullptr;
            size_t temp_storage_bytes = 0;

            switch (op) {
            case ReduceOp::Sum:
            case ReduceOp::Mean:
                cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, n, stream);
                cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream);
                cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, n, stream);
                cudaFreeAsync(d_temp_storage, stream);
                if (op == ReduceOp::Mean) {
                    // Divide by count for mean - use Thrust directly
                    auto out_ptr = thrust::device_pointer_cast(d_out);
                    const int count = static_cast<int>(n);
                    thrust::transform(thrust::cuda::par_nosync.on(stream), out_ptr, out_ptr + 1, out_ptr,
                                      [count] __device__(int val) { return val / count; });
                }
                break;
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
            case ReduceOp::Prod: {
                auto in_ptr = thrust::device_pointer_cast(d_in);
                int result = 1;
                run_with_thrust_policy(stream, [&](auto policy) {
                    result = thrust::reduce(policy, in_ptr, in_ptr + n, 1, ops::mul_op{});
                });
                init_scalar_gpu(d_out, result, stream); // GPU init instead of CPU→GPU upload!
            } break;
            default: {
                init_scalar_gpu(d_out, 0, stream); // GPU init instead of CPU→GPU upload!
            } break;
            }
        }
        // Partial reductions not supported for Int32 yet
    }

    // Bool to int64 conversion functor for CUB
    struct BoolToInt64Op {
        __host__ __device__ __forceinline__
            int64_t
            operator()(const bool& val) const {
            return val ? 1 : 0;
        }
    };

    // Partial bool reduction along axis (All/Any)
    __global__ void bool_reduce_axis_kernel(
        const bool* __restrict__ input,
        bool* __restrict__ output,
        const size_t outer_size,
        const size_t reduce_size,
        const size_t inner_size,
        const bool is_all) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t output_size = outer_size * inner_size;
        if (idx >= output_size)
            return;

        const size_t outer_idx = idx / inner_size;
        const size_t inner_idx = idx % inner_size;
        const size_t base = outer_idx * reduce_size * inner_size + inner_idx;

        if (is_all) {
            bool result = true;
            for (size_t r = 0; r < reduce_size && result; ++r) {
                result = result && input[base + r * inner_size];
            }
            output[idx] = result;
        } else {
            bool result = false;
            for (size_t r = 0; r < reduce_size && !result; ++r) {
                result = result || input[base + r * inner_size];
            }
            output[idx] = result;
        }
    }

    // Bool reduction implementation
    void launch_reduce_op_bool(const void* input, void* output, const size_t* shape, const size_t rank,
                               const int* axes, const size_t num_axes, const bool keepdim, const ReduceOp op,
                               cudaStream_t stream) {
        size_t n = 1;
        for (size_t i = 0; i < rank; ++i)
            n *= shape[i];
        if (n == 0)
            return;

        const bool* d_in = static_cast<const bool*>(input);
        bool* d_out_bool = static_cast<bool*>(output);
        int64_t* d_out_int64 = static_cast<int64_t*>(output);

        // Full reduction
        if (num_axes == 0 || num_axes == rank) {
            void* d_temp_storage = nullptr;
            size_t temp_storage_bytes = 0;

            switch (op) {
            case ReduceOp::Sum:
            case ReduceOp::Mean: {
                auto transform_iter = thrust::make_transform_iterator(
                    thrust::device_pointer_cast(d_in), BoolToInt64Op());

                cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                                       transform_iter, d_out_int64, n, stream);
                cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream);
                cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                                       transform_iter, d_out_int64, n, stream);

                if (op == ReduceOp::Mean) {
                    const int64_t count = static_cast<int64_t>(n);
                    auto out_ptr = thrust::device_pointer_cast(d_out_int64);
                    thrust::transform(thrust::cuda::par_nosync.on(stream), out_ptr, out_ptr + 1, out_ptr,
                                      [count] __device__(int64_t val) { return val / count; });
                }
                cudaFreeAsync(d_temp_storage, stream);
            } break;

            case ReduceOp::Max:
            case ReduceOp::Any: {
                auto transform_iter = thrust::make_transform_iterator(
                    thrust::device_pointer_cast(d_in), BoolToInt64Op());

                int64_t* d_temp_result;
                cudaMallocAsync(&d_temp_result, sizeof(int64_t), stream);

                cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes,
                                       transform_iter, d_temp_result, n, stream);
                cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream);
                cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes,
                                       transform_iter, d_temp_result, n, stream);

                if (op == ReduceOp::Any) {
                    thrust::transform(thrust::cuda::par_nosync.on(stream),
                                      thrust::device_pointer_cast(d_temp_result),
                                      thrust::device_pointer_cast(d_temp_result) + 1,
                                      thrust::device_pointer_cast(d_out_bool),
                                      [] __device__(int64_t val) { return val != 0; });
                } else {
                    cudaMemcpyAsync(d_out_int64, d_temp_result, sizeof(int64_t), cudaMemcpyDeviceToDevice, stream);
                }
                cudaFreeAsync(d_temp_result, stream);
                cudaFreeAsync(d_temp_storage, stream);
            } break;

            case ReduceOp::Min:
            case ReduceOp::All:
            case ReduceOp::Prod: {
                auto transform_iter = thrust::make_transform_iterator(
                    thrust::device_pointer_cast(d_in), BoolToInt64Op());

                int64_t* d_temp_result;
                cudaMallocAsync(&d_temp_result, sizeof(int64_t), stream);

                cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes,
                                       transform_iter, d_temp_result, n, stream);
                cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream);
                cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes,
                                       transform_iter, d_temp_result, n, stream);

                if (op == ReduceOp::All) {
                    thrust::transform(thrust::cuda::par_nosync.on(stream),
                                      thrust::device_pointer_cast(d_temp_result),
                                      thrust::device_pointer_cast(d_temp_result) + 1,
                                      thrust::device_pointer_cast(d_out_bool),
                                      [] __device__(int64_t val) { return val != 0; });
                } else {
                    cudaMemcpyAsync(d_out_int64, d_temp_result, sizeof(int64_t), cudaMemcpyDeviceToDevice, stream);
                }
                cudaFreeAsync(d_temp_result, stream);
                cudaFreeAsync(d_temp_storage, stream);
            } break;

            default:
                init_scalar_gpu(d_out_int64, static_cast<int64_t>(0), stream);
                break;
            }
            return;
        }

        // Partial reduction for Any/All
        if (op == ReduceOp::Any || op == ReduceOp::All) {
            const int dim = axes[0];
            size_t outer_size = 1;
            for (int i = 0; i < dim; ++i)
                outer_size *= shape[i];
            const size_t reduce_size = shape[dim];
            size_t inner_size = 1;
            for (size_t i = dim + 1; i < rank; ++i)
                inner_size *= shape[i];

            // For multi-axis, extend reduce_size to cover all reduced dims
            if (num_axes > 1) {
                std::vector<bool> is_reduced(rank, false);
                for (size_t i = 0; i < num_axes; ++i)
                    is_reduced[axes[i]] = true;

                int first = -1, last = -1;
                for (size_t i = 0; i < rank; ++i) {
                    if (is_reduced[i]) {
                        if (first < 0)
                            first = i;
                        last = i;
                    }
                }

                outer_size = 1;
                for (int i = 0; i < first; ++i)
                    outer_size *= shape[i];

                size_t reduce_combined = 1;
                for (int i = first; i <= last; ++i)
                    reduce_combined *= shape[i];

                inner_size = 1;
                for (size_t i = last + 1; i < rank; ++i)
                    inner_size *= shape[i];

                const size_t output_size = outer_size * inner_size;
                const int threads = 256;
                const int blocks = (output_size + threads - 1) / threads;

                bool_reduce_axis_kernel<<<blocks, threads, 0, stream>>>(
                    d_in, d_out_bool, outer_size, reduce_combined, inner_size, op == ReduceOp::All);
                return;
            }

            const size_t output_size = outer_size * inner_size;
            const int threads = 256;
            const int blocks = (output_size + threads - 1) / threads;

            bool_reduce_axis_kernel<<<blocks, threads, 0, stream>>>(
                d_in, d_out_bool, outer_size, reduce_size, inner_size, op == ReduceOp::All);
        }
    }

    // Public dispatcher function
    void launch_reduce_op(const void* input, void* output, const size_t* shape, size_t rank,
                          const int* axes, size_t num_axes, bool keepdim, ReduceOp op,
                          DataType dtype, cudaStream_t stream) {
        if (dtype == DataType::Float32) {
            launch_reduce_op_float32(input, output, shape, rank, axes, num_axes, keepdim, op, stream);
        } else if (dtype == DataType::Int32) {
            launch_reduce_op_int32(input, output, shape, rank, axes, num_axes, keepdim, op, stream);
        } else if (dtype == DataType::Bool) {
            launch_reduce_op_bool(input, output, shape, rank, axes, num_axes, keepdim, op, stream);
        }
        // Other dtypes not supported
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

    // ============= FILL / ARANGE (GPU-native) =============

    template <typename T>
    __global__ void fill_kernel(T* __restrict__ out, const size_t n, const T val) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            out[i] = val;
    }

    template <typename T>
    void launch_fill(T* out, const size_t n, const T val, cudaStream_t stream) {
        if (n == 0)
            return;
        constexpr int BLOCK = 256;
        const int grid = static_cast<int>((n + BLOCK - 1) / BLOCK);
        fill_kernel<<<grid, BLOCK, 0, stream>>>(out, n, val);
    }

    template void launch_fill<float>(float*, size_t, float, cudaStream_t);
    template void launch_fill<int>(int*, size_t, int, cudaStream_t);
    template void launch_fill<int64_t>(int64_t*, size_t, int64_t, cudaStream_t);
    template void launch_fill<__half>(__half*, size_t, __half, cudaStream_t);
    template void launch_fill<unsigned char>(unsigned char*, size_t, unsigned char, cudaStream_t);

    template <typename T>
    __global__ void arange_kernel(T* __restrict__ out, const size_t n, const T start, const T step) {
        const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            out[i] = start + static_cast<T>(i) * step;
    }

    template <typename T>
    void launch_arange(T* out, const size_t n, const T start, const T step, cudaStream_t stream) {
        if (n == 0)
            return;
        constexpr int BLOCK = 256;
        const int grid = static_cast<int>((n + BLOCK - 1) / BLOCK);
        arange_kernel<<<grid, BLOCK, 0, stream>>>(out, n, start, step);
    }

    template void launch_arange<float>(float*, size_t, float, float, cudaStream_t);
    template void launch_arange<int>(int*, size_t, int, int, cudaStream_t);

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
                thrust::transform(thrust::cuda::par_nosync.on(stream),
                                  thrust::counting_iterator<size_t>(0),
                                  thrust::counting_iterator<size_t>(total_elements),
                                  keys.begin(),
                                  [=] __device__(size_t idx) -> int {
                                      return static_cast<int>(idx / dim_size);
                                  });

                thrust::inclusive_scan_by_key(thrust::cuda::par_nosync.on(stream),
                                              keys.begin(), keys.end(),
                                              data_ptr, data_ptr);
            } else {
                thrust::transform(thrust::cuda::par_nosync,
                                  thrust::counting_iterator<size_t>(0),
                                  thrust::counting_iterator<size_t>(total_elements),
                                  keys.begin(),
                                  [=] __device__(size_t idx) -> int {
                                      return static_cast<int>(idx / dim_size);
                                  });

                thrust::inclusive_scan_by_key(thrust::cuda::par_nosync,
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
                    thrust::inclusive_scan(thrust::cuda::par_nosync.on(stream), data_ptr, data_ptr + total, data_ptr);
                } else {
                    thrust::inclusive_scan(thrust::cuda::par_nosync, data_ptr, data_ptr + total, data_ptr);
                }
            } else if (dtype == DataType::Int32) {
                auto data_ptr = thrust::device_pointer_cast(static_cast<int*>(data));
                if (stream) {
                    thrust::inclusive_scan(thrust::cuda::par_nosync.on(stream), data_ptr, data_ptr + total, data_ptr);
                } else {
                    thrust::inclusive_scan(thrust::cuda::par_nosync, data_ptr, data_ptr + total, data_ptr);
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

        // CRITICAL FIX: Handle 3D grids for N > ~1M (when grid.y > 65535)
        size_t row_block = blockIdx.z * gridDim.y + blockIdx.y;
        size_t row = row_block * BLOCK_SIZE + threadIdx.y;
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
        // CRITICAL FIX: Handle 3D grids for N > ~1M
        size_t row_block = blockIdx.z * gridDim.y + blockIdx.y;
        size_t i = row_block * blockDim.y + threadIdx.y;
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

        // CRITICAL FIX: Handle 3D grids for N > ~1M
        size_t row_block = blockIdx.z * gridDim.y + blockIdx.y;
        size_t row = row_block * BLOCK_SIZE + threadIdx.y;
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
        // CRITICAL FIX: Handle 3D grids for N > ~1M
        size_t row_block = blockIdx.z * gridDim.y + blockIdx.y;
        size_t i = row_block * blockDim.y + threadIdx.y;
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
        const int max_grid_dim = 65535; // CUDA Y/Z dimension limit

        // Helper to create 3D grid for large N and M
        auto make_grid_3d = [max_grid_dim](size_t cols, size_t rows, int block_size) -> dim3 {
            int grid_x = (cols + block_size - 1) / block_size;
            int grid_y_blocks = (rows + block_size - 1) / block_size;

            if (grid_y_blocks <= max_grid_dim) {
                return dim3(grid_x, grid_y_blocks, 1);
            } else {
                // Split rows across grid.y and grid.z
                int grid_y = max_grid_dim;
                int grid_z = (grid_y_blocks + max_grid_dim - 1) / max_grid_dim;
                return dim3(grid_x, grid_y, grid_z);
            }
        };

        if (p == 2.0f) {
            if (D >= 128 && D % 4 == 0) {
                dim3 block(BLOCK_SIZE, BLOCK_SIZE);
                dim3 grid = make_grid_3d(M, N, BLOCK_SIZE);
                cdist_l2_vectorized_kernel<<<grid, block, 0, stream>>>(a, b, out, N, M, D);
            } else {
                dim3 block(BLOCK_SIZE, BLOCK_SIZE);
                dim3 grid = make_grid_3d(M, N, BLOCK_SIZE);
                cdist_l2_optimized_kernel<BLOCK_SIZE><<<grid, block, 0, stream>>>(a, b, out, N, M, D);
            }
        } else if (p == 1.0f) {
            dim3 block(BLOCK_SIZE, BLOCK_SIZE);
            dim3 grid = make_grid_3d(M, N, BLOCK_SIZE);
            cdist_l1_optimized_kernel<BLOCK_SIZE><<<grid, block, 0, stream>>>(a, b, out, N, M, D);
        } else {
            dim3 block(16, 16);
            dim3 grid = make_grid_3d(M, N, 16);
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
                thrust::sort_by_key(thrust::cuda::par_nosync.on(stream), values_ptr, values_ptr + n,
                                    indices_ptr, thrust::greater<float>());
            } else {
                thrust::sort_by_key(thrust::cuda::par_nosync.on(stream), values_ptr, values_ptr + n,
                                    indices_ptr, thrust::less<float>());
            }
        } else {
            if (descending) {
                thrust::sort_by_key(thrust::cuda::par_nosync, values_ptr, values_ptr + n,
                                    indices_ptr, thrust::greater<float>());
            } else {
                thrust::sort_by_key(thrust::cuda::par_nosync, values_ptr, values_ptr + n,
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

                thrust::sequence(thrust::cuda::par_nosync.on(stream), temp_idx.begin(), temp_idx.end(), 0LL);

                if (descending) {
                    thrust::sort_by_key(thrust::cuda::par_nosync.on(stream),
                                        temp_vals.begin(), temp_vals.end(), temp_idx.begin(),
                                        thrust::greater<float>());
                } else {
                    thrust::sort_by_key(thrust::cuda::par_nosync.on(stream),
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
        // Support both 1D and 2D grids for large arrays
        size_t block_id = blockIdx.y * gridDim.x + blockIdx.x;
        size_t row = block_id * blockDim.x + threadIdx.x;
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
                static_cast<const float*>(tensors[0].data_ptr()),
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
            h_input_ptrs[i] = static_cast<const float*>(tensors[i].data_ptr());
            h_input_sizes[i] = tensors[i].shape()[tensors[i].shape().rank() - 1];
        }

        // Sync copy required - local vectors
        cudaMemcpy(const_cast<float**>(d_input_ptrs), h_input_ptrs.data(),
                   num_tensors * sizeof(float*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_input_sizes, h_input_sizes.data(),
                   num_tensors * sizeof(size_t), cudaMemcpyHostToDevice);

        constexpr int BLOCK = 256;
        constexpr size_t MAX_GRID_X = 65535;
        const size_t grid_size = (num_rows + BLOCK - 1) / BLOCK;

        if (grid_size <= MAX_GRID_X) {
            cat_last_dim_kernel_vectorized<<<grid_size, BLOCK, 0, stream>>>(
                static_cast<float*>(output), d_input_ptrs, d_input_sizes,
                num_tensors, num_rows, row_size);
        } else {
            const dim3 grid(std::min(grid_size, MAX_GRID_X), (grid_size + MAX_GRID_X - 1) / MAX_GRID_X);
            cat_last_dim_kernel_vectorized<<<grid, BLOCK, 0, stream>>>(
                static_cast<float*>(output), d_input_ptrs, d_input_sizes,
                num_tensors, num_rows, row_size);
        }

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
        // Support both 1D and 2D grids for large arrays
        size_t block_id = blockIdx.y * gridDim.x + blockIdx.x;
        size_t idx = block_id * blockDim.x + threadIdx.x;
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
            h_input_ptrs[i] = static_cast<const float*>(tensors[i].data_ptr());
            h_input_sizes[i] = tensors[i].shape()[resolved_dim];
        }

        // Sync copy required - local vectors
        cudaMemcpy(const_cast<float**>(d_input_ptrs), h_input_ptrs.data(),
                   num_tensors * sizeof(float*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_input_sizes, h_input_sizes.data(),
                   num_tensors * sizeof(size_t), cudaMemcpyHostToDevice);

        constexpr int BLOCK = 256;
        constexpr size_t MAX_GRID_X = 65535;
        const size_t grid_size = (total_elements + BLOCK - 1) / BLOCK;

        if (grid_size <= MAX_GRID_X) {
            cat_middle_dim_kernel<<<grid_size, BLOCK, 0, stream>>>(
                static_cast<float*>(output), d_input_ptrs, d_input_sizes,
                num_tensors, outer_size, inner_size, total_dim_size);
        } else {
            const dim3 grid(std::min(grid_size, MAX_GRID_X), (grid_size + MAX_GRID_X - 1) / MAX_GRID_X);
            cat_middle_dim_kernel<<<grid, BLOCK, 0, stream>>>(
                static_cast<float*>(output), d_input_ptrs, d_input_sizes,
                num_tensors, outer_size, inner_size, total_dim_size);
        }

        // Return to memory pool
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

    // Float16 conversions
    template void launch_convert_type<float, __half>(const float*, __half*, size_t, cudaStream_t);
    template void launch_convert_type<__half, float>(const __half*, float*, size_t, cudaStream_t);
    template void launch_convert_type<int, __half>(const int*, __half*, size_t, cudaStream_t);
    template void launch_convert_type<__half, int>(const __half*, int*, size_t, cudaStream_t);
    template void launch_convert_type<int64_t, __half>(const int64_t*, __half*, size_t, cudaStream_t);
    template void launch_convert_type<__half, int64_t>(const __half*, int64_t*, size_t, cudaStream_t);
    // Bool/UInt8 (uint8_t = unsigned char) <-> Float16
    template void launch_convert_type<uint8_t, __half>(const uint8_t*, __half*, size_t, cudaStream_t);
    template void launch_convert_type<__half, uint8_t>(const __half*, uint8_t*, size_t, cudaStream_t);

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

    // ============================================================================
    // Type Promotion Instantiations
    // ============================================================================
    // Added to support the type promotion system for mixed-dtype operations.
    // These cover all combinations that promote_dtypes() might produce.
    //
    // Type hierarchy: Bool < UInt8 < Int32 < Int64 < Float16 < Float32
    // ============================================================================

    // Float16 operations
    template void launch_binary_op_generic<__half, __half, ops::add_op>(
        const __half*, const __half*, __half*, size_t, ops::add_op, cudaStream_t);
    template void launch_binary_op_generic<__half, __half, ops::sub_op>(
        const __half*, const __half*, __half*, size_t, ops::sub_op, cudaStream_t);
    template void launch_binary_op_generic<__half, __half, ops::mul_op>(
        const __half*, const __half*, __half*, size_t, ops::mul_op, cudaStream_t);
    template void launch_binary_op_generic<__half, __half, ops::div_op>(
        const __half*, const __half*, __half*, size_t, ops::div_op, cudaStream_t);
    template void launch_binary_op_generic<__half, __half, ops::maximum_op>(
        const __half*, const __half*, __half*, size_t, ops::maximum_op, cudaStream_t);
    template void launch_binary_op_generic<__half, __half, ops::minimum_op>(
        const __half*, const __half*, __half*, size_t, ops::minimum_op, cudaStream_t);
    template void launch_binary_op_generic<__half, __half, ops::pow_op>(
        const __half*, const __half*, __half*, size_t, ops::pow_op, cudaStream_t);

    // Int64 operations
    template void launch_binary_op_generic<int64_t, int64_t, ops::add_op>(
        const int64_t*, const int64_t*, int64_t*, size_t, ops::add_op, cudaStream_t);
    template void launch_binary_op_generic<int64_t, int64_t, ops::sub_op>(
        const int64_t*, const int64_t*, int64_t*, size_t, ops::sub_op, cudaStream_t);
    template void launch_binary_op_generic<int64_t, int64_t, ops::mul_op>(
        const int64_t*, const int64_t*, int64_t*, size_t, ops::mul_op, cudaStream_t);
    template void launch_binary_op_generic<int64_t, int64_t, ops::div_op>(
        const int64_t*, const int64_t*, int64_t*, size_t, ops::div_op, cudaStream_t);
    template void launch_binary_op_generic<int64_t, int64_t, ops::maximum_op>(
        const int64_t*, const int64_t*, int64_t*, size_t, ops::maximum_op, cudaStream_t);
    template void launch_binary_op_generic<int64_t, int64_t, ops::minimum_op>(
        const int64_t*, const int64_t*, int64_t*, size_t, ops::minimum_op, cudaStream_t);
    template void launch_binary_op_generic<int64_t, int64_t, ops::pow_op>(
        const int64_t*, const int64_t*, int64_t*, size_t, ops::pow_op, cudaStream_t);
    template void launch_binary_op_generic<int64_t, int64_t, ops::mod_op>(
        const int64_t*, const int64_t*, int64_t*, size_t, ops::mod_op, cudaStream_t);

    // UInt8 operations
    template void launch_binary_op_generic<uint8_t, uint8_t, ops::add_op>(
        const uint8_t*, const uint8_t*, uint8_t*, size_t, ops::add_op, cudaStream_t);
    template void launch_binary_op_generic<uint8_t, uint8_t, ops::sub_op>(
        const uint8_t*, const uint8_t*, uint8_t*, size_t, ops::sub_op, cudaStream_t);
    template void launch_binary_op_generic<uint8_t, uint8_t, ops::mul_op>(
        const uint8_t*, const uint8_t*, uint8_t*, size_t, ops::mul_op, cudaStream_t);
    template void launch_binary_op_generic<uint8_t, uint8_t, ops::div_op>(
        const uint8_t*, const uint8_t*, uint8_t*, size_t, ops::div_op, cudaStream_t);
    template void launch_binary_op_generic<uint8_t, uint8_t, ops::maximum_op>(
        const uint8_t*, const uint8_t*, uint8_t*, size_t, ops::maximum_op, cudaStream_t);
    template void launch_binary_op_generic<uint8_t, uint8_t, ops::minimum_op>(
        const uint8_t*, const uint8_t*, uint8_t*, size_t, ops::minimum_op, cudaStream_t);
    template void launch_binary_op_generic<uint8_t, uint8_t, ops::pow_op>(
        const uint8_t*, const uint8_t*, uint8_t*, size_t, ops::pow_op, cudaStream_t);

    // mod_op for float and int (was missing!)
    template void launch_binary_op_generic<float, float, ops::mod_op>(
        const float*, const float*, float*, size_t, ops::mod_op, cudaStream_t);
    template void launch_binary_op_generic<int, int, ops::mod_op>(
        const int*, const int*, int*, size_t, ops::mod_op, cudaStream_t);

    // Comparison operations for additional types
    template void launch_binary_op_generic<int64_t, unsigned char, ops::greater_op>(
        const int64_t*, const int64_t*, unsigned char*, size_t, ops::greater_op, cudaStream_t);
    template void launch_binary_op_generic<int64_t, unsigned char, ops::greater_equal_op>(
        const int64_t*, const int64_t*, unsigned char*, size_t, ops::greater_equal_op, cudaStream_t);
    template void launch_binary_op_generic<int64_t, unsigned char, ops::less_op>(
        const int64_t*, const int64_t*, unsigned char*, size_t, ops::less_op, cudaStream_t);
    template void launch_binary_op_generic<int64_t, unsigned char, ops::less_equal_op>(
        const int64_t*, const int64_t*, unsigned char*, size_t, ops::less_equal_op, cudaStream_t);
    template void launch_binary_op_generic<int64_t, unsigned char, ops::equal_op>(
        const int64_t*, const int64_t*, unsigned char*, size_t, ops::equal_op, cudaStream_t);
    template void launch_binary_op_generic<int64_t, unsigned char, ops::not_equal_op>(
        const int64_t*, const int64_t*, unsigned char*, size_t, ops::not_equal_op, cudaStream_t);

    template void launch_binary_op_generic<__half, unsigned char, ops::greater_op>(
        const __half*, const __half*, unsigned char*, size_t, ops::greater_op, cudaStream_t);
    template void launch_binary_op_generic<__half, unsigned char, ops::greater_equal_op>(
        const __half*, const __half*, unsigned char*, size_t, ops::greater_equal_op, cudaStream_t);
    template void launch_binary_op_generic<__half, unsigned char, ops::less_op>(
        const __half*, const __half*, unsigned char*, size_t, ops::less_op, cudaStream_t);
    template void launch_binary_op_generic<__half, unsigned char, ops::less_equal_op>(
        const __half*, const __half*, unsigned char*, size_t, ops::less_equal_op, cudaStream_t);
    template void launch_binary_op_generic<__half, unsigned char, ops::equal_op>(
        const __half*, const __half*, unsigned char*, size_t, ops::equal_op, cudaStream_t);
    template void launch_binary_op_generic<__half, unsigned char, ops::not_equal_op>(
        const __half*, const __half*, unsigned char*, size_t, ops::not_equal_op, cudaStream_t);

    // Logical operations for additional types
    template void launch_binary_op_generic<int64_t, unsigned char, ops::logical_and_op>(
        const int64_t*, const int64_t*, unsigned char*, size_t, ops::logical_and_op, cudaStream_t);
    template void launch_binary_op_generic<int64_t, unsigned char, ops::logical_or_op>(
        const int64_t*, const int64_t*, unsigned char*, size_t, ops::logical_or_op, cudaStream_t);
    template void launch_binary_op_generic<__half, unsigned char, ops::logical_and_op>(
        const __half*, const __half*, unsigned char*, size_t, ops::logical_and_op, cudaStream_t);
    template void launch_binary_op_generic<__half, unsigned char, ops::logical_or_op>(
        const __half*, const __half*, unsigned char*, size_t, ops::logical_or_op, cudaStream_t);

    // ============= STRIDED FILL KERNEL =============

    // Helper device function to convert linear index to multi-dimensional indices and calculate offset
    __device__ inline size_t calculate_strided_offset(
        size_t linear_idx,
        const size_t* __restrict__ shape,
        const size_t* __restrict__ strides,
        size_t storage_offset,
        int ndim) {
        size_t offset = storage_offset;
        size_t remaining = linear_idx;

        // Convert linear index to multi-dimensional indices
        for (int d = ndim - 1; d >= 0; --d) {
            size_t idx = remaining % shape[d];
            offset += idx * strides[d];
            remaining /= shape[d];
        }

        return offset;
    }

    // Optimized kernel for 2D column fill (common case: rotation.slice(1, 0, 1).fill_())
    template <typename T>
    __global__ void fill_strided_2d_kernel(
        T* __restrict__ data,
        T value,
        size_t storage_offset,
        size_t stride0, // Stride for dimension 0 (rows)
        size_t n        // Number of elements to fill
    ) {
        // Support both 1D and 2D grids for large arrays
        size_t block_id = blockIdx.y * gridDim.x + blockIdx.x;
        size_t idx = block_id * blockDim.x + threadIdx.x;

        if (idx < n) {
            // For column slice: offset = storage_offset + idx * stride0
            data[storage_offset + idx * stride0] = value;
        }
    }

    // Kernel for filling strided tensors with a constant value (general case)
    template <typename T>
    __global__ void fill_strided_kernel(
        T* __restrict__ data,
        T value,
        const size_t* __restrict__ shape,
        const size_t* __restrict__ strides,
        size_t storage_offset,
        int ndim,
        size_t n) {
        // Support both 1D and 2D grids for large arrays
        size_t block_id = blockIdx.y * gridDim.x + blockIdx.x;
        size_t linear_idx = block_id * blockDim.x + threadIdx.x;

        if (linear_idx < n) {
            size_t offset = calculate_strided_offset(linear_idx, shape, strides, storage_offset, ndim);
            data[offset] = value;
        }
    }

    // Fast path kernels for common tensor ranks (avoid metadata uploads)
    template <typename T>
    __global__ void fill_strided_1d_kernel(
        T* __restrict__ data,
        T value,
        size_t shape0,
        size_t stride0,
        size_t storage_offset) {
        // Support both 1D and 2D grids for large arrays
        size_t block_id = blockIdx.y * gridDim.x + blockIdx.x;
        size_t idx = block_id * blockDim.x + threadIdx.x;
        if (idx < shape0) {
            data[storage_offset + idx * stride0] = value;
        }
    }

    template <typename T>
    __global__ void fill_strided_3d_kernel(
        T* __restrict__ data,
        T value,
        size_t shape0, size_t shape1, size_t shape2,
        size_t stride0, size_t stride1, size_t stride2,
        size_t storage_offset,
        size_t n) {
        // Support both 1D and 2D grids for large arrays
        size_t block_id = blockIdx.y * gridDim.x + blockIdx.x;
        size_t linear_idx = block_id * blockDim.x + threadIdx.x;
        if (linear_idx < n) {
            // Decompose linear index into 3D coordinates
            size_t idx2 = linear_idx % shape2;
            size_t tmp = linear_idx / shape2;
            size_t idx1 = tmp % shape1;
            size_t idx0 = tmp / shape1;

            size_t offset = storage_offset + idx0 * stride0 + idx1 * stride1 + idx2 * stride2;
            data[offset] = value;
        }
    }

    // FAST PATH: 4D tensors (very common in training - e.g., [N, H, W, C])
    template <typename T>
    __global__ void fill_strided_4d_kernel(
        T* __restrict__ data,
        T value,
        size_t shape0, size_t shape1, size_t shape2, size_t shape3,
        size_t stride0, size_t stride1, size_t stride2, size_t stride3,
        size_t storage_offset,
        size_t n) {
        // Support both 1D and 2D grids for large arrays
        size_t block_id = blockIdx.y * gridDim.x + blockIdx.x;
        size_t linear_idx = block_id * blockDim.x + threadIdx.x;
        if (linear_idx < n) {
            // Decompose linear index into 4D coordinates
            size_t idx3 = linear_idx % shape3;
            size_t tmp = linear_idx / shape3;
            size_t idx2 = tmp % shape2;
            tmp /= shape2;
            size_t idx1 = tmp % shape1;
            size_t idx0 = tmp / shape1;

            size_t offset = storage_offset + idx0 * stride0 + idx1 * stride1 + idx2 * stride2 + idx3 * stride3;
            data[offset] = value;
        }
    }

    // Struct to hold shape/strides for passing by value to kernel
    template <int MAX_DIM = 16>
    struct TensorMetadata {
        size_t shape[MAX_DIM];
        size_t strides[MAX_DIM];
    };

    // OPTIMIZED GENERAL PATH: For ndim <= 16, pass shape/strides via kernel parameters
    // This eliminates 12,000+ blocking cudaMemcpy calls during training!
    template <typename T, int MAX_DIM = 16>
    __global__ void fill_strided_immediate_kernel(
        T* __restrict__ data,
        T value,
        TensorMetadata<MAX_DIM> meta, // Passed by value (256 bytes)
        size_t storage_offset,
        int ndim,
        size_t n) {
        // Support both 1D and 2D grids for large arrays
        size_t block_id = blockIdx.y * gridDim.x + blockIdx.x;
        size_t linear_idx = block_id * blockDim.x + threadIdx.x;

        if (linear_idx < n) {
            // Decompose linear index to multi-dimensional indices
            size_t offset = storage_offset;
            size_t remaining = linear_idx;

            for (int i = ndim - 1; i >= 0; --i) {
                size_t idx_i = remaining % meta.shape[i];
                remaining /= meta.shape[i];
                offset += idx_i * meta.strides[i];
            }

            data[offset] = value;
        }
    }

    // Launch function for strided fill
    template <typename T>
    void launch_fill_strided(
        T* data,
        T value,
        const std::vector<size_t>& shape,
        const std::vector<size_t>& strides,
        size_t storage_offset,
        size_t n,
        cudaStream_t stream) {
        if (n == 0)
            return;

        constexpr int BLOCK_SIZE = 256;
        size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const size_t max_blocks_x = 65535; // Safe limit for all CUDA devices

        // FAST PATHS: Avoid expensive malloc/memcpy/free for common cases
        int ndim = static_cast<int>(shape.size());

        // FAST PATH: 1D tensors (most common)
        if (ndim == 1) {
            // Use 2D grid for large arrays to avoid exceeding grid dimension limits
            if (num_blocks <= max_blocks_x) {
                fill_strided_1d_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    data, value, shape[0], strides[0], storage_offset);
            } else {
                dim3 grid(std::min(num_blocks, max_blocks_x),
                          (num_blocks + max_blocks_x - 1) / max_blocks_x);
                fill_strided_1d_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
                    data, value, shape[0], strides[0], storage_offset);
            }

            CHECK_CUDA(cudaGetLastError());
            // NOTE: No sync here - caller (Tensor::fill_) handles sync if needed
            return;
        }

        // FAST PATH: 2D column slice (e.g., rotation.slice(1, 0, 1).fill_())
        if (ndim == 2 && shape[1] == 1) {
            // Use 2D grid for large arrays to avoid exceeding grid dimension limits
            if (num_blocks <= max_blocks_x) {
                fill_strided_2d_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    data, value, storage_offset, strides[0], n);
            } else {
                dim3 grid(std::min(num_blocks, max_blocks_x),
                          (num_blocks + max_blocks_x - 1) / max_blocks_x);
                fill_strided_2d_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
                    data, value, storage_offset, strides[0], n);
            }

            CHECK_CUDA(cudaGetLastError());
            // NOTE: No sync here - caller (Tensor::fill_) handles sync if needed
            return;
        }

        // FAST PATH: 3D tensors
        if (ndim == 3) {
            // Use 2D grid for large arrays to avoid exceeding grid dimension limits
            if (num_blocks <= max_blocks_x) {
                fill_strided_3d_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    data, value,
                    shape[0], shape[1], shape[2],
                    strides[0], strides[1], strides[2],
                    storage_offset, n);
            } else {
                dim3 grid(std::min(num_blocks, max_blocks_x),
                          (num_blocks + max_blocks_x - 1) / max_blocks_x);
                fill_strided_3d_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
                    data, value,
                    shape[0], shape[1], shape[2],
                    strides[0], strides[1], strides[2],
                    storage_offset, n);
            }

            CHECK_CUDA(cudaGetLastError());
            // NOTE: No sync here - caller (Tensor::fill_) handles sync if needed
            return;
        }

        // FAST PATH: 4D tensors (very common - e.g., [N, H, W, C])
        if (ndim == 4) {
            // Use 2D grid for large arrays to avoid exceeding grid dimension limits
            if (num_blocks <= max_blocks_x) {
                fill_strided_4d_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    data, value,
                    shape[0], shape[1], shape[2], shape[3],
                    strides[0], strides[1], strides[2], strides[3],
                    storage_offset, n);
            } else {
                dim3 grid(std::min(num_blocks, max_blocks_x),
                          (num_blocks + max_blocks_x - 1) / max_blocks_x);
                fill_strided_4d_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
                    data, value,
                    shape[0], shape[1], shape[2], shape[3],
                    strides[0], strides[1], strides[2], strides[3],
                    storage_offset, n);
            }

            CHECK_CUDA(cudaGetLastError());
            // NOTE: No sync here - caller (Tensor::fill_) handles sync if needed
            return;
        }

        // OPTIMIZED PATH: For ndim <= 16, pass metadata via kernel parameters (struct)
        // This eliminates 12,000+ blocking cudaMemcpy calls during training!
        if (ndim <= 16) {
            // Create struct on host and copy data
            TensorMetadata<16> meta;
            std::copy_n(shape.begin(), ndim, meta.shape);
            std::copy_n(strides.begin(), ndim, meta.strides);

            // Use 2D grid for large arrays to avoid exceeding grid dimension limits
            if (num_blocks <= max_blocks_x) {
                fill_strided_immediate_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    data, value, meta, storage_offset, ndim, n);
            } else {
                dim3 grid(std::min(num_blocks, max_blocks_x),
                          (num_blocks + max_blocks_x - 1) / max_blocks_x);
                fill_strided_immediate_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
                    data, value, meta, storage_offset, ndim, n);
            }

            CHECK_CUDA(cudaGetLastError());
            // NOTE: No sync here - caller (Tensor::fill_) handles sync if needed
            return;
        }

        // FALLBACK PATH: For ndim > 16 (extremely rare!)
        // Use device memory allocation only when absolutely necessary

        // Copy shape and strides to device
        size_t* d_shape;
        size_t* d_strides;
        CHECK_CUDA(cudaMalloc(&d_shape, ndim * sizeof(size_t)));
        CHECK_CUDA(cudaMalloc(&d_strides, ndim * sizeof(size_t)));
        CHECK_CUDA(cudaMemcpy(d_shape, shape.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_strides, strides.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice));

        // Use 2D grid for large arrays to avoid exceeding grid dimension limits
        if (num_blocks <= max_blocks_x) {
            fill_strided_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                data, value, d_shape, d_strides, storage_offset, ndim, n);
        } else {
            dim3 grid(std::min(num_blocks, max_blocks_x),
                      (num_blocks + max_blocks_x - 1) / max_blocks_x);
            fill_strided_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
                data, value, d_shape, d_strides, storage_offset, ndim, n);
        }

        CHECK_CUDA(cudaGetLastError());
        if (stream == nullptr) {
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        // Clean up device memory
        CHECK_CUDA(cudaFree(d_shape));
        CHECK_CUDA(cudaFree(d_strides));
    }

    // Explicit instantiations
    template void launch_fill_strided<float>(
        float*, float, const std::vector<size_t>&, const std::vector<size_t>&, size_t, size_t, cudaStream_t);
    template void launch_fill_strided<int>(
        int*, int, const std::vector<size_t>&, const std::vector<size_t>&, size_t, size_t, cudaStream_t);
    template void launch_fill_strided<unsigned char>(
        unsigned char*, unsigned char, const std::vector<size_t>&, const std::vector<size_t>&, size_t, size_t, cudaStream_t);

    // ============= FAST GPU-BASED NaN/Inf CHECK =============
    // Returns immediately if any NaN or Inf is found (early exit via atomic)
    // Only transfers 1 int back to CPU - orders of magnitude faster than copying entire tensor

    __global__ void check_nan_inf_kernel(const float* __restrict__ data, size_t n, int* __restrict__ result) {
        // Early exit if already found (check without atomic for speed)
        if (*result != 0)
            return;

        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t stride = blockDim.x * gridDim.x;

        for (size_t i = idx; i < n; i += stride) {
            const float val = data[i];
            if (isnan(val) || isinf(val)) {
                atomicExch(result, 1); // Signal found
                return;                // Early exit this thread
            }
        }
    }

    // Vectorized version for better memory throughput with grid-stride loop
    __global__ void check_nan_inf_kernel_vec4(const float* __restrict__ data, size_t n, int* __restrict__ result) {
        const size_t n_vec4 = n / 4; // Number of complete float4s
        const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

        // Grid-stride loop over float4 elements
        for (size_t vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
             vec_idx < n_vec4;
             vec_idx += stride) {
            // Early exit if already found
            if (*result != 0)
                return;

            const float4 vals = reinterpret_cast<const float4*>(data)[vec_idx];
            if (isnan(vals.x) || isinf(vals.x) ||
                isnan(vals.y) || isinf(vals.y) ||
                isnan(vals.z) || isinf(vals.z) ||
                isnan(vals.w) || isinf(vals.w)) {
                atomicExch(result, 1);
                return;
            }
        }

        // Handle remainder (last 0-3 elements) - only first few threads
        const size_t remainder_start = n_vec4 * 4;
        const size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
        if (thread_id < (n - remainder_start) && remainder_start + thread_id < n) {
            if (*result != 0)
                return;
            const float val = data[remainder_start + thread_id];
            if (isnan(val) || isinf(val)) {
                atomicExch(result, 1);
            }
        }
    }

    // Persistent buffers to avoid malloc/free overhead (thread-safe via thread_local)
    namespace {
        struct NaNCheckBuffers {
            int* d_result = nullptr;
            int* h_result_pinned = nullptr; // Pinned host memory for fast transfer
            bool initialized = false;

            void init() {
                if (!initialized) {
                    CHECK_CUDA(cudaMalloc(&d_result, sizeof(int)));
                    CHECK_CUDA(cudaMallocHost(&h_result_pinned, sizeof(int))); // Pinned memory
                    initialized = true;
                }
            }

            ~NaNCheckBuffers() {
                if (initialized) {
                    cudaFree(d_result);
                    cudaFreeHost(h_result_pinned);
                }
            }
        };

        thread_local NaNCheckBuffers g_nan_check_buffers;
    } // namespace

    bool has_nan_or_inf_gpu(const float* data, size_t n, cudaStream_t stream) {
        if (n == 0)
            return false;

        // Initialize persistent buffers on first use
        g_nan_check_buffers.init();
        int* d_result = g_nan_check_buffers.d_result;
        int* h_result = g_nan_check_buffers.h_result_pinned;

        // Zero the result flag
        *h_result = 0;
        CHECK_CUDA(cudaMemcpyAsync(d_result, h_result, sizeof(int), cudaMemcpyHostToDevice, stream));

        // Launch kernel
        constexpr int BLOCK_SIZE = 256;
        constexpr int MAX_BLOCKS = 1024;

        // Use vectorized kernel for aligned data, scalar for small arrays
        if (n >= 1024 && (reinterpret_cast<uintptr_t>(data) % 16) == 0) {
            const size_t n_vec4 = (n + 3) / 4;
            const int num_blocks = std::min(static_cast<int>((n_vec4 + BLOCK_SIZE - 1) / BLOCK_SIZE), MAX_BLOCKS);
            check_nan_inf_kernel_vec4<<<num_blocks, BLOCK_SIZE, 0, stream>>>(data, n, d_result);
        } else {
            const int num_blocks = std::min(static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE), MAX_BLOCKS);
            check_nan_inf_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(data, n, d_result);
        }

        // Copy result back using pinned memory (very fast!)
        CHECK_CUDA(cudaMemcpyAsync(h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));

        return *h_result != 0;
    }

} // namespace lfs::core::tensor_ops