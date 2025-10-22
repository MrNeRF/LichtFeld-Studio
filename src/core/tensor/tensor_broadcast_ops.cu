/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "internal/memory_pool.hpp"
#include "internal/tensor_functors.hpp"
#include "internal/tensor_ops.hpp"
#include <cuda_runtime.h>

#ifndef _WIN32
// Thrust headers only needed for non-Windows (Linux uses Thrust iterators)
// Windows uses direct CUDA kernel to avoid nvcc 12.8 ICE with aggregate types
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#endif

namespace gs::tensor_ops {

    // Note: run_with_thrust_policy is now in include/core/tensor_generic_ops.cuh

    // ============================================================================
    // SINGLE-ARRAY BROADCASTING (Generic) - NOT used by binary ops
    // ============================================================================

#ifdef _WIN32
    // Windows nvcc 12.8 workaround: Use direct kernel instead of Thrust transform iterator
    // The Thrust pattern triggers ICE with structs containing arrays
    template <typename T>
    __global__ void broadcast_kernel(const T* __restrict__ src, T* __restrict__ dst,
                                      const int* __restrict__ src_shape,
                                      const int* __restrict__ dst_shape,
                                      const int* __restrict__ src_strides,
                                      const int* __restrict__ dst_strides,
                                      int src_rank, int dst_rank, size_t dst_elements) {
        size_t dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (dst_idx >= dst_elements) return;

        size_t src_idx = 0;
        size_t remaining = dst_idx;

        for (int i = 0; i < dst_rank; ++i) {
            int dst_coord = remaining / dst_strides[i];
            remaining %= dst_strides[i];

            int offset = dst_rank - src_rank;
            if (i >= offset) {
                int src_dim = i - offset;
                int src_coord = (src_shape[src_dim] == 1) ? 0 : dst_coord;
                src_idx += src_coord * src_strides[src_dim];
            }
        }

        dst[dst_idx] = src[src_idx];
    }

    template <typename T>
    void launch_broadcast_generic(const T* src, T* dst,
                                  const size_t* src_shape, const size_t* dst_shape,
                                  size_t src_rank, size_t dst_rank,
                                  size_t dst_elements, cudaStream_t stream) {
        if (dst_elements == 0)
            return;

        // Allocate device memory for shapes and strides
        constexpr int MaxRank = 8;
        int h_src_shape[MaxRank], h_dst_shape[MaxRank];
        int h_src_strides[MaxRank], h_dst_strides[MaxRank];

        // Copy shapes
        for (size_t i = 0; i < src_rank; ++i) {
            h_src_shape[i] = static_cast<int>(src_shape[i]);
        }
        for (size_t i = 0; i < dst_rank; ++i) {
            h_dst_shape[i] = static_cast<int>(dst_shape[i]);
        }

        // Compute strides
        if (src_rank > 0) {
            h_src_strides[src_rank - 1] = 1;
            for (int i = src_rank - 2; i >= 0; --i) {
                h_src_strides[i] = h_src_strides[i + 1] * h_src_shape[i + 1];
            }
        }
        if (dst_rank > 0) {
            h_dst_strides[dst_rank - 1] = 1;
            for (int i = dst_rank - 2; i >= 0; --i) {
                h_dst_strides[i] = h_dst_strides[i + 1] * h_dst_shape[i + 1];
            }
        }

        // Copy to device
        int *d_src_shape, *d_dst_shape, *d_src_strides, *d_dst_strides;
        cudaMalloc(&d_src_shape, MaxRank * sizeof(int));
        cudaMalloc(&d_dst_shape, MaxRank * sizeof(int));
        cudaMalloc(&d_src_strides, MaxRank * sizeof(int));
        cudaMalloc(&d_dst_strides, MaxRank * sizeof(int));

        cudaMemcpyAsync(d_src_shape, h_src_shape, MaxRank * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_dst_shape, h_dst_shape, MaxRank * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_src_strides, h_src_strides, MaxRank * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_dst_strides, h_dst_strides, MaxRank * sizeof(int), cudaMemcpyHostToDevice, stream);

        // Launch kernel
        constexpr int block_size = 256;
        int grid_size = (dst_elements + block_size - 1) / block_size;
        broadcast_kernel<<<grid_size, block_size, 0, stream>>>(
            src, dst, d_src_shape, d_dst_shape, d_src_strides, d_dst_strides,
            static_cast<int>(src_rank), static_cast<int>(dst_rank), dst_elements);

        // Free device memory (sync to avoid use-after-free)
        cudaStreamSynchronize(stream);
        cudaFree(d_src_shape);
        cudaFree(d_dst_shape);
        cudaFree(d_src_strides);
        cudaFree(d_dst_strides);
    }
#else
    // Non-Windows: Use Thrust transform iterator (original code)

    // ============================================================================
    // BROADCASTING INDEX FUNCTOR (for single-array broadcast) - Linux only
    // ============================================================================
    // NOTE: This struct with array members triggers nvcc 12.8 ICE on Windows
    // when passed to Thrust iterators. Keep this in #else block (Linux only).

    template <int MaxRank = 8>
    struct broadcast_index_functor {
        int src_rank, dst_rank;
        int src_shape[MaxRank];
        int dst_shape[MaxRank];
        int src_strides[MaxRank];
        int dst_strides[MaxRank];

        // Default constructor - trivially copyable
        broadcast_index_functor() = default;

        __device__ size_t operator()(size_t dst_linear_idx) const {
            size_t src_idx = 0;
            size_t remaining = dst_linear_idx;

            for (int i = 0; i < dst_rank; ++i) {
                int dst_coord = remaining / dst_strides[i];
                remaining %= dst_strides[i];

                int offset = dst_rank - src_rank;
                if (i >= offset) {
                    int src_dim = i - offset;
                    int src_coord = (src_shape[src_dim] == 1) ? 0 : dst_coord;
                    src_idx += src_coord * src_strides[src_dim];
                }
            }

            return src_idx;
        }
    };

    // Helper function to create and initialize the functor on host
    template <int MaxRank = 8>
    broadcast_index_functor<MaxRank> make_broadcast_functor(
        const std::vector<size_t>& src_shape_vec,
        const std::vector<size_t>& dst_shape_vec) {

        broadcast_index_functor<MaxRank> functor{};
        functor.src_rank = src_shape_vec.size();
        functor.dst_rank = dst_shape_vec.size();

        for (int i = 0; i < functor.src_rank; ++i) {
            functor.src_shape[i] = static_cast<int>(src_shape_vec[i]);
        }
        for (int i = 0; i < functor.dst_rank; ++i) {
            functor.dst_shape[i] = static_cast<int>(dst_shape_vec[i]);
        }

        // Compute row-major strides
        if (functor.src_rank > 0) {
            functor.src_strides[functor.src_rank - 1] = 1;
            for (int i = functor.src_rank - 2; i >= 0; --i) {
                functor.src_strides[i] = functor.src_strides[i + 1] * functor.src_shape[i + 1];
            }
        }

        if (functor.dst_rank > 0) {
            functor.dst_strides[functor.dst_rank - 1] = 1;
            for (int i = functor.dst_rank - 2; i >= 0; --i) {
                functor.dst_strides[i] = functor.dst_strides[i + 1] * functor.dst_shape[i + 1];
            }
        }

        return functor;
    }

    template <typename T>
    void launch_broadcast_generic(const T* src, T* dst,
                                  const size_t* src_shape, const size_t* dst_shape,
                                  size_t src_rank, size_t dst_rank,
                                  size_t dst_elements, cudaStream_t stream) {
        if (dst_elements == 0)
            return;

        std::vector<size_t> src_vec(src_shape, src_shape + src_rank);
        std::vector<size_t> dst_vec(dst_shape, dst_shape + dst_rank);

        auto src_ptr = thrust::device_pointer_cast(src);
        auto dst_ptr = thrust::device_pointer_cast(dst);

        // Use helper function to create functor
        auto index_mapper = make_broadcast_functor(src_vec, dst_vec);

        auto counting = thrust::make_counting_iterator<size_t>(0);
        auto src_index_iter = thrust::make_transform_iterator(counting, index_mapper);
        auto permuted_src = thrust::make_permutation_iterator(src_ptr, src_index_iter);

        run_with_thrust_policy(stream, [&](auto policy) {
            thrust::copy(policy, permuted_src, permuted_src + dst_elements, dst_ptr);
        });
    }
#endif

    void launch_broadcast(const float* src, float* dst,
                          const size_t* src_shape, const size_t* dst_shape,
                          size_t src_rank, size_t dst_rank,
                          size_t dst_elements, cudaStream_t stream) {
        launch_broadcast_generic(src, dst, src_shape, dst_shape, src_rank, dst_rank, dst_elements, stream);
    }

    void launch_broadcast_bool(const unsigned char* src, unsigned char* dst,
                               const size_t* src_shape, const size_t* dst_shape,
                               size_t src_rank, size_t dst_rank,
                               size_t dst_elements, cudaStream_t stream) {
        launch_broadcast_generic(src, dst, src_shape, dst_shape, src_rank, dst_rank, dst_elements, stream);
    }

    // ============================================================================
    // NOTE: launch_broadcast_binary implementation is now in tensor_broadcast_ops.cuh
    // All CUDA kernels and the host function template are defined inline in the header
    // for correct template instantiation with expression template functors.
    // ============================================================================

    // ============================================================================
    // EXPLICIT INSTANTIATIONS FOR C++ FILES
    // C++ files can't see tensor_broadcast_ops.cuh (which is #ifdef __CUDACC__),
    // so we need explicit instantiations for basic binary operations.
    // ============================================================================

    // Arithmetic operations (same input/output type - comprehensive list)
    template void launch_broadcast_binary<float, float, ops::add_op>(
        const float*, const float*, float*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::add_op, cudaStream_t);

    template void launch_broadcast_binary<int, int, ops::add_op>(
        const int*, const int*, int*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::add_op, cudaStream_t);

    template void launch_broadcast_binary<float, float, ops::sub_op>(
        const float*, const float*, float*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::sub_op, cudaStream_t);

    template void launch_broadcast_binary<int, int, ops::sub_op>(
        const int*, const int*, int*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::sub_op, cudaStream_t);

    template void launch_broadcast_binary<float, float, ops::mul_op>(
        const float*, const float*, float*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::mul_op, cudaStream_t);

    template void launch_broadcast_binary<int, int, ops::mul_op>(
        const int*, const int*, int*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::mul_op, cudaStream_t);

    template void launch_broadcast_binary<float, float, ops::div_op>(
        const float*, const float*, float*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::div_op, cudaStream_t);

    template void launch_broadcast_binary<int, int, ops::div_op>(
        const int*, const int*, int*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::div_op, cudaStream_t);

    // Comparison operations (input T -> output unsigned char/bool)
    template void launch_broadcast_binary<float, unsigned char, ops::greater_op>(
        const float*, const float*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::greater_op, cudaStream_t);

    template void launch_broadcast_binary<int, unsigned char, ops::greater_op>(
        const int*, const int*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::greater_op, cudaStream_t);

    template void launch_broadcast_binary<unsigned char, unsigned char, ops::greater_op>(
        const unsigned char*, const unsigned char*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::greater_op, cudaStream_t);

    template void launch_broadcast_binary<float, unsigned char, ops::greater_equal_op>(
        const float*, const float*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::greater_equal_op, cudaStream_t);

    template void launch_broadcast_binary<int, unsigned char, ops::greater_equal_op>(
        const int*, const int*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::greater_equal_op, cudaStream_t);

    template void launch_broadcast_binary<unsigned char, unsigned char, ops::greater_equal_op>(
        const unsigned char*, const unsigned char*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::greater_equal_op, cudaStream_t);

    template void launch_broadcast_binary<float, unsigned char, ops::less_equal_op>(
        const float*, const float*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::less_equal_op, cudaStream_t);

    template void launch_broadcast_binary<int, unsigned char, ops::less_equal_op>(
        const int*, const int*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::less_equal_op, cudaStream_t);

    template void launch_broadcast_binary<unsigned char, unsigned char, ops::less_equal_op>(
        const unsigned char*, const unsigned char*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::less_equal_op, cudaStream_t);

    template void launch_broadcast_binary<float, unsigned char, ops::less_op>(
        const float*, const float*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::less_op, cudaStream_t);

    template void launch_broadcast_binary<int, unsigned char, ops::less_op>(
        const int*, const int*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::less_op, cudaStream_t);

    template void launch_broadcast_binary<unsigned char, unsigned char, ops::less_op>(
        const unsigned char*, const unsigned char*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::less_op, cudaStream_t);

    template void launch_broadcast_binary<float, unsigned char, ops::equal_op>(
        const float*, const float*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::equal_op, cudaStream_t);

    template void launch_broadcast_binary<int, unsigned char, ops::equal_op>(
        const int*, const int*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::equal_op, cudaStream_t);

    template void launch_broadcast_binary<unsigned char, unsigned char, ops::equal_op>(
        const unsigned char*, const unsigned char*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::equal_op, cudaStream_t);

    // Logical operations (bool/unsigned char -> unsigned char)
    template void launch_broadcast_binary<float, unsigned char, ops::logical_and_op>(
        const float*, const float*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::logical_and_op, cudaStream_t);

    template void launch_broadcast_binary<int, unsigned char, ops::logical_and_op>(
        const int*, const int*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::logical_and_op, cudaStream_t);

    template void launch_broadcast_binary<unsigned char, unsigned char, ops::logical_and_op>(
        const unsigned char*, const unsigned char*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::logical_and_op, cudaStream_t);

    template void launch_broadcast_binary<float, unsigned char, ops::logical_or_op>(
        const float*, const float*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::logical_or_op, cudaStream_t);

    template void launch_broadcast_binary<int, unsigned char, ops::logical_or_op>(
        const int*, const int*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::logical_or_op, cudaStream_t);

    template void launch_broadcast_binary<unsigned char, unsigned char, ops::logical_or_op>(
        const unsigned char*, const unsigned char*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::logical_or_op, cudaStream_t);

    // Min/max operations
    template void launch_broadcast_binary<float, float, ops::minimum_op>(
        const float*, const float*, float*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::minimum_op, cudaStream_t);

    template void launch_broadcast_binary<int, int, ops::minimum_op>(
        const int*, const int*, int*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::minimum_op, cudaStream_t);

    template void launch_broadcast_binary<float, float, ops::maximum_op>(
        const float*, const float*, float*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::maximum_op, cudaStream_t);

    template void launch_broadcast_binary<int, int, ops::maximum_op>(
        const int*, const int*, int*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::maximum_op, cudaStream_t);

    // Power operations
    template void launch_broadcast_binary<float, float, ops::pow_op>(
        const float*, const float*, float*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::pow_op, cudaStream_t);

    template void launch_broadcast_binary<int, int, ops::pow_op>(
        const int*, const int*, int*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::pow_op, cudaStream_t);

    // Not equal operation
    template void launch_broadcast_binary<float, unsigned char, ops::not_equal_op>(
        const float*, const float*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::not_equal_op, cudaStream_t);

    template void launch_broadcast_binary<int, unsigned char, ops::not_equal_op>(
        const int*, const int*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::not_equal_op, cudaStream_t);

    template void launch_broadcast_binary<unsigned char, unsigned char, ops::not_equal_op>(
        const unsigned char*, const unsigned char*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::not_equal_op, cudaStream_t);

} // namespace gs::tensor_ops
