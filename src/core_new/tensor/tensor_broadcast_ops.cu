/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "internal/memory_pool.hpp"
#include "internal/tensor_functors.hpp"
#include "internal/tensor_ops.hpp"
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace lfs::core::tensor_ops {

    // Note: run_with_thrust_policy is now in include/core/tensor_generic_ops.cuh

    // ============================================================================
    // BROADCASTING INDEX FUNCTOR (for single-array broadcast)
    // ============================================================================

    template <int MaxRank = 8>
    struct broadcast_index_functor {
        int src_rank, dst_rank;
        int src_shape[MaxRank];
        int dst_shape[MaxRank];
        int src_strides[MaxRank];
        int dst_strides[MaxRank];

        broadcast_index_functor(const std::vector<size_t>& src_shape_vec,
                                const std::vector<size_t>& dst_shape_vec)
            : src_rank(src_shape_vec.size()),
              dst_rank(dst_shape_vec.size()) {

            for (int i = 0; i < src_rank; ++i) {
                src_shape[i] = static_cast<int>(src_shape_vec[i]);
            }
            for (int i = 0; i < dst_rank; ++i) {
                dst_shape[i] = static_cast<int>(dst_shape_vec[i]);
            }

            // Compute row-major strides
            if (src_rank > 0) {
                src_strides[src_rank - 1] = 1;
                for (int i = src_rank - 2; i >= 0; --i) {
                    src_strides[i] = src_strides[i + 1] * src_shape[i + 1];
                }
            }

            if (dst_rank > 0) {
                dst_strides[dst_rank - 1] = 1;
                for (int i = dst_rank - 2; i >= 0; --i) {
                    dst_strides[i] = dst_strides[i + 1] * dst_shape[i + 1];
                }
            }
        }

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

    // ============================================================================
    // SINGLE-ARRAY BROADCASTING (Generic) - NOT used by binary ops
    // ============================================================================

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

        broadcast_index_functor<> index_mapper(src_vec, dst_vec);

        auto counting = thrust::make_counting_iterator<size_t>(0);
        auto src_index_iter = thrust::make_transform_iterator(counting, index_mapper);
        auto permuted_src = thrust::make_permutation_iterator(src_ptr, src_index_iter);

        run_with_thrust_policy(stream, [&](auto policy) {
            thrust::copy(policy, permuted_src, permuted_src + dst_elements, dst_ptr);
        });
    }

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

} // namespace lfs::core::tensor_ops
