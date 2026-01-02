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

    constexpr int MAX_RANK = 8;
    constexpr int BLOCK_SIZE = 256;

    // Strided broadcast kernel params (passed by value, no device alloc)
    struct BroadcastStridedParams {
        size_t src_shape[MAX_RANK];
        size_t src_strides[MAX_RANK];
        size_t dst_strides[MAX_RANK];
        int src_rank;
        int dst_rank;
        size_t dst_elements;
    };

    template <typename T>
    __global__ void broadcast_strided_kernel(const T* __restrict__ src, T* __restrict__ dst,
                                             const BroadcastStridedParams params) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= params.dst_elements) return;

        size_t remaining = idx;
        size_t src_idx = 0;
        const int offset = params.dst_rank - params.src_rank;

        for (int i = 0; i < params.dst_rank; ++i) {
            const size_t coord = remaining / params.dst_strides[i];
            remaining %= params.dst_strides[i];

            if (i >= offset) {
                const int src_dim = i - offset;
                const size_t src_coord = (params.src_shape[src_dim] == 1) ? 0 : coord;
                src_idx += src_coord * params.src_strides[src_dim];
            }
        }
        dst[idx] = src[src_idx];
    }

    template <typename T>
    void launch_broadcast_strided_impl(const T* src, T* dst,
                                       const size_t* src_shape, const size_t* src_strides,
                                       const size_t* dst_shape,
                                       const size_t src_rank, const size_t dst_rank,
                                       const size_t dst_elements, cudaStream_t stream) {
        if (dst_elements == 0) return;
        if (src_rank > MAX_RANK || dst_rank > MAX_RANK) {
            LOG_ERROR("Broadcast strided: rank exceeds {}", MAX_RANK);
            return;
        }

        BroadcastStridedParams params{};
        params.src_rank = static_cast<int>(src_rank);
        params.dst_rank = static_cast<int>(dst_rank);
        params.dst_elements = dst_elements;

        for (size_t i = 0; i < src_rank; ++i) {
            params.src_shape[i] = src_shape[i];
            params.src_strides[i] = src_strides[i];
        }

        if (dst_rank > 0) {
            params.dst_strides[dst_rank - 1] = 1;
            for (int i = static_cast<int>(dst_rank) - 2; i >= 0; --i) {
                params.dst_strides[i] = params.dst_strides[i + 1] * dst_shape[i + 1];
            }
        }

        const int num_blocks = (dst_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        broadcast_strided_kernel<T><<<num_blocks, BLOCK_SIZE, 0, stream>>>(src, dst, params);
    }

    void launch_broadcast_strided(const float* src, float* dst,
                                  const size_t* src_shape, const size_t* src_strides,
                                  const size_t* dst_shape,
                                  const size_t src_rank, const size_t dst_rank,
                                  const size_t dst_elements, cudaStream_t stream) {
        launch_broadcast_strided_impl(src, dst, src_shape, src_strides, dst_shape,
                                      src_rank, dst_rank, dst_elements, stream);
    }

    void launch_broadcast_strided_bool(const unsigned char* src, unsigned char* dst,
                                       const size_t* src_shape, const size_t* src_strides,
                                       const size_t* dst_shape,
                                       const size_t src_rank, const size_t dst_rank,
                                       const size_t dst_elements, cudaStream_t stream) {
        launch_broadcast_strided_impl(src, dst, src_shape, src_strides, dst_shape,
                                      src_rank, dst_rank, dst_elements, stream);
    }

    // Pad kernel params (passed by value, no device alloc)
    struct PadParams {
        size_t src_shape[MAX_RANK];
        size_t src_strides[MAX_RANK];
        size_t dst_strides[MAX_RANK];
        size_t pad_before[MAX_RANK];
        size_t contiguous_strides[MAX_RANK];
        int rank;
        size_t src_elements;
    };

    __global__ void pad_kernel(const float* __restrict__ src, float* __restrict__ dst,
                               const PadParams params) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= params.src_elements) return;

        size_t coords[MAX_RANK];
        size_t remaining = idx;

        for (int i = 0; i < params.rank; ++i) {
            coords[i] = remaining / params.contiguous_strides[i];
            remaining %= params.contiguous_strides[i];
        }

        size_t src_offset = 0;
        size_t dst_offset = 0;
        for (int i = 0; i < params.rank; ++i) {
            src_offset += coords[i] * params.src_strides[i];
            dst_offset += (coords[i] + params.pad_before[i]) * params.dst_strides[i];
        }
        dst[dst_offset] = src[src_offset];
    }

    void launch_pad(const float* src, float* dst,
                    const size_t* src_shape, const size_t* src_strides,
                    const size_t* dst_shape, const size_t* pad_before,
                    const size_t rank, const size_t src_elements, cudaStream_t stream) {
        if (src_elements == 0) return;
        if (rank > MAX_RANK) {
            LOG_ERROR("Pad: rank exceeds {}", MAX_RANK);
            return;
        }

        PadParams params{};
        params.rank = static_cast<int>(rank);
        params.src_elements = src_elements;

        for (size_t i = 0; i < rank; ++i) {
            params.src_shape[i] = src_shape[i];
            params.src_strides[i] = src_strides[i];
            params.pad_before[i] = pad_before[i];
        }

        if (rank > 0) {
            params.contiguous_strides[rank - 1] = 1;
            params.dst_strides[rank - 1] = 1;
            for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
                params.contiguous_strides[i] = params.contiguous_strides[i + 1] * src_shape[i + 1];
                params.dst_strides[i] = params.dst_strides[i + 1] * dst_shape[i + 1];
            }
        }

        const int num_blocks = (src_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        pad_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(src, dst, params);
    }

    // Broadcasting index functor

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

    // ============================================================================
    // Type Promotion Broadcast Instantiations
    // ============================================================================
    // Added to support the type promotion system for mixed-dtype operations
    // with broadcasting.
    // ============================================================================

    // Float16 broadcast operations
    template void launch_broadcast_binary<__half, __half, ops::add_op>(
        const __half*, const __half*, __half*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::add_op, cudaStream_t);
    template void launch_broadcast_binary<__half, __half, ops::sub_op>(
        const __half*, const __half*, __half*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::sub_op, cudaStream_t);
    template void launch_broadcast_binary<__half, __half, ops::mul_op>(
        const __half*, const __half*, __half*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::mul_op, cudaStream_t);
    template void launch_broadcast_binary<__half, __half, ops::div_op>(
        const __half*, const __half*, __half*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::div_op, cudaStream_t);
    template void launch_broadcast_binary<__half, __half, ops::maximum_op>(
        const __half*, const __half*, __half*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::maximum_op, cudaStream_t);
    template void launch_broadcast_binary<__half, __half, ops::minimum_op>(
        const __half*, const __half*, __half*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::minimum_op, cudaStream_t);
    template void launch_broadcast_binary<__half, __half, ops::pow_op>(
        const __half*, const __half*, __half*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::pow_op, cudaStream_t);

    // Int64 broadcast operations
    template void launch_broadcast_binary<int64_t, int64_t, ops::add_op>(
        const int64_t*, const int64_t*, int64_t*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::add_op, cudaStream_t);
    template void launch_broadcast_binary<int64_t, int64_t, ops::sub_op>(
        const int64_t*, const int64_t*, int64_t*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::sub_op, cudaStream_t);
    template void launch_broadcast_binary<int64_t, int64_t, ops::mul_op>(
        const int64_t*, const int64_t*, int64_t*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::mul_op, cudaStream_t);
    template void launch_broadcast_binary<int64_t, int64_t, ops::div_op>(
        const int64_t*, const int64_t*, int64_t*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::div_op, cudaStream_t);
    template void launch_broadcast_binary<int64_t, int64_t, ops::maximum_op>(
        const int64_t*, const int64_t*, int64_t*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::maximum_op, cudaStream_t);
    template void launch_broadcast_binary<int64_t, int64_t, ops::minimum_op>(
        const int64_t*, const int64_t*, int64_t*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::minimum_op, cudaStream_t);
    template void launch_broadcast_binary<int64_t, int64_t, ops::pow_op>(
        const int64_t*, const int64_t*, int64_t*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::pow_op, cudaStream_t);
    template void launch_broadcast_binary<int64_t, int64_t, ops::mod_op>(
        const int64_t*, const int64_t*, int64_t*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::mod_op, cudaStream_t);

    // UInt8 broadcast operations
    template void launch_broadcast_binary<uint8_t, uint8_t, ops::add_op>(
        const uint8_t*, const uint8_t*, uint8_t*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::add_op, cudaStream_t);
    template void launch_broadcast_binary<uint8_t, uint8_t, ops::sub_op>(
        const uint8_t*, const uint8_t*, uint8_t*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::sub_op, cudaStream_t);
    template void launch_broadcast_binary<uint8_t, uint8_t, ops::mul_op>(
        const uint8_t*, const uint8_t*, uint8_t*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::mul_op, cudaStream_t);
    template void launch_broadcast_binary<uint8_t, uint8_t, ops::div_op>(
        const uint8_t*, const uint8_t*, uint8_t*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::div_op, cudaStream_t);
    template void launch_broadcast_binary<uint8_t, uint8_t, ops::maximum_op>(
        const uint8_t*, const uint8_t*, uint8_t*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::maximum_op, cudaStream_t);
    template void launch_broadcast_binary<uint8_t, uint8_t, ops::minimum_op>(
        const uint8_t*, const uint8_t*, uint8_t*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::minimum_op, cudaStream_t);
    template void launch_broadcast_binary<uint8_t, uint8_t, ops::pow_op>(
        const uint8_t*, const uint8_t*, uint8_t*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::pow_op, cudaStream_t);

    // mod_op broadcast (was missing!)
    template void launch_broadcast_binary<float, float, ops::mod_op>(
        const float*, const float*, float*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::mod_op, cudaStream_t);
    template void launch_broadcast_binary<int, int, ops::mod_op>(
        const int*, const int*, int*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::mod_op, cudaStream_t);

    // Comparison operations for additional types
    template void launch_broadcast_binary<int64_t, unsigned char, ops::greater_op>(
        const int64_t*, const int64_t*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::greater_op, cudaStream_t);
    template void launch_broadcast_binary<int64_t, unsigned char, ops::greater_equal_op>(
        const int64_t*, const int64_t*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::greater_equal_op, cudaStream_t);
    template void launch_broadcast_binary<int64_t, unsigned char, ops::less_op>(
        const int64_t*, const int64_t*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::less_op, cudaStream_t);
    template void launch_broadcast_binary<int64_t, unsigned char, ops::less_equal_op>(
        const int64_t*, const int64_t*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::less_equal_op, cudaStream_t);
    template void launch_broadcast_binary<int64_t, unsigned char, ops::equal_op>(
        const int64_t*, const int64_t*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::equal_op, cudaStream_t);
    template void launch_broadcast_binary<int64_t, unsigned char, ops::not_equal_op>(
        const int64_t*, const int64_t*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::not_equal_op, cudaStream_t);

    template void launch_broadcast_binary<__half, unsigned char, ops::greater_op>(
        const __half*, const __half*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::greater_op, cudaStream_t);
    template void launch_broadcast_binary<__half, unsigned char, ops::greater_equal_op>(
        const __half*, const __half*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::greater_equal_op, cudaStream_t);
    template void launch_broadcast_binary<__half, unsigned char, ops::less_op>(
        const __half*, const __half*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::less_op, cudaStream_t);
    template void launch_broadcast_binary<__half, unsigned char, ops::less_equal_op>(
        const __half*, const __half*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::less_equal_op, cudaStream_t);
    template void launch_broadcast_binary<__half, unsigned char, ops::equal_op>(
        const __half*, const __half*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::equal_op, cudaStream_t);
    template void launch_broadcast_binary<__half, unsigned char, ops::not_equal_op>(
        const __half*, const __half*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::not_equal_op, cudaStream_t);

    // Logical operations for additional types
    template void launch_broadcast_binary<int64_t, unsigned char, ops::logical_and_op>(
        const int64_t*, const int64_t*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::logical_and_op, cudaStream_t);
    template void launch_broadcast_binary<int64_t, unsigned char, ops::logical_or_op>(
        const int64_t*, const int64_t*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::logical_or_op, cudaStream_t);
    template void launch_broadcast_binary<__half, unsigned char, ops::logical_and_op>(
        const __half*, const __half*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::logical_and_op, cudaStream_t);
    template void launch_broadcast_binary<__half, unsigned char, ops::logical_or_op>(
        const __half*, const __half*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::logical_or_op, cudaStream_t);

} // namespace lfs::core::tensor_ops
