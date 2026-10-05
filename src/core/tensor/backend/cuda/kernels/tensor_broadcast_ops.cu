/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/assert.hpp"
#include "core/cuda_error.hpp"
#include "core/tensor/backend/cuda/kernels/tensor_ops.hpp"
#include "core/tensor/backend/cuda/runtime/memory_pool.hpp"
#include "core/tensor_fwd.hpp"
#include "internal/tensor_functors.hpp"
#include <cuda_runtime.h>

namespace lfs::core::tensor_ops {

    constexpr int BLOCK_SIZE = 256;

    namespace {
        struct ieee_broadcast_maximum_float_op {
            __host__ __device__ float operator()(const float lhs, const float rhs) const {
                return ops::maximum_op{}(lhs, rhs);
            }
        };

        struct ieee_broadcast_minimum_float_op {
            __host__ __device__ float operator()(const float lhs, const float rhs) const {
                return ops::minimum_op{}(lhs, rhs);
            }
        };
    } // namespace

    // Pad kernel params (passed by value, no device alloc)
    struct PadParams {
        size_t src_shape[MAX_TENSOR_RANK];
        size_t src_strides[MAX_TENSOR_RANK];
        size_t dst_strides[MAX_TENSOR_RANK];
        size_t pad_before[MAX_TENSOR_RANK];
        size_t contiguous_strides[MAX_TENSOR_RANK];
        int rank;
        size_t src_elements;
    };

    __global__ void pad_kernel(const float* __restrict__ src, float* __restrict__ dst,
                               const PadParams params) {
        const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= params.src_elements)
            return;

        size_t coords[MAX_TENSOR_RANK];
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
        if (src_elements == 0)
            return;
        LFS_ASSERT_MSG(rank <= MAX_TENSOR_RANK,
                       "Pad rank exceeds MAX_TENSOR_RANK");

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
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.broadcast.pad");
    }

    void launch_ieee_maximum_float_broadcast(
        const float* lhs, const float* rhs, float* output,
        const size_t* lhs_shape, const size_t* rhs_shape, const size_t* output_shape,
        const size_t lhs_rank, const size_t rhs_rank, const size_t output_rank,
        const size_t output_elements, const cudaStream_t stream) {
        launch_broadcast_binary(
            lhs, rhs, output, lhs_shape, rhs_shape, output_shape,
            lhs_rank, rhs_rank, output_rank, output_elements,
            ieee_broadcast_maximum_float_op{}, stream);
    }

    void launch_ieee_minimum_float_broadcast(
        const float* lhs, const float* rhs, float* output,
        const size_t* lhs_shape, const size_t* rhs_shape, const size_t* output_shape,
        const size_t lhs_rank, const size_t rhs_rank, const size_t output_rank,
        const size_t output_elements, const cudaStream_t stream) {
        launch_broadcast_binary(
            lhs, rhs, output, lhs_shape, rhs_shape, output_shape,
            lhs_rank, rhs_rank, output_rank, output_elements,
            ieee_broadcast_minimum_float_op{}, stream);
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
    template LFS_CORE_API void launch_broadcast_binary<float, float, ops::add_op>(
        const float*, const float*, float*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::add_op, cudaStream_t);

    template LFS_CORE_API void launch_broadcast_binary<int, int, ops::add_op>(
        const int*, const int*, int*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::add_op, cudaStream_t);

    template LFS_CORE_API void launch_broadcast_binary<float, float, ops::sub_op>(
        const float*, const float*, float*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::sub_op, cudaStream_t);

    template LFS_CORE_API void launch_broadcast_binary<int, int, ops::sub_op>(
        const int*, const int*, int*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::sub_op, cudaStream_t);

    template LFS_CORE_API void launch_broadcast_binary<float, float, ops::mul_op>(
        const float*, const float*, float*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::mul_op, cudaStream_t);

    template LFS_CORE_API void launch_broadcast_binary<int, int, ops::mul_op>(
        const int*, const int*, int*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::mul_op, cudaStream_t);

    template LFS_CORE_API void launch_broadcast_binary<float, float, ops::div_op>(
        const float*, const float*, float*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::div_op, cudaStream_t);

    template LFS_CORE_API void launch_broadcast_binary<int, int, ops::div_op>(
        const int*, const int*, int*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::div_op, cudaStream_t);

    // Comparison operations (input T -> output unsigned char/bool)
    template LFS_CORE_API void launch_broadcast_binary<float, unsigned char, ops::greater_op>(
        const float*, const float*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::greater_op, cudaStream_t);

    template LFS_CORE_API void launch_broadcast_binary<int, unsigned char, ops::greater_op>(
        const int*, const int*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::greater_op, cudaStream_t);

    template LFS_CORE_API void launch_broadcast_binary<unsigned char, unsigned char, ops::greater_op>(
        const unsigned char*, const unsigned char*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::greater_op, cudaStream_t);

    template LFS_CORE_API void launch_broadcast_binary<float, unsigned char, ops::greater_equal_op>(
        const float*, const float*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::greater_equal_op, cudaStream_t);

    template LFS_CORE_API void launch_broadcast_binary<int, unsigned char, ops::greater_equal_op>(
        const int*, const int*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::greater_equal_op, cudaStream_t);

    template LFS_CORE_API void launch_broadcast_binary<unsigned char, unsigned char, ops::greater_equal_op>(
        const unsigned char*, const unsigned char*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::greater_equal_op, cudaStream_t);

    template LFS_CORE_API void launch_broadcast_binary<float, unsigned char, ops::less_equal_op>(
        const float*, const float*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::less_equal_op, cudaStream_t);

    template LFS_CORE_API void launch_broadcast_binary<int, unsigned char, ops::less_equal_op>(
        const int*, const int*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::less_equal_op, cudaStream_t);

    template LFS_CORE_API void launch_broadcast_binary<unsigned char, unsigned char, ops::less_equal_op>(
        const unsigned char*, const unsigned char*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::less_equal_op, cudaStream_t);

    template LFS_CORE_API void launch_broadcast_binary<float, unsigned char, ops::less_op>(
        const float*, const float*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::less_op, cudaStream_t);

    template LFS_CORE_API void launch_broadcast_binary<int, unsigned char, ops::less_op>(
        const int*, const int*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::less_op, cudaStream_t);

    template LFS_CORE_API void launch_broadcast_binary<unsigned char, unsigned char, ops::less_op>(
        const unsigned char*, const unsigned char*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::less_op, cudaStream_t);

    template LFS_CORE_API void launch_broadcast_binary<float, unsigned char, ops::equal_op>(
        const float*, const float*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::equal_op, cudaStream_t);

    template LFS_CORE_API void launch_broadcast_binary<int, unsigned char, ops::equal_op>(
        const int*, const int*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::equal_op, cudaStream_t);

    template LFS_CORE_API void launch_broadcast_binary<unsigned char, unsigned char, ops::equal_op>(
        const unsigned char*, const unsigned char*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::equal_op, cudaStream_t);

    // Logical operations (bool/unsigned char -> unsigned char)
    template LFS_CORE_API void launch_broadcast_binary<float, unsigned char, ops::logical_and_op>(
        const float*, const float*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::logical_and_op, cudaStream_t);

    template LFS_CORE_API void launch_broadcast_binary<int, unsigned char, ops::logical_and_op>(
        const int*, const int*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::logical_and_op, cudaStream_t);

    template LFS_CORE_API void launch_broadcast_binary<unsigned char, unsigned char, ops::logical_and_op>(
        const unsigned char*, const unsigned char*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::logical_and_op, cudaStream_t);

    template LFS_CORE_API void launch_broadcast_binary<float, unsigned char, ops::logical_or_op>(
        const float*, const float*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::logical_or_op, cudaStream_t);

    template LFS_CORE_API void launch_broadcast_binary<int, unsigned char, ops::logical_or_op>(
        const int*, const int*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::logical_or_op, cudaStream_t);

    template LFS_CORE_API void launch_broadcast_binary<unsigned char, unsigned char, ops::logical_or_op>(
        const unsigned char*, const unsigned char*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::logical_or_op, cudaStream_t);

    // Min/max operations
    template LFS_CORE_API void launch_broadcast_binary<float, float, ops::minimum_op>(
        const float*, const float*, float*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::minimum_op, cudaStream_t);

    template LFS_CORE_API void launch_broadcast_binary<int, int, ops::minimum_op>(
        const int*, const int*, int*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::minimum_op, cudaStream_t);

    template LFS_CORE_API void launch_broadcast_binary<float, float, ops::maximum_op>(
        const float*, const float*, float*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::maximum_op, cudaStream_t);

    template LFS_CORE_API void launch_broadcast_binary<int, int, ops::maximum_op>(
        const int*, const int*, int*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::maximum_op, cudaStream_t);

    // Power operations
    template LFS_CORE_API void launch_broadcast_binary<float, float, ops::pow_op>(
        const float*, const float*, float*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::pow_op, cudaStream_t);

    template LFS_CORE_API void launch_broadcast_binary<int, int, ops::pow_op>(
        const int*, const int*, int*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::pow_op, cudaStream_t);

    // Not equal operation
    template LFS_CORE_API void launch_broadcast_binary<float, unsigned char, ops::not_equal_op>(
        const float*, const float*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::not_equal_op, cudaStream_t);

    template LFS_CORE_API void launch_broadcast_binary<int, unsigned char, ops::not_equal_op>(
        const int*, const int*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::not_equal_op, cudaStream_t);

    template LFS_CORE_API void launch_broadcast_binary<unsigned char, unsigned char, ops::not_equal_op>(
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
    template LFS_CORE_API void launch_broadcast_binary<__half, __half, ops::add_op>(
        const __half*, const __half*, __half*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::add_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<__half, __half, ops::sub_op>(
        const __half*, const __half*, __half*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::sub_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<__half, __half, ops::mul_op>(
        const __half*, const __half*, __half*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::mul_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<__half, __half, ops::div_op>(
        const __half*, const __half*, __half*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::div_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<__half, __half, ops::maximum_op>(
        const __half*, const __half*, __half*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::maximum_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<__half, __half, ops::minimum_op>(
        const __half*, const __half*, __half*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::minimum_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<__half, __half, ops::pow_op>(
        const __half*, const __half*, __half*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::pow_op, cudaStream_t);

    // Int64 broadcast operations
    template LFS_CORE_API void launch_broadcast_binary<int64_t, int64_t, ops::add_op>(
        const int64_t*, const int64_t*, int64_t*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::add_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<int64_t, int64_t, ops::sub_op>(
        const int64_t*, const int64_t*, int64_t*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::sub_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<int64_t, int64_t, ops::mul_op>(
        const int64_t*, const int64_t*, int64_t*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::mul_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<int64_t, int64_t, ops::div_op>(
        const int64_t*, const int64_t*, int64_t*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::div_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<int64_t, int64_t, ops::maximum_op>(
        const int64_t*, const int64_t*, int64_t*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::maximum_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<int64_t, int64_t, ops::minimum_op>(
        const int64_t*, const int64_t*, int64_t*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::minimum_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<int64_t, int64_t, ops::pow_op>(
        const int64_t*, const int64_t*, int64_t*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::pow_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<int64_t, int64_t, ops::mod_op>(
        const int64_t*, const int64_t*, int64_t*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::mod_op, cudaStream_t);

    // UInt8 broadcast operations
    template LFS_CORE_API void launch_broadcast_binary<uint8_t, uint8_t, ops::add_op>(
        const uint8_t*, const uint8_t*, uint8_t*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::add_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<uint8_t, uint8_t, ops::sub_op>(
        const uint8_t*, const uint8_t*, uint8_t*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::sub_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<uint8_t, uint8_t, ops::mul_op>(
        const uint8_t*, const uint8_t*, uint8_t*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::mul_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<uint8_t, uint8_t, ops::div_op>(
        const uint8_t*, const uint8_t*, uint8_t*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::div_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<uint8_t, uint8_t, ops::maximum_op>(
        const uint8_t*, const uint8_t*, uint8_t*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::maximum_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<uint8_t, uint8_t, ops::minimum_op>(
        const uint8_t*, const uint8_t*, uint8_t*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::minimum_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<uint8_t, uint8_t, ops::pow_op>(
        const uint8_t*, const uint8_t*, uint8_t*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::pow_op, cudaStream_t);

    // mod_op broadcast (was missing!)
    template LFS_CORE_API void launch_broadcast_binary<float, float, ops::mod_op>(
        const float*, const float*, float*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::mod_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<int, int, ops::mod_op>(
        const int*, const int*, int*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::mod_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<unsigned char, unsigned char, ops::mod_op>(
        const unsigned char*, const unsigned char*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::mod_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<__half, __half, ops::mod_op>(
        const __half*, const __half*, __half*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::mod_op, cudaStream_t);

    // Comparison operations for additional types
    template LFS_CORE_API void launch_broadcast_binary<int64_t, unsigned char, ops::greater_op>(
        const int64_t*, const int64_t*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::greater_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<int64_t, unsigned char, ops::greater_equal_op>(
        const int64_t*, const int64_t*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::greater_equal_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<int64_t, unsigned char, ops::less_op>(
        const int64_t*, const int64_t*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::less_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<int64_t, unsigned char, ops::less_equal_op>(
        const int64_t*, const int64_t*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::less_equal_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<int64_t, unsigned char, ops::equal_op>(
        const int64_t*, const int64_t*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::equal_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<int64_t, unsigned char, ops::not_equal_op>(
        const int64_t*, const int64_t*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::not_equal_op, cudaStream_t);

    template LFS_CORE_API void launch_broadcast_binary<__half, unsigned char, ops::greater_op>(
        const __half*, const __half*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::greater_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<__half, unsigned char, ops::greater_equal_op>(
        const __half*, const __half*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::greater_equal_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<__half, unsigned char, ops::less_op>(
        const __half*, const __half*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::less_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<__half, unsigned char, ops::less_equal_op>(
        const __half*, const __half*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::less_equal_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<__half, unsigned char, ops::equal_op>(
        const __half*, const __half*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::equal_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<__half, unsigned char, ops::not_equal_op>(
        const __half*, const __half*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::not_equal_op, cudaStream_t);

    // Logical operations for additional types
    template LFS_CORE_API void launch_broadcast_binary<int64_t, unsigned char, ops::logical_and_op>(
        const int64_t*, const int64_t*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::logical_and_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<int64_t, unsigned char, ops::logical_or_op>(
        const int64_t*, const int64_t*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::logical_or_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<__half, unsigned char, ops::logical_and_op>(
        const __half*, const __half*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::logical_and_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<__half, unsigned char, ops::logical_or_op>(
        const __half*, const __half*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::logical_or_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<float, unsigned char, ops::logical_xor_op>(
        const float*, const float*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::logical_xor_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<int, unsigned char, ops::logical_xor_op>(
        const int*, const int*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::logical_xor_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<unsigned char, unsigned char, ops::logical_xor_op>(
        const unsigned char*, const unsigned char*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::logical_xor_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<int64_t, unsigned char, ops::logical_xor_op>(
        const int64_t*, const int64_t*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::logical_xor_op, cudaStream_t);
    template LFS_CORE_API void launch_broadcast_binary<__half, unsigned char, ops::logical_xor_op>(
        const __half*, const __half*, unsigned char*,
        const size_t*, const size_t*, const size_t*,
        size_t, size_t, size_t, size_t, ops::logical_xor_op, cudaStream_t);

    // UInt32 broadcast operations: arithmetic keeps UInt32, predicates write Bool.
#define LFS_BROADCAST_UINT32(Out, Op)                                      \
    template LFS_CORE_API void launch_broadcast_binary<uint32_t, Out, Op>( \
        const uint32_t*, const uint32_t*, Out*,                            \
        const size_t*, const size_t*, const size_t*,                       \
        size_t, size_t, size_t, size_t, Op, cudaStream_t);
    LFS_BROADCAST_UINT32(uint32_t, ops::add_op)
    LFS_BROADCAST_UINT32(uint32_t, ops::sub_op)
    LFS_BROADCAST_UINT32(uint32_t, ops::mul_op)
    LFS_BROADCAST_UINT32(uint32_t, ops::div_op)
    LFS_BROADCAST_UINT32(uint32_t, ops::maximum_op)
    LFS_BROADCAST_UINT32(uint32_t, ops::minimum_op)
    LFS_BROADCAST_UINT32(uint32_t, ops::pow_op)
    LFS_BROADCAST_UINT32(uint32_t, ops::mod_op)
    LFS_BROADCAST_UINT32(unsigned char, ops::greater_op)
    LFS_BROADCAST_UINT32(unsigned char, ops::greater_equal_op)
    LFS_BROADCAST_UINT32(unsigned char, ops::less_op)
    LFS_BROADCAST_UINT32(unsigned char, ops::less_equal_op)
    LFS_BROADCAST_UINT32(unsigned char, ops::equal_op)
    LFS_BROADCAST_UINT32(unsigned char, ops::not_equal_op)
    LFS_BROADCAST_UINT32(unsigned char, ops::logical_and_op)
    LFS_BROADCAST_UINT32(unsigned char, ops::logical_or_op)
    LFS_BROADCAST_UINT32(unsigned char, ops::logical_xor_op)
#undef LFS_BROADCAST_UINT32
} // namespace lfs::core::tensor_ops
