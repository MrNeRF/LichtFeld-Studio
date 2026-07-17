/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include "core/device_fault.hpp"
#include "internal/cub_workspace.hpp"
#include "internal/cuda_memory_guard.hpp"
#include "internal/tensor_functors.hpp"
#include "internal/tensor_ops.hpp"
#include <cfloat>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <limits>

// Thrust headers
#include "core/assert.hpp"
#include "core/tensor_fwd.hpp"
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

namespace lfs::core::tensor_ops {

    inline constexpr size_t BINARY_BROADCAST_SHAPE_SLOTS = 3 * MAX_TENSOR_RANK;
    inline constexpr size_t TERNARY_BROADCAST_SHAPE_SLOTS = 4 * MAX_TENSOR_RANK;

    struct BinaryBroadcastShapes {
        size_t values[BINARY_BROADCAST_SHAPE_SLOTS]{};
    };

    struct TernaryBroadcastShapes {
        size_t values[TERNARY_BROADCAST_SHAPE_SLOTS]{};
    };

    BinaryBroadcastShapes pack_binary_broadcast_shapes(
        const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
        const size_t a_rank, const size_t b_rank, const size_t c_rank) {
        BinaryBroadcastShapes shapes;
        std::copy(a_shape, a_shape + a_rank, shapes.values);
        std::copy(b_shape, b_shape + b_rank, shapes.values + MAX_TENSOR_RANK);
        std::copy(c_shape, c_shape + c_rank, shapes.values + 2 * MAX_TENSOR_RANK);
        return shapes;
    }

    namespace {
        template <typename T>
        void launch_masked_fill_impl(T* data, const unsigned char* mask, T val, size_t n, cudaStream_t stream) {
            auto data_ptr = thrust::device_pointer_cast(data);
            auto mask_ptr = thrust::device_pointer_cast(mask);
            auto begin = thrust::make_zip_iterator(thrust::make_tuple(data_ptr, mask_ptr));
            auto end = thrust::make_zip_iterator(thrust::make_tuple(data_ptr + n, mask_ptr + n));
            thrust::transform(thrust::cuda::par.on(stream), begin, end, data_ptr,
                              ops::masked_fill_op<T>(val));
        }

        template <typename T>
        void launch_masked_select_impl(const T* input, const unsigned char* mask,
                                       T* output, size_t n, size_t output_size, cudaStream_t stream) {
            if (n == 0 || output_size == 0)
                return;

            auto input_ptr = thrust::device_pointer_cast(input);
            auto mask_ptr = thrust::device_pointer_cast(mask);
            auto output_ptr = thrust::device_pointer_cast(output);

            auto begin = thrust::make_zip_iterator(thrust::make_tuple(input_ptr, mask_ptr));
            auto end = thrust::make_zip_iterator(thrust::make_tuple(input_ptr + n, mask_ptr + n));

            auto transform_begin = thrust::make_transform_iterator(begin, ops::extract_value_op());
            auto transform_end = thrust::make_transform_iterator(end, ops::extract_value_op());
            auto mask_begin = thrust::make_transform_iterator(begin, ops::extract_mask_op());

            thrust::copy_if(thrust::cuda::par.on(stream),
                            transform_begin, transform_end, mask_begin, output_ptr,
                            [] __device__(bool x) { return x; });
        }
    } // namespace

    // ============= Import broadcast index calculator =============
    __device__ inline size_t compute_broadcast_index(
        size_t idx, const size_t* src_shape, size_t src_rank,
        const size_t* dst_shape, size_t dst_rank) {

        size_t src_idx = 0, dst_stride = 1;

#pragma unroll 8
        for (int i = dst_rank - 1; i >= 0; --i) {
            size_t dst_coord = (idx / dst_stride) % dst_shape[i];
            int src_dim = i - (dst_rank - src_rank);

            if (src_dim >= 0) {
                size_t src_coord = (src_shape[src_dim] == 1) ? 0 : dst_coord;
                size_t src_stride = 1;
                for (int j = src_dim + 1; j < src_rank; ++j) {
                    src_stride *= src_shape[j];
                }
                src_idx += src_coord * src_stride;
            }

            dst_stride *= dst_shape[i];
        }

        return src_idx;
    }

    // ============= Comparison Kernels (with broadcasting) =============
    __global__ void compare_eq_kernel(const float* a, const float* b, unsigned char* c,
                                      const BinaryBroadcastShapes shape_storage,
                                      size_t info, size_t total) {
        const size_t* shapes = shape_storage.values;
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total)
            return;

        size_t a_rank = info & 0x1F;
        size_t b_rank = (info >> 5) & 0x1F;
        size_t c_rank = (info >> 10) & 0x1F;
        bool fast_path = info & 0x8000;

        if (fast_path) {
            c[idx] = (a[idx] == b[idx]) ? 1 : 0;
        } else {
            size_t a_idx = compute_broadcast_index(
                idx, shapes, a_rank, shapes + 2 * MAX_TENSOR_RANK, c_rank);
            size_t b_idx = compute_broadcast_index(
                idx, shapes + MAX_TENSOR_RANK, b_rank,
                shapes + 2 * MAX_TENSOR_RANK, c_rank);
            c[idx] = (a[a_idx] == b[b_idx]) ? 1 : 0;
        }
    }

    __global__ void compare_lt_kernel(const float* a, const float* b, unsigned char* c,
                                      const BinaryBroadcastShapes shape_storage,
                                      size_t info, size_t total) {
        const size_t* shapes = shape_storage.values;
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total)
            return;

        size_t a_rank = info & 0x1F;
        size_t b_rank = (info >> 5) & 0x1F;
        size_t c_rank = (info >> 10) & 0x1F;
        bool fast_path = info & 0x8000;

        if (fast_path) {
            c[idx] = (a[idx] < b[idx]) ? 1 : 0;
        } else {
            size_t a_idx = compute_broadcast_index(
                idx, shapes, a_rank, shapes + 2 * MAX_TENSOR_RANK, c_rank);
            size_t b_idx = compute_broadcast_index(
                idx, shapes + MAX_TENSOR_RANK, b_rank,
                shapes + 2 * MAX_TENSOR_RANK, c_rank);
            c[idx] = (a[a_idx] < b[b_idx]) ? 1 : 0;
        }
    }

    __global__ void compare_gt_kernel(const float* a, const float* b, unsigned char* c,
                                      const BinaryBroadcastShapes shape_storage,
                                      size_t info, size_t total) {
        const size_t* shapes = shape_storage.values;
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total)
            return;

        size_t a_rank = info & 0x1F;
        size_t b_rank = (info >> 5) & 0x1F;
        size_t c_rank = (info >> 10) & 0x1F;
        bool fast_path = info & 0x8000;

        if (fast_path) {
            c[idx] = (a[idx] > b[idx]) ? 1 : 0;
        } else {
            size_t a_idx = compute_broadcast_index(
                idx, shapes, a_rank, shapes + 2 * MAX_TENSOR_RANK, c_rank);
            size_t b_idx = compute_broadcast_index(
                idx, shapes + MAX_TENSOR_RANK, b_rank,
                shapes + 2 * MAX_TENSOR_RANK, c_rank);
            c[idx] = (a[a_idx] > b[b_idx]) ? 1 : 0;
        }
    }

    // ============= Scalar Comparison Launch Functions =============
    void launch_compare_scalar_eq(const float* a, float val, unsigned char* r, size_t n, cudaStream_t s) {
        auto a_ptr = thrust::device_pointer_cast(a);
        auto r_ptr = thrust::device_pointer_cast(r);
        thrust::transform(thrust::cuda::par.on(s), a_ptr, a_ptr + n, r_ptr,
                          ops::equal_scalar_op<float>(val));
    }

    void launch_compare_scalar_lt(const float* a, float val, unsigned char* r, size_t n, cudaStream_t s) {
        auto a_ptr = thrust::device_pointer_cast(a);
        auto r_ptr = thrust::device_pointer_cast(r);
        thrust::transform(thrust::cuda::par.on(s), a_ptr, a_ptr + n, r_ptr,
                          ops::less_scalar_op<float>(val));
    }

    void launch_compare_scalar_gt(const float* a, float val, unsigned char* r, size_t n, cudaStream_t s) {
        auto a_ptr = thrust::device_pointer_cast(a);
        auto r_ptr = thrust::device_pointer_cast(r);
        thrust::transform(thrust::cuda::par.on(s), a_ptr, a_ptr + n, r_ptr,
                          ops::greater_scalar_op<float>(val));
    }

    // ============= Logical Operation Kernels (with broadcasting) =============
    __global__ void logical_and_kernel(const unsigned char* a, const unsigned char* b,
                                       unsigned char* c, const BinaryBroadcastShapes shape_storage,
                                       size_t info, size_t total) {
        const size_t* shapes = shape_storage.values;
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total)
            return;

        size_t a_rank = info & 0x1F;
        size_t b_rank = (info >> 5) & 0x1F;
        size_t c_rank = (info >> 10) & 0x1F;
        bool fast_path = info & 0x8000;

        if (fast_path) {
            c[idx] = (a[idx] && b[idx]) ? 1 : 0;
        } else {
            size_t a_idx = compute_broadcast_index(
                idx, shapes, a_rank, shapes + 2 * MAX_TENSOR_RANK, c_rank);
            size_t b_idx = compute_broadcast_index(
                idx, shapes + MAX_TENSOR_RANK, b_rank,
                shapes + 2 * MAX_TENSOR_RANK, c_rank);
            c[idx] = (a[a_idx] && b[b_idx]) ? 1 : 0;
        }
    }

    __global__ void logical_or_kernel(const unsigned char* a, const unsigned char* b,
                                      unsigned char* c, const BinaryBroadcastShapes shape_storage,
                                      size_t info, size_t total) {
        const size_t* shapes = shape_storage.values;
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total)
            return;

        size_t a_rank = info & 0x1F;
        size_t b_rank = (info >> 5) & 0x1F;
        size_t c_rank = (info >> 10) & 0x1F;
        bool fast_path = info & 0x8000;

        if (fast_path) {
            c[idx] = (a[idx] || b[idx]) ? 1 : 0;
        } else {
            size_t a_idx = compute_broadcast_index(
                idx, shapes, a_rank, shapes + 2 * MAX_TENSOR_RANK, c_rank);
            size_t b_idx = compute_broadcast_index(
                idx, shapes + MAX_TENSOR_RANK, b_rank,
                shapes + 2 * MAX_TENSOR_RANK, c_rank);
            c[idx] = (a[a_idx] || b[b_idx]) ? 1 : 0;
        }
    }

    __global__ void logical_xor_kernel(const unsigned char* a, const unsigned char* b,
                                       unsigned char* c, const BinaryBroadcastShapes shape_storage,
                                       size_t info, size_t total) {
        const size_t* shapes = shape_storage.values;
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total)
            return;

        size_t a_rank = info & 0x1F;
        size_t b_rank = (info >> 5) & 0x1F;
        size_t c_rank = (info >> 10) & 0x1F;
        bool fast_path = info & 0x8000;

        if (fast_path) {
            c[idx] = ((a[idx] != 0) != (b[idx] != 0)) ? 1 : 0;
        } else {
            size_t a_idx = compute_broadcast_index(
                idx, shapes, a_rank, shapes + 2 * MAX_TENSOR_RANK, c_rank);
            size_t b_idx = compute_broadcast_index(
                idx, shapes + MAX_TENSOR_RANK, b_rank,
                shapes + 2 * MAX_TENSOR_RANK, c_rank);
            c[idx] = ((a[a_idx] != 0) != (b[b_idx] != 0)) ? 1 : 0;
        }
    }

    void launch_logical_not(const unsigned char* a, unsigned char* r, size_t n, cudaStream_t s) {
        auto a_ptr = thrust::device_pointer_cast(a);
        auto r_ptr = thrust::device_pointer_cast(r);
        thrust::transform(thrust::cuda::par.on(s), a_ptr, a_ptr + n, r_ptr,
                          ops::logical_not_op());
    }

    // ============= Launch Functions for Comparison/Logical Ops =============
    void launch_compare_eq(const float* a, const float* b, unsigned char* c,
                           const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                           size_t a_rank, size_t b_rank, size_t c_rank,
                           size_t c_elements, cudaStream_t stream) {
        LFS_ASSERT_MSG(a_rank <= MAX_TENSOR_RANK && b_rank <= MAX_TENSOR_RANK &&
                           c_rank <= MAX_TENSOR_RANK,
                       "comparison broadcast rank exceeds MAX_TENSOR_RANK");
        const auto shapes = pack_binary_broadcast_shapes(
            a_shape, b_shape, c_shape, a_rank, b_rank, c_rank);

        bool fast_path = (a_rank == c_rank && b_rank == c_rank &&
                          std::equal(a_shape, a_shape + a_rank, c_shape) &&
                          std::equal(b_shape, b_shape + b_rank, c_shape));
        size_t info = a_rank | (b_rank << 5) | (c_rank << 10) | (fast_path << 15);

        int blocks = (c_elements + 255) / 256;
        compare_eq_kernel<<<blocks, 256, 0, stream>>>(a, b, c, shapes, info, c_elements);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.masking.compare_eq");
    }

    void launch_compare_lt(const float* a, const float* b, unsigned char* c,
                           const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                           size_t a_rank, size_t b_rank, size_t c_rank,
                           size_t c_elements, cudaStream_t stream) {
        LFS_ASSERT_MSG(a_rank <= MAX_TENSOR_RANK && b_rank <= MAX_TENSOR_RANK &&
                           c_rank <= MAX_TENSOR_RANK,
                       "comparison broadcast rank exceeds MAX_TENSOR_RANK");
        const auto shapes = pack_binary_broadcast_shapes(
            a_shape, b_shape, c_shape, a_rank, b_rank, c_rank);

        bool fast_path = (a_rank == c_rank && b_rank == c_rank &&
                          std::equal(a_shape, a_shape + a_rank, c_shape) &&
                          std::equal(b_shape, b_shape + b_rank, c_shape));
        size_t info = a_rank | (b_rank << 5) | (c_rank << 10) | (fast_path << 15);

        int blocks = (c_elements + 255) / 256;
        compare_lt_kernel<<<blocks, 256, 0, stream>>>(a, b, c, shapes, info, c_elements);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.masking.compare_lt");
    }

    void launch_compare_gt(const float* a, const float* b, unsigned char* c,
                           const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                           size_t a_rank, size_t b_rank, size_t c_rank,
                           size_t c_elements, cudaStream_t stream) {
        LFS_ASSERT_MSG(a_rank <= MAX_TENSOR_RANK && b_rank <= MAX_TENSOR_RANK &&
                           c_rank <= MAX_TENSOR_RANK,
                       "comparison broadcast rank exceeds MAX_TENSOR_RANK");
        const auto shapes = pack_binary_broadcast_shapes(
            a_shape, b_shape, c_shape, a_rank, b_rank, c_rank);

        bool fast_path = (a_rank == c_rank && b_rank == c_rank &&
                          std::equal(a_shape, a_shape + a_rank, c_shape) &&
                          std::equal(b_shape, b_shape + b_rank, c_shape));
        size_t info = a_rank | (b_rank << 5) | (c_rank << 10) | (fast_path << 15);

        int blocks = (c_elements + 255) / 256;
        compare_gt_kernel<<<blocks, 256, 0, stream>>>(a, b, c, shapes, info, c_elements);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.masking.compare_gt");
    }

    void launch_logical_and(const unsigned char* a, const unsigned char* b, unsigned char* c,
                            const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                            size_t a_rank, size_t b_rank, size_t c_rank,
                            size_t c_elements, cudaStream_t stream) {
        LFS_ASSERT_MSG(a_rank <= MAX_TENSOR_RANK && b_rank <= MAX_TENSOR_RANK &&
                           c_rank <= MAX_TENSOR_RANK,
                       "logical broadcast rank exceeds MAX_TENSOR_RANK");
        const auto shapes = pack_binary_broadcast_shapes(
            a_shape, b_shape, c_shape, a_rank, b_rank, c_rank);

        bool fast_path = (a_rank == c_rank && b_rank == c_rank &&
                          std::equal(a_shape, a_shape + a_rank, c_shape) &&
                          std::equal(b_shape, b_shape + b_rank, c_shape));
        size_t info = a_rank | (b_rank << 5) | (c_rank << 10) | (fast_path << 15);

        int blocks = (c_elements + 255) / 256;
        logical_and_kernel<<<blocks, 256, 0, stream>>>(a, b, c, shapes, info, c_elements);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.masking.logical_and");
    }

    void launch_logical_or(const unsigned char* a, const unsigned char* b, unsigned char* c,
                           const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                           size_t a_rank, size_t b_rank, size_t c_rank,
                           size_t c_elements, cudaStream_t stream) {
        LFS_ASSERT_MSG(a_rank <= MAX_TENSOR_RANK && b_rank <= MAX_TENSOR_RANK &&
                           c_rank <= MAX_TENSOR_RANK,
                       "logical broadcast rank exceeds MAX_TENSOR_RANK");
        const auto shapes = pack_binary_broadcast_shapes(
            a_shape, b_shape, c_shape, a_rank, b_rank, c_rank);

        bool fast_path = (a_rank == c_rank && b_rank == c_rank &&
                          std::equal(a_shape, a_shape + a_rank, c_shape) &&
                          std::equal(b_shape, b_shape + b_rank, c_shape));
        size_t info = a_rank | (b_rank << 5) | (c_rank << 10) | (fast_path << 15);

        int blocks = (c_elements + 255) / 256;
        logical_or_kernel<<<blocks, 256, 0, stream>>>(a, b, c, shapes, info, c_elements);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.masking.logical_or");
    }

    void launch_logical_xor(const unsigned char* a, const unsigned char* b, unsigned char* c,
                            const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                            size_t a_rank, size_t b_rank, size_t c_rank,
                            size_t c_elements, cudaStream_t stream) {
        LFS_ASSERT_MSG(a_rank <= MAX_TENSOR_RANK && b_rank <= MAX_TENSOR_RANK &&
                           c_rank <= MAX_TENSOR_RANK,
                       "logical broadcast rank exceeds MAX_TENSOR_RANK");
        const auto shapes = pack_binary_broadcast_shapes(
            a_shape, b_shape, c_shape, a_rank, b_rank, c_rank);

        bool fast_path = (a_rank == c_rank && b_rank == c_rank &&
                          std::equal(a_shape, a_shape + a_rank, c_shape) &&
                          std::equal(b_shape, b_shape + b_rank, c_shape));
        size_t info = a_rank | (b_rank << 5) | (c_rank << 10) | (fast_path << 15);

        int blocks = (c_elements + 255) / 256;
        logical_xor_kernel<<<blocks, 256, 0, stream>>>(a, b, c, shapes, info, c_elements);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.masking.logical_xor");
    }

    // ============= Masking Operations =============
    void launch_masked_fill(float* data, const unsigned char* mask, float val, size_t n, cudaStream_t s) {
        launch_masked_fill_impl(data, mask, val, n, s);
    }

    void launch_masked_fill(int32_t* data, const unsigned char* mask, int32_t val, size_t n, cudaStream_t s) {
        launch_masked_fill_impl(data, mask, val, n, s);
    }

    void launch_masked_fill(int64_t* data, const unsigned char* mask, int64_t val, size_t n, cudaStream_t s) {
        launch_masked_fill_impl(data, mask, val, n, s);
    }

    void launch_masked_fill(uint8_t* data, const unsigned char* mask, uint8_t val, size_t n, cudaStream_t s) {
        launch_masked_fill_impl(data, mask, val, n, s);
    }

    void launch_masked_fill(__half* data, const unsigned char* mask, __half val, size_t n, cudaStream_t s) {
        launch_masked_fill_impl(data, mask, val, n, s);
    }

    void launch_masked_select(const float* input, const unsigned char* mask,
                              float* output, size_t n, size_t output_size, cudaStream_t stream) {
        launch_masked_select_impl(input, mask, output, n, output_size, stream);
    }

    void launch_masked_select(const __half* input, const unsigned char* mask,
                              __half* output, size_t n, size_t output_size, cudaStream_t stream) {
        launch_masked_select_impl(input, mask, output, n, output_size, stream);
    }

    void launch_masked_select(const int32_t* input, const unsigned char* mask,
                              int32_t* output, size_t n, size_t output_size, cudaStream_t stream) {
        launch_masked_select_impl(input, mask, output, n, output_size, stream);
    }

    void launch_masked_select(const int64_t* input, const unsigned char* mask,
                              int64_t* output, size_t n, size_t output_size, cudaStream_t stream) {
        launch_masked_select_impl(input, mask, output, n, output_size, stream);
    }

    void launch_masked_select(const uint8_t* input, const unsigned char* mask,
                              uint8_t* output, size_t n, size_t output_size, cudaStream_t stream) {
        launch_masked_select_impl(input, mask, output, n, output_size, stream);
    }

    template <typename T>
    __global__ void masked_scatter_compact_kernel(T* data, const unsigned char* mask,
                                                  const T* src, const int* scan, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n && mask[idx]) {
            data[idx] = src[scan[idx]];
        }
    }

    template <typename T>
    void launch_masked_scatter_impl(T* data, const unsigned char* mask,
                                    const T* src, size_t n, size_t src_size, cudaStream_t stream) {
        if (n == 0 || src_size == 0)
            return;

        ScopedDeviceBuffer scan_result(
            n * sizeof(int), stream, "tensor.masked_scatter_scan");
        run_cub_operation(
            "cub::DeviceScan::ExclusiveSum", stream,
            [&](void* workspace, size_t& workspace_bytes) {
                return cub::DeviceScan::ExclusiveSum(
                    workspace, workspace_bytes, mask, scan_result.as<int>(), n, stream);
            });

        int blocks = (n + 255) / 256;
        masked_scatter_compact_kernel<T><<<blocks, 256, 0, stream>>>(
            data, mask, src, scan_result.as<int>(), n);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.masking.masked_scatter_compact");
    }

    void launch_masked_scatter(float* data, const unsigned char* mask,
                               const float* src, size_t n, size_t src_size, cudaStream_t stream) {
        launch_masked_scatter_impl(data, mask, src, n, src_size, stream);
    }

    void launch_masked_scatter(__half* data, const unsigned char* mask,
                               const __half* src, size_t n, size_t src_size, cudaStream_t stream) {
        launch_masked_scatter_impl(data, mask, src, n, src_size, stream);
    }

    void launch_masked_scatter(int32_t* data, const unsigned char* mask,
                               const int32_t* src, size_t n, size_t src_size, cudaStream_t stream) {
        launch_masked_scatter_impl(data, mask, src, n, src_size, stream);
    }

    void launch_masked_scatter(int64_t* data, const unsigned char* mask,
                               const int64_t* src, size_t n, size_t src_size, cudaStream_t stream) {
        launch_masked_scatter_impl(data, mask, src, n, src_size, stream);
    }

    void launch_masked_scatter(uint8_t* data, const unsigned char* mask,
                               const uint8_t* src, size_t n, size_t src_size, cudaStream_t stream) {
        launch_masked_scatter_impl(data, mask, src, n, src_size, stream);
    }

    // ============= Where Operation =============
    __global__ void where_kernel(const unsigned char* cond, const float* x, const float* y,
                                 float* r, const TernaryBroadcastShapes shape_storage,
                                 size_t cr, size_t xr, size_t yr, size_t rr, size_t n) {
        const size_t* shapes = shape_storage.values;
        // Support both 1D and 2D grids for large arrays
        size_t block_id = blockIdx.y * gridDim.x + blockIdx.x;
        size_t idx = block_id * blockDim.x + threadIdx.x;
        if (idx >= n)
            return;

        size_t c_idx = compute_broadcast_index(
            idx, shapes, cr, shapes + 3 * MAX_TENSOR_RANK, rr);
        size_t x_idx = compute_broadcast_index(
            idx, shapes + MAX_TENSOR_RANK, xr,
            shapes + 3 * MAX_TENSOR_RANK, rr);
        size_t y_idx = compute_broadcast_index(
            idx, shapes + 2 * MAX_TENSOR_RANK, yr,
            shapes + 3 * MAX_TENSOR_RANK, rr);

        r[idx] = cond[c_idx] ? x[x_idx] : y[y_idx];
    }

    void launch_where(const unsigned char* cond, const float* x, const float* y, float* r,
                      const size_t* cond_shape, const size_t* x_shape,
                      const size_t* y_shape, const size_t* r_shape,
                      size_t cond_rank, size_t x_rank, size_t y_rank, size_t r_rank,
                      size_t total, cudaStream_t stream) {

        LFS_ASSERT_MSG(cond_rank <= MAX_TENSOR_RANK && x_rank <= MAX_TENSOR_RANK &&
                           y_rank <= MAX_TENSOR_RANK && r_rank <= MAX_TENSOR_RANK,
                       "where rank exceeds MAX_TENSOR_RANK");
        TernaryBroadcastShapes shapes;
        std::copy(cond_shape, cond_shape + cond_rank, shapes.values);
        std::copy(x_shape, x_shape + x_rank, shapes.values + MAX_TENSOR_RANK);
        std::copy(y_shape, y_shape + y_rank, shapes.values + 2 * MAX_TENSOR_RANK);
        std::copy(r_shape, r_shape + r_rank, shapes.values + 3 * MAX_TENSOR_RANK);

        // Use 2D grid for large arrays to avoid exceeding grid dimension limits
        size_t num_blocks = (total + 255) / 256;
        const size_t max_blocks_x = 65535;

        if (num_blocks <= max_blocks_x) {
            where_kernel<<<num_blocks, 256, 0, stream>>>(
                cond, x, y, r, shapes, cond_rank, x_rank, y_rank, r_rank, total);
            LFS_CUDA_LAUNCH_CHECK(stream, "tensor.masking.where");
        } else {
            dim3 grid(std::min(num_blocks, max_blocks_x),
                      (num_blocks + max_blocks_x - 1) / max_blocks_x);
            where_kernel<<<grid, 256, 0, stream>>>(
                cond, x, y, r, shapes, cond_rank, x_rank, y_rank, r_rank, total);
            LFS_CUDA_LAUNCH_CHECK(stream, "tensor.masking.where");
        }
    }

    // ============= Count Nonzero =============
    void launch_count_nonzero_bool(const unsigned char* data, size_t* count, size_t n, cudaStream_t stream) {
        if (n == 0) {
            LFS_CUDA_CHECK(cudaMemsetAsync(count, 0, sizeof(size_t), stream));
            return;
        }

        auto data_ptr = thrust::device_pointer_cast(data);

        // Thrust count_if returns to host, so we get it first
        size_t result = thrust::count_if(thrust::cuda::par.on(stream),
                                         data_ptr, data_ptr + n,
                                         ops::is_nonzero_bool_op());

        // Write to device memory (use sync copy since result is stack variable)
        // OPTIMIZATION: cudaMemcpy (blocking) is more efficient than cudaMemcpyAsync + cudaStreamSynchronize
        LFS_CUDA_CHECK(cudaMemcpy(count, &result, sizeof(size_t), cudaMemcpyHostToDevice));
    }

    void launch_count_nonzero_float(const float* data, size_t* count, size_t n, cudaStream_t stream) {
        if (n == 0) {
            LFS_CUDA_CHECK(cudaMemsetAsync(count, 0, sizeof(size_t), stream));
            return;
        }

        auto data_ptr = thrust::device_pointer_cast(data);

        // Thrust count_if returns to host, so we get it first
        size_t result = thrust::count_if(thrust::cuda::par.on(stream),
                                         data_ptr, data_ptr + n,
                                         ops::is_nonzero_op<float>());

        // Then write to device memory
        // Write to device memory (use sync copy since result is stack variable)
        // OPTIMIZATION: cudaMemcpy (blocking) is more efficient than cudaMemcpyAsync + cudaStreamSynchronize
        LFS_CUDA_CHECK(cudaMemcpy(count, &result, sizeof(size_t), cudaMemcpyHostToDevice));
    }

    // ============= Index Operations =============
    // Phase 6C-P3: index_select is THE reference device-fault kernel (spec §9
    // sign-off 1). On Assert-mode OOB: first-fault record then zero/skip.
    // Clamp/Wrap unchanged (no BoundsViolation). gather/scatter untouched.
    //
    // fault / op_id / trap_after_record: host pass-through from the launch-prep
    // helper. Production Assert path always arms a per-stream slot. Unchecked
    // fast path (ValidatedIndexToken match) passes fault=nullptr and skips
    // reset/harvest — host-only token never appears in this .cu TU.
    template <typename T>
    __global__ void index_select_kernel(const T* in, const int* idx, T* out,
                                        size_t outer, size_t dim_size, size_t inner,
                                        size_t idx_size, int boundary,
                                        DeviceFaultRecord* fault,
                                        std::uint32_t op_id,
                                        bool trap_after_record) {
        // Support both 1D and 2D grids for large arrays
        size_t block_id = blockIdx.y * gridDim.x + blockIdx.x;
        size_t tid = block_id * blockDim.x + threadIdx.x;
        size_t total = outer * idx_size * inner;
        if (tid >= total)
            return;

        size_t o = tid / (idx_size * inner);
        size_t i = (tid / inner) % idx_size;
        size_t j = tid % inner;
        int sel = idx[i];

        if (boundary == 1)
            sel = max(0, min((int)dim_size - 1, sel));
        else if (boundary == 2)
            sel = ((sel % (int)dim_size) + dim_size) % dim_size;
        else if (sel < 0 || sel >= static_cast<int>(dim_size)) {
            LFS_DEBUG_ASSERT_MSG(sel >= 0 && sel < static_cast<int>(dim_size),
                                 detail::format_cuda_safe("index_select index must be in range "
                                                          "(selected_index={}, dimension_size={}, "
                                                          "index_position={}, output_index={}, boundary_mode={})",
                                                          sel, dim_size, i, tid, boundary));
            // Failure branch only: first-fault CAS, then zero/skip (spec §1.4).
            // Hot success path never touches the fault record.
            if (fault != nullptr) {
                const std::uint64_t thread_id =
                    (static_cast<std::uint64_t>(blockIdx.x) << 32) |
                    static_cast<std::uint64_t>(threadIdx.x);
                device_fault_try_record_first(
                    fault,
                    op_id,
                    static_cast<std::int64_t>(sel),
                    static_cast<std::int64_t>(dim_size),
                    thread_id,
                    trap_after_record);
            }
            out[tid] = 0;
            return;
        }

        const size_t src_idx = o * dim_size * inner + static_cast<size_t>(sel) * inner + j;
        LFS_DEBUG_ASSERT_MSG(src_idx < outer * dim_size * inner,
                             detail::format_cuda_safe("index_select source offset must be in range "
                                                      "(source_offset={}, source_numel={}, outer={}, "
                                                      "dimension_size={}, inner={}, output_index={})",
                                                      src_idx, outer * dim_size * inner, outer,
                                                      dim_size, inner, tid));
        out[tid] = in[src_idx];
    }

    // Host-side launch prep for the checked Assert path (spec §1.5 steps 1–4).
    // Graph capture is rejected by device_fault_slot_enqueue_reset (returns
    // cudaErrorStreamCaptureUnsupported); host .cpp also pre-checks and throws
    // typed Unsupported. Clamp/Wrap leave the fault unarmed.
    // Unchecked ValidatedIndexToken fast path is host-only / test-stub (not here).
    inline void prepare_index_select_device_fault(
        const int boundary,
        const cudaStream_t stream,
        const char* launch_tag,
        DeviceFaultRecord** out_fault,
        std::uint32_t* out_op_id,
        bool* out_trap) {
        *out_fault = nullptr;
        *out_op_id = 0;
        *out_trap = false;
        if (boundary != 0) {
            return;
        }
        // §1.9: reject capture before any reset/kernel/harvest enqueue.
        const cudaError_t reset_status = device_fault_slot_enqueue_reset(stream);
        if (reset_status != cudaSuccess) {
            LFS_ENSURE_CUDA_SUCCESS_MSG(
                reset_status, "device_fault_slot_enqueue_reset(index_select)",
                launch_tag);
        }
        DeviceFaultRecord* fault = nullptr;
        const cudaError_t acquire_status = device_fault_slot_acquire(stream, &fault);
        if (acquire_status != cudaSuccess) {
            LFS_ENSURE_CUDA_SUCCESS_MSG(
                acquire_status, "device_fault_slot_acquire(index_select)",
                launch_tag);
        }
        *out_fault = fault;
        // Correlate op_id with the breadcrumb sequence that LAUNCH_CHECK will also bump.
        *out_op_id = static_cast<std::uint32_t>(
            current_cuda_breadcrumb_sequence() & 0xffffffffu);
        *out_trap = device_fault_trap_after_record_for_launch();
    }

    inline void finish_index_select_device_fault(
        DeviceFaultRecord* fault,
        const cudaStream_t stream,
        const char* launch_tag) {
        if (fault == nullptr) {
            return;
        }
        const cudaError_t harvest_status = device_fault_slot_enqueue_harvest(stream);
        if (harvest_status != cudaSuccess) {
            LFS_ENSURE_CUDA_SUCCESS_MSG(
                harvest_status, "device_fault_slot_enqueue_harvest(index_select)",
                launch_tag);
        }
    }

    // Shared geometry + fault arming for the four dtype overloads. LAUNCH_CHECK
    // tags must remain string literals at each call site (macro static_assert).
    [[nodiscard]] bool index_select_compute_geometry(const size_t* shape, size_t rank, int dim,
                                                     size_t idx_size,
                                                     size_t* outer, size_t* inner, size_t* total) {
        if (idx_size == 0) {
            return false;
        }
        *outer = 1;
        *inner = 1;
        for (int i = 0; i < dim; ++i)
            *outer *= shape[i];
        for (size_t i = dim + 1; i < rank; ++i)
            *inner *= shape[i];
        *total = (*outer) * idx_size * (*inner);
        return *total != 0;
    }

    // Float32 overload
    void launch_index_select(const float* in, const int* idx, float* out,
                             const size_t* shape, size_t rank, int dim,
                             size_t idx_size, int boundary, cudaStream_t stream) {
        size_t outer = 0, inner = 0, total = 0;
        if (!index_select_compute_geometry(shape, rank, dim, idx_size, &outer, &inner, &total)) {
            return;
        }

        DeviceFaultRecord* fault = nullptr;
        std::uint32_t op_id = 0;
        bool trap_after_record = false;
        prepare_index_select_device_fault(boundary, stream, "tensor.masking.index_select_f32",
                                          &fault, &op_id, &trap_after_record);

        size_t num_blocks = (total + 255) / 256;
        const size_t max_blocks_x = 65535;
        if (num_blocks <= max_blocks_x) {
            index_select_kernel<float><<<num_blocks, 256, 0, stream>>>(
                in, idx, out, outer, shape[dim], inner, idx_size, boundary,
                fault, op_id, trap_after_record);
            LFS_CUDA_LAUNCH_CHECK(stream, "tensor.masking.index_select_f32");
        } else {
            dim3 grid(std::min(num_blocks, max_blocks_x),
                      (num_blocks + max_blocks_x - 1) / max_blocks_x);
            index_select_kernel<float><<<grid, 256, 0, stream>>>(
                in, idx, out, outer, shape[dim], inner, idx_size, boundary,
                fault, op_id, trap_after_record);
            LFS_CUDA_LAUNCH_CHECK(stream, "tensor.masking.index_select_f32");
        }
        finish_index_select_device_fault(fault, stream, "tensor.masking.index_select_f32");
    }

    // Int64 overload
    void launch_index_select(const int64_t* in, const int* idx, int64_t* out,
                             const size_t* shape, size_t rank, int dim,
                             size_t idx_size, int boundary, cudaStream_t stream) {
        size_t outer = 0, inner = 0, total = 0;
        if (!index_select_compute_geometry(shape, rank, dim, idx_size, &outer, &inner, &total)) {
            return;
        }

        DeviceFaultRecord* fault = nullptr;
        std::uint32_t op_id = 0;
        bool trap_after_record = false;
        prepare_index_select_device_fault(boundary, stream, "tensor.masking.index_select_i64",
                                          &fault, &op_id, &trap_after_record);

        size_t num_blocks = (total + 255) / 256;
        const size_t max_blocks_x = 65535;
        if (num_blocks <= max_blocks_x) {
            index_select_kernel<int64_t><<<num_blocks, 256, 0, stream>>>(
                in, idx, out, outer, shape[dim], inner, idx_size, boundary,
                fault, op_id, trap_after_record);
            LFS_CUDA_LAUNCH_CHECK(stream, "tensor.masking.index_select_i64");
        } else {
            dim3 grid(std::min(num_blocks, max_blocks_x),
                      (num_blocks + max_blocks_x - 1) / max_blocks_x);
            index_select_kernel<int64_t><<<grid, 256, 0, stream>>>(
                in, idx, out, outer, shape[dim], inner, idx_size, boundary,
                fault, op_id, trap_after_record);
            LFS_CUDA_LAUNCH_CHECK(stream, "tensor.masking.index_select_i64");
        }
        finish_index_select_device_fault(fault, stream, "tensor.masking.index_select_i64");
    }

    // Int32 overload
    void launch_index_select(const int32_t* in, const int* idx, int32_t* out,
                             const size_t* shape, size_t rank, int dim,
                             size_t idx_size, int boundary, cudaStream_t stream) {
        size_t outer = 0, inner = 0, total = 0;
        if (!index_select_compute_geometry(shape, rank, dim, idx_size, &outer, &inner, &total)) {
            return;
        }

        DeviceFaultRecord* fault = nullptr;
        std::uint32_t op_id = 0;
        bool trap_after_record = false;
        prepare_index_select_device_fault(boundary, stream, "tensor.masking.index_select_i32",
                                          &fault, &op_id, &trap_after_record);

        size_t num_blocks = (total + 255) / 256;
        const size_t max_blocks_x = 65535;
        if (num_blocks <= max_blocks_x) {
            index_select_kernel<int32_t><<<num_blocks, 256, 0, stream>>>(
                in, idx, out, outer, shape[dim], inner, idx_size, boundary,
                fault, op_id, trap_after_record);
            LFS_CUDA_LAUNCH_CHECK(stream, "tensor.masking.index_select_i32");
        } else {
            dim3 grid(std::min(num_blocks, max_blocks_x),
                      (num_blocks + max_blocks_x - 1) / max_blocks_x);
            index_select_kernel<int32_t><<<grid, 256, 0, stream>>>(
                in, idx, out, outer, shape[dim], inner, idx_size, boundary,
                fault, op_id, trap_after_record);
            LFS_CUDA_LAUNCH_CHECK(stream, "tensor.masking.index_select_i32");
        }
        finish_index_select_device_fault(fault, stream, "tensor.masking.index_select_i32");
    }

    // UInt8 overload
    void launch_index_select(const uint8_t* in, const int* idx, uint8_t* out,
                             const size_t* shape, size_t rank, int dim,
                             size_t idx_size, int boundary, cudaStream_t stream) {
        size_t outer = 0, inner = 0, total = 0;
        if (!index_select_compute_geometry(shape, rank, dim, idx_size, &outer, &inner, &total)) {
            return;
        }

        DeviceFaultRecord* fault = nullptr;
        std::uint32_t op_id = 0;
        bool trap_after_record = false;
        prepare_index_select_device_fault(boundary, stream, "tensor.masking.index_select_u8",
                                          &fault, &op_id, &trap_after_record);

        size_t num_blocks = (total + 255) / 256;
        const size_t max_blocks_x = 65535;
        if (num_blocks <= max_blocks_x) {
            index_select_kernel<uint8_t><<<num_blocks, 256, 0, stream>>>(
                in, idx, out, outer, shape[dim], inner, idx_size, boundary,
                fault, op_id, trap_after_record);
            LFS_CUDA_LAUNCH_CHECK(stream, "tensor.masking.index_select_u8");
        } else {
            dim3 grid(std::min(num_blocks, max_blocks_x),
                      (num_blocks + max_blocks_x - 1) / max_blocks_x);
            index_select_kernel<uint8_t><<<grid, 256, 0, stream>>>(
                in, idx, out, outer, shape[dim], inner, idx_size, boundary,
                fault, op_id, trap_after_record);
            LFS_CUDA_LAUNCH_CHECK(stream, "tensor.masking.index_select_u8");
        }
        finish_index_select_device_fault(fault, stream, "tensor.masking.index_select_u8");
    }

    template <typename T>
    __global__ void gather_kernel(const T* in, const int* idx, T* out,
                                  const size_t* in_shape, const size_t* idx_shape,
                                  size_t in_rank, size_t idx_rank, int dim, size_t total, int boundary) {
        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= total)
            return;

        size_t in_strides[MAX_TENSOR_RANK];
        in_strides[in_rank - 1] = 1;
        for (int i = in_rank - 2; i >= 0; --i) {
            in_strides[i] = in_strides[i + 1] * in_shape[i + 1];
        }

        size_t out_strides[MAX_TENSOR_RANK];
        out_strides[idx_rank - 1] = 1;
        for (int i = idx_rank - 2; i >= 0; --i) {
            out_strides[i] = out_strides[i + 1] * idx_shape[i + 1];
        }

        size_t out_coords[MAX_TENSOR_RANK] = {0};
        size_t temp = tid;
        for (size_t d = 0; d < idx_rank; ++d) {
            out_coords[d] = temp / out_strides[d];
            temp %= out_strides[d];
        }

        int gather_idx = idx[tid];

        if (boundary == 1) {
            gather_idx = max(0, min((int)in_shape[dim] - 1, gather_idx));
        } else if (boundary == 2) {
            gather_idx = ((gather_idx % (int)in_shape[dim]) + in_shape[dim]) % in_shape[dim];
        } else if (gather_idx < 0 || gather_idx >= in_shape[dim]) {
            LFS_DEBUG_ASSERT_MSG(gather_idx >= 0 && gather_idx < static_cast<int>(in_shape[dim]),
                                 detail::format_cuda_safe("gather index must be in range "
                                                          "(gather_index={}, dimension={}, dimension_size={}, "
                                                          "output_index={}, boundary_mode={})",
                                                          gather_idx, dim, in_shape[dim], tid, boundary));
            out[tid] = 0;
            return;
        }

        size_t src_idx = 0;
        for (size_t d = 0; d < in_rank; ++d) {
            size_t coord;
            if (d == dim) {
                coord = gather_idx;
            } else if (d < idx_rank) {
                coord = out_coords[d];
            } else {
                coord = 0;
            }

            if (coord >= in_shape[d]) {
                LFS_DEBUG_ASSERT_MSG(coord < in_shape[d],
                                     detail::format_cuda_safe("gather coordinate must be in range "
                                                              "(dimension={}, coordinate={}, dimension_size={}, "
                                                              "output_index={}, gather_dimension={})",
                                                              d, coord, in_shape[d], tid, dim));
                out[tid] = 0;
                return;
            }

            src_idx += coord * in_strides[d];
        }

        size_t input_elements = 1;
        for (size_t d = 0; d < in_rank; ++d) {
            input_elements *= in_shape[d];
        }
        LFS_DEBUG_ASSERT_MSG(src_idx < input_elements,
                             detail::format_cuda_safe("gather source offset must be in range "
                                                      "(source_offset={}, source_numel={}, output_index={}, "
                                                      "input_rank={}, gather_dimension={})",
                                                      src_idx, input_elements, tid, in_rank, dim));
        out[tid] = in[src_idx];
    }

    void launch_gather(const float* in, const int* idx, float* out,
                       const size_t* in_shape, const size_t* idx_shape,
                       size_t rank, int dim, size_t total, int boundary, cudaStream_t stream) {
        LFS_ASSERT_MSG(rank > 0 && rank <= MAX_TENSOR_RANK,
                       "gather rank exceeds MAX_TENSOR_RANK");
        if (total == 0) {
            return;
        }
        CudaDeviceMemory<size_t> d_in_shape(rank);
        CudaDeviceMemory<size_t> d_idx_shape(rank);
        LFS_CUDA_CHECK(d_in_shape.copy_from_host(in_shape, rank));
        LFS_CUDA_CHECK(d_idx_shape.copy_from_host(idx_shape, rank));

        int blocks = (total + 255) / 256;
        gather_kernel<float><<<blocks, 256, 0, stream>>>(
            in, idx, out, d_in_shape.get(), d_idx_shape.get(),
            rank, rank, dim, total, boundary);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.masking.gather_f32");
    }

    void launch_gather(const int64_t* in, const int* idx, int64_t* out,
                       const size_t* in_shape, const size_t* idx_shape,
                       size_t rank, int dim, size_t total, int boundary, cudaStream_t stream) {
        LFS_ASSERT_MSG(rank > 0 && rank <= MAX_TENSOR_RANK,
                       "gather rank exceeds MAX_TENSOR_RANK");
        if (total == 0) {
            return;
        }
        CudaDeviceMemory<size_t> d_in_shape(rank);
        CudaDeviceMemory<size_t> d_idx_shape(rank);
        LFS_CUDA_CHECK(d_in_shape.copy_from_host(in_shape, rank));
        LFS_CUDA_CHECK(d_idx_shape.copy_from_host(idx_shape, rank));

        int blocks = (total + 255) / 256;
        gather_kernel<int64_t><<<blocks, 256, 0, stream>>>(
            in, idx, out, d_in_shape.get(), d_idx_shape.get(),
            rank, rank, dim, total, boundary);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.masking.gather_i64");
    }

    void launch_take(const float* in, const int* idx, float* out,
                     size_t in_size, size_t out_size, cudaStream_t stream) {
        auto in_ptr = thrust::device_pointer_cast(in);
        auto idx_ptr = thrust::device_pointer_cast(idx);
        auto out_ptr = thrust::device_pointer_cast(out);
        auto transform_idx = thrust::make_transform_iterator(idx_ptr,
                                                             ops::index_clamp_op(in_size));
        thrust::gather(thrust::cuda::par.on(stream), transform_idx, transform_idx + out_size,
                       in_ptr, out_ptr);
    }

    // ============= OPTIMIZED: Fused Gather + Unary Operation =============
    // This uses thrust::permutation_iterator for ZERO-COPY gather combined with
    // thrust::transform for fusion - inspired by NVIDIA's parrot library
    template <typename UnaryOp>
    void launch_gather_fused_unary(const float* in, const int* idx, float* out,
                                   size_t in_size, size_t out_size,
                                   UnaryOp op, cudaStream_t stream) {
        auto in_ptr = thrust::device_pointer_cast(in);
        auto idx_ptr = thrust::device_pointer_cast(idx);
        auto out_ptr = thrust::device_pointer_cast(out);

        // Clamp indices to valid range
        auto clamped_idx = thrust::make_transform_iterator(idx_ptr,
                                                           ops::index_clamp_op(in_size));

        // Create zero-copy permutation view: applies gather WITHOUT materializing
        auto permuted_view = thrust::make_permutation_iterator(in_ptr, clamped_idx);

        // Single fused kernel: gather + unary operation!
        thrust::transform(thrust::cuda::par.on(stream),
                          permuted_view, permuted_view + out_size,
                          out_ptr,
                          op);
    }

    template <typename T>
    __device__ inline void scatter_add(T* dst, T value) {
        atomicAdd(dst, value);
    }

    template <>
    __device__ inline void scatter_add<uint8_t>(uint8_t* dst, uint8_t value) {
        *dst = static_cast<uint8_t>(*dst + value);
    }

    template <typename T>
    __global__ void scatter_kernel(T* out, const int* idx, const T* in,
                                   size_t outer, size_t dim_sz, size_t inner,
                                   size_t idx_sz, int mode) {
        // Support both 1D and 2D grids for large arrays
        size_t block_id = blockIdx.y * gridDim.x + blockIdx.x;
        size_t tid = block_id * blockDim.x + threadIdx.x;
        size_t n = outer * idx_sz * inner;
        if (tid >= n)
            return;

        size_t outer_idx = tid / (idx_sz * inner);
        size_t idx_pos = (tid / inner) % idx_sz;
        size_t inner_idx = tid % inner;

        int scatter_idx = idx[idx_pos];
        if (scatter_idx < 0 || scatter_idx >= dim_sz) {
            LFS_DEBUG_ASSERT_MSG(scatter_idx >= 0 && scatter_idx < static_cast<int>(dim_sz),
                                 detail::format_cuda_safe("scatter index must be in range "
                                                          "(scatter_index={}, dimension_size={}, "
                                                          "index_position={}, input_index={}, mode={})",
                                                          scatter_idx, dim_sz, idx_pos, tid, mode));
            return;
        }

        size_t dst_idx = outer_idx * dim_sz * inner + scatter_idx * inner + inner_idx;
        LFS_DEBUG_ASSERT_MSG(dst_idx < outer * dim_sz * inner,
                             detail::format_cuda_safe("scatter destination offset must be in range "
                                                      "(destination_offset={}, destination_numel={}, outer={}, "
                                                      "dimension_size={}, inner={}, input_index={})",
                                                      dst_idx, outer * dim_sz * inner, outer,
                                                      dim_sz, inner, tid));

        if (mode == 1) {
            scatter_add(&out[dst_idx], in[tid]);
        } else {
            out[dst_idx] = in[tid];
        }
    }

    template <typename T>
    void launch_scatter(T* out, const int* idx, const T* in,
                        const size_t* out_shape, const size_t* in_shape,
                        size_t rank, int dim, size_t /*total*/, int mode, cudaStream_t stream) {
        LFS_ASSERT_MSG(rank > 0 && rank <= MAX_TENSOR_RANK,
                       "scatter rank exceeds MAX_TENSOR_RANK");
        size_t outer = 1, inner = 1;
        for (int i = 0; i < dim; ++i)
            outer *= out_shape[i];
        for (size_t i = dim + 1; i < rank; ++i)
            inner *= out_shape[i];

        const size_t idx_count = in_shape[dim];
        const size_t total_threads = outer * idx_count * inner;
        const size_t num_blocks = (total_threads + 255) / 256;
        constexpr size_t MAX_BLOCKS_X = 65535;

        if (num_blocks <= MAX_BLOCKS_X) {
            scatter_kernel<T><<<num_blocks, 256, 0, stream>>>(
                out, idx, in, outer, out_shape[dim], inner, in_shape[dim], mode);
            LFS_CUDA_LAUNCH_CHECK(stream, "tensor.masking.scatter");
        } else {
            const dim3 grid(std::min(num_blocks, MAX_BLOCKS_X),
                            (num_blocks + MAX_BLOCKS_X - 1) / MAX_BLOCKS_X);
            scatter_kernel<T><<<grid, 256, 0, stream>>>(
                out, idx, in, outer, out_shape[dim], inner, in_shape[dim], mode);
            LFS_CUDA_LAUNCH_CHECK(stream, "tensor.masking.scatter");
        }
    }

    template <typename T>
    void launch_index_fill(T* data, const int* idx, T val,
                           const size_t* shape, size_t rank, int dim,
                           size_t n_idx, cudaStream_t stream) {
        CudaDeviceMemory<T> val_buffer(n_idx);
        auto val_ptr = thrust::device_pointer_cast(val_buffer.get());
        thrust::fill(thrust::cuda::par.on(stream), val_ptr, val_ptr + n_idx, val);

        size_t in_shape[MAX_TENSOR_RANK] = {0};
        std::copy(shape, shape + rank, in_shape);
        in_shape[dim] = n_idx;
        launch_scatter<T>(data, idx, val_buffer.get(), shape, in_shape, rank, dim, n_idx, 0, stream);
    }

    template <typename T>
    void launch_index_copy(T* dst, const int* idx, const T* src,
                           const size_t* shape, size_t rank, int dim,
                           size_t n_idx, cudaStream_t stream) {
        size_t in_shape[MAX_TENSOR_RANK] = {0};
        std::copy(shape, shape + rank, in_shape);
        in_shape[dim] = n_idx;
        launch_scatter<T>(dst, idx, src, shape, in_shape, rank, dim, n_idx, 0, stream);
    }

    template <typename T>
    void launch_index_add(T* dst, const int* idx, const T* src,
                          const size_t* shape, size_t rank, int dim,
                          size_t n_idx, cudaStream_t stream) {
        size_t in_shape[MAX_TENSOR_RANK] = {0};
        std::copy(shape, shape + rank, in_shape);
        in_shape[dim] = n_idx;
        launch_scatter<T>(dst, idx, src, shape, in_shape, rank, dim, n_idx, 1, stream);
    }

    void launch_index_put(float* data, const int* idx, const float* vals,
                          size_t data_size, size_t idx_size, cudaStream_t stream) {
        auto data_ptr = thrust::device_pointer_cast(data);
        auto idx_ptr = thrust::device_pointer_cast(idx);
        auto vals_ptr = thrust::device_pointer_cast(vals);
        auto transform_idx = thrust::make_transform_iterator(idx_ptr,
                                                             ops::index_clamp_op(data_size));
        thrust::scatter(thrust::cuda::par.on(stream), vals_ptr, vals_ptr + idx_size,
                        transform_idx, data_ptr);
    }

    // ============= Nonzero Operations =============

    size_t launch_nonzero(const float* data, int64_t* indices, size_t n, size_t output_size, cudaStream_t stream) {
        if (n == 0 || output_size == 0)
            return 0;
        auto data_ptr = thrust::device_pointer_cast(data);
        auto indices_ptr = thrust::device_pointer_cast(indices);
        auto counting = thrust::counting_iterator<int64_t>(0);
        auto end_it = thrust::copy_if(thrust::cuda::par.on(stream), counting, counting + n, data_ptr,
                                      indices_ptr, ops::nonzero_predicate<float>());
        // Return actual count (fixes potential mismatch)
        return end_it - indices_ptr;
    }

    size_t launch_nonzero_bool(const unsigned char* data, int64_t* indices, size_t n, size_t output_size, cudaStream_t stream) {
        if (n == 0 || output_size == 0)
            return 0;
        auto data_ptr = thrust::device_pointer_cast(data);
        auto indices_ptr = thrust::device_pointer_cast(indices);
        auto counting = thrust::counting_iterator<int64_t>(0);
        auto end_it = thrust::copy_if(thrust::cuda::par.on(stream), counting, counting + n, data_ptr,
                                      indices_ptr, ops::nonzero_bool_predicate());
        // Return actual count (fixes potential mismatch)
        return end_it - indices_ptr;
    }

    // ============= Multi-Tensor Gather (Zip Gather) =============
    // Gather from multiple tensors simultaneously using the same indices
    // Uses zip_iterator to fuse multiple gathers into single memory transaction

    void launch_zip_gather_2(const float* input1, const float* input2,
                             const int* indices,
                             float* output1, float* output2,
                             size_t input_size, size_t index_size,
                             size_t stride1, size_t stride2,
                             cudaStream_t stream) {
        if (input_size == 0 || index_size == 0)
            return;

        auto in1_ptr = thrust::device_pointer_cast(input1);
        auto in2_ptr = thrust::device_pointer_cast(input2);
        auto idx_ptr = thrust::device_pointer_cast(indices);
        auto out1_ptr = thrust::device_pointer_cast(output1);
        auto out2_ptr = thrust::device_pointer_cast(output2);

        // Clamp indices to valid range
        auto clamped_idx = thrust::make_transform_iterator(idx_ptr,
                                                           ops::index_clamp_op(input_size));

        auto input1_idx = thrust::make_transform_iterator(clamped_idx, ops::index_stride_op(stride1));
        auto input2_idx = thrust::make_transform_iterator(clamped_idx, ops::index_stride_op(stride2));
        auto gathered1 = thrust::make_permutation_iterator(in1_ptr, input1_idx);
        auto gathered2 = thrust::make_permutation_iterator(in2_ptr, input2_idx);
        auto gathered = thrust::make_zip_iterator(thrust::make_tuple(gathered1, gathered2));

        // Create zip iterator for outputs
        auto zipped_output = thrust::make_zip_iterator(
            thrust::make_tuple(out1_ptr, out2_ptr));

        // Single gather operation copies both tensors!
        thrust::copy(thrust::cuda::par.on(stream),
                     gathered, gathered + index_size,
                     zipped_output);
    }

    void launch_zip_gather_3(const float* input1, const float* input2, const float* input3,
                             const int* indices,
                             float* output1, float* output2, float* output3,
                             size_t input_size, size_t index_size,
                             size_t stride1, size_t stride2, size_t stride3,
                             cudaStream_t stream) {
        if (input_size == 0 || index_size == 0)
            return;

        auto in1_ptr = thrust::device_pointer_cast(input1);
        auto in2_ptr = thrust::device_pointer_cast(input2);
        auto in3_ptr = thrust::device_pointer_cast(input3);
        auto idx_ptr = thrust::device_pointer_cast(indices);
        auto out1_ptr = thrust::device_pointer_cast(output1);
        auto out2_ptr = thrust::device_pointer_cast(output2);
        auto out3_ptr = thrust::device_pointer_cast(output3);

        // Clamp indices
        auto clamped_idx = thrust::make_transform_iterator(idx_ptr,
                                                           ops::index_clamp_op(input_size));

        auto input1_idx = thrust::make_transform_iterator(clamped_idx, ops::index_stride_op(stride1));
        auto input2_idx = thrust::make_transform_iterator(clamped_idx, ops::index_stride_op(stride2));
        auto input3_idx = thrust::make_transform_iterator(clamped_idx, ops::index_stride_op(stride3));
        auto gathered1 = thrust::make_permutation_iterator(in1_ptr, input1_idx);
        auto gathered2 = thrust::make_permutation_iterator(in2_ptr, input2_idx);
        auto gathered3 = thrust::make_permutation_iterator(in3_ptr, input3_idx);
        auto gathered = thrust::make_zip_iterator(thrust::make_tuple(gathered1, gathered2, gathered3));

        // Zip three output sequences
        auto zipped_output = thrust::make_zip_iterator(
            thrust::make_tuple(out1_ptr, out2_ptr, out3_ptr));

        // Single gather for all three tensors!
        thrust::copy(thrust::cuda::par.on(stream),
                     gathered, gathered + index_size,
                     zipped_output);
    }

    // ============= Explicit Instantiations for Fused Gather =============
    // We need to explicitly instantiate the common functor types used with gather
    template LFS_CORE_API void launch_gather_fused_unary<ops::abs_op>(const float*, const int*, float*, size_t, size_t, ops::abs_op, cudaStream_t);
    template LFS_CORE_API void launch_gather_fused_unary<ops::sqrt_op>(const float*, const int*, float*, size_t, size_t, ops::sqrt_op, cudaStream_t);
    template LFS_CORE_API void launch_gather_fused_unary<ops::neg_op>(const float*, const int*, float*, size_t, size_t, ops::neg_op, cudaStream_t);

    // ============= Explicit Instantiations for Scatter Operations =============
    // Instantiate for float, int, and byte-sized mask types
    template LFS_CORE_API void launch_scatter<float>(float*, const int*, const float*, const size_t*, const size_t*, size_t, int, size_t, int, cudaStream_t);
    template LFS_CORE_API void launch_scatter<int>(int*, const int*, const int*, const size_t*, const size_t*, size_t, int, size_t, int, cudaStream_t);
    template LFS_CORE_API void launch_scatter<uint8_t>(uint8_t*, const int*, const uint8_t*, const size_t*, const size_t*, size_t, int, size_t, int, cudaStream_t);

    template LFS_CORE_API void launch_index_add<float>(float*, const int*, const float*, const size_t*, size_t, int, size_t, cudaStream_t);
    template LFS_CORE_API void launch_index_add<int>(int*, const int*, const int*, const size_t*, size_t, int, size_t, cudaStream_t);

    template LFS_CORE_API void launch_index_copy<float>(float*, const int*, const float*, const size_t*, size_t, int, size_t, cudaStream_t);
    template LFS_CORE_API void launch_index_copy<int>(int*, const int*, const int*, const size_t*, size_t, int, size_t, cudaStream_t);
    template LFS_CORE_API void launch_index_copy<uint8_t>(uint8_t*, const int*, const uint8_t*, const size_t*, size_t, int, size_t, cudaStream_t);

    template LFS_CORE_API void launch_index_fill<float>(float*, const int*, float, const size_t*, size_t, int, size_t, cudaStream_t);
    template LFS_CORE_API void launch_index_fill<int>(int*, const int*, int, const size_t*, size_t, int, size_t, cudaStream_t);
    template LFS_CORE_API void launch_index_fill<uint8_t>(uint8_t*, const int*, uint8_t, const size_t*, size_t, int, size_t, cudaStream_t);

} // namespace lfs::core::tensor_ops
