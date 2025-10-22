/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/cuda_memory_guard.hpp"
#include "internal/tensor_functors.hpp"
#include "internal/tensor_ops.hpp"
#include <cfloat>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <limits>

// Thrust headers
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
                                      const size_t* shapes, size_t info, size_t total) {
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
            size_t a_idx = compute_broadcast_index(idx, shapes, a_rank, shapes + 20, c_rank);
            size_t b_idx = compute_broadcast_index(idx, shapes + 10, b_rank, shapes + 20, c_rank);
            c[idx] = (a[a_idx] == b[b_idx]) ? 1 : 0;
        }
    }

    __global__ void compare_lt_kernel(const float* a, const float* b, unsigned char* c,
                                      const size_t* shapes, size_t info, size_t total) {
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
            size_t a_idx = compute_broadcast_index(idx, shapes, a_rank, shapes + 20, c_rank);
            size_t b_idx = compute_broadcast_index(idx, shapes + 10, b_rank, shapes + 20, c_rank);
            c[idx] = (a[a_idx] < b[b_idx]) ? 1 : 0;
        }
    }

    __global__ void compare_gt_kernel(const float* a, const float* b, unsigned char* c,
                                      const size_t* shapes, size_t info, size_t total) {
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
            size_t a_idx = compute_broadcast_index(idx, shapes, a_rank, shapes + 20, c_rank);
            size_t b_idx = compute_broadcast_index(idx, shapes + 10, b_rank, shapes + 20, c_rank);
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
                                       unsigned char* c, const size_t* shapes,
                                       size_t info, size_t total) {
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
            size_t a_idx = compute_broadcast_index(idx, shapes, a_rank, shapes + 20, c_rank);
            size_t b_idx = compute_broadcast_index(idx, shapes + 10, b_rank, shapes + 20, c_rank);
            c[idx] = (a[a_idx] && b[b_idx]) ? 1 : 0;
        }
    }

    __global__ void logical_or_kernel(const unsigned char* a, const unsigned char* b,
                                      unsigned char* c, const size_t* shapes,
                                      size_t info, size_t total) {
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
            size_t a_idx = compute_broadcast_index(idx, shapes, a_rank, shapes + 20, c_rank);
            size_t b_idx = compute_broadcast_index(idx, shapes + 10, b_rank, shapes + 20, c_rank);
            c[idx] = (a[a_idx] || b[b_idx]) ? 1 : 0;
        }
    }

    __global__ void logical_xor_kernel(const unsigned char* a, const unsigned char* b,
                                       unsigned char* c, const size_t* shapes,
                                       size_t info, size_t total) {
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
            size_t a_idx = compute_broadcast_index(idx, shapes, a_rank, shapes + 20, c_rank);
            size_t b_idx = compute_broadcast_index(idx, shapes + 10, b_rank, shapes + 20, c_rank);
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
        static thread_local CudaDeviceMemory<size_t> shapes(30);
        size_t h_shapes[30] = {0};
        std::copy(a_shape, a_shape + a_rank, h_shapes);
        std::copy(b_shape, b_shape + b_rank, h_shapes + 10);
        std::copy(c_shape, c_shape + c_rank, h_shapes + 20);
        shapes.copy_from_host(h_shapes, 30);

        bool fast_path = (a_rank == c_rank && b_rank == c_rank &&
                          std::equal(a_shape, a_shape + a_rank, c_shape) &&
                          std::equal(b_shape, b_shape + b_rank, c_shape));
        size_t info = a_rank | (b_rank << 5) | (c_rank << 10) | (fast_path << 15);

        int blocks = (c_elements + 255) / 256;
        compare_eq_kernel<<<blocks, 256, 0, stream>>>(a, b, c, shapes.get(), info, c_elements);
    }

    void launch_compare_lt(const float* a, const float* b, unsigned char* c,
                           const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                           size_t a_rank, size_t b_rank, size_t c_rank,
                           size_t c_elements, cudaStream_t stream) {
        static thread_local CudaDeviceMemory<size_t> shapes(30);
        size_t h_shapes[30] = {0};
        std::copy(a_shape, a_shape + a_rank, h_shapes);
        std::copy(b_shape, b_shape + b_rank, h_shapes + 10);
        std::copy(c_shape, c_shape + c_rank, h_shapes + 20);
        shapes.copy_from_host(h_shapes, 30);

        bool fast_path = (a_rank == c_rank && b_rank == c_rank &&
                          std::equal(a_shape, a_shape + a_rank, c_shape) &&
                          std::equal(b_shape, b_shape + b_rank, c_shape));
        size_t info = a_rank | (b_rank << 5) | (c_rank << 10) | (fast_path << 15);

        int blocks = (c_elements + 255) / 256;
        compare_lt_kernel<<<blocks, 256, 0, stream>>>(a, b, c, shapes.get(), info, c_elements);
    }

    void launch_compare_gt(const float* a, const float* b, unsigned char* c,
                           const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                           size_t a_rank, size_t b_rank, size_t c_rank,
                           size_t c_elements, cudaStream_t stream) {
        static thread_local CudaDeviceMemory<size_t> shapes(30);
        size_t h_shapes[30] = {0};
        std::copy(a_shape, a_shape + a_rank, h_shapes);
        std::copy(b_shape, b_shape + b_rank, h_shapes + 10);
        std::copy(c_shape, c_shape + c_rank, h_shapes + 20);
        shapes.copy_from_host(h_shapes, 30);

        bool fast_path = (a_rank == c_rank && b_rank == c_rank &&
                          std::equal(a_shape, a_shape + a_rank, c_shape) &&
                          std::equal(b_shape, b_shape + b_rank, c_shape));
        size_t info = a_rank | (b_rank << 5) | (c_rank << 10) | (fast_path << 15);

        int blocks = (c_elements + 255) / 256;
        compare_gt_kernel<<<blocks, 256, 0, stream>>>(a, b, c, shapes.get(), info, c_elements);
    }

    void launch_logical_and(const unsigned char* a, const unsigned char* b, unsigned char* c,
                            const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                            size_t a_rank, size_t b_rank, size_t c_rank,
                            size_t c_elements, cudaStream_t stream) {
        static thread_local CudaDeviceMemory<size_t> shapes(30);
        size_t h_shapes[30] = {0};
        std::copy(a_shape, a_shape + a_rank, h_shapes);
        std::copy(b_shape, b_shape + b_rank, h_shapes + 10);
        std::copy(c_shape, c_shape + c_rank, h_shapes + 20);
        shapes.copy_from_host(h_shapes, 30);

        bool fast_path = (a_rank == c_rank && b_rank == c_rank &&
                          std::equal(a_shape, a_shape + a_rank, c_shape) &&
                          std::equal(b_shape, b_shape + b_rank, c_shape));
        size_t info = a_rank | (b_rank << 5) | (c_rank << 10) | (fast_path << 15);

        int blocks = (c_elements + 255) / 256;
        logical_and_kernel<<<blocks, 256, 0, stream>>>(a, b, c, shapes.get(), info, c_elements);
    }

    void launch_logical_or(const unsigned char* a, const unsigned char* b, unsigned char* c,
                           const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                           size_t a_rank, size_t b_rank, size_t c_rank,
                           size_t c_elements, cudaStream_t stream) {
        static thread_local CudaDeviceMemory<size_t> shapes(30);
        size_t h_shapes[30] = {0};
        std::copy(a_shape, a_shape + a_rank, h_shapes);
        std::copy(b_shape, b_shape + b_rank, h_shapes + 10);
        std::copy(c_shape, c_shape + c_rank, h_shapes + 20);
        shapes.copy_from_host(h_shapes, 30);

        bool fast_path = (a_rank == c_rank && b_rank == c_rank &&
                          std::equal(a_shape, a_shape + a_rank, c_shape) &&
                          std::equal(b_shape, b_shape + b_rank, c_shape));
        size_t info = a_rank | (b_rank << 5) | (c_rank << 10) | (fast_path << 15);

        int blocks = (c_elements + 255) / 256;
        logical_or_kernel<<<blocks, 256, 0, stream>>>(a, b, c, shapes.get(), info, c_elements);
    }

    void launch_logical_xor(const unsigned char* a, const unsigned char* b, unsigned char* c,
                            const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                            size_t a_rank, size_t b_rank, size_t c_rank,
                            size_t c_elements, cudaStream_t stream) {
        static thread_local CudaDeviceMemory<size_t> shapes(30);
        size_t h_shapes[30] = {0};
        std::copy(a_shape, a_shape + a_rank, h_shapes);
        std::copy(b_shape, b_shape + b_rank, h_shapes + 10);
        std::copy(c_shape, c_shape + c_rank, h_shapes + 20);
        shapes.copy_from_host(h_shapes, 30);

        bool fast_path = (a_rank == c_rank && b_rank == c_rank &&
                          std::equal(a_shape, a_shape + a_rank, c_shape) &&
                          std::equal(b_shape, b_shape + b_rank, c_shape));
        size_t info = a_rank | (b_rank << 5) | (c_rank << 10) | (fast_path << 15);

        int blocks = (c_elements + 255) / 256;
        logical_xor_kernel<<<blocks, 256, 0, stream>>>(a, b, c, shapes.get(), info, c_elements);
    }

    // ============= Masking Operations =============
    void launch_masked_fill(float* data, const unsigned char* mask, float val, size_t n, cudaStream_t s) {
        auto data_ptr = thrust::device_pointer_cast(data);
        auto mask_ptr = thrust::device_pointer_cast(mask);
        auto begin = thrust::make_zip_iterator(thrust::make_tuple(data_ptr, mask_ptr));
        auto end = thrust::make_zip_iterator(thrust::make_tuple(data_ptr + n, mask_ptr + n));
        thrust::transform(thrust::cuda::par.on(s), begin, end, data_ptr,
                          ops::masked_fill_op<float>(val));
    }

    void launch_masked_select(const float* input, const unsigned char* mask,
                              float* output, size_t n, size_t output_size, cudaStream_t stream) {
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
                        transform_begin, transform_end, mask_begin, output_ptr, thrust::identity<bool>());
    }

    __global__ void masked_scatter_compact_kernel(float* data, const unsigned char* mask,
                                                  const float* src, const int* scan, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n && mask[idx]) {
            data[idx] = src[scan[idx]];
        }
    }

    void launch_masked_scatter(float* data, const unsigned char* mask,
                               const float* src, size_t n, size_t src_size, cudaStream_t stream) {
        if (n == 0 || src_size == 0)
            return;

        CudaDeviceMemory<int> scan_result(n);
        if (!scan_result.valid())
            return;

        size_t temp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, mask, scan_result.get(), n, stream);

        CudaDeviceMemory<uint8_t> temp_storage(temp_bytes);
        if (!temp_storage.valid())
            return;

        cub::DeviceScan::ExclusiveSum(temp_storage.get(), temp_bytes,
                                      mask, scan_result.get(), n, stream);

        if (stream != 0)
            cudaStreamSynchronize(stream);

        int blocks = (n + 255) / 256;
        masked_scatter_compact_kernel<<<blocks, 256, 0, stream>>>(
            data, mask, src, scan_result.get(), n);
    }

    // ============= Where Operation =============
    __global__ void where_kernel(const unsigned char* cond, const float* x, const float* y,
                                 float* r, const size_t* shapes,
                                 size_t cr, size_t xr, size_t yr, size_t rr, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n)
            return;

        size_t c_idx = compute_broadcast_index(idx, shapes, cr, shapes + 30, rr);
        size_t x_idx = compute_broadcast_index(idx, shapes + 10, xr, shapes + 30, rr);
        size_t y_idx = compute_broadcast_index(idx, shapes + 20, yr, shapes + 30, rr);

        r[idx] = cond[c_idx] ? x[x_idx] : y[y_idx];
    }

    void launch_where(const unsigned char* cond, const float* x, const float* y, float* r,
                      const size_t* cond_shape, const size_t* x_shape,
                      const size_t* y_shape, const size_t* r_shape,
                      size_t cond_rank, size_t x_rank, size_t y_rank, size_t r_rank,
                      size_t total, cudaStream_t stream) {

        static thread_local CudaDeviceMemory<size_t> shapes(40);
        size_t h_shapes[40] = {0};
        std::copy(cond_shape, cond_shape + cond_rank, h_shapes);
        std::copy(x_shape, x_shape + x_rank, h_shapes + 10);
        std::copy(y_shape, y_shape + y_rank, h_shapes + 20);
        std::copy(r_shape, r_shape + r_rank, h_shapes + 30);
        shapes.copy_from_host(h_shapes, 40);

        where_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
            cond, x, y, r, shapes.get(), cond_rank, x_rank, y_rank, r_rank, total);
    }

    // ============= Count Nonzero =============
    void launch_count_nonzero_bool(const unsigned char* data, size_t* count, size_t n, cudaStream_t stream) {
        if (n == 0) {
            cudaMemsetAsync(count, 0, sizeof(size_t), stream);
            return;
        }

        auto data_ptr = thrust::device_pointer_cast(data);

        // Thrust count_if returns to host, so we get it first
        size_t result = thrust::count_if(thrust::cuda::par.on(stream),
                                         data_ptr, data_ptr + n,
                                         ops::is_nonzero_bool_op());

        // Then write to device memory
        cudaMemcpyAsync(count, &result, sizeof(size_t), cudaMemcpyHostToDevice, stream);

        // Ensure the copy completes before we return (and potentially destroy 'result')
        if (stream) {
            cudaStreamSynchronize(stream);
        }
    }

    void launch_count_nonzero_float(const float* data, size_t* count, size_t n, cudaStream_t stream) {
        if (n == 0) {
            cudaMemsetAsync(count, 0, sizeof(size_t), stream);
            return;
        }

        auto data_ptr = thrust::device_pointer_cast(data);

        // Thrust count_if returns to host, so we get it first
        size_t result = thrust::count_if(thrust::cuda::par.on(stream),
                                         data_ptr, data_ptr + n,
                                         ops::is_nonzero_op<float>());

        // Then write to device memory
        cudaMemcpyAsync(count, &result, sizeof(size_t), cudaMemcpyHostToDevice, stream);

        // Ensure the copy completes before we return (and potentially destroy 'result')
        if (stream) {
            cudaStreamSynchronize(stream);
        }
    }

    // ============= Index Operations =============
    __global__ void index_select_kernel(const float* in, const int* idx, float* out,
                                        size_t outer, size_t dim_size, size_t inner,
                                        size_t idx_size, int boundary) {
        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
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
        else if (sel < 0 || sel >= dim_size) {
            out[tid] = 0;
            return;
        }

        out[tid] = in[o * dim_size * inner + sel * inner + j];
    }

    void launch_index_select(const float* in, const int* idx, float* out,
                             const size_t* shape, size_t rank, int dim,
                             size_t idx_size, int boundary, cudaStream_t stream) {
        size_t outer = 1, inner = 1;
        for (int i = 0; i < dim; ++i)
            outer *= shape[i];
        for (size_t i = dim + 1; i < rank; ++i)
            inner *= shape[i];
        size_t total = outer * idx_size * inner;
        index_select_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
            in, idx, out, outer, shape[dim], inner, idx_size, boundary);
    }

    __global__ void gather_kernel(const float* in, const int* idx, float* out,
                                  const size_t* in_shape, const size_t* idx_shape,
                                  size_t in_rank, size_t idx_rank, int dim, size_t total, int boundary) {
        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= total)
            return;

        size_t in_strides[10];
        in_strides[in_rank - 1] = 1;
        for (int i = in_rank - 2; i >= 0; --i) {
            in_strides[i] = in_strides[i + 1] * in_shape[i + 1];
        }

        size_t out_strides[10];
        out_strides[idx_rank - 1] = 1;
        for (int i = idx_rank - 2; i >= 0; --i) {
            out_strides[i] = out_strides[i + 1] * idx_shape[i + 1];
        }

        size_t out_coords[10] = {0};
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
                out[tid] = 0;
                return;
            }

            src_idx += coord * in_strides[d];
        }

        out[tid] = in[src_idx];
    }

    void launch_gather(const float* in, const int* idx, float* out,
                       const size_t* in_shape, const size_t* idx_shape,
                       size_t rank, int dim, size_t total, int boundary, cudaStream_t stream) {
        CudaDeviceMemory<size_t> d_in_shape(10);
        CudaDeviceMemory<size_t> d_idx_shape(10);

        size_t h_in_shape[10] = {0};
        size_t h_idx_shape[10] = {0};

        size_t idx_rank = rank;
        size_t idx_elements = 1;
        for (size_t i = 0; i < rank; ++i) {
            if (idx_shape[i] > 0) {
                h_idx_shape[i] = idx_shape[i];
                idx_elements *= idx_shape[i];
            } else {
                break;
            }
        }

        idx_rank = 0;
        size_t check_elements = 1;
        for (size_t i = 0; i < 10; ++i) {
            if (idx_shape[i] > 0) {
                check_elements *= idx_shape[i];
                idx_rank++;
                if (check_elements == total)
                    break;
            } else {
                break;
            }
        }

        if (idx_rank == 0)
            idx_rank = 1;

        for (size_t i = 0; i < rank; ++i) {
            h_in_shape[i] = in_shape[i];
        }

        d_in_shape.copy_from_host(h_in_shape, 10);
        d_idx_shape.copy_from_host(h_idx_shape, 10);

        int blocks = (total + 255) / 256;
        gather_kernel<<<blocks, 256, 0, stream>>>(
            in, idx, out, d_in_shape.get(), d_idx_shape.get(),
            rank, idx_rank, dim, total, boundary);
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

    __global__ void scatter_kernel(float* out, const int* idx, const float* in,
                                   size_t outer, size_t dim_sz, size_t inner,
                                   size_t idx_sz, int mode) {
        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        size_t n = outer * idx_sz * inner;
        if (tid >= n)
            return;

        size_t outer_idx = tid / (idx_sz * inner);
        size_t idx_pos = (tid / inner) % idx_sz;
        size_t inner_idx = tid % inner;

        int scatter_idx = idx[idx_pos];
        if (scatter_idx < 0 || scatter_idx >= dim_sz)
            return;

        size_t dst_idx = outer_idx * dim_sz * inner + scatter_idx * inner + inner_idx;

        if (mode == 1) {
            atomicAdd(&out[dst_idx], in[tid]);
        } else {
            out[dst_idx] = in[tid];
        }
    }

    void launch_scatter(float* out, const int* idx, const float* in,
                        const size_t* out_shape, const size_t* in_shape,
                        size_t rank, int dim, size_t total, int mode, cudaStream_t stream) {
        size_t outer = 1, inner = 1;
        for (int i = 0; i < dim; ++i)
            outer *= out_shape[i];
        for (size_t i = dim + 1; i < rank; ++i)
            inner *= out_shape[i];
        scatter_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
            out, idx, in, outer, out_shape[dim], inner, in_shape[dim], mode);
    }

    void launch_index_fill(float* data, const int* idx, float val,
                           const size_t* shape, size_t rank, int dim,
                           size_t n_idx, cudaStream_t stream) {
        CudaDeviceMemory<float> val_buffer(n_idx);
        auto val_ptr = thrust::device_pointer_cast(val_buffer.get());
        thrust::fill(thrust::cuda::par.on(stream), val_ptr, val_ptr + n_idx, val);

        size_t in_shape[10] = {0};
        std::copy(shape, shape + rank, in_shape);
        in_shape[dim] = n_idx;
        launch_scatter(data, idx, val_buffer.get(), shape, in_shape, rank, dim, n_idx, 0, stream);
    }

    void launch_index_copy(float* dst, const int* idx, const float* src,
                           const size_t* shape, size_t rank, int dim,
                           size_t n_idx, cudaStream_t stream) {
        size_t in_shape[10] = {0};
        std::copy(shape, shape + rank, in_shape);
        in_shape[dim] = n_idx;
        launch_scatter(dst, idx, src, shape, in_shape, rank, dim, n_idx, 0, stream);
    }

    void launch_index_add(float* dst, const int* idx, const float* src,
                          const size_t* shape, size_t rank, int dim,
                          size_t n_idx, cudaStream_t stream) {
        size_t in_shape[10] = {0};
        std::copy(shape, shape + rank, in_shape);
        in_shape[dim] = n_idx;
        launch_scatter(dst, idx, src, shape, in_shape, rank, dim, n_idx, 1, stream);
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
    void launch_nonzero(const float* data, int64_t* indices, size_t n, size_t output_size, cudaStream_t stream) {
        if (n == 0 || output_size == 0)
            return;
        auto data_ptr = thrust::device_pointer_cast(data);
        auto indices_ptr = thrust::device_pointer_cast(indices);
        auto counting = thrust::counting_iterator<int64_t>(0);
        thrust::copy_if(thrust::cuda::par.on(stream), counting, counting + n, data_ptr,
                        indices_ptr, ops::nonzero_predicate<float>());
    }

    void launch_nonzero_bool(const unsigned char* data, int64_t* indices, size_t n, size_t output_size, cudaStream_t stream) {
        if (n == 0 || output_size == 0)
            return;
        auto data_ptr = thrust::device_pointer_cast(data);
        auto indices_ptr = thrust::device_pointer_cast(indices);
        auto counting = thrust::counting_iterator<int64_t>(0);
        thrust::copy_if(thrust::cuda::par.on(stream), counting, counting + n, data_ptr,
                        indices_ptr, ops::nonzero_bool_predicate());
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
        auto in1_ptr = thrust::device_pointer_cast(input1);
        auto in2_ptr = thrust::device_pointer_cast(input2);
        auto idx_ptr = thrust::device_pointer_cast(indices);
        auto out1_ptr = thrust::device_pointer_cast(output1);
        auto out2_ptr = thrust::device_pointer_cast(output2);

        // Clamp indices to valid range
        auto clamped_idx = thrust::make_transform_iterator(idx_ptr,
                                                           ops::index_clamp_op(input_size));

        // Create zip iterator for inputs - combines two sequences
        auto zipped_input = thrust::make_zip_iterator(
            thrust::make_tuple(in1_ptr, in2_ptr));

        // Create permutation iterator - applies gather lazily
        auto permuted = thrust::make_permutation_iterator(zipped_input, clamped_idx);

        // Create zip iterator for outputs
        auto zipped_output = thrust::make_zip_iterator(
            thrust::make_tuple(out1_ptr, out2_ptr));

        // Single gather operation copies both tensors!
        thrust::copy(thrust::cuda::par.on(stream),
                     permuted, permuted + index_size,
                     zipped_output);
    }

    void launch_zip_gather_3(const float* input1, const float* input2, const float* input3,
                             const int* indices,
                             float* output1, float* output2, float* output3,
                             size_t input_size, size_t index_size,
                             size_t stride1, size_t stride2, size_t stride3,
                             cudaStream_t stream) {
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

        // Zip three input sequences
        auto zipped_input = thrust::make_zip_iterator(
            thrust::make_tuple(in1_ptr, in2_ptr, in3_ptr));

        // Permuted gather view
        auto permuted = thrust::make_permutation_iterator(zipped_input, clamped_idx);

        // Zip three output sequences
        auto zipped_output = thrust::make_zip_iterator(
            thrust::make_tuple(out1_ptr, out2_ptr, out3_ptr));

        // Single gather for all three tensors!
        thrust::copy(thrust::cuda::par.on(stream),
                     permuted, permuted + index_size,
                     zipped_output);
    }

    // ============= Explicit Instantiations for Fused Gather =============
    // We need to explicitly instantiate the common functor types used with gather
    template void launch_gather_fused_unary<ops::abs_op>(const float*, const int*, float*, size_t, size_t, ops::abs_op, cudaStream_t);
    template void launch_gather_fused_unary<ops::sqrt_op>(const float*, const int*, float*, size_t, size_t, ops::sqrt_op, cudaStream_t);
    template void launch_gather_fused_unary<ops::neg_op>(const float*, const int*, float*, size_t, size_t, ops::neg_op, cudaStream_t);

} // namespace lfs::core::tensor_ops