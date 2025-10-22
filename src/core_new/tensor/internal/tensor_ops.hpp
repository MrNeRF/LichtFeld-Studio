/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "tensor_functors.hpp"
#include <cuda_runtime.h>
#include <vector>

namespace lfs::core {
    // Forward declarations
    class Tensor;

    enum class Device : uint8_t;
    enum class DataType : uint8_t;
    enum class ReduceOp : uint8_t;
    enum class LoadOp : uint8_t;
} // namespace lfs::core

// ============= Generic CUDA Operations =============
// Include template implementation for inline instantiation
// Only include in CUDA compilation units - C++ files will link to .cu implementations
#ifdef __CUDACC__
#include "tensor_generic_ops.cuh"
#else
// Forward declaration for C++ files - implementation in tensor_ops.cu
namespace lfs::core::tensor_ops {
    template <typename InT, typename OutT, typename Op>
    void launch_binary_op_generic(const InT* a, const InT* b, OutT* c, size_t n,
                                  Op op, cudaStream_t stream = nullptr);

    template <typename T, typename OutT, typename Op>
    void launch_unary_op_generic(const T* input, OutT* output, size_t n,
                                 Op op, cudaStream_t stream = nullptr);

    template <typename T, typename OutputT, typename Op>
    void launch_scalar_op_generic(const T* data, T scalar, OutputT* result, size_t n,
                                  Op op, cudaStream_t stream = nullptr);
} // namespace lfs::core::tensor_ops
#endif

// ============= CPU Helpers (Generic, Header-Only) =============
namespace lfs::core {
    // CPU helper for unary operations
    template <typename T, typename OutT, typename Op>
    void apply_unary_cpu(const T* input, OutT* output, size_t n, Op op) {
        for (size_t i = 0; i < n; ++i) {
            output[i] = op(input[i]);
        }
    }

    // CPU helper for binary operations
    template <typename T, typename OutputT, typename Op>
    void apply_binary_cpu(const T* a, const T* b, OutputT* c, size_t n, Op op) {
        for (size_t i = 0; i < n; ++i) {
            c[i] = op(a[i], b[i]);
        }
    }
} // namespace lfs::core

namespace lfs::core::tensor_ops {

    // ============= Clamp Scalar Operations =============
    void launch_clamp_scalar(float* data, float min_val, float max_val, size_t n, cudaStream_t stream);
    void launch_clamp_fused(const float* src, float* dst, float min_val, float max_val, size_t n, cudaStream_t stream);
    void launch_clamp_scalar_int(int* data, int min_val, int max_val, size_t n, cudaStream_t stream);

    void launch_reduce_op(const void* input, void* output,
                          const size_t* shape, size_t rank,
                          const int* axes, size_t num_axes,
                          bool keepdim, ReduceOp op,
                          DataType dtype, cudaStream_t stream);

    // ============= WARP-LEVEL REDUCTIONS (OPTIMIZED) =============
    // Fast reductions using warp shuffle instructions (5-10x faster than CUB for small-medium tensors)
    void launch_warp_reduce_full(const float* input, float* output, size_t n,
                                 ReduceOp op, cudaStream_t stream);

    void launch_warp_segmented_reduce(const float* input, float* output,
                                      size_t num_segments, size_t segment_size,
                                      ReduceOp op, cudaStream_t stream);

    void launch_warp_strided_reduce(const float* input, float* output,
                                    size_t outer_size, size_t reduce_size, size_t inner_size,
                                    ReduceOp op, cudaStream_t stream);

    void launch_warp_multi_axis_reduce(const float* input, float* output,
                                       size_t output_size, size_t reduce_count,
                                       ReduceOp op, cudaStream_t stream);

    bool should_use_warp_reduce(size_t n, size_t num_segments);

    // ============= Load Operations =============
    void launch_load_op(void* output, const size_t* shape, size_t rank,
                        LoadOp op, const void* args,
                        DataType dtype, cudaStream_t stream);

    // Unified Type Conversion Template
    template <typename SrcT, typename DstT>
    void launch_convert_type(const SrcT* src, DstT* dst, size_t n, cudaStream_t stream);

    // ============= Broadcasting =============
    void launch_broadcast(const float* src, float* dst,
                          const size_t* src_shape, const size_t* dst_shape,
                          size_t src_rank, size_t dst_rank,
                          size_t dst_elements, cudaStream_t stream);

    void launch_broadcast_bool(const unsigned char* src, unsigned char* dst,
                               const size_t* src_shape, const size_t* dst_shape,
                               size_t src_rank, size_t dst_rank,
                               size_t dst_elements, cudaStream_t stream);

    // ============= Broadcasting Binary Operations - UNIFIED INTERFACE =============

    // Forward declare operation functors
    template <typename T>
    struct add_op {
        __device__ T operator()(T a, T b) const { return a + b; }
    };
    template <typename T>
    struct sub_op {
        __device__ T operator()(T a, T b) const { return a - b; }
    };
    template <typename T>
    struct mul_op {
        __device__ T operator()(T a, T b) const { return a * b; }
    };
    template <typename T>
    struct div_op {
        __device__ T operator()(T a, T b) const { return a / b; }
    };
    template <typename T>
    struct pow_op {
        __device__ T operator()(T a, T b) const { return powf(a, b); }
    };
    template <typename T>
    struct eq_op {
        __device__ unsigned char operator()(T a, T b) const { return a == b ? 1 : 0; }
    };
    template <typename T>
    struct ne_op {
        __device__ unsigned char operator()(T a, T b) const { return a != b ? 1 : 0; }
    };
    template <typename T>
    struct lt_op {
        __device__ unsigned char operator()(T a, T b) const { return a < b ? 1 : 0; }
    };
    template <typename T>
    struct le_op {
        __device__ unsigned char operator()(T a, T b) const { return a <= b ? 1 : 0; }
    };
    template <typename T>
    struct gt_op {
        __device__ unsigned char operator()(T a, T b) const { return a > b ? 1 : 0; }
    };
    template <typename T>
    struct ge_op {
        __device__ unsigned char operator()(T a, T b) const { return a >= b ? 1 : 0; }
    };
    struct logical_and_op {
        __device__ unsigned char operator()(unsigned char a, unsigned char b) const { return (a && b) ? 1 : 0; }
    };
    struct logical_or_op {
        __device__ unsigned char operator()(unsigned char a, unsigned char b) const { return (a || b) ? 1 : 0; }
    };
    struct logical_xor_op {
        __device__ unsigned char operator()(unsigned char a, unsigned char b) const { return (a != b) ? 1 : 0; }
    };

} // namespace lfs::core::tensor_ops

// Include template implementation for inline instantiation
// Only include in CUDA compilation units - C++ files will link to .cu implementations
#ifdef __CUDACC__
#include "tensor_broadcast_ops.cuh"
#else
// Forward declaration for C++ files - implementation in tensor_broadcast_ops.cu
namespace lfs::core::tensor_ops {
    template <typename T, typename OutputT, typename BinaryOp>
    void launch_broadcast_binary(const T* a, const T* b, OutputT* c,
                                 const size_t* a_shape, const size_t* b_shape, const size_t* c_shape,
                                 size_t a_rank, size_t b_rank, size_t c_rank,
                                 size_t c_elements, BinaryOp op, cudaStream_t stream);
}
#endif

namespace lfs::core::tensor_ops {

    // ============= Matrix Operations =============
    void launch_matmul(const float* a, const float* b, float* c,
                       size_t m, size_t n, size_t k,
                       cudaStream_t stream);

    void launch_batch_matmul(const float* a, const float* b, float* c,
                             size_t batch_size, size_t m, size_t n, size_t k,
                             cudaStream_t stream);

    void launch_transpose(const float* input, float* output,
                          size_t rows, size_t cols,
                          cudaStream_t stream);

    void launch_dot_product(const float* a, const float* b, float* result,
                            size_t n, cudaStream_t stream);

    // ============= Random Operations =============
    void launch_uniform(float* data, size_t n, float low, float high,
                        unsigned long long seed, cudaStream_t stream);

    void launch_normal(float* data, size_t n, float mean, float std,
                       unsigned long long seed, cudaStream_t stream);

    void launch_bernoulli(float* data, size_t n, float p,
                          unsigned long long seed, cudaStream_t stream);

    void launch_randint(int* data, size_t n, int low, int high,
                        unsigned long long seed, cudaStream_t stream);

    void launch_multinomial(const float* weights, int* samples,
                            unsigned long n, unsigned long num_samples, bool replacement,
                            unsigned long long seed, cudaStream_t stream);

    // ============= Matrix Creation Operations =============
    void launch_eye(float* data, size_t m, size_t n, cudaStream_t stream);
    void launch_diag(const float* diagonal, float* matrix, size_t n, cudaStream_t stream);
    void launch_extract_diag(const float* matrix, float* diagonal, size_t n, cudaStream_t stream);

    // ============= Masking Operations =============
    void launch_masked_select(const float* input, const unsigned char* mask,
                              float* output, size_t n, size_t output_size, cudaStream_t stream);

    void launch_masked_fill(float* data, const unsigned char* mask,
                            float value, size_t n, cudaStream_t stream);

    void launch_masked_scatter(float* data, const unsigned char* mask,
                               const float* src, size_t n, size_t src_size, cudaStream_t stream);

    void launch_where(const unsigned char* condition,
                      const float* x, const float* y, float* result,
                      const size_t* cond_shape, const size_t* x_shape,
                      const size_t* y_shape, const size_t* result_shape,
                      size_t cond_rank, size_t x_rank, size_t y_rank, size_t result_rank,
                      size_t result_elements, cudaStream_t stream);

    void launch_count_nonzero_bool(const unsigned char* data, size_t* count,
                                   size_t n, cudaStream_t stream);

    void launch_count_nonzero_float(const float* data, size_t* count,
                                    size_t n, cudaStream_t stream);

    // ============= Indexing Operations =============
    void launch_index_select(const float* input, const int* indices, float* output,
                             const size_t* shape, size_t rank, int dim,
                             size_t index_size, int boundary_mode, cudaStream_t stream);

    void launch_gather(const float* input, const int* indices, float* output,
                       const size_t* input_shape, const size_t* index_shape,
                       size_t rank, int dim, size_t total_elements,
                       int boundary_mode, cudaStream_t stream);

    void launch_take(const float* input, const int* indices, float* output,
                     size_t input_size, size_t index_size, cudaStream_t stream);

    // Fused gather + unary operation using thrust::permutation_iterator for zero-copy
    template <typename UnaryOp>
    void launch_gather_fused_unary(const float* input, const int* indices, float* output,
                                   size_t input_size, size_t index_size,
                                   UnaryOp op, cudaStream_t stream = nullptr);

    // Multi-tensor gather using zip_iterator - gather from multiple tensors with same indices
    // Perfect for: gather positions AND colors, or gather multiple Gaussian properties
    void launch_zip_gather_2(const float* input1, const float* input2,
                             const int* indices,
                             float* output1, float* output2,
                             size_t input_size, size_t index_size,
                             size_t stride1, size_t stride2,
                             cudaStream_t stream = nullptr);

    void launch_zip_gather_3(const float* input1, const float* input2, const float* input3,
                             const int* indices,
                             float* output1, float* output2, float* output3,
                             size_t input_size, size_t index_size,
                             size_t stride1, size_t stride2, size_t stride3,
                             cudaStream_t stream = nullptr);

    void launch_scatter(float* output, const int* indices, const float* src,
                        const size_t* output_shape, const size_t* index_shape,
                        size_t rank, int dim, size_t total_elements,
                        int scatter_mode, cudaStream_t stream);

    void launch_index_fill(float* data, const int* indices, float value,
                           const size_t* shape, size_t rank, int dim,
                           size_t index_size, cudaStream_t stream);

    void launch_index_copy(float* data, const int* indices, const float* src,
                           const size_t* shape, size_t rank, int dim,
                           size_t index_size, cudaStream_t stream);

    void launch_index_add(float* data, const int* indices, const float* src,
                          const size_t* shape, size_t rank, int dim,
                          size_t index_size, cudaStream_t stream);

    void launch_index_put(float* data, const int* indices, const float* values,
                          size_t data_size, size_t index_size, cudaStream_t stream);

    void launch_nonzero(const float* data, int64_t* indices,
                        size_t n, size_t output_size, cudaStream_t stream);

    void launch_nonzero_bool(const unsigned char* data, int64_t* indices,
                             size_t n, size_t output_size, cudaStream_t stream);

    // ============= Cumulative Sum Operation =============
    void launch_cumsum(void* data, const size_t* shape, size_t rank,
                       int dim, DataType dtype, cudaStream_t stream);

    // ============= Pairwise Distance Operations =============
    void launch_cdist(const float* a, const float* b, float* out,
                      size_t N, size_t M, size_t D, float p, cudaStream_t stream);

    // ============= Sorting Operations =============
    void launch_sort_1d(float* values, int64_t* indices, size_t n,
                        bool descending, cudaStream_t stream);

    void launch_sort_2d(float* values, int64_t* indices,
                        size_t outer_size, size_t dim_size, size_t inner_size,
                        int dim, bool descending, cudaStream_t stream);

    // ============= Concatenation Operations =============
    void launch_cat_last_dim(void* output, const std::vector<Tensor>& tensors, size_t num_rows,
                             size_t row_size, size_t element_size, cudaStream_t stream);

    void launch_cat_middle_dim(void* output, const std::vector<Tensor>& tensors, size_t outer_size, size_t inner_size,
                               int resolved_dim, size_t element_size, cudaStream_t stream);

    // ============= Strided Tensor Operations =============
    void launch_strided_copy(
        const void* input,
        void* output,
        const size_t* shape,
        const size_t* strides,
        size_t rank,
        size_t total_elements,
        DataType dtype,
        cudaStream_t stream = nullptr);

    // Fused strided upload: reads from pinned HOST memory with strides,
    // writes contiguously to GPU memory. Eliminates CPU materialization!
    void launch_strided_upload(
        const void* host_input,  // Pinned host memory (non-contiguous)
        void* gpu_output,        // GPU memory (contiguous output)
        const size_t* d_shape,   // Device memory: tensor shape
        const size_t* d_strides, // Device memory: stride information
        size_t rank,
        size_t total_elements,
        DataType dtype,
        cudaStream_t stream = nullptr);

} // namespace lfs::core::tensor_ops
