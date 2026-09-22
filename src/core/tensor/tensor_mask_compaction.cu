/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/tensor_functors.hpp"
#include "internal/tensor_ops.hpp"
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

// Keep the Thrust selection instantiations separate from the other masking
// kernels so CUDA device-debug generation stays within the compiler's limits.
namespace lfs::core::tensor_ops {
    namespace {
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

} // namespace lfs::core::tensor_ops
