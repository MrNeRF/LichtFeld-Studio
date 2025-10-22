/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform.h>

// Include vectorized operations for float4 optimizations
#include "tensor_vectorized_ops.cuh"

namespace lfs::core::tensor_ops {

    // ============================================================================
    // THRUST POLICY HELPER - Stream support for Thrust operations
    // ============================================================================

    template <typename Func>
    inline void run_with_thrust_policy(cudaStream_t stream, Func&& func) {
        if (stream) {
            func(thrust::cuda::par.on(stream));
        } else {
            func(thrust::cuda::par);
        }
    }

    // ============================================================================
    // GENERIC OPERATIONS - Header-only for optimal inlining and specialization
    // ============================================================================

    // These are in a header (not .cu) to allow:
    // 1. Compiler to inline Thrust operations across translation units
    // 2. Template specialization for specific functors (e.g., composed_unary_op)
    // 3. Better devirtualization of functors at compile-time
    // 4. No need for explicit template instantiations
    // 5. Works with expression template fusion (composed functors)

    /**
     * Binary operation: Applies binary functor element-wise to two arrays
     * Supports different input/output types (e.g., float -> bool for comparisons)
     *
     * @tparam InT Input element type
     * @tparam OutT Output element type
     * @tparam Op Binary functor type (must be __device__ callable)
     */
    template <typename InT, typename OutT, typename Op>
    void launch_binary_op_generic(const InT* a, const InT* b, OutT* c, size_t n,
                                  Op op, cudaStream_t stream = nullptr) {
        if (n == 0)
            return;

        // FAST PATH 1: Vectorized comparison operations (float -> unsigned char)
        // Perfect for <, >, ==, != operations
        if constexpr (std::is_same_v<InT, float> && std::is_same_v<OutT, unsigned char>) {
            bool a_aligned = (reinterpret_cast<uintptr_t>(a) % 16) == 0;
            bool b_aligned = (reinterpret_cast<uintptr_t>(b) % 16) == 0;
            bool c_aligned = (reinterpret_cast<uintptr_t>(c) % 4) == 0; // uchar4 needs 4-byte alignment

            if (a_aligned && b_aligned && c_aligned && n > 1024) {
                launch_vectorized_comparison(a, b, c, n, op, stream);
                return;
            }
        }

        // FAST PATH 2: Vectorized float->float operations (arithmetic)
        if constexpr (std::is_same_v<InT, float> && std::is_same_v<OutT, float>) {
            bool a_aligned = (reinterpret_cast<uintptr_t>(a) % 16) == 0;
            bool b_aligned = (reinterpret_cast<uintptr_t>(b) % 16) == 0;
            bool c_aligned = (reinterpret_cast<uintptr_t>(c) % 16) == 0;

            if (a_aligned && b_aligned && c_aligned && n > 1024) {
                launch_vectorized_binary(a, b, c, n, op, stream);
                return;
            }
        }

        // FALLBACK: Use Thrust for unaligned or non-float types
        auto a_ptr = thrust::device_pointer_cast(a);
        auto b_ptr = thrust::device_pointer_cast(b);
        auto c_ptr = thrust::device_pointer_cast(c);

        run_with_thrust_policy(stream, [&](auto policy) {
            thrust::transform(policy, a_ptr, a_ptr + n, b_ptr, c_ptr, op);
        });
    }

    /**
     * Unary operation: Applies unary functor element-wise to an array
     * Supports different input/output types (e.g., float -> bool for predicates)
     *
     * @tparam InT Input element type
     * @tparam OutT Output element type
     * @tparam Op Unary functor type (must be __device__ callable)
     */
    template <typename InT, typename OutT, typename Op>
    void launch_unary_op_generic(const InT* input, OutT* output, size_t n,
                                 Op op, cudaStream_t stream = nullptr) {
        if (n == 0)
            return;

        // FAST PATH: Use vectorized kernel for float->float operations
        if constexpr (std::is_same_v<InT, float> && std::is_same_v<OutT, float>) {
            bool input_aligned = (reinterpret_cast<uintptr_t>(input) % 16) == 0;
            bool output_aligned = (reinterpret_cast<uintptr_t>(output) % 16) == 0;

            if (input_aligned && output_aligned && n > 1024) {
                launch_vectorized_unary(input, output, n, op, stream);
                return;
            }
        }

        // FALLBACK: Use Thrust for unaligned or non-float types
        auto in_ptr = thrust::device_pointer_cast(input);
        auto out_ptr = thrust::device_pointer_cast(output);

        run_with_thrust_policy(stream, [&](auto policy) {
            thrust::transform(policy, in_ptr, in_ptr + n, out_ptr, op);
        });
    }

    /**
     * Scalar operation: Applies binary operation with a scalar value
     * OPTIMIZED: Uses constant_iterator for zero-memory scalar broadcasting
     *
     * @tparam T Element type
     * @tparam OutputT Output element type
     * @tparam Op Binary functor type (must be __device__ callable)
     */
    template <typename T, typename OutputT, typename Op>
    void launch_scalar_op_generic(const T* data, T scalar, OutputT* result, size_t n,
                                  Op op, cudaStream_t stream = nullptr) {
        if (n == 0)
            return;

        // FAST PATH: Use vectorized scalar broadcast for float operations
        if constexpr (std::is_same_v<T, float> && std::is_same_v<OutputT, float>) {
            bool data_aligned = (reinterpret_cast<uintptr_t>(data) % 16) == 0;
            bool result_aligned = (reinterpret_cast<uintptr_t>(result) % 16) == 0;

            if (data_aligned && result_aligned && n > 512) {
                launch_vectorized_scalar_broadcast(data, scalar, result, n, op, stream);
                return;
            }
        }

        // FALLBACK: Use Thrust constant_iterator for unaligned or non-float types
        auto data_ptr = thrust::device_pointer_cast(data);
        auto result_ptr = thrust::device_pointer_cast(result);

        // Use constant_iterator for zero-memory scalar - generated on-the-fly!
        auto constant_scalar = thrust::make_constant_iterator(scalar);

        run_with_thrust_policy(stream, [&](auto policy) {
            // Binary transform: tensor op constant_iterator
            thrust::transform(policy, data_ptr, data_ptr + n, constant_scalar, result_ptr, op);
        });
    }

} // namespace lfs::core::tensor_ops
