/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Vectorized element-wise operations inspired by tiny-cuda-nn / llm.c.
 * float4 + SM-capped grid-stride (multi float4/thread); half2/Packed128 fp16 path.
 */

#pragma once

#include "core/cuda_error.hpp"
#include "gpu_config.hpp"
#include "packed128.cuh"
#include "tensor_functors.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace lfs::core {
    namespace tensor_ops {

        // Cutoff below which Thrust is used (launch overhead dominates).
        // Tuned down from 1024 after SM-capped grid-stride made medium tensors
        // competitive with Thrust transform.
        inline constexpr size_t kVectorizedMinElems = 256;
        inline constexpr size_t kVectorizedScalarMinElems = 128;

        // ============= VECTORIZED UNARY OPERATIONS (float4 + grid-stride) =============

        /**
         * SM-capped grid-stride: each thread processes multiple float4 packs
         * (16 B loads) until the vectorized range is covered. Remainder is
         * handled by thread 0 of block 0 after the loop (or by first few tids).
         */
        template <typename Op>
        __global__ void vectorized_unary_kernel(
            const float* __restrict__ input,
            float* __restrict__ output,
            size_t n,
            Op op) {
            const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
            const size_t n_vec = n / 4;

            for (size_t vec_idx = tid; vec_idx < n_vec; vec_idx += stride) {
                float4 vals = reinterpret_cast<const float4*>(input)[vec_idx];
                vals.x = op(vals.x);
                vals.y = op(vals.y);
                vals.z = op(vals.z);
                vals.w = op(vals.w);
                reinterpret_cast<float4*>(output)[vec_idx] = vals;
            }

            // Scalar tail: only one thread writes the last 0–3 elements.
            if (tid == 0) {
                for (size_t i = n_vec * 4; i < n; ++i) {
                    output[i] = op(input[i]);
                }
            }
        }

        template <typename Op>
        __global__ void vectorized_binary_kernel(
            const float* __restrict__ a,
            const float* __restrict__ b,
            float* __restrict__ output,
            size_t n,
            Op op) {
            const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
            const size_t n_vec = n / 4;

            for (size_t vec_idx = tid; vec_idx < n_vec; vec_idx += stride) {
                float4 a_vals = reinterpret_cast<const float4*>(a)[vec_idx];
                float4 b_vals = reinterpret_cast<const float4*>(b)[vec_idx];
                float4 result;
                result.x = op(a_vals.x, b_vals.x);
                result.y = op(a_vals.y, b_vals.y);
                result.z = op(a_vals.z, b_vals.z);
                result.w = op(a_vals.w, b_vals.w);
                reinterpret_cast<float4*>(output)[vec_idx] = result;
            }

            if (tid == 0) {
                for (size_t i = n_vec * 4; i < n; ++i) {
                    output[i] = op(a[i], b[i]);
                }
            }
        }

        template <typename Op>
        __global__ void vectorized_comparison_kernel(
            const float* __restrict__ a,
            const float* __restrict__ b,
            unsigned char* __restrict__ output,
            size_t n,
            Op op) {
            const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
            const size_t n_vec = n / 4;

            for (size_t vec_idx = tid; vec_idx < n_vec; vec_idx += stride) {
                float4 a_vals = reinterpret_cast<const float4*>(a)[vec_idx];
                float4 b_vals = reinterpret_cast<const float4*>(b)[vec_idx];
                uchar4 result;
                result.x = op(a_vals.x, b_vals.x);
                result.y = op(a_vals.y, b_vals.y);
                result.z = op(a_vals.z, b_vals.z);
                result.w = op(a_vals.w, b_vals.w);
                reinterpret_cast<uchar4*>(output)[vec_idx] = result;
            }

            if (tid == 0) {
                for (size_t i = n_vec * 4; i < n; ++i) {
                    output[i] = op(a[i], b[i]);
                }
            }
        }

        template <typename Op>
        __global__ void vectorized_scalar_broadcast_kernel(
            const float* __restrict__ tensor,
            float scalar,
            float* __restrict__ output,
            size_t n,
            Op op) {
            const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
            const size_t n_vec = n / 4;

            for (size_t vec_idx = tid; vec_idx < n_vec; vec_idx += stride) {
                float4 vals = reinterpret_cast<const float4*>(tensor)[vec_idx];
                vals.x = op(vals.x, scalar);
                vals.y = op(vals.y, scalar);
                vals.z = op(vals.z, scalar);
                vals.w = op(vals.w, scalar);
                reinterpret_cast<float4*>(output)[vec_idx] = vals;
            }

            if (tid == 0) {
                for (size_t i = n_vec * 4; i < n; ++i) {
                    output[i] = op(tensor[i], scalar);
                }
            }
        }

        // ============= FP16 Packed128 / half2 vectorized path =============
        // 8× __half per 16 B load via Packed128. Functors applied element-wise
        // after float promotion where the op is float-oriented; otherwise direct.

        template <typename Op>
        __global__ void vectorized_unary_half_kernel(
            const __half* __restrict__ input,
            __half* __restrict__ output,
            size_t n,
            Op op) {
            const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
            constexpr size_t kPack = half128::size; // 8
            const size_t n_vec = n / kPack;

            for (size_t vec_idx = tid; vec_idx < n_vec; vec_idx += stride) {
                const size_t base = vec_idx * kPack;
                half128 vals = load128(input + base);
#pragma unroll
                for (int k = 0; k < static_cast<int>(kPack); ++k) {
                    vals[k] = op(vals[k]);
                }
                store128(output + base, vals);
            }

            if (tid == 0) {
                for (size_t i = n_vec * kPack; i < n; ++i) {
                    output[i] = op(input[i]);
                }
            }
        }

        template <typename Op>
        __global__ void vectorized_binary_half_kernel(
            const __half* __restrict__ a,
            const __half* __restrict__ b,
            __half* __restrict__ output,
            size_t n,
            Op op) {
            const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
            constexpr size_t kPack = half128::size;
            const size_t n_vec = n / kPack;

            for (size_t vec_idx = tid; vec_idx < n_vec; vec_idx += stride) {
                const size_t base = vec_idx * kPack;
                half128 av = load128(a + base);
                half128 bv = load128(b + base);
                half128 out;
#pragma unroll
                for (int k = 0; k < static_cast<int>(kPack); ++k) {
                    out[k] = op(av[k], bv[k]);
                }
                store128(output + base, out);
            }

            if (tid == 0) {
                for (size_t i = n_vec * kPack; i < n; ++i) {
                    output[i] = op(a[i], b[i]);
                }
            }
        }

        // ============= HOST LAUNCH HELPERS =============

        inline int sm_capped_grid(size_t work_items, int block_size) {
            if (work_items == 0)
                return 0;
            const int need = static_cast<int>((work_items + static_cast<size_t>(block_size) - 1) /
                                              static_cast<size_t>(block_size));
            const int optimal = GPUConfig::get().optimal_grid_size(block_size);
            // Cap at SM-saturating size when work is large; never exceed need.
            return need < optimal ? need : optimal;
        }

        template <typename Op>
        void launch_vectorized_unary(
            const float* input,
            float* output,
            size_t n,
            Op op,
            cudaStream_t stream = nullptr) {
            if (n == 0)
                return;

            constexpr int BLOCK_SIZE = 256;
            const size_t n_vec = n / 4;
            // Ensure at least 1 block so the scalar tail can run when n < 4.
            const size_t work = n_vec > 0 ? n_vec : size_t{1};
            int grid_size = sm_capped_grid(work, BLOCK_SIZE);
            if (grid_size < 1)
                grid_size = 1;

            vectorized_unary_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                input, output, n, op);
            LFS_CUDA_LAUNCH_CHECK(stream, "tensor.vectorized.unary");
        }

        template <typename Op>
        void launch_vectorized_binary(
            const float* a,
            const float* b,
            float* output,
            size_t n,
            Op op,
            cudaStream_t stream = nullptr) {
            if (n == 0)
                return;

            constexpr int BLOCK_SIZE = 256;
            const size_t n_vec = n / 4;
            const size_t work = n_vec > 0 ? n_vec : size_t{1};
            int grid_size = sm_capped_grid(work, BLOCK_SIZE);
            if (grid_size < 1)
                grid_size = 1;

            vectorized_binary_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                a, b, output, n, op);
            LFS_CUDA_LAUNCH_CHECK(stream, "tensor.vectorized.binary");
        }

        template <typename Op>
        void launch_vectorized_comparison(
            const float* a,
            const float* b,
            unsigned char* output,
            size_t n,
            Op op,
            cudaStream_t stream = nullptr) {
            if (n == 0)
                return;

            constexpr int BLOCK_SIZE = 256;
            const size_t n_vec = n / 4;
            const size_t work = n_vec > 0 ? n_vec : size_t{1};
            int grid_size = sm_capped_grid(work, BLOCK_SIZE);
            if (grid_size < 1)
                grid_size = 1;

            vectorized_comparison_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                a, b, output, n, op);
            LFS_CUDA_LAUNCH_CHECK(stream, "tensor.vectorized.comparison");
        }

        template <typename Op>
        void launch_vectorized_scalar_broadcast(
            const float* tensor,
            float scalar,
            float* output,
            size_t n,
            Op op,
            cudaStream_t stream = nullptr) {
            if (n == 0)
                return;

            constexpr int BLOCK_SIZE = 256;
            const size_t n_vec = n / 4;
            const size_t work = n_vec > 0 ? n_vec : size_t{1};
            int grid_size = sm_capped_grid(work, BLOCK_SIZE);
            if (grid_size < 1)
                grid_size = 1;

            vectorized_scalar_broadcast_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                tensor, scalar, output, n, op);
            LFS_CUDA_LAUNCH_CHECK(stream, "tensor.vectorized.scalar_broadcast");
        }

        template <typename Op>
        void launch_vectorized_unary_half(
            const __half* input,
            __half* output,
            size_t n,
            Op op,
            cudaStream_t stream = nullptr) {
            if (n == 0)
                return;

            constexpr int BLOCK_SIZE = 256;
            constexpr size_t kPack = half128::size;
            const size_t n_vec = n / kPack;
            const size_t work = n_vec > 0 ? n_vec : size_t{1};
            int grid_size = sm_capped_grid(work, BLOCK_SIZE);
            if (grid_size < 1)
                grid_size = 1;

            vectorized_unary_half_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                input, output, n, op);
            LFS_CUDA_LAUNCH_CHECK(stream, "tensor.vectorized.unary_half");
        }

        template <typename Op>
        void launch_vectorized_binary_half(
            const __half* a,
            const __half* b,
            __half* output,
            size_t n,
            Op op,
            cudaStream_t stream = nullptr) {
            if (n == 0)
                return;

            constexpr int BLOCK_SIZE = 256;
            constexpr size_t kPack = half128::size;
            const size_t n_vec = n / kPack;
            const size_t work = n_vec > 0 ? n_vec : size_t{1};
            int grid_size = sm_capped_grid(work, BLOCK_SIZE);
            if (grid_size < 1)
                grid_size = 1;

            vectorized_binary_half_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                a, b, output, n, op);
            LFS_CUDA_LAUNCH_CHECK(stream, "tensor.vectorized.binary_half");
        }

    } // namespace tensor_ops
} // namespace lfs::core
