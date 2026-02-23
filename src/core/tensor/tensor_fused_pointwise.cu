/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/tensor_ops.hpp"
#include <cassert>
#include <cuda_runtime.h>

namespace lfs::core::tensor_ops {

    namespace {

        constexpr int BLOCK_SIZE = 256;

        __global__ void affine_transform_vec4_kernel(const float* __restrict__ input,
                                                     float* __restrict__ output,
                                                     size_t n, float a, float b) {
            const size_t vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
            const size_t idx = vec_idx * 4;

            if (idx + 3 < n) {
                float4 vals = reinterpret_cast<const float4*>(input)[vec_idx];
                vals.x = fmaf(a, vals.x, b);
                vals.y = fmaf(a, vals.y, b);
                vals.z = fmaf(a, vals.z, b);
                vals.w = fmaf(a, vals.w, b);
                reinterpret_cast<float4*>(output)[vec_idx] = vals;
            } else if (idx < n) {
                for (size_t i = idx; i < n; ++i) {
                    output[i] = fmaf(a, input[i], b);
                }
            }
        }

        __global__ void affine_transform_scalar_kernel(const float* __restrict__ input,
                                                       float* __restrict__ output,
                                                       size_t n, float a, float b) {
            const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                output[idx] = fmaf(a, input[idx], b);
            }
        }

    } // namespace

    void launch_fused_affine_transform(const float* input, float* output,
                                       size_t n, float a, float b,
                                       cudaStream_t stream) {
        if (n == 0)
            return;
        assert(input != nullptr);
        assert(output != nullptr);

        const bool src_aligned = (reinterpret_cast<uintptr_t>(input) % 16) == 0;
        const bool dst_aligned = (reinterpret_cast<uintptr_t>(output) % 16) == 0;

        if (src_aligned && dst_aligned && n >= 4) {
            const size_t vec_n = (n + 3) / 4;
            const int grid = static_cast<int>((vec_n + BLOCK_SIZE - 1) / BLOCK_SIZE);
            affine_transform_vec4_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(input, output, n, a, b);
        } else {
            const int grid = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
            affine_transform_scalar_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(input, output, n, a, b);
        }
    }

    namespace {

        __device__ __forceinline__ float apply_pointwise_op(float x, const FusedPointwiseOp& op) {
            switch (op.kind) {
            case 0: return x + op.scalar;             // AddScalar
            case 1: return x - op.scalar;             // SubScalar
            case 2: return x * op.scalar;             // MulScalar
            case 3: return x / op.scalar;             // DivScalar
            case 10: return fabsf(x);                 // Abs
            case 11: return -x;                       // Neg
            case 12: return expf(x);                  // Exp
            case 13: return logf(fmaxf(x, 1e-10f));   // Log
            case 14: return sqrtf(fmaxf(x, 0.0f));    // Sqrt
            case 15: return 1.0f / (1.0f + expf(-x)); // Sigmoid
            case 16: return fmaxf(x, 0.0f);           // Relu
            case 17: return x * x;                    // Square
            case 18: return tanhf(x);                 // Tanh
            case 19: return rsqrtf(fmaxf(x, 1e-10f)); // Rsqrt
            case 20: return float((x > 0) - (x < 0)); // Sign
            case 21: return 1.0f / (x + 1e-8f);       // Reciprocal
            case 22: return floorf(x);                // Floor
            case 23: return ceilf(x);                 // Ceil
            case 24: return roundf(x);                // Round
            default: return x;
            }
        }

        __device__ __forceinline__ float apply_chain(float x, const FusedPointwiseOpChain& chain) {
            for (int i = 0; i < chain.num_ops; ++i) {
                x = apply_pointwise_op(x, chain.ops[i]);
            }
            return x;
        }

        __global__ void pointwise_chain_vec4_kernel(const float* __restrict__ input,
                                                    float* __restrict__ output,
                                                    size_t n, FusedPointwiseOpChain chain) {
            const size_t vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
            const size_t idx = vec_idx * 4;

            if (idx + 3 < n) {
                float4 vals = reinterpret_cast<const float4*>(input)[vec_idx];
                vals.x = apply_chain(vals.x, chain);
                vals.y = apply_chain(vals.y, chain);
                vals.z = apply_chain(vals.z, chain);
                vals.w = apply_chain(vals.w, chain);
                reinterpret_cast<float4*>(output)[vec_idx] = vals;
            } else if (idx < n) {
                for (size_t i = idx; i < n; ++i) {
                    output[i] = apply_chain(input[i], chain);
                }
            }
        }

        __global__ void pointwise_chain_scalar_kernel(const float* __restrict__ input,
                                                      float* __restrict__ output,
                                                      size_t n, FusedPointwiseOpChain chain) {
            const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                output[idx] = apply_chain(input[idx], chain);
            }
        }

    } // namespace

    void launch_fused_pointwise_chain(const float* input, float* output,
                                      size_t n, const FusedPointwiseOpChain& chain,
                                      cudaStream_t stream) {
        if (n == 0)
            return;
        assert(input != nullptr);
        assert(output != nullptr);
        assert(chain.num_ops > 0 && chain.num_ops <= FUSED_POINTWISE_MAX_OPS);

        const bool src_aligned = (reinterpret_cast<uintptr_t>(input) % 16) == 0;
        const bool dst_aligned = (reinterpret_cast<uintptr_t>(output) % 16) == 0;

        if (src_aligned && dst_aligned && n >= 4) {
            const size_t vec_n = (n + 3) / 4;
            const int grid = static_cast<int>((vec_n + BLOCK_SIZE - 1) / BLOCK_SIZE);
            pointwise_chain_vec4_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(input, output, n, chain);
        } else {
            const int grid = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
            pointwise_chain_scalar_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(input, output, n, chain);
        }
    }

} // namespace lfs::core::tensor_ops
