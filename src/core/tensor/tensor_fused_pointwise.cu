/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_debug.hpp"
#include "internal/gpu_config.hpp"
#include "internal/lazy_executor.hpp"
#include "internal/tensor_impl.hpp"
#include "internal/tensor_ops.hpp"
#include "internal/warp_reduce.cuh"
#include <cassert>
#include <cfloat>
#include <cuda_runtime.h>

static_assert(static_cast<uint8_t>(lfs::core::internal::LazyPointwiseOpKind::AddScalar) == 0);
static_assert(static_cast<uint8_t>(lfs::core::internal::LazyPointwiseOpKind::Abs) == 10);
static_assert(static_cast<uint8_t>(lfs::core::internal::LazyPointwiseOpKind::Round) == 24);

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

    // ============= Fused Transform-Reduce Kernels =============

    namespace {

        __device__ __forceinline__ float reduce_identity(int reduce_op_int) {
            switch (reduce_op_int) {
            case 0: return 0.0f;     // Sum
            case 1: return 0.0f;     // Mean (accumulates like sum)
            case 2: return -FLT_MAX; // Max
            case 3: return FLT_MAX;  // Min
            case 4: return 1.0f;     // Prod
            default: return 0.0f;
            }
        }

        __device__ __forceinline__ float reduce_combine(float a, float b, int reduce_op_int) {
            switch (reduce_op_int) {
            case 0: return a + b;
            case 1: return a + b;
            case 2: return fmaxf(a, b);
            case 3: return fminf(a, b);
            case 4: return a * b;
            default: return a + b;
            }
        }

        __device__ __forceinline__ float block_reduce_op(float val, int reduce_op_int) {
            switch (reduce_op_int) {
            case 0: return warp_ops::block_reduce_sum(val);
            case 1: return warp_ops::block_reduce_sum(val);
            case 2: return warp_ops::block_reduce_max(val);
            case 3: return warp_ops::block_reduce_min(val);
            case 4: return warp_ops::block_reduce_prod(val);
            default: return warp_ops::block_reduce_sum(val);
            }
        }

        __global__ void fused_transform_reduce_stage1_kernel(
            const float* __restrict__ input,
            float* __restrict__ partial_results,
            size_t n,
            FusedPointwiseOpChain chain,
            int reduce_op_int) {

            const float identity = reduce_identity(reduce_op_int);
            float acc = identity;

            const bool is_aligned = (reinterpret_cast<uintptr_t>(input) % 16) == 0;
            const size_t total_threads = static_cast<size_t>(gridDim.x) * blockDim.x;

            if (is_aligned) {
                const size_t vec_n = n / 4;
                for (size_t vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
                     vec_idx < vec_n;
                     vec_idx += total_threads) {
                    float4 vals = reinterpret_cast<const float4*>(input)[vec_idx];
                    acc = reduce_combine(acc, apply_chain(vals.x, chain), reduce_op_int);
                    acc = reduce_combine(acc, apply_chain(vals.y, chain), reduce_op_int);
                    acc = reduce_combine(acc, apply_chain(vals.z, chain), reduce_op_int);
                    acc = reduce_combine(acc, apply_chain(vals.w, chain), reduce_op_int);
                }
                // Handle tail elements
                const size_t tail_start = vec_n * 4;
                for (size_t i = tail_start + blockIdx.x * blockDim.x + threadIdx.x;
                     i < n;
                     i += total_threads) {
                    acc = reduce_combine(acc, apply_chain(input[i], chain), reduce_op_int);
                }
            } else {
                for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
                     i < n;
                     i += total_threads) {
                    acc = reduce_combine(acc, apply_chain(input[i], chain), reduce_op_int);
                }
            }

            acc = block_reduce_op(acc, reduce_op_int);

            if (threadIdx.x == 0) {
                partial_results[blockIdx.x] = acc;
            }
        }

        __global__ void fused_reduce_stage2_kernel(
            const float* __restrict__ partial_results,
            float* __restrict__ output,
            int num_partials,
            int reduce_op_int) {

            const float identity = reduce_identity(reduce_op_int);
            float acc = identity;

            for (int i = threadIdx.x; i < num_partials; i += blockDim.x) {
                acc = reduce_combine(acc, partial_results[i], reduce_op_int);
            }

            acc = block_reduce_op(acc, reduce_op_int);

            if (threadIdx.x == 0) {
                output[0] = acc;
            }
        }

    } // namespace

    void launch_fused_transform_reduce(const float* input, float* output, size_t n,
                                       const FusedPointwiseOpChain& chain,
                                       ReduceOp reduce_op, cudaStream_t stream) {
        if (n == 0)
            return;
        assert(input != nullptr);
        assert(output != nullptr);
        assert(chain.num_ops > 0 && chain.num_ops <= FUSED_POINTWISE_MAX_OPS);

        const int reduce_op_int = static_cast<int>(reduce_op);
        assert(reduce_op_int >= 0 && reduce_op_int <= 4);

        const auto& gpu = GPUConfig::get();
        const int grid_size = gpu.optimal_grid_size(BLOCK_SIZE);

        float* partial = nullptr;
        CHECK_CUDA(cudaMallocAsync(&partial, grid_size * sizeof(float), stream));
        assert(partial != nullptr);

        fused_transform_reduce_stage1_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
            input, partial, n, chain, reduce_op_int);

        fused_reduce_stage2_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
            partial, output, grid_size, reduce_op_int);

        if (reduce_op == ReduceOp::Mean) {
            const float scale = 1.0f / static_cast<float>(n);
            launch_fused_affine_transform(output, output, 1, scale, 0.0f, stream);
        }

        cudaFreeAsync(partial, stream);
    }

    // ============= Fused Segmented Transform-Reduce (Last-Dim) =============

    namespace {

        __global__ void fused_segmented_transform_reduce_kernel(
            const float* __restrict__ input,
            float* __restrict__ output,
            size_t num_segments,
            size_t segment_size,
            FusedPointwiseOpChain chain,
            int reduce_op_int) {

            for (size_t seg = blockIdx.x; seg < num_segments; seg += gridDim.x) {
                const float* seg_input = input + seg * segment_size;
                const float identity = reduce_identity(reduce_op_int);
                float acc = identity;

                const bool is_aligned = (reinterpret_cast<uintptr_t>(seg_input) % 16) == 0;

                if (is_aligned && segment_size >= 4) {
                    const size_t vec_n = segment_size / 4;
                    for (size_t vec_idx = threadIdx.x; vec_idx < vec_n; vec_idx += blockDim.x) {
                        float4 vals = reinterpret_cast<const float4*>(seg_input)[vec_idx];
                        acc = reduce_combine(acc, apply_chain(vals.x, chain), reduce_op_int);
                        acc = reduce_combine(acc, apply_chain(vals.y, chain), reduce_op_int);
                        acc = reduce_combine(acc, apply_chain(vals.z, chain), reduce_op_int);
                        acc = reduce_combine(acc, apply_chain(vals.w, chain), reduce_op_int);
                    }
                    const size_t tail_start = vec_n * 4;
                    for (size_t i = tail_start + threadIdx.x; i < segment_size; i += blockDim.x) {
                        acc = reduce_combine(acc, apply_chain(seg_input[i], chain), reduce_op_int);
                    }
                } else {
                    for (size_t i = threadIdx.x; i < segment_size; i += blockDim.x) {
                        acc = reduce_combine(acc, apply_chain(seg_input[i], chain), reduce_op_int);
                    }
                }

                acc = block_reduce_op(acc, reduce_op_int);

                if (threadIdx.x == 0) {
                    if (reduce_op_int == 1) { // Mean
                        acc *= (1.0f / static_cast<float>(segment_size));
                    }
                    output[seg] = acc;
                }
            }
        }

    } // namespace

    void launch_fused_segmented_transform_reduce(
        const float* input, float* output,
        size_t num_segments, size_t segment_size,
        const FusedPointwiseOpChain& chain,
        ReduceOp reduce_op, cudaStream_t stream) {

        if (num_segments == 0 || segment_size == 0)
            return;
        assert(input != nullptr);
        assert(output != nullptr);
        assert(chain.num_ops > 0 && chain.num_ops <= FUSED_POINTWISE_MAX_OPS);

        const int reduce_op_int = static_cast<int>(reduce_op);
        assert(reduce_op_int >= 0 && reduce_op_int <= 4);

        const int grid_size = static_cast<int>(std::min(num_segments, size_t(2048)));
        fused_segmented_transform_reduce_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
            input, output, num_segments, segment_size, chain, reduce_op_int);
    }

} // namespace lfs::core::tensor_ops
