/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Optimized reduction kernels inspired by llm.c/llmc
 *
 * KEY OPTIMIZATIONS:
 * 1. Packed128 vectorized loads (4× memory bandwidth vs scalar)
 * 2. Two-stage reduction (eliminates atomic contention)
 * 3. Streaming cache hints (__ldcs - bypass L1 for activations)
 * 4. GPU-aware grid sizing (fill all SMs optimally)
 *
 * Expected: 2-4× speedup on large reductions!
 */

#include "internal/gpu_config.hpp"
#include "internal/packed128.cuh"
#include "internal/tensor_functors.hpp"
#include "internal/tensor_impl.hpp"
#include "internal/tensor_ops.hpp"
#include "internal/warp_reduce.cuh"
#include <cfloat>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

namespace lfs::core::tensor_ops {

    // ============= OPTIMIZED FULL REDUCTION TO SCALAR =============

    /**
     * @brief Fast full reduction using warp shuffles + vectorized loads
     *
     * This kernel combines:
     * 1. Vectorized float4 loads (4x memory bandwidth)
     * 2. Warp-level reductions (5-10x faster than shared memory)
     * 3. Atomic add for final aggregation
     *
     * Expected speedup: 10-20x over naive implementation!
     */
    template <typename T, typename Op>
    __global__ void warp_reduce_full_kernel(
        const T* __restrict__ input,
        T* __restrict__ output,
        size_t n,
        T init_value,
        Op op) {
        size_t vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t idx = vec_idx * 4;

        T val = init_value;

        // Vectorized load: 4 elements per thread
        if constexpr (std::is_same_v<T, float>) {
            if (idx + 3 < n) {
                // Load 4 floats in one transaction (16 bytes aligned)
                float4 vals = reinterpret_cast<const float4*>(input)[vec_idx];

                // Apply operation and combine
                T a = vals.x;
                T b = vals.y;
                T c = vals.z;
                T d = vals.w;

                val = op(op(op(a, b), c), d);
            } else if (idx < n) {
                // Handle remainder (last 1-3 elements)
                for (size_t i = idx; i < n && i < idx + 4; ++i) {
                    val = op(val, input[i]);
                }
            }
        } else {
            // Fallback for non-float types
            if (idx < n) {
                for (size_t i = idx; i < n && i < idx + 4; ++i) {
                    val = op(val, input[i]);
                }
            }
        }

        // Block-level warp reduction
        val = warp_ops::block_reduce_sum(val);

        // First thread in each block writes result
        if (threadIdx.x == 0) {
            atomicAdd(output, val);
        }
    }

    /**
     * @brief TWO-STAGE sum reduction with Packed128 (OPTIMIZED - llm.c pattern)
     *
     * Stage 1: Each block reduces to a partial sum (no atomics!)
     * Stage 2: Single-block aggregation of partial sums (fast!)
     *
     * This eliminates atomic contention and provides deterministic results.
     * Expected 2-4× speedup on large reductions!
     *
     * IMPORTANT: Uses grid-stride loop to handle n > (grid_size * block_size * vec_size)
     */
    __global__ void warp_reduce_sum_stage1_kernel(
        const float* __restrict__ input,
        float* __restrict__ partial_sums,
        size_t n,
        bool use_packed128) {
        float val = 0.0f;
        const size_t total_threads = gridDim.x * blockDim.x;

        if (use_packed128) {
            // Grid-stride loop: process multiple vectors per thread if needed
            for (size_t vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
                 vec_idx * f128::size < n;
                 vec_idx += total_threads) {
                size_t idx = vec_idx * f128::size;
                if (idx + f128::size - 1 < n) {
                    // Streaming load: bypass L1 cache (activations won't be reused)
                    f128 packed = load128cs(input + idx);
#pragma unroll
                    for (int k = 0; k < f128::size; ++k) {
                        val += packed[k];
                    }
                } else if (idx < n) {
                    // Handle remainder
                    for (size_t i = idx; i < n && i < idx + f128::size; ++i) {
                        val += input[i];
                    }
                }
            }
        } else {
            // Fallback for unaligned data - grid-stride loop
            for (size_t vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
                 vec_idx * 4 < n;
                 vec_idx += total_threads) {
                size_t idx = vec_idx * 4;
                for (size_t i = idx; i < n && i < idx + 4; ++i) {
                    val += input[i];
                }
            }
        }

        // Block-level warp reduction
        val = warp_ops::block_reduce_sum(val);

        // Each block writes its partial sum (NO ATOMIC!)
        if (threadIdx.x == 0) {
            partial_sums[blockIdx.x] = val;
        }
    }

    /**
     * @brief TWO-STAGE sum reduction - Stage 2: Final aggregation
     *
     * Single block aggregates all partial sums deterministically.
     * Much faster than atomic contention!
     */
    __global__ void warp_reduce_sum_stage2_kernel(
        const float* __restrict__ partial_sums,
        float* __restrict__ output,
        int num_partials) {
        // Single block processes all partial sums
        float thread_sum = 0.0f;
        for (int i = threadIdx.x; i < num_partials; i += blockDim.x) {
            thread_sum += partial_sums[i];
        }

        // Final block reduction
        float result = warp_ops::block_reduce_sum(thread_sum);

        if (threadIdx.x == 0) {
            *output = result; // Direct write, no atomic!
        }
    }

    // Single-stage sum reduction with grid-stride loop
    __global__ void warp_reduce_sum_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        const size_t n,
        const bool use_vectorized) {
        constexpr size_t VECTOR_SIZE = 4;
        const size_t total_threads = gridDim.x * blockDim.x;
        float val = 0.0f;

        for (size_t vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
             vec_idx * VECTOR_SIZE < n;
             vec_idx += total_threads) {
            const size_t idx = vec_idx * VECTOR_SIZE;

            if (use_vectorized && idx + 3 < n) {
                const float4 vals = reinterpret_cast<const float4*>(input)[vec_idx];
                val += vals.x + vals.y + vals.z + vals.w;
            } else if (idx < n) {
                for (size_t i = idx; i < n && i < idx + VECTOR_SIZE; ++i) {
                    val += input[i];
                }
            }
        }

        val = warp_ops::block_reduce_sum(val);

        if (threadIdx.x == 0) {
            atomicAdd(output, val);
        }
    }

    // Max reduction with grid-stride loop
    __global__ void warp_reduce_max_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        const size_t n,
        const bool use_vectorized) {
        constexpr size_t VECTOR_SIZE = 4;
        const size_t total_threads = gridDim.x * blockDim.x;
        float val = -CUDA_INFINITY;

        for (size_t vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
             vec_idx * VECTOR_SIZE < n;
             vec_idx += total_threads) {
            const size_t idx = vec_idx * VECTOR_SIZE;

            if (use_vectorized && idx + 3 < n) {
                const float4 vals = reinterpret_cast<const float4*>(input)[vec_idx];
                val = fmaxf(val, fmaxf(fmaxf(vals.x, vals.y), fmaxf(vals.z, vals.w)));
            } else if (idx < n) {
                for (size_t i = idx; i < n && i < idx + VECTOR_SIZE; ++i) {
                    val = fmaxf(val, input[i]);
                }
            }
        }

        val = warp_ops::block_reduce_max(val);

        if (threadIdx.x == 0) {
            int* output_as_int = reinterpret_cast<int*>(output);
            int old = *output_as_int;
            int assumed;
            do {
                assumed = old;
                const float new_val = fmaxf(__int_as_float(assumed), val);
                old = atomicCAS(output_as_int, assumed, __float_as_int(new_val));
            } while (assumed != old);
        }
    }

    // Min reduction with grid-stride loop
    __global__ void warp_reduce_min_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        const size_t n,
        const bool use_vectorized) {
        constexpr size_t VECTOR_SIZE = 4;
        const size_t total_threads = gridDim.x * blockDim.x;
        float val = CUDA_INFINITY;

        for (size_t vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
             vec_idx * VECTOR_SIZE < n;
             vec_idx += total_threads) {
            const size_t idx = vec_idx * VECTOR_SIZE;

            if (use_vectorized && idx + 3 < n) {
                const float4 vals = reinterpret_cast<const float4*>(input)[vec_idx];
                val = fminf(val, fminf(fminf(vals.x, vals.y), fminf(vals.z, vals.w)));
            } else if (idx < n) {
                for (size_t i = idx; i < n && i < idx + VECTOR_SIZE; ++i) {
                    val = fminf(val, input[i]);
                }
            }
        }

        val = warp_ops::block_reduce_min(val);

        if (threadIdx.x == 0) {
            int* output_as_int = reinterpret_cast<int*>(output);
            int old = *output_as_int;
            int assumed;
            do {
                assumed = old;
                const float new_val = fminf(__int_as_float(assumed), val);
                old = atomicCAS(output_as_int, assumed, __float_as_int(new_val));
            } while (assumed != old);
        }
    }

    // Product reduction with grid-stride loop
    __global__ void warp_reduce_prod_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        const size_t n,
        const bool use_vectorized) {
        constexpr size_t VECTOR_SIZE = 4;
        const size_t total_threads = gridDim.x * blockDim.x;
        float val = 1.0f;

        for (size_t vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
             vec_idx * VECTOR_SIZE < n;
             vec_idx += total_threads) {
            const size_t idx = vec_idx * VECTOR_SIZE;

            if (use_vectorized && idx + 3 < n) {
                const float4 vals = reinterpret_cast<const float4*>(input)[vec_idx];
                val *= vals.x * vals.y * vals.z * vals.w;
            } else if (idx < n) {
                for (size_t i = idx; i < n && i < idx + VECTOR_SIZE; ++i) {
                    val *= input[i];
                }
            }
        }

        val = warp_ops::block_reduce_prod(val);

        if (threadIdx.x == 0) {
            int* output_as_int = reinterpret_cast<int*>(output);
            int old = *output_as_int;
            int assumed;
            do {
                assumed = old;
                const float new_val = __int_as_float(assumed) * val;
                old = atomicCAS(output_as_int, assumed, __float_as_int(new_val));
            } while (assumed != old);
        }
    }

    // ============= SEGMENTED REDUCTION KERNELS (CONTIGUOUS) =============

    /**
     * @brief FAST kernel for MEDIUM segments (32-2048 elements)
     *
     * Each WARP processes one segment entirely using warp shuffle reductions.
     * NO block synchronization needed!
     *
     * OPTIMIZATION: COALESCED float4 loads - all lanes read consecutive memory.
     * For segment_size=1024: 32 lanes * 4 floats = 128 floats per iteration, 8 iterations.
     */
    __global__ void warp_medium_segment_reduce_sum_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        size_t num_segments,
        size_t segment_size) {
        const int warp_id = threadIdx.x / 32;
        const int lane = threadIdx.x % 32;
        const int warps_per_block = blockDim.x / 32;

        const bool can_vectorize = (segment_size % 4) == 0;

        for (size_t seg_idx = blockIdx.x * warps_per_block + warp_id;
             seg_idx < num_segments;
             seg_idx += gridDim.x * warps_per_block) {
            const float* segment_start = input + seg_idx * segment_size;

            float sum = 0.0f;

            if (can_vectorize) {
                const size_t num_float4s = segment_size / 4;
                for (size_t base = 0; base < num_float4s; base += 32) {
                    size_t idx = base + lane;
                    if (idx < num_float4s) {
                        float4 v = reinterpret_cast<const float4*>(segment_start)[idx];
                        sum += v.x + v.y + v.z + v.w;
                    }
                }
            } else {
                for (size_t i = lane; i < segment_size; i += 32) {
                    sum += segment_start[i];
                }
            }

            sum = warp_ops::warp_reduce_sum(sum);

            if (lane == 0) {
                output[seg_idx] = sum;
            }
        }
    }

    /**
     * @brief FAST kernel for MEDIUM segments - MEAN variant with coalesced loads
     */
    __global__ void warp_medium_segment_reduce_mean_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        size_t num_segments,
        size_t segment_size) {
        const int warp_id = threadIdx.x / 32;
        const int lane = threadIdx.x % 32;
        const int warps_per_block = blockDim.x / 32;
        const float inv_size = 1.0f / static_cast<float>(segment_size);
        const bool can_vectorize = (segment_size % 4) == 0;

        for (size_t seg_idx = blockIdx.x * warps_per_block + warp_id;
             seg_idx < num_segments;
             seg_idx += gridDim.x * warps_per_block) {
            const float* segment_start = input + seg_idx * segment_size;

            float sum = 0.0f;
            if (can_vectorize) {
                const size_t num_float4s = segment_size / 4;
                for (size_t base = 0; base < num_float4s; base += 32) {
                    size_t idx = base + lane;
                    if (idx < num_float4s) {
                        float4 v = reinterpret_cast<const float4*>(segment_start)[idx];
                        sum += v.x + v.y + v.z + v.w;
                    }
                }
            } else {
                for (size_t i = lane; i < segment_size; i += 32) {
                    sum += segment_start[i];
                }
            }

            sum = warp_ops::warp_reduce_sum(sum);

            if (lane == 0) {
                output[seg_idx] = sum * inv_size;
            }
        }
    }

    /**
     * @brief FAST kernel for MEDIUM segments - MAX variant with coalesced loads
     */
    __global__ void warp_medium_segment_reduce_max_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        size_t num_segments,
        size_t segment_size) {
        const int warp_id = threadIdx.x / 32;
        const int lane = threadIdx.x % 32;
        const int warps_per_block = blockDim.x / 32;
        const bool can_vectorize = (segment_size % 4) == 0;

        for (size_t seg_idx = blockIdx.x * warps_per_block + warp_id;
             seg_idx < num_segments;
             seg_idx += gridDim.x * warps_per_block) {
            const float* segment_start = input + seg_idx * segment_size;

            float val = -CUDA_INFINITY;
            if (can_vectorize) {
                const size_t num_float4s = segment_size / 4;
                for (size_t base = 0; base < num_float4s; base += 32) {
                    size_t idx = base + lane;
                    if (idx < num_float4s) {
                        float4 v = reinterpret_cast<const float4*>(segment_start)[idx];
                        val = fmaxf(val, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
                    }
                }
            } else {
                for (size_t i = lane; i < segment_size; i += 32) {
                    val = fmaxf(val, segment_start[i]);
                }
            }

            val = warp_ops::warp_reduce_max(val);

            if (lane == 0) {
                output[seg_idx] = val;
            }
        }
    }

    /**
     * @brief FAST kernel for MEDIUM segments - MIN variant with coalesced loads
     */
    __global__ void warp_medium_segment_reduce_min_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        size_t num_segments,
        size_t segment_size) {
        const int warp_id = threadIdx.x / 32;
        const int lane = threadIdx.x % 32;
        const int warps_per_block = blockDim.x / 32;
        const bool can_vectorize = (segment_size % 4) == 0;

        for (size_t seg_idx = blockIdx.x * warps_per_block + warp_id;
             seg_idx < num_segments;
             seg_idx += gridDim.x * warps_per_block) {
            const float* segment_start = input + seg_idx * segment_size;

            float val = CUDA_INFINITY;
            if (can_vectorize) {
                const size_t num_float4s = segment_size / 4;
                for (size_t base = 0; base < num_float4s; base += 32) {
                    size_t idx = base + lane;
                    if (idx < num_float4s) {
                        float4 v = reinterpret_cast<const float4*>(segment_start)[idx];
                        val = fminf(val, fminf(fminf(v.x, v.y), fminf(v.z, v.w)));
                    }
                }
            } else {
                for (size_t i = lane; i < segment_size; i += 32) {
                    val = fminf(val, segment_start[i]);
                }
            }

            val = warp_ops::warp_reduce_min(val);

            if (lane == 0) {
                output[seg_idx] = val;
            }
        }
    }

    __global__ void warp_medium_segment_reduce_prod_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        size_t num_segments,
        size_t segment_size) {
        const int warp_id = threadIdx.x / 32;
        const int lane = threadIdx.x % 32;
        const int warps_per_block = blockDim.x / 32;
        const bool can_vectorize = (segment_size % 4) == 0;

        for (size_t seg_idx = blockIdx.x * warps_per_block + warp_id;
             seg_idx < num_segments;
             seg_idx += gridDim.x * warps_per_block) {
            const float* segment_start = input + seg_idx * segment_size;

            float val = 1.0f;
            if (can_vectorize) {
                const size_t num_float4s = segment_size / 4;
                for (size_t base = 0; base < num_float4s; base += 32) {
                    size_t idx = base + lane;
                    if (idx < num_float4s) {
                        float4 v = reinterpret_cast<const float4*>(segment_start)[idx];
                        val *= v.x * v.y * v.z * v.w;
                    }
                }
            } else {
                for (size_t i = lane; i < segment_size; i += 32) {
                    val *= segment_start[i];
                }
            }

            val = warp_ops::warp_reduce_prod(val);

            if (lane == 0) {
                output[seg_idx] = val;
            }
        }
    }

    __global__ void warp_tiny_segment_reduce_sum_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        size_t num_segments,
        size_t segment_size) {
        size_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;

        // Each thread processes complete segments
        for (size_t seg_idx = global_tid; seg_idx < num_segments; seg_idx += stride) {
            const float* segment_start = input + seg_idx * segment_size;

            // Sequential reduction of small segment
            // Use double accumulation to avoid FP32 precision loss
            double sum = 0.0;
#pragma unroll 8
            for (size_t i = 0; i < segment_size; ++i) {
                sum += (double)segment_start[i];
            }

            output[seg_idx] = (float)sum;
        }
    }

    /**
     * @brief OPTIMIZED segmented sum reduction with grid-stride loop
     *
     * Uses grid-stride loop to process MULTIPLE segments per block.
     * Much more efficient for medium segments (32-500K elements).
     *
     * NOTE: This is for contiguous segments only (inner_size == 1)
     */
    __global__ void warp_segmented_reduce_sum_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        size_t num_segments,
        size_t segment_size) {
        // Grid-stride loop: Each block processes multiple segments
        for (size_t seg_idx = blockIdx.x; seg_idx < num_segments; seg_idx += gridDim.x) {
            const float* segment_start = input + seg_idx * segment_size;
            float result = warp_ops::vectorized_segment_reduce_sum(segment_start, segment_size);

            if (threadIdx.x == 0) {
                output[seg_idx] = result;
            }
        }
    }

    /**
     * @brief FUSED mean kernel - avoids separate division kernel
     */
    __global__ void warp_segmented_reduce_mean_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        size_t num_segments,
        size_t segment_size) {
        const float inv_size = 1.0f / static_cast<float>(segment_size);
        for (size_t seg_idx = blockIdx.x; seg_idx < num_segments; seg_idx += gridDim.x) {
            const float* segment_start = input + seg_idx * segment_size;
            float result = warp_ops::vectorized_segment_reduce_sum(segment_start, segment_size);

            if (threadIdx.x == 0) {
                output[seg_idx] = result * inv_size;
            }
        }
    }

    /**
     * @brief FUSED mean kernel for tiny segments - avoids separate division
     */
    __global__ void warp_tiny_segment_reduce_mean_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        size_t num_segments,
        size_t segment_size) {
        size_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;
        const float inv_size = 1.0f / static_cast<float>(segment_size);

        for (size_t seg_idx = global_tid; seg_idx < num_segments; seg_idx += stride) {
            const float* segment_start = input + seg_idx * segment_size;
            float sum = 0.0f;
#pragma unroll 8
            for (size_t i = 0; i < segment_size; ++i) {
                sum += segment_start[i];
            }
            output[seg_idx] = sum * inv_size;
        }
    }

    /**
     * @brief SPECIALIZED kernel for TINY segments (< 32 elements) - MAX
     */
    __global__ void warp_tiny_segment_reduce_max_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        size_t num_segments,
        size_t segment_size) {
        size_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;

        for (size_t seg_idx = global_tid; seg_idx < num_segments; seg_idx += stride) {
            const float* segment_start = input + seg_idx * segment_size;

            float max_val = -CUDA_INFINITY;
#pragma unroll 8
            for (size_t i = 0; i < segment_size; ++i) {
                max_val = fmaxf(max_val, segment_start[i]);
            }

            output[seg_idx] = max_val;
        }
    }

    /**
     * @brief SPECIALIZED kernel for TINY segments (< 32 elements) - MIN
     */
    __global__ void warp_tiny_segment_reduce_min_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        size_t num_segments,
        size_t segment_size) {
        size_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;

        for (size_t seg_idx = global_tid; seg_idx < num_segments; seg_idx += stride) {
            const float* segment_start = input + seg_idx * segment_size;

            float min_val = CUDA_INFINITY;
#pragma unroll 8
            for (size_t i = 0; i < segment_size; ++i) {
                min_val = fminf(min_val, segment_start[i]);
            }

            output[seg_idx] = min_val;
        }
    }

    __global__ void warp_tiny_segment_reduce_prod_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        size_t num_segments,
        size_t segment_size) {
        size_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;

        for (size_t seg_idx = global_tid; seg_idx < num_segments; seg_idx += stride) {
            const float* segment_start = input + seg_idx * segment_size;

            float prod_val = 1.0f;
#pragma unroll 8
            for (size_t i = 0; i < segment_size; ++i) {
                prod_val *= segment_start[i];
            }

            output[seg_idx] = prod_val;
        }
    }

    __global__ void warp_segmented_reduce_max_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        size_t num_segments,
        size_t segment_size) {
        // Grid-stride loop: Each block processes multiple segments
        for (size_t seg_idx = blockIdx.x; seg_idx < num_segments; seg_idx += gridDim.x) {
            const float* segment_start = input + seg_idx * segment_size;
            float result = warp_ops::vectorized_segment_reduce_max(segment_start, segment_size);

            if (threadIdx.x == 0) {
                output[seg_idx] = result;
            }
        }
    }

    /**
     * @brief OPTIMIZED segmented min reduction with grid-stride loop
     *
     * Uses grid-stride loop to process MULTIPLE segments per block.
     * Much more efficient for medium segments (32-500K elements).
     */
    __global__ void warp_segmented_reduce_min_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        size_t num_segments,
        size_t segment_size) {
        // Grid-stride loop: Each block processes multiple segments
        for (size_t seg_idx = blockIdx.x; seg_idx < num_segments; seg_idx += gridDim.x) {
            const float* segment_start = input + seg_idx * segment_size;
            float result = warp_ops::vectorized_segment_reduce_min(segment_start, segment_size);

            if (threadIdx.x == 0) {
                output[seg_idx] = result;
            }
        }
    }

    __global__ void warp_segmented_reduce_prod_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        size_t num_segments,
        size_t segment_size) {
        for (size_t seg_idx = blockIdx.x; seg_idx < num_segments; seg_idx += gridDim.x) {
            const float* segment_start = input + seg_idx * segment_size;
            float result = warp_ops::vectorized_segment_reduce_prod(segment_start, segment_size);

            if (threadIdx.x == 0) {
                output[seg_idx] = result;
            }
        }
    }

    // ============= STRIDED REDUCTION KERNELS (NON-CONTIGUOUS) =============

    /**
     * @brief OPTIMIZED strided sum reduction for non-contiguous segments
     *
     * Handles reductions where inner_size > 1 (e.g., reducing along dim 0 or dim 1).
     * Each thread processes multiple output elements using a grid-stride loop.
     *
     * KEY OPTIMIZATIONS:
     * 1. Unrolled accumulation (process 8 elements at a time)
     * 2. Balanced tree reduction to minimize FP rounding errors
     * 3. Grid-stride loop for perfect load balancing
     * 4. Coalesced writes to output
     *
     * Memory pattern: output[outer*inner + inner_idx] = reduce(input[outer*reduce*inner + r*inner + inner_idx] for r in 0..reduce-1)
     */
    __global__ void warp_strided_reduce_sum_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        size_t outer_size,
        size_t reduce_size,
        size_t inner_size) {
        size_t output_elements = outer_size * inner_size;
        size_t stride = blockDim.x * gridDim.x;

        // Grid-stride loop for good occupancy
        for (size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
             out_idx < output_elements;
             out_idx += stride) {
            size_t outer_idx = out_idx / inner_size;
            size_t inner_idx = out_idx % inner_size;

            // Accumulate across the reduce dimension with strided access
            // Use double accumulation to avoid FP32 precision loss
            double sum = 0.0;
            size_t base_idx = outer_idx * reduce_size * inner_size + inner_idx;

            // OPTIMIZATION: Unroll 8× for better ILP (Instruction Level Parallelism)
            size_t r = 0;
            if (reduce_size >= 8) {
#pragma unroll 2
                for (; r + 7 < reduce_size; r += 8) {
                    // Load 8 values with strided access
                    float v0 = input[base_idx + (r + 0) * inner_size];
                    float v1 = input[base_idx + (r + 1) * inner_size];
                    float v2 = input[base_idx + (r + 2) * inner_size];
                    float v3 = input[base_idx + (r + 3) * inner_size];
                    float v4 = input[base_idx + (r + 4) * inner_size];
                    float v5 = input[base_idx + (r + 5) * inner_size];
                    float v6 = input[base_idx + (r + 6) * inner_size];
                    float v7 = input[base_idx + (r + 7) * inner_size];

                    // Accumulate in double precision
                    sum += (double)v0 + (double)v1 + (double)v2 + (double)v3 +
                           (double)v4 + (double)v5 + (double)v6 + (double)v7;
                }
            }

// Handle remainder (< 8 elements)
#pragma unroll 4
            for (; r < reduce_size; ++r) {
                sum += (double)input[base_idx + r * inner_size];
            }

            output[out_idx] = (float)sum;
        }
    }

    /**
     * @brief OPTIMIZED strided max reduction for non-contiguous segments
     *
     * OPTIMIZATION: 8× unrolling + balanced tree reduction for better ILP
     */
    __global__ void warp_strided_reduce_max_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        size_t outer_size,
        size_t reduce_size,
        size_t inner_size) {
        size_t output_elements = outer_size * inner_size;
        size_t stride = blockDim.x * gridDim.x;

        for (size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
             out_idx < output_elements;
             out_idx += stride) {
            size_t outer_idx = out_idx / inner_size;
            size_t inner_idx = out_idx % inner_size;

            float max_val = -CUDA_INFINITY;
            size_t base_idx = outer_idx * reduce_size * inner_size + inner_idx;

            // OPTIMIZATION: Unroll 8× for better ILP
            size_t r = 0;
            if (reduce_size >= 8) {
#pragma unroll 2
                for (; r + 7 < reduce_size; r += 8) {
                    float v0 = input[base_idx + (r + 0) * inner_size];
                    float v1 = input[base_idx + (r + 1) * inner_size];
                    float v2 = input[base_idx + (r + 2) * inner_size];
                    float v3 = input[base_idx + (r + 3) * inner_size];
                    float v4 = input[base_idx + (r + 4) * inner_size];
                    float v5 = input[base_idx + (r + 5) * inner_size];
                    float v6 = input[base_idx + (r + 6) * inner_size];
                    float v7 = input[base_idx + (r + 7) * inner_size];

                    // Balanced tree reduction
                    float m01 = fmaxf(v0, v1);
                    float m23 = fmaxf(v2, v3);
                    float m45 = fmaxf(v4, v5);
                    float m67 = fmaxf(v6, v7);
                    float m0123 = fmaxf(m01, m23);
                    float m4567 = fmaxf(m45, m67);
                    max_val = fmaxf(max_val, fmaxf(m0123, m4567));
                }
            }

// Handle remainder
#pragma unroll 4
            for (; r < reduce_size; ++r) {
                max_val = fmaxf(max_val, input[base_idx + r * inner_size]);
            }

            output[out_idx] = max_val;
        }
    }

    /**
     * @brief FUSED strided mean reduction - avoids separate division kernel
     *
     * OPTIMIZATION: 8× unrolling + fused division (no separate Thrust transform!)
     */
    __global__ void warp_strided_reduce_mean_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        size_t outer_size,
        size_t reduce_size,
        size_t inner_size) {
        size_t output_elements = outer_size * inner_size;
        size_t stride = blockDim.x * gridDim.x;
        const float inv_size = 1.0f / static_cast<float>(reduce_size);

        for (size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
             out_idx < output_elements;
             out_idx += stride) {
            size_t outer_idx = out_idx / inner_size;
            size_t inner_idx = out_idx % inner_size;

            float sum = 0.0f;
            size_t base_idx = outer_idx * reduce_size * inner_size + inner_idx;

            // OPTIMIZATION: Unroll 8× for better ILP
            size_t r = 0;
            if (reduce_size >= 8) {
#pragma unroll 2
                for (; r + 7 < reduce_size; r += 8) {
                    float v0 = input[base_idx + (r + 0) * inner_size];
                    float v1 = input[base_idx + (r + 1) * inner_size];
                    float v2 = input[base_idx + (r + 2) * inner_size];
                    float v3 = input[base_idx + (r + 3) * inner_size];
                    float v4 = input[base_idx + (r + 4) * inner_size];
                    float v5 = input[base_idx + (r + 5) * inner_size];
                    float v6 = input[base_idx + (r + 6) * inner_size];
                    float v7 = input[base_idx + (r + 7) * inner_size];

                    sum += v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7;
                }
            }

// Handle remainder
#pragma unroll 4
            for (; r < reduce_size; ++r) {
                sum += input[base_idx + r * inner_size];
            }

            // FUSED: Division happens here, no separate kernel!
            output[out_idx] = sum * inv_size;
        }
    }

    /**
     * @brief OPTIMIZED strided min reduction for non-contiguous segments
     *
     * OPTIMIZATION: 8× unrolling + balanced tree reduction for better ILP
     */
    __global__ void warp_strided_reduce_min_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        size_t outer_size,
        size_t reduce_size,
        size_t inner_size) {
        size_t output_elements = outer_size * inner_size;
        size_t stride = blockDim.x * gridDim.x;

        for (size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
             out_idx < output_elements;
             out_idx += stride) {
            size_t outer_idx = out_idx / inner_size;
            size_t inner_idx = out_idx % inner_size;

            float min_val = CUDA_INFINITY;
            size_t base_idx = outer_idx * reduce_size * inner_size + inner_idx;

            // OPTIMIZATION: Unroll 8× for better ILP
            size_t r = 0;
            if (reduce_size >= 8) {
#pragma unroll 2
                for (; r + 7 < reduce_size; r += 8) {
                    float v0 = input[base_idx + (r + 0) * inner_size];
                    float v1 = input[base_idx + (r + 1) * inner_size];
                    float v2 = input[base_idx + (r + 2) * inner_size];
                    float v3 = input[base_idx + (r + 3) * inner_size];
                    float v4 = input[base_idx + (r + 4) * inner_size];
                    float v5 = input[base_idx + (r + 5) * inner_size];
                    float v6 = input[base_idx + (r + 6) * inner_size];
                    float v7 = input[base_idx + (r + 7) * inner_size];

                    // Balanced tree reduction
                    float m01 = fminf(v0, v1);
                    float m23 = fminf(v2, v3);
                    float m45 = fminf(v4, v5);
                    float m67 = fminf(v6, v7);
                    float m0123 = fminf(m01, m23);
                    float m4567 = fminf(m45, m67);
                    min_val = fminf(min_val, fminf(m0123, m4567));
                }
            }

// Handle remainder
#pragma unroll 4
            for (; r < reduce_size; ++r) {
                min_val = fminf(min_val, input[base_idx + r * inner_size]);
            }

            output[out_idx] = min_val;
        }
    }

    // ============= HOST LAUNCH FUNCTIONS =============

    /**
     * @brief Launch optimized TWO-STAGE reduction (llm.c pattern)
     *
     * KEY OPTIMIZATIONS:
     * 1. Two-stage reduction (eliminates atomic contention)
     * 2. Packed128 vectorized loads (4× bandwidth)
     * 3. Streaming cache hints (bypass L1 for better cache utilization)
     * 4. GPU-aware grid sizing (fill all SMs optimally)
     *
     * Expected: 2-4× speedup on large reductions!
     *
     * @param input Input tensor data
     * @param output Output scalar
     * @param n Number of elements
     * @param op Reduction operation (ReduceOp enum)
     * @param stream CUDA stream
     * @param partial_buffer Pre-allocated buffer for partial sums (or nullptr to allocate)
     */
    void launch_warp_reduce_full_two_stage(
        const float* input,
        float* output,
        size_t n,
        ReduceOp op,
        cudaStream_t stream,
        float* partial_buffer = nullptr) {
        if (n == 0)
            return;

        // Check alignment for Packed128 loads
        bool is_aligned = is_aligned_128(input);

        // OPTIMIZATION: GPU-aware grid sizing!
        // Fill all SMs optimally (no tail effects)
        constexpr int BLOCK_SIZE = 256;
        const auto& gpu = GPUConfig::get();
        int grid_size = gpu.optimal_grid_size(BLOCK_SIZE);

        // No cap needed - two-stage reduction handles any size efficiently

        // Allocate partial buffer using stream-ordered allocation (CUDA 12.8+)
        float* partial = partial_buffer;
        bool need_free = false;

        if (partial == nullptr) {
            cudaMallocAsync(&partial, grid_size * sizeof(float), stream);
            need_free = true;
        }

        switch (op) {
        case ReduceOp::Sum:
        case ReduceOp::Mean: // Mean handled as sum, then divided by caller
            // TWO-STAGE REDUCTION (eliminates atomic contention!)
            // Stage 1: Each block reduces to a partial sum (no atomics!)
            warp_reduce_sum_stage1_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                input, partial, n, is_aligned);

            // Stage 2: Single block aggregates partial sums (deterministic!)
            warp_reduce_sum_stage2_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
                partial, output, grid_size);
            break;

        case ReduceOp::Max:
            warp_reduce_max_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                input, output, n, is_aligned);
            break;
        case ReduceOp::Min:
            warp_reduce_min_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                input, output, n, is_aligned);
            break;
        case ReduceOp::Prod:
            warp_reduce_prod_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                input, output, n, is_aligned);
            break;
        default:
            break;
        }

        // Free partial buffer if we allocated it
        if (need_free) {
            cudaFreeAsync(partial, stream);
        }
    }

    /**
     * @brief OLD single-stage launch (kept for backward compatibility)
     *
     * Use launch_warp_reduce_full_two_stage() for better performance!
     */
    void launch_warp_reduce_full(
        const float* input,
        float* output,
        size_t n,
        ReduceOp op,
        cudaStream_t stream) {
        if (n == 0)
            return;

        // For large reductions, use two-stage (much faster!)
        if (n > 100000) {
            return launch_warp_reduce_full_two_stage(input, output, n, op, stream);
        }

        // For small reductions, single-stage is fine
        bool is_aligned = (reinterpret_cast<uintptr_t>(input) % 16) == 0;

        constexpr int BLOCK_SIZE = 256;
        int num_vec_elements = (n + 3) / 4;
        int grid_size = (num_vec_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        // No cap - single-stage warp reduce handles any size

        switch (op) {
        case ReduceOp::Sum:
        case ReduceOp::Mean:
            warp_reduce_sum_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                input, output, n, is_aligned);
            break;
        case ReduceOp::Max:
            warp_reduce_max_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                input, output, n, is_aligned);
            break;
        case ReduceOp::Min:
            warp_reduce_min_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                input, output, n, is_aligned);
            break;
        case ReduceOp::Prod:
            warp_reduce_prod_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                input, output, n, is_aligned);
            break;
        default:
            break;
        }
    }

    /**
     * @brief Launch optimized warp-level segmented reduction
     *
     * This is the fast path for axis reductions when:
     * - Segments are contiguous in memory
     * - Segment size is small to medium (< 100K elements)
     * - Number of segments is reasonable (< 10M)
     *
     * @param input Input tensor data
     * @param output Output array (one value per segment)
     * @param num_segments Number of segments
     * @param segment_size Size of each segment
     * @param op Reduction operation
     * @param stream CUDA stream
     */
    void launch_warp_segmented_reduce(
        const float* input,
        float* output,
        size_t num_segments,
        size_t segment_size,
        ReduceOp op,
        cudaStream_t stream) {
        if (num_segments == 0 || segment_size == 0)
            return;

        // Special case: TINY segments (< 32 elements)
        // Each thread processes one complete segment sequentially
        // Much more efficient than using a whole block per segment!
        if (segment_size < 32) {
            constexpr int BLOCK_SIZE = 256;
            int grid_size = (num_segments + BLOCK_SIZE - 1) / BLOCK_SIZE;
            // No cap - each thread handles one segment

            switch (op) {
            case ReduceOp::Sum:
                warp_tiny_segment_reduce_sum_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                    input, output, num_segments, segment_size);
                break;
            case ReduceOp::Mean:
                // FUSED: Division happens in kernel, no separate transform needed
                warp_tiny_segment_reduce_mean_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                    input, output, num_segments, segment_size);
                break;
            case ReduceOp::Max:
                warp_tiny_segment_reduce_max_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                    input, output, num_segments, segment_size);
                break;
            case ReduceOp::Min:
                warp_tiny_segment_reduce_min_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                    input, output, num_segments, segment_size);
                break;
            case ReduceOp::Prod:
                warp_tiny_segment_reduce_prod_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                    input, output, num_segments, segment_size);
                break;
            default:
                break;
            }
            return;
        }

        // MEDIUM segments (32-2048 elements): Use warp-per-segment kernel
        // Each warp processes one segment using only warp shuffles (NO __syncthreads__!)
        if (segment_size <= 2048) {
            constexpr int BLOCK_SIZE = 256; // 8 warps per block
            constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / 32;
            // Use optimal grid size for GPU occupancy with grid-stride loop
            int min_blocks = (num_segments + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
            const auto& gpu = GPUConfig::get();
            int grid_size = std::max(min_blocks, gpu.optimal_grid_size(BLOCK_SIZE));
            LOG_DEBUG("[REDUCE] Medium kernel: segments={} segment_size={} grid={}", num_segments, segment_size, grid_size);

            switch (op) {
            case ReduceOp::Sum:
                warp_medium_segment_reduce_sum_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                    input, output, num_segments, segment_size);
                break;
            case ReduceOp::Mean:
                warp_medium_segment_reduce_mean_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                    input, output, num_segments, segment_size);
                break;
            case ReduceOp::Max:
                warp_medium_segment_reduce_max_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                    input, output, num_segments, segment_size);
                break;
            case ReduceOp::Min:
                warp_medium_segment_reduce_min_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                    input, output, num_segments, segment_size);
                break;
            case ReduceOp::Prod:
                warp_medium_segment_reduce_prod_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                    input, output, num_segments, segment_size);
                break;
            default:
                break;
            }
            return;
        }

        // LARGE segments (> 2048 elements): Use block-level reduction
        // Each block processes one segment with shared memory reduction
        constexpr int BLOCK_SIZE = 256;
        int grid_size = num_segments; // One block per segment (or less if very many)

        switch (op) {
        case ReduceOp::Sum:
            warp_segmented_reduce_sum_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                input, output, num_segments, segment_size);
            break;
        case ReduceOp::Mean:
            // FUSED: Division happens in kernel, no separate transform needed
            warp_segmented_reduce_mean_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                input, output, num_segments, segment_size);
            break;
        case ReduceOp::Max:
            warp_segmented_reduce_max_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                input, output, num_segments, segment_size);
            break;
        case ReduceOp::Min:
            warp_segmented_reduce_min_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                input, output, num_segments, segment_size);
            break;
        case ReduceOp::Prod:
            warp_segmented_reduce_prod_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                input, output, num_segments, segment_size);
            break;
        default:
            break;
        }
    }

    /**
     * @brief Launch optimized strided reduction for non-contiguous segments
     *
     * This handles reductions where inner_size > 1 (e.g., reducing along dim 0 or dim 1).
     * Uses a grid-stride kernel with good occupancy.
     *
     * @param input Input tensor data
     * @param output Output array
     * @param outer_size Number of outer dimensions
     * @param reduce_size Size of reduction dimension
     * @param inner_size Size of inner dimensions (stride between reduction elements)
     * @param op Reduction operation
     * @param stream CUDA stream
     */
    void launch_warp_strided_reduce(
        const float* input,
        float* output,
        size_t outer_size,
        size_t reduce_size,
        size_t inner_size,
        ReduceOp op,
        cudaStream_t stream) {
        if (outer_size == 0 || reduce_size == 0 || inner_size == 0)
            return;

        size_t output_elements = outer_size * inner_size;

        // Optimal configuration: 256 threads per block
        constexpr int BLOCK_SIZE = 256;
        int grid_size = (output_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        // No cap - warp-level reduction scales well

        switch (op) {
        case ReduceOp::Sum:
            warp_strided_reduce_sum_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                input, output, outer_size, reduce_size, inner_size);
            break;
        case ReduceOp::Mean:
            // FUSED: Division happens in kernel, no separate transform needed
            warp_strided_reduce_mean_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                input, output, outer_size, reduce_size, inner_size);
            break;
        case ReduceOp::Max:
            warp_strided_reduce_max_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                input, output, outer_size, reduce_size, inner_size);
            break;
        case ReduceOp::Min:
            warp_strided_reduce_min_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                input, output, outer_size, reduce_size, inner_size);
            break;
        default:
            break;
        }
    }

    /**
     * @brief Multi-axis reduction kernel for contiguous axes
     *
     * When reducing multiple contiguous axes, we can treat it as a single-axis
     * reduction with a larger reduce_size. This is much faster than the generic
     * multi-axis kernel.
     *
     * Example: sum({0, 1}) on [256, 256, 64] reduces 256*256=65536 elements
     *          to produce each of the 64 output values.
     *
     * Each block processes one output element using warp reductions.
     */
    __global__ void warp_multi_axis_reduce_sum_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        size_t output_size,
        size_t reduce_count) {
        size_t out_idx = blockIdx.x;
        if (out_idx >= output_size)
            return;

        // Each output element requires summing reduce_count input elements
        const float* segment_start = input + out_idx * reduce_count;

        // Use vectorized segment reduction
        float result = warp_ops::vectorized_segment_reduce_sum(segment_start, reduce_count);

        if (threadIdx.x == 0) {
            output[out_idx] = result;
        }
    }

    /**
     * @brief Multi-axis max reduction kernel for contiguous axes
     */
    __global__ void warp_multi_axis_reduce_max_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        size_t output_size,
        size_t reduce_count) {
        size_t out_idx = blockIdx.x;
        if (out_idx >= output_size)
            return;

        const float* segment_start = input + out_idx * reduce_count;
        float result = warp_ops::vectorized_segment_reduce_max(segment_start, reduce_count);

        if (threadIdx.x == 0) {
            output[out_idx] = result;
        }
    }

    /**
     * @brief Multi-axis min reduction kernel for contiguous axes
     */
    __global__ void warp_multi_axis_reduce_min_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        size_t output_size,
        size_t reduce_count) {
        size_t out_idx = blockIdx.x;
        if (out_idx >= output_size)
            return;

        const float* segment_start = input + out_idx * reduce_count;
        float result = warp_ops::vectorized_segment_reduce_min(segment_start, reduce_count);

        if (threadIdx.x == 0) {
            output[out_idx] = result;
        }
    }

    /**
     * @brief Launch optimized multi-axis warp reduction
     *
     * This is for reducing multiple contiguous axes. Much faster than the generic
     * multi-axis kernel which has poor index computation overhead.
     *
     * @param input Input tensor data
     * @param output Output array
     * @param output_size Number of output elements
     * @param reduce_count Number of elements to reduce per output
     * @param op Reduction operation
     * @param stream CUDA stream
     */
    void launch_warp_multi_axis_reduce(
        const float* input,
        float* output,
        size_t output_size,
        size_t reduce_count,
        ReduceOp op,
        cudaStream_t stream) {
        if (output_size == 0 || reduce_count == 0)
            return;

        // Each block processes one output element
        // 256 threads per block for good warp utilization
        constexpr int BLOCK_SIZE = 256;
        int grid_size = output_size;

        switch (op) {
        case ReduceOp::Sum:
        case ReduceOp::Mean: // Mean handled as sum, then divided by caller
            warp_multi_axis_reduce_sum_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                input, output, output_size, reduce_count);
            break;
        case ReduceOp::Max:
            warp_multi_axis_reduce_max_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                input, output, output_size, reduce_count);
            break;
        case ReduceOp::Min:
            warp_multi_axis_reduce_min_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
                input, output, output_size, reduce_count);
            break;
        default:
            break;
        }
    }

    // Column reduction for 2D matrices [M, N] -> [N]
    // Uses 2D grid: X for columns, Y for row partitioning with atomic accumulation

    __global__ void column_reduce_sum_kernel(
        const float* __restrict__ input, float* __restrict__ output,
        size_t M, size_t N) {
        const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= N)
            return;

        const size_t rows_per_block = (M + gridDim.y - 1) / gridDim.y;
        const size_t row_start = blockIdx.y * rows_per_block;
        const size_t row_end = min(row_start + rows_per_block, M);

        float sum = 0.0f;
        size_t row = row_start;
        for (; row + 3 < row_end; row += 4) {
            sum += input[row * N + col] + input[(row + 1) * N + col] +
                   input[(row + 2) * N + col] + input[(row + 3) * N + col];
        }
        for (; row < row_end; row++) {
            sum += input[row * N + col];
        }

        if (gridDim.y == 1) {
            output[col] = sum;
        } else {
            atomicAdd(&output[col], sum);
        }
    }

    __global__ void column_reduce_max_kernel(
        const float* __restrict__ input, float* __restrict__ output,
        size_t M, size_t N) {
        const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= N)
            return;

        const size_t rows_per_block = (M + gridDim.y - 1) / gridDim.y;
        const size_t row_start = blockIdx.y * rows_per_block;
        const size_t row_end = min(row_start + rows_per_block, M);

        float val = -FLT_MAX;
        for (size_t row = row_start; row < row_end; row++) {
            val = fmaxf(val, input[row * N + col]);
        }

        if (gridDim.y == 1) {
            output[col] = val;
        } else {
            int* out_int = reinterpret_cast<int*>(output + col);
            int old = *out_int, assumed;
            do {
                assumed = old;
                old = atomicCAS(out_int, assumed, __float_as_int(fmaxf(__int_as_float(assumed), val)));
            } while (assumed != old);
        }
    }

    __global__ void column_reduce_min_kernel(
        const float* __restrict__ input, float* __restrict__ output,
        size_t M, size_t N) {
        const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= N)
            return;

        const size_t rows_per_block = (M + gridDim.y - 1) / gridDim.y;
        const size_t row_start = blockIdx.y * rows_per_block;
        const size_t row_end = min(row_start + rows_per_block, M);

        float val = FLT_MAX;
        for (size_t row = row_start; row < row_end; row++) {
            val = fminf(val, input[row * N + col]);
        }

        if (gridDim.y == 1) {
            output[col] = val;
        } else {
            int* out_int = reinterpret_cast<int*>(output + col);
            int old = *out_int, assumed;
            do {
                assumed = old;
                old = atomicCAS(out_int, assumed, __float_as_int(fminf(__int_as_float(assumed), val)));
            } while (assumed != old);
        }
    }

    void launch_column_reduce(const float* input, float* output,
                              size_t M, size_t N, ReduceOp op, cudaStream_t stream) {
        constexpr int BLOCK = 256;
        int grid_x = (N + BLOCK - 1) / BLOCK;
        int grid_y = 1;
        if (M > 512) {
            int sm_count = GPUConfig::get().sm_count;
            int target = std::max(1, sm_count * 2 / std::max(grid_x, 1));
            grid_y = std::min((int)((M + 127) / 128), target);
        }
        dim3 grid(grid_x, grid_y);

        switch (op) {
        case ReduceOp::Sum:
        case ReduceOp::Mean:
            if (grid_y > 1)
                cudaMemsetAsync(output, 0, N * sizeof(float), stream);
            column_reduce_sum_kernel<<<grid, BLOCK, 0, stream>>>(input, output, M, N);
            if (op == ReduceOp::Mean) {
                float inv_M = 1.0f / static_cast<float>(M);
                thrust::transform(thrust::cuda::par.on(stream),
                                  thrust::device_ptr<float>(output), thrust::device_ptr<float>(output + N),
                                  thrust::device_ptr<float>(output), [inv_M] __device__(float x) { return x * inv_M; });
            }
            break;
        case ReduceOp::Max:
            if (grid_y > 1) {
                thrust::fill(thrust::cuda::par.on(stream),
                             thrust::device_ptr<float>(output), thrust::device_ptr<float>(output + N), -FLT_MAX);
            }
            column_reduce_max_kernel<<<grid, BLOCK, 0, stream>>>(input, output, M, N);
            break;
        case ReduceOp::Min:
            if (grid_y > 1) {
                thrust::fill(thrust::cuda::par.on(stream),
                             thrust::device_ptr<float>(output), thrust::device_ptr<float>(output + N), FLT_MAX);
            }
            column_reduce_min_kernel<<<grid, BLOCK, 0, stream>>>(input, output, M, N);
            break;
        default:
            break;
        }
    }

    /**
     * @brief Determine if warp-level reduction should be used
     *
     * Warp reductions are faster for:
     * - Small to medium tensors (< 10M elements)
     * - Full reductions (entire tensor to scalar)
     * - Contiguous segments (good cache locality)
     *
     * For strided reductions or very large tensors, CUB is still better.
     */
    bool should_use_warp_reduce(size_t n, size_t num_segments) {
        // SCALAR REDUCTIONS: Always use CUB DeviceReduce (much faster!)
        // Benchmarks show CUB is 3-7x faster than our warp kernels for scalar reductions.
        if (num_segments == 1) {
            return false; // Use CUB path in tensor_ops.cu
        }

        // SEGMENTED REDUCTIONS: Use warp kernels for small-medium tensors
        // Our warp kernels are competitive for segmented reductions with good locality.
        size_t segment_size = n / num_segments;

        // Use warp reduce when:
        // - Reasonable segment sizes (< 100K elements per segment)
        // - Reasonable number of segments (< 1M segments)
        // - Total tensor size under 10M elements
        bool use_warp = num_segments > 1 &&
                        num_segments < 1000000 &&
                        segment_size < 100000 &&
                        n < 10000000;

        return use_warp;
    }

} // namespace lfs::core::tensor_ops
