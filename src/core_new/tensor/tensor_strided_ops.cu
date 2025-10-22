/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/tensor_impl.hpp"
#include "internal/tensor_ops.hpp"
#include <cuda_runtime.h>

namespace lfs::core {
    namespace tensor_ops {

        template <typename T>
        __global__ void strided_copy_kernel(
            const T* __restrict__ input,
            T* __restrict__ output,
            const size_t* __restrict__ shape,
            const size_t* __restrict__ strides,
            size_t rank,
            size_t total_elements) {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= total_elements)
                return;

            // Convert flat output index to multi-dimensional coordinates
            size_t tmp = idx;
            size_t input_offset = 0;

            for (int d = rank - 1; d >= 0; --d) {
                size_t coord = tmp % shape[d];
                tmp /= shape[d];
                input_offset += coord * strides[d];
            }

            // Copy element from strided input to contiguous output
            output[idx] = input[input_offset];
        }

        void launch_strided_copy(
            const void* input,
            void* output,
            const size_t* shape,
            const size_t* strides,
            size_t rank,
            size_t total_elements,
            DataType dtype,
            cudaStream_t stream) {
            const int block_size = 256;
            const int num_blocks = (total_elements + block_size - 1) / block_size;

            switch (dtype) {
            case DataType::Float32:
                strided_copy_kernel<<<num_blocks, block_size, 0, stream>>>(
                    static_cast<const float*>(input),
                    static_cast<float*>(output),
                    shape, strides, rank, total_elements);
                break;
            case DataType::Int32:
                strided_copy_kernel<<<num_blocks, block_size, 0, stream>>>(
                    static_cast<const int32_t*>(input),
                    static_cast<int32_t*>(output),
                    shape, strides, rank, total_elements);
                break;
            case DataType::Int64:
                strided_copy_kernel<<<num_blocks, block_size, 0, stream>>>(
                    static_cast<const int64_t*>(input),
                    static_cast<int64_t*>(output),
                    shape, strides, rank, total_elements);
                break;
            case DataType::UInt8:
                strided_copy_kernel<<<num_blocks, block_size, 0, stream>>>(
                    static_cast<const uint8_t*>(input),
                    static_cast<uint8_t*>(output),
                    shape, strides, rank, total_elements);
                break;
            case DataType::Bool:
                strided_copy_kernel<<<num_blocks, block_size, 0, stream>>>(
                    static_cast<const bool*>(input),
                    static_cast<bool*>(output),
                    shape, strides, rank, total_elements);
                break;
            default:
                // Unsupported dtype - do nothing
                break;
            }
        }

        // ============= Fused Strided Upload Kernel =============
        // Reads from PINNED HOST memory with strides, writes contiguously to GPU
        // This eliminates CPU-side materialization entirely!

        // ============= GATHER KERNEL (Good for small strides) =============
        // OPTIMIZED: Rank-3 gather - good when input stride pattern is cache-friendly
        // Iteration order: i2 (fastest) → i1 → i0 (slowest)
        // Best when: s2 is small (e.g., HWC→CHW where s2=1)
        template <typename T>
        __global__ void strided_upload_kernel_rank3_gather(
            const T* __restrict__ host_input, // Pinned host memory (non-contiguous)
            T* __restrict__ gpu_output,       // GPU memory (contiguous)
            size_t d0, size_t d1, size_t d2,  // Shape (immediate parameters)
            size_t s0, size_t s1, size_t s2,  // Strides (immediate parameters)
            size_t total_elements) {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= total_elements)
                return;

            // Convert flat output index to multi-dimensional coordinates
            // Iteration order: i2 → i1 → i0
            size_t i2 = idx % d2;
            size_t tmp = idx / d2;
            size_t i1 = tmp % d1;
            size_t i0 = tmp / d1;

            size_t input_offset = i0 * s0 + i1 * s1 + i2 * s2;

            // Read from pinned host memory and write to GPU memory
            gpu_output[idx] = host_input[input_offset];
        }

        // ============= SPECIALIZED HWC→CHW KERNEL =============
        // CRITICAL OPTIMIZATION for image uploads (e.g., 720×820×3 → 3×720×820)
        // Input: [H, W, C] with strides [W*C, C, 1] (contiguous HWC in host memory)
        // Output: [C, H, W] contiguous (GPU memory)
        //
        // KEY: Iterate over INPUT in native HWC order to maximize PCIe bandwidth!
        // Threads process: h → w → c (keeps host reads contiguous)
        template <typename T>
        __global__ void strided_upload_kernel_hwc_to_chw(
            const T* __restrict__ host_input,
            T* __restrict__ gpu_output,
            size_t H, size_t W, size_t C) {

            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            size_t total = H * W * C;
            if (idx >= total)
                return;

            // Decompose linear index in HWC order (INPUT layout)
            // idx = h*W*C + w*C + c
            size_t c = idx % C;
            size_t tmp = idx / C;
            size_t w = tmp % W;
            size_t h = tmp / W;

            // Input offset (already in HWC order, contiguous)
            size_t input_offset = idx;

            // Output offset in CHW order: [C, H, W]
            size_t output_offset = c * (H * W) + h * W + w;

            // Consecutive threads read consecutive host memory (FAST PCIe!)
            gpu_output[output_offset] = host_input[input_offset];
        }

        // ============= ADAPTIVE GATHER KERNELS (Different iteration orders) =============
        // These kernels iterate in different orders to optimize for different stride patterns

        // Iteration order: i1 → i2 → i0 (good when s1 is smallest)
        template <typename T>
        __global__ void strided_upload_kernel_rank3_gather_order_120(
            const T* __restrict__ host_input,
            T* __restrict__ gpu_output,
            size_t d0, size_t d1, size_t d2,
            size_t s0, size_t s1, size_t s2,
            size_t total_elements) {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= total_elements)
                return;

            // Iteration order: i1 → i2 → i0
            size_t i1 = idx % d1;
            size_t tmp = idx / d1;
            size_t i2 = tmp % d2;
            size_t i0 = tmp / d2;

            // Input offset with strides
            size_t input_offset = i0 * s0 + i1 * s1 + i2 * s2;

            // Output offset in standard row-major order [d0, d1, d2]
            size_t output_offset = i0 * (d1 * d2) + i1 * d2 + i2;

            gpu_output[output_offset] = host_input[input_offset];
        }

        // Iteration order: i0 → i1 → i2 (good when s0 is smallest)
        template <typename T>
        __global__ void strided_upload_kernel_rank3_gather_order_012(
            const T* __restrict__ host_input,
            T* __restrict__ gpu_output,
            size_t d0, size_t d1, size_t d2,
            size_t s0, size_t s1, size_t s2,
            size_t total_elements) {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= total_elements)
                return;

            // Iteration order: i0 → i1 → i2
            size_t i0 = idx % d0;
            size_t tmp = idx / d0;
            size_t i1 = tmp % d1;
            size_t i2 = tmp / d1;

            // Input offset with strides
            size_t input_offset = i0 * s0 + i1 * s1 + i2 * s2;

            // Output offset in standard row-major order [d0, d1, d2]
            size_t output_offset = i0 * (d1 * d2) + i1 * d2 + i2;

            gpu_output[output_offset] = host_input[input_offset];
        }

        // Generic fallback for arbitrary rank
        template <typename T>
        __global__ void strided_upload_kernel(
            const T* __restrict__ host_input, // Pinned host memory (non-contiguous)
            T* __restrict__ gpu_output,       // GPU memory (contiguous)
            const size_t* __restrict__ shape,
            const size_t* __restrict__ strides,
            size_t rank,
            size_t total_elements) {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= total_elements)
                return;

            // Convert flat output index to multi-dimensional coordinates
            size_t tmp = idx;
            size_t input_offset = 0;

            // Generic path for arbitrary rank
            for (int d = rank - 1; d >= 0; --d) {
                size_t coord = tmp % shape[d];
                tmp /= shape[d];
                input_offset += coord * strides[d];
            }

            // Read from pinned host memory and write to GPU memory
            gpu_output[idx] = host_input[input_offset];
        }

        void launch_strided_upload(
            const void* host_input,
            void* gpu_output,
            const size_t* d_shape,
            const size_t* d_strides,
            size_t rank,
            size_t total_elements,
            DataType dtype,
            cudaStream_t stream) {

            // Use larger blocks for better occupancy during PCIe reads
            const int block_size = 256;
            const int num_blocks = (total_elements + block_size - 1) / block_size;

            // FAST PATH: Rank-3 with adaptive kernel selection based on stride pattern
            if (rank == 3) {
                size_t s0 = d_strides[0];
                size_t s1 = d_strides[1];
                size_t s2 = d_strides[2];

                // ============= DETECT HWC→CHW PATTERN =============
                // This is the CRITICAL case for image uploads!
                // Pattern after permute({2, 0, 1}):
                //   d_shape = [C, H, W] (e.g., [3, 720, 820])
                //   d_strides = [1, W*C, C] (e.g., [1, 2460, 3])
                //
                // Detection: s0==1, s2==C, s1==W*C
                bool is_hwc_to_chw = (s0 == 1) &&
                                     (s2 == d_shape[0]) &&
                                     (s1 == s2 * d_shape[2]) &&
                                     (d_shape[0] <= 4); // C is typically 1, 3, or 4

                if (is_hwc_to_chw) {
                    // Use specialized HWC→CHW kernel (H, W, C order in params)
                    size_t C = d_shape[0];
                    size_t H = d_shape[1];
                    size_t W = d_shape[2];

                    switch (dtype) {
                    case DataType::Float32:
                        strided_upload_kernel_hwc_to_chw<<<num_blocks, block_size, 0, stream>>>(
                            static_cast<const float*>(host_input),
                            static_cast<float*>(gpu_output),
                            H, W, C);
                        break;
                    case DataType::UInt8:
                        strided_upload_kernel_hwc_to_chw<<<num_blocks, block_size, 0, stream>>>(
                            static_cast<const uint8_t*>(host_input),
                            static_cast<uint8_t*>(gpu_output),
                            H, W, C);
                        break;
                    default:
                        // Fall through to generic path
                        break;
                    }
                    return;
                }

                // ============= GENERIC STRIDE SELECTION =============
                // Analyze strides to determine optimal iteration order
                // Key insight: We want consecutive threads to access consecutive (or nearby) memory
                // Best strategy: Iterate over the dimension with smallest stride AND large size
                //
                // Heuristic: Compute "access score" = stride * warp_jump_penalty
                // Lower score = better memory access pattern

                // Warp size is 32 threads - if dimension is smaller than 32,
                // we lose coalescing efficiency due to wrap-around
                constexpr size_t WARP_SIZE = 32;

                // Compute "effective stride" considering warp coalescing
                // If dim < WARP_SIZE, threads will wrap around and jump by next_stride
                auto effective_stride = [](size_t stride, size_t dim, size_t next_stride) {
                    if (dim >= WARP_SIZE) {
                        return stride; // Good: full warp stays contiguous
                    } else {
                        // Bad: warp wraps around after 'dim' elements
                        return stride + (WARP_SIZE / dim) * next_stride;
                    }
                };

                size_t eff_s0 = effective_stride(s0, d_shape[0], s1);
                size_t eff_s1 = effective_stride(s1, d_shape[1], s2);
                size_t eff_s2 = s2; // Last dimension, no next stride

                // Determine which dimension has the best effective stride
                enum class IterOrder { ORDER_210,
                                       ORDER_120,
                                       ORDER_012 };
                IterOrder order;

                if (eff_s2 <= eff_s1 && eff_s2 <= eff_s0) {
                    // s2 is best: iterate i2 → i1 → i0 (default, best for HWC→CHW)
                    order = IterOrder::ORDER_210;
                } else if (eff_s1 <= eff_s0 && eff_s1 <= eff_s2) {
                    // s1 is best: iterate i1 → i2 → i0
                    order = IterOrder::ORDER_120;
                } else {
                    // s0 is best: iterate i0 → i1 → i2
                    order = IterOrder::ORDER_012;
                }

                // Dispatch to appropriate kernel based on dtype and iteration order
                switch (dtype) {
                case DataType::Float32:
                    if (order == IterOrder::ORDER_210) {
                        strided_upload_kernel_rank3_gather<<<num_blocks, block_size, 0, stream>>>(
                            static_cast<const float*>(host_input),
                            static_cast<float*>(gpu_output),
                            d_shape[0], d_shape[1], d_shape[2],
                            d_strides[0], d_strides[1], d_strides[2],
                            total_elements);
                    } else if (order == IterOrder::ORDER_120) {
                        strided_upload_kernel_rank3_gather_order_120<<<num_blocks, block_size, 0, stream>>>(
                            static_cast<const float*>(host_input),
                            static_cast<float*>(gpu_output),
                            d_shape[0], d_shape[1], d_shape[2],
                            d_strides[0], d_strides[1], d_strides[2],
                            total_elements);
                    } else {
                        strided_upload_kernel_rank3_gather_order_012<<<num_blocks, block_size, 0, stream>>>(
                            static_cast<const float*>(host_input),
                            static_cast<float*>(gpu_output),
                            d_shape[0], d_shape[1], d_shape[2],
                            d_strides[0], d_strides[1], d_strides[2],
                            total_elements);
                    }
                    break;
                case DataType::Int32:
                    if (order == IterOrder::ORDER_210) {
                        strided_upload_kernel_rank3_gather<<<num_blocks, block_size, 0, stream>>>(
                            static_cast<const int32_t*>(host_input),
                            static_cast<int32_t*>(gpu_output),
                            d_shape[0], d_shape[1], d_shape[2],
                            d_strides[0], d_strides[1], d_strides[2],
                            total_elements);
                    } else if (order == IterOrder::ORDER_120) {
                        strided_upload_kernel_rank3_gather_order_120<<<num_blocks, block_size, 0, stream>>>(
                            static_cast<const int32_t*>(host_input),
                            static_cast<int32_t*>(gpu_output),
                            d_shape[0], d_shape[1], d_shape[2],
                            d_strides[0], d_strides[1], d_strides[2],
                            total_elements);
                    } else {
                        strided_upload_kernel_rank3_gather_order_012<<<num_blocks, block_size, 0, stream>>>(
                            static_cast<const int32_t*>(host_input),
                            static_cast<int32_t*>(gpu_output),
                            d_shape[0], d_shape[1], d_shape[2],
                            d_strides[0], d_strides[1], d_strides[2],
                            total_elements);
                    }
                    break;
                case DataType::Int64:
                    if (order == IterOrder::ORDER_210) {
                        strided_upload_kernel_rank3_gather<<<num_blocks, block_size, 0, stream>>>(
                            static_cast<const int64_t*>(host_input),
                            static_cast<int64_t*>(gpu_output),
                            d_shape[0], d_shape[1], d_shape[2],
                            d_strides[0], d_strides[1], d_strides[2],
                            total_elements);
                    } else if (order == IterOrder::ORDER_120) {
                        strided_upload_kernel_rank3_gather_order_120<<<num_blocks, block_size, 0, stream>>>(
                            static_cast<const int64_t*>(host_input),
                            static_cast<int64_t*>(gpu_output),
                            d_shape[0], d_shape[1], d_shape[2],
                            d_strides[0], d_strides[1], d_strides[2],
                            total_elements);
                    } else {
                        strided_upload_kernel_rank3_gather_order_012<<<num_blocks, block_size, 0, stream>>>(
                            static_cast<const int64_t*>(host_input),
                            static_cast<int64_t*>(gpu_output),
                            d_shape[0], d_shape[1], d_shape[2],
                            d_strides[0], d_strides[1], d_strides[2],
                            total_elements);
                    }
                    break;
                case DataType::UInt8:
                    if (order == IterOrder::ORDER_210) {
                        strided_upload_kernel_rank3_gather<<<num_blocks, block_size, 0, stream>>>(
                            static_cast<const uint8_t*>(host_input),
                            static_cast<uint8_t*>(gpu_output),
                            d_shape[0], d_shape[1], d_shape[2],
                            d_strides[0], d_strides[1], d_strides[2],
                            total_elements);
                    } else if (order == IterOrder::ORDER_120) {
                        strided_upload_kernel_rank3_gather_order_120<<<num_blocks, block_size, 0, stream>>>(
                            static_cast<const uint8_t*>(host_input),
                            static_cast<uint8_t*>(gpu_output),
                            d_shape[0], d_shape[1], d_shape[2],
                            d_strides[0], d_strides[1], d_strides[2],
                            total_elements);
                    } else {
                        strided_upload_kernel_rank3_gather_order_012<<<num_blocks, block_size, 0, stream>>>(
                            static_cast<const uint8_t*>(host_input),
                            static_cast<uint8_t*>(gpu_output),
                            d_shape[0], d_shape[1], d_shape[2],
                            d_strides[0], d_strides[1], d_strides[2],
                            total_elements);
                    }
                    break;
                case DataType::Bool:
                    if (order == IterOrder::ORDER_210) {
                        strided_upload_kernel_rank3_gather<<<num_blocks, block_size, 0, stream>>>(
                            static_cast<const bool*>(host_input),
                            static_cast<bool*>(gpu_output),
                            d_shape[0], d_shape[1], d_shape[2],
                            d_strides[0], d_strides[1], d_strides[2],
                            total_elements);
                    } else if (order == IterOrder::ORDER_120) {
                        strided_upload_kernel_rank3_gather_order_120<<<num_blocks, block_size, 0, stream>>>(
                            static_cast<const bool*>(host_input),
                            static_cast<bool*>(gpu_output),
                            d_shape[0], d_shape[1], d_shape[2],
                            d_strides[0], d_strides[1], d_strides[2],
                            total_elements);
                    } else {
                        strided_upload_kernel_rank3_gather_order_012<<<num_blocks, block_size, 0, stream>>>(
                            static_cast<const bool*>(host_input),
                            static_cast<bool*>(gpu_output),
                            d_shape[0], d_shape[1], d_shape[2],
                            d_strides[0], d_strides[1], d_strides[2],
                            total_elements);
                    }
                    break;
                default:
                    // Unsupported dtype - do nothing
                    break;
                }
                return;
            }

            // GENERIC PATH: Uses device memory for shape/strides (requires caller to allocate)
            switch (dtype) {
            case DataType::Float32:
                strided_upload_kernel<<<num_blocks, block_size, 0, stream>>>(
                    static_cast<const float*>(host_input),
                    static_cast<float*>(gpu_output),
                    d_shape, d_strides, rank, total_elements);
                break;
            case DataType::Int32:
                strided_upload_kernel<<<num_blocks, block_size, 0, stream>>>(
                    static_cast<const int32_t*>(host_input),
                    static_cast<int32_t*>(gpu_output),
                    d_shape, d_strides, rank, total_elements);
                break;
            case DataType::Int64:
                strided_upload_kernel<<<num_blocks, block_size, 0, stream>>>(
                    static_cast<const int64_t*>(host_input),
                    static_cast<int64_t*>(gpu_output),
                    d_shape, d_strides, rank, total_elements);
                break;
            case DataType::UInt8:
                strided_upload_kernel<<<num_blocks, block_size, 0, stream>>>(
                    static_cast<const uint8_t*>(host_input),
                    static_cast<uint8_t*>(gpu_output),
                    d_shape, d_strides, rank, total_elements);
                break;
            case DataType::Bool:
                strided_upload_kernel<<<num_blocks, block_size, 0, stream>>>(
                    static_cast<const bool*>(host_input),
                    static_cast<bool*>(gpu_output),
                    d_shape, d_strides, rank, total_elements);
                break;
            default:
                // Unsupported dtype - do nothing
                break;
            }
        }

    } // namespace tensor_ops
} // namespace lfs::core
