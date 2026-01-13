/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "morton_encoding.hpp"
#include <cstdint>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

namespace lfs::io {

    namespace {

        // Part1By2 from splat-transform - spreads bits for 10-bit input
        // https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
        __device__ __forceinline__ uint32_t Part1By2(uint32_t x) {
            x &= 0x000003ff;
            x = (x ^ (x << 16)) & 0xff0000ff;
            x = (x ^ (x << 8)) & 0x0300f00f;
            x = (x ^ (x << 4)) & 0x030c30c3;
            x = (x ^ (x << 2)) & 0x09249249;
            return x;
        }

        // Morton encoding: Z-major order
        __device__ __forceinline__ uint32_t encodeMorton3(uint32_t x, uint32_t y, uint32_t z) {
            return (Part1By2(z) << 2) + (Part1By2(y) << 1) + Part1By2(x);
        }

        __global__ void morton_encode_kernel(
            const float* __restrict__ positions,
            int64_t* __restrict__ morton_codes,
            const int n_positions,
            const float min_x, const float min_y, const float min_z,
            const float xmul, const float ymul, const float zmul) {

            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n_positions)
                return;

            const float x = positions[idx * 3 + 0];
            const float y = positions[idx * 3 + 1];
            const float z = positions[idx * 3 + 2];

            // Normalize to [0, 1023] range per-axis
            const uint32_t ix = min(1023u, static_cast<uint32_t>((x - min_x) * xmul));
            const uint32_t iy = min(1023u, static_cast<uint32_t>((y - min_y) * ymul));
            const uint32_t iz = min(1023u, static_cast<uint32_t>((z - min_z) * zmul));

            morton_codes[idx] = static_cast<int64_t>(encodeMorton3(ix, iy, iz));
        }

        struct float3_minmax {
            float3 min_val;
            float3 max_val;

            __host__ __device__
            float3_minmax() : min_val{CUDA_INFINITY, CUDA_INFINITY, CUDA_INFINITY},
                              max_val{-CUDA_INFINITY, -CUDA_INFINITY, -CUDA_INFINITY} {}

            __host__ __device__
            float3_minmax(float3 min_v, float3 max_v) : min_val(min_v),
                                                        max_val(max_v) {}
        };

        struct minmax_op {
            __host__ __device__
                float3_minmax
                operator()(const float3_minmax& a, const float3_minmax& b) const {
                float3_minmax result;
                result.min_val.x = fminf(a.min_val.x, b.min_val.x);
                result.min_val.y = fminf(a.min_val.y, b.min_val.y);
                result.min_val.z = fminf(a.min_val.z, b.min_val.z);
                result.max_val.x = fmaxf(a.max_val.x, b.max_val.x);
                result.max_val.y = fmaxf(a.max_val.y, b.max_val.y);
                result.max_val.z = fmaxf(a.max_val.z, b.max_val.z);
                return result;
            }
        };

        struct position_to_minmax {
            const float* positions;

            __host__ __device__
            position_to_minmax(const float* pos) : positions(pos) {}

            __host__ __device__
                float3_minmax
                operator()(int idx) const {
                float3 pos;
                pos.x = positions[idx * 3 + 0];
                pos.y = positions[idx * 3 + 1];
                pos.z = positions[idx * 3 + 2];
                return float3_minmax(pos, pos);
            }
        };

    } // anonymous namespace

    Tensor morton_encode(const Tensor& positions) {
        using lfs::core::DataType;
        using lfs::core::Device;

        if (!positions.is_valid()) {
            LOG_ERROR("morton_encode: Invalid input tensor");
            return Tensor();
        }

        if (positions.ndim() != 2 || positions.size(1) != 3) {
            LOG_ERROR("morton_encode: Positions must have shape [N, 3], got {}", positions.shape().str());
            return Tensor();
        }

        if (positions.dtype() != DataType::Float32) {
            LOG_ERROR("morton_encode: Positions must be Float32");
            return Tensor();
        }

        if (positions.device() != Device::CUDA) {
            LOG_ERROR("morton_encode: Positions must be on CUDA");
            return Tensor();
        }

        const int n_positions = static_cast<int>(positions.size(0));

        // Compute bounding box
        thrust::counting_iterator<int> first(0);
        thrust::counting_iterator<int> last(n_positions);

        position_to_minmax transform_op(positions.ptr<float>());
        float3_minmax init;

        float3_minmax bbox = thrust::transform_reduce(first, last, transform_op, init, minmax_op());

        // Compute per-axis multipliers
        const float xlen = bbox.max_val.x - bbox.min_val.x;
        const float ylen = bbox.max_val.y - bbox.min_val.y;
        const float zlen = bbox.max_val.z - bbox.min_val.z;

        const float xmul = (xlen == 0.0f) ? 0.0f : 1024.0f / xlen;
        const float ymul = (ylen == 0.0f) ? 0.0f : 1024.0f / ylen;
        const float zmul = (zlen == 0.0f) ? 0.0f : 1024.0f / zlen;

        auto morton_codes = Tensor::empty({static_cast<size_t>(n_positions)}, Device::CUDA, DataType::Int64);

        constexpr int BLOCK_SIZE = 256;
        const int grid_size = (n_positions + BLOCK_SIZE - 1) / BLOCK_SIZE;

        morton_encode_kernel<<<grid_size, BLOCK_SIZE>>>(
            positions.ptr<float>(),
            morton_codes.ptr<int64_t>(),
            n_positions,
            bbox.min_val.x, bbox.min_val.y, bbox.min_val.z,
            xmul, ymul, zmul);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            LOG_ERROR("CUDA error in morton_encode_kernel: {}", cudaGetErrorString(err));
            return Tensor();
        }

        cudaDeviceSynchronize();
        return morton_codes;
    }

    Tensor morton_sort_indices(const Tensor& morton_codes) {
        using lfs::core::DataType;
        using lfs::core::Device;

        if (!morton_codes.is_valid()) {
            LOG_ERROR("morton_sort_indices: Invalid input tensor");
            return Tensor();
        }

        if (morton_codes.ndim() != 1) {
            LOG_ERROR("morton_sort_indices: Morton codes must be 1D tensor");
            return Tensor();
        }

        if (morton_codes.dtype() != DataType::Int64) {
            LOG_ERROR("morton_sort_indices: Morton codes must be Int64");
            return Tensor();
        }

        if (morton_codes.device() != Device::CUDA) {
            LOG_ERROR("morton_sort_indices: Morton codes must be on CUDA");
            return Tensor();
        }

        const size_t n = morton_codes.numel();

        auto indices = Tensor::empty({n}, Device::CUDA, DataType::Int64);

        thrust::device_ptr<int64_t> indices_ptr(indices.ptr<int64_t>());
        thrust::sequence(indices_ptr, indices_ptr + n, 0LL);

        auto morton_copy = morton_codes.clone();

        thrust::device_ptr<int64_t> keys_ptr(morton_copy.ptr<int64_t>());
        thrust::device_ptr<int64_t> values_ptr(indices.ptr<int64_t>());

        thrust::sort_by_key(keys_ptr, keys_ptr + n, values_ptr);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            LOG_ERROR("CUDA error in morton_sort_indices: {}", cudaGetErrorString(err));
            return Tensor();
        }

        cudaDeviceSynchronize();
        return indices;
    }

} // namespace lfs::io
