/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "morton_encoding.hpp"
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

namespace lfs::io {

    namespace {

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

            morton_codes[idx] = static_cast<int64_t>(morton_encode(
                morton_coordinate(x, min_x, xmul),
                morton_coordinate(y, min_y, ymul),
                morton_coordinate(z, min_z, zmul)));
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

        bool validate_positions(const Tensor& positions, const char* op_name) {
            using lfs::core::DataType;
            using lfs::core::Device;

            if (!positions.is_valid()) {
                LOG_ERROR("{}: Invalid input tensor", op_name);
                return false;
            }

            if (positions.ndim() != 2 || positions.size(1) != 3) {
                LOG_ERROR("{}: Positions must have shape [N, 3], got {}", op_name, positions.shape().str());
                return false;
            }

            if (positions.dtype() != DataType::Float32) {
                LOG_ERROR("{}: Positions must be Float32", op_name);
                return false;
            }

            if (positions.device() != Device::CUDA) {
                LOG_ERROR("{}: Positions must be on CUDA", op_name);
                return false;
            }

            if (positions.size(0) > static_cast<size_t>(std::numeric_limits<int>::max())) {
                LOG_ERROR("{}: Position count exceeds INT_MAX", op_name);
                return false;
            }

            return true;
        }

        struct MortonParams {
            float3_minmax bbox;
            float xmul = 0.0f;
            float ymul = 0.0f;
            float zmul = 0.0f;
        };

        MortonParams compute_morton_params(const Tensor& positions, const int n_positions) {
            thrust::counting_iterator<int> first(0);
            thrust::counting_iterator<int> last(n_positions);

            position_to_minmax transform_op(positions.ptr<float>());
            float3_minmax init;

            MortonParams params;
            params.bbox = thrust::transform_reduce(first, last, transform_op, init, minmax_op());

            const float xlen = params.bbox.max_val.x - params.bbox.min_val.x;
            const float ylen = params.bbox.max_val.y - params.bbox.min_val.y;
            const float zlen = params.bbox.max_val.z - params.bbox.min_val.z;

            params.xmul = morton_multiplier(xlen);
            params.ymul = morton_multiplier(ylen);
            params.zmul = morton_multiplier(zlen);
            return params;
        }

    } // anonymous namespace

    Tensor morton_sort_indices_for_positions(const Tensor& positions, Tensor* sorted_keys) {
        using lfs::core::DataType;
        using lfs::core::Device;

        if (!validate_positions(positions, "morton_sort_indices_for_positions")) {
            return Tensor();
        }

        const int n_positions = static_cast<int>(positions.size(0));
        const MortonParams params = compute_morton_params(positions, n_positions);

        auto morton_codes = Tensor::empty({static_cast<size_t>(n_positions)}, Device::CUDA, DataType::Int64);
        auto indices = Tensor::empty({static_cast<size_t>(n_positions)}, Device::CUDA, DataType::Int32);

        constexpr int BLOCK_SIZE = 256;
        const int grid_size = (n_positions + BLOCK_SIZE - 1) / BLOCK_SIZE;

        morton_encode_kernel<<<grid_size, BLOCK_SIZE>>>(
            positions.ptr<float>(),
            morton_codes.ptr<int64_t>(),
            n_positions,
            params.bbox.min_val.x, params.bbox.min_val.y, params.bbox.min_val.z,
            params.xmul, params.ymul, params.zmul);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            LOG_ERROR("CUDA error in morton_encode_kernel: {}", cudaGetErrorString(err));
            return Tensor();
        }

        thrust::device_ptr<int32_t> indices_ptr(indices.ptr<int32_t>());
        thrust::sequence(indices_ptr, indices_ptr + n_positions, 0);

        thrust::device_ptr<int64_t> keys_ptr(morton_codes.ptr<int64_t>());
        thrust::device_ptr<int32_t> values_ptr(indices.ptr<int32_t>());

        // Stable ordering makes ties deterministic and retains source order,
        // matching the stable JavaScript sort used by splat-transform.
        thrust::stable_sort_by_key(keys_ptr, keys_ptr + n_positions, values_ptr);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return Tensor();
        }

        cudaDeviceSynchronize();
        if (sorted_keys)
            *sorted_keys = std::move(morton_codes);
        return indices;
    }

} // namespace lfs::io
