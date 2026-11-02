/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/cuda/splat_transform.hpp"
#include "core/cuda_error.hpp"
#include <algorithm>
#include <cstring>

namespace lfs::core::cuda {
    namespace {
        __global__ void transform_geometry(splat_transform::LinearTransform matrix,
                                           const float* scales, const float* rotations,
                                           float* out_scales, float* out_rotations,
                                           std::size_t count) {
            const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < count)
                splat_transform::affine_geometry(matrix, scales + 3 * i, rotations + 4 * i,
                                                 out_scales + 3 * i, out_rotations + 4 * i);
        }

        // Orders float bits as unsigned so atomicMin/atomicMax reduce floats.
        __device__ unsigned order_key(const float value) {
            const unsigned bits = __float_as_uint(value);
            return (bits & 0x80000000u) ? ~bits : bits | 0x80000000u;
        }

        float key_value(const unsigned key) {
            const unsigned bits = (key & 0x80000000u) ? key & 0x7fffffffu : ~key;
            float value;
            std::memcpy(&value, &bits, sizeof(value));
            return value;
        }

        __global__ void position_bounds(const float* means, const unsigned char* deleted,
                                        std::size_t count, unsigned* keys) {
            unsigned lo[3] = {~0u, ~0u, ~0u};
            unsigned hi[3] = {0u, 0u, 0u};
            for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
                 i += std::size_t(gridDim.x) * blockDim.x) {
                const float* p = means + 3 * i;
                if ((deleted && deleted[i]) || !isfinite(p[0]) || !isfinite(p[1]) || !isfinite(p[2]))
                    continue;
                for (int axis = 0; axis < 3; ++axis) {
                    const unsigned key = order_key(p[axis]);
                    lo[axis] = min(lo[axis], key);
                    hi[axis] = max(hi[axis], key);
                }
            }
            for (int axis = 0; axis < 3; ++axis) {
                for (int offset = 16; offset > 0; offset >>= 1) {
                    lo[axis] = min(lo[axis], __shfl_xor_sync(0xffffffffu, lo[axis], offset));
                    hi[axis] = max(hi[axis], __shfl_xor_sync(0xffffffffu, hi[axis], offset));
                }
            }
            if ((threadIdx.x & 31) == 0) {
                for (int axis = 0; axis < 3; ++axis) {
                    atomicMin(keys + axis, lo[axis]);
                    atomicMax(keys + 3 + axis, hi[axis]);
                }
            }
        }
    } // namespace
    void transform_splat_geometry(const splat_transform::LinearTransform& matrix,
                                  const float* scales, const float* rotations,
                                  float* out_scales, float* out_rotations,
                                  std::size_t count, cudaStream_t stream) {
        if (!count)
            return;
        transform_geometry<<<(count + 255) / 256, 256, 0, stream>>>(
            matrix, scales, rotations, out_scales, out_rotations, count);
        LFS_CUDA_CHECK(cudaGetLastError());
    }

    bool splat_position_bounds(const float* means, const unsigned char* deleted, std::size_t count,
                               unsigned* scratch, float out_min[3], float out_max[3], cudaStream_t stream) {
        if (!count)
            return false;
        LFS_CUDA_CHECK(cudaMemsetAsync(scratch, 0xff, 3 * sizeof(unsigned), stream));
        LFS_CUDA_CHECK(cudaMemsetAsync(scratch + 3, 0, 3 * sizeof(unsigned), stream));
        const auto blocks = static_cast<unsigned>(std::min<std::size_t>((count + 255) / 256, 1024));
        position_bounds<<<blocks, 256, 0, stream>>>(means, deleted, count, scratch);
        LFS_CUDA_CHECK(cudaGetLastError());
        unsigned keys[6];
        LFS_CUDA_CHECK(cudaMemcpyAsync(keys, scratch, sizeof(keys), cudaMemcpyDeviceToHost, stream));
        LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
        if (keys[0] > keys[3])
            return false;
        for (int axis = 0; axis < 3; ++axis) {
            out_min[axis] = key_value(keys[axis]);
            out_max[axis] = key_value(keys[3 + axis]);
        }
        return true;
    }
} // namespace lfs::core::cuda
