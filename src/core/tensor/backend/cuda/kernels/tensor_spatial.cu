/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/cuda_error.hpp"
#include "internal/point_spatial.hpp"
#include "tensor_spatial.hpp"

namespace lfs::core::tensor_ops {
    namespace {
        constexpr int kBlockSize = 256;

        using namespace internal;

        __global__ void build(const float* points, const uint8_t* references, int32_t* heads, int32_t* next,
                              const size_t count, const uint32_t bucket_mask, const float radius) {
            const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= count || !references[i]) {
                return;
            }
            const float* p = points + i * 3;
            if (!finite_point(p)) {
                return;
            }
            const auto bucket = hash_cell(cell(p[0], radius), cell(p[1], radius), cell(p[2], radius), bucket_mask);
            next[i] = atomicExch(heads + bucket, static_cast<int32_t>(i));
        }

        __global__ void query(const float* points, const uint8_t* references, const int32_t* heads,
                              const int32_t* next, bool* output, const size_t count,
                              const uint32_t bucket_mask, const float radius) {
            const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= count) {
                return;
            }
            output[i] = pointHasNeighbor(points, references, heads, next, i, bucket_mask, radius);
        }
    } // namespace

    void launch_radius_neighbors(const float* points, const uint8_t* references, int32_t* heads,
                                 int32_t* next, bool* output, const size_t count, const size_t buckets,
                                 const float radius, const cudaStream_t stream) {
        const auto blocks = static_cast<unsigned int>((count + kBlockSize - 1) / kBlockSize);
        const auto bucket_mask = static_cast<uint32_t>(buckets - 1);
        build<<<blocks, kBlockSize, 0, stream>>>(points, references, heads, next, count, bucket_mask, radius);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.radius_neighbors.build");
        query<<<blocks, kBlockSize, 0, stream>>>(points, references, heads, next, output, count, bucket_mask, radius);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.radius_neighbors.query");
    }
} // namespace lfs::core::tensor_ops
