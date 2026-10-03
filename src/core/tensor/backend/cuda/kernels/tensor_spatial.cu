/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/cuda_error.hpp"
#include "internal/point_spatial.hpp"
#include "tensor_spatial.hpp"

#include <algorithm>

namespace lfs::core::tensor_ops {
    namespace {
        constexpr int kBlockSize = 256;

        using namespace internal;

        __global__ void build(const float* points, const uint8_t* references, int32_t* heads, int32_t* next,
                              const size_t count, const uint32_t bucket_mask, const float radius,
                              const size_t begin) {
            const size_t i = begin + static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
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
                              const uint32_t bucket_mask, const float radius, const bool exclude_self,
                              const size_t begin, const uint8_t* queries) {
            const size_t i = begin + static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= count) {
                return;
            }
            output[i] = (!queries || queries[i]) && pointHasNeighbor(points, references, heads, next, i, bucket_mask, radius, exclude_self);
        }
        __global__ void query_counts(const float* points, const int32_t* heads, const int32_t* next,
                                     int32_t* output, const size_t count, const uint32_t bucket_mask,
                                     const float radius, const int32_t max_count, const size_t begin,
                                     const uint8_t* queries) {
            const size_t i = begin + static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= count)
                return;
            output[i] = (!queries || queries[i]) ? pointNeighborCount(points, heads, next, i, bucket_mask, radius, max_count) : 0;
        }
    } // namespace

    void launch_radius_neighbor_counts(const float* points, const uint8_t* references, int32_t* heads,
                                       int32_t* next, int32_t* output, const size_t count, const size_t buckets,
                                       const float radius, const int32_t max_count, const uint8_t* queries, const cudaStream_t stream) {
        const auto bucket_mask = static_cast<uint32_t>(buckets - 1);
        constexpr size_t batch = 8192;
        for (size_t begin = 0; begin < count; begin += batch) {
            const auto end = std::min(begin + batch, count);
            const auto blocks = static_cast<unsigned int>((end - begin + kBlockSize - 1) / kBlockSize);
            build<<<blocks, kBlockSize, 0, stream>>>(points, references, heads, next, end, bucket_mask, radius, begin);
            LFS_CUDA_LAUNCH_CHECK(stream, "tensor.radius_neighbor_counts.build");
            LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        for (size_t begin = 0; begin < count; begin += batch) {
            const auto end = std::min(begin + batch, count);
            const auto blocks = static_cast<unsigned int>((end - begin + kBlockSize - 1) / kBlockSize);
            query_counts<<<blocks, kBlockSize, 0, stream>>>(points, heads, next, output, end, bucket_mask, radius, max_count, begin, queries);
            LFS_CUDA_LAUNCH_CHECK(stream, "tensor.radius_neighbor_counts.query");
            LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }

    void launch_radius_neighbors(const float* points, const uint8_t* references, int32_t* heads,
                                 int32_t* next, bool* output, const size_t count, const size_t buckets,
                                 const float radius, const bool exclude_self, const uint8_t* queries, const cudaStream_t stream) {
        const auto bucket_mask = static_cast<uint32_t>(buckets - 1);
        const size_t batch = exclude_self ? 8192 : count;
        for (size_t begin = 0; begin < count; begin += batch) {
            const auto end = std::min(begin + batch, count);
            const auto blocks = static_cast<unsigned int>((end - begin + kBlockSize - 1) / kBlockSize);
            build<<<blocks, kBlockSize, 0, stream>>>(points, references, heads, next, end, bucket_mask, radius, begin);
            LFS_CUDA_LAUNCH_CHECK(stream, "tensor.radius_neighbors.build");
            if (exclude_self)
                LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        const size_t query_batch = exclude_self ? 8192 : count;
        for (size_t begin = 0; begin < count; begin += query_batch) {
            const auto end = std::min(begin + query_batch, count);
            const auto query_blocks = static_cast<unsigned int>((end - begin + kBlockSize - 1) / kBlockSize);
            query<<<query_blocks, kBlockSize, 0, stream>>>(points, references, heads, next, output, end, bucket_mask, radius, exclude_self, begin, queries);
            LFS_CUDA_LAUNCH_CHECK(stream, "tensor.radius_neighbors.query");
            if (exclude_self)
                LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }
} // namespace lfs::core::tensor_ops
