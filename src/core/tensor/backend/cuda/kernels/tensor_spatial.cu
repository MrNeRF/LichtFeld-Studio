/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/cuda_error.hpp"
#include "internal/nearest_point.hpp"
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
            if (i >= count || (references && !references[i])) {
                return;
            }
            const float* p = points + i * 3;
            if (!finite_point(p)) {
                return;
            }
            const auto bucket = hash_cell(cell(p[0], radius), cell(p[1], radius), cell(p[2], radius), bucket_mask);
            next[i] = atomicExch(heads + bucket, static_cast<int32_t>(i));
        }

        __global__ void nearest_query(const float* queries, const float* targets, const int32_t* heads,
                                      const int32_t* next, int32_t* output, size_t nq, size_t nt, uint32_t mask, float width) {
            const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < nq)
                output[i] = nearestPoint(queries + 3 * i, targets, heads, next, int(nt), mask, width);
        }
        __global__ void coverage_query(const float* points, const float* cameras, int32_t* output, size_t n, size_t count, float maximum) {
            const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < n)
                output[i] = cameraFrustumCount(points + 3 * i, cameras, int(count), maximum);
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
        template <class T>
        __global__ void query_min(const float* points, const T* values, const int32_t* heads,
                                  const int32_t* next, T* output, const size_t count,
                                  const uint32_t bucket_mask, const float radius) {
            const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < count)
                output[i] = pointNeighborMin(points, values, heads, next, i, bucket_mask, radius);
        }
        __global__ void query_spacing(const float* points, const int32_t* heads, const int32_t* next,
                                      float* output, size_t count, uint32_t bucket_mask, float radius) {
            const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < count)
                output[i] = pointNeighborSpacing(points, heads, next, i, bucket_mask, radius);
        }
    } // namespace

    void launch_point_neighbor_spacing(const float* points, const uint8_t* references, int32_t* heads,
                                       int32_t* next, float* output, size_t count, size_t buckets,
                                       float cell_width, cudaStream_t stream) {
        const auto blocks = static_cast<unsigned int>((count + kBlockSize - 1) / kBlockSize);
        const auto mask = static_cast<uint32_t>(buckets - 1);
        build<<<blocks, kBlockSize, 0, stream>>>(points, references, heads, next, count, mask, cell_width, 0);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.point_neighbor_spacing.build");
        query_spacing<<<blocks, kBlockSize, 0, stream>>>(points, heads, next, output, count, mask, cell_width);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.point_neighbor_spacing.query");
    }

    void launch_radius_neighbor_counts(const float* points, const uint8_t* references, int32_t* heads,
                                       int32_t* next, int32_t* output, const size_t count, const size_t buckets,
                                       const float radius, const int32_t max_count, const uint8_t* queries, const cudaStream_t stream) {
        const auto bucket_mask = static_cast<uint32_t>(buckets - 1);
        const auto blocks = static_cast<unsigned int>((count + kBlockSize - 1) / kBlockSize);
        build<<<blocks, kBlockSize, 0, stream>>>(points, references, heads, next, count, bucket_mask, radius, 0);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.radius_neighbor_counts.build");
        query_counts<<<blocks, kBlockSize, 0, stream>>>(points, heads, next, output, count, bucket_mask, radius, max_count, 0, queries);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.radius_neighbor_counts.query");
    }

    void launch_nearest_point_indices(const float* queries, const float* targets, int32_t* heads, int32_t* next, int32_t* output,
                                      size_t nq, size_t nt, size_t buckets, float width, cudaStream_t stream) {
        build<<<(nt + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(targets, nullptr, heads, next, nt, uint32_t(buckets - 1), width, 0);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.nearest_point_indices.build");
        nearest_query<<<(nq + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(queries, targets, heads, next, output, nq, nt, uint32_t(buckets - 1), width);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.nearest_point_indices.query");
    }
    void launch_camera_frustum_counts(const float* points, const float* cameras, int32_t* output, size_t n, size_t count, float maximum, cudaStream_t stream) {
        coverage_query<<<(n + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(points, cameras, output, n, count, maximum);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.camera_frustum_counts");
    }

    void launch_radius_neighbor_min(const float* points, const void* values, const uint8_t value_is_float,
                                    const uint8_t* references, int32_t* heads, int32_t* next, void* output,
                                    const size_t count, const size_t buckets, const float radius,
                                    const cudaStream_t stream) {
        const auto bucket_mask = static_cast<uint32_t>(buckets - 1);
        const auto blocks = static_cast<unsigned int>((count + kBlockSize - 1) / kBlockSize);
        build<<<blocks, kBlockSize, 0, stream>>>(points, references, heads, next, count, bucket_mask, radius, 0);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.radius_neighbor_min.build");
        if (value_is_float) {
            query_min<<<blocks, kBlockSize, 0, stream>>>(points, static_cast<const float*>(values), heads, next,
                                                         static_cast<float*>(output), count, bucket_mask, radius);
            LFS_CUDA_LAUNCH_CHECK(stream, "tensor.radius_neighbor_min.query_float");
        } else {
            query_min<<<blocks, kBlockSize, 0, stream>>>(points, static_cast<const int32_t*>(values), heads, next,
                                                         static_cast<int32_t*>(output), count, bucket_mask, radius);
            LFS_CUDA_LAUNCH_CHECK(stream, "tensor.radius_neighbor_min.query_int");
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
