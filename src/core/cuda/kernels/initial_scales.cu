/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/assert.hpp"
#include "core/cuda/initial_scales.hpp"
#include "core/cuda/morton.cuh"
#include "core/cuda_allocation.hpp"
#include "core/cuda_error.hpp"
#include "core/tensor_cuda_interop.hpp"
#include <algorithm>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_select.cuh>
#include <cuda_runtime.h>
#include <limits>
#include <nvtx3/nvToolsExt.h>

namespace lfs::core::cuda {

    namespace {
        using CubWorkspace = CudaCubWorkspace<UniqueCudaAllocation<StreamOrderedCudaAllocator>>;

        constexpr int kThreads = 256;
        constexpr uint32_t kLeafSize = 8;
        constexpr uint32_t kMaxLeaves = uint32_t{1} << 28;
        // Rows on each side of a point in Morton order that bound its neighbours before the tree walk.
        constexpr uint32_t kSeedRadius = 16;
        constexpr float kPercentile = 0.75f;
        // Morton grid bounds; far outliers clamp to its border so the dense region keeps its resolution.
        constexpr float kGridQuantile = 0.001f;

        struct Box {
            float low[3];
            float high[3];
        };

        unsigned blocks_for(const size_t count) {
            return static_cast<unsigned>((count + kThreads - 1) / kThreads);
        }

        // nanoflann's L2_Simple_Adaptor accumulation under FMA contraction. box_distance shares the
        // shape, and rounding is monotone, so a box never reports more than the distance to a point in it.
        __device__ __forceinline__ float squared_norm(const float dx, const float dy, const float dz) {
            return __fmaf_rn(dz, dz, __fmaf_rn(dy, dy, __fmul_rn(dx, dx)));
        }

        __device__ __forceinline__ float point_distance(const float4 p, const float4 q) {
            return squared_norm(__fsub_rn(p.x, q.x), __fsub_rn(p.y, q.y), __fsub_rn(p.z, q.z));
        }

        __device__ __forceinline__ float axis_gap(const float x, const float low, const float high) {
            return fmaxf(fmaxf(__fsub_rn(low, x), __fsub_rn(x, high)), 0.0f);
        }

        __device__ __forceinline__ float box_distance(const float4 p, const Box& b) {
            return squared_norm(axis_gap(p.x, b.low[0], b.high[0]),
                                axis_gap(p.y, b.low[1], b.high[1]),
                                axis_gap(p.z, b.low[2], b.high[2]));
        }

        struct IsFinite {
            __device__ bool operator()(const float x) const { return isfinite(x); }
        };

        __global__ void column_kernel(const float* __restrict__ means, float* __restrict__ column,
                                      const uint32_t n, const int axis) {
            const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n)
                column[i] = means[size_t{i} * 3 + axis];
        }

        __global__ void morton_kernel(const float* __restrict__ means, uint64_t* __restrict__ keys,
                                      uint32_t* __restrict__ rows, const uint32_t n,
                                      const float3 low, const float3 multiplier) {
            const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n)
                return;
            const float* p = means + size_t{i} * 3;
            keys[i] = morton_encode(morton_coordinate(p[0], low.x, multiplier.x),
                                    morton_coordinate(p[1], low.y, multiplier.y),
                                    morton_coordinate(p[2], low.z, multiplier.z));
            rows[i] = i;
        }

        __global__ void gather_kernel(const float* __restrict__ means, const uint32_t* __restrict__ order,
                                      float4* __restrict__ points, const uint32_t n) {
            const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
            if (t >= n)
                return;
            const float* p = means + size_t{order[t]} * 3;
            points[t] = make_float4(p[0], p[1], p[2], 0.0f);
        }

        __global__ void leaf_boxes_kernel(const float4* __restrict__ points, Box* __restrict__ boxes,
                                          const uint32_t n, const uint32_t leaves) {
            const uint32_t leaf = blockIdx.x * blockDim.x + threadIdx.x;
            if (leaf >= leaves)
                return;
            Box box{{INFINITY, INFINITY, INFINITY}, {-INFINITY, -INFINITY, -INFINITY}};
            const uint32_t begin = leaf * kLeafSize;
            const uint32_t end = min(n, begin + kLeafSize);
            for (uint32_t row = begin; row < end; ++row) {
                const float4 p = points[row];
                box.low[0] = fminf(box.low[0], p.x);
                box.low[1] = fminf(box.low[1], p.y);
                box.low[2] = fminf(box.low[2], p.z);
                box.high[0] = fmaxf(box.high[0], p.x);
                box.high[1] = fmaxf(box.high[1], p.y);
                box.high[2] = fmaxf(box.high[2], p.z);
            }
            boxes[leaves + leaf] = box;
        }

        __global__ void parent_boxes_kernel(Box* __restrict__ boxes, const uint32_t first) {
            const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
            if (t >= first)
                return;
            const uint32_t node = first + t;
            Box box = boxes[node * 2];
            const Box right = boxes[node * 2 + 1];
            for (int axis = 0; axis < 3; ++axis) {
                box.low[axis] = fminf(box.low[axis], right.low[axis]);
                box.high[axis] = fmaxf(box.high[axis], right.high[axis]);
            }
            boxes[node] = box;
        }

        // Heap-ordered boxes: node 1 is the root, node k has children 2k and 2k + 1, and node
        // leaves + l bounds Morton rows [l * kLeafSize, (l + 1) * kLeafSize).
        __global__ void log_scale_kernel(const float4* __restrict__ points, const Box* __restrict__ boxes,
                                         const uint32_t* __restrict__ order, const uint32_t n,
                                         const uint32_t leaves, const float max_scale,
                                         float* __restrict__ scaling) {
            const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
            if (t >= n)
                return;
            const float4 p = points[t];
            // Three smallest squared distances, the point itself included, ascending.
            float best[3] = {INFINITY, INFINITY, INFINITY};
            const auto consider = [&](const uint32_t row) {
                const float d = point_distance(p, points[row]);
                if (!(d < best[2]))
                    return;
                if (d < best[1]) {
                    best[2] = best[1];
                    if (d < best[0]) {
                        best[1] = best[0];
                        best[0] = d;
                    } else {
                        best[1] = d;
                    }
                } else {
                    best[2] = d;
                }
            };
            const uint32_t seed_begin = t > kSeedRadius ? t - kSeedRadius : 0;
            const uint32_t seed_end = min(n, t + kSeedRadius + 1);
            for (uint32_t row = seed_begin; row < seed_end; ++row)
                consider(row);

            // Depth-first with at most one deferred sibling per level; leaves sit at depth <= 28.
            uint32_t stack_nodes[32];
            float stack_distances[32];
            int top = 0;
            stack_nodes[top] = 1;
            stack_distances[top++] = 0.0f;
            while (top > 0) {
                --top;
                const uint32_t node = stack_nodes[top];
                if (!(stack_distances[top] < best[2]))
                    continue;
                if (node >= leaves) {
                    const uint32_t begin = (node - leaves) * kLeafSize;
                    const uint32_t end = min(n, begin + kLeafSize);
                    for (uint32_t row = begin; row < end; ++row) {
                        if (row < seed_begin || row >= seed_end)
                            consider(row);
                    }
                    continue;
                }
                const uint32_t left = node * 2;
                const float left_distance = box_distance(p, boxes[left]);
                const float right_distance = box_distance(p, boxes[left + 1]);
                const bool left_nearer = left_distance <= right_distance;
                const float near_distance = left_nearer ? left_distance : right_distance;
                const float far_distance = left_nearer ? right_distance : left_distance;
                if (far_distance < best[2]) {
                    stack_nodes[top] = left_nearer ? left + 1 : left;
                    stack_distances[top++] = far_distance;
                }
                if (near_distance < best[2]) {
                    stack_nodes[top] = left_nearer ? left : left + 1;
                    stack_distances[top++] = near_distance;
                }
            }

            const float dist = __fmul_rn(__fadd_rn(__fsqrt_rn(best[1]), __fsqrt_rn(best[2])), 0.25f);
            const float clamped = dist < 1e-3f ? 1e-3f : (max_scale < dist ? max_scale : dist);
            const float log_scale = static_cast<float>(log(static_cast<double>(clamped)));
            float* out = scaling + size_t{order[t]} * 3;
            out[0] = log_scale;
            out[1] = log_scale;
            out[2] = log_scale;
        }

        struct FiniteAxes {
            bool complete = true;
            float grid_low[3]{};
            float grid_high[3]{};
            float half_extent[3]{};
        };

        // Morton grid bounds and half the central kPercentile range of every finite axis.
        FiniteAxes finite_axes(const float* means, const uint32_t n, const cudaStream_t stream) {
            const size_t count = n;
            auto column = Tensor::empty({count}, Device::CUDA);
            auto finite = Tensor::empty({count}, Device::CUDA);
            auto sorted = Tensor::empty({count}, Device::CUDA);
            auto selected = Tensor::empty({1}, Device::CUDA, DataType::Int32);
            FiniteAxes axes;
            for (int axis = 0; axis < 3; ++axis) {
                column_kernel<<<blocks_for(count), kThreads, 0, stream>>>(means, column.ptr<float>(), n, axis);
                LFS_CUDA_CHECK_MSG(cudaGetLastError(), "initial scale axis extraction");
                run_cub_operation<CubWorkspace>("cub::DeviceSelect::If", stream, [&](void* workspace, size_t& bytes) {
                    return cub::DeviceSelect::If(workspace, bytes, column.ptr<float>(), finite.ptr<float>(),
                                                 selected.ptr<int>(), static_cast<int>(n), IsFinite{}, stream);
                });
                int len = 0;
                LFS_CUDA_CHECK_MSG(cudaMemcpyAsync(&len, selected.ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost, stream),
                                   "initial scale finite count readback");
                LFS_CUDA_CHECK_MSG(cudaStreamSynchronize(stream), "initial scale finite count sync");
                if (len == 0) {
                    axes.complete = false;
                    return axes;
                }
                run_cub_operation<CubWorkspace>("cub::DeviceRadixSort::SortKeys", stream, [&](void* workspace, size_t& bytes) {
                    return cub::DeviceRadixSort::SortKeys(workspace, bytes, finite.ptr<float>(), sorted.ptr<float>(),
                                                          len, 0, 32, stream);
                });
                const auto size = static_cast<size_t>(len);
                const auto quantile = [size](const float fraction) {
                    return std::min(size - 1, static_cast<size_t>(fraction * static_cast<float>(size)));
                };
                const size_t picks[4] = {quantile((1.0f - kPercentile) / 2.0f), quantile((1.0f + kPercentile) / 2.0f),
                                         quantile(kGridQuantile), quantile(1.0f - kGridQuantile)};
                float picked[4];
                for (int k = 0; k < 4; ++k) {
                    LFS_CUDA_CHECK_MSG(cudaMemcpyAsync(&picked[k], sorted.ptr<float>() + picks[k], sizeof(float),
                                                       cudaMemcpyDeviceToHost, stream),
                                       "initial scale quantile readback");
                }
                LFS_CUDA_CHECK_MSG(cudaStreamSynchronize(stream), "initial scale quantile sync");
                axes.half_extent[axis] = (picked[1] - picked[0]) * 0.5f;
                axes.grid_low[axis] = picked[2];
                axes.grid_high[axis] = picked[3];
            }
            return axes;
        }

        // Rows of `means` in Morton order over the grid bounds.
        Tensor morton_order(const float* means, const uint32_t n, const FiniteAxes& axes, const cudaStream_t stream) {
            const size_t count = n;
            auto keys = Tensor::empty({count}, Device::CUDA, DataType::Int64);
            auto sorted_keys = Tensor::empty({count}, Device::CUDA, DataType::Int64);
            auto rows = Tensor::empty({count}, Device::CUDA, DataType::Int32);
            auto order = Tensor::empty({count}, Device::CUDA, DataType::Int32);
            const float3 low = make_float3(axes.grid_low[0], axes.grid_low[1], axes.grid_low[2]);
            const float3 multiplier = make_float3(morton_multiplier(axes.grid_high[0] - axes.grid_low[0]),
                                                  morton_multiplier(axes.grid_high[1] - axes.grid_low[1]),
                                                  morton_multiplier(axes.grid_high[2] - axes.grid_low[2]));
            auto* const key_in = reinterpret_cast<uint64_t*>(keys.ptr<int64_t>());
            auto* const key_out = reinterpret_cast<uint64_t*>(sorted_keys.ptr<int64_t>());
            auto* const row_in = reinterpret_cast<uint32_t*>(rows.ptr<int32_t>());
            auto* const row_out = reinterpret_cast<uint32_t*>(order.ptr<int32_t>());
            morton_kernel<<<blocks_for(count), kThreads, 0, stream>>>(means, key_in, row_in, n, low, multiplier);
            LFS_CUDA_CHECK_MSG(cudaGetLastError(), "initial scale Morton keys");
            run_cub_operation<CubWorkspace>("cub::DeviceRadixSort::SortPairs", stream, [&](void* workspace, size_t& bytes) {
                return cub::DeviceRadixSort::SortPairs(workspace, bytes, key_in, key_out, row_in, row_out,
                                                       static_cast<int>(n), 0, 63, stream);
            });
            return order;
        }

        void knn_log_scales(const float* means, const uint32_t n, const FiniteAxes& axes,
                            float* scaling, const cudaStream_t stream) {
            float extents[3] = {axes.half_extent[0], axes.half_extent[1], axes.half_extent[2]};
            std::sort(extents, extents + 3);
            const float median_size = std::max(extents[1] * 2.0f, 0.01f);
            const float max_scale = median_size * 0.1f;

            const auto order = morton_order(means, n, axes, stream);
            const auto* const rows = reinterpret_cast<const uint32_t*>(order.ptr<int32_t>());
            uint32_t leaves = 1;
            while (leaves < (n + kLeafSize - 1) / kLeafSize)
                leaves *= 2;
            LFS_ASSERT(leaves <= kMaxLeaves);

            const size_t count = n;
            auto points_storage = Tensor::empty({count * 4}, Device::CUDA);
            auto boxes_storage = Tensor::empty({size_t{leaves} * 2 * sizeof(Box) / sizeof(float)}, Device::CUDA);
            auto* const points = reinterpret_cast<float4*>(points_storage.ptr<float>());
            auto* const boxes = reinterpret_cast<Box*>(boxes_storage.ptr<float>());

            gather_kernel<<<blocks_for(count), kThreads, 0, stream>>>(means, rows, points, n);
            LFS_CUDA_CHECK_MSG(cudaGetLastError(), "initial scale gather");
            leaf_boxes_kernel<<<blocks_for(leaves), kThreads, 0, stream>>>(points, boxes, n, leaves);
            LFS_CUDA_CHECK_MSG(cudaGetLastError(), "initial scale leaf boxes");
            for (uint32_t first = leaves / 2; first > 0; first /= 2) {
                parent_boxes_kernel<<<blocks_for(first), kThreads, 0, stream>>>(boxes, first);
                LFS_CUDA_CHECK_MSG(cudaGetLastError(), "initial scale parent boxes");
            }
            log_scale_kernel<<<blocks_for(count), kThreads, 0, stream>>>(points, boxes, rows, n, leaves, max_scale, scaling);
            LFS_CUDA_CHECK_MSG(cudaGetLastError(), "initial scale nearest neighbours");
        }

    } // namespace

    void mrnf_knn_log_scales(const Tensor& means, Tensor& scaling) {
        LFS_ASSERT(means.device() == Device::CUDA && means.dtype() == DataType::Float32);
        LFS_ASSERT(means.ndim() == 2 && means.size(1) == 3 && means.is_contiguous());
        LFS_ASSERT(scaling.device() == Device::CUDA && scaling.dtype() == DataType::Float32);
        LFS_ASSERT(scaling.ndim() == 2 && scaling.size(0) == means.size(0) && scaling.size(1) == 3 && scaling.is_contiguous());
        LFS_ASSERT_MSG(means.size(0) >= 3 && means.size(0) <= static_cast<size_t>(std::numeric_limits<int>::max()),
                       "initial scales need between 3 and INT_MAX points");

        nvtxRangePush("mrnf_knn_log_scales");
        const cudaStream_t stream = scaling.stream();
        const CUDAStreamGuard stream_guard(stream);
        means.sync_to_stream(stream);
        const auto n = static_cast<uint32_t>(means.size(0));
        const auto axes = finite_axes(means.ptr<float>(), n, stream);
        if (axes.complete)
            knn_log_scales(means.ptr<float>(), n, axes, scaling.ptr<float>(), stream);
        else
            scaling.zero_();
        nvtxRangePop();
    }

} // namespace lfs::core::cuda
