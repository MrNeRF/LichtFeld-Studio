/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/sh_layout.cuh"
#include "core/cuda_error.hpp"
#include "core/logger.hpp"
#include "kmeans.hpp"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <numeric>
#include <optional>
#include <random>
#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <unordered_map>
#include <vector>

namespace lfs::io {

    using lfs::core::DataType;
    using lfs::core::Device;

    namespace {

        constexpr int BLOCK_SIZE = 256;
        constexpr int ASSIGN_POINT_TILE = 128;
        constexpr int ASSIGN_CENTROID_TILE = 32;
        constexpr int ASSIGN_POINTS_PER_THREAD = 2;
        constexpr int ASSIGN_CENTROIDS_PER_THREAD = 8;
        constexpr int ASSIGN_CENTROID_GROUPS = 4;
        // The extra four floats make rows advance through banks instead of
        // mapping the same dimension of every row to one bank.
        constexpr int ASSIGN_SHARED_K_STRIDE = 49;

        struct CudaTimingState {
            bool enabled = false;
        };

        class CudaEventTimer {
        public:
            explicit CudaEventTimer(CudaTimingState& state)
                : state_(&state) {
                if (!state_->enabled) {
                    return;
                }

                const auto start_status = cudaEventCreate(&start_);
                if (start_status != cudaSuccess) {
                    disable("cudaEventCreate(start)", start_status);
                    return;
                }

                const auto stop_status = cudaEventCreate(&stop_);
                if (stop_status != cudaSuccess) {
                    disable("cudaEventCreate(stop)", stop_status);
                }
            }

            ~CudaEventTimer() {
                destroy(start_, "cudaEventDestroy(start)");
                destroy(stop_, "cudaEventDestroy(stop)");
            }

            CudaEventTimer(const CudaEventTimer&) = delete;
            CudaEventTimer& operator=(const CudaEventTimer&) = delete;

            CudaEventTimer(CudaEventTimer&& other) noexcept
                : state_(other.state_), start_(other.start_), stop_(other.stop_) {
                other.state_ = nullptr;
                other.start_ = nullptr;
                other.stop_ = nullptr;
            }

            void record_start() {
                record(start_, "cudaEventRecord(start)");
            }

            void record_stop() {
                record(stop_, "cudaEventRecord(stop)");
            }

            bool elapsed_ms(float& milliseconds) {
                if (!active()) {
                    return false;
                }
                const auto status = cudaEventElapsedTime(&milliseconds, start_, stop_);
                if (status != cudaSuccess) {
                    disable("cudaEventElapsedTime", status);
                    return false;
                }
                return true;
            }

        private:
            bool active() const {
                return state_ && state_->enabled && start_ && stop_;
            }

            void record(cudaEvent_t event, const char* operation) {
                if (!active()) {
                    return;
                }
                const auto status = cudaEventRecord(event, nullptr);
                if (status != cudaSuccess) {
                    disable(operation, status);
                }
            }

            void disable(const char* operation, const cudaError_t status) {
                LOG_DEBUG("SOG k-means timing disabled after {} failed: {}",
                          operation, cudaGetErrorString(status));
                if (state_) {
                    state_->enabled = false;
                }
                destroy(start_, "cudaEventDestroy(start)");
                destroy(stop_, "cudaEventDestroy(stop)");
            }

            static void destroy(cudaEvent_t& event, const char* operation) noexcept {
                if (!event) {
                    return;
                }
                const auto status = cudaEventDestroy(event);
                event = nullptr;
                if (status != cudaSuccess) {
                    LOG_DEBUG("SOG k-means timing cleanup {} failed: {}",
                              operation, cudaGetErrorString(status));
                }
            }

            CudaTimingState* state_ = nullptr;
            cudaEvent_t start_ = nullptr;
            cudaEvent_t stop_ = nullptr;
        };

        // Tensor::cuda() deep-copies CUDA tensors in this codebase; borrow when possible.
        Tensor as_cuda_contiguous(const Tensor& data) {
            if (data.device() == Device::GPU) {
                return data.is_contiguous() ? data : data.contiguous();
            }
            return data.cuda().contiguous();
        }

        __device__ __forceinline__ std::uint32_t shAt_device(
            std::uint32_t p,
            std::uint32_t k,
            std::uint32_t slots_per_primitive) {
            const std::uint32_t block = p / lfs::core::kShReorderSize;
            const std::uint32_t lane = p % lfs::core::kShReorderSize;
            return block * (slots_per_primitive * lfs::core::kShReorderSize) +
                   k * lfs::core::kShReorderSize + lane;
        }

        __device__ __forceinline__ float float4_component(const float4 v, const std::uint32_t component) {
            switch (component) {
            case 0:
                return v.x;
            case 1:
                return v.y;
            case 2:
                return v.z;
            default:
                return v.w;
            }
        }

        template <int N_DIMS>
        __device__ __forceinline__ float read_swizzled_sh_dim(
            const float4* __restrict__ shN,
            const std::uint32_t primitive_idx,
            const std::uint32_t dim) {
            constexpr std::uint32_t slots_per_primitive = (N_DIMS + 3u) / 4u;
            const std::uint32_t slot = dim / 4u;
            const std::uint32_t component = dim % 4u;
            return float4_component(shN[shAt_device(primitive_idx, slot, slots_per_primitive)], component);
        }

        template <int N_DIMS>
        __device__ __forceinline__ float read_swizzled_sh_slot(
            const float4* __restrict__ point_slots,
            const std::uint32_t dim) {
            return float4_component(point_slots[dim / 4u], dim % 4u);
        }

        template <int N_DIMS>
        __global__ void gather_centroids_kernel(
            const float* __restrict__ data,
            const int* __restrict__ indices,
            float* __restrict__ centroids,
            const int k) {
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid >= k)
                return;

            const int src_idx = indices[tid];
#pragma unroll
            for (int d = 0; d < N_DIMS; ++d) {
                centroids[tid * N_DIMS + d] = data[src_idx * N_DIMS + d];
            }
        }

        template <int N_DIMS>
        __global__ void gather_swizzled_centroids_kernel(
            const float4* __restrict__ shN,
            const int* __restrict__ indices,
            float* __restrict__ centroids,
            const int k) {
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid >= k)
                return;

            const std::uint32_t src_idx = static_cast<std::uint32_t>(indices[tid]);
#pragma unroll
            for (int d = 0; d < N_DIMS; ++d) {
                centroids[tid * N_DIMS + d] = read_swizzled_sh_dim<N_DIMS>(shN, src_idx, static_cast<std::uint32_t>(d));
            }
        }

        std::vector<int> sample_unique_indices(const int n, const int count, const unsigned int seed) {
            std::vector<int> indices(static_cast<size_t>(count));
            std::unordered_map<int, int> swaps;
            swaps.reserve(static_cast<size_t>(count) * 2);

            auto value_for = [&swaps](const int index) {
                auto it = swaps.find(index);
                return it == swaps.end() ? index : it->second;
            };

            std::mt19937 rng(seed);
            for (int i = 0; i < count; ++i) {
                std::uniform_int_distribution<int> dist(i, n - 1);
                const int pick = dist(rng);
                indices[static_cast<size_t>(i)] = value_for(pick);
                swaps[pick] = value_for(i);
            }

            return indices;
        }

        Tensor sample_unique_indices_gpu(const int n, const int count, const unsigned int seed) {
            auto indices = sample_unique_indices(n, count, seed);
            return Tensor::from_vector(indices, {static_cast<size_t>(count)}, Device::GPU);
        }

        void build_csr_offsets_gpu(
            const int* d_membership,
            int* d_offsets,
            int* d_indices,
            const int k,
            const int num_clusters) {
            auto cluster_keys = Tensor::zeros({static_cast<size_t>(k)}, Device::GPU, DataType::Int32);
            auto sorted_indices = Tensor::zeros({static_cast<size_t>(k)}, Device::GPU, DataType::Int32);

            LFS_CUDA_CHECK(cudaMemcpy(cluster_keys.ptr<int>(), d_membership,
                                      k * sizeof(int), cudaMemcpyDeviceToDevice));
            thrust::device_ptr<int> indices_ptr(sorted_indices.ptr<int>());
            thrust::sequence(indices_ptr, indices_ptr + k);

            thrust::device_ptr<int> keys_ptr(cluster_keys.ptr<int>());
            thrust::sort_by_key(keys_ptr, keys_ptr + k, indices_ptr);

            LFS_CUDA_CHECK(cudaMemcpy(d_indices, sorted_indices.ptr<int>(),
                                      k * sizeof(int), cudaMemcpyDeviceToDevice));

            thrust::device_ptr<int> offsets_ptr(d_offsets);
            thrust::lower_bound(
                keys_ptr, keys_ptr + k,
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(num_clusters + 1),
                offsets_ptr);
        }

        template <int N_DIMS>
        __global__ void compute_centroid_norms_kernel(
            const float* __restrict__ centroids,
            float* __restrict__ norms,
            const int k) {
            const int cid = blockIdx.x * blockDim.x + threadIdx.x;
            if (cid >= k) {
                return;
            }

            float norm = 0.0f;
#pragma unroll
            for (int d = 0; d < N_DIMS; ++d) {
                const float value = centroids[static_cast<long long>(cid) * N_DIMS + d];
                norm = fmaf(value, value, norm);
            }
            norms[cid] = norm;
        }

        // Tiled assignment compares squared distances as ||c||^2 - 2*dot(x, c):
        // ||x||^2 is common to every centroid for one point and can be omitted.
        // Equal distances choose the lowest centroid index. In the grouped
        // hierarchy, each point group searches the members of the four nearest
        // supers to its assigned super; the final assignment searches all k
        // centroids exactly.
        template <int N_DIMS>
        __global__ void __launch_bounds__(BLOCK_SIZE, 3)
            assign_nearest_bruteforce_kernel(
                const float* __restrict__ data,
                const float* __restrict__ centroids,
                const float* __restrict__ centroid_norms,
                int* __restrict__ labels,
                const int n_points,
                const int k) {
            __shared__ float shared_points[ASSIGN_POINT_TILE][ASSIGN_SHARED_K_STRIDE];
            __shared__ float shared_centroids[ASSIGN_CENTROID_TILE][ASSIGN_SHARED_K_STRIDE];
            __shared__ float shared_centroid_norms[ASSIGN_CENTROID_TILE];

            const int tid = threadIdx.x;
            const int point_row = (tid / ASSIGN_CENTROID_GROUPS) * ASSIGN_POINTS_PER_THREAD;
            const int centroid_col = (tid % ASSIGN_CENTROID_GROUPS) * ASSIGN_CENTROIDS_PER_THREAD;
            const int point_start = blockIdx.x * ASSIGN_POINT_TILE;

            for (int i = tid; i < ASSIGN_POINT_TILE * N_DIMS; i += BLOCK_SIZE) {
                const int point = i / N_DIMS;
                const int dim = i % N_DIMS;
                const int global_point = point_start + point;
                shared_points[point][dim] = global_point < n_points
                                                ? data[static_cast<long long>(global_point) * N_DIMS + dim]
                                                : 0.0f;
            }
            __syncthreads();

            float best_dist[ASSIGN_POINTS_PER_THREAD];
            int best_idx[ASSIGN_POINTS_PER_THREAD];
#pragma unroll
            for (int p = 0; p < ASSIGN_POINTS_PER_THREAD; ++p) {
                best_dist[p] = 1e30f;
                best_idx[p] = k;
            }

            const int num_tiles = (k + ASSIGN_CENTROID_TILE - 1) / ASSIGN_CENTROID_TILE;
            for (int tile = 0; tile < num_tiles; ++tile) {
                const int centroid_start = tile * ASSIGN_CENTROID_TILE;
                for (int i = tid; i < ASSIGN_CENTROID_TILE * N_DIMS; i += BLOCK_SIZE) {
                    const int centroid = i / N_DIMS;
                    const int dim = i % N_DIMS;
                    const int global_centroid = centroid_start + centroid;
                    shared_centroids[centroid][dim] = global_centroid < k
                                                          ? centroids[static_cast<long long>(global_centroid) * N_DIMS + dim]
                                                          : 0.0f;
                }
                for (int i = tid; i < ASSIGN_CENTROID_TILE; i += BLOCK_SIZE) {
                    const int global_centroid = centroid_start + i;
                    shared_centroid_norms[i] = global_centroid < k ? centroid_norms[global_centroid] : 0.0f;
                }
                __syncthreads();

                float dots[ASSIGN_POINTS_PER_THREAD][ASSIGN_CENTROIDS_PER_THREAD] = {};
#pragma unroll
                for (int d = 0; d < N_DIMS; ++d) {
                    float point_values[ASSIGN_POINTS_PER_THREAD];
                    float centroid_values[ASSIGN_CENTROIDS_PER_THREAD];
#pragma unroll
                    for (int p = 0; p < ASSIGN_POINTS_PER_THREAD; ++p) {
                        point_values[p] = shared_points[point_row + p][d];
                    }
#pragma unroll
                    for (int c = 0; c < ASSIGN_CENTROIDS_PER_THREAD; ++c) {
                        centroid_values[c] = shared_centroids[centroid_col + c][d];
                    }
#pragma unroll
                    for (int p = 0; p < ASSIGN_POINTS_PER_THREAD; ++p) {
#pragma unroll
                        for (int c = 0; c < ASSIGN_CENTROIDS_PER_THREAD; ++c) {
                            dots[p][c] = fmaf(point_values[p], centroid_values[c], dots[p][c]);
                        }
                    }
                }

#pragma unroll
                for (int p = 0; p < ASSIGN_POINTS_PER_THREAD; ++p) {
#pragma unroll
                    for (int c = 0; c < ASSIGN_CENTROIDS_PER_THREAD; ++c) {
                        const int global_centroid = centroid_start + centroid_col + c;
                        if (global_centroid < k) {
                            const float dist = fmaf(-2.0f, dots[p][c],
                                                    shared_centroid_norms[centroid_col + c]);
                            if (dist < best_dist[p] ||
                                (dist == best_dist[p] && global_centroid < best_idx[p])) {
                                best_dist[p] = dist;
                                best_idx[p] = global_centroid;
                            }
                        }
                    }
                }
                __syncthreads();
            }

            const unsigned group_mask = 0xffffffffu;
            const int group_lane = tid % ASSIGN_CENTROID_GROUPS;
#pragma unroll
            for (int p = 0; p < ASSIGN_POINTS_PER_THREAD; ++p) {
                for (int offset = ASSIGN_CENTROID_GROUPS / 2; offset > 0; offset >>= 1) {
                    const float other_dist = __shfl_down_sync(
                        group_mask, best_dist[p], offset, ASSIGN_CENTROID_GROUPS);
                    const int other_idx = __shfl_down_sync(
                        group_mask, best_idx[p], offset, ASSIGN_CENTROID_GROUPS);
                    if (other_dist < best_dist[p] ||
                        (other_dist == best_dist[p] && other_idx < best_idx[p])) {
                        best_dist[p] = other_dist;
                        best_idx[p] = other_idx;
                    }
                }
                if (group_lane == 0) {
                    const int global_point = point_start + point_row + p;
                    if (global_point < n_points) {
                        labels[global_point] = best_idx[p];
                    }
                }
            }
        }

        template <int N_DIMS>
        __global__ void __launch_bounds__(BLOCK_SIZE, 3)
            assign_nearest_swizzled_bruteforce_kernel(
                const float4* __restrict__ shN,
                const float* __restrict__ centroids,
                const float* __restrict__ centroid_norms,
                int* __restrict__ labels,
                const int n_points,
                const int k) {
            __shared__ float shared_points[ASSIGN_POINT_TILE][ASSIGN_SHARED_K_STRIDE];
            __shared__ float shared_centroids[ASSIGN_CENTROID_TILE][ASSIGN_SHARED_K_STRIDE];
            __shared__ float shared_centroid_norms[ASSIGN_CENTROID_TILE];

            const int tid = threadIdx.x;
            const int point_row = (tid / ASSIGN_CENTROID_GROUPS) * ASSIGN_POINTS_PER_THREAD;
            const int centroid_col = (tid % ASSIGN_CENTROID_GROUPS) * ASSIGN_CENTROIDS_PER_THREAD;
            const int point_start = blockIdx.x * ASSIGN_POINT_TILE;
            constexpr int point_slots_count = (N_DIMS + 3) / 4;
            for (int i = tid; i < ASSIGN_POINT_TILE * point_slots_count; i += BLOCK_SIZE) {
                const int point = i / point_slots_count;
                const int slot = i % point_slots_count;
                const int global_point = point_start + point;
                const float4 values = global_point < n_points
                                          ? shN[shAt_device(static_cast<std::uint32_t>(global_point),
                                                            static_cast<std::uint32_t>(slot), point_slots_count)]
                                          : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                shared_points[point][slot * 4 + 0] = values.x;
                shared_points[point][slot * 4 + 1] = values.y;
                shared_points[point][slot * 4 + 2] = values.z;
                shared_points[point][slot * 4 + 3] = values.w;
            }
            __syncthreads();

            float best_dist[ASSIGN_POINTS_PER_THREAD];
            int best_idx[ASSIGN_POINTS_PER_THREAD];
#pragma unroll
            for (int p = 0; p < ASSIGN_POINTS_PER_THREAD; ++p) {
                best_dist[p] = 1e30f;
                best_idx[p] = k;
            }

            const int num_tiles = (k + ASSIGN_CENTROID_TILE - 1) / ASSIGN_CENTROID_TILE;
            for (int tile = 0; tile < num_tiles; ++tile) {
                const int centroid_start = tile * ASSIGN_CENTROID_TILE;
                for (int i = tid; i < ASSIGN_CENTROID_TILE * N_DIMS; i += BLOCK_SIZE) {
                    const int centroid = i / N_DIMS;
                    const int dim = i % N_DIMS;
                    const int global_centroid = centroid_start + centroid;
                    shared_centroids[centroid][dim] = global_centroid < k
                                                          ? centroids[static_cast<long long>(global_centroid) * N_DIMS + dim]
                                                          : 0.0f;
                }
                for (int i = tid; i < ASSIGN_CENTROID_TILE; i += BLOCK_SIZE) {
                    const int global_centroid = centroid_start + i;
                    shared_centroid_norms[i] = global_centroid < k ? centroid_norms[global_centroid] : 0.0f;
                }
                __syncthreads();

                float dots[ASSIGN_POINTS_PER_THREAD][ASSIGN_CENTROIDS_PER_THREAD] = {};
#pragma unroll
                for (int d = 0; d < N_DIMS; ++d) {
                    float point_values[ASSIGN_POINTS_PER_THREAD];
                    float centroid_values[ASSIGN_CENTROIDS_PER_THREAD];
#pragma unroll
                    for (int p = 0; p < ASSIGN_POINTS_PER_THREAD; ++p) {
                        point_values[p] = shared_points[point_row + p][d];
                    }
#pragma unroll
                    for (int c = 0; c < ASSIGN_CENTROIDS_PER_THREAD; ++c) {
                        centroid_values[c] = shared_centroids[centroid_col + c][d];
                    }
#pragma unroll
                    for (int p = 0; p < ASSIGN_POINTS_PER_THREAD; ++p) {
#pragma unroll
                        for (int c = 0; c < ASSIGN_CENTROIDS_PER_THREAD; ++c) {
                            dots[p][c] = fmaf(point_values[p], centroid_values[c], dots[p][c]);
                        }
                    }
                }

#pragma unroll
                for (int p = 0; p < ASSIGN_POINTS_PER_THREAD; ++p) {
#pragma unroll
                    for (int c = 0; c < ASSIGN_CENTROIDS_PER_THREAD; ++c) {
                        const int global_centroid = centroid_start + centroid_col + c;
                        if (global_centroid < k) {
                            const float dist = fmaf(-2.0f, dots[p][c],
                                                    shared_centroid_norms[centroid_col + c]);
                            if (dist < best_dist[p] ||
                                (dist == best_dist[p] && global_centroid < best_idx[p])) {
                                best_dist[p] = dist;
                                best_idx[p] = global_centroid;
                            }
                        }
                    }
                }
                __syncthreads();
            }

            const unsigned group_mask = 0xffffffffu;
            const int group_lane = tid % ASSIGN_CENTROID_GROUPS;
#pragma unroll
            for (int p = 0; p < ASSIGN_POINTS_PER_THREAD; ++p) {
                for (int offset = ASSIGN_CENTROID_GROUPS / 2; offset > 0; offset >>= 1) {
                    const float other_dist = __shfl_down_sync(
                        group_mask, best_dist[p], offset, ASSIGN_CENTROID_GROUPS);
                    const int other_idx = __shfl_down_sync(
                        group_mask, best_idx[p], offset, ASSIGN_CENTROID_GROUPS);
                    if (other_dist < best_dist[p] ||
                        (other_dist == best_dist[p] && other_idx < best_idx[p])) {
                        best_dist[p] = other_dist;
                        best_idx[p] = other_idx;
                    }
                }
                if (group_lane == 0) {
                    const int global_point = point_start + point_row + p;
                    if (global_point < n_points) {
                        labels[global_point] = best_idx[p];
                    }
                }
            }
        }

        template <int N_DIMS>
        __global__ void accumulate_centroids_kernel(
            const float* __restrict__ data,
            const int* __restrict__ labels,
            float* __restrict__ centroid_sums,
            int* __restrict__ counts,
            const int n_points) {
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid >= n_points)
                return;

            const int label = labels[tid];

            for (int d = 0; d < N_DIMS; ++d) {
                atomicAdd(&centroid_sums[label * N_DIMS + d], data[tid * N_DIMS + d]);
            }
            atomicAdd(&counts[label], 1);
        }

        template <int N_DIMS>
        __global__ void accumulate_swizzled_centroids_kernel(
            const float4* __restrict__ shN,
            const int* __restrict__ labels,
            float* __restrict__ centroid_sums,
            int* __restrict__ counts,
            const int n_points) {
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid >= n_points)
                return;

            const int label = labels[tid];
            constexpr int point_slots_count = (N_DIMS + 3) / 4;
            float4 point_slots[point_slots_count];
#pragma unroll
            for (int slot = 0; slot < point_slots_count; ++slot) {
                point_slots[slot] = shN[shAt_device(
                    static_cast<std::uint32_t>(tid), static_cast<std::uint32_t>(slot), point_slots_count)];
            }
#pragma unroll
            for (int d = 0; d < N_DIMS; ++d) {
                atomicAdd(&centroid_sums[label * N_DIMS + d],
                          read_swizzled_sh_slot<N_DIMS>(point_slots, static_cast<std::uint32_t>(d)));
            }
            atomicAdd(&counts[label], 1);
        }

        template <int N_DIMS>
        __global__ void finalize_centroids_kernel(
            float* __restrict__ centroids,
            const float* __restrict__ centroid_sums,
            const int* __restrict__ counts,
            const float* __restrict__ data,
            const int k,
            const int n_points,
            unsigned int seed) {
            const int cid = blockIdx.x * blockDim.x + threadIdx.x;
            if (cid >= k)
                return;

            const int count = counts[cid];

            if (count > 0) {
                for (int d = 0; d < N_DIMS; ++d) {
                    centroids[cid * N_DIMS + d] = centroid_sums[cid * N_DIMS + d] / count;
                }
            } else {
                unsigned int rng = seed ^ (cid * 1664525u + 1013904223u);
                rng = rng * 1664525u + 1013904223u;
                int rand_idx = rng % n_points;
                for (int d = 0; d < N_DIMS; ++d) {
                    centroids[cid * N_DIMS + d] = data[rand_idx * N_DIMS + d];
                }
            }
        }

        template <int N_DIMS>
        __global__ void finalize_swizzled_centroids_kernel(
            float* __restrict__ centroids,
            const float* __restrict__ centroid_sums,
            const int* __restrict__ counts,
            const float4* __restrict__ shN,
            const int k,
            const int n_points,
            unsigned int seed) {
            const int cid = blockIdx.x * blockDim.x + threadIdx.x;
            if (cid >= k)
                return;

            const int count = counts[cid];
            if (count > 0) {
#pragma unroll
                for (int d = 0; d < N_DIMS; ++d) {
                    centroids[cid * N_DIMS + d] = centroid_sums[cid * N_DIMS + d] / count;
                }
            } else {
                unsigned int rng = seed ^ (cid * 1664525u + 1013904223u);
                rng = rng * 1664525u + 1013904223u;
                const std::uint32_t rand_idx = static_cast<std::uint32_t>(rng % n_points);
#pragma unroll
                for (int d = 0; d < N_DIMS; ++d) {
                    centroids[cid * N_DIMS + d] =
                        read_swizzled_sh_dim<N_DIMS>(shN, rand_idx, static_cast<std::uint32_t>(d));
                }
            }
        }

        template <int N_DIMS>
        std::tuple<Tensor, Tensor> kmeans_swizzled_bruteforce_impl(
            const Tensor& shN_swizzled,
            const int n,
            const int k,
            const int iterations) {
            auto shN_gpu = as_cuda_contiguous(shN_swizzled);
            const auto* d_shN = reinterpret_cast<const float4*>(shN_gpu.ptr<float>());

            if (n <= k) {
                auto centroids = Tensor::zeros({static_cast<size_t>(n), static_cast<size_t>(N_DIMS)},
                                               Device::GPU, DataType::Float32);
                auto labels = Tensor::arange(n).to(DataType::Int32).cuda();
                const int grid_n = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
                const auto kmeans_ticket = ::lfs::core::cuda_record_range(
                    /*stream=*/nullptr, "io.kmeans.swizzled_gather_points");
                gather_swizzled_centroids_kernel<N_DIMS><<<grid_n, BLOCK_SIZE>>>(
                    d_shN, labels.ptr<int>(), centroids.ptr<float>(), n);
                LFS_CUDA_LAUNCH_CHECK(nullptr, "io.kmeans.gather_swizzled_centroids");
                LFS_CUDA_AWAIT(kmeans_ticket, cudaDeviceSynchronize(), "io.kmeans.swizzled_gather_points_sync");
                return {centroids, labels};
            }

            auto centroids = Tensor::zeros({static_cast<size_t>(k), static_cast<size_t>(N_DIMS)},
                                           Device::GPU, DataType::Float32);
            float* d_centroids = centroids.ptr<float>();

            {
                std::random_device rd;
                auto perm = sample_unique_indices_gpu(n, k, rd());

                const int grid_k = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
                gather_swizzled_centroids_kernel<N_DIMS><<<grid_k, BLOCK_SIZE>>>(
                    d_shN, perm.ptr<int>(), d_centroids, k);
                LFS_CUDA_LAUNCH_CHECK(nullptr, "io.kmeans.gather_swizzled_centroids");
            }

            auto labels = Tensor::zeros({static_cast<size_t>(n)}, Device::GPU, DataType::Int32);
            auto centroid_sums = Tensor::zeros({static_cast<size_t>(k), static_cast<size_t>(N_DIMS)},
                                               Device::GPU, DataType::Float32);
            auto counts = Tensor::zeros({static_cast<size_t>(k)}, Device::GPU, DataType::Int32);
            auto centroid_norms = Tensor::zeros({static_cast<size_t>(k)}, Device::GPU, DataType::Float32);

            const int grid_n = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const int grid_n_assign = (n + ASSIGN_POINT_TILE - 1) / ASSIGN_POINT_TILE;
            const int grid_k = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;

            const auto kmeans_ticket = ::lfs::core::cuda_record_range(
                /*stream=*/nullptr, "io.kmeans.swizzled_bruteforce_iteration");
            for (int iter = 0; iter < iterations; ++iter) {
                compute_centroid_norms_kernel<N_DIMS><<<grid_k, BLOCK_SIZE>>>(
                    d_centroids, centroid_norms.ptr<float>(), k);
                LFS_CUDA_LAUNCH_CHECK(nullptr, "io.kmeans.compute_centroid_norms");
                assign_nearest_swizzled_bruteforce_kernel<N_DIMS><<<grid_n_assign, BLOCK_SIZE>>>(
                    d_shN, d_centroids, centroid_norms.ptr<float>(), labels.ptr<int>(), n, k);
                LFS_CUDA_LAUNCH_CHECK(nullptr, "io.kmeans.assign_nearest_swizzled");

                centroid_sums.zero_();
                counts.zero_();

                accumulate_swizzled_centroids_kernel<N_DIMS><<<grid_n, BLOCK_SIZE>>>(
                    d_shN, labels.ptr<int>(), centroid_sums.ptr<float>(), counts.ptr<int>(), n);
                LFS_CUDA_LAUNCH_CHECK(nullptr, "io.kmeans.accumulate_swizzled_centroids");

                const unsigned int seed = static_cast<unsigned int>(iter * 12345 + 67890);
                finalize_swizzled_centroids_kernel<N_DIMS><<<grid_k, BLOCK_SIZE>>>(
                    d_centroids, centroid_sums.ptr<float>(), counts.ptr<int>(),
                    d_shN, k, n, seed);
                LFS_CUDA_LAUNCH_CHECK(nullptr, "io.kmeans.finalize_swizzled_centroids");
            }

            LFS_CUDA_AWAIT(kmeans_ticket, cudaDeviceSynchronize(), "io.kmeans.swizzled_bruteforce_iteration_sync");
            return {centroids, labels};
        }

        // Hierarchical K-Means for large k
        constexpr int NUM_SUPER_CLUSTERS = 256;
        constexpr int NUM_NEAREST_SUPERS = 4;

        template <int N_DIMS>
        __global__ void __launch_bounds__(BLOCK_SIZE, 4)
            assign_nearest_top_supers_kernel(
                const float* __restrict__ data,
                const float* __restrict__ centroids,
                const float* __restrict__ centroid_norms,
                int* __restrict__ nearest_supers,
                const int n_points,
                const int k) {
            __shared__ float shared_centroids[ASSIGN_CENTROID_TILE][ASSIGN_SHARED_K_STRIDE];
            __shared__ float shared_centroid_norms[ASSIGN_CENTROID_TILE];

            const int point = blockIdx.x * blockDim.x + threadIdx.x;
            float best_dists[NUM_NEAREST_SUPERS];
            int best_idxs[NUM_NEAREST_SUPERS];
#pragma unroll
            for (int i = 0; i < NUM_NEAREST_SUPERS; ++i) {
                best_dists[i] = 1e30f;
                best_idxs[i] = k;
            }

            const int num_tiles = (k + ASSIGN_CENTROID_TILE - 1) / ASSIGN_CENTROID_TILE;
            for (int tile = 0; tile < num_tiles; ++tile) {
                const int centroid_start = tile * ASSIGN_CENTROID_TILE;
                for (int i = threadIdx.x; i < ASSIGN_CENTROID_TILE * N_DIMS; i += BLOCK_SIZE) {
                    const int centroid = i / N_DIMS;
                    const int dim = i % N_DIMS;
                    const int global_centroid = centroid_start + centroid;
                    shared_centroids[centroid][dim] = global_centroid < k
                                                          ? centroids[static_cast<long long>(global_centroid) * N_DIMS + dim]
                                                          : 0.0f;
                }
                for (int i = threadIdx.x; i < ASSIGN_CENTROID_TILE; i += BLOCK_SIZE) {
                    const int global_centroid = centroid_start + i;
                    shared_centroid_norms[i] = global_centroid < k ? centroid_norms[global_centroid] : 0.0f;
                }
                __syncthreads();

                if (point < n_points) {
                    const float* point_values = data + static_cast<long long>(point) * N_DIMS;
                    for (int c = 0; c < ASSIGN_CENTROID_TILE; ++c) {
                        const int global_centroid = centroid_start + c;
                        if (global_centroid >= k) {
                            continue;
                        }
                        float dot = 0.0f;
#pragma unroll
                        for (int d = 0; d < N_DIMS; ++d) {
                            dot = fmaf(point_values[d], shared_centroids[c][d], dot);
                        }
                        const float dist = fmaf(-2.0f, dot, shared_centroid_norms[c]);
                        int insert = NUM_NEAREST_SUPERS;
                        for (int i = 0; i < NUM_NEAREST_SUPERS; ++i) {
                            if (dist < best_dists[i] ||
                                (dist == best_dists[i] && global_centroid < best_idxs[i])) {
                                insert = i;
                                break;
                            }
                        }
                        if (insert < NUM_NEAREST_SUPERS) {
                            for (int i = NUM_NEAREST_SUPERS - 1; i > insert; --i) {
                                best_dists[i] = best_dists[i - 1];
                                best_idxs[i] = best_idxs[i - 1];
                            }
                            best_dists[insert] = dist;
                            best_idxs[insert] = global_centroid;
                        }
                    }
                }
                __syncthreads();
            }

            if (point < n_points) {
#pragma unroll
                for (int i = 0; i < NUM_NEAREST_SUPERS; ++i) {
                    nearest_supers[point * NUM_NEAREST_SUPERS + i] = best_idxs[i];
                }
            }
        }

        template <int N_DIMS>
        __global__ void __launch_bounds__(BLOCK_SIZE, 3)
            assign_grouped_swizzled_kernel(
                const float4* __restrict__ shN,
                const float* __restrict__ centroids,
                const float* __restrict__ centroid_norms,
                const int* __restrict__ sorted_point_idx,
                const int* __restrict__ group_offsets,
                const int* __restrict__ group_task_offsets,
                const int* __restrict__ group_candidate_offsets,
                const int* __restrict__ group_candidate_supers,
                const int* __restrict__ super_offsets,
                const int* __restrict__ super_indices,
                int* __restrict__ labels,
                const int n_points,
                const int k) {
            __shared__ float shared_points[ASSIGN_POINT_TILE][ASSIGN_SHARED_K_STRIDE];
            __shared__ float shared_centroids[ASSIGN_CENTROID_TILE][ASSIGN_SHARED_K_STRIDE];
            __shared__ float shared_centroid_norms[ASSIGN_CENTROID_TILE];
            __shared__ int shared_candidate_ids[ASSIGN_CENTROID_TILE];

            const int task = blockIdx.x;
            int group_lo = 0;
            int group_hi = NUM_SUPER_CLUSTERS;
            while (group_lo + 1 < group_hi) {
                const int mid = (group_lo + group_hi) / 2;
                if (group_task_offsets[mid] <= task) {
                    group_lo = mid;
                } else {
                    group_hi = mid;
                }
            }
            const int group = group_lo;
            const int point_start = group_offsets[group] +
                                    (task - group_task_offsets[group]) * ASSIGN_POINT_TILE;
            const int point_end = group_offsets[group + 1];
            const int point_count = min(ASSIGN_POINT_TILE, point_end - point_start);
            if (point_start >= n_points || point_count <= 0) {
                return;
            }

            const int tid = threadIdx.x;
            const int point_row = (tid / ASSIGN_CENTROID_GROUPS) * ASSIGN_POINTS_PER_THREAD;
            const int centroid_col = (tid % ASSIGN_CENTROID_GROUPS) * ASSIGN_CENTROIDS_PER_THREAD;
            constexpr int point_slots_count = (N_DIMS + 3) / 4;
            for (int i = tid; i < ASSIGN_POINT_TILE * point_slots_count; i += BLOCK_SIZE) {
                const int point = i / point_slots_count;
                const int slot = i % point_slots_count;
                const bool valid = point < point_count;
                const int original_point = valid ? sorted_point_idx[point_start + point] : 0;
                const float4 values = valid
                                          ? shN[shAt_device(static_cast<std::uint32_t>(original_point),
                                                            static_cast<std::uint32_t>(slot), point_slots_count)]
                                          : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                shared_points[point][slot * 4 + 0] = values.x;
                shared_points[point][slot * 4 + 1] = values.y;
                shared_points[point][slot * 4 + 2] = values.z;
                shared_points[point][slot * 4 + 3] = values.w;
            }
            __syncthreads();

            float best_dist[ASSIGN_POINTS_PER_THREAD];
            int best_idx[ASSIGN_POINTS_PER_THREAD];
#pragma unroll
            for (int p = 0; p < ASSIGN_POINTS_PER_THREAD; ++p) {
                best_dist[p] = 1e30f;
                best_idx[p] = k;
            }

            const int candidate_start = group_candidate_offsets[group];
            const int candidate_count = group_candidate_offsets[group + 1] - candidate_start;
            const int num_tiles = (candidate_count + ASSIGN_CENTROID_TILE - 1) /
                                  ASSIGN_CENTROID_TILE;
            for (int tile = 0; tile < num_tiles; ++tile) {
                const int tile_start = tile * ASSIGN_CENTROID_TILE;
                for (int i = tid; i < ASSIGN_CENTROID_TILE; i += BLOCK_SIZE) {
                    const int candidate_pos = tile_start + i;
                    int candidate_idx = -1;
                    if (candidate_pos < candidate_count) {
                        int member_offset = candidate_pos;
#pragma unroll
                        for (int rank = 0; rank < NUM_NEAREST_SUPERS; ++rank) {
                            const int super_idx = group_candidate_supers[group * NUM_NEAREST_SUPERS + rank];
                            const int member_count = super_offsets[super_idx + 1] - super_offsets[super_idx];
                            if (member_offset < member_count) {
                                candidate_idx = super_indices[super_offsets[super_idx] + member_offset];
                                break;
                            }
                            member_offset -= member_count;
                        }
                    }
                    shared_candidate_ids[i] = candidate_idx;
                    if (candidate_idx >= 0) {
#pragma unroll
                        for (int d = 0; d < N_DIMS; ++d) {
                            shared_centroids[i][d] =
                                centroids[static_cast<long long>(candidate_idx) * N_DIMS + d];
                        }
                        shared_centroid_norms[i] = centroid_norms[candidate_idx];
                    } else {
#pragma unroll
                        for (int d = 0; d < N_DIMS; ++d) {
                            shared_centroids[i][d] = 0.0f;
                        }
                        shared_centroid_norms[i] = 0.0f;
                    }
                }
                __syncthreads();

                float dots[ASSIGN_POINTS_PER_THREAD][ASSIGN_CENTROIDS_PER_THREAD] = {};
#pragma unroll
                for (int d = 0; d < N_DIMS; ++d) {
                    float point_values[ASSIGN_POINTS_PER_THREAD];
                    float centroid_values[ASSIGN_CENTROIDS_PER_THREAD];
#pragma unroll
                    for (int p = 0; p < ASSIGN_POINTS_PER_THREAD; ++p) {
                        point_values[p] = shared_points[point_row + p][d];
                    }
#pragma unroll
                    for (int c = 0; c < ASSIGN_CENTROIDS_PER_THREAD; ++c) {
                        centroid_values[c] = shared_centroids[centroid_col + c][d];
                    }
#pragma unroll
                    for (int p = 0; p < ASSIGN_POINTS_PER_THREAD; ++p) {
#pragma unroll
                        for (int c = 0; c < ASSIGN_CENTROIDS_PER_THREAD; ++c) {
                            dots[p][c] = fmaf(point_values[p], centroid_values[c], dots[p][c]);
                        }
                    }
                }

#pragma unroll
                for (int p = 0; p < ASSIGN_POINTS_PER_THREAD; ++p) {
#pragma unroll
                    for (int c = 0; c < ASSIGN_CENTROIDS_PER_THREAD; ++c) {
                        const int candidate_idx = shared_candidate_ids[centroid_col + c];
                        if (candidate_idx >= 0) {
                            const float dist = fmaf(-2.0f, dots[p][c],
                                                    shared_centroid_norms[centroid_col + c]);
                            if (dist < best_dist[p] ||
                                (dist == best_dist[p] && candidate_idx < best_idx[p])) {
                                best_dist[p] = dist;
                                best_idx[p] = candidate_idx;
                            }
                        }
                    }
                }
                __syncthreads();
            }

            const unsigned group_mask = 0xffffffffu;
            const int group_lane = tid % ASSIGN_CENTROID_GROUPS;
#pragma unroll
            for (int p = 0; p < ASSIGN_POINTS_PER_THREAD; ++p) {
                for (int offset = ASSIGN_CENTROID_GROUPS / 2; offset > 0; offset >>= 1) {
                    const float other_dist = __shfl_down_sync(
                        group_mask, best_dist[p], offset, ASSIGN_CENTROID_GROUPS);
                    const int other_idx = __shfl_down_sync(
                        group_mask, best_idx[p], offset, ASSIGN_CENTROID_GROUPS);
                    if (other_dist < best_dist[p] ||
                        (other_dist == best_dist[p] && other_idx < best_idx[p])) {
                        best_dist[p] = other_dist;
                        best_idx[p] = other_idx;
                    }
                }
                if (group_lane == 0) {
                    const int sorted_position = point_start + point_row + p;
                    if (sorted_position < point_end && sorted_position < point_start + point_count) {
                        const int original_point = sorted_point_idx[sorted_position];
                        labels[original_point] = best_idx[p];
                    }
                }
            }
        }

        __global__ void build_group_tile_counts_kernel(
            const int* __restrict__ group_offsets,
            int* __restrict__ group_tile_counts) {
            const int group = blockIdx.x * blockDim.x + threadIdx.x;
            if (group < NUM_SUPER_CLUSTERS) {
                const int count = group_offsets[group + 1] - group_offsets[group];
                group_tile_counts[group] = (count + ASSIGN_POINT_TILE - 1) / ASSIGN_POINT_TILE;
            }
        }

        void build_grouped_point_order_gpu(
            const int* d_membership,
            const int n_points,
            Tensor& sorted_point_idx,
            Tensor& group_offsets,
            Tensor& group_task_offsets,
            int& num_tasks) {
            auto group_keys = Tensor::zeros({static_cast<size_t>(n_points)}, Device::GPU, DataType::Int32);
            LFS_CUDA_CHECK(cudaMemcpy(group_keys.ptr<int>(), d_membership,
                                      static_cast<size_t>(n_points) * sizeof(int), cudaMemcpyDeviceToDevice));

            thrust::device_ptr<int> sorted_ptr(sorted_point_idx.ptr<int>());
            thrust::sequence(sorted_ptr, sorted_ptr + n_points);
            thrust::device_ptr<int> keys_ptr(group_keys.ptr<int>());
            thrust::stable_sort_by_key(keys_ptr, keys_ptr + n_points, sorted_ptr);

            thrust::device_ptr<int> offsets_ptr(group_offsets.ptr<int>());
            thrust::lower_bound(
                keys_ptr, keys_ptr + n_points,
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(NUM_SUPER_CLUSTERS + 1),
                offsets_ptr);

            auto group_tile_counts = Tensor::zeros(
                {static_cast<size_t>(NUM_SUPER_CLUSTERS)}, Device::GPU, DataType::Int32);
            build_group_tile_counts_kernel<<<1, BLOCK_SIZE>>>(
                group_offsets.ptr<int>(), group_tile_counts.ptr<int>());
            LFS_CUDA_LAUNCH_CHECK(nullptr, "io.kmeans.build_group_tile_counts");

            thrust::device_ptr<int> tile_counts_ptr(group_tile_counts.ptr<int>());
            thrust::device_ptr<int> task_offsets_ptr(group_task_offsets.ptr<int>());
            thrust::inclusive_scan(tile_counts_ptr, tile_counts_ptr + NUM_SUPER_CLUSTERS,
                                   task_offsets_ptr + 1);
            LFS_CUDA_CHECK(cudaMemset(group_task_offsets.ptr<int>(), 0, sizeof(int)));
            LFS_CUDA_CHECK(cudaMemcpy(&num_tasks, group_task_offsets.ptr<int>() + NUM_SUPER_CLUSTERS,
                                      sizeof(int), cudaMemcpyDeviceToHost));
        }

        template <int N_DIMS>
        void build_group_candidate_supers_gpu(
            const float* d_super_centroids,
            const float* d_super_norms,
            const int* d_super_offsets,
            Tensor& nearest_supers,
            Tensor& group_candidate_offsets,
            Tensor& group_candidate_supers) {
            assign_nearest_top_supers_kernel<N_DIMS><<<1, BLOCK_SIZE>>>(
                d_super_centroids, d_super_centroids, d_super_norms,
                nearest_supers.ptr<int>(), NUM_SUPER_CLUSTERS, NUM_SUPER_CLUSTERS);
            LFS_CUDA_LAUNCH_CHECK(nullptr, "io.kmeans.assign_nearest_super_neighbors");

            std::vector<int> h_super_offsets(NUM_SUPER_CLUSTERS + 1);
            std::vector<int> h_nearest_supers(NUM_SUPER_CLUSTERS * NUM_NEAREST_SUPERS);
            LFS_CUDA_CHECK(cudaMemcpy(h_super_offsets.data(), d_super_offsets,
                                      h_super_offsets.size() * sizeof(int), cudaMemcpyDeviceToHost));
            LFS_CUDA_CHECK(cudaMemcpy(h_nearest_supers.data(), nearest_supers.ptr<int>(),
                                      h_nearest_supers.size() * sizeof(int), cudaMemcpyDeviceToHost));

            int first_nonempty_super = 0;
            while (first_nonempty_super < NUM_SUPER_CLUSTERS &&
                   h_super_offsets[first_nonempty_super] == h_super_offsets[first_nonempty_super + 1]) {
                ++first_nonempty_super;
            }

            std::vector<int> h_candidate_offsets(NUM_SUPER_CLUSTERS + 1, 0);
            std::vector<int> h_candidate_supers(
                NUM_SUPER_CLUSTERS * NUM_NEAREST_SUPERS, first_nonempty_super);
            for (int group = 0; group < NUM_SUPER_CLUSTERS; ++group) {
                int candidate_count = 0;
                for (int rank = 0; rank < NUM_NEAREST_SUPERS; ++rank) {
                    const int super_idx = h_nearest_supers[group * NUM_NEAREST_SUPERS + rank];
                    h_candidate_supers[group * NUM_NEAREST_SUPERS + rank] = super_idx;
                    candidate_count += h_super_offsets[super_idx + 1] - h_super_offsets[super_idx];
                }
                if (candidate_count == 0) {
                    h_candidate_supers[group * NUM_NEAREST_SUPERS + NUM_NEAREST_SUPERS - 1] =
                        first_nonempty_super;
                    candidate_count = h_super_offsets[first_nonempty_super + 1] -
                                      h_super_offsets[first_nonempty_super];
                }
                h_candidate_offsets[group + 1] = h_candidate_offsets[group] + candidate_count;
            }

            const int candidate_count = h_candidate_offsets.back();
            if (candidate_count > 0) {
                // The four super ids are the compact per-group candidate index list. The
                // grouped kernel expands each list through the super CSR ranges per tile.
                LFS_CUDA_CHECK(cudaMemcpy(group_candidate_offsets.ptr<int>(), h_candidate_offsets.data(),
                                          h_candidate_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
                LFS_CUDA_CHECK(cudaMemcpy(group_candidate_supers.ptr<int>(), h_candidate_supers.data(),
                                          h_candidate_supers.size() * sizeof(int), cudaMemcpyHostToDevice));
            }
        }

        template <int N_DIMS>
        std::tuple<Tensor, Tensor> kmeans_swizzled_hierarchical_impl(
            const Tensor& shN_swizzled,
            const int n,
            const int k,
            const int iterations) {
            auto shN_gpu = as_cuda_contiguous(shN_swizzled);
            const auto* d_shN = reinterpret_cast<const float4*>(shN_gpu.ptr<float>());

            if (n <= k) {
                auto centroids = Tensor::zeros({static_cast<size_t>(n), static_cast<size_t>(N_DIMS)},
                                               Device::GPU, DataType::Float32);
                auto labels = Tensor::arange(n).to(DataType::Int32).cuda();
                const int grid_n = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
                const auto kmeans_ticket = ::lfs::core::cuda_record_range(
                    /*stream=*/nullptr, "io.kmeans.swizzled_hierarchical_gather");
                gather_swizzled_centroids_kernel<N_DIMS><<<grid_n, BLOCK_SIZE>>>(
                    d_shN, labels.ptr<int>(), centroids.ptr<float>(), n);
                LFS_CUDA_LAUNCH_CHECK(nullptr, "io.kmeans.gather_swizzled_centroids");
                LFS_CUDA_AWAIT(kmeans_ticket, cudaDeviceSynchronize(), "io.kmeans.swizzled_hierarchical_gather_sync");
                return {centroids, labels};
            }

            CudaTimingState timing_state{
                lfs::core::Logger::get().is_enabled(lfs::core::LogLevel::Debug)};
            std::optional<CudaEventTimer> init_sampling_timer;
            if (timing_state.enabled) {
                init_sampling_timer.emplace(timing_state);
                init_sampling_timer->record_start();
            }

            auto centroids = Tensor::zeros({static_cast<size_t>(k), static_cast<size_t>(N_DIMS)},
                                           Device::GPU, DataType::Float32);
            float* d_centroids = centroids.ptr<float>();

            {
                std::random_device rd;
                auto perm = sample_unique_indices_gpu(n, k, rd());

                const int grid_k = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
                gather_swizzled_centroids_kernel<N_DIMS><<<grid_k, BLOCK_SIZE>>>(
                    d_shN, perm.ptr<int>(), d_centroids, k);
                LFS_CUDA_LAUNCH_CHECK(nullptr, "io.kmeans.gather_swizzled_centroids");
            }

            auto super_centroids = Tensor::zeros({static_cast<size_t>(NUM_SUPER_CLUSTERS), static_cast<size_t>(N_DIMS)},
                                                 Device::GPU, DataType::Float32);
            auto super_membership = Tensor::zeros({static_cast<size_t>(k)}, Device::GPU, DataType::Int32);
            auto super_norms = Tensor::zeros({static_cast<size_t>(NUM_SUPER_CLUSTERS)}, Device::GPU, DataType::Float32);

            {
                std::random_device rd;
                auto perm = sample_unique_indices_gpu(k, NUM_SUPER_CLUSTERS, rd());

                const int grid_super = (NUM_SUPER_CLUSTERS + BLOCK_SIZE - 1) / BLOCK_SIZE;
                gather_centroids_kernel<N_DIMS><<<grid_super, BLOCK_SIZE>>>(
                    d_centroids, perm.ptr<int>(), super_centroids.ptr<float>(), NUM_SUPER_CLUSTERS);
                LFS_CUDA_LAUNCH_CHECK(nullptr, "io.kmeans.gather_centroids");
            }

            auto super_sums = Tensor::zeros({static_cast<size_t>(NUM_SUPER_CLUSTERS), static_cast<size_t>(N_DIMS)},
                                            Device::GPU, DataType::Float32);
            auto super_counts = Tensor::zeros({static_cast<size_t>(NUM_SUPER_CLUSTERS)}, Device::GPU, DataType::Int32);

            const int grid_k = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const int grid_k_assign = (k + ASSIGN_POINT_TILE - 1) / ASSIGN_POINT_TILE;
            const int grid_super = (NUM_SUPER_CLUSTERS + BLOCK_SIZE - 1) / BLOCK_SIZE;

            for (int iter = 0; iter < 5; ++iter) {
                compute_centroid_norms_kernel<N_DIMS><<<grid_super, BLOCK_SIZE>>>(
                    super_centroids.ptr<float>(), super_norms.ptr<float>(), NUM_SUPER_CLUSTERS);
                LFS_CUDA_LAUNCH_CHECK(nullptr, "io.kmeans.compute_super_norms");
                assign_nearest_bruteforce_kernel<N_DIMS><<<grid_k_assign, BLOCK_SIZE>>>(
                    d_centroids, super_centroids.ptr<float>(), super_norms.ptr<float>(),
                    super_membership.ptr<int>(), k, NUM_SUPER_CLUSTERS);
                LFS_CUDA_LAUNCH_CHECK(nullptr, "io.kmeans.assign_nearest");

                super_sums.zero_();
                super_counts.zero_();

                accumulate_centroids_kernel<N_DIMS><<<grid_k, BLOCK_SIZE>>>(
                    d_centroids, super_membership.ptr<int>(), super_sums.ptr<float>(), super_counts.ptr<int>(), k);
                LFS_CUDA_LAUNCH_CHECK(nullptr, "io.kmeans.accumulate_centroids");

                const unsigned int seed = static_cast<unsigned int>(iter * 12345 + 67890);
                finalize_centroids_kernel<N_DIMS><<<grid_super, BLOCK_SIZE>>>(
                    super_centroids.ptr<float>(), super_sums.ptr<float>(), super_counts.ptr<int>(),
                    d_centroids, NUM_SUPER_CLUSTERS, k, seed);
                LFS_CUDA_LAUNCH_CHECK(nullptr, "io.kmeans.finalize_centroids");
            }

            compute_centroid_norms_kernel<N_DIMS><<<grid_super, BLOCK_SIZE>>>(
                super_centroids.ptr<float>(), super_norms.ptr<float>(), NUM_SUPER_CLUSTERS);
            LFS_CUDA_LAUNCH_CHECK(nullptr, "io.kmeans.compute_super_norms");
            assign_nearest_bruteforce_kernel<N_DIMS><<<grid_k_assign, BLOCK_SIZE>>>(
                d_centroids, super_centroids.ptr<float>(), super_norms.ptr<float>(),
                super_membership.ptr<int>(), k, NUM_SUPER_CLUSTERS);
            LFS_CUDA_LAUNCH_CHECK(nullptr, "io.kmeans.assign_nearest");

            auto super_offsets = Tensor::zeros({static_cast<size_t>(NUM_SUPER_CLUSTERS + 1)}, Device::GPU, DataType::Int32);
            auto super_indices = Tensor::zeros({static_cast<size_t>(k)}, Device::GPU, DataType::Int32);
            build_csr_offsets_gpu(super_membership.ptr<int>(), super_offsets.ptr<int>(),
                                  super_indices.ptr<int>(), k, NUM_SUPER_CLUSTERS);
            if (init_sampling_timer) {
                init_sampling_timer->record_stop();
            }

            auto labels = Tensor::zeros({static_cast<size_t>(n)}, Device::GPU, DataType::Int32);
            auto point_super_membership = Tensor::zeros(
                {static_cast<size_t>(n)}, Device::GPU, DataType::Int32);
            auto sorted_point_idx = Tensor::zeros(
                {static_cast<size_t>(n)}, Device::GPU, DataType::Int32);
            auto group_offsets = Tensor::zeros(
                {static_cast<size_t>(NUM_SUPER_CLUSTERS + 1)}, Device::GPU, DataType::Int32);
            auto group_task_offsets = Tensor::zeros(
                {static_cast<size_t>(NUM_SUPER_CLUSTERS + 1)}, Device::GPU, DataType::Int32);
            auto nearest_supers = Tensor::zeros(
                {static_cast<size_t>(NUM_SUPER_CLUSTERS), static_cast<size_t>(NUM_NEAREST_SUPERS)},
                Device::GPU, DataType::Int32);
            auto group_candidate_offsets = Tensor::zeros(
                {static_cast<size_t>(NUM_SUPER_CLUSTERS + 1)}, Device::GPU, DataType::Int32);
            auto group_candidate_supers = Tensor::zeros(
                {static_cast<size_t>(NUM_SUPER_CLUSTERS * NUM_NEAREST_SUPERS)},
                Device::GPU, DataType::Int32);
            auto centroid_sums = Tensor::zeros({static_cast<size_t>(k), static_cast<size_t>(N_DIMS)},
                                               Device::GPU, DataType::Float32);
            auto centroid_counts = Tensor::zeros({static_cast<size_t>(k)}, Device::GPU, DataType::Int32);
            auto centroid_norms = Tensor::zeros({static_cast<size_t>(k)}, Device::GPU, DataType::Float32);

            const int grid_n = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const int grid_n_exact_assign = (n + ASSIGN_POINT_TILE - 1) / ASSIGN_POINT_TILE;
            const int grid_n_super_assign = (n + ASSIGN_POINT_TILE - 1) / ASSIGN_POINT_TILE;

            const int timed_iterations = std::max(0, iterations);
            std::vector<CudaEventTimer> assign_timers;
            std::vector<CudaEventTimer> supers_timers;
            std::vector<CudaEventTimer> group_timers;
            std::vector<CudaEventTimer> members_timers;
            std::vector<CudaEventTimer> accumulate_finalize_timers;
            assign_timers.reserve(static_cast<size_t>(timed_iterations));
            supers_timers.reserve(static_cast<size_t>(timed_iterations));
            group_timers.reserve(static_cast<size_t>(timed_iterations));
            members_timers.reserve(static_cast<size_t>(timed_iterations));
            accumulate_finalize_timers.reserve(static_cast<size_t>(timed_iterations));
            for (int iter = 0; iter < timed_iterations; ++iter) {
                assign_timers.emplace_back(timing_state);
                supers_timers.emplace_back(timing_state);
                group_timers.emplace_back(timing_state);
                members_timers.emplace_back(timing_state);
                accumulate_finalize_timers.emplace_back(timing_state);
            }

            const auto kmeans_ticket = ::lfs::core::cuda_record_range(
                /*stream=*/nullptr, "io.kmeans.swizzled_hierarchical_iteration");
            for (int iter = 0; iter < iterations; ++iter) {
                if (iter > 0) {
                    compute_centroid_norms_kernel<N_DIMS><<<grid_super, BLOCK_SIZE>>>(
                        super_centroids.ptr<float>(), super_norms.ptr<float>(), NUM_SUPER_CLUSTERS);
                    LFS_CUDA_LAUNCH_CHECK(nullptr, "io.kmeans.compute_super_norms");
                    assign_nearest_bruteforce_kernel<N_DIMS><<<grid_k_assign, BLOCK_SIZE>>>(
                        d_centroids, super_centroids.ptr<float>(), super_norms.ptr<float>(),
                        super_membership.ptr<int>(), k, NUM_SUPER_CLUSTERS);
                    LFS_CUDA_LAUNCH_CHECK(nullptr, "io.kmeans.assign_nearest");

                    super_sums.zero_();
                    super_counts.zero_();
                    accumulate_centroids_kernel<N_DIMS><<<grid_k, BLOCK_SIZE>>>(
                        d_centroids, super_membership.ptr<int>(), super_sums.ptr<float>(), super_counts.ptr<int>(), k);
                    LFS_CUDA_LAUNCH_CHECK(nullptr, "io.kmeans.accumulate_centroids");
                    finalize_centroids_kernel<N_DIMS><<<grid_super, BLOCK_SIZE>>>(
                        super_centroids.ptr<float>(), super_sums.ptr<float>(), super_counts.ptr<int>(),
                        d_centroids, NUM_SUPER_CLUSTERS, k, iter * 111);
                    LFS_CUDA_LAUNCH_CHECK(nullptr, "io.kmeans.finalize_centroids");

                    build_csr_offsets_gpu(super_membership.ptr<int>(), super_offsets.ptr<int>(),
                                          super_indices.ptr<int>(), k, NUM_SUPER_CLUSTERS);
                }

                const bool use_exact = (iter == iterations - 1);
                compute_centroid_norms_kernel<N_DIMS><<<grid_k, BLOCK_SIZE>>>(
                    d_centroids, centroid_norms.ptr<float>(), k);
                LFS_CUDA_LAUNCH_CHECK(nullptr, "io.kmeans.compute_centroid_norms");
                assign_timers[static_cast<size_t>(iter)].record_start();

                if (use_exact) {
                    assign_nearest_swizzled_bruteforce_kernel<N_DIMS><<<grid_n_exact_assign, BLOCK_SIZE>>>(
                        d_shN, d_centroids, centroid_norms.ptr<float>(), labels.ptr<int>(), n, k);
                    LFS_CUDA_LAUNCH_CHECK(nullptr, "io.kmeans.assign_nearest_swizzled");
                } else {
                    supers_timers[static_cast<size_t>(iter)].record_start();
                    compute_centroid_norms_kernel<N_DIMS><<<grid_super, BLOCK_SIZE>>>(
                        super_centroids.ptr<float>(), super_norms.ptr<float>(), NUM_SUPER_CLUSTERS);
                    LFS_CUDA_LAUNCH_CHECK(nullptr, "io.kmeans.compute_super_norms");
                    // The refinement needs only the nearest super id per point. The
                    // four-neighbor list is computed once per super below, not per point.
                    assign_nearest_swizzled_bruteforce_kernel<N_DIMS><<<grid_n_super_assign, BLOCK_SIZE>>>(
                        d_shN, super_centroids.ptr<float>(), super_norms.ptr<float>(),
                        point_super_membership.ptr<int>(), n, NUM_SUPER_CLUSTERS);
                    LFS_CUDA_LAUNCH_CHECK(nullptr, "io.kmeans.assign_points_to_supers");
                    supers_timers[static_cast<size_t>(iter)].record_stop();

                    group_timers[static_cast<size_t>(iter)].record_start();
                    int num_group_tasks = 0;
                    build_grouped_point_order_gpu(
                        point_super_membership.ptr<int>(), n, sorted_point_idx,
                        group_offsets, group_task_offsets, num_group_tasks);
                    group_timers[static_cast<size_t>(iter)].record_stop();

                    members_timers[static_cast<size_t>(iter)].record_start();
                    build_group_candidate_supers_gpu<N_DIMS>(
                        super_centroids.ptr<float>(), super_norms.ptr<float>(), super_offsets.ptr<int>(),
                        nearest_supers, group_candidate_offsets, group_candidate_supers);
                    members_timers[static_cast<size_t>(iter)].record_stop();

                    if (num_group_tasks > 0) {
                        // Refinement is approximate: each point is searched only in the
                        // union of the four supers nearest to its assigned super. The
                        // final iteration below remains an exact argmin over all k.
                        assign_grouped_swizzled_kernel<N_DIMS><<<num_group_tasks, BLOCK_SIZE>>>(
                            d_shN, d_centroids, centroid_norms.ptr<float>(),
                            sorted_point_idx.ptr<int>(), group_offsets.ptr<int>(),
                            group_task_offsets.ptr<int>(), group_candidate_offsets.ptr<int>(),
                            group_candidate_supers.ptr<int>(), super_offsets.ptr<int>(),
                            super_indices.ptr<int>(), labels.ptr<int>(), n, k);
                        LFS_CUDA_LAUNCH_CHECK(nullptr, "io.kmeans.assign_grouped_candidates");
                    }
                }
                assign_timers[static_cast<size_t>(iter)].record_stop();

                accumulate_finalize_timers[static_cast<size_t>(iter)].record_start();
                centroid_sums.zero_();
                centroid_counts.zero_();

                accumulate_swizzled_centroids_kernel<N_DIMS><<<grid_n, BLOCK_SIZE>>>(
                    d_shN, labels.ptr<int>(), centroid_sums.ptr<float>(), centroid_counts.ptr<int>(), n);
                LFS_CUDA_LAUNCH_CHECK(nullptr, "io.kmeans.accumulate_swizzled_centroids");

                const unsigned int seed = static_cast<unsigned int>(iter * 12345 + 67890);
                finalize_swizzled_centroids_kernel<N_DIMS><<<grid_k, BLOCK_SIZE>>>(
                    d_centroids, centroid_sums.ptr<float>(), centroid_counts.ptr<int>(),
                    d_shN, k, n, seed);
                LFS_CUDA_LAUNCH_CHECK(nullptr, "io.kmeans.finalize_swizzled_centroids");
                accumulate_finalize_timers[static_cast<size_t>(iter)].record_stop();
            }

            LFS_CUDA_AWAIT(kmeans_ticket, cudaDeviceSynchronize(), "io.kmeans.swizzled_hierarchical_iteration_sync");

            if (timing_state.enabled && init_sampling_timer) {
                float init_sampling_ms = 0.0f;
                float hierarchical_assign_ms = 0.0f;
                float supers_ms = 0.0f;
                float group_ms = 0.0f;
                float members_ms = 0.0f;
                float final_exact_assign_ms = 0.0f;
                float accumulate_finalize_ms = 0.0f;
                bool timings_valid = init_sampling_timer->elapsed_ms(init_sampling_ms);
                for (int iter = 0; iter < iterations; ++iter) {
                    float assign_ms = 0.0f;
                    float accumulate_finalize_iter_ms = 0.0f;
                    timings_valid = assign_timers[static_cast<size_t>(iter)].elapsed_ms(assign_ms) &&
                                    accumulate_finalize_timers[static_cast<size_t>(iter)].elapsed_ms(
                                        accumulate_finalize_iter_ms) &&
                                    timings_valid;
                    if (iter == iterations - 1) {
                        final_exact_assign_ms = assign_ms;
                    } else {
                        hierarchical_assign_ms += assign_ms;
                        float supers_iter_ms = 0.0f;
                        float group_iter_ms = 0.0f;
                        float members_iter_ms = 0.0f;
                        timings_valid = supers_timers[static_cast<size_t>(iter)].elapsed_ms(supers_iter_ms) &&
                                        group_timers[static_cast<size_t>(iter)].elapsed_ms(group_iter_ms) &&
                                        members_timers[static_cast<size_t>(iter)].elapsed_ms(members_iter_ms) &&
                                        timings_valid;
                        supers_ms += supers_iter_ms;
                        group_ms += group_iter_ms;
                        members_ms += members_iter_ms;
                    }
                    accumulate_finalize_ms += accumulate_finalize_iter_ms;
                }
                if (timings_valid && timing_state.enabled) {
                    LOG_DEBUG(
                        "SH k-means CUDA stages: init_sampling_ms=%0.3f hierarchical_assign_ms=%0.3f "
                        "supers_ms=%0.3f group_ms=%0.3f members_ms=%0.3f "
                        "final_exact_assign_ms=%0.3f accumulate_finalize_ms=%0.3f",
                        init_sampling_ms, hierarchical_assign_ms, supers_ms, group_ms, members_ms,
                        final_exact_assign_ms, accumulate_finalize_ms);
                }
            }
            return {centroids, labels};
        }

    } // anonymous namespace

    std::tuple<Tensor, Tensor> kmeans_sh_swizzled(
        const Tensor& shN_swizzled,
        const int n_points,
        const int sh_coeffs,
        const int k,
        const int iterations) {
        if (!shN_swizzled.is_valid() || shN_swizzled.ndim() != 1 ||
            shN_swizzled.dtype() != DataType::Float32) {
            LOG_ERROR("kmeans_sh_swizzled expects a 1D float32 swizzled shN tensor");
            return {Tensor(), Tensor()};
        }
        if (n_points <= 0 || k <= 0) {
            LOG_ERROR("kmeans_sh_swizzled expects positive n_points and k");
            return {Tensor(), Tensor()};
        }
        const auto required_floats = static_cast<int64_t>(
            lfs::core::sh_swizzled_float_count(static_cast<size_t>(n_points),
                                               static_cast<std::uint32_t>(sh_coeffs)));
        if (shN_swizzled.numel() < required_floats) {
            LOG_ERROR("kmeans_sh_swizzled input has {} floats but needs at least {} for {} points",
                      shN_swizzled.numel(), required_floats, n_points);
            return {Tensor(), Tensor()};
        }

        const int n_dims = sh_coeffs * 3;
        switch (n_dims) {
        case 9:
            return kmeans_swizzled_bruteforce_impl<9>(shN_swizzled, n_points, k, iterations);
        case 24:
            return kmeans_swizzled_bruteforce_impl<24>(shN_swizzled, n_points, k, iterations);
        case 45:
            if (k >= 4096) {
                return kmeans_swizzled_hierarchical_impl<45>(shN_swizzled, n_points, k, iterations);
            }
            return kmeans_swizzled_bruteforce_impl<45>(shN_swizzled, n_points, k, iterations);
        default:
            LOG_ERROR("kmeans_sh_swizzled unsupported SH coefficient count {} (dims {}). Supported: 3, 8, 15",
                      sh_coeffs, n_dims);
            return {Tensor(), Tensor()};
        }
    }

} // namespace lfs::io
