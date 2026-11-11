/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/sh_layout.cuh"
#include "core/cuda_error.hpp"
#include "core/logger.hpp"
#include "kmeans.hpp"
#include <algorithm>
#include <cmath>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>
#include <mma.h>
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
                LOG_DEBUG("SOG k-means timing disabled after %s failed: %s",
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
                    LOG_DEBUG("SOG k-means timing cleanup %s failed: %s",
                              operation, cudaGetErrorString(status));
                }
            }

            CudaTimingState* state_ = nullptr;
            cudaEvent_t start_ = nullptr;
            cudaEvent_t stop_ = nullptr;
        };

        // Tensor::cuda() deep-copies CUDA tensors in this codebase; borrow when possible.
        Tensor as_cuda_contiguous(const Tensor& data) {
            if (data.device() == Device::CUDA) {
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
            return Tensor::from_vector(indices, {static_cast<size_t>(count)}, Device::CUDA);
        }

        void build_csr_offsets_gpu(
            const int* d_membership,
            int* d_offsets,
            int* d_indices,
            const int k,
            const int num_clusters) {
            auto cluster_keys = Tensor::zeros({static_cast<size_t>(k)}, Device::CUDA, DataType::Int32);
            auto sorted_indices = Tensor::zeros({static_cast<size_t>(k)}, Device::CUDA, DataType::Int32);

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

        __global__ void prepare_half_sh_kernel(const float4* sh, const float* centroids, half* points, half* palette, int n, int k) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            const int row = i / 48, d = i % 48;
            const int np = (n + 127) / 128 * 128, kp = (k + 31) / 32 * 32;
            if (row < np) {
                float v = 0;
                if (row < n && d < 45) {
                    const auto slot = sh[shAt_device(row, d / 4, 12)];
                    v = reinterpret_cast<const float*>(&slot)[d % 4];
                }
                points[i] = __float2half_rn(v);
            }
            if (row < kp)
                palette[i] = __float2half_rn(row < k && d < 45 ? centroids[row * 45 + d] : 0);
        }

        // Half-precision products only reject distant candidates. Every possible
        // winner is evaluated with the reference FP32 FMA sequence and tie rule.
        __global__ void assign_sh3_screened_kernel(
            const float4* __restrict__ shN, const float* __restrict__ centroids,
            const float* __restrict__ centroid_norms, int* __restrict__ labels,
            const int n, const int k, const half* half_points, const half* half_centroids, bool have_labels) {
            constexpr int P = 128, C = 32, D = 48;
            __shared__ float points[P][49];
            // Four lanes consume each row; padding avoids eight-way bank
            // conflicts between the eight different rows read by a warp.
            __shared__ __align__(32) float approx[P][C + 4];
            __shared__ float point_norm[P], norms[C];
            __shared__ int point_safe[P];
            const int tid = threadIdx.x;
            const int row = tid / 4, lane = tid % 4;
            const int start = blockIdx.x * P;
            for (int i = tid; i < P * D; i += 512) {
                const int p = i / D, d = i % D;
                float v = 0;
                if (start + p < n && d < 45) {
                    const auto slot = shN[shAt_device(start + p, d / 4, 12)];
                    v = reinterpret_cast<const float*>(&slot)[d % 4];
                }
                points[p][d] = v;
            }
            __syncthreads();
            if (tid < P) {
                float norm = 0;
                int safe = 1;
                for (int d = 0; d < 45; ++d) {
                    const float v = points[tid][d];
                    norm = fmaf(v, v, norm);
                    safe &= isfinite(v) && fabsf(v) <= 65000;
                }
                point_norm[tid] = norm;
                point_safe[tid] = safe;
            }
            float best = 1e30f;
            int best_id = k;
            if (have_labels && start + row < n) {
                const int seed = labels[start + row];
                if (seed >= 0 && seed < k) {
                    float dot = 0;
#pragma unroll
                    for (int d = 0; d < 45; ++d)
                        dot = fmaf(points[row][d], centroids[seed * 45 + d], dot);
                    const float dist = fmaf(-2.0f, dot, centroid_norms[seed]);
                    if (dist <= best) {
                        best = dist;
                        best_id = seed;
                    }
                }
            }
            __syncthreads();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
            const int warp = tid / 32;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a[3];
            for (int d = 0; d < D; d += 16)
                nvcuda::wmma::load_matrix_sync(a[d / 16], half_points + (start + (warp / 2) * 16) * D + d, D);
#endif
            for (int base = 0; base < k; base += C) {
                if (tid < C)
                    norms[tid] = base + tid < k ? centroid_norms[base + tid] : 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> b;
                nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc;
                nvcuda::wmma::fill_fragment(acc, 0.0f);
                for (int d = 0; d < D; d += 16) {
                    nvcuda::wmma::load_matrix_sync(b, half_centroids + (base + (warp % 2) * 16) * D + d, D);
                    nvcuda::wmma::mma_sync(acc, a[d / 16], b, acc);
                }
                nvcuda::wmma::store_matrix_sync(&approx[(warp / 2) * 16][(warp % 2) * 16], acc, C + 4, nvcuda::wmma::mem_row_major);
                __syncthreads();
#else
                __syncthreads();
#endif
                for (int c = lane; c < C && base + c < k; c += 4) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
                    const float estimate = fmaf(-2.0f, approx[row][c], norms[c]);
                    // Half rounding contributes at most (2*u+u*u)*(||x||^2+
                    // ||c||^2), u=2^-11, to the score. The remaining relative
                    // allowance and 1e-8 cover half subnormals and FP32
                    // accumulation/norm/score rounding (45 coefficients).
                    // Out-of-range values always take the reference path.
                    const float margin = 0.0011f * (point_norm[row] + norms[c]) + 1e-8f;
                    if (!point_safe[row] || !(norms[c] >= 0 && norms[c] < 4.0e9f) || !isfinite(estimate) || estimate <= best + margin) {
#endif
                        float dot = 0;
#pragma unroll
                        for (int d = 0; d < 45; ++d)
                            dot = fmaf(points[row][d], centroids[(base + c) * 45 + d], dot);
                        const float dist = fmaf(-2.0f, dot, norms[c]);
                        const int id = base + c;
                        if (dist < best || (dist == best && id < best_id)) {
                            best = dist;
                            best_id = id;
                        }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
                    }
#endif
                }
                for (int delta = 2; delta > 0; delta /= 2) {
                    const float other = __shfl_xor_sync(0xffffffffu, best, delta, 4);
                    const int id = __shfl_xor_sync(0xffffffffu, best_id, delta, 4);
                    if (other < best || (other == best && id < best_id)) {
                        best = other;
                        best_id = id;
                    }
                }
                __syncthreads();
            }
            if (lane == 0 && start + row < n)
                labels[start + row] = best_id;
        }

        template <int N_DIMS, int POINTS_PER_THREAD = ASSIGN_POINTS_PER_THREAD, int THREAD_COUNT = BLOCK_SIZE>
        __global__ void __launch_bounds__(THREAD_COUNT, 3)
            assign_nearest_swizzled_bruteforce_kernel(
                const float4* __restrict__ shN,
                const float* __restrict__ centroids,
                const float* __restrict__ centroid_norms,
                int* __restrict__ labels,
                const int n_points,
                const int k) {
            constexpr int POINT_TILE = THREAD_COUNT / ASSIGN_CENTROID_GROUPS * POINTS_PER_THREAD;
            __shared__ float shared_points[POINT_TILE][ASSIGN_SHARED_K_STRIDE];
            __shared__ float shared_centroids[ASSIGN_CENTROID_TILE][ASSIGN_SHARED_K_STRIDE];
            __shared__ float shared_centroid_norms[ASSIGN_CENTROID_TILE];

            const int tid = threadIdx.x;
            const int point_row = (tid / ASSIGN_CENTROID_GROUPS) * POINTS_PER_THREAD;
            const int centroid_col = (tid % ASSIGN_CENTROID_GROUPS) * ASSIGN_CENTROIDS_PER_THREAD;
            const int point_start = blockIdx.x * POINT_TILE;
            constexpr int point_slots_count = (N_DIMS + 3) / 4;
            for (int i = tid; i < POINT_TILE * point_slots_count; i += THREAD_COUNT) {
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

            float best_dist[POINTS_PER_THREAD];
            int best_idx[POINTS_PER_THREAD];
#pragma unroll
            for (int p = 0; p < POINTS_PER_THREAD; ++p) {
                best_dist[p] = 1e30f;
                best_idx[p] = k;
            }

            const int num_tiles = (k + ASSIGN_CENTROID_TILE - 1) / ASSIGN_CENTROID_TILE;
            for (int tile = 0; tile < num_tiles; ++tile) {
                const int centroid_start = tile * ASSIGN_CENTROID_TILE;
                for (int i = tid; i < ASSIGN_CENTROID_TILE * N_DIMS; i += THREAD_COUNT) {
                    const int centroid = i / N_DIMS;
                    const int dim = i % N_DIMS;
                    const int global_centroid = centroid_start + centroid;
                    shared_centroids[centroid][dim] = global_centroid < k
                                                          ? centroids[static_cast<long long>(global_centroid) * N_DIMS + dim]
                                                          : 0.0f;
                }
                for (int i = tid; i < ASSIGN_CENTROID_TILE; i += THREAD_COUNT) {
                    const int global_centroid = centroid_start + i;
                    shared_centroid_norms[i] = global_centroid < k ? centroid_norms[global_centroid] : 0.0f;
                }
                __syncthreads();

                float dots[POINTS_PER_THREAD][ASSIGN_CENTROIDS_PER_THREAD] = {};
#pragma unroll
                for (int d = 0; d < N_DIMS; ++d) {
                    float point_values[POINTS_PER_THREAD];
                    float centroid_values[ASSIGN_CENTROIDS_PER_THREAD];
#pragma unroll
                    for (int p = 0; p < POINTS_PER_THREAD; ++p) {
                        point_values[p] = shared_points[point_row + p][d];
                    }
#pragma unroll
                    for (int c = 0; c < ASSIGN_CENTROIDS_PER_THREAD; ++c) {
                        centroid_values[c] = shared_centroids[centroid_col + c][d];
                    }
#pragma unroll
                    for (int p = 0; p < POINTS_PER_THREAD; ++p) {
#pragma unroll
                        for (int c = 0; c < ASSIGN_CENTROIDS_PER_THREAD; ++c) {
                            dots[p][c] = fmaf(point_values[p], centroid_values[c], dots[p][c]);
                        }
                    }
                }

#pragma unroll
                for (int p = 0; p < POINTS_PER_THREAD; ++p) {
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
            for (int p = 0; p < POINTS_PER_THREAD; ++p) {
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
                                               Device::CUDA, DataType::Float32);
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
                                           Device::CUDA, DataType::Float32);
            float* d_centroids = centroids.ptr<float>();

            {
                std::random_device rd;
                auto perm = sample_unique_indices_gpu(n, k, rd());

                const int grid_k = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
                gather_swizzled_centroids_kernel<N_DIMS><<<grid_k, BLOCK_SIZE>>>(
                    d_shN, perm.ptr<int>(), d_centroids, k);
                LFS_CUDA_LAUNCH_CHECK(nullptr, "io.kmeans.gather_swizzled_centroids");
            }

            auto labels = Tensor::zeros({static_cast<size_t>(n)}, Device::CUDA, DataType::Int32);
            auto centroid_sums = Tensor::zeros({static_cast<size_t>(k), static_cast<size_t>(N_DIMS)},
                                               Device::CUDA, DataType::Float32);
            auto counts = Tensor::zeros({static_cast<size_t>(k)}, Device::CUDA, DataType::Int32);
            auto centroid_norms = Tensor::zeros({static_cast<size_t>(k)}, Device::CUDA, DataType::Float32);

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

        template <int N_DIMS, int POINTS_PER_THREAD = ASSIGN_POINTS_PER_THREAD, int THREAD_COUNT = BLOCK_SIZE>
        __global__ void __launch_bounds__(THREAD_COUNT, 3)
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
            static_assert(THREAD_COUNT / ASSIGN_CENTROID_GROUPS * POINTS_PER_THREAD == ASSIGN_POINT_TILE);
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
            const int point_row = (tid / ASSIGN_CENTROID_GROUPS) * POINTS_PER_THREAD;
            const int centroid_col = (tid % ASSIGN_CENTROID_GROUPS) * ASSIGN_CENTROIDS_PER_THREAD;
            constexpr int point_slots_count = (N_DIMS + 3) / 4;
            for (int i = tid; i < ASSIGN_POINT_TILE * point_slots_count; i += THREAD_COUNT) {
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

            float best_dist[POINTS_PER_THREAD];
            int best_idx[POINTS_PER_THREAD];
#pragma unroll
            for (int p = 0; p < POINTS_PER_THREAD; ++p) {
                best_dist[p] = 1e30f;
                best_idx[p] = k;
            }

            const int candidate_start = group_candidate_offsets[group];
            const int candidate_count = group_candidate_offsets[group + 1] - candidate_start;
            const int num_tiles = (candidate_count + ASSIGN_CENTROID_TILE - 1) /
                                  ASSIGN_CENTROID_TILE;
            for (int tile = 0; tile < num_tiles; ++tile) {
                const int tile_start = tile * ASSIGN_CENTROID_TILE;
                for (int i = tid; i < ASSIGN_CENTROID_TILE; i += THREAD_COUNT) {
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

                float dots[POINTS_PER_THREAD][ASSIGN_CENTROIDS_PER_THREAD] = {};
#pragma unroll
                for (int d = 0; d < N_DIMS; ++d) {
                    float point_values[POINTS_PER_THREAD];
                    float centroid_values[ASSIGN_CENTROIDS_PER_THREAD];
#pragma unroll
                    for (int p = 0; p < POINTS_PER_THREAD; ++p) {
                        point_values[p] = shared_points[point_row + p][d];
                    }
#pragma unroll
                    for (int c = 0; c < ASSIGN_CENTROIDS_PER_THREAD; ++c) {
                        centroid_values[c] = shared_centroids[centroid_col + c][d];
                    }
#pragma unroll
                    for (int p = 0; p < POINTS_PER_THREAD; ++p) {
#pragma unroll
                        for (int c = 0; c < ASSIGN_CENTROIDS_PER_THREAD; ++c) {
                            dots[p][c] = fmaf(point_values[p], centroid_values[c], dots[p][c]);
                        }
                    }
                }

#pragma unroll
                for (int p = 0; p < POINTS_PER_THREAD; ++p) {
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
            for (int p = 0; p < POINTS_PER_THREAD; ++p) {
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

        // Both tilings compute identical FP32 distances and labels. 4 points x 8 centroids
        // per thread needs fewer shared-memory reads per FMA, which wins at 45 dims; at
        // 9/24 dims the 128-thread blocks lose more to lower occupancy than they gain.
        template <int N_DIMS>
        void launch_assign_grouped(const int num_tasks, const float4* shN, const float* centroids,
                                   const float* centroid_norms, const int* sorted_point_idx,
                                   const int* group_offsets, const int* group_task_offsets,
                                   const int* group_candidate_offsets, const int* group_candidate_supers,
                                   const int* super_offsets, const int* super_indices, int* labels,
                                   const int n_points, const int k) {
            if constexpr (N_DIMS >= 45) {
                assign_grouped_swizzled_kernel<N_DIMS, 4, 128><<<num_tasks, 128>>>(
                    shN, centroids, centroid_norms, sorted_point_idx, group_offsets, group_task_offsets,
                    group_candidate_offsets, group_candidate_supers, super_offsets, super_indices,
                    labels, n_points, k);
            } else {
                assign_grouped_swizzled_kernel<N_DIMS><<<num_tasks, BLOCK_SIZE>>>(
                    shN, centroids, centroid_norms, sorted_point_idx, group_offsets, group_task_offsets,
                    group_candidate_offsets, group_candidate_supers, super_offsets, super_indices,
                    labels, n_points, k);
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
            auto group_keys = Tensor::zeros({static_cast<size_t>(n_points)}, Device::CUDA, DataType::Int32);
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
                {static_cast<size_t>(NUM_SUPER_CLUSTERS)}, Device::CUDA, DataType::Int32);
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
            const int iterations, const bool fast_assignment) {
            auto shN_gpu = as_cuda_contiguous(shN_swizzled);
            const auto* d_shN = reinterpret_cast<const float4*>(shN_gpu.ptr<float>());

            if (n <= k) {
                auto centroids = Tensor::zeros({static_cast<size_t>(n), static_cast<size_t>(N_DIMS)},
                                               Device::CUDA, DataType::Float32);
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
                                           Device::CUDA, DataType::Float32);
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
                                                 Device::CUDA, DataType::Float32);
            auto super_membership = Tensor::zeros({static_cast<size_t>(k)}, Device::CUDA, DataType::Int32);
            auto super_norms = Tensor::zeros({static_cast<size_t>(NUM_SUPER_CLUSTERS)}, Device::CUDA, DataType::Float32);

            {
                std::random_device rd;
                auto perm = sample_unique_indices_gpu(k, NUM_SUPER_CLUSTERS, rd());

                const int grid_super = (NUM_SUPER_CLUSTERS + BLOCK_SIZE - 1) / BLOCK_SIZE;
                gather_centroids_kernel<N_DIMS><<<grid_super, BLOCK_SIZE>>>(
                    d_centroids, perm.ptr<int>(), super_centroids.ptr<float>(), NUM_SUPER_CLUSTERS);
                LFS_CUDA_LAUNCH_CHECK(nullptr, "io.kmeans.gather_centroids");
            }

            auto super_sums = Tensor::zeros({static_cast<size_t>(NUM_SUPER_CLUSTERS), static_cast<size_t>(N_DIMS)},
                                            Device::CUDA, DataType::Float32);
            auto super_counts = Tensor::zeros({static_cast<size_t>(NUM_SUPER_CLUSTERS)}, Device::CUDA, DataType::Int32);

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

            auto super_offsets = Tensor::zeros({static_cast<size_t>(NUM_SUPER_CLUSTERS + 1)}, Device::CUDA, DataType::Int32);
            auto super_indices = Tensor::zeros({static_cast<size_t>(k)}, Device::CUDA, DataType::Int32);
            build_csr_offsets_gpu(super_membership.ptr<int>(), super_offsets.ptr<int>(),
                                  super_indices.ptr<int>(), k, NUM_SUPER_CLUSTERS);
            if (init_sampling_timer) {
                init_sampling_timer->record_stop();
            }

            auto labels = Tensor::zeros({static_cast<size_t>(n)}, Device::CUDA, DataType::Int32);
            auto point_super_membership = Tensor::zeros(
                {static_cast<size_t>(n)}, Device::CUDA, DataType::Int32);
            auto sorted_point_idx = Tensor::zeros(
                {static_cast<size_t>(n)}, Device::CUDA, DataType::Int32);
            auto group_offsets = Tensor::zeros(
                {static_cast<size_t>(NUM_SUPER_CLUSTERS + 1)}, Device::CUDA, DataType::Int32);
            auto group_task_offsets = Tensor::zeros(
                {static_cast<size_t>(NUM_SUPER_CLUSTERS + 1)}, Device::CUDA, DataType::Int32);
            auto nearest_supers = Tensor::zeros(
                {static_cast<size_t>(NUM_SUPER_CLUSTERS), static_cast<size_t>(NUM_NEAREST_SUPERS)},
                Device::CUDA, DataType::Int32);
            auto group_candidate_offsets = Tensor::zeros(
                {static_cast<size_t>(NUM_SUPER_CLUSTERS + 1)}, Device::CUDA, DataType::Int32);
            auto group_candidate_supers = Tensor::zeros(
                {static_cast<size_t>(NUM_SUPER_CLUSTERS * NUM_NEAREST_SUPERS)},
                Device::CUDA, DataType::Int32);
            auto centroid_sums = Tensor::zeros({static_cast<size_t>(k), static_cast<size_t>(N_DIMS)},
                                               Device::CUDA, DataType::Float32);
            auto centroid_counts = Tensor::zeros({static_cast<size_t>(k)}, Device::CUDA, DataType::Int32);
            auto centroid_norms = Tensor::zeros({static_cast<size_t>(k)}, Device::CUDA, DataType::Float32);

            const int grid_n = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
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
                    assign_sh3_labels(shN_gpu, centroids, centroid_norms, labels, fast_assignment, iterations > 1);
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
                        launch_assign_grouped<N_DIMS>(
                            num_group_tasks, d_shN, d_centroids, centroid_norms.ptr<float>(),
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

    void assign_sh3_labels(const Tensor& shN_swizzled, const Tensor& centroids,
                           const Tensor& centroid_norms, Tensor& labels, bool fast, bool have_labels) {
        const int n = static_cast<int>(labels.numel());
        const int k = static_cast<int>(centroids.size(0));
        const auto* sh = reinterpret_cast<const float4*>(shN_swizzled.ptr<float>());
        if (fast) {
            // Half products screen distant candidates only. Possible winners
            // retain the ordinary FP32 dot-product order and lowest-index tie.
            const size_t np = (n + 127) / 128 * 128, kp = (k + 31) / 32 * 32;
            auto half_storage = Tensor::empty({(np + kp) * 24}, Device::CUDA, DataType::Float32);
            auto* half_points = reinterpret_cast<half*>(half_storage.ptr<float>());
            auto* half_centroids = half_points + np * 48;
            prepare_half_sh_kernel<<<(std::max(np, kp) * 48 + 255) / 256, 256>>>(sh, centroids.ptr<float>(), half_points, half_centroids, n, k);
            LFS_CUDA_LAUNCH_CHECK(nullptr, "io.kmeans.prepare_half_sh");
            assign_sh3_screened_kernel<<<(n + 127) / 128, 512>>>(
                sh, centroids.ptr<float>(), centroid_norms.ptr<float>(), labels.ptr<int>(), n, k, half_points, half_centroids, have_labels);
            LFS_CUDA_LAUNCH_CHECK(nullptr, "io.kmeans.assign_sh3_labels");
        } else {
            assign_nearest_swizzled_bruteforce_kernel<45><<<(n + ASSIGN_POINT_TILE - 1) / ASSIGN_POINT_TILE, BLOCK_SIZE>>>(
                sh, centroids.ptr<float>(), centroid_norms.ptr<float>(), labels.ptr<int>(), n, k);
            LFS_CUDA_LAUNCH_CHECK(nullptr, "io.kmeans.assign_sh3_labels");
        }
    }

    std::tuple<Tensor, Tensor> kmeans_sh_swizzled(
        const Tensor& shN_swizzled,
        const int n_points,
        const int sh_coeffs,
        const int k,
        const int iterations, const bool fast_assignment) {
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
            LOG_ERROR("kmeans_sh_swizzled input has %zu floats but needs at least %lld for %d points",
                      shN_swizzled.numel(), static_cast<long long>(required_floats), n_points);
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
                return kmeans_swizzled_hierarchical_impl<45>(shN_swizzled, n_points, k, iterations, fast_assignment);
            }
            return kmeans_swizzled_bruteforce_impl<45>(shN_swizzled, n_points, k, iterations);
        default:
            LOG_ERROR("kmeans_sh_swizzled unsupported SH coefficient count %d (dims %d). Supported: 3, 8, 15",
                      sh_coeffs, n_dims);
            return {Tensor(), Tensor()};
        }
    }

    // ------------------------------------------------------------------------
    // e-means: EMA mini-batch k-means (arXiv:2611.09501). Each epoch visits a
    // fresh permutation of the points in floor(n / B) full batches. Batches
    // are assigned to the current centroids and blended in with a cosine-
    // annealed step: N <- (1 - a) N + a n_b, S <- (1 - a) S + a s_b, c = S / N.
    // Every five batches clusters below 10% of the mean count are revived by
    // splitting a donor drawn proportional to its count. The final centroids
    // are the mean of the points assigned during the last epoch. Large
    // palettes search candidates through a 256-super-cluster inverted file
    // whose buckets are rebuilt on device every batch, without host syncs.
    // ------------------------------------------------------------------------
    namespace {

        constexpr double EMEANS_PI = 3.14159265358979323846;
        constexpr double EMEANS_SCHEDULE_FLOOR = 0.1;
        constexpr float EMEANS_DEAD_FRACTION = 0.1f;
        constexpr int EMEANS_REVIVAL_PERIOD = 5;
        constexpr float EMEANS_SPLIT_DELTA = 0.0078125f; // BF16 machine epsilon (2^-7)
        constexpr float EMEANS_INITIAL_COUNT = 1e-10f;
        constexpr unsigned long long EMEANS_EMPTY_KEY = ~0ull;
        constexpr int EMEANS_IVF_MIN_K = 4096;

        struct EMeansSchedule {
            int batch_size = 0;
            int batches_per_epoch = 0;
            int total_updates = 0;
            double alpha0 = 0.0;
        };

        // Scaling laws of the paper (Section 3.3): the batch size is rounded to
        // a power of two (which keeps batches aligned to the 32-row swizzle
        // blocks) and the initial decay rate follows from k, B, n and T.
        EMeansSchedule emeans_schedule(const int n, const int k, const int epochs) {
            const double nd = static_cast<double>(n);
            const double kd = static_cast<double>(k);
            const double td = static_cast<double>(epochs);
            const double batch_coefficient = n > 2'000'000 ? 0.0044 : 0.035;
            const double raw_batch = batch_coefficient * std::pow(kd, 0.13) * std::pow(nd, 0.88) * std::pow(td, 0.81);
            const int log2_batch = std::clamp(static_cast<int>(std::lround(std::log2(std::max(raw_batch, 1.0)))),
                                              static_cast<int>(std::log2(lfs::core::kShReorderSize)), 30);
            EMeansSchedule schedule;
            schedule.batch_size = static_cast<int>(std::min<long long>(1LL << log2_batch, n));
            schedule.batches_per_epoch = std::max(1, n / schedule.batch_size);
            schedule.total_updates = schedule.batches_per_epoch * epochs;
            schedule.alpha0 = std::min(1.0, 0.824 * std::pow(kd, -0.46) *
                                                std::pow(static_cast<double>(schedule.batch_size), 0.90) *
                                                std::pow(nd, -0.45) * std::pow(td, -0.11));
            return schedule;
        }

        __device__ __forceinline__ std::uint32_t emeans_hash(std::uint32_t x) {
            // lowbias32: a bijection on uint32, so shuffle keys never tie.
            x ^= x >> 16;
            x *= 0x7feb352du;
            x ^= x >> 15;
            x *= 0x846ca68bu;
            x ^= x >> 16;
            return x;
        }

        __global__ void emeans_shuffle_keys_kernel(std::uint32_t* __restrict__ keys,
                                                   int* __restrict__ order,
                                                   const int n,
                                                   const std::uint32_t seed) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                keys[i] = emeans_hash(static_cast<std::uint32_t>(i) * 0x9e3779b9u + seed);
                order[i] = i;
            }
        }

        // Writes are linear in the destination swizzled layout; padded rows are zero.
        template <int SLOTS>
        __global__ void emeans_gather_permuted_kernel(const float4* __restrict__ src,
                                                      const int* __restrict__ order,
                                                      float4* __restrict__ dst,
                                                      const int n,
                                                      const int padded_n) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            constexpr int R = static_cast<int>(lfs::core::kShReorderSize);
            if (i >= padded_n * SLOTS)
                return;
            const int block = i / (SLOTS * R);
            const int rem = i - block * (SLOTS * R);
            const int slot = rem / R;
            const int row = block * R + (rem - slot * R);
            dst[i] = row < n ? src[shAt_device(static_cast<std::uint32_t>(order[row]),
                                               static_cast<std::uint32_t>(slot), SLOTS)]
                             : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        template <int N_DIMS>
        __global__ void emeans_init_centroids_kernel(const float4* __restrict__ shuffled,
                                                     float* __restrict__ centroids,
                                                     float* __restrict__ norms,
                                                     float* __restrict__ ema_counts,
                                                     const int k) {
            const int c = blockIdx.x * blockDim.x + threadIdx.x;
            if (c >= k)
                return;
            float norm = 0.0f;
#pragma unroll
            for (int d = 0; d < N_DIMS; ++d) {
                const float v = read_swizzled_sh_dim<N_DIMS>(shuffled, static_cast<std::uint32_t>(c),
                                                             static_cast<std::uint32_t>(d));
                centroids[static_cast<long long>(c) * N_DIMS + d] = v;
                norm = fmaf(v, v, norm);
            }
            norms[c] = norm;
            ema_counts[c] = EMEANS_INITIAL_COUNT;
        }

        __device__ __forceinline__ unsigned long long emeans_pack_key(const float dist, const int idx) {
            // Order-preserving float bits in the high word, index in the low word:
            // atomicMin keeps the smallest distance and, on ties, the lowest index.
            std::uint32_t bits = __float_as_uint(dist);
            bits = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
            return (static_cast<unsigned long long>(bits) << 32) | static_cast<std::uint32_t>(idx);
        }

        // Exact FP32 nearest-centroid search for one batch. Batches are small
        // relative to the GPU, so the centroid range is split across blockIdx.y
        // and the per-split winners are merged with a packed 64-bit atomicMin.
        template <int N_DIMS, int POINTS_PER_THREAD = ASSIGN_POINTS_PER_THREAD, int THREAD_COUNT = BLOCK_SIZE>
        __global__ void __launch_bounds__(THREAD_COUNT, 3)
            emeans_assign_split_kernel(
                const float4* __restrict__ shN,
                const float* __restrict__ centroids,
                const float* __restrict__ centroid_norms,
                unsigned long long* __restrict__ best_keys,
                const int n_points,
                const int k,
                const int centroids_per_split) {
            constexpr int POINT_TILE = THREAD_COUNT / ASSIGN_CENTROID_GROUPS * POINTS_PER_THREAD;
            __shared__ float shared_points[POINT_TILE][ASSIGN_SHARED_K_STRIDE];
            __shared__ float shared_centroids[ASSIGN_CENTROID_TILE][ASSIGN_SHARED_K_STRIDE];
            __shared__ float shared_centroid_norms[ASSIGN_CENTROID_TILE];

            const int tid = threadIdx.x;
            const int point_row = (tid / ASSIGN_CENTROID_GROUPS) * POINTS_PER_THREAD;
            const int centroid_col = (tid % ASSIGN_CENTROID_GROUPS) * ASSIGN_CENTROIDS_PER_THREAD;
            const int point_start = blockIdx.x * POINT_TILE;
            const int centroid_begin = blockIdx.y * centroids_per_split;
            const int centroid_end = min(k, centroid_begin + centroids_per_split);
            if (centroid_begin >= centroid_end)
                return;

            constexpr int point_slots_count = (N_DIMS + 3) / 4;
            for (int i = tid; i < POINT_TILE * point_slots_count; i += THREAD_COUNT) {
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

            float best_dist[POINTS_PER_THREAD];
            int best_idx[POINTS_PER_THREAD];
#pragma unroll
            for (int p = 0; p < POINTS_PER_THREAD; ++p) {
                best_dist[p] = 1e30f;
                best_idx[p] = k;
            }

            for (int centroid_start = centroid_begin; centroid_start < centroid_end;
                 centroid_start += ASSIGN_CENTROID_TILE) {
                for (int i = tid; i < ASSIGN_CENTROID_TILE * N_DIMS; i += THREAD_COUNT) {
                    const int centroid = i / N_DIMS;
                    const int dim = i % N_DIMS;
                    const int global_centroid = centroid_start + centroid;
                    shared_centroids[centroid][dim] = global_centroid < centroid_end
                                                          ? centroids[static_cast<long long>(global_centroid) * N_DIMS + dim]
                                                          : 0.0f;
                }
                for (int i = tid; i < ASSIGN_CENTROID_TILE; i += THREAD_COUNT) {
                    const int global_centroid = centroid_start + i;
                    shared_centroid_norms[i] = global_centroid < centroid_end ? centroid_norms[global_centroid] : 0.0f;
                }
                __syncthreads();

                float dots[POINTS_PER_THREAD][ASSIGN_CENTROIDS_PER_THREAD] = {};
#pragma unroll
                for (int d = 0; d < N_DIMS; ++d) {
                    float point_values[POINTS_PER_THREAD];
                    float centroid_values[ASSIGN_CENTROIDS_PER_THREAD];
#pragma unroll
                    for (int p = 0; p < POINTS_PER_THREAD; ++p) {
                        point_values[p] = shared_points[point_row + p][d];
                    }
#pragma unroll
                    for (int c = 0; c < ASSIGN_CENTROIDS_PER_THREAD; ++c) {
                        centroid_values[c] = shared_centroids[centroid_col + c][d];
                    }
#pragma unroll
                    for (int p = 0; p < POINTS_PER_THREAD; ++p) {
#pragma unroll
                        for (int c = 0; c < ASSIGN_CENTROIDS_PER_THREAD; ++c) {
                            dots[p][c] = fmaf(point_values[p], centroid_values[c], dots[p][c]);
                        }
                    }
                }

#pragma unroll
                for (int p = 0; p < POINTS_PER_THREAD; ++p) {
#pragma unroll
                    for (int c = 0; c < ASSIGN_CENTROIDS_PER_THREAD; ++c) {
                        const int global_centroid = centroid_start + centroid_col + c;
                        if (global_centroid < centroid_end) {
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
            for (int p = 0; p < POINTS_PER_THREAD; ++p) {
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
                    if (global_point < n_points && best_idx[p] < k) {
                        atomicMin(&best_keys[global_point], emeans_pack_key(best_dist[p], best_idx[p]));
                    }
                }
            }
        }

        // Accumulates batch sums/counts. In the last epoch the batch labels are
        // also scattered to source order as hints for the exact final search.
        template <int N_DIMS, bool FROM_KEYS>
        __global__ void emeans_accumulate_kernel(
            const float4* __restrict__ shN,
            const unsigned long long* __restrict__ best_keys,
            const int* __restrict__ batch_labels,
            const int* __restrict__ batch_order,
            int* __restrict__ source_labels,
            float* __restrict__ sums,
            int* __restrict__ counts,
            const int n_points,
            const int k) {
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid >= n_points)
                return;
            int label;
            if constexpr (FROM_KEYS) {
                const unsigned long long key = best_keys[tid];
                label = key == EMEANS_EMPTY_KEY ? -1 : static_cast<int>(key & 0xffffffffull);
            } else {
                label = batch_labels[tid];
            }
            if (label < 0 || label >= k)
                return;
            if (source_labels)
                source_labels[batch_order[tid]] = label;

            constexpr int point_slots_count = (N_DIMS + 3) / 4;
            float4 point_slots[point_slots_count];
#pragma unroll
            for (int slot = 0; slot < point_slots_count; ++slot) {
                point_slots[slot] = shN[shAt_device(
                    static_cast<std::uint32_t>(tid), static_cast<std::uint32_t>(slot), point_slots_count)];
            }
#pragma unroll
            for (int d = 0; d < N_DIMS; ++d) {
                atomicAdd(&sums[static_cast<long long>(label) * N_DIMS + d],
                          read_swizzled_sh_slot<N_DIMS>(point_slots, static_cast<std::uint32_t>(d)));
            }
            atomicAdd(&counts[label], 1);
        }

        // (c, N) form of the EMA update, equivalent to updating S = N c:
        // N' = (1 - a) N + a n_b and c' = ((1 - a) N c + a s_b) / N'.
        // One 16- or 32-lane group per cluster keeps the row accesses coalesced.
        template <int N_DIMS>
        constexpr int emeans_update_group() {
            return N_DIMS <= 16 ? 16 : 32;
        }

        template <int N_DIMS>
        __global__ void emeans_update_kernel(
            float* __restrict__ centroids,
            float* __restrict__ norms,
            float* __restrict__ ema_counts,
            float* __restrict__ sums,
            int* __restrict__ counts,
            float* __restrict__ epoch_sums,
            int* __restrict__ epoch_counts,
            const int k,
            const float alpha) {
            constexpr int G = emeans_update_group<N_DIMS>();
            const int global = blockIdx.x * blockDim.x + threadIdx.x;
            const int c = global / G;
            const int lane = global % G;
            if (c >= k)
                return;
            const unsigned mask = G == 32 ? 0xffffffffu : (0xffffu << (threadIdx.x & 16));
            const int batch_count = counts[c];
            const float previous = __shfl_sync(mask, lane == 0 ? ema_counts[c] : 0.0f, 0, G);
            const float decayed = (1.0f - alpha) * previous;
            const float updated = fmaf(alpha, static_cast<float>(batch_count), decayed);
            if (batch_count == 0) {
                if (lane == 0)
                    ema_counts[c] = updated;
                return;
            }

            const float inv_updated = 1.0f / updated;
            float norm = 0.0f;
#pragma unroll
            for (int d = lane; d < N_DIMS; d += G) {
                const long long idx = static_cast<long long>(c) * N_DIMS + d;
                const float s = sums[idx];
                const float v = fmaf(decayed, centroids[idx], alpha * s) * inv_updated;
                centroids[idx] = v;
                norm = fmaf(v, v, norm);
                if (epoch_sums)
                    epoch_sums[idx] += s;
                sums[idx] = 0.0f;
            }
#pragma unroll
            for (int offset = G / 2; offset > 0; offset >>= 1)
                norm += __shfl_xor_sync(mask, norm, offset, G);
            if (lane == 0) {
                norms[c] = norm;
                ema_counts[c] = updated;
                if (epoch_counts)
                    epoch_counts[c] += batch_count;
                counts[c] = 0;
            }
        }

        template <int N_DIMS>
        __global__ void emeans_split_kernel(
            const int2* __restrict__ pairs,
            const int pair_count,
            float* __restrict__ centroids,
            float* __restrict__ norms,
            float* __restrict__ ema_counts) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= pair_count)
                return;
            const int dead = pairs[i].x;
            const int donor = pairs[i].y;
            float* dead_centroid = centroids + static_cast<long long>(dead) * N_DIMS;
            float* donor_centroid = centroids + static_cast<long long>(donor) * N_DIMS;
            float dead_norm = 0.0f;
            float donor_norm = 0.0f;
#pragma unroll
            for (int d = 0; d < N_DIMS; ++d) {
                const float v = donor_centroid[d];
                const float sign = (d & 1) ? -1.0f : 1.0f;
                const float revived = v * (1.0f + sign * EMEANS_SPLIT_DELTA);
                const float kept = v * (1.0f - sign * EMEANS_SPLIT_DELTA);
                dead_centroid[d] = revived;
                donor_centroid[d] = kept;
                dead_norm = fmaf(revived, revived, dead_norm);
                donor_norm = fmaf(kept, kept, donor_norm);
            }
            norms[dead] = dead_norm;
            norms[donor] = donor_norm;
            const float half_count = 0.5f * ema_counts[donor];
            ema_counts[dead] = half_count;
            ema_counts[donor] = half_count;
        }

        // Replaces each centroid with the mean of its points; empty clusters keep their centroid.
        template <int N_DIMS>
        __global__ void emeans_mean_kernel(
            float* __restrict__ centroids,
            const float* __restrict__ sums,
            const int* __restrict__ counts,
            const int k) {
            const int c = blockIdx.x * blockDim.x + threadIdx.x;
            if (c >= k)
                return;
            const int count = counts[c];
            if (count <= 0)
                return;
            const float inv = 1.0f / static_cast<float>(count);
#pragma unroll
            for (int d = 0; d < N_DIMS; ++d) {
                centroids[static_cast<long long>(c) * N_DIMS + d] =
                    sums[static_cast<long long>(c) * N_DIMS + d] * inv;
            }
        }

        template <int N_DIMS>
        __global__ void palette_squared_error_kernel(
            const float4* __restrict__ shN,
            const float* __restrict__ centroids,
            const int* __restrict__ labels,
            const int n_points,
            const int k,
            double* __restrict__ total) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            float err = 0.0f;
            if (i < n_points) {
                const int label = labels[i];
                if (label >= 0 && label < k) {
#pragma unroll
                    for (int d = 0; d < N_DIMS; ++d) {
                        const float diff = read_swizzled_sh_dim<N_DIMS>(shN, static_cast<std::uint32_t>(i),
                                                                        static_cast<std::uint32_t>(d)) -
                                           centroids[static_cast<long long>(label) * N_DIMS + d];
                        err = fmaf(diff, diff, err);
                    }
                }
            }
            for (int offset = 16; offset > 0; offset >>= 1)
                err += __shfl_down_sync(0xffffffffu, err, offset);
            __shared__ float warp_sums[BLOCK_SIZE / 32];
            if ((threadIdx.x & 31) == 0)
                warp_sums[threadIdx.x / 32] = err;
            __syncthreads();
            if (threadIdx.x < 32) {
                float block_sum = threadIdx.x < BLOCK_SIZE / 32 ? warp_sums[threadIdx.x] : 0.0f;
                for (int offset = 16; offset > 0; offset >>= 1)
                    block_sum += __shfl_down_sync(0xffffffffu, block_sum, offset);
                if (threadIdx.x == 0)
                    atomicAdd(total, static_cast<double>(block_sum));
            }
        }

        int emeans_device_sm_count() {
            static const int sm_count = [] {
                int device = 0;
                int count = 0;
                if (cudaGetDevice(&device) != cudaSuccess ||
                    cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
                    return 64;
                }
                return std::max(1, count);
            }();
            return sm_count;
        }

        // Counting sort of ids by a [0, NUM_SUPER_CLUSTERS) bucket: histogram,
        // exclusive scan, then an atomic-cursor scatter. Bucket order within a
        // group is irrelevant because candidate ties resolve to the lowest id.
        __device__ __forceinline__ int emeans_bucket(const int membership) {
            return min(max(membership, 0), NUM_SUPER_CLUSTERS - 1);
        }

        __global__ void emeans_bucket_histogram_kernel(const int* __restrict__ membership,
                                                       const int n,
                                                       int* __restrict__ bucket_counts) {
            __shared__ int local[NUM_SUPER_CLUSTERS];
            for (int i = threadIdx.x; i < NUM_SUPER_CLUSTERS; i += blockDim.x)
                local[i] = 0;
            __syncthreads();
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
                atomicAdd(&local[emeans_bucket(membership[i])], 1);
            __syncthreads();
            for (int i = threadIdx.x; i < NUM_SUPER_CLUSTERS; i += blockDim.x) {
                if (local[i] != 0)
                    atomicAdd(&bucket_counts[i], local[i]);
            }
        }

        __global__ void emeans_bucket_scatter_kernel(const int* __restrict__ membership,
                                                     const int n,
                                                     int* __restrict__ cursor,
                                                     int* __restrict__ sorted_ids) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n)
                sorted_ids[atomicAdd(&cursor[emeans_bucket(membership[i])], 1)] = i;
        }

        // Exclusive offsets [NUM_SUPER_CLUSTERS + 1] and scatter cursors from bucket counts.
        __global__ void __launch_bounds__(NUM_SUPER_CLUSTERS)
            emeans_bucket_offsets_kernel(const int* __restrict__ bucket_counts,
                                         int* __restrict__ offsets,
                                         int* __restrict__ cursor) {
            using Scan = cub::BlockScan<int, NUM_SUPER_CLUSTERS>;
            __shared__ typename Scan::TempStorage scan_storage;
            const int g = threadIdx.x;
            const int count = bucket_counts[g];
            int offset = 0;
            int total = 0;
            Scan(scan_storage).ExclusiveSum(count, offset, total);
            offsets[g] = offset;
            cursor[g] = offset;
            if (g == 0)
                offsets[NUM_SUPER_CLUSTERS] = total;
        }

        // Per-batch launch plan for assign_grouped_swizzled_kernel, computed on
        // the device: point groups, per-group tile task offsets and candidate
        // counts from the current palette buckets, with the same empty-candidate
        // fallback as build_group_candidate_supers_gpu.
        __global__ void __launch_bounds__(NUM_SUPER_CLUSTERS)
            emeans_group_plan_kernel(const int* __restrict__ point_counts,
                                     const int* __restrict__ super_offsets,
                                     const int* __restrict__ nearest_supers,
                                     int* __restrict__ group_offsets,
                                     int* __restrict__ cursor,
                                     int* __restrict__ group_task_offsets,
                                     int* __restrict__ group_candidate_offsets,
                                     int* __restrict__ group_candidate_supers) {
            using Scan = cub::BlockScan<int, NUM_SUPER_CLUSTERS>;
            __shared__ typename Scan::TempStorage scan_storage;
            __shared__ int first_nonempty_super;
            const int g = threadIdx.x;
            if (g == 0)
                first_nonempty_super = NUM_SUPER_CLUSTERS - 1;
            __syncthreads();
            if (super_offsets[g + 1] > super_offsets[g])
                atomicMin(&first_nonempty_super, g);
            __syncthreads();

            int candidate_count = 0;
#pragma unroll
            for (int rank = 0; rank < NUM_NEAREST_SUPERS; ++rank) {
                const int super_idx = nearest_supers[g * NUM_NEAREST_SUPERS + rank];
                group_candidate_supers[g * NUM_NEAREST_SUPERS + rank] = super_idx;
                candidate_count += super_offsets[super_idx + 1] - super_offsets[super_idx];
            }
            if (candidate_count == 0) {
                const int fallback = first_nonempty_super;
                group_candidate_supers[g * NUM_NEAREST_SUPERS + NUM_NEAREST_SUPERS - 1] = fallback;
                candidate_count = super_offsets[fallback + 1] - super_offsets[fallback];
            }

            const int point_count = point_counts[g];
            int value = 0;
            int total = 0;
            Scan(scan_storage).ExclusiveSum(point_count, value, total);
            group_offsets[g] = value;
            cursor[g] = value;
            if (g == 0)
                group_offsets[NUM_SUPER_CLUSTERS] = total;
            __syncthreads();
            Scan(scan_storage).ExclusiveSum((point_count + ASSIGN_POINT_TILE - 1) / ASSIGN_POINT_TILE, value, total);
            group_task_offsets[g] = value;
            if (g == 0)
                group_task_offsets[NUM_SUPER_CLUSTERS] = total;
            __syncthreads();
            Scan(scan_storage).ExclusiveSum(candidate_count, value, total);
            group_candidate_offsets[g] = value;
            if (g == 0)
                group_candidate_offsets[NUM_SUPER_CLUSTERS] = total;
        }

        // IVF-style candidate search (Section 3.4 / Appendix M.2): 256 meta-
        // centroids index the palette, and each batch point is searched only in
        // the members of the four meta-centroids nearest to its own, the same
        // approximation the hierarchical SH3 k-means uses between exact passes.
        // The meta-centroids are fit once to the initial palette (a uniform
        // sample of the data); palette membership is rebuilt before every batch
        // so moved and revived centroids are found. Everything stays on the
        // device without host synchronization.
        template <int N_DIMS>
        struct EMeansIvfIndex {
            Tensor super_centroids;
            Tensor super_norms;
            Tensor super_membership;
            Tensor super_sums;
            Tensor super_counts;
            Tensor super_offsets;
            Tensor super_indices;
            Tensor nearest_supers;
            Tensor group_candidate_offsets;
            Tensor group_candidate_supers;
            Tensor point_super_membership;
            Tensor sorted_point_idx;
            Tensor group_offsets;
            Tensor group_task_offsets;
            Tensor bucket_counts;
            Tensor bucket_cursor;
            int histogram_blocks = 1;

            void initialize(const float* d_centroids, const int k, const int batch_size, const unsigned int seed) {
                const auto S = static_cast<size_t>(NUM_SUPER_CLUSTERS);
                super_centroids = Tensor::zeros({S, static_cast<size_t>(N_DIMS)}, Device::CUDA, DataType::Float32);
                super_norms = Tensor::zeros({S}, Device::CUDA, DataType::Float32);
                super_membership = Tensor::zeros({static_cast<size_t>(k)}, Device::CUDA, DataType::Int32);
                super_sums = Tensor::zeros({S, static_cast<size_t>(N_DIMS)}, Device::CUDA, DataType::Float32);
                super_counts = Tensor::zeros({S}, Device::CUDA, DataType::Int32);
                super_offsets = Tensor::zeros({S + 1}, Device::CUDA, DataType::Int32);
                super_indices = Tensor::zeros({static_cast<size_t>(k)}, Device::CUDA, DataType::Int32);
                nearest_supers = Tensor::zeros({S, static_cast<size_t>(NUM_NEAREST_SUPERS)}, Device::CUDA, DataType::Int32);
                group_candidate_offsets = Tensor::zeros({S + 1}, Device::CUDA, DataType::Int32);
                group_candidate_supers = Tensor::zeros({S * NUM_NEAREST_SUPERS}, Device::CUDA, DataType::Int32);
                point_super_membership = Tensor::zeros({static_cast<size_t>(batch_size)}, Device::CUDA, DataType::Int32);
                sorted_point_idx = Tensor::zeros({static_cast<size_t>(batch_size)}, Device::CUDA, DataType::Int32);
                group_offsets = Tensor::zeros({S + 1}, Device::CUDA, DataType::Int32);
                group_task_offsets = Tensor::zeros({S + 1}, Device::CUDA, DataType::Int32);
                bucket_counts = Tensor::zeros({S}, Device::CUDA, DataType::Int32);
                bucket_cursor = Tensor::zeros({S}, Device::CUDA, DataType::Int32);
                histogram_blocks = emeans_device_sm_count() * 2;

                auto perm = sample_unique_indices_gpu(k, NUM_SUPER_CLUSTERS, seed);
                const int grid_super = (NUM_SUPER_CLUSTERS + BLOCK_SIZE - 1) / BLOCK_SIZE;
                gather_centroids_kernel<N_DIMS><<<grid_super, BLOCK_SIZE>>>(
                    d_centroids, perm.ptr<int>(), super_centroids.ptr<float>(), NUM_SUPER_CLUSTERS);
                LFS_CUDA_LAUNCH_CHECK(nullptr, "io.emeans.gather_supers");
                refresh_supers(d_centroids, k, 5, 67890);
            }

            // Warm-started Lloyd steps on the super-clusters, then the fixed top-k super neighbors.
            void refresh_supers(const float* d_centroids, const int k, const int iterations, const unsigned int seed) {
                const int grid_super = (NUM_SUPER_CLUSTERS + BLOCK_SIZE - 1) / BLOCK_SIZE;
                for (int iter = 0; iter < iterations; ++iter) {
                    compute_centroid_norms_kernel<N_DIMS><<<grid_super, BLOCK_SIZE>>>(
                        super_centroids.ptr<float>(), super_norms.ptr<float>(), NUM_SUPER_CLUSTERS);
                    LFS_CUDA_LAUNCH_CHECK(nullptr, "io.emeans.compute_super_norms");
                    assign_palette(d_centroids, k);
                    super_sums.zero_();
                    super_counts.zero_();
                    accumulate_centroids_kernel<N_DIMS><<<(k + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                        d_centroids, super_membership.ptr<int>(), super_sums.ptr<float>(), super_counts.ptr<int>(), k);
                    LFS_CUDA_LAUNCH_CHECK(nullptr, "io.emeans.accumulate_supers");
                    finalize_centroids_kernel<N_DIMS><<<grid_super, BLOCK_SIZE>>>(
                        super_centroids.ptr<float>(), super_sums.ptr<float>(), super_counts.ptr<int>(),
                        d_centroids, NUM_SUPER_CLUSTERS, k, seed + static_cast<unsigned int>(iter) * 12345u);
                    LFS_CUDA_LAUNCH_CHECK(nullptr, "io.emeans.finalize_supers");
                }
                compute_centroid_norms_kernel<N_DIMS><<<grid_super, BLOCK_SIZE>>>(
                    super_centroids.ptr<float>(), super_norms.ptr<float>(), NUM_SUPER_CLUSTERS);
                LFS_CUDA_LAUNCH_CHECK(nullptr, "io.emeans.compute_super_norms");
                assign_nearest_top_supers_kernel<N_DIMS><<<1, BLOCK_SIZE>>>(
                    super_centroids.ptr<float>(), super_centroids.ptr<float>(), super_norms.ptr<float>(),
                    nearest_supers.ptr<int>(), NUM_SUPER_CLUSTERS, NUM_SUPER_CLUSTERS);
                LFS_CUDA_LAUNCH_CHECK(nullptr, "io.emeans.assign_nearest_super_neighbors");
            }

            void rebuild_palette_buckets(const float* d_centroids, const int k) {
                assign_palette(d_centroids, k);
                bucket(super_membership.ptr<int>(), k, super_offsets.ptr<int>(), super_indices.ptr<int>(), nullptr);
            }

            void assign(const float4* d_batch, const float* d_centroids, const float* d_norms,
                        int* d_labels, const int batch_size, const int k) {
                assign_nearest_swizzled_bruteforce_kernel<N_DIMS><<<(batch_size + ASSIGN_POINT_TILE - 1) / ASSIGN_POINT_TILE, BLOCK_SIZE>>>(
                    d_batch, super_centroids.ptr<float>(), super_norms.ptr<float>(),
                    point_super_membership.ptr<int>(), batch_size, NUM_SUPER_CLUSTERS);
                LFS_CUDA_LAUNCH_CHECK(nullptr, "io.emeans.assign_points_to_supers");
                bucket(point_super_membership.ptr<int>(), batch_size, group_offsets.ptr<int>(),
                       sorted_point_idx.ptr<int>(), &EMeansIvfIndex::plan_groups);

                // Upper bound on tile tasks; surplus blocks find an empty tile and exit.
                const int max_tasks = (batch_size + ASSIGN_POINT_TILE - 1) / ASSIGN_POINT_TILE + NUM_SUPER_CLUSTERS;
                launch_assign_grouped<N_DIMS>(
                    max_tasks, d_batch, d_centroids, d_norms, sorted_point_idx.ptr<int>(),
                    group_offsets.ptr<int>(), group_task_offsets.ptr<int>(), group_candidate_offsets.ptr<int>(),
                    group_candidate_supers.ptr<int>(), super_offsets.ptr<int>(), super_indices.ptr<int>(),
                    d_labels, batch_size, k);
                LFS_CUDA_LAUNCH_CHECK(nullptr, "io.emeans.assign_grouped_candidates");
            }

        private:
            void assign_palette(const float* d_centroids, const int k) {
                assign_nearest_bruteforce_kernel<N_DIMS><<<(k + ASSIGN_POINT_TILE - 1) / ASSIGN_POINT_TILE, BLOCK_SIZE>>>(
                    d_centroids, super_centroids.ptr<float>(), super_norms.ptr<float>(),
                    super_membership.ptr<int>(), k, NUM_SUPER_CLUSTERS);
                LFS_CUDA_LAUNCH_CHECK(nullptr, "io.emeans.assign_palette_to_supers");
            }

            void plan_groups() {
                emeans_group_plan_kernel<<<1, NUM_SUPER_CLUSTERS>>>(
                    bucket_counts.ptr<int>(), super_offsets.ptr<int>(), nearest_supers.ptr<int>(),
                    group_offsets.ptr<int>(), bucket_cursor.ptr<int>(), group_task_offsets.ptr<int>(),
                    group_candidate_offsets.ptr<int>(), group_candidate_supers.ptr<int>());
                LFS_CUDA_LAUNCH_CHECK(nullptr, "io.emeans.group_plan");
            }

            void bucket(const int* d_membership, const int n, int* d_offsets, int* d_sorted,
                        void (EMeansIvfIndex::*plan)()) {
                LFS_CUDA_CHECK(cudaMemsetAsync(bucket_counts.ptr<int>(), 0, NUM_SUPER_CLUSTERS * sizeof(int)));
                emeans_bucket_histogram_kernel<<<std::min(histogram_blocks, (n + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE>>>(
                    d_membership, n, bucket_counts.ptr<int>());
                LFS_CUDA_LAUNCH_CHECK(nullptr, "io.emeans.bucket_histogram");
                if (plan) {
                    (this->*plan)();
                } else {
                    emeans_bucket_offsets_kernel<<<1, NUM_SUPER_CLUSTERS>>>(
                        bucket_counts.ptr<int>(), d_offsets, bucket_cursor.ptr<int>());
                    LFS_CUDA_LAUNCH_CHECK(nullptr, "io.emeans.bucket_offsets");
                }
                emeans_bucket_scatter_kernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                    d_membership, n, bucket_cursor.ptr<int>(), d_sorted);
                LFS_CUDA_LAUNCH_CHECK(nullptr, "io.emeans.bucket_scatter");
            }
        };

        template <int N_DIMS>
        std::tuple<Tensor, Tensor> emeans_swizzled_impl(
            const Tensor& shN_swizzled,
            const int n,
            const int k,
            const int epochs,
            const bool fast_assignment) {
            auto shN_gpu = as_cuda_contiguous(shN_swizzled);
            const auto* d_shN = reinterpret_cast<const float4*>(shN_gpu.ptr<float>());
            constexpr int SLOTS = (N_DIMS + 3) / 4;

            if (n <= k) {
                auto centroids = Tensor::zeros({static_cast<size_t>(n), static_cast<size_t>(N_DIMS)},
                                               Device::CUDA, DataType::Float32);
                auto labels = Tensor::arange(n).to(DataType::Int32).cuda();
                const auto ticket = ::lfs::core::cuda_record_range(nullptr, "io.emeans.gather_points");
                gather_swizzled_centroids_kernel<N_DIMS><<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                    d_shN, labels.ptr<int>(), centroids.ptr<float>(), n);
                LFS_CUDA_LAUNCH_CHECK(nullptr, "io.emeans.gather_swizzled_centroids");
                LFS_CUDA_AWAIT(ticket, cudaDeviceSynchronize(), "io.emeans.gather_points_sync");
                return {centroids, labels};
            }

            const auto schedule = emeans_schedule(n, k, epochs);
            const int B = schedule.batch_size;
            // Large palettes use the inverted-file candidate search during training; the
            // final labels always come from an exact assignment.
            const bool use_ivf = k >= EMEANS_IVF_MIN_K;

            const size_t padded_n = lfs::core::sh_swizzled_padded_n(static_cast<size_t>(n));
            auto shuffled = Tensor::empty({padded_n * SLOTS * 4}, Device::CUDA, DataType::Float32);
            auto* d_shuffled = reinterpret_cast<float4*>(shuffled.ptr<float>());
            auto shuffle_keys = Tensor::empty({static_cast<size_t>(n)}, Device::CUDA, DataType::Int32);
            auto sorted_keys = Tensor::empty({static_cast<size_t>(n)}, Device::CUDA, DataType::Int32);
            auto identity = Tensor::empty({static_cast<size_t>(n)}, Device::CUDA, DataType::Int32);
            auto order = Tensor::empty({static_cast<size_t>(n)}, Device::CUDA, DataType::Int32);
            size_t sort_temp_bytes = 0;
            LFS_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
                nullptr, sort_temp_bytes, shuffle_keys.ptr<std::uint32_t>(), sorted_keys.ptr<std::uint32_t>(),
                identity.ptr<int>(), order.ptr<int>(), n));
            auto sort_temp = Tensor::empty({std::max<size_t>(sort_temp_bytes, 1)}, Device::CUDA, DataType::UInt8);

            auto centroids = Tensor::empty({static_cast<size_t>(k), static_cast<size_t>(N_DIMS)},
                                           Device::CUDA, DataType::Float32);
            auto norms = Tensor::empty({static_cast<size_t>(k)}, Device::CUDA, DataType::Float32);
            auto ema_counts = Tensor::empty({static_cast<size_t>(k)}, Device::CUDA, DataType::Float32);
            auto sums = Tensor::zeros({static_cast<size_t>(k), static_cast<size_t>(N_DIMS)}, Device::CUDA, DataType::Float32);
            auto counts = Tensor::zeros({static_cast<size_t>(k)}, Device::CUDA, DataType::Int32);
            auto epoch_sums = Tensor::zeros({static_cast<size_t>(k), static_cast<size_t>(N_DIMS)}, Device::CUDA, DataType::Float32);
            auto epoch_counts = Tensor::zeros({static_cast<size_t>(k)}, Device::CUDA, DataType::Int32);
            auto labels = Tensor::empty({static_cast<size_t>(n)}, Device::CUDA, DataType::Int32);
            LFS_CUDA_CHECK(cudaMemset(labels.ptr<int>(), 0xff, static_cast<size_t>(n) * sizeof(int)));
            auto best_keys = Tensor::empty({static_cast<size_t>(B)}, Device::CUDA, DataType::Int64);
            auto batch_labels = Tensor::empty({static_cast<size_t>(B)}, Device::CUDA, DataType::Int32);
            auto* d_centroids = centroids.ptr<float>();
            auto* d_norms = norms.ptr<float>();
            auto* d_ema_counts = ema_counts.ptr<float>();
            auto* d_best_keys = reinterpret_cast<unsigned long long*>(best_keys.ptr<int64_t>());

            std::random_device rd;
            std::mt19937 rng(rd());
            const std::uint32_t shuffle_seed = rng();
            const int grid_k = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const int grid_n = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const int grid_b = (B + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const int update_blocks = static_cast<int>(
                (static_cast<long long>(k) * emeans_update_group<N_DIMS>() + BLOCK_SIZE - 1) / BLOCK_SIZE);

            auto shuffle = [&](const int epoch) {
                emeans_shuffle_keys_kernel<<<grid_n, BLOCK_SIZE>>>(
                    shuffle_keys.ptr<std::uint32_t>(), identity.ptr<int>(), n,
                    shuffle_seed + static_cast<std::uint32_t>(epoch) * 0x85ebca6bu);
                LFS_CUDA_LAUNCH_CHECK(nullptr, "io.emeans.shuffle_keys");
                LFS_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
                    sort_temp.ptr<std::uint8_t>(), sort_temp_bytes, shuffle_keys.ptr<std::uint32_t>(),
                    sorted_keys.ptr<std::uint32_t>(), identity.ptr<int>(), order.ptr<int>(), n));
                const int total = static_cast<int>(padded_n) * SLOTS;
                emeans_gather_permuted_kernel<SLOTS><<<(total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                    d_shN, order.ptr<int>(), d_shuffled, n, static_cast<int>(padded_n));
                LFS_CUDA_LAUNCH_CHECK(nullptr, "io.emeans.gather_permuted");
            };

            const auto ticket = ::lfs::core::cuda_record_range(nullptr, "io.emeans.train");
            shuffle(0);
            // RandomChoice initialization: the first k rows of a uniform permutation.
            emeans_init_centroids_kernel<N_DIMS><<<grid_k, BLOCK_SIZE>>>(
                d_shuffled, d_centroids, d_norms, d_ema_counts, k);
            LFS_CUDA_LAUNCH_CHECK(nullptr, "io.emeans.init_centroids");

            std::optional<EMeansIvfIndex<N_DIMS>> ivf;
            if (use_ivf) {
                ivf.emplace();
                ivf->initialize(d_centroids, k, B, rng());
            }

            const int point_blocks = (B + ASSIGN_POINT_TILE - 1) / ASSIGN_POINT_TILE;
            const int centroid_tiles = (k + ASSIGN_CENTROID_TILE - 1) / ASSIGN_CENTROID_TILE;
            const int target_blocks = emeans_device_sm_count() * 3 * 2;
            const int splits = std::clamp((target_blocks + point_blocks - 1) / point_blocks, 1, centroid_tiles);
            const int centroids_per_split = (centroid_tiles + splits - 1) / splits * ASSIGN_CENTROID_TILE;

            std::vector<float> host_counts(static_cast<size_t>(k));
            std::vector<int> dead;
            std::vector<std::pair<float, int>> donor_keys;
            std::vector<int2> pairs;
            auto revival_pairs = Tensor::empty({static_cast<size_t>(2 * k)}, Device::CUDA, DataType::Int32);
            std::uniform_real_distribution<float> uniform(std::numeric_limits<float>::min(), 1.0f);
            long long revived_total = 0;

            auto revive = [&]() {
                LFS_CUDA_CHECK(cudaMemcpy(host_counts.data(), d_ema_counts,
                                          static_cast<size_t>(k) * sizeof(float), cudaMemcpyDeviceToHost));
                double total = 0.0;
                for (const float c : host_counts)
                    total += c;
                const float threshold = static_cast<float>(EMEANS_DEAD_FRACTION * total / k);
                dead.clear();
                donor_keys.clear();
                for (int c = 0; c < k; ++c) {
                    if (host_counts[static_cast<size_t>(c)] < threshold) {
                        dead.push_back(c);
                    } else {
                        // Efraimidis-Spirakis keys: the largest log(u) / w are a
                        // count-weighted sample without replacement.
                        donor_keys.emplace_back(std::log(uniform(rng)) / host_counts[static_cast<size_t>(c)], c);
                    }
                }
                const size_t pair_count = std::min(dead.size(), donor_keys.size());
                if (pair_count == 0)
                    return;
                if (pair_count < dead.size())
                    std::shuffle(dead.begin(), dead.end(), rng);
                std::nth_element(donor_keys.begin(), donor_keys.begin() + static_cast<std::ptrdiff_t>(pair_count - 1),
                                 donor_keys.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
                pairs.resize(pair_count);
                for (size_t i = 0; i < pair_count; ++i)
                    pairs[i] = make_int2(dead[i], donor_keys[i].second);
                LFS_CUDA_CHECK(cudaMemcpy(revival_pairs.ptr<int>(), pairs.data(), pair_count * sizeof(int2),
                                          cudaMemcpyHostToDevice));
                emeans_split_kernel<N_DIMS><<<(static_cast<int>(pair_count) + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                    reinterpret_cast<const int2*>(revival_pairs.ptr<int>()), static_cast<int>(pair_count),
                    d_centroids, d_norms, d_ema_counts);
                LFS_CUDA_LAUNCH_CHECK(nullptr, "io.emeans.split_donors");
                revived_total += static_cast<long long>(pair_count);
            };

            int step = 0;
            for (int epoch = 0; epoch < epochs; ++epoch) {
                if (epoch > 0)
                    shuffle(epoch);
                const bool last_epoch = epoch == epochs - 1;
                for (int batch = 0; batch < schedule.batches_per_epoch; ++batch, ++step) {
                    const double progress = static_cast<double>(step) / schedule.total_updates;
                    const float alpha = static_cast<float>(
                        schedule.alpha0 * (EMEANS_SCHEDULE_FLOOR + (1.0 - EMEANS_SCHEDULE_FLOOR) * 0.5 *
                                                                       (1.0 + std::cos(EMEANS_PI * progress))));
                    const size_t batch_offset = static_cast<size_t>(batch) * static_cast<size_t>(B);
                    const float4* d_batch = d_shuffled + batch_offset * SLOTS;
                    const int* d_batch_order = order.ptr<int>() + batch_offset;
                    int* d_source_labels = last_epoch ? labels.ptr<int>() : nullptr;

                    if (ivf) {
                        ivf->rebuild_palette_buckets(d_centroids, k);
                        ivf->assign(d_batch, d_centroids, d_norms, batch_labels.ptr<int>(), B, k);
                        emeans_accumulate_kernel<N_DIMS, false><<<grid_b, BLOCK_SIZE>>>(
                            d_batch, nullptr, batch_labels.ptr<int>(), d_batch_order, d_source_labels,
                            sums.ptr<float>(), counts.ptr<int>(), B, k);
                    } else {
                        LFS_CUDA_CHECK(cudaMemsetAsync(d_best_keys, 0xff, static_cast<size_t>(B) * sizeof(unsigned long long)));
                        emeans_assign_split_kernel<N_DIMS><<<dim3(point_blocks, splits), BLOCK_SIZE>>>(
                            d_batch, d_centroids, d_norms, d_best_keys, B, k, centroids_per_split);
                        LFS_CUDA_LAUNCH_CHECK(nullptr, "io.emeans.assign_split");
                        emeans_accumulate_kernel<N_DIMS, true><<<grid_b, BLOCK_SIZE>>>(
                            d_batch, d_best_keys, nullptr, d_batch_order, d_source_labels,
                            sums.ptr<float>(), counts.ptr<int>(), B, k);
                    }
                    LFS_CUDA_LAUNCH_CHECK(nullptr, "io.emeans.accumulate");

                    emeans_update_kernel<N_DIMS><<<update_blocks, BLOCK_SIZE>>>(
                        d_centroids, d_norms, d_ema_counts, sums.ptr<float>(), counts.ptr<int>(),
                        last_epoch ? epoch_sums.ptr<float>() : nullptr,
                        last_epoch ? epoch_counts.ptr<int>() : nullptr, k, alpha);
                    LFS_CUDA_LAUNCH_CHECK(nullptr, "io.emeans.update");

                    if ((step + 1) % EMEANS_REVIVAL_PERIOD == 0)
                        revive();
                }
            }

            // Last-epoch-mean finalization, then one exact full assignment and
            // refit, mirroring the final pass of the k-means path.
            emeans_mean_kernel<N_DIMS><<<grid_k, BLOCK_SIZE>>>(
                d_centroids, epoch_sums.ptr<float>(), epoch_counts.ptr<int>(), k);
            LFS_CUDA_LAUNCH_CHECK(nullptr, "io.emeans.last_epoch_mean");
            compute_centroid_norms_kernel<N_DIMS><<<grid_k, BLOCK_SIZE>>>(d_centroids, d_norms, k);
            LFS_CUDA_LAUNCH_CHECK(nullptr, "io.emeans.compute_centroid_norms");
            if constexpr (N_DIMS == 45) {
                assign_sh3_labels(shN_gpu, centroids, norms, labels, fast_assignment, true);
            } else {
                assign_nearest_swizzled_bruteforce_kernel<N_DIMS><<<(n + ASSIGN_POINT_TILE - 1) / ASSIGN_POINT_TILE, BLOCK_SIZE>>>(
                    d_shN, d_centroids, d_norms, labels.ptr<int>(), n, k);
                LFS_CUDA_LAUNCH_CHECK(nullptr, "io.emeans.final_assign");
            }
            sums.zero_();
            counts.zero_();
            accumulate_swizzled_centroids_kernel<N_DIMS><<<grid_n, BLOCK_SIZE>>>(
                d_shN, labels.ptr<int>(), sums.ptr<float>(), counts.ptr<int>(), n);
            LFS_CUDA_LAUNCH_CHECK(nullptr, "io.emeans.final_accumulate");
            emeans_mean_kernel<N_DIMS><<<grid_k, BLOCK_SIZE>>>(d_centroids, sums.ptr<float>(), counts.ptr<int>(), k);
            LFS_CUDA_LAUNCH_CHECK(nullptr, "io.emeans.final_refit");
            LFS_CUDA_AWAIT(ticket, cudaDeviceSynchronize(), "io.emeans.train_sync");

            LOG_DEBUG("SH e-means: n=%d k=%d dims=%d epochs=%d batch=%d batches_per_epoch=%d updates=%d "
                      "alpha0=%.4f assignment=%s splits=%d revived=%lld",
                      n, k, N_DIMS, epochs, B, schedule.batches_per_epoch, schedule.total_updates,
                      schedule.alpha0, use_ivf ? "ivf" : "exact", use_ivf ? 0 : splits, revived_total);
            return {centroids, labels};
        }

        template <int N_DIMS>
        double palette_mse_impl(const Tensor& shN_swizzled, const int n, const Tensor& centroids, const Tensor& labels) {
            auto shN_gpu = as_cuda_contiguous(shN_swizzled);
            auto centroids_gpu = as_cuda_contiguous(centroids);
            auto labels_gpu = as_cuda_contiguous(labels);
            double* d_total = nullptr;
            LFS_CUDA_CHECK(cudaMalloc(&d_total, sizeof(double)));
            LFS_CUDA_CHECK(cudaMemset(d_total, 0, sizeof(double)));
            palette_squared_error_kernel<N_DIMS><<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                reinterpret_cast<const float4*>(shN_gpu.ptr<float>()), centroids_gpu.ptr<float>(),
                labels_gpu.ptr<int>(), n, static_cast<int>(centroids_gpu.size(0)), d_total);
            LFS_CUDA_LAUNCH_CHECK(nullptr, "io.kmeans.palette_squared_error");
            double total = 0.0;
            LFS_CUDA_CHECK(cudaMemcpy(&total, d_total, sizeof(double), cudaMemcpyDeviceToHost));
            LFS_CUDA_CHECK(cudaFree(d_total));
            return total / (static_cast<double>(n) * N_DIMS);
        }

    } // anonymous namespace

    std::tuple<Tensor, Tensor> emeans_sh_swizzled(
        const Tensor& shN_swizzled,
        const int n_points,
        const int sh_coeffs,
        const int k,
        const int epochs,
        const bool fast_assignment) {
        if (!shN_swizzled.is_valid() || shN_swizzled.ndim() != 1 ||
            shN_swizzled.dtype() != DataType::Float32) {
            LOG_ERROR("emeans_sh_swizzled expects a 1D float32 swizzled shN tensor");
            return {Tensor(), Tensor()};
        }
        if (n_points <= 0 || k <= 0 || epochs <= 0) {
            LOG_ERROR("emeans_sh_swizzled expects positive n_points, k and epochs");
            return {Tensor(), Tensor()};
        }
        const auto required_floats = static_cast<int64_t>(
            lfs::core::sh_swizzled_float_count(static_cast<size_t>(n_points),
                                               static_cast<std::uint32_t>(sh_coeffs)));
        if (shN_swizzled.numel() < required_floats) {
            LOG_ERROR("emeans_sh_swizzled input has %zu floats but needs at least %lld for %d points",
                      shN_swizzled.numel(), static_cast<long long>(required_floats), n_points);
            return {Tensor(), Tensor()};
        }

        switch (sh_coeffs * 3) {
        case 9:
            return emeans_swizzled_impl<9>(shN_swizzled, n_points, k, epochs, fast_assignment);
        case 24:
            return emeans_swizzled_impl<24>(shN_swizzled, n_points, k, epochs, fast_assignment);
        case 45:
            return emeans_swizzled_impl<45>(shN_swizzled, n_points, k, epochs, fast_assignment);
        default:
            LOG_ERROR("emeans_sh_swizzled unsupported SH coefficient count %d. Supported: 3, 8, 15", sh_coeffs);
            return {Tensor(), Tensor()};
        }
    }

    double sh_palette_mse(const Tensor& shN_swizzled, const int n_points, const int sh_coeffs,
                          const Tensor& centroids, const Tensor& labels) {
        if (!shN_swizzled.is_valid() || !centroids.is_valid() || !labels.is_valid() || n_points <= 0 ||
            centroids.ndim() != 2 || centroids.size(1) != static_cast<size_t>(sh_coeffs * 3) ||
            labels.numel() < static_cast<size_t>(n_points)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        switch (sh_coeffs * 3) {
        case 9:
            return palette_mse_impl<9>(shN_swizzled, n_points, centroids, labels);
        case 24:
            return palette_mse_impl<24>(shN_swizzled, n_points, centroids, labels);
        case 45:
            return palette_mse_impl<45>(shN_swizzled, n_points, centroids, labels);
        default:
            return std::numeric_limits<double>::quiet_NaN();
        }
    }

} // namespace lfs::io
