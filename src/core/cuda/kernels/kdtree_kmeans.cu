/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "core/tensor.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <numeric>
#include <random>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>
#include <vector>

namespace lfs::core::cuda {

    namespace {

        // Chunk size for shared memory - 128 centroids at a time (like splat-transform)
        constexpr int CHUNK_SIZE = 128;
        constexpr int BLOCK_SIZE = 256;

        // GPU-based centroid initialization - gather k random points directly on GPU
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

        // Initialize random indices on GPU using thrust
        void init_random_indices_gpu(int* d_indices, const int n, const unsigned int seed) {
            thrust::device_ptr<int> indices_ptr(d_indices);
            thrust::sequence(indices_ptr, indices_ptr + n);
            thrust::default_random_engine rng(seed);
            thrust::shuffle(indices_ptr, indices_ptr + n, rng);
        }

        // Build CSR offsets from membership counts using thrust
        void build_csr_offsets_gpu(
            const int* d_membership,
            int* d_offsets,
            int* d_indices,
            const int k,
            const int num_clusters) {
            // Count elements per cluster using thrust histogram
            auto counts = Tensor::zeros({static_cast<size_t>(num_clusters)}, Device::CUDA, DataType::Int32);

            // Simple counting kernel
            struct count_op {
                int* counts;
                __device__ void operator()(int label) { atomicAdd(&counts[label], 1); }
            };

            // Use thrust to build sorted indices by cluster
            // 1. Create (cluster, index) pairs
            auto cluster_keys = Tensor::zeros({static_cast<size_t>(k)}, Device::CUDA, DataType::Int32);
            auto sorted_indices = Tensor::zeros({static_cast<size_t>(k)}, Device::CUDA, DataType::Int32);

            // Copy membership to keys, init indices to 0..k-1
            cudaMemcpy(cluster_keys.ptr<int>(), d_membership, k * sizeof(int), cudaMemcpyDeviceToDevice);
            thrust::device_ptr<int> indices_ptr(sorted_indices.ptr<int>());
            thrust::sequence(indices_ptr, indices_ptr + k);

            // Sort by cluster key
            thrust::device_ptr<int> keys_ptr(cluster_keys.ptr<int>());
            thrust::sort_by_key(keys_ptr, keys_ptr + k, indices_ptr);

            // Copy sorted indices to output
            cudaMemcpy(d_indices, sorted_indices.ptr<int>(), k * sizeof(int), cudaMemcpyDeviceToDevice);

            // Build offsets using exclusive scan of histogram
            // First count occurrences
            auto h_membership = std::vector<int>(k);
            cudaMemcpy(h_membership.data(), d_membership, k * sizeof(int), cudaMemcpyDeviceToHost);

            std::vector<int> counts_vec(num_clusters + 1, 0);
            for (int i = 0; i < k; ++i) {
                counts_vec[h_membership[i] + 1]++;
            }

            // Prefix sum for offsets
            for (int i = 1; i <= num_clusters; ++i) {
                counts_vec[i] += counts_vec[i - 1];
            }

            cudaMemcpy(d_offsets, counts_vec.data(), (num_clusters + 1) * sizeof(int), cudaMemcpyHostToDevice);
        }

        // Brute force nearest centroid with shared memory caching
        template <int N_DIMS>
        __global__ void assign_nearest_bruteforce_kernel(
            const float* __restrict__ data,      // [n_points, N_DIMS]
            const float* __restrict__ centroids, // [k, N_DIMS]
            int* __restrict__ labels,            // [n_points]
            const int n_points,
            const int k) {
            // Shared memory for centroid chunk
            __shared__ float shared_centroids[CHUNK_SIZE * N_DIMS];

            const int tid = blockIdx.x * blockDim.x + threadIdx.x;

            // Load this thread's point into registers
            float point[N_DIMS];
            if (tid < n_points) {
                for (int d = 0; d < N_DIMS; ++d) {
                    point[d] = data[tid * N_DIMS + d];
                }
            }

            float min_dist = 1e30f;
            int min_idx = 0;

            // Process centroids in chunks through shared memory
            const int num_chunks = (k + CHUNK_SIZE - 1) / CHUNK_SIZE;

            for (int chunk = 0; chunk < num_chunks; ++chunk) {
                const int chunk_start = chunk * CHUNK_SIZE;
                const int chunk_end = min(chunk_start + CHUNK_SIZE, k);
                const int chunk_size = chunk_end - chunk_start;

                // Collaboratively load centroid chunk into shared memory
                // Each thread loads some elements
                for (int i = threadIdx.x; i < chunk_size * N_DIMS; i += blockDim.x) {
                    int centroid_idx = chunk_start + i / N_DIMS;
                    int dim = i % N_DIMS;
                    if (centroid_idx < k) {
                        shared_centroids[i] = centroids[centroid_idx * N_DIMS + dim];
                    }
                }

                __syncthreads();

                // Find nearest centroid in this chunk
                if (tid < n_points) {
                    for (int c = 0; c < chunk_size; ++c) {
                        float dist = 0.0f;
#pragma unroll
                        for (int d = 0; d < N_DIMS; ++d) {
                            float diff = point[d] - shared_centroids[c * N_DIMS + d];
                            dist += diff * diff;
                        }

                        if (dist < min_dist) {
                            min_dist = dist;
                            min_idx = chunk_start + c;
                        }
                    }
                }

                __syncthreads();
            }

            if (tid < n_points) {
                labels[tid] = min_idx;
            }
        }

        // Specialized kernel for common dimensions with unrolling
        __global__ void assign_nearest_bruteforce_3d_kernel(
            const float* __restrict__ data,
            const float* __restrict__ centroids,
            int* __restrict__ labels,
            const int n_points,
            const int k) {
            __shared__ float shared_centroids[CHUNK_SIZE * 3];

            const int tid = blockIdx.x * blockDim.x + threadIdx.x;

            float px, py, pz;
            if (tid < n_points) {
                px = data[tid * 3 + 0];
                py = data[tid * 3 + 1];
                pz = data[tid * 3 + 2];
            }

            float min_dist = 1e30f;
            int min_idx = 0;

            const int num_chunks = (k + CHUNK_SIZE - 1) / CHUNK_SIZE;

            for (int chunk = 0; chunk < num_chunks; ++chunk) {
                const int chunk_start = chunk * CHUNK_SIZE;
                const int chunk_end = min(chunk_start + CHUNK_SIZE, k);
                const int chunk_size = chunk_end - chunk_start;

                for (int i = threadIdx.x; i < chunk_size * 3; i += blockDim.x) {
                    int idx = chunk_start * 3 + i;
                    if (idx < k * 3) {
                        shared_centroids[i] = centroids[idx];
                    }
                }

                __syncthreads();

                if (tid < n_points) {
                    for (int c = 0; c < chunk_size; ++c) {
                        float dx = px - shared_centroids[c * 3 + 0];
                        float dy = py - shared_centroids[c * 3 + 1];
                        float dz = pz - shared_centroids[c * 3 + 2];
                        float dist = dx * dx + dy * dy + dz * dz;

                        if (dist < min_dist) {
                            min_dist = dist;
                            min_idx = chunk_start + c;
                        }
                    }
                }

                __syncthreads();
            }

            if (tid < n_points) {
                labels[tid] = min_idx;
            }
        }

        // Accumulate points for centroid update
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

        // Finalize centroids (divide sums by counts)
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
                // Re-seed to random point
                unsigned int rng = seed ^ (cid * 1664525u + 1013904223u);
                rng = rng * 1664525u + 1013904223u;
                int rand_idx = rng % n_points;
                for (int d = 0; d < N_DIMS; ++d) {
                    centroids[cid * N_DIMS + d] = data[rand_idx * N_DIMS + d];
                }
            }
        }

        // Templated k-means using GPU brute force
        template <int N_DIMS>
        std::tuple<Tensor, Tensor> kmeans_gpu_bruteforce_impl(
            const Tensor& data,
            int k,
            int iterations) {
            const int n = data.shape()[0];

            LOG_INFO("kmeans_gpu_bruteforce<{}>: n={}, k={}, iterations={}", N_DIMS, n, k, iterations);

            if (n <= k) {
                auto centroids = data.clone();
                auto labels = Tensor::arange(n).to(DataType::Int32).cuda();
                return {centroids, labels};
            }

            // Ensure data is on GPU and contiguous
            auto data_gpu = data.cuda().contiguous();
            const float* d_data = data_gpu.ptr<float>();

            // Allocate centroids
            auto centroids = Tensor::zeros({static_cast<size_t>(k), static_cast<size_t>(N_DIMS)},
                                           Device::CUDA, DataType::Float32);
            float* d_centroids = centroids.ptr<float>();

            // Initialize centroids by sampling k random points (GPU-based, no CPU transfer)
            {
                auto perm = Tensor::zeros({static_cast<size_t>(n)}, Device::CUDA, DataType::Int32);
                std::random_device rd;
                init_random_indices_gpu(perm.ptr<int>(), n, rd());

                const int grid_k = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
                gather_centroids_kernel<N_DIMS><<<grid_k, BLOCK_SIZE>>>(
                    d_data, perm.ptr<int>(), d_centroids, k);
            }

            // Allocate labels
            auto labels = Tensor::zeros({static_cast<size_t>(n)}, Device::CUDA, DataType::Int32);
            int* d_labels = labels.ptr<int>();

            // Allocate accumulation buffers
            auto centroid_sums = Tensor::zeros({static_cast<size_t>(k), static_cast<size_t>(N_DIMS)},
                                               Device::CUDA, DataType::Float32);
            auto counts = Tensor::zeros({static_cast<size_t>(k)}, Device::CUDA, DataType::Int32);

            const int grid_size_points = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const int grid_size_centroids = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;

            for (int iter = 0; iter < iterations; ++iter) {
                // Step 1: Assign points to nearest centroid (brute force with shared memory)
                if constexpr (N_DIMS == 3) {
                    assign_nearest_bruteforce_3d_kernel<<<grid_size_points, BLOCK_SIZE>>>(
                        d_data, d_centroids, d_labels, n, k);
                } else {
                    assign_nearest_bruteforce_kernel<N_DIMS><<<grid_size_points, BLOCK_SIZE>>>(
                        d_data, d_centroids, d_labels, n, k);
                }

                // Step 2: Reset accumulation buffers
                centroid_sums.zero_();
                counts.zero_();

                // Step 3: Accumulate points
                accumulate_centroids_kernel<N_DIMS><<<grid_size_points, BLOCK_SIZE>>>(
                    d_data, d_labels, centroid_sums.ptr<float>(), counts.ptr<int>(), n);

                // Step 4: Finalize centroids
                unsigned int seed = static_cast<unsigned int>(iter * 12345 + 67890);
                finalize_centroids_kernel<N_DIMS><<<grid_size_centroids, BLOCK_SIZE>>>(
                    d_centroids, centroid_sums.ptr<float>(), counts.ptr<int>(),
                    d_data, k, n, seed);
            }

            cudaDeviceSynchronize();

            return {centroids, labels};
        }

        // Binary search for 1D k-means
        __global__ void binary_search_1d_kernel(
            const float* __restrict__ data,
            const float* __restrict__ sorted_centroids,
            int* __restrict__ labels,
            const int n_points,
            const int k) {
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid >= n_points)
                return;

            const float val = data[tid];

            int lo = 0, hi = k - 1;
            while (lo < hi) {
                int mid = (lo + hi) / 2;
                if (sorted_centroids[mid] < val) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }

            int best = lo;
            float best_dist = fabsf(val - sorted_centroids[lo]);

            if (lo > 0) {
                float dist_prev = fabsf(val - sorted_centroids[lo - 1]);
                if (dist_prev < best_dist) {
                    best = lo - 1;
                    best_dist = dist_prev;
                }
            }
            if (lo < k - 1) {
                float dist_next = fabsf(val - sorted_centroids[lo + 1]);
                if (dist_next < best_dist) {
                    best = lo + 1;
                }
            }

            labels[tid] = best;
        }

        __global__ void update_centroids_atomic_1d_kernel(
            const float* __restrict__ data,
            const int* __restrict__ labels,
            float* __restrict__ centroid_sums,
            int* __restrict__ counts,
            const int n_points) {
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid >= n_points)
                return;

            const int label = labels[tid];
            atomicAdd(&centroid_sums[label], data[tid]);
            atomicAdd(&counts[label], 1);
        }

        // Hierarchical K-Means for large k: two-level search via super-clusters.
        // Hybrid approach: hierarchical for early iterations, exact for final iteration.
        constexpr int NUM_SUPER_CLUSTERS = 256;
        constexpr int NUM_NEAREST_SUPERS = 4;
        constexpr int SUPER_CHUNK_SIZE = 64;

        // Find nearest super-clusters for each point
        template <int N_DIMS>
        __global__ void find_nearest_super_clusters_kernel(
            const float* __restrict__ data,            // [n_points, N_DIMS]
            const float* __restrict__ super_centroids, // [NUM_SUPER_CLUSTERS, N_DIMS]
            int* __restrict__ nearest_supers,          // [n_points, NUM_NEAREST_SUPERS]
            const int n_points) {
            __shared__ float shared_supers[SUPER_CHUNK_SIZE * N_DIMS];

            const int tid = blockIdx.x * blockDim.x + threadIdx.x;

            // Load point into registers
            float point[N_DIMS];
            if (tid < n_points) {
#pragma unroll
                for (int d = 0; d < N_DIMS; ++d) {
                    point[d] = data[tid * N_DIMS + d];
                }
            }

            // Track top NUM_NEAREST_SUPERS nearest super-clusters
            float best_dists[NUM_NEAREST_SUPERS];
            int best_idxs[NUM_NEAREST_SUPERS];
#pragma unroll
            for (int i = 0; i < NUM_NEAREST_SUPERS; ++i) {
                best_dists[i] = 1e30f;
                best_idxs[i] = 0;
            }

            // Process super-clusters in chunks
            const int num_chunks = (NUM_SUPER_CLUSTERS + SUPER_CHUNK_SIZE - 1) / SUPER_CHUNK_SIZE;

            for (int chunk = 0; chunk < num_chunks; ++chunk) {
                const int chunk_start = chunk * SUPER_CHUNK_SIZE;
                const int chunk_end = min(chunk_start + SUPER_CHUNK_SIZE, NUM_SUPER_CLUSTERS);
                const int chunk_size = chunk_end - chunk_start;

                // Load super-cluster centroids into shared memory
                for (int i = threadIdx.x; i < chunk_size * N_DIMS; i += blockDim.x) {
                    int sc_idx = chunk_start + i / N_DIMS;
                    int dim = i % N_DIMS;
                    if (sc_idx < NUM_SUPER_CLUSTERS) {
                        shared_supers[i] = super_centroids[sc_idx * N_DIMS + dim];
                    }
                }
                __syncthreads();

                // Find distance to each super-cluster
                if (tid < n_points) {
                    for (int sc = 0; sc < chunk_size; ++sc) {
                        float dist = 0.0f;
#pragma unroll
                        for (int d = 0; d < N_DIMS; ++d) {
                            float diff = point[d] - shared_supers[sc * N_DIMS + d];
                            dist += diff * diff;
                        }

                        // Update top-k nearest super-clusters
                        int sc_global = chunk_start + sc;
                        if (dist < best_dists[NUM_NEAREST_SUPERS - 1]) {
                            // Insert into sorted list
                            for (int i = NUM_NEAREST_SUPERS - 1; i >= 0; --i) {
                                if (i == 0 || dist >= best_dists[i - 1]) {
                                    // Insert at position i, shift others down
                                    for (int j = NUM_NEAREST_SUPERS - 1; j > i; --j) {
                                        best_dists[j] = best_dists[j - 1];
                                        best_idxs[j] = best_idxs[j - 1];
                                    }
                                    best_dists[i] = dist;
                                    best_idxs[i] = sc_global;
                                    break;
                                }
                            }
                        }
                    }
                }
                __syncthreads();
            }

            // Write results
            if (tid < n_points) {
                for (int i = 0; i < NUM_NEAREST_SUPERS; ++i) {
                    nearest_supers[tid * NUM_NEAREST_SUPERS + i] = best_idxs[i];
                }
            }
        }

        // Search within super-clusters to find nearest centroid
        template <int N_DIMS>
        __global__ void hierarchical_search_kernel(
            const float* __restrict__ data,           // [n_points, N_DIMS]
            const float* __restrict__ centroids,      // [k, N_DIMS]
            const int* __restrict__ nearest_supers,   // [n_points, NUM_NEAREST_SUPERS]
            const int* __restrict__ super_membership, // [k] - which super-cluster each centroid belongs to
            const int* __restrict__ super_offsets,    // [NUM_SUPER_CLUSTERS + 1] - CSR offsets
            const int* __restrict__ super_indices,    // [k] - centroid indices sorted by super-cluster
            int* __restrict__ labels,                 // [n_points]
            const int n_points,
            const int k) {
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;

            // Load point into registers
            float point[N_DIMS];
            if (tid < n_points) {
#pragma unroll
                for (int d = 0; d < N_DIMS; ++d) {
                    point[d] = data[tid * N_DIMS + d];
                }
            }

            float min_dist = 1e30f;
            int min_idx = 0;

            // For each super-cluster this point should search
            for (int si = 0; si < NUM_NEAREST_SUPERS; ++si) {
                int super_idx = 0;
                if (tid < n_points) {
                    super_idx = nearest_supers[tid * NUM_NEAREST_SUPERS + si];
                }
                // Broadcast super_idx from first thread (all threads in block need same super_idx for shared mem)
                // Actually, different threads have different points, so we process per-point
                // This is less efficient but correct

                if (tid < n_points) {
                    const int start = super_offsets[super_idx];
                    const int end = super_offsets[super_idx + 1];

                    // Search all centroids in this super-cluster
                    for (int c = start; c < end; ++c) {
                        int centroid_idx = super_indices[c];
                        float dist = 0.0f;
#pragma unroll
                        for (int d = 0; d < N_DIMS; ++d) {
                            float diff = point[d] - centroids[centroid_idx * N_DIMS + d];
                            dist += diff * diff;
                        }

                        if (dist < min_dist) {
                            min_dist = dist;
                            min_idx = centroid_idx;
                        }
                    }
                }
            }

            if (tid < n_points) {
                labels[tid] = min_idx;
            }
        }

        // Optimized hierarchical search with coalesced access
        template <int N_DIMS>
        __global__ void hierarchical_search_optimized_kernel(
            const float* __restrict__ data,         // [n_points, N_DIMS]
            const float* __restrict__ centroids,    // [k, N_DIMS]
            const int* __restrict__ nearest_supers, // [n_points, NUM_NEAREST_SUPERS]
            const int* __restrict__ super_offsets,  // [NUM_SUPER_CLUSTERS + 1]
            const int* __restrict__ super_indices,  // [k] - centroid indices sorted by super-cluster
            int* __restrict__ labels,               // [n_points]
            const int n_points,
            const int k) {
            // Shared memory for loading centroids
            extern __shared__ float shared_mem[];
            float* shared_centroids = shared_mem;

            const int tid = blockIdx.x * blockDim.x + threadIdx.x;

            // Load point into registers
            float point[N_DIMS];
            if (tid < n_points) {
#pragma unroll
                for (int d = 0; d < N_DIMS; ++d) {
                    point[d] = data[tid * N_DIMS + d];
                }
            }

            float min_dist = 1e30f;
            int min_idx = 0;

            // Process each super-cluster separately
            for (int si = 0; si < NUM_NEAREST_SUPERS; ++si) {
                int super_idx = (tid < n_points) ? nearest_supers[tid * NUM_NEAREST_SUPERS + si] : 0;
                int start = super_offsets[super_idx];
                int end = super_offsets[super_idx + 1];
                int n_centroids_in_super = end - start;

                // Process centroids in chunks through shared memory
                int num_chunks = (n_centroids_in_super + CHUNK_SIZE - 1) / CHUNK_SIZE;

                for (int chunk = 0; chunk < num_chunks; ++chunk) {
                    int chunk_start = chunk * CHUNK_SIZE;
                    int chunk_end = min(chunk_start + CHUNK_SIZE, n_centroids_in_super);
                    int chunk_size = chunk_end - chunk_start;

                    // Load centroid chunk into shared memory
                    for (int i = threadIdx.x; i < chunk_size * N_DIMS; i += blockDim.x) {
                        int local_c = i / N_DIMS;
                        int dim = i % N_DIMS;
                        int global_c = super_indices[start + chunk_start + local_c];
                        shared_centroids[i] = centroids[global_c * N_DIMS + dim];
                    }
                    __syncthreads();

                    // Find nearest in chunk
                    if (tid < n_points) {
                        for (int c = 0; c < chunk_size; ++c) {
                            int global_c = super_indices[start + chunk_start + c];
                            float dist = 0.0f;
#pragma unroll
                            for (int d = 0; d < N_DIMS; ++d) {
                                float diff = point[d] - shared_centroids[c * N_DIMS + d];
                                dist += diff * diff;
                            }

                            if (dist < min_dist) {
                                min_dist = dist;
                                min_idx = global_c;
                            }
                        }
                    }
                    __syncthreads();
                }
            }

            if (tid < n_points) {
                labels[tid] = min_idx;
            }
        }

        // Hierarchical k-means with super-cluster structure
        template <int N_DIMS>
        std::tuple<Tensor, Tensor> kmeans_hierarchical_impl(
            const Tensor& data,
            int k,
            int iterations) {
            const int n = static_cast<int>(data.shape()[0]);

            LOG_INFO("kmeans_hierarchical<{}>: n={}, k={}, iterations={}", N_DIMS, n, k, iterations);

            if (n <= k) {
                auto centroids = data.clone();
                auto labels = Tensor::arange(n).to(DataType::Int32).cuda();
                return {centroids, labels};
            }

            // Ensure data is on GPU and contiguous
            auto data_gpu = data.cuda().contiguous();
            const float* d_data = data_gpu.ptr<float>();

            // Allocate main centroids
            auto centroids = Tensor::zeros({static_cast<size_t>(k), static_cast<size_t>(N_DIMS)},
                                           Device::CUDA, DataType::Float32);
            float* d_centroids = centroids.ptr<float>();

            // Initialize centroids by sampling k random points (GPU-based, no CPU transfer)
            {
                auto perm = Tensor::zeros({static_cast<size_t>(n)}, Device::CUDA, DataType::Int32);
                std::random_device rd;
                init_random_indices_gpu(perm.ptr<int>(), n, rd());

                const int grid_k = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
                gather_centroids_kernel<N_DIMS><<<grid_k, BLOCK_SIZE>>>(
                    d_data, perm.ptr<int>(), d_centroids, k);
            }

            // Step 1: Build super-clusters by clustering the k centroids into NUM_SUPER_CLUSTERS groups
            auto super_centroids = Tensor::zeros({static_cast<size_t>(NUM_SUPER_CLUSTERS), static_cast<size_t>(N_DIMS)},
                                                 Device::CUDA, DataType::Float32);
            auto super_membership = Tensor::zeros({static_cast<size_t>(k)}, Device::CUDA, DataType::Int32);

            // Initialize super-centroids by sampling NUM_SUPER_CLUSTERS random centroids (GPU-based)
            {
                auto perm = Tensor::zeros({static_cast<size_t>(k)}, Device::CUDA, DataType::Int32);
                std::random_device rd;
                init_random_indices_gpu(perm.ptr<int>(), k, rd());

                const int grid_super = (NUM_SUPER_CLUSTERS + BLOCK_SIZE - 1) / BLOCK_SIZE;
                gather_centroids_kernel<N_DIMS><<<grid_super, BLOCK_SIZE>>>(
                    d_centroids, perm.ptr<int>(), super_centroids.ptr<float>(), NUM_SUPER_CLUSTERS);
            }

            // Run a few iterations of k-means to cluster centroids into super-clusters
            auto super_sums = Tensor::zeros({static_cast<size_t>(NUM_SUPER_CLUSTERS), static_cast<size_t>(N_DIMS)},
                                            Device::CUDA, DataType::Float32);
            auto super_counts = Tensor::zeros({static_cast<size_t>(NUM_SUPER_CLUSTERS)}, Device::CUDA, DataType::Int32);

            const int grid_k = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const int grid_super = (NUM_SUPER_CLUSTERS + BLOCK_SIZE - 1) / BLOCK_SIZE;

            // 5 iterations to cluster centroids into super-clusters
            for (int iter = 0; iter < 5; ++iter) {
                // Assign each centroid to nearest super-cluster
                assign_nearest_bruteforce_kernel<N_DIMS><<<grid_k, BLOCK_SIZE>>>(
                    d_centroids, super_centroids.ptr<float>(), super_membership.ptr<int>(), k, NUM_SUPER_CLUSTERS);

                // Update super-centroids
                super_sums.zero_();
                super_counts.zero_();

                accumulate_centroids_kernel<N_DIMS><<<grid_k, BLOCK_SIZE>>>(
                    d_centroids, super_membership.ptr<int>(), super_sums.ptr<float>(), super_counts.ptr<int>(), k);

                unsigned int seed = static_cast<unsigned int>(iter * 12345 + 67890);
                finalize_centroids_kernel<N_DIMS><<<grid_super, BLOCK_SIZE>>>(
                    super_centroids.ptr<float>(), super_sums.ptr<float>(), super_counts.ptr<int>(),
                    d_centroids, NUM_SUPER_CLUSTERS, k, seed);
            }

            // Final assignment of centroids to super-clusters
            assign_nearest_bruteforce_kernel<N_DIMS><<<grid_k, BLOCK_SIZE>>>(
                d_centroids, super_centroids.ptr<float>(), super_membership.ptr<int>(), k, NUM_SUPER_CLUSTERS);

            // Build CSR-style index using GPU-accelerated sort
            auto super_offsets = Tensor::zeros({static_cast<size_t>(NUM_SUPER_CLUSTERS + 1)}, Device::CUDA, DataType::Int32);
            auto super_indices = Tensor::zeros({static_cast<size_t>(k)}, Device::CUDA, DataType::Int32);
            build_csr_offsets_gpu(super_membership.ptr<int>(), super_offsets.ptr<int>(),
                                  super_indices.ptr<int>(), k, NUM_SUPER_CLUSTERS);

            // Allocate buffers for main k-means iterations
            auto labels = Tensor::zeros({static_cast<size_t>(n)}, Device::CUDA, DataType::Int32);
            auto nearest_supers = Tensor::zeros({static_cast<size_t>(n), static_cast<size_t>(NUM_NEAREST_SUPERS)},
                                                Device::CUDA, DataType::Int32);
            auto centroid_sums = Tensor::zeros({static_cast<size_t>(k), static_cast<size_t>(N_DIMS)},
                                               Device::CUDA, DataType::Float32);
            auto centroid_counts = Tensor::zeros({static_cast<size_t>(k)}, Device::CUDA, DataType::Int32);

            const int grid_n = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

            // Main k-means iterations using hierarchical search
            for (int iter = 0; iter < iterations; ++iter) {
                // Step 1: Rebuild super-cluster structure (centroids have moved)
                if (iter > 0) {
                    // Re-assign centroids to super-clusters
                    assign_nearest_bruteforce_kernel<N_DIMS><<<grid_k, BLOCK_SIZE>>>(
                        d_centroids, super_centroids.ptr<float>(), super_membership.ptr<int>(), k, NUM_SUPER_CLUSTERS);

                    // Update super-centroids
                    super_sums.zero_();
                    super_counts.zero_();
                    accumulate_centroids_kernel<N_DIMS><<<grid_k, BLOCK_SIZE>>>(
                        d_centroids, super_membership.ptr<int>(), super_sums.ptr<float>(), super_counts.ptr<int>(), k);
                    finalize_centroids_kernel<N_DIMS><<<grid_super, BLOCK_SIZE>>>(
                        super_centroids.ptr<float>(), super_sums.ptr<float>(), super_counts.ptr<int>(),
                        d_centroids, NUM_SUPER_CLUSTERS, k, iter * 111);

                    // Rebuild CSR index using GPU-accelerated sort
                    build_csr_offsets_gpu(super_membership.ptr<int>(), super_offsets.ptr<int>(),
                                          super_indices.ptr<int>(), k, NUM_SUPER_CLUSTERS);
                }

                // Step 2 & 3: Assignment - use hierarchical for early iterations, brute-force for final
                const bool use_exact = (iter == iterations - 1); // Exact search on last iteration for quality

                if (use_exact) {
                    // Final iteration: use exact brute-force for best quality
                    assign_nearest_bruteforce_kernel<N_DIMS><<<grid_n, BLOCK_SIZE>>>(
                        d_data, d_centroids, labels.ptr<int>(), n, k);
                } else {
                    // Early iterations: use fast hierarchical search
                    find_nearest_super_clusters_kernel<N_DIMS><<<grid_n, BLOCK_SIZE>>>(
                        d_data, super_centroids.ptr<float>(), nearest_supers.ptr<int>(), n);

                    hierarchical_search_kernel<N_DIMS><<<grid_n, BLOCK_SIZE>>>(
                        d_data, d_centroids, nearest_supers.ptr<int>(),
                        super_membership.ptr<int>(), super_offsets.ptr<int>(), super_indices.ptr<int>(),
                        labels.ptr<int>(), n, k);
                }

                // Step 4: Update centroids
                centroid_sums.zero_();
                centroid_counts.zero_();

                accumulate_centroids_kernel<N_DIMS><<<grid_n, BLOCK_SIZE>>>(
                    d_data, labels.ptr<int>(), centroid_sums.ptr<float>(), centroid_counts.ptr<int>(), n);

                unsigned int seed = static_cast<unsigned int>(iter * 12345 + 67890);
                finalize_centroids_kernel<N_DIMS><<<grid_k, BLOCK_SIZE>>>(
                    d_centroids, centroid_sums.ptr<float>(), centroid_counts.ptr<int>(),
                    d_data, k, n, seed);
            }

            cudaDeviceSynchronize();
            return {centroids, labels};
        }

    } // anonymous namespace

    // Forward declaration
    std::tuple<Tensor, Tensor> kmeans_1d_binary(
        const Tensor& data,
        int k,
        int iterations);

    std::tuple<Tensor, Tensor> kmeans_kdtree(
        const Tensor& data,
        int k,
        int iterations,
        float tolerance) {
        (void)tolerance; // Not used in GPU implementation

        if (data.ndim() != 2) {
            LOG_ERROR("kmeans_kdtree expects 2D input [n_points, n_dims]");
            return {Tensor(), Tensor()};
        }

        const int n_dims = static_cast<int>(data.shape()[1]);

        // Dispatch based on dimension
        // Shared memory bruteforce is faster than batched cuBLAS when memory is limited
        switch (n_dims) {
        case 1:
            return kmeans_1d_binary(data, k, iterations);
        case 2:
            return kmeans_gpu_bruteforce_impl<2>(data, k, iterations);
        case 3:
            return kmeans_gpu_bruteforce_impl<3>(data, k, iterations);
        case 4:
            return kmeans_gpu_bruteforce_impl<4>(data, k, iterations);
        case 9:
            return kmeans_gpu_bruteforce_impl<9>(data, k, iterations);
        case 12:
            return kmeans_gpu_bruteforce_impl<12>(data, k, iterations);
        case 15:
            return kmeans_gpu_bruteforce_impl<15>(data, k, iterations);
        case 24:
            return kmeans_gpu_bruteforce_impl<24>(data, k, iterations);
        case 27:
            return kmeans_gpu_bruteforce_impl<27>(data, k, iterations);
        case 45:
            // Use hierarchical k-means for large k (65536) - much faster
            if (k >= 4096) {
                return kmeans_hierarchical_impl<45>(data, k, iterations);
            }
            return kmeans_gpu_bruteforce_impl<45>(data, k, iterations);
        case 48:
            return kmeans_gpu_bruteforce_impl<48>(data, k, iterations);
        default:
            LOG_ERROR("kmeans_kdtree: unsupported dimension {}. "
                      "Supported: 1, 2, 3, 4, 9, 12, 15, 24, 27, 45, 48",
                      n_dims);
            return {Tensor(), Tensor()};
        }
    }

    // 1D k-means using binary search
    std::tuple<Tensor, Tensor> kmeans_1d_binary(
        const Tensor& data,
        int k,
        int iterations) {
        Tensor data_2d;
        if (data.ndim() == 1) {
            data_2d = data.unsqueeze(1);
        } else if (data.ndim() == 2 && data.shape()[1] == 1) {
            data_2d = data;
        } else {
            LOG_ERROR("kmeans_1d_binary expects 1D data");
            return {Tensor(), Tensor()};
        }

        auto data_gpu = data_2d.cuda().contiguous();
        const int n = data_gpu.shape()[0];

        if (n <= k) {
            auto sorted = std::get<0>(data_gpu.sort(0));
            auto labels = Tensor::arange(n).to(DataType::Int32).cuda();
            return {sorted, labels};
        }

        // Initialize centroids evenly across range
        float min_val = data_gpu.min().item();
        float max_val = data_gpu.max().item();

        std::vector<float> centroid_vals(k);
        float step = (k > 1) ? (max_val - min_val) / (k - 1) : 0.0f;
        for (int i = 0; i < k; ++i) {
            centroid_vals[i] = min_val + i * step;
        }
        auto centroids = Tensor::from_vector(centroid_vals, {static_cast<size_t>(k), 1}, Device::CUDA);
        auto labels = Tensor::zeros({static_cast<size_t>(n)}, Device::CUDA, DataType::Int32);

        auto centroid_sums = Tensor::zeros({static_cast<size_t>(k)}, Device::CUDA, DataType::Float32);
        auto counts = Tensor::zeros({static_cast<size_t>(k)}, Device::CUDA, DataType::Int32);

        const int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for (int iter = 0; iter < iterations; ++iter) {
            auto centroids_1d = centroids.squeeze();
            auto [sorted_centroids, sort_idx] = centroids_1d.sort(0);

            binary_search_1d_kernel<<<grid_size, BLOCK_SIZE>>>(
                data_gpu.ptr<float>(),
                sorted_centroids.ptr<float>(),
                labels.ptr<int>(),
                n, k);

            centroid_sums.zero_();
            counts.zero_();

            update_centroids_atomic_1d_kernel<<<grid_size, BLOCK_SIZE>>>(
                data_gpu.ptr<float>(),
                labels.ptr<int>(),
                centroid_sums.ptr<float>(),
                counts.ptr<int>(),
                n);

            auto counts_cpu = counts.cpu();
            auto sums_cpu = centroid_sums.cpu();
            const int* cnt = counts_cpu.ptr<int>();
            const float* sum = sums_cpu.ptr<float>();

            for (int c = 0; c < k; ++c) {
                if (cnt[c] > 0) {
                    centroid_vals[c] = sum[c] / cnt[c];
                }
            }

            centroids = Tensor::from_vector(centroid_vals, {static_cast<size_t>(k), 1}, Device::CUDA);
        }

        cudaDeviceSynchronize();

        // Final sort
        auto centroids_1d = centroids.squeeze();
        auto [final_sorted, final_idx] = centroids_1d.sort(0);
        centroids = final_sorted.unsqueeze(1);

        // Remap labels
        auto inv_map = Tensor::zeros({static_cast<size_t>(k)}, Device::CPU, DataType::Int32);
        auto idx_cpu = final_idx.to(DataType::Int32).cpu();
        int* inv_ptr = inv_map.ptr<int>();
        const int* idx_ptr = idx_cpu.ptr<int>();
        for (int i = 0; i < k; ++i) {
            inv_ptr[idx_ptr[i]] = i;
        }

        auto remapped = inv_map.cuda().index_select(0, labels);

        return {centroids, remapped};
    }

} // namespace lfs::core::cuda
