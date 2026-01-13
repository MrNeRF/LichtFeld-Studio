/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/kmeans.cuh"
#include "core/tensor.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace lfs::core {
    namespace cuda {

        namespace {

            // Kernel to compute distances and assign labels
            template <int BLOCK_SIZE = 256>
            __global__ void assign_clusters_kernel(
                const float* __restrict__ data,
                const float* __restrict__ centroids,
                int* __restrict__ labels,
                float* __restrict__ distances,
                const int n_points,
                const int n_clusters,
                const int n_dims) {
                const int tid = blockIdx.x * blockDim.x + threadIdx.x;

                if (tid >= n_points)
                    return;

                float min_dist = CUDA_INFINITY;
                int min_idx = 0;

                // For each centroid
                for (int c = 0; c < n_clusters; ++c) {
                    float dist = 0.0f;

                    // Compute squared Euclidean distance
                    for (int d = 0; d < n_dims; ++d) {
                        float diff = data[tid * n_dims + d] - centroids[c * n_dims + d];
                        dist += diff * diff;
                    }

                    if (dist < min_dist) {
                        min_dist = dist;
                        min_idx = c;
                    }
                }

                labels[tid] = min_idx;
                if (distances != nullptr) {
                    distances[tid] = min_dist;
                }
            }

            // Optimized kernel for 1D clustering
            __global__ void assign_clusters_1d_kernel(
                const float* __restrict__ data,
                const float* __restrict__ centroids,
                int* __restrict__ labels,
                const int n_points,
                const int n_clusters) {
                const int tid = blockIdx.x * blockDim.x + threadIdx.x;

                if (tid >= n_points)
                    return;

                const float point = data[tid];
                float min_dist = CUDA_INFINITY;
                int best = 0;

                // Linear search (sorted centroids)
                for (int c = 0; c < n_clusters; ++c) {
                    float dist = fabsf(point - centroids[c]);
                    if (dist < min_dist) {
                        min_dist = dist;
                        best = c;
                    }
                }

                labels[tid] = best;
            }

            // Kernel to update centroids
            __global__ void update_centroids_kernel(
                const float* __restrict__ data,
                const int* __restrict__ labels,
                float* __restrict__ new_centroids,
                int* __restrict__ counts,
                const int n_points,
                const int n_clusters,
                const int n_dims) {
                const int cluster_id = blockIdx.x;
                const int dim = threadIdx.x;

                if (cluster_id >= n_clusters || dim >= n_dims)
                    return;

                float sum = 0.0f;
                int count = 0;

                // Sum all points belonging to this cluster
                for (int i = 0; i < n_points; ++i) {
                    if (labels[i] == cluster_id) {
                        sum += data[i * n_dims + dim];
                        if (dim == 0)
                            count++;
                    }
                }

                // Store result
                if (dim == 0) {
                    counts[cluster_id] = count;
                }

                __syncthreads();

                if (counts[cluster_id] > 0) {
                    new_centroids[cluster_id * n_dims + dim] = sum / counts[cluster_id];
                }
            }

            // Initialize centroids using k-means++ algorithm
            Tensor initialize_centroids_plusplus(
                const Tensor& data,
                int k) {
                const int n = data.shape()[0];
                const int d = data.shape()[1];

                auto centroids = Tensor::zeros({static_cast<size_t>(k), static_cast<size_t>(d)},
                                               Device::CUDA, DataType::Float32);
                auto distances = Tensor::full({static_cast<size_t>(n)}, CUDA_INFINITY,
                                              Device::CUDA, DataType::Float32);

                // Choose first centroid randomly
                int first_idx = Tensor::randint({1}, 0, n, Device::CUDA, DataType::Int32).item<int>();

                // Copy first centroid row
                centroids[0] = data[first_idx];

                // Choose remaining centroids
                for (int c = 1; c < k; ++c) {
                    // Compute distances to nearest centroid
                    auto centroid_view = centroids.slice(0, 0, c);
                    auto dists = data.cdist(centroid_view);
                    distances = dists.min(1);

                    // Choose next centroid with probability proportional to squared distance
                    auto probs = distances.square();
                    probs = probs.div(probs.sum());

                    // Sample from distribution using cumsum
                    auto cumsum = probs.cumsum(0);
                    float rand_val = Tensor::rand({1}, Device::CUDA).item();
                    auto ge_mask = cumsum.ge(rand_val);
                    auto indices = ge_mask.nonzero(); // Returns [count, 1] for 1D input

                    if (indices.numel() == 0) {
                        // Fallback: if no index found (shouldn't happen), pick randomly
                        int next_idx = Tensor::randint({1}, 0, n, Device::CUDA, DataType::Int32).item<int>();
                        centroids[c] = data[next_idx];
                    } else {
                        // Clean extraction using row proxy - no CPU transfer needed!
                        int64_t next_idx = indices[0].item_int64();
                        centroids[c] = data[static_cast<int>(next_idx)];
                    }
                }

                return centroids;
            }

        } // anonymous namespace

        std::tuple<Tensor, Tensor> kmeans_new(
            const Tensor& data,
            int k,
            int iterations,
            float tolerance) {

            if (data.ndim() != 2) {
                LOG_ERROR("Data must be 2D tensor [N, D]");
                return {Tensor(), Tensor()};
            }

            if (data.device() != Device::CUDA) {
                LOG_ERROR("Data must be on CUDA");
                return {Tensor(), Tensor()};
            }

            if (data.dtype() != DataType::Float32) {
                LOG_ERROR("Data must be float32");
                return {Tensor(), Tensor()};
            }

            const int n = data.shape()[0];
            const int d = data.shape()[1];

            if (n <= k) {
                const auto centroids = data.clone();
                auto labels = Tensor::arange(n).to(DataType::Int32).cuda();
                return {centroids, labels};
            }

            // Initialize centroids using k-means++
            auto centroids = initialize_centroids_plusplus(data, k);
            auto labels = Tensor::zeros({static_cast<size_t>(n)}, Device::CUDA, DataType::Int32);
            auto old_centroids = Tensor::zeros_like(centroids);

            // Allocate workspace
            auto counts = Tensor::zeros({static_cast<size_t>(k)}, Device::CUDA, DataType::Int32);

            const int block_size = 256;
            const int grid_size_points = (n + block_size - 1) / block_size;

            for (int iter = 0; iter < iterations; ++iter) {
                old_centroids.copy_from(centroids);

                // Assign clusters
                assign_clusters_kernel<block_size><<<grid_size_points, block_size>>>(
                    data.ptr<float>(),
                    centroids.ptr<float>(),
                    labels.ptr<int>(),
                    nullptr,
                    n, k, d);

                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    LOG_ERROR("CUDA error in assign_clusters: {}", cudaGetErrorString(err));
                    return {centroids, labels};
                }

                cudaDeviceSynchronize();

                err = cudaGetLastError();
                if (err != cudaSuccess) {
                    LOG_ERROR("CUDA error after sync: {}", cudaGetErrorString(err));
                    return {centroids, labels};
                }

                // Update centroids
                counts.zero_();
                dim3 block(d, 1);
                dim3 grid(k, 1);

                update_centroids_kernel<<<grid, block>>>(
                    data.ptr<float>(),
                    labels.ptr<int>(),
                    centroids.ptr<float>(),
                    counts.ptr<int>(),
                    n, k, d);

                err = cudaGetLastError();
                if (err != cudaSuccess) {
                    LOG_ERROR("CUDA error in update_centroids: {}", cudaGetErrorString(err));
                    return {centroids, labels};
                }

                cudaDeviceSynchronize();

                err = cudaGetLastError();
                if (err != cudaSuccess) {
                    LOG_ERROR("CUDA error after centroid update sync: {}", cudaGetErrorString(err));
                    return {centroids, labels};
                }

                // Check convergence
                float max_movement = (centroids - old_centroids).abs().max().item();

                if (max_movement < tolerance) {
                    break;
                }
            }

            return {centroids, labels};
        }

        std::tuple<Tensor, Tensor> kmeans_1d_new(
            const Tensor& data,
            int k,
            int iterations) {

            // Reshape to [N, 1] properly
            Tensor data_2d;
            if (data.ndim() == 1) {
                data_2d = data.unsqueeze(1); // [N] -> [N, 1]
            } else if (data.ndim() == 2 && data.shape()[1] == 1) {
                data_2d = data.clone();
            } else {
                LOG_ERROR("kmeans_1d expects 1D data, got shape {}", data.shape().str());
                return {Tensor(), Tensor()};
            }

            const int n = data_2d.shape()[0];

            if (n <= k) {
                // Special case: fewer points than clusters
                auto sorted_result = data_2d.sort(0);
                auto sorted = std::get<0>(sorted_result);
                auto labels = Tensor::arange(n).to(DataType::Int32).cuda();
                return {sorted, labels};
            }

            // For 1D, initialize centroids evenly across range
            float min_val = data_2d.min().item();
            float max_val = data_2d.max().item();

            // Create centroids as [k, 1] tensor - initialize manually to avoid linspace dependency
            std::vector<float> centroid_values(k);
            float step = (k > 1) ? (max_val - min_val) / (k - 1) : 0.0f;
            for (int i = 0; i < k; ++i) {
                centroid_values[i] = min_val + i * step;
            }
            auto centroids = Tensor::from_vector(centroid_values, {static_cast<size_t>(k), 1}, Device::CUDA);
            auto labels = Tensor::zeros({static_cast<size_t>(n)}, Device::CUDA, DataType::Int32);

            const int block_size = 256;
            const int grid_size = (n + block_size - 1) / block_size;

            for (int iter = 0; iter < iterations; ++iter) {
                // Sort centroids for efficient 1D assignment
                auto centroids_1d = centroids.squeeze(); // [k, 1] -> [k]
                auto sort_result = centroids_1d.sort(0);
                auto sorted_centroids = std::get<0>(sort_result);
                auto sort_idx = std::get<1>(sort_result);

                // Assign clusters using optimized 1D kernel
                assign_clusters_1d_kernel<<<grid_size, block_size>>>(
                    data_2d.ptr<float>(),
                    sorted_centroids.ptr<float>(),
                    labels.ptr<int>(),
                    n, k);

                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    LOG_ERROR("CUDA error in 1D assign_clusters: {}", cudaGetErrorString(err));
                    return {centroids, labels};
                }

                cudaDeviceSynchronize();

                // OPTIMIZATION: Update centroids using parallel GPU kernel instead of slow CPU loop
                // This eliminates k GPU->CPU syncs and processes all clusters in parallel
                auto counts = Tensor::zeros({static_cast<size_t>(k)}, Device::CUDA, DataType::Int32);
                dim3 block(1, 1); // 1D data, only need 1 thread per cluster
                dim3 grid(k, 1);

                update_centroids_kernel<<<grid, block>>>(
                    data_2d.ptr<float>(),
                    labels.ptr<int>(),
                    centroids.ptr<float>(),
                    counts.ptr<int>(),
                    n, k, 1); // n_dims = 1 for 1D data

                err = cudaGetLastError();
                if (err != cudaSuccess) {
                    LOG_ERROR("CUDA error in 1D update_centroids: {}", cudaGetErrorString(err));
                    return {centroids, labels};
                }

                cudaDeviceSynchronize();
            }

            // Final sort of centroids and remap labels
            auto centroids_1d = centroids.squeeze();
            auto final_sort = centroids_1d.sort(0);
            auto final_sorted = std::get<0>(final_sort);
            auto final_idx = std::get<1>(final_sort);
            centroids = final_sorted.unsqueeze(1);

            // Create inverse mapping for labels using thrust::scatter (GPU-parallel, no CPU syncs)
            auto inv_map = Tensor::zeros({static_cast<size_t>(k)}, Device::CUDA, DataType::Int32);
            auto final_idx_int = final_idx.to(DataType::Int32);

            // Generate sequence [0, 1, 2, ..., k-1] for scatter values
            auto seq = Tensor::arange(k).to(DataType::Int32).cuda();

            // Scatter: inv_map[final_idx_int[i]] = seq[i] for all i
            // This builds the inverse permutation in parallel on GPU
            thrust::scatter(
                thrust::device,
                seq.ptr<int>(),
                seq.ptr<int>() + k,
                final_idx_int.ptr<int>(),
                inv_map.ptr<int>());

            // Remap labels using thrust::gather
            auto remapped_labels = Tensor::zeros_like(labels);
            thrust::gather(
                thrust::device,
                labels.ptr<int>(),
                labels.ptr<int>() + n,
                inv_map.ptr<int>(),
                remapped_labels.ptr<int>());

            cudaDeviceSynchronize();

            return {centroids, remapped_labels};
        }

    } // namespace cuda
} // namespace lfs::core
