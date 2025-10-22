/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/tensor_functors.hpp"
#include "internal/tensor_ops.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Thrust headers for multinomial without replacement
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace lfs::core::tensor_ops {

    // Note: run_with_thrust_policy is now in include/core/tensor_generic_ops.cuh

    // ============= Random Operations Kernels =============

    // Uniform random generation
    __global__ void uniform_kernel(float* data, size_t n, float low, float high,
                                   unsigned long long seed) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < n) {
            curandState state;
            curand_init(seed, idx, 0, &state);
            float val = curand_uniform(&state);
            data[idx] = val * (high - low) + low;
        }
    }

    // Normal random generation
    __global__ void normal_kernel(float* data, size_t n, float mean, float std,
                                  unsigned long long seed) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < n) {
            curandState state;
            curand_init(seed, idx, 0, &state);
            data[idx] = curand_normal(&state) * std + mean;
        }
    }

    // Bernoulli random generation
    __global__ void bernoulli_kernel(float* data, size_t n, float p,
                                     unsigned long long seed) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < n) {
            curandState state;
            curand_init(seed, idx, 0, &state);
            float val = curand_uniform(&state);
            data[idx] = (val < p) ? 1.0f : 0.0f;
        }
    }

    // Random integer generation
    __global__ void randint_kernel(int* data, size_t n, int low, int high,
                                   unsigned long long seed) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < n) {
            curandState state;
            curand_init(seed, idx, 0, &state);

            float val = curand_uniform(&state);
            int range = high - low;

            int result = low + static_cast<int>(val * range);

            if (result >= high)
                result = high - 1;
            if (result < low)
                result = low;

            data[idx] = result;
        }
    }

    // Kernel for multinomial sampling with replacement
    __global__ void multinomial_with_replacement_kernel(const float* weights, int* samples,
                                                        unsigned long n, unsigned long num_samples,
                                                        float sum, unsigned long long seed) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= num_samples)
            return;

        curandState state;
        curand_init(seed, idx, 0, &state);

        float u = curand_uniform(&state) * sum;

        float cumsum = 0.0f;
        for (unsigned long i = 0; i < n; ++i) {
            cumsum += weights[i];
            if (u <= cumsum) {
                samples[idx] = i;
                return;
            }
        }

        samples[idx] = n - 1;
    }

    // Kernel to generate random keys for each index (Gumbel-max trick)
    __global__ void generate_gumbel_keys_kernel(const float* weights, float* keys,
                                                unsigned long n, unsigned long long seed) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= n)
            return;

        curandState state;
        curand_init(seed, idx, 0, &state);

        float u = curand_uniform(&state);

        u = fmaxf(u, 1e-10f);
        u = fminf(u, 1.0f - 1e-10f);

        float gumbel = -logf(-logf(u));

        float log_weight = logf(fmaxf(weights[idx], 1e-10f));
        keys[idx] = log_weight + gumbel;
    }

    // ============= Launch Functions =============

    void launch_uniform(float* data, size_t n, float low, float high,
                        unsigned long long seed, cudaStream_t stream) {
        if (n == 0)
            return;

        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        uniform_kernel<<<grid_size, block_size, 0, stream>>>(data, n, low, high, seed);
    }

    void launch_normal(float* data, size_t n, float mean, float std,
                       unsigned long long seed, cudaStream_t stream) {
        if (n == 0)
            return;

        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        normal_kernel<<<grid_size, block_size, 0, stream>>>(data, n, mean, std, seed);
    }

    void launch_bernoulli(float* data, size_t n, float p,
                          unsigned long long seed, cudaStream_t stream) {
        if (n == 0)
            return;

        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        bernoulli_kernel<<<grid_size, block_size, 0, stream>>>(data, n, p, seed);
    }

    void launch_randint(int* data, size_t n, int low, int high,
                        unsigned long long seed, cudaStream_t stream) {
        if (n == 0)
            return;

        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        randint_kernel<<<grid_size, block_size, 0, stream>>>(data, n, low, high, seed);
    }

    void launch_multinomial(const float* weights, int* samples,
                            unsigned long n, unsigned long num_samples, bool replacement,
                            unsigned long long seed, cudaStream_t stream) {
        if (n == 0 || num_samples == 0)
            return;

        // Compute sum of weights using Thrust with centralized sum_op
        auto weights_ptr = thrust::device_pointer_cast(weights);

        float sum;
        run_with_thrust_policy(stream, [&](auto policy) {
            sum = thrust::reduce(
                policy,
                weights_ptr, weights_ptr + n,
                0.0f,
                ops::sum_op());
        });

        if (sum <= 0) {
            cudaMemsetAsync(samples, 0, num_samples * sizeof(int), stream);
            return;
        }

        if (replacement) {
            int block_size = 256;
            int grid_size = (num_samples + block_size - 1) / block_size;
            multinomial_with_replacement_kernel<<<grid_size, block_size, 0, stream>>>(
                weights, samples, n, num_samples, sum, seed);
        } else {
            thrust::device_vector<float> keys(n);
            thrust::device_vector<int> indices(n);

            int block_size = 256;
            int grid_size = (n + block_size - 1) / block_size;
            generate_gumbel_keys_kernel<<<grid_size, block_size, 0, stream>>>(
                weights, thrust::raw_pointer_cast(keys.data()), n, seed);

            run_with_thrust_policy(stream, [&](auto policy) {
                thrust::sequence(
                    policy,
                    indices.begin(), indices.end());

                thrust::sort_by_key(
                    policy,
                    keys.begin(), keys.end(),
                    indices.begin(),
                    thrust::greater<float>());

                thrust::copy_n(
                    policy,
                    indices.begin(),
                    num_samples,
                    thrust::device_pointer_cast(samples));
            });
        }
    }

} // namespace lfs::core::tensor_ops