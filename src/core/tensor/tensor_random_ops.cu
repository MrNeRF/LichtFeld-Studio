/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/assert.hpp"
#include "core/cuda_error.hpp"
#include "internal/tensor_functors.hpp"
#include "internal/tensor_ops.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Thrust headers for multinomial without replacement
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

namespace lfs::core::tensor_ops {

    namespace {

        __device__ float uniform_unit_interval(curandState* state) {
            constexpr float SCALE = 1.0f / 16777216.0f;
            return static_cast<float>(curand(state) >> 8) * SCALE;
        }

        struct invalid_multinomial_weight {
            __host__ __device__ bool operator()(float weight) const {
                return !isfinite(weight) || weight < 0.0f;
            }
        };

        struct multinomial_weight_to_double {
            __host__ __device__ double operator()(float weight) const {
                return static_cast<double>(weight);
            }
        };

    } // namespace

    // Note: run_with_thrust_policy is now in include/core/tensor_generic_ops.cuh

    // ============= Random Operations Kernels =============

    // Uniform random generation
    __global__ void uniform_kernel(float* data, size_t n, float low, float high,
                                   unsigned long long seed) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < n) {
            curandState state;
            curand_init(seed, idx, 0, &state);
            if (low == high) {
                data[idx] = low;
                return;
            }
            const float val = uniform_unit_interval(&state);
            const float result = fmaf(val, high - low, low);
            data[idx] = result < high ? result : nextafterf(high, low);
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
            const float val = uniform_unit_interval(&state);
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

            const uint64_t random = curand(&state);
            const uint64_t range = static_cast<uint64_t>(
                static_cast<int64_t>(high) - static_cast<int64_t>(low));
            const uint64_t offset = (random * range) >> 32;
            data[idx] = static_cast<int>(static_cast<int64_t>(low) +
                                         static_cast<int64_t>(offset));
        }
    }

    // Kernel for multinomial sampling with replacement
    __global__ void multinomial_with_replacement_kernel(const float* weights, int64_t* samples,
                                                        unsigned long n, unsigned long num_samples,
                                                        double sum, unsigned long long seed) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= num_samples)
            return;

        curandState state;
        curand_init(seed, idx, 0, &state);

        const double u = static_cast<double>(uniform_unit_interval(&state)) * sum;

        double cumsum = 0.0;
        for (unsigned long i = 0; i < n; ++i) {
            cumsum += static_cast<double>(weights[i]);
            if (u < cumsum) {
                samples[idx] = static_cast<int64_t>(i);
                return;
            }
        }

        samples[idx] = static_cast<int64_t>(n - 1);
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
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.random.uniform");
    }

    void launch_normal(float* data, size_t n, float mean, float std,
                       unsigned long long seed, cudaStream_t stream) {
        if (n == 0)
            return;

        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        normal_kernel<<<grid_size, block_size, 0, stream>>>(data, n, mean, std, seed);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.random.normal");
    }

    void launch_bernoulli(float* data, size_t n, float p,
                          unsigned long long seed, cudaStream_t stream) {
        if (n == 0)
            return;

        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        bernoulli_kernel<<<grid_size, block_size, 0, stream>>>(data, n, p, seed);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.random.bernoulli");
    }

    void launch_randint(int* data, size_t n, int low, int high,
                        unsigned long long seed, cudaStream_t stream) {
        if (n == 0)
            return;

        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        randint_kernel<<<grid_size, block_size, 0, stream>>>(data, n, low, high, seed);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.random.randint");
    }

    void launch_multinomial(const float* weights, int64_t* samples,
                            unsigned long n, unsigned long num_samples, bool replacement,
                            unsigned long long seed, cudaStream_t stream) {
        if (n == 0 || num_samples == 0)
            return;

        // Compute sum of weights using Thrust with centralized sum_op
        auto weights_ptr = thrust::device_pointer_cast(weights);

        size_t invalid_count = 0;
        double sum = 0.0;
        run_with_thrust_policy(stream, [&](auto policy) {
            invalid_count = thrust::count_if(
                policy,
                weights_ptr, weights_ptr + n,
                invalid_multinomial_weight{});
            sum = thrust::transform_reduce(
                policy,
                weights_ptr, weights_ptr + n,
                multinomial_weight_to_double{},
                0.0,
                thrust::plus<double>());
        });

        LFS_ASSERT_MSG(invalid_count == 0,
                       "multinomial weights must be finite and non-negative");
        LFS_ASSERT_MSG(std::isfinite(sum) && sum > 0.0,
                       "multinomial weights must have a positive finite sum");

        if (replacement) {
            int block_size = 256;
            int grid_size = (num_samples + block_size - 1) / block_size;
            multinomial_with_replacement_kernel<<<grid_size, block_size, 0, stream>>>(
                weights, samples, n, num_samples, sum, seed);
            LFS_CUDA_LAUNCH_CHECK(stream, "tensor.random.multinomial_with_replacement");
        } else {
            thrust::device_vector<float> keys(n);
            thrust::device_vector<int64_t> indices(n);

            int block_size = 256;
            int grid_size = (n + block_size - 1) / block_size;
            generate_gumbel_keys_kernel<<<grid_size, block_size, 0, stream>>>(
                weights, thrust::raw_pointer_cast(keys.data()), n, seed);
            LFS_CUDA_LAUNCH_CHECK(stream, "tensor.random.multinomial_gumbel_keys");

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
