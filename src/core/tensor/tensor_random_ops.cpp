/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "internal/tensor_impl.hpp"
#include "internal/tensor_ops.hpp"
#include <atomic>
#include <curand.h>
#include <curand_kernel.h>
#include <mutex>
#include <random>

#define CHECK_CURAND(call)                                                   \
    do {                                                                     \
        const curandStatus_t error = (call);                                 \
        LFS_ASSERT_MSG(error == CURAND_STATUS_SUCCESS,                       \
                       std::string("CURAND operation failed with status ") + \
                           std::to_string(static_cast<int>(error)));         \
    } while (0)

namespace lfs::core {

    // ============= RandomGenerator Implementation =============

    // Private implementation class to hide atomic counter
    class RandomGeneratorImpl {
    public:
        std::atomic<uint64_t> call_counter_{0};
        uint64_t cuda_offset_ = 0;
        uint64_t seed_ = 42;
        void* cuda_generator_ = nullptr;
        std::mt19937_64 cpu_generator_;
        std::mutex cuda_mutex_;

        RandomGeneratorImpl() : seed_(42),
                                cpu_generator_(seed_) {
            // Initialize CUDA random generator with Philox (same as PyTorch - much faster!)
            curandGenerator_t* gen = new curandGenerator_t;
            CHECK_CURAND(curandCreateGenerator(gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
            CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(*gen, seed_));
            cuda_generator_ = gen;
        }

        ~RandomGeneratorImpl() {
            if (cuda_generator_) {
                curandGenerator_t* gen = static_cast<curandGenerator_t*>(cuda_generator_);
                curandDestroyGenerator(*gen);
                delete gen;
            }
        }
    };

    RandomGenerator& RandomGenerator::instance() {
        static RandomGenerator instance;
        return instance;
    }

    RandomGenerator::RandomGenerator() : seed_(42),
                                         cpu_generator_(seed_) {
        // Create implementation
        impl_ = new RandomGeneratorImpl();
    }

    RandomGenerator::~RandomGenerator() {
        if (impl_) {
            delete static_cast<RandomGeneratorImpl*>(impl_);
        }
    }

    void RandomGenerator::manual_seed(uint64_t seed) {
        seed_ = seed;
        cpu_generator_.seed(seed);

        auto* impl = static_cast<RandomGeneratorImpl*>(impl_);
        std::lock_guard lock(impl->cuda_mutex_);
        impl->seed_ = seed;
        impl->cpu_generator_.seed(seed);
        impl->call_counter_.store(0);
        impl->cuda_offset_ = 0;

        if (impl->cuda_generator_) {
            curandGenerator_t* gen = static_cast<curandGenerator_t*>(impl->cuda_generator_);
            CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(*gen, seed));
            // IMPORTANT: Reset the offset to ensure reproducibility
            CHECK_CURAND(curandSetGeneratorOffset(*gen, 0));
        }
    }

    uint64_t RandomGenerator::get_next_cuda_seed() {
        auto* impl = static_cast<RandomGeneratorImpl*>(impl_);
        uint64_t base_seed = impl->seed_;
        uint64_t counter = impl->call_counter_.fetch_add(1);
        return base_seed + counter * 1000000ULL;
    }

    void RandomGenerator::generate_cuda_normal(float* output, const size_t count,
                                               const float mean, const float std,
                                               const cudaStream_t stream) {
        LFS_ASSERT_MSG(output != nullptr,
                       "CUDA normal generation requires a valid output pointer");
        LFS_ASSERT_MSG(count > 0 && count % 2 == 0,
                       "CUDA normal generation requires a positive even value count");
        auto* impl = static_cast<RandomGeneratorImpl*>(impl_);
        std::lock_guard lock(impl->cuda_mutex_);
        LFS_ASSERT_MSG(count <= std::numeric_limits<uint64_t>::max() - impl->cuda_offset_,
                       "CUDA random generator offset overflow");

        curandGenerator_t* gen = static_cast<curandGenerator_t*>(impl->cuda_generator_);
        impl->cuda_offset_ += count;
        CHECK_CURAND(curandSetStream(*gen, stream));
        CHECK_CURAND(curandGenerateNormal(*gen, output, count, mean, std));
    }

    void* RandomGenerator::get_generator(Device device) {
        auto* impl = static_cast<RandomGeneratorImpl*>(impl_);
        LFS_ASSERT_MSG(device == Device::CPU || device == Device::CUDA,
                       "random generator received an invalid device");
        if (device == Device::CUDA) {
            return impl->cuda_generator_;
        } else {
            return &impl->cpu_generator_;
        }
    }

    // ============= In-place Random Operations =============

    Tensor& Tensor::uniform_(float low, float high) {
        LFS_ASSERT_MSG(is_valid(),
                       "uniform_ requires a valid tensor");
        LFS_ASSERT_MSG(dtype_ == DataType::Float32,
                       "uniform_ requires Float32 dtype");
        LFS_ASSERT_MSG(std::isfinite(low) && std::isfinite(high) && low <= high,
                       "uniform_ bounds must be finite and ordered");
        if (numel() == 0) {
            return *this;
        }

        if (!is_contiguous()) {
            return mutate_logical_view(
                [&](Tensor& materialized) {
                    materialized.uniform_(low, high);
                });
        }

        size_t n = numel();

        if (device_ == Device::CUDA) {
            // Use kernel-based generation with advancing seed
            uint64_t seed = RandomGenerator::instance().get_next_cuda_seed();
            tensor_ops::launch_uniform(ptr<float>(), n, low, high, seed, stream());
            // No sync - in-place operation returns *this
        } else {
            // CPU uses stateful generator
            auto* impl = static_cast<RandomGeneratorImpl*>(
                RandomGenerator::instance().get_impl());
            std::uniform_real_distribution<float> dist(low, high);

            float* data = ptr<float>();
            for (size_t i = 0; i < n; ++i) {
                data[i] = dist(impl->cpu_generator_);
            }
        }

        return *this;
    }

    Tensor& Tensor::normal_(float mean, float std) {
        LFS_ASSERT_MSG(is_valid(),
                       "normal_ requires a valid tensor");
        LFS_ASSERT_MSG(dtype_ == DataType::Float32,
                       "normal_ requires Float32 dtype");
        LFS_ASSERT_MSG(std::isfinite(mean) && std::isfinite(std) && std >= 0.0f,
                       "normal_ mean and standard deviation must be finite, with std >= 0");
        if (numel() == 0) {
            return *this;
        }

        if (std == 0.0f) {
            return fill_(mean);
        }

        if (!is_contiguous()) {
            return mutate_logical_view(
                [&](Tensor& materialized) {
                    materialized.normal_(mean, std);
                });
        }

        size_t n = numel();

        if (device_ == Device::CUDA) {
            // curandGenerateNormal requires even number of elements
            if (n % 2 == 1) {
                // Generate into an n+1 scratch allocation. Writing n+1 values into
                // the n-element destination was a one-float buffer overflow.
                auto scratch = Tensor::empty({n + 1}, Device::CUDA, DataType::Float32);
                RandomGenerator::instance().generate_cuda_normal(
                    scratch.ptr<float>(), n + 1, mean, std, stream());
                LFS_CUDA_CHECK(cudaMemcpyAsync(ptr<float>(), scratch.ptr<float>(), n * sizeof(float),
                                               cudaMemcpyDeviceToDevice, stream()));
                LFS_CUDA_CHECK(cudaStreamSynchronize(stream()));
            } else {
                RandomGenerator::instance().generate_cuda_normal(
                    ptr<float>(), n, mean, std, stream());
            }
        } else {
            // CPU uses stateful generator
            auto* impl = static_cast<RandomGeneratorImpl*>(
                RandomGenerator::instance().get_impl());
            std::normal_distribution<float> dist(mean, std);

            float* data = ptr<float>();
            for (size_t i = 0; i < n; ++i) {
                data[i] = dist(impl->cpu_generator_);
            }
        }

        return *this;
    }

#undef CHECK_CURAND

} // namespace lfs::core
