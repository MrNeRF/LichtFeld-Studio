/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "internal/tensor_impl.hpp"
#include "internal/tensor_ops.hpp"
#include <atomic>
#include <curand.h>
#include <curand_kernel.h>
#include <random>

#define CHECK_CUDA(call)                                        \
    do {                                                        \
        const cudaError_t error = (call);                       \
        LFS_ASSERT_MSG(error == cudaSuccess,                    \
                       std::string("CUDA operation failed: ") + \
                           cudaGetErrorString(error));          \
    } while (0)

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
        uint64_t seed_ = 42;
        void* cuda_generator_ = nullptr;
        std::mt19937_64 cpu_generator_;

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
        impl->seed_ = seed;
        impl->cpu_generator_.seed(seed);
        impl->call_counter_.store(0); // Reset call counter when seed is set

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

    uint64_t RandomGenerator::get_next_cuda_offset() {
        auto* impl = static_cast<RandomGeneratorImpl*>(impl_);
        uint64_t counter = impl->call_counter_.fetch_add(1);
        return counter * 1000000ULL;
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
        LFS_ASSERT_MSG(is_valid(), "uniform_ requires a valid tensor");
        LFS_ASSERT_MSG(dtype_ == DataType::Float32, "uniform_ requires Float32 dtype");
        LFS_ASSERT_MSG(std::isfinite(low) && std::isfinite(high) && low <= high,
                       "uniform_ bounds must be finite and ordered");
        if (numel() == 0) {
            return *this;
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
        LFS_ASSERT_MSG(is_valid(), "normal_ requires a valid tensor");
        LFS_ASSERT_MSG(dtype_ == DataType::Float32, "normal_ requires Float32 dtype");
        LFS_ASSERT_MSG(std::isfinite(mean) && std::isfinite(std) && std > 0.0f,
                       "normal_ mean and standard deviation must be finite, with std > 0");
        if (numel() == 0) {
            return *this;
        }

        size_t n = numel();

        if (device_ == Device::CUDA) {
            // OPTIMIZATION: Use curandGenerateNormal for bulk generation (much faster!)
            // This avoids the slow per-element curand_init in the kernel
            curandGenerator_t* gen = static_cast<curandGenerator_t*>(
                RandomGenerator::instance().get_generator(Device::CUDA));

            // Advance the generator offset (not the seed!) for reproducibility
            uint64_t offset = RandomGenerator::instance().get_next_cuda_offset();
            CHECK_CURAND(curandSetGeneratorOffset(*gen, offset));

            // curandGenerateNormal requires even number of elements
            if (n % 2 == 1) {
                // Generate into an n+1 scratch allocation. Writing n+1 values into
                // the n-element destination was a one-float buffer overflow.
                auto scratch = Tensor::empty({n + 1}, Device::CUDA, DataType::Float32);
                CHECK_CURAND(curandGenerateNormal(*gen, scratch.ptr<float>(), n + 1, mean, std));
                CHECK_CUDA(cudaMemcpyAsync(ptr<float>(), scratch.ptr<float>(), n * sizeof(float),
                                           cudaMemcpyDeviceToDevice, stream()));
                CHECK_CUDA(cudaStreamSynchronize(stream()));
            } else {
                CHECK_CURAND(curandGenerateNormal(*gen, ptr<float>(), n, mean, std));
            }
            // Note: No need for cudaDeviceSynchronize() - curandGenerateNormal is blocking
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

#undef CHECK_CUDA
#undef CHECK_CURAND

} // namespace lfs::core
