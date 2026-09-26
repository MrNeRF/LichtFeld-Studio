/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "internal/tensor_impl.hpp"
#include <atomic>
#if LFS_HAS_CUDA
#include <curand.h>
#include <curand_kernel.h>
#endif
#include <mutex>
#include <random>

#if LFS_HAS_CUDA
#define CHECK_CURAND(call)                                                   \
    do {                                                                     \
        const curandStatus_t error = (call);                                 \
        LFS_ASSERT_MSG(error == CURAND_STATUS_SUCCESS,                       \
                       std::string("CURAND operation failed with status ") + \
                           std::to_string(static_cast<int>(error)));         \
    } while (0)
#endif

namespace lfs::core {

    // ============= RandomGenerator Implementation =============

    // Private implementation class to hide atomic counter
    class RandomGeneratorImpl {
    public:
        std::atomic<uint64_t> call_counter_{0};
        uint64_t cuda_offset_ = 0;
        uint64_t seed_ = 42;
#if LFS_HAS_CUDA
        curandGenerator_t cuda_generator_ = nullptr;
#endif
        std::mt19937_64 cpu_generator_;
        std::mutex cuda_mutex_;

        RandomGeneratorImpl() : seed_(42),
                                cpu_generator_(seed_) {}

        void ensure_cuda_generator() {
#if LFS_HAS_CUDA
            if (cuda_generator_)
                return;
            CHECK_CURAND(curandCreateGenerator(&cuda_generator_, CURAND_RNG_PSEUDO_PHILOX4_32_10));
            const auto status = curandSetPseudoRandomGeneratorSeed(cuda_generator_, seed_);
            if (status != CURAND_STATUS_SUCCESS) {
                curandDestroyGenerator(cuda_generator_);
                cuda_generator_ = nullptr;
                CHECK_CURAND(status);
            }
#else
            throw TensorError("CUDA random generation is not available in this build");
#endif
        }

        ~RandomGeneratorImpl() {
#if LFS_HAS_CUDA
            if (cuda_generator_)
                curandDestroyGenerator(cuda_generator_);
#endif
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

#if LFS_HAS_CUDA
        if (impl->cuda_generator_) {
            CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(impl->cuda_generator_, seed));
            // IMPORTANT: Reset the offset to ensure reproducibility
            CHECK_CURAND(curandSetGeneratorOffset(impl->cuda_generator_, 0));
        }
#endif
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
#if LFS_HAS_CUDA
        LFS_ASSERT_MSG(output != nullptr,
                       "CUDA normal generation requires a valid output pointer");
        LFS_ASSERT_MSG(count > 0 && count % 2 == 0,
                       "CUDA normal generation requires a positive even value count");
        auto* impl = static_cast<RandomGeneratorImpl*>(impl_);
        std::lock_guard lock(impl->cuda_mutex_);
        LFS_ASSERT_MSG(count <= std::numeric_limits<uint64_t>::max() - impl->cuda_offset_,
                       "CUDA random generator offset overflow");

        impl->ensure_cuda_generator();
        impl->cuda_offset_ += count;
        CHECK_CURAND(curandSetStream(impl->cuda_generator_, stream));
        CHECK_CURAND(curandGenerateNormal(impl->cuda_generator_, output, count, mean, std));
#else
        throw TensorError("CUDA random generation is not available in this build");
#endif
    }

    void* RandomGenerator::get_generator(Device device) {
        auto* impl = static_cast<RandomGeneratorImpl*>(impl_);
        LFS_ASSERT_MSG(device == Device::CPU || device == Device::GPU,
                       "random generator received an invalid device");
        if (device == Device::GPU) {
#if LFS_HAS_CUDA
            std::lock_guard lock(impl->cuda_mutex_);
            impl->ensure_cuda_generator();
            return &impl->cuda_generator_;
#else
            throw TensorError("CUDA random generator access is not available in this build");
#endif
        } else {
            return &impl->cpu_generator_;
        }
    }

    // ============= In-place Random Operations =============

    Tensor& Tensor::uniform_(float low, float high) {

        preserve_lazy_snapshots_before_write();
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

        if (device_ == Device::GPU) {
            // Use kernel-based generation with advancing seed
            uint64_t seed = RandomGenerator::instance().get_next_cuda_seed();
            internal::backend_ops_for(*this).uniform(
                internal::storage_ref(*this),
                internal::RandomProgram{
                    .count = n,
                    .first = low,
                    .second = high,
                    .seed = seed,
                },
                internal::ExecContext{stream()});
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

        preserve_lazy_snapshots_before_write();
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

        if (device_ == Device::GPU) {
            const uint64_t seed = RandomGenerator::instance().get_next_cuda_seed();
            const internal::RandomProgram program{
                .count = n,
                .first = mean,
                .second = std,
                .seed = seed};
            if (n % 2 == 1) {
                auto scratch = internal::allocate_like(
                    *this, TensorShape{n + 1}, DataType::Float32);
                internal::backend_ops_for(*this).normal(
                    internal::storage_ref(*this), internal::storage_ref(scratch),
                    program, internal::ExecContext{stream()});
            } else {
                internal::backend_ops_for(*this).normal(
                    internal::storage_ref(*this), internal::storage_ref(*this),
                    program, internal::ExecContext{stream()});
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
