/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "backend/random_state.hpp"
#include "core/logger.hpp"
#include "core/tensor_fused.hpp"
#include "core/tensor_random.hpp"
#include "internal/tensor_impl.hpp"
#include <atomic>
#include <format>
#include <mutex>
#include <random>

namespace lfs::core {

    Tensor random_from_indices(const Tensor& indices, const uint32_t seed) {
        LFS_ASSERT_MSG(indices.is_valid() && indices.ndim() == 1 && indices.dtype() == DataType::Int32,
                       std::format("random_from_indices requires Int32 [N] (valid={}, rank={}, dtype={})",
                                   indices.is_valid(), indices.ndim(), int(indices.dtype())));
        if (!indices.numel())
            return internal::allocate_like(indices, indices.shape(), DataType::Float32);
        if (indices.device() == Device::GPU) {
            static const auto kernel = [] {
                fused::Builder builder(1);
                const auto index = builder.input(DataType::Int32, 1);
                const auto key = builder.input(DataType::UInt32, 1);
                auto hash = index.load().cast(DataType::UInt32) ^ key.at({0});
                hash = (hash ^ (hash >> 16u)) * 0x7feb352du;
                hash = (hash ^ (hash >> 15u)) * 0x846ca68bu;
                hash = hash ^ (hash >> 16u);
                builder.output((hash >> 8u).cast(DataType::Float32) * 0x1p-24f, DataType::Float32);
                return fused::Kernel(builder);
            }();
            auto key = Tensor::empty({1}, Device::CPU, DataType::UInt32);
            key.ptr<uint32_t>()[0] = seed;
            return kernel({indices.numel()}, {indices, key})[0];
        }
        const auto contiguous = indices.contiguous();
        auto output = Tensor::empty(indices.shape(), Device::CPU);
        const auto* source = contiguous.ptr<int32_t>();
        auto* destination = output.ptr<float>();
        for (size_t i = 0; i < indices.numel(); ++i) {
            uint32_t hash = static_cast<uint32_t>(source[i]) ^ seed;
            hash = (hash ^ (hash >> 16u)) * 0x7feb352du;
            hash = (hash ^ (hash >> 15u)) * 0x846ca68bu;
            hash ^= hash >> 16u;
            destination[i] = static_cast<float>(hash >> 8u) * 0x1p-24f;
        }
        return output;
    }

    // ============= RandomGenerator Implementation =============

    // Private implementation class to hide atomic counter
    class RandomGeneratorImpl {
    public:
        std::atomic<uint64_t> call_counter_{0};
        uint64_t seed_ = 42;
        std::unique_ptr<internal::NativeRandomState> native_;
        std::mt19937_64 cpu_generator_;
        std::mutex generator_mutex_;

        RandomGeneratorImpl() : seed_(42),
                                cpu_generator_(seed_) {}

        internal::NativeRandomState& native(const char* unavailable_message) {
            if (!native_)
                native_ = internal::make_native_random_state(seed_, unavailable_message);
            return *native_;
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
        std::lock_guard lock(impl->generator_mutex_);
        impl->seed_ = seed;
        impl->cpu_generator_.seed(seed);
        impl->call_counter_.store(0);
        if (impl->native_)
            impl->native_->reseed(seed);
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
        auto* impl = static_cast<RandomGeneratorImpl*>(impl_);
        std::lock_guard lock(impl->generator_mutex_);
        impl->native("CUDA random generation is not available in this build").normal(output, count, mean, std, stream);
    }

    void* RandomGenerator::get_generator(Device device) {
        auto* impl = static_cast<RandomGeneratorImpl*>(impl_);
        LFS_ASSERT_MSG(device == Device::CPU || device == Device::GPU,
                       "random generator received an invalid device");
        if (device == Device::GPU) {
            std::lock_guard lock(impl->generator_mutex_);
            return impl->native("CUDA random generator access is not available in this build").handle_address();
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
            internal::backend_ops_for(*this).normal(
                internal::storage_ref(*this), program, internal::ExecContext{stream()});
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
