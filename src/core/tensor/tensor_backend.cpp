/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_backend.hpp"

#include "core/logger.hpp"
#include "internal/tensor_impl.hpp"

#include <atomic>
#include <cctype>
#include <cstdlib>
#include <cuda_runtime.h>
#include <format>
#include <string>
#ifdef LFS_TENSOR_VULKAN
#include "backend/gpu_backend_ops.hpp"
#include "backend/vulkan/vk_context.hpp"
#endif

namespace lfs::core {
    namespace {

        constexpr int kUnconfigured = -1;
        constexpr int kConfiguredCuda = -2;
        constexpr int kConfiguredVulkan = -3;

        std::atomic<int> process_backend_state{kUnconfigured};
        thread_local std::optional<GpuBackend> scoped_backend;

        bool is_resolved(const int state) {
            return state == static_cast<int>(GpuBackend::CUDA) ||
                   state == static_cast<int>(GpuBackend::Vulkan);
        }

        GpuBackend configured_backend(const int state) {
            return state == kConfiguredVulkan ? GpuBackend::Vulkan : GpuBackend::CUDA;
        }

        int configured_state(const GpuBackend backend) {
            return backend == GpuBackend::Vulkan ? kConfiguredVulkan : kConfiguredCuda;
        }

        std::optional<GpuBackend> backend_from_environment() {
            const char* value = std::getenv("LFS_TENSOR_BACKEND");
            if (value == nullptr) {
                return std::nullopt;
            }

            std::string normalized(value);
            for (char& character : normalized) {
                character = static_cast<char>(
                    std::tolower(static_cast<unsigned char>(character)));
            }
            if (normalized == "cuda") {
                return GpuBackend::CUDA;
            }
            if (normalized == "vulkan") {
                return GpuBackend::Vulkan;
            }

            LOG_WARN("Ignoring invalid LFS_TENSOR_BACKEND='{}'; using CUDA", value);
            return GpuBackend::CUDA;
        }

        [[noreturn]] void throw_backend_unavailable(const GpuBackend backend) {
            throw TensorError(std::format(
                "GPU backend '{}' is unavailable", gpu_backend_name(backend)));
        }

    } // namespace

    const char* gpu_backend_name(const GpuBackend backend) {
        switch (backend) {
        case GpuBackend::CUDA: return "CUDA";
        case GpuBackend::Vulkan: return "Vulkan";
        }
        return "Unknown";
    }

    GpuBackend default_gpu_backend() {
        for (;;) {
            int state = process_backend_state.load(std::memory_order_acquire);
            if (is_resolved(state)) {
                return static_cast<GpuBackend>(state);
            }

            const GpuBackend selected = state == kUnconfigured
                                            ? backend_from_environment().value_or(GpuBackend::CUDA)
                                            : configured_backend(state);
            if (process_backend_state.compare_exchange_weak(
                    state, static_cast<int>(selected),
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                return selected;
            }
        }
    }

    lfs::Status set_default_gpu_backend(const GpuBackend backend) {
        int state = process_backend_state.load(std::memory_order_acquire);
        for (;;) {
            if (is_resolved(state)) {
                const auto frozen = static_cast<GpuBackend>(state);
                return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
                    .code = lfs::ErrorCode::FailedPrecondition,
                    .domain = lfs::ErrorDomain::Core,
                    .user_message = std::format(
                        "GPU backend default is frozen as {}; cannot change it to {}",
                        gpu_backend_name(frozen), gpu_backend_name(backend)),
                    .detection = LFS_SOURCE_SITE_CURRENT(),
                }));
            }
            if (process_backend_state.compare_exchange_weak(
                    state, configured_state(backend),
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                return {};
            }
        }
    }

    bool gpu_backend_available(const GpuBackend backend) {
        if (backend == GpuBackend::Vulkan) {
#ifdef LFS_TENSOR_VULKAN
            return internal::vulkan_backend_probe_available();
#else
            return false;
#endif
        }

        static const bool cuda_available = [] {
            int device_count = 0;
            const cudaError_t status = cudaGetDeviceCount(&device_count);
            if (status != cudaSuccess) {
                (void)cudaGetLastError();
                return false;
            }
            return device_count > 0;
        }();
        return cuda_available;
    }

    std::optional<GpuBackend> gpu_backend_of(const Tensor& tensor) {
        if (!tensor.is_valid() || tensor.device_ != Device::CUDA) {
            return std::nullopt;
        }
        return tensor.storage_meta_ ? tensor.storage_meta_->backend : GpuBackend::CUDA;
    }

    GpuBackendScope::GpuBackendScope(const GpuBackend backend)
        : previous_(scoped_backend) {
        scoped_backend = backend;
    }

    GpuBackendScope::~GpuBackendScope() {
        scoped_backend = previous_;
    }

    MemoryInfo gpu_backend_memory_info(const GpuBackend backend) {
        if (backend == GpuBackend::CUDA) {
            return MemoryInfo::cuda();
        }
#ifdef LFS_TENSOR_VULKAN
        return internal::backend_ops(GpuBackend::Vulkan).stats();
#else
        return {};
#endif
    }

    lfs::Status shutdown_gpu_backend(const GpuBackend backend) {
#ifdef LFS_TENSOR_VULKAN
        if (backend == GpuBackend::Vulkan) {
            try {
                internal::backend_ops(backend).shutdown();
            } catch (...) {
                return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
                    .code = lfs::ErrorCode::Internal,
                    .domain = lfs::ErrorDomain::Vulkan,
                    .user_message = "Vulkan tensor backend shutdown failed",
                    .detection = LFS_SOURCE_SITE_CURRENT(),
                }));
            }
        }
#else
        (void)backend;
#endif
        return {};
    }

    namespace internal {

        GpuBackend resolve_new_gpu_storage_backend() {
            const GpuBackend backend = scoped_backend ? *scoped_backend : default_gpu_backend();
            if (!gpu_backend_available(backend)) {
                throw_backend_unavailable(backend);
            }
            return backend;
        }

        Tensor allocate_like(const Tensor& input,
                             const TensorShape& shape,
                             const DataType dtype) {
            LFS_ASSERT_MSG(input.is_valid(), "allocate_like requires a valid input tensor");
            if (input.device() == Device::CPU) {
                return Tensor::empty(shape, Device::CPU, dtype);
            }

            const GpuBackend backend = gpu_backend_of(input).value();
            GpuBackendScope scope(backend);
            return Tensor::empty(shape, Device::CUDA, dtype);
        }

        Tensor allocate_like(const Tensor& input,
                             const TensorShape& shape,
                             const DataType dtype,
                             const float value) {
            LFS_ASSERT_MSG(input.is_valid(), "allocate_like requires a valid input tensor");
            if (input.device() == Device::CPU) {
                return Tensor::full(shape, value, Device::CPU, dtype);
            }

            const GpuBackend backend = gpu_backend_of(input).value();
            GpuBackendScope scope(backend);
            return Tensor::full(shape, value, Device::CUDA, dtype);
        }

        Tensor allocate_zeros_like(const Tensor& input,
                                   const TensorShape& shape,
                                   const DataType dtype) {
            LFS_ASSERT_MSG(input.is_valid(), "allocate_zeros_like requires a valid input tensor");
            if (input.device() == Device::CPU) {
                return Tensor::zeros(shape, Device::CPU, dtype);
            }

            GpuBackendScope scope(gpu_backend_of(input).value());
            return Tensor::zeros(shape, Device::CUDA, dtype);
        }

        Tensor allocate_ones_like(const Tensor& input,
                                  const TensorShape& shape,
                                  const DataType dtype) {
            LFS_ASSERT_MSG(input.is_valid(), "allocate_ones_like requires a valid input tensor");
            if (input.device() == Device::CPU) {
                return Tensor::ones(shape, Device::CPU, dtype);
            }

            GpuBackendScope scope(gpu_backend_of(input).value());
            return Tensor::ones(shape, Device::CUDA, dtype);
        }

        Tensor allocate_rand_like(const Tensor& input,
                                  const TensorShape& shape,
                                  const DataType dtype) {
            LFS_ASSERT_MSG(input.is_valid(), "allocate_rand_like requires a valid input tensor");
            if (input.device() == Device::CPU) {
                return Tensor::rand(shape, Device::CPU, dtype);
            }

            GpuBackendScope scope(gpu_backend_of(input).value());
            return Tensor::rand(shape, Device::CUDA, dtype);
        }

        Tensor allocate_randn_like(const Tensor& input,
                                   const TensorShape& shape,
                                   const DataType dtype) {
            LFS_ASSERT_MSG(input.is_valid(), "allocate_randn_like requires a valid input tensor");
            if (input.device() == Device::CPU) {
                return Tensor::randn(shape, Device::CPU, dtype);
            }

            GpuBackendScope scope(gpu_backend_of(input).value());
            return Tensor::randn(shape, Device::CUDA, dtype);
        }

        Tensor copy_to_backend(const Tensor& source, const GpuBackend target) {
            LFS_ASSERT_MSG(source.is_valid(), "copy_to_backend requires a valid source tensor");
            if (gpu_backend_of(source) == target) {
                return source.clone();
            }

            GpuBackendScope scope(target);
            if (source.device() == Device::CPU) {
                return source.to(Device::CUDA);
            }
            return source.to(Device::CPU).to(Device::CUDA);
        }

        void require_same_gpu_backend(const Tensor& reference,
                                      const Tensor& other,
                                      const std::string_view operation) {
            const auto reference_backend = gpu_backend_of(reference);
            const auto other_backend = gpu_backend_of(other);
            if (!reference_backend || !other_backend || *reference_backend == *other_backend) {
                return;
            }
            LFS_ASSERT_MSG(
                false,
                std::format(
                    "{} requires matching GPU backends, got {} and {}",
                    operation, gpu_backend_name(*reference_backend),
                    gpu_backend_name(*other_backend)));
        }

        LFS_CORE_API void gpu_backend_reset_for_testing() {
            process_backend_state.store(kUnconfigured, std::memory_order_release);
        }

    } // namespace internal

} // namespace lfs::core
