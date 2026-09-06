/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_backend.hpp"

#include "core/logger.hpp"
#include "internal/tensor_impl.hpp"

#include <array>
#include <atomic>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <format>
#include <string>
#ifdef LFS_TENSOR_VULKAN
#include "backend/gpu_backend_ops.hpp"
#include "backend/vulkan/vk_context.hpp"
#include "backend/vulkan/vk_cuda_bridge.hpp"
#include "backend/vulkan/vk_memory.hpp"
#include "backend/vulkan/vk_recorder.hpp"
#include "core/cuda_error.hpp"
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
#ifdef LFS_TENSOR_VULKAN
            if (backend == GpuBackend::Vulkan && internal::vulkan_backend_lost()) {
                throw lfs::Exception(lfs::make_error(lfs::ErrorInit{
                    .code = lfs::ErrorCode::DeviceLost,
                    .domain = lfs::ErrorDomain::Vulkan,
                    .user_message = "Vulkan tensor backend device lost; shut the backend down "
                                    "to create a new context",
                    .detection = LFS_SOURCE_SITE_CURRENT(),
                }));
            }
#endif
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
        return internal::gpu_backend_tag(tensor);
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

    lfs::Status adopt_vulkan_device(const VulkanDeviceHandles& handles) {
#ifdef LFS_TENSOR_VULKAN
        return internal::adopt_vulkan_context(internal::AdoptedDevice{
            .instance = static_cast<VkInstance>(handles.instance),
            .physical_device = static_cast<VkPhysicalDevice>(handles.physical_device),
            .device = static_cast<VkDevice>(handles.device),
            .queue = static_cast<VkQueue>(handles.queue),
            .queue_family = handles.queue_family,
            .shader_atomic_float = handles.shader_atomic_float,
            .memory_budget = handles.memory_budget,
            .shader_float16 = handles.shader_float16,
            .external_memory = handles.external_memory,
            .external_semaphore = handles.external_semaphore,
        });
#else
        (void)handles;
        return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
            .code = lfs::ErrorCode::Unsupported,
            .domain = lfs::ErrorDomain::Vulkan,
            .user_message = "this build has no Vulkan tensor backend",
            .detection = LFS_SOURCE_SITE_CURRENT(),
        }));
#endif
    }

    bool vulkan_backend_adopted() {
#ifdef LFS_TENSOR_VULKAN
        return internal::vulkan_context_adopted();
#else
        return false;
#endif
    }

    void* vulkan_backend_timeline() {
#ifdef LFS_TENSOR_VULKAN
        const auto context = internal::try_live_vulkan_context();
        if (!context) {
            return nullptr;
        }
        const VkSemaphore timeline = context->timeline();
        if (timeline == VK_NULL_HANDLE) {
            return nullptr;
        }
        return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(timeline));
#else
        return nullptr;
#endif
    }

    std::optional<TensorVulkanBuffer> tensor_vulkan_buffer(const Tensor& tensor) {
#ifdef LFS_TENSOR_VULKAN
        if (!tensor.is_valid()) {
            return std::nullopt;
        }
        tensor.materialize_if_deferred();
        if (gpu_backend_of(tensor) != GpuBackend::Vulkan) {
            return std::nullopt;
        }
        const auto context = internal::try_live_vulkan_context();
        if (!context) {
            return std::nullopt;
        }
        const internal::StorageRef storage = internal::storage_ref(tensor);
        if (storage.meta == nullptr ||
            storage.backend != GpuBackend::Vulkan ||
            storage.meta->gpu_descriptor.native_buffer == 0) {
            return std::nullopt;
        }
        context->recorders().flush_storage(storage);
        TensorVulkanBuffer result;
        result.buffer = reinterpret_cast<void*>(static_cast<uintptr_t>(
            storage.meta->gpu_descriptor.native_buffer));
        result.offset = storage.byte_offset;
        result.device_address =
            storage.meta->gpu_descriptor.base_address + storage.byte_offset;
        result.bytes = tensor.bytes();
        result.pending_timeline_value =
            storage.meta->pending_value.load(std::memory_order_acquire);
        result.keep_alive = tensor.data_owner_;
        return result;
#else
        (void)tensor;
        return std::nullopt;
#endif
    }

    namespace {
#ifdef LFS_TENSOR_VULKAN
        lfs::Error cuda_view_error(const lfs::ErrorCode code,
                                   const lfs::ErrorDomain domain,
                                   std::string message) {
            return lfs::make_error(lfs::ErrorInit{
                .code = code,
                .domain = domain,
                .user_message = std::move(message),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }

        std::string uuid_hex(const std::array<uint8_t, 16>& uuid) {
            std::string text;
            text.resize(32);
            static constexpr char kHex[] = "0123456789abcdef";
            for (size_t i = 0; i < uuid.size(); ++i) {
                text[i * 2] = kHex[uuid[i] >> 4];
                text[i * 2 + 1] = kHex[uuid[i] & 0x0f];
            }
            return text;
        }

        struct CudaVulkanViewOwner {
            std::shared_ptr<void> vulkan_keep_alive;
            std::shared_ptr<internal::VulkanCudaMemoryImport> import;
            void* registered = nullptr;

            ~CudaVulkanViewOwner() {
                if (registered != nullptr) {
                    unregister_cuda_address_range(registered);
                }
            }
        };
#endif
    } // namespace

    bool vulkan_backend_exports_memory() {
#ifdef LFS_TENSOR_VULKAN
        try {
            const auto context = internal::try_live_vulkan_context();
            if (!context || context->dead() || !context->cuda_imports()) {
                return false;
            }
            return context->memory().exports_memory() &&
                   context->cuda_imports()->timeline_imported();
        } catch (...) {
            return false;
        }
#else
        return false;
#endif
    }

    lfs::Result<Tensor> cuda_view_of_vulkan_tensor(const Tensor& tensor,
                                                   const cudaStream_t stream) {
#ifdef LFS_TENSOR_VULKAN
        if (!tensor.is_valid()) {
            return cuda_view_error(lfs::ErrorCode::InvalidArgument, lfs::ErrorDomain::Tensor,
                                   "CUDA view of a Vulkan tensor requires a valid tensor");
        }
        const auto vulkan = tensor_vulkan_buffer(tensor);
        if (!vulkan) {
            return cuda_view_error(
                lfs::ErrorCode::InvalidArgument, lfs::ErrorDomain::Tensor,
                gpu_backend_of(tensor) != GpuBackend::Vulkan
                    ? "CUDA view requires a Vulkan-backend tensor"
                    : "CUDA view could not read the Vulkan tensor buffer");
        }
        if (!tensor.is_contiguous()) {
            return cuda_view_error(
                lfs::ErrorCode::InvalidArgument, lfs::ErrorDomain::Tensor,
                "CUDA view of a Vulkan tensor requires contiguous storage");
        }
        if (!vulkan_backend_exports_memory()) {
            return cuda_view_error(
                lfs::ErrorCode::Unsupported, lfs::ErrorDomain::Vulkan,
                "Vulkan tensor backend is not exporting device memory for CUDA");
        }
        const auto context = internal::try_live_vulkan_context();
        if (!context || context->dead() || !context->cuda_imports()) {
            return cuda_view_error(lfs::ErrorCode::FailedPrecondition, lfs::ErrorDomain::Vulkan,
                                   "Vulkan tensor backend is not live");
        }
        const std::array<uint8_t, 16>& vulkan_uuid = context->caps().device_uuid;
        int cuda_device = 0;
        cudaError_t status = cudaGetDevice(&cuda_device);
        if (status != cudaSuccess) {
            (void)cudaGetLastError();
            return cuda_view_error(lfs::ErrorCode::Internal, lfs::ErrorDomain::CUDA,
                                   std::format("cudaGetDevice failed: {}",
                                               cudaGetErrorString(status)));
        }
        cudaDeviceProp properties{};
        status = cudaGetDeviceProperties(&properties, cuda_device);
        if (status != cudaSuccess) {
            (void)cudaGetLastError();
            return cuda_view_error(lfs::ErrorCode::Internal, lfs::ErrorDomain::CUDA,
                                   std::format("cudaGetDeviceProperties failed: {}",
                                               cudaGetErrorString(status)));
        }
        std::array<uint8_t, 16> cuda_uuid{};
        std::memcpy(cuda_uuid.data(), properties.uuid.bytes, cuda_uuid.size());
        if (cuda_uuid != vulkan_uuid) {
            return cuda_view_error(
                lfs::ErrorCode::InvalidArgument, lfs::ErrorDomain::CUDA,
                std::format(
                    "CUDA device {} (UUID {}) does not match the Vulkan tensor device (UUID {})",
                    cuda_device, uuid_hex(cuda_uuid), uuid_hex(vulkan_uuid)));
        }

        const internal::StorageRef storage = internal::storage_ref(tensor);
        const auto block = context->memory().cuda_block_info(storage);
        if (!block) {
            return cuda_view_error(lfs::ErrorCode::Internal, lfs::ErrorDomain::Vulkan,
                                   "CUDA view could not resolve the Vulkan memory block");
        }
        if (block->host_visible) {
            return cuda_view_error(
                lfs::ErrorCode::Unsupported, lfs::ErrorDomain::Vulkan,
                "CUDA view of host-visible Vulkan readback storage is not supported");
        }
        if (!block->exportable) {
            return cuda_view_error(
                lfs::ErrorCode::Unsupported, lfs::ErrorDomain::Vulkan,
                "CUDA view needs pool-backed Vulkan storage; direct (large) allocations are not exportable");
        }
        auto imported = context->cuda_imports()->import_memory(
            block->memory, block->block_size, block->dedicated);
        if (!imported) {
            return imported.error();
        }
        const auto wait = context->cuda_imports()->wait_timeline(vulkan->pending_timeline_value,
                                                                 stream);
        if (!wait) {
            return wait.error();
        }

        auto owner = std::make_shared<CudaVulkanViewOwner>();
        owner->vulkan_keep_alive = vulkan->keep_alive;
        owner->import = *imported;
        void* const data = static_cast<std::byte*>((*imported)->mapped) +
                           block->allocation_offset + storage.byte_offset;
        register_cuda_address_range(data, tensor.bytes(), "vulkan.cuda_view");
        owner->registered = data;
        return Tensor::from_external_owner(data, tensor.shape(), Device::CUDA, tensor.dtype(),
                                           std::move(owner), 0, stream, "vulkan.cuda_view");
#else
        (void)tensor;
        (void)stream;
        return lfs::make_error(lfs::ErrorInit{
            .code = lfs::ErrorCode::Unsupported,
            .domain = lfs::ErrorDomain::Vulkan,
            .user_message = "this build has no Vulkan tensor backend",
            .detection = LFS_SOURCE_SITE_CURRENT(),
        });
#endif
    }

    namespace internal {

        void order_legacy_after_home(const Tensor& tensor) {
            if (tensor.device() != Device::CUDA || tensor.stream() == nullptr) {
                return;
            }
            backend_ops_for(tensor).bridge(ExecContext{tensor.stream()}, ExecContext{nullptr});
        }

        void order_home_after_legacy(const Tensor& tensor) {
            if (tensor.device() != Device::CUDA || tensor.stream() == nullptr) {
                return;
            }
            backend_ops_for(tensor).bridge(ExecContext{nullptr}, ExecContext{tensor.stream()});
        }

        void trim_live_gpu_backends() {
            backend_ops(GpuBackend::CUDA).trim();
#ifdef LFS_TENSOR_VULKAN
            if (vulkan_backend_live()) {
                backend_ops(GpuBackend::Vulkan).trim();
            }
#endif
        }

        void trim_live_gpu_backends_if_reserved_unused_exceeds(const size_t threshold_bytes) {
            backend_ops(GpuBackend::CUDA).trim_if_reserved_unused_exceeds(threshold_bytes);
#ifdef LFS_TENSOR_VULKAN
            if (vulkan_backend_live()) {
                backend_ops(GpuBackend::Vulkan).trim_if_reserved_unused_exceeds(threshold_bytes);
            }
#endif
        }

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

        void throw_gpu_backend_mismatch(const Tensor& reference,
                                        const Tensor& other,
                                        const std::string_view operation) {
            const auto reference_backend = gpu_backend_of(reference);
            const auto other_backend = gpu_backend_of(other);
            throw TensorError(std::format(
                "{} requires matching GPU backends, got {} and {}",
                operation,
                reference_backend ? gpu_backend_name(*reference_backend) : "CPU",
                other_backend ? gpu_backend_name(*other_backend) : "CPU"));
        }

        LFS_CORE_API void gpu_backend_reset_for_testing() {
            process_backend_state.store(kUnconfigured, std::memory_order_release);
        }

    } // namespace internal

} // namespace lfs::core
