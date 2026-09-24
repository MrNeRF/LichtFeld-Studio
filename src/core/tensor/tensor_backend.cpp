/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_backend.hpp"
#include "backend/tensor_completion.hpp"
#include "backend/tensor_vulkan_interop.hpp"
#include "core/tensor_completion.hpp"

#include "backend/cuda/runtime/cuda_event_pool.hpp"
#include "core/cuda_error.hpp"
#include "core/device_fault.hpp"
#include "core/gpu_device_info.hpp"
#include "core/logger.hpp"
#include "internal/tensor_impl.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <exception>
#include <format>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#ifdef LFS_TENSOR_VULKAN
#include "backend/gpu_backend_ops.hpp"
#include "backend/vulkan/vk_context.hpp"
#include "backend/vulkan/vk_cuda_bridge.hpp"
#include "backend/vulkan/vk_memory.hpp"
#include "backend/vulkan/vk_recorder.hpp"
#endif

namespace lfs::core {
    Tensor Tensor::empty_like(const Tensor& other, const TensorShape& shape, DataType dtype) {
        return internal::allocate_like(other, shape, dtype);
    }

    Tensor Tensor::to(const GpuBackend backend) const {
        return internal::copy_to_backend(*this, backend);
    }

    struct TensorVulkanInterop::Impl {
        VulkanInteropDevice device;
        std::mutex mutex;
        std::array<std::shared_ptr<internal::TensorVulkanInteropBackend>, kGpuBackendCount> backends;

        internal::TensorVulkanInteropBackend& backend(GpuBackend backend) {
            std::lock_guard lock(mutex);
            auto& result = backends[static_cast<size_t>(backend)];
            if (!result) {
#ifdef LFS_TENSOR_VULKAN
                result = backend == GpuBackend::CUDA
                             ? internal::make_cuda_vulkan_interop(device)
                             : internal::make_vulkan_vulkan_interop(device);
#else
                throw TensorError("Vulkan tensor interop is unavailable in this build");
#endif
            }
            return *result;
        }
    };

    TensorVulkanInterop::TensorVulkanInterop(VulkanInteropDevice device)
        : impl_(std::make_unique<Impl>()) {
        if (!device.device || !device.physical_device)
            throw TensorError("Vulkan tensor interop requires a live consumer device");
        impl_->device = device;
    }

    TensorVulkanInterop::~TensorVulkanInterop() {
        for (auto& backend : impl_->backends) {
            if (backend)
                backend->shutdown();
        }
    }

    std::function<Tensor(TensorShape, size_t, DataType, std::string_view)>
    TensorVulkanInterop::splat_allocator(bool preserve_float_shN) {
        const auto backend = internal::resolve_new_gpu_storage_backend();
        return [this, backend, preserve_float_shN](TensorShape shape, size_t capacity,
                                                   DataType dtype, std::string_view name) {
            auto tensor = impl_->backend(backend).empty_splat(std::move(shape), capacity, dtype, name, preserve_float_shN);
            tensor.set_name(std::string(name));
            return tensor;
        };
    }

    void TensorVulkanInterop::drain(GpuBackend backend) {
        impl_->backend(backend).drain();
    }

    std::shared_ptr<void> TensorVulkanInterop::execution_scope(GpuBackend backend) {
        return impl_->backend(backend).execution_scope();
    }

    Tensor TensorVulkanInterop::empty(TensorShape shape, DataType dtype, GpuBackend backend, size_t capacity) {
        return impl_->backend(backend).empty(std::move(shape), dtype, capacity);
    }

    std::optional<TensorVulkanBuffer> TensorVulkanInterop::buffer(const Tensor& tensor) {
        const auto backend = gpu_backend_of(tensor);
        if (!backend || !tensor.is_valid())
            return std::nullopt;
        return impl_->backend(*backend).buffer(tensor);
    }

    void TensorVulkanInterop::upload_host(Tensor& tensor, std::span<const std::byte> bytes) {
#ifdef LFS_TENSOR_VULKAN
        if (!tensor.is_valid() || tensor.device() != Device::GPU ||
            !tensor.is_contiguous() || bytes.size() > tensor.bytes())
            throw TensorError("Vulkan-visible host upload requires a contiguous GPU tensor with sufficient capacity");
        if (bytes.empty())
            return;
        internal::backend_ops_for(tensor).copy_host_to_device(internal::CopyRequest{
            .src = internal::raw_storage_ref(const_cast<std::byte*>(bytes.data())),
            .dst = internal::storage_ref(tensor),
            .bytes = bytes.size(),
            .synchronous = false,
            .context = internal::ExecContext{tensor.stream()},
        });
#else
        (void)tensor;
        (void)bytes;
        throw TensorError("Vulkan tensor interop is unavailable in this build");
#endif
    }

    TensorCompletion TensorVulkanInterop::ready(std::span<const Tensor* const> tensors) {
        std::optional<GpuBackend> selected;
        for (const auto* tensor : tensors) {
            if (!tensor || !tensor->is_valid() || !gpu_backend_of(*tensor))
                continue;
            const auto backend = *gpu_backend_of(*tensor);
            if (selected && *selected != backend)
                throw TensorError("A Vulkan interop batch must use one tensor backend");
            selected = backend;
        }
        return selected ? impl_->backend(*selected).ready(tensors) : TensorCompletion{};
    }

    void TensorVulkanInterop::release_timeline(void* semaphore) {
        if (!semaphore)
            return;
        std::lock_guard lock(impl_->mutex);
        for (auto& backend : impl_->backends) {
            if (backend)
                backend->release_timeline(semaphore);
        }
    }

    void TensorVulkanInterop::wait(std::span<const Tensor* const> tensors, VulkanTimelinePoint point) {
        if (!point.semaphore || point.value == 0)
            return;
        for (const GpuBackend backend : kGpuBackends) {
            std::vector<const Tensor*> selected;
            for (const auto* tensor : tensors) {
                if (tensor && tensor->is_valid() && gpu_backend_of(*tensor) == backend)
                    selected.push_back(tensor);
            }
            if (!selected.empty())
                impl_->backend(backend).wait(selected, point);
        }
    }

    void where_into(Tensor& output, const Tensor& condition, float value, const Tensor& source) {
        const auto backend = gpu_backend_of(output);
        if (!output.is_valid() || !source.is_valid() || !condition.is_valid() || !backend ||
            gpu_backend_of(source) != backend || gpu_backend_of(condition) != backend ||
            !output.is_contiguous() || !source.is_contiguous() || !condition.is_contiguous() ||
            output.shape() != source.shape() || output.numel() != condition.numel() ||
            output.dtype() != source.dtype() || condition.dtype() != DataType::Bool ||
            (output.dtype() != DataType::Float32 && output.dtype() != DataType::Float16))
            throw TensorError("where_into requires matching contiguous Float32/Float16 GPU tensors and a Bool mask");
        internal::preserve_lazy_snapshots_before_write(output);
        if (output.numel() == 0)
            return;
        if (*backend == GpuBackend::CUDA) {
            internal::cuda_where_into(output, condition, value, source);
        } else {
#ifdef LFS_TENSOR_VULKAN
            internal::vulkan_where_into(output, condition, value, source);
#else
            throw TensorError("Vulkan tensor backend is unavailable");
#endif
        }
    }

    namespace {

        constexpr int kUnconfigured = -1;
        constexpr int kConfiguredBase = -2;

        std::atomic<int> process_backend_state{kUnconfigured};
        thread_local std::optional<GpuBackend> scoped_backend;
        std::mutex options_mutex;
        TensorBackendOptions backend_options;
        bool options_frozen = false;

        bool is_resolved(const int state) {
            return state >= 0 && state < static_cast<int>(kGpuBackendCount);
        }

        GpuBackend configured_backend(const int state) {
            return static_cast<GpuBackend>(kConfiguredBase - state);
        }

        int configured_state(const GpuBackend backend) {
            return kConfiguredBase - static_cast<int>(backend);
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

    lfs::Status set_tensor_backend_options(const TensorBackendOptions& options) {
        std::lock_guard lock(options_mutex);
        if (options_frozen || options.vulkan_validation < 0 || options.vulkan_validation > 2) {
            return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::FailedPrecondition,
                .domain = lfs::ErrorDomain::Core,
                .user_message = options_frozen ? "Tensor backend options require a restart"
                                               : "Invalid Vulkan validation mode",
                .detection = LFS_SOURCE_SITE_CURRENT(),
            }));
        }
        backend_options = options;
        return {};
    }

    TensorBackendOptions tensor_backend_options() {
        std::lock_guard lock(options_mutex);
        options_frozen = true;
        return backend_options;
    }

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
                                            ? GpuBackend::CUDA
                                            : configured_backend(state);
            if (process_backend_state.compare_exchange_weak(
                    state, static_cast<int>(selected),
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                return selected;
            }
        }
    }

    lfs::Status set_default_gpu_backend(const GpuBackend backend) {
        if (std::ranges::find(kGpuBackends, backend) == kGpuBackends.end()) {
            return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::InvalidArgument,
                .domain = lfs::ErrorDomain::Core,
                .user_message = "Unknown tensor GPU backend",
                .detection = LFS_SOURCE_SITE_CURRENT(),
            }));
        }
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

    bool gpu_backend_live(const GpuBackend backend) {
        if (backend == GpuBackend::Vulkan) {
#ifdef LFS_TENSOR_VULKAN
            return internal::vulkan_backend_live();
#else
            return false;
#endif
        }
        if (!gpu_backend_available(GpuBackend::CUDA))
            return false;
        int ordinal = 0;
        CUdevice device{};
        unsigned flags = 0;
        int active = 0;
        if (cudaGetDevice(&ordinal) != cudaSuccess || cuDeviceGet(&device, ordinal) != CUDA_SUCCESS ||
            cuDevicePrimaryCtxGetState(device, &flags, &active) != CUDA_SUCCESS || active == 0)
            return false;
        return true;
    }

    std::optional<GpuBackend> gpu_backend_of(const Tensor& tensor) {
        if (!tensor.is_valid() || tensor.device_ != Device::GPU) {
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

    struct TensorCompletion::Impl {
        std::mutex mutex;
        std::array<bool, kGpuBackendCount> backends{};
        VulkanTimelinePoint point;
        void* device = nullptr;
        bool settle_on_release = true;
        void wait();
        ~Impl();
        std::vector<std::pair<cudaStream_t, cudaEvent_t>> events;
#ifdef LFS_TENSOR_VULKAN
        std::shared_ptr<internal::VulkanContext> context;
        uint64_t value = 0;
#endif

        void capture(const std::span<const Tensor* const> tensors) {
            std::vector<cudaStream_t> streams;
#ifdef LFS_TENSOR_VULKAN
            internal::StorageRef latest{};
            uint64_t pending = 0;
#endif
            for (const auto* tensor : tensors) {
                if (!tensor || !tensor->is_valid())
                    continue;
                const auto backend = gpu_backend_of(*tensor);
                if (!backend)
                    continue;
                const auto storage = internal::storage_ref(*tensor);
                if (*backend == GpuBackend::CUDA) {
                    if (std::find(streams.begin(), streams.end(), tensor->stream()) == streams.end())
                        streams.push_back(tensor->stream());
                }
#ifdef LFS_TENSOR_VULKAN
                if (*backend == GpuBackend::Vulkan) {
                    if (!storage.meta)
                        continue;
                    const auto storage_value = storage.meta->pending_value.load(std::memory_order_acquire);
                    if (storage_value > pending) {
                        latest = storage;
                        pending = storage_value;
                    }
                }
#endif
            }
            for (const auto stream : streams) {
                auto found = std::find_if(events.begin(), events.end(),
                                          [stream](const auto& event) { return event.first == stream; });
                if (found == events.end()) {
                    events.emplace_back(stream, nullptr);
                    found = std::prev(events.end());
                    found->second = CudaEventPool::instance().acquire();
                    if (!found->second) {
                        // A failed capture must still settle the submitted work.
                        const auto status = cudaStreamSynchronize(stream);
                        events.pop_back();
                        ensure_cuda_success(status, "cudaStreamSynchronize(completion cleanup)", {}, LFS_SOURCE_SITE_CURRENT());
                        throw TensorError("Cannot acquire tensor completion event");
                    }
                }
                ensure_cuda_success(cudaEventRecord(found->second, stream), "cudaEventRecord(completion)", {}, LFS_SOURCE_SITE_CURRENT());
            }
#ifdef LFS_TENSOR_VULKAN
            if (pending != 0) {
                context = internal::acquire_vulkan_context();
                value = std::max(value, pending);
                context->recorders().flush_storage(latest);
            }
#endif
        }
    };

    TensorCompletion::TensorCompletion(const std::span<const Tensor* const> tensors)
        : impl_(std::make_shared<Impl>()) {
        try {
            impl_->capture(tensors);
        } catch (...) {
            const auto failure = std::current_exception();
            try {
                wait();
            } catch (const std::exception& error) {
                LOG_WARN("Tensor capture cleanup failed: {}", error.what());
            }
            std::rethrow_exception(failure);
        }
    }

    void TensorCompletion::include(const GpuBackend backend) {
        if (!impl_)
            impl_ = std::make_shared<Impl>();
        std::lock_guard lock(impl_->mutex);
        impl_->settle_on_release = true;
        impl_->backends[static_cast<size_t>(backend)] = true;
    }

    void TensorCompletion::include(const Tensor& tensor) {
        if (!impl_)
            impl_ = std::make_shared<Impl>();
        std::lock_guard lock(impl_->mutex);
        impl_->settle_on_release = true;
        const Tensor* tensors[] = {&tensor};
        impl_->capture(tensors);
    }

    void TensorCompletion::include_current_gpu() {
        include(internal::resolve_new_gpu_storage_backend());
    }

    bool TensorCompletion::ready() const {
        if (!impl_)
            return true;
        std::lock_guard lock(impl_->mutex);
        if (std::ranges::any_of(impl_->backends, [](bool pending) { return pending; }))
            return false;
        for (const auto& [stream, event] : impl_->events) {
            const auto status = cudaEventQuery(event);
            if (status == cudaErrorNotReady)
                return false;
            ensure_cuda_success(status, "cudaEventQuery(completion)", {}, LFS_SOURCE_SITE_CURRENT());
        }
#ifdef LFS_TENSOR_VULKAN
        if (impl_->device && impl_->point.value) {
            uint64_t completed = 0;
            const auto status = vkGetSemaphoreCounterValue(static_cast<VkDevice>(impl_->device),
                                                           static_cast<VkSemaphore>(impl_->point.semaphore), &completed);
            if (status != VK_SUCCESS)
                throw TensorError("Cannot query tensor completion timeline");
            if (completed < impl_->point.value)
                return false;
        }
        if (impl_->value != 0) {
            if (impl_->context->dead())
                throw_backend_unavailable(GpuBackend::Vulkan);
            if (impl_->context->completed_timeline() < impl_->value)
                return false;
        }
#endif
        return true;
    }

    void TensorCompletion::wait() const {
        if (impl_)
            impl_->wait();
    }

    VulkanTimelinePoint TensorCompletion::timeline() const {
        return impl_ ? impl_->point : VulkanTimelinePoint{};
    }

    void TensorCompletion::Impl::wait() {
        std::lock_guard lock(mutex);
        auto* impl_ = this;
        std::exception_ptr failure;
        const auto settle = [&](auto&& operation) {
            try {
                operation();
            } catch (...) {
                // LFS-CENSUS-OK(empty-catch): Rethrow after settling every captured obligation.
                if (!failure)
                    failure = std::current_exception();
            }
        };
        for (size_t i = 0; i < impl_->backends.size(); ++i) {
            if (std::exchange(impl_->backends[i], false))
                settle([&] { internal::backend_ops(kGpuBackends[i]).synchronize_device(); });
        }
        for (const auto& [stream, event] : impl_->events) {
            settle([&] {
                ensure_cuda_success(cudaEventSynchronize(event), "cudaEventSynchronize(completion)", {}, LFS_SOURCE_SITE_CURRENT());
                device_fault_registry_consume_or_throw(LFS_SOURCE_SITE_CURRENT(), false);
            });
            CudaEventPool::instance().release(event);
        }
        impl_->events.clear();
#ifdef LFS_TENSOR_VULKAN
        if (impl_->device && impl_->point.value) {
            settle([&] {
                const auto semaphore = static_cast<VkSemaphore>(impl_->point.semaphore);
                VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
                info.semaphoreCount = 1;
                info.pSemaphores = &semaphore;
                info.pValues = &impl_->point.value;
                VkResult status;
                do {
                    status = vkWaitSemaphores(static_cast<VkDevice>(impl_->device), &info, 1'000'000'000);
                } while (status == VK_TIMEOUT);
                if (status != VK_SUCCESS)
                    throw TensorError("Cannot wait for tensor completion timeline");
                impl_->device = nullptr;
            });
        }
        if (const auto value = std::exchange(impl_->value, 0))
            settle([&] {
                impl_->context->wait(value);
                impl_->context->check_fault_buffer();
            });
        impl_->context.reset();
#endif
        if (failure)
            std::rethrow_exception(failure);
    }

    TensorCompletion::~TensorCompletion() = default;

    TensorCompletion::Impl::~Impl() {
        if (!settle_on_release)
            return;
        try {
            wait();
        } catch (const std::exception& error) {
            LOG_WARN("Tensor work cleanup failed: {}", error.what());
        } catch (...) {
            LOG_WARN("Tensor work cleanup failed");
        }
    }

    TensorCompletion TensorCompletionAccess::cuda(cudaStream_t stream, VulkanTimelinePoint point) {
        TensorCompletion result;
        result.impl_ = std::make_shared<TensorCompletion::Impl>();
        result.impl_->point = std::move(point);
        const auto event = CudaEventPool::instance().acquire();
        if (!event) {
            ensure_cuda_success(cudaStreamSynchronize(stream), "cudaStreamSynchronize(completion cleanup)", {}, LFS_SOURCE_SITE_CURRENT());
            throw TensorError("Cannot acquire tensor completion event");
        }
        result.impl_->events.emplace_back(stream, event);
        ensure_cuda_success(cudaEventRecord(event, stream), "cudaEventRecord(completion)", {}, LFS_SOURCE_SITE_CURRENT());
        return result;
    }

    TensorCompletion TensorCompletionAccess::vulkan(uint64_t value) {
        TensorCompletion result;
#ifdef LFS_TENSOR_VULKAN
        result.impl_ = std::make_shared<TensorCompletion::Impl>();
        result.impl_->settle_on_release = false;
        result.impl_->context = internal::acquire_vulkan_context();
        result.impl_->value = value;
        result.impl_->point = {result.impl_->context->timeline(), value, result.impl_->context};
#endif
        return result;
    }

    TensorCompletion TensorCompletionAccess::external(void* device, VulkanTimelinePoint point) {
        TensorCompletion result;
        result.impl_ = std::make_shared<TensorCompletion::Impl>();
        result.impl_->settle_on_release = false;
        result.impl_->device = device;
        result.impl_->point = std::move(point);
        return result;
    }

    std::optional<GpuDeviceInfo> gpu_backend_device_info(const GpuBackend backend) {
        if (backend == GpuBackend::CUDA) {
            int device = 0;
            if (cudaGetDevice(&device) != cudaSuccess) {
                return std::nullopt;
            }
            return gpu_backend_device_info(backend, device);
        }
#ifdef LFS_TENSOR_VULKAN
        if (backend == GpuBackend::Vulkan) {
            const auto context = internal::try_live_vulkan_context();
            if (!context || context->dead()) {
                return std::nullopt;
            }
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(context->physical_device(), &properties);
            GpuDeviceInfo result{.name = properties.deviceName};
            VkPhysicalDeviceIDProperties identity{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
            VkPhysicalDeviceProperties2 properties2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            properties2.pNext = &identity;
            vkGetPhysicalDeviceProperties2(context->physical_device(), &properties2);
            std::copy_n(identity.deviceUUID, result.uuid.size(), result.uuid.begin());
            const auto& memory = context->memory_properties();
            for (uint32_t i = 0; i < memory.memoryHeapCount; ++i) {
                if ((memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
                    result.total_memory_bytes += memory.memoryHeaps[i].size;
                }
            }
            // Process-budget support belongs to the physical device, independently
            // of the allocator's enabled budget policy.
            static std::mutex capability_mutex;
            static uint64_t capability_context = 0;
            static bool supports_budget = false;
            {
                std::lock_guard lock(capability_mutex);
                if (capability_context != context->context_id()) {
                    uint32_t count = 0;
                    if (vkEnumerateDeviceExtensionProperties(context->physical_device(), nullptr, &count, nullptr) != VK_SUCCESS)
                        return result;
                    std::vector<VkExtensionProperties> extensions(count);
                    if (vkEnumerateDeviceExtensionProperties(context->physical_device(), nullptr, &count, extensions.data()) != VK_SUCCESS)
                        return result;
                    supports_budget = std::any_of(extensions.begin(), extensions.begin() + count, [](const auto& extension) {
                        return std::strcmp(extension.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0;
                    });
                    capability_context = context->context_id();
                }
                result.supports_process_memory_budget = supports_budget;
            }
            if (result.supports_process_memory_budget) {
                VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
                VkPhysicalDeviceMemoryProperties2 memory_info{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
                memory_info.pNext = &budget;
                vkGetPhysicalDeviceMemoryProperties2(context->physical_device(), &memory_info);
                for (uint32_t i = 0; i < memory_info.memoryProperties.memoryHeapCount; ++i) {
                    if ((memory_info.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
                        result.process_memory_budget_bytes += budget.heapBudget[i];
                        result.process_memory_used_bytes += budget.heapUsage[i];
                    }
                }
            }
            return result;
        }
#endif
        return std::nullopt;
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
            .sharing_queue_families = handles.sharing_queue_families,
            .sharing_queue_family_count = handles.sharing_queue_family_count,
            .shader_atomic_float = handles.shader_atomic_float,
            .memory_budget = handles.memory_budget,
            .shader_float64 = handles.shader_float64,
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

    std::optional<TensorVulkanBuffer> tensor_vulkan_buffer(const Tensor& tensor) {
#ifdef LFS_TENSOR_VULKAN
        auto result = internal::native_vulkan_buffer(tensor);
        if (result)
            internal::try_live_vulkan_context()->recorders().flush_storage(internal::storage_ref(tensor));
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
            if (!context->memory().exports_memory() || !context->timeline_exportable()) {
                return false;
            }
            context->cuda_imports()->import_timeline(context->timeline());
            return context->cuda_imports()->timeline_imported();
        } catch (...) { // LFS-CENSUS-OK(empty-catch): export probe fails closed when CUDA import is absent
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
        return Tensor::from_external_owner(data, tensor.shape(), Device::GPU, tensor.dtype(),
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
            if (tensor.device() != Device::GPU || tensor.stream() == nullptr) {
                return;
            }
            backend_ops_for(tensor).bridge(ExecContext{tensor.stream()}, ExecContext{nullptr});
        }

        void order_home_after_legacy(const Tensor& tensor) {
            if (tensor.device() != Device::GPU || tensor.stream() == nullptr) {
                return;
            }
            backend_ops_for(tensor).bridge(ExecContext{nullptr}, ExecContext{tensor.stream()});
        }

        void trim_live_gpu_backends() {
            for (const GpuBackend backend : kGpuBackends) {
                if (gpu_backend_live(backend))
                    backend_ops(backend).trim();
            }
        }

        void trim_live_gpu_backends_if_reserved_unused_exceeds(const size_t threshold_bytes) {
            for (const GpuBackend backend : kGpuBackends) {
                if (gpu_backend_live(backend))
                    backend_ops(backend).trim_if_reserved_unused_exceeds(threshold_bytes);
            }
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
            return Tensor::empty(shape, Device::GPU, dtype);
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
            return Tensor::full(shape, value, Device::GPU, dtype);
        }

        Tensor allocate_zeros_like(const Tensor& input,
                                   const TensorShape& shape,
                                   const DataType dtype) {
            LFS_ASSERT_MSG(input.is_valid(), "allocate_zeros_like requires a valid input tensor");
            if (input.device() == Device::CPU) {
                return Tensor::zeros(shape, Device::CPU, dtype);
            }

            GpuBackendScope scope(gpu_backend_of(input).value());
            return Tensor::zeros(shape, Device::GPU, dtype);
        }

        Tensor allocate_ones_like(const Tensor& input,
                                  const TensorShape& shape,
                                  const DataType dtype) {
            LFS_ASSERT_MSG(input.is_valid(), "allocate_ones_like requires a valid input tensor");
            if (input.device() == Device::CPU) {
                return Tensor::ones(shape, Device::CPU, dtype);
            }

            GpuBackendScope scope(gpu_backend_of(input).value());
            return Tensor::ones(shape, Device::GPU, dtype);
        }

        Tensor allocate_rand_like(const Tensor& input,
                                  const TensorShape& shape,
                                  const DataType dtype) {
            LFS_ASSERT_MSG(input.is_valid(), "allocate_rand_like requires a valid input tensor");
            if (input.device() == Device::CPU) {
                return Tensor::rand(shape, Device::CPU, dtype);
            }

            GpuBackendScope scope(gpu_backend_of(input).value());
            return Tensor::rand(shape, Device::GPU, dtype);
        }

        Tensor allocate_randn_like(const Tensor& input,
                                   const TensorShape& shape,
                                   const DataType dtype) {
            LFS_ASSERT_MSG(input.is_valid(), "allocate_randn_like requires a valid input tensor");
            if (input.device() == Device::CPU) {
                return Tensor::randn(shape, Device::CPU, dtype);
            }

            GpuBackendScope scope(gpu_backend_of(input).value());
            return Tensor::randn(shape, Device::GPU, dtype);
        }

        Tensor copy_to_backend(const Tensor& source, const GpuBackend target) {
            LFS_ASSERT_MSG(source.is_valid(), "copy_to_backend requires a valid source tensor");
            if (gpu_backend_of(source) == target) {
                return source.clone();
            }

            GpuBackendScope scope(target);
            if (source.device() == Device::CPU) {
                return source.to(Device::GPU);
            }
            return source.to(Device::CPU).to(Device::GPU);
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
            std::lock_guard lock(options_mutex);
            options_frozen = false;
        }

    } // namespace internal

} // namespace lfs::core
