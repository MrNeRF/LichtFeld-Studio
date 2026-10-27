/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "backend/tensor_completion.hpp"
#include "backend/tensor_vulkan_interop.hpp"
#include "core/cuda/driver_entry.hpp"
#include "core/tensor_backend_vulkan.hpp"
#include "core/tensor_completion.hpp"
#include <charconv>

#if LFS_HAS_CUDA
#include "backend/cuda/runtime/cuda_event_pool.hpp"
#endif
#include "core/cuda_error.hpp"
#include "core/device_fault.hpp"
#include "core/gpu_device_info.hpp"
#include "core/gpu_device_runtime.hpp"
#include "core/gpu_device_selection.hpp"
#include "core/logger.hpp"
#include "core/pinned_allocator_stats.hpp"
#include "internal/tensor_impl.hpp"
#if LFS_HAS_CUDA
#include "core/pinned_memory_allocator.hpp"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#if LFS_HAS_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#include "backend/backend_driver.hpp"
#include "backend/gpu_backend_ops.hpp"
#include <exception>
#include <format>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#ifdef LFS_TENSOR_METAL
#include "backend/metal/metal_context.hpp"
#endif
#ifdef LFS_TENSOR_VULKAN
#include "backend/vulkan/vk_context.hpp"
#if LFS_HAS_CUDA
#include "backend/vulkan/vk_cuda_bridge.hpp"
#endif
#include "backend/vulkan/vk_memory.hpp"
#include "backend/vulkan/vk_recorder.hpp"
#include "core/vulkan_helpers.hpp"
#endif

#include "backend/gpu_backend_ops.hpp"
#include "backend/readback_buffer.hpp"

namespace lfs::core::tensor_ops {
    namespace {
        std::atomic<uint64_t> g_tensor_kernel_launch_count{0};
    }

    LFS_CORE_API void reset_tensor_kernel_launch_count() noexcept {
        g_tensor_kernel_launch_count.store(0, std::memory_order_relaxed);
    }

    LFS_CORE_API uint64_t tensor_kernel_launch_count() noexcept {
        return g_tensor_kernel_launch_count.load(std::memory_order_relaxed);
    }

    void record_tensor_kernel_launch(uint64_t n) noexcept {
        g_tensor_kernel_launch_count.fetch_add(n, std::memory_order_relaxed);
        internal::telemetry_record_kernel_launch(n);
    }
} // namespace lfs::core::tensor_ops

namespace lfs::core {
    std::vector<VulkanHardwareDevice> enumerate_vulkan_hardware() {
        return internal::backend_driver(GpuBackend::Vulkan).enumerate_hardware();
    }

    std::vector<VulkanHardwareDevice> enumerate_cuda_hardware() {
        return internal::backend_driver(GpuBackend::CUDA).enumerate_hardware();
    }

    void gpu_trim_cached_memory(const GpuBackend backend) {
        if (gpu_backend_live(backend))
            internal::backend_ops(backend).trim();
    }

    void gpu_device_barrier(const GpuBackend backend) {
        internal::backend_ops(backend).device_barrier();
    }

    int gpu_device_count(const GpuBackend backend) {
        return internal::backend_driver(backend).device_count();
    }

    std::optional<size_t> reserved_allocation_bytes(const Tensor& tensor) {
        if (!tensor.is_valid() || tensor.is_external_storage())
            return std::nullopt;
        if (tensor.device() == Device::CPU) {
            if (!tensor.owns_memory())
                return tensor.is_empty() && !tensor.is_view() ? std::optional<size_t>{0} : std::nullopt;
            const auto rows = tensor.ndim() == 0 ? size_t{1} : tensor.shape()[0];
            const auto capacity = std::max(rows, tensor.capacity());
            if (capacity == 0)
                return size_t{0};
            size_t row_bytes = dtype_size(tensor.dtype());
            for (size_t axis = 1; axis < tensor.ndim(); ++axis)
                row_bytes *= tensor.shape()[axis];
            return row_bytes * capacity;
        }
        const auto storage = internal::storage_ref(tensor);
        if (storage.meta && storage.meta->gpu_descriptor.byte_size != 0)
            return static_cast<size_t>(storage.meta->gpu_descriptor.byte_size);
        return tensor.is_empty() ? std::optional<size_t>{0} : std::nullopt;
    }

    size_t gpu_allocation_bytes(const GpuBackend backend, const size_t bytes) {
        return internal::backend_driver(backend).allocation_bytes(bytes);
    }

    PinnedAllocatorStats pinned_allocator_stats() {
#if LFS_HAS_CUDA
        const auto stats = PinnedMemoryAllocator::instance().get_stats();
        return {stats.allocated_bytes, stats.cached_bytes};
#else
        return {};
#endif
    }

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
            if (!result)
                result = internal::backend_driver(backend).make_vulkan_interop(device);
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

    namespace {
        std::atomic<bool> cuda_direct_interop{true};
    }
    void set_cuda_vulkan_direct_interop(const bool enabled) {
        cuda_direct_interop.store(enabled, std::memory_order_release);
    }
    bool cuda_vulkan_direct_interop() {
        return cuda_direct_interop.load(std::memory_order_acquire);
    }

    SplatPublication splat_publication(const GpuBackend backend) {
        return internal::backend_driver(backend).publication();
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

    void TensorVulkanInterop::run_while_idle(GpuBackend backend, const std::function<void()>& release) {
        impl_->backend(backend).run_while_idle(release);
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
        LFS_ASSERT_MSG(output.is_valid() && source.is_valid() && condition.is_valid() &&
                           output.device() == source.device() && output.device() == condition.device() &&
                           gpu_backend_of(source) == backend && gpu_backend_of(condition) == backend &&
                           output.is_contiguous() && source.is_contiguous() && condition.is_contiguous() &&
                           output.shape() == source.shape() && output.numel() == condition.numel() &&
                           output.dtype() == source.dtype() && condition.dtype() == DataType::Bool &&
                           (output.dtype() == DataType::Float32 || output.dtype() == DataType::Float16),
                       "where_into requires matching contiguous Float32/Float16 tensors and a Bool mask");
        internal::preserve_lazy_snapshots_before_write(output);
        if (output.numel() == 0)
            return;
        if (output.device() == Device::CPU) {
            const auto* mask = condition.ptr<unsigned char>();
            if (output.dtype() == DataType::Float32) {
                const auto values = source.clone();
                for (size_t i = 0; i < output.numel(); ++i)
                    output.ptr<float>()[i] = mask[i] ? value : values.ptr<float>()[i];
            } else {
                const auto values = source.clone();
                const auto scalar = detail::tensor_float_to_half(value);
                for (size_t i = 0; i < output.numel(); ++i)
                    output.ptr<detail::tensor_half_t>()[i] = mask[i] ? scalar : values.ptr<detail::tensor_half_t>()[i];
            }
            return;
        }
        if (output.storage_ptr() == source.storage_ptr() && output.data_ptr() != source.data_ptr()) {
            const auto snapshot = source.clone();
            where_into(output, condition, value, snapshot);
            return;
        }
        internal::backend_ops(*backend).where_scalar(output, condition, value, source);
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

        // Without a configuration: CUDA where it is built, else Metal where the
        // GPU supports it, else Vulkan.
        GpuBackend automatic_backend() {
            for (size_t i = 0; i + 1 < kDefaultGpuBackends.size(); ++i)
                if (internal::backend_driver(kDefaultGpuBackends[i]).available())
                    return kDefaultGpuBackends[i];
            return kDefaultGpuBackends.back();
        }

        GpuBackend backend_of_state(const int state) {
            if (is_resolved(state))
                return static_cast<GpuBackend>(state);
            return state == kUnconfigured ? automatic_backend() : configured_backend(state);
        }

        [[noreturn]] void throw_backend_unavailable(const GpuBackend backend) {
            internal::backend_driver(backend).throw_unavailable();
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
        std::array<int, kGpuBackendCount> selections;
        selections.fill(-1);
        for (const auto backend : kCompiledGpuBackends) {
            auto& ordinal = selections[static_cast<size_t>(backend)];
            if (const auto error = internal::backend_driver(backend).validate_options(options, ordinal))
                return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
                    .code = lfs::ErrorCode::InvalidArgument,
                    .domain = lfs::ErrorDomain::Core,
                    .user_message = *error,
                    .detection = LFS_SOURCE_SITE_CURRENT(),
                }));
        }
        backend_options = options;
        for (const auto backend : kCompiledGpuBackends)
            internal::backend_driver(backend).set_selection(selections[static_cast<size_t>(backend)]);
        internal::clear_backend_activation_cache();
        return {};
    }

    int selected_hip_device() {
        return internal::backend_driver(GpuBackend::HIP).selected_device();
    }

    int selected_cuda_device() {
        return internal::backend_driver(GpuBackend::CUDA).selected_device();
    }

    void activate_selected_cuda_device() {
        internal::backend_driver(GpuBackend::CUDA).bind_thread();
    }

    TensorBackendOptions tensor_backend_options() {
        std::lock_guard lock(options_mutex);
        options_frozen = true;
        return backend_options;
    }

    TensorBackendOptions tensor_backend_options_snapshot() {
        std::lock_guard lock(options_mutex);
        return backend_options;
    }

    const char* gpu_backend_name(const GpuBackend backend) {
        return internal::backend_driver(backend).name();
    }

    GpuBackend default_gpu_backend() {
        for (;;) {
            int state = process_backend_state.load(std::memory_order_acquire);
            const GpuBackend selected = backend_of_state(state);
            if (is_resolved(state) ||
                process_backend_state.compare_exchange_weak(
                    state, static_cast<int>(selected),
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                return selected;
            }
        }
    }

    GpuBackend configured_gpu_backend() {
        return backend_of_state(process_backend_state.load(std::memory_order_acquire));
    }

    lfs::Status set_default_gpu_backend(const GpuBackend backend) {
        if (std::ranges::find(kCompiledGpuBackends, backend) == kCompiledGpuBackends.end()) {
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
        return internal::backend_driver(backend).available();
    }

    bool gpu_backend_live(const GpuBackend backend) {
        return internal::backend_driver(backend).live();
    }

    std::optional<GpuBackend> gpu_backend_of(const Tensor& tensor) {
        if (!tensor.is_valid() || tensor.device_ != Device::GPU) {
            return std::nullopt;
        }
        return internal::gpu_backend_tag(tensor);
    }

    TensorExecutionTarget TensorExecutionTarget::current() {
        const auto backend = scoped_backend ? *scoped_backend : default_gpu_backend();
        return {backend, backend == GpuBackend::CUDA ? getCurrentCUDAStream() : nullptr};
    }

    GpuBackendScope::GpuBackendScope(const GpuBackend backend)
        : previous_(scoped_backend) {
        internal::backend_driver(backend).bind_thread();
        scoped_backend = backend;
    }

    GpuBackendScope::~GpuBackendScope() {
        scoped_backend = previous_;
    }

    MemoryInfo gpu_backend_memory_info(const GpuBackend backend, const bool include_pool_stats) {
        return internal::backend_driver(backend).memory_info(include_pool_stats);
    }

    struct TensorCompletion::Impl {
        std::mutex mutex;
        std::array<bool, kGpuBackendCount> backends{};
        std::array<std::unique_ptr<internal::BackendCompletion>, kGpuBackendCount> captures;
        VulkanTimelinePoint point;
        bool settle_on_release = true;

        internal::BackendCompletion& completion(GpuBackend backend, internal::BackendCompletionTarget target = {}) {
            auto& item = captures[static_cast<size_t>(backend)];
            if (!item) {
                item = internal::backend_ops(backend).create_completion(std::move(target));
                if (!item)
                    throw TensorError("Tensor backend does not support the requested completion target");
            }
            return *item;
        }
        void capture(std::span<const Tensor* const> tensors) {
            std::array<std::vector<const Tensor*>, kGpuBackendCount> selected;
            for (const auto* tensor : tensors) {
                if (!tensor || !tensor->is_valid())
                    continue;
                if (const auto backend = gpu_backend_of(*tensor))
                    selected[static_cast<size_t>(*backend)].push_back(tensor);
            }
            for (size_t i = 0; i < selected.size(); ++i)
                if (!selected[i].empty())
                    completion(kGpuBackends[i]).capture(selected[i]);
        }
        void wait() {
            std::lock_guard lock(mutex);
            std::exception_ptr failure;
            const auto settle = [&](auto&& operation) {
                try {
                    operation();
                } catch (...) {
                    // LFS-CENSUS-OK(empty-catch): Settle every obligation before rethrowing its first failure.
                    if (!failure)
                        failure = std::current_exception();
                }
            };
            for (size_t i = 0; i < backends.size(); ++i)
                if (std::exchange(backends[i], false))
                    settle([&] { internal::backend_ops(kGpuBackends[i]).synchronize_device(); });
            for (auto& capture : captures)
                if (capture)
                    settle([&] { capture->wait(); });
            if (failure)
                std::rethrow_exception(failure);
        }
        ~Impl() {
            if (!settle_on_release)
                return;
            try {
                wait();
            } catch (const std::exception& error) { LOG_WARN("Tensor work cleanup failed: {}", error.what()); } catch (...) {
                LOG_WARN("Tensor work cleanup failed");
            }
        }
    };
    TensorCompletion::TensorCompletion(std::span<const Tensor* const> tensors) : impl_(std::make_shared<Impl>()) {
        try {
            impl_->capture(tensors);
        } catch (...) {
            const auto failure = std::current_exception();
            try {
                wait();
            } catch (const std::exception& error) { LOG_WARN("Tensor capture cleanup failed: {}", error.what()); }
            std::rethrow_exception(failure);
        }
    }
    void TensorCompletion::include(GpuBackend backend) {
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
    void TensorCompletion::include_current_gpu() { include(internal::resolve_new_gpu_storage_backend()); }
    bool TensorCompletion::ready() const {
        if (!impl_)
            return true;
        std::lock_guard lock(impl_->mutex);
        if (std::ranges::any_of(impl_->backends, [](bool pending) { return pending; }))
            return false;
        for (auto& capture : impl_->captures)
            if (capture && !capture->ready())
                return false;
        return true;
    }
    void TensorCompletion::wait() const {
        if (impl_)
            impl_->wait();
    }
    VulkanTimelinePoint TensorCompletion::timeline() const { return impl_ ? impl_->point : VulkanTimelinePoint{}; }
    TensorCompletion::~TensorCompletion() = default;

    TensorCompletion TensorCompletionAccess::hip() {
        TensorCompletion result;
        result.impl_ = std::make_shared<TensorCompletion::Impl>();
        result.impl_->completion(GpuBackend::HIP).record({}, 0);
        return result;
    }
    TensorCompletion TensorCompletionAccess::cuda(cudaStream_t stream, VulkanTimelinePoint point) {
        TensorCompletion result;
        result.impl_ = std::make_shared<TensorCompletion::Impl>();
        result.impl_->point = std::move(point);
        result.impl_->completion(GpuBackend::CUDA).record({stream}, 0);
        return result;
    }
    TensorCompletion TensorCompletionAccess::vulkan(uint64_t value) {
        TensorCompletion result;
        result.impl_ = std::make_shared<TensorCompletion::Impl>();
        result.impl_->settle_on_release = false;
        auto& completion = result.impl_->completion(GpuBackend::Vulkan);
        completion.record({}, value);
        result.impl_->point = completion.timeline();
        return result;
    }
    TensorCompletion TensorCompletionAccess::metal(uint64_t serial, VulkanTimelinePoint point) {
        TensorCompletion result;
        result.impl_ = std::make_shared<TensorCompletion::Impl>();
        result.impl_->settle_on_release = false;
        result.impl_->point = std::move(point);
        result.impl_->completion(GpuBackend::Metal).record({}, serial);
        return result;
    }
    TensorCompletion TensorCompletionAccess::external(void* device, VulkanTimelinePoint point) {
        TensorCompletion result;
        result.impl_ = std::make_shared<TensorCompletion::Impl>();
        result.impl_->settle_on_release = false;
        result.impl_->point = point;
        result.impl_->completion(GpuBackend::Vulkan, {.external_device = device, .external_point = std::move(point)});
        return result;
    }

    std::optional<GpuDeviceInfo> gpu_backend_device_info(const GpuBackend backend) {

        try {
            return internal::backend_driver(backend).device_info(-1);
        } catch (...) { /* LFS-CENSUS-OK(empty-catch): Device metadata discovery reports unavailable without constructing Tensor storage. */
            return std::nullopt;
        }
    }

    std::optional<GpuDeviceInfo> gpu_backend_device_info(const GpuBackend backend,
                                                         const int device_index) {
        if (device_index < 0)
            return std::nullopt;
        return internal::backend_driver(backend).device_info(device_index);
    }

    lfs::Status shutdown_gpu_backend(const GpuBackend backend) {

        auto status = internal::backend_driver(backend).shutdown();
        if (status)
            internal::clear_backend_activation_cache();
        return status;
    }

    lfs::Status adopt_vulkan_device(const VulkanDeviceHandles& handles) {
        return adopt_vulkan_device(handles, false);
    }

    lfs::Status adopt_vulkan_device(const VulkanDeviceHandles& handles,
                                    const bool subgroup_size_control_enabled) {
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
            .vulkan_memory_model = handles.vulkan_memory_model,
            .vulkan_memory_model_device_scope = handles.vulkan_memory_model_device_scope,
            .cooperative_matrix = handles.cooperative_matrix,
            .external_memory = handles.external_memory,
            .external_semaphore = handles.external_semaphore,
#ifdef __APPLE__
            .metal_objects = handles.metal_objects,
#endif
            .subgroup_size_control = subgroup_size_control_enabled,
            .consumer_queue = static_cast<VkQueue>(handles.consumer_queue),
            .consumer_queue_mutex = handles.consumer_queue_mutex,
        });
#else
        (void)handles;
        (void)subgroup_size_control_enabled;
        return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
            .code = lfs::ErrorCode::Unsupported,
            .domain = lfs::ErrorDomain::Vulkan,
            .user_message = "this build has no Vulkan tensor backend",
            .detection = LFS_SOURCE_SITE_CURRENT(),
        }));
#endif
    }

    bool tensor_supports_metal_access(const Tensor& tensor) {
#ifdef __APPLE__
        const auto backend = gpu_backend_of(tensor);
        return backend == GpuBackend::Metal || backend == GpuBackend::Vulkan;
#else
        return false;
#endif
    }

    bool tensor_backend_supports_metal_access() {
#ifdef __APPLE__
        const auto backend = default_gpu_backend();
        return backend == GpuBackend::Metal || backend == GpuBackend::Vulkan;
#else
        return false;
#endif
    }

    bool tensor_backend_shares_vulkan_device() {
        if (gpu_backend_compiled(GpuBackend::HIP) && default_gpu_backend() == GpuBackend::HIP)
            return false;
        return default_gpu_backend() != GpuBackend::Metal;
    }

    bool tensor_backend_needs_vulkan_interop() {
        if (gpu_backend_compiled(GpuBackend::HIP) && default_gpu_backend() == GpuBackend::HIP)
            return true;
        return default_gpu_backend() == GpuBackend::CUDA;
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
#if defined(LFS_TENSOR_VULKAN) && LFS_HAS_CUDA
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
#if defined(LFS_TENSOR_VULKAN) && LFS_HAS_CUDA
        try {
            const auto context = internal::try_live_vulkan_context();
            if (!context || context->dead() || !context->cuda_imports()) {
                return false;
            }
            if (!context->memory().exports_memory() || !context->timeline_exportable()) {
                return false;
            }
            const auto cuda = gpu_backend_device_info(GpuBackend::CUDA, selected_cuda_device());
            VkPhysicalDeviceIDProperties identity{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
            VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            properties.pNext = &identity;
            vkGetPhysicalDeviceProperties2(context->physical_device(), &properties);
            if (!cuda || !std::equal(cuda->uuid.begin(), cuda->uuid.end(), identity.deviceUUID))
                return false;
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
#if defined(LFS_TENSOR_VULKAN) && LFS_HAS_CUDA
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

        void hold_freed_gpu_memory(const bool hold) {
            for (const GpuBackend backend : kGpuBackends) {
                if (gpu_backend_live(backend))
                    backend_ops(backend).hold_freed_memory(hold);
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
            for (const auto backend : kCompiledGpuBackends)
                internal::backend_driver(backend).set_selection(-1);
            internal::clear_backend_activation_cache();
            set_cuda_vulkan_direct_interop(true);
        }

    } // namespace internal

} // namespace lfs::core
