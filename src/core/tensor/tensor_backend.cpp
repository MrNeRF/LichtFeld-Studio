/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "backend/tensor_completion.hpp"
#include "backend/tensor_vulkan_interop.hpp"
#include "core/tensor_backend_vulkan.hpp"
#include "core/tensor_completion.hpp"
#include <charconv>

#include "core/cuda_error.hpp"
#include "core/device_fault.hpp"
#include "core/gpu_device_info.hpp"
#include "core/gpu_device_runtime.hpp"
#include "core/gpu_device_selection.hpp"
#include "core/logger.hpp"
#include "core/pinned_allocator_stats.hpp"
#include "internal/tensor_impl.hpp"

#include "backend/backend_driver.hpp"
#include "backend/gpu_backend_ops.hpp"
#include "backend/native_pressure.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <format>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

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

    MemoryInfo gpu_pressure_memory_info(GpuBackend backend, bool free_query) {
        return internal::backend_driver(backend).pressure_memory_info(free_query);
    }
    bool gpu_pressure_recovery_blocked(GpuBackend backend) {
        return internal::backend_driver(backend).pressure_recovery_blocked();
    }
    void register_gpu_pressure_clients(MemoryPressureCoordinator& coordinator) {
        for (const auto backend : kCompiledGpuBackends)
            internal::backend_driver(backend).register_pressure_clients(coordinator);
    }
    void record_gpu_pressure_breadcrumb() { internal::record_pressure_breadcrumb(); }

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
        // Native policy is immutable. Cache it once per thread, without
        // caching dynamic availability or skipping required device rebinding.
        thread_local std::array<std::optional<internal::BackendDriver*>, kGpuBackendCount> scope_bindings;
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
            std::unreachable();
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
        return {backend, internal::backend_driver(backend).current_target()};
    }

    GpuBackendScope::GpuBackendScope(const GpuBackend backend)
        : previous_(scoped_backend) {
        const auto index = static_cast<size_t>(backend);
        LFS_ASSERT_MSG(index < kGpuBackendCount, "Unknown GPU backend");
        auto& binding = scope_bindings[index];
        if (!binding) {
            auto& driver = internal::backend_driver(backend);
            binding = driver.activation_scope() == internal::BackendActivationScope::Thread ? &driver : nullptr;
        }
        if (*binding)
            (*binding)->bind_thread();
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
            internal::CompletionDrain drain;
            for (size_t i = 0; i < backends.size(); ++i)
                if (std::exchange(backends[i], false))
                    drain.settle([&] { internal::backend_ops(kGpuBackends[i]).synchronize_device(); });
            for (auto& capture : captures)
                if (capture)
                    drain.settle([&] { capture->wait(); });
            drain.rethrow();
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

    TensorCompletion TensorCompletionAccess::record(GpuBackend backend, internal::NativeCompletionRecord record) {
        TensorCompletion result;
        result.impl_ = std::make_shared<TensorCompletion::Impl>();
        result.impl_->settle_on_release = record.settle_on_release;
        result.impl_->point = std::move(record.timeline);
        internal::BackendCompletionTarget imported;
        if (!record.record)
            imported = {.external_device = record.external_device, .external_point = result.impl_->point};
        auto& completion = result.impl_->completion(backend, std::move(imported));
        if (record.record)
            completion.record(record.target, record.value);
        if (record.use_recorded_timeline)
            result.impl_->point = completion.timeline();
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

    bool tensor_supports_metal_access(const Tensor& tensor) {
        const auto backend = gpu_backend_of(tensor);
        return backend && internal::backend_driver(*backend).renderer_access().metal;
    }
    bool tensor_backend_supports_metal_access() {
        return internal::backend_driver(default_gpu_backend()).renderer_access().metal;
    }
    bool tensor_backend_shares_vulkan_device() {
        return internal::backend_driver(default_gpu_backend()).renderer_access().shares_vulkan;
    }
    bool tensor_backend_needs_vulkan_interop() {
        return internal::backend_driver(default_gpu_backend()).renderer_access().needs_vulkan_interop;
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

        namespace {
            template <class Allocate>
            Tensor allocate_like_on_backend(const Tensor& input, const char* validation, Allocate&& allocate) {
                LFS_ASSERT_MSG(input.is_valid(), validation);
                if (input.device() == Device::CPU)
                    return allocate(Device::CPU);
                const GpuBackendScope scope(gpu_backend_of(input).value());
                return allocate(Device::GPU);
            }
        } // namespace

        Tensor allocate_like(const Tensor& input, const TensorShape& shape, DataType dtype) {
            return allocate_like_on_backend(input, "allocate_like requires a valid input tensor",
                                            [&](Device device) { return Tensor::empty(shape, device, dtype); });
        }
        Tensor allocate_like(const Tensor& input, const TensorShape& shape, DataType dtype, float value) {
            return allocate_like_on_backend(input, "allocate_like requires a valid input tensor",
                                            [&](Device device) { return Tensor::full(shape, value, device, dtype); });
        }
        Tensor allocate_zeros_like(const Tensor& input, const TensorShape& shape, DataType dtype) {
            return allocate_like_on_backend(input, "allocate_zeros_like requires a valid input tensor",
                                            [&](Device device) { return Tensor::zeros(shape, device, dtype); });
        }
        Tensor allocate_ones_like(const Tensor& input, const TensorShape& shape, DataType dtype) {
            return allocate_like_on_backend(input, "allocate_ones_like requires a valid input tensor",
                                            [&](Device device) { return Tensor::ones(shape, device, dtype); });
        }
        Tensor allocate_rand_like(const Tensor& input, const TensorShape& shape, DataType dtype) {
            return allocate_like_on_backend(input, "allocate_rand_like requires a valid input tensor",
                                            [&](Device device) { return Tensor::rand(shape, device, dtype); });
        }
        Tensor allocate_randn_like(const Tensor& input, const TensorShape& shape, DataType dtype) {
            return allocate_like_on_backend(input, "allocate_randn_like requires a valid input tensor",
                                            [&](Device device) { return Tensor::randn(shape, device, dtype); });
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
