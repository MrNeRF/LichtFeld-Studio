/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor_upload.hpp"
#include "../internal/tensor_impl.hpp"
#include "core/tensor_readback.hpp"
#include "metal/metal_queue.hpp"
#if LFS_HAS_CUDA
#include "core/cuda_error_typed.hpp"
#include <cuda_runtime.h>
#endif
#include "core/logger.hpp"
#include "core/tensor_backend.hpp"
#if LFS_HAS_CUDA
#include "cuda/runtime/memory_pool.hpp"
#endif
#include "tensor_completion.hpp"
#include "tensor_vulkan_interop.hpp"
#include "vulkan/vk_context.hpp"
#include "vulkan/vk_memory.hpp"
#if LFS_HAS_CUDA
#include "vulkan/vk_cuda_bridge.hpp"
#endif
#include "vulkan/vk_recorder.hpp"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#if LFS_HAS_CUDA && !defined(_WIN32)
#include <unistd.h>
#endif
namespace lfs::core {
#if LFS_HAS_CUDA
    namespace {
        void check(cudaError_t e) {
            if (e != cudaSuccess)
                throw std::runtime_error(cudaGetErrorString(e));
        }
    } // namespace
#endif
    namespace {
        uint64_t metal_target(void* const target) {
            const auto id = reinterpret_cast<uint64_t>(target);
            if (!internal::metal_queue::valid(id))
                throw std::invalid_argument("Execution target is not a Metal tensor queue");
            return id;
        }
    } // namespace
    struct TensorUpload::Impl {
        Tensor source, destination, staging;
        TensorCompletion completion;
        std::shared_ptr<internal::VulkanContext> vk;
#if LFS_HAS_CUDA
        cudaEvent_t cuda_completion = nullptr;
#endif
        bool pending = false;
    };
    TensorUpload::TensorUpload() = default;
    TensorUpload::~TensorUpload() {
        try {
            wait();
        } catch (const std::exception& e) {
            LOG_ERROR("Tensor upload storage retained after incomplete transfer: {}", e.what());
            (void)impl_.release();
        }
#if LFS_HAS_CUDA
        if (impl_ && impl_->cuda_completion)
            (void)cudaEventDestroy(impl_->cuda_completion);
#endif
    }
    TensorUpload::TensorUpload(TensorUpload&&) noexcept = default;
    TensorUpload& TensorUpload::operator=(TensorUpload&& other) noexcept {
        if (this != &other) {
            TensorUpload previous;
            previous.impl_ = std::move(impl_);
            impl_ = std::move(other.impl_);
        }
        return *this;
    }
    bool TensorUpload::pending() const noexcept { return impl_ && impl_->pending; }
    void TensorUpload::enqueue(Tensor destination, const Tensor& source, TensorExecutionTarget target) {
        if (gpu_backend_of(destination) != target.backend())
            throw std::invalid_argument("TensorUpload queue backend mismatch");
        enqueue(std::move(destination), source, target.native_handle());
    }
    void TensorUpload::enqueue(Tensor destination, std::span<const std::byte> source, TensorExecutionTarget target) {
        if (gpu_backend_of(destination) != target.backend())
            throw std::invalid_argument("TensorUpload queue backend mismatch");
        enqueue(std::move(destination), source, target.native_handle());
    }
    void TensorUpload::enqueue(Tensor destination, const Tensor& source) {
        const auto stream = gpu_backend_of(destination) == GpuBackend::CUDA
                                ? reinterpret_cast<void*>(getCurrentCUDAStream())
                                : nullptr;
        enqueue(std::move(destination), source, stream);
    }
    void TensorUpload::enqueue(Tensor destination,
                               const std::span<const std::byte> source,
                               void* const execution_target) {
        if (!destination.is_valid() || !destination.is_contiguous() ||
            destination.bytes() != source.size())
            throw std::invalid_argument("TensorUpload byte span must match a contiguous destination");
        if (pending())
            throw std::logic_error("TensorUpload already pending");
        if (gpu_backend_of(destination) == GpuBackend::Metal && !source.empty()) {
            // Metal copies the bytes during enqueue, so the span needs no staging copy.
            const size_t count = destination.numel();
            const DataType dtype = destination.dtype();
            enqueue(std::move(destination),
                    Tensor::from_blob(const_cast<std::byte*>(source.data()), {count}, Device::CPU, dtype),
                    execution_target);
            return;
        }
        if (!impl_)
            impl_ = std::make_unique<Impl>();
        auto& state = *impl_;
        if (!state.staging.is_valid() || state.staging.dtype() != destination.dtype() ||
            state.staging.numel() < destination.numel()) {
            state.staging = Tensor::empty({destination.numel()}, Device::CPU,
                                          destination.dtype());
        }
        auto staging = state.staging.slice(0, 0, destination.numel());
        if (!source.empty())
            std::memcpy(staging.data_ptr(), source.data(), source.size());
        enqueue(std::move(destination), staging, execution_target);
    }
    void TensorUpload::enqueue(Tensor destination, const Tensor& source,
                               void* const execution_target) {
        if (pending())
            throw std::logic_error("TensorUpload already pending");
        if (!gpu_backend_of(destination) || source.device() != Device::CPU || !source.is_contiguous() ||
            !destination.is_contiguous() || destination.dtype() != source.dtype() || destination.bytes() != source.bytes())
            throw std::invalid_argument("TensorUpload requires matching contiguous CPU and GPU tensors");
        const auto backend = gpu_backend_of(destination);
        if (backend == GpuBackend::Vulkan && execution_target != nullptr) {
            const auto id = reinterpret_cast<uint64_t>(execution_target);
            if (!internal::acquire_vulkan_context()->recorders().owns_queue(id))
                throw std::invalid_argument("TensorUpload execution targets are CUDA streams");
        } else if (backend == GpuBackend::Metal) {
            (void)metal_target(execution_target);
        } else if (backend != GpuBackend::CUDA && execution_target != nullptr) {
            throw std::invalid_argument("TensorUpload execution targets are CUDA streams");
        }
        if (!impl_)
            impl_ = std::make_unique<Impl>();
        auto& s = *impl_;
        s.source = source;
        s.destination = std::move(destination);
        s.completion = {};
        s.vk.reset();
        pin_operands({&s.source, &s.destination});
        const auto stream = backend == GpuBackend::CUDA
                                ? reinterpret_cast<cudaStream_t>(execution_target)
                                : nullptr;
        s.destination.set_stream(stream);
        s.pending = true;
        struct VulkanQueueBind {
            bool active = false;
            VulkanQueueBind(GpuBackend queue_backend, void* target) {
                if (queue_backend != GpuBackend::Vulkan || target == nullptr)
                    return;
                internal::acquire_vulkan_context()->recorders().bind_queue(
                    reinterpret_cast<uint64_t>(target));
                active = true;
            }
            ~VulkanQueueBind() {
                if (active)
                    if (const auto context = internal::try_live_vulkan_context())
                        context->recorders().unbind_queue();
            }
            VulkanQueueBind(const VulkanQueueBind&) = delete;
            VulkanQueueBind& operator=(const VulkanQueueBind&) = delete;
        };
        try {
            const VulkanQueueBind queue_bind(*backend, execution_target);
            internal::backend_ops_for(s.destination).copy_host_to_device({.src = internal::raw_storage_ref(const_cast<void*>(s.source.data_ptr()), s.source.dtype()), .dst = internal::storage_ref(s.destination), .bytes = s.source.bytes(), .synchronous = false, .context = internal::ExecContext{stream}});
            if (backend == GpuBackend::CUDA) {
#if LFS_HAS_CUDA
                if (!s.cuda_completion)
                    check(cudaEventCreateWithFlags(&s.cuda_completion, cudaEventDisableTiming));
                check(cudaEventRecord(s.cuda_completion, stream));
#endif
            } else if (backend == GpuBackend::Vulkan) {
                s.vk = internal::acquire_vulkan_context();
                s.completion = TensorCompletionAccess::vulkan(
                    s.vk->recorders().pending_value(internal::storage_ref(s.destination)));
            } else {
                // Metal has copied the bytes, into the destination or, while the
                // GPU still uses it, into staging that a queued copy drains.
                s.source = {};
                const auto* const meta = internal::storage_ref(s.destination).meta;
                const uint64_t serial = meta ? meta->pending_value.load(std::memory_order_acquire) : 0;
                if (internal::metal_queue::ready(serial)) {
                    s.pending = false;
                    s.destination = {};
                } else {
                    s.completion = TensorCompletionAccess::metal(serial);
                }
            }
        } catch (...) {
            internal::backend_ops_for(s.destination).synchronize_stream(internal::ExecContext{stream});
            s.pending = false;
            throw;
        }
    }
    bool TensorUpload::poll() {
        if (!pending())
            return true;
        auto& s = *impl_;
        if (s.vk) {
            s.vk->recorders().flush_storage(internal::storage_ref(s.destination));
            if (!s.completion.ready())
                return false;
        } else if (gpu_backend_of(s.destination) == GpuBackend::Metal) {
            if (!s.completion.ready())
                return false;
        } else {
#if LFS_HAS_CUDA
            const cudaError_t status = cudaEventQuery(s.cuda_completion);
            if (status == cudaErrorNotReady)
                return false;
            if (status != cudaSuccess) {
                (void)cudaGetLastError();
                throw std::runtime_error("Tensor upload completion query failed");
            }
#else
            return false;
#endif
        }
        s.pending = false;
        s.source = {};
        s.destination = {};
        return true;
    }
    void TensorUpload::wait() {
        if (!pending())
            return;
        auto& s = *impl_;
        if (s.vk || gpu_backend_of(s.destination) == GpuBackend::Metal) {
            if (s.vk)
                s.vk->recorders().flush_storage(internal::storage_ref(s.destination));
            s.completion.wait();
            (void)poll();
            return;
        }
#if LFS_HAS_CUDA
        check(cudaEventSynchronize(s.cuda_completion));
        (void)poll();
#endif
    }
    struct TensorFence::Impl {
        GpuBackend backend = GpuBackend::CUDA;
        mutable std::shared_ptr<internal::VulkanContext> vulkan;
        uint64_t timeline = 0;
#if LFS_HAS_CUDA
        cudaEvent_t event = nullptr;
        ~Impl() {
            if (event)
                (void)cudaEventDestroy(event);
        }
#endif
    };
    TensorFence::TensorFence(GpuBackend backend) : impl_(std::make_unique<Impl>()) {
        impl_->backend = backend;
        if (backend == GpuBackend::Vulkan) {
            impl_->vulkan = internal::acquire_vulkan_context();
            return;
        }
        if (backend == GpuBackend::Metal) {
            if (!gpu_backend_available(GpuBackend::Metal))
                throw std::runtime_error("Metal tensor fences are unavailable");
            return;
        }
        if (backend != GpuBackend::CUDA)
            throw std::runtime_error("Reusable tensor fences are unsupported on Vulkan");
#if LFS_HAS_CUDA
        check(cudaEventCreateWithFlags(&impl_->event, cudaEventDisableTiming));
#else
        throw std::runtime_error("CUDA tensor fences are unavailable in this build");
#endif
    }
    TensorFence::~TensorFence() = default;
    TensorFence::TensorFence(TensorFence&&) noexcept = default;
    TensorFence& TensorFence::operator=(TensorFence&&) noexcept = default;
    TensorFence::TensorFence(GpuBackend backend, void* event) : impl_(std::make_unique<Impl>()) {
        impl_->backend = backend;
        // Vulkan producers hand over a context-timeline value and Metal ones a
        // batch serial. There is no event object to destroy; zero is signaled.
        if (backend == GpuBackend::Vulkan || backend == GpuBackend::Metal) {
            if (backend == GpuBackend::Vulkan)
                impl_->vulkan = internal::acquire_vulkan_context();
            impl_->timeline = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(event));
            return;
        }
        if (backend != GpuBackend::CUDA || !event)
            throw std::invalid_argument("Adopted tensor fence requires a CUDA event");
#if LFS_HAS_CUDA
        impl_->event = static_cast<cudaEvent_t>(event);
#else
        throw std::runtime_error("CUDA tensor fences are unavailable in this build");
#endif
    }
    TensorFence TensorFence::adopt(GpuBackend backend, void* event) {
        return TensorFence(backend, event);
    }
    void TensorFence::record(TensorExecutionTarget target) {
        if (target.backend() != impl_->backend)
            throw std::invalid_argument("TensorFence queue backend mismatch");
        record(target.native_handle());
    }
    void TensorFence::wait_on(TensorExecutionTarget target) const {
        if (target.backend() != impl_->backend)
            throw std::invalid_argument("TensorFence queue backend mismatch");
        wait_on(target.native_handle());
    }
    void TensorFence::record(void* target) {
        if (impl_->backend == GpuBackend::Vulkan) {
            const auto id = reinterpret_cast<uint64_t>(target);
            if (id != 0 && !impl_->vulkan->recorders().owns_queue(id))
                throw std::invalid_argument("TensorFence queue backend mismatch");
            impl_->timeline = impl_->vulkan->recorders().flush_queue(id);
            return;
        }
        if (impl_->backend == GpuBackend::Metal) {
            (void)metal_target(target);
            impl_->timeline = internal::metal_queue::submit();
            return;
        }
#if LFS_HAS_CUDA
        check(cudaEventRecord(impl_->event, static_cast<cudaStream_t>(target)));
#endif
    }
    void TensorFence::wait_on(void* target) const {
        if (impl_->backend == GpuBackend::Vulkan) {
            const auto id = reinterpret_cast<uint64_t>(target);
            if (id != 0 && !impl_->vulkan->recorders().owns_queue(id))
                throw std::invalid_argument("TensorFence queue backend mismatch");
            impl_->vulkan->recorders().queue_defer_wait(id, impl_->timeline);
            return;
        }
        if (impl_->backend == GpuBackend::Metal) {
            // Later batches of every Metal queue already follow the fence's batch.
            (void)metal_target(target);
            return;
        }
#if LFS_HAS_CUDA
        check(cudaStreamWaitEvent(static_cast<cudaStream_t>(target), impl_->event, 0));
#endif
    }
    void TensorFence::wait() const {
        if (impl_->backend == GpuBackend::Vulkan) {
            if (impl_->timeline != 0)
                impl_->vulkan->wait(impl_->timeline);
            return;
        }
        if (impl_->backend == GpuBackend::Metal) {
            internal::metal_queue::wait(impl_->timeline);
            return;
        }
#if LFS_HAS_CUDA
        check(cudaEventSynchronize(impl_->event));
#endif
    }
    bool TensorFence::ready() const {
        if (impl_->backend == GpuBackend::Vulkan)
            return impl_->timeline == 0 ||
                   impl_->vulkan->completed_timeline() >= impl_->timeline;
        if (impl_->backend == GpuBackend::Metal)
            return internal::metal_queue::ready(impl_->timeline);
#if LFS_HAS_CUDA
        const auto status = cudaEventQuery(impl_->event);
        if (status == cudaErrorNotReady)
            return false;
        check(status);
#endif
        return true;
    }
    struct TensorWorkQueue::Impl {
        GpuBackend backend = GpuBackend::CUDA;
        cudaStream_t stream = nullptr;
        bool owns_stream = true;
        VulkanTimelinePoint consumer_point;
        VkDevice device = VK_NULL_HANDLE;
        VkSemaphore ready = VK_NULL_HANDLE, consumer = VK_NULL_HANDLE;
#if LFS_HAS_CUDA
        cudaExternalSemaphore_t ready_cuda = nullptr, consumer_cuda = nullptr;
#endif
#if LFS_HAS_CUDA
        std::vector<std::pair<cudaExternalSemaphore_t, VulkanTimelinePoint>> retired_consumers;
#endif
        std::unique_ptr<internal::MetalVulkanQueue> metal;
        mutable std::shared_ptr<internal::VulkanContext> queue_context;
        uint64_t recorder_id = 0;
        bool vulkan_queue = false;
        bool borrowed_queue = false;
        // Id of an owned or borrowed Metal queue; 0 is the default queue.
        uint64_t metal_queue = 0;
        bool owns_metal_queue = false;
        uint64_t counter = 0;
        TensorCompletion last;
        struct HostCallback {
            uint64_t timeline = 0;
            void (*function)(void*) = nullptr;
            void* user = nullptr;
        };
        std::mutex callback_mutex;
        std::condition_variable callback_cv;
        std::deque<HostCallback> callbacks;
        std::thread callback_thread;
        std::atomic<int> callbacks_outstanding{0};
        bool callback_stop = false;
        std::vector<VulkanTimelinePoint> retired_timelines;
        void ensure_callback_thread() {
            if (callback_thread.joinable())
                return;
            callback_thread = std::thread([this] {
                for (;;) {
                    HostCallback callback;
                    {
                        std::unique_lock lock(callback_mutex);
                        callback_cv.wait(lock, [&] { return callback_stop || !callbacks.empty(); });
                        if (callbacks.empty())
                            return;
                        callback = callbacks.front();
                        callbacks.pop_front();
                    }
                    try {
                        if (callback.timeline != 0 && backend == GpuBackend::Metal)
                            internal::metal_queue::wait_completed(callback.timeline);
                        else if (callback.timeline != 0 && queue_context)
                            queue_context->wait(callback.timeline);
                        if (callback.function)
                            callback.function(callback.user);
                    } catch (const std::exception& error) {
                        LOG_ERROR("Tensor queue host callback failed: {}", error.what());
                    } catch (...) {
                        LOG_ERROR("Tensor queue host callback failed");
                    }
                    {
                        std::lock_guard lock(callback_mutex);
                        callbacks_outstanding.fetch_sub(1, std::memory_order_acq_rel);
                        callback_cv.notify_all();
                    }
                }
            });
        }
        void stop_callbacks() {
            {
                std::lock_guard lock(callback_mutex);
                callback_stop = true;
            }
            callback_cv.notify_all();
            if (callback_thread.joinable())
                callback_thread.join();
        }
        ~Impl() {
            stop_callbacks();
#if LFS_HAS_CUDA
            if (stream && owns_stream) {
                (void)cudaStreamSynchronize(stream);
                CudaMemoryPool::instance().release_stream(stream);
                (void)cudaStreamDestroy(stream);
            }
            if (ready_cuda)
                (void)cudaDestroyExternalSemaphore(ready_cuda);
            if (consumer_cuda)
                (void)cudaDestroyExternalSemaphore(consumer_cuda);
            for (auto& [semaphore, point] : retired_consumers)
                (void)cudaDestroyExternalSemaphore(semaphore);
            if (ready && backend == GpuBackend::CUDA)
                vkDestroySemaphore(device, ready, nullptr);
#endif
        }
#if LFS_HAS_CUDA
        cudaExternalSemaphore_t import(VkSemaphore semaphore) {
            cudaExternalSemaphoreHandleDesc desc{};
#ifdef _WIN32
            const auto get = reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(vkGetDeviceProcAddr(device, "vkGetSemaphoreWin32HandleKHR"));
            VkSemaphoreGetWin32HandleInfoKHR info{VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR};
            info.semaphore = semaphore;
            info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
            HANDLE handle = nullptr;
            if (!get || get(device, &info, &handle) != VK_SUCCESS)
                throw std::runtime_error("Cannot export tensor queue semaphore");
            desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
            desc.handle.win32.handle = handle;
            cudaExternalSemaphore_t result = nullptr;
            const auto status = cudaImportExternalSemaphore(&result, &desc);
            CloseHandle(handle);
            check(status);
#else
            const auto get = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(vkGetDeviceProcAddr(device, "vkGetSemaphoreFdKHR"));
            VkSemaphoreGetFdInfoKHR info{VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
            info.semaphore = semaphore;
            info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
            int fd = -1;
            if (!get || get(device, &info, &fd) != VK_SUCCESS)
                throw std::runtime_error("Cannot export tensor queue semaphore");
            desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
            desc.handle.fd = fd;
            cudaExternalSemaphore_t result = nullptr;
            const auto status = cudaImportExternalSemaphore(&result, &desc);
            if (status != cudaSuccess) {
                close(fd);
                check(status);
            }
#endif
            return result;
        }
#endif
    };
    TensorWorkQueue::TensorWorkQueue(GpuBackend backend, Mode mode) : impl_(std::make_unique<Impl>()) {
        auto& s = *impl_;
        s.backend = backend;
        if (backend == GpuBackend::Vulkan) {
            s.queue_context = internal::acquire_vulkan_context();
            s.recorder_id = s.queue_context->recorders().create_queue(mode == Mode::LegacyOrdered);
            s.vulkan_queue = true;
            return;
        }
        if (backend == GpuBackend::Metal) {
            // Metal work runs in submission order, so every mode is ordered.
            s.metal_queue = internal::metal_queue::create();
            s.owns_metal_queue = true;
            return;
        }
        if (backend != GpuBackend::CUDA)
            throw std::runtime_error("Independent tensor queues are unsupported on Vulkan");
#if LFS_HAS_CUDA
        check(cudaStreamCreateWithFlags(&s.stream,
                                        mode == Mode::LegacyOrdered ? cudaStreamDefault : cudaStreamNonBlocking));
#else
        throw std::runtime_error("CUDA tensor queues are unavailable in this build");
#endif
    }
    TensorWorkQueue::TensorWorkQueue(GpuBackend backend, void* target) : impl_(std::make_unique<Impl>()) {
        if (backend == GpuBackend::Vulkan) {
            auto context = internal::acquire_vulkan_context();
            const auto id = reinterpret_cast<uint64_t>(target);
            if (target != nullptr && !context->recorders().owns_queue(id))
                throw std::runtime_error("Borrowed tensor queues are unsupported on Vulkan");
            impl_->backend = backend;
            impl_->owns_stream = false;
            impl_->queue_context = std::move(context);
            impl_->recorder_id = id;
            impl_->vulkan_queue = true;
            impl_->borrowed_queue = true;
            return;
        }
        if (backend == GpuBackend::Metal) {
            const auto id = reinterpret_cast<uint64_t>(target);
            if (!gpu_backend_available(GpuBackend::Metal) || !internal::metal_queue::valid(id))
                throw std::runtime_error("Borrowed Metal tensor queues must be live Metal queues");
            impl_->backend = backend;
            impl_->owns_stream = false;
            impl_->metal_queue = id;
            return;
        }
        if (backend != GpuBackend::CUDA)
            throw std::runtime_error("Borrowed tensor queues are unsupported on Vulkan");
        impl_->backend = backend;
        impl_->owns_stream = false;
#if LFS_HAS_CUDA
        impl_->stream = static_cast<cudaStream_t>(target);
#else
        throw std::runtime_error("CUDA tensor queues are unavailable in this build");
#endif
    }
    TensorWorkQueue::Scope::Scope(const TensorWorkQueue& queue)
        : backend_scope_(queue.backend()),
          previous_target_(queue.backend() == GpuBackend::CUDA || !queue.impl_->vulkan_queue
                               ? getCurrentCUDAStream()
                               : nullptr) {
        if (queue.impl_->vulkan_queue) {
            queue.impl_->queue_context->recorders().bind_queue(queue.impl_->recorder_id);
            rebound_vulkan_ = true;
            return;
        }
        // Metal queues share one timeline; there is nothing to bind.
        if (queue.backend() == GpuBackend::Metal)
            return;
        setCurrentCUDAStream(static_cast<cudaStream_t>(queue.native_handle()));
    }
    TensorWorkQueue::Scope::~Scope() {
        if (rebound_vulkan_) {
            if (const auto context = internal::try_live_vulkan_context())
                context->recorders().unbind_queue();
            return;
        }
        setCurrentCUDAStream(static_cast<cudaStream_t>(previous_target_));
    }
    void* TensorWorkQueue::native_handle() const {
        if (impl_->vulkan_queue)
            return reinterpret_cast<void*>(impl_->recorder_id);
        if (impl_->backend == GpuBackend::Metal)
            return reinterpret_cast<void*>(impl_->metal_queue);
        return impl_->stream;
    }
    GpuBackend TensorWorkQueue::backend() const { return impl_->backend; }
    bool TensorWorkQueue::ready() const {
        if (impl_->vulkan_queue) {
            if (impl_->callbacks_outstanding.load(std::memory_order_acquire) != 0)
                return false;
            return impl_->queue_context->recorders().queue_ready(impl_->recorder_id);
        }
        if (impl_->backend == GpuBackend::Metal)
            return impl_->callbacks_outstanding.load(std::memory_order_acquire) == 0 &&
                   internal::metal_queue::ready(internal::metal_queue::submit());
        if (impl_->backend != GpuBackend::CUDA)
            throw std::runtime_error("Queue polling is unsupported on Vulkan");
#if LFS_HAS_CUDA
        const auto status = cudaStreamQuery(impl_->stream);
        if (status == cudaErrorNotReady)
            return false;
        check(status);
#endif
        return true;
    }
    void TensorWorkQueue::wait() const {
        if (impl_->vulkan_queue || impl_->backend == GpuBackend::Metal) {
            if (impl_->vulkan_queue)
                impl_->queue_context->recorders().queue_wait(impl_->recorder_id);
            else
                internal::metal_queue::wait(internal::metal_queue::submit());
            std::unique_lock lock(impl_->callback_mutex);
            impl_->callback_cv.wait(lock, [&] {
                return impl_->callbacks_outstanding.load(std::memory_order_acquire) == 0;
            });
            return;
        }
        if (impl_->backend != GpuBackend::CUDA)
            throw std::runtime_error("Queue waits are unsupported on Vulkan");
#if LFS_HAS_CUDA
        check(cudaStreamSynchronize(impl_->stream));
#endif
    }
    void TensorWorkQueue::record(TensorFence& fence) const {
        if (impl_->vulkan_queue || impl_->backend == GpuBackend::Metal) {
            fence.record(TensorExecutionTarget(*this));
            return;
        }
        if (backend() != GpuBackend::CUDA)
            throw std::runtime_error("Reusable fences are unsupported on Vulkan");
        fence.record(native_handle());
    }
    void TensorWorkQueue::wait_for(const TensorFence& fence) const {
        if (impl_->vulkan_queue || impl_->backend == GpuBackend::Metal) {
            fence.wait_on(TensorExecutionTarget(*this));
            return;
        }
        if (backend() != GpuBackend::CUDA)
            throw std::runtime_error("Reusable fences are unsupported on Vulkan");
        fence.wait_on(native_handle());
    }
    void TensorWorkQueue::enqueue_host_callback(void (*callback)(void*), void* user) {
        if (impl_->vulkan_queue || impl_->backend == GpuBackend::Metal) {
            const uint64_t timeline = impl_->vulkan_queue
                                          ? impl_->queue_context->recorders().flush_queue(impl_->recorder_id)
                                          : internal::metal_queue::submit();
            {
                std::lock_guard lock(impl_->callback_mutex);
                impl_->callbacks.push_back({timeline, callback, user});
                impl_->callbacks_outstanding.fetch_add(1, std::memory_order_acq_rel);
                impl_->ensure_callback_thread();
            }
            impl_->callback_cv.notify_all();
            return;
        }
        if (impl_->backend != GpuBackend::CUDA)
            throw std::runtime_error("Queue host callbacks are unsupported on Vulkan");
#if LFS_HAS_CUDA
        check(cudaLaunchHostFunc(impl_->stream, callback, user));
#endif
    }
    void TensorWorkQueue::set_consumer_timeline(void* device, VulkanTimelinePoint point) {
        auto& s = *impl_;
        if (s.vulkan_queue) {
            const auto context_device = s.queue_context->device();
            if (device != nullptr && static_cast<VkDevice>(device) != context_device)
                throw std::invalid_argument("Cannot change the device of an exported tensor queue");
            if (s.consumer == point.semaphore && s.device == device)
                return;
            if (s.consumer != VK_NULL_HANDLE || s.consumer_point.keep_alive)
                s.retired_timelines.push_back(std::move(s.consumer_point));
            s.device = static_cast<VkDevice>(device);
            s.consumer = static_cast<VkSemaphore>(point.semaphore);
            s.consumer_point = std::move(point);
            return;
        }
        if (s.backend == GpuBackend::Metal && !s.metal) {
            if (s.consumer == point.semaphore && s.device == device)
                return;
            if (s.consumer != VK_NULL_HANDLE || s.consumer_point.keep_alive)
                s.retired_timelines.push_back(std::move(s.consumer_point));
            s.device = static_cast<VkDevice>(device);
            s.consumer = static_cast<VkSemaphore>(point.semaphore);
            s.consumer_point = std::move(point);
            return;
        }
        if (s.backend != GpuBackend::CUDA)
            throw std::runtime_error("Consumer timeline replacement is unsupported on Vulkan");
        if (s.consumer == point.semaphore && s.device == device)
            return;
        if (s.ready && s.device != device)
            throw std::invalid_argument("Cannot change the device of an exported tensor queue");
#if LFS_HAS_CUDA
        const auto old_device = s.device;
        s.device = static_cast<VkDevice>(device);
        cudaExternalSemaphore_t replacement = nullptr;
        try {
            if (point.semaphore)
                replacement = s.import(static_cast<VkSemaphore>(point.semaphore));
        } catch (...) {
            s.device = old_device;
            throw;
        }
        if (s.consumer_cuda)
            s.retired_consumers.emplace_back(s.consumer_cuda, std::move(s.consumer_point));
        s.consumer_cuda = replacement;
        s.consumer = static_cast<VkSemaphore>(point.semaphore);
        s.consumer_point = std::move(point);
#endif
    }
    void TensorWorkQueue::wait_timeline(uint64_t value) {
        auto& s = *impl_;
        if (s.vulkan_queue) {
            if (value == 0 || s.consumer == VK_NULL_HANDLE)
                return;
            s.queue_context->recorders().queue_wait_external(
                s.recorder_id, s.consumer, value, s.consumer_point.keep_alive);
            return;
        }
        if (s.backend == GpuBackend::Metal && !s.metal) {
            if (value != 0 && s.consumer != VK_NULL_HANDLE)
                internal::metal_queue::wait_vulkan_timeline(s.device, s.consumer, value);
            return;
        }
        if (s.backend != GpuBackend::CUDA)
            throw std::runtime_error("Consumer timeline waits are unsupported on Vulkan");
#if LFS_HAS_CUDA
        if (value && s.consumer_cuda) {
            cudaExternalSemaphoreWaitParams params{};
            params.params.fence.value = value;
            check(cudaWaitExternalSemaphoresAsync(&s.consumer_cuda, &params, 1, s.stream));
        }
#endif
    }
    TensorWorkQueue::TensorWorkQueue(GpuBackend backend, void* device, void* consumer) : impl_(std::make_unique<Impl>()) {
        auto& s = *impl_;
        s.backend = backend;
        s.device = static_cast<VkDevice>(device);
        s.consumer = static_cast<VkSemaphore>(consumer);
        if (backend == GpuBackend::Vulkan) {
            // The null device/consumer pair is the session storage timeline.
            // A private recorder would not compare equal to TensorExecutionTarget::current().
            s.ready = internal::acquire_vulkan_context()->timeline();
            return;
        }
        if (backend == GpuBackend::Metal) {
#ifdef LFS_TENSOR_METAL
            s.metal = internal::make_metal_vulkan_queue(device, consumer);
            s.ready = static_cast<VkSemaphore>(s.metal->timeline());
            return;
#else
            throw std::runtime_error("Metal tensor work queues are unavailable in this build");
#endif
        }
#if LFS_HAS_CUDA
        check(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking));
        if (!s.device)
            return;
        VkExportSemaphoreCreateInfo export_info{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
        export_info.handleTypes = internal::kVulkanExportSemaphoreHandleType;
        VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        type.pNext = &export_info;
        VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        info.pNext = &type;
        if (vkCreateSemaphore(s.device, &info, nullptr, &s.ready) != VK_SUCCESS)
            throw std::runtime_error("Cannot create tensor queue semaphore");
        s.ready_cuda = s.import(s.ready);
        if (s.consumer)
            s.consumer_cuda = s.import(s.consumer);
#else
        throw std::runtime_error("CUDA tensor work queues are unavailable in this build");
#endif
    }
    TensorWorkQueue::~TensorWorkQueue() {
        try {
            if (impl_->vulkan_queue && !impl_->borrowed_queue && impl_->recorder_id != 0 &&
                impl_->queue_context)
                impl_->queue_context->recorders().destroy_queue(impl_->recorder_id);
            if (impl_->owns_metal_queue)
                internal::metal_queue::destroy(impl_->metal_queue);
            // Stream ownership and import lifetime are independent. In particular,
            // a borrowed default stream can still have outstanding timeline waits.
            if ((impl_->vulkan_queue || impl_->backend == GpuBackend::Metal) &&
                (impl_->consumer != VK_NULL_HANDLE || !impl_->retired_timelines.empty() ||
                 impl_->callbacks_outstanding.load(std::memory_order_acquire) != 0))
                wait();
#if LFS_HAS_CUDA
            if (impl_->backend == GpuBackend::CUDA &&
                (impl_->ready_cuda || impl_->consumer_cuda || !impl_->retired_consumers.empty()))
                wait();
#endif
            impl_->last.wait();
        } catch (const std::exception& e) {
            LOG_ERROR("Tensor queue retained after incomplete work: {}", e.what());
            (void)impl_.release();
        }
    }
    void* TensorWorkQueue::timeline() const { return impl_->ready; }
    TensorCompletion TensorWorkQueue::execute(const std::function<void()>& commands, uint64_t value) {
        auto& s = *impl_;
        const GpuBackendScope backend(s.backend);
        const CUDAStreamGuard stream(s.stream);
        std::optional<TensorWorkQueue::Scope> vulkan_scope;
        if (s.vulkan_queue)
            vulkan_scope.emplace(*this);
        uint64_t consumer_completion = 0;
        if (value && s.consumer) {
            if (s.backend == GpuBackend::CUDA) {
#if LFS_HAS_CUDA
                cudaExternalSemaphoreWaitParams params{};
                params.params.fence.value = value;
                check(cudaWaitExternalSemaphoresAsync(&s.consumer_cuda, &params, 1, s.stream));
#else
                throw std::runtime_error("CUDA tensor work queues are unavailable in this build");
#endif
            } else if (s.metal) {
                s.metal->wait(value);
            } else if (s.backend == GpuBackend::Metal) {
                internal::metal_queue::wait_vulkan_timeline(s.device, s.consumer, value);
            } else {
                const auto ctx = internal::acquire_vulkan_context();
                consumer_completion = ctx->recorders().wait_external({}, s.consumer, value, {});
            }
        }
        TensorCompletion result;
        try {
            commands();
            if (s.backend == GpuBackend::CUDA) {
#if LFS_HAS_CUDA
                ++s.counter;
                if (s.ready_cuda) {
                    cudaExternalSemaphoreSignalParams params{};
                    params.params.fence.value = s.counter;
                    check(cudaSignalExternalSemaphoresAsync(&s.ready_cuda, &params, 1, s.stream));
                }
                result = TensorCompletionAccess::cuda(s.stream, {s.ready, s.counter, {}});
#else
                throw std::runtime_error("CUDA tensor work queues are unavailable in this build");
#endif
            } else if (s.metal) {
                result = s.metal->signal();
            } else if (s.backend == GpuBackend::Metal) {
                result = TensorCompletionAccess::metal(internal::metal_queue::submit());
            } else {
                result = TensorCompletionAccess::vulkan(
                    std::max(consumer_completion, internal::acquire_vulkan_context()->recorders().flush_current()));
            }
        } catch (...) {
            if (s.backend == GpuBackend::CUDA) {
#if LFS_HAS_CUDA
                (void)cudaStreamSynchronize(s.stream);
#endif
            } else if (s.metal) {
                s.metal->signal().wait();
            } else if (s.backend == GpuBackend::Metal) {
                internal::metal_queue::wait_completed(internal::metal_queue::submit());
            } else {
                internal::acquire_vulkan_context()->recorders().wait_all();
            }
            throw;
        }
        s.last = result;
        return result;
    }
    namespace {
        void require_readback_staging(const Tensor& staging, const GpuBackend backend, const size_t bytes) {
            if (gpu_backend_of(staging) != backend || !staging.is_contiguous() ||
                staging.dtype() != DataType::UInt8 || staging.bytes() < bytes)
                throw std::invalid_argument("Readback staging must cover a slot on the queue backend");
        }
    } // namespace
    struct TensorReadbackRing::Impl {
        struct Slot {
            void* host = nullptr;
            std::optional<internal::StorageRef> vulkan_host;
            // Metal storage is shared, so a slot is a tensor the CPU reads in place.
            Tensor metal_host;
            std::unique_ptr<TensorFence> fence;
            std::vector<Tensor> sources;
            bool sealed = false;
        };
        GpuBackend backend = GpuBackend::CUDA;
        TensorWorkQueue* queue = nullptr;
        std::shared_ptr<internal::VulkanContext> vulkan;
        size_t bytes = 0;
        std::vector<Slot> slots;
        const Tensor* staging = nullptr;
        Tensor scratch;
        ~Impl() {
            if (!queue)
                return;
            try {
                queue->wait();
            } catch (const std::exception& error) { LOG_WARN("Readback ring drain failed: {}", error.what()); }
            if (vulkan) {
                for (auto& slot : slots) {
                    if (slot.vulkan_host)
                        vulkan->memory().deallocate(*slot.vulkan_host);
                }
            }
#if LFS_HAS_CUDA
            if (backend == GpuBackend::CUDA) {
                for (auto& slot : slots)
                    if (slot.host)
                        (void)cudaFreeHost(slot.host);
            }
#endif
        }
    };
    TensorReadbackRing::TensorReadbackRing(GpuBackend backend, size_t slots, size_t bytes, TensorWorkQueue& queue, const Tensor* staging)
        : impl_(std::make_unique<Impl>()) {
        if (backend == GpuBackend::Vulkan) {
            if (!queue.impl_->vulkan_queue)
                throw std::runtime_error("Packed tensor readback rings need an owned Vulkan work queue");
        } else if (backend != GpuBackend::CUDA && backend != GpuBackend::Metal) {
            throw std::runtime_error("Packed tensor readback rings are unsupported on this backend");
        }
        if (queue.backend() != backend)
            throw std::invalid_argument("Readback ring queue backend mismatch");
        if (!slots || !bytes)
            throw std::invalid_argument("Readback ring dimensions must be nonzero");
        auto& s = *impl_;
        s.backend = backend;
        s.queue = &queue;
        s.bytes = bytes;
        s.staging = staging;
        if (staging && staging->is_valid())
            require_readback_staging(*staging, backend, bytes);
        s.slots.resize(slots);
        if (backend == GpuBackend::Vulkan) {
            s.vulkan = internal::acquire_vulkan_context();
            for (auto& slot : s.slots) {
                auto storage = s.vulkan->memory().allocate_readback(bytes);
                slot.host = s.vulkan->memory().mapped_pointer(storage);
                slot.vulkan_host = storage;
                slot.fence = std::make_unique<TensorFence>(backend);
            }
            return;
        }
        if (backend == GpuBackend::Metal) {
            const GpuBackendScope scope(backend);
            for (auto& slot : s.slots) {
                slot.metal_host = Tensor::zeros({bytes}, Device::GPU, DataType::UInt8);
                slot.host = internal::metal_queue::host(internal::storage_ref(slot.metal_host));
                slot.fence = std::make_unique<TensorFence>(backend);
            }
            return;
        }
#if LFS_HAS_CUDA
        for (auto& slot : s.slots) {
            check(cudaHostAlloc(&slot.host, bytes, cudaHostAllocPortable));
            std::memset(slot.host, 0, bytes);
            slot.fence = std::make_unique<TensorFence>(backend);
        }
#else
        throw std::runtime_error("CUDA readback rings are unavailable in this build");
#endif
    }
    TensorReadbackRing::~TensorReadbackRing() = default;
    bool TensorReadbackRing::prefers_recycled_host_staging() const noexcept {
        return impl_->backend == GpuBackend::Metal;
    }
    void TensorReadbackRing::enqueue(const Tensor& source, size_t offset, size_t bytes,
                                     size_t index, size_t destination, bool stage) {
        auto& s = *impl_;
        auto& slot = s.slots.at(index);
        if (slot.sealed)
            throw std::logic_error("Readback slot must be released before reuse");
        if (!source.is_valid() || gpu_backend_of(source) != s.backend)
            throw std::invalid_argument("Readback ring requires a tensor on its queue backend");
        if (offset > source.bytes() || bytes > source.bytes() - offset ||
            destination > s.bytes || bytes > s.bytes - destination)
            throw std::out_of_range("Readback ring byte range exceeds source or slot");
        if (!bytes)
            return;
        TensorWorkQueue::Scope scope(*s.queue);
        slot.sources.push_back(source.contiguous());
        const auto& retained = slot.sources.back();
        pin_operands({&retained});
        if (s.backend == GpuBackend::Vulkan || s.backend == GpuBackend::Metal) {
            const bool metal = s.backend == GpuBackend::Metal;
            auto input = internal::storage_ref(retained);
            input.byte_offset += offset;
            auto output = metal ? internal::storage_ref(slot.metal_host) : *slot.vulkan_host;
            output.byte_offset += destination;
            // Staging only serves the GPU copy.
            if (metal && internal::metal_queue::host_copy_if_idle(output, input, bytes))
                return;
            const auto copy = [&](const internal::StorageRef from, const internal::StorageRef to) {
                const internal::CopyRequest request{.src = from, .dst = to, .bytes = bytes};
                if (metal)
                    internal::backend_ops(GpuBackend::Metal).copy_device_to_device(request);
                else
                    s.vulkan->memory().copy_device_to_device(request);
            };
            if (stage) {
                const bool borrowed = s.staging && s.staging->is_valid();
                if (borrowed) {
                    require_readback_staging(*s.staging, s.backend, s.bytes);
                    // Retained with the slot so a released owner cannot free it mid-copy.
                    slot.sources.push_back(*s.staging);
                } else if (!s.scratch.is_valid()) {
                    s.scratch = Tensor::empty({s.bytes}, Device::GPU, DataType::UInt8);
                }
                Tensor& staging = borrowed ? slot.sources.back() : s.scratch;
                copy(input, internal::storage_ref(staging));
                input = internal::storage_ref(staging);
            }
            if (metal)
                copy(input, output);
            else
                (void)s.vulkan->memory().copy_to_readback(input, output, bytes);
            return;
        }
#if LFS_HAS_CUDA
        const auto stream = static_cast<cudaStream_t>(s.queue->native_handle());
        prepare_inputs_for_stream({&retained}, stream);
        const void* input = static_cast<const std::byte*>(retained.data_ptr()) + offset;
        if (stage && bytes) {
            const bool borrowed = s.staging && s.staging->is_valid();
            if (borrowed) {
                require_readback_staging(*s.staging, s.backend, s.bytes);
                // Retained with the slot so a released owner cannot free it mid-copy.
                slot.sources.push_back(*s.staging);
            } else if (!s.scratch.is_valid()) {
                s.scratch = Tensor::empty({s.bytes}, Device::GPU, DataType::UInt8);
            }
            Tensor& staging = borrowed ? slot.sources.back() : s.scratch;
            check(cudaMemcpyAsync(staging.data_ptr(), input, bytes, cudaMemcpyDeviceToDevice, stream));
            input = staging.data_ptr();
        }
        if (bytes)
            check(cudaMemcpyAsync(static_cast<std::byte*>(slot.host) + destination,
                                  input, bytes, cudaMemcpyDeviceToHost, stream));
#endif
    }
    const TensorFence& TensorReadbackRing::seal(size_t index) {
        auto& slot = impl_->slots.at(index);
        if (slot.sealed)
            throw std::logic_error("Readback slot is already sealed");
        impl_->queue->record(*slot.fence);
        slot.sealed = true;
        return *slot.fence;
    }
    bool TensorReadbackRing::poll(size_t index) const {
        const auto& slot = impl_->slots.at(index);
        if (!slot.sealed)
            throw std::logic_error("Readback slot must be sealed before polling");
        return slot.fence->ready();
    }
    void TensorReadbackRing::wait(size_t index) const {
        const auto& slot = impl_->slots.at(index);
        if (!slot.sealed)
            throw std::logic_error("Readback slot must be sealed before waiting");
        slot.fence->wait();
    }
    void TensorReadbackRing::release(size_t index) {
        auto& slot = impl_->slots.at(index);
        if ((slot.sealed && !slot.fence->ready()) ||
            (!slot.sealed && !slot.sources.empty() && !impl_->queue->ready()))
            throw std::logic_error("Cannot release a pending readback slot");
        slot.sources.clear();
        slot.sealed = false;
    }
    std::span<std::byte> TensorReadbackRing::slot_bytes(size_t index) {
        return {static_cast<std::byte*>(impl_->slots.at(index).host), impl_->bytes};
    }
} // namespace lfs::core
