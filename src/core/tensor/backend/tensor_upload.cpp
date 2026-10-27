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
#include "gpu_backend_ops.hpp"
#include "tensor_completion.hpp"
#include "tensor_vulkan_interop.hpp"
#ifdef LFS_TENSOR_VULKAN
#include "vulkan/vk_context.hpp"
#include "vulkan/vk_memory.hpp"
#if LFS_HAS_CUDA
#include "vulkan/vk_cuda_bridge.hpp"
#endif
#include "vulkan/vk_recorder.hpp"
#endif
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
        Tensor source, destination;
        internal::GpuBackendOps* ops = nullptr;
        std::unique_ptr<internal::BackendUploadBuffer> upload;
        bool pending = false;

        void prepare(Tensor target) {
            auto& backend = internal::backend_ops_for(target);
            if (ops != &backend) {
                auto replacement = backend.create_upload_buffer();
                if (!replacement)
                    throw std::runtime_error("Tensor backend does not support uploads");
                upload = std::move(replacement);
                ops = &backend;
            }
            destination = std::move(target);
            source = {};
            pending = false;
        }
        void finish(internal::BackendUploadSubmission status) {
            if (status.source_consumed)
                source = {};
            if (status.complete) {
                pending = false;
                source = {};
                destination = {};
            }
        }
        void failed_enqueue() {
            upload->settle_enqueue_failure();
            pending = false;
            source = {};
            destination = {};
            upload.reset();
            ops = nullptr;
        }
        void enqueue(Tensor target, const Tensor& input, internal::BackendQueueTarget queue) {
            prepare(std::move(target));
            source = input;
            pin_operands({&source, &destination});
            pending = true;
            try {
                finish(upload->enqueue(destination, source, queue, false));
            } catch (...) {
                failed_enqueue();
                throw;
            }
        }
        void enqueue_bytes(Tensor target, std::span<const std::byte> input, internal::BackendQueueTarget queue, bool in_batch) {
            prepare(std::move(target));
            pin_operands({&destination});
            pending = true;
            try {
                finish(upload->enqueue_bytes(destination, input, queue, in_batch));
            } catch (...) {
                failed_enqueue();
                throw;
            }
        }
    };
    TensorUpload::TensorUpload() = default;
    TensorUpload::~TensorUpload() {
        try {
            wait();
        } catch (const std::exception& error) {
            if (impl_ && impl_->upload && impl_->upload->abandon_after_failure()) {
                LOG_ERROR("Tensor upload failed; cleanup delegated to native ownership: {}", error.what());
            } else {
                LOG_ERROR("Tensor upload storage retained after incomplete transfer: {}", error.what());
                (void)impl_.release();
            }
        }
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
    namespace {
        void validate_upload(const Tensor& destination, const Tensor& source) {
            if (!gpu_backend_of(destination) || source.device() != Device::CPU || !source.is_contiguous() ||
                !destination.is_contiguous() || destination.dtype() != source.dtype() || destination.bytes() != source.bytes())
                throw std::invalid_argument("TensorUpload requires matching contiguous CPU and GPU tensors");
        }
        void validate_upload(const Tensor& destination, std::span<const std::byte> source) {
            if (!gpu_backend_of(destination) || !destination.is_contiguous() || destination.bytes() != source.size())
                throw std::invalid_argument("TensorUpload byte span must match a contiguous destination");
        }
    } // namespace
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
    void TensorUpload::enqueue_in_batch(Tensor destination, std::span<const std::byte> source) {
        if (pending())
            throw std::logic_error("TensorUpload already pending");
        validate_upload(destination, source);
        if (!impl_)
            impl_ = std::make_unique<Impl>();
        impl_->enqueue_bytes(std::move(destination), source, {nullptr, false}, true);
    }
    void TensorUpload::enqueue(Tensor destination, const Tensor& source) {
        if (pending())
            throw std::logic_error("TensorUpload already pending");
        validate_upload(destination, source);
        if (!impl_)
            impl_ = std::make_unique<Impl>();
        impl_->enqueue(std::move(destination), source, {});
    }
    void TensorUpload::enqueue(Tensor destination, const Tensor& source, void* target) {
        if (pending())
            throw std::logic_error("TensorUpload already pending");
        validate_upload(destination, source);
        if (!impl_)
            impl_ = std::make_unique<Impl>();
        impl_->enqueue(std::move(destination), source, {target, false});
    }
    void TensorUpload::enqueue(Tensor destination, std::span<const std::byte> source, void* target) {
        if (pending())
            throw std::logic_error("TensorUpload already pending");
        validate_upload(destination, source);
        if (!impl_)
            impl_ = std::make_unique<Impl>();
        impl_->enqueue_bytes(std::move(destination), source, {target, false}, false);
    }
    bool TensorUpload::poll() {
        if (!pending())
            return true;
        if (!impl_->upload->poll())
            return false;
        impl_->pending = false;
        impl_->source = {};
        impl_->destination = {};
        return true;
    }
    void TensorUpload::wait() {
        if (!pending())
            return;
        impl_->upload->wait();
        (void)poll();
    }

    struct TensorFence::Impl {
        GpuBackend backend;
        std::unique_ptr<internal::BackendFence> fence;
    };
    TensorFence::TensorFence(GpuBackend backend) : impl_(std::make_unique<Impl>(Impl{backend, {}})) {
        impl_->fence = internal::backend_ops(backend).create_fence();
        if (!impl_->fence)
            throw std::runtime_error("Tensor backend does not support reusable fences");
    }
    TensorFence::TensorFence(GpuBackend backend, void* imported) : impl_(std::make_unique<Impl>(Impl{backend, {}})) {
        impl_->fence = internal::backend_ops(backend).create_fence(imported);
        if (!impl_->fence)
            throw std::invalid_argument("Tensor backend cannot import this fence handle");
    }
    TensorFence::~TensorFence() = default;
    TensorFence::TensorFence(TensorFence&&) noexcept = default;
    TensorFence& TensorFence::operator=(TensorFence&&) noexcept = default;
    TensorFence TensorFence::adopt(GpuBackend backend, void* event) { return TensorFence(backend, event); }
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
    void TensorFence::record(void* target) { impl_->fence->record(internal::ExecContext{static_cast<cudaStream_t>(target)}); }
    void TensorFence::wait_on(void* target) const { impl_->fence->wait_on(internal::ExecContext{static_cast<cudaStream_t>(target)}); }
    void TensorFence::wait() const { impl_->fence->wait(); }
    bool TensorFence::ready() const { return impl_->fence->ready(); }
    struct TensorWorkQueue::Impl {
        GpuBackend backend = GpuBackend::CUDA;
        [[no_unique_address]] internal::CompiledBackendState<GpuBackend::HIP, internal::BackendQueueLease> hip_queue;
        cudaStream_t stream = nullptr;
        bool owns_stream = true;
        VulkanTimelinePoint consumer_point;
        bool host_consumer = false;
        std::optional<bool> consumer_external_interop;
        void wait_host_consumer(uint64_t value) {
#ifdef LFS_TENSOR_VULKAN
            if (!value || !consumer)
                return;
            VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
            info.semaphoreCount = 1;
            info.pSemaphores = &consumer;
            info.pValues = &value;
            internal::vk_check(nullptr, vkWaitSemaphores(device, &info, 30'000'000'000ull),
                               "vkWaitSemaphores(foreign GPU consumer)");
#else
            (void)value;
            throw std::runtime_error("Foreign GPU timeline waits require Vulkan");
#endif
        }
#ifdef LFS_TENSOR_VULKAN
        VkDevice device = VK_NULL_HANDLE;
        VkSemaphore ready = VK_NULL_HANDLE, consumer = VK_NULL_HANDLE;
#else
        void* device = nullptr;
        void* ready = nullptr;
        void* consumer = nullptr;
#endif
#if LFS_HAS_CUDA
        cudaExternalSemaphore_t ready_cuda = nullptr, consumer_cuda = nullptr;
#endif
#if LFS_HAS_CUDA
        std::vector<std::pair<cudaExternalSemaphore_t, VulkanTimelinePoint>> retired_consumers;
#endif
        std::unique_ptr<internal::MetalVulkanQueue> metal;
#ifdef LFS_TENSOR_VULKAN
        mutable std::shared_ptr<internal::VulkanContext> queue_context;
#endif
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
                        else if (callback.timeline != 0
#ifdef LFS_TENSOR_VULKAN
                                 && queue_context
#endif
                        )
#ifdef LFS_TENSOR_VULKAN
                            queue_context->wait(callback.timeline);
#else
                            throw std::runtime_error("Vulkan tensor callbacks are unavailable in this build");
#endif
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
            if (ready && backend == GpuBackend::CUDA) {
#ifdef LFS_TENSOR_VULKAN
                vkDestroySemaphore(device, ready, nullptr);
#endif
            }
#endif
        }
#if LFS_HAS_CUDA && defined(LFS_TENSOR_VULKAN)
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
        if (gpu_backend_compiled(GpuBackend::HIP) && backend == GpuBackend::HIP) {
            // Logical queues share the native HIP submission queue, as Metal
            // work queues share submission ordering. No concurrency is implied.
            s.hip_queue.get() = internal::backend_ops(backend).execution_lease({});
            s.owns_stream = false;
            return;
        }

        if (backend == GpuBackend::Vulkan) {
#ifdef LFS_TENSOR_VULKAN
            s.queue_context = internal::acquire_vulkan_context();
            s.recorder_id = s.queue_context->recorders().create_queue(mode == Mode::LegacyOrdered);
            s.vulkan_queue = true;
            return;
#else
            throw std::runtime_error("Vulkan tensor queues are unavailable in this build");
#endif
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
        activate_selected_cuda_device();
        check(cudaStreamCreateWithFlags(&s.stream,
                                        mode == Mode::LegacyOrdered ? cudaStreamDefault : cudaStreamNonBlocking));
#else
        throw std::runtime_error("CUDA tensor queues are unavailable in this build");
#endif
    }
    TensorWorkQueue::TensorWorkQueue(GpuBackend backend, void* target) : impl_(std::make_unique<Impl>()) {
        if (gpu_backend_compiled(GpuBackend::HIP) && backend == GpuBackend::HIP) {
            if (target)
                throw std::invalid_argument("HIP borrowed queues require the native default target");
            impl_->backend = backend;
            impl_->owns_stream = false;
            impl_->hip_queue.get() = internal::backend_ops(backend).execution_lease({});
            return;
        }

        if (backend == GpuBackend::Vulkan) {
#ifdef LFS_TENSOR_VULKAN
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
#else
            throw std::runtime_error("Vulkan tensor queues are unavailable in this build");
#endif
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
#ifdef LFS_TENSOR_VULKAN
            queue.impl_->queue_context->recorders().bind_queue(queue.impl_->recorder_id);
            rebound_vulkan_ = true;
            return;
#else
            throw std::runtime_error("Vulkan tensor queues are unavailable in this build");
#endif
        }
        // Metal queues share one timeline; there is nothing to bind.
        if (queue.backend() == GpuBackend::Metal)
            return;
        setCurrentCUDAStream(static_cast<cudaStream_t>(queue.native_handle()));
    }
    TensorWorkQueue::Scope::~Scope() {
        if (rebound_vulkan_) {
#ifdef LFS_TENSOR_VULKAN
            if (const auto context = internal::try_live_vulkan_context())
                context->recorders().unbind_queue();
            return;
#else
            std::terminate();
#endif
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
        if (gpu_backend_compiled(GpuBackend::HIP) && impl_->backend == GpuBackend::HIP)
            return internal::backend_ops(impl_->backend).queue_ready({});

        if (impl_->vulkan_queue) {
#ifdef LFS_TENSOR_VULKAN
            if (impl_->callbacks_outstanding.load(std::memory_order_acquire) != 0)
                return false;
            return impl_->queue_context->recorders().queue_ready(impl_->recorder_id);
#else
            return false;
#endif
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
        if (gpu_backend_compiled(GpuBackend::HIP) && impl_->backend == GpuBackend::HIP) {
            internal::backend_ops(impl_->backend).synchronize_stream({});
            return;
        }

        if (impl_->vulkan_queue || impl_->backend == GpuBackend::Metal) {
            if (impl_->vulkan_queue) {
#ifdef LFS_TENSOR_VULKAN
                impl_->queue_context->recorders().queue_wait(impl_->recorder_id);
#else
                throw std::runtime_error("Vulkan tensor queues are unavailable in this build");
#endif
            } else
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
        fence.record(TensorExecutionTarget(*this));
    }
    void TensorWorkQueue::wait_for(const TensorFence& fence) const {
        fence.wait_on(TensorExecutionTarget(*this));
    }
    void TensorWorkQueue::enqueue_host_callback(void (*callback)(void*), void* user) {
        if (gpu_backend_compiled(GpuBackend::HIP) && backend() == GpuBackend::HIP) {
            internal::backend_ops(backend()).enqueue_host_callback(callback, user, {});
            return;
        }

        if (impl_->vulkan_queue || impl_->backend == GpuBackend::Metal) {
            uint64_t timeline = 0;
            if (impl_->vulkan_queue) {
#ifdef LFS_TENSOR_VULKAN
                timeline = impl_->queue_context->recorders().flush_queue(impl_->recorder_id);
#else
                throw std::runtime_error("Vulkan tensor queues are unavailable in this build");
#endif
            } else {
                timeline = internal::metal_queue::submit();
            }
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
        if (gpu_backend_compiled(GpuBackend::HIP) && s.backend == GpuBackend::HIP) {
#ifdef LFS_TENSOR_VULKAN
            if (!device)
                throw std::invalid_argument("HIP consumer timeline requires a Vulkan device");
            if (s.consumer == point.semaphore && s.device == device)
                return;
            if (s.consumer || s.consumer_point.keep_alive)
                s.retired_timelines.push_back(std::move(s.consumer_point));
            s.host_consumer = true;
            s.device = static_cast<VkDevice>(device);
            s.consumer = static_cast<VkSemaphore>(point.semaphore);
            s.consumer_point = std::move(point);
            return;
#else
            throw std::runtime_error("HIP consumer timeline requires Vulkan support");
#endif
        }
        if (s.vulkan_queue || s.backend == GpuBackend::Vulkan) {
#ifdef LFS_TENSOR_VULKAN
            if (!s.vulkan_queue && device == nullptr)
                throw std::runtime_error("A backend-owned Vulkan queue requires an explicit consumer device");
            const auto context_device = s.queue_context ? s.queue_context->device() : internal::acquire_vulkan_context()->device();
            s.host_consumer = device != nullptr && static_cast<VkDevice>(device) != context_device;
            if (s.consumer == point.semaphore && s.device == device)
                return;
            if (s.consumer != VK_NULL_HANDLE || s.consumer_point.keep_alive)
                s.retired_timelines.push_back(std::move(s.consumer_point));
            s.device = static_cast<VkDevice>(device);
            s.consumer = static_cast<VkSemaphore>(point.semaphore);
            s.consumer_point = std::move(point);
            return;
#else
            throw std::runtime_error("Vulkan tensor queues are unavailable in this build");
#endif
        }
        if (s.backend == GpuBackend::Metal && !s.metal) {
            if (s.consumer == point.semaphore && s.device == device)
                return;
            if (s.consumer != nullptr || s.consumer_point.keep_alive)
                s.retired_timelines.push_back(std::move(s.consumer_point));
#ifdef LFS_TENSOR_VULKAN
            s.device = static_cast<VkDevice>(device);
            s.consumer = static_cast<VkSemaphore>(point.semaphore);
#else
            s.device = device;
            s.consumer = point.semaphore;
#endif
            s.consumer_point = std::move(point);
            return;
        }
        if (s.backend != GpuBackend::CUDA)
            throw std::runtime_error("Consumer timeline replacement is unsupported on Vulkan");
        if (s.consumer == point.semaphore && s.device == device)
            return;
        if (!s.consumer_external_interop.value_or(cuda_vulkan_direct_interop())) {
#ifdef LFS_TENSOR_VULKAN
            s.host_consumer = true;
            if (s.consumer || s.consumer_point.keep_alive)
                s.retired_timelines.push_back(std::move(s.consumer_point));
            s.device = static_cast<VkDevice>(device);
            s.consumer = static_cast<VkSemaphore>(point.semaphore);
            s.consumer_point = std::move(point);
            return;
#endif
        }
        if (s.ready && s.device != device)
            throw std::invalid_argument("Cannot change the device of an exported tensor queue");
#if LFS_HAS_CUDA
#ifdef LFS_TENSOR_VULKAN
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
#else
        throw std::runtime_error("CUDA/Vulkan timeline interop is unavailable in this build");
#endif
#endif
    }
    void TensorWorkQueue::wait_timeline(uint64_t value) {
        auto& s = *impl_;
        if (s.host_consumer) {
            s.wait_host_consumer(value);
            return;
        }
        if (s.vulkan_queue) {
#ifdef LFS_TENSOR_VULKAN
            if (value == 0 || s.consumer == VK_NULL_HANDLE)
                return;
            s.queue_context->recorders().queue_wait_external(
                s.recorder_id, s.consumer, value, s.consumer_point.keep_alive);
            return;
#else
            throw std::runtime_error("Vulkan tensor queues are unavailable in this build");
#endif
        }
        if (s.backend == GpuBackend::Metal && !s.metal) {
            if (value != 0 && s.consumer != nullptr) {
#ifdef LFS_TENSOR_VULKAN
                internal::metal_queue::wait_vulkan_timeline(s.device, s.consumer, value);
#else
                throw std::runtime_error("Metal/Vulkan timeline interop is unavailable in this build");
#endif
            }
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
    TensorWorkQueue::TensorWorkQueue(GpuBackend backend, void* device, void* consumer)
        : TensorWorkQueue(backend, device, consumer, true) {}

    TensorWorkQueue::TensorWorkQueue(GpuBackend backend, void* device, void* consumer, bool external_interop) : impl_(std::make_unique<Impl>()) {
        auto& s = *impl_;
        s.backend = backend;
        s.consumer_external_interop = external_interop;
#ifdef LFS_TENSOR_VULKAN
        s.device = static_cast<VkDevice>(device);
        s.consumer = static_cast<VkSemaphore>(consumer);
#else
        s.device = device;
        s.consumer = consumer;
#endif
        if (gpu_backend_compiled(GpuBackend::HIP) && backend == GpuBackend::HIP) {
            if (consumer && !device)
                throw std::runtime_error("HIP consumer timeline requires a Vulkan device");
#ifndef LFS_TENSOR_VULKAN
            if (device)
                throw std::runtime_error("HIP/Vulkan consumer queues are unavailable in this build");
#endif
            s.hip_queue.get() = internal::backend_ops(backend).execution_lease({});
            s.owns_stream = false;
            s.host_consumer = true;
            s.consumer_external_interop = false;
            return;
        }
        if (backend == GpuBackend::Vulkan) {
#ifdef LFS_TENSOR_VULKAN
            // The null device/consumer pair is the session storage timeline.
            // A private recorder would not compare equal to TensorExecutionTarget::current().
            const auto context = internal::acquire_vulkan_context();
            s.host_consumer = s.device != VK_NULL_HANDLE && s.device != context->device();
            s.ready = context->timeline();
            return;
#else
            throw std::runtime_error("Vulkan tensor queues are unavailable in this build");
#endif
        }
        if (backend == GpuBackend::Metal) {
#if defined(LFS_TENSOR_METAL) && defined(LFS_TENSOR_VULKAN)
            s.metal = internal::make_metal_vulkan_queue(device, consumer);
            s.ready = static_cast<VkSemaphore>(s.metal->timeline());
            return;
#elif defined(LFS_TENSOR_METAL)
            // No Vulkan device: the consumer is the renderer's Metal shared event.
            s.metal = internal::make_metal_work_queue(consumer);
            return;
#else
            throw std::runtime_error("Metal/Vulkan interoperable work queues are unavailable in this build");
#endif
        }
#if LFS_HAS_CUDA
#ifdef LFS_TENSOR_VULKAN
        activate_selected_cuda_device();
        check(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking));
        if (!s.consumer_external_interop.value_or(cuda_vulkan_direct_interop())) {
            s.host_consumer = true;
            return;
        }
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
        throw std::runtime_error("CUDA/Vulkan interoperable work queues are unavailable in this build");
#endif
#else
        throw std::runtime_error("CUDA tensor work queues are unavailable in this build");
#endif
    }
    TensorWorkQueue::~TensorWorkQueue() {
        if (gpu_backend_compiled(GpuBackend::HIP) && impl_->backend == GpuBackend::HIP) {
            if (!internal::backend_ops(impl_->backend).drain_queue_for_destruction()) {
                (void)impl_.release();
                return;
            }
            // The entire native queue has drained, including callbacks. A saved
            // snapshot may still report an earlier, already consumed error.
            // Its owned runtime resources handle any deferred event cleanup;
            // retaining the outer queue cannot make that cleanup safer.
            try {
                impl_->last.wait();
            } catch (const std::exception& e) {
                LOG_ERROR("HIP tensor queue completion failed after synchronization: {}", e.what());
            }
            return;
        }
        try {
#ifdef LFS_TENSOR_VULKAN
            if (impl_->vulkan_queue && !impl_->borrowed_queue && impl_->recorder_id != 0 &&
                impl_->queue_context)
                impl_->queue_context->recorders().destroy_queue(impl_->recorder_id);
#endif
            if (impl_->owns_metal_queue)
                internal::metal_queue::destroy(impl_->metal_queue);
            // Stream ownership and import lifetime are independent. In particular,
            // a borrowed default stream can still have outstanding timeline waits.
            if ((impl_->vulkan_queue || impl_->backend == GpuBackend::Metal) &&
                (impl_->consumer != nullptr || !impl_->retired_timelines.empty() ||
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
    void* TensorWorkQueue::timeline() const { return impl_->host_consumer ? nullptr : impl_->ready; }
    TensorCompletion TensorWorkQueue::execute(const std::function<void()>& commands, uint64_t value) {
        auto& s = *impl_;
        const GpuBackendScope backend(s.backend);
        const CUDAStreamGuard stream(s.stream);
        std::optional<TensorWorkQueue::Scope> vulkan_scope;
        if (s.vulkan_queue)
            vulkan_scope.emplace(*this);
        uint64_t consumer_completion = 0;
        if (value && s.consumer) {
            if (s.host_consumer) {
                s.wait_host_consumer(value);
            } else if (s.backend == GpuBackend::CUDA) {
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
#ifdef LFS_TENSOR_VULKAN
                internal::metal_queue::wait_vulkan_timeline(s.device, s.consumer, value);
#else
                throw std::runtime_error("Metal/Vulkan timeline interop is unavailable in this build");
#endif
            } else {
#ifdef LFS_TENSOR_VULKAN
                const auto ctx = internal::acquire_vulkan_context();
                consumer_completion = ctx->recorders().wait_external({}, s.consumer, value, {});
#else
                throw std::runtime_error("Vulkan tensor queues are unavailable in this build");
#endif
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
            } else if (gpu_backend_compiled(GpuBackend::HIP) && s.backend == GpuBackend::HIP) {
                result = TensorCompletionAccess::hip();
            } else if (s.metal) {
                result = s.metal->signal();
            } else if (s.backend == GpuBackend::Metal) {
                result = TensorCompletionAccess::metal(internal::metal_queue::submit());
            } else {
#ifdef LFS_TENSOR_VULKAN
                result = TensorCompletionAccess::vulkan(
                    std::max(consumer_completion, internal::acquire_vulkan_context()->recorders().flush_current()));
#else
                throw std::runtime_error("Vulkan tensor queues are unavailable in this build");
#endif
            }
        } catch (...) {
            if (s.backend == GpuBackend::CUDA) {
#if LFS_HAS_CUDA
                (void)cudaStreamSynchronize(s.stream);
#endif
            } else if (gpu_backend_compiled(GpuBackend::HIP) && s.backend == GpuBackend::HIP) {
                internal::backend_ops(s.backend).synchronize_stream({});
            } else if (s.metal) {
                s.metal->signal().wait();
            } else if (s.backend == GpuBackend::Metal) {
                internal::metal_queue::wait_completed(internal::metal_queue::submit());
            } else {
#ifdef LFS_TENSOR_VULKAN
                internal::acquire_vulkan_context()->recorders().wait_all();
#endif
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
            std::unique_ptr<internal::BackendReadbackSlot> native;
            std::unique_ptr<TensorFence> fence;
            std::vector<Tensor> sources;
            bool sealed = false;
        };
        GpuBackend backend;
        TensorWorkQueue* queue = nullptr;
        internal::GpuBackendOps* ops = nullptr;
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
        }
    };
    TensorReadbackRing::TensorReadbackRing(GpuBackend backend, size_t slots, size_t bytes, TensorWorkQueue& queue, const Tensor* staging)
        : impl_(std::make_unique<Impl>()) {
        if (queue.backend() != backend)
            throw std::invalid_argument("Readback ring queue backend mismatch");
        if (!slots || !bytes)
            throw std::invalid_argument("Readback ring dimensions must be nonzero");
        auto& s = *impl_;
        s.backend = backend;
        s.queue = &queue;
        s.ops = &internal::backend_ops(backend);
        s.bytes = bytes;
        s.staging = staging;
        if (staging && staging->is_valid())
            require_readback_staging(*staging, backend, bytes);
        s.slots.resize(slots);
        for (auto& slot : s.slots) {
            slot.native = s.ops->create_readback_slot(bytes);
            if (!slot.native)
                throw std::runtime_error("Tensor backend does not support packed readback");
            slot.fence = std::make_unique<TensorFence>(backend);
        }
    }
    TensorReadbackRing::~TensorReadbackRing() = default;
    bool TensorReadbackRing::prefers_recycled_host_staging() const noexcept {
        return impl_->slots.front().native->prefers_recycled_host_staging();
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
        const internal::ExecContext execution{static_cast<cudaStream_t>(s.queue->native_handle())};
        if (slot.native->begin_append(retained, offset, bytes, destination, execution))
            return;
        auto input = internal::storage_ref(retained);
        input.byte_offset += offset;
        if (stage) {
            const bool borrowed = s.staging && s.staging->is_valid();
            if (borrowed) {
                require_readback_staging(*s.staging, s.backend, s.bytes);
                slot.sources.push_back(*s.staging);
            } else if (!s.scratch.is_valid()) {
                s.scratch = Tensor::empty({s.bytes}, Device::GPU, DataType::UInt8);
            }
            Tensor& scratch = borrowed ? slot.sources.back() : s.scratch;
            const auto output = internal::storage_ref(scratch);
            s.ops->copy_device_to_device({.src = input, .dst = output, .bytes = bytes, .context = execution});
            input = output;
        }
        slot.native->append(input, bytes, destination, execution);
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
        if (!slot.fence->ready())
            return false;
        return slot.native->ready();
    }
    void TensorReadbackRing::wait(size_t index) const {
        const auto& slot = impl_->slots.at(index);
        if (!slot.sealed)
            throw std::logic_error("Readback slot must be sealed before waiting");
        slot.fence->wait();
        slot.native->wait();
    }
    void TensorReadbackRing::release(size_t index) {
        auto& slot = impl_->slots.at(index);
        if ((slot.sealed && !slot.fence->ready()) ||
            (!slot.sealed && !slot.sources.empty() && !impl_->queue->ready()))
            throw std::logic_error("Cannot release a pending readback slot");
        slot.native->release();
        slot.sources.clear();
        slot.sealed = false;
    }
    std::span<std::byte> TensorReadbackRing::slot_bytes(size_t index) {
        auto& slot = impl_->slots.at(index);
        if (!slot.native->ready())
            throw std::logic_error("Packed readback bytes are not complete");
        return {static_cast<std::byte*>(slot.native->data()), impl_->bytes};
    }
} // namespace lfs::core
