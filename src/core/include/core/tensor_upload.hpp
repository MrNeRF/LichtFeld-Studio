/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/export.hpp"
#include "core/gpu_backend_fwd.hpp"
#include "core/tensor_completion.hpp"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
namespace lfs::core {
    class Tensor;
    // Reusable asynchronous H2D slot. Owns both tensors until the queued copy
    // completes. Destruction waits; poll never blocks. Does not alter copy_from.
    // Metal copies through unified memory during enqueue, after earlier GPU
    // use of the destination.
    class LFS_CORE_API TensorUpload {
    public:
        TensorUpload();
        ~TensorUpload();
        TensorUpload(TensorUpload&&) noexcept;
        TensorUpload& operator=(TensorUpload&&) noexcept;
        void enqueue(Tensor destination, const Tensor& source);
        // execution_target is a CUDA stream (nullptr selects the CUDA default
        // stream). Vulkan and Metal use their own queues and require nullptr.
        void enqueue(Tensor destination, const Tensor& source, void* execution_target);
        void enqueue(Tensor destination, std::span<const std::byte> source,
                     void* execution_target);
        [[nodiscard]] bool pending() const noexcept;
        [[nodiscard]] bool poll();
        void wait();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
    // Reusable ordering marker. Recording and GPU waits never block the host.
    // Does not retain tensor storage. The caller owns storage through completion.
    // Re-record only after every consumer has submitted its wait. Vulkan is
    // explicitly unsupported until reusable backend queue markers are available.
    class LFS_CORE_API TensorFence {
    public:
        explicit TensorFence(GpuBackend backend);
        ~TensorFence();
        TensorFence(TensorFence&&) noexcept;
        TensorFence& operator=(TensorFence&&) noexcept;
        TensorFence(const TensorFence&) = delete;
        TensorFence& operator=(const TensorFence&) = delete;
        void record(void* execution_target);
        void wait_on(void* execution_target) const;
        void wait() const;
        [[nodiscard]] bool ready() const;
        // Takes ownership of an event from an external backend producer.
        static TensorFence adopt(GpuBackend backend, void* event);

    private:
        TensorFence(GpuBackend backend, void* event);
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    // Device and consumer timeline outlive the queue.
    // CUDA timeline imports are private to tensorlib. execute batches every
    // command in the callback and returns one completion ticket.
    class LFS_CORE_API TensorWorkQueue {
    public:
        enum class Mode { Independent,
                          LegacyOrdered };
        explicit TensorWorkQueue(GpuBackend backend, Mode mode = Mode::Independent);
        // Non-owning adapter for a backend execution target, including default.
        TensorWorkQueue(GpuBackend backend, void* execution_target);
        TensorWorkQueue(GpuBackend backend, void* vulkan_device, void* consumer_timeline);
        class LFS_CORE_API Scope {
        public:
            explicit Scope(const TensorWorkQueue& queue);
            ~Scope();
            Scope(const Scope&) = delete;
            Scope& operator=(const Scope&) = delete;

        private:
            GpuBackendScope backend_scope_;
            void* previous_target_ = nullptr;
        };
        [[nodiscard]] void* native_handle() const;
        [[nodiscard]] GpuBackend backend() const;
        [[nodiscard]] bool ready() const;
        void wait() const;
        void record(TensorFence& fence) const;
        void wait_for(const TensorFence& fence) const;
        // Callback must not call GPU APIs or throw. Runs off the submitting
        // thread after prior queue work. CUDA only; Vulkan throws unsupported.
        void enqueue_host_callback(void (*callback)(void*), void* user);
        // Imports consumer semaphores without a host wait. Imports and keep_alive
        // tokens survive replacement until queue destruction.
        void set_consumer_timeline(void* vulkan_device, VulkanTimelinePoint point);
        void wait_timeline(uint64_t value);
        ~TensorWorkQueue();
        TensorWorkQueue(const TensorWorkQueue&) = delete;
        TensorWorkQueue& operator=(const TensorWorkQueue&) = delete;
        TensorCompletion execute(const std::function<void()>& commands, uint64_t consumer_value = 0);
        [[nodiscard]] void* timeline() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::core
