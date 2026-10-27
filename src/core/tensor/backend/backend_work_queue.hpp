/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/logger.hpp"
#include "core/tensor_completion.hpp"
#include "core/tensor_cuda_interop.hpp"
#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <vector>
namespace lfs::core::internal {
    using BackendTargetBinding = std::array<uintptr_t, 2>;
    struct BackendQueueCreateInfo {
        enum class Kind { Owned,
                          Borrowed,
                          Consumer };
        Kind kind = Kind::Owned;
        bool legacy_ordered = false;
        void* target = nullptr;
        void* consumer_device = nullptr;
        void* consumer = nullptr;
        std::optional<bool> external_interop;
    };
    // CMake selects the native timeline implementation, including explicit
    // unavailable behavior in builds without Vulkan. Tokens stay owned here.
    void validate_external_queue_device(void* device);
    void wait_external_queue_timeline(void* device, void* semaphore, uint64_t value);
    class QueueConsumerTimeline {
        void* device_ = nullptr;
        VulkanTimelinePoint current_;
        std::vector<VulkanTimelinePoint> retired_;

    public:
        QueueConsumerTimeline() = default;
        QueueConsumerTimeline(void* device, void* semaphore) : device_(device), current_{semaphore, 0, {}} {}
        bool matches(void* device, const VulkanTimelinePoint& point) const { return device_ == device && current_.semaphore == point.semaphore; }
        void reserve_replacement() {
            if (current_.semaphore || current_.keep_alive)
                retired_.reserve(retired_.size() + 1);
        }
        void replace(void* device, VulkanTimelinePoint point) {
            if (matches(device, point))
                return;
            if (current_.semaphore || current_.keep_alive)
                retired_.push_back(std::move(current_));
            device_ = device;
            current_ = std::move(point);
        }
        void* device() const { return device_; }
        const VulkanTimelinePoint& point() const { return current_; }
        bool retained() const { return current_.semaphore || !retired_.empty(); }
        void wait(uint64_t value) const {
            if (value && current_.semaphore)
                wait_external_queue_timeline(device_, current_.semaphore, value);
        }
    };
    class BackendWorkQueue {
    protected:
        TensorCompletion last_;
        template <class Queue>
        TensorCompletion execute_commands(Queue& queue, GpuBackend backend, const std::function<void()>& commands, uint64_t value) {
            const GpuBackendScope scope(backend);
            struct Binding {
                Queue& queue;
                BackendTargetBinding token;
                ~Binding() { queue.unbind(token); }
            } binding{queue, queue.bind(true)};
            queue.begin(value);
            TensorCompletion result;
            try {
                commands();
                result = queue.finish();
            } catch (...) {
                queue.settle_failed();
                throw;
            }
            last_ = result;
            return result;
        }

    public:
        virtual ~BackendWorkQueue() = default;
        virtual void* native_handle() const = 0;
        // No allocation: scope state is returned inline and restored by value.
        virtual BackendTargetBinding bind(bool execution) = 0;
        virtual void unbind(BackendTargetBinding) noexcept = 0;
        virtual bool ready() = 0;
        virtual void wait() = 0;
        virtual void enqueue_host_callback(void (*)(void*), void*) = 0;
        virtual void set_consumer_timeline(void*, VulkanTimelinePoint) = 0;
        virtual void wait_timeline(uint64_t) = 0;
        virtual void* timeline() const = 0;
        virtual TensorCompletion execute(const std::function<void()>&, uint64_t) = 0;
        // False means ownership must remain alive after a real drain failure.
        virtual bool close() = 0;
        virtual void validate_readback_ring() const = 0;
    };
    std::unique_ptr<BackendWorkQueue> make_cuda_work_queue(const BackendQueueCreateInfo&);
    std::unique_ptr<BackendWorkQueue> make_vulkan_work_queue(const BackendQueueCreateInfo&);
    std::unique_ptr<BackendWorkQueue> make_metal_backend_work_queue(const BackendQueueCreateInfo&);
    std::unique_ptr<BackendWorkQueue> make_hip_work_queue(const BackendQueueCreateInfo&);
} // namespace lfs::core::internal
