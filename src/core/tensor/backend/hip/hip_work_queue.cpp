/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../backend_work_queue.hpp"
#include "../gpu_backend_ops.hpp"
#include "../tensor_completion.hpp"
#include "hip_backend_ops.hpp"
namespace lfs::core::internal {
    namespace {
        class HipWorkQueue final : public BackendWorkQueue {
            HipBackendOps& ops_;
            std::shared_ptr<void> lease_;
            QueueConsumerTimeline consumer_;

        public:
            explicit HipWorkQueue(const BackendQueueCreateInfo& info)
                : ops_(static_cast<HipBackendOps&>(backend_ops(GpuBackend::HIP))), lease_(ops_.execution_lease()), consumer_(info.consumer_device, info.consumer) {
            }
            void* native_handle() const override { return nullptr; }
            BackendTargetBinding bind(bool) override {
                const auto previous = reinterpret_cast<uintptr_t>(getCurrentCUDAStream());
                setCurrentCUDAStream(nullptr);
                return {previous, 0};
            }
            void unbind(BackendTargetBinding token) noexcept override { setCurrentCUDAStream(reinterpret_cast<cudaStream_t>(token[0])); }
            bool ready() override { return ops_.queue_ready({}); }
            void wait() override { ops_.synchronize_stream({}); }
            void enqueue_host_callback(void (*callback)(void*), void* user) override { ops_.enqueue_host_callback(callback, user, {}); }
            void set_consumer_timeline(void* device, VulkanTimelinePoint point) override {
                if (!device)
                    throw std::invalid_argument("HIP consumer timeline requires a Vulkan device");
                validate_external_queue_device(device);
                consumer_.replace(device, std::move(point));
            }
            void wait_timeline(uint64_t value) override { consumer_.wait(value); }
            void* timeline() const override { return nullptr; }
            void begin(uint64_t value) { consumer_.wait(value); }
            TensorCompletion finish() { return TensorCompletionAccess::record(GpuBackend::HIP, {}); }
            void settle_failed() { ops_.synchronize_stream({}); }
            TensorCompletion execute(const std::function<void()>& commands, uint64_t value) override { return execute_commands(*this, GpuBackend::HIP, commands, value); }
            void validate_readback_ring() const override {}
            bool close() override {
                if (!ops_.drain_queue_for_destruction())
                    return false;
                try {
                    last_.wait();
                } catch (const std::exception& error) { LOG_ERROR("HIP tensor queue completion failed after synchronization: {}", error.what()); }
                return true;
            }
        };
    } // namespace
    std::unique_ptr<BackendWorkQueue> make_hip_work_queue(const BackendQueueCreateInfo& info) {
        if (info.kind == BackendQueueCreateInfo::Kind::Borrowed && info.target)
            throw std::invalid_argument("HIP borrowed queues require the native default target");
        if (info.kind == BackendQueueCreateInfo::Kind::Consumer) {
            if (info.consumer && !info.consumer_device)
                throw std::runtime_error("HIP consumer timeline requires a Vulkan device");
            validate_external_queue_device(info.consumer_device);
        }
        return std::make_unique<HipWorkQueue>(info);
    }
} // namespace lfs::core::internal
