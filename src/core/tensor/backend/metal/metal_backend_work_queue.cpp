/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../backend_work_queue.hpp"
#include "../queue_callback_worker.hpp"
#include "../tensor_completion.hpp"
#include "core/tensor_backend.hpp"
#include "metal_queue.hpp"
#include "metal_queue_consumer.hpp"
namespace lfs::core::internal {
    namespace {
        class MetalBackendWorkQueue final : public BackendWorkQueue {
            QueueConsumerTimeline consumer_;
            std::unique_ptr<MetalVulkanQueue> bridge_;
            uint64_t queue_ = 0;
            bool owns_ = false;
            QueueCallbackWorker callbacks_{[](uint64_t value) { metal_queue::wait_completed(value); }};

        public:
            explicit MetalBackendWorkQueue(const BackendQueueCreateInfo& info) : consumer_(info.consumer_device, info.consumer) {
                if (info.kind == BackendQueueCreateInfo::Kind::Consumer)
                    bridge_ = make_metal_queue_consumer(info.consumer_device, info.consumer);
                else if (info.kind == BackendQueueCreateInfo::Kind::Owned) {
                    queue_ = metal_queue::create();
                    owns_ = true;
                } else {
                    queue_ = reinterpret_cast<uint64_t>(info.target);
                    if (!gpu_backend_available(GpuBackend::Metal) || !metal_queue::valid(queue_))
                        throw std::runtime_error("Borrowed Metal tensor queues must be live Metal queues");
                }
            }
            void* native_handle() const override { return reinterpret_cast<void*>(queue_); }
            BackendTargetBinding bind(bool execution) override {
                const auto previous = reinterpret_cast<uintptr_t>(getCurrentCUDAStream());
                if (execution)
                    setCurrentCUDAStream(nullptr);
                return {previous, 0};
            }
            void unbind(BackendTargetBinding token) noexcept override { setCurrentCUDAStream(reinterpret_cast<cudaStream_t>(token[0])); }
            bool ready() override { return callbacks_.ready() && metal_queue::ready(metal_queue::submit()); }
            void wait() override {
                metal_queue::wait(metal_queue::submit());
                callbacks_.wait();
            }
            void enqueue_host_callback(void (*callback)(void*), void* user) override { callbacks_.enqueue(metal_queue::submit(), callback, user); }
            void set_consumer_timeline(void* device, VulkanTimelinePoint point) override {
                if (bridge_)
                    throw std::runtime_error("Consumer timeline replacement is unsupported on Vulkan");
                consumer_.replace(device, std::move(point));
            }
            void wait_timeline(uint64_t value) override {
                if (bridge_)
                    throw std::runtime_error("Consumer timeline waits are unsupported on Vulkan");
                if (value && consumer_.point().semaphore)
                    wait_metal_queue_consumer(consumer_.device(), consumer_.point().semaphore, value);
            }
            void* timeline() const override { return bridge_ ? bridge_->timeline() : nullptr; }
            void begin(uint64_t value) {
                if (!value || !consumer_.point().semaphore)
                    return;
                if (bridge_)
                    bridge_->wait(value);
                else
                    wait_metal_queue_consumer(consumer_.device(), consumer_.point().semaphore, value);
            }
            TensorCompletion finish() { return bridge_ ? bridge_->signal() : TensorCompletionAccess::metal(metal_queue::submit()); }
            void settle_failed() {
                if (bridge_)
                    bridge_->signal().wait();
                else
                    metal_queue::wait_completed(metal_queue::submit());
            }
            TensorCompletion execute(const std::function<void()>& commands, uint64_t value) override { return execute_commands(*this, GpuBackend::Metal, commands, value); }
            void validate_readback_ring() const override {}
            bool close() override {
                if (consumer_.retained() || !callbacks_.ready())
                    wait();
                if (owns_) {
                    metal_queue::destroy(queue_);
                    owns_ = false;
                }
                last_.wait();
                return true;
            }
        };
    } // namespace
    std::unique_ptr<BackendWorkQueue> make_metal_backend_work_queue(const BackendQueueCreateInfo& info) { return std::make_unique<MetalBackendWorkQueue>(info); }
} // namespace lfs::core::internal
