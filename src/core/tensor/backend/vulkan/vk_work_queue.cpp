/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../backend_work_queue.hpp"
#include "../queue_callback_worker.hpp"
#include "../tensor_completion.hpp"
#include "vk_context.hpp"
#include "vk_recorder.hpp"
#include <algorithm>
namespace lfs::core::internal {
    namespace {
        class VulkanWorkQueue final : public BackendWorkQueue {
            std::shared_ptr<VulkanContext> context_ = acquire_vulkan_context();
            QueueConsumerTimeline consumer_;
            uint64_t recorder_ = 0, consumer_completion_ = 0;
            bool explicit_queue_, owns_ = false, host_consumer_ = false;
            QueueCallbackWorker callbacks_{[this](uint64_t value) { context_->wait(value); }};

        public:
            explicit VulkanWorkQueue(const BackendQueueCreateInfo& info)
                : consumer_(info.consumer_device, info.consumer), explicit_queue_(info.kind != BackendQueueCreateInfo::Kind::Consumer) {
                if (info.kind == BackendQueueCreateInfo::Kind::Owned) {
                    recorder_ = context_->recorders().create_queue(info.legacy_ordered);
                    owns_ = true;
                } else if (info.kind == BackendQueueCreateInfo::Kind::Borrowed) {
                    recorder_ = reinterpret_cast<uint64_t>(info.target);
                    if (recorder_ && !context_->recorders().owns_queue(recorder_))
                        throw std::runtime_error("Borrowed tensor queues are unsupported on Vulkan");
                }
                host_consumer_ = info.consumer_device && info.consumer_device != context_->device();
            }
            void* native_handle() const override { return reinterpret_cast<void*>(recorder_); }
            BackendTargetBinding bind(bool execution) override {
                const auto previous = reinterpret_cast<uintptr_t>(getCurrentCUDAStream());
                if (explicit_queue_)
                    context_->recorders().bind_queue(recorder_);
                if (execution)
                    setCurrentCUDAStream(nullptr);
                return {previous, (execution ? 1u : 0u) | (explicit_queue_ ? 2u : 0u)};
            }
            void unbind(BackendTargetBinding token) noexcept override {
                if ((token[1] & 2u) && try_live_vulkan_context())
                    context_->recorders().unbind_queue();
                if (token[1] & 1u)
                    setCurrentCUDAStream(reinterpret_cast<cudaStream_t>(token[0]));
            }
            bool ready() override {
                if (!explicit_queue_)
                    throw std::runtime_error("Queue polling is unsupported on Vulkan");
                return callbacks_.ready() && context_->recorders().queue_ready(recorder_);
            }
            void wait() override {
                if (!explicit_queue_)
                    throw std::runtime_error("Queue waits are unsupported on Vulkan");
                context_->recorders().queue_wait(recorder_);
                callbacks_.wait();
            }
            void enqueue_host_callback(void (*callback)(void*), void* user) override {
                if (!explicit_queue_)
                    throw std::runtime_error("Queue host callbacks are unsupported on Vulkan");
                callbacks_.enqueue(context_->recorders().flush_queue(recorder_), callback, user);
            }
            void set_consumer_timeline(void* device, VulkanTimelinePoint point) override {
                if (!explicit_queue_ && !device)
                    throw std::runtime_error("A backend-owned Vulkan queue requires an explicit consumer device");
                host_consumer_ = device && device != context_->device();
                consumer_.replace(device, std::move(point));
            }
            void wait_timeline(uint64_t value) override {
                if (host_consumer_) {
                    consumer_.wait(value);
                    return;
                }
                if (!explicit_queue_)
                    throw std::runtime_error("Consumer timeline waits are unsupported on Vulkan");
                if (value && consumer_.point().semaphore)
                    context_->recorders().queue_wait_external(recorder_, static_cast<VkSemaphore>(consumer_.point().semaphore), value, consumer_.point().keep_alive);
            }
            void* timeline() const override { return host_consumer_ || explicit_queue_ ? nullptr : context_->timeline(); }
            void begin(uint64_t value) {
                consumer_completion_ = 0;
                if (!value || !consumer_.point().semaphore)
                    return;
                if (host_consumer_)
                    consumer_.wait(value);
                else
                    consumer_completion_ = context_->recorders().wait_external({}, static_cast<VkSemaphore>(consumer_.point().semaphore), value, {});
            }
            TensorCompletion finish() { return TensorCompletionAccess::record(GpuBackend::Vulkan, {.value = std::max(consumer_completion_, context_->recorders().flush_current()), .settle_on_release = false, .use_recorded_timeline = true}); }
            void settle_failed() { context_->recorders().wait_all(); }
            TensorCompletion execute(const std::function<void()>& commands, uint64_t value) override { return execute_commands(*this, GpuBackend::Vulkan, commands, value); }
            void validate_readback_ring() const override {
                if (!explicit_queue_)
                    throw std::runtime_error("Packed tensor readback rings need an owned Vulkan work queue");
            }
            bool close() override {
                if (explicit_queue_ && (consumer_.retained() || !callbacks_.ready()))
                    wait();
                if (owns_ && recorder_) {
                    context_->recorders().destroy_queue(recorder_);
                    owns_ = false;
                }
                last_.wait();
                return true;
            }
        };
    } // namespace
    std::unique_ptr<BackendWorkQueue> make_vulkan_work_queue(const BackendQueueCreateInfo& info) { return std::make_unique<VulkanWorkQueue>(info); }
} // namespace lfs::core::internal
