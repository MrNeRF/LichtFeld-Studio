/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../../internal/expression_runtime.hpp"
#include "../backend_driver.hpp"
#include "../tensor_vulkan_interop.hpp"
#include "vk_memory.hpp"
#include "vk_pipelines.hpp"

#include "../tensor_completion.hpp"
#include "../upload_staging.hpp"
#include "vk_backend_ops.hpp"
#include "vk_context.hpp"
#include "vk_recorder.hpp"
#include <exception>
#include <utility>
#include <vector>

namespace lfs::core::internal {
    namespace {
        class VulkanUpload final : public BackendUploadBuffer {
            VulkanBackendOps& ops_;
            UploadStaging staging_;
            std::shared_ptr<VulkanContext> context_;
            StorageRef destination_{};
            TensorCompletion completion_;
            uint64_t queue_ = 0;

        public:
            explicit VulkanUpload(VulkanBackendOps& ops) : ops_(ops) {}
            BackendUploadSubmission enqueue(Tensor& destination, const Tensor& source, BackendQueueTarget target, bool) override {
                context_ = acquire_vulkan_context();
                struct Binding {
                    std::shared_ptr<VulkanContext> context;
                    bool bound = false;
                    ~Binding() {
                        if (bound)
                            context->recorders().unbind_queue();
                    }
                } binding{context_};
                if (!target.current && target.handle) {
                    const auto queue = reinterpret_cast<uint64_t>(target.handle);
                    if (!context_->recorders().owns_queue(queue))
                        throw std::invalid_argument("TensorUpload execution target is not a Vulkan queue");
                    context_->recorders().bind_queue(queue);
                    binding.bound = true;
                }
                destination.set_stream(nullptr);
                queue_ = context_->recorders().current_queue();
                destination_ = storage_ref(destination);
                ops_.copy_host_to_device({.src = raw_storage_ref(const_cast<void*>(source.data_ptr()), source.dtype()),
                                          .dst = destination_,
                                          .bytes = source.bytes(),
                                          .synchronous = false});
                completion_ = TensorCompletionAccess::vulkan(context_->recorders().pending_value(destination_));
                return {};
            }
            BackendUploadSubmission enqueue_bytes(Tensor& destination, std::span<const std::byte> source, BackendQueueTarget target, bool in_batch) override {
                const auto snapshot = staging_.snapshot(source, destination.dtype());
                return enqueue(destination, snapshot, target, in_batch);
            }
            void wait() override {
                context_->recorders().flush_storage(destination_);
                completion_.wait();
            }
            bool poll() override {
                context_->recorders().flush_storage(destination_);
                return completion_.ready();
            }
            bool abandon_after_failure() noexcept override { return false; }
            void settle_enqueue_failure() override {
                if (context_)
                    context_->recorders().queue_wait(queue_);
            }
        };
        class VulkanTimestamps final : public BackendTimestamps {
            struct Mark {
                uint64_t queue = 0, timeline = 0;
                bool written = false;
            };
            std::shared_ptr<VulkanContext> context_;
            VkQueryPool queries_ = VK_NULL_HANDLE;
            std::vector<Mark> marks_;
            float period_ = 0.f;
            bool read(size_t index, uint64_t& ticks, VkQueryResultFlags flags) const {
                return vkGetQueryPoolResults(context_->device(), queries_, static_cast<uint32_t>(index), 1,
                                             sizeof(ticks), &ticks, sizeof(ticks), flags) == VK_SUCCESS;
            }

        public:
            explicit VulkanTimestamps(size_t count) {
                auto context = acquire_vulkan_context();
                uint32_t families = 0;
                vkGetPhysicalDeviceQueueFamilyProperties(context->physical_device(), &families, nullptr);
                std::vector<VkQueueFamilyProperties> properties(families);
                if (families)
                    vkGetPhysicalDeviceQueueFamilyProperties(context->physical_device(), &families, properties.data());
                const auto family = context->queue_family();
                if (family >= families || !properties[family].timestampValidBits || context->caps().timestamp_period <= 0.f)
                    return;
                VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
                info.queryType = VK_QUERY_TYPE_TIMESTAMP;
                info.queryCount = static_cast<uint32_t>(count);
                marks_.resize(count);
                context_ = context;
                if (vkCreateQueryPool(context->device(), &info, nullptr, &queries_) != VK_SUCCESS)
                    return;
                period_ = context_->caps().timestamp_period;
            }
            ~VulkanTimestamps() override {
                if (queries_ != VK_NULL_HANDLE)
                    vkDestroyQueryPool(context_->device(), queries_, nullptr);
            }
            bool available() const { return queries_ != VK_NULL_HANDLE; }
            bool mark(size_t index, void* target) override {
                if (index >= marks_.size())
                    return false;
                const auto queue = target ? reinterpret_cast<uint64_t>(target) : context_->recorders().current_queue();
                const auto timeline = context_->recorders().write_timestamp(queue, queries_, static_cast<uint32_t>(index));
                marks_[index] = {queue, timeline, true};
                return true;
            }
            bool wait_event(size_t index) override {
                if (index >= marks_.size() || !marks_[index].written)
                    return false;
                const auto& mark = marks_[index];
                (void)context_->recorders().flush_queue(mark.queue);
                context_->wait(mark.timeline);
                uint64_t ticks = 0;
                return read(index, ticks, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            }
            bool wait_queue(void* target) override {
                const auto queue = target ? reinterpret_cast<uint64_t>(target) : context_->recorders().current_queue();
                context_->recorders().queue_wait(queue);
                // MoltenVK resolves query values after signaling the timeline.
                for (size_t index = 0; index < marks_.size(); ++index) {
                    if (!marks_[index].written || marks_[index].queue != queue)
                        continue;
                    uint64_t ticks = 0;
                    if (!read(index, ticks, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT))
                        return false;
                }
                return true;
            }
            std::optional<float> milliseconds(size_t begin, size_t end) const override {
                if (begin >= marks_.size() || end >= marks_.size() || !marks_[begin].written || !marks_[end].written)
                    return std::nullopt;
                uint64_t first = 0, last = 0;
                if (!read(begin, first, VK_QUERY_RESULT_64_BIT) || !read(end, last, VK_QUERY_RESULT_64_BIT) || last < first)
                    return std::nullopt;
                return static_cast<float>(static_cast<double>(last - first) * period_ / 1.0e6);
            }
        };
    } // namespace
    std::unique_ptr<BackendTimestamps> VulkanBackendOps::create_timestamps(size_t count) {
        auto result = std::make_unique<VulkanTimestamps>(count);
        return result->available() ? std::move(result) : nullptr;
    }
    std::unique_ptr<BackendUploadBuffer> VulkanBackendOps::create_upload_buffer() {
        return std::make_unique<VulkanUpload>(*this);
    }
} // namespace lfs::core::internal

namespace lfs::core::internal {
    namespace {
        class VulkanFence final : public BackendFence {
            std::shared_ptr<VulkanContext> context_ = acquire_vulkan_context();
            uint64_t value_ = 0;
            uint64_t queue(ExecContext target) const {
                const auto id = reinterpret_cast<uint64_t>(target.cuda_stream);
                if (id && !context_->recorders().owns_queue(id))
                    throw std::invalid_argument("TensorFence queue backend mismatch");
                return id;
            }

        public:
            explicit VulkanFence(std::optional<void*> imported) {
                if (imported)
                    value_ = reinterpret_cast<uint64_t>(*imported);
            }
            void record(ExecContext target) override { value_ = context_->recorders().flush_queue(queue(target)); }
            void wait_on(ExecContext target) const override { context_->recorders().queue_defer_wait(queue(target), value_); }
            void wait() const override {
                if (value_)
                    context_->wait(value_);
            }
            bool ready() const override { return !value_ || context_->completed_timeline() >= value_; }
        };
    } // namespace
    std::unique_ptr<BackendFence> VulkanBackendOps::create_fence(std::optional<void*> imported) {
        return std::make_unique<VulkanFence>(imported);
    }
} // namespace lfs::core::internal

namespace lfs::core::internal {
    void VulkanBackendOps::where_scalar(Tensor& output, const Tensor& mask, float value, const Tensor& source) {
        vulkan_where_into(output, mask, value, source);
    }
    ExpressionCacheStats VulkanBackendOps::expression_cache_stats() { return acquire_vulkan_context()->pipelines().expressions().stats(); }
} // namespace lfs::core::internal

namespace lfs::core::internal {
    namespace {
        class VulkanReadbackSlot final : public BackendReadbackSlot {
            std::shared_ptr<VulkanContext> context_ = acquire_vulkan_context();
            StorageRef host_;

        public:
            explicit VulkanReadbackSlot(size_t bytes) : host_(context_->memory().allocate_readback(bytes)) {}
            ~VulkanReadbackSlot() override { context_->memory().deallocate(host_); }
            void* data() const override { return context_->memory().mapped_pointer(host_); }
            bool begin_append(const Tensor&, size_t, size_t, size_t, ExecContext) override { return false; }
            void append(StorageRef source, size_t bytes, size_t destination, ExecContext) override {
                auto output = host_;
                output.byte_offset += destination;
                (void)context_->memory().copy_to_readback(source, output, bytes);
            }
            bool prefers_recycled_host_staging() const noexcept override { return false; }
            bool ready() override { return true; }
            void wait() override {}
            void release() override {}
        };
    } // namespace
    std::unique_ptr<BackendReadbackSlot> VulkanBackendOps::create_readback_slot(size_t bytes) {
        return std::make_unique<VulkanReadbackSlot>(bytes);
    }
} // namespace lfs::core::internal

namespace lfs::core::internal {
    namespace {
        class VulkanCompletion final : public BackendCompletion {
            std::shared_ptr<VulkanContext> context_;
            uint64_t value_ = 0;
            void* external_device_ = nullptr;
            VulkanTimelinePoint external_point_;

        public:
            explicit VulkanCompletion(BackendCompletionTarget target) : external_device_(target.external_device), external_point_(std::move(target.external_point)) {}
            void capture(std::span<const Tensor* const> tensors) override {
                StorageRef latest{};
                uint64_t pending = 0;
                for (const auto* tensor : tensors) {
                    const auto storage = storage_ref(*tensor);
                    if (!storage.meta)
                        continue;
                    const auto value = storage.meta->pending_value.load(std::memory_order_acquire);
                    if (value > pending) {
                        pending = value;
                        latest = storage;
                    }
                }
                if (pending) {
                    context_ = acquire_vulkan_context();
                    value_ = std::max(value_, pending);
                    context_->recorders().flush_storage(latest);
                }
            }
            void record(ExecContext, uint64_t value) override {
                context_ = acquire_vulkan_context();
                value_ = value;
            }
            VulkanTimelinePoint timeline() const override {
                return external_device_ ? external_point_ : context_ ? VulkanTimelinePoint{context_->timeline(), value_, context_}
                                                                     : VulkanTimelinePoint{};
            }
            bool ready() override {
                if (external_device_ && external_point_.value) {
                    uint64_t completed = 0;
                    const auto status = vkGetSemaphoreCounterValue(static_cast<VkDevice>(external_device_), static_cast<VkSemaphore>(external_point_.semaphore), &completed);
                    if (status != VK_SUCCESS)
                        throw TensorError("Cannot query tensor completion timeline");
                    if (completed < external_point_.value)
                        return false;
                }
                if (value_) {
                    if (context_->dead())
                        backend_driver(GpuBackend::Vulkan).throw_unavailable();
                    if (context_->completed_timeline() < value_)
                        return false;
                }
                return true;
            }
            void wait() override {
                std::exception_ptr failure;
                if (external_device_ && external_point_.value) {
                    try {
                        const auto semaphore = static_cast<VkSemaphore>(external_point_.semaphore);
                        VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
                        info.semaphoreCount = 1;
                        info.pSemaphores = &semaphore;
                        info.pValues = &external_point_.value;
                        VkResult status;
                        do {
                            status = vkWaitSemaphores(static_cast<VkDevice>(external_device_), &info, 1'000'000'000);
                        } while (status == VK_TIMEOUT);
                        if (status != VK_SUCCESS)
                            throw TensorError("Cannot wait for tensor completion timeline");
                        external_device_ = nullptr;
                    } catch (...) {
                        // LFS-CENSUS-OK(empty-catch): Settle the storage timeline too before rethrowing the external wait failure.
                        failure = std::current_exception();
                    }
                }
                if (const auto value = std::exchange(value_, 0)) {
                    try {
                        context_->wait(value);
                        context_->check_fault_buffer();
                    } catch (...) {
                        // LFS-CENSUS-OK(empty-catch): Release the captured context before rethrowing the first wait failure.
                        if (!failure)
                            failure = std::current_exception();
                    }
                }
                context_.reset();
                if (failure)
                    std::rethrow_exception(failure);
            }
        };
    } // namespace
    std::unique_ptr<BackendCompletion> VulkanBackendOps::create_completion(BackendCompletionTarget target) {
        return std::make_unique<VulkanCompletion>(std::move(target));
    }
} // namespace lfs::core::internal

namespace lfs::core::internal {
    StorageRef VulkanBackendOps::borrow_native_range([[maybe_unused]] void* pointer, [[maybe_unused]] size_t bytes) { return acquire_vulkan_context()->memory().borrow_address(pointer, bytes); }
} // namespace lfs::core::internal

namespace lfs::core::internal {
    void VulkanBackendOps::name_queue(ExecContext, const char*) {}
} // namespace lfs::core::internal
