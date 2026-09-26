/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vk_recorder.hpp"

#include "../../internal/tensor_impl.hpp"
#include "core/assert.hpp"
#include "vk_context.hpp"
#include "vk_memory.hpp"

#include <algorithm>
#include <array>
#include <deque>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

namespace lfs::core::internal {
    namespace {
        constexpr uint32_t kCommandLimit = 64;

        struct ThreadRecorderToken {
            uint64_t context_id = 0;
            uint64_t recorder_id = 0;

            ~ThreadRecorderToken() {
                if (recorder_id == 0) {
                    return;
                }
                if (const auto context = try_live_vulkan_context();
                    context && context->context_id() == context_id && context->accepting_work()) {
                    context->recorders().release_thread(recorder_id);
                }
            }
        };

        thread_local ThreadRecorderToken tls_recorder;

    } // namespace

    struct VulkanRecorderRegistry::Recorder {
        struct Submitted {
            VkCommandBuffer command = VK_NULL_HANDLE;
            uint64_t value = 0;
            std::vector<std::shared_ptr<void>> lifetimes;
        };

        uint64_t id = 0;
        VkCommandPool pool = VK_NULL_HANDLE;
        VkCommandBuffer command = VK_NULL_HANDLE;
        // Submitted batches whose completion the host has not observed yet,
        // oldest first. Recording continues while they run; the host waits
        // only when kInFlightLimit batches are outstanding.
        std::deque<Submitted> in_flight;
        std::vector<VkCommandBuffer> available_commands;
        uint64_t reserved_value = 0;
        uint64_t submitted_value = 0;
        uint32_t command_count = 0;
        bool owner_alive = true;
        std::vector<std::shared_ptr<void>> lifetimes;
    };

    namespace {
        constexpr size_t kInFlightLimit = 4;
    } // namespace

    void VulkanRecorderRegistry::retire_completed_locked(Recorder& recorder,
                                                         const uint64_t completed) {
        while (!recorder.in_flight.empty() && recorder.in_flight.front().value <= completed) {
            recorder.available_commands.push_back(recorder.in_flight.front().command);
            recorder.in_flight.pop_front();
        }
    }

    VulkanRecorderRegistry::VulkanRecorderRegistry(VulkanContext& context)
        : context_(context) {}

    VulkanRecorderRegistry::~VulkanRecorderRegistry() {
        shutdown();
    }

    VulkanRecorderRegistry::Recorder& VulkanRecorderRegistry::current_locked() {
        LFS_ASSERT_MSG(!shutting_down_ && context_.accepting_work(),
                       "Vulkan backend is shutting down; new recording is rejected");
        if (tls_recorder.context_id == context_.context_id() &&
            tls_recorder.recorder_id != 0) {
            return *recorders_.at(tls_recorder.recorder_id);
        }

        auto recorder = std::make_unique<Recorder>();
        recorder->id = next_recorder_id_++;
        VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
                          VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        pool_info.queueFamilyIndex = context_.queue_family();
        vk_check(&context_, vkCreateCommandPool(context_.device(), &pool_info, nullptr, &recorder->pool),
                 "vkCreateCommandPool");
        const uint64_t id = recorder->id;
        recorders_.emplace(id, std::move(recorder));
        tls_recorder.context_id = context_.context_id();
        tls_recorder.recorder_id = id;
        return *recorders_.at(id);
    }

    void VulkanRecorderRegistry::begin_locked(Recorder& recorder) {
        if (recorder.command != VK_NULL_HANDLE) {
            return;
        }
        retire_completed_locked(recorder, context_.completed_timeline());
        if (recorder.in_flight.size() >= kInFlightLimit) {
            context_.wait(recorder.in_flight.front().value);
            retire_completed_locked(recorder, context_.completed_timeline());
        }
        if (recorder.available_commands.empty()) {
            VkCommandBufferAllocateInfo allocate_info{
                VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            allocate_info.commandPool = recorder.pool;
            allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocate_info.commandBufferCount = 1;
            vk_check(&context_, vkAllocateCommandBuffers(context_.device(), &allocate_info, &recorder.command),
                     "vkAllocateCommandBuffers");
        } else {
            recorder.command = recorder.available_commands.back();
            recorder.available_commands.pop_back();
        }
        VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vk_check(&context_, vkBeginCommandBuffer(recorder.command, &begin_info),
                 "vkBeginCommandBuffer");
        recorder.reserved_value = context_.reserve_timeline_value();
        recorder.command_count = 0;
    }

    void VulkanRecorderRegistry::stamp(const StorageRef storage,
                                       const uint64_t recorder_id,
                                       const uint64_t value) {
        if (storage.meta == nullptr || storage.backend != GpuBackend::Vulkan) {
            return;
        }
        auto* const meta = const_cast<StorageMeta*>(storage.meta);
        meta->pending_recorder.store(recorder_id, std::memory_order_release);
        meta->pending_value.store(value, std::memory_order_release);
    }

    void VulkanRecorderRegistry::ensure_submitted_locked(const StorageRef storage) {
        if (storage.meta == nullptr || storage.backend != GpuBackend::Vulkan) {
            return;
        }
        const uint64_t value = storage.meta->pending_value.load(std::memory_order_acquire);
        const uint64_t recorder_id =
            storage.meta->pending_recorder.load(std::memory_order_acquire);
        if (value == 0 || recorder_id == 0) {
            return;
        }
        const auto iterator = recorders_.find(recorder_id);
        if (iterator != recorders_.end() && iterator->second->reserved_value == value &&
            iterator->second->command != VK_NULL_HANDLE) {
            flush_through_locked(value);
        }
    }

    uint64_t VulkanRecorderRegistry::record(
        const std::span<const StorageRef> reads,
        const std::span<const StorageRef> writes,
        const std::function<void(VkCommandBuffer)>& command,
        const VkPipelineStageFlags2 stage, const VkDeviceSize bytes,
        std::shared_ptr<void> lifetime) {
        std::lock_guard lock(mutex_);
        collect_completed_locked(context_.completed_timeline());
        Recorder& recorder = current_locked();
        const uint64_t foreign = context_.memory().foreign_use(reads, writes, recorder.reserved_value);
        if (foreign != 0) {
            flush_through_locked(foreign);
        }
        begin_locked(recorder);
        context_.memory().prepare_accesses(recorder.command, reads, writes,
                                           recorder.reserved_value, stage, bytes);
        command(recorder.command);
        if (lifetime)
            recorder.lifetimes.push_back(std::move(lifetime));
        ++recorder.command_count;
        for (const StorageRef storage : writes) {
            stamp(storage, recorder.id, recorder.reserved_value);
        }
        const uint64_t value = recorder.reserved_value;
        if (recorder.command_count >= kCommandLimit) {
            flush_through_locked(value);
        }
        return value;
    }

    void VulkanRecorderRegistry::submit_locked(Recorder& recorder) {
        if (recorder.command == VK_NULL_HANDLE) {
            return;
        }
        const VkCommandBuffer command = std::exchange(recorder.command, VK_NULL_HANDLE);
        const uint64_t value = std::exchange(recorder.reserved_value, uint64_t{0});
        recorder.command_count = 0;
        vk_check(&context_, vkEndCommandBuffer(command), "vkEndCommandBuffer");
        context_.submit(command, value);
        recorder.submitted_value = value;
        recorder.in_flight.push_back({command, value, std::move(recorder.lifetimes)});
    }

    uint64_t VulkanRecorderRegistry::flush_through_locked(const uint64_t value) {
        std::vector<Recorder*> pending;
        for (auto& [id, recorder] : recorders_) {
            (void)id;
            if (recorder->command != VK_NULL_HANDLE && recorder->reserved_value <= value) {
                pending.push_back(recorder.get());
            }
        }
        std::ranges::sort(pending, {}, &Recorder::reserved_value);
        uint64_t submitted = 0;
        for (Recorder* const recorder : pending) {
            submitted = recorder->reserved_value;
            submit_locked(*recorder);
        }
        return submitted;
    }

    void VulkanRecorderRegistry::collect_completed_locked(const uint64_t completed) {
        std::erase_if(recorders_, [&](const auto& entry) {
            Recorder& recorder = *entry.second;
            if (recorder.owner_alive || recorder.command != VK_NULL_HANDLE ||
                recorder.submitted_value > completed) {
                return false;
            }
            retire_completed_locked(recorder, completed);
            vkDestroyCommandPool(context_.device(), recorder.pool, nullptr);
            return true;
        });
    }

    uint64_t VulkanRecorderRegistry::flush_storages(std::span<const StorageRef> storage) {
        std::lock_guard lock(mutex_);
        uint64_t pending = 0;
        for (const auto& tensor : storage)
            pending = std::max(pending, pending_value(tensor));
        if (pending)
            flush_through_locked(pending);
        return pending;
    }

    uint64_t VulkanRecorderRegistry::wait_external(std::span<const StorageRef> storage,
                                                   VkSemaphore semaphore, uint64_t value, std::shared_ptr<void> keep_alive) {
        std::lock_guard lock(mutex_);
        flush_through_locked(std::numeric_limits<uint64_t>::max());
        const uint64_t signal = context_.reserve_timeline_value();
        context_.submit_external_wait(semaphore, value, signal);
        for (const auto& tensor : storage)
            stamp(tensor, 0, signal);
        const auto completed = context_.completed_timeline();
        std::erase_if(external_owners_, [completed](const auto& owner) { return owner.first <= completed; });
        external_owners_.emplace_back(signal, std::move(keep_alive));
        return signal;
    }

    void VulkanRecorderRegistry::flush_storage(const StorageRef storage) {
        std::lock_guard lock(mutex_);
        ensure_submitted_locked(storage);
    }

    uint64_t VulkanRecorderRegistry::flush_current() {
        std::lock_guard lock(mutex_);
        if (tls_recorder.context_id != context_.context_id() ||
            tls_recorder.recorder_id == 0) {
            return 0;
        }
        Recorder& recorder = *recorders_.at(tls_recorder.recorder_id);
        return recorder.command == VK_NULL_HANDLE
                   ? recorder.submitted_value
                   : flush_through_locked(recorder.reserved_value);
    }

    uint64_t VulkanRecorderRegistry::flush_all() {
        std::lock_guard lock(mutex_);
        uint64_t submitted =
            flush_through_locked(std::numeric_limits<uint64_t>::max());
        for (const auto& [id, recorder] : recorders_) {
            (void)id;
            submitted = std::max(submitted, recorder->submitted_value);
        }
        if (!external_owners_.empty())
            submitted = std::max(submitted, external_owners_.back().first);
        collect_completed_locked(context_.completed_timeline());
        return submitted;
    }

    void VulkanRecorderRegistry::wait_all() {
        const uint64_t value = flush_all();
        if (value != 0) {
            context_.wait(value);
        }
        std::lock_guard lock(mutex_);
        collect_completed_locked(context_.completed_timeline());
    }

    void VulkanRecorderRegistry::release_thread(const uint64_t recorder_id) {
        std::lock_guard lock(mutex_);
        const auto iterator = recorders_.find(recorder_id);
        if (iterator == recorders_.end()) {
            return;
        }
        Recorder& recorder = *iterator->second;
        if (recorder.command != VK_NULL_HANDLE) {
            if (context_.dead()) {
                // Nothing reaches a lost device; the buffer goes with the pool at
                // shutdown. Submitting here would throw out of a thread-exit
                // destructor.
                recorder.command = VK_NULL_HANDLE;
                recorder.reserved_value = 0;
                recorder.command_count = 0;
            } else {
                flush_through_locked(recorder.reserved_value);
            }
        }
        recorder.owner_alive = false;
    }

    uint64_t VulkanRecorderRegistry::pending_value(const StorageRef storage) const {
        return storage.meta != nullptr
                   ? storage.meta->pending_value.load(std::memory_order_acquire)
                   : 0;
    }

    size_t VulkanRecorderRegistry::dead_recorder_count() const {
        std::lock_guard lock(mutex_);
        return std::ranges::count_if(recorders_, [](const auto& entry) {
            return !entry.second->owner_alive;
        });
    }

    void VulkanRecorderRegistry::shutdown() {
        std::lock_guard lock(mutex_);
        if (shutting_down_) {
            return;
        }
        shutting_down_ = true;
        if (!context_.dead()) {
            try {
                const uint64_t submitted =
                    flush_through_locked(std::numeric_limits<uint64_t>::max());
                if (submitted != 0) {
                    context_.wait(submitted);
                }
                for (auto& [id, recorder] : recorders_) {
                    (void)id;
                    if (recorder->submitted_value != 0) {
                        context_.wait(recorder->submitted_value);
                    }
                }
            } catch (...) {
                // LFS-CENSUS-OK(empty-catch): mark_device_lost_once reports the loss.
                context_.mark_device_lost_once();
            }
        }
        if (context_.dead()) {
            static_cast<void>(vkDeviceWaitIdle(context_.device()));
        }
        for (auto& [id, recorder] : recorders_) {
            (void)id;
            if (recorder->pool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(context_.device(), recorder->pool, nullptr);
            }
        }
        if (!external_owners_.empty() && !context_.dead())
            context_.wait(external_owners_.back().first);
        external_owners_.clear();
        recorders_.clear();
        if (tls_recorder.context_id == context_.context_id()) {
            tls_recorder = {};
        }
    }

} // namespace lfs::core::internal
