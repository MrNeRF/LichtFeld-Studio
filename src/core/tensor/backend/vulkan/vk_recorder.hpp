/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/export.hpp"
#include "core/tensor/internal/private_access.hpp"

#include "../descriptors.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>

namespace lfs::core::internal {

    class VulkanContext;

    class VulkanRecorderRegistry final {
    public:
        explicit VulkanRecorderRegistry(VulkanContext& context);
        ~VulkanRecorderRegistry();

        VulkanRecorderRegistry(const VulkanRecorderRegistry&) = delete;
        VulkanRecorderRegistry& operator=(const VulkanRecorderRegistry&) = delete;

        LFS_CORE_API uint64_t record(std::span<const StorageRef> reads,
                                     std::span<const StorageRef> writes,
                                     const std::function<void(VkCommandBuffer)>& command,
                                     VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                     VkDeviceSize bytes = VK_WHOLE_SIZE,
                                     std::shared_ptr<void> lifetime = {});
        void flush_storage(StorageRef storage);
        uint64_t flush_storages(std::span<const StorageRef> storage);
        uint64_t wait_external(std::span<const StorageRef> storage, VkSemaphore semaphore,
                               uint64_t value, std::shared_ptr<void> keep_alive);
        uint64_t flush_current();
        [[nodiscard]] uint64_t current_queue();
        LFS_CORE_API uint64_t flush_all();
        LFS_CORE_API void wait_all();
        void release_thread(uint64_t recorder_id);
        void shutdown();

        // Dedicated recorder used as a tensor work queue. The caller binds it
        // onto the thread that records commands. Destruction waits for its
        // submitted timeline value. id 0 is the thread's implicit recorder.
        [[nodiscard]] uint64_t create_queue(bool legacy_ordered);
        void destroy_queue(uint64_t id);
        void bind_queue(uint64_t id);
        void unbind_queue();
        [[nodiscard]] bool owns_queue(uint64_t id) const;
        [[nodiscard]] uint64_t flush_queue(uint64_t id);
        [[nodiscard]] bool queue_ready(uint64_t id);
        void queue_wait(uint64_t id);
        // Submits work already recorded on the queue, then makes the next
        // submission wait for value. value 0 is already signaled.
        void queue_defer_wait(uint64_t id, uint64_t value);
        // Flush producer, then make consumer's next submit wait for that value.
        // id 0 is the calling thread's current recorder (bound queue, else implicit).
        void bridge_queues(uint64_t consumer, uint64_t producer);
        // Submit an empty wait for an external timeline semaphore on this device,
        // then order the queue's later submits behind it. Does not wait on the host.
        void queue_wait_external(uint64_t id, VkSemaphore semaphore, uint64_t value,
                                 std::shared_ptr<void> keep_alive);
        // Reset and write one timestamp into the queue's open command buffer.
        // The returned value is signalled when that buffer completes.
        [[nodiscard]] uint64_t write_timestamp(uint64_t id, VkQueryPool pool, uint32_t query);

        [[nodiscard]] uint64_t pending_value(StorageRef storage) const;
        [[nodiscard]] size_t dead_recorder_count() const;

    private:
        struct Recorder;

        Recorder& ensure_implicit_locked();
        Recorder& current_locked();
        Recorder* queue_locked(uint64_t id);
        void ensure_submitted_locked(StorageRef storage);
        uint64_t flush_through_locked(uint64_t value);
        void submit_locked(Recorder& recorder);
        void begin_locked(Recorder& recorder);
        void retire_completed_locked(Recorder& recorder, uint64_t completed);
        void collect_completed_locked(uint64_t completed);
        static void stamp(StorageRef storage, uint64_t recorder_id, uint64_t value);

        VulkanContext& context_;
        // Lock order: VulkanMemory::staging_mutex_, mutex_, then
        // VulkanMemory::allocations_mutex_. Allocator paths must release the
        // allocation lock before entering this registry.
        mutable std::mutex mutex_;
        std::unordered_map<uint64_t, std::unique_ptr<Recorder>> recorders_;
        uint64_t next_recorder_id_ = 1;
        bool shutting_down_ = false;
        uint64_t implicit_submitted_ = 0;
        uint64_t legacy_submitted_ = 0;
        std::vector<std::pair<uint64_t, std::shared_ptr<void>>> external_owners_;
    };

} // namespace lfs::core::internal
