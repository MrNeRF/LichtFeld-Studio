/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "../descriptors.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vulkan/vulkan.h>

namespace lfs::core::internal {

    class VulkanContext;

    class VulkanRecorderRegistry final {
    public:
        explicit VulkanRecorderRegistry(VulkanContext& context);
        ~VulkanRecorderRegistry();

        VulkanRecorderRegistry(const VulkanRecorderRegistry&) = delete;
        VulkanRecorderRegistry& operator=(const VulkanRecorderRegistry&) = delete;

        uint64_t record(std::span<const StorageRef> reads,
                        std::span<const StorageRef> writes,
                        const std::function<void(VkCommandBuffer)>& command);
        void flush_storage(StorageRef storage);
        uint64_t flush_current();
        uint64_t flush_all();
        void wait_all();
        void release_thread(uint64_t recorder_id);
        void shutdown();

        [[nodiscard]] uint64_t pending_value(StorageRef storage) const;
        [[nodiscard]] size_t dead_recorder_count() const;

    private:
        struct Recorder;

        Recorder& current_locked();
        void ensure_submitted_locked(StorageRef storage);
        uint64_t flush_through_locked(uint64_t value);
        void submit_locked(Recorder& recorder);
        void begin_locked(Recorder& recorder);
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
    };

} // namespace lfs::core::internal
