/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/vulkan_queue_sync.hpp"
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

namespace lfs::rendering {
    namespace {
        struct DeviceAccess {
            std::shared_mutex mutex;
        };
        struct QueueAccess {
            VkDevice device;
            std::shared_ptr<DeviceAccess> owner;
            std::mutex mutex;
            QueueAccess(VkDevice d, std::shared_ptr<DeviceAccess> state) : device(d), owner(std::move(state)) {}
        };
        struct Registry {
            std::mutex mutex;
            std::unordered_map<VkDevice, std::shared_ptr<DeviceAccess>> devices;
            std::unordered_map<VkQueue, std::shared_ptr<QueueAccess>> queues;
        };
        Registry& registry() {
            // Backend owners can destroy devices from static destructors. Keep the
            // registry available through process teardown; device destruction still
            // retires each device and its queues.
            static auto* state = new Registry;
            return *state;
        }

        std::shared_ptr<DeviceAccess> device_access(VkDevice device) {
            auto& state_registry = registry();
            std::lock_guard lock(state_registry.mutex);
            auto& state = state_registry.devices[device];
            if (!state)
                state = std::make_shared<DeviceAccess>();
            return state;
        }
        template <class Operation>
        VkResult queue_access(VkQueue queue, Operation&& operation) {
            std::shared_ptr<QueueAccess> state;
            {
                auto& state_registry = registry();
                std::lock_guard lock(state_registry.mutex);
                const auto found = state_registry.queues.find(queue);
                if (found == state_registry.queues.end())
                    return VK_ERROR_INITIALIZATION_FAILED;
                state = found->second;
            }
            const std::shared_lock device_lock(state->owner->mutex);
            const std::lock_guard queue_lock(state->mutex);
            return operation();
        }
    } // namespace
    void register_vulkan_queue(VkDevice device, VkQueue queue) {
        if (!device || !queue)
            return;
        auto owner = device_access(device);
        auto& state_registry = registry();
        std::lock_guard lock(state_registry.mutex);
        if (!state_registry.queues.contains(queue))
            state_registry.queues.emplace(queue, std::make_shared<QueueAccess>(device, std::move(owner)));
    }
    void vk_get_device_queue_synced(VkDevice device, uint32_t family, uint32_t index, VkQueue* queue) {
        vkGetDeviceQueue(device, family, index, queue);
        register_vulkan_queue(device, *queue);
    }
    void vk_destroy_device_synced(VkDevice device, const VkAllocationCallbacks* allocator) {
        if (!device)
            return;
        const auto state = device_access(device);
        const std::unique_lock access(state->mutex);
        vkDestroyDevice(device, allocator);
        auto& state_registry = registry();
        std::lock_guard lock(state_registry.mutex);
        std::erase_if(state_registry.queues, [device](const auto& entry) { return entry.second->device == device; });
        state_registry.devices.erase(device);
    }
    VkResult vk_queue_submit_synced(VkQueue queue, uint32_t count, const VkSubmitInfo* submits, VkFence fence) {
        return queue_access(queue, [&] { return vkQueueSubmit(queue, count, submits, fence); });
    }
    VkResult vk_queue_submit2_synced(VkQueue queue, uint32_t count, const VkSubmitInfo2* submits, VkFence fence) {
        return queue_access(queue, [&] { return vkQueueSubmit2(queue, count, submits, fence); });
    }
    VkResult vk_queue_present_synced(VkQueue queue, const VkPresentInfoKHR* info) {
        return queue_access(queue, [&] { return vkQueuePresentKHR(queue, info); });
    }
    VkResult vk_queue_wait_idle_synced(VkQueue queue) {
        return queue_access(queue, [&] { return vkQueueWaitIdle(queue); });
    }
    VkResult vk_queue_bind_sparse_synced(VkQueue queue, uint32_t count, const VkBindSparseInfo* binds, VkFence fence) {
        return queue_access(queue, [&] { return vkQueueBindSparse(queue, count, binds, fence); });
    }
    VkResult vk_device_wait_idle_synced(VkDevice device) {
        const auto state = device_access(device);
        const std::unique_lock lock(state->mutex);
        return vkDeviceWaitIdle(device);
    }
} // namespace lfs::rendering
