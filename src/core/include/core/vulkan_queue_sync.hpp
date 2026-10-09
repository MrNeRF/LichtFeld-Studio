/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/export.hpp"
#include <vulkan/vulkan.h>

namespace lfs::rendering {
    // Every queue belongs to a device synchronization domain. Queue operations
    // share device access and serialize per queue; device idle takes exclusive access.
    LFS_CORE_API void register_vulkan_queue(VkDevice device, VkQueue queue);
    LFS_CORE_API void vk_get_device_queue_synced(VkDevice device, uint32_t family, uint32_t index, VkQueue* queue);
    LFS_CORE_API void vk_destroy_device_synced(VkDevice device, const VkAllocationCallbacks* allocator);
    LFS_CORE_API VkResult vk_queue_submit_synced(VkQueue queue, uint32_t count, const VkSubmitInfo* submits, VkFence fence);
    LFS_CORE_API VkResult vk_queue_submit2_synced(VkQueue queue, uint32_t count, const VkSubmitInfo2* submits, VkFence fence);
    LFS_CORE_API VkResult vk_queue_present_synced(VkQueue queue, const VkPresentInfoKHR* info);
    LFS_CORE_API VkResult vk_queue_wait_idle_synced(VkQueue queue);
    LFS_CORE_API VkResult vk_queue_bind_sparse_synced(VkQueue queue, uint32_t count, const VkBindSparseInfo* binds, VkFence fence);
    LFS_CORE_API VkResult vk_device_wait_idle_synced(VkDevice device);
} // namespace lfs::rendering
