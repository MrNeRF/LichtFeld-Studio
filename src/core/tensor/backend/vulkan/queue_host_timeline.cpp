/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../backend_work_queue.hpp"
#include "vk_context.hpp"
#include <vulkan/vulkan.h>
namespace lfs::core::internal {
    void validate_external_queue_device(void*) {}
    void wait_external_queue_timeline(void* device, void* semaphore, uint64_t value) {
        const auto native = static_cast<VkSemaphore>(semaphore);
        VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        info.semaphoreCount = 1;
        info.pSemaphores = &native;
        info.pValues = &value;
        vk_check(nullptr, vkWaitSemaphores(static_cast<VkDevice>(device), &info, 30'000'000'000ull), "vkWaitSemaphores(foreign GPU consumer)");
    }
} // namespace lfs::core::internal
