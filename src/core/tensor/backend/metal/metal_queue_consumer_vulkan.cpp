/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "metal_queue.hpp"
#include "metal_queue_consumer.hpp"
namespace lfs::core::internal {
    std::unique_ptr<MetalVulkanQueue> make_metal_queue_consumer(void* device, void* consumer) { return make_metal_vulkan_queue(device, consumer); }
    void wait_metal_queue_consumer(void* device, void* semaphore, uint64_t value) { metal_queue::wait_vulkan_timeline(device, semaphore, value); }
} // namespace lfs::core::internal
