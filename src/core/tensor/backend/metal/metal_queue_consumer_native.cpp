/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "metal_queue_consumer.hpp"
namespace lfs::core::internal {
    std::unique_ptr<MetalVulkanQueue> make_metal_queue_consumer(void*, void* consumer) { return make_metal_work_queue(consumer); }
    void wait_metal_queue_consumer(void*, void*, uint64_t) { throw std::runtime_error("Metal/Vulkan timeline interop is unavailable in this build"); }
} // namespace lfs::core::internal
