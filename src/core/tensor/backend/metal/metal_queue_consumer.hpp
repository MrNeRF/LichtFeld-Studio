/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "../tensor_vulkan_interop.hpp"
namespace lfs::core::internal {
    std::unique_ptr<MetalVulkanQueue> make_metal_queue_consumer(void*, void*);
    void wait_metal_queue_consumer(void*, void*, uint64_t);
} // namespace lfs::core::internal
