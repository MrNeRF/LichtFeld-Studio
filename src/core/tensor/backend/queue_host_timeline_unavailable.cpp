/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "backend_work_queue.hpp"
#include <stdexcept>
namespace lfs::core::internal {
    void validate_external_queue_device(void* device) {
        if (device)
            throw std::runtime_error("Foreign GPU timeline waits require Vulkan");
    }
    void wait_external_queue_timeline(void*, void*, uint64_t) { throw std::runtime_error("Foreign GPU timeline waits require Vulkan"); }
} // namespace lfs::core::internal
