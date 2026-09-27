/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "metal_queue.hpp"

#include <stdexcept>

namespace lfs::core::internal::metal_queue {
    namespace {
        [[noreturn]] void unavailable() {
            throw std::runtime_error("Metal tensor queues are unavailable in this build");
        }
    } // namespace

    uint64_t create() { unavailable(); }
    void destroy(uint64_t) noexcept {}
    bool valid(const uint64_t id) { return id == 0; }
    uint64_t submit() { return 0; }
    bool ready(uint64_t) { return true; }
    void wait(uint64_t) {}
    void wait_completed(uint64_t) {}
    std::byte* host(const StorageRef&) { unavailable(); }
    bool host_copy_if_idle(const StorageRef&, const StorageRef&, size_t) { unavailable(); }
    void wait_vulkan_timeline(void*, void*, uint64_t) { unavailable(); }
    std::unique_ptr<Timestamps> timestamps(size_t) { return nullptr; }
} // namespace lfs::core::internal::metal_queue
