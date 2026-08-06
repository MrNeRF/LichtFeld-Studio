/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/alloc_counter.hpp"

#include <atomic>

namespace lfs::core::alloc_counter {
    namespace {
        // Process-wide counter living in lfs_core.so so every SO shares one value.
        std::atomic<std::uint64_t> g_device_allocs{0};
    } // namespace

    Snapshot snapshot() noexcept {
        return g_device_allocs.load(std::memory_order_relaxed);
    }

    std::uint64_t delta_since(const Snapshot s) noexcept {
        return snapshot() - s;
    }

    std::uint64_t total() noexcept {
        return snapshot();
    }

    void record(const std::uint64_t n) noexcept {
        if (n == 0) {
            return;
        }
        g_device_allocs.fetch_add(n, std::memory_order_relaxed);
    }

} // namespace lfs::core::alloc_counter
