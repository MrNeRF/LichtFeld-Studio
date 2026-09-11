/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/alloc_counter.hpp"

#include <atomic>

namespace lfs::core::alloc_counter {
    namespace {
        // Process-wide counter living in lfs_core.so so every SO shares one value.
        std::atomic<std::uint64_t> g_device_allocs{0};
        std::atomic<std::uint64_t> g_site_counts[static_cast<std::size_t>(Site::Count)]{};
        constexpr int kMaxLogicalDepth = 8;
        thread_local const char* t_logical_stack[kMaxLogicalDepth]{};
        thread_local int t_logical_depth = 0;
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
        record_site(Site::Unknown, n);
    }

    void record_site(const Site site, const std::uint64_t n) noexcept {
        if (n == 0) {
            return;
        }
        g_device_allocs.fetch_add(n, std::memory_order_relaxed);
        const auto idx = static_cast<std::size_t>(site);
        if (idx < static_cast<std::size_t>(Site::Count)) {
            g_site_counts[idx].fetch_add(n, std::memory_order_relaxed);
        }
    }

    std::uint64_t site_count(const Site site) noexcept {
        const auto idx = static_cast<std::size_t>(site);
        if (idx >= static_cast<std::size_t>(Site::Count)) {
            return 0;
        }
        return g_site_counts[idx].load(std::memory_order_relaxed);
    }

    void reset_site_counts() noexcept {
        for (auto& c : g_site_counts) {
            c.store(0, std::memory_order_relaxed);
        }
    }

    const char* site_name(const Site site) noexcept {
        switch (site) {
        case Site::Unknown:
            return "unknown";
        case Site::PoolBucket:
            return "pool_bucket";
        case Site::PoolAsync:
            return "pool_async";
        case Site::PoolDirect:
            return "pool_direct";
        case Site::Slab:
            return "slab";
        case Site::ZerosDirect:
            return "zeros_direct";
        case Site::Arena:
            return "arena";
        case Site::FastgsSort:
            return "fastgs_sort";
        case Site::Count:
            break;
        }
        return "invalid";
    }

    void push_site(const char* name) noexcept {
        if (name == nullptr || name[0] == '\0') {
            return;
        }
        if (t_logical_depth >= kMaxLogicalDepth) {
            return;
        }
        t_logical_stack[t_logical_depth++] = name;
    }

    void pop_site() noexcept {
        if (t_logical_depth > 0) {
            --t_logical_depth;
        }
    }

    const char* current_logical_site() noexcept {
        if (t_logical_depth <= 0) {
            return "";
        }
        const char* n = t_logical_stack[t_logical_depth - 1];
        return n ? n : "";
    }

    ScopedSite::ScopedSite(const char* name) noexcept {
        if (name != nullptr && name[0] != '\0') {
            push_site(name);
            active_ = true;
        }
    }

    ScopedSite::~ScopedSite() noexcept {
        if (active_) {
            pop_site();
        }
    }

} // namespace lfs::core::alloc_counter
