/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "facade_trace.hpp"

namespace lfs::core::internal {

    std::atomic<bool> g_facade_trace_enabled{false};
    std::array<std::atomic<uint64_t>, kFacadeEntryCount> g_facade_trace_counters{};

    void facade_trace_enable_for_testing(const bool enabled) {
        g_facade_trace_enabled.store(enabled, std::memory_order_relaxed);
    }

    bool facade_trace_enabled_for_testing() {
        return g_facade_trace_enabled.load(std::memory_order_relaxed);
    }

    void facade_trace_reset_for_testing() {
        for (auto& counter : g_facade_trace_counters) {
            counter.store(0, std::memory_order_relaxed);
        }
    }

    std::array<uint64_t, kFacadeEntryCount> facade_trace_snapshot_for_testing() {
        std::array<uint64_t, kFacadeEntryCount> snapshot{};
        for (size_t index = 0; index < kFacadeEntryCount; ++index) {
            snapshot[index] = g_facade_trace_counters[index].load(std::memory_order_relaxed);
        }
        return snapshot;
    }

    std::string_view facade_entry_name(const FacadeEntry entry) {
        static constexpr std::array<std::string_view, kFacadeEntryCount> names{
#define LFS_FACADE_ENTRY(name) #name,
#include "facade_entries.def"
#undef LFS_FACADE_ENTRY
        };
        const auto index = static_cast<size_t>(entry);
        return index < names.size() ? names[index] : std::string_view{"invalid"};
    }

} // namespace lfs::core::internal
