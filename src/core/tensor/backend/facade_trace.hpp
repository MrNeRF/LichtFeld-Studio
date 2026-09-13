/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string_view>

// Per-entry call counters for the backend facade. The proof harness uses them
// to record which facade entry a corpus row or a gtest case actually executed,
// so launcher coverage is measured rather than inferred from method names.
// Disabled, the hot path is one relaxed load of a global flag.
namespace lfs::core::internal {

    enum class FacadeEntry : uint8_t {
#define LFS_FACADE_ENTRY(name) name,
#include "facade_entries.def"
#undef LFS_FACADE_ENTRY
        Count
    };

    inline constexpr size_t kFacadeEntryCount = static_cast<size_t>(FacadeEntry::Count);

    LFS_CORE_API extern std::atomic<bool> g_facade_trace_enabled;
    LFS_CORE_API extern std::array<std::atomic<uint64_t>, kFacadeEntryCount> g_facade_trace_counters;

    inline void facade_trace(const FacadeEntry entry) {
        if (g_facade_trace_enabled.load(std::memory_order_relaxed)) {
            g_facade_trace_counters[static_cast<size_t>(entry)].fetch_add(
                1, std::memory_order_relaxed);
        }
    }

    LFS_CORE_API void facade_trace_enable_for_testing(bool enabled);
    LFS_CORE_API bool facade_trace_enabled_for_testing();
    LFS_CORE_API void facade_trace_reset_for_testing();
    LFS_CORE_API std::array<uint64_t, kFacadeEntryCount> facade_trace_snapshot_for_testing();
    LFS_CORE_API std::string_view facade_entry_name(FacadeEntry entry);

} // namespace lfs::core::internal

#define LFS_FACADE_TRACE(name) \
    ::lfs::core::internal::facade_trace(::lfs::core::internal::FacadeEntry::name)
