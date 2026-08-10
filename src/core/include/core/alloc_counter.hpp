/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

/**
 * @file alloc_counter.hpp
 * @brief Lightweight always-on counter of real device allocations.
 *
 * Counts only true driver-level commits:
 *   - cudaMalloc / cudaMallocAsync issued by CudaMemoryPool tiers
 *     (slab growth, bucket miss, async exact, direct tier)
 *   - zeros_direct / reserve direct allocs (memory_pressure path)
 *   - rasterizer arena physical commits (cudaMalloc fallback + VMM cuMemCreate)
 *
 * Pool cache hits (slab free-list, size-bucket reuse) must NOT call record().
 * Designed for always-on release builds: one relaxed atomic increment per
 * real driver alloc site.
 *
 *
 * optional per-site attribution. Coarse pool-tier tags are always
 * recorded; a TLS logical site (densify / joint_bounds / …) can be
 * stacked via ScopedSite.
 */

#include "core/export.hpp"

#include <cstdint>
#include <string_view>

namespace lfs::core::alloc_counter {

    /// Opaque monotonic counter value. Compare with delta_since().
    using Snapshot = std::uint64_t;

    /// Coarse physical site of a driver alloc (pool tier / path).
    enum class Site : std::uint8_t {
        Unknown = 0,
        PoolBucket,
        PoolAsync,
        PoolDirect,
        Slab,
        ZerosDirect,
        Arena,
        FastgsSort,
        Count
    };

    /// Current total of real device allocations since process start.
    [[nodiscard]] LFS_CORE_API Snapshot snapshot() noexcept;

    /// Allocations that occurred after @p s was taken (wrap-safe unsigned).
    [[nodiscard]] LFS_CORE_API std::uint64_t delta_since(Snapshot s) noexcept;

    /// Same as snapshot(); named for readability at log sites.
    [[nodiscard]] LFS_CORE_API std::uint64_t total() noexcept;

    /// Increment the counter. Call ONLY at real driver alloc success sites.
    LFS_CORE_API void record(std::uint64_t n = 1) noexcept;

    /// Increment and attribute to a physical @p site (+ current TLS logical tag).
    LFS_CORE_API void record_site(Site site, std::uint64_t n = 1) noexcept;

    /// Per-site counts since last reset_site_counts() (or process start).
    [[nodiscard]] LFS_CORE_API std::uint64_t site_count(Site site) noexcept;

    /// Zero all per-site counters (does not zero the global total()).
    LFS_CORE_API void reset_site_counts() noexcept;

    /// Human-readable name for a Site enum value.
    [[nodiscard]] LFS_CORE_API const char* site_name(Site site) noexcept;

    /// Push a logical subsystem tag for nested attribution (TLS stack).
    /// Empty / null name is ignored. Pop via ScopedSite dtor or pop_site().
    LFS_CORE_API void push_site(const char* name) noexcept;
    LFS_CORE_API void pop_site() noexcept;

    /// Current logical site tag ("" when none).
    [[nodiscard]] LFS_CORE_API const char* current_logical_site() noexcept;

    /// RAII push/pop of a logical site tag.
    class LFS_CORE_API ScopedSite {
    public:
        explicit ScopedSite(const char* name) noexcept;
        ~ScopedSite() noexcept;
        ScopedSite(const ScopedSite&) = delete;
        ScopedSite& operator=(const ScopedSite&) = delete;

    private:
        bool active_ = false;
    };

} // namespace lfs::core::alloc_counter
