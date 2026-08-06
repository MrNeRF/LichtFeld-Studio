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
 * Phase 0.1 of SPEED_VRAM_OPTIMIZATION_PLAN (gate G2).
 */

#include "core/export.hpp"

#include <cstdint>

namespace lfs::core::alloc_counter {

    /// Opaque monotonic counter value. Compare with delta_since().
    using Snapshot = std::uint64_t;

    /// Current total of real device allocations since process start.
    [[nodiscard]] LFS_CORE_API Snapshot snapshot() noexcept;

    /// Allocations that occurred after @p s was taken (wrap-safe unsigned).
    [[nodiscard]] LFS_CORE_API std::uint64_t delta_since(Snapshot s) noexcept;

    /// Same as snapshot(); named for readability at log sites.
    [[nodiscard]] LFS_CORE_API std::uint64_t total() noexcept;

    /// Increment the counter. Call ONLY at real driver alloc success sites.
    LFS_CORE_API void record(std::uint64_t n = 1) noexcept;

} // namespace lfs::core::alloc_counter
