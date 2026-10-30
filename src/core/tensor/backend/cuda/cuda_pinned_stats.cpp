/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/pinned_allocator_stats.hpp"
#include "core/pinned_memory_allocator.hpp"
namespace lfs::core {
    PinnedAllocatorStats pinned_allocator_stats() {

        const auto stats = PinnedMemoryAllocator::instance().get_stats();
        return {stats.allocated_bytes, stats.cached_bytes};
    }
} // namespace lfs::core
