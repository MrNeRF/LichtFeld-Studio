/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"

#include <cstddef>

namespace lfs::core {

    struct PinnedAllocatorStats {
        std::size_t allocated_bytes = 0;
        std::size_t cached_bytes = 0;
    };

    LFS_CORE_API PinnedAllocatorStats pinned_allocator_stats();

} // namespace lfs::core
