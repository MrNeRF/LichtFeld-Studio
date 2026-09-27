/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/raster_arena.hpp"

#if LFS_HAS_CUDA
#include "core/cuda/memory_arena.hpp"
#endif

namespace lfs::core {

    bool raster_arena_under_memory_pressure() {
#if LFS_HAS_CUDA
        const auto* arena = GlobalArenaManager::instance().try_get_arena();
        return arena != nullptr && arena->is_under_memory_pressure();
#else
        return false;
#endif
    }

    void clear_raster_arena_external_backing(const void* device_ptr) {
#if LFS_HAS_CUDA
        GlobalArenaManager::instance().clear_external_backing(device_ptr);
#else
        (void)device_ptr;
#endif
    }

} // namespace lfs::core
