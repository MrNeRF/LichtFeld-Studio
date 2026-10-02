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

    RasterArenaTurns raster_arena_turns() {
#if LFS_HAS_CUDA
        if (const auto* arena = GlobalArenaManager::instance().try_get_arena()) {
            const auto stats = arena->turn_stats();
            return {.viewer_turn_ms = stats.viewer_turn_ms, .viewer_record_ms = stats.viewer_record_ms};
        }
#endif
        return {};
    }

    void raster_arena_predict_viewer_frame(const bool predict) {
#if LFS_HAS_CUDA
        if (auto* arena = GlobalArenaManager::instance().try_get_arena()) {
            if (predict)
                arena->prepare_viewer_frame();
            else
                arena->cancel_viewer_prediction();
        }
#else
        (void)predict;
#endif
    }

} // namespace lfs::core
