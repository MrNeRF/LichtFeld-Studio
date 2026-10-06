/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

namespace lfs::core {

    // The raster arena is a CUDA training resource. These calls are no-ops
    // when that arena is not part of the build.
    [[nodiscard]] LFS_CORE_API bool raster_arena_under_memory_pressure();
    LFS_CORE_API void clear_raster_arena_external_backing(const void* device_ptr = nullptr);

    // Measured viewer turns on the arena that training shares; zero without one.
    struct RasterArenaTurns {
        double viewer_turn_ms = 0.0;
        double viewer_record_ms = 0.0;
    };
    [[nodiscard]] LFS_CORE_API RasterArenaTurns raster_arena_turns();
    // Predicts the next viewer turn for navigation, or withdraws that prediction.
    LFS_CORE_API void raster_arena_predict_viewer_frame(bool predict);

} // namespace lfs::core
