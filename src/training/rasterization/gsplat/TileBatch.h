/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <cstdint>

namespace gsplat_lfs {
    // Linear, half-open range in the original full-frame tile grid.
    struct TileRange {
        uint32_t begin = 0;
        uint32_t end = UINT32_MAX;
    };
    struct TileBatch {
        TileRange tiles;
        int64_t count;
    };
} // namespace gsplat_lfs
