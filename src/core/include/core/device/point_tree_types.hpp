/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <cstdint>

namespace lfs::core::internal {
    inline constexpr uint32_t kPointTreeFanoutBits = 5;
    inline constexpr uint32_t kPointTreeFanout = 1u << kPointTreeFanoutBits;
    inline constexpr uint32_t kPointTreeMaxLevels = 8;
    struct PointTreeProgram {
        uint32_t points = 0;
        uint32_t references = 0;
        uint32_t levels = 0;
        uint32_t level_offset[kPointTreeMaxLevels] = {};
        uint32_t level_count[kPointTreeMaxLevels] = {};
        // Used when no per-point radii are given.
        float radius = 0;
        int32_t max_count = 0;
    };
} // namespace lfs::core::internal
