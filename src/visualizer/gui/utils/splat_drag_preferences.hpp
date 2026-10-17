/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <algorithm>
#include <optional>
#include <string>

namespace lfs::vis {
    struct SplatDragPreferenceState {
        std::string format = "ply";
        int sh_degree = 3;
        bool blender_axes = true;
    };
    // Preserve the old active format's SH choice while removing the quality
    // switch. A negative old value meant all available coefficients.
    inline int resolveSplatDragShLimit(std::optional<int> limit, bool legacy_maximum, int legacy_limit) {
        return std::clamp(limit.value_or(legacy_maximum || legacy_limit < 0 ? 3 : legacy_limit), 0, 3);
    }
} // namespace lfs::vis
