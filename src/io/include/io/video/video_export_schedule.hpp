/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

namespace lfs::io::video {

    // N frame intervals of 1/fps cover the clip. Frame i is the camera at t = i / fps.
    [[nodiscard]] inline int videoExportFrameCount(const float duration_seconds, const int fps, const int denominator = 1) {
        if (!std::isfinite(duration_seconds) || duration_seconds <= 0 || fps <= 0 || denominator <= 0)
            return 1;
        const double count = static_cast<double>(duration_seconds) * fps / denominator;
        if (count >= std::numeric_limits<int>::max())
            return std::numeric_limits<int>::max();
        return std::max(1, static_cast<int>(std::llround(count)));
    }

    [[nodiscard]] inline float videoExportSampleTime(const int frame_index, const int fps, const int denominator = 1) {
        return static_cast<float>(static_cast<double>(frame_index) * denominator / fps);
    }

} // namespace lfs::io::video
