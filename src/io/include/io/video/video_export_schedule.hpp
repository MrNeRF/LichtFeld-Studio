/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cmath>

namespace lfs::io::video {

    // N frame intervals of 1/fps cover the clip. Frame i is the camera at t = i / fps.
    [[nodiscard]] inline int videoExportFrameCount(const float duration_seconds, const int fps) {
        const long long rounded = std::llround(
            static_cast<double>(duration_seconds) * static_cast<double>(fps));
        if (rounded < 1)
            return 1;
        return static_cast<int>(rounded);
    }

    [[nodiscard]] inline float videoExportSampleTime(const int frame_index, const int fps) {
        return static_cast<float>(frame_index) / static_cast<float>(fps);
    }

} // namespace lfs::io::video
