/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/status_bar_tessle.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>

namespace lfs::vis::gui::tessle {
    Frame buildFrame(float progress, const float bar_dp, const int64_t animation_ms) {
        Frame frame;
        const float width = std::isfinite(bar_dp) ? std::max(0.0f, bar_dp) : 0.0f;
        const float travel = std::max(0.0f, width - 14.0f);
        progress = std::isfinite(progress) ? std::clamp(progress, 0.0f, 1.0f) : 0.0f;
        frame.left_dp = progress * travel;
        const int tick = static_cast<int>((std::max(int64_t{0}, animation_ms) / 100) % 4);
        constexpr std::array kMouthFrames{"closed", "half", "open", "half"};
        frame.sprite = std::format("../icon/tessle/{}.png", kMouthFrames[tick]);
        if (width < 28.0f)
            return frame;

        // Leaf snacks disappear into the jaw, leaving a clear trail for the percentage.
        for (float x = 22.0f; x < width - 4.0f; x += 20.0f) {
            if (x > frame.left_dp + 12.0f)
                frame.scenery_rml += std::format(
                    "<img class=\"tessle-leaf\" src=\"../icon/tessle/leaf.png\" style=\"left:{:.2f}dp;\"/>", x);
        }
        return frame;
    }
} // namespace lfs::vis::gui::tessle
