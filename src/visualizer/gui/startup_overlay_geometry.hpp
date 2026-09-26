/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>
#include <cmath>
#include <optional>

namespace lfs::vis::gui {

    struct StartupOverlayRect {
        float left = 0.0f;
        float top = 0.0f;
        float right = 0.0f;
        float bottom = 0.0f;

        [[nodiscard]] bool contains(const float x, const float y) const {
            return x >= left && y >= top && x < right && y < bottom;
        }
    };

    inline bool startupOverlayBlocksPointer(const StartupOverlayRect& card,
                                            const std::optional<StartupOverlayRect>& dropdown,
                                            const float x,
                                            const float y) {
        return card.contains(x, y) || (dropdown && dropdown->contains(x, y));
    }

    inline float startupOverlayFitDpRatio(const float maximum_ratio,
                                          const float layout_ratio,
                                          const float window_width,
                                          const float window_height,
                                          const StartupOverlayRect& visual_bounds) {
        if (maximum_ratio <= 0.0f || layout_ratio <= 0.0f ||
            window_width <= 0.0f || window_height <= 0.0f)
            return maximum_ratio;

        const float center_x = window_width * 0.5f;
        const float center_y = window_height * 0.5f;
        const float horizontal_extent =
            std::max(center_x - visual_bounds.left, visual_bounds.right - center_x);
        const float vertical_extent =
            std::max(center_y - visual_bounds.top, visual_bounds.bottom - center_y);

        float fit_scale = 1.0f;
        if (horizontal_extent > 0.0f)
            fit_scale = std::min(fit_scale, center_x / horizontal_extent);
        if (vertical_extent > 0.0f)
            fit_scale = std::min(fit_scale, center_y / vertical_extent);

        return std::min(maximum_ratio,
                        std::max(0.01f, layout_ratio * fit_scale));
    }

} // namespace lfs::vis::gui
