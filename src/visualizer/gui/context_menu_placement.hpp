/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>

namespace lfs::vis::gui {

    struct ContextMenuPlacement {
        float left = 0.0f;
        float top = 0.0f;
        bool flipped_x = false;
        bool flipped_y = false;
    };

    // Inputs and output use context pixels. Cursor and origin are screen-space
    // coordinates so menus remain correct on viewports positioned off-origin.
    [[nodiscard]] inline ContextMenuPlacement placeContextMenu(
        const float cursor_x, const float cursor_y,
        const float screen_x, const float screen_y,
        const float viewport_width, const float viewport_height,
        const float menu_width, const float menu_height,
        const float margin) {
        const float safe_margin = std::max(0.0f, margin);
        const float local_x = cursor_x - screen_x;
        const float local_y = cursor_y - screen_y;
        const bool can_fit_x = menu_width <= viewport_width - safe_margin * 2.0f;
        const bool can_fit_y = menu_height <= viewport_height - safe_margin * 2.0f;

        float left = local_x;
        float top = local_y;
        bool flipped_x = false;
        bool flipped_y = false;

        if (can_fit_x && left + menu_width > viewport_width - safe_margin &&
            local_x - menu_width >= safe_margin) {
            left = local_x - menu_width;
            flipped_x = true;
        }
        if (can_fit_y && top + menu_height > viewport_height - safe_margin &&
            local_y - menu_height >= safe_margin) {
            top = local_y - menu_height;
            flipped_y = true;
        }

        const float max_left = viewport_width - menu_width - safe_margin;
        const float max_top = viewport_height - menu_height - safe_margin;
        if (can_fit_x)
            left = std::clamp(left, safe_margin, max_left);
        else
            left = safe_margin;
        if (can_fit_y)
            top = std::clamp(top, safe_margin, max_top);
        else
            top = safe_margin;

        return {left, top, flipped_x, flipped_y};
    }

} // namespace lfs::vis::gui
