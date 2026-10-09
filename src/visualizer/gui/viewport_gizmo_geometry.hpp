/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>

namespace lfs::vis::gui {

    [[nodiscard]] inline bool viewportGizmoFits(const float width, const float height, const float scale) {
        // Reserve room beside the two-column tool rail and below the axis widget
        // for its home/focus controls. Hidden widgets must not capture input.
        const float dpi = std::max(scale, 1.0f);
        return width >= 200.0f * dpi && height >= 150.0f * dpi;
    }

} // namespace lfs::vis::gui
