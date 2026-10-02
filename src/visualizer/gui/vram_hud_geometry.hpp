/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

namespace lfs::vis::gui::vram_hud_geometry {

    inline constexpr float kViewportPadding = 16.0f;

    [[nodiscard]] inline float clampExtent(float requested,
                                           float min_extent,
                                           float visible_extent,
                                           float position) noexcept {
        if (!std::isfinite(requested))
            requested = min_extent;
        if (requested <= 0.0f)
            requested = min_extent;
        if (!std::isfinite(visible_extent) || visible_extent <= 0.0f)
            return std::max(min_extent, requested);

        const float leading_padding = position >= 0.0f ? position : kViewportPadding;
        const float max_extent =
            std::max(1.0f, visible_extent - leading_padding - kViewportPadding);
        const float effective_min = std::min(min_extent, max_extent);
        return std::clamp(requested, effective_min, max_extent);
    }

    [[nodiscard]] inline float toLocal(float absolute, float parent_origin) noexcept {
        return absolute - parent_origin;
    }

    [[nodiscard]] inline float clampPosition(float requested,
                                             float extent,
                                             float visible_extent) noexcept {
        if (!std::isfinite(requested) || requested < 0.0f)
            return -1.0f;
        if (!std::isfinite(visible_extent) || visible_extent <= 0.0f)
            return std::max(0.0f, requested);

        const float safe_extent = std::max(1.0f, std::isfinite(extent) ? extent : 1.0f);
        const float max_pos = std::max(0.0f, visible_extent - safe_extent - kViewportPadding);
        return std::clamp(requested, 0.0f, max_pos);
    }

} // namespace lfs::vis::gui::vram_hud_geometry
