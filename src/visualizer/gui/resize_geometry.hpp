/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>

namespace lfs::vis::gui {

    inline constexpr float RESIZE_GRAB_HALF_WIDTH_DP = 4.0f;

    struct ResizeHitZone {
        float min;
        float max;

        [[nodiscard]] bool contains(float position) const {
            return position >= min && position < max;
        }
    };

    [[nodiscard]] inline ResizeHitZone resizeHitZone(float edge, float scale,
                                                     float half_width_dp = RESIZE_GRAB_HALF_WIDTH_DP) {
        const float half_width = half_width_dp * std::max(scale, 1.0f);
        return {edge - half_width, edge + half_width};
    }

    [[nodiscard]] inline float resizeContentWidth(float dock_width, float scale) {
        return std::max(0.0f, dock_width - RESIZE_GRAB_HALF_WIDTH_DP * std::max(scale, 1.0f));
    }

    [[nodiscard]] inline ResizeHitZone bottomDockResizeHitZone(float edge, float scale, float grip_height) {
        const auto edge_zone = resizeHitZone(edge, scale);
        return {edge_zone.min, edge + grip_height + RESIZE_GRAB_HALF_WIDTH_DP * std::max(scale, 1.0f)};
    }

    struct ResizeDrag {
        float start_edge;
        float start_mouse;

        [[nodiscard]] float edgeAt(float mouse, float minimum, float maximum) const {
            return std::clamp(start_edge + mouse - start_mouse, minimum, maximum);
        }
    };

} // namespace lfs::vis::gui
