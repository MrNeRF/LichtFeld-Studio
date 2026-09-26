/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

namespace lfs::vis {

    struct DepthWindowState {
        float near_plane = 0.01f;
        float far_plane = 100.0f;
        float scale_x = 0.35f;
        float scale_y = 0.35f;
        float offset_x = 0.0f;
        float offset_y = 0.0f;

        [[nodiscard]] friend bool operator==(const DepthWindowState& lhs,
                                             const DepthWindowState& rhs) {
            return lhs.near_plane == rhs.near_plane && lhs.far_plane == rhs.far_plane &&
                   lhs.scale_x == rhs.scale_x && lhs.scale_y == rhs.scale_y &&
                   lhs.offset_x == rhs.offset_x && lhs.offset_y == rhs.offset_y;
        }
    };

} // namespace lfs::vis
