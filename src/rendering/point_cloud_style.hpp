/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

namespace lfs::rendering::pcraster {

    // Point styling of the software raster. These mirror the constants of the
    // same names in viewport/point_cloud.vert; a test keeps the two in step.
    inline constexpr unsigned SELECTION_PREVIEW_COLOR_INDEX = 256u;
    inline constexpr float DESATURATE_BLEND = 0.75f;
    inline constexpr float SELECTION_COMMITTED_BLEND = 0.75f;
    inline constexpr float SELECTION_PREVIEW_BLEND = 0.9f;
    inline constexpr float SELECTION_COMMITTED_SCALE = 1.35f;
    inline constexpr float SELECTION_PREVIEW_SCALE = 1.6f;

} // namespace lfs::rendering::pcraster
