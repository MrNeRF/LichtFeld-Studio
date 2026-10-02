/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/vram_hud_geometry.hpp"

#include <gtest/gtest.h>

namespace lfs::vis::gui::vram_hud_geometry {

    TEST(VramHudGeometry, ConvertsAbsolutePositionToViewportLocal) {
        constexpr float dock_origin = 742.0f;
        constexpr float window_absolute = 902.0f;
        constexpr float drag_delta = 120.0f;

        const float local_start = toLocal(window_absolute, dock_origin);
        const float local_after_drag = local_start + drag_delta;

        EXPECT_FLOAT_EQ(local_start, 160.0f);
        EXPECT_FLOAT_EQ(dock_origin + local_after_drag, 1022.0f);
    }

    TEST(VramHudGeometry, ClampsToVisibleViewportAndKeepsInitialMinimumWidth) {
        constexpr float visible_viewport_width = 1120.0f;

        EXPECT_FLOAT_EQ(clampPosition(1000.0f, 600.0f, visible_viewport_width), 504.0f);
        EXPECT_FLOAT_EQ(clampExtent(30.0f, 360.0f, visible_viewport_width, 0.0f), 360.0f);
    }

} // namespace lfs::vis::gui::vram_hud_geometry
