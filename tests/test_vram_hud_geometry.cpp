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

    TEST(VramHudGeometry, CompactHudCapturesOnlyItsVisibleRectangle) {
        EXPECT_FALSE(pointerTargetEnabled(true, true, false));
        EXPECT_TRUE(pointerTargetEnabled(true, true, true));
        EXPECT_TRUE(pointerTargetEnabled(true, false, false));
        EXPECT_FALSE(pointerTargetEnabled(false, true, true));
        EXPECT_TRUE(capturesPointer(true, 40.0f, 60.0f, 300.0f, 142.0f, 80.0f, 90.0f));
        EXPECT_FALSE(capturesPointer(true, 40.0f, 60.0f, 300.0f, 142.0f, 400.0f, 160.0f));
        EXPECT_FALSE(capturesPointer(false, 40.0f, 60.0f, 300.0f, 142.0f, 80.0f, 90.0f));
    }

    TEST(VramHudGeometry, CompactDragClampsUsingVisibleHudExtent) {
        constexpr float viewport_width = 1120.0f;
        constexpr float expanded_width = 620.0f;
        constexpr float compact_width = 300.0f;
        const auto extent = dragExtent(true, expanded_width, compact_width);
        EXPECT_FLOAT_EQ(extent, compact_width);
        EXPECT_FLOAT_EQ(clampDragPosition(900.0f, extent, viewport_width), 804.0f);
    }

} // namespace lfs::vis::gui::vram_hud_geometry
