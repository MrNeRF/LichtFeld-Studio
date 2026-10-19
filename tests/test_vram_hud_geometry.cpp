/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/vram_hud_geometry.hpp"
#include "gui/vram_hud_sample_policy.hpp"

#include <chrono>
#include <gtest/gtest.h>

#include <limits>

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

    TEST(VramHudGeometry, SnappedResizeDoesNotReservePaddingBeyondItsAnchoredEdge) {
        EXPECT_FLOAT_EQ(clampExtent(620, 360, 800, 180, 0), 620);
        EXPECT_FLOAT_EQ(clampExtent(620, 360, 800, 180), 604);
        EXPECT_FLOAT_EQ(clampExtent(400, 200, 600, 200, 0), 400);
        EXPECT_FLOAT_EQ(clampExtent(340, 200, 600, 200, 0), 340);
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

    TEST(VramHudGeometry, PressBecomesDragOnlyPastClickSlop) {
        EXPECT_FALSE(movedPastClickSlop(0.0f, 0.0f));
        EXPECT_FALSE(movedPastClickSlop(2.0f, -2.0f));
        EXPECT_TRUE(movedPastClickSlop(25.0f, 0.0f));
        EXPECT_TRUE(movedPastClickSlop(-3.0f, 3.0f));
        EXPECT_FALSE(movedPastClickSlop(std::numeric_limits<float>::quiet_NaN(), 10.0f));
    }

    TEST(VramHudSamplePolicy, PublishesAtCadenceOrWhenExpandedChanges) {
        using namespace std::chrono_literals;
        const auto start = std::chrono::steady_clock::time_point{};
        EXPECT_TRUE(vram_hud_sample_policy::shouldPublish(false, false, start, start));
        EXPECT_FALSE(vram_hud_sample_policy::shouldPublish(true, false, start + 249ms, start));
        EXPECT_TRUE(vram_hud_sample_policy::shouldPublish(true, false, start + 250ms, start));
        EXPECT_TRUE(vram_hud_sample_policy::shouldPublish(true, true, start + 1ms, start));
    }

} // namespace lfs::vis::gui::vram_hud_geometry
