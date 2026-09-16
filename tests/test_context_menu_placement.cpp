/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/context_menu_placement.hpp"

#include <gtest/gtest.h>

namespace lfs::vis::gui {

    TEST(ContextMenuPlacement, KeepsMenuAtCursorWhenItFits) {
        const auto result = placeContextMenu(120.0f, 80.0f, 0.0f, 0.0f,
                                             800.0f, 600.0f, 180.0f, 120.0f, 8.0f);

        EXPECT_FLOAT_EQ(result.left, 120.0f);
        EXPECT_FLOAT_EQ(result.top, 80.0f);
        EXPECT_FALSE(result.flipped_x);
        EXPECT_FALSE(result.flipped_y);
    }

    TEST(ContextMenuPlacement, FlipsAtRightAndBottomEdges) {
        const auto result = placeContextMenu(770.0f, 575.0f, 0.0f, 0.0f,
                                             800.0f, 600.0f, 180.0f, 120.0f, 8.0f);

        EXPECT_FLOAT_EQ(result.left, 590.0f);
        EXPECT_FLOAT_EQ(result.top, 455.0f);
        EXPECT_TRUE(result.flipped_x);
        EXPECT_TRUE(result.flipped_y);
    }

    TEST(ContextMenuPlacement, AccountsForViewportOriginAndClampsOversizedMenus) {
        const auto result = placeContextMenu(-1450.0f, 900.0f, -1600.0f, 700.0f,
                                             240.0f, 180.0f, 300.0f, 250.0f, 10.0f);

        EXPECT_FLOAT_EQ(result.left, 10.0f);
        EXPECT_FLOAT_EQ(result.top, 10.0f);
        EXPECT_FALSE(result.flipped_x);
        EXPECT_FALSE(result.flipped_y);
    }

    TEST(ContextMenuPlacement, FlipsRelativeToViewportWithNonzeroOrigin) {
        const auto result = placeContextMenu(-1410.0f, 850.0f, -1600.0f, 700.0f,
                                             240.0f, 180.0f, 80.0f, 60.0f, 8.0f);

        EXPECT_FLOAT_EQ(result.left, 110.0f);
        EXPECT_FLOAT_EQ(result.top, 90.0f);
        EXPECT_TRUE(result.flipped_x);
        EXPECT_TRUE(result.flipped_y);
    }

    TEST(ContextMenuPlacement, ReopeningAtTheSameAnchorProducesTheSamePlacement) {
        const auto first = placeContextMenu(770.0f, 575.0f, 0.0f, 0.0f,
                                            800.0f, 600.0f, 180.0f, 120.0f, 8.0f);
        const auto reopened = placeContextMenu(770.0f, 575.0f, 0.0f, 0.0f,
                                               800.0f, 600.0f, 180.0f, 120.0f, 8.0f);

        EXPECT_FLOAT_EQ(reopened.left, first.left);
        EXPECT_FLOAT_EQ(reopened.top, first.top);
        EXPECT_TRUE(reopened.flipped_x);
        EXPECT_TRUE(reopened.flipped_y);
    }

    TEST(ContextMenuPlacement, FlipsOnlyWhenTheMenuCanFitOnThatSide) {
        const auto result = placeContextMenu(790.0f, 300.0f, 0.0f, 0.0f,
                                             800.0f, 600.0f, 790.0f, 120.0f, 8.0f);

        EXPECT_FLOAT_EQ(result.left, 8.0f);
        EXPECT_FALSE(result.flipped_x);
    }

} // namespace lfs::vis::gui
