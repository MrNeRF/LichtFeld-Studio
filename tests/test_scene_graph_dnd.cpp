/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/elements/scene_graph_drop_target.hpp"

#include <gtest/gtest.h>

namespace lfs::vis::gui {

    TEST(SceneGraphDropTargetTest, KeepsFirstBoundaryInsideContent) {
        const auto target = computeDropTarget(0, 0, 0.1f, false, 5);
        EXPECT_EQ(target.index, 0);
        EXPECT_TRUE(target.show_line);
        EXPECT_EQ(target.line_top_dp, 1);
    }

    TEST(SceneGraphDropTargetTest, ComputesBeforeBoundary) {
        const auto target = computeDropTarget(2, 1, 0.1f, false, 5);
        EXPECT_EQ(target.index, 2);
        EXPECT_FALSE(target.into_group);
        EXPECT_TRUE(target.show_line);
        EXPECT_EQ(target.line_top_dp, target.index * 20);
        EXPECT_EQ(target.line_left_dp, 37);
    }

    TEST(SceneGraphDropTargetTest, ComputesAfterBoundary) {
        const auto target = computeDropTarget(2, 1, 0.9f, false, 5);
        EXPECT_EQ(target.index, 3);
        EXPECT_EQ(target.line_top_dp, target.index * 20);
    }

    TEST(SceneGraphDropTargetTest, ComputesGroupInterior) {
        const auto target = computeDropTarget(1, 0, 0.5f, true, 5);
        EXPECT_TRUE(target.into_group);
        EXPECT_FALSE(target.show_line);
        EXPECT_EQ(target.index, -1);
    }

    TEST(SceneGraphDropTargetTest, ComputesEndOfListBoundary) {
        const auto target = computeDropTarget(5, 0, 1.0f, false, 5);
        EXPECT_EQ(target.index, 5);
        EXPECT_TRUE(target.show_line);
        EXPECT_EQ(target.line_top_dp, 5 * 20 - 2);
        EXPECT_EQ(target.line_left_dp, 4);
    }

} // namespace lfs::vis::gui
