// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

#include "visualizer/gui/rmlui/rmlui_manager.hpp"

#include <gtest/gtest.h>

namespace lfs::vis::gui {

    TEST(RmlDragPayloadTest, PublishesReleasesAndConsumesOneTypedPayload) {
        RmlUIManager manager;

        const auto token = manager.beginDragPayload(
            "application/x-lichtfeld-project", "/tmp/example.licht", "Example");

        ASSERT_NE(token, 0);
        const auto active = manager.dragPayload();
        ASSERT_TRUE(active.has_value());
        EXPECT_EQ(active->token, token);
        EXPECT_EQ(active->type, "application/x-lichtfeld-project");
        EXPECT_EQ(active->data, "/tmp/example.licht");
        EXPECT_EQ(active->label, "Example");
        EXPECT_FALSE(active->released);

        EXPECT_TRUE(manager.endDragPayload(token));
        const auto released = manager.takeReleasedDragPayload();
        ASSERT_TRUE(released.has_value());
        EXPECT_TRUE(released->released);
        EXPECT_FALSE(manager.dragPayload().has_value());
        EXPECT_FALSE(manager.takeReleasedDragPayload().has_value());
    }

    TEST(RmlDragPayloadTest, ReplacementInvalidatesStaleSourceToken) {
        RmlUIManager manager;
        const auto stale = manager.beginDragPayload("first/type", "first", "First");
        const auto current = manager.beginDragPayload("second/type", "second", "Second");

        EXPECT_FALSE(manager.endDragPayload(stale));
        EXPECT_FALSE(manager.cancelDragPayload(stale));
        ASSERT_TRUE(manager.dragPayload().has_value());
        EXPECT_EQ(manager.dragPayload()->token, current);
        EXPECT_TRUE(manager.cancelDragPayload(current));
        EXPECT_FALSE(manager.dragPayload().has_value());
    }

    TEST(RmlDragPayloadTest, RejectsEmptyTypeOrData) {
        RmlUIManager manager;

        EXPECT_EQ(manager.beginDragPayload({}, "data"), 0);
        EXPECT_EQ(manager.beginDragPayload("type", {}), 0);
        EXPECT_FALSE(manager.dragPayload().has_value());
    }

} // namespace lfs::vis::gui
