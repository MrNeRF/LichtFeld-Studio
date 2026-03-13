/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/viewport.hpp"

#include <gtest/gtest.h>

TEST(ViewportTest, InvalidWorldPositionUsesNamedSentinel) {
    Viewport viewport(100, 100);

    const glm::vec3 invalid = viewport.unprojectPixel(50.0f, 50.0f, -1.0f);

    EXPECT_FALSE(Viewport::isValidWorldPosition(invalid));
    EXPECT_FLOAT_EQ(invalid.x, Viewport::INVALID_WORLD_POS);
    EXPECT_FLOAT_EQ(invalid.y, Viewport::INVALID_WORLD_POS);
    EXPECT_FLOAT_EQ(invalid.z, Viewport::INVALID_WORLD_POS);
}

TEST(ViewportTest, UnprojectPixelDependsOnScreenPixel) {
    Viewport viewport(100, 100);
    viewport.camera.R = glm::mat3(1.0f);
    viewport.camera.t = glm::vec3(0.0f);

    const glm::vec3 center = viewport.unprojectPixel(50.0f, 50.0f, 10.0f);
    const glm::vec3 top_left = viewport.unprojectPixel(0.0f, 0.0f, 10.0f);

    ASSERT_TRUE(Viewport::isValidWorldPosition(center));
    ASSERT_TRUE(Viewport::isValidWorldPosition(top_left));
    EXPECT_NEAR(center.x, 0.0f, 1e-4f);
    EXPECT_NEAR(center.y, 0.0f, 1e-4f);
    EXPECT_NEAR(center.z, 10.0f, 1e-4f);
    EXPECT_LT(top_left.x, center.x);
    EXPECT_LT(top_left.y, center.y);
    EXPECT_NEAR(top_left.z, center.z, 1e-4f);
}
