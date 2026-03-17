/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/elements/loss_graph_element.hpp"

#include <gtest/gtest.h>

#include <deque>

TEST(LossGraphElementTest, AutoScaleAddsMarginForVaryingData) {
    const auto [data_min, data_max] =
        lfs::vis::gui::computeLossGraphRange(std::deque<float>{1.0f, 2.0f, 3.0f});

    EXPECT_NEAR(data_min, 0.9f, 1e-5f);
    EXPECT_NEAR(data_max, 3.1f, 1e-5f);
}

TEST(LossGraphElementTest, AutoScaleExpandsFlatData) {
    const auto [data_min, data_max] =
        lfs::vis::gui::computeLossGraphRange(std::deque<float>{5.0f, 5.0f, 5.0f});

    EXPECT_FLOAT_EQ(data_min, 4.0f);
    EXPECT_FLOAT_EQ(data_max, 6.0f);
}

TEST(LossGraphElementTest, ExplicitScaleOverridesAutoScale) {
    const auto [data_min, data_max] = lfs::vis::gui::computeLossGraphRange(
        std::deque<float>{10.0f, 20.0f, 30.0f}, std::pair{0.0f, 100.0f});

    EXPECT_FLOAT_EQ(data_min, 0.0f);
    EXPECT_FLOAT_EQ(data_max, 100.0f);
}

TEST(LossGraphElementTest, ExplicitScaleNormalizationClampsOutOfRangeSamples) {
    EXPECT_FLOAT_EQ(lfs::vis::gui::normalizeLossGraphValue(-1.0f, 0.0f, 1.0f), 0.0f);
    EXPECT_FLOAT_EQ(lfs::vis::gui::normalizeLossGraphValue(2.0f, 0.0f, 1.0f), 1.0f);
    EXPECT_FLOAT_EQ(lfs::vis::gui::normalizeLossGraphValue(0.25f, 0.0f, 1.0f), 0.25f);
}
