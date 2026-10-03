/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "framerate_controller.hpp"
#include <gtest/gtest.h>

using namespace std::chrono_literals;
using lfs::vis::FramerateController;

TEST(FramerateWindow, EmptyAndSingleFrameDoNotInventIntervals) {
    FramerateController rate;
    const auto start = FramerateController::Clock::time_point{};
    EXPECT_FLOAT_EQ(rate.getAverageFPS(start), 0);
    rate.beginFrame(start + 10s);
    EXPECT_FLOAT_EQ(rate.getAverageFPS(start + 10s), 1);
    EXPECT_FLOAT_EQ(rate.getAverageFPS(start + 11s), 0);
}

TEST(FramerateWindow, CountsOnlyTrailingSecond) {
    FramerateController rate;
    const auto start = FramerateController::Clock::time_point{};
    for (int i = 0; i < 120; ++i)
        rate.beginFrame(start + i * 10ms);
    EXPECT_FLOAT_EQ(rate.getAverageFPS(start + 1190ms), 100);
    EXPECT_FLOAT_EQ(rate.getAverageFPS(start + 1690ms), 50);
    EXPECT_FLOAT_EQ(rate.getAverageFPS(start + 2190ms), 0);
}

TEST(FramerateWindow, SparsePreviewDoesNotUseReciprocalFrameTime) {
    FramerateController rate;
    const auto start = FramerateController::Clock::time_point{};
    for (int i = 0; i < 3; ++i)
        rate.beginFrame(start + i * 330ms);
    EXPECT_FLOAT_EQ(rate.getAverageFPS(start + 900ms), 3);
    EXPECT_FLOAT_EQ(rate.getAverageFPS(start + 1100ms), 2);
    EXPECT_FLOAT_EQ(rate.getAverageFPS(start + 1660ms), 0);
    rate.beginFrame(start + 30s);
    EXPECT_FLOAT_EQ(rate.getAverageFPS(start + 30s), 1);
}

TEST(FramerateWindow, PresentationDoesNotCountAsViewportRender) {
    FramerateController ui, viewport;
    const auto start = FramerateController::Clock::time_point{};
    viewport.beginFrame(start);
    for (int i = 0; i < 150; ++i)
        ui.beginFrame(start + i * 10ms);
    EXPECT_FLOAT_EQ(ui.getAverageFPS(start + 1490ms), 100);
    EXPECT_FLOAT_EQ(viewport.getAverageFPS(start + 1490ms), 0);
}
