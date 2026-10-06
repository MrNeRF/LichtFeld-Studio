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

using lfs::vis::FramePlan;
using lfs::vis::FrameRateTracker;
using lfs::vis::FrameReason;

namespace {
    FramePlan plan(FrameReason reason) {
        FramePlan result;
        result.reasons.set(static_cast<std::size_t>(reason));
        return result;
    }
} // namespace

TEST(FrameRateTracker, UiOnlyActivityDoesNotCountViews) {
    FrameRateTracker rates;
    const auto start = FrameRateTracker::Clock::time_point{};
    rates.countView(start);
    rates.countPresented(plan(FrameReason::SceneChange), start);
    rates.countPresented(plan(FrameReason::Input), start + 900ms);
    const auto sample = rates.sample(start + 1100ms);
    EXPECT_FLOAT_EQ(sample.ui, 1);
    EXPECT_FLOAT_EQ(sample.view, 0);
}

TEST(FrameRateTracker, IdlePresentationExpiresOnceWithoutCountingOrRearming) {
    FrameRateTracker rates;
    const auto start = FrameRateTracker::Clock::time_point{};
    EXPECT_FALSE(rates.idleDeadline());
    rates.countPresented(plan(FrameReason::Input), start);
    EXPECT_EQ(rates.idleDeadline(), start + 1s);
    EXPECT_FALSE(rates.idleDue(start + 999ms));
    EXPECT_TRUE(rates.idleDue(start + 1s));
    // A failed presentation must leave the expiry available for a retry.
    EXPECT_TRUE(rates.idleDue(start + 1100ms));
    rates.countPresented(plan(FrameReason::FpsIdle), start + 1100ms);
    EXPECT_FALSE(rates.idleDeadline());
    EXPECT_FALSE(rates.idleDue(start + 2s));
    EXPECT_FLOAT_EQ(rates.sample(start + 1100ms).ui, 0);
    EXPECT_FLOAT_EQ(rates.sample(start + 1100ms).view, 0);
    rates.countPresented(plan(FrameReason::Input), start + 3s);
    EXPECT_EQ(rates.idleDeadline(), start + 4s);
    EXPECT_FLOAT_EQ(rates.sample(start + 3s).ui, 1);
}

TEST(FrameRateTracker, ActivityAlongsideIdleRearmsTheDeadline) {
    FrameRateTracker rates;
    const auto start = FrameRateTracker::Clock::time_point{};
    rates.countPresented(plan(FrameReason::Input), start);
    auto mixed = plan(FrameReason::FpsIdle);
    mixed.reasons.set(static_cast<std::size_t>(FrameReason::Input));
    rates.countPresented(mixed, start + 1s);
    EXPECT_EQ(rates.idleDeadline(), start + 2s);
    EXPECT_FLOAT_EQ(rates.sample(start + 1s).ui, 1);
}
