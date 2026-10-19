/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "input/gt_navigation_repeat.hpp"
#include <gtest/gtest.h>

namespace {
    using Repeat = lfs::vis::input::GTNavigationRepeat;
    using namespace std::chrono_literals;
} // namespace

TEST(GTNavigationRepeat, TapAndReleaseDoNotRepeat) {
    Repeat repeat;
    const Repeat::Clock::time_point now{};
    EXPECT_EQ(repeat.poll(now, 5), 0);
    repeat.start(10, 1, now);
    EXPECT_EQ(repeat.poll(now + 100ms, 5), 0);
    repeat.release(10);
    EXPECT_FALSE(repeat.active());
    EXPECT_EQ(repeat.poll(now + 2s, 5), 0);
}

TEST(GTNavigationRepeat, InitialDelayThenConfiguredCadence) {
    Repeat repeat;
    const Repeat::Clock::time_point now{};
    repeat.start(10, 1, now);
    EXPECT_EQ(repeat.poll(now + 349ms, 5), 0);
    EXPECT_EQ(repeat.poll(now + 350ms, 5), 1);
    EXPECT_EQ(repeat.poll(now + 549ms, 5), 0);
    EXPECT_EQ(repeat.poll(now + 550ms, 5), 1);
    EXPECT_EQ(repeat.poll(now + 599ms, 20), 0);
    EXPECT_EQ(repeat.poll(now + 600ms, 20), 1);
}

TEST(GTNavigationRepeat, SlowRateHasNoBurstAfterDelayedFrame) {
    Repeat repeat;
    const Repeat::Clock::time_point now{};
    repeat.start(10, -1, now);
    EXPECT_EQ(repeat.poll(now + 999ms, 1), 0);
    EXPECT_EQ(repeat.poll(now + 1s, 1), -1);
    EXPECT_EQ(repeat.poll(now + 10s, 1), -1);
    EXPECT_EQ(repeat.poll(now + 10s, 1), 0);
    EXPECT_EQ(repeat.poll(now + 10500ms, 1), 0);
}

TEST(GTNavigationRepeat, DirectionChangeAndCancellation) {
    Repeat repeat;
    const Repeat::Clock::time_point now{};
    repeat.start(10, 1, now);
    repeat.start(11, -1, now + 100ms);
    repeat.release(10);
    EXPECT_TRUE(repeat.active());
    EXPECT_EQ(repeat.key(), 11);
    EXPECT_EQ(repeat.poll(now + 450ms, 5), -1);
    repeat.stop();
    EXPECT_EQ(repeat.poll(now + 1s, 5), 0);
}
