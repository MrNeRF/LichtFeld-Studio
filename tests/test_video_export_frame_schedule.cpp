/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/video/video_export_schedule.hpp"

#include <cmath>
#include <gtest/gtest.h>

namespace {

    struct ScheduleCase {
        float duration;
        int fps;
        int frames;
        float last;
    };

    TEST(VideoExportFrameSchedule, CountAndSampleTimesMatchPathDuration) {
        const ScheduleCase cases[] = {
            {1.0f, 2, 2, 0.5f},
            {30.0f, 2, 60, 29.5f},
            {10.0f, 30, 300, 299.0f / 30.0f},
            {1.5f, 3, 5, 4.0f / 3.0f},           // 4.5 rounds away from zero
            {1.25f, 10, 13, 1.2f},               // 12.5 rounds away from zero
            {1.1f, 24, 26, 25.0f / 24.0f},       // 26.4
            {1.0f / 3.0f, 30, 10, 9.0f / 30.0f}, // fractional product on an integer
            {0.04f, 10, 1, 0.0f},                // 0.4 rounds to 0, so at least one frame
        };

        for (const auto& row : cases) {
            SCOPED_TRACE(::testing::Message() << row.duration << "s @" << row.fps << " fps");
            ASSERT_EQ(lfs::io::video::videoExportFrameCount(row.duration, row.fps), row.frames);
            for (int i = 0; i < row.frames; ++i) {
                const float time = lfs::io::video::videoExportSampleTime(i, row.fps);
                EXPECT_TRUE(std::isfinite(time));
                EXPECT_FLOAT_EQ(time, static_cast<float>(i) / static_cast<float>(row.fps));
            }
            const float last = lfs::io::video::videoExportSampleTime(row.frames - 1, row.fps);
            EXPECT_NEAR(last, row.last, 1.0e-5f);
            if (row.frames > 1)
                EXPECT_LT(last, row.duration);
            const double product = static_cast<double>(row.duration) * static_cast<double>(row.fps);
            if (std::abs(product - std::round(product)) < 1.0e-4)
                EXPECT_NEAR(static_cast<double>(row.frames) / row.fps, row.duration, 1.0e-4);
        }
    }

} // namespace
