/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/video/video_export_options.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <string>

namespace {

    TEST(VideoResolutionTest, RejectsOverflowInEitherDimension) {
        using lfs::io::video::parseVideoResolution;

        for (const auto* dimension : {"2147483648", "4294967312", "4294969216",
                                      "4294971136", "18446744073709551615"}) {
            SCOPED_TRACE(dimension);
            EXPECT_FALSE(parseVideoResolution(std::string(dimension) + "x1080").has_value());
            EXPECT_FALSE(parseVideoResolution("1920x" + std::string(dimension)).has_value());
        }
        EXPECT_FALSE(parseVideoResolution("4294969216x4294968376").has_value());
    }

    TEST(VideoResolutionTest, PreservesClampingForRepresentableDimensions) {
        using lfs::io::video::parseVideoResolution;

        const auto max_dimension = std::to_string(std::numeric_limits<int>::max());
        const auto parsed = parseVideoResolution(max_dimension + "x" + max_dimension);
        ASSERT_TRUE(parsed.has_value());
        EXPECT_EQ(parsed->width, lfs::io::video::MAX_VIDEO_SIDE);
        EXPECT_EQ(parsed->height, lfs::io::video::MAX_VIDEO_SIDE);

        const auto small = parseVideoResolution("1x17");
        ASSERT_TRUE(small.has_value());
        EXPECT_EQ(small->width, lfs::io::video::MIN_VIDEO_SIDE);
        EXPECT_EQ(small->height, 16);
    }

    TEST(VideoResolutionTest, LeadingZerosDoNotChangeOverflowHandling) {
        using lfs::io::video::parseVideoResolution;

        const std::string zeros(128, '0');
        const auto parsed = parseVideoResolution(zeros + "1920x" + zeros + "1080");
        ASSERT_TRUE(parsed.has_value());
        EXPECT_EQ(parsed->width, 1920);
        EXPECT_EQ(parsed->height, 1080);
        EXPECT_FALSE(parseVideoResolution(zeros + "4294969216x1080").has_value());
        EXPECT_FALSE(parseVideoResolution("1920x" + zeros + "4294968376").has_value());
    }

    TEST(VideoResolutionTest, RejectsVeryLongDimensions) {
        using lfs::io::video::parseVideoResolution;

        const std::string dimension(4096, '9');
        EXPECT_FALSE(parseVideoResolution(dimension + "x1080").has_value());
        EXPECT_FALSE(parseVideoResolution("1920x" + dimension).has_value());
    }

} // namespace
