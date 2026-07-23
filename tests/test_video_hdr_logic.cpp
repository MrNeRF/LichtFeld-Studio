/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/hdr_tonemap.hpp"
#include "io/video_frame_extractor.hpp"

extern "C" {
#include <libavutil/pixfmt.h>
}

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <string>

namespace {

    using lfs::io::ExtractionMode;
    using lfs::io::HdrFormat;
    using lfs::io::ResolutionMode;
    using lfs::io::VideoFrameExtractor;

    VideoFrameExtractor::Params validParams() {
        VideoFrameExtractor::Params params;
        params.mode = ExtractionMode::FPS;
        params.fps = 1.0;
        params.start_time = 0.0;
        params.end_time = 10.0;
        return params;
    }

} // namespace

TEST(SparseExtractionContract, TargetCountUsesHalfOpenTrimRange) {
    EXPECT_EQ(lfs::io::calculateFpsSampleCount(0.0, 10.0, 1.0), 10u);
    EXPECT_EQ(lfs::io::calculateFpsSampleCount(2.25, 2.26, 1.0), 1u);
    EXPECT_EQ(lfs::io::calculateFpsSampleCount(0.0, 1.0001, 2.0), 3u);
    EXPECT_EQ(lfs::io::calculateFpsSampleCount(0.0, 1.0, 0.0), 0u);

    const std::size_t target_count =
        lfs::io::calculateFpsSampleCount(2.0, 10.0, 1.0);
    ASSERT_EQ(target_count, 8u);
    EXPECT_DOUBLE_EQ(
        lfs::io::fpsSampleTime(2.0, 10.0, 1.0, target_count - 1), 9.0);
    EXPECT_LT(lfs::io::fpsSampleTime(2.0, 10.0, 1.0, 8), 10.0);
}

TEST(SparseExtractionContract, FinalFrameMaySatisfyCoveredTailTarget) {
    EXPECT_TRUE(lfs::io::frameCoversSampleTime(9.0, 1.0, 9.75));
    EXPECT_FALSE(lfs::io::frameCoversSampleTime(9.0, 1.0, 10.0));
    EXPECT_FALSE(lfs::io::frameCoversSampleTime(9.0, 0.5, 9.75));
}

TEST(SparseExtractionContract, PastEndFillsCoveredRetainedTail) {
    constexpr double start_time = 0.0;
    constexpr double end_time = 10.0;
    constexpr double target_fps = 1.0;
    constexpr double frame_duration = 2.0;
    constexpr std::array frame_times{0.0, 2.0, 4.0, 8.0};
    constexpr double next_decoded_frame_time = 10.5;

    const std::size_t target_count =
        lfs::io::calculateFpsSampleCount(start_time, end_time, target_fps);
    std::size_t next_target_index = 8;
    const bool reached_eof = false;
    const bool reached_end = next_decoded_frame_time >= end_time;

    ASSERT_EQ(target_count, 10u);
    EXPECT_TRUE(lfs::io::frameCoversSampleTime(
        frame_times.back(), frame_duration,
        lfs::io::fpsSampleTime(start_time, end_time, target_fps, 8)));
    EXPECT_TRUE(lfs::io::frameCoversSampleTime(
        frame_times.back(), frame_duration,
        lfs::io::fpsSampleTime(start_time, end_time, target_fps, 9)));
    ASSERT_TRUE(lfs::io::shouldFillRetainedFpsTail(reached_eof, reached_end));
    while (lfs::io::shouldFillRetainedFpsTail(reached_eof, reached_end) &&
           next_target_index < target_count) {
        const double target_time = lfs::io::fpsSampleTime(
            start_time, end_time, target_fps, next_target_index);
        if (!lfs::io::frameCoversSampleTime(
                frame_times.back(), frame_duration, target_time)) {
            break;
        }
        ++next_target_index;
    }

    EXPECT_EQ(next_target_index, target_count);
}

TEST(HdrFormatClassification, NativeProfileFivePrecedesCompatibilityMetadata) {
    EXPECT_EQ(
        lfs::io::detectDolbyVisionFormat(AVCOL_TRC_SMPTE2084, 5, 1),
        HdrFormat::DOLBY_VISION_NATIVE);
    EXPECT_EQ(
        lfs::io::detectDolbyVisionFormat(AVCOL_TRC_SMPTE2084, 8, 1),
        HdrFormat::DOLBY_VISION_HDR10);
    EXPECT_EQ(
        lfs::io::detectDolbyVisionFormat(AVCOL_TRC_UNSPECIFIED, 8, 4),
        HdrFormat::DOLBY_VISION_HLG);
    EXPECT_EQ(
        lfs::io::detectDolbyVisionFormat(AVCOL_TRC_SMPTE2084, 7, 1),
        HdrFormat::DOLBY_VISION_NATIVE);
}

TEST(HdrFormatClassification, StaticHdrMetadataRequiresTenBitSource) {
    EXPECT_EQ(
        lfs::io::detectHdrFormat(AVCOL_TRC_UNSPECIFIED, 10, true, false),
        HdrFormat::HDR10);
    EXPECT_EQ(
        lfs::io::detectHdrFormat(AVCOL_TRC_UNSPECIFIED, 10, false, true),
        HdrFormat::HDR10);
    EXPECT_EQ(
        lfs::io::detectHdrFormat(AVCOL_TRC_UNSPECIFIED, 8, true, true),
        HdrFormat::SDR);
    EXPECT_EQ(
        lfs::io::detectHdrFormat(AVCOL_TRC_UNSPECIFIED, 10, false, false),
        HdrFormat::SDR);
    EXPECT_EQ(
        lfs::io::detectHdrFormat(AVCOL_TRC_SMPTE2084, 8),
        HdrFormat::HDR10);
    EXPECT_EQ(
        lfs::io::detectHdrFormat(AVCOL_TRC_ARIB_STD_B67, 8),
        HdrFormat::HLG);
}

TEST(VideoFrameExtractorParams, ComputesCheckedOutputLayout) {
    auto params = validParams();
    params.resolution_mode = ResolutionMode::Scale;
    params.scale = 0.5f;

    VideoFrameExtractor::ValidatedLayout layout;
    std::string error;
    ASSERT_TRUE(VideoFrameExtractor::validateParams(
        params, 1920, 1080, 1.0 / 90000.0, layout, error))
        << error;
    EXPECT_EQ(layout.width, 960);
    EXPECT_EQ(layout.height, 540);
    EXPECT_EQ(layout.rgb_bytes, 960u * 540u * 3u);
}

TEST(VideoFrameExtractorParams, RejectsInvalidRangesAndDimensions) {
    VideoFrameExtractor::ValidatedLayout layout;
    std::string error;

    auto params = validParams();
    EXPECT_FALSE(VideoFrameExtractor::validateParams(
        params, 0, 1080, 1.0 / 90000.0, layout, error));
    EXPECT_FALSE(VideoFrameExtractor::validateParams(
        params, 1920, 1080, 0.0, layout, error));

    params.start_time = 10.0;
    params.end_time = 10.0;
    EXPECT_FALSE(VideoFrameExtractor::validateParams(
        params, 1920, 1080, 1.0 / 90000.0, layout, error));

    params = validParams();
    params.rotation = 45;
    EXPECT_FALSE(VideoFrameExtractor::validateParams(
        params, 1920, 1080, 1.0 / 90000.0, layout, error));

    params = validParams();
    params.fps = std::numeric_limits<double>::max();
    EXPECT_FALSE(VideoFrameExtractor::validateParams(
        params, 1920, 1080, 1.0 / 90000.0, layout, error));

    params = validParams();
    params.end_time = 1.0e9;
    params.fps = 3.0;
    EXPECT_FALSE(VideoFrameExtractor::validateParams(
        params, 1920, 1080, 1.0 / 90000.0, layout, error));

    params = validParams();
    params.resolution_mode = ResolutionMode::Scale;
    params.scale = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(VideoFrameExtractor::validateParams(
        params, 1920, 1080, 1.0 / 90000.0, layout, error));

    params = validParams();
    params.resolution_mode = ResolutionMode::Custom;
    params.custom_width = std::numeric_limits<int>::max() / 2;
    params.custom_height = 1;
    EXPECT_FALSE(VideoFrameExtractor::validateParams(
        params, 1920, 1080, 1.0 / 90000.0, layout, error));
}
