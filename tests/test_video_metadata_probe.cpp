/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/video/video_metadata.hpp"

#include <filesystem>
#include <gtest/gtest.h>
#include <string>

namespace {

    std::filesystem::path fixture(const char* name) {
        return std::filesystem::path(PROJECT_ROOT_PATH) / "tests" / "data" / "video" / name;
    }

    TEST(VideoMetadataProbe, ReportsDimensionsAndNoSphericalTag) {
        lfs::io::video::VideoMetadata metadata;
        std::string error;
        ASSERT_TRUE(lfs::io::video::probeVideoFileMetadata(fixture("h264_420.mp4"), metadata, error)) << error;
        EXPECT_EQ(metadata.width, 64);
        EXPECT_EQ(metadata.height, 48);
        EXPECT_FALSE(metadata.spherical);
        EXPECT_EQ(metadata.rotation, 0);
        EXPECT_GT(metadata.fps, 0.0);
    }

    TEST(VideoMetadataProbe, ReportsSquareWebmDimensions) {
        lfs::io::video::VideoMetadata metadata;
        std::string error;
        ASSERT_TRUE(lfs::io::video::probeVideoFileMetadata(fixture("vp9_timestamps.webm"), metadata, error)) << error;
        EXPECT_EQ(metadata.width, 128);
        EXPECT_EQ(metadata.height, 128);
        EXPECT_FALSE(metadata.spherical);
    }

    TEST(VideoMetadataProbe, MissingFileFails) {
        lfs::io::video::VideoMetadata metadata;
        std::string error;
        EXPECT_FALSE(lfs::io::video::probeVideoFileMetadata(fixture("does_not_exist.mp4"), metadata, error));
        EXPECT_FALSE(error.empty());
        EXPECT_EQ(metadata.width, 0);
        EXPECT_EQ(metadata.height, 0);
    }

} // namespace
