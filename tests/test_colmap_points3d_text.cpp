/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/formats/colmap.hpp"
#include <cuda_runtime.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <stdexcept>

namespace fs = std::filesystem;

namespace {
    fs::path fixture_dir(const std::string& name) {
        return fs::path(PROJECT_ROOT_PATH) / "tests" / "fixtures" / "colmap_points3d_text" / name;
    }

    bool has_cuda_device() {
        int device_count = 0;
        return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
    }
} // namespace

TEST(ColmapPoints3DText, LoadsTextPointCloudThroughPublicApi) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required for COLMAP point cloud load";
    }

    const auto result = lfs::io::read_colmap_point_cloud_text(fixture_dir("basic"));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->value.size(), 3u);
}

TEST(ColmapPoints3DText, ReportsStatsAndFiltersByMinimumTrackLength) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required for COLMAP point cloud load";
    }

    const auto result = lfs::io::read_colmap_point_cloud_text_with_stats(
        fixture_dir("filter"),
        lfs::io::LoadOptions{.min_track_length = 3});

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->value.track_filter_applied);
    EXPECT_EQ(result->value.total_points, 3u);
    EXPECT_EQ(result->value.points_after_filtering, 1u);
    EXPECT_EQ(result->value.point_cloud.size(), 1u);
}

TEST(ColmapPoints3DText, SkipsDanglingOddTrackTokenWhenFiltering) {
    const auto result = lfs::io::read_colmap_point_cloud_text_with_stats(
        fixture_dir("dangling_track_token"),
        lfs::io::LoadOptions{.min_track_length = 2});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->value.total_points, 1u);
    EXPECT_EQ(result->value.points_after_filtering, 1u);
    EXPECT_EQ(result->value.point_cloud.size(), 1u);
    ASSERT_EQ(result->warnings.size(), 1u);
    EXPECT_EQ(result->warnings.front().code, lfs::ErrorCode::DataLoss);
}

TEST(ColmapPoints3DText, LoadsSinglePointStatsThroughPublicApi) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required for COLMAP point cloud load";
    }

    const auto result = lfs::io::read_colmap_point_cloud_text_with_stats(fixture_dir("single_point"));

    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(result->value.track_filter_applied);
    EXPECT_EQ(result->value.total_points, 1u);
    EXPECT_EQ(result->value.points_after_filtering, 1u);
    EXPECT_EQ(result->value.point_cloud.size(), 1u);
}

TEST(ColmapPoints3DText, SucceedsEmptyOnCommentOnlyFileThroughPublicApi) {
    const auto result = lfs::io::read_colmap_point_cloud_text_with_stats(fixture_dir("empty_comment_only"));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->value.total_points, 0u);
    EXPECT_EQ(result->value.point_cloud.size(), 0u);
    EXPECT_TRUE(result->warnings.empty());

    const auto fast_result = lfs::io::read_colmap_point_cloud_text(fixture_dir("empty_comment_only"));
    ASSERT_TRUE(fast_result.has_value());
    EXPECT_EQ(fast_result->value.size(), 0u);
    EXPECT_TRUE(fast_result->warnings.empty());
}

TEST(ColmapPoints3DText, SkipsMalformedLineThroughPublicApi) {
    const auto result = lfs::io::read_colmap_point_cloud_text_with_stats(fixture_dir("malformed"));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->value.total_points, 0u);
    EXPECT_EQ(result->value.point_cloud.size(), 0u);
    ASSERT_EQ(result->warnings.size(), 1u);
    EXPECT_EQ(result->warnings.front().code, lfs::ErrorCode::DataLoss);
}
