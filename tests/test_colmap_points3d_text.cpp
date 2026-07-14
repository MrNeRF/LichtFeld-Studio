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

    const auto point_cloud = lfs::io::read_colmap_point_cloud_text(fixture_dir("basic"));
    EXPECT_EQ(point_cloud.size(), 3u);
}

TEST(ColmapPoints3DText, ReportsStatsAndFiltersByMinimumTrackLength) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required for COLMAP point cloud load";
    }

    const auto result = lfs::io::read_colmap_point_cloud_text_with_stats(
        fixture_dir("filter"),
        lfs::io::LoadOptions{.min_track_length = 3});

    EXPECT_TRUE(result.track_filter_applied);
    EXPECT_EQ(result.total_points, 3u);
    EXPECT_EQ(result.points_after_filtering, 1u);
    EXPECT_EQ(result.point_cloud.size(), 1u);
}

TEST(ColmapPoints3DText, RejectsDanglingOddTrackTokenWhenFiltering) {
    EXPECT_THROW(
        (void)lfs::io::read_colmap_point_cloud_text_with_stats(
            fixture_dir("dangling_track_token"),
            lfs::io::LoadOptions{.min_track_length = 2}),
        std::runtime_error);
}

TEST(ColmapPoints3DText, LoadsSinglePointStatsThroughPublicApi) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required for COLMAP point cloud load";
    }

    const auto result = lfs::io::read_colmap_point_cloud_text_with_stats(fixture_dir("single_point"));

    EXPECT_FALSE(result.track_filter_applied);
    EXPECT_EQ(result.total_points, 1u);
    EXPECT_EQ(result.points_after_filtering, 1u);
    EXPECT_EQ(result.point_cloud.size(), 1u);
}

TEST(ColmapPoints3DText, ThrowsOnEmptyOrCommentOnlyFileThroughPublicApi) {
    EXPECT_THROW(
        (void)lfs::io::read_colmap_point_cloud_text_with_stats(fixture_dir("empty_comment_only")),
        std::runtime_error);
}

TEST(ColmapPoints3DText, ThrowsOnMalformedLineThroughPublicApi) {
    EXPECT_THROW(
        (void)lfs::io::read_colmap_point_cloud_text_with_stats(fixture_dir("malformed")),
        std::runtime_error);
}
