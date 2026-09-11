/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/formats/colmap.hpp"

#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace {
    namespace fs = std::filesystem;

    template <class T>
    void append_pod(std::vector<char>& bytes, const T value) {
        const auto* begin = reinterpret_cast<const char*>(&value);
        bytes.insert(bytes.end(), begin, begin + sizeof(T));
    }

    void append_camera(std::vector<char>& bytes,
                       const uint32_t id,
                       const int32_t model_id,
                       const uint64_t width,
                       const uint64_t height,
                       const std::vector<double>& params) {
        append_pod(bytes, id);
        append_pod(bytes, model_id);
        append_pod(bytes, width);
        append_pod(bytes, height);
        for (const double value : params) {
            append_pod(bytes, value);
        }
    }

    void append_image(std::vector<char>& bytes,
                      const uint32_t id,
                      const double quaternion_w,
                      const uint32_t camera_id,
                      const std::string_view name,
                      const double point_x = 0.0,
                      const bool include_point = false) {
        append_pod(bytes, id);
        append_pod(bytes, quaternion_w);
        append_pod(bytes, 0.0);
        append_pod(bytes, 0.0);
        append_pod(bytes, 0.0);
        append_pod(bytes, 0.0);
        append_pod(bytes, 0.0);
        append_pod(bytes, 0.0);
        append_pod(bytes, camera_id);
        bytes.insert(bytes.end(), name.begin(), name.end());
        bytes.push_back('\0');
        append_pod(bytes, uint64_t{include_point ? 1u : 0u});
        if (include_point) {
            append_pod(bytes, point_x);
            append_pod(bytes, 2.0);
            append_pod(bytes, std::numeric_limits<uint64_t>::max());
        }
    }

    void append_point(std::vector<char>& bytes,
                      const uint64_t id,
                      const double x,
                      const uint8_t red = 10) {
        append_pod(bytes, id);
        append_pod(bytes, x);
        append_pod(bytes, 2.0);
        append_pod(bytes, 3.0);
        bytes.push_back(static_cast<char>(red));
        bytes.push_back(static_cast<char>(20));
        bytes.push_back(static_cast<char>(30));
        append_pod(bytes, 0.25);
        append_pod(bytes, uint64_t{0});
    }

    class ColmapBinaryErrorTaxonomyTest : public ::testing::Test {
    protected:
        void SetUp() override {
            temp_dir_ = fs::temp_directory_path() / "lfs_colmap_binary_error_taxonomy";
            std::error_code ec;
            fs::remove_all(temp_dir_, ec);
            ASSERT_TRUE(fs::create_directories(temp_dir_));
        }

        void TearDown() override {
            std::error_code ec;
            fs::remove_all(temp_dir_, ec);
        }

        void write_bytes(const fs::path& path, const std::vector<char>& bytes) {
            fs::create_directories(path.parent_path());
            std::ofstream stream(path, std::ios::binary | std::ios::trunc);
            ASSERT_TRUE(stream.is_open());
            stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
            ASSERT_TRUE(stream.good());
        }

        void write_text(const fs::path& path, const std::string_view text) {
            fs::create_directories(path.parent_path());
            std::ofstream stream(path, std::ios::binary | std::ios::trunc);
            ASSERT_TRUE(stream.is_open());
            stream << text;
            ASSERT_TRUE(stream.good());
        }

        void write_png(const fs::path& path) {
            static constexpr unsigned char png[] = {
                0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
                0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
                0x08, 0x06, 0x00, 0x00, 0x00, 0x1f, 0x15, 0xc4, 0x89, 0x00, 0x00, 0x00,
                0x0d, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0x00, 0x01, 0x00, 0x00,
                0x05, 0x00, 0x01, 0x0d, 0x0a, 0x2d, 0xb4, 0x00, 0x00, 0x00, 0x00, 0x49,
                0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82};
            fs::create_directories(path.parent_path());
            std::ofstream stream(path, std::ios::binary | std::ios::trunc);
            ASSERT_TRUE(stream.is_open());
            stream.write(reinterpret_cast<const char*>(png), sizeof(png));
            ASSERT_TRUE(stream.good());
        }

        std::vector<char> valid_camera_file() const {
            std::vector<char> bytes;
            append_pod(bytes, uint64_t{1});
            append_camera(bytes, 1, 1, 1, 1, {1.0, 1.0, 0.5, 0.5});
            return bytes;
        }

        std::vector<char> valid_image_file() const {
            std::vector<char> bytes;
            append_pod(bytes, uint64_t{1});
            append_image(bytes, 1, 1.0, 1, "valid.png");
            return bytes;
        }

        fs::path temp_dir_;
    };

    bool has_cuda_device() {
        int count = 0;
        return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
    }

    TEST_F(ColmapBinaryErrorTaxonomyTest, SkipsInvalidBinaryCameraAndPreservesValidIntrinsics) {
        std::vector<char> mixed_cameras;
        append_pod(mixed_cameras, uint64_t{2});
        append_camera(mixed_cameras, 1, 1, 1, 1, {1.0, 1.0, 0.5, 0.5});
        append_camera(mixed_cameras, 2, 1, 0, 1, {2.0, 2.0, 0.5, 0.5});
        write_bytes(temp_dir_ / "cameras.bin", mixed_cameras);
        write_bytes(temp_dir_ / "images.bin", valid_image_file());
        write_png(temp_dir_ / "images" / "valid.png");

        const auto result = lfs::io::read_colmap_cameras_and_images(temp_dir_);
        ASSERT_TRUE(result.has_value()) << result.error().format();
        const auto& cameras = std::get<0>(result->value);
        ASSERT_EQ(cameras.size(), 1u);
        EXPECT_FLOAT_EQ(cameras.front()->focal_x(), 1.0f);
        EXPECT_FLOAT_EQ(cameras.front()->center_x(), 0.5f);
        ASSERT_EQ(result->warnings.size(), 1u);
        EXPECT_EQ(result->warnings.front().code, lfs::ErrorCode::DataLoss);
        EXPECT_NE(result->warnings.front().message.find("1"), std::string::npos);

        const fs::path baseline = temp_dir_ / "baseline";
        write_bytes(baseline / "cameras.bin", valid_camera_file());
        write_bytes(baseline / "images.bin", valid_image_file());
        write_png(baseline / "images" / "valid.png");
        const auto baseline_result = lfs::io::read_colmap_cameras_and_images(baseline);
        ASSERT_TRUE(baseline_result.has_value()) << baseline_result.error().format();
        const auto& baseline_camera = std::get<0>(baseline_result->value).front();
        EXPECT_EQ(cameras.front()->camera_width(), baseline_camera->camera_width());
        EXPECT_EQ(cameras.front()->camera_height(), baseline_camera->camera_height());
        EXPECT_FLOAT_EQ(cameras.front()->focal_x(), baseline_camera->focal_x());
        EXPECT_FLOAT_EQ(cameras.front()->center_x(), baseline_camera->center_x());
        EXPECT_EQ(std::memcmp(cameras.front()->R().ptr<float>(), baseline_camera->R().ptr<float>(),
                              9 * sizeof(float)),
                  0);
        EXPECT_EQ(std::memcmp(cameras.front()->T().ptr<float>(), baseline_camera->T().ptr<float>(),
                              3 * sizeof(float)),
                  0);
    }

    TEST_F(ColmapBinaryErrorTaxonomyTest, SkippedImageLeavesNoCameraUidHoles) {
        std::vector<char> images;
        append_pod(images, uint64_t{2});
        append_image(images, 1, 1.0, 7, "orphan.png");
        append_image(images, 2, 1.0, 1, "valid.png");
        write_bytes(temp_dir_ / "cameras.bin", valid_camera_file());
        write_bytes(temp_dir_ / "images.bin", images);
        write_png(temp_dir_ / "images" / "orphan.png");
        write_png(temp_dir_ / "images" / "valid.png");

        const auto result = lfs::io::read_colmap_cameras_and_images(temp_dir_);
        ASSERT_TRUE(result.has_value()) << result.error().format();
        const auto& cameras = std::get<0>(result->value);
        ASSERT_EQ(cameras.size(), 1u);
        for (const auto& camera : cameras) {
            EXPECT_LT(static_cast<size_t>(camera->uid()), cameras.size());
        }
        ASSERT_EQ(result->warnings.size(), 1u);
        EXPECT_EQ(result->warnings.front().code, lfs::ErrorCode::DataLoss);
    }

    TEST_F(ColmapBinaryErrorTaxonomyTest, SkipsInvalidBinaryPoseAndPoint2DObservation) {
        std::vector<char> images;
        append_pod(images, uint64_t{2});
        append_image(images, 1, 1.0, 1, "valid.png", std::numeric_limits<double>::quiet_NaN(), true);
        append_image(images, 2, 2.0, 1, "invalid.png");
        write_bytes(temp_dir_ / "cameras.bin", valid_camera_file());
        write_bytes(temp_dir_ / "images.bin", images);
        write_png(temp_dir_ / "images" / "valid.png");

        const auto result = lfs::io::read_colmap_cameras_and_images(temp_dir_);
        ASSERT_TRUE(result.has_value()) << result.error().format();
        EXPECT_EQ(std::get<0>(result->value).size(), 1u);
        ASSERT_EQ(result->warnings.size(), 2u);
        EXPECT_EQ(result->warnings[0].code, lfs::ErrorCode::DataLoss);
        EXPECT_EQ(result->warnings[1].code, lfs::ErrorCode::DataLoss);
    }

    TEST_F(ColmapBinaryErrorTaxonomyTest, SkipsInvalidBinaryPointAndCapsSamples) {
        if (!has_cuda_device()) {
            GTEST_SKIP() << "CUDA device required";
        }
        std::vector<char> points;
        append_pod(points, uint64_t{11});
        append_point(points, 1, 1.0, 7);
        for (uint64_t id = 2; id <= 11; ++id) {
            append_point(points, id, std::numeric_limits<double>::quiet_NaN());
        }
        write_bytes(temp_dir_ / "points3D.bin", points);

        const auto result = lfs::io::read_colmap_point_cloud(temp_dir_);
        ASSERT_TRUE(result.has_value()) << result.error().format();
        EXPECT_EQ(result->value.size(), 1u);
        ASSERT_EQ(result->warnings.size(), 1u);
        EXPECT_NE(result->warnings.front().message.find("10"), std::string::npos);
        EXPECT_NE(result->warnings.front().message.find("8 sampled"), std::string::npos);
        const auto means = result->value.means.cpu();
        EXPECT_FLOAT_EQ(means.ptr<float>()[0], 1.0f);
        const auto colors = result->value.colors.cpu();
        EXPECT_EQ(colors.ptr<uint8_t>()[0], 7u);

        const fs::path baseline = temp_dir_ / "baseline";
        std::vector<char> baseline_points;
        append_pod(baseline_points, uint64_t{1});
        append_point(baseline_points, 1, 1.0, 7);
        write_bytes(baseline / "points3D.bin", baseline_points);
        const auto baseline_result = lfs::io::read_colmap_point_cloud(baseline);
        ASSERT_TRUE(baseline_result.has_value()) << baseline_result.error().format();
        const auto baseline_means = baseline_result->value.means.cpu();
        const auto baseline_colors = baseline_result->value.colors.cpu();
        EXPECT_EQ(std::memcmp(means.ptr<float>(), baseline_means.ptr<float>(), 3 * sizeof(float)), 0);
        EXPECT_EQ(std::memcmp(colors.ptr<uint8_t>(), baseline_colors.ptr<uint8_t>(), 3), 0);
    }

    TEST_F(ColmapBinaryErrorTaxonomyTest, RejectsUnknownBinaryCameraModel) {
        std::vector<char> cameras;
        append_pod(cameras, uint64_t{1});
        append_camera(cameras, 1, 999, 1, 1, {});
        write_bytes(temp_dir_ / "cameras.bin", cameras);
        write_bytes(temp_dir_ / "images.bin", valid_image_file());

        const auto result = lfs::io::read_colmap_cameras_only(temp_dir_);
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code, lfs::io::ErrorCode::UNSUPPORTED_FORMAT);
    }

    TEST_F(ColmapBinaryErrorTaxonomyTest, RejectsAllInvalidBinaryCameras) {
        std::vector<char> cameras;
        append_pod(cameras, uint64_t{1});
        append_camera(cameras, 1, 1, 0, 1, {1.0, 1.0, 0.5, 0.5});
        write_bytes(temp_dir_ / "cameras.bin", cameras);
        write_bytes(temp_dir_ / "images.bin", valid_image_file());

        const auto result = lfs::io::read_colmap_cameras_only(temp_dir_);
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code, lfs::io::ErrorCode::CORRUPTED_DATA);
    }

    TEST_F(ColmapBinaryErrorTaxonomyTest, RejectsAllInvalidBinaryImages) {
        std::vector<char> images;
        append_pod(images, uint64_t{1});
        append_image(images, 1, 2.0, 1, "invalid.png");
        write_bytes(temp_dir_ / "cameras.bin", valid_camera_file());
        write_bytes(temp_dir_ / "images.bin", images);

        const auto result = lfs::io::read_colmap_cameras_only(temp_dir_);
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code, lfs::io::ErrorCode::CORRUPTED_DATA);
    }

    TEST_F(ColmapBinaryErrorTaxonomyTest, AllInvalidBinaryPointsSucceedEmptyWithWarning) {
        std::vector<char> points;
        append_pod(points, uint64_t{1});
        append_point(points, 1, std::numeric_limits<double>::quiet_NaN());
        write_bytes(temp_dir_ / "points3D.bin", points);

        const auto result = lfs::io::read_colmap_point_cloud_with_stats(temp_dir_);
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(result->value.total_points, 0u);
        ASSERT_EQ(result->warnings.size(), 1u);
    }

    TEST_F(ColmapBinaryErrorTaxonomyTest, RejectsAllInvalidTextCamerasAndImages) {
        write_text(temp_dir_ / "cameras.txt", "bad camera line\n");
        write_text(temp_dir_ / "images.txt", "1 1 0 0 0 0 0 0 1 valid.png\n");
        auto result = lfs::io::read_colmap_cameras_only(temp_dir_);
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code, lfs::io::ErrorCode::CORRUPTED_DATA);

        write_text(temp_dir_ / "cameras.txt", "1 PINHOLE 1 1 1 1 0.5 0.5\n");
        write_text(temp_dir_ / "images.txt", "1 2 0 0 0 0 0 0 1 invalid.png\n");
        result = lfs::io::read_colmap_cameras_only(temp_dir_);
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code, lfs::io::ErrorCode::CORRUPTED_DATA);
    }

    TEST_F(ColmapBinaryErrorTaxonomyTest, SeededShortReadSweepReturnsOnlyResults) {
        constexpr uint32_t seed = 0x3b202607u;
        std::vector<char> valid_empty_points;
        append_pod(valid_empty_points, uint64_t{0});
        std::vector<size_t> lengths(valid_empty_points.size() + 1);
        std::iota(lengths.begin(), lengths.end(), size_t{0});
        std::mt19937 generator(seed);
        std::shuffle(lengths.begin(), lengths.end(), generator);

        for (const size_t length : lengths) {
            write_bytes(temp_dir_ / "points3D.bin",
                        std::vector<char>(valid_empty_points.begin(), valid_empty_points.begin() + length));
            const auto result = lfs::io::read_colmap_point_cloud(temp_dir_);
            if (length == valid_empty_points.size()) {
                ASSERT_TRUE(result.has_value()) << "length=" << length;
                EXPECT_EQ(result->value.size(), 0u);
            } else {
                ASSERT_FALSE(result.has_value()) << "length=" << length;
                EXPECT_EQ(result.error().code, lfs::io::ErrorCode::CORRUPTED_DATA);
            }
        }
    }
} // namespace
