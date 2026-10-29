/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/formats/colmap.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <variant>
#include <vector>

namespace {
    namespace fs = std::filesystem;

    // A 1x1 RGBA PNG.
    constexpr unsigned char kPng1x1[] = {
        0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
        0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x06, 0x00, 0x00, 0x00, 0x1f, 0x15, 0xc4, 0x89, 0x00, 0x00, 0x00,
        0x0d, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0x00, 0x01, 0x00, 0x00,
        0x05, 0x00, 0x01, 0x0d, 0x0a, 0x2d, 0xb4, 0x00, 0x00, 0x00, 0x00, 0x49,
        0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82};

    class ColmapTextImagesTest : public ::testing::Test {
    protected:
        void SetUp() override {
            dataset_ = fs::temp_directory_path() / "lfs_colmap_text_images_test";
            std::error_code ec;
            fs::remove_all(dataset_, ec);
            fs::create_directories(dataset_ / "images");
        }

        void TearDown() override {
            std::error_code ec;
            fs::remove_all(dataset_, ec);
        }

        // One 1x1 camera, the given images.txt, and a PNG for each named image.
        void write_dataset(const std::string& images_txt, const std::vector<std::string>& image_files) {
            write_file(dataset_ / "cameras.txt", "1 PINHOLE 1 1 1 1 0.5 0.5\n");
            write_file(dataset_ / "images.txt", images_txt);
            for (const auto& name : image_files) {
                std::ofstream png(dataset_ / "images" / name, std::ios::binary | std::ios::trunc);
                png.write(reinterpret_cast<const char*>(kPng1x1), sizeof(kPng1x1));
                ASSERT_TRUE(png.good()) << name;
            }
        }

        static void write_file(const fs::path& path, const std::string& text) {
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            out << text;
            ASSERT_TRUE(out.good()) << path;
        }

        fs::path dataset_;
    };

    // The image count of the "skipped ... image(s) with an invalid pose" warning; 0 without one.
    std::int64_t skipped_images(const std::vector<lfs::io::Diagnostic>& warnings) {
        for (const auto& warning : warnings) {
            if (warning.message.find("invalid pose") == std::string::npos)
                continue;
            for (const auto& field : warning.fields.entries())
                if (field.key == "skipped_count")
                    return std::get<std::int64_t>(field.value);
        }
        return 0;
    }
} // namespace

// POINTS2D lines as COLMAP writes them. Their first X has a fraction, so integer
// reads of IMAGE_ID and CAMERA_ID stop inside a number instead of failing, and each
// line used to count as one more image with an invalid pose.
TEST_F(ColmapTextImagesTest, Points2DLinesAreNotCountedAsSkippedImages) {
    write_dataset("# Image list with two lines of data per image:\n"
                  "1 1 0 0 0 0 0 0 1 a.png\n"
                  "2353.88 48.8696 0 3755.43 265.458 1 2618.74 327.631 2 1924.22 535.631 3\n"
                  "2 1 0 0 0 0 0 1 1 b.png\n"
                  "2303.19 253.097 0 3870.18 518.035 1 2592.94 618.849 2 1780.14 913.333 3\n",
                  {"a.png", "b.png"});

    const auto result = lfs::io::read_colmap_cameras_and_images_text(dataset_, "images");
    ASSERT_TRUE(result.has_value()) << result.error().format();
    EXPECT_EQ(std::get<0>(result->value).size(), 2u);
    EXPECT_EQ(skipped_images(result->warnings), 0);
}

// An image with an invalid pose is skipped and counted once, together with its POINTS2D line.
TEST_F(ColmapTextImagesTest, SkippedImageIsCountedOnce) {
    write_dataset("1 2 0 0 0 0 0 0 1 bad.png\n"
                  "1000.5 20.25 -1 1100.5 30.75 -1 1200.5 40.5 -1 1300.25 50.5 -1\n"
                  "2 1 0 0 0 0 0 0 1 good.png\n"
                  "1000.5 20.25 -1 1100.5 30.75 -1 1200.5 40.5 -1 1300.25 50.5 -1\n",
                  {"good.png"});

    const auto result = lfs::io::read_colmap_cameras_and_images_text(dataset_, "images");
    ASSERT_TRUE(result.has_value()) << result.error().format();
    const auto& cameras = std::get<0>(result->value);
    ASSERT_EQ(cameras.size(), 1u);
    EXPECT_EQ(cameras[0]->image_name(), "good.png");
    EXPECT_EQ(skipped_images(result->warnings), 1);
}

// Writers that leave out the POINTS2D lines still give every image.
TEST_F(ColmapTextImagesTest, ImagesWithoutPoints2DLinesAllLoad) {
    write_dataset("1 1 0 0 0 0 0 0 1 a.png\n"
                  "2 1 0 0 0 0 0 1 1 b.png\n"
                  "3 1 0 0 0 0 0 2 1 c.png\n",
                  {"a.png", "b.png", "c.png"});

    const auto result = lfs::io::read_colmap_cameras_and_images_text(dataset_, "images");
    ASSERT_TRUE(result.has_value()) << result.error().format();
    EXPECT_EQ(std::get<0>(result->value).size(), 3u);
    EXPECT_EQ(skipped_images(result->warnings), 0);
}
