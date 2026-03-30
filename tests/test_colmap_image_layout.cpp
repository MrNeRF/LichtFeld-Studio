/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/formats/colmap.hpp"

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

namespace {

    class ColmapImageLayoutTest : public ::testing::Test {
    protected:
        void SetUp() override {
            temp_dir_ = fs::temp_directory_path() / "lfs_colmap_image_layout_test";
            std::error_code ec;
            fs::remove_all(temp_dir_, ec);
            fs::create_directories(temp_dir_);
        }

        void TearDown() override {
            std::error_code ec;
            fs::remove_all(temp_dir_, ec);
        }

        void write_text_file(const fs::path& path, const std::string& contents) {
            fs::create_directories(path.parent_path());
            std::ofstream out(path, std::ios::binary);
            ASSERT_TRUE(out.is_open()) << "Failed to open " << path;
            out << contents;
            out.close();
            ASSERT_TRUE(out.good()) << "Failed to write " << path;
        }

        void write_png(const fs::path& path) {
            static const std::vector<unsigned char> PNG_1X1 = {
                0x89,
                0x50,
                0x4E,
                0x47,
                0x0D,
                0x0A,
                0x1A,
                0x0A,
                0x00,
                0x00,
                0x00,
                0x0D,
                0x49,
                0x48,
                0x44,
                0x52,
                0x00,
                0x00,
                0x00,
                0x01,
                0x00,
                0x00,
                0x00,
                0x01,
                0x08,
                0x06,
                0x00,
                0x00,
                0x00,
                0x1F,
                0x15,
                0xC4,
                0x89,
                0x00,
                0x00,
                0x00,
                0x0D,
                0x49,
                0x44,
                0x41,
                0x54,
                0x78,
                0x9C,
                0x63,
                0x00,
                0x01,
                0x00,
                0x00,
                0x05,
                0x00,
                0x01,
                0x0D,
                0x0A,
                0x2D,
                0xB4,
                0x00,
                0x00,
                0x00,
                0x00,
                0x49,
                0x45,
                0x4E,
                0x44,
                0xAE,
                0x42,
                0x60,
                0x82,
            };

            fs::create_directories(path.parent_path());
            std::ofstream out(path, std::ios::binary);
            ASSERT_TRUE(out.is_open()) << "Failed to open " << path;
            out.write(reinterpret_cast<const char*>(PNG_1X1.data()),
                      static_cast<std::streamsize>(PNG_1X1.size()));
            out.close();
            ASSERT_TRUE(out.good()) << "Failed to write " << path;
        }

        void write_minimal_colmap_text_dataset(const fs::path& dataset_dir,
                                               const std::string& image_name) {
            write_text_file(dataset_dir / "cameras.txt",
                            "1 PINHOLE 1 1 1 1 0.5 0.5\n");
            write_text_file(dataset_dir / "images.txt",
                            "1 1 0 0 0 0 0 0 1 " + image_name + "\n");
        }

        fs::path temp_dir_;
    };

} // namespace

TEST_F(ColmapImageLayoutTest, ResolvesNestedImagesByBasename) {
    const fs::path dataset_dir = temp_dir_ / "dataset";
    const fs::path nested_image =
        dataset_dir / "images" / "Photogrammetry Sekal pipes" / "frame_0001.png";

    write_minimal_colmap_text_dataset(dataset_dir, "frame_0001.png");
    write_png(nested_image);

    auto result =
        lfs::io::read_colmap_cameras_and_images_text(dataset_dir, "images");
    ASSERT_TRUE(result.has_value()) << result.error().format();

    auto& [cameras, scene_center] = *result;
    (void)scene_center;

    ASSERT_EQ(cameras.size(), 1u);
    EXPECT_EQ(cameras[0]->image_name(), "frame_0001.png");
    EXPECT_TRUE(fs::equivalent(cameras[0]->image_path(), nested_image));
}

TEST_F(ColmapImageLayoutTest, FailsEarlyWhenReferencedImageIsMissing) {
    const fs::path dataset_dir = temp_dir_ / "dataset";

    write_minimal_colmap_text_dataset(dataset_dir, "missing_frame.png");

    auto result =
        lfs::io::read_colmap_cameras_and_images_text(dataset_dir, "images");
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, lfs::io::ErrorCode::PATH_NOT_FOUND);
}
