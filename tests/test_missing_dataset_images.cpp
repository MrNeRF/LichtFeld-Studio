/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/formats/colmap.hpp"
#include "io/formats/transforms.hpp"
#include "io/loaders/blender_loader.hpp"
#include "io/loaders/colmap_loader.hpp"
#include "training/dataset.hpp"

#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

    bool has_cuda_device() {
        int device_count = 0;
        return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
    }

    class MissingDatasetImagesTest : public ::testing::Test {
    protected:
        void SetUp() override {
            temp_dir_ = fs::temp_directory_path() / "lfs_missing_dataset_images_test";
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

        void write_bytes(const fs::path& path, const std::vector<char>& bytes) {
            fs::create_directories(path.parent_path());
            std::ofstream out(path, std::ios::binary);
            ASSERT_TRUE(out.is_open()) << "Failed to open " << path;
            out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
            out.close();
            ASSERT_TRUE(out.good()) << "Failed to write " << path;
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
            std::ofstream out(path, std::ios::binary);
            ASSERT_TRUE(out.is_open()) << "Failed to open " << path;
            out.write(reinterpret_cast<const char*>(png), sizeof(png));
            out.close();
            ASSERT_TRUE(out.good()) << "Failed to write " << path;
        }

        void write_colmap_text(const fs::path& dataset_dir, const std::vector<std::string>& image_names) {
            write_text_file(dataset_dir / "cameras.txt", "1 PINHOLE 1 1 1 1 0.5 0.5\n");
            std::ostringstream images;
            for (size_t i = 0; i < image_names.size(); ++i) {
                images << (i + 1) << " 1 0 0 0 0 0 0 1 " << image_names[i] << "\n";
            }
            write_text_file(dataset_dir / "images.txt", images.str());
            fs::create_directories(dataset_dir / "images");
        }

        template <class T>
        void append_pod(std::vector<char>& bytes, const T value) {
            const auto* begin = reinterpret_cast<const char*>(&value);
            bytes.insert(bytes.end(), begin, begin + sizeof(T));
        }

        void write_colmap_binary(const fs::path& dataset_dir, const std::vector<std::string>& image_names) {
            std::vector<char> cameras;
            append_pod(cameras, uint64_t{1});
            append_pod(cameras, uint32_t{1});
            append_pod(cameras, int32_t{1});
            append_pod(cameras, uint64_t{1});
            append_pod(cameras, uint64_t{1});
            append_pod(cameras, 1.0);
            append_pod(cameras, 1.0);
            append_pod(cameras, 0.5);
            append_pod(cameras, 0.5);
            write_bytes(dataset_dir / "cameras.bin", cameras);

            std::vector<char> images;
            append_pod(images, static_cast<uint64_t>(image_names.size()));
            for (size_t i = 0; i < image_names.size(); ++i) {
                append_pod(images, static_cast<uint32_t>(i + 1));
                append_pod(images, 1.0);
                append_pod(images, 0.0);
                append_pod(images, 0.0);
                append_pod(images, 0.0);
                append_pod(images, 0.0);
                append_pod(images, 0.0);
                append_pod(images, 0.0);
                append_pod(images, uint32_t{1});
                images.insert(images.end(), image_names[i].begin(), image_names[i].end());
                images.push_back('\0');
                append_pod(images, uint64_t{0});
            }
            write_bytes(dataset_dir / "images.bin", images);
            fs::create_directories(dataset_dir / "images");
        }

        void write_transforms(const fs::path& dataset_dir, const std::vector<std::string>& image_names) {
            const nlohmann::json identity = {
                {1.0, 0.0, 0.0, 0.0},
                {0.0, 1.0, 0.0, 0.0},
                {0.0, 0.0, 1.0, 0.0},
                {0.0, 0.0, 0.0, 1.0},
            };
            nlohmann::json frames = nlohmann::json::array();
            for (const auto& name : image_names) {
                frames.push_back({
                    {"file_path", name},
                    {"transform_matrix", identity},
                });
            }
            nlohmann::json transforms = {
                {"w", 1},
                {"h", 1},
                {"fl_x", 1.0},
                {"fl_y", 1.0},
                {"cx", 0.5},
                {"cy", 0.5},
                {"frames", frames},
            };
            write_text_file(dataset_dir / "transforms.json", transforms.dump());
        }

        void expect_partial_missing(
            const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras,
            const std::vector<std::string>& warnings,
            const std::string& missing_name,
            const std::string& present_name) {
            ASSERT_EQ(cameras.size(), 2u);
            std::vector<std::string> missing;
            std::vector<std::string> present;
            for (const auto& camera : cameras) {
                if (camera->has_image()) {
                    present.push_back(camera->image_name());
                } else {
                    missing.push_back(camera->image_name());
                }
            }
            ASSERT_EQ(missing.size(), 1u);
            EXPECT_EQ(missing.front(), missing_name);
            ASSERT_EQ(present.size(), 1u);
            EXPECT_EQ(present.front(), present_name);

            ASSERT_FALSE(warnings.empty());
            bool found_warning = false;
            for (const auto& warning : warnings) {
                if (warning.find(missing_name) != std::string::npos &&
                    warning.find("missing") != std::string::npos) {
                    found_warning = true;
                }
                EXPECT_EQ(warning.find(present_name), std::string::npos);
            }
            EXPECT_TRUE(found_warning);

            lfs::training::DatasetConfig config;
            lfs::training::CameraDataset dataset(cameras, config, lfs::training::CameraDataset::Split::ALL);
            EXPECT_EQ(dataset.size(), 1u);
            EXPECT_EQ(dataset.get_camera(0)->image_name(), present_name);
            EXPECT_EQ(dataset.get_cameras().size(), 2u);
        }

        fs::path temp_dir_;
    };

} // namespace

TEST_F(MissingDatasetImagesTest, ColmapTextLoadsWhenOneImageIsMissing) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }

    const fs::path dataset_dir = temp_dir_ / "colmap_text";
    write_colmap_text(dataset_dir, {"present.png", "missing.png"});
    write_png(dataset_dir / "images" / "present.png");

    auto result = lfs::io::read_colmap_cameras_and_images_text(dataset_dir, "images");
    ASSERT_TRUE(result.has_value()) << result.error().format();

    const auto& cameras = std::get<0>(result->value);
    std::vector<std::string> warnings;
    for (const auto& diagnostic : result->warnings) {
        warnings.push_back(diagnostic.message);
        EXPECT_EQ(diagnostic.code, lfs::ErrorCode::NotFound);
    }
    ASSERT_NO_FATAL_FAILURE(expect_partial_missing(cameras, warnings, "missing.png", "present.png"));
}

TEST_F(MissingDatasetImagesTest, ColmapTextAllMissingFails) {
    const fs::path dataset_dir = temp_dir_ / "colmap_text_all_missing";
    write_colmap_text(dataset_dir, {"missing_a.png", "missing_b.png"});

    auto result = lfs::io::read_colmap_cameras_and_images_text(dataset_dir, "images");
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, lfs::io::ErrorCode::EMPTY_DATASET);
    EXPECT_NE(result.error().message.find("All 2 dataset image files are missing"), std::string::npos);
    EXPECT_NE(result.error().message.find("missing_a.png"), std::string::npos);
}

TEST_F(MissingDatasetImagesTest, ColmapTextValidationAllowsPartialMissing) {
    const fs::path dataset_dir = temp_dir_ / "colmap_text_validate";
    write_colmap_text(dataset_dir, {"present.png", "missing.png"});
    write_png(dataset_dir / "images" / "present.png");

    auto result = lfs::io::validate_colmap_dataset_layout(dataset_dir, "images");
    ASSERT_TRUE(result.has_value()) << result.error().format();
}

TEST_F(MissingDatasetImagesTest, ColmapTextStillFailsWhenImagesFolderIsMissing) {
    const fs::path dataset_dir = temp_dir_ / "colmap_text_no_folder";
    write_text_file(dataset_dir / "cameras.txt", "1 PINHOLE 1 1 1 1 0.5 0.5\n");
    write_text_file(dataset_dir / "images.txt", "1 1 0 0 0 0 0 0 1 missing.png\n");

    auto result = lfs::io::validate_colmap_dataset_layout(dataset_dir, "images");
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, lfs::io::ErrorCode::PATH_NOT_FOUND);
}

TEST_F(MissingDatasetImagesTest, ColmapTextResolvesMissingAgainstLoadResolutionFolder) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }

    const fs::path dataset_dir = temp_dir_ / "colmap_text_images_2";
    write_colmap_text(dataset_dir, {"present.png", "missing.png"});
    // images_2 applies scale_factor 2; 1x1 cameras become 0x0 and are skipped.
    write_text_file(dataset_dir / "cameras.txt", "1 PINHOLE 2 2 1 1 0.5 0.5\n");
    write_png(dataset_dir / "images" / "present.png");
    write_png(dataset_dir / "images" / "missing.png");
    write_png(dataset_dir / "images_2" / "present.png");

    auto full_result = lfs::io::read_colmap_cameras_and_images_text(dataset_dir, "images");
    ASSERT_TRUE(full_result.has_value()) << full_result.error().format();
    EXPECT_TRUE(full_result->warnings.empty());

    auto scaled_result = lfs::io::read_colmap_cameras_and_images_text(dataset_dir, "images_2");
    ASSERT_TRUE(scaled_result.has_value()) << scaled_result.error().format();
    const auto& cameras = std::get<0>(scaled_result->value);
    std::vector<std::string> warnings;
    for (const auto& diagnostic : scaled_result->warnings) {
        warnings.push_back(diagnostic.message);
    }
    ASSERT_NO_FATAL_FAILURE(expect_partial_missing(cameras, warnings, "missing.png", "present.png"));
}

TEST_F(MissingDatasetImagesTest, ColmapBinaryLoadsWhenOneImageIsMissing) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }

    const fs::path dataset_dir = temp_dir_ / "colmap_bin";
    write_colmap_binary(dataset_dir, {"present.png", "missing.png"});
    write_png(dataset_dir / "images" / "present.png");

    auto result = lfs::io::read_colmap_cameras_and_images(dataset_dir, "images");
    ASSERT_TRUE(result.has_value()) << result.error().format();

    const auto& cameras = std::get<0>(result->value);
    std::vector<std::string> warnings;
    for (const auto& diagnostic : result->warnings) {
        warnings.push_back(diagnostic.message);
        EXPECT_EQ(diagnostic.code, lfs::ErrorCode::NotFound);
    }
    ASSERT_NO_FATAL_FAILURE(expect_partial_missing(cameras, warnings, "missing.png", "present.png"));
}

TEST_F(MissingDatasetImagesTest, ColmapBinaryAllMissingFails) {
    const fs::path dataset_dir = temp_dir_ / "colmap_bin_all_missing";
    write_colmap_binary(dataset_dir, {"missing_a.png", "missing_b.png"});

    auto result = lfs::io::read_colmap_cameras_and_images(dataset_dir, "images");
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, lfs::io::ErrorCode::EMPTY_DATASET);
    EXPECT_NE(result.error().message.find("All 2 dataset image files are missing"), std::string::npos);
}

TEST_F(MissingDatasetImagesTest, ColmapLoaderReportsMissingSet) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }

    const fs::path dataset_dir = temp_dir_ / "colmap_loader";
    write_colmap_text(dataset_dir, {"present.png", "missing.png"});
    write_png(dataset_dir / "images" / "present.png");

    lfs::io::ColmapLoader loader;
    auto result = loader.load(dataset_dir, {});
    ASSERT_TRUE(result.has_value()) << result.error().format();
    auto* scene = std::get_if<lfs::io::LoadedScene>(&result->data);
    ASSERT_NE(scene, nullptr);
    ASSERT_NO_FATAL_FAILURE(expect_partial_missing(scene->cameras, result->warnings, "missing.png", "present.png"));
}

TEST_F(MissingDatasetImagesTest, TransformsLoadsWhenOneImageIsMissing) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }

    const fs::path dataset_dir = temp_dir_ / "transforms";
    write_transforms(dataset_dir, {"present.png", "missing.png"});
    write_png(dataset_dir / "present.png");

    lfs::io::BlenderLoader loader;
    auto result = loader.load(dataset_dir, {});
    ASSERT_TRUE(result.has_value()) << result.error().format();
    auto* scene = std::get_if<lfs::io::LoadedScene>(&result->data);
    ASSERT_NE(scene, nullptr);
    ASSERT_NO_FATAL_FAILURE(expect_partial_missing(scene->cameras, result->warnings, "missing.png", "present.png"));
}

TEST_F(MissingDatasetImagesTest, TransformsAllMissingFails) {
    const fs::path dataset_dir = temp_dir_ / "transforms_all_missing";
    write_transforms(dataset_dir, {"missing_a.png", "missing_b.png"});

    lfs::io::BlenderLoader loader;
    auto result = loader.load(dataset_dir, {});
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, lfs::io::ErrorCode::EMPTY_DATASET);
    EXPECT_NE(result.error().message.find("All 2 dataset image files are missing"), std::string::npos);
}

TEST_F(MissingDatasetImagesTest, TransformsValidateOnlyAllMissingFails) {
    const fs::path dataset_dir = temp_dir_ / "transforms_validate_all_missing";
    write_transforms(dataset_dir, {"missing_a.png", "missing_b.png"});

    lfs::io::BlenderLoader loader;
    auto result = loader.load(dataset_dir, {.validate_only = true});
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, lfs::io::ErrorCode::EMPTY_DATASET);
}

TEST_F(MissingDatasetImagesTest, TransformsRawReaderKeepsMissingCameraRecords) {
    const fs::path dataset_dir = temp_dir_ / "transforms_raw";
    write_transforms(dataset_dir, {"present.png", "missing.png"});
    write_png(dataset_dir / "present.png");

    auto [cameras, center, splits] =
        lfs::io::read_transforms_cameras_and_images(dataset_dir / "transforms.json", {});
    (void)center;
    (void)splits;
    ASSERT_EQ(cameras.size(), 2u);
    EXPECT_TRUE(cameras[0]._has_image);
    EXPECT_FALSE(cameras[1]._has_image);
    EXPECT_EQ(cameras[1]._image_name, "missing.png");
}
