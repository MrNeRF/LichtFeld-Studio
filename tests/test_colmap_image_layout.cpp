/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/filesystem_utils.hpp"
#include "io/formats/colmap.hpp"
#include "io/loaders/colmap_loader.hpp"

#include <atomic>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <sstream>
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
                                               const std::vector<std::string>& image_names) {
            write_text_file(dataset_dir / "cameras.txt",
                            "1 PINHOLE 1 1 1 1 0.5 0.5\n");

            std::ostringstream images;
            for (size_t i = 0; i < image_names.size(); ++i) {
                images << (i + 1) << " 1 0 0 0 0 0 0 1 " << image_names[i] << "\n";
            }
            write_text_file(dataset_dir / "images.txt", images.str());
        }

        void write_minimal_colmap_text_dataset(const fs::path& dataset_dir,
                                               const std::string& image_name) {
            write_minimal_colmap_text_dataset(dataset_dir, std::vector<std::string>{image_name});
        }

        void write_ascii_double_ply(const fs::path& path, const size_t vertex_count) {
            fs::create_directories(path.parent_path());
            std::ofstream out(path, std::ios::binary);
            ASSERT_TRUE(out.is_open()) << "Failed to open " << path;
            out << "ply\n";
            out << "format ascii 1.0\n";
            out << "element vertex " << vertex_count << "\n";
            out << "property double x\n";
            out << "property double y\n";
            out << "property double z\n";
            out << "end_header\n";
            for (size_t i = 0; i < vertex_count; ++i) {
                out << static_cast<double>(i) << ' '
                    << static_cast<double>(i) * 0.5 << ' '
                    << static_cast<double>(i) * -0.25 << '\n';
            }
            out.close();
            ASSERT_TRUE(out.good()) << "Failed to write " << path;
        }

        fs::path temp_dir_;
    };

    bool has_cuda_device() {
        int device_count = 0;
        return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
    }

} // namespace

TEST_F(ColmapImageLayoutTest, ResolvesNestedImagesByBasename) {
    const fs::path dataset_dir = temp_dir_ / "dataset";
    const fs::path nested_image =
        dataset_dir / "images" / "Photogrammetry Sekal pipes" / "frame_0001.png";
    const fs::path nested_mask =
        dataset_dir / "masks" / "Photogrammetry Sekal pipes" / "frame_0001.png";

    write_minimal_colmap_text_dataset(dataset_dir, "frame_0001.png");
    write_png(nested_image);
    write_png(nested_mask);

    auto result =
        lfs::io::read_colmap_cameras_and_images_text(dataset_dir, "images");
    ASSERT_TRUE(result.has_value()) << result.error().format();

    const auto& cameras = std::get<0>(*result);

    ASSERT_EQ(cameras.size(), 1u);
    EXPECT_EQ(cameras[0]->image_name(), "frame_0001.png");
    EXPECT_TRUE(fs::equivalent(cameras[0]->image_path(), nested_image));
    EXPECT_TRUE(fs::equivalent(cameras[0]->mask_path(), nested_mask));
}

TEST_F(ColmapImageLayoutTest, FiltersTextPointCloudByMinimumTrackLength) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required for COLMAP point cloud load";
    }

    const fs::path dataset_dir = temp_dir_ / "dataset";
    write_text_file(dataset_dir / "points3D.txt",
                    "1 0 0 0 255 0 0 0.1 1 0\n"
                    "2 1 0 0 0 255 0 0.2 1 0 2 0 3 0\n"
                    "3 2 0 0 0 0 255 0.3 1 0 2 0\n");

    const auto result = lfs::io::read_colmap_point_cloud_text_with_stats(
        dataset_dir,
        lfs::io::LoadOptions{.min_track_length = 3});

    EXPECT_TRUE(result.track_filter_applied);
    EXPECT_EQ(result.total_points, 3u);
    EXPECT_EQ(result.points_after_filtering, 1u);
    EXPECT_EQ(result.point_cloud.size(), 1u);
}

TEST_F(ColmapImageLayoutTest, ResolvesDepthMapsByImageName) {
    const fs::path dataset_dir = temp_dir_ / "dataset";
    const fs::path image_path = dataset_dir / "images" / "frame_0000.png";
    const fs::path depth_path = dataset_dir / "depth" / "frame_0000.png";

    write_minimal_colmap_text_dataset(dataset_dir, "frame_0000.png");
    write_png(image_path);
    write_png(depth_path);

    auto result =
        lfs::io::read_colmap_cameras_and_images_text(dataset_dir, "images");
    ASSERT_TRUE(result.has_value()) << result.error().format();

    const auto& cameras = std::get<0>(*result);

    ASSERT_EQ(cameras.size(), 1u);
    EXPECT_TRUE(cameras[0]->has_depth());
    EXPECT_TRUE(fs::equivalent(cameras[0]->depth_path(), depth_path));
}

TEST_F(ColmapImageLayoutTest, DepthDirCacheResolvesDepthsFolderAndDepthExtension) {
    const fs::path dataset_dir = temp_dir_ / "dataset";
    const fs::path depth_path = dataset_dir / "depths" / "nested" / "frame_0001.depth.png";

    write_png(depth_path);

    const lfs::io::DepthDirCache cache(dataset_dir);
    const auto result = cache.lookup("nested/frame_0001.png");

    ASSERT_TRUE(result.found());
    EXPECT_TRUE(fs::equivalent(result.path, depth_path));
}

TEST_F(ColmapImageLayoutTest, DepthDirCacheFallsBackToTrailingFrameNumber) {
    const fs::path dataset_dir = temp_dir_ / "dataset";
    const fs::path depth_path = dataset_dir / "depth" / "DEPTH_0042.png";

    write_png(depth_path);

    const lfs::io::DepthDirCache cache(dataset_dir);
    const auto result = cache.lookup("RENDER_0042.png");

    ASSERT_TRUE(result.found());
    EXPECT_TRUE(fs::equivalent(result.path, depth_path));
}

TEST_F(ColmapImageLayoutTest, DepthDirCacheFrameNumberFallbackSkipsAmbiguousMatches) {
    const fs::path dataset_dir = temp_dir_ / "dataset";
    write_png(dataset_dir / "depth" / "DEPTH_0042.png");
    write_png(dataset_dir / "depth" / "CONF_0042.png");

    const lfs::io::DepthDirCache cache(dataset_dir);
    const auto result = cache.lookup("RENDER_0042.png");

    EXPECT_FALSE(result.found());
    EXPECT_FALSE(result.ambiguous());
}

TEST_F(ColmapImageLayoutTest, ResolvesDuplicateNestedImagesAndMasksByRelativePath) {
    const fs::path dataset_dir = temp_dir_ / "dataset";
    const fs::path image_a = dataset_dir / "images" / "img1" / "frame_0001.png";
    const fs::path image_b = dataset_dir / "images" / "img2" / "frame_0001.png";
    const fs::path mask_a = dataset_dir / "masks" / "img1" / "frame_0001.png";
    const fs::path mask_b = dataset_dir / "masks" / "img2" / "frame_0001.png";

    write_minimal_colmap_text_dataset(
        dataset_dir,
        std::vector<std::string>{"img1/frame_0001.png", "img2/frame_0001.png"});
    write_png(image_a);
    write_png(image_b);
    write_png(mask_a);
    write_png(mask_b);

    auto result =
        lfs::io::read_colmap_cameras_and_images_text(dataset_dir, "images");
    ASSERT_TRUE(result.has_value()) << result.error().format();

    auto& [cameras, scene_center] = *result;
    (void)scene_center;

    ASSERT_EQ(cameras.size(), 2u);
    EXPECT_EQ(cameras[0]->image_name(), "img1/frame_0001.png");
    EXPECT_EQ(cameras[1]->image_name(), "img2/frame_0001.png");
    EXPECT_TRUE(fs::equivalent(cameras[0]->image_path(), image_a));
    EXPECT_TRUE(fs::equivalent(cameras[1]->image_path(), image_b));
    EXPECT_TRUE(fs::equivalent(cameras[0]->mask_path(), mask_a));
    EXPECT_TRUE(fs::equivalent(cameras[1]->mask_path(), mask_b));
}

TEST_F(ColmapImageLayoutTest, FailsWhenDuplicateNestedImagesAreReferencedByBasename) {
    const fs::path dataset_dir = temp_dir_ / "dataset";
    const fs::path image_a = dataset_dir / "images" / "img1" / "frame_0001.png";
    const fs::path image_b = dataset_dir / "images" / "img2" / "frame_0001.png";

    write_minimal_colmap_text_dataset(
        dataset_dir,
        std::vector<std::string>{"frame_0001.png", "frame_0001.png"});
    write_png(image_a);
    write_png(image_b);

    EXPECT_THROW(
        (void)lfs::io::read_colmap_cameras_and_images_text(dataset_dir, "images"),
        std::runtime_error);
}

TEST_F(ColmapImageLayoutTest, ValidationFailsWhenDuplicateBasenameWasCollapsedInMetadata) {
    const fs::path dataset_dir = temp_dir_ / "dataset";

    write_minimal_colmap_text_dataset(dataset_dir, "frame_0001.png");
    write_png(dataset_dir / "images" / "img1" / "frame_0001.png");
    write_png(dataset_dir / "images" / "img2" / "frame_0001.png");

    auto result = lfs::io::validate_colmap_dataset_layout(dataset_dir, "images");
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, lfs::io::ErrorCode::INVALID_DATASET);
    EXPECT_NE(result.error().message.find("basename only"), std::string::npos);
    EXPECT_NE(result.error().message.find("Metadata contains 1 record"), std::string::npos);
    EXPECT_NE(result.error().message.find("flattened or dropped"), std::string::npos);
}

TEST_F(ColmapImageLayoutTest, ValidationFailsWhenMasksDoNotMirrorRelativeImageLayout) {
    const fs::path dataset_dir = temp_dir_ / "dataset";

    write_minimal_colmap_text_dataset(
        dataset_dir,
        std::vector<std::string>{"img1/frame_0001.png", "img2/frame_0001.png"});
    write_png(dataset_dir / "images" / "img1" / "frame_0001.png");
    write_png(dataset_dir / "images" / "img2" / "frame_0001.png");
    write_png(dataset_dir / "masks" / "cam_a" / "frame_0001.png");
    write_png(dataset_dir / "masks" / "cam_b" / "frame_0001.png");

    auto result = lfs::io::validate_colmap_dataset_layout(dataset_dir, "images");
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, lfs::io::ErrorCode::INVALID_DATASET);
    EXPECT_NE(result.error().message.find("mask"), std::string::npos);
    EXPECT_NE(result.error().message.find("masks/img1/frame_0001.png"), std::string::npos);
}

TEST_F(ColmapImageLayoutTest, FailsEarlyWhenReferencedImageIsMissing) {
    const fs::path dataset_dir = temp_dir_ / "dataset";

    write_minimal_colmap_text_dataset(dataset_dir, "missing_frame.png");

    auto result =
        lfs::io::read_colmap_cameras_and_images_text(dataset_dir, "images");
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, lfs::io::ErrorCode::PATH_NOT_FOUND);
}

TEST_F(ColmapImageLayoutTest, ValidateOnlyRunsColmapPreflight) {
    const fs::path dataset_dir = temp_dir_ / "dataset";

    write_minimal_colmap_text_dataset(dataset_dir, "frame_0001.png");
    write_png(dataset_dir / "images" / "img1" / "frame_0001.png");
    write_png(dataset_dir / "images" / "img2" / "frame_0001.png");

    lfs::io::ColmapLoader loader;
    auto result = loader.load(dataset_dir, {.validate_only = true});

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, lfs::io::ErrorCode::INVALID_DATASET);
    EXPECT_NE(result.error().message.find("basename only"), std::string::npos);
}

TEST_F(ColmapImageLayoutTest, ValidationCanBeCancelledDuringFilesystemScan) {
    const fs::path dataset_dir = temp_dir_ / "dataset";

    write_minimal_colmap_text_dataset(dataset_dir, "frame_0001.png");
    write_png(dataset_dir / "images" / "frame_0001.png");

    int cancel_checks = 0;
    auto result = lfs::io::validate_colmap_dataset_layout(dataset_dir, "images", {
                                                                                     .cancel_requested = [&cancel_checks]() {
                                                                                         return ++cancel_checks >= 1;
                                                                                     },
                                                                                 });

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, lfs::io::ErrorCode::CANCELLED);
}

TEST_F(ColmapImageLayoutTest, LoadCanBeCancelledDuringMetadataParse) {
    const fs::path dataset_dir = temp_dir_ / "dataset";

    write_minimal_colmap_text_dataset(dataset_dir, "frame_0001.png");
    write_png(dataset_dir / "images" / "frame_0001.png");

    int cancel_checks = 0;
    lfs::io::ColmapLoader loader;
    auto result = loader.load(dataset_dir, {
                                               .cancel_requested = [&cancel_checks]() {
                                                   return ++cancel_checks >= 2;
                                               },
                                           });

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, lfs::io::ErrorCode::CANCELLED);
}

TEST_F(ColmapImageLayoutTest, LoadCancelsInsteadOfFallingBackFromCustomPointCloudPly) {
    const fs::path dataset_dir = temp_dir_ / "dataset";

    write_minimal_colmap_text_dataset(dataset_dir, "frame_0001.png");
    write_png(dataset_dir / "images" / "frame_0001.png");
    write_ascii_double_ply(dataset_dir / "points3D.ply", 300000);

    std::atomic<bool> cancel_requested{false};
    std::atomic<int> cancel_checks{0};
    cancel_requested.store(true, std::memory_order_relaxed);

    lfs::io::ColmapLoader loader;
    auto result = loader.load(dataset_dir, {
                                               .cancel_requested = [&cancel_requested, &cancel_checks]() {
                                                   const int check_count = cancel_checks.fetch_add(1, std::memory_order_relaxed) + 1;
                                                   return cancel_requested.load(std::memory_order_relaxed) && check_count >= 50;
                                               },
                                           });

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, lfs::io::ErrorCode::CANCELLED);
}

TEST_F(ColmapImageLayoutTest, DetectDatasetInfoCountsNestedImagesAndMasks) {
    const fs::path dataset_dir = temp_dir_ / "dataset";

    write_png(dataset_dir / "images" / "img1" / "frame_0001.png");
    write_png(dataset_dir / "images" / "img2" / "frame_0002.png");
    write_png(dataset_dir / "masks" / "img1" / "frame_0001.png");
    write_png(dataset_dir / "masks" / "img2" / "frame_0002.png");
    write_png(dataset_dir / "depth" / "img1" / "frame_0001.png");
    write_png(dataset_dir / "depth" / "img2" / "frame_0002.png");
    write_text_file(dataset_dir / "sparse" / "0" / "cameras.txt",
                    "1 PINHOLE 1 1 1 1 0.5 0.5\n");

    const lfs::io::DatasetInfo info = lfs::io::detect_dataset_info(dataset_dir);

    EXPECT_TRUE(fs::equivalent(info.images_path, dataset_dir / "images"));
    EXPECT_TRUE(fs::equivalent(info.masks_path, dataset_dir / "masks"));
    EXPECT_EQ(info.image_count, 2);
    EXPECT_TRUE(info.has_masks);
    EXPECT_EQ(info.mask_count, 2);
    EXPECT_TRUE(fs::equivalent(info.depths_path, dataset_dir / "depth"));
    EXPECT_TRUE(info.has_depths);
    EXPECT_EQ(info.depth_count, 2);
}

TEST_F(ColmapImageLayoutTest, WriteBackAppliesSceneTransformsToTextSparseModel) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required for Camera-backed COLMAP write-back";
    }

    const fs::path dataset_dir = temp_dir_ / "dataset";
    const fs::path output_dir = temp_dir_ / "out_sparse";

    write_text_file(dataset_dir / "cameras.txt",
                    "1 PINHOLE 640 480 500 500 320 240\n");
    write_text_file(dataset_dir / "images.txt",
                    "1 1 0 0 0 0 0 0 1 frame_0001.png\n"
                    "12 34 7\n");
    write_text_file(dataset_dir / "points3D.txt",
                    "7 10 20 30 1 2 3 0.25 1 0\n");

    auto cameras_result = lfs::io::read_colmap_cameras_only(dataset_dir);
    ASSERT_TRUE(cameras_result.has_value()) << cameras_result.error().format();
    auto [cameras, scene_center] = std::move(*cameras_result);
    (void)scene_center;
    ASSERT_EQ(cameras.size(), 1u);

    lfs::io::PointCloud point_cloud(
        lfs::io::Tensor::from_vector({10.0f, 20.0f, 30.0f}, {1, 3}, lfs::io::Device::CPU),
        lfs::io::Tensor::from_vector({1.0f / 255.0f, 2.0f / 255.0f, 3.0f / 255.0f}, {1, 3}, lfs::io::Device::CPU));
    const glm::mat4 transform = glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 2.0f, 3.0f));
    const std::vector<lfs::io::ColmapCameraWriteData> camera_data{
        lfs::io::ColmapCameraWriteData{
            .camera = cameras[0],
            .data_world_transform = transform,
        },
    };

    auto write_result = lfs::io::write_colmap_reconstruction(
        dataset_dir,
        output_dir,
        camera_data,
        &point_cloud,
        transform,
        lfs::io::ColmapWriteOptions{.format = lfs::io::ColmapWriteFormat::Text});
    ASSERT_TRUE(write_result.has_value()) << write_result.error().format();

    std::ifstream images_file(output_dir / "images.txt");
    std::stringstream images_contents;
    images_contents << images_file.rdbuf();
    const std::string images_text = images_contents.str();
    EXPECT_NE(images_text.find("12 34 7"), std::string::npos);

    std::istringstream image_lines(images_text);
    std::string image_line;
    while (std::getline(image_lines, image_line) && (image_line.empty() || image_line.starts_with("#"))) {
    }
    ASSERT_FALSE(image_line.empty());

    uint32_t image_id = 0;
    uint32_t camera_id = 0;
    double qw = 0.0;
    double qx = 0.0;
    double qy = 0.0;
    double qz = 0.0;
    double tx = 0.0;
    double ty = 0.0;
    double tz = 0.0;
    std::string image_name;
    std::istringstream image_pose(image_line);
    ASSERT_TRUE(image_pose >> image_id >> qw >> qx >> qy >> qz >> tx >> ty >> tz >> camera_id >> image_name);
    EXPECT_EQ(image_id, 1u);
    EXPECT_EQ(camera_id, 1u);
    EXPECT_EQ(image_name, "frame_0001.png");
    EXPECT_NEAR(qw, 1.0, 1e-6);
    EXPECT_NEAR(qx, 0.0, 1e-6);
    EXPECT_NEAR(qy, 0.0, 1e-6);
    EXPECT_NEAR(qz, 0.0, 1e-6);
    EXPECT_NEAR(tx, -1.0, 1e-6);
    EXPECT_NEAR(ty, -2.0, 1e-6);
    EXPECT_NEAR(tz, -3.0, 1e-6);

    std::ifstream points_file(output_dir / "points3D.txt");
    std::stringstream points_contents;
    points_contents << points_file.rdbuf();
    const std::string points_text = points_contents.str();
    EXPECT_NE(points_text.find("0.25 1 0"), std::string::npos);

    std::istringstream point_lines(points_text);
    std::string point_line;
    while (std::getline(point_lines, point_line) && (point_line.empty() || point_line.starts_with("#"))) {
    }
    ASSERT_FALSE(point_line.empty());

    uint64_t point_id = 0;
    int r = 0;
    int g = 0;
    int b = 0;
    double px = 0.0;
    double py = 0.0;
    double pz = 0.0;
    double error = 0.0;
    uint32_t track_image_id = 0;
    uint32_t track_point_idx = 0;
    std::istringstream point_record(point_line);
    ASSERT_TRUE(point_record >> point_id >> px >> py >> pz >> r >> g >> b >> error >> track_image_id >> track_point_idx);
    EXPECT_EQ(point_id, 7u);
    EXPECT_NEAR(px, 11.0, 1e-6);
    EXPECT_NEAR(py, 22.0, 1e-6);
    EXPECT_NEAR(pz, 33.0, 1e-6);
    EXPECT_EQ(r, 1);
    EXPECT_EQ(g, 2);
    EXPECT_EQ(b, 3);
    EXPECT_NEAR(error, 0.25, 1e-6);
    EXPECT_EQ(track_image_id, 1u);
    EXPECT_EQ(track_point_idx, 0u);
}

TEST_F(ColmapImageLayoutTest, WriteBackRemovesStaleOppositeFormatSparseFiles) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required for Camera-backed COLMAP write-back";
    }

    const fs::path dataset_dir = temp_dir_ / "dataset";
    const fs::path output_dir = temp_dir_ / "out_sparse";

    write_text_file(dataset_dir / "cameras.txt",
                    "1 PINHOLE 640 480 500 500 320 240\n");
    write_text_file(dataset_dir / "images.txt",
                    "1 1 0 0 0 0 0 0 1 frame_0001.png\n"
                    "\n");
    write_text_file(dataset_dir / "points3D.txt",
                    "7 10 20 30 1 2 3 0.25\n");

    write_text_file(output_dir / "cameras.bin", "stale cameras\n");
    write_text_file(output_dir / "images.bin", "stale images\n");
    write_text_file(output_dir / "points3D.bin", "stale points\n");

    auto cameras_result = lfs::io::read_colmap_cameras_only(dataset_dir);
    ASSERT_TRUE(cameras_result.has_value()) << cameras_result.error().format();
    auto [cameras, scene_center] = std::move(*cameras_result);
    (void)scene_center;
    ASSERT_EQ(cameras.size(), 1u);

    const std::vector<lfs::io::ColmapCameraWriteData> camera_data{
        lfs::io::ColmapCameraWriteData{
            .camera = cameras[0],
            .data_world_transform = glm::mat4(1.0f),
        },
    };

    auto write_result = lfs::io::write_colmap_reconstruction(
        dataset_dir,
        output_dir,
        camera_data,
        nullptr,
        glm::mat4(1.0f),
        lfs::io::ColmapWriteOptions{.format = lfs::io::ColmapWriteFormat::Text});
    ASSERT_TRUE(write_result.has_value()) << write_result.error().format();

    EXPECT_TRUE(fs::exists(output_dir / "cameras.txt"));
    EXPECT_TRUE(fs::exists(output_dir / "images.txt"));
    EXPECT_TRUE(fs::exists(output_dir / "points3D.txt"));
    EXPECT_FALSE(fs::exists(output_dir / "cameras.bin"));
    EXPECT_FALSE(fs::exists(output_dir / "images.bin"));
    EXPECT_FALSE(fs::exists(output_dir / "points3D.bin"));
}

TEST_F(ColmapImageLayoutTest, WriteBackStagesAndPublishesOneValidatedGeneration) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required for Camera-backed COLMAP write-back";
    }

    const fs::path sparse_dir = temp_dir_ / "sparse";
    write_text_file(sparse_dir / "cameras.txt",
                    "1 PINHOLE 640 480 500 500 320 240\n");
    write_text_file(sparse_dir / "images.txt",
                    "1 1 0 0 0 0 0 0 1 frame_0001.png\n"
                    "\n");
    write_text_file(sparse_dir / "rigs.bin", "auxiliary sparse metadata\n");

    auto cameras_result = lfs::io::read_colmap_cameras_only(sparse_dir);
    ASSERT_TRUE(cameras_result.has_value()) << cameras_result.error().format();
    auto [cameras, scene_center] = std::move(*cameras_result);
    (void)scene_center;
    ASSERT_EQ(cameras.size(), 1u);

    const glm::mat4 transform = glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 2.0f, 3.0f));
    const std::vector<lfs::io::ColmapCameraWriteData> camera_data{
        lfs::io::ColmapCameraWriteData{
            .camera = cameras[0],
            .data_world_transform = transform,
        },
    };

    const auto write_result = lfs::io::write_colmap_reconstruction(
        sparse_dir,
        sparse_dir,
        camera_data,
        nullptr,
        glm::mat4(1.0f),
        lfs::io::ColmapWriteOptions{.format = lfs::io::ColmapWriteFormat::Text});
    ASSERT_TRUE(write_result.has_value()) << write_result.error().format();

    const auto reopened = lfs::io::read_colmap_cameras_only(sparse_dir);
    ASSERT_TRUE(reopened.has_value()) << reopened.error().format();
    EXPECT_EQ(std::get<0>(*reopened).size(), 1u);

    std::ifstream images_stream(sparse_dir / "images.txt");
    const std::string images_text{std::istreambuf_iterator<char>(images_stream),
                                  std::istreambuf_iterator<char>()};
    EXPECT_NE(images_text.find(" -1 -2 -3 1 frame_0001.png"), std::string::npos);

    std::ifstream auxiliary_stream(sparse_dir / "rigs.bin");
    const std::string auxiliary_text{std::istreambuf_iterator<char>(auxiliary_stream),
                                     std::istreambuf_iterator<char>()};
    EXPECT_EQ(auxiliary_text, "auxiliary sparse metadata\n");

    for (const auto& entry : fs::directory_iterator(temp_dir_)) {
        EXPECT_FALSE(entry.path().filename().string().starts_with("sparse."))
            << "staging generation was not cleaned up: " << entry.path();
    }
}
