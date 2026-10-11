/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/scene.hpp"
#include "visualizer/scene/scene_manager.hpp"

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

    class ColmapCameraImportTest : public ::testing::Test {
    protected:
        void SetUp() override {
            temp_dir_ = fs::temp_directory_path() / "lfs_colmap_camera_import_test";
            std::error_code ec;
            fs::remove_all(temp_dir_, ec);
            fs::create_directories(temp_dir_);
        }

        void TearDown() override {
            std::error_code ec;
            fs::remove_all(temp_dir_, ec);
        }

        // A text COLMAP model with identity rotations; image i sits at translation z = depths[i].
        fs::path write_model(const std::string& name, const std::vector<double>& depths) {
            const fs::path sparse = temp_dir_ / name;
            fs::create_directories(sparse);
            std::ofstream cameras(sparse / "cameras.txt");
            cameras << "1 PINHOLE 64 48 50 50 32 24\n";
            std::ofstream images(sparse / "images.txt");
            for (size_t i = 0; i < depths.size(); ++i)
                images << i + 1 << " 1 0 0 0 0 0 " << depths[i] << " 1 " << name << "_" << i << ".png\n\n";
            return sparse;
        }

        fs::path temp_dir_;
    };

} // namespace

TEST_F(ColmapCameraImportTest, SecondImportGetsUidsAboveTheFirstAndKeepsItsPoses) {
    const fs::path first = write_model("first", {1.0, 2.0});
    const fs::path second = write_model("second", {30.0, 40.0, 50.0});

    lfs::vis::SceneManager scene_manager;
    scene_manager.loadColmapCamerasOnly(first);
    scene_manager.loadColmapCamerasOnly(second);

    const auto& scene = scene_manager.getScene();
    const auto cameras = scene.getAllCameras();
    ASSERT_EQ(cameras.size(), 5u);

    const std::map<std::string, double> depth_by_image = {
        {"first_0.png", 1.0},
        {"first_1.png", 2.0},
        {"second_0.png", 30.0},
        {"second_1.png", 40.0},
        {"second_2.png", 50.0}};
    std::set<int> uids;
    for (const auto& camera : cameras) {
        uids.insert(camera->uid());
        const auto found = scene.getCameraByUid(camera->uid());
        ASSERT_NE(found, nullptr);
        EXPECT_EQ(found->image_name(), camera->image_name());
        EXPECT_FLOAT_EQ(found->T().to_vector()[2], depth_by_image.at(camera->image_name()));
    }
    EXPECT_EQ(uids.size(), cameras.size());
}
