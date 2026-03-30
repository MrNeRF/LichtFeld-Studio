/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/mesh_data.hpp"
#include "core/point_cloud.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "gui/video_export_utils.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/render_constants.hpp"
#include "scene/scene_manager.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

using lfs::core::Device;
using lfs::core::Tensor;

namespace {

    std::unique_ptr<lfs::core::SplatData> make_test_splat(const std::vector<float>& xyz) {
        const size_t count = xyz.size() / 3;
        auto means = Tensor::from_vector(xyz, {count, size_t{3}}, Device::CUDA).to(lfs::core::DataType::Float32);
        auto sh0 = Tensor::zeros({count, size_t{1}, size_t{3}}, Device::CUDA, lfs::core::DataType::Float32);
        auto shN = Tensor::zeros({count, size_t{3}, size_t{3}}, Device::CUDA, lfs::core::DataType::Float32);
        auto scaling = Tensor::zeros({count, size_t{3}}, Device::CUDA, lfs::core::DataType::Float32);

        std::vector<float> rotation_data(count * 4, 0.0f);
        for (size_t i = 0; i < count; ++i) {
            rotation_data[i * 4] = 1.0f;
        }
        auto rotation = Tensor::from_vector(rotation_data, {count, size_t{4}}, Device::CUDA).to(lfs::core::DataType::Float32);
        auto opacity = Tensor::zeros({count, size_t{1}}, Device::CUDA, lfs::core::DataType::Float32);

        return std::make_unique<lfs::core::SplatData>(
            1,
            std::move(means),
            std::move(sh0),
            std::move(shN),
            std::move(scaling),
            std::move(rotation),
            std::move(opacity),
            1.0f);
    }

    std::shared_ptr<lfs::core::PointCloud> make_test_point_cloud() {
        auto means = Tensor::from_vector(
                         std::vector<float>{0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f},
                         {size_t{2}, size_t{3}},
                         Device::CUDA)
                         .to(lfs::core::DataType::Float32);
        auto colors = Tensor::from_vector(
                          std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f},
                          {size_t{2}, size_t{3}},
                          Device::CUDA)
                          .to(lfs::core::DataType::Float32);
        return std::make_shared<lfs::core::PointCloud>(std::move(means), std::move(colors));
    }

    std::shared_ptr<lfs::core::MeshData> make_test_mesh() {
        auto mesh = std::make_shared<lfs::core::MeshData>();
        mesh->vertices = Tensor::from_vector(
                             std::vector<float>{
                                 0.0f,
                                 0.0f,
                                 0.0f,
                                 1.0f,
                                 0.0f,
                                 0.0f,
                                 0.0f,
                                 1.0f,
                                 0.0f,
                             },
                             {size_t{3}, size_t{3}},
                             Device::CPU)
                             .to(lfs::core::DataType::Float32);
        mesh->indices = Tensor::from_vector(
                            std::vector<int>{0, 1, 2},
                            {size_t{1}, size_t{3}},
                            Device::CPU)
                            .to(lfs::core::DataType::Int32);
        return mesh;
    }

    void expect_translation(const glm::mat4& transform, const glm::vec3& expected) {
        EXPECT_FLOAT_EQ(transform[3][0], expected.x);
        EXPECT_FLOAT_EQ(transform[3][1], expected.y);
        EXPECT_FLOAT_EQ(transform[3][2], expected.z);
    }

    void expect_visualizer_translation_from_data(const glm::mat4& transform, const glm::vec3& data_translation) {
        expect_translation(transform, lfs::rendering::visualizerWorldPointFromDataWorld(data_translation));
    }

} // namespace

TEST(VideoExportUtilsTest, CaptureSnapshotUsesRenderableModelAndTransforms) {
    lfs::vis::SceneManager scene_manager;
    auto& scene = scene_manager.getScene();

    scene.addNode("left", make_test_splat({0.0f, 0.0f, 0.0f}));
    scene.addNode("right", make_test_splat({0.0f, 0.0f, 0.0f}));
    scene.setNodeTransform("left", glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 2.0f, 3.0f)));
    scene.setNodeTransform("right", glm::translate(glm::mat4(1.0f), glm::vec3(-4.0f, 0.5f, 2.0f)));

    auto snapshot_result = lfs::vis::gui::captureVideoExportSceneSnapshot(scene_manager);
    ASSERT_TRUE(snapshot_result.has_value()) << snapshot_result.error();

    const auto& snapshot = *snapshot_result;
    ASSERT_TRUE(snapshot.combined_model);
    EXPECT_EQ(snapshot.combined_model->size(), 2u);
    EXPECT_FALSE(snapshot.point_cloud);
    ASSERT_EQ(snapshot.model_transforms.size(), 2u);
    expect_visualizer_translation_from_data(snapshot.model_transforms[0], {1.0f, 2.0f, 3.0f});
    expect_visualizer_translation_from_data(snapshot.model_transforms[1], {-4.0f, 0.5f, 2.0f});

    ASSERT_TRUE(snapshot.transform_indices);
    EXPECT_EQ(snapshot.transform_indices->cpu().to_vector_int(), (std::vector<int>{0, 1}));
}

TEST(VideoExportUtilsTest, CaptureSnapshotPrefersSplatsOverPointCloudAndKeepsMeshes) {
    lfs::vis::SceneManager scene_manager;
    auto& scene = scene_manager.getScene();

    scene.addNode("splat", make_test_splat({0.0f, 0.0f, 0.0f}));
    scene.addPointCloud("points", make_test_point_cloud());
    scene.addMesh("mesh", make_test_mesh());

    auto snapshot_result = lfs::vis::gui::captureVideoExportSceneSnapshot(scene_manager);
    ASSERT_TRUE(snapshot_result.has_value()) << snapshot_result.error();

    const auto& snapshot = *snapshot_result;
    ASSERT_TRUE(snapshot.combined_model);
    EXPECT_FALSE(snapshot.point_cloud);
    ASSERT_EQ(snapshot.meshes.size(), 1u);
    ASSERT_TRUE(snapshot.meshes[0].mesh);
}

TEST(VideoExportUtilsTest, CaptureSnapshotKeepsPointCloudTransformWhenNoModelExists) {
    lfs::vis::SceneManager scene_manager;
    auto& scene = scene_manager.getScene();

    scene.addPointCloud("points", make_test_point_cloud());
    scene.setNodeTransform("points", glm::translate(glm::mat4(1.0f), glm::vec3(3.0f, -2.0f, 5.0f)));

    auto snapshot_result = lfs::vis::gui::captureVideoExportSceneSnapshot(scene_manager);
    ASSERT_TRUE(snapshot_result.has_value()) << snapshot_result.error();

    const auto& snapshot = *snapshot_result;
    EXPECT_FALSE(snapshot.combined_model);
    ASSERT_TRUE(snapshot.point_cloud);
    EXPECT_EQ(snapshot.point_cloud->size(), 2);
    expect_visualizer_translation_from_data(snapshot.point_cloud_transform, {3.0f, -2.0f, 5.0f});
}

TEST(VideoExportUtilsTest, CaptureSnapshotSupportsMeshOnlyScenes) {
    lfs::vis::SceneManager scene_manager;
    auto& scene = scene_manager.getScene();

    scene.addMesh("mesh", make_test_mesh());
    scene.setNodeTransform("mesh", glm::translate(glm::mat4(1.0f), glm::vec3(-1.5f, 0.0f, 4.0f)));

    auto snapshot_result = lfs::vis::gui::captureVideoExportSceneSnapshot(scene_manager);
    ASSERT_TRUE(snapshot_result.has_value()) << snapshot_result.error();

    const auto& snapshot = *snapshot_result;
    EXPECT_FALSE(snapshot.combined_model);
    EXPECT_FALSE(snapshot.point_cloud);
    ASSERT_EQ(snapshot.meshes.size(), 1u);
    ASSERT_TRUE(snapshot.meshes[0].mesh);
    expect_visualizer_translation_from_data(snapshot.meshes[0].transform, {-1.5f, 0.0f, 4.0f});
}

TEST(VideoExportUtilsTest, ValidateVideoExportOptionsRejectsInvalidValues) {
    EXPECT_FALSE(lfs::vis::gui::validateVideoExportOptions({.width = 0,
                                                            .height = 1080,
                                                            .framerate = 30,
                                                            .crf = 18}));
    EXPECT_FALSE(lfs::vis::gui::validateVideoExportOptions({.width = 1920,
                                                            .height = -1,
                                                            .framerate = 30,
                                                            .crf = 18}));
    EXPECT_FALSE(lfs::vis::gui::validateVideoExportOptions({.width = 1920,
                                                            .height = 1080,
                                                            .framerate = 0,
                                                            .crf = 18}));
    EXPECT_FALSE(lfs::vis::gui::validateVideoExportOptions({.width = lfs::rendering::MAX_VIEWPORT_SIZE + 1,
                                                            .height = 1080,
                                                            .framerate = 30,
                                                            .crf = 18}));
    EXPECT_FALSE(lfs::vis::gui::validateVideoExportOptions({.width = 1920,
                                                            .height = 1080,
                                                            .framerate = 30,
                                                            .crf = 99}));
}

TEST(VideoExportUtilsTest, ValidateVideoExportOptionsAcceptsTypicalPreset) {
    auto result = lfs::vis::gui::validateVideoExportOptions({.width = 1920,
                                                             .height = 1080,
                                                             .framerate = 30,
                                                             .crf = 18});

    ASSERT_TRUE(result.has_value()) << result.error();
    EXPECT_EQ(result->width, 1920);
    EXPECT_EQ(result->height, 1080);
    EXPECT_EQ(result->framerate, 30);
    EXPECT_EQ(result->crf, 18);
}
