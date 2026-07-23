/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bus.hpp"
#include "core/events.hpp"
#include "core/editor_context.hpp"
#include "core/image_io.hpp"
#include "core/image_loader.hpp"
#include "core/point_cloud.hpp"
#include "core/services.hpp"
#include "core/tensor.hpp"
#include "io/cache_image_loader.hpp"
#include "operation/undo_history.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "visualizer/gui_capabilities.hpp"
#include "visualizer/rendering/render_pass.hpp"
#include "visualizer/rendering/rendering_manager.hpp"
#include "visualizer/rendering/split_view_composition.hpp"
#include "visualizer/rendering/split_view_service.hpp"
#include "visualizer/rendering/viewport_artifact_service.hpp"
#include "visualizer/rendering/viewport_frame_lifecycle_service.hpp"
#include "visualizer/rendering/viewport_request_builder.hpp"
#include "visualizer/scene/scene_manager.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

namespace lfs::vis {

    namespace {
        std::unique_ptr<lfs::core::SplatData> makeTestSplat(const float x) {
            using lfs::core::DataType;
            using lfs::core::Device;
            using lfs::core::Tensor;

            return std::make_unique<lfs::core::SplatData>(
                0,
                Tensor::from_vector({x, 0.0f, 2.0f}, {size_t{1}, size_t{3}}, Device::CPU),
                Tensor::from_vector({1.0f, 1.0f, 1.0f}, {size_t{1}, size_t{1}, size_t{3}}, Device::CPU),
                Tensor::zeros({size_t{1}, size_t{0}, size_t{3}}, Device::CPU, DataType::Float32),
                Tensor::from_vector({0.0f, 0.0f, 0.0f}, {size_t{1}, size_t{3}}, Device::CPU),
                Tensor::from_vector({1.0f, 0.0f, 0.0f, 0.0f}, {size_t{1}, size_t{4}}, Device::CPU),
                Tensor::from_vector({8.0f}, {size_t{1}, size_t{1}}, Device::CPU),
                1.0f);
        }

        std::unique_ptr<lfs::core::SplatData> makeTwoPointTestSplat(const float x0, const float x1) {
            using lfs::core::DataType;
            using lfs::core::Device;
            using lfs::core::Tensor;

            return std::make_unique<lfs::core::SplatData>(
                0,
                Tensor::from_vector({x0, 0.0f, 2.0f,
                                     x1, 0.0f, 2.0f},
                                    {size_t{2}, size_t{3}}, Device::CPU),
                Tensor::from_vector({1.0f, 1.0f, 1.0f,
                                     1.0f, 1.0f, 1.0f},
                                    {size_t{2}, size_t{1}, size_t{3}}, Device::CPU),
                Tensor::zeros({size_t{2}, size_t{0}, size_t{3}}, Device::CPU, DataType::Float32),
                Tensor::from_vector({0.0f, 0.0f, 0.0f,
                                     0.0f, 0.0f, 0.0f},
                                    {size_t{2}, size_t{3}}, Device::CPU),
                Tensor::from_vector({1.0f, 0.0f, 0.0f, 0.0f,
                                     1.0f, 0.0f, 0.0f, 0.0f},
                                    {size_t{2}, size_t{4}}, Device::CPU),
                Tensor::from_vector({8.0f,
                                     8.0f},
                                    {size_t{2}, size_t{1}}, Device::CPU),
                1.0f);
        }
        std::shared_ptr<lfs::core::PointCloud> makeTestPointCloud() {
            using lfs::core::Device;
            using lfs::core::Tensor;

            auto means = Tensor::from_vector(
                {0.0f, 0.0f, 0.0f,
                 1.0f, 0.0f, 0.0f},
                {size_t{2}, size_t{3}},
                Device::CPU);
            auto colors = Tensor::from_vector(
                {1.0f, 0.0f, 0.0f,
                 0.0f, 1.0f, 0.0f},
                {size_t{2}, size_t{3}},
                Device::CPU);
            return std::make_shared<lfs::core::PointCloud>(std::move(means), std::move(colors));
        }

        void expectVisualizerTranslationFromData(const glm::mat4& transform, const glm::vec3& data_translation) {
            const glm::vec3 expected =
                lfs::rendering::visualizerWorldPointFromDataWorld(data_translation);
            EXPECT_FLOAT_EQ(transform[3][0], expected.x);
            EXPECT_FLOAT_EQ(transform[3][1], expected.y);
            EXPECT_FLOAT_EQ(transform[3][2], expected.z);
        }

        void expectMat3Near(const glm::mat3& actual, const glm::mat3& expected, const float epsilon = 1e-5f) {
            for (int col = 0; col < 3; ++col) {
                for (int row = 0; row < 3; ++row) {
                    EXPECT_NEAR(actual[col][row], expected[col][row], epsilon);
                }
            }
        }

        void waitUntilResizeSettleReady(ViewportFrameLifecycleService& service) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
            while (!service.resizeSettleReady() && std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            ASSERT_TRUE(service.resizeSettleReady());
        }

        void ensureCameraImageLoader() {
            static bool initialized = false;
            if (initialized) {
                return;
            }

            lfs::io::CacheLoader::getInstance(false, false);
            lfs::core::set_image_loader([](const lfs::core::ImageLoadParams& p) {
                return lfs::io::CacheLoader::getInstance().load_cached_image(
                    p.path,
                    {.resize_factor = p.resize_factor,
                     .max_width = p.max_width,
                     .cuda_stream = p.stream,
                     .output_uint8 = p.output_uint8});
            });
            initialized = true;
        }
    } // namespace

    class RenderingManagerEventsTest : public ::testing::Test {
    protected:
        void SetUp() override {
            lfs::event::EventBridge::instance().clear_all();
            lfs::core::event::bus().clear_all();
        }

        void TearDown() override {
            lfs::event::EventBridge::instance().clear_all();
            lfs::core::event::bus().clear_all();
        }
    };

    class SceneManagerRenderStateTest : public ::testing::Test {
    protected:
        void SetUp() override {
            lfs::event::EventBridge::instance().clear_all();
            lfs::core::event::bus().clear_all();
            services().clear();
            op::undoHistory().clear();
        }

        void TearDown() override {
            op::undoHistory().clear();
            services().clear();
            lfs::event::EventBridge::instance().clear_all();
            lfs::core::event::bus().clear_all();
        }
    };

    TEST(SplitViewServiceTest, ToggleGtComparisonRestoresPreviousProjectionMode) {
        SplitViewService service;
        RenderSettings settings;
        settings.equirectangular = true;

        const auto enable = service.toggleMode(settings, SplitViewMode::GTComparison);
        EXPECT_TRUE(enable.mode_changed);
        EXPECT_EQ(enable.previous_mode, SplitViewMode::Disabled);
        EXPECT_EQ(enable.current_mode, SplitViewMode::GTComparison);
        EXPECT_EQ(settings.split_view_mode, SplitViewMode::GTComparison);

        settings.equirectangular = false;

        const auto disable = service.toggleMode(settings, SplitViewMode::GTComparison);
        EXPECT_TRUE(disable.mode_changed);
        EXPECT_EQ(disable.previous_mode, SplitViewMode::GTComparison);
        EXPECT_EQ(disable.current_mode, SplitViewMode::Disabled);
        ASSERT_TRUE(disable.restore_equirectangular.has_value());
        EXPECT_TRUE(*disable.restore_equirectangular);
        EXPECT_TRUE(settings.equirectangular);
        EXPECT_EQ(settings.split_view_mode, SplitViewMode::Disabled);
    }

    TEST(SplitViewServiceTest, UpdateInfoClearsStaleSplitViewLabels) {
        SplitViewService service;

        FrameResources active_resources;
        active_resources.split_view_executed = true;
        active_resources.split_info = {.enabled = true, .left_name = "Left", .right_name = "Right"};
        service.updateInfo(active_resources);

        const auto active_info = service.getInfo();
        EXPECT_TRUE(active_info.enabled);
        EXPECT_EQ(active_info.left_name, "Left");
        EXPECT_EQ(active_info.right_name, "Right");

        FrameResources idle_resources;
        service.updateInfo(idle_resources);

        const auto idle_info = service.getInfo();
        EXPECT_FALSE(idle_info.enabled);
        EXPECT_TRUE(idle_info.left_name.empty());
        EXPECT_TRUE(idle_info.right_name.empty());
    }

    TEST(SplitViewServiceTest, SceneClearedDisablesSplitViewAndResetsOffset) {
        SplitViewService service;
        RenderSettings settings;
        settings.split_view_mode = SplitViewMode::PLYComparison;
        settings.split_view_offset = 3;

        const auto result = service.handleSceneCleared(settings);

        EXPECT_TRUE(result.mode_changed);
        EXPECT_EQ(settings.split_view_mode, SplitViewMode::Disabled);
        EXPECT_EQ(settings.split_view_offset, 0);
    }

    TEST(SplitViewServiceTest, IndependentDualCopiesPrimaryViewportAndResetsFocus) {
        SplitViewService service;
        RenderSettings settings;
        Viewport primary_viewport(640, 480);
        primary_viewport.setViewMatrix(glm::mat3(1.0f), glm::vec3(1.0f, 2.0f, 3.0f));
        service.setFocusedPanel(SplitViewPanelId::Right);

        const auto result = service.toggleMode(
            settings, SplitViewMode::IndependentDual, &primary_viewport);

        EXPECT_TRUE(result.mode_changed);
        EXPECT_EQ(settings.split_view_mode, SplitViewMode::IndependentDual);
        EXPECT_EQ(service.focusedPanel(), SplitViewPanelId::Left);
        EXPECT_EQ(service.secondaryViewport().getTranslation(), primary_viewport.getTranslation());
        EXPECT_EQ(service.secondaryViewport().getRotationMatrix(), primary_viewport.getRotationMatrix());
    }

    TEST(SplitViewServiceTest, IndependentDualToggleOffDisablesModeAndResetsFocus) {
        SplitViewService service;
        RenderSettings settings;
        Viewport primary_viewport(640, 480);

        ASSERT_TRUE(service.toggleMode(settings, SplitViewMode::IndependentDual, &primary_viewport).mode_changed);
        service.setFocusedPanel(SplitViewPanelId::Right);

        const auto result = service.toggleMode(
            settings, SplitViewMode::IndependentDual, &primary_viewport);

        EXPECT_TRUE(result.mode_changed);
        EXPECT_EQ(result.current_mode, SplitViewMode::Disabled);
        EXPECT_EQ(settings.split_view_mode, SplitViewMode::Disabled);
        EXPECT_EQ(service.focusedPanel(), SplitViewPanelId::Left);
    }

    TEST(SplitViewServiceTest, GtRenderCameraUsesVisualizerCameraAxesAndNormalizedSceneRotation) {
        using lfs::core::Camera;
        using lfs::core::CameraModelType;
        using lfs::core::Device;
        using lfs::core::Tensor;

        Camera camera(
            Tensor::from_vector(
                {1.0f, 0.0f, 0.0f,
                 0.0f, 1.0f, 0.0f,
                 0.0f, 0.0f, 1.0f},
                {size_t{3}, size_t{3}},
                Device::CPU),
            Tensor::from_vector({0.0f, 0.0f, 0.0f}, {size_t{3}}, Device::CPU),
            500.0f,
            600.0f,
            320.0f,
            240.0f,
            Tensor(),
            Tensor(),
            CameraModelType::PINHOLE,
            "test.png",
            {},
            {},
            640,
            480,
            7);

        glm::mat4 scene_transform(1.0f);
        scene_transform = glm::translate(scene_transform, glm::vec3(1.0f, 2.0f, 3.0f));
        scene_transform = glm::scale(scene_transform, glm::vec3(2.0f, 3.0f, 4.0f));

        const auto render_camera =
            detail::buildGTRenderCamera(camera, {1280, 960}, scene_transform);
        ASSERT_TRUE(render_camera.has_value());

        expectMat3Near(
            render_camera->rotation,
            lfs::rendering::DATA_TO_VISUALIZER_CAMERA_AXES);
        EXPECT_EQ(render_camera->translation, glm::vec3(1.0f, 2.0f, 3.0f));
        ASSERT_TRUE(render_camera->intrinsics.has_value());
        EXPECT_FLOAT_EQ(render_camera->intrinsics->focal_x, 1000.0f);
        EXPECT_FLOAT_EQ(render_camera->intrinsics->focal_y, 1200.0f);
        EXPECT_FLOAT_EQ(render_camera->intrinsics->center_x, 640.0f);
        EXPECT_FLOAT_EQ(render_camera->intrinsics->center_y, 480.0f);
        EXPECT_FALSE(render_camera->equirectangular);
    }

    TEST(CameraImageLoadTest, PreviewLoadsCanAvoidMutatingCameraImageDimensions) {
        using lfs::core::Camera;
        using lfs::core::CameraModelType;
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::Tensor;

        ensureCameraImageLoader();

        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        const auto image_path = std::filesystem::temp_directory_path() /
                                ("lfs_camera_preview_" + std::to_string(now) + ".png");
        auto image = Tensor::zeros({size_t{6}, size_t{8}, size_t{3}}, Device::CPU, DataType::UInt8);
        ASSERT_NO_THROW(lfs::core::save_image(image_path, image));

        Camera camera(
            Tensor::from_vector(
                {1.0f, 0.0f, 0.0f,
                 0.0f, 1.0f, 0.0f,
                 0.0f, 0.0f, 1.0f},
                {size_t{3}, size_t{3}},
                Device::CPU),
            Tensor::from_vector({0.0f, 0.0f, 0.0f}, {size_t{3}}, Device::CPU),
            500.0f,
            500.0f,
            4.0f,
            3.0f,
            Tensor(),
            Tensor(),
            CameraModelType::PINHOLE,
            "preview.png",
            image_path,
            {},
            8,
            6,
            42);

        auto preview = camera.load_and_get_image(-1, 4, false, false);
        ASSERT_TRUE(preview.is_valid());
        ASSERT_EQ(preview.ndim(), 3);
        EXPECT_EQ(static_cast<int>(preview.shape()[1]), 3);
        EXPECT_EQ(static_cast<int>(preview.shape()[2]), 4);
        EXPECT_EQ(camera.image_width(), 8);
        EXPECT_EQ(camera.image_height(), 6);

        auto published = camera.load_and_get_image(-1, 4, false, true);
        ASSERT_TRUE(published.is_valid());
        EXPECT_EQ(camera.image_width(), 4);
        EXPECT_EQ(camera.image_height(), 3);

        std::filesystem::remove(image_path);
    }

    TEST(SplitViewServiceTest, SharedCameraPoseHelperNormalizesSceneRotationAndAppliesVisualizerAxes) {
        const glm::mat3 world_to_camera = glm::mat3(1.0f);
        const glm::vec3 world_to_camera_translation(0.0f, 0.0f, 0.0f);

        glm::mat4 scene_transform(1.0f);
        scene_transform = glm::translate(scene_transform, glm::vec3(1.0f, 2.0f, 3.0f));
        scene_transform = glm::scale(scene_transform, glm::vec3(2.0f, 3.0f, 4.0f));

        const auto pose = lfs::rendering::visualizerCameraPoseFromDataWorldToCamera(
            world_to_camera,
            world_to_camera_translation,
            scene_transform);

        expectMat3Near(pose.rotation, lfs::rendering::DATA_TO_VISUALIZER_CAMERA_AXES);
        EXPECT_EQ(pose.translation, glm::vec3(1.0f, 2.0f, 3.0f));
    }

    TEST_F(RenderingManagerEventsTest, OrthographicTogglePreservesApparentZoomAtPivotInBothDirections) {
        RenderingManager manager;
        auto settings = manager.getSettings();
        settings.focal_length_mm = 50.0f;
        manager.updateSettings(settings);

        constexpr float viewport_height = 900.0f;
        constexpr float distance_to_pivot = 7.5f;

        manager.setOrthographic(true, viewport_height, distance_to_pivot);
        const auto ortho_settings = manager.getSettings();
        ASSERT_TRUE(ortho_settings.orthographic);

        const float expected_scale = viewport_height /
                                     (2.0f * distance_to_pivot *
                                      std::tan(glm::radians(lfs::rendering::focalLengthToVFov(50.0f)) * 0.5f));
        EXPECT_NEAR(ortho_settings.ortho_scale, expected_scale, 1e-4f);

        settings = ortho_settings;
        settings.ortho_scale *= 1.75f;
        manager.updateSettings(settings);

        manager.setOrthographic(false, viewport_height, distance_to_pivot);
        const auto perspective_settings = manager.getSettings();
        ASSERT_FALSE(perspective_settings.orthographic);

        const float expected_vfov = glm::degrees(2.0f * std::atan(
                                                            viewport_height / (2.0f * distance_to_pivot * settings.ortho_scale)));
        const float expected_focal = lfs::rendering::vFovToFocalLength(expected_vfov);
        EXPECT_NEAR(perspective_settings.focal_length_mm, expected_focal, 1e-4f);
    }

    TEST(SplitViewServiceTest, GtComparisonPlanPreservesGtTextureOrigin) {
        Viewport viewport(640, 480);
        RenderSettings settings;
        settings.split_view_mode = SplitViewMode::GTComparison;
        settings.split_position = 0.4f;

        FrameContext ctx{
            .viewport = viewport,
            .settings = settings,
            .render_size = {640, 480},
            .current_camera_id = 7,
        };

        FrameResources res;
        res.gt_context = GTComparisonContext{
            .gt_image_handle = 11,
            .camera_id = 7,
            .dimensions = {320, 240},
            .gpu_aligned_dims = {320, 256},
            .render_texcoord_scale = {1.0f, 240.0f / 256.0f},
            .gt_texcoord_scale = {1.0f, 1.0f},
            .gt_texture_origin = lfs::rendering::TextureOrigin::TopLeft,
        };
        res.cached_gpu_frame = lfs::rendering::GpuFrame{
            .color = {.id = 22, .size = {320, 240}},
        };

        const auto plan = buildSplitViewCompositionPlan(ctx, res);
        ASSERT_TRUE(plan.has_value());
        ASSERT_TRUE(plan->panels[0].panel.presentation.flip_y.has_value());
        EXPECT_TRUE(*plan->panels[0].panel.presentation.flip_y);
        EXPECT_FALSE(plan->panels[1].panel.presentation.flip_y.has_value());
    }

    TEST(SplitViewServiceTest, CurrentSceneTransformUsesIdentityForMultipleVisiblePointClouds) {
        SceneManager manager;
        auto& scene = manager.getScene();

        const auto left_parent = scene.addGroup("LeftParent");
        const auto right_parent = scene.addGroup("RightParent");

        scene.setNodeTransform(
            "LeftParent",
            glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 2.0f, 3.0f)));
        scene.setNodeTransform(
            "RightParent",
            glm::translate(glm::mat4(1.0f), glm::vec3(-4.0f, 5.0f, 6.0f)));

        scene.addPointCloud("LeftCloud", makeTestPointCloud(), left_parent);
        scene.addPointCloud("RightCloud", makeTestPointCloud(), right_parent);

        EXPECT_EQ(
            detail::currentSceneTransform(&manager, -1),
            lfs::rendering::dataWorldTransformToVisualizerWorld(glm::mat4(1.0f)));
    }

    TEST_F(SceneManagerRenderStateTest, DatasetReadyStateKeepsVisiblePointCloudWhenTrainingModelIsEmpty) {
        SceneManager manager;
        manager.changeContentType(SceneManager::ContentType::Dataset);

        auto& scene = manager.getScene();
        const auto dataset_id = scene.addGroup("Dataset");

        auto means_empty = lfs::core::Tensor::zeros({size_t{0}, size_t{3}}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
        auto sh0_empty = lfs::core::Tensor::zeros({size_t{0}, size_t{1}, size_t{3}}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
        auto shN_empty = lfs::core::Tensor::zeros({size_t{0}, size_t{3}, size_t{3}}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
        auto scaling_empty = lfs::core::Tensor::zeros({size_t{0}, size_t{3}}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
        auto rotation_empty = lfs::core::Tensor::zeros({size_t{0}, size_t{4}}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
        auto opacity_empty = lfs::core::Tensor::zeros({size_t{0}, size_t{1}}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
        scene.addSplat(
            "Model",
            std::make_unique<lfs::core::SplatData>(
                1,
                std::move(means_empty),
                std::move(sh0_empty),
                std::move(shN_empty),
                std::move(scaling_empty),
                std::move(rotation_empty),
                std::move(opacity_empty),
                1.0f),
            dataset_id);
        scene.setTrainingModelNode("Model");

        auto means = lfs::core::Tensor::from_vector({0.0f, 0.0f, 0.0f}, {size_t{1}, size_t{3}}, lfs::core::Device::CPU);
        auto colors = lfs::core::Tensor::from_vector({1.0f, 0.0f, 0.0f}, {size_t{1}, size_t{3}}, lfs::core::Device::CPU);
        scene.addPointCloud("PointCloud", std::make_shared<lfs::core::PointCloud>(std::move(means), std::move(colors)), dataset_id);

        const auto state = manager.buildRenderState();
        ASSERT_NE(state.combined_model, nullptr);
        EXPECT_TRUE(state.combined_model->means_raw().is_valid());
        EXPECT_EQ(state.combined_model->size(), 0u);
        ASSERT_NE(state.point_cloud, nullptr);
        EXPECT_EQ(state.point_cloud->size(), 1);
        EXPECT_EQ(state.point_cloud_transform,
                  lfs::rendering::dataWorldTransformToVisualizerWorld(glm::mat4(1.0f)));
    }

    TEST_F(SceneManagerRenderStateTest, HiddenDatasetTrainingModelStaysResidentAndIsCulledByMask) {
        SceneManager manager;
        manager.changeContentType(SceneManager::ContentType::Dataset);

        auto& scene = manager.getScene();
        scene.addSplat("Model", makeTestSplat(0.0f));
        scene.setTrainingModelNode("Model");

        scene.setNodeVisibility("Model", false);

        const auto state = manager.buildRenderState();
        ASSERT_NE(state.combined_model, nullptr);
        EXPECT_EQ(state.combined_model->size(), 1u);
        EXPECT_EQ(state.visible_splat_count, 0u);
        ASSERT_EQ(state.node_visibility_mask.size(), 1u);
        EXPECT_FALSE(state.node_visibility_mask[0]);
        EXPECT_EQ(manager.getModelForRendering(), state.combined_model);
    }

    TEST_F(SceneManagerRenderStateTest, PointCloudTransformIsTrackedSeparatelyFromModelTransforms) {
        SceneManager manager;
        auto& scene = manager.getScene();

        scene.addPointCloud("PointCloud", makeTestPointCloud());
        scene.setNodeTransform(
            "PointCloud",
            glm::translate(glm::mat4(1.0f), glm::vec3(3.0f, -2.0f, 5.0f)));

        const auto state = manager.buildRenderState();
        ASSERT_NE(state.point_cloud, nullptr);
        EXPECT_TRUE(state.model_transforms.empty());
        expectVisualizerTranslationFromData(state.point_cloud_transform, {3.0f, -2.0f, 5.0f});
    }

    TEST_F(SceneManagerRenderStateTest, VisiblePointCloudDoesNotPolluteModelTransformArray) {
        SceneManager manager;
        auto& scene = manager.getScene();

        scene.addPointCloud("PointCloud", makeTestPointCloud());
        scene.setNodeTransform(
            "PointCloud",
            glm::translate(glm::mat4(1.0f), glm::vec3(9.0f, 8.0f, 7.0f)));
        scene.addSplat("Model", makeTestSplat(0.0f));
        scene.setNodeTransform(
            "Model",
            glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 2.0f, 3.0f)));

        const auto state = manager.buildRenderState();
        ASSERT_EQ(state.model_transforms.size(), 1u);
        expectVisualizerTranslationFromData(state.model_transforms[0], {1.0f, 2.0f, 3.0f});
    }

    TEST_F(SceneManagerRenderStateTest, MultipleVisiblePointCloudsAreMergedAcrossParentTransforms) {
        SceneManager manager;
        auto& scene = manager.getScene();

        const auto left_parent = scene.addGroup("LeftParent");
        const auto right_parent = scene.addGroup("RightParent");

        scene.setNodeTransform(
            "LeftParent",
            glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 2.0f, 3.0f)));
        scene.setNodeTransform(
            "RightParent",
            glm::translate(glm::mat4(1.0f), glm::vec3(-4.0f, 5.0f, 6.0f)));

        auto left_means = lfs::core::Tensor::from_vector(
            {0.0f, 0.0f, 0.0f,
             1.0f, 0.0f, 0.0f},
            {size_t{2}, size_t{3}},
            lfs::core::Device::CPU);
        auto left_colors = lfs::core::Tensor::from_vector(
            {1.0f, 0.0f, 0.0f,
             0.0f, 1.0f, 0.0f},
            {size_t{2}, size_t{3}},
            lfs::core::Device::CPU);
        scene.addPointCloud(
            "LeftCloud",
            std::make_shared<lfs::core::PointCloud>(std::move(left_means), std::move(left_colors)),
            left_parent);

        auto right_means = lfs::core::Tensor::from_vector(
            {0.0f, 1.0f, 0.0f},
            {size_t{1}, size_t{3}},
            lfs::core::Device::CPU);
        auto right_colors = lfs::core::Tensor::from_vector(
            {0.0f, 0.0f, 1.0f},
            {size_t{1}, size_t{3}},
            lfs::core::Device::CPU);
        scene.addPointCloud(
            "RightCloud",
            std::make_shared<lfs::core::PointCloud>(std::move(right_means), std::move(right_colors)),
            right_parent);

        const auto state = manager.buildRenderState();
        ASSERT_NE(state.point_cloud, nullptr);
        EXPECT_EQ(state.point_cloud->size(), 3);
        EXPECT_TRUE(state.model_transforms.empty());
        EXPECT_EQ(state.point_cloud_transform,
                  lfs::rendering::dataWorldTransformToVisualizerWorld(glm::mat4(1.0f)));

        auto means_cpu = state.point_cloud->means.cpu();
        auto acc = means_cpu.accessor<float, 2>();
        EXPECT_FLOAT_EQ(acc(0, 0), 1.0f);
        EXPECT_FLOAT_EQ(acc(0, 1), 2.0f);
        EXPECT_FLOAT_EQ(acc(0, 2), 3.0f);
        EXPECT_FLOAT_EQ(acc(1, 0), 2.0f);
        EXPECT_FLOAT_EQ(acc(1, 1), 2.0f);
        EXPECT_FLOAT_EQ(acc(1, 2), 3.0f);
        EXPECT_FLOAT_EQ(acc(2, 0), -4.0f);
        EXPECT_FLOAT_EQ(acc(2, 1), 6.0f);
        EXPECT_FLOAT_EQ(acc(2, 2), 6.0f);
    }

    TEST_F(SceneManagerRenderStateTest, MultipleVisiblePointCloudMergeRefreshesWhenSourceDataChanges) {
        SceneManager manager;
        auto& scene = manager.getScene();

        const auto left_parent = scene.addGroup("LeftParent");
        const auto right_parent = scene.addGroup("RightParent");

        auto left_point_cloud = std::make_shared<lfs::core::PointCloud>(
            lfs::core::Tensor::from_vector(
                {0.0f, 0.0f, 0.0f},
                {size_t{1}, size_t{3}},
                lfs::core::Device::CPU),
            lfs::core::Tensor::from_vector(
                {1.0f, 0.0f, 0.0f},
                {size_t{1}, size_t{3}},
                lfs::core::Device::CPU));
        auto right_point_cloud = std::make_shared<lfs::core::PointCloud>(
            lfs::core::Tensor::from_vector(
                {1.0f, 1.0f, 1.0f},
                {size_t{1}, size_t{3}},
                lfs::core::Device::CPU),
            lfs::core::Tensor::from_vector(
                {0.0f, 1.0f, 0.0f},
                {size_t{1}, size_t{3}},
                lfs::core::Device::CPU));

        scene.addPointCloud("LeftCloud", left_point_cloud, left_parent);
        scene.addPointCloud("RightCloud", right_point_cloud, right_parent);

        const auto initial_state = manager.buildRenderState();
        ASSERT_NE(initial_state.point_cloud, nullptr);
        ASSERT_EQ(initial_state.point_cloud->size(), 2);

        right_point_cloud->means = lfs::core::Tensor::from_vector(
            {10.0f, 20.0f, 30.0f},
            {size_t{1}, size_t{3}},
            lfs::core::Device::CPU);

        const auto updated_state = manager.buildRenderState();
        ASSERT_NE(updated_state.point_cloud, nullptr);
        ASSERT_EQ(updated_state.point_cloud->size(), 2);

        auto means_cpu = updated_state.point_cloud->means.cpu();
        auto acc = means_cpu.accessor<float, 2>();
        EXPECT_FLOAT_EQ(acc(0, 0), 0.0f);
        EXPECT_FLOAT_EQ(acc(0, 1), 0.0f);
        EXPECT_FLOAT_EQ(acc(0, 2), 0.0f);
        EXPECT_FLOAT_EQ(acc(1, 0), 10.0f);
        EXPECT_FLOAT_EQ(acc(1, 1), 20.0f);
        EXPECT_FLOAT_EQ(acc(1, 2), 30.0f);
    }

    TEST_F(SceneManagerRenderStateTest, MultipleVisiblePointCloudMergeRefreshesWhenSourceTensorChangesInPlace) {
        SceneManager manager;
        auto& scene = manager.getScene();

        const auto left_parent = scene.addGroup("LeftParent");
        const auto right_parent = scene.addGroup("RightParent");

        auto left_point_cloud = std::make_shared<lfs::core::PointCloud>(
            lfs::core::Tensor::from_vector(
                {0.0f, 0.0f, 0.0f},
                {size_t{1}, size_t{3}},
                lfs::core::Device::CPU),
            lfs::core::Tensor::from_vector(
                {1.0f, 0.0f, 0.0f},
                {size_t{1}, size_t{3}},
                lfs::core::Device::CPU));
        auto right_point_cloud = std::make_shared<lfs::core::PointCloud>(
            lfs::core::Tensor::from_vector(
                {1.0f, 1.0f, 1.0f},
                {size_t{1}, size_t{3}},
                lfs::core::Device::CPU),
            lfs::core::Tensor::from_vector(
                {0.0f, 1.0f, 0.0f},
                {size_t{1}, size_t{3}},
                lfs::core::Device::CPU));

        scene.addPointCloud("LeftCloud", left_point_cloud, left_parent);
        scene.addPointCloud("RightCloud", right_point_cloud, right_parent);

        const auto initial_state = manager.buildRenderState();
        ASSERT_NE(initial_state.point_cloud, nullptr);
        ASSERT_EQ(initial_state.point_cloud->size(), 2);

        right_point_cloud->means.copy_(lfs::core::Tensor::from_vector(
            {7.0f, 8.0f, 9.0f},
            {size_t{1}, size_t{3}},
            lfs::core::Device::CPU));

        const auto updated_state = manager.buildRenderState();
        ASSERT_NE(updated_state.point_cloud, nullptr);
        ASSERT_EQ(updated_state.point_cloud->size(), 2);

        auto means_cpu = updated_state.point_cloud->means.cpu();
        auto acc = means_cpu.accessor<float, 2>();
        EXPECT_FLOAT_EQ(acc(0, 0), 0.0f);
        EXPECT_FLOAT_EQ(acc(0, 1), 0.0f);
        EXPECT_FLOAT_EQ(acc(0, 2), 0.0f);
        EXPECT_FLOAT_EQ(acc(1, 0), 7.0f);
        EXPECT_FLOAT_EQ(acc(1, 1), 8.0f);
        EXPECT_FLOAT_EQ(acc(1, 2), 9.0f);
    }

    TEST_F(SceneManagerRenderStateTest, PlyComparisonBuildsFullFrameWipeFromCombinedSceneMasks) {
        SceneManager manager;
        manager.changeContentType(SceneManager::ContentType::SplatFiles);

        auto& scene = manager.getScene();
        const auto left_id = scene.addSplat("left", makeTestSplat(0.0f));
        const auto right_id = scene.addSplat("right", makeTestSplat(1.0f));

        RenderSettings settings;
        settings.split_view_mode = SplitViewMode::PLYComparison;
        settings.split_position = 0.35f;
        settings.show_rings = true;
        settings.depth_filter_enabled = true;
        settings.depth_filter_min = {-1.0f, -1.0f, -1.0f};
        settings.depth_filter_max = {1.0f, 1.0f, 1.0f};

        Viewport viewport(640, 480);
        const auto scene_state = manager.buildRenderState();
        ASSERT_NE(scene_state.combined_model, nullptr);

        const FrameContext ctx{
            .viewport = viewport,
            .scene_manager = &manager,
            .model = manager.getModelForRendering(),
            .scene_state = scene_state,
            .settings = settings,
            .render_size = {640, 480},
            .viewport_pos = {0, 0},
        };

        const auto plan = buildSplitViewCompositionPlan(ctx, FrameResources{});
        ASSERT_TRUE(plan.has_value());
        ASSERT_EQ(plan->panels.size(), 2u);

        EXPECT_EQ(plan->panels[0].panel.content.model, ctx.model);
        EXPECT_EQ(plan->panels[1].panel.content.model, ctx.model);
        EXPECT_EQ(plan->panels[0].panel.content.model_transform, glm::mat4(1.0f));
        EXPECT_EQ(plan->panels[1].panel.content.model_transform, glm::mat4(1.0f));

        for (size_t i = 0; i < plan->panels.size(); ++i) {
            const auto& panel = plan->panels[i].panel;
            ASSERT_TRUE(panel.content.gaussian_render.has_value());
            EXPECT_EQ(panel.content.gaussian_render->frame_view.size, ctx.render_size);
            EXPECT_FALSE(panel.presentation.normalize_x_to_panel);
            EXPECT_EQ(panel.content.gaussian_render->scene.model_transforms, &ctx.scene_state.model_transforms);
            EXPECT_EQ(panel.content.gaussian_render->scene.transform_indices, ctx.scene_state.transform_indices);
            ASSERT_EQ(panel.content.gaussian_render->scene.node_visibility_mask.size(), 2u);
            EXPECT_EQ(panel.content.gaussian_render->scene.node_visibility_mask[0], i == 0);
            EXPECT_EQ(panel.content.gaussian_render->scene.node_visibility_mask[1], i == 1);
            EXPECT_TRUE(panel.content.gaussian_render->filters.view_volume.has_value());
            EXPECT_TRUE(panel.content.gaussian_render->overlay.markers.show_rings);
            EXPECT_EQ(panel.content.gaussian_render->overlay.emphasis.mask, ctx.scene_state.selection_mask);
            EXPECT_FALSE(panel.content.gaussian_render->overlay.cursor.enabled);
            EXPECT_EQ(panel.content.gaussian_render->overlay.emphasis.transient_mask.mask, nullptr);
            EXPECT_EQ(panel.content.gaussian_render->overlay.emphasis.focused_gaussian_id, -1);
        }

        EXPECT_EQ(scene.getVisibleNodeIndex(left_id), 0);
        EXPECT_EQ(scene.getVisibleNodeIndex(right_id), 1);
    }

    TEST_F(SceneManagerRenderStateTest, SwitchToEditModePlyComparisonUsesCombinedSceneMasks) {
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::Tensor;

        SceneManager manager;
        manager.changeContentType(SceneManager::ContentType::Dataset);

        auto& scene = manager.getScene();
        scene.addSplat("Model", makeTestSplat(0.0f));
        scene.setTrainingModelNode("Model");

        manager.switchToEditMode();
        const auto trained_id = scene.getNodeIdByName("Trained Model");
        const auto bike_id = scene.addSplat("bike", makeTestSplat(1.0f));

        const auto cropbox_id = scene.getOrCreateCropBoxForSplat(trained_id);
        auto* cropbox = scene.getCropBoxData(cropbox_id);
        ASSERT_NE(cropbox, nullptr);
        cropbox->min = {-1.0f, -1.0f, -1.0f};
        cropbox->max = {1.0f, 1.0f, 1.0f};
        cropbox->enabled = true;

        auto scene_state = manager.buildRenderState();
        scene_state.selection_mask = std::make_shared<Tensor>(
            Tensor::zeros({size_t{2}}, Device::CPU, DataType::UInt8));
        scene_state.selected_node_mask = {true, false};

        Tensor transient_selection =
            Tensor::zeros({size_t{2}}, Device::CPU, DataType::Bool);

        RenderSettings settings;
        settings.split_view_mode = SplitViewMode::PLYComparison;
        settings.split_position = 0.4f;
        settings.show_rings = true;
        settings.depth_filter_enabled = true;
        settings.depth_filter_min = {-2.0f, -2.0f, -2.0f};
        settings.depth_filter_max = {2.0f, 2.0f, 2.0f};
        settings.desaturate_unselected = true;

        Viewport viewport(640, 480);
        const FrameContext ctx{
            .viewport = viewport,
            .scene_manager = &manager,
            .model = manager.getModelForRendering(),
            .scene_state = std::move(scene_state),
            .settings = settings,
            .render_size = {640, 480},
            .viewport_pos = {0, 0},
            .cursor_preview =
                {.active = true,
                 .x = 32.0f,
                 .y = 24.0f,
                 .radius = 10.0f,
                 .add_mode = true,
                 .selection_tensor = &transient_selection,
                 .preview_selection = &transient_selection,
                 .focused_gaussian_id = 0},
        };

        const auto plan = buildSplitViewCompositionPlan(ctx, FrameResources{});
        ASSERT_TRUE(plan.has_value());
        ASSERT_EQ(plan->panels.size(), 2u);

        for (size_t i = 0; i < plan->panels.size(); ++i) {
            const auto& panel = plan->panels[i].panel;
            ASSERT_TRUE(panel.content.gaussian_render.has_value());
            EXPECT_EQ(panel.content.model, ctx.model);
            EXPECT_EQ(panel.content.model_transform, glm::mat4(1.0f));
            EXPECT_EQ(panel.content.gaussian_render->scene.model_transforms, &ctx.scene_state.model_transforms);
            EXPECT_EQ(panel.content.gaussian_render->scene.transform_indices, ctx.scene_state.transform_indices);
            ASSERT_EQ(panel.content.gaussian_render->scene.node_visibility_mask.size(), 2u);
            EXPECT_EQ(panel.content.gaussian_render->scene.node_visibility_mask[0], i == 0);
            EXPECT_EQ(panel.content.gaussian_render->scene.node_visibility_mask[1], i == 1);

            EXPECT_TRUE(panel.content.gaussian_render->filters.crop_region.has_value());
            EXPECT_EQ(panel.content.gaussian_render->filters.crop_region->parent_node_index, 0);
            EXPECT_TRUE(panel.content.gaussian_render->filters.view_volume.has_value());
            EXPECT_TRUE(panel.content.gaussian_render->overlay.markers.show_rings);
            EXPECT_EQ(panel.content.gaussian_render->overlay.emphasis.mask, ctx.scene_state.selection_mask);
            EXPECT_EQ(panel.content.gaussian_render->overlay.emphasis.emphasized_node_mask,
                      ctx.scene_state.selected_node_mask);
            EXPECT_TRUE(panel.content.gaussian_render->overlay.emphasis.dim_non_emphasized);
            EXPECT_FALSE(panel.content.gaussian_render->overlay.cursor.enabled);
            EXPECT_EQ(panel.content.gaussian_render->overlay.emphasis.transient_mask.mask, nullptr);
            EXPECT_EQ(panel.content.gaussian_render->overlay.emphasis.focused_gaussian_id, -1);
        }

        EXPECT_EQ(scene.getVisibleNodeIndex(trained_id), 0);
        EXPECT_EQ(scene.getVisibleNodeIndex(bike_id), 1);
    }

    TEST_F(SceneManagerRenderStateTest, HiddenEnabledCropBoxStillFiltersRender) {
        SceneManager manager;
        auto& scene = manager.getScene();
        const auto model_id = scene.addSplat("Model", makeTestSplat(0.0f));
        const auto cropbox_id = scene.getOrCreateCropBoxForSplat(model_id);
        ASSERT_NE(cropbox_id, lfs::core::NULL_NODE);

        auto* cropbox = scene.getCropBoxData(cropbox_id);
        ASSERT_NE(cropbox, nullptr);
        cropbox->min = {-0.25f, -0.25f, -0.25f};
        cropbox->max = {0.25f, 0.25f, 0.25f};
        cropbox->enabled = true;
        scene.setNodeVisibility(cropbox_id, false);

        Viewport viewport(640, 480);
        RenderSettings settings;
        const auto scene_state = manager.buildRenderState();
        const FrameContext ctx{
            .viewport = viewport,
            .scene_manager = &manager,
            .model = manager.getModelForRendering(),
            .scene_state = scene_state,
            .settings = settings,
            .render_size = {640, 480},
        };

        const auto request = buildViewportRenderRequest(ctx, {640, 480});
        ASSERT_TRUE(request.filters.crop_region.has_value());
        EXPECT_EQ(request.filters.crop_region->bounds.min, cropbox->min);
        EXPECT_EQ(request.filters.crop_region->bounds.max, cropbox->max);
    }

    TEST_F(SceneManagerRenderStateTest, CropBoxWireframeSelectionHonorsParentSelectionAndVisibility) {
        SceneManager manager;
        auto& scene = manager.getScene();
        const auto model_a_id = scene.addSplat("ModelA", makeTestSplat(0.0f));
        const auto model_b_id = scene.addSplat("ModelB", makeTestSplat(2.0f));
        ASSERT_NE(model_a_id, lfs::core::NULL_NODE);
        ASSERT_NE(model_b_id, lfs::core::NULL_NODE);

        const auto cropbox_id = scene.getOrCreateCropBoxForSplat(model_a_id);
        ASSERT_NE(cropbox_id, lfs::core::NULL_NODE);
        auto* cropbox = scene.getCropBoxData(cropbox_id);
        ASSERT_NE(cropbox, nullptr);
        cropbox->enabled = true;

        manager.selectNode("ModelB");
        auto state = manager.buildRenderState();
        EXPECT_EQ(manager.getSelectedNodeCropBoxId(), lfs::core::NULL_NODE);
        EXPECT_EQ(manager.getActiveSelectionCropBoxId(), lfs::core::NULL_NODE);
        EXPECT_EQ(state.selected_cropbox_index, -1);
        ASSERT_EQ(state.cropboxes.size(), 1u);
        EXPECT_TRUE(state.cropboxes.front().effectively_visible);

        Viewport viewport(640, 480);
        FrameContext ctx{
            .viewport = viewport,
            .scene_manager = &manager,
            .model = manager.getModelForRendering(),
            .scene_state = state,
            .settings = RenderSettings{},
            .render_size = {640, 480},
        };
        auto request = buildViewportRenderRequest(ctx, {640, 480});
        ASSERT_TRUE(request.filters.crop_region.has_value());
        ASSERT_EQ(request.filters.crop_regions.size(), 1u);
        EXPECT_EQ(request.filters.crop_regions.front().parent_node_index, scene.getVisibleNodeIndex(model_a_id));

        manager.selectNode("ModelA");
        state = manager.buildRenderState();
        EXPECT_EQ(manager.getSelectedNodeCropBoxId(), cropbox_id);
        EXPECT_EQ(manager.getActiveSelectionCropBoxId(), cropbox_id);
        ASSERT_GE(state.selected_cropbox_index, 0);
        ASSERT_LT(static_cast<size_t>(state.selected_cropbox_index), state.cropboxes.size());
        EXPECT_EQ(state.cropboxes[static_cast<size_t>(state.selected_cropbox_index)].node_id, cropbox_id);
        EXPECT_TRUE(state.cropboxes[static_cast<size_t>(state.selected_cropbox_index)].effectively_visible);

        scene.setNodeVisibility(cropbox_id, false);
        state = manager.buildRenderState();
        EXPECT_EQ(manager.getSelectedNodeCropBoxId(), cropbox_id);
        EXPECT_EQ(manager.getActiveSelectionCropBoxId(), cropbox_id);
        ASSERT_EQ(state.cropboxes.size(), 1u);
        EXPECT_EQ(state.cropboxes.front().node_id, cropbox_id);
        EXPECT_FALSE(state.cropboxes.front().effectively_visible);

        ctx.scene_state = state;
        request = buildViewportRenderRequest(ctx, {640, 480});
        ASSERT_TRUE(request.filters.crop_region.has_value());
    }

    TEST_F(SceneManagerRenderStateTest, EnabledCropBoxesRemainScopedToTheirParents) {
        SceneManager manager;
        auto& scene = manager.getScene();
        const auto model_a_id = scene.addSplat("ModelA", makeTestSplat(0.0f));
        const auto model_b_id = scene.addSplat("ModelB", makeTestSplat(2.0f));
        ASSERT_NE(model_a_id, lfs::core::NULL_NODE);
        ASSERT_NE(model_b_id, lfs::core::NULL_NODE);

        const auto cropbox_a_id = scene.getOrCreateCropBoxForSplat(model_a_id);
        ASSERT_NE(cropbox_a_id, lfs::core::NULL_NODE);
        auto* cropbox_a = scene.getCropBoxData(cropbox_a_id);
        ASSERT_NE(cropbox_a, nullptr);
        cropbox_a->min = {-2.0f, -2.0f, -2.0f};
        cropbox_a->max = {-1.0f, -1.0f, -1.0f};
        cropbox_a->enabled = true;

        Viewport viewport(640, 480);
        RenderSettings settings;
        settings.desaturate_cropping = true;

        manager.selectNode("ModelB");
        FrameContext ctx{
            .viewport = viewport,
            .scene_manager = &manager,
            .model = manager.getModelForRendering(),
            .scene_state = manager.buildRenderState(),
            .settings = settings,
            .render_size = {640, 480},
        };
        auto request = buildViewportRenderRequest(ctx, {640, 480});
        ASSERT_TRUE(request.filters.crop_region.has_value());
        ASSERT_EQ(request.filters.crop_regions.size(), 1u);
        EXPECT_EQ(request.filters.crop_regions.front().bounds.min, cropbox_a->min);
        EXPECT_EQ(request.filters.crop_regions.front().bounds.max, cropbox_a->max);
        EXPECT_EQ(request.filters.crop_regions.front().parent_node_index, scene.getVisibleNodeIndex(model_a_id));

        const auto cropbox_b_id = scene.getOrCreateCropBoxForSplat(model_b_id);
        ASSERT_NE(cropbox_b_id, lfs::core::NULL_NODE);
        auto* cropbox_b = scene.getCropBoxData(cropbox_b_id);
        ASSERT_NE(cropbox_b, nullptr);
        cropbox_b->min = {1.0f, 1.0f, 1.0f};
        cropbox_b->max = {2.0f, 2.0f, 2.0f};
        cropbox_b->enabled = true;

        ctx.scene_state = manager.buildRenderState();
        request = buildViewportRenderRequest(ctx, {640, 480});
        ASSERT_TRUE(request.filters.crop_region.has_value());
        ASSERT_EQ(request.filters.crop_regions.size(), 2u);
        const auto* filter_a = request.filters.crop_regions[0].parent_node_index == scene.getVisibleNodeIndex(model_a_id)
                                   ? &request.filters.crop_regions[0]
                                   : &request.filters.crop_regions[1];
        const auto* filter_b = request.filters.crop_regions[0].parent_node_index == scene.getVisibleNodeIndex(model_b_id)
                                   ? &request.filters.crop_regions[0]
                                   : &request.filters.crop_regions[1];
        EXPECT_EQ(filter_a->bounds.min, cropbox_a->min);
        EXPECT_EQ(filter_a->bounds.max, cropbox_a->max);
        EXPECT_EQ(filter_b->bounds.min, cropbox_b->min);
        EXPECT_EQ(filter_b->bounds.max, cropbox_b->max);
        EXPECT_TRUE(filter_a->desaturate);
        EXPECT_TRUE(filter_b->desaturate);

        ctx.settings.desaturate_cropping = false;
        request = buildViewportRenderRequest(ctx, {640, 480});
        ASSERT_EQ(request.filters.crop_regions.size(), 2u);
        EXPECT_FALSE(request.filters.crop_regions[0].desaturate);
        EXPECT_FALSE(request.filters.crop_regions[1].desaturate);

        scene.setNodeVisibility(model_a_id, false);
        ctx.scene_state = manager.buildRenderState();
        request = buildViewportRenderRequest(ctx, {640, 480});
        ASSERT_TRUE(request.filters.crop_region.has_value());
        ASSERT_EQ(request.filters.crop_regions.size(), 1u);
        EXPECT_EQ(request.filters.crop_regions.front().parent_node_index, scene.getVisibleNodeIndex(model_b_id));
        EXPECT_EQ(request.filters.crop_regions.front().bounds.min, cropbox_b->min);
        EXPECT_EQ(request.filters.crop_regions.front().bounds.max, cropbox_b->max);
    }

    TEST_F(SceneManagerRenderStateTest, MultipleEnabledCropBoxesWithoutSelectionRemainParentScoped) {
        SceneManager manager;
        auto& scene = manager.getScene();
        const auto model_a_id = scene.addSplat("ModelA", makeTestSplat(0.0f));
        const auto model_b_id = scene.addSplat("ModelB", makeTestSplat(2.0f));
        ASSERT_NE(model_a_id, lfs::core::NULL_NODE);
        ASSERT_NE(model_b_id, lfs::core::NULL_NODE);

        const auto cropbox_a_id = scene.getOrCreateCropBoxForSplat(model_a_id);
        const auto cropbox_b_id = scene.getOrCreateCropBoxForSplat(model_b_id);
        ASSERT_NE(cropbox_a_id, lfs::core::NULL_NODE);
        ASSERT_NE(cropbox_b_id, lfs::core::NULL_NODE);
        ASSERT_NE(scene.getCropBoxData(cropbox_a_id), nullptr);
        ASSERT_NE(scene.getCropBoxData(cropbox_b_id), nullptr);
        scene.getCropBoxData(cropbox_a_id)->enabled = true;
        scene.getCropBoxData(cropbox_b_id)->enabled = true;
        manager.clearSelection();

        Viewport viewport(640, 480);
        const FrameContext ctx{
            .viewport = viewport,
            .scene_manager = &manager,
            .model = manager.getModelForRendering(),
            .scene_state = manager.buildRenderState(),
            .settings = RenderSettings{},
            .render_size = {640, 480},
        };

        const auto request = buildViewportRenderRequest(ctx, {640, 480});
        ASSERT_TRUE(request.filters.crop_region.has_value());
        ASSERT_EQ(request.filters.crop_regions.size(), 2u);
        EXPECT_NE(request.filters.crop_regions[0].parent_node_index, request.filters.crop_regions[1].parent_node_index);
        EXPECT_FALSE(request.filters.crop_regions[0].desaturate);
        EXPECT_FALSE(request.filters.crop_regions[1].desaturate);
    }

    TEST_F(SceneManagerRenderStateTest, LiveCropboxPreviewOverridesOnlyEditedParent) {
        SceneManager manager;
        auto& scene = manager.getScene();
        const auto model_a_id = scene.addSplat("ModelA", makeTestSplat(0.0f));
        const auto model_b_id = scene.addSplat("ModelB", makeTestSplat(2.0f));
        ASSERT_NE(model_a_id, lfs::core::NULL_NODE);
        ASSERT_NE(model_b_id, lfs::core::NULL_NODE);

        const auto cropbox_a_id = scene.getOrCreateCropBoxForSplat(model_a_id);
        const auto cropbox_b_id = scene.getOrCreateCropBoxForSplat(model_b_id);
        ASSERT_NE(cropbox_a_id, lfs::core::NULL_NODE);
        ASSERT_NE(cropbox_b_id, lfs::core::NULL_NODE);
        auto* cropbox_a = scene.getCropBoxData(cropbox_a_id);
        auto* cropbox_b = scene.getCropBoxData(cropbox_b_id);
        ASSERT_NE(cropbox_a, nullptr);
        ASSERT_NE(cropbox_b, nullptr);
        cropbox_a->min = {-2.0f, -2.0f, -2.0f};
        cropbox_a->max = {-1.0f, -1.0f, -1.0f};
        cropbox_a->enabled = true;
        cropbox_b->min = {1.0f, 1.0f, 1.0f};
        cropbox_b->max = {2.0f, 2.0f, 2.0f};
        cropbox_b->enabled = true;

        Viewport viewport(640, 480);
        const FrameContext ctx{
            .viewport = viewport,
            .scene_manager = &manager,
            .model = manager.getModelForRendering(),
            .scene_state = manager.buildRenderState(),
            .settings = RenderSettings{},
            .render_size = {640, 480},
            .gizmo = {
                .cropbox_active = true,
                .cropbox_min = {-0.5f, -0.5f, -0.5f},
                .cropbox_max = {0.5f, 0.5f, 0.5f},
                .cropbox_transform = glm::mat4(1.0f),
                .cropbox_affects_render = true,
                .cropbox_parent_node_index = scene.getVisibleNodeIndex(model_a_id),
            },
        };

        const auto request = buildViewportRenderRequest(ctx, {640, 480});
        ASSERT_EQ(request.filters.crop_regions.size(), 2u);
        const auto* filter_a = request.filters.crop_regions[0].parent_node_index == scene.getVisibleNodeIndex(model_a_id)
                                   ? &request.filters.crop_regions[0]
                                   : &request.filters.crop_regions[1];
        const auto* filter_b = request.filters.crop_regions[0].parent_node_index == scene.getVisibleNodeIndex(model_b_id)
                                   ? &request.filters.crop_regions[0]
                                   : &request.filters.crop_regions[1];
        EXPECT_EQ(filter_a->bounds.min, glm::vec3(-0.5f));
        EXPECT_EQ(filter_a->bounds.max, glm::vec3(0.5f));
        EXPECT_EQ(filter_b->bounds.min, cropbox_b->min);
        EXPECT_EQ(filter_b->bounds.max, cropbox_b->max);
    }
    TEST_F(SceneManagerRenderStateTest, ApplyCropBoxTargetsOnlyParentSplat) {
        SceneManager manager;
        services().set(&manager);
        auto& scene = manager.getScene();
        const auto model_a_id = scene.addSplat("ModelA", makeTwoPointTestSplat(0.0f, 2.0f));
        const auto model_b_id = scene.addSplat("ModelB", makeTwoPointTestSplat(0.0f, 2.0f));
        ASSERT_NE(model_a_id, lfs::core::NULL_NODE);
        ASSERT_NE(model_b_id, lfs::core::NULL_NODE);
        auto* model_a = scene.getMutableNode("ModelA");
        auto* model_b = scene.getMutableNode("ModelB");
        ASSERT_NE(model_a, nullptr);
        ASSERT_NE(model_b, nullptr);
        ASSERT_TRUE(model_a->model);
        ASSERT_TRUE(model_b->model);
        ASSERT_EQ(model_a->model->visible_count(), 2u);
        ASSERT_EQ(model_b->model->visible_count(), 2u);
        ASSERT_FALSE(model_a->model->has_deleted_mask());
        ASSERT_FALSE(model_b->model->has_deleted_mask());

        lfs::geometry::BoundingBox crop_box;
        crop_box.setBounds(glm::vec3(-0.5f, -0.5f, -0.5f), glm::vec3(0.5f, 0.5f, 3.0f));
        lfs::core::events::cmd::CropPLY{
            .crop_box = crop_box,
            .inverse = false,
            .target_node_id = static_cast<int32_t>(model_a_id),
        }.emit();

        model_a = scene.getMutableNode("ModelA");
        model_b = scene.getMutableNode("ModelB");
        ASSERT_NE(model_a, nullptr);
        ASSERT_NE(model_b, nullptr);
        ASSERT_TRUE(model_a->model);
        ASSERT_TRUE(model_b->model);
        EXPECT_TRUE(model_a->model->has_deleted_mask());
        EXPECT_FALSE(model_b->model->has_deleted_mask());
        EXPECT_EQ(model_b->model->visible_count(), 2u);
    }
    TEST_F(SceneManagerRenderStateTest, EnsureEllipsoidConvertsExistingCropBoxInPlace) {
        SceneManager manager;
        RenderingManager rendering_manager;
        auto& scene = manager.getScene();
        const auto parent_id = scene.addPointCloud("Model", makeTestPointCloud());
        ASSERT_NE(parent_id, lfs::core::NULL_NODE);

        const auto cropbox_id = scene.addCropBox("Model_cropbox", parent_id);
        ASSERT_NE(cropbox_id, lfs::core::NULL_NODE);
        auto* cropbox_node = scene.getNodeById(cropbox_id);
        ASSERT_NE(cropbox_node, nullptr);
        ASSERT_NE(cropbox_node->cropbox, nullptr);
        cropbox_node->cropbox->min = {-3.0f, -5.0f, -7.0f};
        cropbox_node->cropbox->max = {3.0f, 5.0f, 7.0f};
        cropbox_node->cropbox->enabled = true;
        cropbox_node->cropbox->inverse = true;
        cropbox_node->cropbox->color = {0.25f, 0.5f, 0.75f};
        cropbox_node->cropbox->line_width = 5.0f;
        cropbox_node->cropbox->flash_intensity = 0.4f;
        const glm::mat4 transform =
            glm::rotate(glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 2.0f, 3.0f)),
                        glm::radians(30.0f),
                        glm::vec3(0.0f, 1.0f, 0.0f));
        scene.setNodeTransform(cropbox_node->name, transform);
        scene.setNodeVisibility(cropbox_id, false);
        manager.selectNode(cropbox_node->name);

        auto result = cap::ensureEllipsoid(manager, &rendering_manager, parent_id);
        ASSERT_TRUE(result) << result.error();
        EXPECT_EQ(*result, cropbox_id);
        EXPECT_EQ(scene.getCropBoxForSplat(parent_id), lfs::core::NULL_NODE);
        EXPECT_EQ(scene.getEllipsoidForSplat(parent_id), cropbox_id);

        const auto* converted = scene.getNodeById(cropbox_id);
        ASSERT_NE(converted, nullptr);
        EXPECT_EQ(converted->type, lfs::core::NodeType::ELLIPSOID);
        EXPECT_EQ(converted->name, "Model_cropbox");
        EXPECT_FALSE(converted->cropbox);
        ASSERT_TRUE(converted->ellipsoid);
        EXPECT_EQ(converted->parent_id, parent_id);
        EXPECT_FALSE(converted->visible.get());
        EXPECT_TRUE(converted->ellipsoid->enabled);
        EXPECT_TRUE(converted->ellipsoid->inverse);
        EXPECT_EQ(converted->ellipsoid->color, glm::vec3(0.25f, 0.5f, 0.75f));
        EXPECT_FLOAT_EQ(converted->ellipsoid->line_width, 5.0f);
        EXPECT_FLOAT_EQ(converted->ellipsoid->flash_intensity, 0.4f);
        EXPECT_EQ(converted->ellipsoid->radii, glm::vec3(3.0f, 5.0f, 7.0f));
        EXPECT_EQ(scene.getNodeTransform(converted->name), transform);
        EXPECT_EQ(manager.getSelectedNodeName(), converted->name);

        const auto settings = rendering_manager.getSettings();
        EXPECT_FALSE(settings.show_crop_box);
        EXPECT_FALSE(settings.use_crop_box);
        EXPECT_FALSE(settings.show_ellipsoid);
        EXPECT_TRUE(settings.use_ellipsoid);

        const auto state = manager.buildRenderState();
        EXPECT_TRUE(state.cropboxes.empty());
        ASSERT_EQ(state.ellipsoids.size(), 1u);
        EXPECT_EQ(state.ellipsoids.front().node_id, cropbox_id);
    }

    TEST_F(SceneManagerRenderStateTest, DefaultCropBoxConvertsToEllipsoidAtCropCenter) {
        SceneManager manager;
        RenderingManager rendering_manager;
        auto& scene = manager.getScene();
        const auto parent_id = scene.addPointCloud("Model", makeTestPointCloud());
        ASSERT_NE(parent_id, lfs::core::NULL_NODE);

        auto cropbox_result = cap::ensureCropBox(manager, &rendering_manager, parent_id);
        ASSERT_TRUE(cropbox_result) << cropbox_result.error();
        const auto cropbox_id = *cropbox_result;
        const auto* cropbox_node = scene.getNodeById(cropbox_id);
        ASSERT_NE(cropbox_node, nullptr);
        ASSERT_TRUE(cropbox_node->cropbox);
        EXPECT_EQ(cropbox_node->cropbox->min, glm::vec3(-0.5f, -1e-4f, -1e-4f));
        EXPECT_EQ(cropbox_node->cropbox->max, glm::vec3(0.5f, 1e-4f, 1e-4f));
        const glm::mat4 cropbox_transform = scene.getNodeTransform(cropbox_node->name);
        EXPECT_EQ(glm::vec3(cropbox_transform[3]), glm::vec3(0.5f, 0.0f, 0.0f));

        auto ellipsoid_result = cap::ensureEllipsoid(manager, &rendering_manager, parent_id);
        ASSERT_TRUE(ellipsoid_result) << ellipsoid_result.error();
        EXPECT_EQ(*ellipsoid_result, cropbox_id);

        const auto* ellipsoid_node = scene.getNodeById(cropbox_id);
        ASSERT_NE(ellipsoid_node, nullptr);
        EXPECT_EQ(ellipsoid_node->type, lfs::core::NodeType::ELLIPSOID);
        ASSERT_TRUE(ellipsoid_node->ellipsoid);
        EXPECT_EQ(ellipsoid_node->ellipsoid->radii, glm::vec3(0.5f, 1e-4f, 1e-4f));
        EXPECT_EQ(scene.getNodeTransform(ellipsoid_node->name), cropbox_transform);
    }

    TEST_F(SceneManagerRenderStateTest, AddCropCommandsConvertSelectedCropVolumeViaParent) {
        SceneManager manager;
        RenderingManager rendering_manager;
        services().set(&rendering_manager);
        EditorContext editor;
        services().set(&editor);
        auto& scene = manager.getScene();
        const auto parent_id = scene.addPointCloud("Model", makeTestPointCloud());
        ASSERT_NE(parent_id, lfs::core::NULL_NODE);

        auto cropbox_result = cap::ensureCropBox(manager, &rendering_manager, parent_id);
        ASSERT_TRUE(cropbox_result) << cropbox_result.error();
        const auto cropbox_id = *cropbox_result;
        const auto* cropbox_node = scene.getNodeById(cropbox_id);
        ASSERT_NE(cropbox_node, nullptr);
        ASSERT_TRUE(cropbox_node->cropbox);

        lfs::core::events::cmd::AddCropEllipsoid{.node_name = cropbox_node->name}.emit();

        const auto* ellipsoid_node = scene.getNodeById(cropbox_id);
        ASSERT_NE(ellipsoid_node, nullptr);
        EXPECT_EQ(ellipsoid_node->type, lfs::core::NodeType::ELLIPSOID);
        EXPECT_EQ(scene.getEllipsoidForSplat(parent_id), cropbox_id);
        EXPECT_EQ(scene.getCropBoxForSplat(parent_id), lfs::core::NULL_NODE);
        EXPECT_EQ(manager.getSelectedNodeName(), ellipsoid_node->name);
        EXPECT_EQ(editor.getActiveOperator(), "builtin.cropbox");

        lfs::core::events::cmd::AddCropBox{.node_name = ellipsoid_node->name}.emit();

        const auto* converted_node = scene.getNodeById(cropbox_id);
        ASSERT_NE(converted_node, nullptr);
        EXPECT_EQ(converted_node->type, lfs::core::NodeType::CROPBOX);
        EXPECT_EQ(scene.getCropBoxForSplat(parent_id), cropbox_id);
        EXPECT_EQ(scene.getEllipsoidForSplat(parent_id), lfs::core::NULL_NODE);
        EXPECT_EQ(manager.getSelectedNodeName(), converted_node->name);
        EXPECT_EQ(editor.getActiveOperator(), "builtin.cropbox");
    }

    TEST_F(SceneManagerRenderStateTest, AddCropCommandsRevealExistingHiddenCropVolume) {
        SceneManager manager;
        RenderingManager rendering_manager;
        services().set(&rendering_manager);
        auto& scene = manager.getScene();
        const auto parent_id = scene.addPointCloud("Model", makeTestPointCloud());
        ASSERT_NE(parent_id, lfs::core::NULL_NODE);

        auto cropbox_result = cap::ensureCropBox(manager, &rendering_manager, parent_id);
        ASSERT_TRUE(cropbox_result) << cropbox_result.error();
        const auto cropbox_id = *cropbox_result;
        const auto* cropbox_node = scene.getNodeById(cropbox_id);
        ASSERT_NE(cropbox_node, nullptr);

        manager.setNodeVisibility(cropbox_id, false);
        lfs::core::events::cmd::AddCropBox{.node_name = "Model"}.emit();

        cropbox_node = scene.getNodeById(cropbox_id);
        ASSERT_NE(cropbox_node, nullptr);
        EXPECT_TRUE(cropbox_node->visible);
        EXPECT_EQ(cropbox_node->type, lfs::core::NodeType::CROPBOX);
        EXPECT_EQ(manager.getSelectedNodeName(), cropbox_node->name);

        manager.setNodeVisibility(cropbox_id, false);
        lfs::core::events::cmd::AddCropEllipsoid{.node_name = cropbox_node->name}.emit();

        const auto* ellipsoid_node = scene.getNodeById(cropbox_id);
        ASSERT_NE(ellipsoid_node, nullptr);
        EXPECT_TRUE(ellipsoid_node->visible);
        EXPECT_EQ(ellipsoid_node->type, lfs::core::NodeType::ELLIPSOID);
        EXPECT_EQ(scene.getEllipsoidForSplat(parent_id), cropbox_id);
        EXPECT_EQ(manager.getSelectedNodeName(), ellipsoid_node->name);
    }

    TEST_F(SceneManagerRenderStateTest, ResetAndFitPreserveEnabledCropEffects) {
        SceneManager manager;
        RenderingManager rendering_manager;
        services().set(&rendering_manager);
        auto& scene = manager.getScene();
        const auto parent_id = scene.addSplat("Model", makeTestSplat(0.0f));
        ASSERT_NE(parent_id, lfs::core::NULL_NODE);

        auto cropbox_result = cap::ensureCropBox(manager, &rendering_manager, parent_id);
        ASSERT_TRUE(cropbox_result) << cropbox_result.error();
        const auto cropbox_id = *cropbox_result;
        auto* cropbox_node = scene.getNodeById(cropbox_id);
        ASSERT_NE(cropbox_node, nullptr);
        ASSERT_TRUE(cropbox_node->cropbox);
        cropbox_node->cropbox->enabled = true;
        manager.selectNode(cropbox_node->name);

        auto settings = rendering_manager.getSettings();
        settings.use_crop_box = false;
        rendering_manager.updateSettings(settings);
        lfs::core::events::cmd::ResetCropBox{}.emit();

        cropbox_node = scene.getNodeById(cropbox_id);
        ASSERT_NE(cropbox_node, nullptr);
        ASSERT_TRUE(cropbox_node->cropbox);
        EXPECT_TRUE(cropbox_node->cropbox->enabled);
        EXPECT_TRUE(rendering_manager.getSettings().use_crop_box);

        Viewport viewport(640, 480);
        auto scene_state = manager.buildRenderState();
        FrameContext ctx{
            .viewport = viewport,
            .scene_manager = &manager,
            .model = manager.getModelForRendering(),
            .scene_state = scene_state,
            .settings = rendering_manager.getSettings(),
            .render_size = {640, 480},
        };
        auto request = buildViewportRenderRequest(ctx, {640, 480});
        EXPECT_TRUE(request.filters.crop_region.has_value());

        settings = rendering_manager.getSettings();
        settings.use_crop_box = false;
        rendering_manager.updateSettings(settings);
        lfs::core::events::cmd::FitCropBoxToScene{.use_percentile = false}.emit();
        cropbox_node = scene.getNodeById(cropbox_id);
        ASSERT_NE(cropbox_node, nullptr);
        ASSERT_TRUE(cropbox_node->cropbox);
        EXPECT_TRUE(cropbox_node->cropbox->enabled);
        EXPECT_TRUE(rendering_manager.getSettings().use_crop_box);
        scene_state = manager.buildRenderState();
        ctx.scene_state = scene_state;
        ctx.settings = rendering_manager.getSettings();
        request = buildViewportRenderRequest(ctx, {640, 480});
        EXPECT_TRUE(request.filters.crop_region.has_value());

        lfs::core::events::cmd::AddCropEllipsoid{.node_name = cropbox_node->name}.emit();
        auto* ellipsoid_node = scene.getNodeById(cropbox_id);
        ASSERT_NE(ellipsoid_node, nullptr);
        ASSERT_TRUE(ellipsoid_node->ellipsoid);
        ellipsoid_node->ellipsoid->enabled = true;
        settings = rendering_manager.getSettings();
        settings.use_ellipsoid = false;
        rendering_manager.updateSettings(settings);
        lfs::core::events::cmd::ResetEllipsoid{}.emit();

        ellipsoid_node = scene.getNodeById(cropbox_id);
        ASSERT_NE(ellipsoid_node, nullptr);
        ASSERT_TRUE(ellipsoid_node->ellipsoid);
        EXPECT_TRUE(ellipsoid_node->ellipsoid->enabled);
        EXPECT_TRUE(rendering_manager.getSettings().use_ellipsoid);
        scene_state = manager.buildRenderState();
        ctx.scene_state = scene_state;
        ctx.settings = rendering_manager.getSettings();
        request = buildViewportRenderRequest(ctx, {640, 480});
        EXPECT_TRUE(request.filters.ellipsoid_region.has_value());

        settings = rendering_manager.getSettings();
        settings.use_ellipsoid = false;
        rendering_manager.updateSettings(settings);
        lfs::core::events::cmd::FitEllipsoidToScene{.use_percentile = false}.emit();
        ellipsoid_node = scene.getNodeById(cropbox_id);
        ASSERT_NE(ellipsoid_node, nullptr);
        ASSERT_TRUE(ellipsoid_node->ellipsoid);
        EXPECT_TRUE(ellipsoid_node->ellipsoid->enabled);
        EXPECT_TRUE(rendering_manager.getSettings().use_ellipsoid);
        scene_state = manager.buildRenderState();
        ctx.scene_state = scene_state;
        ctx.settings = rendering_manager.getSettings();
        request = buildViewportRenderRequest(ctx, {640, 480});
        EXPECT_TRUE(request.filters.ellipsoid_region.has_value());
    }

    TEST_F(SceneManagerRenderStateTest, DeletingSelectedCropVolumeSelectsParentAndClearsRenderState) {
        SceneManager manager;
        RenderingManager rendering_manager;
        services().set(&manager);
        services().set(&rendering_manager);
        auto& scene = manager.getScene();
        const auto parent_id = scene.addPointCloud("Model", makeTestPointCloud());
        ASSERT_NE(parent_id, lfs::core::NULL_NODE);

        auto cropbox_result = cap::ensureCropBox(manager, &rendering_manager, parent_id);
        ASSERT_TRUE(cropbox_result) << cropbox_result.error();
        const auto cropbox_id = *cropbox_result;
        const auto* cropbox_node = scene.getNodeById(cropbox_id);
        ASSERT_NE(cropbox_node, nullptr);
        manager.selectNode(cropbox_node->name);

        auto settings = rendering_manager.getSettings();
        settings.show_crop_box = true;
        settings.use_crop_box = true;
        rendering_manager.updateSettings(settings);

        manager.removePLY(cropbox_node->name);

        EXPECT_EQ(scene.getCropBoxForSplat(parent_id), lfs::core::NULL_NODE);
        EXPECT_EQ(manager.getSelectedNodeName(), "Model");
        settings = rendering_manager.getSettings();
        EXPECT_FALSE(settings.show_crop_box);
        EXPECT_FALSE(settings.use_crop_box);
        EXPECT_TRUE(manager.buildRenderState().cropboxes.empty());
    }

    TEST_F(SceneManagerRenderStateTest, EnsureCropBoxConvertsExistingEllipsoidAndUndoRedoRestoresShape) {
        SceneManager manager;
        RenderingManager rendering_manager;
        auto& scene = manager.getScene();
        const auto parent_id = scene.addPointCloud("Model", makeTestPointCloud());
        ASSERT_NE(parent_id, lfs::core::NULL_NODE);

        const auto ellipsoid_id = scene.addEllipsoid("Model_ellipsoid", parent_id);
        ASSERT_NE(ellipsoid_id, lfs::core::NULL_NODE);
        auto* ellipsoid_node = scene.getNodeById(ellipsoid_id);
        ASSERT_NE(ellipsoid_node, nullptr);
        ASSERT_NE(ellipsoid_node->ellipsoid, nullptr);
        ellipsoid_node->ellipsoid->radii = {2.0f, 3.0f, 4.0f};
        ellipsoid_node->ellipsoid->enabled = true;
        ellipsoid_node->ellipsoid->inverse = true;
        ellipsoid_node->ellipsoid->color = {0.8f, 0.6f, 0.4f};
        ellipsoid_node->ellipsoid->line_width = 6.0f;
        ellipsoid_node->ellipsoid->flash_intensity = 0.3f;
        const glm::mat4 transform = glm::scale(
            glm::translate(glm::mat4(1.0f), glm::vec3(-1.0f, 2.0f, -3.0f)),
            glm::vec3(1.0f, 2.0f, 3.0f));
        scene.setNodeTransform(ellipsoid_node->name, transform);
        scene.setNodeVisibility(ellipsoid_id, false);
        manager.selectNode(ellipsoid_node->name);

        auto result = cap::ensureCropBox(manager, &rendering_manager, parent_id);
        ASSERT_TRUE(result) << result.error();
        EXPECT_EQ(*result, ellipsoid_id);
        EXPECT_EQ(scene.getCropBoxForSplat(parent_id), ellipsoid_id);
        EXPECT_EQ(scene.getEllipsoidForSplat(parent_id), lfs::core::NULL_NODE);

        auto* converted = scene.getNodeById(ellipsoid_id);
        ASSERT_NE(converted, nullptr);
        EXPECT_EQ(converted->type, lfs::core::NodeType::CROPBOX);
        EXPECT_EQ(converted->name, "Model_ellipsoid");
        ASSERT_TRUE(converted->cropbox);
        EXPECT_FALSE(converted->ellipsoid);
        EXPECT_EQ(converted->cropbox->min, glm::vec3(-2.0f, -3.0f, -4.0f));
        EXPECT_EQ(converted->cropbox->max, glm::vec3(2.0f, 3.0f, 4.0f));
        EXPECT_TRUE(converted->cropbox->enabled);
        EXPECT_TRUE(converted->cropbox->inverse);
        EXPECT_EQ(converted->cropbox->color, glm::vec3(0.8f, 0.6f, 0.4f));
        EXPECT_FLOAT_EQ(converted->cropbox->line_width, 6.0f);
        EXPECT_FLOAT_EQ(converted->cropbox->flash_intensity, 0.3f);
        EXPECT_EQ(scene.getNodeTransform(converted->name), transform);
        EXPECT_EQ(manager.getSelectedNodeName(), converted->name);

        auto settings = rendering_manager.getSettings();
        EXPECT_FALSE(settings.show_ellipsoid);
        EXPECT_FALSE(settings.use_ellipsoid);
        EXPECT_FALSE(settings.show_crop_box);
        EXPECT_TRUE(settings.use_crop_box);

        auto state = manager.buildRenderState();
        ASSERT_EQ(state.cropboxes.size(), 1u);
        EXPECT_TRUE(state.ellipsoids.empty());

        auto undo_result = op::undoHistory().undo();
        ASSERT_TRUE(undo_result.success);
        const auto* undone = scene.getNode("Model_ellipsoid");
        ASSERT_NE(undone, nullptr);
        EXPECT_EQ(undone->type, lfs::core::NodeType::ELLIPSOID);
        ASSERT_TRUE(undone->ellipsoid);
        EXPECT_FALSE(undone->cropbox);
        EXPECT_EQ(undone->ellipsoid->radii, glm::vec3(2.0f, 3.0f, 4.0f));
        EXPECT_TRUE(undone->ellipsoid->enabled);
        EXPECT_TRUE(undone->ellipsoid->inverse);
        EXPECT_FALSE(undone->visible.get());
        EXPECT_EQ(manager.getSelectedNodeName(), "Model_ellipsoid");

        auto redo_result = op::undoHistory().redo();
        ASSERT_TRUE(redo_result.success);
        const auto* redone = scene.getNode("Model_ellipsoid");
        ASSERT_NE(redone, nullptr);
        EXPECT_EQ(redone->type, lfs::core::NodeType::CROPBOX);
        ASSERT_TRUE(redone->cropbox);
        EXPECT_FALSE(redone->ellipsoid);
        EXPECT_EQ(redone->cropbox->min, glm::vec3(-2.0f, -3.0f, -4.0f));
        EXPECT_EQ(redone->cropbox->max, glm::vec3(2.0f, 3.0f, 4.0f));
        EXPECT_TRUE(redone->cropbox->enabled);
        EXPECT_TRUE(redone->cropbox->inverse);
    }

    TEST_F(SceneManagerRenderStateTest, PointCloudRequestUsesSelectedEnabledEllipsoidCrop) {
        SceneManager manager;
        auto& scene = manager.getScene();
        const auto parent_id = scene.addPointCloud("Model", makeTestPointCloud());
        ASSERT_NE(parent_id, lfs::core::NULL_NODE);

        auto ellipsoid_result = cap::ensureEllipsoid(manager, nullptr, parent_id);
        ASSERT_TRUE(ellipsoid_result) << ellipsoid_result.error();
        auto* ellipsoid_node = scene.getNodeById(*ellipsoid_result);
        ASSERT_NE(ellipsoid_node, nullptr);
        ASSERT_TRUE(ellipsoid_node->ellipsoid);
        ellipsoid_node->ellipsoid->enabled = true;
        ellipsoid_node->ellipsoid->inverse = true;
        ellipsoid_node->ellipsoid->radii = {2.0f, 3.0f, 4.0f};
        const auto transform = glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 2.0f, 3.0f));
        scene.setNodeTransform(ellipsoid_node->name, transform);
        manager.selectNode(ellipsoid_node->name);

        Viewport viewport(640, 480);
        auto scene_state = manager.buildRenderState();
        ASSERT_EQ(scene_state.ellipsoids.size(), 1u);
        RenderSettings settings;
        const FrameContext ctx{
            .viewport = viewport,
            .scene_manager = &manager,
            .scene_state = scene_state,
            .settings = settings,
            .render_size = {640, 480},
        };
        const std::vector<glm::mat4> transforms = scene_state.model_transforms;

        const auto request = buildPointCloudRenderRequest(ctx, {640, 480}, transforms);

        ASSERT_TRUE(request.filters.crop_ellipsoid.has_value());
        EXPECT_FALSE(request.filters.crop_box.has_value());
        EXPECT_EQ(request.filters.crop_ellipsoid->radii, glm::vec3(2.0f, 3.0f, 4.0f));
        EXPECT_EQ(request.filters.crop_ellipsoid->transform,
                  glm::inverse(scene_state.ellipsoids.front().world_transform));
        EXPECT_TRUE(request.filters.crop_inverse);
        EXPECT_FALSE(request.filters.crop_desaturate);
    }

    TEST_F(SceneManagerRenderStateTest, PointCloudRequestIgnoresDisabledSelectedEllipsoidCrop) {
        SceneManager manager;
        auto& scene = manager.getScene();
        const auto parent_id = scene.addPointCloud("Model", makeTestPointCloud());
        ASSERT_NE(parent_id, lfs::core::NULL_NODE);

        auto ellipsoid_result = cap::ensureEllipsoid(manager, nullptr, parent_id);
        ASSERT_TRUE(ellipsoid_result) << ellipsoid_result.error();
        auto* ellipsoid_node = scene.getNodeById(*ellipsoid_result);
        ASSERT_NE(ellipsoid_node, nullptr);
        ASSERT_TRUE(ellipsoid_node->ellipsoid);
        ellipsoid_node->ellipsoid->enabled = false;
        manager.selectNode(ellipsoid_node->name);

        Viewport viewport(640, 480);
        auto scene_state = manager.buildRenderState();
        RenderSettings settings;
        const FrameContext ctx{
            .viewport = viewport,
            .scene_manager = &manager,
            .scene_state = scene_state,
            .settings = settings,
            .render_size = {640, 480},
        };
        const std::vector<glm::mat4> transforms = scene_state.model_transforms;

        const auto request = buildPointCloudRenderRequest(ctx, {640, 480}, transforms);

        EXPECT_FALSE(request.filters.crop_ellipsoid.has_value());
        EXPECT_FALSE(request.filters.crop_box.has_value());
    }

    TEST_F(SceneManagerRenderStateTest, DeleteSelectedGaussiansRejectsSelectedCropVolumeWithoutRemovingIt) {
        SceneManager manager;
        auto& scene = manager.getScene();
        const auto parent_id = scene.addPointCloud("Model", makeTestPointCloud());
        ASSERT_NE(parent_id, lfs::core::NULL_NODE);

        auto cropbox_result = cap::ensureCropBox(manager, nullptr, parent_id);
        ASSERT_TRUE(cropbox_result) << cropbox_result.error();
        const auto cropbox_id = *cropbox_result;
        auto* cropbox_node = scene.getNodeById(cropbox_id);
        ASSERT_NE(cropbox_node, nullptr);
        const auto cropbox_name = cropbox_node->name;
        manager.selectNode(cropbox_name);

        const auto result = manager.deleteSelectedGaussiansWithHistory();

        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error(), "Use the Crop toolbar Delete action to remove selected crop volumes");
        EXPECT_NE(scene.getNodeById(cropbox_id), nullptr);
        EXPECT_EQ(manager.getSelectedNodeName(), cropbox_name);
    }


    TEST_F(SceneManagerRenderStateTest, PointCloudRequestKeepsSingleEnabledCropBoxAfterDeselection) {
        SceneManager manager;
        auto& scene = manager.getScene();
        const auto parent_id = scene.addPointCloud("Model", makeTestPointCloud());
        ASSERT_NE(parent_id, lfs::core::NULL_NODE);

        auto cropbox_result = cap::ensureCropBox(manager, nullptr, parent_id);
        ASSERT_TRUE(cropbox_result) << cropbox_result.error();
        auto* cropbox_node = scene.getNodeById(*cropbox_result);
        ASSERT_NE(cropbox_node, nullptr);
        ASSERT_TRUE(cropbox_node->cropbox);
        cropbox_node->cropbox->enabled = true;
        cropbox_node->cropbox->inverse = true;
        cropbox_node->cropbox->min = {-1.0f, -2.0f, -3.0f};
        cropbox_node->cropbox->max = {1.0f, 2.0f, 3.0f};
        scene.setNodeTransform(cropbox_node->name, glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, 3.0f, 4.0f)));
        scene.setNodeVisibility(*cropbox_result, false);
        manager.clearSelection();

        Viewport viewport(640, 480);
        auto scene_state = manager.buildRenderState();
        ASSERT_EQ(scene_state.cropboxes.size(), 1u);
        EXPECT_LT(scene_state.cropboxes.front().parent_node_index, 0);
        EXPECT_FALSE(scene_state.cropboxes.front().effectively_visible);
        EXPECT_TRUE(scene_state.cropboxes.front().parent_effectively_visible);
        RenderSettings settings;
        settings.desaturate_cropping = false;
        const FrameContext ctx{
            .viewport = viewport,
            .scene_manager = &manager,
            .scene_state = scene_state,
            .settings = settings,
            .render_size = {640, 480},
        };
        const std::vector<glm::mat4> transforms = scene_state.model_transforms;

        const auto request = buildPointCloudRenderRequest(ctx, {640, 480}, transforms);

        ASSERT_TRUE(request.filters.crop_box.has_value());
        EXPECT_FALSE(request.filters.crop_ellipsoid.has_value());
        EXPECT_EQ(request.filters.crop_box->min, glm::vec3(-1.0f, -2.0f, -3.0f));
        EXPECT_EQ(request.filters.crop_box->max, glm::vec3(1.0f, 2.0f, 3.0f));
        EXPECT_EQ(request.filters.crop_box->transform, glm::inverse(scene_state.cropboxes.front().world_transform));
        EXPECT_TRUE(request.filters.crop_inverse);
        EXPECT_FALSE(request.filters.crop_desaturate);
    }

    TEST_F(SceneManagerRenderStateTest, PointCloudRequestIgnoresHiddenSplatCropBoxFallback) {
        SceneManager manager;
        auto& scene = manager.getScene();
        const auto hidden_splat_id = scene.addSplat("HiddenSplat", makeTestSplat(0.0f));
        ASSERT_NE(hidden_splat_id, lfs::core::NULL_NODE);
        const auto point_cloud_id = scene.addPointCloud("VisiblePoints", makeTestPointCloud());
        ASSERT_NE(point_cloud_id, lfs::core::NULL_NODE);

        auto cropbox_result = cap::ensureCropBox(manager, nullptr, hidden_splat_id);
        ASSERT_TRUE(cropbox_result) << cropbox_result.error();
        auto* cropbox_node = scene.getNodeById(*cropbox_result);
        ASSERT_NE(cropbox_node, nullptr);
        ASSERT_TRUE(cropbox_node->cropbox);
        cropbox_node->cropbox->enabled = true;
        scene.setNodeVisibility("HiddenSplat", false);
        manager.clearSelection();

        Viewport viewport(640, 480);
        auto scene_state = manager.buildRenderState();
        ASSERT_EQ(scene_state.cropboxes.size(), 1u);
        EXPECT_FALSE(scene_state.cropboxes.front().effectively_visible);
        EXPECT_FALSE(scene_state.cropboxes.front().parent_effectively_visible);
        RenderSettings settings;
        const FrameContext ctx{
            .viewport = viewport,
            .scene_manager = &manager,
            .scene_state = scene_state,
            .settings = settings,
            .render_size = {640, 480},
        };
        const std::vector<glm::mat4> transforms = scene_state.model_transforms;

        const auto request = buildPointCloudRenderRequest(ctx, {640, 480}, transforms);

        EXPECT_FALSE(request.filters.crop_box.has_value());
        EXPECT_FALSE(request.filters.crop_ellipsoid.has_value());
    }

    TEST_F(SceneManagerRenderStateTest, PointCloudRequestKeepsSingleEnabledEllipsoidAfterDeselection) {
        SceneManager manager;
        auto& scene = manager.getScene();
        const auto parent_id = scene.addPointCloud("Model", makeTestPointCloud());
        ASSERT_NE(parent_id, lfs::core::NULL_NODE);

        auto ellipsoid_result = cap::ensureEllipsoid(manager, nullptr, parent_id);
        ASSERT_TRUE(ellipsoid_result) << ellipsoid_result.error();
        auto* ellipsoid_node = scene.getNodeById(*ellipsoid_result);
        ASSERT_NE(ellipsoid_node, nullptr);
        ASSERT_TRUE(ellipsoid_node->ellipsoid);
        ellipsoid_node->ellipsoid->enabled = true;
        ellipsoid_node->ellipsoid->inverse = true;
        ellipsoid_node->ellipsoid->radii = {2.0f, 3.0f, 4.0f};
        scene.setNodeTransform(ellipsoid_node->name, glm::translate(glm::mat4(1.0f), glm::vec3(3.0f, 4.0f, 5.0f)));
        scene.setNodeVisibility(*ellipsoid_result, false);
        manager.clearSelection();

        Viewport viewport(640, 480);
        auto scene_state = manager.buildRenderState();
        ASSERT_EQ(scene_state.ellipsoids.size(), 1u);
        EXPECT_LT(scene_state.ellipsoids.front().parent_node_index, 0);
        EXPECT_FALSE(scene_state.ellipsoids.front().effectively_visible);
        EXPECT_TRUE(scene_state.ellipsoids.front().parent_effectively_visible);
        RenderSettings settings;
        settings.desaturate_cropping = false;
        const FrameContext ctx{
            .viewport = viewport,
            .scene_manager = &manager,
            .scene_state = scene_state,
            .settings = settings,
            .render_size = {640, 480},
        };
        const std::vector<glm::mat4> transforms = scene_state.model_transforms;

        const auto request = buildPointCloudRenderRequest(ctx, {640, 480}, transforms);

        ASSERT_TRUE(request.filters.crop_ellipsoid.has_value());
        EXPECT_FALSE(request.filters.crop_box.has_value());
        EXPECT_EQ(request.filters.crop_ellipsoid->radii, glm::vec3(2.0f, 3.0f, 4.0f));
        EXPECT_EQ(request.filters.crop_ellipsoid->transform, glm::inverse(scene_state.ellipsoids.front().world_transform));
        EXPECT_TRUE(request.filters.crop_inverse);
        EXPECT_FALSE(request.filters.crop_desaturate);
    }

    TEST_F(SceneManagerRenderStateTest, PointCloudRequestIgnoresHiddenSplatEllipsoidFallback) {
        SceneManager manager;
        auto& scene = manager.getScene();
        const auto hidden_splat_id = scene.addSplat("HiddenSplat", makeTestSplat(0.0f));
        ASSERT_NE(hidden_splat_id, lfs::core::NULL_NODE);
        const auto point_cloud_id = scene.addPointCloud("VisiblePoints", makeTestPointCloud());
        ASSERT_NE(point_cloud_id, lfs::core::NULL_NODE);

        auto ellipsoid_result = cap::ensureEllipsoid(manager, nullptr, hidden_splat_id);
        ASSERT_TRUE(ellipsoid_result) << ellipsoid_result.error();
        auto* ellipsoid_node = scene.getNodeById(*ellipsoid_result);
        ASSERT_NE(ellipsoid_node, nullptr);
        ASSERT_TRUE(ellipsoid_node->ellipsoid);
        ellipsoid_node->ellipsoid->enabled = true;
        scene.setNodeVisibility("HiddenSplat", false);
        manager.clearSelection();

        Viewport viewport(640, 480);
        auto scene_state = manager.buildRenderState();
        ASSERT_EQ(scene_state.ellipsoids.size(), 1u);
        EXPECT_FALSE(scene_state.ellipsoids.front().effectively_visible);
        EXPECT_FALSE(scene_state.ellipsoids.front().parent_effectively_visible);
        RenderSettings settings;
        const FrameContext ctx{
            .viewport = viewport,
            .scene_manager = &manager,
            .scene_state = scene_state,
            .settings = settings,
            .render_size = {640, 480},
        };
        const std::vector<glm::mat4> transforms = scene_state.model_transforms;

        const auto request = buildPointCloudRenderRequest(ctx, {640, 480}, transforms);

        EXPECT_FALSE(request.filters.crop_box.has_value());
        EXPECT_FALSE(request.filters.crop_ellipsoid.has_value());
    }

    TEST(ViewportRequestBuilderTest, PointCloudRequestUsesCropBoxWhenBothPointCloudGizmosAffectRender) {
        Viewport viewport(640, 480);
        SceneRenderState scene_state;
        RenderSettings settings;
        settings.desaturate_cropping = false;
        const auto ellipsoid_transform = glm::translate(glm::mat4(1.0f), glm::vec3(4.0f, 5.0f, 6.0f));
        const FrameContext ctx{
            .viewport = viewport,
            .scene_state = scene_state,
            .settings = settings,
            .render_size = {640, 480},
            .gizmo =
                {.cropbox_active = true,
                 .cropbox_min = {-1.0f, -1.0f, -1.0f},
                 .cropbox_max = {1.0f, 1.0f, 1.0f},
                 .cropbox_transform = glm::mat4(1.0f),
                 .cropbox_affects_render = true,
                 .cropbox_parent_node_index = 0,
                 .ellipsoid_active = true,
                 .ellipsoid_radii = {3.0f, 4.0f, 5.0f},
                 .ellipsoid_transform = ellipsoid_transform,
                 .ellipsoid_affects_render = true,
                 .ellipsoid_parent_node_index = 0},
        };
        const std::vector<glm::mat4> transforms{glm::mat4(1.0f)};

        const auto request = buildPointCloudRenderRequest(ctx, {640, 480}, transforms);

        ASSERT_TRUE(request.filters.crop_box.has_value());
        EXPECT_FALSE(request.filters.crop_ellipsoid.has_value());
        EXPECT_EQ(request.filters.crop_box->min, glm::vec3(-1.0f, -1.0f, -1.0f));
        EXPECT_EQ(request.filters.crop_box->max, glm::vec3(1.0f, 1.0f, 1.0f));
        EXPECT_EQ(request.filters.crop_box->transform, glm::mat4(1.0f));
        EXPECT_FALSE(request.filters.crop_inverse);
        EXPECT_FALSE(request.filters.crop_desaturate);
    }

    TEST(ViewportRequestBuilderTest, PointCloudRequestKeepsActiveCropBoxBehavior) {
        Viewport viewport(640, 480);
        SceneRenderState scene_state;
        RenderSettings settings;
        settings.desaturate_cropping = true;
        const auto cropbox_transform = glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 0.0f, 0.0f));
        const FrameContext ctx{
            .viewport = viewport,
            .scene_state = scene_state,
            .settings = settings,
            .render_size = {640, 480},
            .gizmo =
                {.cropbox_active = true,
                 .cropbox_min = {-2.0f, -3.0f, -4.0f},
                 .cropbox_max = {2.0f, 3.0f, 4.0f},
                 .cropbox_transform = cropbox_transform,
                 .cropbox_affects_render = true,
                 .cropbox_parent_node_index = -1},
        };
        const std::vector<glm::mat4> transforms{glm::mat4(1.0f)};

        const auto request = buildPointCloudRenderRequest(ctx, {640, 480}, transforms);

        ASSERT_TRUE(request.filters.crop_box.has_value());
        EXPECT_FALSE(request.filters.crop_ellipsoid.has_value());
        EXPECT_EQ(request.filters.crop_box->min, glm::vec3(-2.0f, -3.0f, -4.0f));
        EXPECT_EQ(request.filters.crop_box->max, glm::vec3(2.0f, 3.0f, 4.0f));
        EXPECT_EQ(request.filters.crop_box->transform, glm::inverse(cropbox_transform));
        EXPECT_FALSE(request.filters.crop_inverse);
        EXPECT_TRUE(request.filters.crop_desaturate);
    }

    TEST(ViewportRequestBuilderTest, PointCloudRequestCarriesSelectionOverlay) {
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::Tensor;

        Viewport viewport(640, 480);
        SceneRenderState scene_state;
        scene_state.selection_mask = std::make_shared<Tensor>(
            Tensor::zeros({size_t{2}}, Device::CPU, DataType::UInt8));
        scene_state.selection_mask->ptr<std::uint8_t>()[0] = 1;

        Tensor preview_selection =
            Tensor::zeros({size_t{2}}, Device::CPU, DataType::UInt8);
        preview_selection.ptr<std::uint8_t>()[1] = 1;

        RenderSettings settings;
        settings.selection_color_committed = {0.25f, 0.5f, 0.75f};
        settings.selection_color_preview = {0.1f, 0.9f, 0.2f};
        settings.voxel_size = 0.02f;

        const FrameContext ctx{
            .viewport = viewport,
            .scene_state = scene_state,
            .settings = settings,
            .render_size = {640, 480},
            .viewport_pos = {0, 0},
            .cursor_preview =
                {.add_mode = false,
                 .preview_selection = &preview_selection},
        };
        const std::vector<glm::mat4> transforms{glm::mat4(1.0f)};

        const auto request = buildPointCloudRenderRequest(ctx, {640, 480}, transforms);

        EXPECT_EQ(request.overlay.selection_mask, scene_state.selection_mask);
        EXPECT_EQ(request.overlay.transient_mask.mask, &preview_selection);
        EXPECT_FALSE(request.overlay.transient_mask.additive);
        EXPECT_EQ(request.overlay.selection_colors[1], glm::vec4(settings.selection_color_committed, 1.0f));
        EXPECT_EQ(request.overlay.selection_colors[lfs::rendering::kSelectionPreviewColorIndex],
                  glm::vec4(settings.selection_color_preview, 1.0f));
        EXPECT_EQ(request.render.voxel_size, settings.voxel_size);
    }

    TEST(ViewportFrameLifecycleServiceTest, ResizeActiveDefersFullRefreshUntilDebounceCompletes) {
        ViewportFrameLifecycleService service;

        const auto initial_resize = service.handleViewportResize({640, 480});
        EXPECT_EQ(initial_resize.dirty, DirtyFlag::VIEWPORT | DirtyFlag::CAMERA | DirtyFlag::OVERLAY);
        EXPECT_FALSE(initial_resize.completed);

        EXPECT_EQ(service.setViewportResizeActive(true), 0u);

        const auto active_resize = service.handleViewportResize({800, 600});
        EXPECT_EQ(active_resize.dirty, DirtyFlag::OVERLAY);
        EXPECT_FALSE(active_resize.completed);
        EXPECT_TRUE(service.isResizeDeferring());

        EXPECT_EQ(service.setViewportResizeActive(false),
                  DirtyFlag::VIEWPORT | DirtyFlag::CAMERA | DirtyFlag::OVERLAY);

        const auto debounce_step_1 = service.handleViewportResize({800, 600});
        EXPECT_EQ(debounce_step_1.dirty, DirtyFlag::OVERLAY);
        EXPECT_FALSE(debounce_step_1.completed);

        const auto debounce_step_2 = service.handleViewportResize({800, 600});
        EXPECT_EQ(debounce_step_2.dirty, DirtyFlag::OVERLAY);
        EXPECT_FALSE(debounce_step_2.completed);

        waitUntilResizeSettleReady(service);
        const auto debounce_step_3 = service.handleViewportResize({800, 600});
        EXPECT_EQ(debounce_step_3.dirty, DirtyFlag::VIEWPORT | DirtyFlag::CAMERA);
        EXPECT_TRUE(debounce_step_3.completed);
        EXPECT_FALSE(service.isResizeDeferring());
    }

    TEST(ViewportFrameLifecycleServiceTest, PassiveWindowResizeDefersFullRefreshUntilDebounceCompletes) {
        ViewportFrameLifecycleService service;

        EXPECT_EQ(service.handleViewportResize({640, 480}).dirty,
                  DirtyFlag::VIEWPORT | DirtyFlag::CAMERA | DirtyFlag::OVERLAY);

        const auto passive_resize = service.handleViewportResize({800, 600});
        EXPECT_EQ(passive_resize.dirty, DirtyFlag::OVERLAY);
        EXPECT_FALSE(passive_resize.completed);
        EXPECT_TRUE(service.isResizeDeferring());

        EXPECT_EQ(service.handleViewportResize({800, 600}).dirty, DirtyFlag::OVERLAY);
        EXPECT_EQ(service.handleViewportResize({800, 600}).dirty, DirtyFlag::OVERLAY);

        waitUntilResizeSettleReady(service);
        const auto completed = service.handleViewportResize({800, 600});
        EXPECT_EQ(completed.dirty, DirtyFlag::VIEWPORT | DirtyFlag::CAMERA);
        EXPECT_TRUE(completed.completed);
        EXPECT_FALSE(service.isResizeDeferring());
    }

    TEST(ViewportFrameLifecycleServiceTest, ExplicitRefreshDeferralCompletesAfterStableFrames) {
        ViewportFrameLifecycleService service;

        EXPECT_EQ(service.handleViewportResize({640, 480}).dirty,
                  DirtyFlag::VIEWPORT | DirtyFlag::CAMERA | DirtyFlag::OVERLAY);
        EXPECT_EQ(service.deferViewportRefresh(), DirtyFlag::OVERLAY);
        EXPECT_TRUE(service.isResizeDeferring());

        EXPECT_EQ(service.handleViewportResize({640, 480}).dirty, DirtyFlag::OVERLAY);
        EXPECT_EQ(service.handleViewportResize({640, 480}).dirty, DirtyFlag::OVERLAY);

        waitUntilResizeSettleReady(service);
        const auto completed = service.handleViewportResize({640, 480});
        EXPECT_EQ(completed.dirty, DirtyFlag::VIEWPORT | DirtyFlag::CAMERA);
        EXPECT_TRUE(completed.completed);
        EXPECT_FALSE(service.isResizeDeferring());
    }

    TEST(ViewportFrameLifecycleServiceTest, ModelChangeClearsCachedViewportArtifactsOncePerModelPointer) {
        ViewportFrameLifecycleService service;
        ViewportArtifactService artifacts;

        const auto generation_before = artifacts.artifactGeneration();
        const auto first_change = service.handleModelChange(0x1234, artifacts);
        EXPECT_TRUE(first_change.changed);
        EXPECT_EQ(first_change.previous_model_ptr, 0u);
        EXPECT_GT(artifacts.artifactGeneration(), generation_before);

        const auto generation_after_first_change = artifacts.artifactGeneration();
        const auto repeated_change = service.handleModelChange(0x1234, artifacts);
        EXPECT_FALSE(repeated_change.changed);
        EXPECT_EQ(artifacts.artifactGeneration(), generation_after_first_change);
    }

    TEST(ViewportArtifactServiceTest, ExplicitSplitPanelSamplingUsesPanelLocalCoordinates) {
        ViewportArtifactService artifacts;

        auto left_depth = lfs::core::Tensor::from_vector(
                              std::vector<float>(512, 1.0f),
                              {size_t{1}, size_t{1}, size_t{512}},
                              lfs::core::Device::CPU)
                              .cuda();
        auto right_values = std::vector<float>(512, 2.0f);
        right_values[256] = 42.0f;
        auto right_depth = lfs::core::Tensor::from_vector(
                               right_values,
                               {size_t{1}, size_t{1}, size_t{512}},
                               lfs::core::Device::CPU)
                               .cuda();

        FrameResources resources;
        resources.cached_metadata = CachedRenderMetadata{
            .depth_panels =
                {CachedRenderPanelMetadata{
                     .depth = std::make_shared<lfs::core::Tensor>(std::move(left_depth)),
                     .start_position = 0.0f,
                     .end_position = 0.5f,
                 },
                 CachedRenderPanelMetadata{
                     .depth = std::make_shared<lfs::core::Tensor>(std::move(right_depth)),
                     .start_position = 0.5f,
                     .end_position = 1.0f,
                 }},
            .depth_panel_count = 2,
            .valid = true,
            .depth_is_ndc = false,
        };
        resources.cached_result_size = {1024, 1};
        artifacts.updateFromFrameResources(resources, false);

        EXPECT_FLOAT_EQ(
            artifacts.sampleLinearDepthAt(256, 0, {1024, 1}, SplitViewPanelId::Right),
            42.0f);
    }

    TEST(ViewportFrameLifecycleServiceTest, MissingViewportOutputForcesFreshRedraw) {
        ViewportFrameLifecycleService service;

        EXPECT_EQ(
            service.requiredDirtyMask(false, true, SplitViewMode::Disabled),
            DirtyFlag::ALL);
        EXPECT_EQ(
            service.requiredDirtyMask(false, false, SplitViewMode::PLYComparison),
            DirtyFlag::ALL | DirtyFlag::SPLIT_VIEW);
        EXPECT_EQ(
            service.requiredDirtyMask(true, true, SplitViewMode::PLYComparison),
            0u);
    }

    TEST(ViewportRequestBuilderTest, CursorPreviewTargetsOnlyItsSplitPanel) {
        Viewport viewport;
        RenderSettings settings;
        FrameContext ctx{
            .viewport = viewport,
            .settings = settings,
            .render_size = {800, 600},
            .cursor_preview =
                {.active = true,
                 .x = 120.0f,
                 .y = 80.0f,
                 .radius = 24.0f,
                 .add_mode = true,
                 .panel = SplitViewPanelId::Right},
        };

        const auto left_request = buildViewportRenderRequest(
            ctx, {400, 600}, &ctx.viewport, SplitViewPanelId::Left);
        const auto right_request = buildViewportRenderRequest(
            ctx, {400, 600}, &ctx.viewport, SplitViewPanelId::Right);

        EXPECT_FALSE(left_request.overlay.cursor.enabled);
        EXPECT_TRUE(right_request.overlay.cursor.enabled);
    }

    TEST(ViewportRequestBuilderTest, TrainingSuppressesInteractiveSelectionOverlayButKeepsRenderMarkers) {
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::Tensor;

        Viewport viewport(640, 480);
        SceneRenderState scene_state;
        scene_state.selection_mask = std::make_shared<Tensor>(
            Tensor::zeros({size_t{2}}, Device::CPU, DataType::UInt8));
        scene_state.selected_node_mask = {true, false};

        Tensor preview_selection =
            Tensor::zeros({size_t{2}}, Device::CPU, DataType::UInt8);

        RenderSettings settings;
        settings.show_rings = true;
        settings.show_center_markers = true;
        settings.desaturate_unselected = true;
        settings.selection_color_center_marker = glm::vec3(0.25f, 0.5f, 0.75f);

        const FrameContext ctx{
            .viewport = viewport,
            .scene_state = scene_state,
            .settings = settings,
            .render_size = {640, 480},
            .viewport_pos = {0, 0},
            .training_active = true,
            .cursor_preview =
                {.active = true,
                 .x = 120.0f,
                 .y = 80.0f,
                 .radius = 24.0f,
                 .add_mode = true,
                 .preview_selection = &preview_selection,
                 .focused_gaussian_id = 1,
                 .selection_mode = SelectionPreviewMode::Rings},
            .selection_flash_intensity = 0.75f,
        };

        const auto request = buildViewportRenderRequest(ctx, {640, 480});

        EXPECT_TRUE(request.overlay.markers.show_rings);
        EXPECT_TRUE(request.overlay.markers.show_center_markers);
        EXPECT_EQ(request.overlay.selection_colors[0], glm::vec4(settings.selection_color_center_marker, 1.0f));
        EXPECT_FALSE(request.overlay.cursor.enabled);
        EXPECT_EQ(request.overlay.emphasis.mask, nullptr);
        EXPECT_EQ(request.overlay.emphasis.transient_mask.mask, nullptr);
        EXPECT_FALSE(request.overlay.emphasis.transient_mask.additive);
        EXPECT_TRUE(request.overlay.emphasis.emphasized_node_mask.empty());
        EXPECT_FALSE(request.overlay.emphasis.dim_non_emphasized);
        EXPECT_FLOAT_EQ(request.overlay.emphasis.flash_intensity, 0.0f);
        EXPECT_EQ(request.overlay.emphasis.focused_gaussian_id, -1);

        const std::vector<glm::mat4> transforms{glm::mat4(1.0f)};
        const auto point_cloud_request = buildPointCloudRenderRequest(ctx, {640, 480}, transforms);
        EXPECT_EQ(point_cloud_request.overlay.selection_mask, nullptr);
        EXPECT_EQ(point_cloud_request.overlay.transient_mask.mask, nullptr);
    }

    TEST_F(RenderingManagerEventsTest, SceneLoadedDisablesGtComparison) {
        RenderingManager manager;
        lfs::core::events::cmd::ToggleGTComparison{}.emit();
        EXPECT_EQ(manager.getSettings().split_view_mode, SplitViewMode::GTComparison);

        lfs::core::events::state::SceneLoaded{
            .scene = nullptr,
            .path = std::filesystem::path{},
            .type = lfs::core::events::state::SceneLoaded::Type::PLY,
            .num_gaussians = 0}
            .emit();

        EXPECT_EQ(manager.getSettings().split_view_mode, SplitViewMode::Disabled);
    }

    TEST_F(RenderingManagerEventsTest, SceneClearedDisablesGtComparison) {
        RenderingManager manager;
        lfs::core::events::cmd::ToggleGTComparison{}.emit();
        EXPECT_EQ(manager.getSettings().split_view_mode, SplitViewMode::GTComparison);

        lfs::core::events::state::SceneCleared{}.emit();

        EXPECT_EQ(manager.getSettings().split_view_mode, SplitViewMode::Disabled);
    }

    TEST_F(RenderingManagerEventsTest, ToggleIndependentSplitViewInitializesSecondaryViewport) {
        RenderingManager manager;
        Viewport primary_viewport(800, 600);
        primary_viewport.setViewMatrix(glm::mat3(1.0f), glm::vec3(4.0f, 5.0f, 6.0f));

        lfs::core::events::cmd::ToggleIndependentSplitView{
            .viewport = &primary_viewport,
        }
            .emit();

        EXPECT_EQ(manager.getSettings().split_view_mode, SplitViewMode::IndependentDual);
        const auto& secondary = manager.resolvePanelViewport(primary_viewport, SplitViewPanelId::Right);
        EXPECT_EQ(secondary.getTranslation(), primary_viewport.getTranslation());
        EXPECT_EQ(secondary.getRotationMatrix(), primary_viewport.getRotationMatrix());
    }

    TEST_F(RenderingManagerEventsTest, ToggleIndependentSplitViewTwiceDisablesMode) {
        RenderingManager manager;
        Viewport primary_viewport(800, 600);

        lfs::core::events::cmd::ToggleIndependentSplitView{
            .viewport = &primary_viewport,
        }
            .emit();
        ASSERT_EQ(manager.getSettings().split_view_mode, SplitViewMode::IndependentDual);

        lfs::core::events::cmd::ToggleIndependentSplitView{
            .viewport = &primary_viewport,
        }
            .emit();

        EXPECT_EQ(manager.getSettings().split_view_mode, SplitViewMode::Disabled);
        EXPECT_EQ(manager.getFocusedSplitPanel(), SplitViewPanelId::Left);
    }

    TEST_F(RenderingManagerEventsTest, IndependentSplitGridPlaneTracksPanelsIndependently) {
        RenderingManager manager;
        Viewport primary_viewport(800, 600);

        auto settings = manager.getSettings();
        settings.grid_plane = 2;
        manager.updateSettings(settings);

        lfs::core::events::cmd::ToggleIndependentSplitView{
            .viewport = &primary_viewport,
        }
            .emit();

        ASSERT_EQ(manager.getSettings().split_view_mode, SplitViewMode::IndependentDual);
        EXPECT_EQ(manager.getGridPlaneForPanel(SplitViewPanelId::Left), 2);
        EXPECT_EQ(manager.getGridPlaneForPanel(SplitViewPanelId::Right), 2);

        manager.setGridPlaneForPanel(SplitViewPanelId::Left, 0);
        manager.setGridPlaneForPanel(SplitViewPanelId::Right, 1);

        EXPECT_EQ(manager.getGridPlaneForPanel(SplitViewPanelId::Left), 0);
        EXPECT_EQ(manager.getGridPlaneForPanel(SplitViewPanelId::Right), 1);

        manager.setFocusedSplitPanel(SplitViewPanelId::Left);
        EXPECT_EQ(manager.getSettings().grid_plane, 0);

        manager.setFocusedSplitPanel(SplitViewPanelId::Right);
        EXPECT_EQ(manager.getSettings().grid_plane, 1);
    }

    TEST_F(RenderingManagerEventsTest, GridSettingsChangedOnlyUpdatesFocusedPanelInIndependentSplit) {
        RenderingManager manager;
        Viewport primary_viewport(800, 600);

        lfs::core::events::cmd::ToggleIndependentSplitView{
            .viewport = &primary_viewport,
        }
            .emit();

        ASSERT_EQ(manager.getSettings().split_view_mode, SplitViewMode::IndependentDual);

        manager.setGridPlaneForPanel(SplitViewPanelId::Left, 0);
        manager.setGridPlaneForPanel(SplitViewPanelId::Right, 1);
        manager.setFocusedSplitPanel(SplitViewPanelId::Right);

        lfs::core::events::ui::GridSettingsChanged{
            .enabled = true,
            .plane = 2,
            .opacity = 0.25f,
        }
            .emit();

        EXPECT_EQ(manager.getGridPlaneForPanel(SplitViewPanelId::Left), 0);
        EXPECT_EQ(manager.getGridPlaneForPanel(SplitViewPanelId::Right), 2);
        EXPECT_EQ(manager.getSettings().grid_plane, 2);
    }

    TEST_F(RenderingManagerEventsTest, RenderSettingsChangedEquirectangularForcesGutBackend) {
        using Backend = lfs::rendering::GaussianRasterBackend;

        RenderingManager manager;
        auto settings = manager.getSettings();
        settings.raster_backend = Backend::ThreeDgs;
        settings.gut = false;
        settings.equirectangular = false;
        manager.updateSettings(settings);

        auto event = lfs::core::events::ui::RenderSettingsChanged{};
        event.equirectangular = true;
        event.emit();

        settings = manager.getSettings();
        EXPECT_TRUE(settings.equirectangular);
        EXPECT_EQ(settings.raster_backend, Backend::ThreeDgut);
        EXPECT_TRUE(settings.gut);
    }

} // namespace lfs::vis
