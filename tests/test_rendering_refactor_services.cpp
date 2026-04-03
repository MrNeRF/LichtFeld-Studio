/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bus.hpp"
#include "core/events.hpp"
#include "core/camera.hpp"
#include "core/point_cloud.hpp"
#include "core/services.hpp"
#include "core/tensor.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "visualizer/rendering/passes/mesh_pass.hpp"
#include "visualizer/rendering/passes/point_cloud_pass.hpp"
#include "visualizer/rendering/passes/splat_raster_pass.hpp"
#include "visualizer/rendering/render_pass.hpp"
#include "visualizer/rendering/rendering_manager.hpp"
#include "visualizer/rendering/split_view_composition.hpp"
#include "visualizer/rendering/split_view_service.hpp"
#include "visualizer/rendering/viewport_artifact_service.hpp"
#include "visualizer/rendering/viewport_frame_lifecycle_service.hpp"
#include "visualizer/rendering/viewport_request_builder.hpp"
#include "visualizer/scene/scene_manager.hpp"

#include <filesystem>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
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
        }

        void TearDown() override {
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
            .gt_texture_id = 11,
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

        auto scene_state = manager.buildRenderState();
        scene_state.selection_mask = std::make_shared<Tensor>(
            Tensor::zeros({size_t{2}}, Device::CPU, DataType::UInt8));
        scene_state.selected_node_mask = {true, false};

        Tensor transient_selection =
            Tensor::zeros({size_t{2}}, Device::CPU, DataType::Bool);

        RenderSettings settings;
        settings.split_view_mode = SplitViewMode::PLYComparison;
        settings.split_position = 0.4f;
        settings.use_crop_box = true;
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

        const auto debounce_step_3 = service.handleViewportResize({800, 600});
        EXPECT_EQ(debounce_step_3.dirty, DirtyFlag::VIEWPORT | DirtyFlag::CAMERA);
        EXPECT_TRUE(debounce_step_3.completed);
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
            artifacts.sampleLinearDepthAt(256, 0, {1024, 1}, nullptr, SplitViewPanelId::Right),
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

    TEST(RenderPassSensitivityTest, SplitViewToggleInvalidatesBaseViewportContent) {
        SplatRasterPass splat_pass;
        PointCloudPass point_cloud_pass;

        EXPECT_NE(splat_pass.sensitivity() & DirtyFlag::SPLIT_VIEW, 0u);
        EXPECT_NE(point_cloud_pass.sensitivity() & DirtyFlag::SPLIT_VIEW, 0u);
        EXPECT_NE(MeshPass::MESH_GEOMETRY_MASK & DirtyFlag::SPLIT_VIEW, 0u);
    }

    TEST(RenderPassSensitivityTest, InvalidTrainingModelRoutesToPointCloudFallback) {
        Viewport viewport;
        RenderSettings settings;
        SceneManager manager;
        const lfs::core::SplatData invalid_model;

        FrameContext ctx{
            .viewport = viewport,
            .scene_manager = &manager,
            .model = &invalid_model,
            .settings = settings,
            .render_size = viewport.windowSize};

        SplatRasterPass splat_pass;
        PointCloudPass point_cloud_pass;

        EXPECT_FALSE(splat_pass.shouldExecute(DirtyFlag::SPLATS, ctx));
        EXPECT_TRUE(point_cloud_pass.shouldExecute(DirtyFlag::SPLATS, ctx));
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

} // namespace lfs::vis
