/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bus.hpp"
#include "core/events.hpp"
#include "core/point_cloud.hpp"
#include "core/services.hpp"
#include "core/tensor.hpp"
#include "visualizer/rendering/passes/mesh_pass.hpp"
#include "visualizer/rendering/passes/point_cloud_pass.hpp"
#include "visualizer/rendering/passes/splat_raster_pass.hpp"
#include "visualizer/rendering/render_pass.hpp"
#include "visualizer/rendering/rendering_manager.hpp"
#include "visualizer/rendering/split_view_service.hpp"
#include "visualizer/rendering/viewport_artifact_service.hpp"
#include "visualizer/rendering/viewport_frame_lifecycle_service.hpp"
#include "visualizer/scene/scene_manager.hpp"

#include <filesystem>
#include <gtest/gtest.h>
#include <vector>

namespace lfs::vis {

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

        const auto enable = service.toggleGTComparison(settings);
        EXPECT_TRUE(enable.enabled);
        EXPECT_EQ(settings.split_view_mode, SplitViewMode::GTComparison);

        settings.equirectangular = false;

        const auto disable = service.toggleGTComparison(settings);
        EXPECT_FALSE(disable.enabled);
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

        service.handleSceneCleared(settings);

        EXPECT_EQ(settings.split_view_mode, SplitViewMode::Disabled);
        EXPECT_EQ(settings.split_view_offset, 0);
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

    TEST(ViewportFrameLifecycleServiceTest, MissingViewportOutputForcesFreshRedraw) {
        ViewportFrameLifecycleService service;

        EXPECT_EQ(
            service.requiredDirtyMask(false, true, SplitViewMode::Disabled),
            DirtyFlag::ALL);
        EXPECT_EQ(
            service.requiredDirtyMask(false, false, SplitViewMode::PLYComparison),
            DirtyFlag::ALL | DirtyFlag::SPLIT_VIEW);
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

    TEST_F(RenderingManagerEventsTest, SceneLoadedDisablesGtComparisonAndEmitsEvent) {
        std::vector<bool> gt_mode_events;
        lfs::core::events::ui::GTComparisonModeChanged::when(
            [&gt_mode_events](const auto& event) { gt_mode_events.push_back(event.enabled); });

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
        ASSERT_EQ(gt_mode_events.size(), 2u);
        EXPECT_TRUE(gt_mode_events[0]);
        EXPECT_FALSE(gt_mode_events[1]);
    }

    TEST_F(RenderingManagerEventsTest, SceneClearedDisablesGtComparisonAndEmitsEvent) {
        std::vector<bool> gt_mode_events;
        lfs::core::events::ui::GTComparisonModeChanged::when(
            [&gt_mode_events](const auto& event) { gt_mode_events.push_back(event.enabled); });

        RenderingManager manager;
        lfs::core::events::cmd::ToggleGTComparison{}.emit();
        EXPECT_EQ(manager.getSettings().split_view_mode, SplitViewMode::GTComparison);

        lfs::core::events::state::SceneCleared{}.emit();

        EXPECT_EQ(manager.getSettings().split_view_mode, SplitViewMode::Disabled);
        ASSERT_EQ(gt_mode_events.size(), 2u);
        EXPECT_TRUE(gt_mode_events[0]);
        EXPECT_FALSE(gt_mode_events[1]);
    }

} // namespace lfs::vis
