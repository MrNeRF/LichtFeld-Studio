/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bus.hpp"
#include "core/events.hpp"
#include "core/tensor.hpp"
#include "operation/undo_history.hpp"
#include "operator/operator_registry.hpp"
#include "visualizer/rendering/depth_window_state.hpp"
#include "visualizer/rendering/dirty_flags.hpp"
#include "visualizer/rendering/render_pass.hpp"
#include "visualizer/rendering/rendering_manager.hpp"
#include "visualizer/rendering/rendering_types.hpp"
#include "visualizer/rendering/viewport_request_builder.hpp"
#include "visualizer_impl.hpp"

#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace lfs::vis {

    namespace {

        DepthWindowState makeWindow(const float depth_near_val,
                                    const float depth_far_val,
                                    const float scale_x = 0.35f,
                                    const float scale_y = 0.35f,
                                    const float offset_x = 0.0f,
                                    const float offset_y = 0.0f) {
            return {
                .near_plane = depth_near_val,
                .far_plane = depth_far_val,
                .scale_x = scale_x,
                .scale_y = scale_y,
                .offset_x = offset_x,
                .offset_y = offset_y,
            };
        }

    } // namespace

    class DepthWindowPanelsTest : public ::testing::Test {
    protected:
        void SetUp() override {
            lfs::event::EventBridge::instance().clear_all();
            lfs::core::event::bus().clear_all();
            lfs::vis::services().clear();
            lfs::vis::op::undoHistory().clear();

            options_.show_startup_overlay = false;
            options_.width = 200;
            options_.height = 200;
            viewer_ = std::make_unique<lfs::vis::VisualizerImpl>(options_);
            rendering_manager_ = viewer_->getRenderingManager();
            ASSERT_NE(rendering_manager_, nullptr);

            const std::vector<float> means_data{0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
            const std::vector<float> rotation_data{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f};
            auto means = lfs::core::Tensor::from_vector(means_data, {size_t{2}, size_t{3}}, lfs::core::Device::CUDA).to(lfs::core::DataType::Float32);
            auto sh0 = lfs::core::Tensor::zeros({size_t{2}, size_t{1}, size_t{3}}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
            auto shN = lfs::core::Tensor::zeros({size_t{2}, size_t{3}, size_t{3}}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
            auto scaling = lfs::core::Tensor::zeros({size_t{2}, size_t{3}}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
            auto rotation = lfs::core::Tensor::from_vector(rotation_data, {size_t{2}, size_t{4}}, lfs::core::Device::CUDA).to(lfs::core::DataType::Float32);
            auto opacity = lfs::core::Tensor::zeros({size_t{2}, size_t{1}}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
            viewer_->getSceneManager()->getScene().addSplat(
                "depth_window_test",
                std::make_unique<lfs::core::SplatData>(1, std::move(means), std::move(sh0), std::move(shN), std::move(scaling), std::move(rotation), std::move(opacity), 1.0f));
            viewer_->getSceneManager()->initSelectionService();
            if (auto* const service = viewer_->getSceneManager()->getSelectionService()) {
                service->setTestingViewport({
                    .x = 0.0f,
                    .y = 0.0f,
                    .width = static_cast<float>(options_.width),
                    .height = static_cast<float>(options_.height),
                    .render_width = options_.width,
                    .render_height = options_.height,
                });
            }
            auto settings = rendering_manager_->getSettings();
            settings.depth_filter_enabled = true;
            settings.depth_filter_scale_x = 0.5f;
            settings.depth_filter_scale_y = 0.5f;
            rendering_manager_->updateSettings(settings);
        }

        void TearDown() override {
            lfs::vis::op::operators().cancelModalOperator();
            lfs::event::EventBridge::instance().clear_all();
            lfs::core::event::bus().clear_all();
            lfs::vis::services().clear();
            viewer_.reset();
            lfs::vis::op::undoHistory().clear();
        }

        void enterIndependentDual() {
            lfs::core::events::cmd::ToggleIndependentSplitView{.viewport = &viewer_->getViewport()}.emit();
        }

        lfs::vis::ViewerOptions options_{};
        std::unique_ptr<lfs::vis::VisualizerImpl> viewer_;
        lfs::vis::RenderingManager* rendering_manager_ = nullptr;
    };

    TEST_F(DepthWindowPanelsTest, SetGetRoundtripAndClamping) {
        RenderingManager manager;
        DepthWindowState extreme = makeWindow(500.0f, -1.0f, 2.0f, -2.0f, 5.0f, -5.0f);
        manager.setDepthWindowForPanel(SplitViewPanelId::Left, extreme);
        const auto stored = manager.getDepthWindowForPanel(SplitViewPanelId::Left);
        EXPECT_FLOAT_EQ(stored.near_plane, 500.0f);
        EXPECT_FLOAT_EQ(stored.far_plane, 500.01f);
        EXPECT_FLOAT_EQ(stored.scale_x, 1.0f);
        EXPECT_FLOAT_EQ(stored.scale_y, 0.05f);
        EXPECT_FLOAT_EQ(stored.offset_x, 1.0f);
        EXPECT_FLOAT_EQ(stored.offset_y, -1.0f);
    }

    TEST_F(DepthWindowPanelsTest, FocusSwitchSwapsProjection) {
        // Sync defaults OFF: per-panel independence is the default experience.
        EXPECT_FALSE(rendering_manager_->getDepthWindowSync());
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(false);

        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left, makeWindow(1.0f, 10.0f, 0.4f));
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Right, makeWindow(2.0f, 20.0f, 0.5f));

        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Left);
        const auto left_settings = rendering_manager_->getSettings();
        EXPECT_FLOAT_EQ(-left_settings.depth_filter_max.z, 1.0f);
        EXPECT_FLOAT_EQ(-left_settings.depth_filter_min.z, 10.0f);
        EXPECT_FLOAT_EQ(left_settings.depth_filter_scale_x, 0.4f);

        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Right);
        const auto right_settings = rendering_manager_->getSettings();
        EXPECT_FLOAT_EQ(-right_settings.depth_filter_max.z, 2.0f);
        EXPECT_FLOAT_EQ(-right_settings.depth_filter_min.z, 20.0f);
        EXPECT_FLOAT_EQ(right_settings.depth_filter_scale_x, 0.5f);
    }

    TEST_F(DepthWindowPanelsTest, UpdateSettingsBackRoutesFocusedSlotWhenUnsynced) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(false);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Right, makeWindow(2.0f, 20.0f, 0.45f));
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Left);

        auto settings = rendering_manager_->getSettings();
        settings.depth_filter_scale_x = 0.55f;
        settings.depth_filter_min.z = -15.0f;
        settings.depth_filter_max.z = -3.0f;
        rendering_manager_->updateSettings(settings, DirtyFlag::SELECTION);

        const auto left = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        const auto right = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right);
        EXPECT_FLOAT_EQ(left.scale_x, 0.55f);
        EXPECT_FLOAT_EQ(left.near_plane, 3.0f);
        EXPECT_FLOAT_EQ(left.far_plane, 15.0f);
        EXPECT_FLOAT_EQ(right.near_plane, 2.0f);
        EXPECT_FLOAT_EQ(right.far_plane, 20.0f);
        EXPECT_FLOAT_EQ(right.scale_x, 0.45f);
    }

    TEST_F(DepthWindowPanelsTest, FanOutWhenSynced) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(true);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Right, makeWindow(4.0f, 40.0f, 0.6f));

        const auto left = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        const auto right = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right);
        EXPECT_FLOAT_EQ(left.near_plane, 4.0f);
        EXPECT_FLOAT_EQ(right.near_plane, 4.0f);
        EXPECT_FLOAT_EQ(left.scale_x, 0.6f);
        EXPECT_FLOAT_EQ(right.scale_x, 0.6f);
    }

    TEST_F(DepthWindowPanelsTest, ModeEnterSeedsBothSlots) {
        auto settings = rendering_manager_->getSettings();
        settings.depth_filter_scale_x = 0.42f;
        settings.depth_filter_min.z = -25.0f;
        settings.depth_filter_max.z = -5.0f;
        rendering_manager_->updateSettings(settings);

        enterIndependentDual();

        const auto left = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        const auto right = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right);
        EXPECT_FLOAT_EQ(left.scale_x, 0.42f);
        EXPECT_FLOAT_EQ(right.scale_x, 0.42f);
        EXPECT_FLOAT_EQ(left.near_plane, 5.0f);
        EXPECT_FLOAT_EQ(right.near_plane, 5.0f);
        EXPECT_FLOAT_EQ(left.far_plane, 25.0f);
        EXPECT_FLOAT_EQ(right.far_plane, 25.0f);
    }

    TEST_F(DepthWindowPanelsTest, ModeLeaveCollapsesToPreTransitionFocus) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(false);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left, makeWindow(1.0f, 10.0f, 0.3f));
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Right, makeWindow(2.0f, 20.0f, 0.7f));
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Right);

        lfs::core::events::cmd::ToggleIndependentSplitView{.viewport = &viewer_->getViewport()}.emit();

        const auto settings = rendering_manager_->getSettings();
        EXPECT_EQ(rendering_manager_->getSettings().split_view_mode, SplitViewMode::Disabled);
        EXPECT_FLOAT_EQ(-settings.depth_filter_max.z, 2.0f);
        EXPECT_FLOAT_EQ(-settings.depth_filter_min.z, 20.0f);
        EXPECT_FLOAT_EQ(settings.depth_filter_scale_x, 0.7f);

        // The sync choice survives leaving and re-entering split view.
        EXPECT_FALSE(rendering_manager_->getDepthWindowSync());
        lfs::core::events::cmd::ToggleIndependentSplitView{.viewport = &viewer_->getViewport()}.emit();
        EXPECT_EQ(rendering_manager_->getSettings().split_view_mode, SplitViewMode::IndependentDual);
        EXPECT_FALSE(rendering_manager_->getDepthWindowSync());
    }

    TEST_F(DepthWindowPanelsTest, StaleProxyWithoutSelectionMaskDoesNotMoveOtherPanel) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(false);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left, makeWindow(1.0f, 10.0f, 0.35f));
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Right, makeWindow(2.0f, 20.0f, 0.45f));

        auto stale = rendering_manager_->getSettings();
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Right);
        stale.background_color = glm::vec3(0.1f, 0.2f, 0.3f);
        rendering_manager_->updateSettings(stale, DirtyFlag::BACKGROUND);

        const auto right = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right);
        const auto applied = rendering_manager_->getSettings();
        EXPECT_FLOAT_EQ(right.near_plane, 2.0f);
        EXPECT_FLOAT_EQ(right.far_plane, 20.0f);
        EXPECT_FLOAT_EQ(right.scale_x, 0.45f);
        EXPECT_FLOAT_EQ(applied.background_color.x, 0.1f);
        EXPECT_FLOAT_EQ(applied.background_color.y, 0.2f);
        EXPECT_FLOAT_EQ(applied.background_color.z, 0.3f);
    }

    TEST_F(DepthWindowPanelsTest, SyncOnCopiesFocusedToOther) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(false);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left, makeWindow(1.0f, 10.0f, 0.31f));
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Right, makeWindow(2.0f, 20.0f, 0.62f));
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Left);

        rendering_manager_->setDepthWindowSync(true);

        const auto right = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right);
        EXPECT_FLOAT_EQ(right.near_plane, 1.0f);
        EXPECT_FLOAT_EQ(right.far_plane, 10.0f);
        EXPECT_FLOAT_EQ(right.scale_x, 0.31f);
    }

    TEST_F(DepthWindowPanelsTest, UpdateSettingsSyncedFansOutToBothSlots) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(true);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left, makeWindow(1.0f, 10.0f, 0.35f));
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Right, makeWindow(2.0f, 20.0f, 0.45f));

        auto settings = rendering_manager_->getSettings();
        settings.depth_filter_scale_x = 0.66f;
        settings.depth_filter_min.z = -30.0f;
        settings.depth_filter_max.z = -6.0f;
        rendering_manager_->updateSettings(settings, DirtyFlag::SELECTION);

        const auto left = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        const auto right = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right);
        EXPECT_FLOAT_EQ(left.scale_x, 0.66f);
        EXPECT_FLOAT_EQ(right.scale_x, 0.66f);
        EXPECT_FLOAT_EQ(left.near_plane, 6.0f);
        EXPECT_FLOAT_EQ(right.near_plane, 6.0f);
        EXPECT_FLOAT_EQ(left.far_plane, 30.0f);
        EXPECT_FLOAT_EQ(right.far_plane, 30.0f);
    }

    TEST(ViewportRequestBuilderTest, DisabledModeUsesGlobalSettingsNotDefaultSlots) {
        Viewport viewport;
        RenderSettings settings;
        settings.depth_filter_enabled = true;
        settings.split_view_mode = SplitViewMode::Disabled;
        settings.depth_filter_min = glm::vec3(-50.0f, -50.0f, -80.0f);
        settings.depth_filter_max = glm::vec3(50.0f, 50.0f, -7.0f);
        settings.depth_filter_scale_x = 0.42f;
        settings.depth_filter_scale_y = 0.51f;
        settings.depth_filter_offset_x = 0.11f;
        settings.depth_filter_offset_y = -0.22f;

        FrameContext ctx{
            .viewport = viewport,
            .settings = settings,
            .render_size = {800, 600},
            .panel_depth_windows = {
                makeWindow(1.0f, 10.0f, 0.25f, 0.30f, 0.1f, 0.0f),
                makeWindow(2.0f, 20.0f, 0.55f, 0.60f, -0.2f, 0.15f),
            },
        };

        const auto request = buildViewportRenderRequest(ctx, {800, 600}, &ctx.viewport, std::nullopt);
        ASSERT_TRUE(request.filters.screen_window.has_value());
        ASSERT_TRUE(request.filters.view_volume.has_value());
        EXPECT_FLOAT_EQ(request.filters.screen_window->scale_x, 0.42f);
        EXPECT_FLOAT_EQ(request.filters.screen_window->scale_y, 0.51f);
        EXPECT_FLOAT_EQ(request.filters.screen_window->offset_x, 0.11f);
        EXPECT_FLOAT_EQ(request.filters.screen_window->offset_y, -0.22f);
        EXPECT_FLOAT_EQ(request.filters.view_volume->min.z, -80.0f);
        EXPECT_FLOAT_EQ(request.filters.view_volume->max.z, -7.0f);
    }

    TEST(ViewportRequestBuilderTest, IndependentDualPanelsResolveDifferentDepthWindows) {
        Viewport viewport;
        RenderSettings settings;
        settings.depth_filter_enabled = true;
        settings.split_view_mode = SplitViewMode::IndependentDual;
        settings.depth_filter_min = glm::vec3(-50.0f, -50.0f, -100.0f);
        settings.depth_filter_max = glm::vec3(50.0f, 50.0f, 0.0f);

        FrameContext ctx{
            .viewport = viewport,
            .settings = settings,
            .render_size = {800, 600},
            .panel_depth_windows = {
                makeWindow(1.0f, 10.0f, 0.25f, 0.30f, 0.1f, 0.0f),
                makeWindow(2.0f, 20.0f, 0.55f, 0.60f, -0.2f, 0.15f),
            },
        };

        const auto left_request = buildViewportRenderRequest(
            ctx, {400, 600}, &ctx.viewport, SplitViewPanelId::Left);
        const auto right_request = buildViewportRenderRequest(
            ctx, {400, 600}, &ctx.viewport, SplitViewPanelId::Right);

        ASSERT_TRUE(left_request.filters.screen_window.has_value());
        ASSERT_TRUE(right_request.filters.screen_window.has_value());
        ASSERT_TRUE(left_request.filters.view_volume.has_value());
        ASSERT_TRUE(right_request.filters.view_volume.has_value());
        EXPECT_FLOAT_EQ(left_request.filters.screen_window->scale_x, 0.25f);
        EXPECT_FLOAT_EQ(right_request.filters.screen_window->scale_x, 0.55f);
        EXPECT_FLOAT_EQ(left_request.filters.view_volume->min.z, -10.0f);
        EXPECT_FLOAT_EQ(right_request.filters.view_volume->min.z, -20.0f);
        EXPECT_FLOAT_EQ(left_request.filters.view_volume->max.z, -1.0f);
        EXPECT_FLOAT_EQ(right_request.filters.view_volume->max.z, -2.0f);
    }

} // namespace lfs::vis
