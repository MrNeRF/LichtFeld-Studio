/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bus.hpp"
#include "core/services.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/gui_manager.hpp"
#include "input/input_controller.hpp"
#include "licht_test_support.hpp"
#include "rendering/render_pass.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer/visualizer_impl.hpp"
#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <optional>
#include <string>

namespace lfs::vis {
    class OrthographicZoomTest : public ::testing::Test {
    protected:
        void SetUp() override {
            if (const auto* previous = std::getenv("LFS_HOME"))
                previous_home_ = previous;
            setHome(temporary_.path.string());
            previous_persistence_enabled_ = input::InputBindings::isPersistenceEnabled();
            input::InputBindings::setPersistenceEnabled(false);
            services().clear();
            gui::guiFocusState().reset();
            lfs::event::EventBridge::instance().clear_all();
            lfs::core::event::bus().clear_all();
        }

        void TearDown() override {
            services().clear();
            gui::guiFocusState().reset();
            lfs::event::EventBridge::instance().clear_all();
            lfs::core::event::bus().clear_all();
            input::InputBindings::setPersistenceEnabled(previous_persistence_enabled_);
            setHome(previous_home_);
        }

        static void setHome(const std::optional<std::string>& value) {
#ifdef _WIN32
            (void)_putenv_s("LFS_HOME", value ? value->c_str() : "");
#else
            if (value)
                (void)setenv("LFS_HOME", value->c_str(), 1);
            else
                (void)unsetenv("LFS_HOME");
#endif
        }

        static void configure(RenderingManager& rendering) {
            services().set(&rendering);
            auto settings = rendering.getSettings();
            settings.orthographic = true;
            settings.ortho_scale = 70.0f;
            rendering.updateSettings(settings);
        }

        static float renderedScale(const Viewport& viewport, const RenderingManager& rendering) {
            FrameContext context{.viewport = viewport,
                                 .settings = rendering.getSettings(),
                                 .render_size = viewport.windowSize};
            return context.makeFrameView().ortho_scale;
        }

        ViewerOptions projectOptions() const {
            ViewerOptions options;
            options.show_startup_overlay = false;
            options.project_lifecycle_settings_path = temporary_.path / "lifecycle.json";
            return options;
        }

        static VulkanViewportPassParams overlays(VisualizerImpl& viewer) {
            auto* gui = viewer.getGuiManager();
            gui->viewport_layout_.pos = {0.0f, 0.0f};
            gui->viewport_layout_.size = {400.0f, 200.0f};
            return gui->buildVulkanViewportParams({400, 200}, 0);
        }

        lfs::test::licht::TemporaryDirectory temporary_{"lfs-orthographic-zoom"};
        std::optional<std::string> previous_home_;
        bool previous_persistence_enabled_ = true;
    };

    TEST_F(OrthographicZoomTest, WheelZoomsSceneWithRestoredOverrideWithoutMovingCamera) {
        Viewport viewport(200, 200);
        viewport.ortho_scale_override = 100.0f;
        InputController controller(nullptr, viewport);
        RenderingManager rendering;
        configure(rendering);
        const auto eye = viewport.camera.t;
        const auto pivot = viewport.camera.pivot;

        controller.handleScroll(0.0, 1.0);
        EXPECT_NEAR(renderedScale(viewport, rendering), 110.0f, 1e-4f);
        EXPECT_NEAR(rendering.getSettings().ortho_scale, 110.0f, 1e-4f);
        controller.handleScroll(0.0, -1.0);
        EXPECT_NEAR(renderedScale(viewport, rendering), 99.0f, 1e-4f);
        EXPECT_EQ(viewport.camera.t, eye);
        EXPECT_EQ(viewport.camera.pivot, pivot);
    }

    TEST_F(OrthographicZoomTest, WheelUsesGlobalScaleWithoutCreatingPrimaryOverride) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        RenderingManager rendering;
        configure(rendering);
        controller.handleScroll(0.0, 1.0);
        EXPECT_NEAR(renderedScale(viewport, rendering), 77.0f, 1e-4f);
        EXPECT_FALSE(viewport.ortho_scale_override);
    }

    TEST_F(OrthographicZoomTest, WheelKeepsRestoredScaleWithinLimits) {
        Viewport viewport(200, 200);
        InputController controller(nullptr, viewport);
        RenderingManager rendering;
        configure(rendering);
        viewport.ortho_scale_override = 10000.0f;
        controller.handleScroll(0.0, 1.0);
        EXPECT_FLOAT_EQ(renderedScale(viewport, rendering), 10000.0f);
        viewport.ortho_scale_override = 1.0f;
        controller.handleScroll(0.0, -1.0);
        EXPECT_FLOAT_EQ(renderedScale(viewport, rendering), 1.0f);
    }

    TEST_F(OrthographicZoomTest, SecondaryWheelZoomDoesNotChangePrimaryScale) {
        Viewport primary(200, 200);
        primary.ortho_scale_override = 100.0f;
        InputController controller(nullptr, primary);
        RenderingManager rendering;
        configure(rendering);
        rendering.restoreSplitViewMode(SplitViewMode::IndependentDual, primary);
        auto& secondary = rendering.resolvePanelViewport(primary, SplitViewPanelId::Right);
        secondary.ortho_scale_override = 200.0f;
        // SDL's uninitialized pointer is at (0, 0); place that inside the right panel.
        controller.updateViewportBounds(-150, 0, 200, 200);
        controller.handleScroll(0.0, 1.0);
        EXPECT_NEAR(renderedScale(secondary, rendering), 220.0f, 1e-4f);
        EXPECT_FLOAT_EQ(renderedScale(primary, rendering), 100.0f);
        EXPECT_FLOAT_EQ(rendering.getSettings().ortho_scale, 70.0f);
    }

    TEST_F(OrthographicZoomTest, FrustumCacheAndGridRespectViewportScale) {
        VisualizerImpl viewer(projectOptions());
        auto& viewport = viewer.getViewport();
        viewport.windowSize = {400, 200};
        viewport.setViewMatrix(glm::mat3(1.0f), {0.0f, 0.0f, 10.0f});
        viewport.ortho_scale_override = 100.0f;
        auto* rendering = viewer.getRenderingManager();
        configure(*rendering);
        auto settings = rendering->getSettings();
        settings.show_camera_frustums = true;
        rendering->updateSettings(settings);
        auto& scene = viewer.getSceneManager()->getScene();
        const auto group = scene.addCameraGroup("Cameras", scene.addGroup("Dataset"), 1);
        scene.addCamera("camera", group, std::make_shared<core::Camera>(core::Tensor::eye(3, core::Device::CPU), core::Tensor::zeros({3}, core::Device::CPU), 100.0f, 100.0f, 32.0f, 32.0f, core::Tensor(), core::Tensor(), core::CameraModelType::PINHOLE, "camera", std::filesystem::path{}, std::filesystem::path{}, 64, 64, 0));

        const auto first = overlays(viewer);
        ASSERT_FALSE(first.grid_overlays.empty());
        ASSERT_NE(first.frustum_overlay_data, nullptr);
        ASSERT_FALSE(first.frustum_overlay_data->frustum_batches.empty());
        EXPECT_FLOAT_EQ(first.frustum_overlay_data->frustum_batches.front().focal_x, 100.0f);
        EXPECT_NEAR(first.grid_overlays.front().projection[0][0], 0.5f, 1e-5f);

        // Subpixel scale drift must not rebuild the cached screen-space outlines.
        viewport.ortho_scale_override = std::nextafter(100.0f, 101.0f);
        const auto stable = overlays(viewer);
        EXPECT_TRUE(stable.frustum_overlay_data->frustum_batches.front().focal_x == 100.0f);

        // A panel override can change without global settings or camera pose changing.
        viewport.ortho_scale_override = 200.0f;
        const auto second = overlays(viewer);
        ASSERT_FALSE(second.frustum_overlay_data->frustum_batches.empty());
        EXPECT_FLOAT_EQ(second.frustum_overlay_data->frustum_batches.front().focal_x, 200.0f);
        EXPECT_NEAR(second.grid_overlays.front().projection[0][0], 1.0f, 1e-5f);

        // Python/project overrides can be much smaller than wheel zoom's minimum.
        viewport.ortho_scale_override = 1.0e-6f;
        (void)overlays(viewer);
        viewport.ortho_scale_override = 2.0e-6f;
        const auto small = overlays(viewer);
        EXPECT_EQ(small.frustum_overlay_data->frustum_batches.front().focal_x, 2.0e-6f);
    }
} // namespace lfs::vis
