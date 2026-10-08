/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bus.hpp"
#include "core/services.hpp"
#include "core/tensor_backend.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/gui_manager.hpp"
#include "input/input_controller.hpp"
#include "licht_test_support.hpp"
#include "rendering/render_pass.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "screen/screen_service.hpp"
#include "test_view_targets.hpp"
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

        static ViewportFrameDesc overlays(VisualizerImpl& viewer) {
            auto* gui = viewer.getGuiManager();
            gui->viewport_layout_.pos = {0.0f, 0.0f};
            gui->viewport_layout_.size = {400.0f, 200.0f};
            gui->ui_hidden_ = true;
            gui->viewport_layout_.view = viewer.activeView().id;
            return gui->buildViewportFrameDesc(viewer.activeView().id, {400, 200}, 0);
        }

        lfs::test::licht::TemporaryDirectory temporary_{"lfs-orthographic-zoom"};
        std::optional<std::string> previous_home_;
        bool previous_persistence_enabled_ = true;
    };

    TEST_F(OrthographicZoomTest, WheelZoomsRestoredViewScaleWithoutMovingCamera) {
        Viewport viewport(200, 200);
        TestViewTargets targets(viewport);
        InputController controller(nullptr, targets);
        screen::ScreenService screens;
        RenderingManager rendering(screens);
        configure(rendering);
        rendering.editViewSettings(screens.activeView(), [](ViewSettings& settings) { settings.ortho_scale = 100.0f; });
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

    TEST_F(OrthographicZoomTest, WheelUsesInitialViewScale) {
        Viewport viewport(200, 200);
        TestViewTargets targets(viewport);
        InputController controller(nullptr, targets);
        screen::ScreenService screens;
        RenderingManager rendering(screens);
        configure(rendering);
        controller.handleScroll(0.0, 1.0);
        EXPECT_NEAR(renderedScale(viewport, rendering), 77.0f, 1e-4f);
        EXPECT_NEAR(rendering.getSettings().ortho_scale, 77.0f, 1e-4f);
    }

    TEST_F(OrthographicZoomTest, WheelKeepsRestoredScaleWithinLimits) {
        Viewport viewport(200, 200);
        TestViewTargets targets(viewport);
        InputController controller(nullptr, targets);
        screen::ScreenService screens;
        RenderingManager rendering(screens);
        configure(rendering);
        rendering.editViewSettings(screens.activeView(), [](ViewSettings& settings) { settings.ortho_scale = 10000.0f; });
        controller.handleScroll(0.0, 1.0);
        EXPECT_FLOAT_EQ(renderedScale(viewport, rendering), 10000.0f);
        rendering.editViewSettings(screens.activeView(), [](ViewSettings& settings) { settings.ortho_scale = 1.0f; });
        controller.handleScroll(0.0, -1.0);
        EXPECT_FLOAT_EQ(renderedScale(viewport, rendering), 1.0f);
    }

    TEST_F(OrthographicZoomTest, SecondaryWheelZoomDoesNotChangePrimaryScale) {
        screen::ScreenService screens;
        const auto primary_id = screens.screen().activeView();
        const auto secondary_id = screens.screen().split(primary_id, screen::SplitAxis::Columns, 0.5f);
        ASSERT_TRUE(secondary_id.valid());
        auto& primary = screens.view3D(primary_id)->camera;
        auto& secondary = screens.view3D(secondary_id)->camera;
        primary.windowSize = secondary.windowSize = {200, 200};
        RenderingManager rendering(screens);
        configure(rendering);
        rendering.editViewSettings(secondary_id.value, [](ViewSettings& settings) {
            settings.orthographic = true;
            settings.ortho_scale = 200.0f;
        });
        struct SecondaryTargets : ViewTargets {
            SecondaryTargets(Viewport& camera, ViewId id) : camera(camera), id(id) {}
            ViewTarget activeView() override { return {id, &camera, {0, 0}, {200, 200}}; }
            ViewTarget viewAt(float, float) override { return activeView(); }
            ViewTarget findView(ViewId value) override { return value == id ? activeView() : ViewTarget{}; }
            ViewId viewId(const Viewport&) const override { return id; }
            std::uint64_t viewEpoch() const override { return 1; }
            void activateView(ViewId) override {}
            bool runViewCommand(ViewId, std::string_view) override { return false; }
            Viewport& camera;
            ViewId id;
        } targets(secondary, secondary_id.value);
        InputController controller(nullptr, targets);
        controller.handleScroll(0.0, 1.0);
        const FrameContext secondary_context{.viewport = secondary, .settings = rendering.settingsForView(secondary_id.value), .render_size = secondary.windowSize};
        EXPECT_NEAR(secondary_context.makeFrameView().ortho_scale, 220.0f, 1e-4f);
        EXPECT_NEAR(rendering.settingsForView(secondary_id.value).ortho_scale, 220.0f, 1e-4f);
        EXPECT_FLOAT_EQ(rendering.settingsForView(primary_id.value).ortho_scale, 70.0f);
        EXPECT_FLOAT_EQ(rendering.getSettings().ortho_scale, 70.0f);
    }

    TEST_F(OrthographicZoomTest, FrustumCacheAndGridRespectViewportScale) {
        // Camera construction uploads its derived pose even when R/T are CPU tensors.
        // Keep the four wheel/scale contracts above runnable without a GPU.
        const auto backend = core::default_gpu_backend();
        if (!core::gpu_backend_available(backend)) {
            GTEST_SKIP() << "Camera pose requires GPU backend " << core::gpu_backend_name(backend);
        }
        VisualizerImpl viewer(projectOptions());
        auto& viewport = viewer.getViewport();
        viewport.windowSize = {400, 200};
        viewport.setViewMatrix(glm::mat3(1.0f), {0.0f, 0.0f, 10.0f});
        auto* rendering = viewer.getRenderingManager();
        configure(*rendering);
        auto settings = rendering->getSettings();
        settings.show_camera_frustums = true;
        settings.ortho_scale = 100.0f;
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
        rendering->editViewSettings(viewer.activeView().id, [](ViewSettings& settings) { settings.ortho_scale = std::nextafter(100.0f, 101.0f); });
        const auto stable = overlays(viewer);
        EXPECT_TRUE(stable.frustum_overlay_data->frustum_batches.front().focal_x == 100.0f);

        // A view scale can change without scene settings or camera pose changing.
        rendering->editViewSettings(viewer.activeView().id, [](ViewSettings& settings) { settings.ortho_scale = 200.0f; });
        const auto second = overlays(viewer);
        ASSERT_FALSE(second.frustum_overlay_data->frustum_batches.empty());
        EXPECT_FLOAT_EQ(second.frustum_overlay_data->frustum_batches.front().focal_x, 200.0f);
        EXPECT_NEAR(second.grid_overlays.front().projection[0][0], 1.0f, 1e-5f);

        // Python/project settings can be much smaller than wheel zoom's minimum.
        rendering->editViewSettings(viewer.activeView().id, [](ViewSettings& settings) { settings.ortho_scale = 1.0e-6f; });
        (void)overlays(viewer);
        rendering->editViewSettings(viewer.activeView().id, [](ViewSettings& settings) { settings.ortho_scale = 2.0e-6f; });
        const auto small = overlays(viewer);
        EXPECT_EQ(small.frustum_overlay_data->frustum_batches.front().focal_x, 2.0e-6f);
    }
} // namespace lfs::vis
