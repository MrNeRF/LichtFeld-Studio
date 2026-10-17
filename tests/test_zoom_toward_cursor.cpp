/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/event_bridge.hpp"
#include "core/services.hpp"
#include "gui/gui_focus_state.hpp"
#include "input/injected_pointer.hpp"
#include "input/input_controller.hpp"
#include "input/zoom_toward_cursor.hpp"
#include "licht_test_support.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/rendering_manager.hpp"
#include "screen/screen_service.hpp"
#include "test_view_targets.hpp"

#include <SDL3/SDL_events.h>
#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <optional>
#include <string>

namespace lfs::vis {
    namespace {
        constexpr float kFocalLengthMm = 35.0f;

        void expectNear(const glm::vec3& actual, const glm::vec3& expected, const float tolerance) {
            EXPECT_NEAR(actual.x, expected.x, tolerance);
            EXPECT_NEAR(actual.y, expected.y, tolerance);
            EXPECT_NEAR(actual.z, expected.z, tolerance);
        }

        glm::vec2 projectToScreen(const Viewport& viewport, const glm::vec3& world) {
            const auto screen = lfs::rendering::projectWorldPoint(viewport.camera.R, viewport.camera.t,
                                                                  viewport.windowSize, world, kFocalLengthMm);
            EXPECT_TRUE(screen.has_value());
            return screen.value_or(glm::vec2(0.0f));
        }

        // A camera 10 units in front of the origin, looking at it.
        Viewport lookingAtOrigin() {
            Viewport viewport(400, 300);
            viewport.camera.t = glm::vec3(2.0f, 3.0f, 10.0f);
            viewport.camera.pivot = glm::vec3(0.0f);
            viewport.camera.R = lfs::rendering::makeVisualizerLookAtRotation(viewport.camera.t, viewport.camera.pivot);
            return viewport;
        }

        glm::vec3 orthoWorldAt(const glm::mat3& rotation, const glm::vec3& translation, const glm::ivec2 render_size,
                               const glm::vec2 render_pixel, const float ortho_scale) {
            return lfs::rendering::unprojectScreenPoint(rotation, translation, render_size, render_pixel.x,
                                                        render_pixel.y, 5.0f, kFocalLengthMm, true, ortho_scale);
        }

        // Clears the injected pointer so later tests see the real (absent) mouse.
        void releaseInjectedPointer() {
            SDL_Event move{};
            move.type = SDL_EVENT_MOUSE_MOTION;
            (void)input::prepareInjectedPointerEvent(move);
            input::finishInjectedPointerFrame();
        }
    } // namespace

    TEST(ZoomTowardCursorMath, DollyTowardKeepsAnchorAtTheSameScreenPosition) {
        auto viewport = lookingAtOrigin();
        const glm::vec2 cursor(310.0f, 70.0f);
        const glm::vec3 anchor = viewport.unprojectPixel(cursor.x, cursor.y, 8.0f, kFocalLengthMm);

        for (const float fraction : {0.3f, -0.4f}) {
            const glm::vec2 before = projectToScreen(viewport, anchor);
            viewport.camera.dollyToward(anchor, fraction);
            const glm::vec2 after = projectToScreen(viewport, anchor);
            EXPECT_NEAR(after.x, before.x, 1e-2f);
            EXPECT_NEAR(after.y, before.y, 1e-2f);
        }
        const glm::vec2 final_position = projectToScreen(viewport, anchor);
        EXPECT_NEAR(final_position.x, cursor.x, 1e-2f);
        EXPECT_NEAR(final_position.y, cursor.y, 1e-2f);
    }

    TEST(ZoomTowardCursorMath, DollyTowardKeepsPivotOnTheViewAxisAndScalesItsDistance) {
        auto viewport = lookingAtOrigin();
        const glm::vec3 anchor = viewport.unprojectPixel(320.0f, 220.0f, 6.0f, kFocalLengthMm);
        const float distance = glm::length(viewport.camera.pivot - viewport.camera.t);

        viewport.camera.dollyToward(anchor, 0.25f);

        const glm::vec3 to_pivot = viewport.camera.pivot - viewport.camera.t;
        expectNear(glm::normalize(to_pivot), lfs::rendering::cameraForward(viewport.camera.R), 1e-5f);
        EXPECT_NEAR(glm::length(to_pivot), 0.75f * distance, 1e-4f);
    }

    TEST(ZoomTowardCursorMath, NearAnchorFallsBackToCenteredDollyRelativeToSceneSize) {
        auto viewport = lookingAtOrigin();
        const glm::vec3 forward = lfs::rendering::cameraForward(viewport.camera.R);
        const glm::vec3 anchor = viewport.camera.t + 0.5f * glm::normalize(forward + glm::vec3(0.1f, 0.0f, 0.0f));

        // In a 1000-unit scene half a unit is "at the surface": zoom keeps going straight ahead.
        auto large = viewport;
        auto centered = viewport;
        large.camera.setSceneExtent(1000.0f);
        large.camera.dollyToward(anchor, 0.2f);
        centered.camera.dolly(0.2f);
        expectNear(large.camera.t, centered.camera.t, 1e-5f);

        // In a 1-unit scene the same half unit is still far away: zoom heads for the anchor.
        auto small = viewport;
        small.camera.setSceneExtent(1.0f);
        small.camera.dollyToward(anchor, 0.2f);
        expectNear(small.camera.t, viewport.camera.t + 0.2f * (anchor - viewport.camera.t), 1e-5f);
    }

    TEST(ZoomTowardCursorMath, AnchorBehindTheCameraNeverPullsZoomInBackwards) {
        auto viewport = lookingAtOrigin();
        const glm::vec3 forward = lfs::rendering::cameraForward(viewport.camera.R);
        // Pushed through the surface during a gesture: the cached anchor is now behind.
        const glm::vec3 anchor = viewport.camera.t - 3.0f * forward + glm::vec3(0.2f, 0.0f, 0.0f);
        viewport.camera.setSceneExtent(10.0f);
        const glm::vec3 eye = viewport.camera.t;

        viewport.camera.dollyToward(anchor, 0.2f);

        EXPECT_GT(glm::dot(viewport.camera.t - eye, forward), 0.0f);
    }

    TEST(ZoomTowardCursorMath, ZoomOutAfterPassingTheAnchorMovesBackwards) {
        auto viewport = lookingAtOrigin();
        const glm::vec3 forward = lfs::rendering::cameraForward(viewport.camera.R);
        const glm::vec3 anchor = viewport.camera.t - 3.0f * forward + glm::vec3(0.2f, 0.0f, 0.0f);
        viewport.camera.setSceneExtent(10.0f);
        const glm::vec3 eye = viewport.camera.t;

        viewport.camera.dollyToward(anchor, -0.2f);

        EXPECT_LT(glm::dot(viewport.camera.t - eye, forward), 0.0f);
    }

    TEST(ZoomTowardCursorMath, RepeatedZoomOutStaysBoundedAndCanZoomBackIn) {
        auto viewport = lookingAtOrigin();
        const glm::vec3 anchor = viewport.unprojectPixel(310.0f, 70.0f, 8.0f, kFocalLengthMm);
        // Zoom speed 100: every notch doubles the distances without a bound.
        for (int notch = 0; notch < 200; ++notch)
            viewport.camera.dollyToward(anchor, -1.0f);

        const float pivot_distance = glm::length(viewport.camera.pivot - viewport.camera.t);
        EXPECT_TRUE(std::isfinite(pivot_distance));
        EXPECT_LE(pivot_distance, lfs::rendering::DEFAULT_FAR_PLANE * 1.001f);
        EXPECT_LE(glm::length(anchor - viewport.camera.t), lfs::rendering::DEFAULT_FAR_PLANE * 1.001f);

        viewport.camera.dollyToward(anchor, 0.5f);
        const float closer = glm::length(viewport.camera.pivot - viewport.camera.t);
        EXPECT_NEAR(closer, 0.5f * pivot_distance, 1e-3f * pivot_distance);
    }

    TEST(ZoomTowardCursorMath, OrthographicShiftKeepsCursorPointWhenRenderMatchesPanel) {
        const glm::mat3 rotation = lfs::rendering::makeVisualizerLookAtRotation({1.0f, 4.0f, 6.0f}, glm::vec3(0.0f));
        const glm::vec3 translation(1.0f, 4.0f, 6.0f);
        const glm::ivec2 render_size(400, 300);
        const glm::vec2 local(330.0f, 40.0f);

        const glm::vec2 offset = input::orthoCursorOffset(local, glm::vec2(render_size), render_size);
        EXPECT_FLOAT_EQ(offset.x, 130.0f);
        EXPECT_FLOAT_EQ(offset.y, -110.0f);

        const glm::vec3 before = orthoWorldAt(rotation, translation, render_size, local, 40.0f);
        const glm::vec3 shift = input::orthoZoomShift(rotation, offset, 40.0f, 44.0f);
        const glm::vec3 after = orthoWorldAt(rotation, translation + shift, render_size, local, 44.0f);
        expectNear(after, before, 1e-4f);
    }

    TEST(ZoomTowardCursorMath, OrthographicShiftUsesRenderPixelsWhenRenderDiffersFromPanel) {
        const glm::mat3 rotation(1.0f);
        const glm::vec3 translation(0.0f, 0.0f, 10.0f);
        // Render scale 0.5 (or a HiDPI panel): 800x600 on screen, 400x300 rendered.
        const glm::vec2 panel_size(800.0f, 600.0f);
        const glm::ivec2 render_size(400, 300);
        const glm::vec2 local(600.0f, 100.0f);
        const glm::vec2 render_pixel = local * 0.5f;

        const glm::vec2 offset = input::orthoCursorOffset(local, panel_size, render_size);
        EXPECT_FLOAT_EQ(offset.x, 100.0f);
        EXPECT_FLOAT_EQ(offset.y, -100.0f);

        const glm::vec3 before = orthoWorldAt(rotation, translation, render_size, render_pixel, 2.0f);
        const glm::vec3 shift = input::orthoZoomShift(rotation, offset, 2.0f, 1.5f);
        const glm::vec3 after = orthoWorldAt(rotation, translation + shift, render_size, render_pixel, 1.5f);
        expectNear(after, before, 1e-4f);
    }

    TEST(ZoomTowardCursorMath, AnchorIsReadOncePerGesture) {
        using Clock = input::ZoomAnchorCache::Clock;
        input::ZoomAnchorCache cache;
        const glm::mat3 rotation(1.0f);
        const glm::vec3 eye(0.0f, 0.0f, 10.0f);
        const auto start = Clock::now();
        const glm::dvec2 cursor(120.0, 80.0);
        EXPECT_FALSE(cache.matches(1, cursor, start, rotation, eye));

        cache.store(1, cursor, glm::vec3(1.0f, 2.0f, 3.0f), start);
        // The anchor serves only once a zoom step has recorded its resulting pose.
        EXPECT_FALSE(cache.matches(1, cursor, start, rotation, eye));
        cache.settle(rotation, eye);
        const auto next = start + std::chrono::milliseconds(50);
        ASSERT_TRUE(cache.matches(1, cursor + glm::dvec2(1.0, 1.0), next, rotation, eye));
        EXPECT_EQ(cache.reuse(next), std::optional(glm::vec3(1.0f, 2.0f, 3.0f)));

        // Each event extends the gesture: 330 ms after the first read is still
        // the same gesture because the event at 50 ms restarted the idle timer.
        const auto later = next + std::chrono::milliseconds(280);
        EXPECT_TRUE(cache.matches(1, cursor, later, rotation, eye));

        EXPECT_FALSE(cache.matches(1, cursor + glm::dvec2(5.0, 0.0), next, rotation, eye));
        EXPECT_FALSE(cache.matches(2, cursor, next, rotation, eye));
        EXPECT_FALSE(cache.matches(1, cursor, next + input::ZoomAnchorCache::kGestureIdle + std::chrono::milliseconds(1),
                                   rotation, eye));

        // Any other camera motion between zoom events (orbit, pan, WASD) starts a new gesture.
        EXPECT_FALSE(cache.matches(1, cursor, next, rotation, eye + glm::vec3(0.0f, 0.01f, 0.0f)));
        const glm::mat3 orbited = lfs::rendering::makeVisualizerLookAtRotation(eye, glm::vec3(1.0f, 0.0f, 0.0f));
        EXPECT_FALSE(cache.matches(1, cursor, next, orbited, eye));

        // Background is cached too, so it is not re-read on every notch either.
        cache.store(1, cursor, std::nullopt, later);
        cache.settle(rotation, eye);
        ASSERT_TRUE(cache.matches(1, cursor, later, rotation, eye));
        EXPECT_FALSE(cache.reuse(later).has_value());
        cache.clear();
        EXPECT_FALSE(cache.matches(1, cursor, later, rotation, eye));
    }

    class ZoomTowardCursorControllerTest : public ::testing::Test {
    protected:
        void SetUp() override {
            if (const auto* previous = std::getenv("LFS_HOME"))
                previous_home_ = previous;
            setHome(temporary_.path.string());
            services().clear();
            gui::guiFocusState().reset();
            lfs::event::EventBridge::instance().clear_all();
        }

        void TearDown() override {
            releaseInjectedPointer();
            services().clear();
            gui::guiFocusState().reset();
            lfs::event::EventBridge::instance().clear_all();
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

        static void configureOrthographic(RenderingManager& rendering, const ViewId view) {
            services().set(&rendering);
            rendering.editViewSettings(view, [](ViewSettings& settings) {
                settings.orthographic = true;
                settings.ortho_scale = 50.0f;
            });
        }

        lfs::test::licht::TemporaryDirectory temporary_{"lfs-zoom-toward-cursor"};
        std::optional<std::string> previous_home_;
    };

    TEST_F(ZoomTowardCursorControllerTest, OffByDefaultSoWheelZoomStaysCentered) {
        Viewport viewport(200, 200);
        TestViewTargets targets(viewport);
        InputController controller(nullptr, targets);
        screen::ScreenService screens;
        RenderingManager rendering(screens);
        configureOrthographic(rendering, screens.activeView());
        EXPECT_FALSE(controller.zoomTowardCursorEnabled());

        const auto eye = viewport.camera.t;
        input::injectPointerMove(170.0f, 40.0f);
        controller.handleScroll(0.0, 1.0);

        EXPECT_NEAR(rendering.settingsForView(screens.activeView()).ortho_scale, 55.0f, 1e-4f);
        EXPECT_EQ(viewport.camera.t, eye);
    }

    TEST_F(ZoomTowardCursorControllerTest, OrthographicWheelKeepsThePointUnderTheCursor) {
        Viewport viewport(200, 200);
        TestViewTargets targets(viewport);
        InputController controller(nullptr, targets);
        controller.setZoomTowardCursorEnabled(true);
        screen::ScreenService screens;
        RenderingManager rendering(screens);
        configureOrthographic(rendering, screens.activeView());
        const glm::vec2 cursor(170.0f, 40.0f);
        const glm::vec3 before = orthoWorldAt(viewport.camera.R, viewport.camera.t, viewport.windowSize, cursor, 50.0f);

        input::injectPointerMove(cursor.x, cursor.y);
        controller.handleScroll(0.0, 1.0);

        const float scale = rendering.settingsForView(screens.activeView()).ortho_scale;
        EXPECT_NEAR(scale, 55.0f, 1e-4f);
        const glm::vec3 after = orthoWorldAt(viewport.camera.R, viewport.camera.t, viewport.windowSize, cursor, scale);
        expectNear(after, before, 1e-4f);
    }

    TEST_F(ZoomTowardCursorControllerTest, PerspectiveZoomOverBackgroundFallsBackToCenteredZoom) {
        Viewport viewport(200, 200);
        TestViewTargets targets(viewport);
        InputController controller(nullptr, targets);
        controller.setZoomTowardCursorEnabled(true);
        screen::ScreenService screens;
        RenderingManager rendering(screens);
        services().set(&rendering);
        ASSERT_FALSE(rendering.settingsForView(screens.activeView()).orthographic);
        // Nothing has been rendered, so there is no depth under the cursor.
        auto centered = viewport;
        centered.camera.zoom(1.0f);

        input::injectPointerMove(170.0f, 40.0f);
        controller.handleScroll(0.0, 1.0);

        expectNear(viewport.camera.t, centered.camera.t, 1e-5f);
        expectNear(viewport.camera.pivot, centered.camera.pivot, 1e-5f);
    }

    TEST_F(ZoomTowardCursorControllerTest, ComparisonSplitFallsBackToCenteredZoom) {
        Viewport viewport(200, 200);
        TestViewTargets targets(viewport);
        InputController controller(nullptr, targets);
        controller.setZoomTowardCursorEnabled(true);
        screen::ScreenService screens;
        RenderingManager rendering(screens);
        configureOrthographic(rendering, screens.activeView());
        rendering.editViewSettings(screens.activeView(), [](ViewSettings& settings) {
            settings.split_view_mode = SplitViewMode::PLYComparison;
        });
        const auto eye = viewport.camera.t;

        input::injectPointerMove(170.0f, 40.0f);
        controller.handleScroll(0.0, 1.0);

        EXPECT_NEAR(rendering.settingsForView(screens.activeView()).ortho_scale, 55.0f, 1e-4f);
        EXPECT_EQ(viewport.camera.t, eye);
    }
} // namespace lfs::vis
