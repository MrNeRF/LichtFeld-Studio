/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Trackpad navigation reads two-finger swipes as orbit/pan/zoom while mouse
// mode keeps wheel zoom; automatic mode swipes only while two fingers rest on
// the trackpad, and pinch zooms in every mode. Drives the controller without a
// window, so the pointer sits at the viewport origin.

#include "input/input_controller.hpp"
#include "input/key_codes.hpp"
#include "internal/viewport.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/rendering_manager.hpp"
#include "screen/screen_service.hpp"
#include "single_view_targets.hpp"
#include "visualizer/app_store.hpp"

#include <SDL3/SDL_keyboard.h>
#include <cmath>
#include <glm/glm.hpp>
#include <gtest/gtest.h>

namespace lfs::vis {

    namespace {
        struct NonActiveViewTargets final : ViewTargets {
            NonActiveViewTargets(screen::ScreenService& screens, const ViewId secondary)
                : screens(screens), secondary(secondary) {}
            ViewTarget activeView() override {
                const auto id = screens.activeView();
                auto* view = screens.view3D(id);
                return {id, &view->camera, {0, 0}, glm::vec2(view->camera.windowSize)};
            }
            ViewTarget viewAt(float, float) override {
                auto* view = screens.view3D(secondary);
                return {secondary, &view->camera, {0, 0}, glm::vec2(view->camera.windowSize)};
            }
            ViewTarget findView(ViewId id) override {
                auto* view = screens.view3D(id);
                return view ? ViewTarget{id, &view->camera, {0, 0}, glm::vec2(view->camera.windowSize)} : ViewTarget{};
            }
            ViewId viewId(const Viewport& viewport) const override {
                for (const auto id : screens.screen().views())
                    if (&screens.view3D(id.value)->camera == &viewport)
                        return id.value;
                return kNoView;
            }
            std::uint64_t viewEpoch() const override { return screens.screenEpoch(); }
            void activateView(ViewId id) override { screens.screen().setActiveView(screen::AreaId{id}); }
            bool runViewCommand(ViewId, std::string_view) override { return false; }

            screen::ScreenService& screens;
            ViewId secondary;
        };

        struct TestViewTargets final : ViewTargets {
            explicit TestViewTargets(Viewport& camera) : camera(camera) {}
            ViewTarget activeView() override { return {1, &camera, {0, 0}, {200, 200}}; }
            ViewTarget viewAt(float x, float y) override {
                auto target = activeView();
                return target.contains(x, y) ? target : ViewTarget{};
            }
            ViewTarget findView(ViewId id) override { return id == 1 ? activeView() : ViewTarget{}; }
            ViewId viewId(const Viewport&) const override { return 1; }
            std::uint64_t viewEpoch() const override { return 1; }
            void activateView(ViewId) override {}
            bool runViewCommand(ViewId, std::string_view) override { return false; }
            Viewport& camera;
        };

        constexpr float kStartDistance = 5.0f;

        class TrackpadNavigationTest : public ::testing::Test {
        protected:
            void SetUp() override {
                SDL_SetModState(SDL_KMOD_NONE);
                // The Viewport constructor reads the user's zoom speed.
                viewport.camera.setZoomSpeed(11.0f);
                viewport.camera.R = glm::mat3(1.0f);
                viewport.camera.t = glm::vec3(0.0f, 0.0f, kStartDistance);
                viewport.camera.pivot = glm::vec3(0.0f);
            }

            void TearDown() override { SDL_SetModState(SDL_KMOD_NONE); }

            [[nodiscard]] float pivotDistance() const {
                return glm::distance(viewport.camera.t, viewport.camera.pivot);
            }

            [[nodiscard]] float rotationChange() const {
                const glm::mat3 identity(1.0f);
                float change = 0.0f;
                for (int col = 0; col < 3; ++col)
                    change += glm::distance(viewport.camera.R[col], identity[col]);
                return change;
            }

            Viewport viewport{200, 200};
            TestViewTargets views{viewport};
            InputController controller{nullptr, views};
        };
    } // namespace

    TEST_F(TrackpadNavigationTest, MouseModeScrollZoomsAndIgnoresHorizontal) {
        controller.handleScroll(1.0, 0.0);
        EXPECT_FLOAT_EQ(pivotDistance(), kStartDistance);
        EXPECT_FLOAT_EQ(rotationChange(), 0.0f);

        controller.handleScroll(0.0, 1.0);
        EXPECT_LT(pivotDistance(), kStartDistance);
        EXPECT_FLOAT_EQ(rotationChange(), 0.0f);
    }

    TEST(TrackpadNavigationViews, WheelZoomUsesTheNonActiveViewsProjectionSettings) {
        screen::ScreenService screens;
        const auto primary = screens.screen().activeView();
        const auto secondary = screens.screen().split(primary, screen::SplitAxis::Columns, 0.5f);
        ASSERT_TRUE(secondary.valid());
        auto* primary_view = screens.view3D(primary);
        auto* secondary_view = screens.view3D(secondary);
        ASSERT_NE(primary_view, nullptr);
        ASSERT_NE(secondary_view, nullptr);
        primary_view->camera.windowSize = {200, 200};
        secondary_view->camera.windowSize = {200, 200};
        screens.editViewSettings(secondary.value, [](ViewSettings& settings) {
            settings.orthographic = true;
            settings.ortho_scale = 20.0f;
        });

        RenderingManager rendering(screens);
        services().set(&rendering);
        NonActiveViewTargets targets(screens, secondary.value);
        InputController controller(nullptr, targets);
        const auto primary_position = primary_view->camera.camera.t;
        const auto secondary_position = secondary_view->camera.camera.t;
        controller.handleScroll(0.0, 1.0);

        EXPECT_FLOAT_EQ(primary_view->settings.ortho_scale, ViewSettings{}.ortho_scale);
        EXPECT_FLOAT_EQ(secondary_view->settings.ortho_scale, 22.0f);
        EXPECT_EQ(primary_view->camera.camera.t, primary_position);
        EXPECT_EQ(secondary_view->camera.camera.t, secondary_position);
        services().clear();
    }

    TEST_F(TrackpadNavigationTest, PinchZoomsInEveryMode) {
        // A pinch arrives as many small scale updates.
        const auto pinch = [this](const float total_scale) {
            constexpr int kUpdates = 20;
            for (int i = 0; i < kUpdates; ++i)
                controller.handlePinch(std::pow(total_scale, 1.0f / kUpdates));
        };
        for (const auto device : {NavigationDevice::Mouse, NavigationDevice::Trackpad, NavigationDevice::Automatic}) {
            SCOPED_TRACE(navigationDeviceName(device));
            SetUp();
            controller.setTrackpadPreferences({.device = device});

            // At the default speed the zoom is the square of the finger scale.
            pinch(1.25f);
            EXPECT_NEAR(pivotDistance(), kStartDistance / (1.25f * 1.25f), 1e-3f);
            pinch(1.0f / 1.25f);
            EXPECT_NEAR(pivotDistance(), kStartDistance, 1e-3f);
            EXPECT_FLOAT_EQ(rotationChange(), 0.0f);
        }
    }

    TEST_F(TrackpadNavigationTest, AutomaticSwipesOnlyWithTwoFingersOnTheTrackpad) {
        controller.setTrackpadPreferences({.device = NavigationDevice::Automatic});

        // Without fingers on the trackpad the scroll comes from a mouse wheel.
        controller.handleScroll(2.0, 0.0);
        EXPECT_FLOAT_EQ(rotationChange(), 0.0f);
        controller.handleScroll(0.0, 1.0);
        const float zoomed = pivotDistance();
        EXPECT_LT(zoomed, kStartDistance);

        // A resting thumb does not turn the wheel into a swipe.
        controller.handleTrackpadTouch(true);
        controller.handleScroll(2.0, 0.0);
        EXPECT_FLOAT_EQ(rotationChange(), 0.0f);

        controller.handleTrackpadTouch(true);
        controller.handleScroll(2.0, 0.0);
        const float orbited = rotationChange();
        EXPECT_GT(orbited, 1e-3f);
        EXPECT_NEAR(pivotDistance(), zoomed, 1e-4f);

        controller.handleTrackpadTouch(false);
        controller.handleTrackpadTouch(false);
        controller.handleScroll(2.0, 0.0);
        EXPECT_FLOAT_EQ(rotationChange(), orbited);
    }

    TEST_F(TrackpadNavigationTest, SwipeOrbitsAroundThePivot) {
        controller.setTrackpadPreferences({.device = NavigationDevice::Trackpad});

        // A horizontal-only swipe must orbit too.
        controller.handleScroll(2.0, 0.0);
        const float after_horizontal = rotationChange();
        EXPECT_GT(after_horizontal, 1e-3f);

        controller.handleScroll(0.0, 2.0);
        EXPECT_GT(rotationChange(), after_horizontal);
        EXPECT_NEAR(pivotDistance(), kStartDistance, 1e-4f);
        EXPECT_FLOAT_EQ(glm::length(viewport.camera.pivot), 0.0f);
    }

    TEST_F(TrackpadNavigationTest, ShiftSwipePansCameraAndPivotTogether) {
        controller.setTrackpadPreferences({.device = NavigationDevice::Trackpad});
        SDL_SetModState(SDL_KMOD_LSHIFT);

        // Scrolling right and up moves the view right and up (content left and down).
        controller.handleScroll(3.0, 1.0);
        const glm::vec3 moved = viewport.camera.t - glm::vec3(0.0f, 0.0f, kStartDistance);
        EXPECT_GT(moved.x, 0.0f);
        EXPECT_GT(moved.y, 0.0f);
        EXPECT_NEAR(glm::distance(viewport.camera.pivot, moved), 0.0f, 1e-5f);
        EXPECT_FLOAT_EQ(rotationChange(), 0.0f);
    }

    TEST_F(TrackpadNavigationTest, CtrlSwipeZooms) {
        controller.setTrackpadPreferences({.device = NavigationDevice::Trackpad});
        SDL_SetModState(SDL_KMOD_LCTRL);

        controller.handleScroll(0.0, 1.0);
        EXPECT_LT(pivotDistance(), kStartDistance);
        EXPECT_FLOAT_EQ(rotationChange(), 0.0f);
    }

    TEST_F(TrackpadNavigationTest, SwipePansAndShiftSwipeOrbitsWhenSwapped) {
        controller.setTrackpadPreferences({.device = NavigationDevice::Trackpad, .swipe_pans = true});

        controller.handleScroll(3.0, 1.0);
        EXPECT_GT(glm::distance(viewport.camera.t, glm::vec3(0.0f, 0.0f, kStartDistance)), 0.0f);
        EXPECT_FLOAT_EQ(rotationChange(), 0.0f);

        SDL_SetModState(SDL_KMOD_LSHIFT);
        controller.handleScroll(3.0, 0.0);
        EXPECT_GT(rotationChange(), 1e-3f);
    }

    TEST_F(TrackpadNavigationTest, SpeedLevelsScaleSwipesAndPinch) {
        // Level 50 is 1x and every 25 levels double the speed.
        const auto yaw_after_swipe = [this](const float swipe_speed) {
            SetUp();
            controller.setTrackpadPreferences({.device = NavigationDevice::Trackpad, .swipe_speed = swipe_speed});
            controller.handleScroll(1.0, 0.0);
            const glm::vec3 forward = lfs::rendering::cameraForward(viewport.camera.R);
            return std::atan2(forward.x, -forward.z);
        };
        EXPECT_NEAR(yaw_after_swipe(75.0f), 2.0f * yaw_after_swipe(50.0f), 1e-4f);

        SetUp();
        controller.setTrackpadPreferences({.device = NavigationDevice::Trackpad, .zoom_speed = 75.0f});
        controller.handlePinch(1.25f);
        EXPECT_NEAR(pivotDistance(), kStartDistance / std::pow(1.25f, 4.0f), 1e-3f);
    }

    TEST_F(TrackpadNavigationTest, RollChordWinsOverSwipeOrbit) {
        controller.setTrackpadPreferences({.device = NavigationDevice::Trackpad});
        const glm::vec3 forward = lfs::rendering::cameraForward(viewport.camera.R);

        controller.handleKey(input::KEY_R, input::ACTION_PRESS, input::MODIFIER_NONE);
        controller.handleScroll(0.0, 1.0);
        controller.handleKey(input::KEY_R, input::ACTION_RELEASE, input::MODIFIER_NONE);

        EXPECT_GT(rotationChange(), 1e-3f);
        EXPECT_NEAR(glm::distance(lfs::rendering::cameraForward(viewport.camera.R), forward), 0.0f, 1e-5f);
        EXPECT_FLOAT_EQ(glm::distance(viewport.camera.t, glm::vec3(0.0f, 0.0f, kStartDistance)), 0.0f);
    }

    TEST_F(TrackpadNavigationTest, FpvSwipeLooksAroundInPlace) {
        controller.setCameraNavigationMode(InputController::CameraNavigationMode::FPV);
        controller.setTrackpadPreferences({.device = NavigationDevice::Trackpad});

        controller.handleScroll(2.0, 0.0);
        EXPECT_GT(rotationChange(), 1e-3f);
        EXPECT_FLOAT_EQ(glm::distance(viewport.camera.t, glm::vec3(0.0f, 0.0f, kStartDistance)), 0.0f);
    }

} // namespace lfs::vis
