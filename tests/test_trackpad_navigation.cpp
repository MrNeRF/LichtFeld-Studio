/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Trackpad navigation reads two-finger swipes as orbit/pan/zoom while mouse
// mode keeps wheel zoom; pinch zooms in both. Drives the controller without a
// window, so the pointer sits at the viewport origin.

#include "input/input_controller.hpp"
#include "input/key_codes.hpp"
#include "internal/viewport.hpp"
#include "rendering/coordinate_conventions.hpp"

#include <SDL3/SDL_keyboard.h>
#include <cmath>
#include <glm/glm.hpp>
#include <gtest/gtest.h>

namespace lfs::vis {

    namespace {
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
            InputController controller{nullptr, viewport};
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

    TEST_F(TrackpadNavigationTest, PinchZoomsInBothModes) {
        // A pinch arrives as many small scale updates.
        const auto pinch = [this](const float total_scale) {
            constexpr int kUpdates = 20;
            for (int i = 0; i < kUpdates; ++i)
                controller.handlePinch(std::pow(total_scale, 1.0f / kUpdates));
        };
        for (const bool trackpad : {false, true}) {
            SCOPED_TRACE(trackpad);
            SetUp();
            controller.setTrackpadPreferences({.enabled = trackpad});

            // At the default speed the zoom is the square of the finger scale.
            pinch(1.25f);
            EXPECT_NEAR(pivotDistance(), kStartDistance / (1.25f * 1.25f), 1e-3f);
            pinch(1.0f / 1.25f);
            EXPECT_NEAR(pivotDistance(), kStartDistance, 1e-3f);
            EXPECT_FLOAT_EQ(rotationChange(), 0.0f);
        }
    }

    TEST_F(TrackpadNavigationTest, SwipeOrbitsAroundThePivot) {
        controller.setTrackpadPreferences({.enabled = true});

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
        controller.setTrackpadPreferences({.enabled = true});
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
        controller.setTrackpadPreferences({.enabled = true});
        SDL_SetModState(SDL_KMOD_LCTRL);

        controller.handleScroll(0.0, 1.0);
        EXPECT_LT(pivotDistance(), kStartDistance);
        EXPECT_FLOAT_EQ(rotationChange(), 0.0f);
    }

    TEST_F(TrackpadNavigationTest, SwipePansAndShiftSwipeOrbitsWhenSwapped) {
        controller.setTrackpadPreferences({.enabled = true, .swipe_pans = true});

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
            controller.setTrackpadPreferences({.enabled = true, .swipe_speed = swipe_speed});
            controller.handleScroll(1.0, 0.0);
            const glm::vec3 forward = lfs::rendering::cameraForward(viewport.camera.R);
            return std::atan2(forward.x, -forward.z);
        };
        EXPECT_NEAR(yaw_after_swipe(75.0f), 2.0f * yaw_after_swipe(50.0f), 1e-4f);

        SetUp();
        controller.setTrackpadPreferences({.enabled = true, .zoom_speed = 75.0f});
        controller.handlePinch(1.25f);
        EXPECT_NEAR(pivotDistance(), kStartDistance / std::pow(1.25f, 4.0f), 1e-3f);
    }

    TEST_F(TrackpadNavigationTest, RollChordWinsOverSwipeOrbit) {
        controller.setTrackpadPreferences({.enabled = true});
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
        controller.setTrackpadPreferences({.enabled = true});

        controller.handleScroll(2.0, 0.0);
        EXPECT_GT(rotationChange(), 1e-3f);
        EXPECT_FLOAT_EQ(glm::distance(viewport.camera.t, glm::vec3(0.0f, 0.0f, kStartDistance)), 0.0f);
    }

} // namespace lfs::vis
