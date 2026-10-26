/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "input/gamepad_navigation.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <string_view>

namespace {
    using namespace lfs::vis::input;
    struct NavigationTest : testing::Test {
        GamepadNavigationSession session;
        GamepadNavigationSettings settings{true};
        GamepadNavigationContext context{true, true, 123, 0};
        GamepadSample sample;
        void SetUp() override {
            sample.id = 7;
            sample.generation = 1;
            sample.has_axis.fill(true);
            sample.has_button.fill(true);
        }
        GamepadNavigationIntent tick(float dt = 0.05f) { return session.update(sample, settings, context, dt); }
        void arm() {
            ++context.click_epoch;
            for (int i = 0; i < 3; ++i)
                tick();
            ASSERT_STREQ(session.status(), "ready");
        }
    };
} // namespace

TEST_F(NavigationTest, ConnectionAndHeldSticksCannotArmWithoutViewportClickAndStableNeutral) {
    tick();
    EXPECT_STREQ(session.status(), "viewport");
    sample.axes[SDL_GAMEPAD_AXIS_LEFTX] = 0.7f;
    ++context.click_epoch;
    for (int i = 0; i < 20; ++i)
        EXPECT_FALSE(tick().active);
    EXPECT_STREQ(session.status(), "neutral");
    EXPECT_FALSE(session.awaitingNeutral());
    sample.axes[SDL_GAMEPAD_AXIS_LEFTX] = 0;
    tick();
    EXPECT_TRUE(session.awaitingNeutral());
    tick();
    EXPECT_STREQ(session.status(), "neutral");
    tick();
    EXPECT_STREQ(session.status(), "ready");
}

TEST_F(NavigationTest, FocusAndTargetChangesRequireNewClick) {
    arm();
    context.allowed = false;
    EXPECT_FALSE(tick().active);
    context.allowed = true;
    tick();
    EXPECT_STREQ(session.status(), "viewport");
    arm();
    context.target = 456;
    tick();
    EXPECT_STREQ(session.status(), "viewport");
    arm();
    sample.generation++;
    tick();
    EXPECT_STREQ(session.status(), "viewport");
    sample.id = 0;
    EXPECT_FALSE(tick().active);
    EXPECT_STREQ(session.status(), "no_device");
}

TEST_F(NavigationTest, ModeChangesAndExplicitSuspensionInvalidatePendingInput) {
    arm();
    context.mode = GamepadNavigationMode::Drone;
    EXPECT_FALSE(tick().active);
    EXPECT_STREQ(session.status(), "viewport");
    arm();
    session.suspend(context.click_epoch);
    sample.axes[SDL_GAMEPAD_AXIS_LEFTX] = 1;
    EXPECT_FALSE(tick().active);
}

TEST_F(NavigationTest, RadialDeadZonePreservesPartialDeflectionAndBoundsDiagonals) {
    EXPECT_EQ(GamepadNavigationSession::filterStick({0.05f, 0.05f}, 0.15f), glm::vec2(0));
    EXPECT_NEAR(GamepadNavigationSession::filterStick({0.575f, 0}, 0.15f).x, 0.5f, 1e-6f);
    EXPECT_NEAR(glm::length(GamepadNavigationSession::filterStick({1, 1}, 0.15f)), 1.0f, 1e-6f);
    arm();
    sample.axes[SDL_GAMEPAD_AXIS_LEFTY] = -0.575f;
    sample.axes[SDL_GAMEPAD_AXIS_RIGHTX] = 0.575f;
    const auto intent = tick();
    EXPECT_NEAR(intent.move.y, 0.5f, 1e-6f);
    EXPECT_NEAR(intent.look.x, -0.75f, 1e-6f);
    EXPECT_TRUE(intent.active);
}

TEST_F(NavigationTest, TriggerDifferenceAndPrecisionHaveExplicitPrecedence) {
    arm();
    sample.axes[SDL_GAMEPAD_AXIS_LEFT_TRIGGER] = 1;
    sample.axes[SDL_GAMEPAD_AXIS_RIGHT_TRIGGER] = 1;
    EXPECT_FLOAT_EQ(tick().vertical, 0);
    sample.axes[SDL_GAMEPAD_AXIS_LEFT_TRIGGER] = 0;
    EXPECT_FLOAT_EQ(tick().vertical, 1);
    sample.buttons[SDL_GAMEPAD_BUTTON_LEFT_SHOULDER] = true;
    sample.buttons[SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER] = true;
    EXPECT_FLOAT_EQ(tick().speed, 0.35f);
}

TEST_F(NavigationTest, WalkDoesNotFlyOrSprintAndOnlyEastCanExitWalk) {
    context.mode = GamepadNavigationMode::Walk;
    arm();
    sample.axes[SDL_GAMEPAD_AXIS_RIGHT_TRIGGER] = 1;
    sample.buttons[SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER] = true;
    sample.pressed[SDL_GAMEPAD_BUTTON_SOUTH] = true;
    const auto intent = tick();
    EXPECT_FLOAT_EQ(intent.vertical, 0);
    EXPECT_FLOAT_EQ(intent.speed, 1);
    EXPECT_FALSE(intent.exit_walk);
    EXPECT_FALSE(intent.active);
    sample.pressed[SDL_GAMEPAD_BUTTON_EAST] = true;
    EXPECT_TRUE(tick().exit_walk);
    context.mode = GamepadNavigationMode::Orbit;
    EXPECT_FALSE(tick().exit_walk);
}

TEST_F(NavigationTest, ProjectionCapabilitiesAndNonFiniteInputBlockAllMotion) {
    arm();
    context.supported_projection = false;
    EXPECT_FALSE(tick().active);
    EXPECT_STREQ(session.status(), "projection");
    context.supported_projection = true;
    sample.has_axis[SDL_GAMEPAD_AXIS_RIGHTY] = false;
    EXPECT_FALSE(tick().active);
    EXPECT_STREQ(session.status(), "mapping");
    sample.has_axis.fill(true);
    arm();
    sample.axes[SDL_GAMEPAD_AXIS_LEFTX] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(tick().active);
    EXPECT_STREQ(session.status(), "suspended");
}

TEST_F(NavigationTest, NeutralRequiresReleasedButtonsAndTriggers) {
    sample.buttons[SDL_GAMEPAD_BUTTON_SOUTH] = true;
    ++context.click_epoch;
    for (int i = 0; i < 10; ++i)
        EXPECT_FALSE(tick().active);
    EXPECT_STREQ(session.status(), "neutral");
    sample.buttons.fill(false);
    sample.axes[SDL_GAMEPAD_AXIS_LEFT_TRIGGER] = 0.1f;
    for (int i = 0; i < 10; ++i)
        tick();
    EXPECT_STREQ(session.status(), "neutral");
    sample.axes.fill(0);
    arm();
}

TEST_F(NavigationTest, DisabledAndInvalidTimeCannotMoveOrAccumulateRearmTime) {
    arm();
    settings.enabled = false;
    EXPECT_FALSE(tick().active);
    EXPECT_STREQ(session.status(), "disabled");
    settings.enabled = true;
    arm();
    EXPECT_FALSE(tick(std::numeric_limits<float>::infinity()).active);
    EXPECT_FALSE(tick(-1).active);
    EXPECT_FALSE(tick(0).active);
}

TEST_F(NavigationTest, InvalidSettingsAreRejectedBeforeConfiguration) {
    for (const float value : {-1.0f, 1.0f, std::numeric_limits<float>::quiet_NaN()}) {
        settings.dead_zone = value;
        EXPECT_THROW(settings.validate(), std::invalid_argument);
    }
    settings = GamepadNavigationSettings{};
    EXPECT_NO_THROW(settings.validate());
    settings.move_speed = std::numeric_limits<float>::infinity();
    EXPECT_THROW(settings.validate(), std::invalid_argument);
}

TEST_F(NavigationTest, DiagonalDriftCannotPassTheRadialNeutralGate) {
    ++context.click_epoch;
    sample.axes[SDL_GAMEPAD_AXIS_LEFTX] = 0.12f;
    sample.axes[SDL_GAMEPAD_AXIS_LEFTY] = 0.12f;
    for (int i = 0; i < 20; ++i)
        EXPECT_FALSE(tick().active);
    EXPECT_STREQ(session.status(), "neutral");
    sample.axes.fill(0);
    arm();
}

TEST_F(NavigationTest, NeutralDurationUsesTimeRatherThanFrameCount) {
    for (int fps : {30, 60, 144}) {
        session = GamepadNavigationSession{};
        ++context.click_epoch;
        int frames = 0;
        do {
            tick(1.0f / fps);
            ++frames;
            ASSERT_LE(frames, fps);
        } while (std::string_view(session.status()) != "ready");
        const float seconds = float(frames) / fps;
        EXPECT_GE(seconds, 0.12f);
        EXPECT_LE(seconds, 0.12f + 1.0f / fps);
    }
}
