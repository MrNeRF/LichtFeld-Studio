/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "input/gamepad_navigation.hpp"
#include "input/gamepad_profiles.hpp"
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

TEST_F(NavigationTest, ShortTriggerCommandsAreConsumedEvenAfterTheAxisReturnsToZero) {
    settings.bindings.common[actionIndex(GamepadAction::Vertical)] = {};
    settings.bindings.common[actionIndex(GamepadAction::NextNavigation)] = {SDL_GAMEPAD_AXIS_LEFT_TRIGGER};
    arm();
    sample.trigger_pressed[0] = true;
    EXPECT_EQ(tick().navigation_step, 1);
    EXPECT_TRUE(GamepadNavigationSession::needsSampleUpdate(sample, sample, settings.dead_zone));
    sample.trigger_pressed.fill(false);
    EXPECT_EQ(tick().navigation_step, 0);
    sample.axes[SDL_GAMEPAD_AXIS_LEFT_TRIGGER] = .8f;
    EXPECT_EQ(tick().navigation_step, 1);
    EXPECT_EQ(tick().navigation_step, 0);
}

TEST_F(NavigationTest, SequencerTriggerCrossingDoesNotReplayAfterBlockedTimelineEditing) {
    context.sequencer_open = true;
    settings.bindings.common[actionIndex(GamepadAction::Vertical)] = {};
    settings.bindings.common[actionIndex(GamepadAction::AddKeyframe)] = {SDL_GAMEPAD_AXIS_RIGHT_TRIGGER};
    arm();
    sample.trigger_pressed[1] = true;
    EXPECT_TRUE(tick().add_keyframe);
    sample.trigger_pressed.fill(false);
    EXPECT_FALSE(tick().add_keyframe);
    context.sequencer_commands_allowed = false;
    sample.trigger_pressed[1] = true;
    EXPECT_FALSE(tick().add_keyframe);
    sample.trigger_pressed.fill(false);
    context.sequencer_commands_allowed = true;
    EXPECT_FALSE(tick().add_keyframe);
}

TEST_F(NavigationTest, SequencerButtonsDoNotRequireCameraProjectionOrStickMapping) {
    context.sequencer_open = true;
    context.supported_projection = false;
    context.mode = GamepadNavigationMode::FPV;
    sample.has_axis.fill(false);
    arm();
    sample.pressed[SDL_GAMEPAD_BUTTON_WEST] = true;
    const auto command = tick();
    EXPECT_TRUE(command.add_keyframe);
    EXPECT_EQ(command.move, glm::vec2(0));
    EXPECT_EQ(command.look, glm::vec2(0));
    context.allowed = false;
    EXPECT_FALSE(tick().add_keyframe);
}

TEST_F(NavigationTest, WakeFilteringIgnoresNoiseButRetainsNeutralAndButtonTransitions) {
    auto next = sample;
    next.axes[0] = .02f;
    next.axes[2] = -.03f;
    next.axes[4] = .01f;
    EXPECT_FALSE(GamepadNavigationSession::needsSampleUpdate(sample, next, .15f));
    next.axes[0] = .13f; // crosses the arming hysteresis threshold
    EXPECT_TRUE(GamepadNavigationSession::needsSampleUpdate(sample, next, .15f));
    next = sample;
    next.pressed[SDL_GAMEPAD_BUTTON_WEST] = true;
    EXPECT_TRUE(GamepadNavigationSession::needsSampleUpdate(sample, next, .15f));
    next = sample;
    ++next.generation;
    EXPECT_TRUE(GamepadNavigationSession::needsSampleUpdate(sample, next, .15f));
}

TEST_F(NavigationTest, TemporaryInterruptionRetainsAssignmentButTargetAndDeviceChangesRequireClick) {
    arm();
    context.allowed = false;
    EXPECT_FALSE(tick().active);
    context.allowed = true;
    for (int i = 0; i < 3; ++i)
        tick();
    EXPECT_STREQ(session.status(), "ready");
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

TEST_F(NavigationTest, ModeChangesRequireNeutralButExplicitSuspensionRevokesAssignment) {
    arm();
    context.mode = GamepadNavigationMode::Drone;
    EXPECT_FALSE(tick().active);
    EXPECT_STREQ(session.status(), "neutral");
    for (int i = 0; i < 2; ++i)
        tick();
    EXPECT_STREQ(session.status(), "ready");
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
    settings.bindings.common[5] = {int(SDL_GAMEPAD_AXIS_COUNT) + SDL_GAMEPAD_BUTTON_LEFT_STICK};
    settings.bindings.common[6] = {int(SDL_GAMEPAD_AXIS_COUNT) + SDL_GAMEPAD_BUTTON_RIGHT_STICK};
    arm();
    sample.axes[SDL_GAMEPAD_AXIS_LEFT_TRIGGER] = 1;
    sample.axes[SDL_GAMEPAD_AXIS_RIGHT_TRIGGER] = 1;
    EXPECT_FLOAT_EQ(tick().vertical, 0);
    sample.axes[SDL_GAMEPAD_AXIS_LEFT_TRIGGER] = 0;
    EXPECT_FLOAT_EQ(tick().vertical, 1);
    sample.buttons[SDL_GAMEPAD_BUTTON_LEFT_STICK] = true;
    sample.buttons[SDL_GAMEPAD_BUTTON_RIGHT_STICK] = true;
    EXPECT_FLOAT_EQ(tick().speed, 0.35f);
}

TEST_F(NavigationTest, WalkDoesNotFlyOrSprintAndOnlyEastCanExitWalk) {
    context.mode = GamepadNavigationMode::Walk;
    settings.bindings.common[6] = {int(SDL_GAMEPAD_AXIS_COUNT) + SDL_GAMEPAD_BUTTON_RIGHT_STICK};
    arm();
    sample.axes[SDL_GAMEPAD_AXIS_RIGHT_TRIGGER] = 1;
    sample.buttons[SDL_GAMEPAD_BUTTON_RIGHT_STICK] = true;
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

TEST_F(NavigationTest, ProfileRoundTripPreservesEnablementButExcludesDeviceOwnership) {
    settings.bindings.common[8] = {SDL_GAMEPAD_AXIS_COUNT + SDL_GAMEPAD_BUTTON_NORTH};
    const auto serialized = settings.json();
    EXPECT_TRUE(serialized["enabled"].get<bool>());
    EXPECT_FALSE(serialized.contains("selected"));
    const auto restored = GamepadNavigationSettings::fromJson(serialized);
    EXPECT_TRUE(restored.enabled);
    EXPECT_EQ(restored.bindings.common[8], settings.bindings.common[8]);
    EXPECT_EQ(restored.json(), serialized);
}

TEST_F(NavigationTest, InvalidProfilesAndConflictsAreRejected) {
    auto profile = settings.json();
    profile["schema_version"] = 1.5;
    EXPECT_THROW(GamepadNavigationSettings::fromJson(profile), std::invalid_argument);
    profile = settings.json();
    profile["device"] = 42;
    EXPECT_THROW(GamepadNavigationSettings::fromJson(profile), std::invalid_argument);
    settings.bindings.common[8] = settings.bindings.common[10];
    EXPECT_THROW(settings.validate(), std::invalid_argument);
    settings = GamepadNavigationSettings{true};
    settings.bindings.modes[4][6] = GamepadBinding{SDL_GAMEPAD_AXIS_COUNT + SDL_GAMEPAD_BUTTON_NORTH};
    EXPECT_THROW(settings.validate(), std::invalid_argument);
    EXPECT_THROW(controlFromName("button.a"), std::invalid_argument);
}

TEST_F(NavigationTest, ModeOverrideCanExplicitlyClearOrInherit) {
    settings.bindings.modes[0][0] = GamepadBinding{};
    auto restored = GamepadNavigationSettings::fromJson(settings.json());
    EXPECT_EQ(restored.bindings.resolve(0, 0).positive, -1);
    EXPECT_EQ(restored.bindings.resolve(1, 0).positive, SDL_GAMEPAD_AXIS_LEFTX);
    restored.bindings.modes[0][0].reset();
    restored.bindings.modes[4][7].reset();
    const auto again = GamepadNavigationSettings::fromJson(restored.json());
    EXPECT_EQ(again.bindings.resolve(0, 0).positive, SDL_GAMEPAD_AXIS_LEFTX);
    EXPECT_EQ(again.bindings.resolve(4, 7).positive, -1);
}

TEST_F(NavigationTest, ButtonsCanDriveSignedMovementAndInversion) {
    settings.bindings.common[0] = {SDL_GAMEPAD_AXIS_COUNT + SDL_GAMEPAD_BUTTON_NORTH,
                                   SDL_GAMEPAD_AXIS_COUNT + SDL_GAMEPAD_BUTTON_WEST, true};
    settings.validate();
    arm();
    sample.buttons[SDL_GAMEPAD_BUTTON_NORTH] = true;
    EXPECT_FLOAT_EQ(tick().move.x, -1);
    sample.buttons[SDL_GAMEPAD_BUTTON_NORTH] = false;
    sample.buttons[SDL_GAMEPAD_BUTTON_WEST] = true;
    EXPECT_FLOAT_EQ(tick().move.x, 1);
}

TEST_F(NavigationTest, ConfigRevisionRequiresReleaseAndNewViewportClick) {
    arm();
    ++settings.revision;
    EXPECT_FALSE(tick().active);
    EXPECT_STREQ(session.status(), "viewport");
    arm();
    EXPECT_STREQ(session.status(), "ready");
}

TEST_F(NavigationTest, DiscreteTriggerActionFiresOncePerPress) {
    settings.bindings.common[4] = {};
    settings.bindings.common[8] = {SDL_GAMEPAD_AXIS_RIGHT_TRIGGER};
    settings.validate();
    arm();
    sample.axes[SDL_GAMEPAD_AXIS_RIGHT_TRIGGER] = 1;
    EXPECT_EQ(tick().navigation_step, 1);
    EXPECT_EQ(tick().navigation_step, 0);
    sample.axes[SDL_GAMEPAD_AXIS_RIGHT_TRIGGER] = 0;
    tick();
    sample.axes[SDL_GAMEPAD_AXIS_RIGHT_TRIGGER] = 1;
    EXPECT_EQ(tick().navigation_step, 1);
}

TEST_F(NavigationTest, SemanticCommandsAreIndependentOfVendorButtonLetters) {
    settings.bindings.common[10] = {SDL_GAMEPAD_AXIS_COUNT + SDL_GAMEPAD_BUTTON_NORTH};
    settings.validate();
    arm();
    sample.buttons[SDL_GAMEPAD_BUTTON_NORTH] = true;
    EXPECT_EQ(tick().rendering_step, 1);
    EXPECT_EQ(tick().rendering_step, 0);
    sample.buttons[SDL_GAMEPAD_BUTTON_NORTH] = false;
    tick();
    sample.pressed[SDL_GAMEPAD_BUTTON_NORTH] = true;
    EXPECT_EQ(tick().rendering_step, 1);
    sample.has_button[SDL_GAMEPAD_BUTTON_NORTH] = false;
    EXPECT_EQ(tick().rendering_step, 0);
}

TEST_F(NavigationTest, LegacyProfileDefaultsToDisabledAndEnablementRemainsAnExplicitPreference) {
    auto profile = settings.json();
    profile.erase("enabled");
    EXPECT_FALSE(GamepadNavigationSettings::fromJson(profile).enabled);
    profile["enabled"] = true;
    EXPECT_TRUE(GamepadNavigationSettings::fromJson(profile).enabled);
    profile["enabled"] = false;
    EXPECT_FALSE(GamepadNavigationSettings::fromJson(profile).enabled);
    profile["enabled"] = "true";
    EXPECT_THROW(GamepadNavigationSettings::fromJson(profile), nlohmann::json::exception);
}

TEST_F(NavigationTest, ShoulderFactoryPresetCyclesBothCommandsOncePerPress) {
    settings.validate();
    arm();
    sample.buttons[SDL_GAMEPAD_BUTTON_LEFT_SHOULDER] = true;
    auto intent = tick();
    EXPECT_EQ(intent.navigation_step, 1);
    EXPECT_FLOAT_EQ(intent.speed, settings.move_speed);
    EXPECT_EQ(tick().navigation_step, 0);
    sample.buttons[SDL_GAMEPAD_BUTTON_LEFT_SHOULDER] = false;
    sample.buttons[SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER] = true;
    EXPECT_EQ(tick().rendering_step, 1);
    EXPECT_EQ(tick().rendering_step, 0);
}

TEST_F(NavigationTest, DeliberateControllerModeChangeRearmsOnlyAfterNeutralWithoutNewMouseClick) {
    arm();
    sample.buttons[SDL_GAMEPAD_BUTTON_LEFT_SHOULDER] = true;
    EXPECT_EQ(tick().navigation_step, 1);
    session.pause();
    context.mode = GamepadNavigationMode::Trackball;
    for (int i = 0; i < 10; ++i)
        EXPECT_FALSE(tick().active);
    EXPECT_STREQ(session.status(), "neutral");
    sample.buttons.fill(false);
    for (int i = 0; i < 3; ++i)
        tick();
    EXPECT_STREQ(session.status(), "ready");
    sample.buttons[SDL_GAMEPAD_BUTTON_LEFT_SHOULDER] = true;
    EXPECT_EQ(tick().navigation_step, 1);
}

TEST(GamepadModeCycle, SkipsUnavailableWalkAndWrapsInBothDirections) {
    using namespace lfs::vis::input;
    std::vector<GamepadNavigationMode> tried;
    EXPECT_TRUE(cycleGamepadNavigation(GamepadNavigationMode::Drone, 1, false, [&](auto mode) {
        tried.push_back(mode);
        return mode != GamepadNavigationMode::Walk;
    }));
    EXPECT_EQ(tried, (std::vector{GamepadNavigationMode::Walk, GamepadNavigationMode::Orbit}));
    tried.clear();
    EXPECT_TRUE(cycleGamepadNavigation(GamepadNavigationMode::Orbit, -1, false, [&](auto mode) {
        tried.push_back(mode);
        return mode != GamepadNavigationMode::Walk;
    }));
    EXPECT_EQ(tried, (std::vector{GamepadNavigationMode::Walk, GamepadNavigationMode::Drone}));
}

TEST(GamepadModeCycle, OrthographicCyclesOnlySupportedModesAndFailureIsBounded) {
    using namespace lfs::vis::input;
    std::vector<GamepadNavigationMode> tried;
    EXPECT_TRUE(cycleGamepadNavigation(GamepadNavigationMode::Trackball, 1, true, [&](auto mode) {
        tried.push_back(mode);
        return true;
    }));
    EXPECT_EQ(tried, (std::vector{GamepadNavigationMode::Orbit}));
    int attempts = 0;
    EXPECT_FALSE(cycleGamepadNavigation(GamepadNavigationMode::Orbit, 1, false, [&](auto) { ++attempts; return false; }));
    EXPECT_EQ(attempts, 4);
}

TEST_F(NavigationTest, ModalPauseCannotQueueMovementOrShoulderCommands) {
    arm();
    context.allowed = false;
    sample.axes[SDL_GAMEPAD_AXIS_LEFTX] = 1;
    sample.buttons[SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER] = true;
    for (int i = 0; i < 5; ++i)
        EXPECT_FALSE(tick().active);
    context.allowed = true;
    for (int i = 0; i < 5; ++i)
        EXPECT_FALSE(tick().active);
    EXPECT_STREQ(session.status(), "neutral");
    sample.axes.fill(0);
    sample.buttons.fill(false);
    sample.pressed[SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER] = true;
    EXPECT_FALSE(tick().active);
    sample.pressed.fill(false);
    for (int i = 0; i < 3; ++i)
        EXPECT_EQ(tick().rendering_step, 0);
    EXPECT_STREQ(session.status(), "ready");
    sample.buttons[SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER] = true;
    EXPECT_EQ(tick().rendering_step, 1);
}

TEST_F(NavigationTest, PauseDoesNotGrantAnUnassignedViewport) {
    tick();
    session.pause();
    for (int i = 0; i < 10; ++i)
        EXPECT_FALSE(tick().active);
    EXPECT_STREQ(session.status(), "viewport");
}

TEST_F(NavigationTest, ReconnectDuringPauseRevokesPreviousGrant) {
    arm();
    context.allowed = false;
    tick();
    ++sample.generation;
    tick();
    context.allowed = true;
    for (int i = 0; i < 10; ++i)
        EXPECT_FALSE(tick().active);
    EXPECT_STREQ(session.status(), "viewport");
    arm();
}

TEST_F(NavigationTest, BlockingDialogBetweenFramesRequiresNeutralAgain) {
    arm();
    session.pause(); // native picker opened and closed while the UI loop was blocked
    sample.axes[SDL_GAMEPAD_AXIS_RIGHTX] = 0.9f;
    for (int i = 0; i < 10; ++i)
        EXPECT_FALSE(tick().active);
    sample.axes.fill(0);
    for (int i = 0; i < 3; ++i)
        tick();
    EXPECT_STREQ(session.status(), "ready");
}

TEST(GamepadProfiles, MigrationKeepsTuningAndModeOverridesButExcludesEnablement) {
    using namespace lfs::vis::input;
    GamepadNavigationSettings settings{true};
    settings.move_speed = 2.7f;
    settings.bindings.modes[2][0] = GamepadBinding{};
    const auto profiles = GamepadProfiles::migrate(settings);
    EXPECT_EQ(profiles.value["active"], "personal");
    EXPECT_FALSE(profiles.value["profiles"][0]["settings"].contains("enabled"));
    EXPECT_TRUE(profiles.value["device_profiles"].empty());
    EXPECT_FALSE(profiles.activeSettings(false).enabled);
    EXPECT_FLOAT_EQ(profiles.activeSettings(true).move_speed, 2.7f);
    EXPECT_TRUE(profiles.activeSettings(true).bindings.modes[2][0].has_value());
}

TEST(GamepadProfiles, LibraryRejectsDuplicateIdentitiesNamesAndDanglingAssociations) {
    using namespace lfs::vis::input;
    const auto original = GamepadProfiles::migrate({}).value;
    auto value = original;
    value["profiles"].push_back(value["profiles"][0]);
    EXPECT_THROW(GamepadProfiles::fromJson(value), std::invalid_argument);
    value["profiles"][1]["id"] = "second";
    EXPECT_THROW(GamepadProfiles::fromJson(value), std::invalid_argument);
    value["profiles"][1]["name"] = "Second";
    EXPECT_NO_THROW(GamepadProfiles::fromJson(value));
    value["device_profiles"][std::string(32, 'a')] = "missing";
    EXPECT_THROW(GamepadProfiles::fromJson(value), std::invalid_argument);
    value["device_profiles"] = {{"123", "personal"}};
    EXPECT_THROW(GamepadProfiles::fromJson(value), std::invalid_argument);
    value = original;
    value["profiles"][0]["settings"]["enabled"] = true;
    EXPECT_THROW(GamepadProfiles::fromJson(value), std::invalid_argument);
    value = original;
    value["schema_version"] = 2;
    EXPECT_THROW(GamepadProfiles::fromJson(value), std::invalid_argument);
}

TEST(GamepadProfiles, EditsAndResetAreConfinedToActiveProfile) {
    using namespace lfs::vis::input;
    auto profiles = GamepadProfiles::migrate({});
    auto other = profiles.value["profiles"][0];
    other["id"] = "second";
    other["name"] = "Second";
    other["settings"]["move_speed"] = 3.0;
    profiles.value["profiles"].push_back(other);
    auto edited = profiles.activeSettings(true);
    edited.look_speed = 2.1f;
    profiles.updateActive(edited);
    EXPECT_EQ(profiles.value["profiles"][1], other);
    profiles.value["active"] = "second";
    EXPECT_FLOAT_EQ(profiles.activeSettings(false).move_speed, 3.0f);
    profiles.updateActive({});
    EXPECT_FLOAT_EQ(profiles.value["profiles"][0]["settings"]["look_speed"].get<float>(), 2.1f);
    EXPECT_EQ(GamepadProfiles::fromJson(profiles.value).value, profiles.value);
}

TEST_F(NavigationTest, SequencerDefaultsAreContextualAndFireOnlyOncePerPress) {
    arm();
    sample.buttons[SDL_GAMEPAD_BUTTON_WEST] = true;
    EXPECT_FALSE(tick().add_keyframe);
    sample.buttons.fill(false);
    context.sequencer_open = true;
    for (int i = 0; i < 3; ++i)
        tick();
    sample.buttons[SDL_GAMEPAD_BUTTON_WEST] = true;
    EXPECT_TRUE(tick().add_keyframe);
    EXPECT_FALSE(tick().add_keyframe);
    sample.buttons.fill(false);
    tick();
    sample.buttons[SDL_GAMEPAD_BUTTON_NORTH] = true;
    EXPECT_TRUE(tick().remove_selected_keyframe);
    sample.buttons.fill(false);
    tick();
    sample.buttons[SDL_GAMEPAD_BUTTON_DPAD_LEFT] = true;
    EXPECT_EQ(tick().keyframe_step, -1);
    EXPECT_EQ(tick().keyframe_step, 0);
    sample.buttons.fill(false);
    tick();
    sample.buttons[SDL_GAMEPAD_BUTTON_DPAD_RIGHT] = true;
    EXPECT_EQ(tick().keyframe_step, 1);
}

TEST_F(NavigationTest, OpeningAndClosingSequencerRequireNeutralAndNeverQueueHeldButtons) {
    arm();
    sample.buttons[SDL_GAMEPAD_BUTTON_NORTH] = true;
    context.sequencer_open = true;
    for (int i = 0; i < 10; ++i)
        EXPECT_FALSE(tick().active);
    sample.buttons.fill(false);
    for (int i = 0; i < 3; ++i)
        tick();
    EXPECT_STREQ(session.status(), "ready");
    context.allowed = false;
    sample.pressed[SDL_GAMEPAD_BUTTON_WEST] = true;
    EXPECT_FALSE(tick().active);
    context.allowed = true;
    EXPECT_FALSE(tick().active);
    sample.pressed.fill(false);
    for (int i = 0; i < 3; ++i)
        EXPECT_FALSE(tick().add_keyframe);
    context.sequencer_open = false;
    EXPECT_FALSE(tick().active);
    EXPECT_STREQ(session.status(), "neutral");
}

TEST_F(NavigationTest, SequencerRemappingOverridesSharedCameraControlsOnlyWhileVisible) {
    settings.bindings.common[0] = {SDL_GAMEPAD_AXIS_COUNT + SDL_GAMEPAD_BUTTON_NORTH};
    settings.validate();
    arm();
    sample.buttons[SDL_GAMEPAD_BUTTON_NORTH] = true;
    EXPECT_EQ(tick().move.x, 1);
    sample.buttons.fill(false);
    context.sequencer_open = true;
    for (int i = 0; i < 3; ++i)
        tick();
    sample.buttons[SDL_GAMEPAD_BUTTON_NORTH] = true;
    auto intent = tick();
    EXPECT_TRUE(intent.remove_selected_keyframe);
    EXPECT_EQ(intent.move.x, 0);
    sample.buttons.fill(false);
    tick();
    settings.bindings.modes[gamepad_sequencer_scope][13] = GamepadBinding{};
    intent = tick();
    sample.buttons[SDL_GAMEPAD_BUTTON_NORTH] = true;
    EXPECT_FALSE(tick().remove_selected_keyframe);
    EXPECT_EQ(tick().move.x, 1);
}

TEST_F(NavigationTest, SimultaneousSequencerEditingCommandsAreIgnored) {
    context.sequencer_open = true;
    arm();
    sample.buttons[SDL_GAMEPAD_BUTTON_WEST] = true;
    sample.buttons[SDL_GAMEPAD_BUTTON_NORTH] = true;
    const auto intent = tick();
    EXPECT_FALSE(intent.add_keyframe);
    EXPECT_FALSE(intent.remove_selected_keyframe);
    EXPECT_EQ(intent.keyframe_step, 0);
}

TEST_F(NavigationTest, DeselectIsOptionalRemappableAndReservedOnlyInsideSequencer) {
    constexpr auto action = actionIndex(GamepadAction::DeselectKeyframe);
    EXPECT_EQ(settings.bindings.resolve(gamepad_sequencer_scope, action).positive, -1);
    settings.bindings.modes[gamepad_sequencer_scope][action] = GamepadBinding{SDL_GAMEPAD_AXIS_COUNT + SDL_GAMEPAD_BUTTON_DPAD_DOWN};
    context.sequencer_open = true;
    arm();
    sample.pressed[SDL_GAMEPAD_BUTTON_DPAD_DOWN] = true;
    EXPECT_TRUE(tick().deselect_keyframe);
    sample.pressed.fill(false);
    context.sequencer_open = false;
    for (int i = 0; i < 3; ++i)
        tick();
    sample.pressed[SDL_GAMEPAD_BUTTON_DPAD_DOWN] = true;
    EXPECT_FALSE(tick().deselect_keyframe);
    const auto restored = GamepadNavigationSettings::fromJson(settings.json());
    EXPECT_EQ(restored.bindings.resolve(gamepad_sequencer_scope, action).positive,
              SDL_GAMEPAD_AXIS_COUNT + SDL_GAMEPAD_BUTTON_DPAD_DOWN);
}

TEST_F(NavigationTest, TimelineGestureBlocksEditingButKeepsCameraMotionAndDoesNotReplayHeldButtons) {
    context.sequencer_open = true;
    arm();
    context.sequencer_commands_allowed = false;
    sample.axes[SDL_GAMEPAD_AXIS_LEFTX] = .8f;
    sample.buttons[SDL_GAMEPAD_BUTTON_WEST] = true;
    sample.pressed[SDL_GAMEPAD_BUTTON_WEST] = true;
    auto intent = tick();
    EXPECT_FALSE(intent.add_keyframe);
    EXPECT_GT(intent.move.x, 0.f);
    EXPECT_TRUE(intent.active);
    sample.pressed.fill(false);
    context.sequencer_commands_allowed = true;
    EXPECT_FALSE(tick().add_keyframe); // held through the drag, no delayed capture
    sample.buttons.fill(false);
    tick();
    sample.pressed[SDL_GAMEPAD_BUTTON_WEST] = true;
    EXPECT_TRUE(tick().add_keyframe);
}

TEST_F(NavigationTest, LegacyProfilesGainSequencerDefaultsAndKeepExplicitOverrides) {
    auto legacy = settings.json();
    for (auto name : {"add_keyframe", "remove_selected_keyframe", "previous_keyframe", "next_keyframe", "deselect_keyframe", "update_selected_keyframe"})
        legacy["bindings"]["common"].erase(name);
    auto restored = GamepadNavigationSettings::fromJson(legacy);
    EXPECT_EQ(restored.bindings.resolve(gamepad_sequencer_scope, actionIndex(GamepadAction::DeselectKeyframe)).positive, -1);
    EXPECT_EQ(restored.bindings.resolve(gamepad_sequencer_scope, actionIndex(GamepadAction::UpdateSelectedKeyframe)).positive, -1);
    EXPECT_EQ(restored.bindings.resolve(gamepad_sequencer_scope, 12).positive,
              SDL_GAMEPAD_AXIS_COUNT + SDL_GAMEPAD_BUTTON_WEST);
    restored.bindings.modes[gamepad_sequencer_scope][12] = GamepadBinding{};
    const auto roundtrip = GamepadNavigationSettings::fromJson(restored.json());
    EXPECT_EQ(roundtrip.bindings.resolve(gamepad_sequencer_scope, 12).positive, -1);
    EXPECT_EQ(gamepad_modes.size(), 5); // Sequencer is an overlay, never a camera mode.
}
