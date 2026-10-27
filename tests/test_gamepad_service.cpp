/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "input/gamepad_service.hpp"
#include <SDL3/SDL.h>
#include <algorithm>
#include <gtest/gtest.h>

namespace {
    class GamepadServiceTest : public testing::Test {
    protected:
        lfs::vis::input::GamepadService service;
        std::vector<SDL_JoystickID> attached;
        std::vector<SDL_Joystick*> handles;

        void SetUp() override { ASSERT_TRUE(service.initialize()) << SDL_GetError(); }
        void TearDown() override {
            for (auto* handle : handles)
                SDL_CloseJoystick(handle);
            for (auto id : attached)
                if (SDL_IsJoystickVirtual(id))
                    SDL_DetachVirtualJoystick(id);
            pump();
            service.shutdown();
        }
        SDL_JoystickID attach(bool mapped = true) {
            SDL_VirtualJoystickDesc desc;
            SDL_INIT_INTERFACE(&desc);
            desc.type = mapped ? SDL_JOYSTICK_TYPE_GAMEPAD : SDL_JOYSTICK_TYPE_FLIGHT_STICK;
            desc.naxes = SDL_GAMEPAD_AXIS_COUNT;
            desc.nbuttons = SDL_GAMEPAD_BUTTON_COUNT;
            desc.axis_mask = (1u << SDL_GAMEPAD_AXIS_COUNT) - 1;
            desc.button_mask = (1u << SDL_GAMEPAD_BUTTON_COUNT) - 1;
            desc.name = "LichtFeld test controller";
            const auto id = SDL_AttachVirtualJoystick(&desc);
            if (id)
                attached.push_back(id);
            return id;
        }
        void pump() {
            SDL_Event event;
            while (SDL_PollEvent(&event))
                service.handleEvent(event);
            service.sample();
        }
    };
} // namespace

TEST_F(GamepadServiceTest, TwoIdenticalPadsNeverAutomaticallyAcquireSelection) {
    const auto first = attach();
    const auto second = attach();
    ASSERT_NE(first, 0u);
    ASSERT_NE(second, 0u);
    pump();
    const auto state = service.snapshot();
    EXPECT_EQ(state.selected, 0u);
    EXPECT_EQ(std::count_if(state.devices.begin(), state.devices.end(), [&](const auto& d) {
                  return d.id == first || d.id == second;
              }),
              2);
    ASSERT_TRUE(service.select(first));
    ASSERT_TRUE(SDL_DetachVirtualJoystick(second));
    pump();
    EXPECT_EQ(service.snapshot().selected, first);
    const auto replacement = attach();
    pump();
    EXPECT_EQ(service.snapshot().selected, first);
    ASSERT_TRUE(SDL_DetachVirtualJoystick(first));
    pump();
    EXPECT_EQ(service.snapshot().selected, 0u);
    EXPECT_EQ(service.snapshot().sample.id, 0u);
    EXPECT_FALSE(service.select(first));
    EXPECT_TRUE(service.select(replacement));
}

TEST_F(GamepadServiceTest, PressesSurviveRepeatedPumpsUntilTheirOwnConsumerAcknowledges) {
    const auto id = attach();
    ASSERT_NE(id, 0u);
    pump();
    ASSERT_TRUE(service.selectNavigation(id));
    ASSERT_TRUE(service.select(id));
    SDL_Event event{};
    event.type = SDL_EVENT_GAMEPAD_BUTTON_DOWN;
    event.gbutton.which = id;
    event.gbutton.button = SDL_GAMEPAD_BUTTON_WEST;
    service.handleEvent(event);
    pump();
    pump();
    ASSERT_TRUE(service.navigationSample().pressed[SDL_GAMEPAD_BUTTON_WEST]);
    ASSERT_TRUE(service.takeDiagnosticSnapshot().sample.pressed[SDL_GAMEPAD_BUTTON_WEST]);
    EXPECT_FALSE(service.takeDiagnosticSnapshot().sample.pressed[SDL_GAMEPAD_BUTTON_WEST]);
    EXPECT_TRUE(service.navigationSample().pressed[SDL_GAMEPAD_BUTTON_WEST]);
    service.consumeNavigationPresses();
    EXPECT_FALSE(service.navigationSample().pressed[SDL_GAMEPAD_BUTTON_WEST]);
}

TEST_F(GamepadServiceTest, StartupEnumerationAndDuplicateAddedEventsKeepTheSelection) {
    const auto id = attach();
    ASSERT_NE(id, 0u);
    pump();
    lfs::vis::input::GamepadService keepalive;
    ASSERT_TRUE(keepalive.initialize());
    service.shutdown();
    ASSERT_TRUE(service.initialize());
    ASSERT_TRUE(service.select(id));
    const auto generation = service.snapshot().sample.generation;
    SDL_Event event{};
    event.type = SDL_EVENT_GAMEPAD_ADDED;
    event.gdevice.which = id;
    service.handleEvent(event);
    pump();
    EXPECT_EQ(service.snapshot().selected, id);
    EXPECT_EQ(service.snapshot().sample.generation, generation);
}

TEST_F(GamepadServiceTest, SamplesNormalizeSticksAndTriggersAndKeepShortPresses) {
    const auto id = attach();
    ASSERT_NE(id, 0u);
    auto* joystick = SDL_OpenJoystick(id);
    ASSERT_NE(joystick, nullptr);
    handles.push_back(joystick);
    pump();
    ASSERT_TRUE(service.select(id));
    ASSERT_TRUE(SDL_SetJoystickVirtualAxis(joystick, SDL_GAMEPAD_AXIS_LEFTX, -32768));
    ASSERT_TRUE(SDL_SetJoystickVirtualAxis(joystick, SDL_GAMEPAD_AXIS_RIGHTX, 32767));
    ASSERT_TRUE(SDL_SetJoystickVirtualAxis(joystick, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, 32767));
    SDL_UpdateJoysticks();
    pump();
    const auto sample = service.snapshot().sample;
    EXPECT_FLOAT_EQ(sample.axes[SDL_GAMEPAD_AXIS_LEFTX], -1);
    EXPECT_FLOAT_EQ(sample.axes[SDL_GAMEPAD_AXIS_RIGHTX], 1);
    EXPECT_FLOAT_EQ(sample.axes[SDL_GAMEPAD_AXIS_LEFT_TRIGGER], 1);
    service.beginFrame();
    SDL_Event press{};
    press.type = SDL_EVENT_GAMEPAD_BUTTON_DOWN;
    press.gbutton.which = id;
    press.gbutton.button = SDL_GAMEPAD_BUTTON_SOUTH;
    service.handleEvent(press);
    press.type = SDL_EVENT_GAMEPAD_BUTTON_UP;
    service.handleEvent(press);
    EXPECT_TRUE(service.sample());
    EXPECT_TRUE(service.snapshot().sample.pressed[SDL_GAMEPAD_BUTTON_SOUTH]);
    EXPECT_FALSE(service.snapshot().sample.buttons[SDL_GAMEPAD_BUTTON_SOUTH]);
    service.beginFrame();
    EXPECT_FALSE(service.snapshot().sample.pressed[SDL_GAMEPAD_BUTTON_SOUTH]);
    EXPECT_FALSE(service.sample());
}

TEST_F(GamepadServiceTest, RemappingInvalidatesSelectionAndOldSnapshotsStayUsable) {
    const auto id = attach();
    ASSERT_NE(id, 0u);
    pump();
    ASSERT_TRUE(service.select(id));
    const auto old = service.snapshot();
    SDL_Event event{};
    event.type = SDL_EVENT_GAMEPAD_REMAPPED;
    event.gdevice.which = id;
    service.handleEvent(event);
    EXPECT_EQ(service.snapshot().selected, 0u);
    EXPECT_EQ(old.selected, id);
    ASSERT_TRUE(service.select(id));
    EXPECT_GT(service.snapshot().sample.generation, old.sample.generation);
    EXPECT_TRUE(service.select(0));
    EXPECT_EQ(service.snapshot().sample.id, 0u);
}

TEST_F(GamepadServiceTest, UnmappedJoysticksAreListedButCannotBeSelectedForGamepadSampling) {
    const auto id = attach(false);
    ASSERT_NE(id, 0u);
    pump();
    const auto state = service.snapshot();
    const auto found = std::find_if(state.devices.begin(), state.devices.end(),
                                    [id](const auto& d) { return d.id == id; });
    ASSERT_NE(found, state.devices.end());
    EXPECT_FALSE(found->mapped);
    EXPECT_FALSE(service.select(id));
}

TEST_F(GamepadServiceTest, RemovalDiscardsQueuedPressesAndRepeatedShutdownIsSafe) {
    const auto id = attach();
    ASSERT_NE(id, 0u);
    pump();
    ASSERT_TRUE(service.select(id));
    SDL_Event event{};
    event.type = SDL_EVENT_GAMEPAD_REMOVED;
    event.gdevice.which = id;
    service.handleEvent(event);
    event.type = SDL_EVENT_GAMEPAD_BUTTON_DOWN;
    event.gbutton.which = id;
    event.gbutton.button = SDL_GAMEPAD_BUTTON_SOUTH;
    service.handleEvent(event);
    EXPECT_FALSE(service.snapshot().sample.pressed[SDL_GAMEPAD_BUTTON_SOUTH]);
    service.shutdown();
    service.shutdown();
    EXPECT_FALSE(service.snapshot().available);
    EXPECT_TRUE(service.snapshot().devices.empty());
}

TEST_F(GamepadServiceTest, DiagnosticSelectionDoesNotStealNavigationAndBothInvalidateOnRemoval) {
    const auto navigation = attach();
    const auto diagnostic = attach();
    ASSERT_NE(navigation, 0u);
    ASSERT_NE(diagnostic, 0u);
    pump();
    ASSERT_TRUE(service.selectNavigation(navigation));
    ASSERT_TRUE(service.select(diagnostic));
    EXPECT_EQ(service.navigationSample().id, navigation);
    EXPECT_EQ(service.snapshot().selected, diagnostic);
    ASSERT_TRUE(service.select(0));
    EXPECT_EQ(service.navigationSample().id, navigation);
    ASSERT_TRUE(SDL_DetachVirtualJoystick(navigation));
    pump();
    EXPECT_EQ(service.navigationSample().id, 0u);
    EXPECT_EQ(service.snapshot().navigation_selected, 0u);
}

TEST_F(GamepadServiceTest, FirstUsableConnectionIsPreselectedWithoutSelectingDiagnostics) {
    // Physical pads can coexist with virtual fixtures on developer machines.
    const auto initial = service.snapshot();
    const bool existing = std::any_of(initial.devices.begin(), initial.devices.end(),
                                      [](const auto& d) { return d.mapped && d.error.empty(); });
    const auto first = attach();
    pump();
    const auto expected = existing ? initial.navigation_selected : first;
    EXPECT_EQ(service.navigationSample().id, expected);
    EXPECT_EQ(service.snapshot().selected, 0u);
    const auto second = attach();
    pump();
    EXPECT_EQ(service.navigationSample().id, expected);
    ASSERT_TRUE(service.selectNavigation(second));
    const auto third = attach();
    pump();
    EXPECT_EQ(service.navigationSample().id, second);
    ASSERT_TRUE(SDL_DetachVirtualJoystick(second));
    pump();
    EXPECT_EQ(service.navigationSample().id, 0u);
    ASSERT_TRUE(service.selectNavigation(first));
    EXPECT_EQ(service.navigationSample().id, first);
}

TEST_F(GamepadServiceTest, StartupSelectsOnlyUnambiguousMappedController) {
    const auto initial = service.snapshot();
    const auto count = std::count_if(initial.devices.begin(), initial.devices.end(),
                                     [](const auto& d) { return d.mapped && d.error.empty(); });
    lfs::vis::input::GamepadService startup;
    ASSERT_TRUE(startup.initialize());
    EXPECT_EQ(startup.navigationSample().id, initial.navigation_selected);
    startup.shutdown();
    const auto first = attach();
    pump();
    ASSERT_TRUE(startup.initialize());
    EXPECT_EQ(startup.navigationSample().id, count == 0 ? first : 0u);
    startup.shutdown();
    const auto second = attach();
    pump();
    ASSERT_NE(second, 0u);
    ASSERT_TRUE(startup.initialize());
    EXPECT_EQ(startup.navigationSample().id, 0u);
    EXPECT_EQ(startup.snapshot().selected, 0u);
}

TEST_F(GamepadServiceTest, ReconnectIsPreselectedButExplicitDeselectIsStableAcrossFrames) {
    const auto initial = service.snapshot();
    const bool existing = std::any_of(initial.devices.begin(), initial.devices.end(),
                                      [](const auto& d) { return d.mapped && d.error.empty(); });
    const auto first = attach();
    pump();
    EXPECT_EQ(service.navigationSample().id, existing ? initial.navigation_selected : first);
    ASSERT_TRUE(service.selectNavigation(0));
    pump();
    EXPECT_EQ(service.navigationSample().id, 0u);
    ASSERT_TRUE(SDL_DetachVirtualJoystick(first));
    pump();
    const auto replacement = attach();
    pump();
    EXPECT_NE(first, replacement);
    EXPECT_EQ(service.navigationSample().id, existing ? 0u : replacement);
}

TEST_F(GamepadServiceTest, ModelIdentitySurvivesReconnectAndIsSharedByIdenticalControllers) {
    const auto first = attach();
    pump();
    ASSERT_TRUE(service.selectNavigation(first));
    const auto key = service.navigationProfileKey();
    ASSERT_EQ(key.size(), 32);
    EXPECT_NE(key, std::string(32, '0'));
    const auto second = attach();
    pump();
    ASSERT_TRUE(service.selectNavigation(second));
    EXPECT_EQ(service.navigationProfileKey(), key);
    ASSERT_TRUE(SDL_DetachVirtualJoystick(first));
    ASSERT_TRUE(SDL_DetachVirtualJoystick(second));
    pump();
    EXPECT_TRUE(service.navigationProfileKey().empty());
    const auto replacement = attach();
    pump();
    ASSERT_TRUE(service.selectNavigation(replacement));
    EXPECT_NE(replacement, first);
    EXPECT_EQ(service.navigationProfileKey(), key);
}
