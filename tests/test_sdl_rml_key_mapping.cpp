/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/sdl_rml_key_mapping.hpp"
#include "input/frame_input_buffer.hpp"
#include "input/key_codes.hpp"
#include "input/sdl_key_mapping.hpp"
#include <array>
#include <gtest/gtest.h>

namespace {

    TEST(SdlRmlKeyMappingTest, LetterScancodeMapsToAlphabetKey) {
        EXPECT_EQ(lfs::vis::gui::sdlScancodeToRml(SDL_SCANCODE_A), Rml::Input::KI_A);
    }

    TEST(SdlRmlKeyMappingTest, TextEditingShortcutsFollowLayoutKeycodes) {
        using namespace lfs::vis::gui;
        for (const SDL_Keycode keycode : {SDLK_A, SDLK_C, SDLK_X, SDLK_V, SDLK_Z, SDLK_Y}) {
            EXPECT_EQ(sdlTextShortcutToRml(SDL_SCANCODE_Q, keycode, SDL_KMOD_CTRL),
                      sdlKeycodeToRml(keycode));
            EXPECT_EQ(sdlTextShortcutToRml(SDL_SCANCODE_Q, keycode, SDL_KMOD_GUI),
                      sdlKeycodeToRml(keycode));
        }
        EXPECT_EQ(sdlTextShortcutToRml(SDL_SCANCODE_A, SDLK_Q, SDL_KMOD_CTRL), Rml::Input::KI_Q);
        EXPECT_EQ(sdlTextShortcutToRml(SDL_SCANCODE_Y, SDLK_Z, SDL_KMOD_CTRL), Rml::Input::KI_Z);
        EXPECT_EQ(sdlTextShortcutToRml(SDL_SCANCODE_Z, SDLK_Y, SDL_KMOD_CTRL), Rml::Input::KI_Y);
        EXPECT_EQ(sdlTextShortcutToRml(SDL_SCANCODE_Q, SDLK_A, SDL_KMOD_NONE), Rml::Input::KI_Q);
        EXPECT_EQ(sdlTextShortcutToRml(SDL_SCANCODE_Q, SDLK_A, static_cast<SDL_Keymod>(SDL_KMOD_CTRL | SDL_KMOD_ALT)), Rml::Input::KI_Q);
    }

    TEST(SdlRmlKeyMappingTest, EqualsKeyMapsToOemPlus) {
        EXPECT_EQ(lfs::vis::gui::sdlKeycodeToRml(SDLK_EQUALS), Rml::Input::KI_OEM_PLUS);
        EXPECT_EQ(lfs::vis::gui::sdlScancodeToRml(SDL_SCANCODE_EQUALS), Rml::Input::KI_OEM_PLUS);
    }

    TEST(SdlRmlKeyMappingTest, ScancodeAndKeycodeMappingsStayInSync) {
        for (int value = 0; value < SDL_SCANCODE_COUNT; ++value) {
            const auto scancode = static_cast<SDL_Scancode>(value);
            const auto keycode = SDL_GetKeyFromScancode(scancode, SDL_KMOD_NONE, false);
            const auto keycode_mapping = lfs::vis::gui::sdlKeycodeToRml(keycode);
            if (keycode_mapping == Rml::Input::KI_UNKNOWN)
                continue;

            EXPECT_EQ(lfs::vis::gui::sdlScancodeToRml(scancode), keycode_mapping)
                << "scancode=" << value << " keycode=" << keycode;
        }
    }

    TEST(SdlKeyMappingTest, NumpadScancodesMapToAppKeys) {
        EXPECT_EQ(lfs::vis::input::sdlScancodeToAppKey(SDL_SCANCODE_KP_0), lfs::vis::input::KEY_KP_0);
        EXPECT_EQ(lfs::vis::input::sdlScancodeToAppKey(SDL_SCANCODE_KP_1), lfs::vis::input::KEY_KP_1);
        EXPECT_EQ(lfs::vis::input::sdlScancodeToAppKey(SDL_SCANCODE_KP_9), lfs::vis::input::KEY_KP_9);
        EXPECT_EQ(lfs::vis::input::sdlScancodeToAppKey(SDL_SCANCODE_KP_PLUS), lfs::vis::input::KEY_KP_ADD);
        EXPECT_EQ(lfs::vis::input::sdlScancodeToAppKey(SDL_SCANCODE_KP_MINUS), lfs::vis::input::KEY_KP_SUBTRACT);
        EXPECT_EQ(lfs::vis::input::sdlScancodeToAppKey(SDL_SCANCODE_KP_ENTER), lfs::vis::input::KEY_KP_ENTER);
    }

    TEST(SdlKeyMappingTest, AppNumpadKeysMapBackToScancodes) {
        EXPECT_EQ(lfs::vis::input::appKeyToSdlScancode(lfs::vis::input::KEY_KP_0), SDL_SCANCODE_KP_0);
        EXPECT_EQ(lfs::vis::input::appKeyToSdlScancode(lfs::vis::input::KEY_KP_1), SDL_SCANCODE_KP_1);
        EXPECT_EQ(lfs::vis::input::appKeyToSdlScancode(lfs::vis::input::KEY_KP_9), SDL_SCANCODE_KP_9);
        EXPECT_EQ(lfs::vis::input::appKeyToSdlScancode(lfs::vis::input::KEY_KP_ADD), SDL_SCANCODE_KP_PLUS);
        EXPECT_EQ(lfs::vis::input::appKeyToSdlScancode(lfs::vis::input::KEY_KP_SUBTRACT), SDL_SCANCODE_KP_MINUS);
        EXPECT_EQ(lfs::vis::input::appKeyToSdlScancode(lfs::vis::input::KEY_KP_ENTER), SDL_SCANCODE_KP_ENTER);
    }

    TEST(SdlKeyMappingTest, LayoutAwareKeycodesMapToAppKeys) {
        EXPECT_EQ(lfs::vis::input::sdlKeycodeToAppKey(SDLK_A), lfs::vis::input::KEY_A);
        EXPECT_EQ(lfs::vis::input::sdlKeycodeToAppKey(SDLK_Z), lfs::vis::input::KEY_Z);
        EXPECT_EQ(lfs::vis::input::sdlKeycodeToAppKey(SDLK_RETURN), lfs::vis::input::KEY_ENTER);
        EXPECT_EQ(lfs::vis::input::sdlKeycodeToAppKey(SDLK_KP_PLUS), lfs::vis::input::KEY_KP_ADD);
    }

    TEST(SdlKeyMappingTest, ShortcutTranslationKeepsNonNumberRowKeysLayoutAware) {
        using namespace lfs::vis::input;
        EXPECT_EQ(sdlShortcutKeyToAppKey(SDL_SCANCODE_APOSTROPHE, SDLK_APOSTROPHE), KEY_APOSTROPHE);
        EXPECT_EQ(sdlShortcutKeyToAppKey(SDL_SCANCODE_MINUS, SDLK_MINUS), KEY_MINUS);
        EXPECT_EQ(sdlShortcutKeyToAppKey(SDL_SCANCODE_KP_4, SDLK_KP_4), KEY_KP_4);
        EXPECT_EQ(sdlShortcutKeyToAppKey(SDL_SCANCODE_A, SDLK_Q), KEY_Q);
    }

    TEST(SdlKeyMappingTest, EntireTopNumberRowFollowsPhysicalDigitsAcrossLayouts) {
        using namespace lfs::vis::input;
        struct Layout {
            const char* name;
            std::array<SDL_Keycode, 10> keycodes;
        };
        const std::array layouts = {
            Layout{"us", {SDLK_1, SDLK_2, SDLK_3, SDLK_4, SDLK_5, SDLK_6, SDLK_7, SDLK_8, SDLK_9, SDLK_0}},
            Layout{"be", {SDLK_AMPERSAND, static_cast<SDL_Keycode>(0xe9), SDLK_DBLAPOSTROPHE, SDLK_APOSTROPHE, SDLK_LEFTPAREN, static_cast<SDL_Keycode>(0xa7), static_cast<SDL_Keycode>(0xe8), SDLK_EXCLAIM, static_cast<SDL_Keycode>(0xe7), static_cast<SDL_Keycode>(0xe0)}},
            Layout{"fr", {SDLK_AMPERSAND, static_cast<SDL_Keycode>(0xe9), SDLK_DBLAPOSTROPHE, SDLK_APOSTROPHE, SDLK_LEFTPAREN, SDLK_MINUS, static_cast<SDL_Keycode>(0xe8), SDLK_UNDERSCORE, static_cast<SDL_Keycode>(0xe7), static_cast<SDL_Keycode>(0xe0)}},
        };
        for (const auto& layout : layouts) {
            for (size_t i = 0; i < layout.keycodes.size(); ++i) {
                SCOPED_TRACE(std::string(layout.name) + " key " + std::to_string((i + 1) % 10));
                const auto scancode = static_cast<SDL_Scancode>(SDL_SCANCODE_1 + i);
                EXPECT_EQ(sdlShortcutKeyToAppKey(scancode, layout.keycodes[i]),
                          KEY_0 + static_cast<int>((i + 1) % 10));
            }
        }
    }

    TEST(FrameInputBufferTest, NumberRowShortcutMappingLeavesTextInputUntouched) {
        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();
        SDL_Event key_event{};
        key_event.type = SDL_EVENT_KEY_DOWN;
        key_event.key.windowID = 11;
        key_event.key.scancode = SDL_SCANCODE_4;
        key_event.key.key = SDLK_APOSTROPHE;
        buffer.processEvent(key_event, 11);
        SDL_Event text_event{};
        text_event.type = SDL_EVENT_TEXT_INPUT;
        text_event.text.windowID = 11;
        text_event.text.text = "'";
        buffer.processEvent(text_event, 11);

        EXPECT_EQ(lfs::vis::input::sdlShortcutKeyToAppKey(key_event.key.scancode, key_event.key.key),
                  lfs::vis::input::KEY_4);
        ASSERT_EQ(buffer.input_events.size(), 2u);
        EXPECT_EQ(buffer.input_events.back().kind, lfs::vis::FrameInputEventKind::Text);
        EXPECT_EQ(buffer.input_events.back().text, "'");
    }

    TEST(SdlRmlKeyMappingTest, ModifierTranslationPreservesMeta) {
        const int mods = lfs::vis::gui::sdlModsToRml(
            true, false, true, true);

        EXPECT_NE(mods & Rml::Input::KM_CTRL, 0);
        EXPECT_NE(mods & Rml::Input::KM_ALT, 0);
        EXPECT_NE(mods & Rml::Input::KM_META, 0);
        EXPECT_EQ(mods & Rml::Input::KM_SHIFT, 0);
    }

    TEST(FrameInputBufferTest, IgnoresForeignWindowEvents) {
        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();

        SDL_Event key_event{};
        key_event.type = SDL_EVENT_KEY_DOWN;
        key_event.key.windowID = 7;
        key_event.key.scancode = SDL_SCANCODE_A;
        buffer.processEvent(key_event, 11);

        SDL_Event mouse_event{};
        mouse_event.type = SDL_EVENT_MOUSE_BUTTON_DOWN;
        mouse_event.button.windowID = 7;
        mouse_event.button.button = SDL_BUTTON_LEFT;
        buffer.processEvent(mouse_event, 11);

        SDL_Event text_event{};
        text_event.type = SDL_EVENT_TEXT_INPUT;
        text_event.text.windowID = 7;
        text_event.text.text = "x";
        buffer.processEvent(text_event, 11);

        EXPECT_TRUE(buffer.keys_pressed.empty());
        EXPECT_FALSE(buffer.mouse_clicked[0]);
        EXPECT_TRUE(buffer.input_events.empty());
        EXPECT_FALSE(buffer.had_event);
        EXPECT_FALSE(buffer.mouse_moved);
    }

    TEST(FrameInputBufferTest, RecordsMatchingWindowEventsAndUtf8) {
        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();

        SDL_Event key_event{};
        key_event.type = SDL_EVENT_KEY_DOWN;
        key_event.key.windowID = 11;
        key_event.key.scancode = SDL_SCANCODE_A;
        buffer.processEvent(key_event, 11);

        SDL_Event mouse_event{};
        mouse_event.type = SDL_EVENT_MOUSE_BUTTON_DOWN;
        mouse_event.button.windowID = 11;
        mouse_event.button.button = SDL_BUTTON_LEFT;
        buffer.processEvent(mouse_event, 11);

        SDL_Event text_event{};
        text_event.type = SDL_EVENT_TEXT_INPUT;
        text_event.text.windowID = 11;
        text_event.text.text = reinterpret_cast<const char*>(u8"é");
        buffer.processEvent(text_event, 11);

        ASSERT_EQ(buffer.keys_pressed.size(), 1u);
        EXPECT_EQ(buffer.keys_pressed.front(), SDL_SCANCODE_A);
        EXPECT_TRUE(buffer.mouse_clicked[0]);
        ASSERT_EQ(buffer.input_events.size(), 3u);
        EXPECT_EQ(buffer.input_events[1].kind, lfs::vis::FrameInputEventKind::MouseButton);
        EXPECT_EQ(buffer.input_events.back().kind, lfs::vis::FrameInputEventKind::Text);
        EXPECT_EQ(buffer.input_events.back().text, "\xc3\xa9");
        EXPECT_TRUE(buffer.had_event);
        EXPECT_FALSE(buffer.mouse_moved);
    }

    TEST(FrameInputBufferTest, PreservesOrderedMouseButtonsAndHorizontalWheel) {
        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();

        SDL_Event first_down{};
        first_down.type = SDL_EVENT_MOUSE_BUTTON_DOWN;
        first_down.button.windowID = 11;
        first_down.button.button = SDL_BUTTON_LEFT;
        first_down.button.x = 10.0f;
        first_down.button.y = 20.0f;
        first_down.button.timestamp = 100;
        first_down.button.clicks = 1;
        buffer.processEvent(first_down, 11);

        SDL_Event first_up = first_down;
        first_up.type = SDL_EVENT_MOUSE_BUTTON_UP;
        first_up.button.timestamp = 101;
        buffer.processEvent(first_up, 11);

        SDL_Event second_down = first_down;
        second_down.button.x = 30.0f;
        second_down.button.y = 40.0f;
        second_down.button.timestamp = 102;
        second_down.button.clicks = 2;
        buffer.processEvent(second_down, 11);

        SDL_Event wheel{};
        wheel.type = SDL_EVENT_MOUSE_WHEEL;
        wheel.wheel.windowID = 11;
        wheel.wheel.x = 2.0f;
        wheel.wheel.y = -3.0f;
        buffer.processEvent(wheel, 11);

        ASSERT_EQ(buffer.mouse_button_events.size(), 3u);
        EXPECT_TRUE(buffer.mouse_button_events[0].down);
        EXPECT_FALSE(buffer.mouse_button_events[1].down);
        EXPECT_TRUE(buffer.mouse_button_events[2].down);
        EXPECT_FLOAT_EQ(buffer.mouse_button_events[0].x, 10.0f);
        EXPECT_FLOAT_EQ(buffer.mouse_button_events[2].x, 30.0f);
        EXPECT_EQ(buffer.mouse_button_events[0].timestamp, 100u);
        EXPECT_EQ(buffer.mouse_button_events[2].clicks, 2u);
        EXPECT_FLOAT_EQ(buffer.mouse_wheel, -3.0f);
        EXPECT_FLOAT_EQ(buffer.mouse_wheel_x, 2.0f);
        EXPECT_TRUE(buffer.mouse_clicked[0]);
        EXPECT_TRUE(buffer.mouse_released[0]);
    }

    TEST(FrameInputBufferTest, TracksMouseWindowAndWakeEventsForRenderDemand) {
        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();

        SDL_Event motion_event{};
        motion_event.type = SDL_EVENT_MOUSE_MOTION;
        motion_event.motion.windowID = 11;
        buffer.processEvent(motion_event, 11);

        EXPECT_TRUE(buffer.had_event);
        EXPECT_TRUE(buffer.mouse_moved);
        EXPECT_FALSE(buffer.window_event);
        EXPECT_FALSE(buffer.user_event);

        buffer.beginFrame();

        SDL_Event window_event{};
        window_event.type = SDL_EVENT_WINDOW_RESIZED;
        window_event.window.windowID = 11;
        buffer.processEvent(window_event, 11);

        EXPECT_TRUE(buffer.had_event);
        EXPECT_TRUE(buffer.window_event);
        EXPECT_FALSE(buffer.mouse_moved);

        buffer.beginFrame();

        SDL_Event wake_event{};
        wake_event.type = SDL_EVENT_USER;
        buffer.processEvent(wake_event, 11);

        EXPECT_TRUE(buffer.had_event);
        EXPECT_TRUE(buffer.user_event);
        EXPECT_FALSE(buffer.window_event);
    }

} // namespace
