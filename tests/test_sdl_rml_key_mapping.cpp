/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/sdl_rml_key_mapping.hpp"
#include "input/frame_input_buffer.hpp"
#include "input/key_codes.hpp"
#include "input/sdl_key_mapping.hpp"
#include <gtest/gtest.h>

namespace {

    TEST(SdlRmlKeyMappingTest, LetterScancodeMapsToAlphabetKey) {
        EXPECT_EQ(lfs::vis::gui::sdlScancodeToRml(SDL_SCANCODE_A), Rml::Input::KI_A);
    }

    TEST(SdlRmlKeyMappingTest, EqualsKeyMapsToOemPlus) {
        EXPECT_EQ(lfs::vis::gui::sdlKeycodeToRml(SDLK_EQUALS), Rml::Input::KI_OEM_PLUS);
        EXPECT_EQ(lfs::vis::gui::sdlScancodeToRml(SDL_SCANCODE_EQUALS), Rml::Input::KI_OEM_PLUS);
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
        EXPECT_TRUE(buffer.text_codepoints.empty());
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
        ASSERT_EQ(buffer.text_codepoints.size(), 1u);
        EXPECT_EQ(buffer.text_codepoints.front(), 0xE9u);
    }

} // namespace
