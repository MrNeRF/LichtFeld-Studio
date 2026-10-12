/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "input/input_bindings.hpp"
#include "input/sdl_key_mapping.hpp"
#include <array>
#include <gtest/gtest.h>
#include <set>

namespace lfs::vis::input {
    namespace {
        TEST(KeymapExtendedKeysTest, FunctionKeysF13ThroughF24StayDistinct) {
            struct RestorePersistence {
                bool enabled = InputBindings::isPersistenceEnabled();
                ~RestorePersistence() { InputBindings::setPersistenceEnabled(enabled); }
            } restore;
            InputBindings::setPersistenceEnabled(false);
            InputBindings bindings;
            bindings.startCapture(ToolMode::GLOBAL, Action::TOOL_SCALE);
            bindings.captureKey(KEY_UNKNOWN, KEY_UNKNOWN, 0);
            EXPECT_TRUE(bindings.isCapturing());
            EXPECT_FALSE(bindings.getAndClearCaptured());
            bindings.cancelCapture();
            bindings.setBinding(ToolMode::GLOBAL, Action::TOOL_SCALE,
                                KeyTrigger{KEY_UNKNOWN});
            EXPECT_EQ(bindings.getActionForShortcut(ToolMode::SELECTION, KEY_UNKNOWN,
                                                    KEY_UNKNOWN, 0),
                      Action::NONE);
            std::set<int> app_keys;
            for (int i = 0; i < 12; ++i) {
                const auto scancode = static_cast<SDL_Scancode>(SDL_SCANCODE_F13 + i);
                const auto keycode = static_cast<SDL_Keycode>(SDLK_F13 + i);
                const int physical = sdlScancodeToAppKey(scancode);
                const int logical = sdlKeycodeToAppKey(keycode);
                SCOPED_TRACE("F" + std::to_string(13 + i));
                ASSERT_NE(physical, KEY_UNKNOWN);
                EXPECT_EQ(logical, physical);
                EXPECT_TRUE(app_keys.insert(physical).second);
                EXPECT_EQ(appKeyToSdlScancode(physical), scancode);
                EXPECT_EQ(getKeyName(physical), "F" + std::to_string(13 + i));
                bindings.startCapture(ToolMode::GLOBAL, Action::TOOL_SCALE);
                bindings.captureKey(physical, logical, 0);
                const auto captured = bindings.getAndClearCaptured();
                ASSERT_TRUE(captured);
                ASSERT_TRUE(std::holds_alternative<KeyTrigger>(*captured));
                EXPECT_EQ(std::get<KeyTrigger>(*captured).key, physical);
                bindings.setBinding(ToolMode::GLOBAL, Action::TOOL_SCALE, *captured);
                EXPECT_EQ(bindings.getActionForShortcut(ToolMode::SELECTION, physical,
                                                        logical, 0),
                          Action::TOOL_SCALE);
            }
            EXPECT_EQ(app_keys.size(), 12u);
            for (const int key : app_keys) {
                bindings.setBinding(ToolMode::GLOBAL, Action::TOOL_SCALE,
                                    KeyTrigger{key});
                for (const int other : app_keys) {
                    if (other != key)
                        EXPECT_NE(bindings.getActionForKey(ToolMode::SELECTION, other, 0),
                                  Action::TOOL_SCALE);
                }
            }
        }

    } // namespace
} // namespace lfs::vis::input
