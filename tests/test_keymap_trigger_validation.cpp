/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "input/input_bindings.hpp"
#include "input/sdl_key_mapping.hpp"
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

namespace lfs::vis::input {
    namespace {
        struct NoProfilePersistence {
            bool previous = InputBindings::isPersistenceEnabled();
            NoProfilePersistence() { InputBindings::setPersistenceEnabled(false); }
            ~NoProfilePersistence() { InputBindings::setPersistenceEnabled(previous); }
        };

        TEST(KeymapTriggerValidationTest, RejectsMovementModifiersAtCaptureAndImport) {
            NoProfilePersistence isolated;
            InputBindings bindings;
            for (const int forbidden : {MODIFIER_CTRL, MODIFIER_ALT, MODIFIER_SUPER}) {
                SCOPED_TRACE(forbidden);
                EXPECT_FALSE(InputBindings::isValidTriggerForAction(
                    Action::CAMERA_MOVE_FORWARD, KeyTrigger{KEY_K, forbidden}));
                bindings.startCapture(ToolMode::GLOBAL, Action::CAMERA_MOVE_FORWARD);
                bindings.captureKey(KEY_K, KEY_K, forbidden);
                EXPECT_TRUE(bindings.isCapturing());
                EXPECT_FALSE(bindings.getAndClearCaptured());
                bindings.cancelCapture();
                bindings.setBinding(ToolMode::GLOBAL, Action::CAMERA_MOVE_FORWARD,
                                    KeyTrigger{KEY_K, forbidden});
                EXPECT_NE(bindings.getKeyForAction(Action::CAMERA_MOVE_FORWARD), KEY_K);
            }
            bindings.startCapture(ToolMode::GLOBAL, Action::CAMERA_MOVE_FORWARD);
            bindings.captureKey(KEY_K, KEY_K, MODIFIER_SHIFT);
            EXPECT_FALSE(bindings.isCapturing());
            EXPECT_EQ(bindings.getKeyForAction(Action::CAMERA_MOVE_FORWARD), KEY_K);

            const auto path = std::filesystem::temp_directory_path() /
                              "lfs_invalid_movement_keymap.json";
            {
                std::ofstream output(path);
                output << R"({"name":"Broken","version":29,
                "bindings":[{"mode":0,"action":5,"description":"Move Forward",
                "trigger_type":"key","key":75,"modifiers":2}]})";
            }
            EXPECT_FALSE(bindings.loadProfileFromFile(path));
            EXPECT_EQ(bindings.getKeyForAction(Action::CAMERA_MOVE_FORWARD), KEY_K);
            std::filesystem::remove(path);
        }

        TEST(KeymapTriggerValidationTest, RejectsImpossibleModifierMasks) {
            NoProfilePersistence isolated;
            InputBindings bindings;
            for (const int invalid : {16, 32, -1}) {
                SCOPED_TRACE(invalid);
                const KeyTrigger key{KEY_4, invalid};
                EXPECT_FALSE(InputBindings::isValidTriggerForAction(Action::TOOL_SCALE, key));
                bindings.setBinding(ToolMode::GLOBAL, Action::TOOL_SCALE, key);
                EXPECT_EQ(bindings.getActionForKey(ToolMode::GLOBAL, KEY_4, 0),
                          Action::TOOL_SCALE);
            }
            EXPECT_TRUE(InputBindings::isValidTriggerForAction(
                Action::TOOL_SCALE, KeyTrigger{KEY_4, MODIFIER_SHIFT | MODIFIER_CTRL}));
        }

        TEST(KeymapTriggerValidationTest, CapturedInsertShowsItsNameAndDispatches) {
            NoProfilePersistence isolated;
            InputBindings bindings;
            const int physical = sdlScancodeToAppKey(SDL_SCANCODE_INSERT);
            const int logical = sdlKeycodeToAppKey(SDLK_INSERT);
            EXPECT_EQ(physical, KEY_INSERT);
            EXPECT_EQ(logical, KEY_INSERT);
            bindings.startCapture(ToolMode::GLOBAL, Action::TOOL_SCALE);
            bindings.captureKey(physical, logical, 0);
            const auto captured = bindings.getAndClearCaptured();
            ASSERT_TRUE(captured);
            EXPECT_EQ(getKeyName(KEY_INSERT), "Insert");
            EXPECT_EQ(bindings.getTriggerDescription(Action::TOOL_SCALE), "Insert");
            EXPECT_EQ(bindings.getActionForShortcut(ToolMode::GLOBAL, physical, logical, 0),
                      Action::TOOL_SCALE);
        }
    } // namespace
} // namespace lfs::vis::input
