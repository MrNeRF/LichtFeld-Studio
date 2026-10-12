/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "input/input_bindings.hpp"
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace lfs::vis::input {
    namespace {
        TEST(KeymapLegacyIdTest, EarlyReleaseScaleAndHomeKeepTheirMeaningAfterRebind) {
            struct RestorePersistence {
                bool enabled = InputBindings::isPersistenceEnabled();
                ~RestorePersistence() { InputBindings::setPersistenceEnabled(enabled); }
            } restore;
            InputBindings::setPersistenceEnabled(false);
            namespace fs = std::filesystem;
            const fs::path root = fs::path(PROJECT_ROOT_PATH) / "tests/data/keymap_defaults";
            const fs::path path = fs::temp_directory_path() / "lfs_early_keymap_rebind.json";
            for (const std::string_view release : {"v0.5.0", "v0.5.1", "v0.5.2"}) {
                SCOPED_TRACE(std::string(release));
                auto saved = nlohmann::json::parse(std::ifstream(root /
                                                                 (std::string(release) + ".json")));
                bool changed_scale = false;
                bool changed_home = false;
                for (auto& binding : saved["bindings"]) {
                    if (binding.value("description", "") == "Scale" &&
                        binding.value("trigger_type", "") == "key" &&
                        binding.value("key", -1) == KEY_4 &&
                        binding.value("modifiers", -1) == 0) {
                        binding["key"] = KEY_F9;
                        changed_scale = true;
                    }
                    if (binding.value("description", "") == "Home" &&
                        binding.value("trigger_type", "") == "key" &&
                        binding.value("key", -1) == KEY_H &&
                        binding.value("modifiers", -1) == 0) {
                        binding["key"] = KEY_F8;
                        changed_home = true;
                    }
                }
                ASSERT_TRUE(changed_scale);
                ASSERT_TRUE(changed_home);
                for (auto& binding : saved["bindings"])
                    binding.erase("action_symbol");
                {
                    std::ofstream output(path);
                    output << saved.dump(2);
                }
                InputBindings bindings;
                ASSERT_TRUE(bindings.loadProfileFromFile(path));
                EXPECT_EQ(bindings.getActionForKey(ToolMode::SELECTION, KEY_F9, 0),
                          Action::TOOL_SCALE);
                EXPECT_EQ(bindings.getActionForKey(ToolMode::SELECTION, KEY_F8, 0),
                          Action::CAMERA_RESET_HOME);
                ASSERT_TRUE(bindings.saveProfileToFile(path));
                InputBindings reloaded;
                ASSERT_TRUE(reloaded.loadProfileFromFile(path));
                EXPECT_EQ(reloaded.getActionForKey(ToolMode::SELECTION, KEY_F9, 0),
                          Action::TOOL_SCALE);
                EXPECT_EQ(reloaded.getActionForKey(ToolMode::SELECTION, KEY_F8, 0),
                          Action::CAMERA_RESET_HOME);
            }
            fs::remove(path);
        }

        TEST(KeymapLegacyIdTest, OldActionIdTableAlsoCoversUserAddedNonDefaultAction) {
            struct RestorePersistence {
                bool enabled = InputBindings::isPersistenceEnabled();
                ~RestorePersistence() { InputBindings::setPersistenceEnabled(enabled); }
            } restore;
            InputBindings::setPersistenceEnabled(false);
            namespace fs = std::filesystem;
            const fs::path path = fs::temp_directory_path() / "lfs_early_nondefault_action.json";
            auto saved = nlohmann::json::parse(std::ifstream(
                fs::path(PROJECT_ROOT_PATH) / "tests/data/keymap_defaults/v0.5.0.json"));
            saved["bindings"].push_back({{"mode", static_cast<int>(ToolMode::SELECTION)},
                                         {"action", 37}, // BRUSH_RESIZE in this release.
                                         {"description", "Resize Brush"},
                                         {"trigger_type", "scroll"},
                                         {"modifiers", MODIFIER_CTRL | MODIFIER_SHIFT | MODIFIER_ALT}});
            {
                std::ofstream output(path);
                output << saved.dump(2);
            }
            InputBindings bindings;
            ASSERT_TRUE(bindings.loadProfileFromFile(path));
            EXPECT_EQ(bindings.getActionForScroll(
                          ToolMode::SELECTION,
                          MODIFIER_CTRL | MODIFIER_SHIFT | MODIFIER_ALT),
                      Action::BRUSH_RESIZE);
            fs::remove(path);
        }
    } // namespace
} // namespace lfs::vis::input
