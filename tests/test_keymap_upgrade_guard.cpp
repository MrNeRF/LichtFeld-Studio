/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "input/input_bindings.hpp"
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <sstream>

namespace lfs::vis::input {
    namespace {
        namespace fs = std::filesystem;
        using json = nlohmann::json;
        struct RestorePersistence {
            bool enabled = InputBindings::isPersistenceEnabled();
            ~RestorePersistence() { InputBindings::setPersistenceEnabled(enabled); }
        };

        std::string bindingDigest(const json& bindings) {
            std::uint64_t digest = 14695981039346656037ull;
            for (const unsigned char byte : bindings.dump())
                digest = (digest ^ byte) * 1099511628211ull;
            std::ostringstream output;
            output << std::hex << std::setfill('0') << std::setw(16) << digest;
            return output.str();
        }

        TEST(KeymapUpgradeGuardTest, CurrentDefaultsHaveARecordedVersion) {
            RestorePersistence restore;
            InputBindings::setPersistenceEnabled(false);
            InputBindings current;
            const auto saved = fs::temp_directory_path() / "lfs_keymap_recorded_defaults.json";
            ASSERT_TRUE(current.saveProfileToFile(saved));
            const auto actual = json::parse(std::ifstream(saved));
            fs::remove(saved);
            const fs::path baseline = fs::path(PROJECT_ROOT_PATH) /
                                      "tests/data/keymap_contract" /
                                      ("current_v" + std::to_string(actual.at("version").get<int>()) +
                                       ".digest");
            ASSERT_TRUE(fs::exists(baseline))
                << "A new keymap profile version needs the previous defaults. Run: "
                   "python3 tools/generate_keymap_defaults.py --record-previous HEAD";
            std::ifstream digest_file(baseline);
            std::string version_label;
            int recorded_version = -1;
            std::string digest_label;
            std::string recorded_digest;
            digest_file >> version_label >> recorded_version >> digest_label >> recorded_digest;
            EXPECT_EQ(version_label, "version");
            EXPECT_EQ(digest_label, "fnv1a64");
            EXPECT_EQ(actual.at("version"), recorded_version)
                << "A new keymap profile version needs the previous defaults. Run: "
                   "python3 tools/generate_keymap_defaults.py --record-previous HEAD";
            EXPECT_EQ(bindingDigest(actual.at("bindings")), recorded_digest)
                << "Default bindings changed without recording the previous profile version. "
                   "Bump PROFILE_VERSION, then run: "
                   "python3 tools/generate_keymap_defaults.py --record-previous HEAD";
        }

        TEST(KeymapUpgradeGuardTest, StaleFullListDoesNotOverrideFutureDefaults) {
            RestorePersistence restore;
            const fs::path path = fs::temp_directory_path() / "lfs_keymap_future_defaults.json";
            InputBindings::setPersistenceEnabled(false);
            InputBindings current;
            ASSERT_TRUE(current.saveProfileToFile(path));
            auto saved = json::parse(std::ifstream(path));
            for (auto& binding : saved["bindings"]) {
                if (binding.value("action", -1) == static_cast<int>(Action::CAMERA_ORBIT)) {
                    binding["button"] = static_cast<int>(MouseButton::RIGHT);
                }
            }
            {
                std::ofstream out(path);
                out << saved.dump(2);
            }
            InputBindings loaded;
            ASSERT_TRUE(loaded.loadProfileFromFile(path));
            EXPECT_EQ(loaded.getActionForDrag(ToolMode::SELECTION, MouseButton::MIDDLE, 0),
                      Action::CAMERA_ORBIT);
            EXPECT_EQ(loaded.getActionForDrag(ToolMode::SELECTION, MouseButton::RIGHT, 0),
                      Action::CAMERA_PAN);
            fs::remove(path);
        }

        TEST(KeymapUpgradeGuardTest, RemovingOneParallelDefaultKeepsItsNeighborInherited) {
            RestorePersistence restore;
            InputBindings::setPersistenceEnabled(false);
            const fs::path root = fs::path(PROJECT_ROOT_PATH) / "tests/data/keymap_defaults";
            const fs::path path = fs::temp_directory_path() / "lfs_keymap_parallel_default.json";
            auto saved = json::parse(std::ifstream(root / "v0.5.3.json"));
            bool removed_shift = false;
            for (auto it = saved["bindings"].begin(); it != saved["bindings"].end();) {
                if (it->value("description", "") == "Brush size" &&
                    it->value("trigger_type", "") == "scroll" &&
                    it->value("modifiers", -1) == MODIFIER_SHIFT) {
                    it = saved["bindings"].erase(it);
                    removed_shift = true;
                } else {
                    ++it;
                }
            }
            ASSERT_TRUE(removed_shift);
            {
                std::ofstream output(path);
                output << saved.dump(2);
            }
            InputBindings loaded;
            ASSERT_TRUE(loaded.loadProfileFromFile(path));
            ASSERT_TRUE(loaded.saveProfileToFile(path));
            const auto upgraded = json::parse(std::ifstream(path));
            EXPECT_EQ(upgraded.at("version"), 31);
            size_t brush_additions = 0;
            size_t brush_removals = 0;
            for (const auto& addition : upgraded["overrides"]["bindings"])
                brush_additions += addition.value("action", -1) ==
                                   static_cast<int>(Action::BRUSH_RESIZE);
            for (const auto& removal : upgraded["overrides"]["removed"])
                brush_removals += removal.value("action", -1) ==
                                  static_cast<int>(Action::BRUSH_RESIZE);
            EXPECT_EQ(brush_additions, 0u);
            EXPECT_EQ(brush_removals, 1u);
            InputBindings reloaded;
            ASSERT_TRUE(reloaded.loadProfileFromFile(path));
            EXPECT_EQ(reloaded.getActionForScroll(ToolMode::SELECTION, MODIFIER_CTRL),
                      Action::BRUSH_RESIZE);
            EXPECT_EQ(reloaded.getActionForScroll(ToolMode::SELECTION, MODIFIER_SHIFT),
                      Action::NONE);
            fs::remove(path);
        }

        TEST(KeymapUpgradeGuardTest, ClearingOneDefaultWritesAnExplicitTombstone) {
            RestorePersistence restore;
            InputBindings::setPersistenceEnabled(false);
            const fs::path path = fs::temp_directory_path() / "lfs_keymap_clear_tombstone.json";
            InputBindings bindings;
            bindings.clearBinding(ToolMode::GLOBAL, Action::TOOL_SCALE);
            ASSERT_TRUE(bindings.saveProfileToFile(path));
            auto saved = json::parse(std::ifstream(path));
            const auto& removed = saved.at("overrides").at("removed");
            EXPECT_TRUE(std::ranges::any_of(removed, [](const json& item) {
                return item.value("mode", -1) == static_cast<int>(ToolMode::GLOBAL) &&
                       item.value("action", -1) == static_cast<int>(Action::TOOL_SCALE);
            }));
            // The override record, rather than the compatibility full list,
            // defines the effective binding when this file is loaded again.
            saved["bindings"].push_back({{"mode", static_cast<int>(ToolMode::GLOBAL)},
                                         {"action", static_cast<int>(Action::TOOL_SCALE)},
                                         {"description", "Scale"},
                                         {"trigger_type", "key"},
                                         {"key", KEY_4},
                                         {"modifiers", 0}});
            {
                std::ofstream output(path);
                output << saved.dump(2);
            }
            InputBindings reloaded;
            ASSERT_TRUE(reloaded.loadProfileFromFile(path));
            EXPECT_EQ(reloaded.getActionForKey(ToolMode::GLOBAL, KEY_4, 0), Action::NONE);
            fs::remove(path);
        }
    } // namespace
} // namespace lfs::vis::input
