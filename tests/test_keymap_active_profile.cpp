/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "input/input_bindings.hpp"
#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>

namespace lfs::vis::input {
    namespace {
        struct ScopedHome {
            bool previous_persistence = InputBindings::isPersistenceEnabled();
            std::string previous;
            bool had_previous = false;
            explicit ScopedHome(const std::filesystem::path& path) {
                if (const char* old = std::getenv("LFS_HOME")) {
                    previous = old;
                    had_previous = true;
                }
#ifdef _WIN32
                _putenv_s("LFS_HOME", path.string().c_str());
#else
                setenv("LFS_HOME", path.string().c_str(), 1);
#endif
                InputBindings::setPersistenceEnabled(true);
            }
            ~ScopedHome() {
#ifdef _WIN32
                _putenv_s("LFS_HOME", had_previous ? previous.c_str() : "");
#else
                if (had_previous)
                    setenv("LFS_HOME", previous.c_str(), 1);
                else
                    unsetenv("LFS_HOME");
#endif
                InputBindings::setPersistenceEnabled(previous_persistence);
            }
        };

        TEST(KeymapActiveProfileTest, NamedProfileRemainsActiveAfterRestart) {
            namespace fs = std::filesystem;
            const auto home = fs::temp_directory_path() / "lfs_active_keymap_guard";
            fs::remove_all(home);
            fs::create_directories(home / "config" / "keymaps");
            {
                ScopedHome environment(home);
                InputBindings first;
                first.setBinding(ToolMode::GLOBAL, Action::TOOL_SCALE, KeyTrigger{KEY_J, 0});
                ASSERT_TRUE(first.saveProfile("Named"));
                EXPECT_EQ(first.getCurrentProfileName(), "Named");
                EXPECT_EQ(first.getActionForKey(ToolMode::GLOBAL, KEY_J, 0), Action::TOOL_SCALE);
                InputBindings restarted;
                EXPECT_EQ(restarted.getCurrentProfileName(), "Named");
                EXPECT_EQ(restarted.getActionForKey(ToolMode::GLOBAL, KEY_J, 0), Action::TOOL_SCALE);
                EXPECT_EQ(restarted.getActionForKey(ToolMode::GLOBAL, KEY_4, 0), Action::NONE);
                restarted.loadProfile("Default");
                InputBindings default_restart;
                EXPECT_EQ(default_restart.getCurrentProfileName(), "Default");
                EXPECT_EQ(default_restart.getActionForKey(ToolMode::GLOBAL, KEY_4, 0), Action::TOOL_SCALE);
            }
            fs::remove_all(home);
        }
    } // namespace
} // namespace lfs::vis::input
