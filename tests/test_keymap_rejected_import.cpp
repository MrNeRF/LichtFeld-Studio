/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/user_paths.hpp"
#include "input/input_bindings.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <optional>

namespace lfs::vis::input {
    namespace {
        class ScopedProfileHome {
        public:
            explicit ScopedProfileHome(const std::filesystem::path& root)
                : root_(root), persistence_(InputBindings::isPersistenceEnabled()) {
                std::error_code error;
                std::filesystem::remove_all(root_, error);
                if (const char* previous = std::getenv("LFS_HOME"))
                    previous_ = previous;
#ifdef _WIN32
                (void)_putenv_s("LFS_HOME", root_.string().c_str());
#else
                (void)setenv("LFS_HOME", root_.string().c_str(), 1);
#endif
                InputBindings::setPersistenceEnabled(true);
                std::filesystem::create_directories(root_);
            }

            ~ScopedProfileHome() {
                InputBindings::setPersistenceEnabled(persistence_);
#ifdef _WIN32
                (void)_putenv_s("LFS_HOME", previous_ ? previous_->c_str() : "");
#else
                if (previous_)
                    (void)setenv("LFS_HOME", previous_->c_str(), 1);
                else
                    (void)unsetenv("LFS_HOME");
#endif
                std::error_code error;
                std::filesystem::remove_all(root_, error);
            }

        private:
            std::filesystem::path root_;
            std::optional<std::string> previous_;
            bool persistence_;
        };

        TEST(KeymapRejectedImportTest, RejectedImportPreservesDisplayedAndLiveBindings) {
            namespace fs = std::filesystem;
            const fs::path root = fs::temp_directory_path() / "lfs_keymap_rejected_import";
            ScopedProfileHome home(root);
            InputBindings bindings;
            bindings.setBinding(ToolMode::GLOBAL, Action::TOOL_SCALE, KeyTrigger{KEY_F9});
            const fs::path valid = root / "valid.json";
            ASSERT_TRUE(bindings.saveProfileToFile(valid));
            ASSERT_TRUE(bindings.loadProfileFromFile(valid));
            const auto before_name = bindings.getCurrentProfileName();
            const auto before_revision = bindings.getBindingsRevision();
            const auto before_display = bindings.getBindingsForMode(ToolMode::GLOBAL);
            ASSERT_EQ(bindings.getActionForKey(ToolMode::SELECTION, KEY_F9, 0),
                      Action::TOOL_SCALE);

            const fs::path invalid = root / "invalid.json";
            {
                std::ofstream output(invalid);
                output << R"({"name":"Broken","version":30,"bindings":[
                    {"mode":0,"action":63,"description":"Translate","trigger_type":"key","key":52,"modifiers":0},
                    {"mode":0,"description":"Missing action","trigger_type":"key","key":53,"modifiers":0}]})";
            }
            EXPECT_FALSE(bindings.loadProfileFromFile(invalid));
            EXPECT_EQ(bindings.getCurrentProfileName(), before_name);
            EXPECT_EQ(bindings.getBindingsRevision(), before_revision);
            EXPECT_EQ(bindings.getBindingsForMode(ToolMode::GLOBAL), before_display);
            EXPECT_EQ(bindings.getActionForKey(ToolMode::SELECTION, KEY_F9, 0),
                      Action::TOOL_SCALE);
        }

    } // namespace
} // namespace lfs::vis::input
