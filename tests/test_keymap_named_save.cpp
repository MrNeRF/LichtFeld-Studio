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

        TEST(KeymapNamedSaveTest, NamedSaveWritesAndLoadsItsOwnActiveName) {
            namespace fs = std::filesystem;
            const fs::path root = fs::temp_directory_path() / "lfs_keymap_named_save";
            ScopedProfileHome home(root);
            const auto paths = core::UserPaths::resolve();
            ASSERT_TRUE(paths);
            ASSERT_TRUE(paths->ensureDirectories());
            InputBindings bindings;
            ASSERT_TRUE(bindings.saveProfile("FirstName"));
            const auto first = nlohmann::json::parse(
                std::ifstream(paths->keymapDir() / "FirstName.json"));
            EXPECT_EQ(first.at("name"), "FirstName");
            bindings.loadProfile("FirstName");
            EXPECT_EQ(bindings.getCurrentProfileName(), "FirstName");
            ASSERT_TRUE(bindings.saveProfile("SecondName"));
            const auto second = nlohmann::json::parse(
                std::ifstream(paths->keymapDir() / "SecondName.json"));
            EXPECT_EQ(second.at("name"), "SecondName");
            bindings.loadProfile("SecondName");
            EXPECT_EQ(bindings.getCurrentProfileName(), "SecondName");
        }
    } // namespace
} // namespace lfs::vis::input
