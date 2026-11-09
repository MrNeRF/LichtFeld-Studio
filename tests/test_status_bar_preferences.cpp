/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/environment.hpp"
#include "core/user_paths.hpp"
#include "preferences.hpp"
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <optional>

namespace {
    class StatusBarPreferenceTest : public ::testing::Test {
    protected:
        std::filesystem::path home;
        std::optional<std::string> previous;
        void SetUp() override {
            if (const auto* value = std::getenv("LFS_HOME"))
                previous = value;
            home = std::filesystem::temp_directory_path() / ("lfs-status-bar-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
            std::filesystem::create_directories(home);
            setHome(home.string());
        }
        void TearDown() override {
            if (previous)
                setHome(*previous);
            else {
#ifdef _WIN32
                (void)_putenv_s("LFS_HOME", "");
#else
                (void)unsetenv("LFS_HOME");
#endif
            }
            std::filesystem::remove_all(home);
        }
        void setHome(const std::string& value) {
#ifdef _WIN32
            (void)_putenv_s("LFS_HOME", value.c_str());
#else
            (void)setenv("LFS_HOME", value.c_str(), 1);
#endif
        }
    };
    TEST_F(StatusBarPreferenceTest, DefaultsKeepCommonItemsAndHideDiagnostics) {
        const auto state = lfs::vis::UserPreferences::instance().statusBar();
        for (const auto& field : lfs::vis::STATUS_BAR_PREFERENCE_FIELDS) {
            const std::string key = field.key;
            const bool expected = key == "enabled" || key == "activity" || key == "training_progress" ||
                                  key == "camera" || key == "gaussians" || key == "selected" || key == "fps" || key == "vram";
            EXPECT_EQ(state.*field.member, expected) << key;
        }
    }
    TEST_F(StatusBarPreferenceTest, PersistsIndependentChoicesAndResetsOnlyStatusBar) {
        auto& preferences = lfs::vis::UserPreferences::instance();
        preferences.setProgressBarStyle("tessle");
        auto state = preferences.statusBar();
        for (const auto& field : lfs::vis::STATUS_BAR_PREFERENCE_FIELDS)
            state.*field.member = !(state.*field.member);
        preferences.setStatusBar(state);
        EXPECT_EQ(preferences.statusBar(), state);
        const auto paths = lfs::core::UserPaths::resolve();
        ASSERT_TRUE(paths);
        std::ifstream input(paths->preferencesFile());
        const auto saved = nlohmann::json::parse(input);
        for (const auto& field : lfs::vis::STATUS_BAR_PREFERENCE_FIELDS)
            EXPECT_EQ(saved.at("status_bar").at(field.key).get<bool>(), state.*field.member);
        preferences.setStatusBar({});
        EXPECT_EQ(preferences.statusBar(), lfs::vis::StatusBarPreferences{});
        EXPECT_EQ(preferences.progressBarStyle(), "tessle");
    }
    TEST_F(StatusBarPreferenceTest, InvalidSavedValuesFallBackPerFieldAndPreserveUnknownSettings) {
        const auto paths = lfs::core::UserPaths::resolve();
        ASSERT_TRUE(paths);
        std::filesystem::create_directories(paths->preferencesFile().parent_path());
        {
            std::ofstream output(paths->preferencesFile());
            output << R"({"future_setting":42,"status_bar":{"fps":"bad","vram":false,"gpu_name":true,"future_item":true}})";
        }
        auto& preferences = lfs::vis::UserPreferences::instance();
        const auto state = preferences.statusBar();
        EXPECT_TRUE(state.fps);
        EXPECT_FALSE(state.vram);
        EXPECT_TRUE(state.gpu_name);
        EXPECT_TRUE(state.activity);
        preferences.setStatusBar({});
        std::ifstream input(paths->preferencesFile());
        const auto saved = nlohmann::json::parse(input);
        EXPECT_EQ(saved.at("future_setting"), 42);
        EXPECT_EQ(saved.at("status_bar").at("future_item"), true);
    }
} // namespace
