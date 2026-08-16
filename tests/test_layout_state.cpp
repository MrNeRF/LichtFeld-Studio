/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include <visualizer/gui/layout_state.hpp>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

namespace {

    class LayoutStatePersistenceTest : public ::testing::Test {
    protected:
        void SetUp() override {
            static std::atomic_uint64_t sequence = 0;
            const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
            temp_dir_ = std::filesystem::temp_directory_path() /
                        ("lfs_layout_state_test_" + std::to_string(stamp) + "_" +
                         std::to_string(sequence.fetch_add(1)));
            std::filesystem::create_directories(temp_dir_);
            legacy_path_ = temp_dir_ / "layout.json";
            preferences_path_ = temp_dir_ / "ui_preferences.json";
        }

        void TearDown() override {
            std::error_code ec;
            std::filesystem::remove_all(temp_dir_, ec);
        }

        static void writeJson(const std::filesystem::path& path,
                              const nlohmann::json& json) {
            std::ofstream file(path);
            ASSERT_TRUE(file);
            file << json.dump(2);
            ASSERT_TRUE(file);
        }

        [[nodiscard]] static nlohmann::json readJson(const std::filesystem::path& path) {
            std::ifstream file(path);
            EXPECT_TRUE(file);
            return nlohmann::json::parse(file);
        }

        std::filesystem::path temp_dir_;
        std::filesystem::path legacy_path_;
        std::filesystem::path preferences_path_;
    };

} // namespace

TEST_F(LayoutStatePersistenceTest, SavesCurrentVersionAndRoundTripsUserPreferences) {
    using lfs::vis::gui::LayoutState;

    LayoutState saved;
    saved.scene_panel_ratio = 0.63f;
    saved.window_visibility = {{"scene_panel", true}};
    saved.file_association = "declined";
    saved.vram_hud_x = 42.0f;
    saved.vram_hud_y = 51.0f;
    saved.vram_hud_width = 640.0f;
    saved.vram_hud_height = 480.0f;
    saved.vram_hud_active_tab = "allocations";
    saved.vram_hud_collapsed_paths = {"root/a", "root/b"};
    saved.perf_hud_visible = true;
    saved.perf_hud_expanded = false;

    ASSERT_TRUE(saved.saveUserPreferencesTo(preferences_path_));
    const auto persisted = readJson(preferences_path_);
    EXPECT_EQ(persisted.at("version"), 1);
    EXPECT_FALSE(persisted.contains("scene_panel_ratio"));
    EXPECT_FALSE(persisted.contains("windows"));

    LayoutState loaded;
    ASSERT_TRUE(loaded.loadFrom(legacy_path_, preferences_path_));
    EXPECT_FLOAT_EQ(loaded.scene_panel_ratio, 0.4f);
    EXPECT_TRUE(loaded.window_visibility.empty());
    EXPECT_EQ(loaded.file_association, saved.file_association);
    EXPECT_FLOAT_EQ(loaded.vram_hud_x, saved.vram_hud_x);
    EXPECT_FLOAT_EQ(loaded.vram_hud_y, saved.vram_hud_y);
    EXPECT_FLOAT_EQ(loaded.vram_hud_width, saved.vram_hud_width);
    EXPECT_FLOAT_EQ(loaded.vram_hud_height, saved.vram_hud_height);
    EXPECT_EQ(loaded.vram_hud_active_tab, saved.vram_hud_active_tab);
    EXPECT_EQ(loaded.vram_hud_collapsed_paths, saved.vram_hud_collapsed_paths);
    EXPECT_EQ(loaded.perf_hud_visible, saved.perf_hud_visible);
    EXPECT_EQ(loaded.perf_hud_expanded, saved.perf_hud_expanded);
}

TEST_F(LayoutStatePersistenceTest, MigratesLegacyLayoutThenAppliesUserPreferences) {
    using lfs::vis::gui::LayoutState;

    writeJson(legacy_path_, {
                                {"scene_panel_ratio", 0.72},
                                {"bottom_dock_height", 360.0},
                                {"file_association", "legacy"},
                                {"windows", {{"python_console", true}}},
                                {"vram_hud", {{"active_tab", "tree"}}},
                            });
    writeJson(preferences_path_, {
                                     {"file_association", "declined"},
                                     {"vram_hud", {{"collapsed", {"root/preference"}}}},
                                     {"perf_hud", {{"visible", true}, {"expanded", false}}},
                                 });

    LayoutState loaded;
    ASSERT_TRUE(loaded.loadFrom(legacy_path_, preferences_path_));
    EXPECT_FLOAT_EQ(loaded.scene_panel_ratio, 0.72f);
    EXPECT_FLOAT_EQ(loaded.bottom_dock_height, 360.0f);
    EXPECT_TRUE(loaded.window_visibility.at("python_console"));
    EXPECT_EQ(loaded.file_association, "declined");
    EXPECT_EQ(loaded.vram_hud_active_tab, "tree");
    EXPECT_EQ(loaded.vram_hud_collapsed_paths,
              std::vector<std::string>{"root/preference"});
    EXPECT_TRUE(loaded.perf_hud_visible);
    EXPECT_FALSE(loaded.perf_hud_expanded);

    ASSERT_TRUE(loaded.saveUserPreferencesTo(preferences_path_));
    EXPECT_EQ(readJson(preferences_path_).at("version"), 1);
}

TEST_F(LayoutStatePersistenceTest, RejectsFutureVersionWithoutMutatingStateOrFile) {
    using lfs::vis::gui::LayoutState;

    const nlohmann::json future_preferences = {
        {"version", 2},
        {"file_association", "future"},
    };
    writeJson(preferences_path_, future_preferences);

    LayoutState loaded;
    loaded.file_association = "unchanged";
    ASSERT_FALSE(loaded.loadFrom(legacy_path_, preferences_path_));
    EXPECT_EQ(loaded.file_association, "unchanged");
    EXPECT_EQ(readJson(preferences_path_), future_preferences);
}

TEST_F(LayoutStatePersistenceTest, RejectsNonIntegerVersionWithoutMutatingState) {
    using lfs::vis::gui::LayoutState;

    writeJson(preferences_path_, {
                                     {"version", "1"},
                                     {"file_association", "changed"},
                                 });

    LayoutState loaded;
    loaded.file_association = "unchanged";
    ASSERT_FALSE(loaded.loadFrom(legacy_path_, preferences_path_));
    EXPECT_EQ(loaded.file_association, "unchanged");
}

TEST_F(LayoutStatePersistenceTest, InvalidPreferencesDoNotCommitLoadedLegacyState) {
    using lfs::vis::gui::LayoutState;

    writeJson(legacy_path_, {{"scene_panel_ratio", 0.72}});
    writeJson(preferences_path_, {{"version", 2}});

    LayoutState loaded;
    loaded.scene_panel_ratio = 0.25f;
    ASSERT_FALSE(loaded.loadFrom(legacy_path_, preferences_path_));
    EXPECT_FLOAT_EQ(loaded.scene_panel_ratio, 0.25f);
}

TEST_F(LayoutStatePersistenceTest, MissingFilesKeepDefaultsAndAllowFirstSave) {
    using lfs::vis::gui::LayoutState;

    LayoutState state;
    ASSERT_TRUE(state.loadFrom(legacy_path_, preferences_path_));
    EXPECT_FLOAT_EQ(state.scene_panel_ratio, 0.4f);
    ASSERT_TRUE(state.saveUserPreferencesTo(preferences_path_));
    EXPECT_EQ(readJson(preferences_path_).at("version"), 1);
}
