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
            layout_path_ = temp_dir_ / "layout.json";
        }

        void TearDown() override {
            std::error_code ec;
            std::filesystem::remove_all(temp_dir_, ec);
        }

        void writeJson(const nlohmann::json& json) const {
            std::ofstream file(layout_path_);
            ASSERT_TRUE(file);
            file << json.dump(2);
            ASSERT_TRUE(file);
        }

        [[nodiscard]] nlohmann::json readJson() const {
            std::ifstream file(layout_path_);
            EXPECT_TRUE(file);
            return nlohmann::json::parse(file);
        }

        std::filesystem::path temp_dir_;
        std::filesystem::path layout_path_;
    };

} // namespace

TEST_F(LayoutStatePersistenceTest, SavesCurrentVersionAndRoundTripsState) {
    using lfs::vis::gui::LayoutState;

    LayoutState saved;
    saved.scene_panel_ratio = 0.63f;
    saved.python_console_width = 410.0f;
    saved.bottom_dock_height = 275.0f;
    saved.show_sequencer = true;
    saved.file_association = "declined";
    saved.window_visibility = {{"scene_panel", true}, {"python_console", false}};
    saved.vram_hud_x = 42.0f;
    saved.vram_hud_y = 51.0f;
    saved.vram_hud_width = 640.0f;
    saved.vram_hud_height = 480.0f;
    saved.vram_hud_active_tab = "allocations";
    saved.vram_hud_collapsed_paths = {"root/a", "root/b"};

    ASSERT_TRUE(saved.saveTo(layout_path_));
    EXPECT_EQ(readJson().at("version"), 1);

    LayoutState loaded;
    ASSERT_TRUE(loaded.loadFrom(layout_path_));
    EXPECT_FLOAT_EQ(loaded.scene_panel_ratio, saved.scene_panel_ratio);
    EXPECT_FLOAT_EQ(loaded.python_console_width, saved.python_console_width);
    EXPECT_FLOAT_EQ(loaded.bottom_dock_height, saved.bottom_dock_height);
    EXPECT_EQ(loaded.show_sequencer, saved.show_sequencer);
    EXPECT_EQ(loaded.file_association, saved.file_association);
    EXPECT_EQ(loaded.window_visibility, saved.window_visibility);
    EXPECT_FLOAT_EQ(loaded.vram_hud_x, saved.vram_hud_x);
    EXPECT_FLOAT_EQ(loaded.vram_hud_y, saved.vram_hud_y);
    EXPECT_FLOAT_EQ(loaded.vram_hud_width, saved.vram_hud_width);
    EXPECT_FLOAT_EQ(loaded.vram_hud_height, saved.vram_hud_height);
    EXPECT_EQ(loaded.vram_hud_active_tab, saved.vram_hud_active_tab);
    EXPECT_EQ(loaded.vram_hud_collapsed_paths, saved.vram_hud_collapsed_paths);
}

TEST_F(LayoutStatePersistenceTest, MigratesUnversionedLayoutAsVersionZero) {
    using lfs::vis::gui::LayoutState;

    writeJson({
        {"scene_panel_ratio", 0.72},
        {"bottom_dock_height", 360.0},
        {"windows", {{"python_console", true}}},
        {"vram_hud", {{"active_tab", "tree"}, {"collapsed", {"root/legacy"}}}},
    });

    LayoutState loaded;
    ASSERT_TRUE(loaded.loadFrom(layout_path_));
    EXPECT_FLOAT_EQ(loaded.scene_panel_ratio, 0.72f);
    EXPECT_FLOAT_EQ(loaded.bottom_dock_height, 360.0f);
    EXPECT_TRUE(loaded.window_visibility.at("python_console"));
    EXPECT_EQ(loaded.vram_hud_active_tab, "tree");
    EXPECT_EQ(loaded.vram_hud_collapsed_paths, std::vector<std::string>{"root/legacy"});

    ASSERT_TRUE(loaded.saveTo(layout_path_));
    EXPECT_EQ(readJson().at("version"), 1);
}

TEST_F(LayoutStatePersistenceTest, RejectsFutureVersionWithoutMutatingStateOrFile) {
    using lfs::vis::gui::LayoutState;

    const nlohmann::json future_layout = {
        {"version", 2},
        {"scene_panel_ratio", 0.9},
    };
    writeJson(future_layout);

    LayoutState loaded;
    loaded.scene_panel_ratio = 0.25f;
    ASSERT_FALSE(loaded.loadFrom(layout_path_));
    EXPECT_FLOAT_EQ(loaded.scene_panel_ratio, 0.25f);
    EXPECT_EQ(readJson(), future_layout);
}

TEST_F(LayoutStatePersistenceTest, RejectsNonIntegerVersionWithoutMutatingState) {
    using lfs::vis::gui::LayoutState;

    writeJson({
        {"version", "1"},
        {"scene_panel_ratio", 0.9},
    });

    LayoutState loaded;
    loaded.scene_panel_ratio = 0.25f;
    ASSERT_FALSE(loaded.loadFrom(layout_path_));
    EXPECT_FLOAT_EQ(loaded.scene_panel_ratio, 0.25f);
}

TEST_F(LayoutStatePersistenceTest, MissingFileKeepsDefaultsAndAllowsFirstSave) {
    using lfs::vis::gui::LayoutState;

    LayoutState state;
    ASSERT_TRUE(state.loadFrom(layout_path_));
    EXPECT_FLOAT_EQ(state.scene_panel_ratio, 0.4f);
    ASSERT_TRUE(state.saveTo(layout_path_));
    EXPECT_EQ(readJson().at("version"), 1);
}
