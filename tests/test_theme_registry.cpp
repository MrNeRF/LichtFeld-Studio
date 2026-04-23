/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/theme/theme.hpp"
#include "visualizer/internal/resource_paths.hpp"
#include "core/path_utils.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

    std::vector<lfs::vis::ThemePresetInfo> themePresetInfos() {
        std::vector<lfs::vis::ThemePresetInfo> infos;
        lfs::vis::visitThemePresetInfos([&infos](const lfs::vis::ThemePresetInfo& info) {
            infos.push_back(info);
        });
        return infos;
    }

} // namespace

TEST(ThemeRegistry, ManifestOwnsCatalogMetadata) {
    const auto manifest_path = lfs::vis::getAssetPath("themes/manifest.json");

    std::ifstream manifest_file;
    ASSERT_TRUE(lfs::core::open_file_for_read(manifest_path, manifest_file));

    nlohmann::json manifest;
    manifest_file >> manifest;

    ASSERT_EQ(manifest.value("schema_version", 0), 1);
    ASSERT_TRUE(manifest.contains("themes"));
    ASSERT_TRUE(manifest["themes"].is_array());

    const auto infos = themePresetInfos();
    std::map<std::string, lfs::vis::ThemePresetInfo> info_by_id;
    for (const auto& info : infos) {
        info_by_id.emplace(info.id, info);
    }

    ASSERT_EQ(info_by_id.size(), manifest["themes"].size());

    for (const auto& entry : manifest["themes"]) {
        ASSERT_TRUE(entry.is_object());
        ASSERT_TRUE(entry.contains("id"));
        ASSERT_TRUE(entry.contains("file"));
        ASSERT_TRUE(entry.contains("fallback"));
        ASSERT_TRUE(entry.contains("label_key"));
        ASSERT_TRUE(entry.contains("mode"));
        ASSERT_TRUE(entry.contains("order"));

        const std::string id = entry["id"].get<std::string>();
        ASSERT_TRUE(info_by_id.contains(id)) << id;

        const auto& info = info_by_id.at(id);
        EXPECT_EQ(info.label_key, entry["label_key"].get<std::string>()) << id;
        EXPECT_EQ(info.mode, entry["mode"].get<std::string>()) << id;
        EXPECT_EQ(info.order, entry["order"].get<int>()) << id;

        const std::string theme_file = entry["file"].get<std::string>();
        const auto theme_path = lfs::vis::getAssetPath("themes/" + theme_file);

        std::ifstream theme_stream;
        ASSERT_TRUE(lfs::core::open_file_for_read(theme_path, theme_stream)) << theme_file;

        nlohmann::json theme;
        theme_stream >> theme;
        EXPECT_FALSE(theme.contains("id")) << theme_file;
        EXPECT_FALSE(theme.contains("label_key")) << theme_file;
        EXPECT_FALSE(theme.contains("mode")) << theme_file;
        EXPECT_FALSE(theme.contains("order")) << theme_file;
    }
}

TEST(ThemeRegistry, CatalogIsStableAndSelfDescribing) {
    const auto infos = themePresetInfos();

    ASSERT_EQ(infos.size(), 6u);

    int previous_order = 0;
    std::set<std::string> ids;
    for (const auto& info : infos) {
        EXPECT_FALSE(info.id.empty());
        EXPECT_FALSE(info.name.empty()) << info.id;
        EXPECT_FALSE(info.label_key.empty()) << info.id;
        EXPECT_TRUE(info.mode == "dark" || info.mode == "light") << info.id;
        EXPECT_GT(info.order, previous_order) << info.id;
        EXPECT_TRUE(ids.insert(info.id).second) << info.id;
        previous_order = info.order;
    }

    EXPECT_TRUE(ids.contains("dark"));
    EXPECT_TRUE(ids.contains("light"));
    EXPECT_TRUE(ids.contains("gruvbox"));
    EXPECT_TRUE(ids.contains("catppuccin_mocha"));
    EXPECT_TRUE(ids.contains("catppuccin_latte"));
    EXPECT_TRUE(ids.contains("nord"));
}

TEST(ThemeRegistry, CurrentThemeUsesStablePresetId) {
    const std::string original_theme = lfs::vis::currentThemeId();

    ASSERT_TRUE(lfs::vis::setThemeByName("Catppuccin Mocha"));
    EXPECT_EQ(lfs::vis::currentThemeId(), "catppuccin_mocha");
    EXPECT_EQ(lfs::vis::theme().name, "Catppuccin Mocha");

    ASSERT_TRUE(lfs::vis::setThemeByName("catppuccin-latte"));
    EXPECT_EQ(lfs::vis::currentThemeId(), "catppuccin_latte");
    EXPECT_EQ(lfs::vis::theme().name, "Catppuccin Latte");

    if (!original_theme.empty()) {
        EXPECT_TRUE(lfs::vis::setThemeByName(original_theme));
    }
}
