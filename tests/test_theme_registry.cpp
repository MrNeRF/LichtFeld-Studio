/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/theme/theme.hpp"

#include <gtest/gtest.h>

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
