/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/rmlui_system_interface.hpp"

#include <gtest/gtest.h>

using lfs::vis::gui::RmlSystemInterface;

TEST(RmlSystemInterfaceTest, TextBelowTheAstralPlaneDoesNotRequestTheEmojiFont) {
    RmlSystemInterface system_interface(nullptr);
    Rml::String translated;

    EXPECT_EQ(system_interface.TranslateString(translated, "Scene 1,024 splats"), 0);
    EXPECT_EQ(translated, "Scene 1,024 splats");
    system_interface.TranslateString(translated, "⚠ Check the dataset ★ ✓");
    system_interface.TranslateString(translated, "プロジェクト 프로젝트");
    EXPECT_FALSE(system_interface.sawAstralText());
}

TEST(RmlSystemInterfaceTest, EmojiTextRequestsTheEmojiFont) {
    RmlSystemInterface system_interface(nullptr);
    Rml::String translated;

    EXPECT_EQ(system_interface.TranslateString(translated, "Tree \U0001F333 scan"), 0);
    EXPECT_EQ(translated, "Tree \U0001F333 scan");
    EXPECT_TRUE(system_interface.sawAstralText());

    system_interface.TranslateString(translated, "plain text");
    EXPECT_TRUE(system_interface.sawAstralText());
}

TEST(RmlSystemInterfaceTest, ShownTextOutsideTranslationRequestsTheEmojiFont) {
    RmlSystemInterface system_interface(nullptr);

    system_interface.noteShownText("A plain description");
    EXPECT_FALSE(system_interface.sawAstralText());
    system_interface.noteShownText("First scan \U0001F680");
    EXPECT_TRUE(system_interface.sawAstralText());
}
