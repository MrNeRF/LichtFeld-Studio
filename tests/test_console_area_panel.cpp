/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/console_area_panel.hpp"

#include <gtest/gtest.h>

namespace {

    using lfs::vis::gui::PythonConsoleAreaPanel;

    TEST(ConsoleAreaPanel, FactoryOwnsIndependentChromeState) {
        auto prototype = std::make_shared<PythonConsoleAreaPanel>(nullptr, "prototype");
        ASSERT_TRUE(prototype->supportsAreaInstances());

        auto first = prototype->createAreaInstance("area:first");
        auto second = prototype->createAreaInstance("area:second");
        ASSERT_NE(first, nullptr);
        ASSERT_NE(second, nullptr);
        EXPECT_NE(first.get(), second.get());

        first->applyChromeJson(R"({"active_tab":2,"splitter_ratio":0.3})");
        EXPECT_NE(first->captureChromeJson(), second->captureChromeJson());
        EXPECT_NE(first->captureChromeJson().find("\"active_tab\":2"), std::string::npos);
        EXPECT_NE(second->captureChromeJson().find("\"active_tab\":-1"), std::string::npos);
    }

    TEST(ConsoleAreaPanel, NullContextDoesNotClaimDraw) {
        PythonConsoleAreaPanel panel(nullptr);
        const lfs::vis::gui::PanelDirectRenderRequest request{
            .mode = lfs::vis::gui::PanelDirectRenderMode::Draw,
            .width = 320.0f,
            .height = 200.0f};
        EXPECT_FALSE(panel.renderDirect(request, {}).handled);
    }

} // namespace
