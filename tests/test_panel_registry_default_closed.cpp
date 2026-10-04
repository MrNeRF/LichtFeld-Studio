/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/localization_manager.hpp"

#include <gtest/gtest.h>

#include <visualizer/gui/panel_registry.hpp>

#include <filesystem>
#include <memory>
#include <string>
#include <utility>

namespace {

    class TestPanel final : public lfs::vis::gui::IPanel {
    public:
        void draw(const lfs::vis::gui::PanelDrawContext&) override {}

        lfs::vis::gui::PanelRenderCapabilities renderCapabilities() const override {
            return {.direct = true};
        }

        lfs::vis::gui::PanelDirectRenderResult renderDirect(
            const lfs::vis::gui::PanelDirectRenderRequest&,
            const lfs::vis::gui::PanelDrawContext&) override {
            return {.handled = true};
        }
    };

    class PanelRegistryDefaultClosedTest : public ::testing::Test {
    protected:
        void SetUp() override {
            lfs::vis::gui::PanelRegistry::instance().unregister_all_non_native();
        }

        void TearDown() override {
            lfs::vis::gui::PanelRegistry::instance().unregister_all_non_native();
        }

        static void registerPanel(std::string id,
                                  lfs::vis::gui::PanelSpace space,
                                  uint32_t options = 0,
                                  bool enabled = true,
                                  std::string label = {}) {
            lfs::vis::gui::PanelInfo info;
            info.id = std::move(id);
            info.label = label.empty() ? info.id : std::move(label);
            info.space = space;
            info.options = options;
            info.enabled = enabled;
            info.is_native = false;
            info.panel = std::make_shared<TestPanel>();
            ASSERT_TRUE(lfs::vis::gui::PanelRegistry::instance().register_panel(
                std::move(info)));
        }
    };

} // namespace

TEST_F(PanelRegistryDefaultClosedTest,
       FloatingDefaultClosedDisablesAndSurvivesEmptyProjectReset) {
    using namespace lfs::vis::gui;

    registerPanel("test.default_closed",
                  PanelSpace::Floating,
                  static_cast<uint32_t>(PanelOption::DEFAULT_CLOSED));

    const auto registered =
        PanelRegistry::instance().get_panel("test.default_closed");
    ASSERT_TRUE(registered.has_value());
    EXPECT_FALSE(registered->enabled);
    EXPECT_FALSE(PanelRegistry::instance().is_panel_enabled("test.default_closed"));

    PanelRegistry::instance().apply_project_state({});

    const auto after_reset =
        PanelRegistry::instance().get_panel("test.default_closed");
    ASSERT_TRUE(after_reset.has_value());
    EXPECT_FALSE(after_reset->enabled);
    EXPECT_FALSE(PanelRegistry::instance().is_panel_enabled("test.default_closed"));
}

TEST_F(PanelRegistryDefaultClosedTest,
       FloatingWithoutDefaultClosedStaysEnabledAfterEmptyProjectReset) {
    using namespace lfs::vis::gui;

    registerPanel("test.default_open", PanelSpace::Floating);

    const auto registered =
        PanelRegistry::instance().get_panel("test.default_open");
    ASSERT_TRUE(registered.has_value());
    EXPECT_TRUE(registered->enabled);
    EXPECT_TRUE(PanelRegistry::instance().is_panel_enabled("test.default_open"));

    PanelRegistry::instance().apply_project_state({});

    const auto after_reset =
        PanelRegistry::instance().get_panel("test.default_open");
    ASSERT_TRUE(after_reset.has_value());
    EXPECT_TRUE(after_reset->enabled);
    EXPECT_TRUE(PanelRegistry::instance().is_panel_enabled("test.default_open"));
}

// Fails if a panel labelled with a localization key keeps the text of the language it was registered in (which
// left hidden tabs in the previous language) or ignores a runtime override, or if a literal label is rewritten.
TEST_F(PanelRegistryDefaultClosedTest, KeyLabelsFollowTheLanguage) {
    using namespace lfs::vis::gui;
    auto& locale = lfs::event::LocalizationManager::getInstance();
    ASSERT_TRUE(locale.initialize(
        (std::filesystem::path(PROJECT_ROOT_PATH) / "src/visualizer/gui/resources/locales").string()));
    ASSERT_TRUE(locale.setLanguage("en"));
    registerPanel("test.keyed", PanelSpace::MainPanelTab, 0, true, "window.training");
    registerPanel("test.literal", PanelSpace::MainPanelTab, 0, true, "Training notes");
    const auto label = [](const char* id) { return PanelRegistry::instance().get_panel(id)->label; };
    EXPECT_EQ(label("test.keyed"), std::string(locale.get("window.training")));

    ASSERT_TRUE(locale.setLanguage("zh"));
    const std::string translated = locale.get("window.training");
    ASSERT_NE(translated, "Training");
    PanelRegistry::instance().refresh_localized_labels();
    EXPECT_EQ(label("test.keyed"), translated);
    EXPECT_EQ(label("test.literal"), "Training notes");

    locale.setOverride("window.training", "Preview");
    PanelRegistry::instance().refresh_localized_labels();
    EXPECT_EQ(label("test.keyed"), "Preview");
    locale.clearOverride("window.training");
    PanelRegistry::instance().refresh_localized_labels();
    EXPECT_EQ(label("test.keyed"), translated);
    EXPECT_TRUE(locale.setLanguage("en"));
}
