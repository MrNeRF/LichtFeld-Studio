/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include <visualizer/gui/panel_registry.hpp>

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
                                  bool enabled = true) {
            lfs::vis::gui::PanelInfo info;
            info.id = std::move(id);
            info.label = info.id;
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
