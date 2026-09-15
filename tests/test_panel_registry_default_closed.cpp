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

namespace {
    class AreaFactoryPanel final : public lfs::vis::gui::IPanel {
    public:
        std::string state;
        void draw(const lfs::vis::gui::PanelDrawContext&) override {}
        bool supportsAreaInstances() const override { return true; }
        std::shared_ptr<IPanel> createAreaInstance(std::string_view instance_id) const override {
            // A plugin constructor may inspect its own registration. This must
            // not deadlock against create_area_instance's registry lookup.
            EXPECT_TRUE(lfs::vis::gui::PanelRegistry::instance().get_panel("test.area_factory"));
            auto instance = std::make_shared<AreaFactoryPanel>();
            instance->state = instance_id;
            return instance;
        }
    };
} // namespace

TEST_F(PanelRegistryDefaultClosedTest, AreaInstancesOwnStateAndMayReenterRegistry) {
    using namespace lfs::vis::gui;
    auto prototype = std::make_shared<AreaFactoryPanel>();
    PanelInfo info;
    info.id = "test.area_factory";
    info.label = "Area factory";
    info.space = lfs::vis::gui::PanelSpace::MainPanelTab;
    info.is_native = false;
    info.panel = prototype;
    auto& registry = PanelRegistry::instance();
    ASSERT_TRUE(registry.register_panel(std::move(info)));
    const auto first = std::dynamic_pointer_cast<AreaFactoryPanel>(
        registry.create_area_instance("test.area_factory", "area.1"));
    const auto second = std::dynamic_pointer_cast<AreaFactoryPanel>(
        registry.create_area_instance("test.area_factory", "area.2"));
    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    EXPECT_NE(first, second);
    EXPECT_NE(first, prototype);
    first->state = "scrolled";
    EXPECT_EQ(second->state, "area.2");
    EXPECT_TRUE(prototype->state.empty());
    registry.unregister_panel("test.area_factory");
    EXPECT_EQ(first->state, "scrolled");
    EXPECT_FALSE(registry.create_area_instance("test.area_factory", "area.3"));
}
