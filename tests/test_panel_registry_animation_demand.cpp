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
        explicit TestPanel(bool animation) : animation_(animation) {}

        void draw(const lfs::vis::gui::PanelDrawContext&) override {}

        bool needsAnimationFrame() const override {
            return animation_;
        }

    private:
        bool animation_ = false;
    };

    class RecordingPanel final : public lfs::vis::gui::IPanel {
    public:
        void draw(const lfs::vis::gui::PanelDrawContext&) override {}

        bool supportsDirectDraw() const override { return true; }

        void drawDirect(float, float, float, float,
                        const lfs::vis::gui::PanelDrawContext&) override {}
    };

    class PanelRegistryAnimationDemandTest : public ::testing::Test {
    protected:
        void SetUp() override {
            lfs::vis::gui::PanelRegistry::instance().unregister_all_non_native();
        }

        void TearDown() override {
            lfs::vis::gui::PanelRegistry::instance().unregister_all_non_native();
        }

        static void registerPanel(std::string id,
                                  lfs::vis::gui::PanelSpace space,
                                  bool animation,
                                  std::string parent_id = {}) {
            lfs::vis::gui::PanelInfo info;
            info.id = std::move(id);
            info.label = info.id;
            info.space = space;
            info.parent_id = std::move(parent_id);
            info.is_native = false;
            info.panel = std::make_shared<TestPanel>(animation);
            ASSERT_TRUE(lfs::vis::gui::PanelRegistry::instance().register_panel(std::move(info)));
        }

        static void registerRecordingPanel(std::string id) {
            lfs::vis::gui::PanelInfo info;
            info.id = std::move(id);
            info.label = info.id;
            info.space = lfs::vis::gui::PanelSpace::Floating;
            info.is_native = false;
            info.panel = std::make_shared<RecordingPanel>();
            ASSERT_TRUE(lfs::vis::gui::PanelRegistry::instance().register_panel(std::move(info)));
        }
    };

} // namespace

TEST_F(PanelRegistryAnimationDemandTest, ViewportOverlayAnimationDoesNotMarkRightPanel) {
    using namespace lfs::vis::gui;

    registerPanel("test.viewport_overlay", PanelSpace::ViewportOverlay, true);

    const auto demand = PanelRegistry::instance().animationDemandForVisiblePanels({
        .active_main_tab = "test.main",
        .ui_visible = true,
        .right_panel_visible = true,
        .bottom_dock_visible = true,
    });

    EXPECT_TRUE(demand.any());
    EXPECT_TRUE(demand.viewport_overlay);
    EXPECT_FALSE(demand.rightPanel());
    EXPECT_FALSE(demand.main_panel_tab);
    EXPECT_FALSE(demand.scene_header);
}

TEST_F(PanelRegistryAnimationDemandTest, RightPanelDemandOnlyTracksVisibleRightPanelPanels) {
    using namespace lfs::vis::gui;

    registerPanel("test.scene_header", PanelSpace::SceneHeader, true);
    registerPanel("test.main.active", PanelSpace::MainPanelTab, true);
    registerPanel("test.main.inactive", PanelSpace::MainPanelTab, true);
    registerPanel("test.child.active", PanelSpace::MainPanelTab, true, "test.main.active");
    registerPanel("test.bottom", PanelSpace::BottomDock, true);

    const auto visible = PanelRegistry::instance().animationDemandForVisiblePanels({
        .active_main_tab = "test.main.active",
        .ui_visible = true,
        .right_panel_visible = true,
        .bottom_dock_visible = true,
    });
    EXPECT_TRUE(visible.rightPanel());
    EXPECT_TRUE(visible.scene_header);
    EXPECT_TRUE(visible.main_panel_tab);
    EXPECT_TRUE(visible.bottom_dock);

    const auto right_hidden = PanelRegistry::instance().animationDemandForVisiblePanels({
        .active_main_tab = "test.main.active",
        .ui_visible = true,
        .right_panel_visible = false,
        .bottom_dock_visible = true,
    });
    EXPECT_FALSE(right_hidden.rightPanel());
    EXPECT_FALSE(right_hidden.scene_header);
    EXPECT_FALSE(right_hidden.main_panel_tab);
    EXPECT_TRUE(right_hidden.bottom_dock);

    const auto bottom_hidden = PanelRegistry::instance().animationDemandForVisiblePanels({
        .active_main_tab = "test.main.active",
        .ui_visible = true,
        .right_panel_visible = true,
        .bottom_dock_visible = false,
    });
    EXPECT_TRUE(bottom_hidden.rightPanel());
    EXPECT_FALSE(bottom_hidden.bottom_dock);
}

TEST_F(PanelRegistryAnimationDemandTest, BringPanelToFrontRaisesEnabledFloatingPanel) {
    using namespace lfs::vis::gui;

    registerRecordingPanel("test.first");
    registerRecordingPanel("test.second");

    const auto first_before = PanelRegistry::instance().get_panel("test.first");
    const auto second_before = PanelRegistry::instance().get_panel("test.second");
    ASSERT_TRUE(first_before.has_value());
    ASSERT_TRUE(second_before.has_value());
    EXPECT_LT(first_before->float_stack_order, second_before->float_stack_order);

    EXPECT_TRUE(PanelRegistry::instance().bring_panel_to_front("test.first"));

    const auto first_after = PanelRegistry::instance().get_panel("test.first");
    const auto second_after = PanelRegistry::instance().get_panel("test.second");
    ASSERT_TRUE(first_after.has_value());
    ASSERT_TRUE(second_after.has_value());
    EXPECT_GT(first_after->float_stack_order, second_after->float_stack_order);
}

TEST_F(PanelRegistryAnimationDemandTest, BringPanelToFrontIgnoresDisabledFloatingPanel) {
    using namespace lfs::vis::gui;

    registerRecordingPanel("test.first");
    registerRecordingPanel("test.second");
    PanelRegistry::instance().set_panel_enabled("test.first", false);

    EXPECT_FALSE(PanelRegistry::instance().bring_panel_to_front("test.first"));

    auto panels = PanelRegistry::instance().get_panels_for_space(PanelSpace::Floating);
    ASSERT_EQ(panels.size(), 1u);
    EXPECT_EQ(panels.front().id, "test.second");
}
