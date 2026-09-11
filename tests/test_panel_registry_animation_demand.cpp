/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include <visualizer/gui/panel_layout.hpp>
#include <visualizer/gui/panel_registry.hpp>

#include <algorithm>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

    class TestPanel final : public lfs::vis::gui::IPanel {
    public:
        explicit TestPanel(bool animation,
                           std::optional<double> scheduled_delay = std::nullopt)
            : animation_(animation),
              scheduled_delay_(scheduled_delay) {}

        void draw(const lfs::vis::gui::PanelDrawContext&) override {}

        bool needsAnimationFrame() const override {
            return animation_;
        }

        std::optional<double> nextScheduledAnimationDelay() const override {
            return scheduled_delay_;
        }

    private:
        bool animation_ = false;
        std::optional<double> scheduled_delay_;
    };

    class ChromePanel final : public lfs::vis::gui::IPanel {
    public:
        void draw(const lfs::vis::gui::PanelDrawContext&) override {}

        std::string captureChromeJson() const override {
            return chrome;
        }

        void applyChromeJson(const std::string_view json) override {
            chrome = json.empty() ? std::string("{}") : std::string(json);
            ++apply_count;
        }

        std::string chrome = R"({"metric_id":"opacity"})";
        int apply_count = 0;
    };

    class RecordingPanel final : public lfs::vis::gui::IPanel {
    public:
        void draw(const lfs::vis::gui::PanelDrawContext&) override {}

        lfs::vis::gui::PanelRenderCapabilities renderCapabilities() const override {
            return {.direct = true};
        }

        lfs::vis::gui::PanelDirectRenderResult renderDirect(
            const lfs::vis::gui::PanelDirectRenderRequest& request,
            const lfs::vis::gui::PanelDrawContext&) override {
            using lfs::vis::gui::PanelDirectRenderMode;
            requests.emplace_back(request.mode, request.space);
            switch (request.mode) {
            case PanelDirectRenderMode::Measure:
                return {.handled = true, .height = height};
            case PanelDirectRenderMode::Draw:
                ++draw_count;
                return {.handled = true, .height = height};
            case PanelDirectRenderMode::Preload:
                ++preload_count;
                return {.handled = true, .height = height};
            case PanelDirectRenderMode::Cached:
                ++cached_count;
                return {.handled = cache_hit, .height = height};
            }
            return {};
        }

        bool poll(const lfs::vis::gui::PanelDrawContext&) override {
            ++poll_count;
            if (poll_action)
                poll_action();
            return poll_result;
        }

        int draw_count = 0;
        int preload_count = 0;
        int cached_count = 0;
        int poll_count = 0;
        float height = 24.0f;
        bool cache_hit = true;
        bool poll_result = true;
        std::function<void()> poll_action;
        std::vector<std::pair<lfs::vis::gui::PanelDirectRenderMode,
                              lfs::vis::gui::PanelSpace>>
            requests;
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
                                  std::string parent_id = {},
                                  std::optional<double> scheduled_delay = std::nullopt) {
            lfs::vis::gui::PanelInfo info;
            info.id = std::move(id);
            info.label = info.id;
            info.space = space;
            info.parent_id = std::move(parent_id);
            info.is_native = false;
            info.panel = std::make_shared<TestPanel>(animation, scheduled_delay);
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

TEST_F(PanelRegistryAnimationDemandTest, LeftDockDemandRespectsLeftDockVisibility) {
    using namespace lfs::vis::gui;

    registerPanel("test.left", PanelSpace::LeftDock, true);

    const auto visible = PanelRegistry::instance().animationDemandForVisiblePanels({
        .active_main_tab = "test.main",
        .ui_visible = true,
        .right_panel_visible = true,
        .bottom_dock_visible = true,
        .left_dock_visible = true,
    });
    EXPECT_TRUE(visible.left_dock);
    EXPECT_TRUE(visible.any());
    EXPECT_TRUE(PanelRegistry::instance().needsAnimationFrameForVisiblePanels({
        .active_main_tab = "test.main",
        .ui_visible = true,
        .right_panel_visible = true,
        .bottom_dock_visible = true,
        .left_dock_visible = true,
    }));

    const auto left_hidden = PanelRegistry::instance().animationDemandForVisiblePanels({
        .active_main_tab = "test.main",
        .ui_visible = true,
        .right_panel_visible = true,
        .bottom_dock_visible = true,
        .left_dock_visible = false,
    });
    EXPECT_FALSE(left_hidden.left_dock);
    EXPECT_FALSE(left_hidden.any());
    EXPECT_FALSE(PanelRegistry::instance().needsAnimationFrameForVisiblePanels({
        .active_main_tab = "test.main",
        .ui_visible = true,
        .right_panel_visible = true,
        .bottom_dock_visible = true,
        .left_dock_visible = false,
    }));

    const auto ui_hidden = PanelRegistry::instance().animationDemandForVisiblePanels({
        .active_main_tab = "test.main",
        .ui_visible = false,
        .right_panel_visible = true,
        .bottom_dock_visible = true,
        .left_dock_visible = true,
    });
    EXPECT_FALSE(ui_hidden.left_dock);
    EXPECT_FALSE(ui_hidden.any());
    EXPECT_FALSE(PanelRegistry::instance().needsAnimationFrameForVisiblePanels({
        .active_main_tab = "test.main",
        .ui_visible = false,
        .right_panel_visible = true,
        .bottom_dock_visible = true,
        .left_dock_visible = true,
    }));
}

TEST_F(PanelRegistryAnimationDemandTest, ScheduledDelaySkipsHiddenLeftDock) {
    using namespace lfs::vis::gui;

    registerPanel("test.left.scheduled", PanelSpace::LeftDock, false, {}, 0.25);

    const auto visible_delay =
        PanelRegistry::instance().nextScheduledAnimationDelayForVisiblePanels({
            .active_main_tab = "test.main",
            .ui_visible = true,
            .right_panel_visible = true,
            .bottom_dock_visible = true,
            .left_dock_visible = true,
        });
    ASSERT_TRUE(visible_delay.has_value());
    EXPECT_NEAR(*visible_delay, 0.25, 1e-9);

    const auto hidden_delay =
        PanelRegistry::instance().nextScheduledAnimationDelayForVisiblePanels({
            .active_main_tab = "test.main",
            .ui_visible = true,
            .right_panel_visible = true,
            .bottom_dock_visible = true,
            .left_dock_visible = false,
        });
    EXPECT_FALSE(hidden_delay.has_value());
}

TEST_F(PanelRegistryAnimationDemandTest, DefaultLeftDockVisibleTrue) {
    using namespace lfs::vis::gui;

    // Callers must set left_dock_visible explicitly; the field defaults to true (#1597).
    const PanelAnimationVisibility v{
        .active_main_tab = "test.main",
        .ui_visible = true,
        .right_panel_visible = true,
        .bottom_dock_visible = true,
    };
    EXPECT_TRUE(v.left_dock_visible);
}

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
TEST_F(PanelRegistryAnimationDemandTest,
       ProjectRectangleSurvivesMachineLocalScaleClamping) {
    using namespace lfs::vis::gui;

    registerRecordingPanel("test.project_rect");
    const PanelProjectState requested{
        .id = "test.project_rect",
        .space = PanelSpace::Floating,
        .float_x = -900.0f,
        .float_y = 1700.0f,
        .float_user_height = 333.0f,
        .float_last_bounds_valid = true,
        .float_last_x = -900.0f,
        .float_last_y = 1700.0f,
        .float_last_w = 640.0f,
        .float_last_h = 333.0f,
        .float_auto_center = false,
        .float_stack_order = 17,
    };
    auto& registry = PanelRegistry::instance();
    registry.apply_project_state({requested});

    // UI scale is user-global. Its live resize must not rewrite the project
    // rectangle merely because this machine applies a different scale/clamp.
    registry.rescale_floating_panels(1.0f, 2.0f);
    const auto captured = registry.capture_project_state();
    const auto found = std::ranges::find(
        captured, requested.id,
        &PanelProjectState::id);
    ASSERT_NE(found, captured.end());
    EXPECT_FLOAT_EQ(found->float_x, requested.float_x);
    EXPECT_FLOAT_EQ(found->float_y, requested.float_y);
    EXPECT_FLOAT_EQ(
        found->float_user_height,
        requested.float_user_height);
    EXPECT_FLOAT_EQ(
        found->float_last_w,
        requested.float_last_w);
    EXPECT_FLOAT_EQ(
        found->float_last_h,
        requested.float_last_h);

    registry.clear_project_state_retention();
    const auto live = registry.capture_project_state();
    const auto live_found = std::ranges::find(
        live, requested.id,
        &PanelProjectState::id);
    ASSERT_NE(live_found, live.end());
    EXPECT_FLOAT_EQ(
        live_found->float_user_height,
        requested.float_user_height * 2.0f);
    EXPECT_FALSE(
        live_found->float_last_bounds_valid);
}

TEST_F(PanelRegistryAnimationDemandTest,
       ApplyProjectStateResetsUnlistedPanelsToRegisterDefaults) {
    using namespace lfs::vis::gui;

    registerRecordingPanel("test.keep");
    registerRecordingPanel("test.drop");
    auto& registry = PanelRegistry::instance();
    registry.set_panel_enabled("test.drop", false);

    const PanelProjectState keep{
        .id = "test.keep",
        .space = PanelSpace::Floating,
        .enabled = true,
        .float_x = 12.0f,
        .float_y = 24.0f,
        .float_auto_center = false,
    };
    registry.apply_project_state({keep});

    const auto kept = registry.get_panel("test.keep");
    auto dropped = registry.get_panel("test.drop");
    ASSERT_TRUE(kept.has_value());
    ASSERT_TRUE(dropped.has_value());
    EXPECT_TRUE(kept->enabled);
    EXPECT_FALSE(dropped->enabled);

    registry.set_panel_enabled("test.drop", true);
    registry.apply_project_state({keep});

    dropped = registry.get_panel("test.drop");
    ASSERT_TRUE(dropped.has_value());
    EXPECT_TRUE(dropped->enabled);
}

TEST_F(PanelRegistryAnimationDemandTest,
       LateRegisterDropsInvalidSavedOverlayAndStillRegisters) {
    using namespace lfs::vis::gui;

    auto& registry = PanelRegistry::instance();
    const PanelProjectState saved{
        .id = "test.corrupt_float",
        .space = PanelSpace::Floating,
        .enabled = false,
        .float_x = 10.0f,
        .float_y = 20.0f,
        .float_auto_center = false,
    };
    registry.apply_project_state({saved});

    PanelInfo declared_invalid_float;
    declared_invalid_float.id = "test.declared_invalid_float";
    declared_invalid_float.label = declared_invalid_float.id;
    declared_invalid_float.space = PanelSpace::Floating;
    declared_invalid_float.is_native = false;
    declared_invalid_float.panel = std::make_shared<TestPanel>(false);
    EXPECT_FALSE(registry.register_panel(
        std::move(declared_invalid_float)));

    PanelInfo info;
    info.id = saved.id;
    info.label = info.id;
    info.space = PanelSpace::MainPanelTab;
    info.enabled = true;
    info.order = 42;
    info.is_native = false;
    info.panel = std::make_shared<TestPanel>(false);
    ASSERT_TRUE(registry.register_panel(std::move(info)));

    const auto registered = registry.get_panel(saved.id);
    ASSERT_TRUE(registered.has_value());
    EXPECT_EQ(registered->space, PanelSpace::MainPanelTab);
    EXPECT_TRUE(registered->enabled);
    EXPECT_EQ(registered->order, 42);

    const auto captured = registry.capture_project_state();
    const auto found = std::ranges::find(
        captured, saved.id, &PanelProjectState::id);
    ASSERT_NE(found, captured.end());
    EXPECT_EQ(found->space, PanelSpace::MainPanelTab);
    EXPECT_TRUE(found->enabled);
}

TEST_F(PanelRegistryAnimationDemandTest,
       LateRegisterAppliesPendingProjectFloatingState) {
    using namespace lfs::vis::gui;

    auto& registry = PanelRegistry::instance();
    const PanelProjectState saved{
        .id = "test.late_float",
        .space = PanelSpace::Floating,
        .enabled = true,
        .float_x = 88.0f,
        .float_y = 99.0f,
        .float_user_height = 140.0f,
        .float_last_bounds_valid = true,
        .float_last_x = 88.0f,
        .float_last_y = 99.0f,
        .float_last_w = 320.0f,
        .float_last_h = 140.0f,
        .float_auto_center = false,
        .float_stack_order = 4,
    };
    registry.apply_project_state({saved});
    registerRecordingPanel("test.late_float");

    const auto captured = registry.capture_project_state();
    const auto found = std::ranges::find(
        captured, saved.id, &PanelProjectState::id);
    ASSERT_NE(found, captured.end());
    EXPECT_TRUE(found->enabled);
    EXPECT_FLOAT_EQ(found->float_x, saved.float_x);
    EXPECT_FLOAT_EQ(found->float_y, saved.float_y);
    EXPECT_FLOAT_EQ(found->float_last_w, saved.float_last_w);
}

TEST_F(PanelRegistryAnimationDemandTest, UnifiedRenderRequestCoversSpacePanelAndChildrenTargets) {
    using namespace lfs::vis::gui;

    auto parent = std::make_shared<RecordingPanel>();
    PanelInfo parent_info;
    parent_info.id = "test.parent";
    parent_info.label = parent_info.id;
    parent_info.space = PanelSpace::MainPanelTab;
    parent_info.is_native = false;
    parent_info.panel = parent;
    ASSERT_TRUE(PanelRegistry::instance().register_panel(std::move(parent_info)));

    auto child = std::make_shared<RecordingPanel>();
    PanelInfo child_info;
    child_info.id = "test.child";
    child_info.label = child_info.id;
    child_info.parent_id = "test.parent";
    child_info.space = PanelSpace::MainPanelTab;
    child_info.is_native = false;
    child_info.panel = child;
    ASSERT_TRUE(PanelRegistry::instance().register_panel(std::move(child_info)));

    PanelDrawContext ctx;
    const float space_height = PanelRegistry::instance().render_panels({
                                                                           .target = PanelRenderTarget::for_space(PanelSpace::MainPanelTab),
                                                                           .mode = PanelRenderMode::Direct,
                                                                           .width = 320.0f,
                                                                           .height = 200.0f,
                                                                       },
                                                                       ctx);
    EXPECT_FLOAT_EQ(space_height, parent->height);
    EXPECT_EQ(parent->draw_count, 1);
    EXPECT_EQ(child->draw_count, 0);

    const float panel_height = PanelRegistry::instance().render_panels({
                                                                           .target = PanelRenderTarget::for_panel("test.parent"),
                                                                           .mode = PanelRenderMode::DirectPreload,
                                                                           .width = 320.0f,
                                                                           .height = 200.0f,
                                                                       },
                                                                       ctx);
    EXPECT_FLOAT_EQ(panel_height, parent->height);
    EXPECT_EQ(parent->preload_count, 1);

    const float children_height = PanelRegistry::instance().render_panels({
                                                                              .target = PanelRenderTarget::for_children("test.parent"),
                                                                              .mode = PanelRenderMode::DirectCached,
                                                                              .width = 320.0f,
                                                                              .height = 200.0f,
                                                                          },
                                                                          ctx);
    EXPECT_FLOAT_EQ(children_height, child->height);
    EXPECT_EQ(child->cached_count, 1);
    EXPECT_EQ(child->draw_count, 0);
}

TEST_F(PanelRegistryAnimationDemandTest, CachedMissPollsAndFallsBackToLiveDraw) {
    using namespace lfs::vis::gui;

    auto panel = std::make_shared<RecordingPanel>();
    panel->cache_hit = false;
    PanelInfo info;
    info.id = "test.cached_miss";
    info.label = info.id;
    info.space = PanelSpace::MainPanelTab;
    info.is_native = false;
    info.panel = panel;
    ASSERT_TRUE(PanelRegistry::instance().register_panel(std::move(info)));

    PanelDrawContext ctx;
    const float height = PanelRegistry::instance().render_panels({
                                                                     .target = PanelRenderTarget::for_panel("test.cached_miss"),
                                                                     .mode = PanelRenderMode::DirectCached,
                                                                     .width = 320.0f,
                                                                     .height = 200.0f,
                                                                 },
                                                                 ctx);

    EXPECT_FLOAT_EQ(height, panel->height);
    EXPECT_EQ(panel->cached_count, 1);
    EXPECT_EQ(panel->poll_count, 1);
    EXPECT_EQ(panel->draw_count, 1);
}

TEST_F(PanelRegistryAnimationDemandTest, FloatingInteractionExistsOnlyWhilePanelIsFloating) {
    using namespace lfs::vis::gui;

    registerRecordingPanel("test.floating");
    const auto floating_before = PanelRegistry::instance().get_panel("test.floating");
    ASSERT_TRUE(floating_before.has_value());
    EXPECT_GT(floating_before->float_stack_order, 0u);

    ASSERT_TRUE(PanelRegistry::instance().set_panel_space(
        "test.floating", PanelSpace::MainPanelTab));
    const auto docked = PanelRegistry::instance().get_panel("test.floating");
    ASSERT_TRUE(docked.has_value());
    EXPECT_EQ(docked->float_stack_order, 0u);

    ASSERT_TRUE(PanelRegistry::instance().set_panel_space(
        "test.floating", PanelSpace::Floating));
    const auto floating_after = PanelRegistry::instance().get_panel("test.floating");
    ASSERT_TRUE(floating_after.has_value());
    EXPECT_GT(floating_after->float_stack_order,
              floating_before->float_stack_order);
}

TEST_F(PanelRegistryAnimationDemandTest, StandardRenderingUsesSnapshotSpace) {
    using namespace lfs::vis::gui;

    auto panel = std::make_shared<RecordingPanel>();
    PanelInfo info;
    info.id = "test.space_snapshot";
    info.label = info.id;
    info.space = PanelSpace::MainPanelTab;
    info.is_native = false;
    info.panel = panel;
    ASSERT_TRUE(PanelRegistry::instance().register_panel(std::move(info)));

    PanelDrawContext ctx;
    PanelRegistry::instance().render_panels({
                                                .target = PanelRenderTarget::for_panel("test.space_snapshot"),
                                                .mode = PanelRenderMode::Standard,
                                            },
                                            ctx);
    ASSERT_FALSE(panel->requests.empty());
    EXPECT_EQ(panel->requests.back().first, PanelDirectRenderMode::Draw);
    EXPECT_EQ(panel->requests.back().second, PanelSpace::MainPanelTab);

    ASSERT_TRUE(PanelRegistry::instance().set_panel_space(
        "test.space_snapshot", PanelSpace::Floating));
    ASSERT_TRUE(PanelRegistry::instance().set_panel_space(
        "test.space_snapshot", PanelSpace::SidePanel));
    PanelRegistry::instance().render_panels({
                                                .target = PanelRenderTarget::for_panel("test.space_snapshot"),
                                                .mode = PanelRenderMode::Standard,
                                            },
                                            ctx);
    EXPECT_EQ(panel->requests.back().second, PanelSpace::SidePanel);
}

TEST_F(PanelRegistryAnimationDemandTest, MeasureReceivesPanelSpace) {
    using namespace lfs::vis::gui;

    auto panel = std::make_shared<RecordingPanel>();
    PanelInfo info;
    info.id = "test.measure_space";
    info.label = info.id;
    info.space = PanelSpace::Floating;
    info.is_native = false;
    info.panel = panel;
    ASSERT_TRUE(PanelRegistry::instance().register_panel(std::move(info)));

    ViewportLayout viewport;
    viewport.size = {1280.0f, 720.0f};
    PanelDrawContext ctx;
    ctx.viewport = &viewport;
    ctx.scene_generation = 1;
    PanelRegistry::instance().render_panels({
                                                .target = PanelRenderTarget::for_space(PanelSpace::Floating),
                                                .mode = PanelRenderMode::Standard,
                                            },
                                            ctx);

    ASSERT_FALSE(panel->requests.empty());
    EXPECT_EQ(panel->requests.front().first, PanelDirectRenderMode::Measure);
    EXPECT_EQ(panel->requests.front().second, PanelSpace::Floating);
}

TEST_F(PanelRegistryAnimationDemandTest, DockDuringFloatingPollSkipsInteractionState) {
    using namespace lfs::vis::gui;

    auto panel = std::make_shared<RecordingPanel>();
    PanelInfo info;
    info.id = "test.dock_during_poll";
    info.label = info.id;
    info.space = PanelSpace::Floating;
    info.is_native = false;
    info.panel = panel;
    ASSERT_TRUE(PanelRegistry::instance().register_panel(std::move(info)));

    ViewportLayout viewport;
    viewport.size = {1280.0f, 720.0f};
    PanelDrawContext ctx;
    ctx.viewport = &viewport;
    PanelRegistry::instance().render_panels({
                                                .target = PanelRenderTarget::for_space(PanelSpace::Floating),
                                                .mode = PanelRenderMode::Standard,
                                            },
                                            ctx);
    EXPECT_TRUE(PanelRegistry::instance().isPositionOverFloatingPanel(640.0, 360.0));

    panel->poll_action = [&] {
        panel->poll_action = {};
        EXPECT_TRUE(PanelRegistry::instance().set_panel_space(
            "test.dock_during_poll", PanelSpace::MainPanelTab));
    };
    ctx.scene_generation = 2;
    PanelRegistry::instance().render_panels({
                                                .target = PanelRenderTarget::for_space(PanelSpace::Floating),
                                                .mode = PanelRenderMode::Standard,
                                            },
                                            ctx);

    EXPECT_FALSE(PanelRegistry::instance().isPositionOverFloatingPanel(640.0, 360.0));
    const auto panel_details = PanelRegistry::instance().get_panel("test.dock_during_poll");
    ASSERT_TRUE(panel_details.has_value());
    EXPECT_EQ(panel_details->space, PanelSpace::MainPanelTab);
    EXPECT_EQ(panel_details->float_stack_order, 0u);
}

TEST_F(PanelRegistryAnimationDemandTest, PanelPayloadRoundTripAndReset) {
    using namespace lfs::vis::gui;

    auto panel = std::make_shared<ChromePanel>();
    panel->chrome = R"({"metric_id":"scale","log_scale":true,"bin_count":64})";
    PanelInfo info;
    info.id = "lfs.histogram";
    info.label = info.id;
    info.space = PanelSpace::BottomDock;
    info.is_native = false;
    info.panel = panel;
    ASSERT_TRUE(PanelRegistry::instance().register_panel(std::move(info)));

    const auto captured =
        PanelRegistry::instance().capture_panel_payloads();
    ASSERT_EQ(captured.size(), 1u);
    ASSERT_TRUE(captured.contains("lfs.histogram"));
    EXPECT_NE(captured.at("lfs.histogram").find("\"metric_id\""), std::string::npos);
    EXPECT_NE(captured.at("lfs.histogram").find("scale"), std::string::npos);

    panel->chrome = R"({"metric_id":"opacity"})";
    PanelRegistry::instance().apply_panel_payloads(captured);
    EXPECT_EQ(panel->chrome, captured.at("lfs.histogram"));

    PanelRegistry::instance().apply_panel_payloads({});
    EXPECT_EQ(panel->chrome, "{}");
}

TEST_F(PanelRegistryAnimationDemandTest, DefaultClosedAppliesToDockedPanelsAndReset) {
    using namespace lfs::vis::gui;

    PanelInfo info;
    info.id = "lfs.asset_manager";
    info.label = info.id;
    info.space = PanelSpace::LeftDock;
    info.options = static_cast<uint32_t>(PanelOption::DEFAULT_CLOSED);
    info.is_native = false;
    info.panel = std::make_shared<TestPanel>(false);
    ASSERT_TRUE(PanelRegistry::instance().register_panel(std::move(info)));
    EXPECT_FALSE(PanelRegistry::instance().is_panel_enabled("lfs.asset_manager"));

    PanelRegistry::instance().set_panel_enabled("lfs.asset_manager", true);
    ASSERT_TRUE(PanelRegistry::instance().is_panel_enabled("lfs.asset_manager"));
    PanelRegistry::instance().reset_project_state();
    EXPECT_FALSE(PanelRegistry::instance().is_panel_enabled("lfs.asset_manager"));
}

TEST_F(PanelRegistryAnimationDemandTest, LateRegisterAppliesPendingPanelPayload) {
    using namespace lfs::vis::gui;

    std::unordered_map<std::string, std::string> payloads;
    payloads.emplace(
        "lfs.histogram",
        R"({"metric_id":"opacity","compare_metric_id":"scale","log_scale":true})");
    PanelRegistry::instance().apply_panel_payloads(payloads);

    auto panel = std::make_shared<ChromePanel>();
    panel->chrome = "{}";
    PanelInfo info;
    info.id = "lfs.histogram";
    info.label = info.id;
    info.space = PanelSpace::BottomDock;
    info.is_native = false;
    info.panel = panel;
    ASSERT_TRUE(PanelRegistry::instance().register_panel(std::move(info)));

    EXPECT_EQ(panel->apply_count, 1);
    EXPECT_NE(panel->chrome.find("opacity"), std::string::npos);
    EXPECT_NE(panel->chrome.find("log_scale"), std::string::npos);
}
