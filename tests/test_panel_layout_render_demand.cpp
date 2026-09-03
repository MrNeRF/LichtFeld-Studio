/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include <python/python_runtime.hpp>
#include <visualizer/app_store.hpp>
#include <visualizer/gui/panel_layout.hpp>
#include <visualizer/gui/panel_registry.hpp>

#include <memory>
#include <string>
#include <unordered_map>

namespace {

    class CountingDirectPanel final : public lfs::vis::gui::IPanel {
    public:
        explicit CountingDirectPanel(float height, bool poll_result = true)
            : height_(height), poll_result_(poll_result) {}

        void draw(const lfs::vis::gui::PanelDrawContext&) override {}

        bool poll(const lfs::vis::gui::PanelDrawContext&) override { return poll_result_; }

        void setPollResult(bool value) { poll_result_ = value; }

        lfs::vis::gui::PanelRenderCapabilities renderCapabilities() const override {
            return {.direct = true};
        }

        lfs::vis::gui::PanelDirectRenderResult renderDirect(
            const lfs::vis::gui::PanelDirectRenderRequest& request,
            const lfs::vis::gui::PanelDrawContext&) override {
            using lfs::vis::gui::PanelDirectRenderMode;
            switch (request.mode) {
            case PanelDirectRenderMode::Measure:
                break;
            case PanelDirectRenderMode::Preload:
                ++preload_count;
                break;
            case PanelDirectRenderMode::Draw:
                ++draw_count;
                last_draw_x = request.x;
                last_draw_y = request.y;
                last_draw_width = request.width;
                last_draw_height = request.height;
                break;
            case PanelDirectRenderMode::Cached:
                ++cached_draw_count;
                last_cached_x = request.x;
                last_cached_width = request.width;
                break;
            }
            return {.handled = true, .height = height_};
        }

        int preload_count = 0;
        int draw_count = 0;
        int cached_draw_count = 0;
        float last_draw_x = 0.0f;
        float last_draw_y = 0.0f;
        float last_draw_width = 0.0f;
        float last_draw_height = 0.0f;
        float last_cached_x = 0.0f;
        float last_cached_width = 0.0f;
        bool poll_result_ = true;

    private:
        float height_ = 0.0f;
    };

    class PollGatedUpdatePanel final : public lfs::vis::gui::IPanel {
    public:
        void draw(const lfs::vis::gui::PanelDrawContext&) override {}

        lfs::vis::gui::PanelRenderCapabilities renderCapabilities() const override {
            return {.direct = true};
        }

        lfs::vis::gui::PanelDirectRenderResult renderDirect(
            const lfs::vis::gui::PanelDirectRenderRequest& request,
            const lfs::vis::gui::PanelDrawContext&) override {
            if (request.mode == lfs::vis::gui::PanelDirectRenderMode::Draw)
                pending_update = false;
            return {.handled = true, .height = 24.0f};
        }

        bool poll(const lfs::vis::gui::PanelDrawContext&) override {
            ++poll_count;
            return poll_result;
        }

        void setPollVisibility(const bool visible) override {
            poll_visible = visible;
        }

        bool isVisibleForAnimation() const override {
            return poll_visible;
        }

        bool needsAnimationFrame() const override {
            return pending_update;
        }

        void setPollResult(const bool result) { poll_result = result; }

        bool poll_result = false;
        bool poll_visible = true;
        bool pending_update = true;
        int poll_count = 0;
    };

    class PanelLayoutRenderDemandTest : public ::testing::Test {
    protected:
        void SetUp() override {
            lfs::vis::gui::PanelRegistry::instance().unregister_all_non_native();
        }

        void TearDown() override {
            lfs::vis::gui::PanelRegistry::instance().unregister_all_non_native();
        }

        static std::shared_ptr<CountingDirectPanel> registerPanel(
            std::string id,
            lfs::vis::gui::PanelSpace space,
            float height,
            uint32_t options = 0,
            float initial_width = 0.0f,
            float initial_height = 0.0f,
            bool enabled = true,
            bool poll_result = true,
            bool is_native = false) {
            auto panel = std::make_shared<CountingDirectPanel>(height, poll_result);
            lfs::vis::gui::PanelInfo info;
            info.id = std::move(id);
            info.label = info.id;
            info.space = space;
            info.options = options;
            info.initial_width = initial_width;
            info.initial_height = initial_height;
            info.is_native = is_native;
            info.enabled = enabled;
            info.panel = panel;
            EXPECT_TRUE(lfs::vis::gui::PanelRegistry::instance().register_panel(std::move(info)));
            return panel;
        }

        static lfs::vis::gui::ScreenState screen() {
            return {
                .work_pos = {0.0f, 0.0f},
                .work_size = {1280.0f, 720.0f},
                .any_item_active = false,
            };
        }
    };

} // namespace

TEST_F(PanelLayoutRenderDemandTest, BottomDockDrawsOnlyActivePanelAtFullContentHeight) {
    using namespace lfs::vis::gui;

    auto first = registerPanel("test.bottom.draw.first", PanelSpace::BottomDock, 20.0f);
    auto second = registerPanel("test.bottom.draw.second", PanelSpace::BottomDock, 20.0f);
    PanelLayoutManager layout;
    UIContext ui;
    PanelDrawContext draw_ctx;
    draw_ctx.ui = &ui;
    PanelInputState input;

    layout.renderBottomDock(draw_ctx, true, false, input, screen());

    ASSERT_TRUE(layout.isBottomDockVisible());
    ASSERT_EQ(layout.bottomDockTabs().size(), 2);
    EXPECT_EQ(layout.getBottomDockActiveTab(), "test.bottom.draw.first");
    EXPECT_EQ(first->draw_count, 1);
    EXPECT_EQ(second->draw_count, 0);
    const auto bar = layout.bottomDockTabBarRect();
    const float dpi = lfs::python::get_shared_dpi_scale();
    EXPECT_FLOAT_EQ(bar.y, layout.bottomDockTopY() + PanelLayoutManager::DOCK_GRIP_H * dpi);
    EXPECT_FLOAT_EQ(first->last_draw_y, bar.y + bar.height);
    EXPECT_FLOAT_EQ(first->last_draw_y,
                    layout.bottomDockTopY() +
                        (PanelLayoutManager::DOCK_GRIP_H + PanelLayoutManager::TAB_BAR_H + 1.0f) * dpi);
    EXPECT_FLOAT_EQ(first->last_draw_height,
                    layout.getBottomDockHeight() - bar.height - PanelLayoutManager::DOCK_GRIP_H * dpi);
}

TEST_F(PanelLayoutRenderDemandTest, BottomDockEnablingPanelActivatesIt) {
    using namespace lfs::vis::gui;

    auto first = registerPanel("test.bottom.enable.first", PanelSpace::BottomDock, 20.0f);
    auto second = registerPanel("test.bottom.enable.second", PanelSpace::BottomDock, 20.0f,
                                0, 0.0f, 0.0f, false);
    PanelLayoutManager layout;
    UIContext ui;
    PanelDrawContext draw_ctx{.ui = &ui};
    PanelInputState input;

    layout.renderBottomDock(draw_ctx, true, false, input, screen());
    PanelRegistry::instance().set_panel_enabled("test.bottom.enable.second", true);
    layout.renderBottomDock(draw_ctx, true, false, input, screen());

    EXPECT_EQ(layout.getBottomDockActiveTab(), "test.bottom.enable.second");
    EXPECT_EQ(first->draw_count, 1);
    EXPECT_EQ(second->draw_count, 1);
}

TEST_F(PanelLayoutRenderDemandTest, BottomDockClosingActivePanelFallsBackToFirst) {
    using namespace lfs::vis::gui;

    auto first = registerPanel("test.bottom.close.first", PanelSpace::BottomDock, 20.0f);
    auto second = registerPanel("test.bottom.close.second", PanelSpace::BottomDock, 20.0f);
    PanelLayoutManager layout;
    layout.setBottomDockActiveTab("test.bottom.close.second");
    UIContext ui;
    PanelDrawContext draw_ctx{.ui = &ui};
    PanelInputState input;

    layout.renderBottomDock(draw_ctx, true, false, input, screen());
    PanelRegistry::instance().set_panel_enabled("test.bottom.close.second", false);
    layout.renderBottomDock(draw_ctx, true, false, input, screen());

    EXPECT_EQ(layout.getBottomDockActiveTab(), "test.bottom.close.first");
    EXPECT_EQ(first->draw_count, 1);
    EXPECT_EQ(second->draw_count, 1);
}

TEST_F(PanelLayoutRenderDemandTest, PollFalsePanelIsNotVisibleAndPollFlipDoesNotActivateIt) {
    using namespace lfs::vis::gui;

    auto first = registerPanel("test.bottom.poll.first", PanelSpace::BottomDock, 20.0f);
    auto hidden = registerPanel("test.bottom.poll.hidden", PanelSpace::BottomDock, 20.0f,
                                0, 0.0f, 0.0f, true, false, false);
    PanelLayoutManager layout;
    UIContext ui;
    PanelDrawContext draw_ctx{.ui = &ui};
    PanelInputState input;

    layout.renderBottomDock(draw_ctx, true, false, input, screen());
    ASSERT_EQ(layout.bottomDockTabs().size(), 1);
    EXPECT_EQ(layout.getBottomDockActiveTab(), "test.bottom.poll.first");
    hidden->setPollResult(true);
    ++draw_ctx.scene_generation;
    layout.renderBottomDock(draw_ctx, true, false, input, screen());

    ASSERT_EQ(layout.bottomDockTabs().size(), 2);
    EXPECT_EQ(layout.getBottomDockActiveTab(), "test.bottom.poll.first");
    EXPECT_EQ(first->draw_count, 2);
    EXPECT_EQ(hidden->draw_count, 0);
}

TEST_F(PanelLayoutRenderDemandTest, ApplyProjectStateRestoresBottomDockActiveTab) {
    using namespace lfs::vis::gui;

    auto first = registerPanel("test.bottom.restore.first", PanelSpace::BottomDock, 20.0f);
    auto second = registerPanel("test.bottom.restore.second", PanelSpace::BottomDock, 20.0f);
    PanelLayoutManager layout;
    PanelLayoutProjectState state;
    state.bottom_dock_active_tab_id = "test.bottom.restore.second";
    layout.applyProjectState(state);
    UIContext ui;
    PanelDrawContext draw_ctx{.ui = &ui};
    PanelInputState input;

    layout.renderBottomDock(draw_ctx, true, false, input, screen());

    EXPECT_EQ(layout.getBottomDockActiveTab(), "test.bottom.restore.second");
    EXPECT_EQ(first->draw_count, 0);
    EXPECT_EQ(second->draw_count, 1);
}

TEST_F(PanelLayoutRenderDemandTest, BottomDockCachedDrawsOnlyLastActivePanel) {
    using namespace lfs::vis::gui;

    auto first = registerPanel("test.bottom.cached.first", PanelSpace::BottomDock, 20.0f);
    auto second = registerPanel("test.bottom.cached.second", PanelSpace::BottomDock, 20.0f);
    PanelLayoutManager layout;
    layout.setBottomDockActiveTab("test.bottom.cached.second");
    UIContext ui;
    PanelDrawContext draw_ctx{.ui = &ui};
    PanelInputState input;

    layout.renderBottomDock(draw_ctx, true, false, input, screen());
    first->draw_count = second->draw_count = 0;
    layout.renderBottomDockCached(draw_ctx, true, false, input, screen());

    EXPECT_EQ(first->cached_draw_count, 0);
    EXPECT_EQ(second->cached_draw_count, 1);
}

TEST_F(PanelLayoutRenderDemandTest, CanCacheSceneHeaderWhileDrawingActiveTabLive) {
    using namespace lfs::vis::gui;

    auto scene_header = registerPanel("test.scene", PanelSpace::SceneHeader, 120.0f);
    auto active_tab = registerPanel("test.active", PanelSpace::MainPanelTab, 200.0f);

    PanelLayoutManager layout;
    layout.setActiveTab("test.active");
    UIContext ui;
    PanelDrawContext draw_ctx;
    draw_ctx.ui = &ui;
    std::unordered_map<std::string, bool> window_states;
    std::string focus_panel_name;
    PanelInputState input;

    layout.renderRightPanel(ui, draw_ctx, true, false, window_states, focus_panel_name,
                            input, screen(),
                            {.scene_header_live = false, .active_tab_live = true});

    EXPECT_EQ(scene_header->preload_count, 0);
    EXPECT_EQ(scene_header->draw_count, 0);
    EXPECT_EQ(scene_header->cached_draw_count, 1);
    EXPECT_EQ(active_tab->preload_count, 1);
    EXPECT_EQ(active_tab->draw_count, 1);
    EXPECT_EQ(active_tab->cached_draw_count, 0);
}

TEST_F(PanelLayoutRenderDemandTest, CanDrawSceneHeaderLiveWhileCachingActiveTab) {
    using namespace lfs::vis::gui;

    auto scene_header = registerPanel("test.scene", PanelSpace::SceneHeader, 120.0f);
    auto active_tab = registerPanel("test.active", PanelSpace::MainPanelTab, 200.0f);

    PanelLayoutManager layout;
    layout.setActiveTab("test.active");
    UIContext ui;
    PanelDrawContext draw_ctx;
    draw_ctx.ui = &ui;
    std::unordered_map<std::string, bool> window_states;
    std::string focus_panel_name;
    PanelInputState input;

    layout.renderRightPanel(ui, draw_ctx, true, false, window_states, focus_panel_name,
                            input, screen(),
                            {.scene_header_live = true, .active_tab_live = false});

    EXPECT_EQ(scene_header->preload_count, 1);
    EXPECT_EQ(scene_header->draw_count, 1);
    EXPECT_EQ(scene_header->cached_draw_count, 0);
    EXPECT_EQ(active_tab->preload_count, 0);
    EXPECT_EQ(active_tab->draw_count, 0);
    EXPECT_EQ(active_tab->cached_draw_count, 1);
}

TEST_F(PanelLayoutRenderDemandTest, BottomDockStartsAfterVisibleLeftDock) {
    using namespace lfs::vis::gui;

    registerPanel("test.left", PanelSpace::LeftDock, 100.0f);
    auto bottom = registerPanel("test.bottom", PanelSpace::BottomDock, 100.0f);

    PanelLayoutManager layout;
    UIContext ui;
    PanelDrawContext draw_ctx;
    draw_ctx.ui = &ui;
    PanelInputState input;

    layout.renderLeftDock(draw_ctx, true, false, input, screen());
    ASSERT_TRUE(layout.isLeftDockVisible());

    const auto viewport = layout.computeViewportLayout(true, false, false, screen());
    layout.renderBottomDock(draw_ctx, true, false, input, screen());

    EXPECT_FLOAT_EQ(bottom->last_draw_x, viewport.pos.x);
    EXPECT_FLOAT_EQ(bottom->last_draw_width, viewport.size.x);

    layout.renderBottomDockCached(draw_ctx, true, false, input, screen());

    EXPECT_FLOAT_EQ(bottom->last_cached_x, viewport.pos.x);
    EXPECT_FLOAT_EQ(bottom->last_cached_width, viewport.size.x);
}

TEST_F(PanelLayoutRenderDemandTest, LeftDockReservationAppliesOnTheFramePanelBecomesVisible) {
    using namespace lfs::vis::gui;

    PanelLayoutManager layout;
    const auto s = screen();

    const auto closed = layout.computeBottomDockHorizontalLayout(true, false, s);
    EXPECT_FLOAT_EQ(closed.x, s.work_pos.x);

    registerPanel("test.left", PanelSpace::LeftDock, 100.0f);

    const auto open = layout.computeBottomDockHorizontalLayout(true, false, s);
    EXPECT_GT(open.x, closed.x);
    EXPECT_LT(open.width, closed.width);

    const auto hidden_ui = layout.computeBottomDockHorizontalLayout(true, true, s);
    EXPECT_FLOAT_EQ(hidden_ui.x, s.work_pos.x);
    EXPECT_FLOAT_EQ(hidden_ui.width, s.work_size.x);

    PanelRegistry::instance().set_panel_enabled("test.left", false);
    const auto disabled = layout.computeBottomDockHorizontalLayout(true, false, s);
    EXPECT_FLOAT_EQ(disabled.x, closed.x);
    EXPECT_FLOAT_EQ(disabled.width, closed.width);
}

TEST_F(PanelLayoutRenderDemandTest, FloatingViewportAnchorStaysInViewportHole) {
    using namespace lfs::vis::gui;

    registerPanel("test.left.viewport_float", PanelSpace::LeftDock, 100.0f);
    auto seq = registerPanel("test.seq.viewport_float",
                             PanelSpace::Floating,
                             200.0f,
                             static_cast<uint32_t>(PanelOption::FLOAT_IN_VIEWPORT),
                             8192.0f);

    PanelLayoutManager layout;
    UIContext ui;
    PanelDrawContext draw_ctx;
    draw_ctx.ui = &ui;
    const auto s = screen();
    const auto viewport = layout.computeViewportLayout(true, false, false, s);
    draw_ctx.viewport = &viewport;
    draw_ctx.screen_bounds = PanelDrawBounds{
        .width = s.work_size.x,
        .height = s.work_size.y,
    };

    PanelRegistry::instance().render_panels({
                                                .target = PanelRenderTarget::for_space(PanelSpace::Floating),
                                                .mode = PanelRenderMode::Standard,
                                            },
                                            draw_ctx);

    EXPECT_EQ(seq->draw_count, 1);
    EXPECT_GE(seq->last_draw_x, viewport.pos.x);
    EXPECT_LE(seq->last_draw_x + seq->last_draw_width, viewport.pos.x + viewport.size.x);
    const float right_edge = s.work_pos.x + s.work_size.x - layout.getRightPanelWidth();
    EXPECT_LE(seq->last_draw_x + seq->last_draw_width, right_edge);
    EXPECT_FLOAT_EQ(seq->last_draw_y + seq->last_draw_height,
                    viewport.pos.y + viewport.size.y);
}

TEST_F(PanelLayoutRenderDemandTest, SetPanelSpaceFloatingMasksSameFrame) {
    using namespace lfs::vis::gui;

    registerPanel("test.left.same_frame", PanelSpace::LeftDock, 100.0f);
    registerPanel("test.seq.same_frame",
                  PanelSpace::BottomDock,
                  200.0f,
                  static_cast<uint32_t>(PanelOption::FLOAT_IN_VIEWPORT),
                  8192.0f,
                  200.0f);

    PanelLayoutManager layout;
    UIContext ui;
    PanelDrawContext draw_ctx;
    draw_ctx.ui = &ui;
    const auto s = screen();
    const auto viewport = layout.computeViewportLayout(true, false, false, s);
    draw_ctx.viewport = &viewport;
    draw_ctx.screen_bounds = PanelDrawBounds{
        .width = s.work_size.x,
        .height = s.work_size.y,
    };

    PanelRegistry::instance().render_panels({
                                                .target = PanelRenderTarget::for_space(PanelSpace::Floating),
                                                .mode = PanelRenderMode::Standard,
                                            },
                                            draw_ctx);

    ASSERT_TRUE(PanelRegistry::instance().set_panel_space("test.seq.same_frame",
                                                          PanelSpace::Floating));

    ASSERT_GT(viewport.pos.x, 32.0f);

    const float band_x = viewport.pos.x + 16.0f;
    const float band_y = viewport.pos.y + viewport.size.y - 100.0f;
    EXPECT_TRUE(PanelRegistry::instance().isPositionOverFloatingPanel(band_x, band_y));

    const float left_x = s.work_pos.x + 8.0f;
    EXPECT_FALSE(PanelRegistry::instance().isPositionOverFloatingPanel(left_x, band_y));
}

TEST_F(PanelLayoutRenderDemandTest,
       PollHiddenPendingUpdateDoesNotDemandAnimationUntilVisible) {
    using namespace lfs::vis::gui;

    auto panel = std::make_shared<PollGatedUpdatePanel>();
    PanelInfo info;
    info.id = "test.poll_gated_update";
    info.label = info.id;
    info.space = PanelSpace::MainPanelTab;
    info.is_native = false;
    info.panel = panel;
    ASSERT_TRUE(PanelRegistry::instance().register_panel(std::move(info)));

    PanelDrawContext ctx;
    const PanelRenderOptions preload{
        .target = PanelRenderTarget::for_panel("test.poll_gated_update"),
        .mode = PanelRenderMode::DirectPreload,
        .width = 320.0f,
        .height = 200.0f,
    };
    const PanelAnimationVisibility visibility{
        .active_main_tab = "test.poll_gated_update",
    };

    PanelRegistry::instance().render_panels(preload, ctx);
    EXPECT_FALSE(panel->poll_visible);
    EXPECT_FALSE(PanelRegistry::instance().needsAnimationFrameForVisiblePanels(visibility));
    EXPECT_TRUE(panel->pending_update);

    panel->setPollResult(true);
    PanelRegistry::instance().invalidate_poll_cache();
    PanelRegistry::instance().render_panels(preload, ctx);
    EXPECT_TRUE(panel->poll_visible);
    EXPECT_TRUE(PanelRegistry::instance().needsAnimationFrameForVisiblePanels(visibility));

    PanelRegistry::instance().render_panels({
                                                .target = preload.target,
                                                .mode = PanelRenderMode::Direct,
                                                .width = preload.width,
                                                .height = preload.height,
                                            },
                                            ctx);
    EXPECT_FALSE(panel->pending_update);
    EXPECT_FALSE(PanelRegistry::instance().needsAnimationFrameForVisiblePanels(visibility));
}

TEST_F(PanelLayoutRenderDemandTest, BottomDockAndSequencerChangesPublishToolbarGeneration) {
    using namespace lfs::vis::gui;

    PanelLayoutManager layout;
    auto& generation = lfs::vis::app_store().viewport_toolbar_generation;
    const auto initial = generation.get();

    layout.setBottomDockActiveTab("test.bottom.first");
    EXPECT_EQ(generation.get(), initial + 1);
    layout.setBottomDockActiveTab("test.bottom.first");
    EXPECT_EQ(generation.get(), initial + 1);
    layout.setBottomDockActiveTab("test.bottom.second");
    EXPECT_EQ(generation.get(), initial + 2);

    layout.setShowSequencer(true);
    EXPECT_EQ(generation.get(), initial + 3);
    layout.setShowSequencer(true);
    EXPECT_EQ(generation.get(), initial + 3);
    layout.setShowSequencer(false);
    EXPECT_EQ(generation.get(), initial + 4);
}
