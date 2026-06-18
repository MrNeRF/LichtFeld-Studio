/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include <visualizer/gui/panel_layout.hpp>
#include <visualizer/gui/panel_registry.hpp>

#include <memory>
#include <string>
#include <unordered_map>

namespace {

    class CountingDirectPanel final : public lfs::vis::gui::IPanel {
    public:
        explicit CountingDirectPanel(float height) : height_(height) {}

        void draw(const lfs::vis::gui::PanelDrawContext&) override {}

        void preloadDirect(float, float, const lfs::vis::gui::PanelDrawContext&,
                           float = -1.0f, float = -1.0f,
                           const lfs::vis::gui::PanelInputState* = nullptr) override {
            ++preload_count;
        }

        void drawDirect(float, float, float, float,
                        const lfs::vis::gui::PanelDrawContext&) override {
            ++draw_count;
        }

        bool drawDirectCached(float, float, float, float,
                              const lfs::vis::gui::PanelDrawContext&) override {
            ++cached_draw_count;
            return true;
        }

        float getDirectDrawHeight() const override { return height_; }

        int preload_count = 0;
        int draw_count = 0;
        int cached_draw_count = 0;

    private:
        float height_ = 0.0f;
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
            float height) {
            auto panel = std::make_shared<CountingDirectPanel>(height);
            lfs::vis::gui::PanelInfo info;
            info.id = std::move(id);
            info.label = info.id;
            info.space = space;
            info.is_native = false;
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
