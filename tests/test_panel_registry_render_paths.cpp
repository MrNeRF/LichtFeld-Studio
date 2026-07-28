/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include <visualizer/gui/panel_layout.hpp>
#include <visualizer/gui/panel_registry.hpp>

#include <memory>
#include <string>
#include <utility>

namespace {

    class RecordingDirectPanel final : public lfs::vis::gui::IPanel {
    public:
        explicit RecordingDirectPanel(float height, bool cache_hit = false)
            : height_(height), cache_hit_(cache_hit) {}

        void draw(const lfs::vis::gui::PanelDrawContext&) override {}

        bool poll(const lfs::vis::gui::PanelDrawContext&) override {
            ++poll_count;
            return poll_result;
        }

        void preloadDirect(float w, float h, const lfs::vis::gui::PanelDrawContext&,
                           float clip_y_min = -1.0f, float clip_y_max = -1.0f,
                           const lfs::vis::gui::PanelInputState* input = nullptr) override {
            ++preload_count;
            preload_width = w;
            preload_height = h;
            preload_clip_y_min = clip_y_min;
            preload_clip_y_max = clip_y_max;
            preload_input = input;
            preload_set_input = current_input;
        }

        void drawDirect(float x, float y, float w, float h,
                        const lfs::vis::gui::PanelDrawContext&) override {
            ++draw_count;
            draw_x = x;
            draw_y = y;
            draw_width = w;
            draw_height = h;
            draw_input = current_input;
            draw_clip_y_min = clip_y_min;
            draw_clip_y_max = clip_y_max;
        }

        bool drawDirectCached(float x, float y, float w, float h,
                              const lfs::vis::gui::PanelDrawContext&) override {
            ++cached_draw_count;
            cached_x = x;
            cached_y = y;
            cached_width = w;
            cached_height = h;
            cached_input = current_input;
            cached_clip_y_min = clip_y_min;
            cached_clip_y_max = clip_y_max;
            return cache_hit_;
        }

        float getDirectDrawHeight() const override { return height_; }

        void setInputClipY(float y_min, float y_max) override {
            clip_y_min = y_min;
            clip_y_max = y_max;
        }

        void setInput(const lfs::vis::gui::PanelInputState* input) override {
            current_input = input;
        }

        void setPanelSpace(lfs::vis::gui::PanelSpace space) override {
            panel_space = space;
        }

        bool poll_result = true;
        int poll_count = 0;
        int preload_count = 0;
        int draw_count = 0;
        int cached_draw_count = 0;
        float preload_width = 0.0f;
        float preload_height = 0.0f;
        float preload_clip_y_min = 0.0f;
        float preload_clip_y_max = 0.0f;
        const lfs::vis::gui::PanelInputState* preload_input = nullptr;
        const lfs::vis::gui::PanelInputState* preload_set_input = nullptr;
        float draw_x = 0.0f;
        float draw_y = 0.0f;
        float draw_width = 0.0f;
        float draw_height = 0.0f;
        const lfs::vis::gui::PanelInputState* draw_input = nullptr;
        float draw_clip_y_min = 0.0f;
        float draw_clip_y_max = 0.0f;
        float cached_x = 0.0f;
        float cached_y = 0.0f;
        float cached_width = 0.0f;
        float cached_height = 0.0f;
        const lfs::vis::gui::PanelInputState* cached_input = nullptr;
        float cached_clip_y_min = 0.0f;
        float cached_clip_y_max = 0.0f;
        lfs::vis::gui::PanelSpace panel_space = lfs::vis::gui::PanelSpace::Floating;
        const lfs::vis::gui::PanelInputState* current_input = nullptr;
        float clip_y_min = -1.0f;
        float clip_y_max = -1.0f;

    private:
        float height_ = 0.0f;
        bool cache_hit_ = false;
    };

    class PanelRegistryRenderPathsTest : public ::testing::Test {
    protected:
        void SetUp() override {
            lfs::vis::gui::PanelRegistry::instance().unregister_all_non_native();
        }

        void TearDown() override {
            lfs::vis::gui::PanelRegistry::instance().unregister_all_non_native();
        }

        static std::shared_ptr<RecordingDirectPanel> registerPanel(
            std::string id,
            lfs::vis::gui::PanelSpace space,
            float height,
            bool cache_hit = false,
            std::string parent_id = {}) {
            auto panel = std::make_shared<RecordingDirectPanel>(height, cache_hit);
            lfs::vis::gui::PanelInfo info;
            info.id = std::move(id);
            info.label = info.id;
            info.space = space;
            info.parent_id = std::move(parent_id);
            info.is_native = false;
            info.panel = panel;
            EXPECT_TRUE(lfs::vis::gui::PanelRegistry::instance().register_panel(std::move(info)));
            return panel;
        }
    };

} // namespace

TEST_F(PanelRegistryRenderPathsTest, DirectSpaceDrawForwardsInputAndReturnsPanelHeight) {
    using namespace lfs::vis::gui;

    auto panel = registerPanel("test.side", PanelSpace::SidePanel, 48.0f);
    PanelDrawContext draw_ctx;
    PanelInputState input;

    const float used = PanelRegistry::instance().draw_panels_direct(
        PanelSpace::SidePanel, 10.0f, 20.0f, 300.0f, 180.0f, draw_ctx, &input);

    EXPECT_FLOAT_EQ(used, 48.0f);
    EXPECT_EQ(panel->poll_count, 1);
    EXPECT_EQ(panel->draw_count, 1);
    EXPECT_EQ(panel->cached_draw_count, 0);
    EXPECT_EQ(panel->draw_input, &input);
    EXPECT_EQ(panel->panel_space, PanelSpace::SidePanel);
    EXPECT_FLOAT_EQ(panel->draw_x, 10.0f);
    EXPECT_FLOAT_EQ(panel->draw_y, 20.0f);
    EXPECT_FLOAT_EQ(panel->draw_width, 300.0f);
    EXPECT_FLOAT_EQ(panel->draw_height, 180.0f);
    EXPECT_EQ(panel->current_input, nullptr);
}

TEST_F(PanelRegistryRenderPathsTest, CachedSingleDirectDrawAvoidsPollAndLiveDrawOnHit) {
    using namespace lfs::vis::gui;

    auto panel = registerPanel("test.cached", PanelSpace::MainPanelTab, 72.0f, true);
    PanelDrawContext draw_ctx;
    PanelInputState input;

    const float used = PanelRegistry::instance().draw_single_panel_direct_cached(
        "test.cached", 4.0f, 8.0f, 240.0f, 160.0f, draw_ctx, 12.0f, 90.0f, &input);

    EXPECT_FLOAT_EQ(used, 72.0f);
    EXPECT_EQ(panel->cached_draw_count, 1);
    EXPECT_EQ(panel->poll_count, 0);
    EXPECT_EQ(panel->draw_count, 0);
    EXPECT_EQ(panel->cached_input, &input);
    EXPECT_FLOAT_EQ(panel->cached_clip_y_min, 12.0f);
    EXPECT_FLOAT_EQ(panel->cached_clip_y_max, 90.0f);
    EXPECT_EQ(panel->current_input, nullptr);
    EXPECT_FLOAT_EQ(panel->clip_y_min, -1.0f);
    EXPECT_FLOAT_EQ(panel->clip_y_max, -1.0f);
}

TEST_F(PanelRegistryRenderPathsTest, CachedSingleDirectDrawFallsBackToLiveDrawOnMiss) {
    using namespace lfs::vis::gui;

    auto panel = registerPanel("test.cache_miss", PanelSpace::MainPanelTab, 72.0f);
    PanelDrawContext draw_ctx;
    PanelInputState input;

    const float used = PanelRegistry::instance().draw_single_panel_direct_cached(
        "test.cache_miss", 4.0f, 8.0f, 240.0f, 160.0f, draw_ctx, 12.0f, 90.0f, &input);

    EXPECT_FLOAT_EQ(used, 72.0f);
    EXPECT_EQ(panel->cached_draw_count, 1);
    EXPECT_EQ(panel->poll_count, 1);
    EXPECT_EQ(panel->draw_count, 1);
    EXPECT_EQ(panel->draw_input, &input);
    EXPECT_FLOAT_EQ(panel->draw_clip_y_min, 12.0f);
    EXPECT_FLOAT_EQ(panel->draw_clip_y_max, 90.0f);
    EXPECT_EQ(panel->current_input, nullptr);
    EXPECT_FLOAT_EQ(panel->clip_y_min, -1.0f);
    EXPECT_FLOAT_EQ(panel->clip_y_max, -1.0f);
}

TEST_F(PanelRegistryRenderPathsTest, PreloadingChildrenUsesRemainingHeightWithoutDrawing) {
    using namespace lfs::vis::gui;

    registerPanel("test.parent", PanelSpace::MainPanelTab, 0.0f);
    auto first = registerPanel("test.first", PanelSpace::MainPanelTab, 30.0f, false, "test.parent");
    auto second = registerPanel("test.second", PanelSpace::MainPanelTab, 25.0f, false, "test.parent");
    PanelDrawContext draw_ctx;
    PanelInputState input;

    const float used = PanelRegistry::instance().preload_child_panels_direct(
        "test.parent", 200.0f, 80.0f, draw_ctx, 5.0f, 70.0f, &input);

    EXPECT_FLOAT_EQ(used, 55.0f);
    EXPECT_EQ(first->preload_count, 1);
    EXPECT_EQ(second->preload_count, 1);
    EXPECT_EQ(first->draw_count, 0);
    EXPECT_EQ(second->draw_count, 0);
    EXPECT_EQ(first->cached_draw_count, 0);
    EXPECT_EQ(second->cached_draw_count, 0);
    EXPECT_FLOAT_EQ(first->preload_width, 200.0f);
    EXPECT_FLOAT_EQ(first->preload_height, 80.0f);
    EXPECT_FLOAT_EQ(second->preload_height, 50.0f);
    EXPECT_FLOAT_EQ(first->preload_clip_y_min, 5.0f);
    EXPECT_FLOAT_EQ(first->preload_clip_y_max, 70.0f);
    EXPECT_EQ(first->preload_input, &input);
    EXPECT_EQ(first->preload_set_input, &input);
    EXPECT_EQ(first->current_input, nullptr);
    EXPECT_EQ(second->current_input, nullptr);
    EXPECT_FLOAT_EQ(first->clip_y_min, -1.0f);
    EXPECT_FLOAT_EQ(first->clip_y_max, -1.0f);
}
