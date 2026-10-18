/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panel_layout.hpp"
#include "gui/rml_right_panel.hpp"
#include <RmlUi/Core.h>
#include <filesystem>
#include <gtest/gtest.h>

namespace lfs::vis::gui {
    class RmlRightPanelTestAccess {
    public:
        static void bind(RmlRightPanel& panel, Rml::Context* context) {
            panel.rml_context_ = context;
            panel.bindTabModel();
        }
        static void attach(RmlRightPanel& panel, Rml::ElementDocument* doc) {
            panel.document_ = doc;
            panel.tab_bar_el_ = doc->GetElementById("tab-bar");
            panel.tab_strip_viewport_el_ = doc->GetElementById("tab-strip-viewport");
        }
        static void sync(RmlRightPanel& panel, const std::vector<TabSnapshot>& tabs, const std::string& active) {
            panel.syncTabData(tabs, active);
            settle(panel);
        }
        static void settle(RmlRightPanel& panel) {
            panel.updateTabModel();
            panel.layoutTabs();
            panel.updateTabModel();
            panel.syncTabNavigation();
            panel.syncTabScrollState();
            panel.updateTabModel();
        }
        static void scroll(RmlRightPanel& panel, float delta) { panel.scrollTabs(delta); }
    };
} // namespace lfs::vis::gui

namespace {
    using namespace lfs::vis::gui;
    class TabRenderer final : public Rml::RenderInterface {
        Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex>, Rml::Span<const int>) override { return 1; }
        void RenderGeometry(Rml::CompiledGeometryHandle, Rml::Vector2f, Rml::TextureHandle) override {}
        void ReleaseGeometry(Rml::CompiledGeometryHandle) override {}
        Rml::TextureHandle LoadTexture(Rml::Vector2i&, const Rml::String&) override { return 0; }
        Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte>, Rml::Vector2i) override { return 1; }
        void ReleaseTexture(Rml::TextureHandle) override {}
        void EnableScissorRegion(bool) override {}
        void SetScissorRegion(Rml::Rectanglei) override {}
    };

    class RightPanelTabsTest : public ::testing::TestWithParam<float> {
    protected:
        TabRenderer renderer;
        Rml::Context* context = nullptr;
        Rml::ElementDocument* doc = nullptr;
        RmlRightPanel panel;
        std::vector<std::string> picks;
        std::vector<std::string> closes;
        void SetUp() override {
            ASSERT_TRUE(Rml::Initialise());
            const auto root = std::filesystem::path(PROJECT_ROOT_PATH);
            ASSERT_TRUE(Rml::LoadFontFace((root / "src/visualizer/gui/assets/fonts/Inter-Regular.ttf").generic_string()));
            context = Rml::CreateContext("vertical-tabs", {400, 600}, &renderer);
            ASSERT_NE(context, nullptr);
            context->SetDensityIndependentPixelRatio(GetParam());
            RmlRightPanelTestAccess::bind(panel, context);
            doc = context->LoadDocument((root / "src/visualizer/gui/rmlui/resources/right_panel.rml").generic_string());
            ASSERT_NE(doc, nullptr);
            RmlRightPanelTestAccess::attach(panel, doc);
            doc->GetElementById("tab-bar")->SetProperty("top", "0px");
            doc->GetElementById("tab-bar")->SetProperty("height", "240px");
            panel.on_tab_changed = [this](const std::string& id) { picks.push_back(id); };
            panel.on_tab_closed = [this](const std::string& id) { closes.push_back(id); };
            doc->Show();
        }
        void TearDown() override {
            panel.shutdown();
            Rml::RemoveContext("vertical-tabs");
            Rml::Shutdown();
        }
        static std::vector<TabSnapshot> tabs(int count) {
            std::vector<TabSnapshot> result;
            for (int i = 0; i < count; ++i) {
                const auto id = "plugin-" + std::to_string(i);
                result.push_back({.id = id, .label = "Plugin " + std::to_string(i), .dom_id = "tab-" + id});
            }
            return result;
        }
        Rml::Element* button(int index) { return doc->GetElementById("tab-plugin-" + std::to_string(index)); }
        PanelInputState inputOverRail() {
            PanelInputState input;
            input.mouse_x = 400.0f - 14.0f * GetParam();
            input.mouse_y = 50.0f;
            return input;
        }
        RightPanelLayout layout() {
            RightPanelLayout result;
            result.size = {400, 600};
            result.splitter_h = 0;
            return result;
        }
    };

    TEST_P(RightPanelTabsTest, TabsStackOnTheRightAndLabelsDetermineHeight) {
        auto data = tabs(3);
        data[0].label = "Rendering";
        data[1].label = "Training";
        data[2].label = "A plugin with a much longer translated label";
        RmlRightPanelTestAccess::sync(panel, data, data[0].id);
        float last_bottom = 0;
        for (int i = 0; i < 3; ++i) {
            auto* tab = button(i);
            const auto position = tab->GetAbsoluteOffset(Rml::BoxArea::Border);
            const auto size = tab->GetBox().GetSize(Rml::BoxArea::Border);
            EXPECT_NEAR(position.x + size.x, 400, 0.5f);
            EXPECT_NEAR(size.x, 28 * GetParam(), 0.5f);
            EXPECT_GE(position.y, last_bottom);
            last_bottom = position.y + size.y;
            EXPECT_NEAR(size.y, tab->GetChild(0)->GetBox().GetSize().x + 20 * GetParam(), 1.0f);
        }
        EXPECT_GT(button(2)->GetBox().GetSize().y, button(0)->GetBox().GetSize().y);
        EXPECT_NE(button(0)->GetProperty("nav-down"), nullptr);
    }

    TEST_P(RightPanelTabsTest, WheelScrollsOverflowWithoutChangingTheActiveTab) {
        auto data = tabs(40);
        RmlRightPanelTestAccess::sync(panel, data, data[0].id);
        auto input = inputOverRail();
        input.mouse_wheel = -2;
        panel.processInput(layout(), input);
        RmlRightPanelTestAccess::settle(panel);
        EXPECT_GT(panel.tabStripScroll(), 0);
        EXPECT_TRUE(picks.empty());
        RmlRightPanelTestAccess::scroll(panel, 100000);
        RmlRightPanelTestAccess::settle(panel);
        auto* strip = doc->GetElementById("tab-strip-viewport");
        EXPECT_NEAR(panel.tabStripScroll(), strip->GetScrollHeight() - strip->GetClientHeight(), 0.5f);
    }

    TEST_P(RightPanelTabsTest, MiddleMousePansTheStripAndReleasesCapture) {
        auto data = tabs(40);
        RmlRightPanelTestAccess::sync(panel, data, data[0].id);
        auto input = inputOverRail();
        input.mouse_clicked[2] = input.mouse_down[2] = true;
        panel.processInput(layout(), input);
        input.mouse_clicked[2] = false;
        input.mouse_y -= 40;
        panel.processInput(layout(), input);
        RmlRightPanelTestAccess::settle(panel);
        EXPECT_NEAR(panel.tabStripScroll(), 40, 0.5f);
        EXPECT_TRUE(panel.isResizeInteractionActive());
        input.mouse_down[2] = false;
        panel.processInput(layout(), input);
        EXPECT_FALSE(panel.isResizeInteractionActive());
    }

    TEST_P(RightPanelTabsTest, ExternalSelectionRevealsTheLastPluginAndRemovalClampsScroll) {
        auto data = tabs(40);
        RmlRightPanelTestAccess::sync(panel, data, data.back().id);
        EXPECT_GT(panel.tabStripScroll(), 0);
        const auto last = button(39)->GetAbsoluteOffset(Rml::BoxArea::Border);
        EXPECT_GE(last.y, -0.5f);
        EXPECT_LE(last.y + button(39)->GetBox().GetSize(Rml::BoxArea::Border).y, 240.5f);
        data.resize(1);
        RmlRightPanelTestAccess::sync(panel, data, data[0].id);
        EXPECT_EQ(panel.tabStripScroll(), 0);
        EXPECT_TRUE(picks.empty());
    }

    TEST_P(RightPanelTabsTest, ReorderingAndRemovingPluginsPreserveSelection) {
        auto data = tabs(4);
        RmlRightPanelTestAccess::sync(panel, data, data[2].id);
        std::swap(data[0], data[2]);
        data.erase(data.begin() + 1);
        RmlRightPanelTestAccess::sync(panel, data, "plugin-2");
        int highlighted = 0;
        for (int i = 0; i < 4; ++i) {
            if (auto* tab = button(i); tab && tab->IsClassSet("active")) {
                ++highlighted;
                EXPECT_EQ(i, 2);
            }
        }
        EXPECT_EQ(highlighted, 1);
        EXPECT_TRUE(picks.empty());
    }

    TEST_P(RightPanelTabsTest, ClickAndCloseDispatchSeparateActions) {
        auto data = tabs(2);
        data[1].closeable = true;
        RmlRightPanelTestAccess::sync(panel, data, data[0].id);
        button(1)->Click();
        ASSERT_EQ(picks.size(), 1);
        EXPECT_EQ(picks.back(), data[1].id);
        button(1)->GetChild(1)->Click();
        ASSERT_EQ(closes.size(), 1);
        EXPECT_EQ(closes.back(), data[1].id);
        EXPECT_EQ(picks.size(), 1);
    }

    TEST_P(RightPanelTabsTest, KeyboardActivatesFocusedCategories) {
        auto data = tabs(3);
        RmlRightPanelTestAccess::sync(panel, data, data[0].id);
        button(1)->Focus();
        context->ProcessKeyDown(Rml::Input::KI_RETURN, 0);
        ASSERT_EQ(picks.size(), 1);
        EXPECT_EQ(picks.back(), data[1].id);
        context->ProcessKeyDown(Rml::Input::KI_SPACE, 0);
        ASSERT_EQ(picks.size(), 2);
        EXPECT_EQ(picks.back(), data[1].id);
    }

    TEST_P(RightPanelTabsTest, ControlWheelStopsAtEndsAndControlTabWraps) {
        auto data = tabs(3);
        RmlRightPanelTestAccess::sync(panel, data, data[2].id);
        auto input = inputOverRail();
        input.key_ctrl = true;
        input.mouse_wheel = -1;
        panel.processInput(layout(), input);
        EXPECT_TRUE(picks.empty());
        input.mouse_wheel = 1;
        panel.processInput(layout(), input);
        ASSERT_EQ(picks.size(), 1);
        EXPECT_EQ(picks.back(), data[1].id);
        input.mouse_wheel = 0;
        input.input_events.push_back({.kind = lfs::vis::FrameInputEventKind::KeyDown, .scancode = SDL_SCANCODE_TAB});
        panel.processInput(layout(), input);
        ASSERT_EQ(picks.size(), 2);
        EXPECT_EQ(picks.back(), data[0].id);
        RmlRightPanelTestAccess::sync(panel, data, data[0].id);
        input.key_shift = true;
        panel.processInput(layout(), input);
        EXPECT_EQ(picks.back(), data[2].id);
    }

    INSTANTIATE_TEST_SUITE_P(UiScale, RightPanelTabsTest, ::testing::Values(1.0f, 1.5f, 2.0f));
} // namespace
