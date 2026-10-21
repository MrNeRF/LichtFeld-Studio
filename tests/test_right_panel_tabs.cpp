/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panel_layout.hpp"
#include "gui/rml_right_panel.hpp"
#include <RmlUi/Core.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <unordered_map>

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
        static void settle(RmlRightPanel& panel) { panel.updateTabs(); }
        static void scroll(RmlRightPanel& panel, float delta) { panel.scrollTabs(delta); }
    };
} // namespace lfs::vis::gui

namespace {
    using namespace lfs::vis::gui;
    class TabRenderer final : public Rml::RenderInterface {
    public:
        struct TextDraw {
            Rml::Rectanglef bounds;
            Rml::Rectanglei clip;
            float direction_y;
        };
        std::vector<TextDraw> text_draws;

    private:
        std::unordered_map<Rml::CompiledGeometryHandle, std::vector<Rml::Vertex>> geometry;
        Rml::CompiledGeometryHandle next_handle = 1;
        Rml::Matrix4f transform = Rml::Matrix4f::Identity();
        Rml::Rectanglei clip;
        bool clipping = false;
        Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex> vertices, Rml::Span<const int>) override {
            const auto handle = next_handle++;
            geometry[handle] = {vertices.begin(), vertices.end()};
            return handle;
        }
        void RenderGeometry(Rml::CompiledGeometryHandle handle, Rml::Vector2f translation, Rml::TextureHandle texture) override {
            if (!texture)
                return;
            auto bounds = Rml::Rectanglef::MakeInvalid();
            for (const auto& vertex : geometry.at(handle)) {
                const auto pos = transform * Rml::Vector4f(vertex.position.x + translation.x, vertex.position.y + translation.y, 0, 1);
                const Rml::Vector2f point(pos.x / pos.w, pos.y / pos.w);
                bounds = bounds.Valid() ? bounds.Join(point) : Rml::Rectanglef::FromPosition(point);
            }
            const auto direction = transform * Rml::Vector4f(1, 0, 0, 0);
            text_draws.push_back({bounds, clipping ? clip : Rml::Rectanglei::FromSize({400, 600}), direction.y});
        }
        void ReleaseGeometry(Rml::CompiledGeometryHandle handle) override { geometry.erase(handle); }
        Rml::TextureHandle LoadTexture(Rml::Vector2i&, const Rml::String&) override { return 0; }
        Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte>, Rml::Vector2i) override { return 1; }
        void ReleaseTexture(Rml::TextureHandle) override {}
        void EnableScissorRegion(bool enabled) override { clipping = enabled; }
        void SetScissorRegion(Rml::Rectanglei region) override { clip = region; }
        void SetTransform(const Rml::Matrix4f* matrix) override {
            transform = matrix ? *matrix : Rml::Matrix4f::Identity();
        }
    };

    class RightPanelTabsTest : public ::testing::TestWithParam<float> {
    protected:
        TabRenderer renderer;
        Rml::Context* context = nullptr;
        Rml::ElementDocument* doc = nullptr;
        RmlRightPanel panel;
        std::vector<std::string> picks;
        void SetUp() override {
            ASSERT_TRUE(Rml::Initialise());
            const auto root = std::filesystem::path(PROJECT_ROOT_PATH);
            ASSERT_TRUE(Rml::LoadFontFace((root / "src/visualizer/gui/assets/fonts/Inter-Regular.ttf").generic_string()));
            ASSERT_TRUE(Rml::LoadFontFace((root / "src/visualizer/gui/assets/fonts/Inter-SemiBold.ttf").generic_string()));
            context = Rml::CreateContext("vertical-tabs", {400, 600}, &renderer);
            ASSERT_NE(context, nullptr);
            context->SetDensityIndependentPixelRatio(GetParam());
            context->SetDefaultScrollBehavior(Rml::ScrollBehavior::Instant, 1.0f);
            RmlRightPanelTestAccess::bind(panel, context);
            doc = context->LoadDocument((root / "src/visualizer/gui/rmlui/resources/right_panel.rml").generic_string());
            ASSERT_NE(doc, nullptr);
            RmlRightPanelTestAccess::attach(panel, doc);
            doc->GetElementById("tab-bar")->SetProperty("top", "0px");
            doc->GetElementById("tab-bar")->SetProperty("height", "240px");
            panel.on_tab_changed = [this](const std::string& id) { picks.push_back(id); };
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
            input.mouse_x = 8.0f + 14.0f * GetParam();
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

    TEST_P(RightPanelTabsTest, TabsStackOnTheLeftAndLabelsDetermineHeight) {
        auto data = tabs(3);
        data[0].label = "Rendering";
        data[1].label = "Training";
        data[2].label = "A plugin with a much longer translated label";
        RmlRightPanelTestAccess::sync(panel, data, data[0].id);
        EXPECT_NEAR(button(0)->GetAbsoluteTop(),
                    doc->GetElementById("tab-strip-viewport")->GetAbsoluteTop(), 0.5f);
        float last_bottom = 0;
        for (int i = 0; i < 3; ++i) {
            auto* tab = button(i);
            const auto position = tab->GetAbsoluteOffset(Rml::BoxArea::Border);
            const auto size = tab->GetBox().GetSize(Rml::BoxArea::Border);
            EXPECT_NEAR(position.x, 8 + (i == 0 ? 0 : 3) * GetParam(), 0.5f);
            EXPECT_NEAR(size.x, (i == 0 ? 28 : 25) * GetParam(), 0.5f);
            EXPECT_GE(position.y, last_bottom);
            last_bottom = position.y + size.y;
            EXPECT_NEAR(size.y, tab->GetChild(0)->GetBox().GetSize().x + 18 * GetParam(), 1.0f);
        }
        EXPECT_GT(button(2)->GetBox().GetSize().y, button(0)->GetBox().GetSize().y);
        EXPECT_NE(button(0)->GetProperty("nav-down"), nullptr);
        EXPECT_FLOAT_EQ(button(0)->GetBox().GetEdge(Rml::BoxArea::Border, Rml::BoxEdge::Right), 0);
        EXPECT_GT(button(1)->GetBox().GetEdge(Rml::BoxArea::Border, Rml::BoxEdge::Right), 0);
    }

    TEST_P(RightPanelTabsTest, RelabelingResizesPreviouslySettledTabs) {
        auto data = tabs(2);
        RmlRightPanelTestAccess::sync(panel, data, data[0].id);
        const float initial_height = button(1)->GetBox().GetSize(Rml::BoxArea::Border).y;
        for (int i = 0; i < 3; ++i)
            RmlRightPanelTestAccess::settle(panel);

        data[1].label = "A plugin with a much longer translated label";
        RmlRightPanelTestAccess::sync(panel, data, data[0].id);
        EXPECT_GT(button(1)->GetBox().GetSize(Rml::BoxArea::Border).y, initial_height);
        EXPECT_NEAR(button(1)->GetBox().GetSize(Rml::BoxArea::Border).y,
                    button(1)->GetChild(0)->GetBox().GetSize().x + 18 * GetParam(), 1.0f);

        data[1].label = "Plugin 1";
        RmlRightPanelTestAccess::sync(panel, data, data[0].id);
        EXPECT_NEAR(button(1)->GetBox().GetSize(Rml::BoxArea::Border).y, initial_height, 0.5f);
    }

    TEST_P(RightPanelTabsTest, RotatedTextFitsTheRenderClipAtFractionalScales) {
        auto data = tabs(2);
        data[0].label = "Rendering";
        data[1].label = "ForestFormer3D";
        doc->GetElementById("tab-bar")->SetProperty("height", "580px");
        auto reference = doc->CreateElement("span");
        reference->SetInnerRML("Camera");
        reference->SetProperty("position", "absolute");
        reference->SetProperty("left", "150px");
        reference->SetProperty("top", "14dp");
        reference->SetProperty("font-size", "12dp");
        reference->SetProperty("line-height", "1.2");
        reference->SetProperty("font-weight", "600");
        doc->AppendChild(std::move(reference));
        for (const auto& selected : data) {
            RmlRightPanelTestAccess::sync(panel, data, selected.id);
            renderer.text_draws.clear();
            context->Render();
            ASSERT_EQ(renderer.text_draws.size(), 3);
            EXPECT_NEAR(renderer.text_draws[0].bounds.Top(), renderer.text_draws.back().bounds.Top(), 1.5f);
            for (int i = 0; i < 2; ++i) {
                auto* tab = button(i);
                auto* label = tab->GetChild(0);
                const auto& draw = renderer.text_draws[i];
                EXPECT_NEAR(draw.direction_y, -1.0f, 0.001f);
                const auto pos = tab->GetAbsoluteOffset(Rml::BoxArea::Border);
                const auto size = tab->GetBox().GetSize(Rml::BoxArea::Border);
                EXPECT_GE(draw.bounds.Left(), pos.x + GetParam());
                EXPECT_LE(draw.bounds.Right(), pos.x + size.x - GetParam());
                EXPECT_GE(draw.bounds.Top(), pos.y + 9.0f * GetParam());
                EXPECT_LE(draw.bounds.Bottom(), pos.y + size.y - 8.0f * GetParam());
                EXPECT_NEAR((draw.bounds.Top() + draw.bounds.Bottom()) * 0.5f,
                            pos.y + size.y * 0.5f, 1.5f);
                EXPECT_NEAR(label->GetAbsoluteLeft(), std::round(label->GetAbsoluteLeft()), 0.01f);
                EXPECT_NEAR(label->GetAbsoluteTop(), std::round(label->GetAbsoluteTop()), 0.01f);
            }
            for (const auto& draw : renderer.text_draws) {
                EXPECT_GE(draw.bounds.Left(), draw.clip.Left() - 1.0f);
                EXPECT_GE(draw.bounds.Top(), draw.clip.Top() - 1.0f);
                EXPECT_LE(draw.bounds.Right(), draw.clip.Right() + 1.0f);
                EXPECT_LE(draw.bounds.Bottom(), draw.clip.Bottom() + 1.0f);
            }
        }
    }

    TEST_P(RightPanelTabsTest, RailEdgeOpensOnlyBesideTheSelectedTab) {
        auto data = tabs(3);
        RmlRightPanelTestAccess::sync(panel, data, data[1].id);
        auto* selected = button(1);
        auto* strip = doc->GetElementById("tab-strip-viewport");
        auto* before = doc->GetElementById("tab-edge-before");
        auto* after = doc->GetElementById("tab-edge-after");
        const float top = selected->GetAbsoluteOffset(Rml::BoxArea::Border).y -
                          strip->GetAbsoluteOffset(Rml::BoxArea::Border).y;
        EXPECT_NEAR(before->GetBox().GetSize().y, top, 1.0f);
        EXPECT_NEAR(before->GetAbsoluteTop(), strip->GetAbsoluteTop(), 0.5f);
        EXPECT_NEAR(after->GetAbsoluteOffset(Rml::BoxArea::Border).y,
                    selected->GetAbsoluteOffset(Rml::BoxArea::Border).y +
                        selected->GetBox().GetSize(Rml::BoxArea::Border).y,
                    1.0f);
        RmlRightPanelTestAccess::scroll(panel, 1000);
        RmlRightPanelTestAccess::settle(panel);
        EXPECT_GE(before->GetBox().GetSize().y, 0);
        EXPECT_GE(after->GetBox().GetSize().y, 0);
        EXPECT_GE(after->GetAbsoluteTop(), strip->GetAbsoluteTop());
        RmlRightPanelTestAccess::sync(panel, data, data[0].id);
        EXPECT_FLOAT_EQ(before->GetBox().GetSize().y, 0);
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

    TEST_P(RightPanelTabsTest, PluginCloseControlsDoNotActivateTheirTab) {
        auto data = tabs(2);
        data[1].closeable = true;
        std::vector<std::string> closed;
        panel.on_tab_closed = [&](const auto& id) { closed.push_back(id); };
        RmlRightPanelTestAccess::sync(panel, data, data[0].id);
        ASSERT_EQ(button(0)->GetNumChildren(), 2);
        EXPECT_FALSE(button(0)->GetChild(1)->IsVisible());
        ASSERT_EQ(button(1)->GetNumChildren(), 2);
        auto* close = button(1)->GetChild(1);
        EXPECT_TRUE(close->IsClassSet("tab-close"));
        EXPECT_TRUE(close->IsVisible());
        const auto close_top = close->GetAbsoluteTop();
        const auto label_top = button(1)->GetChild(0)->GetAbsoluteTop();
        EXPECT_GT(close_top, label_top);
        close->Click();
        EXPECT_TRUE(picks.empty());
        ASSERT_EQ(closed.size(), 1);
        EXPECT_EQ(closed.front(), data[1].id);
        ASSERT_TRUE(close->Focus());
        context->ProcessKeyDown(Rml::Input::KI_RETURN, 0);
        EXPECT_TRUE(picks.empty());
        EXPECT_EQ(closed.size(), 2);
        button(1)->Click();
        ASSERT_EQ(picks.size(), 1);
        EXPECT_EQ(picks.back(), data[1].id);
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

    TEST_P(RightPanelTabsTest, ControlWheelScrollsOnlyTheStripWithoutChangingCategory) {
        auto data = tabs(40);
        RmlRightPanelTestAccess::sync(panel, data, data[0].id);
        auto input = inputOverRail();
        input.key_ctrl = true;
        input.mouse_wheel = -1;
        panel.processInput(layout(), input);
        RmlRightPanelTestAccess::settle(panel);
        EXPECT_TRUE(picks.empty());
        EXPECT_GT(panel.tabStripScroll(), 0);
        const float before = panel.tabStripScroll();
        input.mouse_x = 200;
        panel.processInput(layout(), input);
        RmlRightPanelTestAccess::settle(panel);
        EXPECT_EQ(panel.tabStripScroll(), before);
        EXPECT_TRUE(picks.empty());
    }

    TEST_P(RightPanelTabsTest, OverflowButtonsShowHiddenTabsAndTrackScrollLimits) {
        auto data = tabs(40);
        RmlRightPanelTestAccess::sync(panel, data, data[0].id);
        auto* up = doc->GetElementById("tab-scroll-up");
        auto* down = doc->GetElementById("tab-scroll-down");
        EXPECT_TRUE(up->IsVisible());
        EXPECT_TRUE(down->IsVisible());
        EXPECT_TRUE(up->HasAttribute("disabled"));
        EXPECT_FALSE(down->HasAttribute("disabled"));
        down->Click();
        RmlRightPanelTestAccess::settle(panel);
        EXPECT_GT(panel.tabStripScroll(), 0);
        EXPECT_FALSE(up->HasAttribute("disabled"));
        RmlRightPanelTestAccess::scroll(panel, 100000);
        RmlRightPanelTestAccess::settle(panel);
        EXPECT_TRUE(down->HasAttribute("disabled"));
        EXPECT_FALSE(up->HasAttribute("disabled"));
        up->Click();
        RmlRightPanelTestAccess::settle(panel);
        EXPECT_FALSE(down->HasAttribute("disabled"));
        EXPECT_TRUE(picks.empty());
        data.resize(1);
        RmlRightPanelTestAccess::sync(panel, data, data[0].id);
        EXPECT_FALSE(up->IsVisible());
        EXPECT_FALSE(down->IsVisible());
        EXPECT_FLOAT_EQ(panel.tabStripScroll(), 0);
    }

    TEST_P(RightPanelTabsTest, FocusScrollUpdatesRailAndTheNextWheelOffset) {
        auto data = tabs(40);
        RmlRightPanelTestAccess::sync(panel, data, data[0].id);
        ASSERT_TRUE(button(0)->Focus());
        for (int i = 0; i < 39; ++i)
            context->ProcessKeyDown(Rml::Input::KI_DOWN, 0);
        RmlRightPanelTestAccess::settle(panel);
        auto* strip = doc->GetElementById("tab-strip-viewport");
        const float focused_scroll = strip->GetScrollTop();
        ASSERT_GT(focused_scroll, 100);
        EXPECT_FLOAT_EQ(panel.tabStripScroll(), focused_scroll);
        EXPECT_FLOAT_EQ(doc->GetElementById("tab-edge-before")->GetBox().GetSize().y, 0);
        RmlRightPanelTestAccess::scroll(panel, -30);
        RmlRightPanelTestAccess::settle(panel);
        EXPECT_NEAR(panel.tabStripScroll(), focused_scroll - 30, 0.5f);
        EXPECT_NEAR(strip->GetScrollTop(), focused_scroll - 30, 0.5f);
    }

    INSTANTIATE_TEST_SUITE_P(UiScale, RightPanelTabsTest, ::testing::Values(1.0f, 1.25f, 1.5f, 1.75f, 2.0f));
} // namespace
