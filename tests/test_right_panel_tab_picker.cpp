/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/elements/form_control_layout.hpp"
#include "gui/screen_host.hpp"
#include "screen/screen_service.hpp"

#include <RmlUi/Core.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace lfs::vis::gui {
    class PropertiesTabPickerTestAccess {
    public:
        static void bind(ScreenHost& host, Rml::Context* const context) { host.bindChromeModel(context); }
        static void attach(ScreenHost& host, Rml::ElementDocument* const document) { host.chrome_document_ = document; }
        static void watchPicks(ScreenHost& host, std::vector<std::string>* const picks) {
            host.on_tab_picked_ = [picks](const std::string& id) { picks->push_back(id); };
        }
        static void sync(ScreenHost& host, const std::vector<ChromeItem>& tabs, const std::string& active) {
            host.properties_.setActiveTab(active);
            host.properties_tab_selected_ = active;
            ChromeArea area;
            area.id = 1;
            area.is_properties = true;
            area.show_tab_picker = true;
            area.picker_dom_id = "tab-picker";
            area.active_tab = active;
            area.tabs = tabs;
            host.chrome_areas_ = {std::move(area)};
            host.chrome_model_.DirtyVariable("areas");
            host.reconcileTabPickerSelection();
        }
        static bool tooltipPending(const ScreenHost& host) { return host.chrome_tooltip_.needsFrame(); }
        static std::string active(const ScreenHost& host) {
            return host.chrome_areas_.empty() ? std::string{} : std::string(host.chrome_areas_.front().active_tab);
        }
        static void coverPointer(ScreenHost& host) {
            AreaFrame frame;
            frame.header = {0.0f, 0.0f, 4000.0f, 4000.0f};
            host.frames_ = {frame};
            host.work_ = {0.0f, 0.0f, 400.0f, 600.0f};
        }
        static void release(ScreenHost& host) {
            host.on_tab_picked_ = {};
            host.chrome_tooltip_.setHover({}, nullptr);
            host.chrome_model_ = {};
            host.chrome_document_ = nullptr;
            host.chrome_context_ = nullptr;
            host.chrome_areas_.clear();
        }
    };
} // namespace lfs::vis::gui

namespace {
    using namespace lfs::vis::gui;

    class PickerRenderer final : public Rml::RenderInterface {
        Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex>, Rml::Span<const int>) override { return 1; }
        void RenderGeometry(Rml::CompiledGeometryHandle, Rml::Vector2f, Rml::TextureHandle) override {}
        void ReleaseGeometry(Rml::CompiledGeometryHandle) override {}
        Rml::TextureHandle LoadTexture(Rml::Vector2i&, const Rml::String&) override { return 0; }
        Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte>, Rml::Vector2i) override { return 0; }
        void ReleaseTexture(Rml::TextureHandle) override {}
        void EnableScissorRegion(bool) override {}
        void SetScissorRegion(Rml::Rectanglei) override {}
    };

    class RightPanelTabPickerTest : public ::testing::Test {
    protected:
        PickerRenderer renderer;
        Rml::ElementInstancerGeneric<FormControlLayout<Rml::ElementFormControlSelect>> select;
        lfs::vis::screen::ScreenService screens;
        ScreenHost host;
        Rml::Context* context = nullptr;
        std::vector<std::string> picks;

        RightPanelTabPickerTest() : host(screens) {}

        void SetUp() override {
            ASSERT_TRUE(Rml::Initialise());
            Rml::Factory::RegisterElementInstancer("select", &select);
            context = Rml::CreateContext("tab-picker-review", {400, 600}, &renderer);
            ASSERT_NE(context, nullptr);
            PropertiesTabPickerTestAccess::bind(host, context);
            auto* doc = context->LoadDocumentFromMemory(
                "<rml><body data-model='screen_chrome'><div data-for='area : areas'>"
                "<div id='tab-strip'><div class='tab' data-for='item : area.tabs' data-attr-id='item.dom_id' "
                "data-class-active='item.id == area.active_tab'>{{item.label}}</div></div>"
                "<select id='tab-picker' data-attr-id='area.picker_dom_id' data-value='area.active_tab' "
                "data-event-change='pick_tab'><option data-for='tab : area.tabs' data-attr-value='tab.id'>"
                "{{tab.label}}</option></select></div></body></rml>");
            ASSERT_NE(doc, nullptr);
            PropertiesTabPickerTestAccess::attach(host, doc);
            PropertiesTabPickerTestAccess::watchPicks(host, &picks);
            doc->Show();
        }
        Rml::ElementFormControlSelect* picker() {
            auto* const element = context->GetDocument(0)->GetElementById("tab-picker");
            return dynamic_cast<Rml::ElementFormControlSelect*>(element);
        }
        void TearDown() override {
            PropertiesTabPickerTestAccess::release(host);
            Rml::RemoveContext("tab-picker-review");
            Rml::Shutdown();
        }
        static std::vector<ChromeItem> tabs(std::initializer_list<const char*> ids) {
            std::vector<ChromeItem> result;
            for (const char* id : ids) {
                ChromeItem tab;
                tab.id = id;
                tab.label = id;
                tab.dom_id = std::string("tab-") + id;
                result.push_back(std::move(tab));
            }
            return result;
        }
    };

    TEST_F(RightPanelTabPickerTest, PickerHoverPreparesHelpAndOpenPopupSuppressesIt) {
        PropertiesTabPickerTestAccess::sync(host, tabs({"A", "B"}), "A");
        auto* const picker = this->picker();
        ASSERT_NE(picker, nullptr);
        picker->SetAttribute("title", "All panels");
        context->Update();
        PropertiesTabPickerTestAccess::coverPointer(host);
        PanelInputState input;
        const auto position = picker->GetAbsoluteOffset(Rml::BoxArea::Border);
        const auto size = picker->GetBox().GetSize(Rml::BoxArea::Border);
        input.mouse_x = position.x + size.x * 0.5f;
        input.mouse_y = position.y + size.y * 0.5f;
        host.processInput(input, true);
        EXPECT_TRUE(PropertiesTabPickerTestAccess::tooltipPending(host));
        picker->Click();
        input.mouse_wheel = 1;
        host.processInput(input, true);
        EXPECT_FALSE(PropertiesTabPickerTestAccess::tooltipPending(host));
    }

    TEST_F(RightPanelTabPickerTest, RemovingEarlierTabPreservesActiveSelection) {
        for (const std::string active : {"C", "D"}) {
            PropertiesTabPickerTestAccess::sync(host, tabs({"A", "B", "C", "D"}), active);
            PropertiesTabPickerTestAccess::sync(host, tabs({"A", "C", "D"}), active);
            auto* const picker = this->picker();
            ASSERT_NE(picker, nullptr);
            EXPECT_EQ(picker->GetValue(), active);
            EXPECT_EQ(PropertiesTabPickerTestAccess::active(host), active);
            auto* strip = context->GetDocument(0)->GetElementById("tab-strip");
            int highlighted = 0;
            for (int i = 0; i < strip->GetNumChildren(); ++i) {
                auto* tab = strip->GetChild(i);
                if (tab->IsClassSet("active")) {
                    ++highlighted;
                    EXPECT_EQ(tab->GetId(), "tab-" + active);
                }
            }
            EXPECT_EQ(highlighted, 1);
            EXPECT_TRUE(picks.empty());
        }
    }

    TEST_F(RightPanelTabPickerTest, ReorderingPreservesTheCheckedOptionAndNextUserPick) {
        PropertiesTabPickerTestAccess::sync(host, tabs({"A", "B", "C", "D"}), "A");
        auto* picker = this->picker();
        ASSERT_NE(picker, nullptr);
        picker->Focus();
        picker->Click();
        PropertiesTabPickerTestAccess::sync(host, tabs({"B", "A", "C", "D"}), "A");
        picker = this->picker();
        ASSERT_NE(picker, nullptr);
        ASSERT_NE(picker->GetOption(picker->GetSelection()), nullptr);
        EXPECT_EQ(picker->GetOption(picker->GetSelection())->GetAttribute<Rml::String>("value", ""), "A");
        EXPECT_TRUE(picks.empty());
        context->ProcessKeyDown(Rml::Input::KI_DOWN, 0);
        ASSERT_EQ(picks.size(), 1);
        EXPECT_EQ(picks.front(), "C");
    }

    TEST_F(RightPanelTabPickerTest, InsertingEarlierTabPreservesActiveSelection) {
        PropertiesTabPickerTestAccess::sync(host, tabs({"A", "B", "C", "D"}), "C");
        PropertiesTabPickerTestAccess::sync(host, tabs({"A", "X", "B", "C", "D"}), "C");
        auto* const picker = this->picker();
        ASSERT_NE(picker, nullptr);
        EXPECT_EQ(picker->GetValue(), "C");
        EXPECT_EQ(PropertiesTabPickerTestAccess::active(host), "C");
        EXPECT_TRUE(picks.empty());
    }

    TEST_F(RightPanelTabPickerTest, OpenPopupModelUpdatesDoNotBecomeUserPicks) {
        PropertiesTabPickerTestAccess::sync(host, tabs({"A", "B", "C", "D"}), "C");
        auto* picker = this->picker();
        ASSERT_NE(picker, nullptr);
        picker->Focus();
        picker->Click();
        ASSERT_TRUE(picker->IsSelectBoxVisible());
        PropertiesTabPickerTestAccess::sync(host, tabs({"A", "C", "D"}), "C");
        picker = this->picker();
        ASSERT_NE(picker, nullptr);
        EXPECT_EQ(picker->GetValue(), "C");
        EXPECT_TRUE(picks.empty());
        context->ProcessKeyDown(Rml::Input::KI_DOWN, 0);
        ASSERT_EQ(picks.size(), 1);
        EXPECT_EQ(picks.front(), "D");
    }

} // namespace
