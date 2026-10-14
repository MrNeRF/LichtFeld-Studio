/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rml_right_panel.hpp"
#include "gui/rmlui/elements/form_control_layout.hpp"
#include <RmlUi/Core.h>
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
            panel.tab_picker_el_ = dynamic_cast<Rml::ElementFormControlSelect*>(doc->GetElementById("tab-picker"));
        }
        static void sync(RmlRightPanel& panel, const std::vector<TabSnapshot>& tabs, const std::string& active) {
            panel.syncTabData(tabs, active);
            panel.updateTabModel();
            panel.updateTabModel();
        }
        static std::string active(const RmlRightPanel& panel) { return panel.active_tab_; }
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
        Rml::ElementInstancerGeneric<FormControlLayoutElement<Rml::ElementFormControlSelect>> select;
        Rml::Context* context = nullptr;
        RmlRightPanel panel;
        Rml::ElementFormControlSelect* picker = nullptr;
        std::vector<std::string> picks;
        void SetUp() override {
            ASSERT_TRUE(Rml::Initialise());
            Rml::Factory::RegisterElementInstancer("select", &select);
            context = Rml::CreateContext("tab-picker-review", {400, 600}, &renderer);
            ASSERT_NE(context, nullptr);
            RmlRightPanelTestAccess::bind(panel, context);
            auto* doc = context->LoadDocumentFromMemory(
                "<rml><body data-model='right_panel_tabs'><select id='tab-picker' data-value='active_tab' data-event-change='pick_tab'>"
                "<option data-for='tab : tabs' data-attr-value='tab.id'>{{tab.label}}</option></select></body></rml>");
            ASSERT_NE(doc, nullptr);
            RmlRightPanelTestAccess::attach(panel, doc);
            picker = dynamic_cast<Rml::ElementFormControlSelect*>(doc->GetElementById("tab-picker"));
            ASSERT_NE(picker, nullptr);
            panel.on_tab_changed = [this](const std::string& id) { picks.push_back(id); };
            doc->Show();
        }
        void TearDown() override {
            panel.shutdown();
            Rml::RemoveContext("tab-picker-review");
            Rml::Shutdown();
        }
        static std::vector<TabSnapshot> tabs(std::initializer_list<const char*> ids) {
            std::vector<TabSnapshot> result;
            for (const char* id : ids)
                result.push_back({.id = id, .label = id});
            return result;
        }
    };

    TEST_F(RightPanelTabPickerTest, RemovingEarlierTabPreservesActiveSelection) {
        for (const std::string active : {"C", "D"}) {
            RmlRightPanelTestAccess::sync(panel, tabs({"A", "B", "C", "D"}), active);
            RmlRightPanelTestAccess::sync(panel, tabs({"A", "C", "D"}), active);
            EXPECT_EQ(picker->GetValue(), active);
            EXPECT_EQ(RmlRightPanelTestAccess::active(panel), active);
            EXPECT_TRUE(picks.empty());
        }
    }

    TEST_F(RightPanelTabPickerTest, InsertingEarlierTabPreservesActiveSelection) {
        RmlRightPanelTestAccess::sync(panel, tabs({"A", "B", "C", "D"}), "C");
        RmlRightPanelTestAccess::sync(panel, tabs({"A", "X", "B", "C", "D"}), "C");
        EXPECT_EQ(picker->GetValue(), "C");
        EXPECT_EQ(RmlRightPanelTestAccess::active(panel), "C");
        EXPECT_TRUE(picks.empty());
    }

    TEST_F(RightPanelTabPickerTest, OpenPopupModelUpdatesDoNotBecomeUserPicks) {
        RmlRightPanelTestAccess::sync(panel, tabs({"A", "B", "C", "D"}), "C");
        picker->Focus();
        picker->Click();
        ASSERT_TRUE(picker->IsSelectBoxVisible());
        RmlRightPanelTestAccess::sync(panel, tabs({"A", "C", "D"}), "C");
        EXPECT_EQ(picker->GetValue(), "C");
        EXPECT_TRUE(picks.empty());
        context->ProcessKeyDown(Rml::Input::KI_DOWN, 0);
        ASSERT_EQ(picks.size(), 1);
        EXPECT_EQ(picks.front(), "D");
    }
} // namespace
