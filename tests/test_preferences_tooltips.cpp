/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/event_bridge/localization_manager.hpp"
#include "gui/rmlui/rml_tooltip.hpp"
#include <RmlUi/Core.h>
#include <RmlUi/Core/ElementDocument.h>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <regex>

namespace {
    // Report assertions to GTest rather than opening a modal Windows dialog.
    class TestSystemInterface final : public Rml::SystemInterface {
    public:
        bool LogMessage(Rml::Log::Type type, const Rml::String& message) override {
            if (type == Rml::Log::LT_ASSERT)
                ADD_FAILURE() << message;
            return true;
        }
    };

    class StubRenderInterface final : public Rml::RenderInterface {
    public:
        Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex>,
                                                    Rml::Span<const int>) override {
            return 1;
        }

        void RenderGeometry(Rml::CompiledGeometryHandle,
                            Rml::Vector2f,
                            Rml::TextureHandle) override {}

        void ReleaseGeometry(Rml::CompiledGeometryHandle) override {}

        Rml::TextureHandle LoadTexture(Rml::Vector2i& dimensions,
                                       const Rml::String&) override {
            dimensions = {16, 16};
            return 1;
        }

        Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte>,
                                           Rml::Vector2i) override {
            return 1;
        }

        void ReleaseTexture(Rml::TextureHandle) override {}
        void EnableScissorRegion(bool) override {}
        void SetScissorRegion(Rml::Rectanglei) override {}
    };

    class PreferencesTooltipTest : public ::testing::Test {
    protected:
        static void SetUpTestSuite() {
            Rml::SetSystemInterface(&system_);
            ASSERT_TRUE(Rml::Initialise());
            ASSERT_TRUE(lfs::event::LocalizationManager::getInstance().initialize(
                (std::filesystem::path(PROJECT_ROOT_PATH) / "src/visualizer/gui/resources/locales").string()));
        }
        static void TearDownTestSuite() {
            lfs::event::LocalizationManager::getInstance().reset();
            Rml::Shutdown();
            Rml::SetSystemInterface(nullptr);
        }
        void SetUp() override {
            context_ = Rml::CreateContext("preferences_tooltip_test", {1000, 800}, &renderer_);
            ASSERT_NE(context_, nullptr);
            std::ifstream file(std::filesystem::path(PROJECT_ROOT_PATH) / "src/visualizer/gui/rmlui/resources/preferences.rml");
            ASSERT_TRUE(file.is_open());
            std::string markup{std::istreambuf_iterator<char>(file), {}};
            // Keep production DOM and tooltip attributes; omit unrelated data models.
            markup = std::regex_replace(markup, std::regex(R"re( data-(?!tooltip)[a-z-]+="[^"]*")re"), "");
            document_ = context_->LoadDocumentFromMemory(markup);
            ASSERT_NE(document_, nullptr);
        }
        void TearDown() override { Rml::RemoveContext("preferences_tooltip_test"); }
        inline static TestSystemInterface system_;
        // RmlUi retains the render manager until Shutdown, beyond RemoveContext.
        inline static StubRenderInterface renderer_;
        Rml::Context* context_{};
        Rml::ElementDocument* document_{};
    };

    TEST_F(PreferencesTooltipTest, ProductionRowsOnlyInheritHelpThroughTheirLabels) {
        Rml::ElementList rows;
        document_->QuerySelectorAll(rows, ".preferences-row[data-tooltip], .preferences-setting[data-tooltip]");
        ASSERT_GE(rows.size(), 14u);
        for (auto* row : rows) {
            SCOPED_TRACE(row->GetAttribute<Rml::String>("data-tooltip", ""));
            auto* label = row->QuerySelector(".preferences-label");
            ASSERT_NE(label, nullptr);
            EXPECT_FALSE(lfs::vis::gui::resolveRmlTooltip(label).empty());
            EXPECT_TRUE(lfs::vis::gui::resolveRmlTooltip(row).empty());
            Rml::ElementList controls;
            row->QuerySelectorAll(controls, "input, select, button");
            ASSERT_FALSE(controls.empty());
            for (auto* control : controls) {
                EXPECT_TRUE(lfs::vis::gui::resolveRmlTooltip(control).empty());
                control->SetAttribute("title", "Explicit control help");
                EXPECT_EQ(lfs::vis::gui::resolveRmlTooltip(control), "Explicit control help");
            }
        }
    }

    TEST_F(PreferencesTooltipTest, LiteralResetTitleResolvesBeforeAncestorHelp) {
        auto* row = document_->CreateElement("div").release();
        row->SetClass("preferences-row", true);
        row->SetAttribute("title", "Row help");
        row->SetInnerRML("<span class=\"preferences-label\"><span>Label</span></span><input />");
        document_->AppendChild(Rml::ElementPtr(row));
        auto* label = row->GetChild(0);
        auto* text = label->GetChild(0);
        EXPECT_EQ(lfs::vis::gui::resolveRmlTooltip(text), "Row help");
        // Literal data-tooltip text is intentionally not a localization key.
        label->SetAttribute("data-tooltip", "Row help\nDouble-click to restore the default: 1");
        EXPECT_EQ(lfs::vis::gui::resolveRmlTooltip(text), "Row help");
        label->RemoveAttribute("data-tooltip");
        for (const auto* title : {"Row help\nDouble-click to restore the default: 1",
                                  "Row help\nDouble-click to restore the default: 2"}) {
            label->SetAttribute("title", title);
            EXPECT_EQ(lfs::vis::gui::resolveRmlTooltip(text), title);
            EXPECT_TRUE(lfs::vis::gui::resolveRmlTooltip(row->GetChild(1)).empty());
        }
    }

    TEST_F(PreferencesTooltipTest, NestedLabelTextRetainsHelpButControlChildrenDoNot) {
        auto* row = document_->CreateElement("div").release();
        row->SetClass("preferences-row", true);
        row->SetAttribute("title", "Row help");
        row->SetInnerRML("<span class=\"preferences-label\"><span>Label</span></span><div><span>Value</span></div>");
        document_->AppendChild(Rml::ElementPtr(row));
        EXPECT_EQ(lfs::vis::gui::resolveRmlTooltip(row->GetChild(0)->GetChild(0)), "Row help");
        EXPECT_TRUE(lfs::vis::gui::resolveRmlTooltip(row->GetChild(1)->GetChild(0)).empty());
    }
} // namespace
