/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <RmlUi/Core.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/ElementUtilities.h>
#include <RmlUi/Core/RenderInterface.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>
#include <string>

namespace {
    class SettingRowRenderInterface final : public Rml::RenderInterface {
    public:
        Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex>, Rml::Span<const int>) override { return 1; }
        void RenderGeometry(Rml::CompiledGeometryHandle, Rml::Vector2f, Rml::TextureHandle) override {}
        void ReleaseGeometry(Rml::CompiledGeometryHandle) override {}
        Rml::TextureHandle LoadTexture(Rml::Vector2i& dimensions, const Rml::String&) override {
            dimensions = {16, 16};
            return 1;
        }
        Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte>, Rml::Vector2i) override { return 1; }
        void ReleaseTexture(Rml::TextureHandle) override {}
        void EnableScissorRegion(bool) override {}
        void SetScissorRegion(Rml::Rectanglei) override {}
    };

    class SettingRowLayoutTest : public ::testing::TestWithParam<float> {
    protected:
        static void SetUpTestSuite() {
            ASSERT_TRUE(Rml::Initialise());
            ASSERT_TRUE(Rml::LoadFontFace((std::filesystem::path(PROJECT_ROOT_PATH) /
                                           "src/visualizer/gui/assets/fonts/Inter-Regular.ttf")
                                              .string()));
        }
        static void TearDownTestSuite() { Rml::Shutdown(); }
        void SetUp() override {
            context_ = Rml::CreateContext("setting_rows", {600, 900}, &renderer_);
            ASSERT_NE(context_, nullptr);
            context_->SetDensityIndependentPixelRatio(GetParam());
        }
        void TearDown() override { ASSERT_TRUE(Rml::RemoveContext("setting_rows")); }
        void load(int width, const std::string& label, bool checkbox) {
            context_->SetDimensions({int(width * GetParam()), int(900 * GetParam())});
            const auto resources = std::filesystem::path(PROJECT_ROOT_PATH) / "src/visualizer/gui/rmlui/resources";
            document_ = context_->LoadDocumentFromMemory(
                "<rml><head><link type=\"text/rcss\" href=\"components.rcss\"/></head><body>"
                "<div id=\"row\" class=\"setting-row setting-row--aligned\">"
                "<span id=\"label\" class=\"setting-row__label-col\">" +
                    label + "</span>" +
                    (checkbox ? "<label id=\"control\" class=\"setting-row__control-col setting-row__control-col--checkbox\"><input id=\"input\" type=\"checkbox\"/></label>" : "<div id=\"control\" class=\"setting-row__control-col setting-row__control-col--slider\"><input id=\"input\" type=\"range\" class=\"setting-slider\" min=\"1\" max=\"100\" value=\"11\"/><span class=\"slider-value\">11</span></div>") +
                    "</div></body></rml>",
                (resources / "setting_row_test.rml").string());
            ASSERT_NE(document_, nullptr);
            document_->Show();
            context_->Update();
        }
        void checkBounds() {
            auto* row = document_->GetElementById("row");
            auto* label = document_->GetElementById("label");
            auto* control = document_->GetElementById("control");
            const auto right = [](Rml::Element* e) { return e->GetAbsoluteOffset(Rml::BoxArea::Border).x + e->GetBox().GetSize(Rml::BoxArea::Border).x; };
            EXPECT_LE(right(label), control->GetAbsoluteOffset(Rml::BoxArea::Border).x);
            EXPECT_LE(right(control), right(row) + 1);
            EXPECT_GT(document_->GetElementById("input")->GetBox().GetSize(Rml::BoxArea::Border).x, 10 * GetParam());
        }
        inline static SettingRowRenderInterface renderer_;
        Rml::Context* context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
    };

    TEST_P(SettingRowLayoutTest, CheckboxUsesSpareWidthForFullLabel) {
        for (const std::string text : {"Show Coordinate Axes:", "Afficher les Axes de Coordonnées:", "Toon Coördinaat Assen:"}) {
            load(328, text, true);
            auto* label = document_->GetElementById("label");
            EXPECT_GE(label->GetBox().GetSize().x + 1, Rml::ElementUtilities::GetStringWidth(label, text));
            checkBounds();
            document_->Close();
        }
    }

    TEST_P(SettingRowLayoutTest, WideSliderShowsLocalizedLabel) {
        load(560, "Navigationsgeschwindigkeit:", false);
        auto* label = document_->GetElementById("label");
        EXPECT_GE(label->GetBox().GetSize().x + 1, Rml::ElementUtilities::GetStringWidth(label, "Navigationsgeschwindigkeit:"));
        EXPECT_GE(document_->GetElementById("input")->GetBox().GetSize().x, 200 * GetParam());
        checkBounds();
    }

    TEST_P(SettingRowLayoutTest, NarrowSliderWrapsWithoutHidingTextOrControls) {
        load(240, "Navigationsgeschwindigkeit:", false);
        auto* label = document_->GetElementById("label");
        EXPECT_EQ(label->GetComputedValues().white_space(), Rml::Style::WhiteSpace::Normal);
        EXPECT_GT(label->GetBox().GetSize().y, label->GetComputedValues().line_height().value);
        checkBounds();
    }

    TEST_P(SettingRowLayoutTest, ShortLabelsAndControlsRemainUsableAfterResizing) {
        for (const bool checkbox : {true, false}) {
            load(328, "Show Grid:", checkbox);
            auto* input = document_->GetElementById("input");
            const auto original = input->GetBox().GetSize();
            for (const int width : {560, 240, 328}) {
                context_->SetDimensions({int(width * GetParam()), int(900 * GetParam())});
                context_->Update();
                checkBounds();
                if (checkbox)
                    EXPECT_EQ(input->GetBox().GetSize(), original);
            }
            EXPECT_EQ(input->GetBox().GetSize(), original);
            document_->Close();
        }
    }

    TEST_P(SettingRowLayoutTest, CompactPropertyColumnRetainsOriginalGeometry) {
        load(328, "Short:", false);
        auto* label = document_->GetElementById("label");
        label->SetClassNames("prop-label");
        for (const int width : {560, 240, 328}) {
            context_->SetDimensions({int(width * GetParam()), int(900 * GetParam())});
            context_->Update();
            EXPECT_NEAR(label->GetBox().GetSize().x, std::min(120.0f, width * .4f) * GetParam(), 1);
            EXPECT_EQ(label->GetComputedValues().white_space(), Rml::Style::WhiteSpace::Nowrap);
        }
    }

    TEST_P(SettingRowLayoutTest, CheckboxAndSliderKeepValuesAndInputBehavior) {
        load(328, "Show Grid:", true);
        auto* input = document_->GetElementById("input");
        EXPECT_NEAR(input->GetBox().GetSize().x, 13 * GetParam(), 1);
        EXPECT_NEAR(input->GetBox().GetSize().y, 13 * GetParam(), 1);
        const auto pos = input->GetAbsoluteOffset(Rml::BoxArea::Content);
        context_->ProcessMouseMove(int(pos.x + 5), int(pos.y + 5), 0);
        context_->ProcessMouseButtonDown(0, 0);
        context_->ProcessMouseButtonUp(0, 0);
        EXPECT_TRUE(input->HasAttribute("checked"));
        document_->Close();
        load(560, "Zoom speed:", false);
        input = document_->GetElementById("input");
        EXPECT_EQ(input->GetAttribute<int>("value", 0), 11);
        input->SetAttribute("value", 42);
        context_->SetDimensions({int(240 * GetParam()), int(900 * GetParam())});
        context_->Update();
        EXPECT_EQ(input->GetAttribute<int>("value", 0), 42);
        checkBounds();
    }

    TEST_P(SettingRowLayoutTest, UpdateCost) {
        for (const bool checkbox : {true, false}) {
            load(checkbox ? 328 : 560, checkbox ? "Show Coordinate Axes:" : "Navigationsgeschwindigkeit:", checkbox);
            for (const bool resize : {false, true}) {
                const auto start = std::chrono::steady_clock::now();
                for (int i = 0; i < 10000; ++i) {
                    if (resize)
                        context_->SetDimensions({int(((checkbox ? 328 : 560) + i % 2) * GetParam()), int(900 * GetParam())});
                    context_->Update();
                }
                std::cout << (checkbox ? "checkbox" : "slider") << (resize ? " resize" : " idle") << " update us: "
                          << std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / 10000 << "\n";
            }
            document_->Close();
        }
    }

    INSTANTIATE_TEST_SUITE_P(UiScales, SettingRowLayoutTest, ::testing::Values(1.0f, 1.5f, 2.0f));
} // namespace
