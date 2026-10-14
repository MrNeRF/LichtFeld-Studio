/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "gui/rmlui/elements/form_control_layout.hpp"
#include <RmlUi/Core.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/ElementInstancer.h>
#include <RmlUi/Core/Factory.h>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>

namespace {
    using namespace lfs::vis::gui;
    class Renderer final : public Rml::RenderInterface {
        Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex>, Rml::Span<const int>) override { return 1; }
        void RenderGeometry(Rml::CompiledGeometryHandle, Rml::Vector2f, Rml::TextureHandle) override {}
        void ReleaseGeometry(Rml::CompiledGeometryHandle) override {}
        Rml::TextureHandle LoadTexture(Rml::Vector2i&, const Rml::String&) override { return 0; }
        Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte>, Rml::Vector2i) override { return 1; }
        void ReleaseTexture(Rml::TextureHandle) override {}
        void EnableScissorRegion(bool) override {}
        void SetScissorRegion(Rml::Rectanglei) override {}
    };
    class RmlFormControlLayoutTest : public ::testing::TestWithParam<float> {
    protected:
        Renderer renderer;
        Rml::Context* context = nullptr;
        Rml::ElementInstancerGeneric<FormControlLayout<Rml::ElementFormControlInput>> input;
        Rml::ElementInstancerGeneric<FormControlLayout<Rml::ElementFormControlSelect>> select;
        Rml::ElementInstancerGeneric<FormControlLayout<Rml::ElementFormControlTextArea>> textarea;

        void SetUp() override {
            ASSERT_TRUE(Rml::Initialise());
            Rml::Factory::RegisterElementInstancer("input", &input);
            Rml::Factory::RegisterElementInstancer("select", &select);
            Rml::Factory::RegisterElementInstancer("textarea", &textarea);
            ASSERT_TRUE(Rml::LoadFontFace((std::filesystem::path(PROJECT_ROOT_PATH) /
                                           "src/visualizer/gui/assets/fonts/Inter-Regular.ttf")
                                              .generic_string()));
            context = Rml::CreateContext("form-layout", {800, 1200}, &renderer);
            ASSERT_NE(context, nullptr);
            context->SetDensityIndependentPixelRatio(GetParam());
        }
        void TearDown() override {
            Rml::RemoveContext("form-layout");
            Rml::Shutdown();
        }
        Rml::ElementDocument* load(const Rml::String& body) {
            auto* doc = context->LoadDocumentFromMemory(
                "<rml><head><style>body {font-family:Inter;font-size:12dp;} div {display:block;} scrollbarvertical {width:4px;}</style></head><body>" + body + "</body></rml>");
            if (doc) {
                doc->Show();
                context->Update();
            }
            return doc;
        }
    };

    TEST_P(RmlFormControlLayoutTest, IntrinsicBoxesKeepTheEstablishedLayoutAtEveryScale) {
        auto* doc = load("<input id='range' type='range'/><input id='vertical' type='range' orientation='vertical'/>"
                         "<input id='checkbox' type='checkbox'/><input id='radio' type='radio'/>"
                         "<select id='select'><option>First</option></select><input id='text' type='text'/>"
                         "<textarea id='textarea' rows='3'/>");
        ASSERT_NE(doc, nullptr);
        for (auto [id, expected] : {std::pair{"range", Rml::Vector2f{256, 16}},
                                    {"vertical", {16, 256}},
                                    {"checkbox", {16, 16}},
                                    {"radio", {16, 16}},
                                    {"select", {128, 16}}}) {
            SCOPED_TRACE(id);
            auto* element = doc->GetElementById(id);
            ASSERT_NE(element, nullptr);
            Rml::Vector2f dimensions;
            float ratio = 0;
            ASSERT_TRUE(element->GetIntrinsicDimensions(dimensions, ratio));
            EXPECT_FLOAT_EQ(dimensions.x, expected.x);
            EXPECT_FLOAT_EQ(dimensions.y, expected.y);
        }
        for (const char* id : {"text", "textarea"}) {
            auto* element = doc->GetElementById(id);
            Rml::Vector2f dimensions;
            float ratio = 0;
            ASSERT_TRUE(element->GetIntrinsicDimensions(dimensions, ratio));
            EXPECT_FLOAT_EQ(dimensions.y, Rml::Math::Round(element->GetLineHeight()) * (Rml::String(id) == "textarea" ? 3 : 1));
        }
    }
    TEST_P(RmlFormControlLayoutTest, ExplicitDpDimensionsOverrideIntrinsicDefaults) {
        auto* doc = load("<input id='range' type='range' style='width:42dp;height:22dp'/>");
        ASSERT_NE(doc, nullptr);
        auto size = doc->GetElementById("range")->GetBox().GetSize();
        EXPECT_NEAR(size.x, 42 * GetParam(), 0.01f);
        EXPECT_NEAR(size.y, 22 * GetParam(), 0.01f);
    }
    TEST_P(RmlFormControlLayoutTest, TrailingPaddingAndMarginsDoNotIntroduceANewScrollbar) {
        auto* doc = load("<div id='scroll' style='display:flex;flex-direction:column;width:200px;height:100px;padding:10px;box-sizing:border-box;overflow:auto'>"
                         "<div style='height:90px;flex-shrink:0;margin-bottom:12px'/></div>");
        ASSERT_NE(doc, nullptr);
        auto* scroll = doc->GetElementById("scroll");
        EXPECT_FLOAT_EQ(scroll->GetClientWidth(), 200);
        EXPECT_FLOAT_EQ(scroll->GetClientHeight(), 100);
        EXPECT_FLOAT_EQ(scroll->GetScrollHeight(), 100);
        scroll->SetScrollTop(1000);
        EXPECT_FLOAT_EQ(scroll->GetScrollTop(), 0);
    }
    TEST_P(RmlFormControlLayoutTest, PreferencesPopupCoversValuesInFollowingRows) {
        auto* doc = load("<div class='preferences-row'><span class='preferences-label'>Navigation Mode</span>"
                         "<select id='navigation' class='preferences-select' style='margin-left:0'><option>Orbit</option><option>Free Orbit</option>"
                         "<option>Fly</option><option>Drone</option></select></div>"
                         "<div class='setting-row setting-row--aligned'><span class='setting-row__label-col'>Zoom speed</span>"
                         "<div class='setting-row__control-col setting-row__control-col--slider'>"
                         "<div class='scrub-field'><div class='scrub-field-fill'></div><span id='number' class='scrub-field-display'>11</span></div></div></div>");
        ASSERT_NE(doc, nullptr);
        std::string css = "body { font-family: Inter; width: 600dp; }";
        for (const char* name : {"components.rcss", "preferences.rcss"}) {
            std::ifstream file(std::filesystem::path(PROJECT_ROOT_PATH) / "src/visualizer/gui/rmlui/resources" / name);
            css.append(std::istreambuf_iterator<char>(file), {});
        }
        doc->SetStyleSheetContainer(Rml::Factory::InstanceStyleSheetString(css));
        context->SetDimensions({int(800 * GetParam()), int(600 * GetParam())});
        context->Update();
        auto* select = doc->GetElementById("navigation");
        const auto pos = select->GetAbsoluteOffset();
        context->ProcessMouseMove(int(pos.x + 20 * GetParam()), int(pos.y + 8 * GetParam()), 0);
        context->ProcessMouseButtonDown(0, 0);
        context->ProcessMouseButtonUp(0, 0);
        context->Update();
        context->Render();
        context->Update();
        Rml::Element* popup = nullptr;
        for (int i = 0; i < select->GetNumChildren(true); ++i)
            if (select->GetChild(i)->GetTagName() == "selectbox")
                popup = select->GetChild(i);
        ASSERT_NE(popup, nullptr);
        ASSERT_TRUE(popup->IsPseudoClassSet("checked"));
        const auto number = doc->GetElementById("number")->GetAbsoluteOffset();
        const auto popup_position = popup->GetAbsoluteOffset(Rml::BoxArea::Border);
        const auto popup_size = popup->GetBox().GetSize(Rml::BoxArea::Border);
        const Rml::Vector2f probe{number.x + 5 * GetParam(), number.y + 5 * GetParam()};
        ASSERT_GE(probe.x, popup_position.x);
        ASSERT_GE(probe.y, popup_position.y);
        ASSERT_LT(probe.x, popup_position.x + popup_size.x);
        ASSERT_LT(probe.y, popup_position.y + popup_size.y);
        auto* top = context->GetElementAtPoint(probe);
        ASSERT_NE(top, nullptr);
        SCOPED_TRACE(top->GetTagName() + " " + top->GetId());
        while (top && top != popup)
            top = top->GetParentNode();
        EXPECT_EQ(top, popup);
    }

    INSTANTIATE_TEST_SUITE_P(UiScales, RmlFormControlLayoutTest,
                             ::testing::Values(1.f, 1.25f, 1.5f, 1.75f, 2.f, 3.f));
} // namespace
