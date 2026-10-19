/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <RmlUi/Core.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/ElementInstancer.h>
#include <RmlUi/Core/ElementText.h>
#include <RmlUi/Core/ElementUtilities.h>
#include <RmlUi/Core/Factory.h>
#include <RmlUi/Core/RenderInterface.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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

    struct TrainingStubElement : Rml::Element {
        explicit TrainingStubElement(const Rml::String& tag) : Rml::Element(tag) {}
    };

    std::string german_training_label(const char* key) {
        std::ifstream file(std::filesystem::path(PROJECT_ROOT_PATH) / "src/visualizer/gui/resources/locales/de.json");
        std::ostringstream buffer;
        buffer << file.rdbuf();
        const std::string json = buffer.str();
        const auto section = json.find("\"training_params\"");
        const std::string needle = std::string("\"") + key + "\": \"";
        const auto at = section == std::string::npos ? std::string::npos : json.find(needle, section);
        EXPECT_NE(at, std::string::npos) << key;
        if (at == std::string::npos)
            return {};
        const auto begin = at + needle.size();
        return json.substr(begin, json.find('"', begin) - begin);
    }

    // Loads production training.rml at the default 360px right-panel width.
    class TrainingDocumentLabelTest : public ::testing::Test {
    protected:
        struct Row {
            Rml::String id;
            Rml::String label;
            Rml::String kind;
            Rml::String text;
            Rml::String value;
            Rml::String tooltip_key;
            bool checked = false;
        };

        static void SetUpTestSuite() {
            ASSERT_TRUE(Rml::Initialise());
            ASSERT_TRUE(Rml::LoadFontFace((std::filesystem::path(PROJECT_ROOT_PATH) /
                                           "src/visualizer/gui/assets/fonts/Inter-Regular.ttf")
                                              .string()));
            static Rml::ElementInstancerGeneric<TrainingStubElement> stubs;
            Rml::Factory::RegisterElementInstancer("loss-graph", &stubs);
            Rml::Factory::RegisterElementInstancer("color-picker", &stubs);
        }
        static void TearDownTestSuite() { Rml::Shutdown(); }

        void SetUp() override {
            max_label_ = german_training_label("max_gaussians");
            exposure_label_ = german_training_label("exposure_correction");
            ASSERT_FALSE(max_label_.empty());
            ASSERT_FALSE(exposure_label_.empty());
            struct_rows_.push_back({"max_cap", max_label_, "number", "300000", "", "", false});
            exposure_rows_.push_back({"use_exposure_correction", exposure_label_, "checkbox", "", "", "", false});

            context_ = Rml::CreateContext("training_labels", {360, 900}, &renderer_);
            ASSERT_NE(context_, nullptr);
            context_->SetDensityIndependentPixelRatio(1.0f);
            auto model = context_->CreateDataModel("training");
            ASSERT_TRUE(static_cast<bool>(model));
            auto row = model.RegisterStruct<Row>();
            bool bound = row.RegisterMember("id", &Row::id) && row.RegisterMember("label", &Row::label) &&
                         row.RegisterMember("kind", &Row::kind) && row.RegisterMember("text", &Row::text) &&
                         row.RegisterMember("value", &Row::value) && row.RegisterMember("tooltip_key", &Row::tooltip_key) &&
                         row.RegisterMember("checked", &Row::checked) && model.RegisterArray<std::vector<Row>>();
            show_main_ = true;
            basic_visible_ = true;
            bound = bound && model.Bind("pv_basic_struct_rows", &struct_rows_) &&
                    model.Bind("pv_basic_struct_options", &options_) &&
                    model.Bind("pv_basic_exposure_correction_rows", &exposure_rows_) &&
                    model.Bind("show_main", &show_main_) &&
                    model.Bind("pv_section_basic_params_visible", &basic_visible_) &&
                    model.Bind("pv_section_advanced_params_visible", &advanced_visible_) &&
                    model.Bind("show_training_telemetry", &telemetry_) && model.Bind("show_no_trainer", &no_trainer_) &&
                    model.Bind("show_no_params", &no_params_) && model.Bind("pv_search_active", &search_active_) &&
                    model.Bind("pv_search_query", &search_query_);
            ASSERT_TRUE(bound);
            document_ = context_->LoadDocument((std::filesystem::path(PROJECT_ROOT_PATH) /
                                                "src/visualizer/gui/rmlui/resources/training.rml")
                                                   .string());
            ASSERT_NE(document_, nullptr);
            document_->Show();
            context_->Update();
        }
        void TearDown() override {
            if (context_)
                Rml::RemoveContext("training_labels");
        }

        inline static SettingRowRenderInterface renderer_;
        Rml::Context* context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        std::vector<Row> struct_rows_;
        std::vector<Row> exposure_rows_;
        std::vector<Row> options_;
        Rml::String max_label_;
        Rml::String exposure_label_;
        Rml::String search_query_;
        bool show_main_ = false;
        bool basic_visible_ = false;
        bool advanced_visible_ = false;
        bool telemetry_ = false;
        bool no_trainer_ = false;
        bool no_params_ = false;
        bool search_active_ = false;
    };

    TEST_F(TrainingDocumentLabelTest, GermanLabelsWrapInsideTheColumn) {
        const auto expect_wrapped = [&](const Rml::String& text) {
            Rml::ElementList labels;
            document_->QuerySelectorAll(labels, ".training-scroll-region .setting-row > .prop-label");
            Rml::Element* label = nullptr;
            for (auto* element : labels) {
                std::string contents;
                for (int i = 0; i < element->GetNumChildren(); ++i)
                    if (auto* node = dynamic_cast<Rml::ElementText*>(element->GetChild(i)))
                        contents += node->GetText();
                if (element->IsVisible(true) && contents == text)
                    label = element;
            }
            ASSERT_NE(label, nullptr) << text;
            auto* row = label->GetParentNode();
            Rml::Element* value = nullptr;
            for (int i = 0; i < row->GetNumChildren(); ++i) {
                auto* child = row->GetChild(i);
                if (child->GetTagName() == "input" && child->IsVisible(false))
                    value = child;
            }
            ASSERT_NE(value, nullptr) << text;

            const float width = label->GetBox().GetSize(Rml::BoxArea::Border).x;
            const float height = label->GetBox().GetSize(Rml::BoxArea::Border).y;
            const float text_width = static_cast<float>(Rml::ElementUtilities::GetStringWidth(label, text));
            EXPECT_LE(width, 121.0f) << text;
            EXPECT_GE(width, 100.0f) << text;
            EXPECT_GT(text_width, width + 1.0f) << text;
            EXPECT_EQ(label->GetComputedValues().white_space(), Rml::Style::WhiteSpace::Normal) << text;
            EXPECT_EQ(label->GetComputedValues().overflow_x(), Rml::Style::Overflow::Visible) << text;
            EXPECT_GT(height, label->GetLineHeight() * 1.5f) << text;
            EXPECT_LE(label->GetScrollWidth(), label->GetClientWidth() + 1.0f) << text;
            EXPECT_LE(label->GetScrollHeight(), label->GetClientHeight() + 1.0f) << text;
            const float label_right = label->GetAbsoluteOffset(Rml::BoxArea::Border).x + width;
            const float value_left = value->GetAbsoluteOffset(Rml::BoxArea::Border).x;
            const float value_right = value_left + value->GetBox().GetSize(Rml::BoxArea::Border).x;
            const float row_right = row->GetAbsoluteOffset(Rml::BoxArea::Border).x +
                                    row->GetBox().GetSize(Rml::BoxArea::Border).x;
            EXPECT_GE(value_left, label_right - 1.0f) << text;
            EXPECT_GT(value->GetBox().GetSize(Rml::BoxArea::Border).x, 8.0f) << text;
            EXPECT_LE(value_right, row_right + 1.0f) << text;
        };
        expect_wrapped(max_label_);
        expect_wrapped(exposure_label_);
    }
} // namespace
