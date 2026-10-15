/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <RmlUi/Core.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/Elements/ElementFormControlSelect.h>
#include <RmlUi/Core/RenderInterface.h>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <regex>
#include <string>
#include <vector>

namespace {
    class HistogramRenderInterface final : public Rml::RenderInterface {
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

    class HistogramLayoutTest : public ::testing::TestWithParam<float> {
    protected:
        static void SetUpTestSuite() {
            ASSERT_TRUE(Rml::Initialise());
            ASSERT_TRUE(Rml::LoadFontFace((std::filesystem::path(PROJECT_ROOT_PATH) /
                                           "src/visualizer/gui/assets/fonts/Inter-Regular.ttf")
                                              .string()));
        }
        static void TearDownTestSuite() { Rml::Shutdown(); }
        void SetUp() override {
            context_ = Rml::CreateContext("histogram_layout", {900, 1200}, &renderer_);
            ASSERT_NE(context_, nullptr);
            context_->SetDensityIndependentPixelRatio(GetParam());
            const auto resources = std::filesystem::path(PROJECT_ROOT_PATH) / "src/visualizer/gui/rmlui/resources";
            std::ifstream input(resources / "histogram_panel.rml");
            const std::string source{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
            const auto begin = source.find("<div id=\"histogram-range-of-interest\">");
            const auto end = source.find("<div id=\"histogram-chart\">", begin);
            ASSERT_NE(begin, std::string::npos);
            ASSERT_NE(end, std::string::npos);
            auto row = source.substr(begin, end - begin);
            for (const auto& [binding, label] : {std::pair{"{{ range_of_interest_label }}", "Range of Interest:"},
                                                 std::pair{"{{ reset_range_label }}", "Reset"}}) {
                const auto pos = row.find(binding);
                ASSERT_NE(pos, std::string::npos);
                row.replace(pos, std::string(binding).size(), label);
            }
            document_ = context_->LoadDocumentFromMemory(
                "<rml><head><link type=\"text/rcss\" href=\"components.rcss\"/>"
                "<link type=\"text/rcss\" href=\"histogram_panel.rcss\"/></head>"
                "<body class=\"docked-panel\"><div id=\"histogram-shell\" >"
                "<div id=\"histogram-card\">" +
                    row + "</div></div></body></rml>",
                (resources / "histogram_layout_test.rml").string());
            ASSERT_NE(document_, nullptr);
            document_->Show();
        }
        void TearDown() override { ASSERT_TRUE(Rml::RemoveContext("histogram_layout")); }

        void resize(const int width, const int height = 300) {
            context_->SetDimensions({static_cast<int>(width * GetParam()), static_cast<int>(height * GetParam())});
            context_->Update();
        }
        void checkControls() {
            auto* label = document_->QuerySelector(".range-of-interest-label");
            auto* min = document_->GetElementById("range-min-input");
            auto* max = document_->GetElementById("range-max-input");
            auto* reset = document_->QuerySelector(".range-reset");
            auto* row = document_->GetElementById("histogram-range-of-interest");
            ASSERT_NE(label, nullptr);
            ASSERT_NE(min, nullptr);
            ASSERT_NE(max, nullptr);
            ASSERT_NE(reset, nullptr);
            ASSERT_NE(row, nullptr);
            EXPECT_LE(label->GetBox().GetSize(Rml::BoxArea::Border).y,
                      label->GetComputedValues().line_height().value + 1.0f);
            EXPECT_NEAR(min->GetAbsoluteOffset(Rml::BoxArea::Border).y, max->GetAbsoluteOffset(Rml::BoxArea::Border).y, 1.0f);
            for (auto* element : {min, max, reset}) {
                const auto size = element->GetBox().GetSize(Rml::BoxArea::Border);
                EXPECT_GE(element->GetAbsoluteOffset(Rml::BoxArea::Border).x, row->GetAbsoluteOffset(Rml::BoxArea::Border).x);
                EXPECT_LE(element->GetAbsoluteOffset(Rml::BoxArea::Border).x + size.x,
                          row->GetAbsoluteOffset(Rml::BoxArea::Border).x + row->GetBox().GetSize(Rml::BoxArea::Border).x + 1.0f);
            }
            for (auto* element : {min, max}) {
                EXPECT_EQ(element->GetComputedValues().text_align(), Rml::Style::TextAlign::Center);
                EXPECT_NEAR(element->GetComputedValues().line_height().value, 22.0f * GetParam(), 1.0f);
            }
            EXPECT_GE(min->GetBox().GetSize(Rml::BoxArea::Border).x, 72.0f * GetParam());
            EXPECT_GE(max->GetBox().GetSize(Rml::BoxArea::Border).x, 72.0f * GetParam());
        }
        void loadPanel(const bool compare) {
            document_->Close();
            context_->Update();
            const auto resources = std::filesystem::path(PROJECT_ROOT_PATH) / "src/visualizer/gui/rmlui/resources";
            std::ifstream input(resources / "histogram_panel.rml");
            std::string source{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
            source = source.substr(source.find("  <div id=\"histogram-shell\""));
            source = source.substr(0, source.find("</body>"));
            source = std::regex_replace(source, std::regex(R"re( data-[a-z-]+="[^"]*")re"), "");
            for (const auto& [binding, label] : {
                     std::pair{"{{ range_of_interest_label }}", "Range of Interest:"},
                     std::pair{"{{ reset_range_label }}", "Reset"},
                     std::pair{"{{ field_label }}", "Field"},
                     std::pair{"{{ compare_with_label }}", "Compare With"},
                     std::pair{"{{ metric_description }}", "Activated Gaussian opacity in [0, 1]"},
                     std::pair{"{{ range_text }}", "0.0031 to 0.9998"},
                     std::pair{"{{ sample_count }}", "904,850"},
                     std::pair{"{{ marked_range_text }}", "No marked range"},
                     std::pair{"{{ marked_count_text }}", "0 Gaussians"}}) {
                size_t pos;
                while ((pos = source.find(binding)) != std::string::npos)
                    source.replace(pos, std::string(binding).size(), label);
            }
            source = std::regex_replace(source, std::regex(R"(\{\{[^}]+\}\})"), "Value");
            document_ = context_->LoadDocumentFromMemory(
                "<rml><head><link type=\"text/rcss\" href=\"components.rcss\"/>"
                "<link type=\"text/rcss\" href=\"histogram_panel.rcss\"/></head>"
                "<body class=\"docked-panel\">" +
                    source + "</body></rml>",
                (resources / "histogram_layout_test.rml").string());
            ASSERT_NE(document_, nullptr);
            document_->SetProperty("height", "100%");
            document_->SetProperty("width", "100%");
            document_->GetElementById("histogram-empty")->SetClass("hidden", true);
            document_->GetElementById("histogram-card")->SetClass("hidden", compare);
            document_->GetElementById("compare-card")->SetClass("hidden", !compare);
            document_->GetElementById("compare-content")->SetClass("hidden", false);
            document_->GetElementById("histogram-bin-input")->GetParentNode()->GetParentNode()->SetClass("hidden", compare);
            Rml::ElementList bins;
            bins.push_back(document_->GetElementById("x-bin-input")->GetParentNode()->GetParentNode());
            bins.push_back(document_->GetElementById("y-bin-input")->GetParentNode()->GetParentNode());
            for (auto* element : bins)
                element->SetClass("hidden", !compare);
            // The shared controller replaces each native range with this shell.
            for (const auto* id : {"histogram-bin-input", "x-bin-input", "y-bin-input"}) {
                auto* host = document_->GetElementById(id)->GetParentNode();
                host->SetInnerRML("<div class=\"scrub-field\"><div class=\"scrub-field-fill\"></div>"
                                  "<span class=\"scrub-field-display\">56</span>"
                                  "<input class=\"scrub-field-input\" type=\"text\" value=\"56\"/></div>");
            }
            document_->Show();
        }

        void checkInside(Rml::Element* element, Rml::Element* parent) {
            ASSERT_NE(element, nullptr);
            ASSERT_NE(parent, nullptr);
            EXPECT_GE(element->GetAbsoluteOffset(Rml::BoxArea::Border).x,
                      parent->GetAbsoluteOffset(Rml::BoxArea::Border).x);
            EXPECT_LE(element->GetAbsoluteOffset(Rml::BoxArea::Border).x + element->GetBox().GetSize(Rml::BoxArea::Border).x,
                      parent->GetAbsoluteOffset(Rml::BoxArea::Border).x + parent->GetBox().GetSize(Rml::BoxArea::Border).x + 1.0f);
        }

        inline static HistogramRenderInterface renderer_;
        Rml::Context* context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
    };

    TEST_P(HistogramLayoutTest, NarrowRangeStaysLegibleAcrossRepeatedResizes) {
        for (const int width : {860, 360, 860, 360}) {
            SCOPED_TRACE(width);
            resize(width);
            checkControls();
        }
    }

    TEST_P(HistogramLayoutTest, DockedRangeKeepsControlsLegible) {
        document_->GetElementById("histogram-shell")->SetClass("is-floating", false);
        for (const int width : {860, 360}) {
            resize(width);
            checkControls();
        }
    }

    TEST_P(HistogramLayoutTest, WideRangeKeepsCaptionAndControlsOnOneLine) {
        resize(860);
        checkControls();
        auto* label = document_->QuerySelector(".range-of-interest-label");
        auto* min = document_->GetElementById("range-min-input");
        EXPECT_NEAR(label->GetAbsoluteOffset(Rml::BoxArea::Border).y + label->GetBox().GetSize(Rml::BoxArea::Border).y / 2,
                    min->GetAbsoluteOffset(Rml::BoxArea::Border).y + min->GetBox().GetSize(Rml::BoxArea::Border).y / 2, 1.0f);
        EXPECT_FLOAT_EQ(min->GetBox().GetSize(Rml::BoxArea::Border).x,
                        document_->GetElementById("range-max-input")->GetBox().GetSize(Rml::BoxArea::Border).x);

        // Numeric bounds should stay compact even when the dock spans a monitor.
        for (const int width : {860, 1800}) {
            resize(width);
            for (const auto* id : {"range-min-input", "range-max-input"}) {
                EXPECT_LE(document_->GetElementById(id)->GetBox().GetSize(Rml::BoxArea::Border).x,
                          96.0f * GetParam());
            }
        }
    }

    TEST_P(HistogramLayoutTest, BottomPanelFitsWithoutScrollbarsAtPracticalDockHeights) {
        for (const bool compare : {false, true}) {
            loadPanel(compare);
            for (const auto& [width, height] : {std::pair{1800, 220}, {1000, 300}, {860, 220}, {600, 300}, {360, 400}, {1000, 300}}) {
                SCOPED_TRACE(::testing::Message() << "size=" << width << "x" << height << " compare=" << compare);
                resize(width, height);
                auto* toolbar = document_->GetElementById("histogram-toolbar");
                auto* content = document_->GetElementById("histogram-content");
                auto* stats = document_->GetElementById("histogram-stats");
                EXPECT_LE(toolbar->GetAbsoluteOffset(Rml::BoxArea::Border).y + toolbar->GetBox().GetSize(Rml::BoxArea::Border).y,
                          content->GetAbsoluteOffset(Rml::BoxArea::Border).y);
                EXPECT_GE(stats->GetAbsoluteOffset(Rml::BoxArea::Border).y,
                          content->GetAbsoluteOffset(Rml::BoxArea::Border).y + content->GetBox().GetSize(Rml::BoxArea::Border).y);
                EXPECT_LE(stats->GetAbsoluteOffset(Rml::BoxArea::Border).y + stats->GetBox().GetSize(Rml::BoxArea::Border).y,
                          height * GetParam());
                Rml::ElementList elements;
                document_->QuerySelectorAll(elements, ".histogram-toolbar-item");
                for (auto* element : elements) {
                    if (!element->IsClassSet("hidden"))
                        checkInside(element, toolbar);
                }
                for (const auto* id : {"histogram-field", "histogram-compare-field"})
                    EXPECT_LE(document_->GetElementById(id)->GetBox().GetSize(Rml::BoxArea::Border).x, 128.0f * GetParam());
                Rml::ElementList scrubs;
                document_->QuerySelectorAll(scrubs, ".scrub-field-display, .scrub-field-input");
                for (auto* element : scrubs)
                    EXPECT_EQ(element->GetComputedValues().text_align(), Rml::Style::TextAlign::Center);
                elements.clear();
                document_->QuerySelectorAll(elements, ".histogram-stat");
                for (auto* element : elements)
                    checkInside(element, stats);
                for (const auto* id : {"histogram-shell", "histogram-content", "histogram-stats"}) {
                    auto* element = document_->GetElementById(id);
                    EXPECT_LE(element->GetScrollWidth(), element->GetClientWidth() + 1.0f) << id;
                    EXPECT_LE(element->GetScrollHeight(), element->GetClientHeight() + 1.0f) << id;
                }
                auto* chart = document_->GetElementById(compare ? "compare-chart" : "histogram-chart");
                EXPECT_GE(chart->GetBox().GetSize(Rml::BoxArea::Border).y, 32.0f * GetParam());
                if (compare) {
                    EXPECT_LE(document_->GetElementById("compare-y-axis")->GetBox().GetSize(Rml::BoxArea::Border).x, 52.0f * GetParam());
                    for (const auto* id : {"compare-range-x-min-input", "compare-range-x-max-input", "compare-range-y-min-input", "compare-range-y-max-input"})
                        checkInside(document_->GetElementById(id), document_->GetElementById("compare-range-of-interest"));
                } else {
                    checkControls();
                }
            }
        }
    }

    TEST_P(HistogramLayoutTest, CompactDropdownKeepsLongOptionsReadable) {
        loadPanel(false);
        resize(860);
        auto* select = dynamic_cast<Rml::ElementFormControlSelect*>(document_->GetElementById("histogram-compare-field"));
        ASSERT_NE(select, nullptr);
        select->RemoveAll();
        select->SetInnerRML("<option value=\"\">Off</option><option value=\"erank\">Effective Rank</option>"
                            "<option value=\"world_distance\">World Distance</option>"
                            "<option value=\"world_es\">Distancia en espacio mundo</option>");
        select->SetProperty("width", "48dp");
        select->ShowSelectBox();
        context_->Update();
        context_->Render();
        context_->Update();
        Rml::Element* popup = nullptr;
        for (int index = 0; index < select->GetNumChildren(true); ++index) {
            auto* child = select->GetChild(index);
            if (child->GetTagName() == "selectbox")
                popup = child;
        }
        ASSERT_NE(popup, nullptr);
        EXPECT_LE(select->GetBox().GetSize(Rml::BoxArea::Border).x, 48.0f * GetParam());
        EXPECT_GE(popup->GetBox().GetSize(Rml::BoxArea::Border).x, 180.0f * GetParam());
        EXPECT_LE(popup->GetScrollWidth(), popup->GetClientWidth() + 1.0f);
        EXPECT_EQ(popup->GetNumChildren(), 4);
        select->HideSelectBox();
    }

    INSTANTIATE_TEST_SUITE_P(UiScales, HistogramLayoutTest, ::testing::Values(1.0f, 1.5f, 2.0f));
} // namespace
