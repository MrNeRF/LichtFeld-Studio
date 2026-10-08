/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <RmlUi/Core.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/RenderInterface.h>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
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
                "<body class=\"docked-panel\"><div id=\"histogram-shell\" class=\"is-floating\">"
                "<div id=\"histogram-card\">" +
                    row + "</div></div></body></rml>",
                (resources / "histogram_layout_test.rml").string());
            ASSERT_NE(document_, nullptr);
            document_->Show();
        }
        void TearDown() override { ASSERT_TRUE(Rml::RemoveContext("histogram_layout")); }

        void resize(const int width) {
            context_->SetDimensions({static_cast<int>(width * GetParam()), static_cast<int>(1000 * GetParam())});
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
            EXPECT_GE(min->GetBox().GetSize(Rml::BoxArea::Border).x, 72.0f * GetParam());
            EXPECT_GE(max->GetBox().GetSize(Rml::BoxArea::Border).x, 72.0f * GetParam());
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

        // Compare the wide layout against the original ungrouped controls.
        auto* controls = document_->QuerySelector(".histogram-range-controls");
        if (controls) {
            const auto geometry = [&] {
                std::vector<float> values;
                for (const auto* selector : {".range-of-interest-label", "#range-min-input", "#range-max-input", ".range-reset"}) {
                    auto* element = document_->QuerySelector(selector);
                    const auto offset = element->GetAbsoluteOffset(Rml::BoxArea::Border);
                    const auto size = element->GetBox().GetSize(Rml::BoxArea::Border);
                    values.insert(values.end(), {offset.x, offset.y, size.x, size.y});
                }
                return values;
            };
            const auto grouped = geometry();
            document_->GetElementById("histogram-range-of-interest")->SetInnerRML("<span class=\"range-of-interest-label\">" + label->GetInnerRML() + "</span>" + controls->GetInnerRML());
            context_->Update();
            EXPECT_EQ(geometry(), grouped);
        }
    }

    INSTANTIATE_TEST_SUITE_P(UiScales, HistogramLayoutTest, ::testing::Values(1.0f, 1.5f, 2.0f));
} // namespace
