/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <RmlUi/Core.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/ElementText.h>
#include <RmlUi/Core/ElementUtilities.h>
#include <RmlUi/Core/RenderInterface.h>

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <nlohmann/json.hpp>
#include <regex>
#include <string>
#include <vector>

namespace {

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

    class PreferencesHintLayoutTest : public ::testing::Test {
    protected:
        static std::filesystem::path root() { return PROJECT_ROOT_PATH; }

        static void SetUpTestSuite() {
            ASSERT_TRUE(Rml::Initialise());
            const auto fonts = root() / "src/visualizer/gui/assets/fonts";
            ASSERT_TRUE(Rml::LoadFontFace((fonts / "Inter-Regular.ttf").string(), true));
            ASSERT_TRUE(Rml::LoadFontFace((fonts / "NotoSansJP-Regular.ttf").string(), true));
            ASSERT_TRUE(Rml::LoadFontFace((fonts / "NotoSansKR-Regular.ttf").string(), true));
        }

        static void TearDownTestSuite() { Rml::Shutdown(); }

        void SetUp() override {
            context_ = Rml::CreateContext("preferences_hint_layout", {800, 600}, &renderer_);
            ASSERT_NE(context_, nullptr);
            std::ifstream file(root() / "src/visualizer/gui/rmlui/resources/preferences.rcss");
            ASSERT_TRUE(file.is_open());
            const std::string stylesheet{std::istreambuf_iterator<char>(file), {}};
            document_ = context_->LoadDocumentFromMemory(
                "<rml><head><style>" + stylesheet +
                "body { font-family: Inter; }</style></head>"
                "<body><span id=\"hint\" class=\"preferences-description\"></span></body></rml>");
            ASSERT_NE(document_, nullptr);
            hint_ = document_->GetElementById("hint");
            ASSERT_NE(hint_, nullptr);
            document_->Show();
        }

        void TearDown() override { Rml::RemoveContext("preferences_hint_layout"); }

        std::string translation(const std::string& locale) {
            std::ifstream file(root() / "src/visualizer/gui/resources/locales" / (locale + ".json"));
            return nlohmann::json::parse(file).at("preferences.gallery.shortcuts_hint").get<std::string>();
        }

        std::vector<std::string> lines(const std::string& text, const int width) {
            hint_->SetProperty("width", std::to_string(width) + "px");
            hint_->SetInnerRML(text);
            context_->Update();
            auto* element = rmlui_dynamic_cast<Rml::ElementText*>(hint_->GetChild(0));
            EXPECT_NE(element, nullptr);
            if (!element)
                return {};
            std::vector<std::string> result;
            for (const auto& line : element->GetLines()) {
                EXPECT_LE(Rml::ElementUtilities::GetStringWidth(element, line.text), width)
                    << line.text;
                result.push_back(line.text);
            }
            return result;
        }

        StubRenderInterface renderer_;
        Rml::Context* context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        Rml::Element* hint_ = nullptr;
    };

    TEST_F(PreferencesHintLayoutTest, JapaneseShortcutHintFitsWithoutLosingText) {
        const auto text = translation("ja");
        for (const int width : {280, 560, 760}) {
            SCOPED_TRACE(width);
            const auto wrapped = lines(text, width);
            EXPECT_GT(wrapped.size(), 1u);
            std::string joined;
            for (const auto& line : wrapped)
                joined += line;
            // Normal whitespace handling removes the space at the first line break.
            auto expected = text;
            std::erase(expected, ' ');
            std::erase(joined, ' ');
            EXPECT_EQ(joined, expected);
        }
    }

    TEST_F(PreferencesHintLayoutTest, ExistingWordWrappingIsUnchanged) {
        for (const auto* locale : {"de", "en", "es", "fr", "it", "nl", "pl", "ko", "zh"}) {
            SCOPED_TRACE(locale);
            for (const int width : {560, 760}) {
                SCOPED_TRACE(width);
                const auto text = translation(locale);
                hint_->SetProperty("word-break", "normal");
                const auto reference = lines(text, width);
                hint_->RemoveProperty("word-break");
                EXPECT_EQ(lines(text, width), reference);
            }
        }
    }

    TEST_F(PreferencesHintLayoutTest, ShortDescriptionsStayOnOneLine) {
        for (const auto* text : {"Short description.", "短い説明。", "Ctrl+Shift+C"}) {
            SCOPED_TRACE(text);
            const auto wrapped = lines(text, 560);
            ASSERT_EQ(wrapped.size(), 1u);
            EXPECT_EQ(wrapped.front(), text);
        }
    }

    TEST_F(PreferencesHintLayoutTest, ProjectDefaultsActionFitsRealPreferencesWindow) {
        auto read = [](const std::filesystem::path& path) {
            std::ifstream file(path);
            return std::string{std::istreambuf_iterator<char>(file), {}};
        };
        const auto resources = root() / "src/visualizer/gui/rmlui/resources";
        auto markup = read(resources / "preferences.rml");
        ASSERT_NE(markup.find("data-event-click=\"apply_viewport_defaults\""), std::string::npos);
        // Render the actual Viewport section without starting unrelated Python models.
        // Retain the floating-window template and the complete flex ancestor chain.
        markup = std::regex_replace(markup, std::regex(R"re(data-if="show_(general|appearance|input|interface|file_associations|mcp|portal)")re"),
                                    "style=\"display: none;\"");
        markup = std::regex_replace(markup, std::regex(R"re( data-[a-z-]+="[^"]*")re"), "");
        for (const auto* locale : {"de", "en", "es", "fr", "it", "ja", "ko", "nl", "pl", "zh"}) {
            SCOPED_TRACE(locale);
            const auto json = nlohmann::json::parse(read(root() / "src/visualizer/gui/resources/locales" / (std::string(locale) + ".json")));
            const auto label = json.at("preferences").at("apply_viewport_defaults").get<std::string>();
            for (const float scale : {1.0f, 1.5f, 2.0f}) {
                SCOPED_TRACE(scale);
                context_->SetDimensions({int(1280 * scale), int(720 * scale)});
                context_->SetDensityIndependentPixelRatio(scale);
                for (const int width : {540, 780, 1000}) {
                    SCOPED_TRACE(width);
                    auto translated = markup;
                    const std::string token = "@tr:preferences.apply_viewport_defaults";
                    translated.replace(translated.find(token), token.size(), label);
                    auto* doc = context_->LoadDocumentFromMemory(translated, (resources / "preferences.rml").string());
                    ASSERT_NE(doc, nullptr);
                    doc->SetClass("floating-panel", true);
                    doc->SetProperty("width", std::to_string(width) + "dp");
                    doc->SetProperty("height", "440dp");
                    doc->Show();
                    context_->Update();
                    auto* button = doc->QuerySelector(".preferences-apply-defaults");
                    ASSERT_NE(button, nullptr);
                    auto* container = button->GetParentNode();
                    const auto left = container->GetAbsoluteOffset(Rml::BoxArea::Content).x;
                    const auto button_size = button->GetBox().GetSize(Rml::BoxArea::Border);
                    EXPECT_GE(button->GetAbsoluteOffset(Rml::BoxArea::Border).x, left - 0.5f);
                    EXPECT_LE(button->GetAbsoluteOffset(Rml::BoxArea::Border).x + button_size.x,
                              left + container->GetBox().GetSize(Rml::BoxArea::Content).x + 0.5f);
                    auto* text = rmlui_dynamic_cast<Rml::ElementText*>(button->GetChild(0));
                    ASSERT_NE(text, nullptr);
                    // Containment alone also accepts a one-character-wide column.
                    ASSERT_FALSE(text->GetLines().empty());
                    EXPECT_LE(text->GetLines().size(), 4u);
                    if (std::string(locale) == "en" && width == 780 && scale == 1.0f)
                        EXPECT_EQ(text->GetLines().size(), 1u);
                    std::string joined;
                    for (const auto& line : text->GetLines())
                        joined += line.text;
                    auto expected = label;
                    std::erase(expected, ' ');
                    std::erase(joined, ' ');
                    EXPECT_EQ(joined, expected);
                    doc->Close();
                }
            }
        }
    }

} // namespace
