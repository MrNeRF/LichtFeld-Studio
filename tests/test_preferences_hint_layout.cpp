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

} // namespace
