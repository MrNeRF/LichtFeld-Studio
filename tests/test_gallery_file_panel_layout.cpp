/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <RmlUi/Core.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/RenderInterface.h>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>
#include <map>
#include <string>
#include <vector>

namespace {

    class StubRenderInterface final : public Rml::RenderInterface {
    public:
        Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex>, Rml::Span<const int>) override {
            return 1;
        }
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

    const std::filesystem::path kResources =
        std::filesystem::path(PROJECT_ROOT_PATH) / "src/visualizer/gui/rmlui/resources";

    // Catches the publish dialog giving its checkboxes the text field height (16x24 instead of the app's
    // square checkbox) and top-aligning button labels in fixed-height buttons.
    class GalleryFilePanelLayoutTest : public ::testing::TestWithParam<float> {
    protected:
        static void SetUpTestSuite() {
            ASSERT_TRUE(Rml::Initialise());
            ASSERT_TRUE(Rml::LoadFontFace(
                (std::filesystem::path(PROJECT_ROOT_PATH) / "src/visualizer/gui/assets/fonts/Inter-Regular.ttf").string()));
        }
        static void TearDownTestSuite() { Rml::Shutdown(); }

        void SetUp() override {
            context_ = Rml::CreateContext("gallery_file_panel_layout", {900, 1200}, &renderer_);
            ASSERT_NE(context_, nullptr);
            context_->SetDensityIndependentPixelRatio(GetParam());
            auto model = context_->CreateDataModel("gallery_file");
            for (auto& [name, value] : publish_review_)
                ASSERT_TRUE(model.Bind(name, &value));
            for (auto& [name, value] : labels_)
                ASSERT_TRUE(model.Bind(name, &value));
            document_ = context_->LoadDocument((kResources / "gallery_file_panel.rml").string());
            ASSERT_NE(document_, nullptr);
            document_->Show();
            reference_ = context_->LoadDocumentFromMemory(
                R"(<rml><head><link type="text/rcss" href="components.rcss"/></head><body><input id="reference" type="checkbox"/></body></rml>)",
                (kResources / "reference.rml").string());
            ASSERT_NE(reference_, nullptr);
            reference_->Show();
            context_->Update();
        }

        void TearDown() override { ASSERT_TRUE(Rml::RemoveContext("gallery_file_panel_layout")); }

        std::vector<Rml::Element*> visible(const std::string& selector) {
            Rml::ElementList elements;
            document_->QuerySelectorAll(elements, selector);
            std::erase_if(elements, [](Rml::Element* element) { return !element->IsVisible(true); });
            return elements;
        }

        inline static StubRenderInterface renderer_;
        std::map<std::string, bool> publish_review_{
            {"is_publish", true},
            {"show_format", true},
            {"show_cover", true},
            {"can_cover", true},
            {"show_save_project", true},
            {"use_cover", false},
            {"save_project", true},
            {"can_submit", true},
            {"is_pull", false},
            {"is_conflict", false},
            {"is_replacement", false},
            {"show_local_apply", false},
            {"show_prepared_copy", true},
            {"show_unlinked_hint", false},
            {"show_unsaved_hint", false},
            {"waiting", false},
            {"has_error", false}};
        std::map<std::string, Rml::String> labels_{
            {"panel_label", "Publish..."},
            {"file_name", "project.licht"},
            {"title", "Playground"},
            {"description", ""},
            {"upload_format", "sog"},
            {"submit_label", "Publish"},
            {"cancel_label", "Cancel"},
            {"unlinked_copy", ""},
            {"g_review_title", ""},
            {"g_review_description", ""},
            {"g_review_upload_as", ""},
            {"g_format_studio", ""},
            {"g_format_sog", ""},
            {"g_format_ssog", ""},
            {"g_format_spz", ""},
            {"format_hint", ""},
            {"estimate", ""},
            {"includes", ""},
            {"eligibility_reason", ""},
            {"g_info_folder", ""},
            {"pull_folder", ""},
            {"g_info_filename", ""},
            {"pull_name", ""},
            {"checkpoint_warning", ""},
            {"replacement_warning", ""},
            {"quota", ""},
            {"warning", ""},
            {"connection_reason", ""},
            {"error", ""}};
        Rml::Context* context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        Rml::ElementDocument* reference_ = nullptr;
    };

    TEST_P(GalleryFilePanelLayoutTest, CheckboxesMatchTheAppCheckbox) {
        const auto expected = reference_->GetElementById("reference")->GetBox().GetSize(Rml::BoxArea::Border);
        ASSERT_GT(expected.x, 0.0f);
        const auto checkboxes = visible(".gallery-cover-option input");
        ASSERT_EQ(checkboxes.size(), 2u);
        for (auto* checkbox : checkboxes) {
            const auto size = checkbox->GetBox().GetSize(Rml::BoxArea::Border);
            EXPECT_FLOAT_EQ(size.x, expected.x);
            EXPECT_FLOAT_EQ(size.y, expected.y);
        }
    }

    TEST_P(GalleryFilePanelLayoutTest, ButtonLabelsAreVerticallyCentered) {
        const auto buttons = visible(".gallery-file-actions .btn");
        ASSERT_EQ(buttons.size(), 2u);
        for (auto* button : buttons) {
            const float content_height = button->GetBox().GetSize(Rml::BoxArea::Content).y;
            ASSERT_GT(content_height, 0.0f);
            const float label_center_offset = (button->GetComputedValues().line_height().value - content_height) / 2.0f;
            EXPECT_LE(std::abs(label_center_offset), 0.5f);
        }
    }

    INSTANTIATE_TEST_SUITE_P(UiScales, GalleryFilePanelLayoutTest, ::testing::Values(1.0f, 1.5f, 2.0f));

} // namespace
