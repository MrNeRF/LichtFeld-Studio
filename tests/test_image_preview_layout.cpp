/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <RmlUi/Core.h>
#include <RmlUi/Core/ElementDocument.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <map>
#include <string>
#include <vector>

namespace {
    class PreviewLayoutRenderer final : public Rml::RenderInterface {
    public:
        std::vector<Rml::String> loaded_sources;

    private:
        Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex>, Rml::Span<const int>) override { return 1; }
        void RenderGeometry(Rml::CompiledGeometryHandle, Rml::Vector2f, Rml::TextureHandle) override {}
        void ReleaseGeometry(Rml::CompiledGeometryHandle) override {}
        Rml::TextureHandle LoadTexture(Rml::Vector2i& size, const Rml::String& source) override {
            loaded_sources.push_back(source);
            size = {16, 16};
            return 1;
        }
        Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte>, Rml::Vector2i) override { return 1; }
        void ReleaseTexture(Rml::TextureHandle) override {}
        void EnableScissorRegion(bool) override {}
        void SetScissorRegion(Rml::Rectanglei) override {}
    };

    struct Thumb {
        int index = 0;
        bool selected = false;
        Rml::String name = "frame.png", label = "1", decorator = "image(preview://thumb=256&rev=1&path=frame.png contain)";
    };

    class ImagePreviewLayoutTest : public ::testing::TestWithParam<float> {
    protected:
        void SetUp() override {
            ASSERT_TRUE(Rml::Initialise());
            const auto root = std::filesystem::path(PROJECT_ROOT_PATH);
            ASSERT_TRUE(Rml::LoadFontFace((root / "src/visualizer/gui/assets/fonts/Inter-Regular.ttf").string()));
            context = Rml::CreateContext("preview-layout", {900, 700}, &renderer);
            ASSERT_NE(context, nullptr);
            context->SetDensityIndependentPixelRatio(GetParam());
            auto model = context->CreateDataModel("image_preview");
            auto thumb = model.RegisterStruct<Thumb>();
            ASSERT_TRUE(thumb);
            thumb.RegisterMember("index", &Thumb::index);
            thumb.RegisterMember("selected", &Thumb::selected);
            thumb.RegisterMember("name", &Thumb::name);
            thumb.RegisterMember("label", &Thumb::label);
            thumb.RegisterMember("decorator", &Thumb::decorator);
            model.RegisterArray<std::vector<Thumb>>();
            ASSERT_TRUE(model.Bind("thumbs", &thumbs));
            ASSERT_TRUE(model.Bind("panel_label", &title));
            ASSERT_TRUE(model.Bind("filmstrip_top_spacer_height", &top));
            ASSERT_TRUE(model.Bind("filmstrip_bottom_spacer_height", &bottom));
            const auto resources = root / "src/visualizer/gui/rmlui/resources";
            std::ifstream input(resources / "image_preview.rml");
            const std::string source{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
            doc = context->LoadDocumentFromMemory(source,
                                                  Rml::StringUtilities::Replace((resources / "image_preview.rml").generic_string(), ':', '|'));
            ASSERT_NE(doc, nullptr);
            for (const auto& [id, text] : labels)
                doc->GetElementById(id)->SetInnerRML(text);
            doc->Show();
        }
        void TearDown() override {
            Rml::RemoveContext("preview-layout");
            Rml::Shutdown();
        }
        void resize(int width, int height = 600) {
            context->SetDimensions({int(width * GetParam()), int(height * GetParam())});
            context->Update();
            context->Update();
        }
        void checkWithin(Rml::Element* child, Rml::Element* parent) {
            const auto p = parent->GetAbsoluteOffset(Rml::BoxArea::Border);
            const auto size = parent->GetBox().GetSize(Rml::BoxArea::Border);
            const auto c = child->GetAbsoluteOffset(Rml::BoxArea::Border);
            const auto cs = child->GetBox().GetSize(Rml::BoxArea::Border);
            EXPECT_GE(c.x, p.x - 1);
            EXPECT_GE(c.y, p.y - 1);
            EXPECT_LE(c.x + cs.x, p.x + size.x + 1);
            EXPECT_LE(c.y + cs.y, p.y + size.y + 1);
        }
        PreviewLayoutRenderer renderer;
        Rml::Context* context = nullptr;
        Rml::ElementDocument* doc = nullptr;
        std::vector<Thumb> thumbs = std::vector<Thumb>(12);
        Rml::String title = "Dataset · 1 / 10,000", top = "0dp", bottom = "640000dp";
        const std::map<std::string, std::string> labels{
            {"btn-fit", "Adatta"},
            {"btn-info", "Informazioni"},
            {"btn-thumbnails", "Miniature"},
            {"btn-picker", "Contagocce"},
            {"btn-refresh", "Aggiorna"},
            {"btn-help", "Aiuto"},
            {"btn-apply-render-color", "Applica allo sfondo del rendering"},
            {"btn-apply-training-color", "Applica allo sfondo del training"}};
    };

    TEST_P(ImagePreviewLayoutTest, ToolbarWrapsAndLeavesAUsableViewport) {
        for (int width : {600, 760, 1200}) {
            SCOPED_TRACE(width);
            resize(width);
            auto* toolbar = doc->GetElementById("preview-toolbar");
            Rml::ElementList buttons;
            toolbar->QuerySelectorAll(buttons, "button");
            ASSERT_EQ(buttons.size(), 15u);
            for (auto* button : buttons) {
                checkWithin(button, toolbar);
                EXPECT_NEAR(button->GetBox().GetSize(Rml::BoxArea::Border).y, 28 * GetParam(), 1);
                EXPECT_NEAR(button->GetComputedValues().line_height().value,
                            button->GetBox().GetSize(Rml::BoxArea::Content).y, 1);
            }
            Rml::ElementList text_buttons;
            toolbar->QuerySelectorAll(text_buttons, ".btn");
            ASSERT_EQ(text_buttons.size(), 7u);
            for (auto* button : text_buttons)
                EXPECT_GT(button->GetBox().GetSize(Rml::BoxArea::Border).x, 30 * GetParam());
            auto* viewport = doc->GetElementById("image-viewport");
            EXPECT_GT(viewport->GetClientWidth(), 200 * GetParam());
            EXPECT_GT(viewport->GetClientHeight(), 250 * GetParam());
            checkWithin(doc->GetElementById("status-bar"), doc);
        }
    }

    TEST_P(ImagePreviewLayoutTest, StagingLayerLoadsBehindReadyImageWithoutDisplayNone) {
        resize(900);
        auto* ready = doc->GetElementById("main-image-a");
        auto* pending = doc->GetElementById("main-image-b");
        ASSERT_NE(ready, nullptr);
        ASSERT_NE(pending, nullptr);
        ready->SetClass("preview-active", true);
        pending->SetClass("hidden", false);
        for (auto* layer : {ready, pending}) {
            layer->SetProperty("width", "400dp");
            layer->SetProperty("height", "300dp");
        }
        ASSERT_TRUE(ready->SetProperty("decorator", "image(preview://ready)"));
        ASSERT_TRUE(pending->SetProperty("decorator", "image(preview://pending)"));
        context->Update();
        context->Render();
        EXPECT_TRUE(ready->IsVisible(true));
        EXPECT_TRUE(pending->IsVisible(true));
        EXPECT_NE(std::find(renderer.loaded_sources.begin(), renderer.loaded_sources.end(), "preview://ready"),
                  renderer.loaded_sources.end());
        EXPECT_NE(std::find(renderer.loaded_sources.begin(), renderer.loaded_sources.end(), "preview://pending"),
                  renderer.loaded_sources.end());
    }

    TEST_P(ImagePreviewLayoutTest, ThumbnailRowsKeepTheirStrideWhenScrolling) {
        resize(760);
        auto* film = doc->GetElementById("filmstrip");
        Rml::ElementList rows;
        film->QuerySelectorAll(rows, ".thumb-row");
        std::erase_if(rows, [](Rml::Element* row) { return !row->IsVisible(true); });
        ASSERT_EQ(rows.size(), 12u);
        for (size_t i = 1; i < rows.size(); ++i)
            EXPECT_NEAR(rows[i]->GetAbsoluteOffset().y - rows[i - 1]->GetAbsoluteOffset().y,
                        64 * GetParam(), 0.5f);
        film->SetScrollTop(100000 * GetParam());
        context->Update();
        EXPECT_GT(film->GetScrollTop(), 90000 * GetParam());
    }

    TEST_P(ImagePreviewLayoutTest, SampleCommandsStayInsideTheirRow) {
        doc->GetElementById("sampled-color-controls")->SetClass("hidden", false);
        for (int width : {600, 760}) {
            resize(width);
            auto* row = doc->GetElementById("sampled-color-controls");
            checkWithin(doc->GetElementById("btn-apply-render-color"), row);
            checkWithin(doc->GetElementById("btn-apply-training-color"), row);
            EXPECT_GT(doc->GetElementById("image-viewport")->GetClientHeight(), 200 * GetParam());
        }
    }

    TEST_P(ImagePreviewLayoutTest, ThumbnailDecoratorBindsAndUnavailableCameraLooksDisabled) {
        resize(760);
        auto* thumbnail = doc->QuerySelector(".thumb-item[data-index='0']");
        ASSERT_NE(thumbnail, nullptr);
        ASSERT_NE(thumbnail->GetProperty("decorator"), nullptr);
        EXPECT_NE(thumbnail->GetProperty("decorator")->ToString(), "none");
        auto* camera = doc->GetElementById("btn-go-to-camera");
        camera->SetProperty("transition", "none");
        camera->SetAttribute("disabled", "disabled");
        context->Update();
        EXPECT_NEAR(std::stof(camera->GetProperty("opacity")->ToString()), 0.35f, 0.001f);
        EXPECT_EQ(camera->GetProperty("tab-index")->ToString(), "none");
        EXPECT_EQ(camera->GetProperty("pointer-events")->ToString(), "auto");
        camera->SetAttribute("title", "Camera unavailable");
        const auto center = camera->GetAbsoluteOffset(Rml::BoxArea::Border) +
                            camera->GetBox().GetSize(Rml::BoxArea::Border) * 0.5f;
        context->ProcessMouseMove(int(center.x), int(center.y), 0);
        auto* hover = context->GetHoverElement();
        while (hover && hover != camera)
            hover = hover->GetParentNode();
        ASSERT_EQ(hover, camera);
        EXPECT_EQ(hover->GetAttribute<Rml::String>("title", ""), "Camera unavailable");
    }

    TEST_P(ImagePreviewLayoutTest, SelectionScrollCanBeAppliedAfterPublishingVirtualExtent) {
        thumbs.clear();
        bottom = "0dp";
        auto model = context->GetDataModel("image_preview").GetModelHandle();
        model.DirtyAllVariables();
        resize(900, 700);
        auto* film = doc->GetElementById("filmstrip");
        const float requested = 9000 * 64 * GetParam();
        film->SetScrollTop(requested);
        EXPECT_LT(film->GetScrollTop(), requested - 1000);
        top = "576000dp";
        bottom = "63232dp";
        thumbs.resize(12);
        model.DirtyAllVariables();
        context->Update();
        context->Update();
        film->SetScrollTop(requested);
        EXPECT_NEAR(film->GetScrollTop(), requested, 1);
    }

    TEST_P(ImagePreviewLayoutTest, LoadingAndErrorMessagesHaveVisibleCenteredText) {
        auto* label = doc->GetElementById("preview-state-text");
        ASSERT_NE(label, nullptr);
        label->SetInnerRML("Unable to load image. Refresh to retry.");
        resize(760);
        auto* viewport = doc->GetElementById("image-viewport");
        EXPECT_GT(label->GetBox().GetSize(Rml::BoxArea::Border).x, 80 * GetParam());
        EXPECT_GT(label->GetBox().GetSize(Rml::BoxArea::Border).y, 12 * GetParam());
        checkWithin(label, viewport);
    }

    INSTANTIATE_TEST_SUITE_P(UiScales, ImagePreviewLayoutTest, ::testing::Values(1.0f, 1.5f, 2.0f));
} // namespace
