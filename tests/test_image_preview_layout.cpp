/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <RmlUi/Core.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/FontEngineInterface.h>
#include <algorithm>
#include <array>
#include <core/event_bridge/localization_manager.hpp>
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
        Rml::String mask_decorator = "image(preview://memory=sam:fixture:0:1&thumb=256&mode=mask contain)";
    };

    struct SemanticCategory {
        Rml::String value = "person", label = "Person", remove_label = "Remove Person";
        bool disabled = false;
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
            thumb.RegisterMember("mask_decorator", &Thumb::mask_decorator);
            model.RegisterArray<std::vector<Thumb>>();
            auto category = model.RegisterStruct<SemanticCategory>();
            ASSERT_TRUE(category);
            category.RegisterMember("value", &SemanticCategory::value);
            category.RegisterMember("label", &SemanticCategory::label);
            category.RegisterMember("remove_label", &SemanticCategory::remove_label);
            category.RegisterMember("disabled", &SemanticCategory::disabled);
            model.RegisterArray<std::vector<SemanticCategory>>();
            ASSERT_TRUE(model.Bind("semantic_categories", &categories));
            ASSERT_TRUE(model.Bind("semantic_text", &semantic_text));
            ASSERT_TRUE(model.Bind("semantic_confidence", &semantic_confidence));
            ASSERT_TRUE(model.BindEventCallback("remove_semantic_category",
                                                [](Rml::DataModelHandle, Rml::Event&, const Rml::VariantList&) {}));
            model_handle = model.GetModelHandle();
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
            lfs::event::LocalizationManager::getInstance().reset();
            Rml::RemoveContext("preview-layout");
            Rml::Shutdown();
        }
        void resize(int width, int height = 600) {
            context->SetDimensions({int(width * GetParam()), int(height * GetParam())});
            context->Update();
            context->Update();
        }
        void checkWithin(Rml::Element* child, Rml::Element* parent) {
            SCOPED_TRACE(::testing::Message() << child->GetId() << " in " << parent->GetId()
                                              << " (width " << parent->GetClientWidth() << ")");
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
        Rml::DataModelHandle model_handle;
        std::vector<Thumb> thumbs = std::vector<Thumb>(12);
        std::vector<SemanticCategory> categories = std::vector<SemanticCategory>(1);
        Rml::String semantic_text = "person";
        int semantic_confidence = 40;
        Rml::String title = "Dataset · 1 / 10,000", top = "0dp", bottom = "640000dp";
        const std::map<std::string, std::string> labels{
            {"btn-fit", "Adatta"},
            {"btn-info", "Informazioni"},
            {"btn-thumbnails", "Miniature"},
            {"btn-picker", "Contagocce"},
            {"btn-sam", "SAM2.1"},
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
            ASSERT_EQ(buttons.size(), 16u);
            for (auto* button : buttons) {
                checkWithin(button, toolbar);
                EXPECT_NEAR(button->GetBox().GetSize(Rml::BoxArea::Border).y, 28 * GetParam(), 1);
                EXPECT_NEAR(button->GetComputedValues().line_height().value,
                            button->GetBox().GetSize(Rml::BoxArea::Content).y, 1);
            }
            Rml::ElementList text_buttons;
            toolbar->QuerySelectorAll(text_buttons, ".btn");
            ASSERT_EQ(text_buttons.size(), 8u);
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

    TEST_P(ImagePreviewLayoutTest, SamOverlayAndPromptsStayAboveEitherReadyLayer) {
        auto* overlay = doc->GetElementById("sam-overlay");
        auto* markers = doc->GetElementById("sam-markers");
        ASSERT_NE(overlay, nullptr);
        ASSERT_NE(markers, nullptr);
        overlay->SetClass("hidden", false);
        for (const auto* active_id : {"main-image-a", "main-image-b"}) {
            for (const auto* layer_id : {"main-image-a", "main-image-b"}) {
                auto* layer = doc->GetElementById(layer_id);
                const bool active = Rml::String(layer_id) == active_id;
                layer->SetClass("preview-active", active);
                layer->SetClass("hidden", !active);
            }
            context->Update();
            EXPECT_GT(overlay->GetComputedValues().z_index().value,
                      doc->GetElementById(active_id)->GetComputedValues().z_index().value);
            EXPECT_GT(markers->GetComputedValues().z_index().value, overlay->GetComputedValues().z_index().value);
        }
    }

    TEST_P(ImagePreviewLayoutTest, SamControlsWrapWithoutCoveringTheCanvas) {
        auto* controls = doc->GetElementById("sam-controls");
        controls->SetClass("hidden", false);
        auto* apply = doc->GetElementById("sam-apply-controls");
        apply->SetClass("hidden", false);
        auto* sidebar = doc->GetElementById("sidebar");
        sidebar->SetClass("sam-sidebar", true);
        doc->GetElementById("metadata-info")->SetClass("hidden", true);
        doc->GetElementById("sam-markers")->SetInnerRML("<div id='test-sam-box' class='sam-box'></div>");
        const std::map<std::string, std::string> sam_keys{
            {"sam-method-manual", "sam_method_manual"},
            {"sam-semantic-heading", "sam_semantic_heading"},
            {"sam-semantic-descriptions", "sam_semantic_descriptions"},
            {"sam-semantic-help", "sam_semantic_help"},
            {"sam-semantic-find", "sam_semantic_find"},
            {"sam-semantic-advanced", "sam_semantic_advanced"},
            {"sam-semantic-confidence-label", "sam_semantic_confidence"},
            {"sam-positive", "sam_positive"},
            {"sam-negative", "sam_negative"},
            {"sam-box", "sam_box"},
            {"sam-pan", "sam_pan"},
            {"sam-remove", "sam_remove"},
            {"sam-flip", "sam_flip"},
            {"sam-show", "sam_show"},
            {"sam-undo", "sam_undo"},
            {"sam-clear", "sam_clear"},
            {"sam-generate", "sam_generate"},
            {"sam-refine", "sam_refine"},
            {"sam-cancel", "sam_cancel"},
            {"sam-apply", "sam_apply"},
            {"sam-exclude", "sam_exclude"},
            {"sam-heading", "sam_heading"},
            {"sam-proposals-heading", "sam_proposals_heading"},
            {"sam-apply-heading", "sam_apply_heading"},
            {"sam-edit-hint", "sam_edit_hint"},
            {"sam-quality", "sam_quality"},
            {"sam-status", "sam_ready"},
            {"sam-hint", "sam_hint"}};
        const auto root = std::filesystem::path(PROJECT_ROOT_PATH);
        for (const char* font : {"NotoSansJP-Regular.ttf", "NotoSansKR-Regular.ttf"})
            ASSERT_TRUE(Rml::LoadFontFace((root / "src/visualizer/gui/assets/fonts" / font).string(), true));
        auto& locales = lfs::event::LocalizationManager::getInstance();
        ASSERT_TRUE(locales.initialize((root / "src/visualizer/gui/resources/locales").string()));
        for (const auto& language : locales.getAvailableLanguages()) {
            SCOPED_TRACE(language);
            ASSERT_TRUE(locales.setLanguage(language));
            // Several long selected categories must scroll without moving Find/Apply.
            const std::array category_keys{"person", "truck", "plant", "backpack", "suitcase", "motorcycle"};
            categories.resize(category_keys.size());
            for (size_t i = 0; i < category_keys.size(); ++i) {
                categories[i].value = category_keys[i];
                categories[i].label = locales.get("image_preview.sam_category_" + std::string(category_keys[i]));
                categories[i].remove_label = categories[i].label;
            }
            model_handle.DirtyVariable("semantic_categories");
            std::map<std::string, std::string> sam_labels{
                {"sam-candidate-0", "1"},
                {"sam-candidate-1", "2"},
                {"sam-candidate-2", "3"}};
            for (const auto& [id, key] : sam_keys)
                sam_labels[id] = locales.get("image_preview." + key);
            sam_labels["sam-quality"] = Rml::StringUtilities::Replace(sam_labels["sam-quality"], "{score}", "97");
            for (const auto& [id, text] : sam_labels) {
                auto* element = doc->GetElementById(id);
                ASSERT_NE(element, nullptr) << id;
                element->SetInnerRML(text);
            }
            doc->GetElementById("btn-sam")->SetInnerRML(locales.get("image_preview.sam_title"));
            auto* selection = doc->GetElementById("sam-selection");
            selection->SetInnerRML("<option value='0'>1</option><option value='box'>Box</option>");
            for (const bool semantic : {false, true}) {
                SCOPED_TRACE(semantic);
                doc->GetElementById("sam-semantic-controls")->SetClass("hidden", !semantic);
                doc->GetElementById("sam-semantic-options")->SetClass("hidden", !semantic);
                doc->GetElementById("sam-manual-controls")->SetClass("hidden", semantic);
                doc->GetElementById("sam-semantic-find-row")->SetClass("hidden", !semantic);
                for (const char* id : {"sam-proposals-heading", "sam-generate-row", "sam-candidates-row", "sam-quality"})
                    doc->GetElementById(id)->SetClass("hidden", semantic);
                const char* action_id = semantic ? "sam-semantic-find" : "sam-generate";
                for (const bool help : {false, true}) {
                    SCOPED_TRACE(help);
                    doc->GetElementById("shortcut-help")->SetClass("hidden", !help);
                    for (int width : {600, 760, 900, 1200}) {
                        SCOPED_TRACE(width);
                        doc->GetElementById("sam-pages")->SetClass("hidden", width != 900);
                        doc->GetElementById("sam-heading")->SetClass("hidden", width == 900);
                        resize(width);
                        EXPECT_NEAR(doc->GetElementById("test-sam-box")->GetComputedValues().border_top_width(), 2 * GetParam(), 1);
                        Rml::ElementList buttons;
                        controls->QuerySelectorAll(buttons, "button");
                        Rml::ElementList apply_buttons;
                        apply->QuerySelectorAll(apply_buttons, "button");
                        std::erase_if(buttons, [](auto* button) { return !button->IsVisible(true); });
                        std::erase_if(apply_buttons, [](auto* button) { return !button->IsVisible(true); });
                        buttons.insert(buttons.end(), apply_buttons.begin(), apply_buttons.end());
                        ASSERT_FALSE(buttons.empty());
                        const auto mode_buttons = semantic ? std::array{"sam-semantic-find", "sam-semantic-advanced"} : std::array{"sam-generate", "sam-refine"};
                        for (const auto* id : mode_buttons) {
                            EXPECT_NE(std::find(buttons.begin(), buttons.end(), doc->GetElementById(id)), buttons.end()) << id;
                        }
                        checkWithin(apply, sidebar);
                        EXPECT_NEAR(apply->GetAbsoluteOffset(Rml::BoxArea::Border).y + apply->GetBox().GetSize(Rml::BoxArea::Border).y,
                                    sidebar->GetAbsoluteOffset(Rml::BoxArea::Border).y + sidebar->GetBox().GetSize(Rml::BoxArea::Border).y, 1.5f * GetParam());
                        const float apply_top = apply->GetAbsoluteOffset(Rml::BoxArea::Border).y;
                        checkWithin(doc->GetElementById("btn-sam"), doc->GetElementById("preview-toolbar"));
                        for (auto* button : buttons) {
                            button->ScrollIntoView(false);
                            context->Update();
                            EXPECT_NEAR(apply->GetAbsoluteOffset(Rml::BoxArea::Border).y, apply_top, 0.5f);
                            checkWithin(button, std::find(apply_buttons.begin(), apply_buttons.end(), button) != apply_buttons.end() ? apply : controls);
                            EXPECT_NEAR(button->GetBox().GetSize(Rml::BoxArea::Border).y, 28 * GetParam(), 1);
                            EXPECT_GE(button->GetBox().GetSize(Rml::BoxArea::Border).x, 28 * GetParam() - 1);
                            const auto text_width = Rml::GetFontEngineInterface()->GetStringWidth(
                                button->GetFontFaceHandle(), button->GetInnerRML(),
                                Rml::TextShapingContext{button->GetComputedValues().language()});
                            EXPECT_GE(button->GetBox().GetSize(Rml::BoxArea::Content).x + 1, text_width);
                        }
                        EXPECT_GT(doc->GetElementById("image-viewport")->GetClientHeight(), 180 * GetParam());
                        EXPECT_GT(doc->GetElementById("image-viewport")->GetClientWidth(), 200 * GetParam());

                        // Inference status, empty scores and the proposal legend must not move
                        // the canvas or the controls under the pointer between operations.
                        controls->SetScrollTop(0);
                        context->Update();
                        if (width == 900 && !help) {
                            checkWithin(doc->GetElementById(action_id), apply);
                            for (const char* id : {"sam-cancel", "sam-show"})
                                checkWithin(doc->GetElementById(id), apply);
                        }
                        const auto geometry = [&](const char* id) {
                            auto* element = doc->GetElementById(id);
                            return Rml::Rectanglef::FromPositionSize(element->GetAbsoluteOffset(Rml::BoxArea::Border),
                                                                     element->GetBox().GetSize(Rml::BoxArea::Border));
                        };
                        const auto viewport_bounds = geometry("image-viewport");
                        const auto generate_bounds = geometry(action_id);
                        const auto apply_bounds = geometry("sam-apply");
                        for (const char* phase : {"sam_generate_hint", "sam_loading", "sam_ready", "sam_closing", "sam_error"}) {
                            SCOPED_TRACE(phase);
                            const bool ready = Rml::String(phase) == "sam_ready";
                            std::string text = locales.get("image_preview." + Rml::String(phase));
                            if (Rml::String(phase) == "sam_error") {
                                const auto detail = text;
                                for (int i = 0; i < 20; ++i)
                                    text += " " + detail;
                            }
                            auto* status = doc->GetElementById("sam-status");
                            status->SetInnerRML(text);
                            status->SetScrollTop(0);
                            doc->GetElementById("sam-quality")->SetInnerRML(ready ? sam_labels.at("sam-quality") : "");
                            auto* legend = doc->GetElementById("mask-legend");
                            legend->SetInnerRML(locales.get("image_preview.sam_proposal_legend"));
                            legend->SetClass("hidden", !ready);
                            context->Update();
                            EXPECT_EQ(geometry("image-viewport"), viewport_bounds);
                            EXPECT_EQ(geometry(action_id), generate_bounds);
                            EXPECT_EQ(geometry("sam-apply"), apply_bounds);
                            checkWithin(status, apply);
                            if (ready) {
                                checkWithin(legend, doc->GetElementById("status-bar"));
                                EXPECT_GT(legend->GetClientWidth(), 100 * GetParam());
                                EXPECT_NEAR(legend->GetClientHeight(), 18 * GetParam(), 0.5f);
                            }
                        }
                    }
                }
            }
            doc->GetElementById("metadata-info")->SetClass("hidden", false);
            resize(760);
            checkWithin(apply, sidebar);
            EXPECT_NEAR(apply->GetAbsoluteOffset(Rml::BoxArea::Border).y + apply->GetBox().GetSize(Rml::BoxArea::Border).y,
                        sidebar->GetAbsoluteOffset(Rml::BoxArea::Border).y + sidebar->GetBox().GetSize(Rml::BoxArea::Border).y, 1.5f * GetParam());
            doc->GetElementById("metadata-info")->SetClass("hidden", true);
        }
    }

    TEST_P(ImagePreviewLayoutTest, TrackingStatusAndLocalizedSaveActionsKeepTheCanvasStable) {
        const auto root = std::filesystem::path(PROJECT_ROOT_PATH);
        for (const char* font : {"NotoSansJP-Regular.ttf", "NotoSansKR-Regular.ttf"})
            ASSERT_TRUE(Rml::LoadFontFace((root / "src/visualizer/gui/assets/fonts" / font).string(), true));
        auto& locales = lfs::event::LocalizationManager::getInstance();
        ASSERT_TRUE(locales.initialize((root / "src/visualizer/gui/resources/locales").string()));
        auto* controls = doc->GetElementById("sam-controls");
        auto* apply = doc->GetElementById("sam-apply-controls");
        auto* sidebar = doc->GetElementById("sidebar");
        sidebar->SetClass("sam-sidebar", true);
        doc->GetElementById("metadata-info")->SetClass("hidden", true);
        for (const auto* id : {"sam-pages", "sam-controls", "sam-apply-controls", "sam-tracking", "sam-save-controls"})
            doc->GetElementById(id)->SetClass("hidden", false);
        for (const auto* id : {"sam-frame-controls", "sam-generate-row", "sam-apply", "sam-exclude"})
            doc->GetElementById(id)->SetClass("hidden", true);
        for (const auto& language : locales.getAvailableLanguages()) {
            SCOPED_TRACE(language);
            ASSERT_TRUE(locales.setLanguage(language));
            doc->GetElementById("sam-page-frame")->SetInnerRML(locales.get("image_preview.sam_page_frame"));
            doc->GetElementById("sam-page-sequence")->SetInnerRML(locales.get("image_preview.sam_page_sequence"));
            for (const auto& [id, key] : std::map<std::string, std::string>{{"sam-track", "sam_track"},
                                                                            {"sam-approve-range", "sam_approve_range"},
                                                                            {"sam-save", "sam_save_approved"},
                                                                            {"sam-save-exclude", "sam_save_excluded"},
                                                                            {"sam-apply-heading", "sam_save_heading"},
                                                                            {"sam-approve-label", "sam_approve"},
                                                                            {"sam-tracking-heading", "sam_tracking_heading"},
                                                                            {"sam-track-first-label", "sam_range_first"},
                                                                            {"sam-track-last-label", "sam_range_last"},
                                                                            {"sam-track-direction-label", "sam_direction"}})
                doc->GetElementById(id)->SetInnerRML(locales.get("image_preview." + key));
            doc->GetElementById("sam-apply-heading")->SetInnerRML(std::string(locales.get("image_preview.sam_save_heading")) + " · 999 / 1000");
            doc->GetElementById("sam-track-direction")->SetInnerRML("<option>" + Rml::String(locales.get("image_preview.sam_both")) + "</option>");
            for (int width : {900, 600, 760}) {
                controls->SetScrollTop(0);
                resize(width);
                const auto viewport = doc->GetElementById("image-viewport")->GetBox().GetSize();
                const auto apply_top = apply->GetAbsoluteOffset().y;
                for (const auto* id : {"sam-track-first", "sam-track-last", "sam-track-direction"}) {
                    auto* field = doc->GetElementById(id);
                    if (width != 900)
                        field->ScrollIntoView(false);
                    context->Update();
                    checkWithin(field, controls);
                }
                for (const auto* id : {"sam-track", "sam-approve-range", "sam-save", "sam-save-exclude"}) {
                    auto* button = doc->GetElementById(id);
                    if (width != 900)
                        button->ScrollIntoView(false);
                    context->Update();
                    checkWithin(button, id == Rml::String("sam-track") || id == Rml::String("sam-approve-range") ? controls : apply);
                    const auto key = id == Rml::String("sam-track") ? "sam_track" : id == Rml::String("sam-approve-range") ? "sam_approve_range"
                                                                                : id == Rml::String("sam-save")            ? "sam_save_approved"
                                                                                                                           : "sam_save_excluded";
                    const auto text_width = Rml::GetFontEngineInterface()->GetStringWidth(button->GetFontFaceHandle(), Rml::String(locales.get("image_preview." + Rml::String(key))),
                                                                                          Rml::TextShapingContext{button->GetComputedValues().language()});
                    EXPECT_GE(button->GetBox().GetSize(Rml::BoxArea::Content).x + 1, text_width);
                }
                for (const auto* phase : {"sam_tracking", "sam_saved", "sam_error"}) {
                    doc->GetElementById("sam-status")->SetInnerRML(locales.get("image_preview." + Rml::String(phase)));
                    context->Update();
                    EXPECT_EQ(doc->GetElementById("image-viewport")->GetBox().GetSize(), viewport);
                    EXPECT_NEAR(apply->GetAbsoluteOffset().y, apply_top, .5f);
                    checkWithin(apply, sidebar);
                }
            }
        }
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
