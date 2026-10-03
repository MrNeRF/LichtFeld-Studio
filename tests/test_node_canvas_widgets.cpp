/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/nodes/tree.hpp"
#include "gui/rmlui/elements/color_picker_element.hpp"
#include "gui/rmlui/elements/colour_offset_element.hpp"
#include "gui/rmlui/elements/node_canvas_dom.hpp"
#include "gui/rmlui/elements/node_canvas_element.hpp"
#include "gui/rmlui/elements/node_canvas_widgets.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rml_tooltip.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "scene/scene_manager.hpp"
#include "theme/theme.hpp"
#include "visualizer/nodes/modifier_manager.hpp"
#include "visualizer/operation/undo_history.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/ElementInstancer.h>
#include <RmlUi/Core/ElementText.h>
#include <RmlUi/Core/Elements/ElementFormControlInput.h>
#include <SDL3/SDL_scancode.h>
#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>
#include <thread>

namespace {
    class WidgetRenderer final : public Rml::RenderInterface {
    public:
        Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex>, Rml::Span<const int>) override { return 1; }
        void RenderGeometry(Rml::CompiledGeometryHandle, Rml::Vector2f, Rml::TextureHandle) override {}
        void ReleaseGeometry(Rml::CompiledGeometryHandle) override {}
        Rml::TextureHandle LoadTexture(Rml::Vector2i&, const Rml::String&) override { return 0; }
        Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte>, Rml::Vector2i) override { return 0; }
        void ReleaseTexture(Rml::TextureHandle) override {}
        void EnableScissorRegion(bool) override {}
        void SetScissorRegion(Rml::Rectanglei) override {}
    };

    class NodeCanvasWidgets : public ::testing::Test {
    protected:
        static lfs::vis::SceneManager isolatedScene() {
            // Earlier fixtures also create SceneManagers with global handlers.
            // Clear them before installing this fixture's scene handlers.
            lfs::event::EventBridge::instance().clear_all();
            return lfs::vis::SceneManager{};
        }

        void SetUp() override {
            lfs::vis::op::undoHistory().clear();
            ASSERT_TRUE(Rml::Initialise());
            const auto resources = std::filesystem::path(PROJECT_ROOT_PATH) / "src/visualizer/gui";
            auto& localization = lfs::event::LocalizationManager::getInstance();
            localization.reset();
            ASSERT_TRUE(localization.initialize((resources / "resources/locales").string()));
            ASSERT_TRUE(Rml::LoadFontFace((resources / "assets/fonts/Inter-Regular.ttf").string()));
            Rml::Factory::RegisterElementInstancer("node-canvas", &instancer_);
            Rml::Factory::RegisterElementInstancer("colour-offset", &offset_instancer_);
            Rml::Factory::RegisterElementInstancer("color-picker", &picker_instancer_);
            context_ = Rml::CreateContext("node_widgets", {1000, 700}, &renderer_);
            ASSERT_NE(context_, nullptr);
            context_->SetDensityIndependentPixelRatio(2.0f);
            const auto styles = resources / "rmlui/resources";
            document_ = context_->LoadDocumentFromMemory(
                "<rml><head><link type='text/rcss' href='components.rcss'/>"
                "<link type='text/rcss' href='node_editor.rcss'/></head>"
                "<body><node-canvas id='node-editor-canvas'/></body></rml>",
                (styles / "node_editor.rml").string());
            ASSERT_NE(document_, nullptr);
            auto* canvas = dynamic_cast<lfs::vis::gui::NodeCanvasElement*>(document_->GetElementById("node-editor-canvas"));
            ASSERT_NE(canvas, nullptr);
            scene_.changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);
            canvas->setContext(&scene_, nullptr);
            document_->Show();
            context_->Update();
            auto container = document_->CreateElement("div");
            container->SetProperty("position", "absolute");
            container->SetProperty("left", "20dp");
            container->SetProperty("top", "20dp");
            container->SetProperty("width", "200dp");
            const lfs::nodes::Node node{.name = "Test"};
            const lfs::nodes::SocketDecl socket{
                .identifier = "Value",
                .label = "Value",
                .type = std::string(lfs::nodes::FLOAT_SOCKET),
                .default_value = 0.25f,
                .min = 0.0,
                .max = 1.0,
                .step = 0.01};
            container->SetInnerRML(lfs::vis::gui::node_widgets::input(node, socket, true));
            field_ = container->QuerySelector(".node-scrub");
            input_ = dynamic_cast<Rml::ElementFormControlInput*>(container->QuerySelector("input"));
            canvas->AppendChild(std::move(container));
            context_->Update();
            ASSERT_NE(field_, nullptr);
            ASSERT_NE(input_, nullptr);
        }

        void TearDown() override {
            lfs::vis::setTheme(original_theme_);
            lfs::event::LocalizationManager::getInstance().reset();
            lfs::vis::op::undoHistory().clear();
            // SceneManager's process-lifetime handlers must not outlive this fixture.
            lfs::event::EventBridge::instance().clear_all();
            Rml::RemoveContext("node_widgets");
            Rml::Shutdown();
        }

        void attachGraph() {
            using lfs::core::Device;
            using lfs::core::Tensor;
            context_->SetDimensions({2200, 1600});
            const auto id = scene_.getScene().addPointCloud("Host", std::make_shared<lfs::core::PointCloud>(
                                                                        Tensor::zeros({1, 3}, Device::CPU), Tensor::ones({1, 3}, Device::CPU)));
            host_ = scene_.getScene().getNodeUuid(id);
            scene_.selectNode(id);
            auto& manager = scene_.modifierManager();
            auto& tree = manager.newTree("Interaction");
            tree_ = tree.uuid;
            const auto input = tree.input_node().name;
            const auto output = tree.output_node().name;
            tree.find_node(input)->location = {20, 20};
            tree.find_node(output)->location = {530, 20};
            tree.add_node("lfs.colour_correct", "Correct").location = {270, 20};
            tree.add_node("lfs.value", "Value").location = {20, 280};
            tree.remove_link({input, "Geometry", output, "Geometry"});
            ASSERT_TRUE(tree.add_link({input, "Geometry", "Correct", "Geometry"}));
            ASSERT_TRUE(tree.add_link({"Correct", "Geometry", output, "Geometry"}));
            manager.addModifier(host_, tree.uuid);
            ASSERT_TRUE(manager.evaluate(host_).ok);
            context_->Update();
            auto* canvas = dynamic_cast<lfs::vis::gui::NodeCanvasElement*>(document_->GetElementById("node-editor-canvas"));
            canvas->setView({}, 1.0f);
            context_->Update();
            (void)manager.performance(true);
        }

        void pointer(const char* event, const float x, const float y, const int button = 0) {
            auto* canvas = document_->GetElementById("node-editor-canvas");
            Rml::Dictionary parameters;
            parameters["mouse_x"] = x;
            parameters["mouse_y"] = y;
            parameters["button"] = button;
            canvas->DispatchEvent(event, parameters);
            context_->Update();
            scene_.modifierManager().tick();
        }

        void expectNoEvaluation() {
            const auto metrics = scene_.modifierManager().performance();
            EXPECT_EQ(metrics["requests"], 0);
            EXPECT_EQ(metrics["evaluations"], 0);
        }

        WidgetRenderer renderer_;
        const lfs::vis::Theme original_theme_ = lfs::vis::theme();
        Rml::ElementInstancerGeneric<lfs::vis::gui::NodeCanvasElement> instancer_;
        Rml::ElementInstancerGeneric<lfs::vis::gui::ColourOffsetElement> offset_instancer_;
        Rml::ElementInstancerGeneric<lfs::vis::gui::ColorPickerElement> picker_instancer_;
        lfs::vis::SceneManager scene_ = isolatedScene();
        Rml::Context* context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        Rml::Element* field_ = nullptr;
        Rml::ElementFormControlInput* input_ = nullptr;
        lfs::core::Uuid host_;
        std::string tree_;
    };

    TEST_F(NodeCanvasWidgets, ShiftDuplicateUsesClipboardPathAndKeepsInternalLinks) {
        attachGraph();
        auto& manager = scene_.modifierManager();
        auto* tree = manager.tree(tree_);
        ASSERT_TRUE(tree->add_link({"Value", "Value", "Correct", "Exposure"}));
        manager.markDirty();
        context_->Update();
        auto* canvas = static_cast<lfs::vis::gui::NodeCanvasElement*>(document_->GetElementById("node-editor-canvas"));
        ASSERT_TRUE(canvas->selectNodes({"Value", "Correct"}, std::nullopt));
        ASSERT_TRUE(canvas->handleKey(SDL_SCANCODE_D, true, false, false));
        context_->Update();
        const auto selection = canvas->viewState()["selected_nodes"].get<std::unordered_set<std::string>>();
        ASSERT_EQ(selection.size(), 2u);
        EXPECT_EQ(std::ranges::count_if(tree->links, [&](const auto& link) {
            return selection.contains(link.from_node) && selection.contains(link.to_node);
        }), 1);
    }

    TEST_F(NodeCanvasWidgets, ControlJWrapsSelectionInOneUndoStep) {
        attachGraph();
        auto* canvas = static_cast<lfs::vis::gui::NodeCanvasElement*>(document_->GetElementById("node-editor-canvas"));
        ASSERT_TRUE(canvas->selectNodes({"Value", "Correct"}, std::nullopt));
        lfs::vis::op::undoHistory().clear();
        ASSERT_TRUE(canvas->handleKey(SDL_SCANCODE_J, false, true, false));
        auto* tree = scene_.modifierManager().tree(tree_);
        const auto frame = tree->find_node("Correct")->ui.value("frame", std::string{});
        EXPECT_FALSE(frame.empty());
        EXPECT_EQ(tree->find_node("Value")->ui["frame"], frame);
        EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 1u);
    }

    TEST_F(NodeCanvasWidgets, DroppingInsideFrameJoinsAndDraggingOutsideLeaves) {
        attachGraph();
        auto& manager = scene_.modifierManager();
        manager.tree(tree_)->add_node("lfs.frame", "Frame").location = {100, 400};
        manager.markDirty();
        context_->Update();
        pointer("mousedown", 80, 580);
        pointer("mousemove", 340, 880);
        pointer("mouseup", 340, 880);
        EXPECT_EQ(manager.tree(tree_)->find_node("Value")->ui.value("frame", ""), "Frame");
        pointer("mousedown", 340, 880);
        pointer("mousemove", 900, 300);
        pointer("mouseup", 900, 300);
        EXPECT_FALSE(manager.tree(tree_)->find_node("Value")->ui.contains("frame"));
    }

    TEST_F(NodeCanvasWidgets, FrameDragMovesMembersLiveAndKeepsMembership) {
        attachGraph();
        auto& manager = scene_.modifierManager();
        ASSERT_TRUE(manager.frameWrap(tree_, {"Value"}, "Frame"));
        context_->Update();
        auto* canvas = static_cast<lfs::vis::gui::NodeCanvasElement*>(document_->GetElementById("node-editor-canvas"));
        auto* frame = canvas->QuerySelector(".node-frame");
        auto* member = canvas->QuerySelector(".node-box[data-node=Value]");
        ASSERT_NE(frame, nullptr);
        ASSERT_NE(member, nullptr);
        const auto origin = frame->GetAbsoluteOffset(Rml::BoxArea::Border);
        const auto member_before = member->GetAbsoluteOffset(Rml::BoxArea::Border);
        const auto location = manager.tree(tree_)->find_node("Value")->location;
        pointer("mousedown", origin.x + 70, origin.y + 22);
        pointer("mousemove", origin.x + 270, origin.y + 102);
        EXPECT_FLOAT_EQ(member->GetAbsoluteOffset(Rml::BoxArea::Border).x, member_before.x + 200);
        EXPECT_FLOAT_EQ(member->GetAbsoluteOffset(Rml::BoxArea::Border).y, member_before.y + 80);
        pointer("mouseup", origin.x + 270, origin.y + 102);
        EXPECT_FLOAT_EQ(manager.tree(tree_)->find_node("Value")->location[0], location[0] + 100);
        EXPECT_FLOAT_EQ(manager.tree(tree_)->find_node("Value")->location[1], location[1] + 40);
        EXPECT_EQ(manager.tree(tree_)->find_node("Value")->ui["frame"], "Frame");
    }

    TEST_F(NodeCanvasWidgets, FrameHeaderAndWrappedNoteStayAboveMembersAtEveryZoom) {
        attachGraph();
        auto& manager = scene_.modifierManager();
        ASSERT_TRUE(manager.frameWrap(tree_, {"Value"}, "Selection Grade"));
        manager.tree(tree_)->find_node("Selection Grade")->properties["note"] =
            "Keep the selection outside the reusable grade. This description wraps across multiple lines.";
        manager.markDirty();
        auto* canvas = static_cast<lfs::vis::gui::NodeCanvasElement*>(document_->GetElementById("node-editor-canvas"));
        for (const auto zoom : {1.0f, 0.5f, 1.5f}) {
            canvas->setView({}, zoom);
            context_->Update();
            context_->Update();
            auto* label = canvas->QuerySelector(".frame-label");
            auto* note = canvas->QuerySelector(".frame-note");
            auto* member = canvas->QuerySelector(".node-box[data-node=Value]");
            ASSERT_NE(label, nullptr);
            ASSERT_NE(note, nullptr);
            EXPECT_GE(label->GetComputedValues().font_size(), 13.0f * 2.0f * zoom - 1);
            EXPECT_LT(note->GetAbsoluteOffset().y + note->GetBox().GetSize().y,
                      member->GetAbsoluteOffset(Rml::BoxArea::Border).y);
        }
    }

    TEST_F(NodeCanvasWidgets, NoteHeightFitsWrappedTextAndWidthPropertyAtEveryZoom) {
        attachGraph();
        auto& manager = scene_.modifierManager();
        auto& note = manager.tree(tree_)->add_node("lfs.note", "Note");
        note.location = {300, 500};
        note.properties["text"] = "Colour grade\nThis longer explanation must wrap without losing its second or final line.\nFinal line.";
        note.properties["width"] = 180.0f;
        manager.markDirty();
        auto* canvas = static_cast<lfs::vis::gui::NodeCanvasElement*>(document_->GetElementById("node-editor-canvas"));
        for (const auto zoom : {1.0f, 0.5f, 1.5f}) {
            canvas->setView({}, zoom);
            context_->Update();
            context_->Update();
            auto* card = canvas->QuerySelector(".node-note");
            auto* text = card->QuerySelector(".note-text");
            EXPECT_FLOAT_EQ(card->GetBox().GetSize(Rml::BoxArea::Border).x, 180.0f * 2 * zoom);
            EXPECT_LE(text->GetAbsoluteOffset().y + text->GetBox().GetSize().y,
                      card->GetAbsoluteOffset(Rml::BoxArea::Border).y + card->GetBox().GetSize(Rml::BoxArea::Border).y - 10 * zoom);
        }
    }

    TEST_F(NodeCanvasWidgets, GroupSidebarHidesUuidAndShowsFullTypeAndAddLabels) {
        attachGraph();
        auto& manager = scene_.modifierManager();
        const auto group = manager.makeGroup(tree_, {"Correct"}, "Grade");
        ASSERT_TRUE(group);
        auto* canvas = static_cast<lfs::vis::gui::NodeCanvasElement*>(document_->GetElementById("node-editor-canvas"));
        context_->Update();
        ASSERT_TRUE(canvas->selectNodes({group->group_node}, std::nullopt));
        context_->Update();
        auto* sidebar = canvas->QuerySelector("#node-editor-sidebar");
        EXPECT_EQ(sidebar->QuerySelector("[data-property=tree]"), nullptr);
        EXPECT_EQ(sidebar->QuerySelector(".interface-type")->GetInnerRML(), "Geometry");
        EXPECT_EQ(sidebar->QuerySelector("button[data-action=interface-add]")->GetInnerRML(), "Add input");
        ASSERT_TRUE(canvas->enterGroup(group->group_node));
        context_->Update();
        const auto breadcrumb = canvas->viewState()["breadcrumb"];
        ASSERT_EQ(breadcrumb.size(), 3u);
        EXPECT_EQ(breadcrumb[1]["label"], "Interaction");
        const auto markup = canvas->QuerySelector("#node-editor-breadcrumb")->GetInnerRML();
        EXPECT_EQ(markup.find("\xE2\x96\xB8"), std::string::npos);
    }

    TEST_F(NodeCanvasWidgets, CardContentsRemainVisibleAndContrastingAcrossThemesAndZoom) {
        attachGraph();
        auto* canvas = dynamic_cast<lfs::vis::gui::NodeCanvasElement*>(document_->GetElementById("node-editor-canvas"));
        const auto resources = std::filesystem::path(PROJECT_ROOT_PATH) / "src/visualizer/gui/rmlui/resources";
        const auto base = lfs::vis::gui::rml_theme::loadBaseRCSS((resources / "node_editor.rcss").string());
        const auto style = lfs::vis::gui::rml_theme::loadBaseRCSS((resources / "node_editor.theme.rcss").string());
        const auto luminance = [](const Rml::Colourb colour) {
            const auto linear = [](const Rml::byte channel) {
                const double s = channel / 255.0;
                return s <= 0.04045 ? s / 12.92 : std::pow((s + 0.055) / 1.055, 2.4);
            };
            return 0.2126 * linear(colour.red) + 0.7152 * linear(colour.green) + 0.0722 * linear(colour.blue);
        };
        std::string active_theme;
        for (const auto* mode : {"dark", "light"}) {
            ASSERT_TRUE(lfs::vis::setThemeFamilySelection("lichtfeld", mode));
            if (!active_theme.empty())
                context_->ActivateTheme(active_theme, false);
            active_theme = lfs::vis::currentThemeId();
            context_->ActivateTheme(active_theme, true);
            lfs::vis::gui::rml_theme::applyTheme(document_, base, style);
            for (const float zoom : {1.0f, 0.75f, 0.6f, 0.5f, 0.3f, 1.4f, 1.0f}) {
                SCOPED_TRACE(std::string(mode) + " zoom=" + std::to_string(zoom));
                context_->Update();
                canvas->setView({}, zoom);
                context_->Update();
                context_->Render();
                auto* card = canvas->QuerySelector(".node-box[data-node=Correct]");
                ASSERT_NE(card, nullptr);
                const auto card_bottom = card->GetAbsoluteOffset().y + card->GetBox().GetSize(Rml::BoxArea::Border).y;
                const double background = luminance(card->GetComputedValues().background_color());
                for (const auto* selector : {".socket-row.output .socket-label", ".socket-input.linked > .socket-label", ".node-settings", ".node-footer"}) {
                    SCOPED_TRACE(selector);
                    auto* element = card->QuerySelector(selector);
                    ASSERT_NE(element, nullptr);
                    EXPECT_GT(element->GetBox().GetSize().x, 0.0f);
                    EXPECT_GT(element->GetBox().GetSize().y, 0.0f);
                    EXPECT_FALSE(element->GetInnerRML().empty());
                    EXPECT_LE(element->GetAbsoluteOffset().y + element->GetBox().GetSize().y, card_bottom);
                    if (zoom >= 0.75f)
                        EXPECT_GE(element->GetComputedValues().font_size(), 11.0f * context_->GetDensityIndependentPixelRatio());
                    for (auto* ancestor = element; ancestor && ancestor != canvas; ancestor = ancestor->GetParentNode()) {
                        EXPECT_NE(ancestor->GetComputedValues().display(), Rml::Style::Display::None);
                        EXPECT_EQ(ancestor->GetComputedValues().visibility(), Rml::Style::Visibility::Visible);
                        EXPECT_GT(ancestor->GetComputedValues().opacity(), 0.9f);
                    }
                    const auto colour = element->GetComputedValues().color();
                    EXPECT_EQ(colour.alpha, 255);
                    const double foreground = luminance(colour);
                    EXPECT_GE((std::max(foreground, background) + 0.05) / (std::min(foreground, background) + 0.05), 3.0);
                }
            }
        }
    }

    TEST_F(NodeCanvasWidgets, HelpSurfacesRemainVisibleAndRememberTypePreference) {
        attachGraph();
        auto* canvas = dynamic_cast<lfs::vis::gui::NodeCanvasElement*>(document_->GetElementById("node-editor-canvas"));
        ASSERT_TRUE(canvas->selectNodes({"Correct"}, std::nullopt));
        context_->Update();
        auto* sidebar = canvas->GetElementById("node-editor-sidebar");
        const auto descriptor = scene_.modifierManager().registry().find("lfs.colour_correct");
        auto* description = sidebar->QuerySelector(".node-description");
        auto* help = sidebar->QuerySelector(".node-help");
        ASSERT_NE(description, nullptr);
        ASSERT_NE(help, nullptr);
        EXPECT_EQ(description->GetInnerRML(), descriptor->description);
        EXPECT_LT(description->GetAbsoluteOffset().y, description->GetParentNode()->QuerySelector(".target-type")->GetAbsoluteOffset().y);
        EXPECT_TRUE(help->IsClassSet("expanded"));
        EXPECT_GT(help->QuerySelector(".node-help-text")->GetBox().GetSize().y, 30.0f);
        help->QuerySelector("button")->DispatchEvent("click", {});
        context_->Update();
        EXPECT_FALSE(sidebar->QuerySelector(".node-help")->IsClassSet("expanded"));
        ASSERT_TRUE(canvas->selectNodes({"Value"}, std::nullopt));
        context_->Update();
        EXPECT_TRUE(sidebar->QuerySelector(".node-help")->IsClassSet("expanded"));
        ASSERT_TRUE(canvas->selectNodes({"Correct"}, std::nullopt));
        context_->Update();
        EXPECT_FALSE(sidebar->QuerySelector(".node-help")->IsClassSet("expanded"));
        auto* input = sidebar->QuerySelector("input[data-input=Exposure]");
        ASSERT_NE(input, nullptr);
        const auto socket = std::ranges::find(descriptor->inputs, "Exposure", &lfs::nodes::SocketDecl::identifier);
        ASSERT_NE(socket, descriptor->inputs.end());
        EXPECT_EQ(lfs::vis::gui::resolveRmlTooltip(input), socket->description);
        input->DispatchEvent("focus", {});
        context_->Update();
        auto* row = input;
        while (row && !row->IsClassSet("node-setting-help"))
            row = row->GetParentNode();
        ASSERT_NE(row, nullptr);
        EXPECT_TRUE(row->IsClassSet("show-help"));
        EXPECT_GT(row->QuerySelector(".node-field-help")->GetBox().GetSize().y, 0.0f);
        input->DispatchEvent("blur", {});
        context_->Update();
        EXPECT_FALSE(row->IsClassSet("show-help"));
        auto* title = canvas->QuerySelector(".node-box[data-node=Correct] .node-title-label");
        EXPECT_EQ(lfs::vis::gui::resolveRmlTooltip(title), descriptor->description);
        auto* output = canvas->QuerySelector(".node-box[data-node=Correct] .socket-row.output .socket-label");
        EXPECT_EQ(lfs::vis::gui::resolveRmlTooltip(output), descriptor->outputs.front().description);
    }

    TEST_F(NodeCanvasWidgets, SearchKeyboardHighlightShowsMatchingHelpPreview) {
        attachGraph();
        auto* canvas = dynamic_cast<lfs::vis::gui::NodeCanvasElement*>(document_->GetElementById("node-editor-canvas"));
        canvas->headerAction("add", 40, 100);
        context_->Update();
        auto* search = dynamic_cast<Rml::ElementFormControlInput*>(canvas->GetElementById("node-add-search"));
        ASSERT_NE(search, nullptr);
        EXPECT_EQ(context_->GetFocusElement(), search);
        search->SetValue("Scale Clamp");
        search->DispatchEvent("change", {});
        context_->Update();
        auto* menu = canvas->GetElementById("node-add-menu");
        EXPECT_TRUE(menu->IsClassSet("has-preview"));
        auto* preview = menu->GetElementById("node-add-preview");
        ASSERT_NE(preview, nullptr);
        EXPECT_GT(preview->GetBox().GetSize().x, 200.0f);
        EXPECT_GT(preview->GetBox().GetSize().y, 40.0f);
        EXPECT_EQ(preview->QuerySelector(".node-add-preview-description")->GetInnerRML(),
                  scene_.modifierManager().registry().find("lfs.scale_clamp")->description);
        search->SetValue("Colour");
        search->DispatchEvent("change", {});
        context_->Update();
        const auto previous = menu->QuerySelector(".first-hit")->GetAttribute<Rml::String>("data-type", "");
        ASSERT_TRUE(canvas->handleKey(SDL_SCANCODE_DOWN, false, false, false));
        context_->Update();
        const auto next = menu->QuerySelector(".first-hit")->GetAttribute<Rml::String>("data-type", "");
        EXPECT_NE(previous, next);
        EXPECT_EQ(preview->QuerySelector(".node-add-preview-description")->GetInnerRML(),
                  scene_.modifierManager().registry().find(next)->description);
        EXPECT_TRUE(canvas->handleKey(SDL_SCANCODE_ESCAPE, false, false, false));
        EXPECT_EQ(canvas->GetElementById("node-add-menu"), nullptr);
    }

    TEST_F(NodeCanvasWidgets, LanguageChangesOnlyPresentationAndPreviewRespectsOptOut) {
        attachGraph();
        auto* canvas = dynamic_cast<lfs::vis::gui::NodeCanvasElement*>(document_->GetElementById("node-editor-canvas"));
        auto& registry = scene_.modifierManager().registry();
        const auto english = registry.find("lfs.colour_correct");
        const auto before = scene_.modifierManager().tree(tree_)->to_json();
        ASSERT_TRUE(lfs::event::LocalizationManager::getInstance().setLanguage("de"));
        context_->Update();
        const auto german = registry.find_localized("lfs.colour_correct");
        EXPECT_NE(german->label, english->label);
        EXPECT_NE(german->help, english->help);
        EXPECT_EQ(german->inputs[2].identifier, "Exposure");
        EXPECT_EQ(registry.find("lfs.colour_correct"), english);
        EXPECT_EQ(scene_.modifierManager().tree(tree_)->to_json(), before);
        auto& tree = *scene_.modifierManager().tree(tree_);
        tree.add_node("lfs.hsv_range", "HSV").location = {30, 500};
        scene_.modifierManager().markDirty();
        context_->Update();
        ASSERT_TRUE(canvas->selectNodes({"HSV"}, std::nullopt));
        EXPECT_TRUE(canvas->previewSelection());
        canvas->setPreviewSelection(false);
        ASSERT_TRUE(canvas->selectNodes({"Correct"}, std::nullopt));
        ASSERT_TRUE(canvas->selectNodes({"HSV"}, std::nullopt));
        EXPECT_FALSE(canvas->previewSelection());
    }

    TEST_F(NodeCanvasWidgets, KeepingOverviewLabelsDoesNotEnlargeCardsIntoNeighbours) {
        attachGraph();
        auto* canvas = dynamic_cast<lfs::vis::gui::NodeCanvasElement*>(document_->GetElementById("node-editor-canvas"));
        ASSERT_TRUE(canvas->arrange(false));
        for (const float zoom : {0.3f, 0.5f, 0.6f, 1.0f, 1.4f}) {
            SCOPED_TRACE(zoom);
            canvas->setView({}, zoom);
            context_->Update();
            const auto nodes = canvas->viewState().at("nodes");
            for (std::size_t i = 0; i < nodes.size(); ++i) {
                const auto a = nodes[i].at("bounds").get<std::array<float, 4>>();
                for (std::size_t j = i + 1; j < nodes.size(); ++j) {
                    const auto b = nodes[j].at("bounds").get<std::array<float, 4>>();
                    EXPECT_FALSE(a[0] < b[0] + b[2] && a[0] + a[2] > b[0] &&
                                 a[1] < b[1] + b[3] && a[1] + a[3] > b[1]);
                }
            }
        }
    }

    TEST_F(NodeCanvasWidgets, ModifierNameElidesUntilFocusedAndStatusItemsDoNotSplit) {
        attachGraph();
        auto& manager = scene_.modifierManager();
        const std::string name = "Garden — Colour & Detail with a long modifier name";
        ASSERT_TRUE(manager.renameModifier(host_, manager.stack(host_)->modifiers.front().name, name));
        auto* canvas = dynamic_cast<lfs::vis::gui::NodeCanvasElement*>(document_->GetElementById("node-editor-canvas"));
        ASSERT_TRUE(canvas->selectNodes({"Correct"}, std::nullopt));
        context_->Update();
        auto* rename = dynamic_cast<Rml::ElementFormControlInput*>(canvas->QuerySelector(".modifier-rename"));
        auto* label = canvas->QuerySelector(".modifier-name-label");
        ASSERT_NE(rename, nullptr);
        ASSERT_NE(label, nullptr);
        EXPECT_EQ(rename->GetAttribute<Rml::String>("title", ""), name);
        EXPECT_EQ(lfs::vis::gui::resolveRmlTooltip(rename), name);
        EXPECT_EQ(rename->GetValue(), name);
        EXPECT_EQ(label->GetComputedValues().text_overflow(), Rml::Style::TextOverflow::Ellipsis);
        EXPECT_GT(label->GetBox().GetSize().x, 120.0f);
        EXPECT_EQ(rename->GetComputedValues().opacity(), 0.0f);
        const auto position = rename->GetAbsoluteOffset() + rename->GetBox().GetSize() * 0.5f;
        context_->ProcessMouseMove(static_cast<int>(position.x), static_cast<int>(position.y), 0);
        EXPECT_EQ(lfs::vis::gui::resolveRmlTooltip(context_->GetHoverElement()), name);
        context_->ProcessMouseButtonDown(0, 0);
        context_->ProcessMouseButtonUp(0, 0);
        context_->Update();
        EXPECT_EQ(context_->GetFocusElement(), rename);
        EXPECT_EQ(label->GetComputedValues().display(), Rml::Style::Display::None);
        EXPECT_EQ(rename->GetComputedValues().opacity(), 1.0f);
        rename->SetValue("Renamed");
        Rml::Dictionary change;
        change["linebreak"] = true;
        rename->DispatchEvent("change", change);
        context_->Update();
        EXPECT_EQ(manager.stack(host_)->modifiers.front().name, "Renamed");

        const auto text = lfs::vis::gui::node_widgets::nonBreakingStatus("Last run: 1.00M · 11% selected · cached · 12 ms");
        EXPECT_EQ(text, "Last\u00a0run:\u00a01.00M · 11%\u00a0selected · cached · 12\u00a0ms");
        auto* status = dynamic_cast<Rml::ElementText*>(canvas->GetElementById("node-inspector-last-run")->GetChild(0));
        ASSERT_NE(status, nullptr);
        EXPECT_EQ(status->GetText().find(" ms"), std::string::npos);
        EXPECT_NE(status->GetText().find("\u00a0ms"), std::string::npos);
    }

    TEST_F(NodeCanvasWidgets, DoubleClickOpensEditableNumberAndEnterClampsValue) {
        field_->DispatchEvent("dblclick", {});
        context_->Update();
        EXPECT_TRUE(field_->IsClassSet("is-editing"));
        EXPECT_EQ(context_->GetFocusElement(), input_);
        input_->SetValue("2.5");
        context_->ProcessKeyDown(Rml::Input::KI_RETURN, 0);
        EXPECT_FALSE(field_->IsClassSet("is-editing"));
        EXPECT_DOUBLE_EQ(field_->GetAttribute<double>("data-value", 0.0), 1.0);
    }

    TEST_F(NodeCanvasWidgets, EscapeRestoresTypedValue) {
        field_->DispatchEvent("dblclick", {});
        context_->Update();
        input_->SetValue("0.75");
        context_->ProcessKeyDown(Rml::Input::KI_ESCAPE, 0);
        EXPECT_FALSE(field_->IsClassSet("is-editing"));
        EXPECT_DOUBLE_EQ(field_->GetAttribute<double>("data-value", 0.0), 0.25);
    }

    TEST_F(NodeCanvasWidgets, DetachedFieldCannotAcquireKeyboardFocus) {
        lfs::vis::gui::RmlUIManager manager;
        auto detached = document_->CreateElement("input");
        detached->SetAttribute("type", "text");
        detached->AddEventListener("focus", &manager);
        EXPECT_EQ(detached->GetContext(), nullptr);
        EXPECT_NO_THROW(detached->DispatchEvent("focus", {}));
        detached->RemoveEventListener("focus", &manager);
    }

    TEST_F(NodeCanvasWidgets, PanZoomSelectBoxSelectAndMoveNeverEvaluateOrReplaceCards) {
        attachGraph();
        auto* canvas = document_->GetElementById("node-editor-canvas");
        auto* card = canvas->QuerySelector(".node-box[data-node=Correct]");
        ASSERT_NE(card, nullptr);
        pointer("mousedown", 600, 65);
        pointer("mouseup", 600, 65);
        EXPECT_TRUE(card->IsClassSet("selected"));
        expectNoEvaluation();
        pointer("mousedown", 600, 65);
        for (int step = 1; step <= 20; ++step)
            pointer("mousemove", 600.0f + step, 65.0f + step);
        pointer("mouseup", 620, 85);
        expectNoEvaluation();
        EXPECT_EQ(canvas->QuerySelector(".node-box[data-node=Correct]"), card);
        const auto* moved = scene_.modifierManager().tree(tree_)->find_node("Correct");
        ASSERT_NE(moved, nullptr);
        EXPECT_EQ(moved->location[0], 280.0f);
        pointer("mousedown", 1500, 1300);
        pointer("mousemove", 20, 20);
        pointer("mouseup", 20, 20);
        expectNoEvaluation();
        pointer("mousedown", 1000, 1000, 2);
        pointer("mousemove", 1060, 1030, 2);
        pointer("mouseup", 1060, 1030, 2);
        expectNoEvaluation();
        Rml::Dictionary wheel;
        wheel["mouse_x"] = 800.0f;
        wheel["mouse_y"] = 700.0f;
        wheel["wheel_delta_y"] = 1.0f;
        wheel["ctrl_key"] = 1;
        canvas->DispatchEvent("mousescroll", wheel);
        context_->Update();
        scene_.modifierManager().tick();
        expectNoEvaluation();
        EXPECT_EQ(canvas->QuerySelector(".node-box[data-node=Correct]"), card);
    }

    TEST_F(NodeCanvasWidgets, AddPopupFiltersAndEnterAddsWithoutReplacingExistingCards) {
        attachGraph();
        auto* canvas = dynamic_cast<lfs::vis::gui::NodeCanvasElement*>(document_->GetElementById("node-editor-canvas"));
        auto* card = canvas->QuerySelector(".node-box[data-node=Correct]");
        canvas->headerAction("add", 600, 300);
        context_->Update();
        auto* menu = canvas->GetElementById("node-add-menu");
        ASSERT_NE(menu, nullptr);
        EXPECT_LT(menu->GetBox().GetSize().y, context_->GetDimensions().y * 0.7f);
        auto* search = dynamic_cast<Rml::ElementFormControlInput*>(menu->GetElementById("node-add-search"));
        ASSERT_NE(search, nullptr);
        search->SetValue("HSV");
        search->DispatchEvent("change", {});
        context_->Update();
        Rml::ElementList items;
        menu->QuerySelectorAll(items, ".node-add-item");
        EXPECT_EQ(std::ranges::count_if(items, [](const auto* item) { return !item->IsClassSet("filtered"); }), 1);
        expectNoEvaluation();
        ASSERT_TRUE(canvas->handleKey(SDL_SCANCODE_RETURN, false, false, false));
        context_->Update();
        EXPECT_EQ(canvas->GetElementById("node-add-menu"), nullptr);
        const auto* tree = scene_.modifierManager().tree(tree_);
        EXPECT_EQ(std::ranges::count(tree->nodes, "lfs.hsv_range", &lfs::nodes::Node::type_id), 1);
        EXPECT_EQ(canvas->QuerySelector(".node-box[data-node=Correct]"), card);
    }

    TEST_F(NodeCanvasWidgets, ArrangeSubsetIsOneLayoutUndoAndPreservesUnselectedCards) {
        attachGraph();
        auto* canvas = dynamic_cast<lfs::vis::gui::NodeCanvasElement*>(document_->GetElementById("node-editor-canvas"));
        auto& manager = scene_.modifierManager();
        auto* tree = manager.tree(tree_);
        const auto before = tree->to_json();
        auto* card = canvas->QuerySelector(".node-box[data-node=Correct]");
        ASSERT_TRUE(canvas->selectNodes({"Correct", "Value"}, std::nullopt));
        context_->Update();
        lfs::vis::op::undoHistory().clear();
        const auto view = canvas->viewState();
        ASSERT_TRUE(canvas->arrange());
        context_->Update();
        EXPECT_EQ(tree->input_node().location, (std::array<float, 2>{20, 20}));
        EXPECT_EQ(tree->output_node().location, (std::array<float, 2>{530, 20}));
        EXPECT_EQ(canvas->viewState()["pan"], view["pan"]);
        EXPECT_EQ(canvas->viewState()["zoom"], view["zoom"]);
        EXPECT_EQ(canvas->QuerySelector(".node-box[data-node=Correct]"), card);
        EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 1u);
        expectNoEvaluation();
        ASSERT_TRUE(lfs::vis::op::undoHistory().undo().success);
        EXPECT_EQ(manager.tree(tree_)->to_json(), before);
        EXPECT_FALSE(canvas->arrange(true, std::unordered_set<std::string>{"Missing"}));
    }

    TEST_F(NodeCanvasWidgets, ColourControlsAreCompactAndSignedWheelEditsOneUndo) {
        attachGraph();
        auto* canvas = dynamic_cast<lfs::vis::gui::NodeCanvasElement*>(document_->GetElementById("node-editor-canvas"));
        ASSERT_TRUE(canvas->selectNodes({"Correct"}, std::nullopt));
        context_->Update();
        auto* sidebar = canvas->GetElementById("node-editor-sidebar");
        auto* wheel = sidebar->QuerySelector("colour-offset[data-input=Shadows]");
        ASSERT_NE(wheel, nullptr);
        EXPECT_EQ(sidebar->QuerySelector("input[data-input=Shadows]"), nullptr);
        auto* swatch = sidebar->QuerySelector(".node-swatch[data-input=Shadows]");
        ASSERT_NE(swatch, nullptr);
        EXPECT_EQ(swatch->GetAttribute<int>("data-offset", 0), 1);
        const auto bounds = wheel->GetBox().GetSize();
        EXPECT_FLOAT_EQ(bounds.x, 176);
        EXPECT_FLOAT_EQ(bounds.y, 176);
        lfs::vis::op::undoHistory().clear();
        wheel->ScrollIntoView();
        context_->Update();
        const auto origin = wheel->GetAbsoluteOffset(Rml::BoxArea::Content);
        context_->ProcessMouseMove(static_cast<int>(origin.x + bounds.x * 0.5f),
                                   static_cast<int>(origin.y + bounds.y * 0.5f), 0);
        context_->ProcessMouseButtonDown(0, 0);
        for (int i = 1; i <= 10; ++i) {
            context_->ProcessMouseMove(static_cast<int>(origin.x + bounds.x * 0.5f + i * 4.0f),
                                       static_cast<int>(origin.y + bounds.y * 0.5f), 0);
            context_->Update();
            EXPECT_EQ(sidebar->QuerySelector("colour-offset[data-input=Shadows]"), wheel);
        }
        context_->ProcessMouseButtonUp(0, 0);
        context_->Update();
        auto& manager = scene_.modifierManager();
        const auto values = [&]() {
            return lfs::vis::gui::node_widgets::valuePayload(*manager.tree(tree_)->find_node("Correct"), "Shadows", glm::vec3(0));
        };
        EXPECT_GT(values()[0].get<float>(), 0.4f);
        EXPECT_LT(values()[1].get<float>(), -0.2f);
        EXPECT_LT(values()[2].get<float>(), -0.2f);
        EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 1u);
        wheel->DispatchEvent("dblclick", {});
        context_->Update();
        EXPECT_EQ(values()[0], 0.0f);
        EXPECT_EQ(values()[1], 0.0f);
        EXPECT_EQ(values()[2], 0.0f);
        ASSERT_TRUE(lfs::vis::op::undoHistory().undo().success);
        EXPECT_GT(values()[0].get<float>(), 0.4f);
    }

    TEST_F(NodeCanvasWidgets, SignedSwatchUsesStandardPickerWithLiveSingleUndoDrag) {
        attachGraph();
        auto* canvas = dynamic_cast<lfs::vis::gui::NodeCanvasElement*>(document_->GetElementById("node-editor-canvas"));
        ASSERT_TRUE(canvas->selectNodes({"Correct"}, std::nullopt));
        context_->Update();
        auto* swatch = canvas->GetElementById("node-editor-sidebar")->QuerySelector(".node-swatch[data-input=Shadows]");
        ASSERT_NE(swatch, nullptr);
        swatch->DispatchEvent("click", {});
        context_->Update();
        auto* picker = canvas->QuerySelector("color-picker");
        ASSERT_NE(picker, nullptr);
        EXPECT_FLOAT_EQ(picker->GetAttribute<float>("red", 0), 0.5f);
        EXPECT_FLOAT_EQ(picker->GetAttribute<float>("green", 0), 0.5f);
        EXPECT_FLOAT_EQ(picker->GetAttribute<float>("blue", 0), 0.5f);
        const auto origin = picker->GetAbsoluteOffset(Rml::BoxArea::Content);
        const auto size = picker->GetBox().GetSize();
        lfs::vis::op::undoHistory().clear();
        context_->ProcessMouseMove(static_cast<int>(origin.x + size.x * 0.1f), static_cast<int>(origin.y + size.y * 0.4f), 0);
        context_->ProcessMouseButtonDown(0, 0);
        for (int i = 1; i <= 10; ++i) {
            context_->ProcessMouseMove(static_cast<int>(origin.x + size.x * (0.1f + i * 0.05f)), static_cast<int>(origin.y + size.y * 0.4f), 0);
            context_->Update();
        }
        context_->ProcessMouseButtonUp(0, 0);
        context_->Update();
        const auto values = lfs::vis::gui::node_widgets::valuePayload(
            *scene_.modifierManager().tree(tree_)->find_node("Correct"), "Shadows", glm::vec3(0));
        EXPECT_NEAR(values[0].get<float>(), 0.2f, 0.02f);
        EXPECT_LT(values[1].get<float>(), -0.5f);
        EXPECT_LT(values[2].get<float>(), -0.5f);
        EXPECT_EQ(lfs::vis::op::undoHistory().undoCount(), 1u);
    }

    TEST_F(NodeCanvasWidgets, InspectorSwitchUsesChangeEventAndRequestsOnce) {
        attachGraph();
        pointer("mousedown", 600, 65);
        pointer("mouseup", 600, 65);
        auto* canvas = document_->GetElementById("node-editor-canvas");
        auto* control = canvas->QuerySelector("input[data-action=node-on]");
        ASSERT_NE(control, nullptr);
        control->RemoveAttribute("checked");
        control->DispatchEvent("change", {});
        context_->Update();
        auto& manager = scene_.modifierManager();
        EXPECT_TRUE(manager.tree(tree_)->find_node("Correct")->muted);
        manager.tick();
        ASSERT_TRUE(manager.evaluate(host_).ok);
        const auto metrics = manager.performance();
        EXPECT_EQ(metrics["requests"], 1);
        EXPECT_EQ(metrics["evaluations"], 1);
    }

    TEST_F(NodeCanvasWidgets, WireDragDoesNotEvaluateUntilDropAndDropRequestsOnce) {
        attachGraph();
        auto* canvas = document_->GetElementById("node-editor-canvas");
        auto* field = canvas->QuerySelector(".node-box[data-node=Correct] input[data-input=Exposure]");
        ASSERT_NE(field, nullptr);
        canvas->QuerySelector(".node-box[data-node=Correct] .node-settings")->DispatchEvent("click", {});
        context_->Update();
        pointer("mousedown", 488, 646);
        for (int step = 1; step <= 20; ++step)
            pointer("mousemove", 488.0f + step * 2.6f, 646.0f - step * 15.0f);
        expectNoEvaluation();
        pointer("mouseup", 540, 346);
        SCOPED_TRACE(dynamic_cast<lfs::vis::gui::NodeCanvasElement*>(canvas)->viewState().dump());
        EXPECT_EQ(canvas->QuerySelector(".node-box[data-node=Correct] input[data-input=Exposure]"), field);
        EXPECT_TRUE(field->GetParentNode()->GetParentNode()->GetParentNode()->IsClassSet("linked"));
        auto& manager = scene_.modifierManager();
        const auto* tree = manager.tree(tree_);
        ASSERT_TRUE(std::ranges::any_of(tree->links, [](const auto& link) {
            return link.from_node == "Value" && link.to_node == "Correct" && link.to_socket == "Exposure";
        }));
        ASSERT_TRUE(manager.evaluate(host_).ok);
        const auto metrics = manager.performance();
        EXPECT_EQ(metrics["requests"], 1);
        EXPECT_EQ(metrics["evaluations"], 1);
    }

    TEST_F(NodeCanvasWidgets, CompactSettingsPersistWithoutEvaluationAndLinkedInputsStayVisible) {
        attachGraph();
        auto* canvas = dynamic_cast<lfs::vis::gui::NodeCanvasElement*>(document_->GetElementById("node-editor-canvas"));
        auto* card = canvas->QuerySelector(".node-box[data-node=Correct]");
        auto* exposure = card->QuerySelector("input[data-input=Exposure]");
        auto* row = exposure->GetParentNode()->GetParentNode()->GetParentNode();
        EXPECT_EQ(row->GetComputedValues().display(), Rml::Style::Display::None);
        const float compact_height = card->GetOffsetHeight();
        card->QuerySelector(".node-settings")->DispatchEvent("click", {});
        context_->Update();
        EXPECT_GT(card->GetOffsetHeight(), compact_height + 300.0f);
        auto& manager = scene_.modifierManager();
        auto* tree = manager.tree(tree_);
        const auto restored = lfs::nodes::NodeTree::from_json(tree->to_json(), manager.registry());
        EXPECT_TRUE(restored.find_node("Correct")->ui.at("settings_expanded").get<bool>());
        card->QuerySelector(".node-settings")->DispatchEvent("click", {});
        context_->Update();
        manager.tick();
        expectNoEvaluation();
        const auto before = tree->to_json();
        ASSERT_TRUE(tree->add_link({"Value", "Value", "Correct", "Exposure"}));
        manager.recordTreeEdit(tree_, before);
        context_->Update();
        EXPECT_NE(row->GetComputedValues().display(), Rml::Style::Display::None);
        EXPECT_EQ(card->QuerySelector("input[data-input=Exposure]"), exposure);
    }

    TEST_F(NodeCanvasWidgets, CardsSurviveSubmitBusyInstallAndMoveBeforeRelease) {
        attachGraph();
        auto& manager = scene_.modifierManager();
        lfs::nodes::NodeTypeInfo slow;
        slow.id = "test.slow";
        slow.label = "Slow";
        slow.inputs = {{"Geometry", "Geometry", std::string(lfs::nodes::GEOMETRY_SOCKET)}};
        slow.outputs = slow.inputs;
        slow.evaluate = [](lfs::nodes::NodeContext& context) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            context.set_output("Geometry", context.input("Geometry"));
        };
        ASSERT_TRUE(manager.registry().register_type(std::move(slow)));
        auto* tree = manager.tree(tree_);
        const auto before = tree->to_json();
        tree->find_node("Correct")->type_id = "test.slow";
        manager.recordTreeEdit(tree_, before);
        context_->Update();
        auto* canvas = dynamic_cast<lfs::vis::gui::NodeCanvasElement*>(document_->GetElementById("node-editor-canvas"));
        auto* card = canvas->QuerySelector(".node-box[data-node=Correct]");
        ASSERT_NE(card, nullptr);
        const auto node_count = canvas->viewState()["nodes"].size();
        const auto link_count = canvas->viewState()["links"];
        manager.tick();
        context_->Update();
        EXPECT_FALSE(card->IsClassSet("evaluating"));
        for (int attempt = 0; attempt < 100 && manager.progress().node != "Correct"; ++attempt)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        std::this_thread::sleep_for(std::chrono::milliseconds(90));
        context_->Update();
        context_->Render();
        EXPECT_TRUE(card->IsClassSet("evaluating"));
        EXPECT_EQ(canvas->viewState()["nodes"].size(), node_count);
        EXPECT_EQ(canvas->viewState()["links"], link_count);
        pointer("mousedown", 600, 65);
        pointer("mousemove", 670, 105);
        SCOPED_TRACE(canvas->viewState().dump());
        EXPECT_TRUE(card->IsClassSet("selected"));
        EXPECT_NEAR(card->GetAbsoluteOffset().x, 610.0f, 2.0f);
        EXPECT_EQ(tree->find_node("Correct")->location[0], 270.0f);
        pointer("mouseup", 670, 105);
        ASSERT_TRUE(manager.evaluate(host_).ok);
        context_->Update();
        context_->Render();
        EXPECT_EQ(canvas->QuerySelector(".node-box[data-node=Correct]"), card);
        EXPECT_EQ(canvas->viewState()["nodes"].size(), node_count);
        EXPECT_EQ(canvas->viewState()["links"], link_count);
        EXPECT_FALSE(card->IsClassSet("evaluating"));
    }

    TEST_F(NodeCanvasWidgets, InspectorKeepsFullWidthAcrossPatches) {
        attachGraph();
        context_->SetDimensions({2488, 1246});
        auto* canvas = dynamic_cast<lfs::vis::gui::NodeCanvasElement*>(document_->GetElementById("node-editor-canvas"));
        for (int i = 0; i < 10; ++i) {
            ASSERT_TRUE(canvas->selectNodes({i % 2 == 0 ? "Correct" : "Value"}, std::nullopt));
            context_->Update();
            context_->Render();
            Rml::ElementList sections;
            canvas->QuerySelectorAll(sections, ".sidebar-section");
            ASSERT_EQ(sections.size(), 3u);
            for (auto* section : sections) {
                EXPECT_GE(section->GetBox().GetSize(Rml::BoxArea::Border).x, 520.0f);
            }
        }
    }

    TEST_F(NodeCanvasWidgets, SidebarFillsColumnAtOneAndTwoDpAcrossResizeAndSelection) {
        attachGraph();
        auto* canvas = dynamic_cast<lfs::vis::gui::NodeCanvasElement*>(document_->GetElementById("node-editor-canvas"));
        for (const float dp : {1.0f, 2.0f}) {
            context_->SetDensityIndependentPixelRatio(dp);
            for (const int height : {380, 750, 420}) {
                context_->SetDimensions({static_cast<int>(1244 * dp), static_cast<int>(height * dp)});
                for (const char* name : {"Correct", "Value", "Correct"}) {
                    ASSERT_TRUE(canvas->selectNodes({name}, std::nullopt));
                    context_->Update();
                    context_->Render();
                    Rml::ElementList sections;
                    canvas->QuerySelectorAll(sections, ".sidebar-section");
                    for (auto* section : sections)
                        EXPECT_GE(section->GetBox().GetSize(Rml::BoxArea::Border).x / dp, 260.0f);
                    Rml::ElementList fields;
                    canvas->QuerySelectorAll(fields, "#node-editor-sidebar .node-field-control");
                    ASSERT_FALSE(fields.empty());
                    for (auto* field : fields)
                        EXPECT_GE(field->GetBox().GetSize(Rml::BoxArea::Content).x / dp, 120.0f);
                    auto* add = canvas->QuerySelector("[data-action=add-modifier]");
                    ASSERT_NE(add, nullptr);
                    EXPECT_GE(add->GetBox().GetSize(Rml::BoxArea::Border).x / dp, 230.0f);
                    auto* section = add->GetParentNode();
                    const float content_right = section->GetAbsoluteOffset(Rml::BoxArea::Content).x + section->GetBox().GetSize(Rml::BoxArea::Content).x;
                    const float button_right = add->GetAbsoluteOffset(Rml::BoxArea::Border).x + add->GetBox().GetSize(Rml::BoxArea::Border).x;
                    EXPECT_LE(button_right, content_right + 1.0f);
                    auto* rename = canvas->QuerySelector(".modifier-rename");
                    ASSERT_NE(rename, nullptr);
                    EXPECT_GE(rename->GetBox().GetSize(Rml::BoxArea::Content).x / dp, 120.0f);
                    auto* status = canvas->GetElementById("node-inspector-last-run");
                    ASSERT_NE(status, nullptr);
                    EXPECT_LE(status->GetBox().GetSize(Rml::BoxArea::Content).y / dp, 30.0f);
                }
            }
        }
    }

    TEST_F(NodeCanvasWidgets, NumericSteppersRespectModifiersHoldAndUndoOnce) {
        attachGraph();
        auto* canvas = dynamic_cast<lfs::vis::gui::NodeCanvasElement*>(document_->GetElementById("node-editor-canvas"));
        ASSERT_TRUE(canvas->selectNodes({"Correct"}, std::nullopt));
        context_->Update();
        auto* input = canvas->QuerySelector("#node-editor-sidebar input[data-input=Exposure]");
        ASSERT_NE(input, nullptr);
        auto* field = input->GetParentNode();
        auto* increase = field->QuerySelector("[data-step-direction='1']");
        ASSERT_NE(increase, nullptr);
        auto& manager = scene_.modifierManager();
        const double step = field->GetAttribute<double>("data-step", 0.01);
        const auto pointer = [](Rml::Element* element, const char* type, const char* modifier = "shift_key", int pressed = 0) {
            Rml::Dictionary parameters;
            parameters["button"] = 0;
            parameters[modifier] = pressed;
            element->DispatchEvent(type, parameters);
        };
        lfs::vis::op::undoHistory().clear();
        pointer(increase, "mousedown", "shift_key", 1);
        pointer(increase, "mouseup");
        context_->Update();
        EXPECT_NEAR(std::get<float>(manager.tree(tree_)->find_node("Correct")->input_values.at("Exposure").data), step * 10, 1e-6);
        ASSERT_TRUE(lfs::vis::op::undoHistory().undo().success);
        context_->Update();
        input = canvas->QuerySelector("#node-editor-sidebar input[data-input=Exposure]");
        increase = input->GetParentNode()->QuerySelector("[data-step-direction='1']");
        pointer(increase, "mousedown", "alt_key", 1);
        std::this_thread::sleep_for(std::chrono::milliseconds(420));
        context_->Update();
        pointer(increase, "mouseup");
        context_->Update();
        EXPECT_NEAR(std::get<float>(manager.tree(tree_)->find_node("Correct")->input_values.at("Exposure").data), step * 0.2, 1e-6);
        ASSERT_TRUE(lfs::vis::op::undoHistory().undo().success);
        EXPECT_NEAR(std::get<float>(manager.tree(tree_)->find_node("Correct")->input_values.at("Exposure").data), 0.0, 1e-6);
    }

    TEST_F(NodeCanvasWidgets, EnterStartsNumericTypingAndEscapeCancels) {
        field_->Focus();
        context_->ProcessKeyDown(Rml::Input::KI_RETURN, 0);
        EXPECT_TRUE(field_->IsClassSet("is-editing"));
        input_->SetValue("0.8");
        context_->ProcessKeyDown(Rml::Input::KI_ESCAPE, 0);
        EXPECT_FALSE(field_->IsClassSet("is-editing"));
        EXPECT_DOUBLE_EQ(field_->GetAttribute<double>("data-value", 0.0), 0.25);
    }

    TEST_F(NodeCanvasWidgets, DomPatchNeverReplacesPrivateScrollbarChildren) {
        auto* parent = document_->AppendChild(document_->CreateElement("div"));
        parent->SetInnerRML("<div id='first'>First</div>");
        auto private_child = document_->CreateElement("scrollbarvertical");
        auto* scrollbar = parent->AppendChild(std::move(private_child), false);
        ASSERT_EQ(parent->GetNumChildren(), 1);
        ASSERT_EQ(parent->GetNumChildren(true), 2);
        lfs::vis::gui::node_widgets::patchMarkup(*parent, "<div id='first'>First</div><div id='second'>Second</div>");
        EXPECT_EQ(parent->GetNumChildren(), 2);
        EXPECT_EQ(parent->GetNumChildren(true), 3);
        EXPECT_EQ(parent->GetChild(2), scrollbar);
        lfs::vis::gui::node_widgets::patchMarkup(*parent, "<div id='first'>Changed</div>");
        EXPECT_EQ(parent->GetNumChildren(), 1);
        EXPECT_EQ(parent->GetChild(1), scrollbar);
    }

    TEST_F(NodeCanvasWidgets, AddPopupFocusesSearchAndEnterAddsExactTopHit) {
        attachGraph();
        auto* canvas = dynamic_cast<lfs::vis::gui::NodeCanvasElement*>(document_->GetElementById("node-editor-canvas"));
        canvas->headerAction("add", 400, 400);
        context_->Update();
        auto* search = document_->GetElementById("node-add-search");
        ASSERT_NE(search, nullptr);
        EXPECT_EQ(context_->GetFocusElement(), search);
        context_->ProcessTextInput("HSV");
        context_->Update();
        auto* popup = document_->GetElementById("node-add-menu");
        ASSERT_NE(popup, nullptr);
        EXPECT_TRUE(popup->IsClassSet("searching"));
        auto* top = popup->QuerySelector(".first-hit");
        ASSERT_NE(top, nullptr);
        EXPECT_EQ(top->GetAttribute<Rml::String>("data-type", ""), "lfs.hsv_range");
        const auto count = scene_.modifierManager().tree(tree_)->nodes.size();
        context_->ProcessKeyDown(Rml::Input::KI_RETURN, 0);
        context_->Update();
        EXPECT_EQ(document_->GetElementById("node-add-menu"), nullptr);
        EXPECT_EQ(scene_.modifierManager().tree(tree_)->nodes.size(), count + 1);
        EXPECT_EQ(scene_.modifierManager().tree(tree_)->nodes.back().type_id, "lfs.hsv_range");
    }

    TEST_F(NodeCanvasWidgets, RenderNeverPatchesDomAndReplacingGraphRebindsAndFrames) {
        attachGraph();
        auto& manager = scene_.modifierManager();
        auto* canvas = dynamic_cast<lfs::vis::gui::NodeCanvasElement*>(document_->GetElementById("node-editor-canvas"));
        auto* tree = manager.tree(tree_);
        const auto before = tree->to_json();
        tree->add_node("lfs.value", "Added");
        manager.recordTreeEdit(tree_, before);
        context_->Render();
        EXPECT_EQ(canvas->QuerySelector(".node-box[data-node=Added]"), nullptr);
        context_->Update();
        EXPECT_NE(canvas->QuerySelector(".node-box[data-node=Added]"), nullptr);
        ASSERT_TRUE(manager.removeTree(tree_));
        auto& replacement = manager.newTree("Replacement");
        replacement.add_node("lfs.colour_correct", "Fresh");
        manager.addModifier(host_, replacement.uuid);
        canvas->invalidateView();
        context_->Update();
        context_->Render();
        EXPECT_EQ(canvas->viewState()["tree"], replacement.uuid);
        EXPECT_NE(canvas->QuerySelector(".node-box[data-node=Fresh]"), nullptr);
        EXPECT_EQ(canvas->QuerySelector(".node-box[data-node=Correct]"), nullptr);
        EXPECT_EQ(canvas->viewState()["nodes"].size(), 3u);
        EXPECT_GT(canvas->viewState()["zoom"].get<float>(), 0.3f);
    }
} // namespace
