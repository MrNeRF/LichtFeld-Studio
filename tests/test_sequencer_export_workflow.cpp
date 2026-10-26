/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/localization_manager.hpp"
#include "gui/rml_modal_overlay.hpp"
#include "project/session_state.hpp"
#include "python/python_runtime.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "visualizer_impl.hpp"
#include <RmlUi/Core.h>
#include <RmlUi/Core/Elements/ElementFormControl.h>
#include <RmlUi/Core/Elements/ElementFormControlSelect.h>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <thread>

namespace lfs::vis {
    void installSessionTestInputController(VisualizerImpl& viewer);
}

namespace lfs::vis::gui {
    class SequencerExportWorkflowTest : public ::testing::Test {
    protected:
        class Renderer final : public Rml::RenderInterface {
        public:
            uintptr_t next_handle = 1;
            Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex>, Rml::Span<const int>) override { return next_handle++; }
            void RenderGeometry(Rml::CompiledGeometryHandle, Rml::Vector2f, Rml::TextureHandle) override {}
            void ReleaseGeometry(Rml::CompiledGeometryHandle) override {}
            Rml::TextureHandle LoadTexture(Rml::Vector2i&, const Rml::String&) override { return 0; }
            Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte>, Rml::Vector2i) override { return next_handle++; }
            void ReleaseTexture(Rml::TextureHandle) override {}
            void EnableScissorRegion(bool) override {}
            void SetScissorRegion(Rml::Rectanglei) override {}
        } renderer;
        RmlUIManager manager;
        Rml::Context* context = nullptr;
        Rml::ElementDocument* document = nullptr;

        void SetUp() override {
            ASSERT_TRUE(Rml::Initialise());
            ASSERT_TRUE(Rml::LoadFontFace((std::filesystem::path(PROJECT_ROOT_PATH) / "src/visualizer/gui/assets/fonts/Inter-Regular.ttf").string()));
        }
        void TearDown() override {
            manager.shutdown();
            if (context)
                Rml::RemoveContext("modal_form_behavior");
            Rml::Shutdown();
        }
        void checkExportForm(const std::function<void(VisualizerImpl&, RmlModalOverlay&, Rml::ElementDocument&)>& check,
                             const int width = 800, const int height = 600, const float scale = 1.0f, const char* language = "en") {
            ViewerOptions options;
            options.show_startup_overlay = false;
            VisualizerImpl viewer(options);
            auto& sequencer = viewer.getGuiManager()->sequencerUI();
            auto* modal = viewer.getGuiManager()->modalOverlay();
            auto* const viewer_manager = modal->rml_manager_;
            modal->rml_manager_ = &manager;
            auto& locales = lfs::event::LocalizationManager::getInstance();
            ASSERT_TRUE(locales.initialize((std::filesystem::path(PROJECT_ROOT_PATH) /
                                            "src/visualizer/gui/resources/locales")
                                               .string()));
            ASSERT_TRUE(locales.setLanguage(language));
            sequencer.showExportSettings();
            ASSERT_FALSE(modal->queue_.empty());
            modal->active_ = std::move(modal->queue_.back());
            modal->queue_.clear();
            context = Rml::CreateContext("modal_form_behavior", {width, height}, &renderer);
            ASSERT_NE(context, nullptr);
            context->SetDensityIndependentPixelRatio(scale);
            context->SetDefaultScrollBehavior(Rml::ScrollBehavior::Instant, 1.0f);
            const auto resources = std::filesystem::path(PROJECT_ROOT_PATH) / "src/visualizer/gui/rmlui/resources";
            const auto css = [&](const char* name) {
                std::ifstream file(resources / name);
                return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
            };
            document = context->LoadDocumentFromMemory("<rml><head><style>" + css("components.rcss") + css("modal_overlay.rcss") +
                                                       R"(</style></head><body id="body"><div id="backdrop"></div><div id="dialog" class="modal-dialog form-modal" style="display:block;width:560dp"><div class="modal-title">Video export</div><div class="modal-sep"></div><form id="form"><div id="content" class="modal-content"></div><div class="modal-sep"></div><div id="buttons" class="modal-button-row"></div></form></div></body></rml>)");
            ASSERT_NE(document, nullptr);
            modal->rml_context_ = context;
            modal->document_ = document;
            modal->el_form_ = document->GetElementById("form");
            modal->el_content_ = document->GetElementById("content");
            modal->el_button_row_ = document->GetElementById("buttons");
            modal->el_backdrop_ = document->GetElementById("backdrop");
            modal->el_dialog_ = document->GetElementById("dialog");
            modal->el_content_->SetInnerRML(modal->active_->body_rml);
            modal->updateButtons(modal->active_->buttons);
            modal->el_button_row_->AddEventListener(Rml::EventId::Click, &modal->listener_);
            modal->el_form_->AddEventListener(Rml::EventId::Change, &modal->listener_);
            modal->el_form_->AddEventListener(Rml::EventId::Click, &modal->listener_);
            modal->bindTextInputRevert();
            document->Show();
            context->Update();
            check(viewer, *modal, *document);
            modal->rml_manager_->deactivateInput(context);
            modal->text_input_revert_.clear();
            modal->el_form_->RemoveEventListener(Rml::EventId::Change, &modal->listener_);
            modal->el_form_->RemoveEventListener(Rml::EventId::Click, &modal->listener_);
            modal->el_button_row_->RemoveEventListener(Rml::EventId::Click, &modal->listener_);
            modal->rml_context_ = nullptr;
            modal->document_ = nullptr;
            modal->active_.reset();
            modal->queue_.clear();
            modal->rml_manager_ = viewer_manager;
            manager.shutdown();
            Rml::RemoveContext("modal_form_behavior");
            context = nullptr;
            document = nullptr;
        }

        static Rml::Element* nativeChild(Rml::Element* element, const char* tag) {
            if (element->GetTagName() == tag)
                return element;
            for (int i = 0; i < element->GetNumChildren(true); ++i)
                if (auto* found = nativeChild(element->GetChild(i), tag))
                    return found;
            return nullptr;
        }
        static void clearDirty(RmlModalOverlay& modal) { modal.render_needed_ = false; }
        static bool isDirty(const RmlModalOverlay& modal) { return modal.render_needed_; }
        void deliver(RmlModalOverlay& modal, const PanelInputState& input) {
            const bool cached = modal.elements_cached_;
            modal.elements_cached_ = true;
            modal.rml_manager_->routeInput(context, input, [&](const PanelInputState&) { modal.processInput(input); }, true);
            modal.elements_cached_ = cached;
            context->Update();
            context->Render();
        }
        void clickAt(RmlModalOverlay& modal, float x, float y) {
            PanelInputState input;
            input.mouse_x = x;
            input.mouse_y = y;
            deliver(modal, input);
            input.mouse_clicked[0] = true;
            input.mouse_down[0] = true;
            deliver(modal, input);
            input.mouse_clicked[0] = false;
            input.mouse_down[0] = false;
            input.mouse_released[0] = true;
            deliver(modal, input);
        }

        void checkNativeFpsInput(const float scale) {
            checkExportForm([&](VisualizerImpl&, RmlModalOverlay& modal, Rml::ElementDocument& doc) {
                auto* select = doc.QuerySelector("select[name=framerate_preset]");
                auto* input = dynamic_cast<Rml::ElementFormControl*>(doc.QuerySelector("input[name=framerate]"));
                ASSERT_NE(select, nullptr);
                ASSERT_NE(input, nullptr);
                EXPECT_TRUE(input->IsVisible(true));
                EXPECT_EQ(doc.QuerySelector(".modal-note.dim-text"), nullptr);
                select->ScrollIntoView(false);
                context->Update();
                context->Render();
                PanelInputState help_hover;
                const auto input_pos = input->GetAbsoluteOffset(Rml::BoxArea::Border);
                help_hover.mouse_x = input_pos.x + 10;
                help_hover.mouse_y = input_pos.y + 8;
                deliver(modal, help_hover);
                EXPECT_TRUE(modal.secondsUntilNextUpdate().has_value());
                const auto pos = select->GetAbsoluteOffset(Rml::BoxArea::Border);
                clickAt(modal, pos.x + select->GetOffsetWidth() - 9, pos.y + select->GetOffsetHeight() * 0.5f);
                auto* popup = nativeChild(select, "selectbox");
                ASSERT_NE(popup, nullptr);
                ASSERT_TRUE(popup->IsVisible(true));
                ASSERT_GT(popup->GetScrollHeight(), popup->GetClientHeight());
                const auto box = popup->GetAbsoluteOffset(Rml::BoxArea::Border);
                PanelInputState event;
                event.mouse_x = box.x + 30;
                event.mouse_y = box.y + popup->GetClientHeight() * 0.5f;
                clearDirty(modal);
                deliver(modal, event);
                auto* hovered = context->GetHoverElement();
                while (hovered && hovered->GetTagName() != "option")
                    hovered = hovered->GetParentNode();
                ASSERT_NE(hovered, nullptr);
                EXPECT_TRUE(hovered->IsPseudoClassSet("hover"));
                EXPECT_EQ(hovered->GetProperty<Rml::Colourb>("background-color"), Rml::Colourb(137, 180, 250, 255));
                EXPECT_TRUE(isDirty(modal));
                EXPECT_FALSE(modal.secondsUntilNextUpdate().has_value());
                const float before = popup->GetScrollTop();
                event.mouse_wheel = -1;
                deliver(modal, event);
                EXPECT_GT(popup->GetScrollTop(), before);
                auto* scrollbar = nativeChild(popup, "scrollbarvertical");
                ASSERT_NE(scrollbar, nullptr);
                auto* bar = nativeChild(scrollbar, "sliderbar");
                ASSERT_NE(bar, nullptr);
                const auto bar_pos = bar->GetAbsoluteOffset(Rml::BoxArea::Border);
                event = {};
                event.mouse_x = bar_pos.x + bar->GetOffsetWidth() * 0.5f;
                event.mouse_y = bar_pos.y + bar->GetOffsetHeight() * 0.5f;
                deliver(modal, event);
                const float before_drag = popup->GetScrollTop();
                event.mouse_clicked[0] = true;
                event.mouse_down[0] = true;
                deliver(modal, event);
                event.mouse_clicked[0] = false;
                event.mouse_y += 24;
                deliver(modal, event);
                event.mouse_down[0] = false;
                event.mouse_released[0] = true;
                deliver(modal, event);
                EXPECT_GT(popup->GetScrollTop(), before_drag);
                // Close the popup and enter an arbitrary fractional value in the
                // actual native text control, without selecting a Custom mode.
                clickAt(modal, input->GetAbsoluteOffset(Rml::BoxArea::Border).x + 20,
                        input->GetAbsoluteOffset(Rml::BoxArea::Border).y + 8);
                input->SetValue("");
                input->Focus();
                context->ProcessTextInput("31.25");
                context->Update();
                EXPECT_EQ(input->GetValue(), "31.25");
                EXPECT_TRUE(modal.current());
            },
                            1200, 900, scale);
        }

        void checkFramingPopupOverFps(const float scale) {
            checkExportForm([&](VisualizerImpl&, RmlModalOverlay& modal, Rml::ElementDocument& doc) {
                doc.QuerySelector("[data-form-action=swap]")->DispatchEvent("click", {});
                context->Update();
                context->Render();
                auto* framing = dynamic_cast<Rml::ElementFormControl*>(doc.QuerySelector("select[name=framing]"));
                auto* fps = doc.QuerySelector("input[name=framerate]");
                auto* presets = doc.QuerySelector("select[name=framerate_preset]");
                auto* width = doc.QuerySelector("input[name=width]");
                ASSERT_NE(framing, nullptr);
                ASSERT_NE(fps, nullptr);
                ASSERT_NE(presets, nullptr);
                ASSERT_NE(width, nullptr);
                EXPECT_NEAR(fps->GetOffsetHeight(), width->GetOffsetHeight(), 1.0f);
                EXPECT_NEAR(presets->GetOffsetHeight(), width->GetOffsetHeight(), 1.0f);
                // Use the real popup and pointer hit testing: a large popup z-index
                // alone does not escape an ancestor's stacking context.
                for (const int choice : {1, 2, 0}) {
                    framing->ScrollIntoView(true);
                    context->Update();
                    context->Render();
                    const auto pos = framing->GetAbsoluteOffset(Rml::BoxArea::Border);
                    modal.tooltip_.setHover("Framing help", framing);
                    std::this_thread::sleep_for(std::chrono::milliseconds(520));
                    modal.tooltip_.apply(doc.GetElementById("body"), static_cast<int>(pos.x), static_cast<int>(pos.y), 1200, 900);
                    auto* tooltip = doc.GetElementById("frame-tooltip");
                    ASSERT_NE(tooltip, nullptr);
                    ASSERT_TRUE(tooltip->IsClassSet("visible"));
                    clickAt(modal, pos.x + framing->GetOffsetWidth() - 9 * scale,
                            pos.y + framing->GetOffsetHeight() * 0.5f);
                    EXPECT_FALSE(tooltip->IsClassSet("visible"));
                    auto* popup = nativeChild(framing, "selectbox");
                    ASSERT_NE(popup, nullptr);
                    ASSERT_TRUE(popup->IsVisible(true));
                    Rml::ElementList options;
                    const auto collect_options = [&](auto&& self, Rml::Element* element) -> void {
                        if (element->GetTagName() == "option")
                            options.push_back(element);
                        for (int i = 0; i < element->GetNumChildren(true); ++i)
                            self(self, element->GetChild(i));
                    };
                    collect_options(collect_options, popup);
                    ASSERT_EQ(options.size(), 3);
                    auto* option = options.at(choice);
                    const auto option_pos = option->GetAbsoluteOffset(Rml::BoxArea::Border);
                    const auto fps_pos = fps->GetAbsoluteOffset(Rml::BoxArea::Border);
                    const float x = fps_pos.x + 10 * scale;
                    const float y = option_pos.y + option->GetOffsetHeight() * 0.5f;
                    PanelInputState hover;
                    hover.mouse_x = x;
                    hover.mouse_y = y;
                    deliver(modal, hover);
                    auto* hit = context->GetHoverElement();
                    while (hit && hit->GetTagName() != "option")
                        hit = hit->GetParentNode();
                    EXPECT_EQ(hit, option);
                    clickAt(modal, x, y);
                    EXPECT_EQ(framing->GetValue(), std::to_string(choice));
                    EXPECT_FALSE(popup->IsVisible(true));
                }
            },
                            1200, 900, scale);
        }

        void checkTransportMenuBehavior() {
            ViewerOptions options;
            options.show_startup_overlay = false;
            VisualizerImpl viewer(options);
            auto* gui = viewer.getGuiManager();
            auto& sequencer = gui->sequencerUI();
            auto& menu = gui->globalContextMenu();
            using Target = TransportContextMenuRequest::Target;
            sequencer.showTransportContextMenu({Target::SPEED, 10, 20});
            ASSERT_EQ(menu.pending_items_.size(), 5);
            EXPECT_TRUE(menu.pending_items_[2].is_active);
            const std::array<float, 5> speeds{0.25f, 0.5f, 1.0f, 2.0f, 4.0f};
            for (size_t i = 0; i < speeds.size(); ++i) {
                menu.callback_(menu.pending_items_[i].action);
                EXPECT_EQ(gui->getSequencerUIState().playback_speed, speeds[i]);
                EXPECT_EQ(sequencer.controller_.playbackSpeed(), speeds[i]);
            }
            menu.callback_("speed_999999999999999999999");
            EXPECT_EQ(sequencer.controller_.playbackSpeed(), 4.0f);
            sequencer.showTransportContextMenu({Target::PATH, 10, 20});
            ASSERT_EQ(menu.pending_items_.size(), 3);
            EXPECT_TRUE(menu.pending_items_[1].is_label);
            menu.callback_("load_path");
            EXPECT_TRUE(sequencer.panel_->load_path_requested_);
            sequencer.showTransportContextMenu({Target::PLY, 10, 20});
            menu.callback_("load_ply");
            EXPECT_TRUE(sequencer.panel_->load_sequence_requested_);
        }

        void checkClearCameraTimeline() {
            ViewerOptions options;
            options.show_startup_overlay = false;
            VisualizerImpl viewer(options);
            auto* gui = viewer.getGuiManager();
            auto& sequencer = gui->sequencerUI();
            auto& menu = gui->globalContextMenu();
            sequencer.controller_.setPlySequence("fixture", "PLY", {"frame.ply"}, {"frame"}, 24);
            sequencer.controller_.addKeyframe(sequencer::Keyframe{});
            sequencer.showTransportContextMenu({TransportContextMenuRequest::Target::PATH, 10, 20});
            menu.callback_("clear_path");
            auto* modal = gui->modalOverlay();
            ASSERT_TRUE(modal->current());
            EXPECT_EQ(sequencer.controller_.timeline().size(), 1);
            EXPECT_TRUE(modal->dismiss(modal->current()->button_labels.front()));
            EXPECT_EQ(sequencer.controller_.timeline().size(), 1);
            sequencer.showTransportContextMenu({TransportContextMenuRequest::Target::PATH, 10, 20});
            menu.callback_("clear_path");
            ASSERT_TRUE(modal->current());
            EXPECT_TRUE(modal->dismiss(modal->current()->button_labels.back()));
            EXPECT_EQ(sequencer.controller_.timeline().size(), 0);
            EXPECT_TRUE(sequencer.controller_.hasPlySequence());
            EXPECT_FLOAT_EQ(sequencer.controller_.plySequenceFps(), 24.0f);
        }

        void checkTransportClick(const char* id, const TransportContextMenuRequest::Target expected) {
            SequencerController controller;
            panels::SequencerUIState state;
            RmlSequencerPanel panel(controller, state, &manager);
            context = Rml::CreateContext("modal_form_behavior", {800, 600}, &renderer);
            ASSERT_NE(context, nullptr);
            document = context->LoadDocumentFromMemory(std::string("<rml><body><div id=\"") + id + "\"></div></body></rml>");
            ASSERT_NE(document, nullptr);
            auto* button = document->GetElementById(id);
            button->AddEventListener(Rml::EventId::Click, &panel.transport_listener_);
            button->DispatchEvent("click", {});
            EXPECT_EQ(panel.transport_ctx_request_.target, expected);
            EXPECT_EQ(state.playback_speed, 1.0f);
            EXPECT_FALSE(state.equirectangular);
            button->RemoveEventListener(Rml::EventId::Click, &panel.transport_listener_);
        }

        void checkEasingLayoutInteraction() {
            SequencerController controller;
            panels::SequencerUIState state;
            RmlSequencerPanel panel(controller, state, &manager);
            controller.addKeyframe(sequencer::Keyframe{});
            const auto resources = std::filesystem::path(PROJECT_ROOT_PATH) / "src/visualizer/gui/rmlui/resources";
            const auto read = [&](const char* name) {
                std::ifstream file(resources / name);
                return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
            };
            auto markup = read("sequencer.rml");
            const auto link = markup.find("<link");
            markup.erase(link, markup.find('>', link) - link + 1);
            markup.insert(markup.find("</head>"), "<style>" + read("components.rcss") + read("sequencer.rcss") + "</style>");
            context = Rml::CreateContext("modal_form_behavior", {600, 286}, &renderer);
            ASSERT_NE(context, nullptr);
            context->SetDensityIndependentPixelRatio(2.0f);
            document = context->LoadDocumentFromMemory(markup);
            ASSERT_NE(document, nullptr);
            document->Show();
            context->Update();
            auto* body = document->GetElementById("body");
            auto* stripe = document->GetElementById("easing-stripe");
            auto* timeline = document->GetElementById("timeline");
            ASSERT_GT(body->GetScrollHeight(), body->GetClientHeight());
            body->SetScrollTop(body->GetScrollHeight());
            context->Update();
            panel.el_easing_stripe_ = stripe;
            panel.cached_panel_x_ = 40.0f;
            panel.cached_panel_y_ = 60.0f;
            panel.cached_total_height_ = 286.0f;
            panel.cached_dp_ratio_ = 2.0f;
            // A stale legacy height must not override the rendered stripe.
            panel.cached_height_ = 1000.0f;
            const float timeline_x = 40.0f + timeline->GetAbsoluteOffset(Rml::BoxArea::Border).x;
            lfs::vis::PanelInputState input;
            input.mouse_x = timeline_x + 20.0f;
            input.mouse_y = 60.0f + stripe->GetAbsoluteOffset(Rml::BoxArea::Border).y + stripe->GetOffsetHeight() * 0.5f;
            input.mouse_clicked[1] = true;
            ASSERT_GE(input.mouse_y, 60.0f);
            ASSERT_LE(input.mouse_y, 346.0f);
            panel.handleEasingStripeInteraction(timeline_x, timeline->GetClientWidth(), input);
            EXPECT_TRUE(panel.context_menu_open_);
            panel.context_menu_open_ = false;
            body->SetScrollTop(0.0f);
            context->Update();
            input.mouse_y = 60.0f + stripe->GetAbsoluteOffset(Rml::BoxArea::Border).y + stripe->GetOffsetHeight() * 0.5f;
            ASSERT_GT(input.mouse_y, 346.0f);
            panel.handleEasingStripeInteraction(timeline_x, timeline->GetClientWidth(), input);
            EXPECT_FALSE(panel.context_menu_open_);
        }

        void checkCameraAspectMenu() {
            ViewerOptions options;
            options.show_startup_overlay = false;
            VisualizerImpl viewer(options);
            auto* gui = viewer.getGuiManager();
            auto& sequencer = gui->sequencerUI();
            auto& state = gui->getSequencerUIState();
            const auto viewport = gui->getViewportSize();
            auto& menu = gui->globalContextMenu();
            sequencer.showTransportContextMenu({TransportContextMenuRequest::Target::CAMERA_ASPECT, 10, 20});
            menu.callback_("aspect_9_16");
            EXPECT_EQ(state.camera_width, 9);
            EXPECT_EQ(state.camera_height, 16);
            EXPECT_EQ(state.outputWidth(), 1920);
            EXPECT_EQ(state.outputHeight(), 1080);
            const auto key = sequencer.currentPipPreviewKey();
            EXPECT_LT(key.width, key.height);
            EXPECT_NEAR(static_cast<float>(key.width) / key.height, 9.0f / 16, 0.01f);
            EXPECT_EQ(gui->getViewportSize(), viewport);
            menu.callback_("aspect_999999999999_0");
            EXPECT_EQ(state.camera_width, 9);
            state.equirectangular = true;
            menu.callback_("aspect_1_1");
            EXPECT_EQ(state.camera_width, 9);
            const auto pano = sequencer.currentPipPreviewKey();
            EXPECT_EQ(pano.width, 2 * pano.height);
        }

        void checkGizmoModalGate(const bool active) {
            ViewerOptions options;
            options.show_startup_overlay = false;
            VisualizerImpl viewer(options);
            auto& sequencer = viewer.getGuiManager()->sequencerUI();
            auto* modal = viewer.getGuiManager()->modalOverlay();
            if (active) {
                modal->active_.emplace();
            } else {
                core::ModalRequest request;
                request.buttons = {{"Cancel", "secondary"}};
                modal->enqueue(std::move(request));
                ASSERT_TRUE(modal->hasPendingRequest());
            }
            // No selected keyframe: normal gizmo validation would end editing.
            // A modal must defer even that work until its underlay is usable.
            sequencer.viewport_edit_mode_ = SequencerViewportEditMode::Translate;
            sequencer.renderKeyframeGizmo(UIContext{}, ViewportLayout{});
            EXPECT_EQ(sequencer.viewport_edit_mode_, SequencerViewportEditMode::Translate);
            if (active)
                modal->active_.reset();
            else
                ASSERT_TRUE(modal->dismiss("Cancel"));
            sequencer.renderKeyframeGizmo(UIContext{}, ViewportLayout{});
            EXPECT_EQ(sequencer.viewport_edit_mode_, SequencerViewportEditMode::None);
        }
    };

    TEST_F(SequencerExportWorkflowTest, OpenModalDefersSequencerGizmoWorkUntilDismissed) {
        checkGizmoModalGate(true);
    }

    TEST_F(SequencerExportWorkflowTest, PendingModalDefersSequencerGizmoWorkUntilDismissed) {
        checkGizmoModalGate(false);
    }

    TEST_F(SequencerExportWorkflowTest, ProductionExportFormKeepsWidthControlAndCancelDiscardsCoupledDimensions) {
        checkExportForm([&](VisualizerImpl& viewer, RmlModalOverlay& modal, Rml::ElementDocument& doc) {
            auto* width = doc.QuerySelector("input[name=width]");
            auto* height = doc.QuerySelector("input[name=height]");
            ASSERT_NE(width, nullptr);
            ASSERT_NE(height, nullptr);
            width->SetAttribute("value", "1280");
            width->DispatchEvent("change", {});
            EXPECT_EQ(doc.QuerySelector("input[name=width]"), width);
            EXPECT_EQ(height->GetAttribute<Rml::String>("value", ""), "720");
            EXPECT_EQ(viewer.getGuiManager()->getSequencerUIState().outputWidth(), 1920);
            ASSERT_TRUE(modal.current());
            EXPECT_TRUE(modal.dismiss(modal.current()->button_labels.back()));
            EXPECT_EQ(viewer.getGuiManager()->getSequencerUIState().outputWidth(), 1920);
            EXPECT_EQ(viewer.getGuiManager()->getSequencerUIState().outputHeight(), 1080);
        });
    }

    TEST_F(SequencerExportWorkflowTest, ProductionExportFormRejectsOverflowAndCancelPreservesPreferences) {
        checkExportForm([&](VisualizerImpl& viewer, RmlModalOverlay& modal, Rml::ElementDocument& doc) {
            auto* width = doc.QuerySelector("input[name=width]");
            ASSERT_NE(width, nullptr);
            width->SetAttribute("value", "4294971136");
            width->DispatchEvent("change", {});
            // Programmatic dispatch queues button updates until the next frame.
            modal.render(800, 600);
            const auto snapshot = modal.current();
            ASSERT_TRUE(snapshot);
            EXPECT_FALSE(snapshot->button_enabled[0]);
            EXPECT_TRUE(snapshot->button_enabled[1]);
            EXPECT_FALSE(modal.dismiss(snapshot->button_labels[0]));
            ASSERT_TRUE(modal.dismiss(snapshot->button_labels.back()));
            EXPECT_EQ(viewer.getGuiManager()->getSequencerUIState().outputWidth(), 1920);
        });
    }

    TEST_F(SequencerExportWorkflowTest, InvalidExportDoesNotFallThroughToCancelOnEnter) {
        checkExportForm([&](VisualizerImpl&, RmlModalOverlay& modal, Rml::ElementDocument& doc) {
            auto* width = doc.QuerySelector("input[name=width]");
            ASSERT_NE(width, nullptr);
            width->Focus();
            width->SetAttribute("value", "4294971136");
            width->DispatchEvent("change", {});
            PanelInputState enter;
            enter.input_events.push_back({.kind = FrameInputEventKind::KeyDown, .scancode = SDL_SCANCODE_RETURN, .keycode = SDLK_RETURN});
            deliver(modal, enter);
            ASSERT_TRUE(modal.current());
            EXPECT_FALSE(modal.current()->button_enabled.front());
            doc.GetElementById("body")->Focus();
            deliver(modal, enter);
            EXPECT_TRUE(modal.current());
        });
    }

    TEST_F(SequencerExportWorkflowTest, EscapeRestoresExportDraftAndCoupledDimension) {
        checkExportForm([&](VisualizerImpl&, RmlModalOverlay& modal, Rml::ElementDocument& doc) {
            auto* width = doc.QuerySelector("input[name=width]");
            auto* height = doc.QuerySelector("input[name=height]");
            ASSERT_NE(width, nullptr);
            ASSERT_NE(height, nullptr);
            width->Focus();
            width->SetAttribute("value", "1280");
            width->DispatchEvent("change", {});
            EXPECT_EQ(height->GetAttribute<Rml::String>("value", ""), "720");
            PanelInputState escape;
            escape.input_events.push_back({.kind = FrameInputEventKind::KeyDown, .scancode = SDL_SCANCODE_ESCAPE, .keycode = SDLK_ESCAPE});
            deliver(modal, escape);
            EXPECT_EQ(width->GetAttribute<Rml::String>("value", ""), "1920");
            EXPECT_EQ(height->GetAttribute<Rml::String>("value", ""), "1080");
            EXPECT_TRUE(modal.current());
        });
    }

    TEST_F(SequencerExportWorkflowTest, ProductionExportSwapKeepsControlsUsableForFurtherInput) {
        checkExportForm([&](VisualizerImpl& viewer, RmlModalOverlay& modal, Rml::ElementDocument& doc) {
            auto* swap = doc.QuerySelector("[data-form-action=swap]");
            auto* width = doc.QuerySelector("input[name=width]");
            auto* height = doc.QuerySelector("input[name=height]");
            ASSERT_NE(swap, nullptr);
            swap->DispatchEvent("click", {});
            modal.render(800, 600);
            ASSERT_EQ(doc.QuerySelector("[data-form-action=swap]"), swap);
            ASSERT_EQ(doc.QuerySelector("input[name=width]"), width);
            ASSERT_EQ(doc.QuerySelector("input[name=height]"), height);
            EXPECT_EQ(width->GetAttribute<Rml::String>("value", ""), "1080");
            EXPECT_EQ(height->GetAttribute<Rml::String>("value", ""), "1920");
            height->SetAttribute("value", "1280");
            height->DispatchEvent("change", {});
            EXPECT_EQ(width->GetAttribute<Rml::String>("value", ""), "720");
            EXPECT_EQ(viewer.getGuiManager()->getSequencerUIState().outputWidth(), 1920);
            ASSERT_TRUE(modal.current());
            ASSERT_EQ(modal.current()->button_labels.size(), 2);
            auto* cancel = doc.GetElementById("modal-btn-1");
            ASSERT_NE(cancel, nullptr);
            cancel->DispatchEvent("click", {});
            modal.render(800, 600);
            EXPECT_FALSE(modal.current());
            EXPECT_EQ(viewer.getGuiManager()->getSequencerUIState().outputWidth(), 1920);
            EXPECT_EQ(viewer.getGuiManager()->getSequencerUIState().outputHeight(), 1080);
        });
    }

    TEST_F(SequencerExportWorkflowTest, FittingGroupAppearsOnlyForDifferentRatiosWithoutReplacingControls) {
        checkExportForm([&](VisualizerImpl&, RmlModalOverlay& modal, Rml::ElementDocument& doc) {
            auto* group = doc.QuerySelector("[data-form-visible=framing_visible]");
            auto* select = doc.QuerySelector("select[name=framing]");
            auto* swap = doc.QuerySelector("[data-form-action=swap]");
            ASSERT_NE(group, nullptr);
            ASSERT_NE(select, nullptr);
            ASSERT_NE(swap, nullptr);
            EXPECT_EQ(group->GetProperty<Rml::Style::Display>("display"), Rml::Style::Display::None);
            EXPECT_EQ(doc.QuerySelector("input[name=crop_zoom]"), nullptr);
            swap->DispatchEvent("click", {});
            modal.render(800, 600);
            EXPECT_EQ(group->GetProperty<Rml::Style::Display>("display"), Rml::Style::Display::Block);
            EXPECT_EQ(doc.QuerySelector("select[name=framing]"), select);
            swap->DispatchEvent("click", {});
            modal.render(800, 600);
            EXPECT_EQ(group->GetProperty<Rml::Style::Display>("display"), Rml::Style::Display::None);
            EXPECT_EQ(doc.QuerySelector("select[name=framing]"), select);
            EXPECT_TRUE(modal.current());
        });
    }

    TEST_F(SequencerExportWorkflowTest, CommonFpsMenuAndManualEntryUpdateTheSameDraft) {
        checkExportForm([&](VisualizerImpl& viewer, RmlModalOverlay& modal, Rml::ElementDocument& doc) {
            auto* preset = dynamic_cast<Rml::ElementFormControlSelect*>(doc.QuerySelector("select[name=framerate_preset]"));
            auto* input = doc.QuerySelector("input[name=framerate]");
            ASSERT_NE(preset, nullptr);
            ASSERT_NE(input, nullptr);
            EXPECT_EQ(preset->GetNumOptions(), 22);
            for (int i = 0; i < preset->GetNumOptions(); ++i)
                EXPECT_FALSE(preset->GetOption(i)->GetAttribute<Rml::String>("value", "").empty());
            EXPECT_TRUE(input->IsVisible(true));
            preset->SetValue("60");
            EXPECT_EQ(input->GetAttribute<Rml::String>("value", ""), "60");
            EXPECT_EQ(preset->GetValue(), "60");
            input->SetAttribute("value", "31");
            input->DispatchEvent("change", {});
            EXPECT_EQ(preset->GetValue(), "");
            input->SetAttribute("value", "30");
            input->DispatchEvent("change", {});
            EXPECT_EQ(preset->GetValue(), "30");
            preset->SetValue("29.97");
            EXPECT_EQ(input->GetAttribute<Rml::String>("value", ""), "29.97");
            EXPECT_EQ(preset->GetValue(), "29.97");
            EXPECT_EQ(viewer.getGuiManager()->getSequencerUIState().framerate, 30);
            ASSERT_TRUE(modal.dismiss(modal.current()->button_labels.back()));
            EXPECT_EQ(viewer.getGuiManager()->getSequencerUIState().framerate, 30);
        });
    }

    TEST_F(SequencerExportWorkflowTest, NativeFpsPopupScrollsWithWheelAndScrollbarAndManualInputIsAlwaysEditable) { checkNativeFpsInput(1.0f); }
    TEST_F(SequencerExportWorkflowTest, NativeFpsPopupInputAtDoubleScale) { checkNativeFpsInput(2.0f); }

    TEST_F(SequencerExportWorkflowTest, FramingPopupOverlaysFpsAndUsesStandardControlHeight) { checkFramingPopupOverFps(1.0f); }
    TEST_F(SequencerExportWorkflowTest, FramingPopupOverlaysFpsAtDoubleScale) { checkFramingPopupOverFps(2.0f); }

    TEST_F(SequencerExportWorkflowTest, QualitySliderAndManualCrfStaySynchronizedWithoutReplacingControls) {
        checkExportForm([&](VisualizerImpl& viewer, RmlModalOverlay& modal, Rml::ElementDocument& doc) {
            auto* slider = dynamic_cast<Rml::ElementFormControl*>(doc.QuerySelector("input[name=quality_slider]"));
            auto* input = doc.QuerySelector("input[name=quality]");
            ASSERT_NE(slider, nullptr);
            ASSERT_NE(input, nullptr);
            const auto internal = [](auto&& self, Rml::Element* element, const char* tag) -> Rml::Element* {
                if (element->GetTagName() == tag)
                    return element;
                for (int i = 0; i < element->GetNumChildren(true); ++i)
                    if (auto* found = self(self, element->GetChild(i), tag))
                        return found;
                return nullptr;
            };
            auto* track = internal(internal, slider, "slidertrack");
            auto* thumb = internal(internal, slider, "sliderbar");
            ASSERT_NE(track, nullptr);
            ASSERT_NE(thumb, nullptr);
            EXPECT_NEAR(track->GetAbsoluteOffset(Rml::BoxArea::Border).y + track->GetOffsetHeight() * 0.5f,
                        thumb->GetAbsoluteOffset(Rml::BoxArea::Border).y + thumb->GetOffsetHeight() * 0.5f, 1.0f);
            EXPECT_NEAR(track->GetAbsoluteOffset(Rml::BoxArea::Border).y + track->GetOffsetHeight() * 0.5f,
                        input->GetAbsoluteOffset(Rml::BoxArea::Border).y + input->GetOffsetHeight() * 0.5f, 1.0f);
            EXPECT_GE(slider->GetOffsetHeight(), 28.0f);
            EXPECT_EQ(doc.QuerySelector(".export-quality-legend"), nullptr);
            EXPECT_NE(doc.GetElementById("export-quality-label")->GetAttribute<Rml::String>("data-tooltip", ""), "");
            slider->SetValue("51");
            EXPECT_EQ(input->GetAttribute<Rml::String>("value", ""), "0");
            input->SetAttribute("value", "51");
            input->DispatchEvent("change", {});
            EXPECT_EQ(slider->GetValue(), "0.000000");
            EXPECT_EQ(doc.QuerySelector("input[name=quality]"), input);
            EXPECT_EQ(viewer.getGuiManager()->getSequencerUIState().quality, 18);
            ASSERT_TRUE(modal.dismiss(modal.current()->button_labels.back()));
            EXPECT_EQ(viewer.getGuiManager()->getSequencerUIState().quality, 18);
        });
    }

    TEST_F(SequencerExportWorkflowTest, ExportControlsReserveScrollbarGutterAndFooterFitsScaledWindows) {
        for (const auto* language : {"en", "it", "de", "es", "fr", "ja", "ko", "nl", "pl", "zh"}) {
            for (const auto size : {glm::ivec3{800, 600, 1}, glm::ivec3{1280, 720, 2}, glm::ivec3{1040, 720, 2}, glm::ivec3{800, 720, 2}}) {
                SCOPED_TRACE(std::string(language) + " width=" + std::to_string(size.x));
                checkExportForm([&](VisualizerImpl&, RmlModalOverlay& modal, Rml::ElementDocument& doc) {
                    auto* form = doc.QuerySelector(".sequencer-export-form");
                    ASSERT_NE(form, nullptr);
                    const float right = form->GetAbsoluteOffset(Rml::BoxArea::Border).x + form->GetOffsetWidth() - 8.0f * size.z;
                    Rml::ElementList controls;
                    form->QuerySelectorAll(controls, "input, select, button");
                    for (auto* control : controls) {
                        SCOPED_TRACE(control->GetAttribute<Rml::String>("name", "button"));
                        EXPECT_LE(control->GetAbsoluteOffset(Rml::BoxArea::Border).x + control->GetOffsetWidth(), right + 1.0f);
                    }
                    auto* row = doc.QuerySelector("input[name=quality_slider]");
                    auto* number = doc.QuerySelector("input[name=quality]");
                    auto* range = doc.QuerySelector("input[name=quality_slider]");
                    ASSERT_NE(row, nullptr);
                    ASSERT_NE(number, nullptr);
                    ASSERT_NE(range, nullptr);
                    EXPECT_NEAR(row->GetAbsoluteOffset(Rml::BoxArea::Border).y + row->GetOffsetHeight() * 0.5f,
                                number->GetAbsoluteOffset(Rml::BoxArea::Border).y + number->GetOffsetHeight() * 0.5f, 1.0f);
                    EXPECT_LE(doc.GetElementById("dialog")->GetOffsetHeight(), static_cast<float>(size.y));
                    auto* fps = dynamic_cast<Rml::ElementFormControl*>(doc.QuerySelector("select[name=framerate_preset]"));
                    ASSERT_NE(fps, nullptr);
                    fps->SetValue("");
                    modal.render(size.x, size.y);
                    auto* manual = doc.QuerySelector("input[name=framerate]");
                    ASSERT_NE(manual, nullptr);
                    EXPECT_LE(manual->GetAbsoluteOffset(Rml::BoxArea::Border).x + manual->GetOffsetWidth(), right + 1.0f);
                    EXPECT_LE(doc.GetElementById("dialog")->GetOffsetHeight(), static_cast<float>(size.y));
                },
                                size.x, size.y, static_cast<float>(size.z), language);
            }
        }
    }

    TEST_F(SequencerExportWorkflowTest, ClearingCameraTimelineRequiresConfirmationAndKeepsPly) { checkClearCameraTimeline(); }

    TEST_F(SequencerExportWorkflowTest, TransportMenusApplyExplicitChoicesAndQueueFileActions) { checkTransportMenuBehavior(); }
    TEST_F(SequencerExportWorkflowTest, SpeedClickOpensMenuWithoutCycling) { checkTransportClick("btn-speed", TransportContextMenuRequest::Target::SPEED); }
    TEST_F(SequencerExportWorkflowTest, SnapOptionsClickDoesNotToggleSnap) { checkTransportClick("btn-snap-options", TransportContextMenuRequest::Target::SNAP); }
    TEST_F(SequencerExportWorkflowTest, ProjectionClickOpensMenuWithoutChangingProjection) { checkTransportClick("btn-equirect", TransportContextMenuRequest::Target::PROJECTION); }
    TEST_F(SequencerExportWorkflowTest, EasingHitTestingFollowsRenderedStripeAfterToolbarWrapAndScroll) { checkEasingLayoutInteraction(); }

    TEST_F(SequencerExportWorkflowTest, CameraAndFractionalExportPreferencesRoundTripAndLegacyProjectsKeepFraming) {
        ViewerOptions options;
        options.show_startup_overlay = false;
        VisualizerImpl viewer(options);
        auto& ui = viewer.getGuiManager()->getSequencerUIState();
        ui.camera_width = 9;
        ui.camera_height = 16;
        ui.framerate = 30000;
        ui.framerate_denominator = 1001;
        ui.export_fit = 1;
        installSessionTestInputController(viewer);
        auto captured = project::captureGuiSession(viewer, {}, {});
        ASSERT_TRUE(captured) << captured.error().user_message();
        const auto serialized = io::JsonChapterDom::Json::parse(captured->sequencer.dom().dump());
        EXPECT_EQ(serialized["preferences"]["framerate"], 30);
        EXPECT_EQ(serialized["preferences"]["framerate_numerator"], 30000);
        EXPECT_EQ(serialized["preferences"]["framerate_denominator"], 1001);
        auto prepared = project::prepareGuiSessionRestore(*captured);
        ASSERT_TRUE(prepared);
        ui.camera_width = 1;
        ui.camera_height = 1;
        ui.framerate = 24;
        ui.framerate_denominator = 1;
        ui.export_fit = 0;
        std::vector<project::CameraBookmarkProjectState> bookmarks;
        project::applyGuiSession(viewer, *prepared, bookmarks);
        EXPECT_EQ(ui.camera_width, 9);
        EXPECT_EQ(ui.camera_height, 16);
        EXPECT_EQ(ui.framerate, 30000);
        EXPECT_EQ(ui.framerate_denominator, 1001);
        EXPECT_EQ(ui.export_fit, 1);
        auto legacy = io::JsonChapterDom::Json::parse(captured->sequencer.dom().dump());
        for (const auto* key : {"camera_width", "camera_height", "framerate_numerator", "framerate_denominator", "export_fit"})
            legacy["preferences"].erase(key);
        legacy["preferences"]["preset"] = "tiktok";
        legacy["preferences"]["framerate"] = 25;
        const auto chapter = io::project::SequencerSessionChapter::parse(legacy.dump());
        ASSERT_TRUE(chapter);
        captured->sequencer = *chapter;
        prepared = project::prepareGuiSessionRestore(*captured);
        ASSERT_TRUE(prepared);
        project::applyGuiSession(viewer, *prepared, bookmarks);
        EXPECT_EQ(ui.camera_width, 9);
        EXPECT_EQ(ui.camera_height, 16);
        EXPECT_EQ(ui.framerate, 25);
        EXPECT_EQ(ui.framerate_denominator, 1);
        EXPECT_EQ(ui.export_fit, 0);
    }

    TEST_F(SequencerExportWorkflowTest, CameraAspectChangesPreviewWithoutChangingExportOrViewport) { checkCameraAspectMenu(); }

    TEST_F(SequencerExportWorkflowTest, NtscProjectsKeepLegacyFrameRatesUsableAndExactRatesRoundTrip) {
        ViewerOptions options;
        options.show_startup_overlay = false;
        VisualizerImpl viewer(options);
        installSessionTestInputController(viewer);
        auto& ui = viewer.getGuiManager()->getSequencerUIState();
        for (const auto [numerator, rounded] : {std::pair{24000, 24}, {30000, 30}, {60000, 60}}) {
            ui.framerate = numerator;
            ui.framerate_denominator = 1001;
            auto captured = project::captureGuiSession(viewer, {}, {});
            ASSERT_TRUE(captured);
            const auto json = io::JsonChapterDom::Json::parse(captured->sequencer.dom().dump());
            EXPECT_EQ(json["preferences"]["framerate"], rounded);
            EXPECT_EQ(json["preferences"]["framerate_numerator"], numerator);
            auto prepared = project::prepareGuiSessionRestore(*captured);
            ASSERT_TRUE(prepared);
            ui.framerate = 25;
            ui.framerate_denominator = 1;
            std::vector<project::CameraBookmarkProjectState> bookmarks;
            project::applyGuiSession(viewer, *prepared, bookmarks);
            EXPECT_EQ(ui.framerate, numerator);
            EXPECT_EQ(ui.framerate_denominator, 1001);
        }
    }

    TEST_F(SequencerExportWorkflowTest, ResolutionMenuIsGenericAndHasNoDuplicateSizes) {
        checkExportForm([&](VisualizerImpl&, RmlModalOverlay&, Rml::ElementDocument& doc) {
            auto* presets = dynamic_cast<Rml::ElementFormControlSelect*>(doc.QuerySelector("select[name=preset]"));
            ASSERT_NE(presets, nullptr);
            EXPECT_EQ(presets->GetNumOptions(), io::video::getPresetCount() - 1);
            for (int i = 0; i < presets->GetNumOptions(); ++i) {
                const auto* option = presets->GetOption(i);
                const auto text = option->GetInnerRML();
                EXPECT_EQ(text.find("YouTube"), std::string::npos);
                EXPECT_EQ(text.find("TikTok"), std::string::npos);
                EXPECT_EQ(text.find("Instagram"), std::string::npos);
                EXPECT_NE(option->GetAttribute<Rml::String>("value", ""), std::to_string(static_cast<int>(io::video::VideoPreset::TIKTOK_HD)));
            }
        });
    }

    TEST_F(SequencerExportWorkflowTest, LegacyWriterFrameRateChangesOverridePreservedExactFields) {
        ViewerOptions options;
        options.show_startup_overlay = false;
        VisualizerImpl viewer(options);
        installSessionTestInputController(viewer);
        auto& ui = viewer.getGuiManager()->getSequencerUIState();
        ui.framerate = 30000;
        ui.framerate_denominator = 1001;
        const auto original = project::captureGuiSession(viewer, {}, {});
        ASSERT_TRUE(original);
        for (const int legacy_fps : {30, 60, 24}) {
            auto captured = *original;
            ASSERT_TRUE(captured.sequencer.merge_known_state({{"preferences", {{"framerate", legacy_fps}}}}));
            const auto prepared = project::prepareGuiSessionRestore(captured);
            ASSERT_TRUE(prepared);
            ui.framerate = 25;
            ui.framerate_denominator = 1;
            std::vector<project::CameraBookmarkProjectState> bookmarks;
            project::applyGuiSession(viewer, *prepared, bookmarks);
            EXPECT_EQ(ui.framerate, legacy_fps == 30 ? 30000 : legacy_fps);
            EXPECT_EQ(ui.framerate_denominator, legacy_fps == 30 ? 1001 : 1);
        }
    }

    TEST_F(SequencerExportWorkflowTest, EarlierBranchProjectsMigrateNumeratorFromLegacyKey) {
        ViewerOptions options;
        options.show_startup_overlay = false;
        VisualizerImpl viewer(options);
        installSessionTestInputController(viewer);
        auto& ui = viewer.getGuiManager()->getSequencerUIState();
        for (const auto [n, d] : {std::pair{24000, 1001}, {30000, 1001}, {60000, 1001}, {125, 4}}) {
            ui.framerate = n;
            ui.framerate_denominator = d;
            auto captured = project::captureGuiSession(viewer, {}, {});
            ASSERT_TRUE(captured);
            auto json = io::JsonChapterDom::Json::parse(captured->sequencer.dom().dump());
            json["preferences"].erase("framerate_numerator");
            json["preferences"]["framerate"] = n;
            const auto chapter = io::project::SequencerSessionChapter::parse(json.dump());
            ASSERT_TRUE(chapter);
            captured->sequencer = *chapter;
            const auto prepared = project::prepareGuiSessionRestore(*captured);
            ASSERT_TRUE(prepared);
            ui.framerate = 25;
            ui.framerate_denominator = 1;
            std::vector<project::CameraBookmarkProjectState> bookmarks;
            project::applyGuiSession(viewer, *prepared, bookmarks);
            EXPECT_EQ(ui.framerate, n);
            EXPECT_EQ(ui.framerate_denominator, d);
            const auto upgraded = project::captureGuiSession(viewer, {}, {});
            ASSERT_TRUE(upgraded);
            const auto saved = io::JsonChapterDom::Json::parse(upgraded->sequencer.dom().dump());
            EXPECT_EQ(saved["preferences"]["framerate_numerator"], n);
            EXPECT_EQ(saved["preferences"]["framerate_denominator"], d);
            EXPECT_EQ(saved["preferences"]["framerate"], (n + d / 2) / d);
        }
    }

    TEST_F(SequencerExportWorkflowTest, InvalidExactFrameRatesFallBackWithoutUsingPreviousProjectState) {
        ViewerOptions options;
        options.show_startup_overlay = false;
        VisualizerImpl viewer(options);
        installSessionTestInputController(viewer);
        auto& ui = viewer.getGuiManager()->getSequencerUIState();
        const auto original = project::captureGuiSession(viewer, {}, {});
        ASSERT_TRUE(original);
        using Json = io::JsonChapterDom::Json;
        for (const auto pair : {Json::array({30000, 0}), Json::array({0, 1001}), Json::array({30000, -1}),
                                Json::array({"bad", 1001}), Json::array({30000, "bad"}), Json::array({2147483647, 1}),
                                Json::array({4294997296LL, 1001}), Json::array({30000, 4294968297LL})}) {
            auto captured = *original;
            auto json = Json::parse(captured.sequencer.dom().dump());
            json["preferences"]["framerate"] = 30;
            json["preferences"]["framerate_numerator"] = pair[0];
            json["preferences"]["framerate_denominator"] = pair[1];
            const auto chapter = io::project::SequencerSessionChapter::parse(json.dump());
            ASSERT_TRUE(chapter);
            captured.sequencer = *chapter;
            const auto prepared = project::prepareGuiSessionRestore(captured);
            ASSERT_TRUE(prepared);
            ui.framerate = 60000;
            ui.framerate_denominator = 1001;
            std::vector<project::CameraBookmarkProjectState> bookmarks;
            project::applyGuiSession(viewer, *prepared, bookmarks);
            EXPECT_EQ(ui.framerate, 30);
            EXPECT_EQ(ui.framerate_denominator, 1);
        }
    }
    TEST_F(SequencerExportWorkflowTest, CameraAspectClickOpensMenu) { checkTransportClick("btn-camera-aspect", TransportContextMenuRequest::Target::CAMERA_ASPECT); }
} // namespace lfs::vis::gui
