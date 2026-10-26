/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rml_modal_overlay.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include <RmlUi/Core.h>
#include <RmlUi/Core/Elements/ElementFormControlSelect.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <thread>

namespace lfs::vis::gui {
    class ModalFormBehaviorTest : public ::testing::Test {
    protected:
        class Renderer final : public Rml::RenderInterface {
            uintptr_t next = 1;
            Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex>, Rml::Span<const int>) override { return next++; }
            void RenderGeometry(Rml::CompiledGeometryHandle, Rml::Vector2f, Rml::TextureHandle) override {}
            void ReleaseGeometry(Rml::CompiledGeometryHandle) override {}
            Rml::TextureHandle LoadTexture(Rml::Vector2i&, const Rml::String&) override { return 0; }
            Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte>, Rml::Vector2i) override { return next++; }
            void ReleaseTexture(Rml::TextureHandle) override {}
            void EnableScissorRegion(bool) override {}
            void SetScissorRegion(Rml::Rectanglei) override {}
        } renderer;
        RmlUIManager manager;
        RmlModalOverlay overlay{&manager};
        Rml::Context* context = nullptr;
        Rml::ElementDocument* document = nullptr;
        const std::filesystem::path resources = std::filesystem::path(PROJECT_ROOT_PATH) / "src/visualizer/gui";
        void SetUp() override {
            ASSERT_TRUE(Rml::Initialise());
            ASSERT_TRUE(Rml::LoadFontFace((resources / "assets/fonts/Inter-Regular.ttf").string()));
        }
        void TearDown() override {
            overlay.text_input_revert_.clear();
            manager.shutdown();
            overlay.rml_context_ = nullptr;
            overlay.document_ = nullptr;
            if (context)
                Rml::RemoveContext("modal_form_behavior");
            Rml::Shutdown();
        }
        void openForm(const std::string& controls = "<input id='field' name='field' type='text' value='original' title='Field help' />") {
            context = Rml::CreateContext("modal_form_behavior", {800, 600}, &renderer);
            ASSERT_NE(context, nullptr);
            context->SetDefaultScrollBehavior(Rml::ScrollBehavior::Instant, 1.0f);
            const auto css = [&](const char* name) {
                std::ifstream file(resources / "rmlui/resources" / name);
                return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
            };
            document = context->LoadDocumentFromMemory("<rml><head><style>" + css("components.rcss") + css("modal_overlay.rcss") +
                                                       "</style></head><body id='body'><div id='backdrop'></div><div id='dialog' class='form-modal'><form id='form'><div id='content'>" + controls +
                                                       "</div><div id='buttons'></div></form></div></body></rml>");
            ASSERT_NE(document, nullptr);
            overlay.active_.emplace();
            overlay.active_->key = "test.form";
            overlay.active_->buttons = {{"Cancel", "secondary"}};
            overlay.rml_context_ = context;
            overlay.document_ = document;
            overlay.elements_cached_ = true;
            overlay.el_form_ = document->GetElementById("form");
            overlay.el_content_ = document->GetElementById("content");
            overlay.el_button_row_ = document->GetElementById("buttons");
            overlay.el_backdrop_ = document->GetElementById("backdrop");
            overlay.el_dialog_ = document->GetElementById("dialog");
            overlay.el_form_->AddEventListener(Rml::EventId::Change, &overlay.listener_);
            overlay.bindTextInputRevert();
            document->Show();
            context->Update();
            context->Render();
        }
        void send(const PanelInputState& input) {
            manager.routeInput(context, input, [this, input](const PanelInputState&) { overlay.processInput(input); }, true);
            context->Update();
            context->Render();
        }
        PanelInputState over(Rml::Element* element) {
            const auto pos = element->GetAbsoluteOffset(Rml::BoxArea::Border);
            PanelInputState input;
            input.mouse_x = pos.x + 8;
            input.mouse_y = pos.y + element->GetOffsetHeight() * 0.5f;
            return input;
        }
        void click(PanelInputState input) {
            send(input);
            input.mouse_clicked[0] = input.mouse_down[0] = true;
            send(input);
            input.mouse_clicked[0] = input.mouse_down[0] = false;
            input.mouse_released[0] = true;
            send(input);
        }
        void key(PanelInputState input, SDL_Scancode code, SDL_Keycode symbol) {
            input.input_events.push_back({.kind = FrameInputEventKind::KeyDown, .scancode = code, .keycode = symbol});
            send(input);
        }
        void applyTooltip() { overlay.tooltip_.apply(document->GetElementById("body"), 20, 20, 800, 600); }
        bool tooltipVisible() {
            auto* tip = document->GetElementById("frame-tooltip");
            return tip && tip->IsClassSet("visible");
        }
        void reveal(PanelInputState input) {
            send(input);
            std::this_thread::sleep_for(std::chrono::milliseconds(520));
            applyTooltip();
            ASSERT_TRUE(tooltipVisible());
        }
        void changeHandler(std::function<void(const core::ModalResult&)> callback) { overlay.active_->on_change = std::move(callback); }
        void cancel() { overlay.cancel(); }
        void resultHandler(std::function<void(const core::ModalResult&)> callback) { overlay.active_->on_result = std::move(callback); }
        void cancelHandler(std::function<void()> callback) { overlay.active_->on_cancel = std::move(callback); }
        void type(PanelInputState input) {
            input.input_events.push_back({.kind = FrameInputEventKind::Text, .text = "x"});
            send(input);
        }
        bool replace(const std::string& body) {
            return overlay.updateForm("test.form", body, {{"Cancel", "secondary"}});
        }
        static Rml::Element* nativeChild(Rml::Element* element, const char* tag) {
            if (element->GetTagName() == tag)
                return element;
            for (int i = 0; i < element->GetNumChildren(true); ++i)
                if (auto* found = nativeChild(element->GetChild(i), tag))
                    return found;
            return nullptr;
        }
    };

    TEST_F(ModalFormBehaviorTest, DropdownReceivesWheelAndImmediateHoverWithoutTooltip) {
        std::string options = "<select id='choices' title='Choose' style='width:200dp'>";
        for (int i = 0; i < 40; ++i)
            options += "<option>" + std::to_string(i) + "</option>";
        openForm(options + "</select>");
        auto* select = document->GetElementById("choices");
        reveal(over(select));
        auto input = over(select);
        input.mouse_x += select->GetOffsetWidth() - 16;
        click(input);
        auto* popup = nativeChild(select, "selectbox");
        ASSERT_NE(popup, nullptr);
        ASSERT_TRUE(popup->IsVisible(true));
        EXPECT_FALSE(tooltipVisible());
        input = over(popup);
        send(input);
        auto* hovered = context->GetHoverElement();
        while (hovered && hovered->GetTagName() != "option")
            hovered = hovered->GetParentNode();
        ASSERT_NE(hovered, nullptr);
        EXPECT_EQ(hovered->GetProperty<Rml::Colourb>("background-color"), Rml::Colourb(137, 180, 250, 255));
        const float before = popup->GetScrollTop();
        input.mouse_wheel = -1;
        send(input);
        EXPECT_GT(popup->GetScrollTop(), before);
    }

    TEST_F(ModalFormBehaviorTest, TypingSuppressesTooltipUntilPointerLeavesAndReturns) {
        openForm();
        auto* field = document->GetElementById("field");
        auto input = over(field);
        reveal(input);
        field->Focus();
        auto typed = input;
        typed.input_events.push_back({.kind = FrameInputEventKind::Text, .text = "x"});
        send(typed);
        EXPECT_NE(field->GetAttribute<Rml::String>("value", "").find('x'), Rml::String::npos);
        send(input);
        std::this_thread::sleep_for(std::chrono::milliseconds(650));
        applyTooltip();
        EXPECT_FALSE(tooltipVisible());
        EXPECT_FALSE(overlay.secondsUntilNextUpdate().has_value());
        PanelInputState away;
        away.mouse_x = 700;
        away.mouse_y = 500;
        send(away);
        reveal(input);
    }

    TEST_F(ModalFormBehaviorTest, VisibleTooltipIsRemovedOnFormReplacement) {
        openForm();
        reveal(over(document->GetElementById("field")));
        ASSERT_TRUE(overlay.updateForm("test.form", std::string("<input name='replacement' />"), {{"Cancel", "secondary"}}));
        applyTooltip();
        EXPECT_FALSE(tooltipVisible());
    }

    TEST_F(ModalFormBehaviorTest, VisibleTooltipIsRemovedOnDismiss) {
        openForm();
        reveal(over(document->GetElementById("field")));
        ASSERT_TRUE(overlay.dismiss("Cancel"));
        EXPECT_FALSE(tooltipVisible());
    }

    TEST_F(ModalFormBehaviorTest, VisibleTooltipIsRemovedOnCancel) {
        openForm();
        reveal(over(document->GetElementById("field")));
        cancel();
        EXPECT_FALSE(tooltipVisible());
    }

    TEST_F(ModalFormBehaviorTest, EscapeReportsTheRestoredValueToChangeCallback) {
        openForm();
        auto* field = document->GetElementById("field");
        field->Focus();
        field->SetAttribute("value", "edited");
        int callbacks = 0;
        changeHandler([&](const core::ModalResult& result) { ++callbacks; EXPECT_EQ(result.form_values.at("field"), "original"); });
        key(over(field), SDL_SCANCODE_ESCAPE, SDLK_ESCAPE);
        EXPECT_EQ(callbacks, 1);
        EXPECT_EQ(field->GetAttribute<Rml::String>("value", ""), "original");
        EXPECT_TRUE(overlay.isOpen());
    }

    TEST_F(ModalFormBehaviorTest, EscapeCallbackCanReplaceTheFormDuringDispatch) {
        openForm();
        auto* field = document->GetElementById("field");
        field->Focus();
        field->SetAttribute("value", "edited");
        const auto alive = field->GetObserverPtr();
        changeHandler([&](const core::ModalResult&) {
            EXPECT_TRUE(overlay.updateForm("test.form", std::string("<input id='replacement' />"), {{"Cancel", "secondary"}}));
        });
        key(over(field), SDL_SCANCODE_ESCAPE, SDLK_ESCAPE);
        EXPECT_FALSE(alive);
        EXPECT_NE(document->GetElementById("replacement"), nullptr);
        EXPECT_TRUE(overlay.isOpen());
    }

    TEST_F(ModalFormBehaviorTest, EnterChangeDefersReplacementUntilDispatchReturns) {
        openForm();
        auto* field = document->GetElementById("field");
        field->Focus();
        const auto alive = field->GetObserverPtr();
        int changes = 0;
        changeHandler([&](const core::ModalResult&) {
            ++changes;
            EXPECT_TRUE(replace("<input id='replacement' value='new' />"));
            EXPECT_TRUE(alive);
            EXPECT_EQ(document->GetElementById("replacement"), nullptr);
        });
        bool submitted = false;
        resultHandler([&](const core::ModalResult& result) {
            submitted = true;
            EXPECT_EQ(result.form_values.at("replacement"), "new");
        });
        key(over(field), SDL_SCANCODE_RETURN, SDLK_RETURN);
        EXPECT_EQ(changes, 1);
        EXPECT_TRUE(submitted);
        EXPECT_FALSE(alive);
        EXPECT_NE(document->GetElementById("replacement"), nullptr);
    }

    TEST_F(ModalFormBehaviorTest, TypingChangeDefersReplacementUntilDispatchReturns) {
        openForm();
        auto* field = document->GetElementById("field");
        field->Focus();
        const auto alive = field->GetObserverPtr();
        int changes = 0;
        changeHandler([&](const core::ModalResult&) {
            ++changes;
            EXPECT_TRUE(replace("<input id='replacement' />"));
            EXPECT_TRUE(alive);
            EXPECT_EQ(document->GetElementById("replacement"), nullptr);
        });
        type(over(field));
        EXPECT_EQ(changes, 1);
        EXPECT_FALSE(alive);
        EXPECT_NE(document->GetElementById("replacement"), nullptr);
        EXPECT_TRUE(overlay.isOpen());
    }

    TEST_F(ModalFormBehaviorTest, DropdownChangeDefersReplacementUntilDispatchReturns) {
        openForm("<select id='choices' style='width:200dp'><option value='one'>One</option><option value='two'>Two</option></select>");
        auto* select = document->GetElementById("choices");
        const auto alive = select->GetObserverPtr();
        int changes = 0;
        changeHandler([&](const core::ModalResult& result) {
            ++changes;
            EXPECT_EQ(result.form_values.at("choices"), "two");
            EXPECT_TRUE(replace("<input id='replacement' />"));
            EXPECT_TRUE(alive);
            EXPECT_EQ(document->GetElementById("replacement"), nullptr);
        });
        auto input = over(select);
        input.mouse_x += select->GetOffsetWidth() - 16;
        click(input);
        auto* popup = nativeChild(select, "selectbox");
        ASSERT_NE(popup, nullptr);
        ASSERT_TRUE(popup->IsVisible(true));
        Rml::ElementList options;
        popup->QuerySelectorAll(options, "option");
        ASSERT_EQ(options.size(), 2);
        click(over(options[1]));
        EXPECT_EQ(changes, 1);
        EXPECT_FALSE(alive);
        EXPECT_NE(document->GetElementById("replacement"), nullptr);
    }

    TEST_F(ModalFormBehaviorTest, ChangeDefersCloseUntilDispatchReturns) {
        openForm();
        auto* field = document->GetElementById("field");
        field->Focus();
        bool closed = false;
        resultHandler([&](const core::ModalResult&) { closed = true; });
        changeHandler([&](const core::ModalResult&) {
            EXPECT_TRUE(overlay.dismiss("Cancel"));
            EXPECT_TRUE(overlay.isOpen());
            EXPECT_FALSE(closed);
        });
        type(over(field));
        EXPECT_TRUE(closed);
        EXPECT_FALSE(overlay.isOpen());
    }

    TEST_F(ModalFormBehaviorTest, ChangeAppliesUpdatesInRequestOrder) {
        openForm();
        auto* field = document->GetElementById("field");
        field->Focus();
        const auto alive = field->GetObserverPtr();
        changeHandler([&](const core::ModalResult&) {
            EXPECT_TRUE(replace("<input id='first' />"));
            EXPECT_TRUE(replace("<input id='second' />"));
            EXPECT_TRUE(alive);
            EXPECT_EQ(document->GetElementById("second"), nullptr);
        });
        type(over(field));
        EXPECT_FALSE(alive);
        EXPECT_EQ(document->GetElementById("first"), nullptr);
        EXPECT_NE(document->GetElementById("second"), nullptr);
    }

    TEST_F(ModalFormBehaviorTest, ChangeCloseWinsOverUpdates) {
        openForm();
        auto* field = document->GetElementById("field");
        field->Focus();
        bool closed = false;
        resultHandler([&](const core::ModalResult& result) {
            closed = true;
            EXPECT_EQ(result.form_values.at("first"), "new");
        });
        changeHandler([&](const core::ModalResult&) {
            EXPECT_TRUE(replace("<input id='first' value='new' />"));
            EXPECT_TRUE(overlay.dismiss("Cancel"));
            EXPECT_FALSE(replace("<input id='dropped' />"));
            EXPECT_FALSE(closed);
        });
        type(over(field));
        EXPECT_TRUE(closed);
        EXPECT_FALSE(overlay.isOpen());
        EXPECT_NE(document->GetElementById("first"), nullptr);
        EXPECT_EQ(document->GetElementById("dropped"), nullptr);
    }

    TEST_F(ModalFormBehaviorTest, NestedInputDoesNotFlushOuterDispatchUpdates) {
        openForm();
        auto* field = document->GetElementById("field");
        field->Focus();
        const auto alive = field->GetObserverPtr();
        const auto input = over(field);
        changeHandler([&](const core::ModalResult&) {
            EXPECT_TRUE(replace("<input id='first' />"));
            key(input, SDL_SCANCODE_RIGHT, SDLK_RIGHT);
            EXPECT_TRUE(alive);
            EXPECT_EQ(document->GetElementById("first"), nullptr);
            EXPECT_TRUE(replace("<input id='second' />"));
        });
        type(input);
        EXPECT_FALSE(alive);
        EXPECT_NE(document->GetElementById("second"), nullptr);
    }

    TEST_F(ModalFormBehaviorTest, ChangeDefersCancelAndDropsLaterUpdate) {
        openForm();
        auto* field = document->GetElementById("field");
        field->Focus();
        bool cancelled = false;
        cancelHandler([&] { cancelled = true; });
        changeHandler([&](const core::ModalResult&) {
            cancel();
            EXPECT_TRUE(overlay.isOpen());
            EXPECT_FALSE(cancelled);
            EXPECT_FALSE(replace("<input id='dropped' />"));
        });
        type(over(field));
        EXPECT_TRUE(cancelled);
        EXPECT_FALSE(overlay.isOpen());
        EXPECT_EQ(document->GetElementById("dropped"), nullptr);
    }

    TEST_F(ModalFormBehaviorTest, EnterUsesButtonsFromQueuedUpdate) {
        openForm();
        auto* field = document->GetElementById("field");
        field->Focus();
        bool submitted = false;
        resultHandler([&](const core::ModalResult& result) {
            submitted = true;
            EXPECT_EQ(result.button_label, "Apply");
        });
        changeHandler([&](const core::ModalResult&) {
            EXPECT_TRUE(overlay.updateForm("test.form", std::nullopt, {{"Apply", "primary"}}));
        });
        key(over(field), SDL_SCANCODE_RETURN, SDLK_RETURN);
        EXPECT_TRUE(submitted);
        EXPECT_FALSE(overlay.isOpen());
    }

    TEST_F(ModalFormBehaviorTest, ProgrammaticChangeWaitsUntilNextFrame) {
        openForm();
        auto* field = document->GetElementById("field");
        const auto alive = field->GetObserverPtr();
        changeHandler([&](const core::ModalResult&) {
            EXPECT_TRUE(replace("<input id='replacement' />"));
            EXPECT_TRUE(alive);
        });
        field->DispatchEvent(Rml::EventId::Change, {});
        EXPECT_TRUE(alive);
        EXPECT_TRUE(overlay.needsAnimationFrame());
        overlay.render(0, 0);
        EXPECT_FALSE(alive);
        EXPECT_NE(document->GetElementById("replacement"), nullptr);
    }

    TEST_F(ModalFormBehaviorTest, UpdateOutsideDispatchAppliesImmediately) {
        openForm();
        const auto alive = document->GetElementById("field")->GetObserverPtr();
        EXPECT_TRUE(replace("<input id='replacement' />"));
        EXPECT_FALSE(alive);
        EXPECT_NE(document->GetElementById("replacement"), nullptr);
    }

    TEST_F(ModalFormBehaviorTest, EscapeOnUneditedFieldDoesNotNotifyChange) {
        openForm();
        auto* field = document->GetElementById("field");
        field->Focus();
        int callbacks = 0;
        changeHandler([&](const core::ModalResult&) { ++callbacks; });
        key(over(field), SDL_SCANCODE_ESCAPE, SDLK_ESCAPE);
        EXPECT_EQ(callbacks, 0);
        EXPECT_NE(context->GetFocusElement(), field);
        EXPECT_TRUE(overlay.isOpen());
    }
} // namespace lfs::vis::gui
