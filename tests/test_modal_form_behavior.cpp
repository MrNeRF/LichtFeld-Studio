/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panel_layout.hpp"
#include "gui/rml_modal_overlay.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include <RmlUi/Core.h>
#include <RmlUi/Core/Elements/ElementFormControl.h>
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
        RmlModalOverlay overlay{&manager};
        Rml::Context* context = nullptr;
        Rml::ElementDocument* document = nullptr;

        void SetUp() override {
            ASSERT_TRUE(Rml::Initialise());
            ASSERT_TRUE(Rml::LoadFontFace((std::filesystem::path(__FILE__).parent_path().parent_path() /
                                           "src/visualizer/gui/assets/fonts/Inter-Regular.ttf")
                                              .string()));
        }
        void TearDown() override {
            if (overlay.el_form_)
                unbindForm();
            manager.shutdown();
            if (context) {
                overlay.rml_context_ = nullptr;
                EXPECT_TRUE(Rml::RemoveContext("modal_form_behavior"));
            }
            Rml::Shutdown();
        }
        void installFormDocument(const std::string& native_controls = "", const float scale = 1.0f) {
            context = Rml::CreateContext("modal_form_behavior", {800, 600}, &renderer);
            ASSERT_NE(context, nullptr);
            context->SetDensityIndependentPixelRatio(scale);
            context->SetDefaultScrollBehavior(Rml::ScrollBehavior::Instant, 1.0f);
            const auto resources = std::filesystem::path(__FILE__).parent_path().parent_path() / "src/visualizer/gui/rmlui/resources";
            const auto read = [&](const char* name) {
                std::ifstream file(resources / name);
                return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
            };
            document = context->LoadDocumentFromMemory("<rml><head><style>" + read("components.rcss") + read("modal_overlay.rcss") +
                                                       R"(</style></head><body id="body"><div id="backdrop"></div><div id="dialog" class="form-modal"><div id="content"><input type="text" name="resolution" id="resolution" value="1920x1080" /></div><div id="buttons"></div></div></body></rml>)");
            ASSERT_NE(document, nullptr);
            if (!native_controls.empty())
                document->GetElementById("content")->SetInnerRML(native_controls);
            document->Show();
            context->Update();
            overlay.rml_context_ = context;
            overlay.document_ = document;
            overlay.el_button_row_ = document->GetElementById("buttons");
            overlay.el_backdrop_ = document->GetElementById("backdrop");
            overlay.el_dialog_ = document->GetElementById("dialog");
            bindForm(document->GetElementById("content"));
        }

        void setForm(const std::string& submit_button = "Apply") {
            overlay.active_.emplace();
            overlay.active_->key = "test.form";
            overlay.active_->buttons = {{"Apply", "secondary", true}, {"Cancel", "secondary", false}};
            overlay.active_->submit_button_label = submit_button;
        }
        bool submit() { return overlay.dismissFirstEnabledButton(); }
        void bindForm(Rml::Element* content) {
            overlay.el_content_ = content;
            overlay.el_form_ = content;
            content->AddEventListener(Rml::EventId::Change, &overlay.listener_);
            content->AddEventListener(Rml::EventId::Click, &overlay.listener_);
            overlay.bindTextInputRevert();
        }
        void onChange(std::function<void(const core::ModalResult&)> callback) {
            overlay.active_->on_change = std::move(callback);
        }
        void onResult(std::function<void(const core::ModalResult&)> callback) {
            overlay.active_->on_result = std::move(callback);
        }
        void cancel() { overlay.cancel(); }
        void unbindForm() {
            overlay.text_input_revert_.clear();
            overlay.el_form_->RemoveEventListener(Rml::EventId::Change, &overlay.listener_);
            overlay.el_form_->RemoveEventListener(Rml::EventId::Click, &overlay.listener_);
            overlay.el_form_ = nullptr;
            overlay.el_content_ = nullptr;
        }
        static Rml::Element* nativeChild(Rml::Element* element, const char* tag) {
            if (element->GetTagName() == tag)
                return element;
            for (int i = 0; i < element->GetNumChildren(true); ++i)
                if (auto* found = nativeChild(element->GetChild(i), tag))
                    return found;
            return nullptr;
        }
        void deliver(RmlModalOverlay& modal, const PanelInputState& input) {
            auto* manager = modal.rml_manager_;
            const bool cached = modal.elements_cached_;
            modal.rml_manager_ = nullptr;
            modal.elements_cached_ = true;
            modal.processInput(input);
            modal.rml_manager_ = manager;
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

        // Shared modal lifecycle helpers, also used by the full Sequencer suite.
        void showVisibleTooltip() {
            auto* target = document->GetElementById("resolution");
            if (!target)
                target = document->GetElementById("choices");
            ASSERT_NE(target, nullptr);
            overlay.tooltip_.setHover("Help for the current form", target);
            std::this_thread::sleep_for(std::chrono::milliseconds(520));
            ASSERT_TRUE(overlay.tooltip_.apply(document->GetElementById("body"), 20, 20, 800, 600));
            ASSERT_TRUE(document->GetElementById("frame-tooltip")->IsClassSet("visible"));
        }
        void applyTooltip() { overlay.tooltip_.apply(document->GetElementById("body"), 20, 20, 800, 600); }
        void pressNativeKey(SDL_Scancode scancode, SDL_Keycode keycode) {
            PanelInputState input;
            input.input_events.push_back({.kind = FrameInputEventKind::KeyDown, .scancode = scancode, .keycode = keycode});
            const bool cached = overlay.elements_cached_;
            overlay.elements_cached_ = true;
            // Exercise the real dispatch scope: dismiss/cancel need the manager
            // to remain available while the keyboard callback is running.
            manager.routeInput(context, input, [&](const PanelInputState&) { overlay.processInput(input); }, true);
            overlay.elements_cached_ = cached;
            context->Update();
            context->Render();
        }
        // End shared modal lifecycle helpers.

        void checkNativeDropdown(const float scale) {
            setForm();
            std::string controls = R"(<select id="choices" name="choices" title="Choose a value" style="width:200dp"><option value="0">0</option>)";
            for (int i = 1; i < 30; ++i)
                controls += "<option value=\"" + std::to_string(i) + "\">" + std::to_string(i) + "</option>";
            installFormDocument(controls + "</select>", scale);
            auto* select = document->GetElementById("choices");
            ASSERT_NE(select, nullptr);
            const auto pos = select->GetAbsoluteOffset(Rml::BoxArea::Border);
            PanelInputState input;
            input.mouse_x = pos.x + 10 * scale;
            input.mouse_y = pos.y + select->GetOffsetHeight() * 0.5f;
            deliver(overlay, input);
            EXPECT_TRUE(overlay.secondsUntilNextUpdate().has_value());
            clickAt(overlay, pos.x + select->GetOffsetWidth() - 9 * scale, input.mouse_y);
            auto* popup = nativeChild(select, "selectbox");
            ASSERT_NE(popup, nullptr);
            ASSERT_TRUE(popup->IsVisible(true));
            ASSERT_GT(popup->GetScrollHeight(), popup->GetClientHeight());
            const auto popup_pos = popup->GetAbsoluteOffset(Rml::BoxArea::Border);
            input.mouse_x = popup_pos.x + 10 * scale;
            input.mouse_y = popup_pos.y + popup->GetClientHeight() * 0.5f;
            deliver(overlay, input);
            EXPECT_FALSE(overlay.secondsUntilNextUpdate().has_value());
            auto* hovered = context->GetHoverElement();
            while (hovered && hovered->GetTagName() != "option")
                hovered = hovered->GetParentNode();
            ASSERT_NE(hovered, nullptr);
            EXPECT_EQ(hovered->GetProperty<Rml::Colourb>("background-color"), Rml::Colourb(137, 180, 250, 255));
            const float before = popup->GetScrollTop();
            input.mouse_wheel = -1;
            deliver(overlay, input);
            EXPECT_GT(popup->GetScrollTop(), before);
            auto* scrollbar = nativeChild(popup, "scrollbarvertical");
            ASSERT_NE(scrollbar, nullptr);
            auto* bar = nativeChild(scrollbar, "sliderbar");
            ASSERT_NE(bar, nullptr);
            const auto bar_pos = bar->GetAbsoluteOffset(Rml::BoxArea::Border);
            input = {};
            input.mouse_x = bar_pos.x + bar->GetOffsetWidth() * 0.5f;
            input.mouse_y = bar_pos.y + bar->GetOffsetHeight() * 0.5f;
            deliver(overlay, input);
            const float before_drag = popup->GetScrollTop();
            input.mouse_clicked[0] = true;
            input.mouse_down[0] = true;
            deliver(overlay, input);
            input.mouse_clicked[0] = false;
            input.mouse_y += 24 * scale;
            deliver(overlay, input);
            input.mouse_down[0] = false;
            input.mouse_released[0] = true;
            deliver(overlay, input);
            EXPECT_GT(popup->GetScrollTop(), before_drag);
        }
    };

    TEST_F(ModalFormBehaviorTest, DisabledDefaultDoesNotSubmitAnotherEnabledAction) {
        setForm();
        EXPECT_FALSE(submit());
        EXPECT_TRUE(overlay.current().has_value());
        EXPECT_FALSE(overlay.dismiss("Apply"));
        EXPECT_TRUE(overlay.current().has_value());
    }

    TEST_F(ModalFormBehaviorTest, LegacyModalStillSubmitsItsFirstEnabledButton) {
        setForm("");
        installFormDocument();
        std::string selected;
        onResult([&](const core::ModalResult& result) { selected = result.button_label; });
        EXPECT_TRUE(submit());
        EXPECT_EQ(selected, "Cancel");
    }

    TEST_F(ModalFormBehaviorTest, NativeChangeIdentifiesTheChangedField) {
        setForm();
        installFormDocument();
        int changes = 0;
        onChange([&](const core::ModalResult& result) {
            ++changes;
            EXPECT_EQ(result.changed_field, "resolution");
            EXPECT_TRUE(result.form_action.empty());
            EXPECT_EQ(result.form_values.at("resolution"), "1280x720");
        });
        auto* input = document->GetElementById("resolution");
        input->SetAttribute("value", "1280x720");
        input->DispatchEvent("change", {});
        EXPECT_EQ(changes, 1);
    }

    TEST_F(ModalFormBehaviorTest, DependentValuesEscapeTextAndPreserveCheckboxAndSectionIdentity) {
        setForm();
        installFormDocument(R"(<input id="flag" type="checkbox" name="flag" /><span id="text" data-form-value="text"></span><div id="section" data-form-visible="section"><input type="text" name="nested" /></div>)");
        auto* checkbox = document->GetElementById("flag");
        auto* section = document->GetElementById("section");
        ASSERT_TRUE(overlay.setFormValues("test.form", {{"flag", "1"}, {"text", "<input name='injected' />"}, {"section", ""}}));
        context->Update();
        EXPECT_TRUE(checkbox->HasAttribute("checked"));
        EXPECT_EQ(document->QuerySelector("input[name=injected]"), nullptr);
        EXPECT_EQ(section->GetProperty<Rml::Style::Display>("display"), Rml::Style::Display::None);
        ASSERT_TRUE(overlay.setFormValues("test.form", {{"flag", ""}, {"section", "1"}}));
        context->Update();
        EXPECT_FALSE(checkbox->HasAttribute("checked"));
        EXPECT_EQ(section->GetProperty<Rml::Style::Display>("display"), Rml::Style::Display::Block);
        EXPECT_EQ(document->GetElementById("flag"), checkbox);
        EXPECT_EQ(document->GetElementById("section"), section);
    }

    TEST_F(ModalFormBehaviorTest, EscapeRestoresBothTheControlAndTheFormDraft) {
        setForm();
        installFormDocument();
        std::string draft;
        onChange([&draft](const core::ModalResult& result) { draft = result.form_values.at("resolution"); });
        auto* input = document->GetElementById("resolution");
        input->Blur();
        input->Focus();
        input->SetAttribute("value", "invalid");
        input->DispatchEvent("change", {});
        EXPECT_EQ(draft, "invalid");
        input->DispatchEvent("escapecancel", {});
        EXPECT_EQ(input->GetAttribute<Rml::String>("value", ""), "1920x1080");
        EXPECT_EQ(draft, "1920x1080");
    }

    TEST_F(ModalFormBehaviorTest, DismissReturnsCurrentValues) {
        setForm();
        installFormDocument();
        bool returned = false;
        onResult([&returned](const core::ModalResult& result) {
            returned = true;
            EXPECT_EQ(result.button_label, "Cancel");
            EXPECT_EQ(result.form_values.at("resolution"), "1920x1080");
        });
        EXPECT_TRUE(overlay.dismiss("Cancel"));
        EXPECT_TRUE(returned);
        EXPECT_FALSE(overlay.current().has_value());
    }

    TEST_F(ModalFormBehaviorTest, EscapeCancellationDoesNotSubmit) {
        setForm();
        installFormDocument();
        bool submitted = false;
        onResult([&submitted](const core::ModalResult&) { submitted = true; });
        cancel();
        EXPECT_FALSE(submitted);
        EXPECT_FALSE(overlay.current().has_value());
    }
    TEST_F(ModalFormBehaviorTest, ProgrammaticDependentValuesPreserveControlAndSuppressRecursiveChange) {
        setForm();
        installFormDocument();
        int changes = 0;
        onChange([&](const core::ModalResult&) { ++changes; });
        auto* original = document->GetElementById("resolution");
        EXPECT_FALSE(overlay.setFormValues("wrong.form", {{"resolution", "1280x720"}}));
        EXPECT_TRUE(overlay.setFormValues("test.form", {{"resolution", "1280x720"}}));
        EXPECT_EQ(document->GetElementById("resolution"), original);
        EXPECT_EQ(original->GetAttribute<Rml::String>("value", ""), "1280x720");
        EXPECT_EQ(changes, 0);
    }

    TEST_F(ModalFormBehaviorTest, FormActionDoesNotDismissAndReportsOnlyTheNamedAction) {
        setForm();
        installFormDocument();
        auto button = document->CreateElement("button");
        button->SetAttribute("type", "button");
        button->SetAttribute("data-form-action", "swap");
        auto* target = document->GetElementById("content")->AppendChild(std::move(button));
        int changes = 0;
        onChange([&](const core::ModalResult& result) {
            ++changes;
            EXPECT_EQ(result.form_action, "swap");
            EXPECT_TRUE(result.changed_field.empty());
        });
        target->DispatchEvent("click", {});
        EXPECT_EQ(changes, 1);
        EXPECT_TRUE(overlay.isOpen());
    }

    TEST_F(ModalFormBehaviorTest, NativeDropdownReceivesWheelAndScrollbarAndSuppressesTooltip) { checkNativeDropdown(1.0f); }
    TEST_F(ModalFormBehaviorTest, NativeDropdownInputAtHighDpi) { checkNativeDropdown(1.5f); }
    TEST_F(ModalFormBehaviorTest, NativeDropdownInputAtDoubleScale) { checkNativeDropdown(2.0f); }

    TEST_F(ModalFormBehaviorTest, VisibleTooltipIsHiddenWhenBodyChanges) {
        setForm();
        installFormDocument();
        showVisibleTooltip();
        ASSERT_TRUE(overlay.updateForm("test.form", std::string("<input name='replacement' type='text' />"), {{"Cancel", "secondary"}}));
        applyTooltip();
        EXPECT_FALSE(document->GetElementById("frame-tooltip")->IsClassSet("visible"));
        EXPECT_FALSE(overlay.secondsUntilNextUpdate().has_value());
    }

    TEST_F(ModalFormBehaviorTest, VisibleTooltipIsHiddenOnCancel) {
        setForm();
        installFormDocument();
        showVisibleTooltip();
        cancel();
        EXPECT_FALSE(document->GetElementById("frame-tooltip")->IsClassSet("visible"));
    }

    TEST_F(ModalFormBehaviorTest, VisibleTooltipIsHiddenOnDismiss) {
        setForm();
        installFormDocument();
        showVisibleTooltip();
        ASSERT_TRUE(overlay.dismiss("Cancel"));
        EXPECT_FALSE(document->GetElementById("frame-tooltip")->IsClassSet("visible"));
    }

    TEST_F(ModalFormBehaviorTest, VisibleTooltipHidesWhenPointerLeavesWithoutClicking) {
        setForm();
        installFormDocument();
        showVisibleTooltip();
        PanelInputState input;
        input.mouse_x = 790;
        input.mouse_y = 590;
        deliver(overlay, input);
        applyTooltip();
        EXPECT_FALSE(document->GetElementById("frame-tooltip")->IsClassSet("visible"));
    }

    TEST_F(ModalFormBehaviorTest, VisibleTooltipHidesWhileTyping) {
        setForm();
        installFormDocument();
        auto* input = document->GetElementById("resolution");
        input->Focus();
        showVisibleTooltip();
        pressNativeKey(SDL_SCANCODE_LEFT, SDLK_LEFT);
        EXPECT_FALSE(document->GetElementById("frame-tooltip")->IsClassSet("visible"));
    }

    TEST_F(ModalFormBehaviorTest, OpeningNativePopupHidesVisibleTooltipBeforeSelection) {
        setForm();
        installFormDocument("<select id='choices' name='choices' title='Choose'><option value='a'>A</option><option value='b'>B</option><option value='c'>C</option></select>");
        showVisibleTooltip();
        auto* select = dynamic_cast<Rml::ElementFormControlSelect*>(document->GetElementById("choices"));
        const auto pos = select->GetAbsoluteOffset(Rml::BoxArea::Border);
        clickAt(overlay, pos.x + select->GetOffsetWidth() - 9, pos.y + select->GetOffsetHeight() * 0.5f);
        ASSERT_TRUE(select->IsSelectBoxVisible());
        EXPECT_FALSE(document->GetElementById("frame-tooltip")->IsClassSet("visible"));
        auto* option = select->GetOption(2);
        const auto option_pos = option->GetAbsoluteOffset(Rml::BoxArea::Border);
        clickAt(overlay, option_pos.x + 10, option_pos.y + option->GetOffsetHeight() * 0.5f);
        EXPECT_EQ(select->GetValue(), "c");
    }

    TEST_F(ModalFormBehaviorTest, RadioGroupUpdatesCheckedStateAndPreservesOptionValues) {
        setForm();
        installFormDocument("<input type='radio' name='mode' value='a' id='mode-a' checked='checked' /><input type='radio' name='mode' value='b' id='mode-b' />");
        int changes = 0;
        onChange([&](const core::ModalResult&) { ++changes; });
        ASSERT_TRUE(overlay.setFormValues("test.form", {{"mode", "b"}}));
        context->Update();
        EXPECT_FALSE(document->GetElementById("mode-a")->HasAttribute("checked"));
        EXPECT_TRUE(document->GetElementById("mode-b")->HasAttribute("checked"));
        EXPECT_EQ(document->GetElementById("mode-a")->GetAttribute<Rml::String>("value", ""), "a");
        EXPECT_EQ(document->GetElementById("mode-b")->GetAttribute<Rml::String>("value", ""), "b");
        EXPECT_EQ(changes, 0);
        EXPECT_FALSE(overlay.setFormValues("", {{"mode", "a"}}));
    }

    TEST_F(ModalFormBehaviorTest, NativeEnterHonorsDisabledDefaultInsteadOfCanceling) {
        setForm();
        installFormDocument();
        int results = 0;
        onResult([&](const core::ModalResult& result) { ++results; EXPECT_EQ(result.button_label, "Apply"); });
        pressNativeKey(SDL_SCANCODE_RETURN, SDLK_RETURN);
        EXPECT_TRUE(overlay.isOpen());
        EXPECT_EQ(results, 0);
        ASSERT_TRUE(overlay.updateForm("test.form", std::nullopt, {{"Apply", "primary"}, {"Cancel", "secondary"}}));
        pressNativeKey(SDL_SCANCODE_RETURN, SDLK_RETURN);
        EXPECT_FALSE(overlay.isOpen());
        EXPECT_EQ(results, 1);
    }

    TEST_F(ModalFormBehaviorTest, NativeEscapeRestoresDraftThenCancelsOnSecondPress) {
        setForm();
        installFormDocument();
        auto* input = document->GetElementById("resolution");
        input->Focus();
        input->SetAttribute("value", "invalid");
        std::string restored;
        onChange([&](const core::ModalResult& result) { restored = result.form_values.at("resolution"); });
        pressNativeKey(SDL_SCANCODE_ESCAPE, SDLK_ESCAPE);
        EXPECT_EQ(restored, "1920x1080");
        EXPECT_TRUE(overlay.isOpen());
        pressNativeKey(SDL_SCANCODE_ESCAPE, SDLK_ESCAPE);
        EXPECT_FALSE(overlay.isOpen());
    }

    TEST_F(ModalFormBehaviorTest, EscapeCallbackCanDismissItsOwnForm) {
        setForm();
        installFormDocument();
        auto* input = document->GetElementById("resolution");
        input->Focus();
        input->SetAttribute("value", "invalid");
        onChange([&](const core::ModalResult&) { EXPECT_TRUE(overlay.dismiss("Cancel")); });
        pressNativeKey(SDL_SCANCODE_ESCAPE, SDLK_ESCAPE);
        EXPECT_FALSE(overlay.isOpen());
    }

    TEST_F(ModalFormBehaviorTest, RadioGroupsWithoutIdsKeepTheirOriginalOptionValues) {
        setForm();
        installFormDocument("<input type='radio' name='mode' value='a' checked='checked' /><input type='radio' name='mode' value='b' />");
        ASSERT_TRUE(overlay.setFormValues("test.form", {{"mode", "b"}}));
        Rml::ElementList radios;
        document->QuerySelectorAll(radios, "input[type=radio]");
        ASSERT_EQ(radios.size(), 2);
        EXPECT_EQ(radios[0]->GetAttribute<Rml::String>("value", ""), "a");
        EXPECT_EQ(radios[1]->GetAttribute<Rml::String>("value", ""), "b");
        onResult([](const core::ModalResult& result) { EXPECT_EQ(result.form_values.at("mode"), "b"); });
        EXPECT_TRUE(overlay.dismiss("Cancel"));
    }

    TEST_F(ModalFormBehaviorTest, EscapeCallbackCanReplaceTheFormBody) {
        setForm();
        installFormDocument();
        document->GetElementById("resolution")->Focus();
        onChange([&](const core::ModalResult&) {
            EXPECT_TRUE(overlay.updateForm("test.form", std::string("<input id='replacement' type='text' />"), {{"Cancel", "secondary"}}));
        });
        pressNativeKey(SDL_SCANCODE_ESCAPE, SDLK_ESCAPE);
        EXPECT_TRUE(overlay.isOpen());
        EXPECT_NE(document->GetElementById("replacement"), nullptr);
    }
} // namespace lfs::vis::gui
