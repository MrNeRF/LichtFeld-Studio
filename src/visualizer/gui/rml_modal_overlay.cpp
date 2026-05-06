/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rml_modal_overlay.hpp"
#include "core/logger.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/panel_layout.hpp"
#include "gui/rmlui/rml_document_utils.hpp"
#include "gui/rmlui/rml_input_utils.hpp"
#include "gui/rmlui/rml_text_input_handler.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "internal/resource_paths.hpp"
#include "theme/theme.hpp"

#include "gui/rmlui/sdl_rml_key_mapping.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/Input.h>
#include <cassert>
#include <format>

namespace lfs::vis::gui {

    RmlModalOverlay::RmlModalOverlay(RmlUIManager* rml_manager)
        : rml_manager_(rml_manager) {
        assert(rml_manager_);
        listener_.overlay = this;
    }

    RmlModalOverlay::~RmlModalOverlay() {
        if (Rml::GetSystemInterface())
            text_input_revert_.clear();
        if (rml_context_ && rml_manager_ && rml_manager_->isInitialized())
            rml_manager_->destroyContext("modal_overlay");
    }

    void RmlModalOverlay::enqueue(lfs::core::ModalRequest request) {
        std::lock_guard lock(queue_mutex_);
        queue_.push_back(std::move(request));
    }

    bool RmlModalOverlay::isOpen() const {
        return active_.has_value();
    }

    void RmlModalOverlay::initContext() {
        if (rml_context_)
            return;

        rml_context_ = rml_manager_->createContext("modal_overlay", 800, 600);
        if (!rml_context_) {
            LOG_ERROR("RmlModalOverlay: failed to create context");
            return;
        }

        try {
            const auto rml_path = lfs::vis::getAssetPath("rmlui/modal_overlay.rml");
            document_ = rml_documents::loadDocument(rml_context_, rml_path);
            if (!document_) {
                LOG_ERROR("RmlModalOverlay: failed to load modal_overlay.rml");
                return;
            }
            document_->Show();
            cacheElements();
        } catch (const std::exception& e) {
            LOG_ERROR("RmlModalOverlay: resource not found: {}", e.what());
        }
    }

    void RmlModalOverlay::reloadResources() {
        if (!rml_context_ || active_.has_value())
            return;

        text_input_revert_.clear();
        if (document_) {
            rml_context_->UnloadDocument(document_);
            rml_context_->Update();
        }

        document_ = nullptr;
        el_backdrop_ = nullptr;
        el_dialog_ = nullptr;
        el_title_ = nullptr;
        el_form_ = nullptr;
        el_content_ = nullptr;
        el_input_row_ = nullptr;
        el_input_ = nullptr;
        el_button_row_ = nullptr;
        elements_cached_ = false;
        base_rcss_.clear();
        has_theme_signature_ = false;
        width_ = 0;
        height_ = 0;

        try {
            const auto rml_path = lfs::vis::getAssetPath("rmlui/modal_overlay.rml");
            document_ = rml_documents::loadDocument(rml_context_, rml_path);
            if (!document_) {
                LOG_ERROR("RmlModalOverlay: failed to reload modal_overlay.rml");
                return;
            }
            document_->Show();
            cacheElements();
        } catch (const std::exception& e) {
            LOG_ERROR("RmlModalOverlay: resource not found during reload: {}", e.what());
            return;
        }

        syncTheme();
    }

    void RmlModalOverlay::cacheElements() {
        assert(document_);
        el_backdrop_ = document_->GetElementById("modal-backdrop");
        el_dialog_ = document_->GetElementById("modal-dialog");
        el_title_ = document_->GetElementById("modal-title");
        el_form_ = document_->GetElementById("modal-form");
        el_content_ = document_->GetElementById("modal-content");
        el_input_row_ = document_->GetElementById("modal-input-row");
        el_input_ = document_->GetElementById("modal-input");
        el_button_row_ = document_->GetElementById("modal-button-row");

        elements_cached_ = el_backdrop_ && el_dialog_ && el_title_ && el_form_ && el_content_ &&
                           el_input_row_ && el_input_ && el_button_row_;

        if (!elements_cached_) {
            LOG_ERROR("RmlModalOverlay: missing DOM elements");
            return;
        }

        el_backdrop_->AddEventListener(Rml::EventId::Click, &listener_);
        el_form_->AddEventListener(Rml::EventId::Submit, &listener_);
        el_form_->AddEventListener(Rml::EventId::Change, &listener_);
        el_button_row_->AddEventListener(Rml::EventId::Click, &listener_);
    }

    void RmlModalOverlay::syncTheme() {
        if (!document_)
            return;

        const std::size_t theme_signature = rml_theme::currentThemeSignature();
        if (has_theme_signature_ && theme_signature == last_theme_signature_)
            return;
        last_theme_signature_ = theme_signature;
        has_theme_signature_ = true;

        if (base_rcss_.empty())
            base_rcss_ = rml_theme::loadBaseRCSS("rmlui/modal_overlay.rcss");

        rml_theme::applyTheme(document_, base_rcss_, rml_theme::loadBaseRCSS("rmlui/modal_overlay.theme.rcss"));
    }

    void RmlModalOverlay::showNext() {
        assert(elements_cached_);
        assert(!active_.has_value());

        lfs::core::ModalRequest req;
        {
            std::lock_guard lock(queue_mutex_);
            if (queue_.empty())
                return;
            req = std::move(queue_.front());
            queue_.pop_front();
        }

        text_input_revert_.clear();

        el_title_->SetInnerRML(req.title);
        el_content_->SetInnerRML(req.body_rml);

        if (req.has_input) {
            el_input_row_->SetClass("visible", true);
            el_input_->SetAttribute("value", req.input_default);
        } else {
            el_input_row_->SetClass("visible", false);
        }

        std::string btn_html;
        btn_html.reserve(512);
        for (size_t i = 0; i < req.buttons.size(); ++i) {
            const auto& btn = req.buttons[i];
            const std::string cls = "btn btn--" + btn.style;
            btn_html += std::format(
                R"(<button type="button" class="{}" id="modal-btn-{}"{}>{}</button>)",
                cls, i, btn.disabled ? " disabled=\"disabled\"" : "", btn.label);
        }
        el_button_row_->SetInnerRML(btn_html);

        el_dialog_->SetClass("style-info", req.style == lfs::core::ModalStyle::Info);
        el_dialog_->SetClass("style-warning", req.style == lfs::core::ModalStyle::Warning);
        el_dialog_->SetClass("style-error", req.style == lfs::core::ModalStyle::Error);

        el_dialog_->SetProperty("width", std::format("{}dp", req.width_dp));

        el_backdrop_->SetProperty("display", "block");
        el_dialog_->SetProperty("display", "block");

        active_ = std::move(req);
        bindTextInputRevert();
        if (active_->has_input)
            el_input_->Focus();
    }

    lfs::core::ModalResult RmlModalOverlay::collectFormValues() const {
        lfs::core::ModalResult result;

        if (active_ && active_->has_input) {
            result.input_value = el_input_->GetAttribute<Rml::String>("value", "");
        }

        // Collect all named text-editable controls from the content area.
        rml_input::forEachTextEditableElement(el_content_, [&result](Rml::Element& element) {
            const auto id = element.GetId();
            if (!id.empty()) {
                result.form_values[id] = element.GetAttribute<Rml::String>("value", "");
            }
        });

        return result;
    }

    void RmlModalOverlay::dismiss(const std::string& button_label) {
        if (!active_)
            return;

        text_input_revert_.clear();
        el_backdrop_->SetProperty("display", "none");
        el_dialog_->SetProperty("display", "none");

        auto result = collectFormValues();
        result.button_label = button_label;

        auto on_result = std::move(active_->on_result);
        active_.reset();

        if (on_result)
            on_result(result);
    }

    bool RmlModalOverlay::dismissFirstEnabledButton() {
        if (!active_)
            return false;

        for (const auto& button : active_->buttons) {
            if (!button.disabled) {
                dismiss(button.label);
                return true;
            }
        }
        return false;
    }

    void RmlModalOverlay::cancel() {
        if (!active_)
            return;

        text_input_revert_.clear();
        el_backdrop_->SetProperty("display", "none");
        el_dialog_->SetProperty("display", "none");

        auto on_cancel = std::move(active_->on_cancel);
        active_.reset();

        if (on_cancel)
            on_cancel();
    }

    void RmlModalOverlay::bindTextInputRevert() {
        text_input_revert_.bind(el_input_);
        rml_input::forEachTextEditableElement(el_content_, [this](Rml::Element& element) {
            text_input_revert_.bind(&element);
        });
    }

    void RmlModalOverlay::processInput(const PanelInputState& input) {
        if (!active_ || !rml_context_ || !elements_cached_)
            return;
        if (rml_manager_)
            rml_manager_->trackContextFrame(rml_context_, 0, 0);

        auto& focus = guiFocusState();
        focus.want_capture_mouse = true;
        focus.want_capture_keyboard = true;
        bool has_text_focus = rml_input::isTextEditableElement(rml_context_->GetFocusElement());
        if (has_text_focus)
            focus.want_text_input = true;

        const int mods = sdlModsToRml(input.key_ctrl, input.key_shift,
                                      input.key_alt, input.key_super);

        const float mx = input.mouse_x - input.screen_x;
        const float my = input.mouse_y - input.screen_y;
        rml_context_->ProcessMouseMove(static_cast<int>(mx), static_cast<int>(my), mods);

        if (input.mouse_clicked[0])
            rml_context_->ProcessMouseButtonDown(0, mods);
        if (input.mouse_released[0])
            rml_context_->ProcessMouseButtonUp(0, mods);

        auto* const text_input_handler =
            rml_manager_ ? rml_manager_->getTextInputHandler() : nullptr;
        const bool composing = text_input_handler && text_input_handler->isComposing();

        bool escape_requested = false;
        for (const int sc : input.keys_pressed) {
            if (!composing && sc == SDL_SCANCODE_ESCAPE) {
                if (auto* const focused = rml_context_->GetFocusElement();
                    focused && (rml_input::isTextEditableElement(focused) ||
                                rml_input::isSelectRelatedElement(focused))) {
                    escape_requested = true;
                    continue;
                }
            }
            if (composing &&
                (sc == SDL_SCANCODE_RETURN || sc == SDL_SCANCODE_KP_ENTER ||
                 sc == SDL_SCANCODE_ESCAPE)) {
                continue;
            }
            const auto rml_key = sdlScancodeToRml(static_cast<SDL_Scancode>(sc));
            if (rml_key != Rml::Input::KI_UNKNOWN) {
                if (text_input_handler && text_input_handler->handleKeyDown(rml_key, mods))
                    continue;
                rml_context_->ProcessKeyDown(rml_key, mods);
            }
        }
        for (const int sc : input.keys_released) {
            if (escape_requested && sc == SDL_SCANCODE_ESCAPE)
                continue;
            if (composing && (sc == SDL_SCANCODE_RETURN || sc == SDL_SCANCODE_KP_ENTER))
                continue;
            const auto rml_key = sdlScancodeToRml(static_cast<SDL_Scancode>(sc));
            if (rml_key != Rml::Input::KI_UNKNOWN)
                rml_context_->ProcessKeyUp(rml_key, mods);
        }

        if (!composing && escape_requested) {
            if (rml_input::cancelFocusedElement(*rml_context_)) {
                has_text_focus = rml_input::isTextEditableElement(rml_context_->GetFocusElement());
                return;
            }
        }

        if (has_text_focus && text_input_handler && input.has_text_editing) {
            text_input_handler->handleTextEditing(
                input.text_editing, input.text_editing_start, input.text_editing_length);
        }

        bool forward_text_codepoints = input.text_inputs.empty();
        if (has_text_focus) {
            for (const auto& text_input : input.text_inputs) {
                if (!text_input_handler || !text_input_handler->handleTextInput(text_input))
                    forward_text_codepoints = true;
            }
        }

        if (has_text_focus && forward_text_codepoints) {
            for (uint32_t cp : input.text_codepoints)
                rml_context_->ProcessTextInput(static_cast<Rml::Character>(cp));
        }

        // Re-check active_ since RmlUI event processing above may have triggered
        // dismiss() or cancel() callbacks that reset it
        if (!active_)
            return;

        if (!composing && active_.has_value() && !active_->has_input && rml_context_->GetFocusElement() == nullptr &&
            (hasKey(input.keys_pressed, SDL_SCANCODE_RETURN) ||
             hasKey(input.keys_pressed, SDL_SCANCODE_KP_ENTER))) {
            if (dismissFirstEnabledButton())
                return;
        }
        if (!composing && hasKey(input.keys_pressed, SDL_SCANCODE_ESCAPE)) {
            cancel();
        }
    }

    void RmlModalOverlay::render(int screen_w, int screen_h,
                                 float screen_x, float screen_y,
                                 float vp_x, float vp_y, float vp_w, float vp_h) {
        bool has_pending;
        {
            std::lock_guard lock(queue_mutex_);
            has_pending = !queue_.empty();
        }

        if (!active_ && !has_pending)
            return;

        if (!rml_context_) {
            initContext();
            if (!rml_context_)
                return;
        }

        if (!active_ && has_pending && elements_cached_)
            showNext();

        if (!active_)
            return;

        if (!rml_manager_ || !rml_manager_->getVulkanRenderInterface())
            return;

        rml_manager_->trackContextFrame(rml_context_, 0, 0);
        syncTheme();

        const int w = screen_w;
        const int h = screen_h;

        if (w <= 0 || h <= 0)
            return;

        if (w != width_ || h != height_) {
            width_ = w;
            height_ = h;
            rml_context_->SetDimensions(Rml::Vector2i(w, h));
        }

        rml_context_->Update();

        if (el_dialog_ && active_) {
            const float dp_ratio = rml_manager_->getDpRatio();
            const float dialog_w = static_cast<float>(active_->width_dp) * dp_ratio;
            const float dialog_h = el_dialog_->GetClientHeight();
            const float vp_cx = (vp_x - screen_x) + vp_w * 0.5f;
            const float vp_cy = (vp_y - screen_y) + vp_h * 0.5f;
            el_dialog_->SetProperty("left", std::format("{}px", vp_cx - dialog_w * 0.5f));
            el_dialog_->SetProperty("top", std::format("{}px", vp_cy - dialog_h * 0.5f));
            rml_context_->Update();
        }

        rml_manager_->queueVulkanContext(rml_context_, 0.0f, 0.0f, true);
    }

    void RmlModalOverlay::releaseRendererResources() {
    }

    void RmlModalOverlay::OverlayEventListener::ProcessEvent(Rml::Event& event) {
        assert(overlay);
        auto* target = event.GetTargetElement();
        if (!target)
            return;

        if (event == Rml::EventId::Submit && event.GetCurrentElement() == overlay->el_form_) {
            overlay->dismissFirstEnabledButton();
            event.StopPropagation();
            return;
        }

        if (event == Rml::EventId::Change && event.GetCurrentElement() == overlay->el_form_) {
            if (event.GetParameter<bool>("linebreak", false) &&
                rml_input::isTextEditableElement(event.GetTargetElement())) {
                overlay->dismissFirstEnabledButton();
                event.StopPropagation();
            }
            return;
        }

        const auto& id = target->GetId();

        if (id == "modal-backdrop") {
            overlay->cancel();
            return;
        }

        if (id.starts_with("modal-btn-")) {
            if (target->IsClassSet("disabled"))
                return;

            const std::string idx_str = id.substr(10);
            const size_t idx = std::stoul(idx_str);
            if (overlay->active_ && idx < overlay->active_->buttons.size()) {
                overlay->dismiss(overlay->active_->buttons[idx].label);
            }
        }
    }

} // namespace lfs::vis::gui
