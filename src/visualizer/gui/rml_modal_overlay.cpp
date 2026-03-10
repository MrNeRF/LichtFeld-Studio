/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// clang-format off
#include <glad/glad.h>
// clang-format on

#include "gui/rml_modal_overlay.hpp"
#include "core/logger.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/panel_layout.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "gui/rmlui/rmlui_render_interface.hpp"
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
        fbo_.destroy();
        if (rml_context_ && rml_manager_)
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

        rml_context_->EnableMouseCursor(false);

        try {
            const auto rml_path = lfs::vis::getAssetPath("rmlui/modal_overlay.rml");
            document_ = rml_context_->LoadDocument(rml_path.string());
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

    void RmlModalOverlay::cacheElements() {
        assert(document_);
        el_backdrop_ = document_->GetElementById("modal-backdrop");
        el_dialog_ = document_->GetElementById("modal-dialog");
        el_title_ = document_->GetElementById("modal-title");
        el_content_ = document_->GetElementById("modal-content");
        el_input_row_ = document_->GetElementById("modal-input-row");
        el_input_ = document_->GetElementById("modal-input");
        el_button_row_ = document_->GetElementById("modal-button-row");

        elements_cached_ = el_backdrop_ && el_dialog_ && el_title_ && el_content_ &&
                           el_input_row_ && el_input_ && el_button_row_;

        if (!elements_cached_) {
            LOG_ERROR("RmlModalOverlay: missing DOM elements");
            return;
        }

        el_backdrop_->AddEventListener(Rml::EventId::Click, &listener_);
        el_button_row_->AddEventListener(Rml::EventId::Click, &listener_);
    }

    std::string RmlModalOverlay::generateThemeRCSS(const lfs::vis::Theme& t) const {
        using rml_theme::colorToRml;
        using rml_theme::colorToRmlAlpha;
        const auto& p = t.palette;

        const auto surface = colorToRmlAlpha(p.surface, 0.98f);
        const auto border = colorToRmlAlpha(p.border, 0.4f);
        const auto backdrop = colorToRmlAlpha(
            t.isLightTheme() ? ImVec4{0.12f, 0.14f, 0.18f, 1.0f} : p.background,
            t.isLightTheme() ? 0.18f : 0.44f);
        const auto text = colorToRml(p.text);
        const auto text_dim = colorToRml(p.text_dim);
        const auto sep_color = colorToRmlAlpha(p.border, 0.5f);
        const auto info_border = colorToRml(p.success);
        const auto warn_border = colorToRml(p.warning);
        const auto err_border = colorToRml(p.error);
        const auto error_col = colorToRml(p.error);
        const auto warning_col = colorToRml(p.warning);
        const int rounding = static_cast<int>(t.sizes.window_rounding);

        return std::format(
            ".modal-backdrop {{ background-color: {}; }}\n"
            ".modal-dialog {{ background-color: {}; border-color: {}; border-radius: {}dp; }}\n"
            ".modal-title {{ color: {}; }}\n"
            ".modal-sep {{ background-color: {}; }}\n"
            ".modal-content {{ color: {}; }}\n"
            ".dim-text {{ color: {}; }}\n"
            ".error-text {{ color: {}; }}\n"
            ".warning-text {{ color: {}; }}\n"
            ".modal-dialog.style-info {{ border-color: {}; }}\n"
            ".modal-dialog.style-warning {{ border-color: {}; }}\n"
            ".modal-dialog.style-error {{ border-color: {}; }}\n",
            backdrop, surface, border, rounding,
            text, sep_color, text, text_dim,
            error_col, warning_col,
            info_border, warn_border, err_border);
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

        rml_theme::applyTheme(document_, base_rcss_, rml_theme::generateAllThemeMedia([this](const auto& th) { return generateThemeRCSS(th); }));
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
            const std::string cls = "btn btn--" + btn.style + (btn.disabled ? " disabled" : "");
            btn_html += std::format(
                R"(<div class="{}" id="modal-btn-{}">{}</div>)",
                cls, i, btn.label);
        }
        el_button_row_->SetInnerRML(btn_html);

        el_dialog_->SetClass("style-info", req.style == lfs::core::ModalStyle::Info);
        el_dialog_->SetClass("style-warning", req.style == lfs::core::ModalStyle::Warning);
        el_dialog_->SetClass("style-error", req.style == lfs::core::ModalStyle::Error);

        el_dialog_->SetProperty("width", std::format("{}dp", req.width_dp));

        el_backdrop_->SetProperty("display", "block");
        el_dialog_->SetProperty("display", "block");

        if (req.has_input)
            el_input_->Focus();

        active_ = std::move(req);
    }

    lfs::core::ModalResult RmlModalOverlay::collectFormValues() const {
        lfs::core::ModalResult result;

        if (active_ && active_->has_input) {
            result.input_value = el_input_->GetAttribute<Rml::String>("value", "");
        }

        // Collect all text inputs from the content area
        Rml::ElementList inputs;
        el_content_->GetElementsByTagName(inputs, "input");
        for (auto* input : inputs) {
            const auto id = input->GetId();
            if (!id.empty()) {
                result.form_values[id] = input->GetAttribute<Rml::String>("value", "");
            }
        }

        return result;
    }

    void RmlModalOverlay::dismiss(const std::string& button_label) {
        if (!active_)
            return;

        el_backdrop_->SetProperty("display", "none");
        el_dialog_->SetProperty("display", "none");

        auto result = collectFormValues();
        result.button_label = button_label;

        auto on_result = std::move(active_->on_result);
        active_.reset();

        if (on_result)
            on_result(result);
    }

    void RmlModalOverlay::cancel() {
        if (!active_)
            return;

        el_backdrop_->SetProperty("display", "none");
        el_dialog_->SetProperty("display", "none");

        auto on_cancel = std::move(active_->on_cancel);
        active_.reset();

        if (on_cancel)
            on_cancel();
    }

    void RmlModalOverlay::processInput(const PanelInputState& input) {
        if (!active_ || !rml_context_ || !elements_cached_)
            return;

        auto& focus = guiFocusState();
        focus.want_capture_mouse = true;
        focus.want_capture_keyboard = true;
        if (active_->has_input)
            focus.want_text_input = true;

        const float mx = input.mouse_x - input.screen_x;
        const float my = input.mouse_y - input.screen_y;
        rml_context_->ProcessMouseMove(static_cast<int>(mx), static_cast<int>(my), 0);

        if (input.mouse_clicked[0])
            rml_context_->ProcessMouseButtonDown(0, 0);
        if (input.mouse_released[0])
            rml_context_->ProcessMouseButtonUp(0, 0);

        const int mods = sdlModsToRml(input.key_ctrl, input.key_shift,
                                      input.key_alt, input.key_super);
        for (const int sc : input.keys_pressed) {
            const auto rml_key = sdlScancodeToRml(static_cast<SDL_Scancode>(sc));
            if (rml_key != Rml::Input::KI_UNKNOWN)
                rml_context_->ProcessKeyDown(rml_key, mods);
        }
        for (const int sc : input.keys_released) {
            const auto rml_key = sdlScancodeToRml(static_cast<SDL_Scancode>(sc));
            if (rml_key != Rml::Input::KI_UNKNOWN)
                rml_context_->ProcessKeyUp(rml_key, mods);
        }

        for (uint32_t cp : input.text_codepoints)
            rml_context_->ProcessTextInput(static_cast<Rml::Character>(cp));

        if (hasKey(input.keys_pressed, SDL_SCANCODE_RETURN) ||
            hasKey(input.keys_pressed, SDL_SCANCODE_KP_ENTER)) {
            for (size_t i = 0; i < active_->buttons.size(); ++i) {
                if (!active_->buttons[i].disabled) {
                    dismiss(active_->buttons[i].label);
                    return;
                }
            }
        }
        if (hasKey(input.keys_pressed, SDL_SCANCODE_ESCAPE)) {
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

        if (!rml_manager_->shouldDeferFboUpdate(fbo_)) {
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

            fbo_.ensure(w, h);
            if (!fbo_.valid())
                return;

            auto* render_iface = rml_manager_->getRenderInterface();
            assert(render_iface);
            render_iface->SetViewport(w, h);

            GLint prev_fbo = 0;
            fbo_.bind(&prev_fbo);
            render_iface->SetTargetFramebuffer(fbo_.fbo());

            render_iface->BeginFrame();
            rml_context_->Render();
            render_iface->EndFrame();

            render_iface->SetTargetFramebuffer(0);
            fbo_.unbind(prev_fbo);
        }

        if (fbo_.valid())
            fbo_.blitToScreen(0.0f, 0.0f, static_cast<float>(screen_w), static_cast<float>(screen_h),
                              screen_w, screen_h);
    }

    void RmlModalOverlay::destroyGLResources() {
        fbo_.destroy();
    }

    void RmlModalOverlay::OverlayEventListener::ProcessEvent(Rml::Event& event) {
        assert(overlay);
        auto* target = event.GetTargetElement();
        if (!target)
            return;

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
