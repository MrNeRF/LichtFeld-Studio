/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rml_sequencer_overlay.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/rmlui/rml_document_utils.hpp"
#include "gui/rmlui/rml_text_input_handler.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "gui/string_keys.hpp"
#include "internal/resource_paths.hpp"
#include "sequencer/keyframe.hpp"
#include "sequencer/sequencer_controller.hpp"
#include "theme/theme.hpp"

#include "gui/rmlui/sdl_rml_key_mapping.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/Input.h>
#include <SDL3/SDL_video.h>
#include <algorithm>
#include <cassert>
#include <fmt/format.h>

namespace lfs::vis::gui {

    namespace {
        const char* easingName(const lfs::sequencer::EasingType easing) {
            using namespace lichtfeld::Strings;
            switch (easing) {
            case lfs::sequencer::EasingType::LINEAR: return LOC(Scene::KEYFRAME_EASING_LINEAR);
            case lfs::sequencer::EasingType::EASE_IN: return LOC(Scene::KEYFRAME_EASING_EASE_IN);
            case lfs::sequencer::EasingType::EASE_OUT: return LOC(Scene::KEYFRAME_EASING_EASE_OUT);
            case lfs::sequencer::EasingType::EASE_IN_OUT: return LOC(Scene::KEYFRAME_EASING_EASE_IN_OUT);
            default: return LOC(Scene::KEYFRAME_EASING_LINEAR);
            }
        }
    } // namespace

    RmlSequencerOverlay::RmlSequencerOverlay(SequencerController& controller, RmlUIManager* rml_manager)
        : controller_(controller),
          rml_manager_(rml_manager) {
        assert(rml_manager_);
        listener_.overlay = this;
    }

    RmlSequencerOverlay::~RmlSequencerOverlay() {
        hidePreviewWindow();
        if (rml_context_ && rml_manager_)
            rml_manager_->destroyContext("sequencer_overlay");
    }

    void RmlSequencerOverlay::initContext() {
        if (rml_context_)
            return;

        rml_context_ = rml_manager_->createContext("sequencer_overlay", 800, 600);
        if (!rml_context_) {
            LOG_ERROR("RmlSequencerOverlay: failed to create context");
            return;
        }

        try {
            const auto rml_path = lfs::vis::getAssetPath("rmlui/sequencer_overlay.rml");
            document_ = rml_documents::loadDocument(rml_context_, rml_path);
            if (!document_) {
                LOG_ERROR("RmlSequencerOverlay: failed to load sequencer_overlay.rml");
                return;
            }
            document_->Show();
            cacheElements();
        } catch (const std::exception& e) {
            LOG_ERROR("RmlSequencerOverlay: resource not found: {}", e.what());
        }
    }

    bool RmlSequencerOverlay::ensureContextReady() {
        if (!rml_context_)
            initContext();
        if (!rml_context_ || !document_ || !elements_cached_)
            return false;

        if (width_ <= 0 || height_ <= 0) {
            int window_w = 800;
            int window_h = 600;
            if (auto* const window = rml_manager_ ? rml_manager_->getWindow() : nullptr)
                SDL_GetWindowSize(window, &window_w, &window_h);

            width_ = std::max(window_w, 1);
            height_ = std::max(window_h, 1);
            rml_context_->SetDimensions(Rml::Vector2i(width_, height_));
        }

        syncTheme();
        return true;
    }

    void RmlSequencerOverlay::reloadResources() {
        if (!rml_context_)
            return;

        hideContextMenu();
        hideEditOverlay();
        hidePreviewWindow();
        time_edit_active_ = false;
        focal_edit_active_ = false;
        wants_input_ = false;
        has_text_focus_ = false;

        if (document_) {
            rml_context_->UnloadDocument(document_);
            rml_context_->Update();
        }

        document_ = nullptr;
        el_menu_backdrop_ = nullptr;
        el_context_menu_ = nullptr;
        el_popup_backdrop_ = nullptr;
        el_time_popup_ = nullptr;
        el_focal_popup_ = nullptr;
        el_time_input_ = nullptr;
        el_focal_input_ = nullptr;
        el_edit_overlay_ = nullptr;
        el_edit_label_ = nullptr;
        el_edit_delta_ = nullptr;
        el_edit_apply_ = nullptr;
        el_edit_revert_ = nullptr;
        el_preview_window_ = nullptr;
        el_preview_title_ = nullptr;
        el_preview_image_ = nullptr;
        el_time_popup_title_ = nullptr;
        el_focal_popup_title_ = nullptr;
        el_time_ok_ = nullptr;
        el_time_cancel_ = nullptr;
        el_focal_ok_ = nullptr;
        el_focal_cancel_ = nullptr;
        elements_cached_ = false;
        base_rcss_.clear();
        has_theme_signature_ = false;
        width_ = 0;
        height_ = 0;

        try {
            const auto rml_path = lfs::vis::getAssetPath("rmlui/sequencer_overlay.rml");
            document_ = rml_documents::loadDocument(rml_context_, rml_path);
            if (!document_) {
                LOG_ERROR("RmlSequencerOverlay: failed to reload sequencer_overlay.rml");
                return;
            }
            document_->Show();
            cacheElements();
        } catch (const std::exception& e) {
            LOG_ERROR("RmlSequencerOverlay: resource not found during reload: {}", e.what());
            return;
        }

        syncTheme();
    }

    void RmlSequencerOverlay::cacheElements() {
        assert(document_);
        el_menu_backdrop_ = document_->GetElementById("menu-backdrop");
        el_context_menu_ = document_->GetElementById("keyframe-context-menu");
        el_popup_backdrop_ = document_->GetElementById("popup-backdrop");
        el_time_popup_ = document_->GetElementById("time-edit-popup");
        el_focal_popup_ = document_->GetElementById("focal-edit-popup");
        el_time_input_ = document_->GetElementById("time-edit-input");
        el_focal_input_ = document_->GetElementById("focal-edit-input");
        el_edit_overlay_ = document_->GetElementById("kf-edit-overlay");
        el_edit_label_ = document_->GetElementById("kf-edit-label");
        el_edit_delta_ = document_->GetElementById("kf-edit-delta");
        el_edit_apply_ = document_->GetElementById("kf-edit-apply");
        el_edit_revert_ = document_->GetElementById("kf-edit-revert");
        el_preview_window_ = document_->GetElementById("pip-preview-window");
        el_preview_title_ = document_->GetElementById("pip-preview-title");
        el_preview_image_ = document_->GetElementById("pip-preview-image");
        el_time_ok_ = document_->GetElementById("time-edit-ok");
        el_time_cancel_ = document_->GetElementById("time-edit-cancel");
        el_focal_ok_ = document_->GetElementById("focal-edit-ok");
        el_focal_cancel_ = document_->GetElementById("focal-edit-cancel");

        // Popup titles are the first .popup-title span inside each popup
        {
            Rml::ElementList elems;
            if (el_time_popup_) {
                el_time_popup_->GetElementsByClassName(elems, "popup-title");
                el_time_popup_title_ = elems.empty() ? nullptr : elems[0];
            }
            elems.clear();
            if (el_focal_popup_) {
                el_focal_popup_->GetElementsByClassName(elems, "popup-title");
                el_focal_popup_title_ = elems.empty() ? nullptr : elems[0];
            }
        }

        elements_cached_ = el_menu_backdrop_ && el_context_menu_ && el_popup_backdrop_ &&
                           el_time_popup_ && el_focal_popup_ && el_time_input_ &&
                           el_focal_input_ && el_edit_overlay_ && el_edit_label_ && el_edit_delta_ &&
                           el_preview_window_ && el_preview_title_ && el_preview_image_;

        if (!elements_cached_) {
            LOG_ERROR("RmlSequencerOverlay: missing DOM elements");
            return;
        }

        syncLocalization();

        el_menu_backdrop_->AddEventListener(Rml::EventId::Click, &listener_);
        el_context_menu_->AddEventListener(Rml::EventId::Click, &listener_);
        el_popup_backdrop_->AddEventListener(Rml::EventId::Click, &listener_);
        el_edit_overlay_->AddEventListener(Rml::EventId::Click, &listener_);
        el_time_popup_->AddEventListener(Rml::EventId::Click, &listener_);
        el_focal_popup_->AddEventListener(Rml::EventId::Click, &listener_);
    }

    void RmlSequencerOverlay::syncTheme() {
        if (!document_)
            return;

        const std::size_t theme_signature = rml_theme::currentThemeSignature();
        if (has_theme_signature_ && theme_signature == last_theme_signature_)
            return;
        last_theme_signature_ = theme_signature;
        has_theme_signature_ = true;

        if (base_rcss_.empty())
            base_rcss_ = rml_theme::loadBaseRCSS("rmlui/sequencer_overlay.rcss");

        rml_theme::applyTheme(document_, base_rcss_, rml_theme::loadBaseRCSS("rmlui/sequencer_overlay.theme.rcss"));
        syncLocalization();
    }

    void RmlSequencerOverlay::syncLocalization() {
        using namespace lichtfeld::Strings;
        if (el_edit_apply_)
            el_edit_apply_->SetInnerRML(LOC(Sequencer::APPLY_U));
        if (el_edit_revert_)
            el_edit_revert_->SetInnerRML(LOC(Sequencer::REVERT_ESC));
        if (document_) {
            if (auto* const close_btn = document_->GetElementById("kf-close-btn"))
                close_btn->SetInnerRML(LOC(Common::CLOSE));
        }
        if (el_time_popup_title_)
            el_time_popup_title_->SetInnerRML(LOC(Sequencer::EDIT_KEYFRAME_TIME));
        if (el_focal_popup_title_)
            el_focal_popup_title_->SetInnerRML(LOC(Sequencer::EDIT_FOCAL_LENGTH_TITLE));
        if (el_time_ok_)
            el_time_ok_->SetInnerRML(LOC(Common::OK));
        if (el_time_cancel_)
            el_time_cancel_->SetInnerRML(LOC(Common::CANCEL));
        if (el_focal_ok_)
            el_focal_ok_->SetInnerRML(LOC(Common::OK));
        if (el_focal_cancel_)
            el_focal_cancel_->SetInnerRML(LOC(Common::CANCEL));
    }

    std::string RmlSequencerOverlay::buildContextMenuHTML(
        std::optional<size_t> keyframe,
        const SequencerViewportEditMode edit_mode) const {
        const auto& timeline = controller_.timeline();
        std::string html;
        html.reserve(1024);

        using namespace lichtfeld::Strings;
        html += fmt::format(
            R"(<div class="context-menu-item" id="ctx-add">{}<span class="context-menu-label context-menu-shortcut">K</span></div>)",
            LOC(Sequencer::ADD_KEYFRAME_HERE));

        if (keyframe.has_value() && *keyframe < timeline.size()) {
            const size_t idx = *keyframe;
            const auto* const keyframe_data = timeline.getKeyframe(idx);
            if (!keyframe_data || keyframe_data->is_loop_point)
                return html;
            const bool is_first = (idx == 0);
            const bool is_last = (idx == timeline.size() - 1);

            html += R"(<div class="context-menu-separator"></div>)";
            html += fmt::format(
                R"(<div class="context-menu-item" id="ctx-update">{}<span class="context-menu-label context-menu-shortcut">U</span></div>)",
                LOC(Sequencer::UPDATE_TO_CURRENT_VIEW));
            html += fmt::format(
                R"(<div class="context-menu-item" id="ctx-goto">{}</div>)",
                LOC(Sequencer::GO_TO_KEYFRAME));
            html += fmt::format(
                R"(<div class="context-menu-item" id="ctx-focal">{}</div>)",
                LOC(Sequencer::EDIT_FOCAL_LENGTH));
            html += R"(<div class="context-menu-separator"></div>)";

            const bool translate_active = edit_mode == SequencerViewportEditMode::Translate;
            const bool rotate_active = edit_mode == SequencerViewportEditMode::Rotate;

            html += fmt::format(
                R"(<div class="context-menu-item{}" id="ctx-translate">{}</div>)",
                translate_active ? " active" : "", LOC(Sequencer::MOVE_TRANSLATE));
            html += fmt::format(
                R"(<div class="context-menu-item{}" id="ctx-rotate">{}</div>)",
                rotate_active ? " active" : "", LOC(Sequencer::ROTATE));

            html += R"(<div class="context-menu-separator"></div>)";

            if (!is_last) {
                const auto current_easing = timeline.keyframes()[idx].easing;
                html += fmt::format(R"(<div class="context-menu-label">{}</div>)", LOC(Sequencer::EASING));
                for (int e = 0; e < 4; ++e) {
                    const auto easing = static_cast<lfs::sequencer::EasingType>(e);
                    const bool active = (current_easing == easing);
                    html += fmt::format(
                        R"(<div class="context-menu-item submenu-item{}" id="ctx-easing-{}">{}</div>)",
                        active ? " active" : "", e, easingName(easing));
                }
            } else {
                html += fmt::format(
                    R"(<div class="context-menu-label">{}</div>)", LOC(Sequencer::EASING_LAST_KEYFRAME));
            }

            html += R"(<div class="context-menu-separator"></div>)";

            if (is_first)
                html += fmt::format(
                    R"(<div class="context-menu-item disabled" id="ctx-delete-disabled">{}</div>)",
                    LOC(Sequencer::DELETE_KEYFRAME));
            else
                html += fmt::format(
                    R"(<div class="context-menu-item" id="ctx-delete">{}<span class="context-menu-label context-menu-shortcut">Del</span></div>)",
                    LOC(Sequencer::DELETE_KEYFRAME));
        }

        return html;
    }

    void RmlSequencerOverlay::showContextMenu(float screen_x, float screen_y,
                                              std::optional<size_t> keyframe_index,
                                              const float time,
                                              const SequencerViewportEditMode edit_mode) {
        if (!ensureContextReady())
            return;

        context_menu_keyframe_ = keyframe_index;
        context_menu_time_ = time;
        context_menu_open_ = true;
        skip_next_click_ = true;

        const std::string html = buildContextMenuHTML(keyframe_index, edit_mode);
        el_context_menu_->SetInnerRML(html);
        el_context_menu_->SetClass("visible", true);
        el_menu_backdrop_->SetProperty("display", "block");

        rml_context_->Update();
        const float dp = rml_manager_ ? rml_manager_->getDpRatio() : 1.0f;
        const float menu_h = el_context_menu_->GetClientHeight();
        const float y = (screen_y + menu_h > static_cast<float>(height_))
                            ? std::max(0.0f, screen_y - menu_h)
                            : screen_y;

        el_context_menu_->SetProperty("left", fmt::format("{:.0f}dp", screen_x / dp));
        el_context_menu_->SetProperty("top", fmt::format("{:.0f}dp", y / dp));
    }

    void RmlSequencerOverlay::hideContextMenu() {
        if (!elements_cached_)
            return;

        context_menu_open_ = false;
        context_menu_keyframe_ = std::nullopt;
        context_menu_time_ = 0.0f;
        el_context_menu_->SetClass("visible", false);
        el_menu_backdrop_->SetProperty("display", "none");
    }

    void RmlSequencerOverlay::showTimeEdit(size_t index, float current_time) {
        if (!ensureContextReady())
            return;

        time_edit_active_ = true;
        time_edit_index_ = index;

        el_time_input_->SetAttribute("value", fmt::format("{:.2f}", current_time));

        const float dp = rml_manager_ ? rml_manager_->getDpRatio() : 1.0f;
        const float popup_x = static_cast<float>(width_) / (2.0f * dp) - 110.0f;
        const float popup_y = static_cast<float>(height_) / (2.0f * dp) - 60.0f;
        el_time_popup_->SetProperty("left", fmt::format("{:.0f}dp", popup_x));
        el_time_popup_->SetProperty("top", fmt::format("{:.0f}dp", popup_y));
        el_time_popup_->SetProperty("display", "block");
        el_popup_backdrop_->SetProperty("display", "block");

        el_time_input_->Focus();
        has_text_focus_ = true;
    }

    void RmlSequencerOverlay::showFocalEdit(size_t index, float current_focal_mm) {
        if (!ensureContextReady())
            return;

        focal_edit_active_ = true;
        focal_edit_index_ = index;

        el_focal_input_->SetAttribute("value", fmt::format("{:.1f}", current_focal_mm));

        const float dp = rml_manager_ ? rml_manager_->getDpRatio() : 1.0f;
        const float popup_x = static_cast<float>(width_) / (2.0f * dp) - 110.0f;
        const float popup_y = static_cast<float>(height_) / (2.0f * dp) - 60.0f;
        el_focal_popup_->SetProperty("left", fmt::format("{:.0f}dp", popup_x));
        el_focal_popup_->SetProperty("top", fmt::format("{:.0f}dp", popup_y));
        el_focal_popup_->SetProperty("display", "block");
        el_popup_backdrop_->SetProperty("display", "block");

        el_focal_input_->Focus();
        has_text_focus_ = true;
    }

    void RmlSequencerOverlay::submitTimeEdit() {
        if (!time_edit_active_ || !el_time_input_)
            return;

        const Rml::String val = el_time_input_->GetAttribute<Rml::String>("value", "");
        const float new_time = std::strtof(val.c_str(), nullptr);
        if (new_time > 0.0f && time_edit_index_ > 0)
            pending_time_edit_ = EditResult{time_edit_index_, new_time};

        time_edit_active_ = false;
        has_text_focus_ = false;
        el_time_popup_->SetProperty("display", "none");
        el_popup_backdrop_->SetProperty("display", "none");
    }

    void RmlSequencerOverlay::submitFocalEdit() {
        if (!focal_edit_active_ || !el_focal_input_)
            return;

        const Rml::String val = el_focal_input_->GetAttribute<Rml::String>("value", "");
        const float new_focal = std::strtof(val.c_str(), nullptr);
        if (new_focal > 0.0f)
            pending_focal_edit_ = EditResult{focal_edit_index_, new_focal};

        focal_edit_active_ = false;
        has_text_focus_ = false;
        el_focal_popup_->SetProperty("display", "none");
        el_popup_backdrop_->SetProperty("display", "none");
    }

    void RmlSequencerOverlay::updateEditOverlay(size_t selected, float pos_delta, float rot_delta,
                                                float right_x, float top_y) {
        if (!ensureContextReady())
            return;

        constexpr float MARGIN = 16.0f;
        constexpr float OVERLAY_WIDTH = 200.0f;
        constexpr const char* DEG_SIGN = "\xC2\xB0";

        const float dp = rml_manager_ ? rml_manager_->getDpRatio() : 1.0f;
        const float left = right_x / dp - OVERLAY_WIDTH - MARGIN;
        const float top = top_y / dp + MARGIN;

        el_edit_overlay_->SetProperty("left", fmt::format("{:.0f}dp", left));
        el_edit_overlay_->SetProperty("top", fmt::format("{:.0f}dp", top));

        overlay_px_left_ = left * dp;
        overlay_px_top_ = top * dp;
        overlay_px_width_ = (OVERLAY_WIDTH + 18.0f) * dp;
        overlay_px_height_ = 80.0f * dp;

        const size_t kf_num = selected + 1;
        el_edit_label_->SetInnerRML(
            fmt::format(fmt::runtime(LOC(lichtfeld::Strings::Sequencer::EDITING_KEYFRAME)), kf_num));
        el_edit_delta_->SetInnerRML(fmt::format("{:.3f}m  {:.1f}{}", pos_delta, rot_delta, DEG_SIGN));

        if (!edit_overlay_visible_) {
            el_edit_overlay_->SetProperty("display", "block");
            edit_overlay_visible_ = true;
        }
    }

    void RmlSequencerOverlay::hideEditOverlay() {
        if (!elements_cached_ || !edit_overlay_visible_)
            return;

        el_edit_overlay_->SetProperty("display", "none");
        edit_overlay_visible_ = false;
        overlay_px_left_ = overlay_px_top_ = overlay_px_width_ = overlay_px_height_ = 0.0f;
    }

    void RmlSequencerOverlay::showPreviewWindow(const float left, const float top,
                                                const float width, const float height,
                                                const std::string& title, const bool playing,
                                                const std::uintptr_t texture_id) {
        if (!ensureContextReady() || texture_id == 0)
            return;

        (void)texture_id;
        if (!rml_manager_ || !rml_manager_->getVulkanRenderInterface() ||
            !el_preview_window_ || !el_preview_title_ || !el_preview_image_)
            return;

        if (!preview_source_.empty()) {
            el_preview_image_->SetAttribute("src", "");
            preview_source_.clear();
        }

        el_preview_window_->SetProperty("left", fmt::format("{:.1f}px", left));
        el_preview_window_->SetProperty("top", fmt::format("{:.1f}px", top));
        el_preview_window_->SetProperty("width", fmt::format("{:.1f}px", width + 8.0f));
        el_preview_title_->SetInnerRML(title);
        el_preview_window_->SetClass("playing", playing);
        el_preview_image_->SetProperty("width", fmt::format("{:.1f}px", width));
        el_preview_image_->SetProperty("height", fmt::format("{:.1f}px", height));

        el_preview_window_->SetProperty("display", "block");
        preview_visible_ = true;
    }

    void RmlSequencerOverlay::hidePreviewWindow() {
        if (!preview_source_.empty()) {
            preview_source_.clear();
        }

        if (el_preview_window_)
            el_preview_window_->SetProperty("display", "none");
        if (el_preview_image_)
            el_preview_image_->SetAttribute("src", "");

        preview_visible_ = false;
    }

    void RmlSequencerOverlay::processInput(const lfs::vis::PanelInputState& input) {
        wants_input_ = false;
        if (!rml_context_ || !document_ || !elements_cached_)
            return;

        const bool anything_visible = context_menu_open_ || time_edit_active_ ||
                                      focal_edit_active_ || edit_overlay_visible_ ||
                                      preview_visible_;
        if (!anything_visible)
            return;
        if (rml_manager_)
            rml_manager_->trackContextFrame(rml_context_, 0, 0);

        // Sync any property changes from this frame before hover and click hit-testing.
        rml_context_->Update();

        const float mx = input.mouse_x;
        const float my = input.mouse_y;

        const int mods = gui::sdlModsToRml(input.key_ctrl, input.key_shift,
                                           input.key_alt, input.key_super);

        rml_context_->ProcessMouseMove(static_cast<int>(mx), static_cast<int>(my), mods);

        auto* hover = rml_context_->GetHoverElement();
        const bool over_interactive = hover && hover->GetTagName() != "body" &&
                                      hover->GetId() != "body";

        if (edit_overlay_visible_ && el_edit_overlay_) {
            const float h = el_edit_overlay_->GetOffsetHeight();
            if (h > 0.0f)
                overlay_px_height_ = h;
        }

        const bool over_edit_overlay = isMouseOverEditOverlay(mx, my);

        if (over_interactive || over_edit_overlay ||
            context_menu_open_ || time_edit_active_ || focal_edit_active_) {
            wants_input_ = true;

            if (skip_next_click_) {
                skip_next_click_ = false;
            } else {
                if (input.mouse_clicked[0])
                    rml_context_->ProcessMouseButtonDown(0, mods);
                if (input.mouse_released[0])
                    rml_context_->ProcessMouseButtonUp(0, mods);
                if (input.mouse_clicked[1])
                    rml_context_->ProcessMouseButtonDown(1, mods);
                if (input.mouse_released[1])
                    rml_context_->ProcessMouseButtonUp(1, mods);
            }
        }

        const bool need_keyboard = has_text_focus_ || context_menu_open_ ||
                                   time_edit_active_ || focal_edit_active_;
        if (need_keyboard) {
            auto& focus = gui::guiFocusState();
            focus.want_capture_keyboard = true;
            if (has_text_focus_ || time_edit_active_ || focal_edit_active_)
                focus.want_text_input = true;

            auto* const text_input_handler =
                rml_manager_ ? rml_manager_->getTextInputHandler() : nullptr;
            const bool composing = text_input_handler && text_input_handler->isComposing();

            for (int sc : input.keys_pressed) {
                if (composing &&
                    (sc == SDL_SCANCODE_RETURN || sc == SDL_SCANCODE_KP_ENTER ||
                     sc == SDL_SCANCODE_ESCAPE)) {
                    continue;
                }
                const auto rml_key = gui::sdlScancodeToRml(static_cast<SDL_Scancode>(sc));
                if (rml_key != Rml::Input::KI_UNKNOWN) {
                    if (text_input_handler && text_input_handler->handleKeyDown(rml_key, mods))
                        continue;
                    rml_context_->ProcessKeyDown(rml_key, mods);
                }
            }
            for (int sc : input.keys_released) {
                if (composing && (sc == SDL_SCANCODE_RETURN || sc == SDL_SCANCODE_KP_ENTER ||
                                  sc == SDL_SCANCODE_ESCAPE))
                    continue;
                const auto rml_key = gui::sdlScancodeToRml(static_cast<SDL_Scancode>(sc));
                if (rml_key != Rml::Input::KI_UNKNOWN)
                    rml_context_->ProcessKeyUp(rml_key, mods);
            }

            if (has_text_focus_) {
                if (text_input_handler && input.has_text_editing) {
                    text_input_handler->handleTextEditing(
                        input.text_editing, input.text_editing_start, input.text_editing_length);
                }

                bool forward_text_codepoints = input.text_inputs.empty();
                for (const auto& text_input : input.text_inputs) {
                    if (!text_input_handler || !text_input_handler->handleTextInput(text_input))
                        forward_text_codepoints = true;
                }

                if (forward_text_codepoints) {
                    for (uint32_t cp : input.text_codepoints)
                        rml_context_->ProcessTextInput(static_cast<Rml::Character>(cp));
                }
            }

            if (!composing &&
                (gui::hasKey(input.keys_pressed, SDL_SCANCODE_RETURN) ||
                 gui::hasKey(input.keys_pressed, SDL_SCANCODE_KP_ENTER))) {
                if (time_edit_active_)
                    submitTimeEdit();
                else if (focal_edit_active_)
                    submitFocalEdit();
            }
            if (!composing && gui::hasKey(input.keys_pressed, SDL_SCANCODE_ESCAPE)) {
                if (time_edit_active_) {
                    time_edit_active_ = false;
                    has_text_focus_ = false;
                    el_time_popup_->SetProperty("display", "none");
                    el_popup_backdrop_->SetProperty("display", "none");
                } else if (focal_edit_active_) {
                    focal_edit_active_ = false;
                    has_text_focus_ = false;
                    el_focal_popup_->SetProperty("display", "none");
                    el_popup_backdrop_->SetProperty("display", "none");
                } else if (context_menu_open_) {
                    hideContextMenu();
                }
            }
        }

        if (edit_overlay_visible_ && !need_keyboard) {
            if (gui::hasKey(input.keys_pressed, SDL_SCANCODE_U))
                pending_actions_.push_back({Action::APPLY_EDIT, 0, 0});
            if (gui::hasKey(input.keys_pressed, SDL_SCANCODE_ESCAPE))
                pending_actions_.push_back({Action::REVERT_EDIT, 0, 0});
        }
    }

    void RmlSequencerOverlay::render(int screen_w, int screen_h) {
        const bool anything_visible = context_menu_open_ || time_edit_active_ ||
                                      focal_edit_active_ || edit_overlay_visible_ ||
                                      preview_visible_;
        if (!anything_visible)
            return;

        if (!ensureContextReady())
            return;

        if (!rml_manager_ || !rml_manager_->getVulkanRenderInterface())
            return;

        rml_manager_->trackContextFrame(rml_context_, 0, 0);

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
        rml_manager_->queueVulkanContext(rml_context_, 0.0f, 0.0f, true);
    }

    void RmlSequencerOverlay::compositeToScreen(const int screen_w, const int screen_h) const {
        (void)screen_w;
        (void)screen_h;
    }

    void RmlSequencerOverlay::destroyGraphicsResources() {
        hidePreviewWindow();
    }

    std::optional<RmlSequencerOverlay::PendingAction> RmlSequencerOverlay::consumeAction() {
        if (pending_actions_.empty())
            return std::nullopt;
        auto action = pending_actions_.front();
        pending_actions_.erase(pending_actions_.begin());
        return action;
    }

    std::optional<RmlSequencerOverlay::EditResult> RmlSequencerOverlay::consumeTimeEdit() {
        auto result = pending_time_edit_;
        pending_time_edit_ = std::nullopt;
        return result;
    }

    std::optional<RmlSequencerOverlay::EditResult> RmlSequencerOverlay::consumeFocalEdit() {
        auto result = pending_focal_edit_;
        pending_focal_edit_ = std::nullopt;
        return result;
    }

    void RmlSequencerOverlay::OverlayEventListener::ProcessEvent(Rml::Event& event) {
        assert(overlay);
        auto* target = event.GetTargetElement();
        while (target && target->GetId().empty())
            target = target->GetParentNode();
        if (!target)
            return;

        const auto& id = target->GetId();

        if (id == "menu-backdrop") {
            overlay->hideContextMenu();
            return;
        }

        if (id == "popup-backdrop") {
            if (overlay->time_edit_active_) {
                overlay->time_edit_active_ = false;
                overlay->has_text_focus_ = false;
                overlay->el_time_popup_->SetProperty("display", "none");
            }
            if (overlay->focal_edit_active_) {
                overlay->focal_edit_active_ = false;
                overlay->has_text_focus_ = false;
                overlay->el_focal_popup_->SetProperty("display", "none");
            }
            overlay->el_popup_backdrop_->SetProperty("display", "none");
            return;
        }

        if (id == "ctx-add") {
            overlay->pending_actions_.push_back({RmlSequencerOverlay::Action::ADD_KEYFRAME, 0, 0, overlay->context_menu_time_});
            overlay->hideContextMenu();
        } else if (id == "ctx-update") {
            if (overlay->context_menu_keyframe_) {
                overlay->pending_actions_.push_back(
                    {RmlSequencerOverlay::Action::UPDATE_KEYFRAME,
                     *overlay->context_menu_keyframe_, 0, 0.0f});
            }
            overlay->hideContextMenu();
        } else if (id == "ctx-goto") {
            if (overlay->context_menu_keyframe_) {
                overlay->pending_actions_.push_back(
                    {RmlSequencerOverlay::Action::GOTO_KEYFRAME,
                     *overlay->context_menu_keyframe_, 0, 0.0f});
            }
            overlay->hideContextMenu();
        } else if (id == "ctx-focal") {
            if (overlay->context_menu_keyframe_) {
                overlay->pending_actions_.push_back(
                    {RmlSequencerOverlay::Action::EDIT_FOCAL_LENGTH,
                     *overlay->context_menu_keyframe_, 0, 0.0f});
            }
            overlay->hideContextMenu();
        } else if (id == "ctx-translate") {
            if (overlay->context_menu_keyframe_) {
                overlay->pending_actions_.push_back(
                    {RmlSequencerOverlay::Action::SET_TRANSLATE,
                     *overlay->context_menu_keyframe_, 0, 0.0f});
            }
            overlay->hideContextMenu();
        } else if (id == "ctx-rotate") {
            if (overlay->context_menu_keyframe_) {
                overlay->pending_actions_.push_back(
                    {RmlSequencerOverlay::Action::SET_ROTATE,
                     *overlay->context_menu_keyframe_, 0, 0.0f});
            }
            overlay->hideContextMenu();
        } else if (id.starts_with("ctx-easing-")) {
            const int easing_val = id.back() - '0';
            if (easing_val >= 0 && easing_val < 4 && overlay->context_menu_keyframe_) {
                overlay->pending_actions_.push_back(
                    {RmlSequencerOverlay::Action::SET_EASING,
                     *overlay->context_menu_keyframe_, easing_val, 0.0f});
            }
            overlay->hideContextMenu();
        } else if (id == "ctx-delete") {
            if (overlay->context_menu_keyframe_) {
                overlay->pending_actions_.push_back(
                    {RmlSequencerOverlay::Action::DELETE_KEYFRAME,
                     *overlay->context_menu_keyframe_, 0, 0.0f});
            }
            overlay->hideContextMenu();
        } else if (id == "kf-close-btn") {
            overlay->pending_actions_.push_back(
                {RmlSequencerOverlay::Action::CLOSE_EDIT_PANEL, 0, 0, 0.0f});
        } else if (id == "kf-edit-apply") {
            overlay->pending_actions_.push_back(
                {RmlSequencerOverlay::Action::APPLY_EDIT, 0, 0, 0.0f});
        } else if (id == "kf-edit-revert") {
            overlay->pending_actions_.push_back(
                {RmlSequencerOverlay::Action::REVERT_EDIT, 0, 0, 0.0f});
        } else if (id == "time-edit-ok") {
            overlay->submitTimeEdit();
        } else if (id == "time-edit-cancel") {
            overlay->time_edit_active_ = false;
            overlay->has_text_focus_ = false;
            overlay->el_time_popup_->SetProperty("display", "none");
            overlay->el_popup_backdrop_->SetProperty("display", "none");
        } else if (id == "focal-edit-ok") {
            overlay->submitFocalEdit();
        } else if (id == "focal-edit-cancel") {
            overlay->focal_edit_active_ = false;
            overlay->has_text_focus_ = false;
            overlay->el_focal_popup_->SetProperty("display", "none");
            overlay->el_popup_backdrop_->SetProperty("display", "none");
        }
    }

} // namespace lfs::vis::gui
