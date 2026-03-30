/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// clang-format off
#include <glad/glad.h>
// clang-format on

#include "gui/startup_overlay.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/rmlui/rml_input_utils.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "gui/rmlui/rmlui_render_interface.hpp"
#include "gui/rmlui/sdl_rml_key_mapping.hpp"
#include "gui/string_keys.hpp"
#include "internal/resource_paths.hpp"
#include "theme/theme.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/Elements/ElementFormControlSelect.h>
#include <RmlUi/Core/Input.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <imgui_internal.h>
#include <imgui.h>

#ifdef _WIN32
#include <shellapi.h>
#include <windows.h>
#endif

namespace lfs::vis::gui {

    using rml_theme::colorToRml;
    using rml_theme::colorToRmlAlpha;

    class LinkClickListener final : public Rml::EventListener {
    public:
        void ProcessEvent(Rml::Event& event) override {
            auto* el = event.GetCurrentElement();
            if (!el)
                return;
            auto url = el->GetAttribute("data-url", Rml::String(""));
            if (!url.empty())
                StartupOverlay::openURL(url.c_str());
        }
    };

    class LangChangeListener final : public Rml::EventListener {
    public:
        void ProcessEvent(Rml::Event& event) override {
            auto* el = event.GetCurrentElement();
            if (!el)
                return;
            auto* select = dynamic_cast<Rml::ElementFormControlSelect*>(el);
            if (!select)
                return;
            int idx = select->GetSelection();
            if (idx < 0)
                return;

            auto& loc = lfs::event::LocalizationManager::getInstance();
            const auto available = loc.getAvailableLanguages();
            if (idx < static_cast<int>(available.size()))
                loc.setLanguage(available[idx]);
        }
    };

    void StartupOverlay::openURL(const char* url) {
#ifdef _WIN32
        ShellExecuteA(nullptr, "open", url, nullptr, nullptr, SW_SHOWNORMAL);
#else
        std::string cmd = "xdg-open \"" + std::string(url) + "\" &";
        std::system(cmd.c_str());
#endif
    }

    void StartupOverlay::init(RmlUIManager* mgr) {
        assert(mgr);
        rml_manager_ = mgr;

        rml_context_ = rml_manager_->createContext("startup_overlay", 800, 600);
        if (!rml_context_) {
            LOG_ERROR("StartupOverlay: failed to create RML context");
            return;
        }

        try {
            const auto rml_path = lfs::vis::getAssetPath("rmlui/startup.rml");
            document_ = rml_context_->LoadDocument(rml_path.string());
            if (!document_) {
                LOG_ERROR("StartupOverlay: failed to load startup.rml");
                return;
            }
            document_->Show();
        } catch (const std::exception& e) {
            LOG_ERROR("StartupOverlay: resource not found: {}", e.what());
            return;
        }

        populateLanguages();
        updateLocalizedText();

        link_listener_ = new LinkClickListener();
        for (const char* id : {"link-discord", "link-x", "link-donate"}) {
            auto* el = document_->GetElementById(id);
            if (el)
                el->AddEventListener(Rml::EventId::Click, link_listener_);
        }

        lang_listener_ = new LangChangeListener();
        auto* lang_select = document_->GetElementById("lang-select");
        if (lang_select)
            lang_select->AddEventListener(Rml::EventId::Change, lang_listener_);

        updateTheme();
    }

    void StartupOverlay::shutdown() {
        fbo_.destroy();
        if (rml_context_ && rml_manager_)
            rml_manager_->destroyContext("startup_overlay");
        rml_context_ = nullptr;
        document_ = nullptr;
        delete link_listener_;
        link_listener_ = nullptr;
        delete lang_listener_;
        lang_listener_ = nullptr;
    }

    void StartupOverlay::populateLanguages() {
        auto* select_el = document_->GetElementById("lang-select");
        if (!select_el)
            return;
        auto* select = dynamic_cast<Rml::ElementFormControlSelect*>(select_el);
        if (!select)
            return;

        auto& loc = lfs::event::LocalizationManager::getInstance();
        const auto langs = loc.getAvailableLanguages();
        const auto names = loc.getAvailableLanguageNames();
        const auto& current = loc.getCurrentLanguage();

        for (size_t i = 0; i < langs.size(); ++i) {
            select->Add(names[i], langs[i]);
            if (langs[i] == current)
                select->SetSelection(static_cast<int>(i));
        }
    }

    void StartupOverlay::updateLocalizedText() {
        if (!document_)
            return;

        auto set_text = [&](const char* id, const char* key) {
            auto* el = document_->GetElementById(id);
            if (el)
                el->SetInnerRML(LOC(key));
        };

        set_text("supported-text", lichtfeld::Strings::Startup::SUPPORTED_BY);
        set_text("lang-label", lichtfeld::Strings::Preferences::LANGUAGE);
        set_text("click-hint", lichtfeld::Strings::Startup::CLICK_TO_CONTINUE);
    }

    std::string StartupOverlay::generateThemeRCSS(const lfs::vis::Theme& t) const {
        const auto& p = t.palette;
        const bool is_light = t.isLightTheme();

        auto blend = [](const ImVec4& a, const ImVec4& b, float t_val) -> ImVec4 {
            return {a.x + (b.x - a.x) * t_val,
                    a.y + (b.y - a.y) * t_val,
                    a.z + (b.z - a.z) * t_val,
                    1.0f};
        };

        const auto border = colorToRmlAlpha(p.border, is_light ? 0.75f : 0.62f);
        const auto text = colorToRml(p.text);
        const auto text_dim_85 = colorToRmlAlpha(p.text_dim, 0.85f);
        const auto text_dim_50 = colorToRmlAlpha(p.text_dim, 0.50f);
        const auto primary = colorToRmlAlpha(p.primary, is_light ? 0.78f : 0.62f);
        const auto select_bg = colorToRmlAlpha(p.background, is_light ? 0.90f : 0.78f);
        const auto selectbox_bg = colorToRmlAlpha(p.surface, is_light ? 0.95f : 0.90f);

        const ImVec4 base_color = blend(p.surface, p.text, is_light ? 0.04f : 0.10f);
        const ImVec4 border_color = blend(p.border, p.text, is_light ? 0.28f : 0.38f);
        const float base_alpha = is_light ? 0.82f : 0.86f;
        const float bdr_alpha = is_light ? 0.40f : 0.50f;
        const float inset_alpha = is_light ? 0.08f : 0.05f;

        std::string box_shadow;
        if (t.shadows.enabled)
            box_shadow = std::format("box-shadow: {}, {} 0dp 0dp 0dp 1dp inset;",
                                     rml_theme::layeredShadow(t, 4),
                                     colorToRmlAlpha(RmlColor{1, 1, 1, 1}, inset_alpha));

        return std::format(
            "#overlay-box {{ background-color: {7}; border: 1dp {8}; border-radius: 12dp; {9} }}\n"
            ".dim-text {{ color: {2}; }}\n"
            ".hint-text {{ color: {3}; }}\n"
            ".social-link span {{ color: {2}; }}\n"
            ".social-icon {{ image-color: {2}; }}\n"
            ".heart-icon {{ image-color: rgb(220, 50, 50); }}\n"
            "select {{ color: {1}; background-color: {5}; border-color: {0}; }}\n"
            "select:hover {{ border-color: {4}; }}\n"
            "selectbox {{ background-color: {6}; border-color: {0}; }}\n"
            "selectbox option:hover {{ background-color: {4}; }}\n"
            "#lang-label {{ color: {2}; }}\n",
            border, text, text_dim_85, text_dim_50, primary, select_bg, selectbox_bg,
            colorToRmlAlpha(base_color, base_alpha),
            colorToRmlAlpha(border_color, bdr_alpha),
            box_shadow);
    }

    void StartupOverlay::updateTheme() {
        if (!document_)
            return;

        const auto& t = theme();
        const std::size_t theme_signature = rml_theme::currentThemeSignature();
        if (has_theme_signature_ && theme_signature == last_theme_signature_)
            return;
        last_theme_signature_ = theme_signature;
        has_theme_signature_ = true;

        const bool is_light = t.isLightTheme();
        const auto logo_path = lfs::vis::getAssetPath(
            is_light ? "lichtfeld-splash-logo-dark.png" : "lichtfeld-splash-logo.png");
        auto* logo = document_->GetElementById("logo");
        if (logo) {
            logo->SetAttribute("src", logo_path.string());
            auto [w, h, c] = lfs::core::get_image_info(logo_path);
            if (w > 0 && h > 0) {
                logo->SetProperty("width", std::format("{:.0f}dp", w * 1.3f));
                logo->SetProperty("height", std::format("{:.0f}dp", h * 1.3f));
            }
        }

        const auto core11_path = lfs::vis::getAssetPath(
            is_light ? "core11-logo-dark.png" : "core11-logo.png");
        auto* core11 = document_->GetElementById("core11-logo");
        if (core11) {
            core11->SetAttribute("src", core11_path.string());
            auto [w, h, c] = lfs::core::get_image_info(core11_path);
            if (w > 0 && h > 0) {
                core11->SetProperty("width", std::format("{:.0f}dp", w * 0.5f));
                core11->SetProperty("height", std::format("{:.0f}dp", h * 0.5f));
            }
        }

        auto base_rcss = rml_theme::loadBaseRCSS("rmlui/startup.rcss");
        rml_theme::applyTheme(document_, base_rcss, rml_theme::generateAllThemeMedia([this](const auto& th) { return generateThemeRCSS(th); }));
    }

    bool StartupOverlay::forwardInput(const PanelInputState& input, float overlay_x,
                                      float overlay_y, float overlay_w, float overlay_h) {
        assert(rml_context_);
        if (rml_manager_) {
            rml_manager_->trackContextFrame(rml_context_,
                                            static_cast<int>(overlay_x - input.screen_x),
                                            static_cast<int>(overlay_y - input.screen_y));
        }

        const float local_x = input.mouse_x - overlay_x;
        const float local_y = input.mouse_y - overlay_y;

        const bool hovered = local_x >= 0 && local_y >= 0 &&
                             local_x < overlay_w && local_y < overlay_h;

        if (hovered) {
            rml_context_->ProcessMouseMove(static_cast<int>(local_x),
                                           static_cast<int>(local_y), 0);

            if (input.mouse_clicked[0])
                rml_context_->ProcessMouseButtonDown(0, 0);
            if (input.mouse_released[0])
                rml_context_->ProcessMouseButtonUp(0, 0);

            if (input.mouse_wheel != 0.0f)
                rml_context_->ProcessMouseWheel(Rml::Vector2f(0.0f, -input.mouse_wheel), 0);
        }

        bool escape_consumed = false;
        if (!input.viewport_keyboard_focus &&
            rml_input::hasFocusedKeyboardTarget(rml_context_->GetFocusElement())) {
            const int mods = sdlModsToRml(input.key_ctrl, input.key_shift,
                                          input.key_alt, input.key_super);
            for (int sc : input.keys_pressed) {
                if (sc == SDL_SCANCODE_ESCAPE && rml_input::cancelFocusedElement(*rml_context_)) {
                    escape_consumed = true;
                    continue;
                }

                const auto rml_key = sdlScancodeToRml(static_cast<SDL_Scancode>(sc));
                if (rml_key != Rml::Input::KI_UNKNOWN)
                    rml_context_->ProcessKeyDown(rml_key, mods);
            }

            for (int sc : input.keys_released) {
                if (escape_consumed && sc == SDL_SCANCODE_ESCAPE)
                    continue;

                const auto rml_key = sdlScancodeToRml(static_cast<SDL_Scancode>(sc));
                if (rml_key != Rml::Input::KI_UNKNOWN)
                    rml_context_->ProcessKeyUp(rml_key, mods);
            }
        }

        return escape_consumed;
    }

    void StartupOverlay::render(const ViewportLayout& viewport, bool drag_hovering) {
        if (!visible_)
            return;

        static constexpr float MIN_VIEWPORT_SIZE = 100.0f;
        if (viewport.size.x < MIN_VIEWPORT_SIZE || viewport.size.y < MIN_VIEWPORT_SIZE)
            return;

        if (!rml_context_ || !document_)
            return;

        auto& focus = guiFocusState();
        focus.want_capture_mouse = true;
        focus.want_capture_keyboard = true;

        bool escape_consumed = false;
        const bool defer_fbo_update = rml_manager_->shouldDeferFboUpdate(fbo_);
        if (defer_fbo_update && input_) {
            escape_consumed = forwardInput(*input_, viewport.pos.x, viewport.pos.y,
                                           viewport.size.x, viewport.size.y);
        }

        if (!defer_fbo_update) {
            updateTheme();
            updateLocalizedText();

            const int ctx_w = static_cast<int>(viewport.size.x);
            const int ctx_h = static_cast<int>(viewport.size.y);

            rml_context_->SetDimensions(Rml::Vector2i(ctx_w, ctx_h));
            document_->SetProperty("width", std::format("{}px", ctx_w));
            document_->SetProperty("height", std::format("{}px", ctx_h));
            rml_context_->Update();

            fbo_.ensure(ctx_w, ctx_h);
            if (!fbo_.valid())
                return;

            if (input_) {
                escape_consumed = forwardInput(*input_, viewport.pos.x, viewport.pos.y,
                                               viewport.size.x, viewport.size.y);
            }

            auto* render = rml_manager_->getRenderInterface();
            assert(render);
            render->SetViewport(ctx_w, ctx_h);

            GLint prev_fbo = 0;
            fbo_.bind(&prev_fbo);
            render->SetTargetFramebuffer(fbo_.fbo());

            render->BeginFrame();
            rml_context_->Render();
            render->EndFrame();

            render->SetTargetFramebuffer(0);
            fbo_.unbind(prev_fbo);
        }

        if (fbo_.valid()) {
            auto* main_viewport = ImGui::GetMainViewport();
            ImGui::SetNextWindowPos(ImVec2(viewport.pos.x, viewport.pos.y));
            ImGui::SetNextWindowSize(ImVec2(viewport.size.x, viewport.size.y));
            if (main_viewport)
                ImGui::SetNextWindowViewport(main_viewport->ID);
            ImGui::SetNextWindowBgAlpha(0.0f);

            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
            const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration |
                                           ImGuiWindowFlags_NoDocking |
                                           ImGuiWindowFlags_NoMove |
                                           ImGuiWindowFlags_NoSavedSettings |
                                           ImGuiWindowFlags_NoScrollbar |
                                           ImGuiWindowFlags_NoScrollWithMouse |
                                           ImGuiWindowFlags_NoNav |
                                           ImGuiWindowFlags_NoNavFocus;
            if (ImGui::Begin("##StartupOverlayComposite", nullptr, flags)) {
                ImGui::BringWindowToDisplayFront(ImGui::GetCurrentWindow());
                // Keep the splash in ImGui's item stack so hover tests block viewport tools.
                fbo_.blitAsImage(viewport.size.x, viewport.size.y);
            }
            ImGui::End();
            ImGui::PopStyleVar(3);
        }

        ++shown_frames_;

        auto* lang_el = document_ ? document_->GetElementById("lang-select") : nullptr;
        bool rml_select_open = false;
        if (lang_el) {
            auto* sel = dynamic_cast<Rml::ElementFormControlSelect*>(lang_el);
            if (sel)
                rml_select_open = sel->IsSelectBoxVisible();
        }

        if (shown_frames_ > 2 && !rml_select_open && !drag_hovering && input_) {
            const bool mouse_clicked =
                input_->mouse_clicked[0] || input_->mouse_clicked[1] || input_->mouse_clicked[2];
            const bool key_action = (!escape_consumed &&
                                     hasKey(input_->keys_pressed, SDL_SCANCODE_ESCAPE)) ||
                                    hasKey(input_->keys_pressed, SDL_SCANCODE_SPACE) ||
                                    hasKey(input_->keys_pressed, SDL_SCANCODE_RETURN) ||
                                    hasKey(input_->keys_pressed, SDL_SCANCODE_KP_ENTER);

            if (key_action) {
                LOG_DEBUG("StartupOverlay: dismissed by key action");
                visible_ = false;
            } else if (mouse_clicked) {
                auto* overlay_box = document_->GetElementById("overlay-box");
                bool inside = false;
                if (overlay_box) {
                    const float mx = input_->mouse_x - viewport.pos.x;
                    const float my = input_->mouse_y - viewport.pos.y;
                    auto abs_offset = overlay_box->GetAbsoluteOffset(Rml::BoxArea::Border);
                    float box_w = overlay_box->GetOffsetWidth();
                    float box_h = overlay_box->GetOffsetHeight();
                    inside = mx >= abs_offset.x && mx < abs_offset.x + box_w &&
                             my >= abs_offset.y && my < abs_offset.y + box_h;
                    if (!inside)
                        LOG_DEBUG("StartupOverlay: dismissed by click outside box "
                                  "(mouse={:.0f},{:.0f} box={:.0f},{:.0f} {:.0f}x{:.0f})",
                                  mx, my, abs_offset.x, abs_offset.y, box_w, box_h);
                } else {
                    LOG_DEBUG("StartupOverlay: dismissed - overlay-box element not found");
                }
                if (!inside)
                    visible_ = false;
            }
        }
    }

} // namespace lfs::vis::gui
