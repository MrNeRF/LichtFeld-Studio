/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// clang-format off
#include <glad/glad.h>
// clang-format on

#include "gui/global_context_menu.hpp"
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

    GlobalContextMenu::GlobalContextMenu(RmlUIManager* mgr)
        : mgr_(mgr) {
        assert(mgr_);
        listener_.owner = this;
    }

    GlobalContextMenu::~GlobalContextMenu() {
        menu_model_ = {};
        items_.clear();
        pending_items_.clear();
        fbo_.destroy();
        if (ctx_ && mgr_)
            mgr_->destroyContext("global_context_menu");
    }

    void GlobalContextMenu::initContext() {
        if (ctx_)
            return;

        ctx_ = mgr_->createContext("global_context_menu", 800, 600);
        if (!ctx_) {
            LOG_ERROR("GlobalContextMenu: failed to create context");
            return;
        }

        ctx_->EnableMouseCursor(false);

        auto ctor = ctx_->CreateDataModel("global_context_menu");
        assert(ctor);

        if (auto handle = ctor.RegisterStruct<ContextMenuItem>()) {
            handle.RegisterMember("label", &ContextMenuItem::label);
            handle.RegisterMember("action", &ContextMenuItem::action);
            handle.RegisterMember("separator_before", &ContextMenuItem::separator_before);
            handle.RegisterMember("is_label", &ContextMenuItem::is_label);
            handle.RegisterMember("is_submenu_item", &ContextMenuItem::is_submenu_item);
            handle.RegisterMember("is_active", &ContextMenuItem::is_active);
        }
        ctor.RegisterArray<std::vector<ContextMenuItem>>();
        ctor.Bind("items", &items_);
        menu_model_ = ctor.GetModelHandle();

        try {
            const auto rml_path = lfs::vis::getAssetPath("rmlui/global_context_menu.rml");
            doc_ = ctx_->LoadDocument(rml_path.string());
            if (!doc_) {
                LOG_ERROR("GlobalContextMenu: failed to load global_context_menu.rml");
                return;
            }
            doc_->Show();

            el_backdrop_ = doc_->GetElementById("backdrop");
            el_ctx_menu_ = doc_->GetElementById("ctx-menu");

            if (!el_backdrop_ || !el_ctx_menu_) {
                LOG_ERROR("GlobalContextMenu: missing DOM elements");
                return;
            }

            el_backdrop_->AddEventListener(Rml::EventId::Click, &listener_);
            el_ctx_menu_->AddEventListener(Rml::EventId::Click, &listener_);
        } catch (const std::exception& e) {
            LOG_ERROR("GlobalContextMenu: resource not found: {}", e.what());
        }
    }

    std::string GlobalContextMenu::generateThemeRCSS(const lfs::vis::Theme& t) const {
        using rml_theme::colorToRml;
        using rml_theme::colorToRmlAlpha;
        const auto& p = t.palette;

        const auto surface = colorToRmlAlpha(p.surface, 0.95f);
        const auto border = colorToRmlAlpha(p.border, 0.4f);
        const auto text = colorToRml(p.text);
        const auto text_dim = colorToRml(p.text_dim);
        const int rounding = static_cast<int>(t.sizes.window_rounding);

        return std::format(
            ".context-menu {{ background-color: {}; border-color: {}; border-radius: {}dp; }}\n"
            ".context-menu-item {{ color: {}; }}\n"
            ".context-menu-label {{ color: {}; }}\n"
            ".context-menu-separator {{ background-color: {}; }}\n",
            surface, border, rounding,
            text, text_dim,
            colorToRmlAlpha(p.border, 0.5f));
    }

    void GlobalContextMenu::syncTheme() {
        if (!doc_)
            return;

        const std::size_t theme_signature = rml_theme::currentThemeSignature();
        if (has_theme_signature_ && theme_signature == last_theme_signature_)
            return;
        last_theme_signature_ = theme_signature;
        has_theme_signature_ = true;

        if (base_rcss_.empty())
            base_rcss_ = rml_theme::loadBaseRCSS("rmlui/global_context_menu.rcss");

        rml_theme::applyTheme(doc_, base_rcss_, rml_theme::generateAllThemeMedia([this](const auto& th) { return generateThemeRCSS(th); }));
    }

    void GlobalContextMenu::request(std::vector<ContextMenuItem> items, float screen_x, float screen_y) {
        pending_items_ = std::move(items);
        pending_x_ = screen_x;
        pending_y_ = screen_y;
        pending_open_ = true;
    }

    std::string GlobalContextMenu::pollResult() {
        std::string r;
        r.swap(result_);
        return r;
    }

    void GlobalContextMenu::hide() {
        if (!el_ctx_menu_ || !el_backdrop_)
            return;

        open_ = false;
        el_ctx_menu_->SetClass("visible", false);
        el_backdrop_->SetProperty("display", "none");
    }

    void GlobalContextMenu::processInput(const PanelInputState& input) {
        if (!open_ || !ctx_ || !doc_ || !el_backdrop_ || !el_ctx_menu_)
            return;

        const float mx = input.mouse_x - input.screen_x;
        const float my = input.mouse_y - input.screen_y;

        ctx_->ProcessMouseMove(static_cast<int>(mx), static_cast<int>(my), 0);

        if (input.mouse_clicked[0])
            ctx_->ProcessMouseButtonDown(0, 0);
        if (input.mouse_released[0])
            ctx_->ProcessMouseButtonUp(0, 0);

        if (input.mouse_clicked[1]) {
            hide();
            return;
        }

        guiFocusState().want_capture_mouse = true;

        if (hasKey(input.keys_pressed, SDL_SCANCODE_ESCAPE))
            hide();
    }

    void GlobalContextMenu::render(int screen_w, int screen_h,
                                   float screen_x, float screen_y) {
        if (!open_ && !pending_open_)
            return;

        if (!ctx_) {
            initContext();
            if (!ctx_)
                return;
        }

        if (pending_open_) {
            pending_open_ = false;

            if (el_ctx_menu_ && el_backdrop_) {
                items_ = pending_items_;
                menu_model_.DirtyVariable("items");
                el_ctx_menu_->SetProperty("left", std::format("{:.0f}dp", pending_x_ - screen_x));
                el_ctx_menu_->SetProperty("top", std::format("{:.0f}dp", pending_y_ - screen_y));
                el_ctx_menu_->SetClass("visible", true);
                el_backdrop_->SetProperty("display", "block");
                open_ = true;
            }
            pending_items_.clear();
        }

        if (!open_)
            return;

        syncTheme();

        const int w = screen_w;
        const int h = screen_h;

        if (w <= 0 || h <= 0)
            return;

        if (!mgr_->shouldDeferFboUpdate(fbo_)) {
            if (w != width_ || h != height_) {
                width_ = w;
                height_ = h;
                ctx_->SetDimensions(Rml::Vector2i(w, h));
            }

            ctx_->Update();

            fbo_.ensure(w, h);
            if (!fbo_.valid())
                return;

            auto* render_iface = mgr_->getRenderInterface();
            assert(render_iface);
            render_iface->SetViewport(w, h);

            GLint prev_fbo = 0;
            fbo_.bind(&prev_fbo);
            render_iface->SetTargetFramebuffer(fbo_.fbo());

            render_iface->BeginFrame();
            ctx_->Render();
            render_iface->EndFrame();

            render_iface->SetTargetFramebuffer(0);
            fbo_.unbind(prev_fbo);
        }

        if (fbo_.valid())
            fbo_.blitToScreen(0.0f, 0.0f, static_cast<float>(screen_w), static_cast<float>(screen_h),
                              screen_w, screen_h);
    }

    void GlobalContextMenu::destroyGLResources() {
        fbo_.destroy();
    }

    void GlobalContextMenu::EventListener::ProcessEvent(Rml::Event& event) {
        assert(owner);
        auto* target = event.GetTargetElement();
        if (!target)
            return;

        const auto& id = target->GetId();

        if (id == "backdrop") {
            owner->hide();
            return;
        }

        const auto action = target->GetAttribute<Rml::String>("data-ctx-action", "");
        if (!action.empty()) {
            owner->result_ = action;
            owner->hide();
        }
    }

} // namespace lfs::vis::gui
