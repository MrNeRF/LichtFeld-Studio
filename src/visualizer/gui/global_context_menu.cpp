/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/global_context_menu.hpp"
#include "core/logger.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/panel_layout.hpp"
#include "gui/rmlui/rml_document_utils.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "internal/resource_paths.hpp"
#include "theme/theme.hpp"

#include "gui/rmlui/sdl_rml_key_mapping.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/Input.h>
#include <algorithm>
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
            doc_ = rml_documents::loadDocument(ctx_, rml_path);
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

    void GlobalContextMenu::reloadResources() {
        if (!ctx_)
            return;

        hide();
        open_ = false;
        pending_open_ = false;
        focus_first_item_ = false;
        callback_ = {};

        if (doc_) {
            ctx_->UnloadDocument(doc_);
            ctx_->Update();
        }

        doc_ = nullptr;
        el_backdrop_ = nullptr;
        el_ctx_menu_ = nullptr;
        base_rcss_.clear();
        has_theme_signature_ = false;
        width_ = 0;
        height_ = 0;

        try {
            const auto rml_path = lfs::vis::getAssetPath("rmlui/global_context_menu.rml");
            doc_ = rml_documents::loadDocument(ctx_, rml_path);
            if (!doc_) {
                LOG_ERROR("GlobalContextMenu: failed to reload global_context_menu.rml");
                return;
            }
            doc_->Show();

            el_backdrop_ = doc_->GetElementById("backdrop");
            el_ctx_menu_ = doc_->GetElementById("ctx-menu");

            if (!el_backdrop_ || !el_ctx_menu_) {
                LOG_ERROR("GlobalContextMenu: missing DOM elements after reload");
                return;
            }

            el_backdrop_->AddEventListener(Rml::EventId::Click, &listener_);
            el_ctx_menu_->AddEventListener(Rml::EventId::Click, &listener_);
        } catch (const std::exception& e) {
            LOG_ERROR("GlobalContextMenu: resource not found during reload: {}", e.what());
            return;
        }

        menu_model_.DirtyVariable("items");
        syncTheme();
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

        rml_theme::applyTheme(doc_, base_rcss_, rml_theme::loadBaseRCSS("rmlui/global_context_menu.theme.rcss"));
    }

    void GlobalContextMenu::request(std::vector<ContextMenuItem> items, float screen_x, float screen_y,
                                    ActionCallback callback) {
        pending_items_ = std::move(items);
        callback_ = std::move(callback);
        pending_x_ = screen_x;
        pending_y_ = screen_y;
        pending_open_ = true;
        focus_first_item_ = true;
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
        focus_first_item_ = false;
        callback_ = {};
        el_ctx_menu_->SetClass("visible", false);
        el_backdrop_->SetProperty("display", "none");
    }

    void GlobalContextMenu::focusFirstItem() {
        if (!el_ctx_menu_)
            return;

        Rml::ElementList items;
        el_ctx_menu_->GetElementsByClassName(items, "context-menu-item");
        for (auto* item : items) {
            if (item && item->Focus())
                break;
        }
    }

    void GlobalContextMenu::processInput(const PanelInputState& input) {
        if (!open_ || !ctx_ || !doc_ || !el_backdrop_ || !el_ctx_menu_)
            return;
        if (mgr_)
            mgr_->trackContextFrame(ctx_, 0, 0);

        const float mx = input.mouse_x - input.screen_x;
        const float my = input.mouse_y - input.screen_y;

        const int mods = sdlModsToRml(input.key_ctrl, input.key_shift,
                                      input.key_alt, input.key_super);

        ctx_->ProcessMouseMove(static_cast<int>(mx), static_cast<int>(my), mods);

        if (input.mouse_clicked[0])
            ctx_->ProcessMouseButtonDown(0, mods);
        if (input.mouse_released[0])
            ctx_->ProcessMouseButtonUp(0, mods);

        if (input.mouse_clicked[1]) {
            hide();
            return;
        }

        auto& focus = guiFocusState();
        focus.want_capture_mouse = true;
        focus.want_capture_keyboard = true;

        for (const int sc : input.keys_pressed) {
            if (sc == SDL_SCANCODE_ESCAPE) {
                hide();
                return;
            }
            const auto rml_key = sdlScancodeToRml(static_cast<SDL_Scancode>(sc));
            if (rml_key != Rml::Input::KI_UNKNOWN)
                ctx_->ProcessKeyDown(rml_key, mods);
        }
        for (const int sc : input.keys_released) {
            const auto rml_key = sdlScancodeToRml(static_cast<SDL_Scancode>(sc));
            if (rml_key != Rml::Input::KI_UNKNOWN)
                ctx_->ProcessKeyUp(rml_key, mods);
        }

        if (input.mouse_clicked[0] || input.mouse_clicked[1])
            focus_first_item_ = false;
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
                const float dp = std::max(mgr_ ? mgr_->getDpRatio() : 1.0f, 1.0f);
                el_ctx_menu_->SetProperty("left", std::format("{:.0f}dp", (pending_x_ - screen_x) / dp));
                el_ctx_menu_->SetProperty("top", std::format("{:.0f}dp", (pending_y_ - screen_y) / dp));
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

        if (!mgr_ || !mgr_->getVulkanRenderInterface())
            return;

        if (w != width_ || h != height_) {
            width_ = w;
            height_ = h;
            ctx_->SetDimensions(Rml::Vector2i(w, h));
        }

        ctx_->Update();
        if (focus_first_item_) {
            focusFirstItem();
            focus_first_item_ = false;
            ctx_->Update();
        }

        mgr_->queueVulkanContext(ctx_, 0.0f, 0.0f, true);
    }

    void GlobalContextMenu::releaseRendererResources() {
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
            auto callback = owner->callback_;
            owner->hide();
            if (callback) {
                callback(action);
            } else {
                owner->result_ = action;
            }
        }
    }

} // namespace lfs::vis::gui
