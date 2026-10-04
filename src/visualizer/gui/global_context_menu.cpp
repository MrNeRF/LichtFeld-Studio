/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/global_context_menu.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/logger.hpp"
#include "gui/context_menu_placement.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/gui_input.hpp"
#include "gui/rmlui/rml_document_utils.hpp"
#include "gui/rmlui/rml_input_utils.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "internal/resource_paths.hpp"
#include "theme/theme.hpp"

#include "gui/rmlui/sdl_rml_key_mapping.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/Elements/ElementFormControlInput.h>
#include <RmlUi/Core/Input.h>
#include <algorithm>
#include <cassert>
#include <cctype>
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
        if (mgr_ && mgr_->isInitialized())
            mgr_->releaseCachedContext(direct_cache_);
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
            handle.RegisterMember("icon", &ContextMenuItem::icon);
            handle.RegisterMember("shortcut", &ContextMenuItem::shortcut);
        }
        ctor.RegisterArray<std::vector<ContextMenuItem>>();
        ctor.Bind("items", &items_);
        ctor.Bind("searchable", &searchable_);
        search_label_ = LOC("scene.search");
        ctor.Bind("search_label", &search_label_);
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
            if (auto* search = doc_->GetElementById("ctx-search"))
                search->AddEventListener("input", &listener_);
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
        last_menu_left_dp_ = -1.0f;
        last_menu_top_dp_ = -1.0f;
        last_menu_min_width_dp_ = -1.0f;
        last_menu_max_width_dp_ = -1.0f;
        render_needed_ = true;
        last_mouse_valid_ = false;
        if (mgr_)
            mgr_->releaseCachedContext(direct_cache_);

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
            if (auto* search = doc_->GetElementById("ctx-search"))
                search->AddEventListener("input", &listener_);
        } catch (const std::exception& e) {
            LOG_ERROR("GlobalContextMenu: resource not found during reload: {}", e.what());
            return;
        }

        menu_model_.DirtyVariable("items");
        syncTheme();
    }

    void GlobalContextMenu::preload() {
        initContext();
        syncTheme();
    }

    bool GlobalContextMenu::syncTheme() {
        if (!doc_)
            return false;

        const std::size_t theme_signature = rml_theme::currentThemeSignature();
        if (has_theme_signature_ && theme_signature == last_theme_signature_)
            return false;
        last_theme_signature_ = theme_signature;
        has_theme_signature_ = true;

        if (base_rcss_.empty())
            base_rcss_ = rml_theme::loadBaseRCSS("rmlui/global_context_menu.rcss");

        rml_theme::applyTheme(doc_, base_rcss_, rml_theme::loadBaseRCSS("rmlui/global_context_menu.theme.rcss"));
        render_needed_ = true;
        return true;
    }

    void GlobalContextMenu::request(std::vector<ContextMenuItem> items, float screen_x,
                                    float screen_y, ActionCallback callback,
                                    const bool searchable) {
        pending_items_ = std::move(items);
        searchable_ = searchable;
        callback_ = std::move(callback);
        pending_x_ = screen_x;
        pending_y_ = screen_y;
        pending_open_ = true;
        initContext();
        if (ctx_ && el_ctx_menu_ && el_backdrop_) {
            all_items_ = pending_items_;
            items_ = all_items_;
            menu_model_.DirtyVariable("items");
            menu_model_.DirtyVariable("searchable");
            if (auto* search = dynamic_cast<Rml::ElementFormControlInput*>(
                    doc_->GetElementById("ctx-search")))
                search->SetValue("");
            el_ctx_menu_->SetClass("visible", true);
            el_backdrop_->SetProperty("display", "block");
            open_ = true;
            ctx_->Update();
            mgr_->activateInput(ctx_, [this](const PanelInputState& event) { processInput(event); });
            focusFirstItem();
        }
        focus_first_item_ = true;
        render_needed_ = true;
        last_mouse_valid_ = false;
        last_menu_left_dp_ = -1.0f;
        last_menu_top_dp_ = -1.0f;
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
        pending_open_ = false;
        mgr_->deactivateInput(ctx_);
        focus_first_item_ = false;
        callback_ = {};
        el_ctx_menu_->SetClass("visible", false);
        el_backdrop_->SetProperty("display", "none");
        render_needed_ = true;
        last_mouse_valid_ = false;
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

        if (mgr_ && ctx_ && mgr_->routeInput(ctx_, input, [this](const PanelInputState& event) { processInput(event); }, true))
            return;
        const float mx = input.mouse_x - input.screen_x;
        const float my = input.mouse_y - input.screen_y;

        const int mods = sdlModsToRml(input.key_ctrl, input.key_shift,
                                      input.key_alt, input.key_super);

        const int rml_mx = static_cast<int>(mx);
        const int rml_my = static_cast<int>(my);
        if (!last_mouse_valid_ || rml_mx != last_mouse_x_ || rml_my != last_mouse_y_) {
            ctx_->ProcessMouseMove(rml_mx, rml_my, mods);
            last_mouse_valid_ = true;
            last_mouse_x_ = rml_mx;
            last_mouse_y_ = rml_my;
            render_needed_ = true;
        }

        if (input.mouse_clicked[0]) {
            ctx_->ProcessMouseButtonDown(0, mods);
            render_needed_ = true;
        }
        if (input.mouse_released[0]) {
            ctx_->ProcessMouseButtonUp(0, mods);
            render_needed_ = true;
        }

        if (input.mouse_wheel != 0.0f || input.mouse_wheel_x != 0.0f) {
            ctx_->ProcessMouseWheel(
                Rml::Vector2f(-input.mouse_wheel_x, -input.mouse_wheel), mods);
            render_needed_ = true;
        }

        if (input.mouse_clicked[1]) {
            hide();
            return;
        }

        auto& focus = guiFocusState();
        focus.want_capture_mouse = true;
        focus.want_capture_keyboard = true;

        for (const auto& event : input.input_events) {
            if (event.kind == FrameInputEventKind::KeyDown && event.scancode == SDL_SCANCODE_ESCAPE) {
                hide();
                return;
            }
            render_needed_ |= rml_input::processKeyboardEvent(*ctx_, event);
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
                el_ctx_menu_->SetClass("has-shortcuts", std::ranges::any_of(items_, [](const auto& item) {
                                           return !item.shortcut.empty();
                                       }));
                const float dp = std::max(mgr_ ? mgr_->getDpRatio() : 1.0f, 1.0f);
                // Keep the initial layout near the pointer. Once RmlUi has
                // measured the menu, the final position is clamped to the
                // viewport using the actual menu bounds.
                el_ctx_menu_->SetProperty("left", std::format("{:.0f}dp", (pending_x_ - screen_x) / dp));
                el_ctx_menu_->SetProperty("top", std::format("{:.0f}dp", (pending_y_ - screen_y) / dp));
                el_ctx_menu_->SetClass("visible", true);
                el_backdrop_->SetProperty("display", "block");
                open_ = true;
                render_needed_ = true;
            }
            pending_items_.clear();
        }

        if (!open_)
            return;

        const bool theme_changed = syncTheme();

        const int w = screen_w;
        const int h = screen_h;

        if (w <= 0 || h <= 0)
            return;

        if (!mgr_ || !mgr_->getUiRenderer())
            return;

        const float dp = std::max(mgr_->getDpRatio(), 1.0f);
        const float available_width_dp = std::max(1.0f, static_cast<float>(w) / dp - 8.0f);
        const float minimum_width_dp = std::min(150.0f, available_width_dp);
        if (minimum_width_dp != last_menu_min_width_dp_ ||
            available_width_dp != last_menu_max_width_dp_) {
            el_ctx_menu_->SetProperty("min-width", std::format("{:.2f}dp", minimum_width_dp));
            el_ctx_menu_->SetProperty("max-width", std::format("{:.2f}dp", available_width_dp));
            last_menu_min_width_dp_ = minimum_width_dp;
            last_menu_max_width_dp_ = available_width_dp;
            render_needed_ = true;
        }

        if (w != width_ || h != height_) {
            width_ = w;
            height_ = h;
            ctx_->SetDimensions(Rml::Vector2i(w, h));
            render_needed_ = true;
        }

        bool refresh_cache = render_needed_ || theme_changed || direct_cache_.texture == 0;
        if (refresh_cache)
            ctx_->Update();
        if (focus_first_item_) {
            focusFirstItem();
            focus_first_item_ = false;
            ctx_->Update();
            refresh_cache = true;
        }

        // Position only after layout, so flipping and clamping use the real
        // dimensions of the localized menu and its current font/theme.
        if (open_ && el_ctx_menu_) {
            const float menu_dp = std::max(mgr_->getDpRatio(), 1.0f);
            const auto menu_size = el_ctx_menu_->GetBox().GetSize(Rml::BoxArea::Border);
            const auto placement = placeContextMenu(
                pending_x_, pending_y_, screen_x, screen_y,
                static_cast<float>(w), static_cast<float>(h),
                menu_size.x, menu_size.y, 4.0f * menu_dp);
            const float left_dp = placement.left / menu_dp;
            const float top_dp = placement.top / menu_dp;
            if (left_dp != last_menu_left_dp_ || top_dp != last_menu_top_dp_) {
                el_ctx_menu_->SetProperty("left", std::format("{:.2f}dp", left_dp));
                el_ctx_menu_->SetProperty("top", std::format("{:.2f}dp", top_dp));
                last_menu_left_dp_ = left_dp;
                last_menu_top_dp_ = top_dp;
                ctx_->Update();
                refresh_cache = true;
            }
        }

        render_needed_ = false;
        mgr_->queueCachedContext({
            .context = ctx_,
            .cache = &direct_cache_,
            .cache_width = w,
            .cache_height = h,
            .offset_x = 0.0f,
            .offset_y = 0.0f,
            .draw_width = static_cast<float>(w),
            .draw_height = static_cast<float>(h),
            .refresh = refresh_cache,
            .foreground = true,
            .clip = {},
        });
    }

    void GlobalContextMenu::releaseRendererResources() {
        if (mgr_)
            mgr_->releaseCachedContext(direct_cache_);
    }

    void GlobalContextMenu::EventListener::ProcessEvent(Rml::Event& event) {
        assert(owner);
        auto* target = event.GetTargetElement();
        if (target && target->GetId() == "ctx-search") {
            if (const auto* input = dynamic_cast<Rml::ElementFormControlInput*>(target))
                owner->filterItems(input->GetValue());
            return;
        }
        while (target && target != owner->el_ctx_menu_ && target->GetId() != "backdrop" &&
               !target->HasAttribute("data-ctx-action"))
            target = target->GetParentNode();
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

    void GlobalContextMenu::filterItems(const std::string_view query) {
        const auto lowercase = [](std::string value) {
            std::ranges::transform(value, value.begin(), [](const unsigned char character) {
                return static_cast<char>(std::tolower(character));
            });
            return value;
        };
        const std::string needle = lowercase(std::string(query));
        if (needle.empty()) {
            items_ = all_items_;
        } else {
            items_.clear();
            const ContextMenuItem* category = nullptr;
            bool category_added = false;
            for (const auto& item : all_items_) {
                if (item.is_label) {
                    category = &item;
                    category_added = false;
                    continue;
                }
                const bool matches_label = lowercase(item.label).contains(needle);
                const bool matches_category =
                    category && lowercase(category->label).contains(needle);
                if (!matches_label && !matches_category)
                    continue;
                if (category && !category_added) {
                    items_.push_back(*category);
                    category_added = true;
                }
                items_.push_back(item);
            }
        }
        menu_model_.DirtyVariable("items");
        if (ctx_)
            ctx_->Update();
        render_needed_ = true;
    }

} // namespace lfs::vis::gui
