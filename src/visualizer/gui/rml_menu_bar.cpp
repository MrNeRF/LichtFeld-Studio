/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rml_menu_bar.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/services.hpp"
#include "gui/rmlui/rml_document_utils.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rml_tooltip.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "input/input_controller.hpp"
#include "internal/resource_paths.hpp"
#include "ipc/view_context.hpp"
#include "operator/operator_registry.hpp"
#include "python/python_runtime.hpp"
#include "rendering/dirty_flags.hpp"
#include "rendering/rendering_manager.hpp"
#include "rendering/rendering_types.hpp"
#include "window/window_manager.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Element.h>
#include <cassert>
#include <cmath>
#include <format>
#include <glm/glm.hpp>

namespace lfs::vis::gui {

    namespace {
        MenuDropdownLeafView makeLeafItemView(const MenuItemDesc& item) {
            MenuDropdownLeafView view;
            view.label = item.label;
            view.enabled = item.enabled;
            view.callback_index = item.callback_index;

            switch (item.type) {
            case MenuItemDesc::Type::Operator:
                view.action = "operator";
                view.operator_id = item.operator_id;
                break;
            case MenuItemDesc::Type::Toggle:
                view.action = "callback";
                view.has_shortcut = !item.shortcut.empty();
                view.shortcut = item.shortcut;
                view.show_checkmark = true;
                view.checkmark = item.selected ? "\u2713" : "";
                break;
            case MenuItemDesc::Type::ShortcutItem:
            case MenuItemDesc::Type::Item:
                view.action = "callback";
                view.has_shortcut = !item.shortcut.empty();
                view.shortcut = item.shortcut;
                break;
            case MenuItemDesc::Type::Separator:
            case MenuItemDesc::Type::SubMenuBegin:
            case MenuItemDesc::Type::SubMenuEnd:
                break;
            }

            return view;
        }

        MenuDropdownLeafView makeSubmenuLabelView(const std::string& label) {
            MenuDropdownLeafView view;
            view.label = label;
            view.enabled = false;
            view.is_label = true;
            return view;
        }

        std::vector<MenuDropdownLeafView> buildChildMenuItems(const std::vector<MenuItemDesc>& items,
                                                              std::size_t& pos,
                                                              bool& warned_nested_submenu) {
            std::vector<MenuDropdownLeafView> result;
            bool separator_before = false;

            while (pos < items.size()) {
                const auto& item = items[pos];
                switch (item.type) {
                case MenuItemDesc::Type::Separator:
                    separator_before = true;
                    ++pos;
                    break;
                case MenuItemDesc::Type::SubMenuBegin: {
                    if (!warned_nested_submenu) {
                        warned_nested_submenu = true;
                        LOG_WARN("RmlMenuBar: nested submenu depth > 1 is flattened in the retained Rml menu model.");
                    }
                    MenuDropdownLeafView label_view = makeSubmenuLabelView(item.label);
                    label_view.separator_before = separator_before;
                    separator_before = false;
                    result.push_back(std::move(label_view));
                    ++pos;

                    auto nested_items = buildChildMenuItems(items, pos, warned_nested_submenu);
                    for (auto& nested : nested_items)
                        result.push_back(std::move(nested));
                    break;
                }
                case MenuItemDesc::Type::SubMenuEnd:
                    ++pos;
                    return result;
                case MenuItemDesc::Type::Operator:
                case MenuItemDesc::Type::Toggle:
                case MenuItemDesc::Type::ShortcutItem:
                case MenuItemDesc::Type::Item: {
                    MenuDropdownLeafView view = makeLeafItemView(item);
                    view.separator_before = separator_before;
                    separator_before = false;
                    result.push_back(std::move(view));
                    ++pos;
                    break;
                }
                }
            }

            return result;
        }

        std::vector<MenuDropdownRootView> buildRootMenuItems(const std::vector<MenuItemDesc>& items) {
            std::vector<MenuDropdownRootView> result;
            std::size_t pos = 0;
            bool separator_before = false;
            bool warned_nested_submenu = false;

            while (pos < items.size()) {
                const auto& item = items[pos];
                switch (item.type) {
                case MenuItemDesc::Type::Separator:
                    separator_before = true;
                    ++pos;
                    break;
                case MenuItemDesc::Type::SubMenuEnd:
                    ++pos;
                    break;
                case MenuItemDesc::Type::SubMenuBegin: {
                    MenuDropdownRootView view;
                    view.index = static_cast<int>(result.size());
                    view.label = item.label;
                    view.has_children = true;
                    view.separator_before = separator_before;
                    separator_before = false;
                    ++pos;
                    view.children = buildChildMenuItems(items, pos, warned_nested_submenu);
                    result.push_back(std::move(view));
                    break;
                }
                case MenuItemDesc::Type::Operator:
                case MenuItemDesc::Type::Toggle:
                case MenuItemDesc::Type::ShortcutItem:
                case MenuItemDesc::Type::Item: {
                    MenuDropdownRootView view;
                    view.index = static_cast<int>(result.size());
                    const auto leaf = makeLeafItemView(item);
                    view.label = leaf.label;
                    view.action = leaf.action;
                    view.operator_id = leaf.operator_id;
                    view.shortcut = leaf.shortcut;
                    view.checkmark = leaf.checkmark;
                    view.enabled = leaf.enabled;
                    view.separator_before = separator_before;
                    view.has_shortcut = leaf.has_shortcut;
                    view.show_checkmark = leaf.show_checkmark;
                    view.callback_index = leaf.callback_index;
                    separator_before = false;
                    result.push_back(std::move(view));
                    ++pos;
                    break;
                }
                }
            }

            return result;
        }
    } // namespace

    float RmlMenuBar::barHeight() const {
        const float dp = rml_manager_ ? rml_manager_->getDpRatio() : 1.0f;
        return bar_height_ * dp;
    }

    void RmlMenuBar::init(RmlUIManager* mgr) {
        assert(mgr);
        rml_manager_ = mgr;

        rml_context_ = rml_manager_->createContext("menu_bar", 800, 30);
        if (!rml_context_) {
            LOG_ERROR("RmlMenuBar: failed to create RML context");
            return;
        }

        auto ctor = rml_context_->CreateDataModel("menu_bar");
        assert(ctor);

        if (auto handle = ctor.RegisterStruct<MenuLabelView>()) {
            handle.RegisterMember("index", &MenuLabelView::index);
            handle.RegisterMember("label", &MenuLabelView::label);
            handle.RegisterMember("active", &MenuLabelView::active);
        }
        if (auto handle = ctor.RegisterStruct<MenuDropdownLeafView>()) {
            handle.RegisterMember("label", &MenuDropdownLeafView::label);
            handle.RegisterMember("action", &MenuDropdownLeafView::action);
            handle.RegisterMember("operator_id", &MenuDropdownLeafView::operator_id);
            handle.RegisterMember("shortcut", &MenuDropdownLeafView::shortcut);
            handle.RegisterMember("checkmark", &MenuDropdownLeafView::checkmark);
            handle.RegisterMember("enabled", &MenuDropdownLeafView::enabled);
            handle.RegisterMember("separator_before", &MenuDropdownLeafView::separator_before);
            handle.RegisterMember("has_shortcut", &MenuDropdownLeafView::has_shortcut);
            handle.RegisterMember("show_checkmark", &MenuDropdownLeafView::show_checkmark);
            handle.RegisterMember("is_label", &MenuDropdownLeafView::is_label);
            handle.RegisterMember("callback_index", &MenuDropdownLeafView::callback_index);
        }
        ctor.RegisterArray<std::vector<MenuLabelView>>();
        ctor.RegisterArray<std::vector<MenuDropdownLeafView>>();
        if (auto handle = ctor.RegisterStruct<MenuDropdownRootView>()) {
            handle.RegisterMember("label", &MenuDropdownRootView::label);
            handle.RegisterMember("index", &MenuDropdownRootView::index);
            handle.RegisterMember("action", &MenuDropdownRootView::action);
            handle.RegisterMember("operator_id", &MenuDropdownRootView::operator_id);
            handle.RegisterMember("shortcut", &MenuDropdownRootView::shortcut);
            handle.RegisterMember("checkmark", &MenuDropdownRootView::checkmark);
            handle.RegisterMember("enabled", &MenuDropdownRootView::enabled);
            handle.RegisterMember("separator_before", &MenuDropdownRootView::separator_before);
            handle.RegisterMember("has_shortcut", &MenuDropdownRootView::has_shortcut);
            handle.RegisterMember("show_checkmark", &MenuDropdownRootView::show_checkmark);
            handle.RegisterMember("has_children", &MenuDropdownRootView::has_children);
            handle.RegisterMember("submenu_open", &MenuDropdownRootView::submenu_open);
            handle.RegisterMember("callback_index", &MenuDropdownRootView::callback_index);
            handle.RegisterMember("children", &MenuDropdownRootView::children);
        }
        ctor.RegisterArray<std::vector<MenuDropdownRootView>>();
        if (auto handle = ctor.RegisterStruct<MenuToolbarButtonView>()) {
            handle.RegisterMember("button_id", &MenuToolbarButtonView::button_id);
            handle.RegisterMember("action", &MenuToolbarButtonView::action);
            handle.RegisterMember("value", &MenuToolbarButtonView::value);
            handle.RegisterMember("icon_src", &MenuToolbarButtonView::icon_src);
            handle.RegisterMember("tooltip_key", &MenuToolbarButtonView::tooltip_key);
            handle.RegisterMember("tooltip_text", &MenuToolbarButtonView::tooltip_text);
            handle.RegisterMember("selected", &MenuToolbarButtonView::selected);
        }
        ctor.RegisterArray<std::vector<MenuToolbarButtonView>>();
        ctor.Bind("menu_labels", &menu_labels_);
        ctor.Bind("dropdown_items", &dropdown_items_);
        ctor.Bind("menu_camera_buttons", &camera_buttons_);
        ctor.Bind("menu_render_buttons", &render_buttons_);
        ctor.Bind("menu_projection_buttons", &projection_buttons_);
        menu_model_ = ctor.GetModelHandle();

        try {
            const auto rml_path = lfs::vis::getAssetPath("rmlui/menubar.rml");
            document_ = rml_documents::loadDocument(rml_context_, rml_path);
            if (!document_) {
                LOG_ERROR("RmlMenuBar: failed to load menubar.rml");
                return;
            }
            document_->Show();
        } catch (const std::exception& e) {
            LOG_ERROR("RmlMenuBar: resource not found: {}", e.what());
            return;
        }

        menu_items_ = document_->GetElementById("menu-items");
        dropdown_overlay_ = document_->GetElementById("dropdown-overlay");
        dropdown_container_ = document_->GetElementById("dropdown-container");
        dropdown_popup_ = document_->GetElementById("dropdown-popup");
        brand_logo_ = document_->GetElementById("brand-logo");
        menu_toolbar_ = document_->GetElementById("menu-toolbar");
        menu_window_controls_ = document_->GetElementById("menu-window-controls");
        menu_window_split_view_ = document_->GetElementById("menu-window-split-view");
        menu_window_toggle_ui_ = document_->GetElementById("menu-window-toggle-ui");
        menu_window_maximize_ = document_->GetElementById("menu-window-maximize");
        body_el_ = document_->GetElementById("body");

        render_needed_ = true;
        updateTheme();
    }

    void RmlMenuBar::shutdown() {
        menu_model_ = {};
        menu_labels_.clear();
        dropdown_items_.clear();
        camera_buttons_.clear();
        render_buttons_.clear();
        projection_buttons_.clear();
        open_menu_idname_.clear();
        if (rml_manager_)
            rml_manager_->releaseCachedVulkanContext(direct_cache_);
        if (rml_context_ && rml_manager_)
            rml_manager_->destroyContext("menu_bar");
        rml_context_ = nullptr;
        document_ = nullptr;
        menu_items_ = nullptr;
        dropdown_container_ = nullptr;
        dropdown_popup_ = nullptr;
        dropdown_overlay_ = nullptr;
        brand_logo_ = nullptr;
        menu_toolbar_ = nullptr;
        menu_window_controls_ = nullptr;
        menu_window_split_view_ = nullptr;
        menu_window_toggle_ui_ = nullptr;
        menu_window_maximize_ = nullptr;
        body_el_ = nullptr;
        last_window_split_view_ = false;
        last_ui_hidden_ = false;
        last_window_maximized_ = false;
        clearTitlebarDragRegion();
    }

    void RmlMenuBar::suspend() {
        wants_input_ = false;
        mouse_pos_valid_ = false;
        last_mouse_x_ = 0;
        last_mouse_y_ = 0;
        last_hovered_label_ = -1;
        last_toolbar_hovered_ = false;
        clearTitlebarDragRegion();
        tooltip_.setHover({}, nullptr);

        if (open_menu_index_ >= 0)
            closeDropdown();
    }

    void RmlMenuBar::reloadResources() {
        if (!rml_context_)
            return;

        if (open_menu_index_ >= 0)
            closeDropdown();

        if (document_) {
            rml_context_->UnloadDocument(document_);
            rml_context_->Update();
        }

        document_ = nullptr;
        menu_items_ = nullptr;
        dropdown_container_ = nullptr;
        dropdown_popup_ = nullptr;
        dropdown_overlay_ = nullptr;
        brand_logo_ = nullptr;
        menu_toolbar_ = nullptr;
        menu_window_controls_ = nullptr;
        menu_window_split_view_ = nullptr;
        menu_window_toggle_ui_ = nullptr;
        menu_window_maximize_ = nullptr;
        body_el_ = nullptr;
        tooltip_.setHover({}, nullptr);
        clearTitlebarDragRegion();
        base_rcss_.clear();
        has_theme_signature_ = false;
        wants_input_ = false;
        render_needed_ = true;
        mouse_pos_valid_ = false;
        last_hovered_label_ = -1;
        last_ctx_w_ = 0;
        last_ctx_h_ = 0;
        last_document_h_ = 0;
        if (rml_manager_)
            rml_manager_->releaseCachedVulkanContext(direct_cache_);

        try {
            const auto rml_path = lfs::vis::getAssetPath("rmlui/menubar.rml");
            document_ = rml_documents::loadDocument(rml_context_, rml_path);
            if (!document_) {
                LOG_ERROR("RmlMenuBar: failed to reload menubar.rml");
                return;
            }
            document_->Show();
        } catch (const std::exception& e) {
            LOG_ERROR("RmlMenuBar: resource not found during reload: {}", e.what());
            return;
        }

        menu_items_ = document_->GetElementById("menu-items");
        dropdown_overlay_ = document_->GetElementById("dropdown-overlay");
        dropdown_container_ = document_->GetElementById("dropdown-container");
        dropdown_popup_ = document_->GetElementById("dropdown-popup");
        brand_logo_ = document_->GetElementById("brand-logo");
        menu_toolbar_ = document_->GetElementById("menu-toolbar");
        menu_window_controls_ = document_->GetElementById("menu-window-controls");
        menu_window_split_view_ = document_->GetElementById("menu-window-split-view");
        menu_window_toggle_ui_ = document_->GetElementById("menu-window-toggle-ui");
        menu_window_maximize_ = document_->GetElementById("menu-window-maximize");
        body_el_ = document_->GetElementById("body");
        applied_toolbar_right_ = -1.0f;
        last_window_split_view_ = false;
        last_ui_hidden_ = false;
        last_window_maximized_ = false;

        rebuildLabels();
        menu_model_.DirtyVariable("dropdown_items");
        menu_model_.DirtyVariable("menu_camera_buttons");
        menu_model_.DirtyVariable("menu_render_buttons");
        menu_model_.DirtyVariable("menu_projection_buttons");
        updateTheme();
    }

    void RmlMenuBar::updateLabels(const std::vector<std::string>& labels,
                                  const std::vector<std::string>& idnames) {
        assert(labels.size() == idnames.size());
        if (labels == current_labels_ && idnames == current_idnames_)
            return;
        current_labels_ = labels;
        current_idnames_ = idnames;
        render_needed_ = true;

        if (open_menu_index_ >= static_cast<int>(current_idnames_.size())) {
            closeDropdown();
        }

        rebuildLabels();
    }

    void RmlMenuBar::rebuildLabels() {
        menu_labels_.clear();
        menu_labels_.reserve(current_labels_.size());
        for (size_t i = 0; i < current_labels_.size(); ++i) {
            menu_labels_.push_back(MenuLabelView{
                .index = static_cast<int>(i),
                .label = current_labels_[i],
                .active = static_cast<int>(i) == active_index_,
            });
        }
        menu_model_.DirtyVariable("menu_labels");
        render_needed_ = true;
    }

    void RmlMenuBar::syncActiveLabelState() {
        const int display_active = open_menu_index_ >= 0 ? open_menu_index_ : -1;
        if (display_active == active_index_)
            return;

        active_index_ = display_active;
        for (size_t i = 0; i < menu_labels_.size(); ++i)
            menu_labels_[i].active = static_cast<int>(i) == active_index_;
        menu_model_.DirtyVariable("menu_labels");
        render_needed_ = true;
    }

    void RmlMenuBar::processInput(const PanelInputState& input) {
        if (!menu_items_ || !document_)
            return;

        wants_input_ = false;
        if (rml_manager_)
            rml_manager_->trackContextFrame(rml_context_, 0, 0);

        const float mx = input.mouse_x - input.screen_x;
        const float my = input.mouse_y - input.screen_y;
        const bool is_open = open_menu_index_ >= 0;
        const int ctx_w = input.screen_w;
        const int ctx_h = (is_open || tooltip_.hasActiveState())
                              ? input.screen_h
                              : static_cast<int>(bar_height_ * rml_manager_->getDpRatio());
        const int rml_mx = static_cast<int>(mx);
        const int rml_my = static_cast<int>(my);
        const bool was_in_context = mouse_pos_valid_ &&
                                    last_mouse_x_ >= 0 && last_mouse_x_ < ctx_w &&
                                    last_mouse_y_ >= 0 && last_mouse_y_ < ctx_h;
        const bool is_in_context = rml_mx >= 0 && rml_mx < ctx_w &&
                                   rml_my >= 0 && rml_my < ctx_h;
        const bool mouse_moved =
            !mouse_pos_valid_ || rml_mx != last_mouse_x_ || rml_my != last_mouse_y_;
        const bool pointer_event =
            input.mouse_clicked[0] || input.mouse_released[0] ||
            input.mouse_clicked[1] || input.mouse_released[1] ||
            input.mouse_clicked[2] || input.mouse_released[2] ||
            input.mouse_wheel != 0.0f;
        const bool pointer_down =
            input.mouse_down[0] || input.mouse_down[1] || input.mouse_down[2];
        const bool context_size_unchanged = ctx_w == last_ctx_w_ && ctx_h == last_ctx_h_;
        if (mouse_pos_valid_ && !mouse_moved && !pointer_event && !pointer_down &&
            context_size_unchanged && !render_needed_) {
            wants_input_ = is_open || last_hovered_label_ >= 0 || last_toolbar_hovered_;
            return;
        }
        if (mouse_moved && (is_open || was_in_context || is_in_context)) {
            mouse_pos_valid_ = true;
            last_mouse_x_ = rml_mx;
            last_mouse_y_ = rml_my;
            render_needed_ = true;
            rml_context_->ProcessMouseMove(rml_mx, rml_my, 0);
        }

        const int count = menu_items_->GetNumChildren();
        int hovered_label = -1;
        for (int i = 0; i < count; ++i) {
            auto* child = menu_items_->GetChild(i);
            const auto box = child->GetAbsoluteOffset(Rml::BoxArea::Border);
            const auto size = child->GetBox().GetSize(Rml::BoxArea::Border);
            if (mx >= box.x && mx < box.x + size.x && my >= box.y && my < box.y + size.y) {
                hovered_label = i;
                break;
            }
        }
        last_hovered_label_ = hovered_label;

        Rml::Element* hovered_toolbar_btn =
            (!is_open && hovered_label < 0) ? toolbarButtonAtPoint(mx, my) : nullptr;
        last_toolbar_hovered_ = hovered_toolbar_btn != nullptr;
        if (hovered_toolbar_btn)
            tooltip_.setHover(resolveRmlTooltip(hovered_toolbar_btn), hovered_toolbar_btn);
        else
            tooltip_.setHover({}, nullptr);

        const float dp_ratio = rml_manager_ ? rml_manager_->getDpRatio() : 1.0f;
        const bool in_free_titlebar =
            !is_open &&
            hovered_label < 0 &&
            !hovered_toolbar_btn &&
            my >= 0.0f &&
            my < bar_height_ * dp_ratio;
        if (in_free_titlebar && (input.mouse_clicked[0] || input.mouse_down[0])) {
            wants_input_ = true;
            return;
        }

        if (is_open) {
            wants_input_ = true;

            if (hovered_label >= 0 && hovered_label != open_menu_index_) {
                openDropdown(hovered_label);
                return;
            }

            Rml::Element* hit_element = nullptr;
            if (hovered_label < 0 && dropdown_container_)
                hit_element = dropdownElementAtPoint(mx, my);
            setOpenSubmenu(submenuIndexForElement(hit_element));

            if (input.mouse_clicked[0]) {
                if (hovered_label >= 0 && hovered_label == open_menu_index_) {
                    closeDropdown();
                    return;
                }

                if (hovered_label < 0 && dropdown_container_) {
                    auto* hit = dropdown_container_;
                    {
                        Rml::Element* clicked = hit_element ? hit_element : dropdownElementAtPoint(mx, my);
                        const bool clicked_submenu = submenuIndexForElement(clicked) >= 0;
                        if (clicked) {
                            while (clicked && clicked != hit) {
                                if (clicked->HasAttribute("data-action")) {
                                    const std::string action = clicked->GetAttribute<Rml::String>("data-action", "");
                                    if (action == "operator") {
                                        const std::string op_id = clicked->GetAttribute<Rml::String>("data-operator-id", "");
                                        if (!op_id.empty() && !clicked->HasAttribute("disabled")) {
                                            op::operators().invoke(op_id);
                                        }
                                        closeDropdown();
                                        return;
                                    } else if (action == "callback") {
                                        const int cb_idx = clicked->GetAttribute<int>("data-callback-index", -1);
                                        if (cb_idx >= 0 && !clicked->HasAttribute("disabled")) {
                                            python::execute_menu_callback(open_menu_idname_, cb_idx);
                                        }
                                        closeDropdown();
                                        return;
                                    }
                                }
                                clicked = clicked->GetParentNode();
                            }
                        }
                        if (clicked_submenu)
                            return;
                    }

                    closeDropdown();
                    return;
                }
            }
        } else {
            if (hovered_toolbar_btn) {
                wants_input_ = true;
                if (input.mouse_clicked[0]) {
                    dispatchToolbarAction(
                        hovered_toolbar_btn->GetAttribute<Rml::String>("data-action", ""),
                        hovered_toolbar_btn->GetAttribute<Rml::String>("data-value", ""));
                    return;
                }
            } else if (hovered_label >= 0) {
                wants_input_ = true;
                if (input.mouse_clicked[0]) {
                    openDropdown(hovered_label);
                    return;
                }
            }
        }

        // Update active label highlighting
        syncActiveLabelState();
    }

    void RmlMenuBar::openDropdown(int index) {
        assert(index >= 0 && index < static_cast<int>(current_idnames_.size()));

        open_menu_index_ = index;
        open_submenu_index_ = -1;
        open_menu_idname_ = current_idnames_[index];
        clearTitlebarDragRegion();

        MenuDropdownContent content;
        content.menu_idname = current_idnames_[index];

        python::collect_menu_content(
            current_idnames_[index],
            [](const python::MenuItemInfo* info, void* ctx) {
                auto* c = static_cast<MenuDropdownContent*>(ctx);
                MenuItemDesc item;
                item.type = static_cast<MenuItemDesc::Type>(info->type);
                item.label = info->label ? info->label : "";
                item.operator_id = info->operator_id ? info->operator_id : "";
                item.shortcut = info->shortcut ? info->shortcut : "";
                item.enabled = info->enabled;
                item.selected = info->selected;
                item.callback_index = info->callback_index;
                c->items.push_back(std::move(item));
            },
            &content);

        dropdown_items_ = buildRootMenuItems(content.items);
        rebuildDropdownDOM();
        syncActiveLabelState();
        render_needed_ = true;
    }

    void RmlMenuBar::closeDropdown() {
        open_menu_index_ = -1;
        open_submenu_index_ = -1;
        open_menu_idname_.clear();
        dropdown_items_.clear();
        menu_model_.DirtyVariable("dropdown_items");

        if (dropdown_container_) {
            dropdown_container_->SetClass("visible", false);
        }
        if (dropdown_overlay_)
            dropdown_overlay_->SetClass("visible", false);

        syncActiveLabelState();
        render_needed_ = true;
    }

    void RmlMenuBar::setOpenSubmenu(const int index) {
        if (index == open_submenu_index_)
            return;

        open_submenu_index_ = index;
        bool changed = false;
        for (auto& item : dropdown_items_) {
            const bool open = item.has_children && item.index == open_submenu_index_;
            if (item.submenu_open != open) {
                item.submenu_open = open;
                changed = true;
            }
        }
        if (!changed)
            return;

        menu_model_.DirtyVariable("dropdown_items");
        render_needed_ = true;
    }

    Rml::Element* RmlMenuBar::dropdownElementAtPoint(const float x, const float y) const {
        if (!dropdown_container_)
            return nullptr;

        const auto contains = [x, y](Rml::Element* element) {
            const auto box = element->GetAbsoluteOffset(Rml::BoxArea::Border);
            const auto size = element->GetBox().GetSize(Rml::BoxArea::Border);
            return x >= box.x && x < box.x + size.x && y >= box.y && y < box.y + size.y;
        };

        const auto find_deepest = [&](const auto& self, Rml::Element* element) -> Rml::Element* {
            for (int i = element->GetNumChildren() - 1; i >= 0; --i) {
                if (auto* hit = self(self, element->GetChild(i)))
                    return hit;
            }
            return contains(element) ? element : nullptr;
        };

        return find_deepest(find_deepest, dropdown_container_);
    }

    int RmlMenuBar::submenuIndexForElement(Rml::Element* element) const {
        for (auto* el = element; el; el = el->GetParentNode()) {
            if (!el->HasAttribute("data-root-index"))
                continue;
            return el->GetAttribute<int>("data-root-index", -1);
        }
        return -1;
    }

    void RmlMenuBar::rebuildToolbarButtons() {
        std::vector<MenuToolbarButtonView> camera_buttons;
        std::vector<MenuToolbarButtonView> render_buttons;
        std::vector<MenuToolbarButtonView> projection_buttons;

        const auto make = [](std::string id, std::string action, std::string value,
                             std::string icon, std::string tooltip_key,
                             std::string tooltip_text, bool selected) {
            return MenuToolbarButtonView{
                .button_id = std::move(id),
                .action = std::move(action),
                .value = std::move(value),
                .icon_src = "../icon/" + std::move(icon) + ".png",
                .tooltip_key = std::move(tooltip_key),
                .tooltip_text = std::move(tooltip_text),
                .selected = selected,
            };
        };

        if (const auto* ic = lfs::vis::InputController::instance()) {
            using NavMode = lfs::vis::InputController::CameraNavigationMode;
            struct NavButtonSpec {
                NavMode mode;
                const char* icon;
                const char* tooltip;
            };
            static constexpr NavButtonSpec kNavButtons[] = {
                {NavMode::Orbit, "camera-orbit", "Orbit Camera"},
                {NavMode::Trackball, "world", "Free Orbit Camera"},
                {NavMode::FPV, "camera-fpv", "Fly Camera"},
                {NavMode::Drone, "drone", "Drone Camera"},
            };
            const auto mode = ic->cameraNavigationMode();
            for (const auto& spec : kNavButtons) {
                const std::string name = lfs::vis::InputController::cameraNavigationModeName(spec.mode);
                camera_buttons.push_back(make("menu-camera-" + name, "set_camera_navigation_mode", name,
                                              spec.icon, "", spec.tooltip, mode == spec.mode));
            }
        }

        if (const auto* rm = lfs::vis::services().renderingOrNull()) {
            const auto settings = rm->getSettings();

            std::string active_mode = "splats";
            if (settings.point_cloud_mode)
                active_mode = "points";
            else if (settings.show_rings)
                active_mode = "rings";
            else if (settings.show_center_markers)
                active_mode = "centers";

            render_buttons.push_back(make("menu-render-splats", "set_render_mode", "splats", "blob",
                                          "toolbar.splat_rendering", "Splat Rendering",
                                          active_mode == "splats"));
            render_buttons.push_back(make("menu-render-points", "set_render_mode", "points",
                                          "dots-diagonal", "toolbar.point_cloud", "Point Cloud",
                                          active_mode == "points"));
            render_buttons.push_back(make("menu-render-rings", "set_render_mode", "rings", "ring",
                                          "toolbar.gaussian_rings", "Gaussian Rings",
                                          active_mode == "rings"));
            render_buttons.push_back(make("menu-render-centers", "set_render_mode", "centers",
                                          "circle-dot", "toolbar.center_markers", "Center Markers",
                                          active_mode == "centers"));

            const bool ortho = settings.orthographic;
            projection_buttons.push_back(make("menu-projection", "toggle_projection", "",
                                              ortho ? "box" : "perspective",
                                              ortho ? "toolbar.orthographic" : "toolbar.perspective",
                                              ortho ? "Orthographic" : "Perspective", ortho));
            projection_buttons.push_back(make("menu-depth-view", "toggle_depth_view", "", "depth-map",
                                              "toolbar.depth_map", "Depth Map", settings.depth_view));

            bool view_snap = false;
            if (const auto* ic = lfs::vis::InputController::instance())
                view_snap = ic->cameraViewSnapEnabled();
            projection_buttons.push_back(make("menu-view-snap", "toggle_camera_view_snap", "", "check",
                                              "", "Snap Axis Views", view_snap));
        }

        if (render_buttons != render_buttons_) {
            render_buttons_ = std::move(render_buttons);
            menu_model_.DirtyVariable("menu_render_buttons");
            render_needed_ = true;
        }
        if (camera_buttons != camera_buttons_) {
            camera_buttons_ = std::move(camera_buttons);
            menu_model_.DirtyVariable("menu_camera_buttons");
            render_needed_ = true;
        }
        if (projection_buttons != projection_buttons_) {
            projection_buttons_ = std::move(projection_buttons);
            menu_model_.DirtyVariable("menu_projection_buttons");
            render_needed_ = true;
        }
    }

    void RmlMenuBar::dispatchToolbarAction(const std::string& action, const std::string& value) {
        auto* rm = lfs::vis::services().renderingOrNull();

        if (action == "set_render_mode") {
            if (!rm)
                return;
            auto settings = rm->getSettings();
            const bool enable_point_cloud = value == "points";
            const bool point_cloud_changed = settings.point_cloud_mode != enable_point_cloud;
            settings.point_cloud_mode = enable_point_cloud;
            settings.show_rings = value == "rings";
            settings.show_center_markers = value == "centers";
            rm->updateSettings(settings,
                               point_cloud_changed && enable_point_cloud
                                   ? lfs::vis::DirtyFlag::ALL
                                   : lfs::vis::DirtyFlag::SELECTION);
        } else if (action == "set_camera_navigation_mode") {
            auto* ic = lfs::vis::InputController::instance();
            if (!ic)
                return;
            if (const auto mode = lfs::vis::InputController::cameraNavigationModeFromName(value)) {
                ic->setCameraNavigationMode(*mode);
            }
        } else if (action == "toggle_projection") {
            if (!rm)
                return;
            const bool ortho = rm->getSettings().orthographic;
            float viewport_height = 0.0f;
            float distance_to_pivot = 0.0f;
            if (const auto view = lfs::vis::get_current_view_info(); view.has_value()) {
                viewport_height = static_cast<float>(view->height);
                const glm::vec3 eye(view->translation[0], view->translation[1], view->translation[2]);
                const glm::vec3 pivot(view->pivot[0], view->pivot[1], view->pivot[2]);
                distance_to_pivot = glm::length(pivot - eye);
            }
            rm->setOrthographic(!ortho, viewport_height, distance_to_pivot);
        } else if (action == "toggle_depth_view") {
            if (!rm)
                return;
            auto settings = rm->getSettings();
            settings.depth_view = !settings.depth_view;
            rm->updateSettings(settings, lfs::vis::DirtyFlag::ALL);
        } else if (action == "toggle_camera_view_snap") {
            if (auto* ic = lfs::vis::InputController::instance())
                ic->setCameraViewSnapEnabled(!ic->cameraViewSnapEnabled());
        } else if (action == "toggle_independent_split_view") {
            if (auto* ic = lfs::vis::InputController::instance())
                ic->toggleIndependentSplitView();
        } else if (action == "window_toggle_ui") {
            lfs::core::events::ui::ToggleUI{}.emit();
        } else if (action == "window_minimize") {
            if (auto* wm = lfs::vis::services().windowOrNull())
                wm->minimize();
        } else if (action == "window_toggle_maximize") {
            if (auto* wm = lfs::vis::services().windowOrNull())
                wm->toggleMaximized();
        } else if (action == "window_close") {
            if (auto* wm = lfs::vis::services().windowOrNull()) {
                wm->requestClose();
                wm->wakeEventLoop();
            }
        }

        render_needed_ = true;
    }

    void RmlMenuBar::setUiHidden(const bool hidden) {
        if (ui_hidden_ == hidden)
            return;
        ui_hidden_ = hidden;
        render_needed_ = true;
    }

    Rml::Element* RmlMenuBar::toolbarButtonAtPoint(const float x, const float y) const {
        const auto find_button = [x, y](Rml::Element* root) -> Rml::Element* {
            if (!root)
                return nullptr;
            for (int i = 0; i < root->GetNumChildren(); ++i) {
                auto* child = root->GetChild(i);
                if (!child->HasAttribute("data-action"))
                    continue;
                const auto box = child->GetAbsoluteOffset(Rml::BoxArea::Border);
                const auto size = child->GetBox().GetSize(Rml::BoxArea::Border);
                if (x >= box.x && x < box.x + size.x && y >= box.y && y < box.y + size.y)
                    return child;
            }
            return nullptr;
        };

        if (auto* button = find_button(menu_toolbar_))
            return button;
        return find_button(menu_window_controls_);
    }

    void RmlMenuBar::clearTitlebarDragRegion() {
        if (auto* wm = lfs::vis::services().windowOrNull())
            wm->clearTitlebarDragRegion();
    }

    void RmlMenuBar::updateTitlebarDragRegion(const int bar_height_px) {
        auto* wm = lfs::vis::services().windowOrNull();
        if (!wm)
            return;
        if (open_menu_index_ >= 0 || bar_height_px <= 0) {
            wm->clearTitlebarDragRegion();
            return;
        }

        const auto append_element = [](std::vector<lfs::vis::WindowManager::HitTestRect>& rects,
                                       Rml::Element* element) {
            if (!element)
                return;

            const auto offset = element->GetAbsoluteOffset(Rml::BoxArea::Border);
            const auto size = element->GetBox().GetSize(Rml::BoxArea::Border);
            const int x1 = static_cast<int>(std::floor(offset.x));
            const int y1 = static_cast<int>(std::floor(offset.y));
            const int x2 = static_cast<int>(std::ceil(offset.x + size.x));
            const int y2 = static_cast<int>(std::ceil(offset.y + size.y));
            if (x2 <= x1 || y2 <= y1)
                return;

            rects.push_back({
                .x = x1,
                .y = y1,
                .w = x2 - x1,
                .h = y2 - y1,
            });
        };

        std::vector<lfs::vis::WindowManager::HitTestRect> excluded_rects;
        excluded_rects.reserve(3);
        append_element(excluded_rects, menu_items_);
        append_element(excluded_rects, menu_toolbar_);
        append_element(excluded_rects, menu_window_controls_);
        wm->setTitlebarDragRegion(bar_height_px, std::move(excluded_rects));
    }

    void RmlMenuBar::rebuildDropdownDOM() {
        if (!dropdown_container_ || !dropdown_popup_ || !dropdown_overlay_ || !menu_items_)
            return;

        const int count = menu_items_->GetNumChildren();
        if (open_menu_index_ < 0 || open_menu_index_ >= count)
            return;

        auto* label_el = menu_items_->GetChild(open_menu_index_);
        const auto label_offset = label_el->GetAbsoluteOffset(Rml::BoxArea::Border);
        const auto label_size = label_el->GetBox().GetSize(Rml::BoxArea::Border);

        menu_model_.DirtyVariable("dropdown_items");
        dropdown_container_->SetProperty("left", "0px");
        dropdown_container_->SetProperty("top", "0px");
        dropdown_popup_->SetProperty("left", std::format("{}px", label_offset.x));
        dropdown_popup_->SetProperty("top", std::format("{}px", label_offset.y + label_size.y));
        dropdown_container_->SetClass("visible", true);
        dropdown_overlay_->SetClass("visible", true);
        render_needed_ = true;
    }

    bool RmlMenuBar::updateTheme() {
        if (!document_)
            return false;

        const std::size_t theme_signature = rml_theme::currentThemeSignature();
        if (has_theme_signature_ && theme_signature == last_theme_signature_)
            return false;
        last_theme_signature_ = theme_signature;
        has_theme_signature_ = true;

        if (base_rcss_.empty())
            base_rcss_ = rml_theme::loadBaseRCSS("rmlui/menubar.rcss");

        rml_theme::applyTheme(document_, base_rcss_, rml_theme::loadBaseRCSS("rmlui/menubar.theme.rcss"));

        if (brand_logo_) {
            const auto logo_path = lfs::vis::getAssetPath("lichtfeld-icon.png");
            brand_logo_->SetAttribute("src", rml_theme::pathToRmlImageSource(logo_path));
            brand_logo_->SetProperty("width", "20dp");
            brand_logo_->SetProperty("height", "20dp");
        }
        return true;
    }

    void RmlMenuBar::draw(int screen_w, int screen_h) {
        if (!rml_context_ || !document_)
            return;
        if (!rml_manager_ || !rml_manager_->getVulkanRenderInterface())
            return;
        const bool theme_changed = updateTheme();
        rebuildToolbarButtons();

        if (menu_window_split_view_) {
            const bool split_view = [&] {
                if (auto* rm = lfs::vis::services().renderingOrNull())
                    return rm->getSettings().split_view_mode == lfs::vis::SplitViewMode::IndependentDual;
                return false;
            }();
            if (split_view != last_window_split_view_) {
                menu_window_split_view_->SetClass("selected", split_view);
                menu_window_split_view_->SetAttribute(
                    "title", split_view ? "Exit Independent Split View" : "Independent Split View");
                last_window_split_view_ = split_view;
                render_needed_ = true;
            }
        }

        if (menu_window_toggle_ui_ && ui_hidden_ != last_ui_hidden_) {
            menu_window_toggle_ui_->SetClass("selected", ui_hidden_);
            menu_window_toggle_ui_->SetAttribute("title", ui_hidden_ ? "Show UI" : "Hide UI");
            last_ui_hidden_ = ui_hidden_;
            render_needed_ = true;
        }

        if (menu_window_maximize_) {
            const bool maximized = [&] {
                if (auto* wm = lfs::vis::services().windowOrNull())
                    return wm->isMaximized();
                return false;
            }();
            if (maximized != last_window_maximized_) {
                menu_window_maximize_->SetClass("maximized", maximized);
                menu_window_maximize_->SetAttribute("title", maximized ? "Restore Window" : "Maximize Window");
                last_window_maximized_ = maximized;
                render_needed_ = true;
            }
        }

        const float dp_ratio = rml_manager_->getDpRatio();
        const int bar_h = static_cast<int>(bar_height_ * dp_ratio);

        // Right-align the render/projection toolbar to the viewport edge, but
        // keep it clear of the window-control cluster when there is no dock panel.
        if (menu_toolbar_) {
            const float inset = 8.0f * dp_ratio;
            constexpr float kFallbackRightClusterReserveDp = 184.0f;
            float right_px = kFallbackRightClusterReserveDp * dp_ratio;
            if (menu_window_controls_) {
                const auto offset = menu_window_controls_->GetAbsoluteOffset(Rml::BoxArea::Border);
                if (offset.x > 0.0f)
                    right_px = static_cast<float>(screen_w) - offset.x + 4.0f * dp_ratio;
            }
            if (viewport_right_edge_ > 0.0f)
                right_px = std::max(right_px, static_cast<float>(screen_w) - viewport_right_edge_ + inset);
            if (std::abs(right_px - applied_toolbar_right_) > 0.5f) {
                menu_toolbar_->SetProperty("right", std::format("{:.1f}px", right_px));
                applied_toolbar_right_ = right_px;
                render_needed_ = true;
            }
        }

        int ctx_w = screen_w;
        // A closed menu bar only occupies the bar strip, but a dropdown or a
        // tooltip needs the full height so it is not clipped below the bar.
        int ctx_h = (open_menu_index_ >= 0 || tooltip_.hasActiveState()) ? screen_h : bar_h;

        bool tooltip_changed = false;
        if (tooltip_.hasActiveState()) {
            if (!body_el_)
                body_el_ = document_->GetElementById("body");
            tooltip_changed = tooltip_.apply(body_el_, last_mouse_x_, last_mouse_y_, ctx_w, ctx_h);
        }
        rml_manager_->setContextNeedsPassiveMouseMoveFrames(rml_context_, tooltip_.needsFrame());
        rml_manager_->setContextTooltipRevealDeadline(rml_context_, tooltip_.revealDeadline());

        const bool size_changed = (ctx_w != last_ctx_w_ || ctx_h != last_ctx_h_);
        const bool refresh_cache = render_needed_ || theme_changed || size_changed ||
                                   tooltip_changed || direct_cache_.texture == 0;

        if (refresh_cache) {
            rml_context_->SetDimensions(Rml::Vector2i(ctx_w, ctx_h));
            if (ctx_h != last_document_h_) {
                document_->SetProperty("height", std::format("{}px", ctx_h));
                last_document_h_ = ctx_h;
            }
            rml_context_->Update();
        }
        updateTitlebarDragRegion(bar_h);

        rml_manager_->queueCachedVulkanContext({
            .context = rml_context_,
            .cache = &direct_cache_,
            .cache_width = ctx_w,
            .cache_height = ctx_h,
            .offset_x = 0.0f,
            .offset_y = 0.0f,
            .draw_width = static_cast<float>(screen_w),
            .draw_height = static_cast<float>(ctx_h),
            .refresh = refresh_cache,
            .foreground = true,
            .clip_enabled = true,
            .clip = {
                .x1 = 0.0f,
                .y1 = 0.0f,
                .x2 = static_cast<float>(screen_w),
                .y2 = static_cast<float>(ctx_h),
            },
        });
        last_ctx_w_ = ctx_w;
        last_ctx_h_ = ctx_h;
        render_needed_ = false;
    }

} // namespace lfs::vis::gui
