/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rml_menu_bar.hpp"
#include "core/logger.hpp"
#include "gui/rmlui/rml_document_utils.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "internal/resource_paths.hpp"
#include "operator/operator_registry.hpp"
#include "python/python_runtime.hpp"
#include "theme/theme.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Element.h>
#include <cassert>
#include <format>

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
            handle.RegisterMember("action", &MenuDropdownRootView::action);
            handle.RegisterMember("operator_id", &MenuDropdownRootView::operator_id);
            handle.RegisterMember("shortcut", &MenuDropdownRootView::shortcut);
            handle.RegisterMember("checkmark", &MenuDropdownRootView::checkmark);
            handle.RegisterMember("enabled", &MenuDropdownRootView::enabled);
            handle.RegisterMember("separator_before", &MenuDropdownRootView::separator_before);
            handle.RegisterMember("has_shortcut", &MenuDropdownRootView::has_shortcut);
            handle.RegisterMember("show_checkmark", &MenuDropdownRootView::show_checkmark);
            handle.RegisterMember("has_children", &MenuDropdownRootView::has_children);
            handle.RegisterMember("callback_index", &MenuDropdownRootView::callback_index);
            handle.RegisterMember("children", &MenuDropdownRootView::children);
        }
        ctor.RegisterArray<std::vector<MenuDropdownRootView>>();
        ctor.Bind("menu_labels", &menu_labels_);
        ctor.Bind("dropdown_items", &dropdown_items_);
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

        render_needed_ = true;
        updateTheme();
    }

    void RmlMenuBar::shutdown() {
        menu_model_ = {};
        menu_labels_.clear();
        dropdown_items_.clear();
        open_menu_idname_.clear();
        if (rml_context_ && rml_manager_)
            rml_manager_->destroyContext("menu_bar");
        rml_context_ = nullptr;
        document_ = nullptr;
        menu_items_ = nullptr;
        dropdown_container_ = nullptr;
        dropdown_overlay_ = nullptr;
    }

    void RmlMenuBar::suspend() {
        wants_input_ = false;
        mouse_pos_valid_ = false;
        last_mouse_x_ = 0;
        last_mouse_y_ = 0;

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
        dropdown_overlay_ = nullptr;
        base_rcss_.clear();
        has_theme_signature_ = false;
        wants_input_ = false;
        render_needed_ = true;
        mouse_pos_valid_ = false;
        last_ctx_w_ = 0;
        last_ctx_h_ = 0;
        last_document_h_ = 0;

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

        rebuildLabels();
        menu_model_.DirtyVariable("dropdown_items");
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
        const int ctx_h = is_open
                              ? input.screen_h
                              : static_cast<int>(bar_height_ * rml_manager_->getDpRatio());
        const int rml_mx = static_cast<int>(mx);
        const int rml_my = static_cast<int>(my);
        const bool was_in_context = mouse_pos_valid_ &&
                                    last_mouse_x_ >= 0 && last_mouse_x_ < ctx_w &&
                                    last_mouse_y_ >= 0 && last_mouse_y_ < ctx_h;
        const bool is_in_context = rml_mx >= 0 && rml_mx < ctx_w &&
                                   rml_my >= 0 && rml_my < ctx_h;
        if ((!mouse_pos_valid_ || rml_mx != last_mouse_x_ || rml_my != last_mouse_y_) &&
            (is_open || was_in_context || is_in_context)) {
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

        if (is_open) {
            wants_input_ = true;

            if (hovered_label >= 0 && hovered_label != open_menu_index_) {
                openDropdown(hovered_label);
                return;
            }

            if (input.mouse_clicked[0]) {
                if (hovered_label >= 0 && hovered_label == open_menu_index_) {
                    closeDropdown();
                    return;
                }

                if (hovered_label < 0 && dropdown_container_) {
                    auto* hit = dropdown_container_;
                    {
                        Rml::Element* clicked = rml_context_->GetElementAtPoint(Rml::Vector2f(mx, my));
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
                    }

                    closeDropdown();
                    return;
                }
            }
        } else {
            if (hovered_label >= 0) {
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
        open_menu_idname_ = current_idnames_[index];

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

    void RmlMenuBar::rebuildDropdownDOM() {
        if (!dropdown_container_ || !dropdown_overlay_ || !menu_items_)
            return;

        const int count = menu_items_->GetNumChildren();
        if (open_menu_index_ < 0 || open_menu_index_ >= count)
            return;

        auto* label_el = menu_items_->GetChild(open_menu_index_);
        const auto label_offset = label_el->GetAbsoluteOffset(Rml::BoxArea::Border);
        const auto label_size = label_el->GetBox().GetSize(Rml::BoxArea::Border);

        menu_model_.DirtyVariable("dropdown_items");
        dropdown_container_->SetProperty("left", std::format("{}px", label_offset.x));
        dropdown_container_->SetProperty("top", std::format("{}px", label_offset.y + label_size.y));
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
        return true;
    }

    void RmlMenuBar::draw(int screen_w, int screen_h) {
        if (!rml_context_ || !document_)
            return;
        if (!rml_manager_ || !rml_manager_->getVulkanRenderInterface())
            return;
        const bool theme_changed = updateTheme();

        const float dp_ratio = rml_manager_->getDpRatio();
        const int bar_h = static_cast<int>(bar_height_ * dp_ratio);

        int ctx_w = screen_w;
        int ctx_h;

        if (open_menu_index_ >= 0) {
            ctx_h = screen_h;
        } else {
            ctx_h = bar_h;
        }

        const bool size_changed = (ctx_w != last_ctx_w_ || ctx_h != last_ctx_h_);
        const bool needs_render = render_needed_ || theme_changed || size_changed;
        if (!needs_render) {
            rml_manager_->queueVulkanContext(rml_context_, 0.0f, 0.0f, true,
                                             true, 0.0f, 0.0f,
                                             static_cast<float>(screen_w),
                                             static_cast<float>(ctx_h));
            return;
        }

        rml_context_->SetDimensions(Rml::Vector2i(ctx_w, ctx_h));
        if (ctx_h != last_document_h_) {
            document_->SetProperty("height", std::format("{}px", ctx_h));
            last_document_h_ = ctx_h;
        }
        rml_context_->Update();

        rml_manager_->queueVulkanContext(rml_context_, 0.0f, 0.0f, true,
                                         true, 0.0f, 0.0f,
                                         static_cast<float>(screen_w),
                                         static_cast<float>(ctx_h));
        last_ctx_w_ = ctx_w;
        last_ctx_h_ = ctx_h;
        render_needed_ = false;
    }

} // namespace lfs::vis::gui
