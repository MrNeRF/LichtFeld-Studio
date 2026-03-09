/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/localization_manager.hpp"
#include "core/logger.hpp"
#include "gui/rml_menu_bar.hpp"
#include "operator/operator_registry.hpp"
#include "py_ui.hpp"

#include <algorithm>
#include <mutex>
#include <unordered_set>
#include <imgui.h>

namespace lfs::python {

    PyMenuRegistry& PyMenuRegistry::instance() {
        static PyMenuRegistry registry;
        return registry;
    }

    static const char* menu_location_to_string(MenuLocation loc) {
        switch (loc) {
        case MenuLocation::File: return "FILE";
        case MenuLocation::Edit: return "EDIT";
        case MenuLocation::View: return "VIEW";
        case MenuLocation::Window: return "WINDOW";
        case MenuLocation::Help: return "HELP";
        case MenuLocation::MenuBar: return "MENU_BAR";
        default: return "UNKNOWN";
        }
    }

    static MenuLocation parse_menu_location(const std::string& s) {
        if (s == "EDIT")
            return MenuLocation::Edit;
        if (s == "VIEW")
            return MenuLocation::View;
        if (s == "WINDOW")
            return MenuLocation::Window;
        if (s == "HELP")
            return MenuLocation::Help;
        if (s == "MENU_BAR")
            return MenuLocation::MenuBar;
        return MenuLocation::File;
    }

    static void sort_by_order(std::vector<PyMenuClassInfo>& menus) {
        std::sort(menus.begin(), menus.end(),
                  [](const PyMenuClassInfo& a, const PyMenuClassInfo& b) { return a.order < b.order; });
    }

    namespace {
        nb::object dict_get(const nb::dict& dict, const char* key) {
            const auto py_key = nb::str(key);
            if (dict.contains(py_key))
                return dict[py_key];
            return nb::none();
        }

        std::string object_to_string(const nb::object& obj, const std::string& fallback = "") {
            if (!obj.is_valid() || obj.is_none())
                return fallback;
            try {
                return nb::cast<std::string>(obj);
            } catch (...) {
                return fallback;
            }
        }

        bool object_to_bool(const nb::object& obj, bool fallback = false) {
            if (!obj.is_valid() || obj.is_none())
                return fallback;
            try {
                return nb::cast<bool>(obj);
            } catch (...) {
                return fallback;
            }
        }

        bool is_separator_item(const nb::dict& item) {
            const auto type = object_to_string(dict_get(item, "type"));
            return type == "separator" || object_to_bool(dict_get(item, "separator"), false);
        }

        bool is_submenu_item(const nb::dict& item) {
            return object_to_string(dict_get(item, "type")) == "submenu";
        }

        bool is_operator_item(const nb::dict& item) {
            if (object_to_string(dict_get(item, "type")) == "operator")
                return true;
            const auto operator_id = dict_get(item, "operator_id");
            return operator_id.is_valid() && !operator_id.is_none();
        }

        bool is_callback_item(const nb::dict& item) {
            return !is_separator_item(item) && !is_submenu_item(item) && !is_operator_item(item);
        }

        vis::gui::MenuItemDesc::Type callback_item_type(const nb::dict& item) {
            const auto type = object_to_string(dict_get(item, "type"));
            if (type == "toggle")
                return vis::gui::MenuItemDesc::Type::Toggle;
            if (!object_to_string(dict_get(item, "shortcut")).empty())
                return vis::gui::MenuItemDesc::Type::ShortcutItem;
            return vis::gui::MenuItemDesc::Type::Item;
        }

        nb::object get_schema_items(const PyMenuClassInfo& info) {
            if (!nb::hasattr(info.menu_instance, "menu_items"))
                return nb::none();
            return info.menu_instance.attr("menu_items")();
        }

        bool has_schema_menu_items(const PyMenuClassInfo& info) {
            if (!nb::hasattr(info.menu_instance, "menu_items"))
                return false;

            try {
                nb::module_ types_module = nb::module_::import_("lfs_plugins.types");
                nb::object base_menu = types_module.attr("Menu");
                if (!nb::hasattr(base_menu, "menu_items") || !nb::hasattr(info.menu_class, "menu_items"))
                    return true;

                return !info.menu_class.attr("menu_items").is(base_menu.attr("menu_items"));
            } catch (...) {
                return true;
            }
        }

        void warn_legacy_menu_draw_once(const std::string& idname) {
            static std::mutex mutex;
            static std::unordered_set<std::string> warned;
            std::lock_guard lock(mutex);
            if (warned.emplace(idname).second) {
                LOG_WARN("Rml transition: menu '{}' uses legacy draw(layout) fallback. "
                         "Keep it for compatibility, but prefer declarative menu_items() "
                         "for new or touched menus.",
                         idname);
            }
        }

        void collect_schema_items(const nb::object& items_obj,
                                  vis::gui::MenuDropdownContent& content,
                                  int& callback_index) {
            if (!items_obj.is_valid() || items_obj.is_none())
                return;

            for (nb::handle item_handle : items_obj) {
                if (!nb::isinstance<nb::dict>(item_handle))
                    continue;
                nb::dict item = nb::borrow<nb::dict>(item_handle);

                if (is_separator_item(item)) {
                    vis::gui::MenuItemDesc desc;
                    desc.type = vis::gui::MenuItemDesc::Type::Separator;
                    content.items.push_back(std::move(desc));
                    continue;
                }

                if (is_submenu_item(item)) {
                    vis::gui::MenuItemDesc begin_desc;
                    begin_desc.type = vis::gui::MenuItemDesc::Type::SubMenuBegin;
                    begin_desc.label = object_to_string(dict_get(item, "label"));
                    content.items.push_back(std::move(begin_desc));

                    collect_schema_items(dict_get(item, "items"), content, callback_index);

                    vis::gui::MenuItemDesc end_desc;
                    end_desc.type = vis::gui::MenuItemDesc::Type::SubMenuEnd;
                    content.items.push_back(std::move(end_desc));
                    continue;
                }

                vis::gui::MenuItemDesc desc;
                desc.label = object_to_string(dict_get(item, "label"));
                desc.enabled = object_to_bool(dict_get(item, "enabled"), true);

                if (is_operator_item(item)) {
                    desc.type = vis::gui::MenuItemDesc::Type::Operator;
                    desc.operator_id = object_to_string(dict_get(item, "operator_id"));
                } else {
                    desc.type = callback_item_type(item);
                    desc.shortcut = object_to_string(dict_get(item, "shortcut"));
                    desc.selected = object_to_bool(dict_get(item, "selected"), false);
                    desc.callback_index = callback_index++;
                }

                content.items.push_back(std::move(desc));
            }
        }

        bool execute_schema_callback(const nb::object& items_obj,
                                     const nb::object& menu_instance,
                                     int target_callback_index,
                                     int& next_callback_index) {
            if (!items_obj.is_valid() || items_obj.is_none())
                return false;

            for (nb::handle item_handle : items_obj) {
                if (!nb::isinstance<nb::dict>(item_handle))
                    continue;
                nb::dict item = nb::borrow<nb::dict>(item_handle);

                if (is_submenu_item(item)) {
                    if (execute_schema_callback(
                            dict_get(item, "items"), menu_instance,
                            target_callback_index, next_callback_index)) {
                        return true;
                    }
                    continue;
                }

                if (!is_callback_item(item))
                    continue;

                const int current_index = next_callback_index++;
                if (current_index != target_callback_index)
                    continue;

                nb::object callback = dict_get(item, "callback");
                if (callback.is_valid() && !callback.is_none()) {
                    nb::borrow<nb::callable>(callback)();
                    return true;
                }

                const auto action_id = object_to_string(dict_get(item, "action_id"));
                if (!action_id.empty() && nb::hasattr(menu_instance, "on_menu_action")) {
                    menu_instance.attr("on_menu_action")(action_id);
                    return true;
                }

                return true;
            }

            return false;
        }

        void draw_schema_menu_items(const nb::object& items_obj, const nb::object& menu_instance) {
            if (!items_obj.is_valid() || items_obj.is_none())
                return;

            for (nb::handle item_handle : items_obj) {
                if (!nb::isinstance<nb::dict>(item_handle))
                    continue;
                nb::dict item = nb::borrow<nb::dict>(item_handle);

                if (is_separator_item(item)) {
                    ImGui::Separator();
                    continue;
                }

                if (is_submenu_item(item)) {
                    const auto label = object_to_string(dict_get(item, "label"));
                    if (ImGui::BeginMenu(label.c_str())) {
                        draw_schema_menu_items(dict_get(item, "items"), menu_instance);
                        ImGui::EndMenu();
                    }
                    continue;
                }

                const auto label = object_to_string(dict_get(item, "label"));
                const auto enabled = object_to_bool(dict_get(item, "enabled"), true);

                if (is_operator_item(item)) {
                    const auto operator_id = object_to_string(dict_get(item, "operator_id"));
                    const bool can_execute = enabled && vis::op::operators().poll(operator_id);
                    if (ImGui::MenuItem(label.c_str(), nullptr, false, can_execute))
                        vis::op::operators().invoke(operator_id);
                    continue;
                }

                const auto shortcut = object_to_string(dict_get(item, "shortcut"));
                const auto selected = object_to_bool(dict_get(item, "selected"), false);
                const bool clicked = ImGui::MenuItem(
                    label.c_str(),
                    shortcut.empty() ? nullptr : shortcut.c_str(),
                    selected,
                    enabled);
                if (!clicked)
                    continue;

                nb::object callback = dict_get(item, "callback");
                if (callback.is_valid() && !callback.is_none()) {
                    nb::borrow<nb::callable>(callback)();
                    continue;
                }

                const auto action_id = object_to_string(dict_get(item, "action_id"));
                if (!action_id.empty() && nb::hasattr(menu_instance, "on_menu_action"))
                    menu_instance.attr("on_menu_action")(action_id);
            }
        }
    } // namespace

    void PyMenuRegistry::register_menu(nb::object menu_class) {
        if (!menu_class.is_valid()) {
            LOG_ERROR("register_menu: invalid menu_class");
            return;
        }

        const std::string idname = nb::cast<std::string>(menu_class.attr("__module__")) + "." +
                                   nb::cast<std::string>(menu_class.attr("__qualname__"));

        std::string label;
        MenuLocation location = MenuLocation::File;
        int order = 100;

        try {
            if (nb::hasattr(menu_class, "label"))
                label = nb::cast<std::string>(menu_class.attr("label"));
            if (nb::hasattr(menu_class, "location"))
                location = parse_menu_location(nb::cast<std::string>(menu_class.attr("location")));
            if (nb::hasattr(menu_class, "order"))
                order = nb::cast<int>(menu_class.attr("order"));
        } catch (const std::exception& e) {
            LOG_ERROR("register_menu: failed to extract attributes: {}", e.what());
            return;
        }

        nb::object instance;
        try {
            instance = menu_class();
        } catch (const std::exception& e) {
            LOG_ERROR("register_menu: failed to create instance for '{}': {}", idname, e.what());
            return;
        }

        std::lock_guard lock(mutex_);

        auto it = std::find_if(menu_classes_.begin(), menu_classes_.end(),
                               [&idname](const PyMenuClassInfo& mc) { return mc.idname == idname; });

        if (it != menu_classes_.end()) {
            it->label = label;
            it->location = location;
            it->order = order;
            it->menu_class = menu_class;
            it->menu_instance = instance;
        } else {
            menu_classes_.push_back({idname, label, location, order, menu_class, instance});
        }

        sort_by_order(menu_classes_);
    }

    void PyMenuRegistry::unregister_menu(nb::object menu_class) {
        const std::string idname = nb::cast<std::string>(menu_class.attr("__module__")) + "." +
                                   nb::cast<std::string>(menu_class.attr("__qualname__"));

        std::lock_guard lock(mutex_);
        std::erase_if(menu_classes_, [&idname](const PyMenuClassInfo& mc) { return mc.idname == idname; });
    }

    void PyMenuRegistry::unregister_all() {
        std::lock_guard lock(mutex_);
        menu_classes_.clear();
        synced_from_python_ = false;
    }

    void PyMenuRegistry::draw_menu_items(MenuLocation location) {
        ensure_synced();
        std::vector<PyMenuClassInfo> menu_classes_copy;
        {
            std::lock_guard lock(mutex_);
            for (const auto& mc : menu_classes_) {
                if (mc.location == location) {
                    menu_classes_copy.push_back(mc);
                }
            }
        }

        const char* const section = menu_location_to_string(location);
        const bool has_hooks = PyUIHookRegistry::instance().has_hooks("menu", section);

        if (menu_classes_copy.empty() && !has_hooks) {
            return;
        }

        nb::gil_scoped_acquire gil;

        if (has_hooks) {
            PyUIHookRegistry::instance().invoke("menu", section, PyHookPosition::Prepend);
        }

        for (const auto& mc : menu_classes_copy) {
            if (ImGui::BeginMenu(LOC(mc.label.c_str()))) {
                try {
                    bool should_draw = true;
                    if (nb::hasattr(mc.menu_class, "poll")) {
                        should_draw = nb::cast<bool>(mc.menu_class.attr("poll")(nb::none()));
                    }
                    if (should_draw && has_schema_menu_items(mc)) {
                        draw_schema_menu_items(get_schema_items(mc), mc.menu_instance);
                    } else if (should_draw && nb::hasattr(mc.menu_instance, "draw")) {
                        warn_legacy_menu_draw_once(mc.idname);
                        PyUILayout layout(1);
                        mc.menu_instance.attr("draw")(layout);
                    }
                } catch (const std::exception& e) {
                    LOG_ERROR("Menu class '{}' draw error: {}", mc.idname, e.what());
                }
                ImGui::EndMenu();
            }
        }

        if (has_hooks) {
            PyUIHookRegistry::instance().invoke("menu", section, PyHookPosition::Append);
        }
    }

    bool PyMenuRegistry::has_items(MenuLocation location) const {
        ensure_synced();
        std::lock_guard lock(mutex_);
        for (const auto& mc : menu_classes_) {
            if (mc.location == location) {
                return true;
            }
        }
        const char* const section = menu_location_to_string(location);
        if (PyUIHookRegistry::instance().has_hooks("menu", section)) {
            return true;
        }
        return false;
    }

    void PyMenuRegistry::ensure_synced() const {
        if (!synced_from_python_) {
            synced_from_python_ = true;
            sync_from_python();
        }
    }

    bool PyMenuRegistry::has_menu_bar_entries() const {
        ensure_synced();
        std::lock_guard lock(mutex_);
        for (const auto& mc : menu_classes_) {
            if (mc.location == MenuLocation::MenuBar) {
                return true;
            }
        }
        return false;
    }

    std::vector<PyMenuClassInfo*> PyMenuRegistry::get_menu_bar_entries() {
        ensure_synced();
        std::lock_guard lock(mutex_);
        std::vector<PyMenuClassInfo*> result;
        for (auto& mc : menu_classes_) {
            if (mc.location == MenuLocation::MenuBar) {
                result.push_back(&mc);
            }
        }
        std::sort(result.begin(), result.end(),
                  [](const PyMenuClassInfo* a, const PyMenuClassInfo* b) { return a->order < b->order; });
        return result;
    }

    void PyMenuRegistry::draw_menu_bar_entry(const std::string& idname) {
        PyMenuClassInfo* target = nullptr;
        {
            std::lock_guard lock(mutex_);
            for (auto& mc : menu_classes_) {
                if (mc.idname == idname) {
                    target = &mc;
                    break;
                }
            }
        }

        if (!target || !target->menu_instance.is_valid()) {
            return;
        }

        nb::gil_scoped_acquire gil;

        try {
            bool should_draw = true;
            if (nb::hasattr(target->menu_class, "poll")) {
                should_draw = nb::cast<bool>(target->menu_class.attr("poll")(nb::none()));
            }
            if (!should_draw) {
                return;
            }
            if (has_schema_menu_items(*target)) {
                draw_schema_menu_items(get_schema_items(*target), target->menu_instance);
            } else if (nb::hasattr(target->menu_instance, "draw")) {
                warn_legacy_menu_draw_once(idname);
                PyUILayout layout(1);
                target->menu_instance.attr("draw")(layout);
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Menu '{}' draw error: {}", idname, e.what());
        }
    }

    void PyMenuRegistry::sync_from_python() const {
        // GIL is already held by callers (bridge functions in python_runtime.cpp)
        try {
            auto menus_module = nb::module_::import_("lfs_plugins.layouts.menus");
            auto get_menu_classes = menus_module.attr("get_menu_classes");
            auto menu_classes = get_menu_classes();

            std::lock_guard lock(mutex_);

            menu_classes_.clear();

            for (nb::handle menu_class_handle : menu_classes) {
                nb::object menu_class = nb::borrow(menu_class_handle);
                if (!menu_class.is_valid())
                    continue;

                std::string idname = nb::cast<std::string>(menu_class.attr("__module__")) + "." +
                                     nb::cast<std::string>(menu_class.attr("__qualname__"));

                std::string label;
                MenuLocation location = MenuLocation::File;
                int order = 100;

                if (nb::hasattr(menu_class, "label")) {
                    label = nb::cast<std::string>(menu_class.attr("label"));
                }
                if (nb::hasattr(menu_class, "location")) {
                    location = parse_menu_location(nb::cast<std::string>(menu_class.attr("location")));
                }
                if (nb::hasattr(menu_class, "order")) {
                    order = nb::cast<int>(menu_class.attr("order"));
                }

                nb::object instance = menu_class();

                PyMenuClassInfo info;
                info.idname = idname;
                info.label = label;
                info.location = location;
                info.order = order;
                info.menu_class = menu_class;
                info.menu_instance = instance;

                menu_classes_.push_back(std::move(info));
            }

            sort_by_order(menu_classes_);

            LOG_INFO("Synced {} menus from Python registry", menu_classes_.size());
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to sync menus from Python: {}", e.what());
        }
    }

    vis::gui::MenuDropdownContent PyMenuRegistry::collect_menu_content(const std::string& idname) {
        vis::gui::MenuDropdownContent content;
        content.menu_idname = idname;

        PyMenuClassInfo* target = nullptr;
        {
            std::lock_guard lock(mutex_);
            for (auto& mc : menu_classes_) {
                if (mc.idname == idname) {
                    target = &mc;
                    break;
                }
            }
        }

        if (!target || !target->menu_instance.is_valid())
            return content;

        nb::gil_scoped_acquire gil;

        try {
            bool should_draw = true;
            if (nb::hasattr(target->menu_class, "poll"))
                should_draw = nb::cast<bool>(target->menu_class.attr("poll")(nb::none()));

            if (!should_draw)
                return content;

            if (has_schema_menu_items(*target)) {
                int callback_index = 0;
                collect_schema_items(get_schema_items(*target), content, callback_index);
            } else if (nb::hasattr(target->menu_instance, "draw")) {
                warn_legacy_menu_draw_once(idname);
                PyUILayout layout(1);
                layout.setCollecting(&content);
                target->menu_instance.attr("draw")(layout);
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Menu '{}' collect error: {}", idname, e.what());
        }

        return content;
    }

    void PyMenuRegistry::execute_menu_callback(const std::string& idname, int callback_index) {
        PyMenuClassInfo* target = nullptr;
        {
            std::lock_guard lock(mutex_);
            for (auto& mc : menu_classes_) {
                if (mc.idname == idname) {
                    target = &mc;
                    break;
                }
            }
        }

        if (!target || !target->menu_instance.is_valid())
            return;

        nb::gil_scoped_acquire gil;

        try {
            bool should_draw = true;
            if (nb::hasattr(target->menu_class, "poll"))
                should_draw = nb::cast<bool>(target->menu_class.attr("poll")(nb::none()));

            if (!should_draw)
                return;

            if (has_schema_menu_items(*target)) {
                int next_callback_index = 0;
                execute_schema_callback(
                    get_schema_items(*target), target->menu_instance,
                    callback_index, next_callback_index);
            } else if (nb::hasattr(target->menu_instance, "draw")) {
                warn_legacy_menu_draw_once(idname);
                vis::gui::MenuDropdownContent dummy;
                PyUILayout layout(1);
                layout.setCollecting(&dummy);
                layout.setExecuteAtIndex(callback_index);
                target->menu_instance.attr("draw")(layout);
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Menu '{}' callback execution error: {}", idname, e.what());
        }
    }

    void register_ui_menus(nb::module_& m) {
        nb::enum_<MenuLocation>(m, "MenuLocation")
            .value("FILE", MenuLocation::File)
            .value("EDIT", MenuLocation::Edit)
            .value("VIEW", MenuLocation::View)
            .value("WINDOW", MenuLocation::Window)
            .value("HELP", MenuLocation::Help)
            .value("MENU_BAR", MenuLocation::MenuBar);

        m.def(
            "register_menu",
            [](nb::object cls) { PyMenuRegistry::instance().register_menu(cls); },
            nb::arg("cls"), "Register a menu class");

        m.def(
            "unregister_menu",
            [](nb::object cls) { PyMenuRegistry::instance().unregister_menu(cls); },
            nb::arg("cls"), "Unregister a menu class");

        m.def(
            "unregister_all_menus", []() {
                PyMenuRegistry::instance().unregister_all();
            },
            "Unregister all Python menus");
    }

} // namespace lfs::python
