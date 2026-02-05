/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/localization_manager.hpp"
#include "core/logger.hpp"
#include "py_ui.hpp"

#include <algorithm>
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
                    if (should_draw && nb::hasattr(mc.menu_instance, "draw")) {
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
            if (should_draw && nb::hasattr(target->menu_instance, "draw")) {
                PyUILayout layout(1);
                target->menu_instance.attr("draw")(layout);
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Menu '{}' draw error: {}", idname, e.what());
        }
    }

    void PyMenuRegistry::sync_from_python() const {
        nb::gil_scoped_acquire gil;

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
