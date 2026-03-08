/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "py_rml.hpp"
#include "py_ui.hpp"
#include "python/python_runtime.hpp"
#include "python_panel_adapter.hpp"
#include "rml_im_mode_panel_adapter.hpp"
#include "rml_python_panel_adapter.hpp"
#include "visualizer/gui/panel_registry.hpp"

#include <algorithm>
#include <optional>

namespace lfs::python {

    namespace gui = lfs::vis::gui;

    namespace {
        std::string get_class_id(nb::object cls) {
            auto mod = nb::cast<std::string>(cls.attr("__module__"));
            auto name = nb::cast<std::string>(cls.attr("__qualname__"));
            return mod + "." + name;
        }

        std::optional<PanelSpace> parse_panel_space(const std::string& str) {
            if (str == "SIDE_PANEL" || str == "PROPERTIES")
                return PanelSpace::SidePanel;
            if (str == "VIEWPORT_OVERLAY")
                return PanelSpace::ViewportOverlay;
            if (str == "DOCKABLE")
                return PanelSpace::Dockable;
            if (str == "MAIN_PANEL_TAB")
                return PanelSpace::MainPanelTab;
            if (str == "SCENE_HEADER")
                return PanelSpace::SceneHeader;
            if (str == "STATUS_BAR")
                return PanelSpace::StatusBar;
            if (str == "FLOATING")
                return PanelSpace::Floating;
            return std::nullopt;
        }
    } // namespace

    PyPanelRegistry& PyPanelRegistry::instance() {
        static PyPanelRegistry registry;
        return registry;
    }

    void PyPanelRegistry::register_panel(nb::object panel_class) {
        std::lock_guard lock(mutex_);

        if (!panel_class.is_valid()) {
            LOG_ERROR("register_panel: invalid panel_class");
            return;
        }

        std::string label = "Python Panel";
        std::string idname = get_class_id(panel_class);
        PanelSpace space = PanelSpace::MainPanelTab;
        int order = 100;
        uint32_t options = 0;
        PollDependency poll_deps = PollDependency::ALL;

        try {
            if (nb::hasattr(panel_class, "idname")) {
                idname = nb::cast<std::string>(panel_class.attr("idname"));
            }
            if (nb::hasattr(panel_class, "label")) {
                label = nb::cast<std::string>(panel_class.attr("label"));
            }
            if (nb::hasattr(panel_class, "space")) {
                std::string space_str = nb::cast<std::string>(panel_class.attr("space"));
                if (!space_str.empty()) {
                    if (auto ps = parse_panel_space(space_str)) {
                        space = *ps;
                    } else {
                        LOG_WARN("Unknown panel space '{}' for panel '{}', defaulting to MainPanelTab", space_str, label);
                    }
                }
            }
            if (nb::hasattr(panel_class, "order")) {
                order = nb::cast<int>(panel_class.attr("order"));
            }
            nb::object opts;
            if (nb::hasattr(panel_class, "options")) {
                opts = panel_class.attr("options");
            }
            if (opts.is_valid() && nb::isinstance<nb::set>(opts)) {
                nb::set opts_set = nb::cast<nb::set>(opts);
                for (auto item : opts_set) {
                    std::string opt_str = nb::cast<std::string>(item);
                    if (opt_str == "DEFAULT_CLOSED") {
                        options |= static_cast<uint32_t>(gui::PanelOption::DEFAULT_CLOSED);
                    } else if (opt_str == "HIDE_HEADER") {
                        options |= static_cast<uint32_t>(gui::PanelOption::HIDE_HEADER);
                    }
                }
            }
            if (nb::hasattr(panel_class, "poll_deps")) {
                nb::object deps_obj = panel_class.attr("poll_deps");
                if (deps_obj.is_valid() && nb::isinstance<nb::set>(deps_obj)) {
                    poll_deps = PollDependency::NONE;
                    nb::set deps_set = nb::cast<nb::set>(deps_obj);
                    for (auto item : deps_set) {
                        std::string dep = nb::cast<std::string>(item);
                        if (dep == "SELECTION")
                            poll_deps = poll_deps | PollDependency::SELECTION;
                        else if (dep == "TRAINING")
                            poll_deps = poll_deps | PollDependency::TRAINING;
                        else if (dep == "SCENE")
                            poll_deps = poll_deps | PollDependency::SCENE;
                        else
                            LOG_WARN("Unknown poll dependency '{}' for panel '{}', ignoring", dep, label);
                    }
                }
            }
        } catch (const std::exception& e) {
            LOG_ERROR("register_panel: failed to extract attributes: {}", e.what());
            return;
        }

        LOG_DEBUG("Panel '{}' registered (space={})", label, static_cast<int>(space));

        nb::object instance;
        try {
            instance = panel_class();
        } catch (const std::exception& e) {
            LOG_ERROR("register_panel: failed to create instance for '{}': {}", label, e.what());
            return;
        }

        if (!instance.is_valid()) {
            LOG_ERROR("register_panel: invalid instance for '{}'", label);
            return;
        }

        const bool has_poll = nb::hasattr(panel_class, "poll");
        const bool use_rml = (space != PanelSpace::ViewportOverlay) && lfs::python::get_rml_manager();

        std::shared_ptr<gui::IPanel> adapter;
        if (use_rml) {
            adapter = std::make_shared<gui::RmlImModePanelAdapter>(
                lfs::python::get_rml_manager(), instance, has_poll);
        } else {
            adapter = std::make_shared<PythonPanelAdapter>(instance, has_poll);
        }

        std::string parent_idname;
        try {
            if (nb::hasattr(panel_class, "parent")) {
                parent_idname = nb::cast<std::string>(panel_class.attr("parent"));
            }
        } catch (const std::exception& e) {
            LOG_ERROR("register_panel: failed to extract parent for '{}': {}", label, e.what());
        }

        gui::PanelInfo info;
        info.panel = adapter;
        info.label = label;
        info.idname = idname;
        info.parent_idname = parent_idname;
        info.space = to_gui_space(space);
        info.order = order;
        info.options = options;
        info.poll_deps = static_cast<gui::PollDependency>(poll_deps);
        info.is_native = false;

        const bool default_closed =
            (options & static_cast<uint32_t>(gui::PanelOption::DEFAULT_CLOSED)) &&
            (info.space == gui::PanelSpace::Floating || info.space == gui::PanelSpace::Dockable);
        info.enabled = !default_closed;

        std::string module_prefix;
        try {
            module_prefix = nb::cast<std::string>(panel_class.attr("__module__"));
        } catch (...) {
        }

        gui::PanelRegistry::instance().register_panel(std::move(info));
        panels_[idname] = {adapter, module_prefix};
    }

    void PyPanelRegistry::unregister_panel(nb::object panel_class) {
        std::lock_guard lock(mutex_);

        std::string idname;
        if (nb::hasattr(panel_class, "idname")) {
            idname = nb::cast<std::string>(panel_class.attr("idname"));
        }
        if (idname.empty()) {
            idname = get_class_id(panel_class);
        }

        gui::PanelRegistry::instance().unregister_panel(idname);
        panels_.erase(idname);
    }

    void PyPanelRegistry::unregister_all() {
        std::lock_guard lock(mutex_);
        gui::PanelRegistry::instance().unregister_all_non_native();
        panels_.clear();
    }

    void PyPanelRegistry::unregister_for_module(const std::string& prefix) {
        std::lock_guard lock(mutex_);

        std::vector<std::string> to_remove;
        for (const auto& [idname, entry] : panels_) {
            if (entry.module_prefix == prefix || entry.module_prefix.starts_with(prefix + ".")) {
                to_remove.push_back(idname);
            }
        }

        for (const auto& idname : to_remove) {
            gui::PanelRegistry::instance().unregister_panel(idname);
            panels_.erase(idname);
            LOG_INFO("Unregistered panel '{}' for module '{}'", idname, prefix);
        }
    }

    void PyPanelRegistry::register_rml_panel(nb::object panel_class, void* rml_manager) {
        std::lock_guard lock(mutex_);

        if (!panel_class.is_valid()) {
            LOG_ERROR("register_rml_panel: invalid panel_class");
            return;
        }

        std::string label = "RmlUI Panel";
        std::string idname;
        PanelSpace space = PanelSpace::SceneHeader;
        int order = 0;
        std::string rml_template;
        int height_mode = 0;
        float initial_width = 0;
        float initial_height = 0;
        bool has_poll = false;

        try {
            idname = nb::hasattr(panel_class, "idname")
                         ? nb::cast<std::string>(panel_class.attr("idname"))
                         : get_class_id(panel_class);
            if (nb::hasattr(panel_class, "label"))
                label = nb::cast<std::string>(panel_class.attr("label"));
            if (nb::hasattr(panel_class, "space")) {
                std::string space_str = nb::cast<std::string>(panel_class.attr("space"));
                if (auto ps = parse_panel_space(space_str))
                    space = *ps;
            }
            if (nb::hasattr(panel_class, "order"))
                order = nb::cast<int>(panel_class.attr("order"));
            if (nb::hasattr(panel_class, "rml_template"))
                rml_template = nb::cast<std::string>(panel_class.attr("rml_template"));
            if (nb::hasattr(panel_class, "rml_height_mode")) {
                std::string mode_str = nb::cast<std::string>(panel_class.attr("rml_height_mode"));
                if (mode_str == "content")
                    height_mode = 1;
            }
            if (nb::hasattr(panel_class, "initial_width"))
                initial_width = nb::cast<float>(panel_class.attr("initial_width"));
            if (nb::hasattr(panel_class, "initial_height"))
                initial_height = nb::cast<float>(panel_class.attr("initial_height"));
            has_poll = nb::hasattr(panel_class, "poll");
        } catch (const std::exception& e) {
            LOG_ERROR("register_rml_panel: failed to extract attributes: {}", e.what());
            return;
        }

        if (rml_template.empty()) {
            LOG_ERROR("register_rml_panel: rml_template not set for '{}'", label);
            return;
        }

        nb::object instance;
        try {
            instance = panel_class();
        } catch (const std::exception& e) {
            LOG_ERROR("register_rml_panel: failed to create instance for '{}': {}", label, e.what());
            return;
        }

        auto adapter = std::make_shared<gui::RmlPythonPanelAdapter>(
            rml_manager, std::move(instance), idname, rml_template, has_poll, height_mode);

        const auto gui_space = to_gui_space(space);
        if (gui_space == gui::PanelSpace::Floating)
            adapter->setForeground(true);

        gui::PanelInfo info;
        info.panel = adapter;
        info.label = label;
        info.idname = idname;
        info.space = gui_space;
        info.order = order;
        info.is_native = false;
        info.initial_width = initial_width;
        info.initial_height = initial_height;

        std::string module_prefix;
        try {
            module_prefix = nb::cast<std::string>(panel_class.attr("__module__"));
        } catch (...) {
        }

        gui::PanelRegistry::instance().register_panel(std::move(info));
        panels_[idname] = {adapter, module_prefix};

        LOG_INFO("RmlUI panel '{}' registered", label);
    }

    void register_ui_panels(nb::module_& m) {
        nb::enum_<PanelSpace>(m, "PanelSpace")
            .value("SIDE_PANEL", PanelSpace::SidePanel)
            .value("FLOATING", PanelSpace::Floating)
            .value("VIEWPORT_OVERLAY", PanelSpace::ViewportOverlay)
            .value("DOCKABLE", PanelSpace::Dockable)
            .value("MAIN_PANEL_TAB", PanelSpace::MainPanelTab)
            .value("SCENE_HEADER", PanelSpace::SceneHeader)
            .value("STATUS_BAR", PanelSpace::StatusBar);

        m.def(
            "register_panel",
            [](nb::object cls) { PyPanelRegistry::instance().register_panel(cls); },
            nb::arg("cls"),
            "Register a panel class for rendering in the UI");

        m.def(
            "register_rml_panel",
            [](nb::object cls) {
                auto* mgr = lfs::python::get_rml_manager();
                if (!mgr) {
                    LOG_ERROR("register_rml_panel: RmlUI manager not available");
                    return;
                }
                PyPanelRegistry::instance().register_rml_panel(cls, mgr);
            },
            nb::arg("cls"),
            "Register an RmlUI panel class");

        m.def(
            "unregister_panel",
            [](nb::object cls) { PyPanelRegistry::instance().unregister_panel(cls); },
            nb::arg("cls"),
            "Unregister a panel class");

        m.def(
            "unregister_all_panels", []() {
                PyPanelRegistry::instance().unregister_all();
            },
            "Unregister all Python panels");

        m.def(
            "unregister_panels_for_module",
            [](const std::string& prefix) {
                PyPanelRegistry::instance().unregister_for_module(prefix);
            },
            nb::arg("module_prefix"),
            "Unregister all panels registered by a given module prefix");

        m.def(
            "get_panel_names", [](const std::string& space) {
                auto ps = parse_panel_space(space).value_or(PanelSpace::Floating);
                return gui::PanelRegistry::instance().get_panel_names(to_gui_space(ps));
            },
            nb::arg("space") = "FLOATING", "Get registered panel names for a given space");

        m.def(
            "set_panel_enabled", [](const std::string& idname, bool enabled) {
                gui::PanelRegistry::instance().set_panel_enabled(idname, enabled);
            },
            nb::arg("idname"), nb::arg("enabled"), "Enable or disable a panel by idname");

        m.def(
            "is_panel_enabled", [](const std::string& idname) {
                return gui::PanelRegistry::instance().is_panel_enabled(idname);
            },
            nb::arg("idname"), "Check if a panel is enabled");

        m.def(
            "get_main_panel_tabs", []() {
                auto tabs = gui::PanelRegistry::instance().get_panels_for_space(gui::PanelSpace::MainPanelTab);
                nb::list result;
                for (const auto& tab : tabs) {
                    nb::dict info;
                    info["idname"] = tab.idname;
                    info["label"] = tab.label;
                    info["order"] = tab.order;
                    info["enabled"] = tab.enabled;
                    result.append(info);
                }
                return result;
            },
            "Get all main panel tabs as list of dicts");

        m.def(
            "get_panel", [](const std::string& idname) -> nb::object {
                auto panel = gui::PanelRegistry::instance().get_panel(idname);
                if (!panel.has_value()) {
                    return nb::none();
                }
                nb::dict info;
                info["idname"] = panel->idname;
                info["label"] = panel->label;
                info["order"] = panel->order;
                info["enabled"] = panel->enabled;
                info["space"] = static_cast<int>(panel->space);
                return info;
            },
            nb::arg("idname"), "Get panel info by idname (None if not found)");

        m.def(
            "set_panel_label", [](const std::string& idname, const std::string& new_label) {
                return gui::PanelRegistry::instance().set_panel_label(idname, new_label);
            },
            nb::arg("idname"), nb::arg("label"), "Set the display label for a panel");

        m.def(
            "set_panel_order", [](const std::string& idname, int new_order) {
                return gui::PanelRegistry::instance().set_panel_order(idname, new_order);
            },
            nb::arg("idname"), nb::arg("order"), "Set the sort order for a panel");

        m.def(
            "set_panel_space", [](const std::string& idname, const std::string& space_str) {
                auto ps = parse_panel_space(space_str);
                if (!ps) {
                    LOG_WARN("Unknown panel space '{}' for panel '{}', defaulting to MainPanelTab", space_str, idname);
                }
                auto gui_space = to_gui_space(ps.value_or(PanelSpace::MainPanelTab));
                return gui::PanelRegistry::instance().set_panel_space(idname, gui_space);
            },
            nb::arg("idname"), nb::arg("space"), "Set the panel space (where it renders)");

        m.def(
            "set_panel_parent", [](const std::string& idname, const std::string& parent_idname) {
                return gui::PanelRegistry::instance().set_panel_parent(idname, parent_idname);
            },
            nb::arg("idname"), nb::arg("parent"), "Set the parent panel (embeds as collapsible section)");

        m.def(
            "has_main_panel_tabs", []() {
                return gui::PanelRegistry::instance().has_panels(gui::PanelSpace::MainPanelTab);
            },
            "Check if any main panel tabs are registered");
    }

} // namespace lfs::python
