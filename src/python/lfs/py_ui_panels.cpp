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
#include "visualizer/gui/rmlui/rml_theme.hpp"

#include <algorithm>
#include <array>
#include <mutex>
#include <stdexcept>

namespace lfs::python {

    namespace gui = lfs::vis::gui;

    namespace {
        void throw_type_error(const std::string& message) {
            throw nb::type_error(message.c_str());
        }

        void throw_value_error(const std::string& message) {
            throw nb::value_error(message.c_str());
        }

        void throw_attribute_error(const std::string& message) {
            throw nb::attribute_error(message.c_str());
        }

        std::string get_class_id(nb::object cls) {
            auto mod = nb::cast<std::string>(cls.attr("__module__"));
            auto name = nb::cast<std::string>(cls.attr("__qualname__"));
            return mod + "." + name;
        }

        bool class_declares_attr(nb::handle cls, const char* attr_name) {
            if (!nb::hasattr(cls, "__dict__"))
                return false;
            nb::object class_dict = nb::getattr(cls, "__dict__");
            return PyMapping_HasKeyString(class_dict.ptr(), attr_name) == 1;
        }

        bool class_overrides(nb::object cls, nb::object base, const char* attr_name) {
            if (!nb::hasattr(cls, "__mro__"))
                return false;

            const nb::tuple mro = nb::cast<nb::tuple>(cls.attr("__mro__"));
            for (size_t i = 0; i < mro.size(); ++i) {
                nb::object current = nb::borrow<nb::object>(mro[i]);
                if (current.is(base))
                    return false;
                if (class_declares_attr(current, attr_name))
                    return true;
            }
            return false;
        }

        void reject_legacy_panel_attributes(nb::object panel_class, nb::object panel_base) {
            constexpr std::array legacy_mappings{
                std::pair{"idname", "id"},
                std::pair{"poll_deps", "poll_dependencies"},
                std::pair{"rml_template", "template"},
                std::pair{"rml_height_mode", "height_mode"},
                std::pair{"initial_width", "size"},
                std::pair{"initial_height", "size"},
            };

            for (const auto& [legacy_name, replacement] : legacy_mappings) {
                if (!class_overrides(panel_class, panel_base, legacy_name))
                    continue;
                throw_attribute_error(
                    std::string("Panel attribute '") + legacy_name +
                    "' has been removed. Use '" + replacement + "' instead.");
            }
        }

        std::string parse_string_value(nb::handle value, const char* field_name) {
            if (!nb::isinstance<nb::str>(value))
                throw_type_error(std::string(field_name) + " must be a string");
            return nb::cast<std::string>(value);
        }

        std::string parse_pathlike_value(nb::handle value, const char* field_name) {
            if (nb::isinstance<nb::str>(value))
                return nb::cast<std::string>(value);

            try {
                nb::object os_fspath = nb::module_::import_("os").attr("fspath");
                return nb::cast<std::string>(os_fspath(value));
            } catch (const std::exception&) {
                throw_type_error(std::string(field_name) + " must be a string or os.PathLike");
                return {};
            }
        }

        PanelSpace parse_panel_space_value(nb::handle value, const char* field_name) {
            try {
                return nb::cast<PanelSpace>(value);
            } catch (const std::exception&) {
                throw_type_error(std::string(field_name) + " must be a PanelSpace enum value");
            }
            return PanelSpace::Floating;
        }

        int parse_height_mode_value(nb::handle value, const char* field_name) {
            try {
                return static_cast<int>(nb::cast<PanelHeightMode>(value));
            } catch (const std::exception&) {
                throw_type_error(std::string(field_name) + " must be a PanelHeightMode enum value");
            }
            return 0;
        }

        uint32_t to_panel_option_mask(const gui::PanelOption option) {
            return static_cast<uint32_t>(option);
        }

        gui::PanelOption parse_panel_option_value(nb::handle value) {
            try {
                return nb::cast<gui::PanelOption>(value);
            } catch (const std::exception&) {
                throw_type_error("options entries must be PanelOption enum values");
            }
            return gui::PanelOption::DEFAULT_CLOSED;
        }

        PollDependency parse_poll_dependency_value(nb::handle value) {
            try {
                return nb::cast<PollDependency>(value);
            } catch (const std::exception&) {
                throw_type_error("poll_dependencies entries must be PollDependency enum values");
            }
            return PollDependency::NONE;
        }

        void define_panel_base_type(nb::module_& m) {
            if (nb::hasattr(m, "Panel"))
                return;
            nb::module_::import_("_lfs_panel_contract").attr("install_runtime_panel_base")(m);
        }

        nb::object panel_base_type() {
            static nb::object panel_type = nb::module_::import_("lichtfeld").attr("ui").attr("Panel");
            return panel_type;
        }

        std::string default_template_for_space(const PanelSpace space) {
            if (space == PanelSpace::Floating)
                return "rmlui/floating_window.rml";
            if (space == PanelSpace::StatusBar)
                return "rmlui/status_bar_panel.rml";
            return "rmlui/docked_panel.rml";
        }

        std::string default_immediate_document_for_space(const PanelSpace space) {
            if (space == PanelSpace::StatusBar)
                return "rmlui/im_mode_status_bar_panel.rml";
            return "rmlui/im_mode_panel.rml";
        }

        std::string resolve_template_identifier(const std::string& template_name,
                                                const PanelSpace space) {
            if (template_name.empty())
                return default_template_for_space(space);
            if (template_name == "builtin:docked-panel")
                return "rmlui/docked_panel.rml";
            if (template_name == "builtin:floating-window")
                return "rmlui/floating_window.rml";
            if (template_name == "builtin:status-bar")
                return "rmlui/status_bar_panel.rml";
            return template_name;
        }

        const std::string& retained_immediate_mode_style() {
            static std::string cached = []() {
                try {
                    return gui::rml_theme::loadBaseRCSS("rmlui/im_mode_panel.rcss");
                } catch (const std::exception& e) {
                    LOG_ERROR("Failed to load retained immediate-mode RCSS: {}", e.what());
                    return std::string{};
                }
            }();
            return cached;
        }

        std::string compose_retained_style(const std::string& style,
                                           const bool has_immediate_draw) {
            if (!has_immediate_draw)
                return style;

            const auto& im_mode_style = retained_immediate_mode_style();
            if (im_mode_style.empty())
                return style;
            if (style.empty())
                return im_mode_style;
            return im_mode_style + "\n" + style;
        }

        int parse_height_mode(nb::object panel_class) {
            if (!nb::hasattr(panel_class, "height_mode"))
                return 0;

            return parse_height_mode_value(panel_class.attr("height_mode"), "height_mode");
        }

        std::pair<float, float> parse_panel_size(nb::object panel_class) {
            if (!nb::hasattr(panel_class, "size"))
                return {0.0f, 0.0f};

            nb::object size_obj = panel_class.attr("size");
            if (!size_obj.is_valid() || size_obj.is_none())
                return {0.0f, 0.0f};

            if (!nb::isinstance<nb::tuple>(size_obj))
                throw_type_error("size must be a tuple[float, float] or None");

            const nb::tuple size_tuple = nb::cast<nb::tuple>(size_obj);
            if (size_tuple.size() != 2)
                throw_value_error("size must contain exactly two values");

            const auto width = nb::cast<float>(size_tuple[0]);
            const auto height = nb::cast<float>(size_tuple[1]);
            if (width < 0.0f || height < 0.0f)
                throw_value_error("size values must be non-negative");
            return {width, height};
        }

        nb::set panel_options_to_python_set(const uint32_t options) {
            nb::set values;
            if ((options & static_cast<uint32_t>(gui::PanelOption::DEFAULT_CLOSED)) != 0)
                values.add(nb::cast(gui::PanelOption::DEFAULT_CLOSED));
            if ((options & static_cast<uint32_t>(gui::PanelOption::HIDE_HEADER)) != 0)
                values.add(nb::cast(gui::PanelOption::HIDE_HEADER));
            return values;
        }

        nb::set poll_dependencies_to_python_set(const gui::PollDependency deps) {
            nb::set values;
            if ((deps & gui::PollDependency::SELECTION) != gui::PollDependency::NONE)
                values.add(nb::cast(PollDependency::SELECTION));
            if ((deps & gui::PollDependency::TRAINING) != gui::PollDependency::NONE)
                values.add(nb::cast(PollDependency::TRAINING));
            if ((deps & gui::PollDependency::SCENE) != gui::PollDependency::NONE)
                values.add(nb::cast(PollDependency::SCENE));
            return values;
        }

        nb::object panel_size_to_python_object(const float width, const float height) {
            if (width <= 0.0f && height <= 0.0f)
                return nb::none();
            return nb::make_tuple(width, height);
        }

        bool declares_retained_features(nb::object panel_class, nb::object panel_base,
                                        const std::string& template_name,
                                        const std::string& style, const int height_mode) {
            if (!template_name.empty() || !style.empty() || height_mode != 0)
                return true;

            for (const auto* hook :
                 std::array{"on_bind_model", "on_mount", "on_unmount", "on_update", "on_scene_changed"}) {
                if (class_overrides(panel_class, panel_base, hook))
                    return true;
            }

            return false;
        }
    } // namespace

    PyPanelRegistry& PyPanelRegistry::instance() {
        static PyPanelRegistry registry;
        return registry;
    }

    void PyPanelRegistry::register_panel(nb::object panel_class) {
        std::lock_guard lock(mutex_);

        if (!panel_class.is_valid())
            throw_type_error("register_panel: invalid panel class");

        const nb::object panel_base = panel_base_type();
        reject_legacy_panel_attributes(panel_class, panel_base);

        std::string label;
        std::string panel_id = get_class_id(panel_class);
        PanelSpace space = PanelSpace::MainPanelTab;
        int order = 100;
        uint32_t options = 0;
        PollDependency poll_dependencies = PollDependency::ALL;
        std::string parent_panel_id;
        std::string template_name;
        std::string style;
        int height_mode = 0;
        float initial_width = 0.0f;
        float initial_height = 0.0f;

        if (nb::hasattr(panel_class, "id")) {
            panel_id = parse_string_value(panel_class.attr("id"), "id");
            if (class_overrides(panel_class, panel_base, "id") && panel_id.empty())
                throw_value_error("id must not be empty");
        }

        if (nb::hasattr(panel_class, "label")) {
            label = parse_string_value(panel_class.attr("label"), "label");
        }
        if (nb::hasattr(panel_class, "space")) {
            space = parse_panel_space_value(panel_class.attr("space"), "space");
        }
        if (nb::hasattr(panel_class, "order")) {
            order = nb::cast<int>(panel_class.attr("order"));
        }
        if (nb::hasattr(panel_class, "parent")) {
            parent_panel_id = parse_string_value(panel_class.attr("parent"), "parent");
        }
        if (nb::hasattr(panel_class, "template")) {
            template_name = parse_pathlike_value(panel_class.attr("template"), "template");
        }
        if (nb::hasattr(panel_class, "style")) {
            style = parse_string_value(panel_class.attr("style"), "style");
        }
        height_mode = parse_height_mode(panel_class);
        auto [parsed_width, parsed_height] = parse_panel_size(panel_class);
        initial_width = parsed_width;
        initial_height = parsed_height;

        nb::object opts;
        if (nb::hasattr(panel_class, "options")) {
            opts = panel_class.attr("options");
        }
        if (opts.is_valid() && !opts.is_none()) {
            if (!nb::isinstance<nb::set>(opts))
                throw_type_error("options must be a set of PanelOption values");
            nb::set opts_set = nb::cast<nb::set>(opts);
            for (auto item : opts_set) {
                options |= to_panel_option_mask(parse_panel_option_value(item));
            }
        }
        if (nb::hasattr(panel_class, "poll_dependencies")) {
            nb::object deps_obj = panel_class.attr("poll_dependencies");
            if (deps_obj.is_valid() && !deps_obj.is_none()) {
                if (!nb::isinstance<nb::set>(deps_obj))
                    throw_type_error("poll_dependencies must be a set of PollDependency values");
                poll_dependencies = PollDependency::NONE;
                nb::set deps_set = nb::cast<nb::set>(deps_obj);
                for (auto item : deps_set) {
                    poll_dependencies = poll_dependencies | parse_poll_dependency_value(item);
                }
            }
        }

        if (!parent_panel_id.empty()) {
            if (class_overrides(panel_class, panel_base, "space"))
                throw_value_error("Panels with 'parent' must not also override 'space'");
            if (initial_width > 0.0f || initial_height > 0.0f)
                throw_value_error("Panels with 'parent' must not define 'size'");
        }

        if (label.empty())
            label = panel_id;

        LOG_DEBUG("Panel '{}' registered (space={})", label, static_cast<int>(space));

        nb::object instance = panel_class();

        if (!instance.is_valid())
            throw_value_error("register_panel: panel constructor returned an invalid instance");

        const bool has_poll = class_overrides(panel_class, panel_base, "poll");
        const bool has_immediate_draw = class_overrides(panel_class, panel_base, "draw");
        const bool use_retained = declares_retained_features(
            panel_class, panel_base, template_name, style, height_mode);
        const bool use_rml = (space != PanelSpace::ViewportOverlay) && lfs::python::get_rml_manager();

        if (space == PanelSpace::ViewportOverlay && use_retained) {
            throw_value_error(
                "VIEWPORT_OVERLAY panels do not support retained templates, styles, or lifecycle hooks");
        }

        if (use_retained && !use_rml) {
            throw_value_error(
                "Retained panel features require the retained UI manager and are unavailable in this runtime");
        }

        if (space == PanelSpace::Floating && !use_rml) {
            throw_value_error(
                "FLOATING panels require the retained UI manager. Floating windows do not fall back to immediate mode");
        }

        std::shared_ptr<gui::IPanel> adapter;
        if (use_retained) {
            auto retained_adapter = std::make_shared<gui::RmlPythonPanelAdapter>(
                lfs::python::get_rml_manager(),
                instance,
                panel_id,
                resolve_template_identifier(template_name, space),
                compose_retained_style(style, has_immediate_draw),
                has_poll,
                height_mode,
                has_immediate_draw);
            if (to_gui_space(space) == gui::PanelSpace::Floating)
                retained_adapter->setForeground(true);
            adapter = retained_adapter;
        } else if (use_rml) {
            adapter = std::make_shared<gui::RmlImModePanelAdapter>(
                lfs::python::get_rml_manager(),
                instance,
                has_poll,
                default_immediate_document_for_space(space));
        } else {
            adapter = std::make_shared<PythonPanelAdapter>(instance, has_poll);
        }

        gui::PanelInfo info;
        info.panel = adapter;
        info.label = label;
        info.id = panel_id;
        info.parent_id = parent_panel_id;
        info.space = to_gui_space(space);
        info.order = order;
        info.options = options;
        info.poll_dependencies = static_cast<gui::PollDependency>(poll_dependencies);
        info.is_native = false;
        info.initial_width = initial_width;
        info.initial_height = initial_height;

        const bool default_closed =
            (options & static_cast<uint32_t>(gui::PanelOption::DEFAULT_CLOSED)) &&
            (info.space == gui::PanelSpace::Floating);
        info.enabled = !default_closed;

        std::string module_prefix;
        try {
            module_prefix = nb::cast<std::string>(panel_class.attr("__module__"));
        } catch (const std::exception& e) {
            LOG_DEBUG("Panel '{}': could not read __module__: {}", panel_id, e.what());
        }

        if (!gui::PanelRegistry::instance().register_panel(std::move(info))) {
            throw_value_error(
                std::string("register_panel: runtime rejected panel '") + panel_id + "'");
        }
        panels_[panel_id] = {adapter, module_prefix};
    }

    void PyPanelRegistry::unregister_panel(nb::object panel_class) {
        std::lock_guard lock(mutex_);

        std::string panel_id;
        if (nb::hasattr(panel_class, "id")) {
            panel_id = parse_string_value(panel_class.attr("id"), "id");
        }
        if (panel_id.empty()) {
            panel_id = get_class_id(panel_class);
        }

        if (on_graphics_thread()) {
            gui::PanelRegistry::instance().unregister_panel(panel_id);
        } else {
            schedule_graphics_callback([id = panel_id]() {
                gui::PanelRegistry::instance().unregister_panel(id);
            });
        }
        panels_.erase(panel_id);
    }

    void PyPanelRegistry::unregister_all() {
        std::lock_guard lock(mutex_);
        if (on_graphics_thread()) {
            gui::PanelRegistry::instance().unregister_all_non_native();
        } else {
            schedule_graphics_callback([]() {
                gui::PanelRegistry::instance().unregister_all_non_native();
            });
        }
        panels_.clear();
    }

    void PyPanelRegistry::unregister_for_module(const std::string& prefix) {
        std::lock_guard lock(mutex_);

        if (prefix.empty() || prefix == "lfs_plugins") {
            LOG_WARN("Refusing to unregister panels for broad module prefix '{}'", prefix);
            return;
        }

        std::vector<std::string> to_remove;
        for (const auto& [panel_id, entry] : panels_) {
            if (entry.module_prefix == prefix || entry.module_prefix.starts_with(prefix + ".")) {
                to_remove.push_back(panel_id);
            }
        }

        for (const auto& panel_id : to_remove) {
            if (on_graphics_thread()) {
                gui::PanelRegistry::instance().unregister_panel(panel_id);
            } else {
                schedule_graphics_callback([id = panel_id]() {
                    gui::PanelRegistry::instance().unregister_panel(id);
                });
            }
            panels_.erase(panel_id);
            LOG_INFO("Unregistered panel '{}' for module '{}'", panel_id, prefix);
        }
    }

    void register_ui_panels(nb::module_& m) {
        nb::enum_<PanelSpace>(m, "PanelSpace")
            .value("SIDE_PANEL", PanelSpace::SidePanel)
            .value("FLOATING", PanelSpace::Floating)
            .value("VIEWPORT_OVERLAY", PanelSpace::ViewportOverlay)
            .value("MAIN_PANEL_TAB", PanelSpace::MainPanelTab)
            .value("SCENE_HEADER", PanelSpace::SceneHeader)
            .value("BOTTOM_DOCK", PanelSpace::BottomDock)
            .value("STATUS_BAR", PanelSpace::StatusBar);
        nb::enum_<PanelHeightMode>(m, "PanelHeightMode")
            .value("FILL", PanelHeightMode::Fill)
            .value("CONTENT", PanelHeightMode::Content);
        nb::enum_<gui::PanelOption>(m, "PanelOption")
            .value("DEFAULT_CLOSED", gui::PanelOption::DEFAULT_CLOSED)
            .value("HIDE_HEADER", gui::PanelOption::HIDE_HEADER);
        nb::enum_<PollDependency>(m, "PollDependency")
            .value("NONE", PollDependency::NONE)
            .value("SELECTION", PollDependency::SELECTION)
            .value("TRAINING", PollDependency::TRAINING)
            .value("SCENE", PollDependency::SCENE)
            .value("ALL", PollDependency::ALL);
        define_panel_base_type(m);
        nb::class_<gui::PanelSummary>(m, "PanelSummary")
            .def_prop_ro("id", [](const gui::PanelSummary& summary) { return summary.id; })
            .def_ro("label", &gui::PanelSummary::label)
            .def_ro("space", &gui::PanelSummary::space)
            .def_ro("order", &gui::PanelSummary::order)
            .def_ro("enabled", &gui::PanelSummary::enabled);
        nb::class_<gui::PanelDetails>(m, "PanelInfo")
            .def_prop_ro("id", [](const gui::PanelDetails& info) { return info.id; })
            .def_ro("label", &gui::PanelDetails::label)
            .def_prop_ro("parent", [](const gui::PanelDetails& info) { return info.parent_id; })
            .def_ro("space", &gui::PanelDetails::space)
            .def_ro("order", &gui::PanelDetails::order)
            .def_ro("enabled", &gui::PanelDetails::enabled)
            .def_prop_ro("options", [](const gui::PanelDetails& info) {
                return panel_options_to_python_set(info.options);
            })
            .def_prop_ro("poll_dependencies", [](const gui::PanelDetails& info) {
                return poll_dependencies_to_python_set(info.poll_dependencies);
            })
            .def_ro("is_native", &gui::PanelDetails::is_native)
            .def_prop_ro("size", [](const gui::PanelDetails& info) {
                return panel_size_to_python_object(info.initial_width, info.initial_height);
            });

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
            "get_panel_names", [](PanelSpace space) {
                return gui::PanelRegistry::instance().get_panel_names(to_gui_space(space));
            },
            nb::arg("space") = PanelSpace::Floating, "Get registered panel ids for a given space");

        m.def(
            "set_panel_enabled", [](const std::string& panel_id, bool enabled) {
                gui::PanelRegistry::instance().set_panel_enabled(panel_id, enabled);
            },
            nb::arg("panel_id"), nb::arg("enabled"), "Enable or disable a panel by id");

        m.def(
            "is_panel_enabled", [](const std::string& panel_id) {
                return gui::PanelRegistry::instance().is_panel_enabled(panel_id);
            },
            nb::arg("panel_id"), "Check if a panel is enabled");

        m.def(
            "get_main_panel_tabs", []() {
                return gui::PanelRegistry::instance().get_panels_for_space(gui::PanelSpace::MainPanelTab);
            },
            "Get all main panel tabs as typed panel summaries");

        m.def(
            "get_panel", [](const std::string& panel_id) {
                return gui::PanelRegistry::instance().get_panel(panel_id);
            },
            nb::arg("panel_id"), "Get typed panel info by id (None if not found)");

        m.def(
            "set_panel_label", [](const std::string& panel_id, const std::string& new_label) {
                return gui::PanelRegistry::instance().set_panel_label(panel_id, new_label);
            },
            nb::arg("panel_id"), nb::arg("label"), "Set the display label for a panel");

        m.def(
            "set_panel_order", [](const std::string& panel_id, int new_order) {
                return gui::PanelRegistry::instance().set_panel_order(panel_id, new_order);
            },
            nb::arg("panel_id"), nb::arg("order"), "Set the sort order for a panel");

        m.def(
            "set_panel_space", [](const std::string& panel_id, PanelSpace space) {
                return gui::PanelRegistry::instance().set_panel_space(panel_id, to_gui_space(space));
            },
            nb::arg("panel_id"), nb::arg("space"), "Set the panel space (where it renders)");

        m.def(
            "set_panel_parent", [](const std::string& panel_id, const std::string& parent_id) {
                return gui::PanelRegistry::instance().set_panel_parent(panel_id, parent_id);
            },
            nb::arg("panel_id"), nb::arg("parent"), "Set the parent panel (embeds as collapsible section)");

        m.def(
            "has_main_panel_tabs", []() {
                return gui::PanelRegistry::instance().has_panels(gui::PanelSpace::MainPanelTab);
            },
            "Check if any main panel tabs are registered");
    }

} // namespace lfs::python
