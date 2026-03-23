/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_plugins.hpp"
#include "py_ui.hpp"

#include "core/logger.hpp"

#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace lfs::python {

    namespace {
        nb::object get_plugin_manager() {
            return nb::module_::import_("lfs_plugins").attr("PluginManager").attr("instance")();
        }

        nb::object get_capability_registry() {
            return nb::module_::import_("lfs_plugins").attr("CapabilityRegistry").attr("instance")();
        }

        nb::object get_settings_manager() {
            return nb::module_::import_("lfs_plugins").attr("SettingsManager").attr("instance")();
        }
    } // namespace

    void register_plugins(nb::module_& m) {
        auto plugins = m.def_submodule("plugins", "Plugin system - management, capabilities, panels, and settings");
        plugins.attr("API_VERSION") = "1.0";

        nb::list features;
        features.append("capabilities.v1");
        features.append("menus.v1");
        features.append("operators.v1");
        features.append("panels.v1");
        features.append("settings.v1");
        features.append("signals.v1");
        plugins.attr("FEATURES") = features;

        // ===== Plugin Management =====

        plugins.def(
            "discover", []() { return get_plugin_manager().attr("discover")(); },
            "Discover plugins in ~/.lichtfeld/plugins/");

        plugins.def(
            "load", [](const std::string& name) { return nb::cast<bool>(get_plugin_manager().attr("load")(name)); },
            nb::arg("name"), "Load plugin");

        plugins.def(
            "unload", [](const std::string& name) { return nb::cast<bool>(get_plugin_manager().attr("unload")(name)); },
            nb::arg("name"), "Unload plugin");

        plugins.def(
            "reload", [](const std::string& name) { return nb::cast<bool>(get_plugin_manager().attr("reload")(name)); },
            nb::arg("name"), "Reload plugin");

        plugins.def(
            "load_all", []() { return get_plugin_manager().attr("load_all")(); }, "Load all user-enabled plugins");

        plugins.def(
            "list_loaded", []() { return get_plugin_manager().attr("list_loaded")(); }, "List loaded plugins");

        plugins.def(
            "start_watcher", []() { get_plugin_manager().attr("start_watcher")(); }, "Start file watcher");

        plugins.def(
            "stop_watcher", []() { get_plugin_manager().attr("stop_watcher")(); }, "Stop file watcher");

        plugins.def(
            "get_state", [](const std::string& name) { return get_plugin_manager().attr("get_state")(name); },
            nb::arg("name"), "Get plugin state");

        plugins.def(
            "get_error", [](const std::string& name) { return get_plugin_manager().attr("get_error")(name); },
            nb::arg("name"), "Get plugin error");

        plugins.def(
            "get_traceback", [](const std::string& name) { return get_plugin_manager().attr("get_traceback")(name); },
            nb::arg("name"), "Get plugin error traceback");

        plugins.def(
            "install",
            [](const std::string& url, const bool auto_load, const std::string& transport) {
                return nb::cast<std::string>(
                    get_plugin_manager().attr("install")(url, nb::none(), auto_load, transport));
            },
            nb::arg("url"), nb::arg("auto_load") = true, nb::arg("transport") = "archive",
            "Install from GitHub URL");

        plugins.def(
            "update", [](const std::string& name) { return nb::cast<bool>(get_plugin_manager().attr("update")(name)); },
            nb::arg("name"), "Update plugin");

        plugins.def(
            "uninstall",
            [](const std::string& name) { return nb::cast<bool>(get_plugin_manager().attr("uninstall")(name)); },
            nb::arg("name"), "Uninstall plugin");

        plugins.def(
            "search", [](const std::string& query) { return get_plugin_manager().attr("search")(query); },
            nb::arg("query"), "Search plugin registry");

        plugins.def(
            "install_from_registry",
            [](const std::string& plugin_id, const std::string& version, const bool auto_load,
               const std::string& transport) {
                nb::object ver = version.empty() ? nb::none() : nb::cast(version);
                return nb::cast<std::string>(
                    get_plugin_manager().attr("install_from_registry")(
                        plugin_id, ver, nb::none(), auto_load, transport));
            },
            nb::arg("plugin_id"), nb::arg("version") = "", nb::arg("auto_load") = true,
            nb::arg("transport") = "archive",
            "Install plugin from registry");

        plugins.def(
            "check_updates", []() { return get_plugin_manager().attr("check_updates")(); },
            "Check for plugin updates");

        // ===== Capability API =====

        plugins.def(
            "register_capability",
            [](const std::string& name, nb::callable handler, const std::string& description,
               std::optional<nb::dict> schema, std::optional<std::string> plugin_name, bool requires_gui) {
                auto registry = get_capability_registry();

                nb::object py_schema = nb::none();
                if (schema) {
                    auto CapabilitySchema = nb::module_::import_("lfs_plugins").attr("CapabilitySchema");
                    nb::dict props = schema->contains("properties") ? nb::cast<nb::dict>((*schema)["properties"])
                                                                    : nb::dict();
                    nb::list req =
                        schema->contains("required") ? nb::cast<nb::list>((*schema)["required"]) : nb::list();
                    py_schema = CapabilitySchema(nb::arg("properties") = props, nb::arg("required") = req);
                }

                registry.attr("register")(name, handler, description, py_schema,
                                          plugin_name ? nb::cast(*plugin_name) : nb::none(), requires_gui);
            },
            nb::arg("name"), nb::arg("handler"), nb::arg("description") = "",
            nb::arg("schema") = nb::none(), nb::arg("plugin_name") = nb::none(), nb::arg("requires_gui") = true,
            "Register a capability (handler signature: def handler(args: dict, ctx: PluginContext) -> dict)");

        plugins.def(
            "unregister_capability",
            [](const std::string& name) { return nb::cast<bool>(get_capability_registry().attr("unregister")(name)); },
            nb::arg("name"), "Unregister a capability");

        plugins.def(
            "invoke",
            [](const std::string& name, nb::dict args) {
                return get_capability_registry().attr("invoke")(name, args);
            },
            nb::arg("name"), nb::arg("args") = nb::dict(),
            "Invoke a capability by name");

        plugins.def(
            "has_capability",
            [](const std::string& name) { return nb::cast<bool>(get_capability_registry().attr("has")(name)); },
            nb::arg("name"), "Check if a capability is registered");

        plugins.def(
            "list_capabilities", []() {
                nb::list result;
                auto caps = get_capability_registry().attr("list_all")();
                for (auto cap : caps) {
                    nb::dict d;
                    d["name"] = cap.attr("name");
                    d["description"] = cap.attr("description");
                    d["plugin_name"] = cap.attr("plugin_name");
                    result.append(d);
                }
                return result;
            },
            "List all registered capabilities");

        // ===== Settings API =====

        plugins.def(
            "settings",
            [](const std::string& plugin_name) { return get_settings_manager().attr("get")(plugin_name); },
            nb::arg("plugin_name"), "Get settings object for a plugin");

        // ===== Template Generator =====

        plugins.def(
            "create",
            [](const std::string& name) {
                auto create_plugin = nb::module_::import_("lfs_plugins").attr("create_plugin");
                return nb::cast<std::string>(create_plugin(name).attr("__str__")());
            },
            nb::arg("name"), "Create a new plugin from template (returns path to created plugin)");

        // NOTE: Plugin loading moved to runner.cpp:ensure_plugins_loaded()
        // This avoids circular import when plugins do `import lichtfeld` at module level
    }

} // namespace lfs::python
