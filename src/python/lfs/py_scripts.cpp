/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_scripts.hpp"

#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "python/runner.hpp"
#include "visualizer/gui/panels/python_scripts_panel.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace lfs::python {

    void register_scripts(nb::module_& m) {
        auto scripts = m.def_submodule("scripts", "Python script management");

        using ScriptState = vis::gui::panels::PythonScriptManagerState;

        scripts.def(
            "get_scripts",
            []() {
                auto& state = ScriptState::getInstance();
                nb::list result;
                for (const auto& s : state.scripts()) {
                    nb::dict d;
                    d["path"] = lfs::core::path_to_utf8(s.path);
                    d["enabled"] = s.enabled;
                    d["has_error"] = s.has_error;
                    d["error_message"] = s.error_message;
                    result.append(d);
                }
                return result;
            },
            "Get list of loaded scripts with their state");

        scripts.def(
            "set_script_enabled",
            [](size_t index, bool enabled) {
                ScriptState::getInstance().setScriptEnabled(index, enabled);
            },
            nb::arg("index"), nb::arg("enabled"),
            "Enable or disable a script by index");

        scripts.def(
            "set_script_error",
            [](size_t index, const std::string& error) {
                ScriptState::getInstance().setScriptError(index, error);
            },
            nb::arg("index"), nb::arg("error"),
            "Set error message for a script (empty to clear)");

        scripts.def(
            "clear_errors",
            []() { ScriptState::getInstance().clearErrors(); },
            "Clear all script errors");

        scripts.def(
            "clear",
            []() { ScriptState::getInstance().clear(); },
            "Clear all scripts");

        scripts.def(
            "run",
            [](const std::vector<std::string>& paths) {
                std::vector<std::filesystem::path> fs_paths;
                fs_paths.reserve(paths.size());
                for (const auto& p : paths) {
                    fs_paths.emplace_back(lfs::core::utf8_to_path(p));
                }
                auto result = lfs::python::run_scripts(fs_paths);
                nb::dict ret;
                if (result) {
                    ret["success"] = true;
                    ret["error"] = "";
                } else {
                    ret["success"] = false;
                    ret["error"] = result.error();
                }
                return ret;
            },
            nb::arg("paths"),
            "Run scripts by paths, returns {success: bool, error: str}");

        scripts.def(
            "get_enabled_paths",
            []() {
                auto& state = ScriptState::getInstance();
                std::vector<std::string> result;
                for (const auto& p : state.enabledScripts()) {
                    result.push_back(lfs::core::path_to_utf8(p));
                }
                return result;
            },
            "Get list of enabled script paths");

        scripts.def(
            "count",
            []() { return ScriptState::getInstance().scripts().size(); },
            "Get number of loaded scripts");
    }

} // namespace lfs::python
