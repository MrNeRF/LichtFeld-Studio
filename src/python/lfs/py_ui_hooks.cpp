/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "py_rml.hpp"
#include "py_ui.hpp"

#include <algorithm>

namespace lfs::python {

    namespace {
        PyHookPosition parse_position(const std::string& position) {
            return (position == "prepend" || position == "PREPEND")
                       ? PyHookPosition::Prepend
                       : PyHookPosition::Append;
        }
    } // namespace

    PyUIHookRegistry& PyUIHookRegistry::instance() {
        static PyUIHookRegistry registry;
        return registry;
    }

    void PyUIHookRegistry::add_hook(const std::string& panel,
                                    const std::string& section,
                                    nb::object callback,
                                    PyHookPosition position) {
        std::lock_guard lock(mutex_);
        const std::string key = panel + ":" + section;
        hooks_[key].push_back({std::move(callback), position});
    }

    void PyUIHookRegistry::remove_hook(const std::string& panel,
                                       const std::string& section,
                                       nb::object callback) {
        std::lock_guard lock(mutex_);
        const std::string key = panel + ":" + section;
        auto it = hooks_.find(key);
        if (it == hooks_.end()) {
            return;
        }
        std::erase_if(it->second, [&callback](const HookEntry& entry) {
            return entry.callback.is(callback);
        });
    }

    void PyUIHookRegistry::clear_hooks(const std::string& panel, const std::string& section) {
        std::lock_guard lock(mutex_);
        if (section.empty()) {
            const std::string prefix = panel + ":";
            std::erase_if(hooks_, [&prefix](const auto& kv) { return kv.first.starts_with(prefix); });
        } else {
            hooks_.erase(panel + ":" + section);
        }
    }

    void PyUIHookRegistry::clear_all() {
        std::lock_guard lock(mutex_);
        hooks_.clear();
    }

    void PyUIHookRegistry::invoke(const std::string& panel,
                                  const std::string& section,
                                  PyHookPosition position) {
        invoke_document(panel, section, nullptr, position);
    }

    void PyUIHookRegistry::invoke_document(const std::string& panel,
                                           const std::string& section,
                                           Rml::ElementDocument* document,
                                           PyHookPosition position) {
        std::vector<nb::object> callbacks;
        nb::gil_scoped_acquire gil;
        {
            std::lock_guard lock(mutex_);
            const std::string key = panel + ":" + section;
            auto it = hooks_.find(key);
            if (it == hooks_.end()) {
                return;
            }
            for (const auto& entry : it->second) {
                if (entry.position == position) {
                    callbacks.push_back(entry.callback);
                }
            }
        }

        if (callbacks.empty()) {
            return;
        }

        for (const auto& cb : callbacks) {
            try {
                if (document) {
                    cb(PyRmlDocument(document));
                } else {
                    PyUILayout layout;
                    cb(layout);
                }
            } catch (const std::exception& e) {
                LOG_ERROR("Hook {}:{} error: {}", panel, section, e.what());
            }
        }
    }

    bool PyUIHookRegistry::has_hooks(const std::string& panel, const std::string& section) const {
        std::lock_guard lock(mutex_);
        const std::string key = panel + ":" + section;
        auto it = hooks_.find(key);
        return it != hooks_.end() && !it->second.empty();
    }

    std::vector<std::string> PyUIHookRegistry::get_hook_points() const {
        std::lock_guard lock(mutex_);
        std::vector<std::string> points;
        points.reserve(hooks_.size());
        for (const auto& [key, hooks] : hooks_) {
            if (!hooks.empty()) {
                points.push_back(key);
            }
        }
        return points;
    }

    void register_ui_hooks(nb::module_& m) {
        nb::enum_<PyHookPosition>(m, "HookPosition")
            .value("PREPEND", PyHookPosition::Prepend)
            .value("APPEND", PyHookPosition::Append);

        m.def(
            "add_hook",
            [](const std::string& panel, const std::string& section,
               nb::object callback, const std::string& position) {
                PyUIHookRegistry::instance().add_hook(panel, section, callback, parse_position(position));
            },
            nb::arg("panel"), nb::arg("section"), nb::arg("callback"),
            nb::arg("position") = "append",
            "Add a UI hook callback to a panel section");

        m.def(
            "remove_hook",
            [](const std::string& panel, const std::string& section, nb::object callback) {
                PyUIHookRegistry::instance().remove_hook(panel, section, callback);
            },
            nb::arg("panel"), nb::arg("section"), nb::arg("callback"),
            "Remove a specific UI hook callback");

        m.def(
            "clear_hooks",
            [](const std::string& panel, const std::string& section) {
                PyUIHookRegistry::instance().clear_hooks(panel, section);
            },
            nb::arg("panel"), nb::arg("section") = "",
            "Clear all hooks for a panel or panel/section");

        m.def(
            "clear_all_hooks", []() {
                PyUIHookRegistry::instance().clear_all();
            },
            "Clear all registered UI hooks");

        m.def(
            "get_hook_points", []() {
                return PyUIHookRegistry::instance().get_hook_points();
            },
            "Get all registered hook point identifiers");

        m.def(
            "invoke_hooks",
            [](const std::string& panel, const std::string& section, bool prepend) {
                PyUIHookRegistry::instance().invoke(panel, section,
                                                    prepend ? PyHookPosition::Prepend : PyHookPosition::Append);
            },
            nb::arg("panel"), nb::arg("section"), nb::arg("prepend") = false,
            "Invoke all hooks for a panel/section (prepend=True for prepend hooks, False for append)");

        m.def(
            "hook",
            [](const std::string& panel, const std::string& section, const std::string& position) {
                return nb::cpp_function([panel, section, position](nb::object func) {
                    PyUIHookRegistry::instance().add_hook(panel, section, func, parse_position(position));
                    return func;
                });
            },
            nb::arg("panel"), nb::arg("section"), nb::arg("position") = "append",
            "Decorator to register a UI hook for a panel section");
    }

} // namespace lfs::python
