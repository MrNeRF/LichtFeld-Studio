/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "py_rml.hpp"
#include "py_ui.hpp"
#include "python/gil.hpp"

#include <algorithm>
#include <atomic>
#include <string>

namespace lfs::python {

    namespace {
        PyHookPosition parse_position(const std::string& position) {
            return (position == "prepend" || position == "PREPEND")
                       ? PyHookPosition::Prepend
                       : PyHookPosition::Append;
        }

        const char* position_name(const PyHookPosition position) {
            return position == PyHookPosition::Prepend ? "prepend" : "append";
        }

        std::string python_string_attr(PyObject* obj, const char* attr) {
            if (!obj)
                return {};
            PyObject* value = PyObject_GetAttrString(obj, attr);
            if (!value) {
                PyErr_Clear();
                return {};
            }

            const char* text = PyUnicode_Check(value) ? PyUnicode_AsUTF8(value) : nullptr;
            if (!text) {
                PyErr_Clear();
                Py_DECREF(value);
                return {};
            }

            std::string result = text;
            Py_DECREF(value);
            return result;
        }

        std::string callback_name(const nb::object& callback) {
            PyObject* obj = callback.ptr();
            std::string qualname = python_string_attr(obj, "__qualname__");
            if (qualname.empty())
                qualname = python_string_attr(obj, "__name__");
            if (qualname.empty())
                qualname = "<callable>";

            std::string module = python_string_attr(obj, "__module__");
            if (!module.empty() && qualname != "<callable>")
                return module + "." + qualname;
            return qualname;
        }

        std::string callback_module(const nb::object& callback) {
            PyObject* obj = callback.ptr();
            std::string module = python_string_attr(obj, "__module__");
            if (!module.empty())
                return module;

            PyObject* func = PyObject_GetAttrString(obj, "__func__");
            if (func) {
                module = python_string_attr(func, "__module__");
                Py_DECREF(func);
                if (!module.empty())
                    return module;
            } else {
                PyErr_Clear();
            }

            PyObject* cls = PyObject_GetAttrString(obj, "__class__");
            if (cls) {
                module = python_string_attr(cls, "__module__");
                Py_DECREF(cls);
            } else {
                PyErr_Clear();
            }
            return module;
        }

        bool is_first_party_hook(const nb::object& callback) {
            const std::string module = python_string_attr(callback.ptr(), "__module__");
            return module == "lfs_plugins" || module.starts_with("lfs_plugins.");
        }

        void warn_deprecated_ui_hooks_once(const nb::object& callback) {
            if (is_first_party_hook(callback))
                return;

            static std::atomic_bool warned{false};
            if (warned.exchange(true, std::memory_order_acq_rel))
                return;

            LOG_WARN("Python UI hooks are deprecated and will be removed after the reactive "
                     "RmlUi state migration. Use lfs_plugins.ui.RuntimeState and "
                     "Rml data-model updates instead.");
        }
        bool consume_document_dirty_with_attribution(Rml::ElementDocument* document,
                                                     const std::string& panel,
                                                     const std::string& section,
                                                     const PyHookPosition position,
                                                     const std::string& callback,
                                                     const char* source) {
            if (!document || !consume_document_dirty(document))
                return false;

            LOG_PERF("python_document_hook_dirty panel={} section={} position={} callback={} source={}",
                     panel,
                     section,
                     position_name(position),
                     callback,
                     source);
            return true;
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
        const std::string name = callback_name(callback);
        const std::string module = callback_module(callback);
        hooks_[key].push_back({std::move(callback), position, name, module});
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

    void PyUIHookRegistry::clear_hooks_for_module(const std::string& module_prefix) {
        std::lock_guard lock(mutex_);
        if (module_prefix.empty() || module_prefix == "lfs_plugins") {
            LOG_WARN("Refusing to clear UI hooks for broad module prefix '{}'", module_prefix);
            return;
        }

        const std::string child_prefix = module_prefix + ".";
        for (auto it = hooks_.begin(); it != hooks_.end();) {
            std::erase_if(it->second, [&](const HookEntry& entry) {
                return entry.module == module_prefix || entry.module.starts_with(child_prefix);
            });
            if (it->second.empty())
                it = hooks_.erase(it);
            else
                ++it;
        }
    }

    void PyUIHookRegistry::clear_all() {
        std::lock_guard lock(mutex_);
        hooks_.clear();
    }

    void PyUIHookRegistry::invoke(const std::string& panel,
                                  const std::string& section,
                                  PyHookPosition position) {
        (void)invoke_document(panel, section, nullptr, position);
    }

    bool PyUIHookRegistry::invoke_document(const std::string& panel,
                                           const std::string& section,
                                           Rml::ElementDocument* document,
                                           PyHookPosition position) {
        if (!can_acquire_gil())
            return false;

        nb::gil_scoped_acquire gil;
        std::vector<HookEntry> callbacks;
        {
            std::lock_guard lock(mutex_);
            const std::string key = panel + ":" + section;
            auto it = hooks_.find(key);
            if (it == hooks_.end()) {
                return consume_document_dirty_with_attribution(
                    document, panel, section, position, "<none>", "no_hooks");
            }
            for (const auto& entry : it->second) {
                if (entry.position == position) {
                    callbacks.push_back(entry);
                }
            }
        }

        if (callbacks.empty()) {
            return consume_document_dirty_with_attribution(
                document, panel, section, position, "<none>", "no_callbacks");
        }

        bool dirty = consume_document_dirty_with_attribution(
            document, panel, section, position, "<pre_hooks>", "before_callbacks");
        for (const auto& entry : callbacks) {
            try {
                if (document) {
                    entry.callback(PyRmlDocument(document));
                } else {
                    PyUILayout layout(PyUILayout::Mode::DrawHook);
                    entry.callback(layout);
                }
            } catch (const std::exception& e) {
                LOG_ERROR("Hook {}:{} error: {}", panel, section, e.what());
            }
            dirty |= consume_document_dirty_with_attribution(
                document, panel, section, position, entry.name, "after_callback");
        }
        return dirty;
    }

    bool PyUIHookRegistry::has_hooks(const std::string& panel, const std::string& section) const {
        std::lock_guard lock(mutex_);
        const std::string key = panel + ":" + section;
        auto it = hooks_.find(key);
        return it != hooks_.end() && !it->second.empty();
    }

    bool PyUIHookRegistry::has_hooks(const std::string& panel,
                                     const std::string& section,
                                     const PyHookPosition position) const {
        std::lock_guard lock(mutex_);
        const std::string key = panel + ":" + section;
        auto it = hooks_.find(key);
        if (it == hooks_.end())
            return false;
        return std::any_of(it->second.begin(), it->second.end(), [position](const HookEntry& entry) {
            return entry.position == position;
        });
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
                warn_deprecated_ui_hooks_once(callback);
                PyUIHookRegistry::instance().add_hook(panel, section, callback, parse_position(position));
            },
            nb::arg("panel"), nb::arg("section"), nb::arg("callback"),
            nb::arg("position") = "append",
            "Add a UI hook callback to a panel section. Hook layouts that cannot "
            "host interactive widgets warn once and return inert controls.");

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
            "clear_hooks_for_module",
            [](const std::string& prefix) {
                PyUIHookRegistry::instance().clear_hooks_for_module(prefix);
            },
            nb::arg("module_prefix"),
            "Clear all hooks registered by a given module prefix");

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
                    warn_deprecated_ui_hooks_once(func);
                    PyUIHookRegistry::instance().add_hook(panel, section, func, parse_position(position));
                    return func;
                });
            },
            nb::arg("panel"), nb::arg("section"), nb::arg("position") = "append",
            "Decorator to register a UI hook for a panel section");
    }

} // namespace lfs::python
