/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "python_introspector.hpp"
#include "core/logger.hpp"
#include "python/gil.hpp"
#include <algorithm>

#include <Python.h>

namespace lfs::vis::editor {

    PythonIntrospector::PythonIntrospector() {
        last_refresh_ = std::chrono::steady_clock::now() - REFRESH_INTERVAL * 2; // Force initial refresh
    }

    PythonIntrospector::~PythonIntrospector() = default;

    bool PythonIntrospector::shouldRefresh() const {
        auto now = std::chrono::steady_clock::now();
        return (now - last_refresh_) > REFRESH_INTERVAL;
    }

    void PythonIntrospector::refresh() {
        if (!Py_IsInitialized()) {
            return;
        }

        std::vector<CompletionItem> new_globals;

        {
            const lfs::python::GilAcquire gil;

            // Get __main__ module globals
            PyObject* main_module = PyImport_AddModule("__main__");
            if (main_module) {
                PyObject* globals = PyModule_GetDict(main_module);
                if (globals) {
                    PyObject* keys = PyDict_Keys(globals);
                    if (keys) {
                        Py_ssize_t len = PyList_Size(keys);
                        for (Py_ssize_t i = 0; i < len; ++i) {
                            PyObject* key = PyList_GetItem(keys, i);
                            if (key && PyUnicode_Check(key)) {
                                const char* name = PyUnicode_AsUTF8(key);
                                if (name && name[0] != '_') {
                                    PyObject* value = PyDict_GetItem(globals, key);

                                    CompletionKind kind = CompletionKind::Variable;
                                    std::string type_str = "variable";

                                    if (value) {
                                        if (PyCallable_Check(value)) {
                                            if (PyType_Check(value)) {
                                                kind = CompletionKind::Class;
                                                type_str = "class";
                                            } else {
                                                kind = CompletionKind::Function;
                                                type_str = "function";
                                            }
                                        } else if (PyModule_Check(value)) {
                                            kind = CompletionKind::Module;
                                            type_str = "module";
                                        }
                                    }

                                    new_globals.push_back({std::string(name),
                                                           std::string(name),
                                                           "Runtime " + type_str,
                                                           kind,
                                                           70});
                                }
                            }
                        }
                        Py_DECREF(keys);
                    }
                }
            }
        }

        // Update cache
        {
            std::lock_guard lock(mutex_);
            cached_globals_ = std::move(new_globals);
            last_refresh_ = std::chrono::steady_clock::now();
        }
    }

    void PythonIntrospector::introspectObject(const std::string& obj_expr,
                                              std::vector<CompletionItem>& out) {
        if (!Py_IsInitialized() || obj_expr.empty()) {
            return;
        }

        const lfs::python::GilAcquire gil;

        PyObject* main_module = PyImport_AddModule("__main__");
        PyObject* globals = main_module ? PyModule_GetDict(main_module) : nullptr;

        if (globals) {
            PyObject* obj = PyRun_String(obj_expr.c_str(), Py_eval_input, globals, globals);
            if (obj) {
                PyObject* dir_result = PyObject_Dir(obj);
                if (dir_result) {
                    Py_ssize_t len = PyList_Size(dir_result);
                    for (Py_ssize_t i = 0; i < len; ++i) {
                        PyObject* name_obj = PyList_GetItem(dir_result, i);
                        if (name_obj && PyUnicode_Check(name_obj)) {
                            const char* name = PyUnicode_AsUTF8(name_obj);
                            if (name && name[0] != '_') {
                                PyObject* attr = PyObject_GetAttrString(obj, name);
                                CompletionKind kind = CompletionKind::Property;
                                std::string type_str = "attribute";

                                if (attr) {
                                    if (PyCallable_Check(attr)) {
                                        kind = CompletionKind::Function;
                                        type_str = "method";
                                    }
                                    Py_DECREF(attr);
                                }

                                out.push_back({std::string(name),
                                               obj_expr + "." + name,
                                               type_str,
                                               kind,
                                               75});
                            }
                        }
                    }
                    Py_DECREF(dir_result);
                }
                Py_DECREF(obj);
            } else {
                PyErr_Clear();
            }
        }
    }

    std::vector<CompletionItem> PythonIntrospector::getCompletions(
        const std::string& prefix, const std::string& context) {

        // Auto-refresh if needed
        if (shouldRefresh()) {
            refresh();
        }

        std::vector<CompletionItem> results;

        // Check if context indicates object member access (has a dot)
        size_t dot_pos = context.rfind('.');
        if (dot_pos != std::string::npos && dot_pos > 0) {
            // Extract the object expression before the dot
            std::string obj_expr = context.substr(0, dot_pos);

            // Remove leading whitespace and get the last token
            size_t start = obj_expr.find_last_of(" \t\n(,=");
            if (start != std::string::npos) {
                obj_expr = obj_expr.substr(start + 1);
            }

            if (!obj_expr.empty()) {
                introspectObject(obj_expr, results);
            }
        }

        // Filter by prefix
        auto startsWith = [](const std::string& str, const std::string& pre) {
            if (pre.size() > str.size())
                return false;
            return std::equal(pre.begin(), pre.end(), str.begin(),
                              [](char a, char b) { return std::tolower(a) == std::tolower(b); });
        };

        // Add global symbols
        {
            std::lock_guard lock(mutex_);
            for (const auto& item : cached_globals_) {
                if (prefix.empty() || startsWith(item.text, prefix)) {
                    results.push_back(item);
                }
            }
        }

        // Filter results by prefix for object members too
        if (!prefix.empty()) {
            results.erase(
                std::remove_if(results.begin(), results.end(),
                               [&](const CompletionItem& item) {
                                   return !startsWith(item.text, prefix);
                               }),
                results.end());
        }

        return results;
    }

} // namespace lfs::vis::editor
