/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "plugin_runner.hpp"
#include "gil.hpp"
#include "package_manager.hpp"
#include "runner.hpp"

#include <core/executable_path.hpp>
#include <cstdlib>
#include <print>

#include <Python.h>

namespace lfs::python {

    namespace {

        struct PyObjectGuard {
            PyObject* obj;
            explicit PyObjectGuard(PyObject* p) : obj(p) {}
            ~PyObjectGuard() { Py_XDECREF(obj); }
            PyObject* get() const { return obj; }
            operator bool() const { return obj != nullptr; }
        };

        std::string pyobj_to_string(PyObject* obj) {
            if (!obj)
                return "";
            PyObjectGuard str(PyObject_Str(obj));
            if (!str)
                return "";
            const char* c_str = PyUnicode_AsUTF8(str.get());
            return c_str ? c_str : "";
        }

        void print_python_error() {
            if (PyErr_Occurred()) {
                PyObject *type, *value, *traceback;
                PyErr_Fetch(&type, &value, &traceback);
                PyErr_NormalizeException(&type, &value, &traceback);

                std::string msg = pyobj_to_string(value);
                std::println(stderr, "Error: {}", msg);

                Py_XDECREF(type);
                Py_XDECREF(value);
                Py_XDECREF(traceback);
            }
        }

    } // anonymous namespace

    int run_plugin_command(const lfs::core::args::PluginMode& mode) {
        ensure_initialized();

        int result = 0;
        {
            const GilAcquire gil;

            switch (mode.command) {
            case lfs::core::args::PluginMode::Command::CREATE: {
                PyObjectGuard setup(PyImport_ImportModule("lfs_plugins.dev_setup"));
                if (!setup) {
                    print_python_error();
                    result = 1;
                    break;
                }

                PyObjectGuard func(PyObject_GetAttrString(setup.get(), "create_plugin_with_venv"));
                if (!func) {
                    print_python_error();
                    result = 1;
                    break;
                }

                const auto& pm = PackageManager::instance();
                const std::string uv_path = pm.uv_path().string();
                const std::string site_packages = pm.site_packages_dir().string();
                const std::string typings_dir = lfs::core::getTypingsDir().string();
#ifdef LFS_PYTHON_EXECUTABLE
                const std::string python_path = LFS_PYTHON_EXECUTABLE;
#else
                const std::string python_path;
#endif
                if (uv_path.empty()) {
                    std::println(stderr, "Error: UV not found");
                    result = 1;
                    break;
                }
                if (python_path.empty()) {
                    std::println(stderr, "Error: Python executable not configured");
                    result = 1;
                    break;
                }

                PyObjectGuard args(PyTuple_Pack(1, PyUnicode_FromString(mode.name.c_str())));
                PyObjectGuard kwargs(PyDict_New());
                PyDict_SetItemString(kwargs.get(), "uv_path", PyUnicode_FromString(uv_path.c_str()));
                PyDict_SetItemString(
                    kwargs.get(), "python_path", PyUnicode_FromString(python_path.c_str()));
                PyDict_SetItemString(
                    kwargs.get(), "typings_dir", PyUnicode_FromString(typings_dir.c_str()));
                PyDict_SetItemString(
                    kwargs.get(), "site_packages_dir", PyUnicode_FromString(site_packages.c_str()));

                PyObjectGuard path(PyObject_Call(func.get(), args.get(), kwargs.get()));
                if (!path) {
                    print_python_error();
                    result = 1;
                    break;
                }

                const char* path_str = PyUnicode_AsUTF8(path.get());
                if (path_str) {
                    std::println("Created: {}", path_str);
                    std::println("Open:    code {}", path_str);
                }
                break;
            }

            case lfs::core::args::PluginMode::Command::CHECK: {
                PyObjectGuard validator(PyImport_ImportModule("lfs_plugins.validator"));
                if (!validator) {
                    print_python_error();
                    result = 1;
                    break;
                }

                const auto plugins_dir =
                    std::filesystem::path(getenv("HOME")) / ".lichtfeld" / "plugins";
                const auto plugin_path = plugins_dir / mode.name;

                PyObjectGuard func(PyObject_GetAttrString(validator.get(), "validate_plugin"));
                if (!func) {
                    print_python_error();
                    result = 1;
                    break;
                }

                PyObjectGuard args(
                    PyTuple_Pack(1, PyUnicode_FromString(plugin_path.string().c_str())));
                PyObjectGuard errors(PyObject_CallObject(func.get(), args.get()));
                if (!errors) {
                    print_python_error();
                    result = 1;
                    break;
                }

                const Py_ssize_t len = PyList_Size(errors.get());
                if (len == 0) {
                    std::println("OK");
                } else {
                    for (Py_ssize_t i = 0; i < len; ++i) {
                        PyObject* item = PyList_GetItem(errors.get(), i);
                        const char* msg = PyUnicode_AsUTF8(item);
                        if (msg) {
                            std::println(stderr, "ERROR: {}", msg);
                        }
                    }
                    result = 1;
                }
                break;
            }

            case lfs::core::args::PluginMode::Command::LIST: {
                PyObjectGuard manager_mod(PyImport_ImportModule("lfs_plugins.manager"));
                if (!manager_mod) {
                    print_python_error();
                    result = 1;
                    break;
                }

                PyObjectGuard manager_class(
                    PyObject_GetAttrString(manager_mod.get(), "PluginManager"));
                if (!manager_class) {
                    print_python_error();
                    result = 1;
                    break;
                }

                PyObjectGuard instance_method(
                    PyObject_GetAttrString(manager_class.get(), "instance"));
                if (!instance_method) {
                    print_python_error();
                    result = 1;
                    break;
                }

                PyObjectGuard manager_instance(PyObject_CallObject(instance_method.get(), nullptr));
                if (!manager_instance) {
                    print_python_error();
                    result = 1;
                    break;
                }

                PyObjectGuard discover_method(
                    PyObject_GetAttrString(manager_instance.get(), "discover"));
                if (!discover_method) {
                    print_python_error();
                    result = 1;
                    break;
                }

                PyObjectGuard plugins(PyObject_CallObject(discover_method.get(), nullptr));
                if (!plugins) {
                    print_python_error();
                    result = 1;
                    break;
                }

                PyObjectGuard iter(PyObject_GetIter(plugins.get()));
                if (iter) {
                    PyObject* item;
                    while ((item = PyIter_Next(iter.get())) != nullptr) {
                        PyObjectGuard guard(item);
                        PyObjectGuard name_obj(PyObject_GetAttrString(item, "name"));
                        PyObjectGuard version_obj(PyObject_GetAttrString(item, "version"));
                        PyObjectGuard desc_obj(PyObject_GetAttrString(item, "description"));

                        const char* name = name_obj ? PyUnicode_AsUTF8(name_obj.get()) : "";
                        const char* version = version_obj ? PyUnicode_AsUTF8(version_obj.get()) : "";
                        const char* desc = desc_obj ? PyUnicode_AsUTF8(desc_obj.get()) : "";

                        std::println("{} v{} - {}", name ? name : "", version ? version : "",
                                     desc ? desc : "");
                    }
                }
                break;
            }
            }
        }

        // Skip cleanup to avoid Python/nanobind static destructor issues
        std::fflush(stdout);
        std::fflush(stderr);
        _exit(result);
    }

} // namespace lfs::python
