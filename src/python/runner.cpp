/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "runner.hpp"
#include "package_manager.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <string>
#include <string_view>
#include <thread>

#include <core/executable_path.hpp>
#include <core/logger.hpp>
#include <core/path_utils.hpp>

#include "gil.hpp"
#include "python_compat.hpp"
#include "python_runtime.hpp"
#include "training/control/control_boundary.hpp"
#include <atomic>
#include <mutex>
#ifndef _WIN32
#include <unistd.h>
#endif

namespace lfs::python {

    static bool g_we_initialized_python = false;

    namespace {
        struct EnsureInitializedRegistrar {
            EnsureInitializedRegistrar() { set_ensure_initialized_callback(ensure_initialized); }
        };
        static EnsureInitializedRegistrar g_registrar;
    } // namespace

    static std::function<void(const std::string&, bool)> g_output_callback;
    static std::mutex g_output_mutex;
    static std::mutex g_plugin_init_mutex;
    static std::atomic<bool> g_python_bridge_ready{false};
    static std::atomic<bool> g_builtin_ui_ready{false};
    static std::atomic<bool> g_builtin_ui_deferred_logged{false};
    static std::atomic<bool> g_python_bridge_failed{false};
    static std::mutex g_python_bridge_failure_mutex;
    static std::string g_python_bridge_failure_detail;
    static std::atomic<bool> g_plugin_preload_scheduled{false};

    // RAII wrapper for the plugin preload thread that ensures proper cleanup
    // at static destruction time to avoid crashes from std::thread::~thread()
    // calling std::terminate() on a joinable thread.
    struct PluginPreloadThread {
        std::thread thread;
        ~PluginPreloadThread() {
            if (thread.joinable()) {
                thread.join();
            }
        }
    };
    static PluginPreloadThread g_plugin_preload_thread;

    // Python C extension for capturing output
    static PyObject* capture_write(PyObject* self, PyObject* args) {
        (void)self;
        const char* text = nullptr;
        int is_stderr = 0;
        if (!PyArg_ParseTuple(args, "si", &text, &is_stderr)) {
            return nullptr;
        }
        if (text && *text) {
            std::lock_guard lock(g_output_mutex);
            if (g_output_callback) {
                g_output_callback(text, is_stderr != 0);
            } else {
                if (is_stderr) {
                    LOG_WARN("[Python] {}", text);
                } else {
                    LOG_INFO("[Python] {}", text);
                }
            }
        }
        Py_RETURN_NONE;
    }

    static PyMethodDef g_capture_methods[] = {
        {"write", capture_write, METH_VARARGS, "Write to output callback"},
        {nullptr, nullptr, 0, nullptr}};

    static PyModuleDef g_capture_module = {
        PyModuleDef_HEAD_INIT, "_lfs_output", nullptr, -1, g_capture_methods};

    static PyObject* init_capture_module() {
        return PyModule_Create(&g_capture_module);
    }

    static void register_output_module_post_init() {
        PyObject* modules = PyImport_GetModuleDict();
        if (PyDict_GetItemString(modules, "_lfs_output")) {
            return;
        }
        PyObject* module = PyModule_Create(&g_capture_module);
        if (module) {
            PyDict_SetItemString(modules, "_lfs_output", module);
            Py_DECREF(module);
        }
    }

    static void redirect_output() {
        const char* redirect_code = R"(
import sys
import _lfs_output

class OutputCapture:
    def __init__(self, is_stderr=False):
        self._is_stderr = 1 if is_stderr else 0
    def write(self, text):
        if text:
            _lfs_output.write(text, self._is_stderr)
    def flush(self):
        pass

sys.stdout = OutputCapture(False)
sys.stderr = OutputCapture(True)
)";
        PyRun_SimpleString(redirect_code);
        LOG_DEBUG("Python output redirect installed");
    }

    void set_output_callback(std::function<void(const std::string&, bool)> callback) {
        std::lock_guard lock(g_output_mutex);
        g_output_callback = std::move(callback);
    }

    void write_output(const std::string& text, bool is_error) {
        std::lock_guard lock(g_output_mutex);
        if (g_output_callback) {
            g_output_callback(text, is_error);
        }
    }

    static void add_dll_directories() {
#ifdef _WIN32
        // Python 3.8+ on Windows requires os.add_dll_directory() for DLL loading
        // First add the executable directory using C++ (more reliable)
        const auto exe_dir = lfs::core::getExecutableDir();
        const auto exe_dir_str = lfs::core::path_to_utf8(exe_dir);

        std::string add_dll_code = std::format(R"(
import os
def _add_dll_dirs():
    dirs_to_add = [
        r'{}',  # Executable directory
    ]
    # Also add CUDA path if available
    cuda_path = os.environ.get('CUDA_PATH')
    if cuda_path:
        dirs_to_add.append(os.path.join(cuda_path, 'bin'))

    # Add vcpkg bin if it exists
    vcpkg_bin = os.path.join(r'{}', 'vcpkg_installed', 'x64-windows', 'bin')
    if os.path.isdir(vcpkg_bin):
        dirs_to_add.append(vcpkg_bin)

    for d in dirs_to_add:
        if os.path.isdir(d):
            try:
                os.add_dll_directory(d)
                print(f'[DLL] Added: {{d}}')
            except Exception as e:
                print(f'[DLL] Failed to add {{d}}: {{e}}')
_add_dll_dirs()
)",
                                               exe_dir_str, exe_dir_str);

        PyRun_SimpleString(add_dll_code.c_str());
        LOG_INFO("Windows DLL directories configured for: {}", exe_dir_str);
#endif
    }

    namespace {
        bool env_flag_enabled(const char* name, const bool default_value) {
            const char* value = std::getenv(name);
            if (!value || !*value) {
                return default_value;
            }

            const std::string_view text(value);
            if (text == "1" || text == "true" || text == "TRUE" || text == "on" || text == "yes") {
                return true;
            }
            if (text == "0" || text == "false" || text == "FALSE" || text == "off" || text == "no") {
                return false;
            }
            return default_value;
        }

        bool prepend_sys_path_once(PyObject* const sys_path,
                                   const std::filesystem::path& path,
                                   const char* label) {
            if (!sys_path || path.empty()) {
                return false;
            }

            const auto path_utf8 = lfs::core::path_to_utf8(path);
            PyObject* const py_path = PyUnicode_FromString(path_utf8.c_str());
            if (!py_path) {
                LOG_WARN("Failed to create Python path string for {}: {}", label, path_utf8);
                PyErr_Clear();
                return false;
            }

            const int contains = PySequence_Contains(sys_path, py_path);
            if (contains < 0) {
                LOG_WARN("Failed to inspect sys.path while adding {}: {}", label, path_utf8);
                PyErr_Clear();
                Py_DECREF(py_path);
                return false;
            }

            if (contains == 0) {
                if (PyList_Insert(sys_path, 0, py_path) != 0) {
                    LOG_WARN("Failed to prepend {} to sys.path: {}", label, path_utf8);
                    PyErr_Clear();
                    Py_DECREF(py_path);
                    return false;
                }
                LOG_INFO("Added {} to Python path: {}", label, path_utf8);
            }

            Py_DECREF(py_path);
            return true;
        }

#ifdef LFS_DEV_PYTHON_SOURCE_DIR
        void prepend_dev_python_source_path(PyObject* const sys_path) {
            const auto source_dir = lfs::core::utf8_to_path(LFS_DEV_PYTHON_SOURCE_DIR);
            std::error_code ec;
            if (!std::filesystem::exists(source_dir / "lfs_plugins", ec)) {
                LOG_WARN("Python dev source path is unavailable: {}",
                         lfs::core::path_to_utf8(source_dir));
                return;
            }

            prepend_sys_path_once(sys_path, source_dir, "dev source Python dir");
        }

        void start_dev_python_watcher(PyObject* const lfs_plugins) {
            if (!env_flag_enabled("LFS_PYTHON_HOT_RELOAD", true)) {
                LOG_INFO("Python dev hot reload disabled by LFS_PYTHON_HOT_RELOAD");
                return;
            }
            if (!lfs_plugins) {
                return;
            }

            PyObject* const manager_cls = PyObject_GetAttrString(lfs_plugins, "PluginManager");
            if (!manager_cls) {
                PyErr_Print();
                LOG_WARN("Python dev hot reload: lfs_plugins.PluginManager not found");
                return;
            }

            PyObject* const manager = PyObject_CallMethod(manager_cls, "instance", nullptr);
            Py_DECREF(manager_cls);
            if (!manager) {
                PyErr_Print();
                LOG_WARN("Python dev hot reload: failed to get PluginManager instance");
                return;
            }

            PyObject* const result = PyObject_CallMethod(manager, "start_watcher", nullptr);
            Py_DECREF(manager);
            if (!result) {
                PyErr_Print();
                LOG_WARN("Python dev hot reload: failed to start watcher");
                return;
            }

            Py_DECREF(result);
            LOG_INFO("Python dev hot reload watcher started");
        }
#endif

        std::string consume_python_error_detailed() {
            PyObject* type = nullptr;
            PyObject* value = nullptr;
            PyObject* tb = nullptr;
            PyErr_Fetch(&type, &value, &tb);
            PyErr_NormalizeException(&type, &value, &tb);

            std::string message = "(unknown error)";

            auto pyobject_to_utf8 = [](PyObject* obj) -> std::string {
                if (!obj) {
                    return {};
                }
                PyObject* str = PyObject_Str(obj);
                if (!str) {
                    return {};
                }
                std::string result;
                if (const char* text = PyUnicode_AsUTF8(str)) {
                    result = text;
                }
                Py_DECREF(str);
                return result;
            };

            PyObject* traceback_module = PyImport_ImportModule("traceback");
            if (traceback_module) {
                PyObject* format_exception = PyObject_GetAttrString(traceback_module, "format_exception");
                if (format_exception && PyCallable_Check(format_exception)) {
                    PyObject* args = PyTuple_Pack(3, type ? type : Py_None, value ? value : Py_None, tb ? tb : Py_None);
                    PyObject* lines = args ? PyObject_CallObject(format_exception, args) : nullptr;
                    if (lines) {
                        PyObject* empty = PyUnicode_FromString("");
                        PyObject* joined = empty ? PyUnicode_Join(empty, lines) : nullptr;
                        if (joined) {
                            if (const char* text = PyUnicode_AsUTF8(joined)) {
                                message = text;
                            }
                            Py_DECREF(joined);
                        }
                        Py_XDECREF(empty);
                        Py_DECREF(lines);
                    }
                    Py_XDECREF(args);
                }
                Py_XDECREF(format_exception);
                Py_DECREF(traceback_module);
            }

            if (message == "(unknown error)" || message.empty()) {
                if (const auto value_text = pyobject_to_utf8(value); !value_text.empty()) {
                    message = value_text;
                } else if (const auto type_text = pyobject_to_utf8(type); !type_text.empty()) {
                    message = type_text;
                }
            }

            Py_XDECREF(type);
            Py_XDECREF(value);
            Py_XDECREF(tb);
            return message;
        }

        void remember_python_bridge_failure(const std::string& detail) {
            bool expected = false;
            if (g_python_bridge_failed.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
                std::lock_guard lock(g_python_bridge_failure_mutex);
                g_python_bridge_failure_detail = detail;
            }
        }

        std::string python_bridge_failure_detail() {
            std::lock_guard lock(g_python_bridge_failure_mutex);
            return g_python_bridge_failure_detail;
        }

        PyObject* import_lichtfeld_module(const char* context, const bool latch_failure = false) {
            if (g_python_bridge_failed.load(std::memory_order_acquire)) {
                LOG_ERROR("{}: skipping lichtfeld import after previous initialization failure. Restart required. {}",
                          context, python_bridge_failure_detail());
                return nullptr;
            }

            PyObject* lf = PyImport_ImportModule("lichtfeld");
            if (!lf) {
                const std::string detail = consume_python_error_detailed();
                LOG_ERROR("{}: {}", context, detail);
                if (latch_failure) {
                    remember_python_bridge_failure(detail);
                }
                return nullptr;
            }

            return lf;
        }

        bool ensure_builtin_ui_ready_locked() {
            if (g_builtin_ui_ready.load(std::memory_order_acquire)) {
                return true;
            }

            if (!lfs::python::get_rml_manager()) {
                bool expected = false;
                if (g_builtin_ui_deferred_logged.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
                    LOG_INFO("Builtin Python UI registration deferred until retained UI runtime is available");
                }
                return false;
            }

            PyObject* lfs_plugins = PyImport_ImportModule("lfs_plugins");
            if (!lfs_plugins) {
                PyErr_Print();
                return false;
            }

            bool builtin_panels_registered = false;
            PyObject* register_fn = PyObject_GetAttrString(lfs_plugins, "register_builtin_panels");
            if (register_fn) {
                PyObject* result = PyObject_CallNoArgs(register_fn);
                if (!result) {
                    PyErr_Print();
                    LOG_ERROR("Failed to register builtin panels");
                } else {
                    const int registered = PyObject_IsTrue(result);
                    if (registered < 0) {
                        PyErr_Print();
                    } else {
                        builtin_panels_registered = registered != 0;
                    }
                    Py_DECREF(result);
                }
                Py_DECREF(register_fn);
            } else {
                PyErr_Clear();
                LOG_ERROR("lfs_plugins.register_builtin_panels not found");
            }

#ifdef LFS_DEV_PYTHON_SOURCE_DIR
            if (builtin_panels_registered) {
                start_dev_python_watcher(lfs_plugins);
            } else {
                LOG_INFO("Python dev hot reload watcher skipped because builtin panels were not registered");
            }
#endif
            Py_DECREF(lfs_plugins);

            if (builtin_panels_registered) {
                g_builtin_ui_ready.store(true, std::memory_order_release);
            }
            return builtin_panels_registered;
        }

        bool ensure_python_bridge_ready_locked() {
            if (g_python_bridge_ready.load(std::memory_order_acquire)) {
                ensure_builtin_ui_ready_locked();
                return true;
            }

            add_dll_directories();

            LOG_INFO("Attempting to import lichtfeld module...");
            PyObject* lf = import_lichtfeld_module("Failed to import lichtfeld", true);
            if (!lf) {
                return false;
            }
            LOG_INFO("lichtfeld module imported successfully");

            ensure_builtin_ui_ready_locked();

            // Initialize signal bridge after lfs_plugins.ui.state is available
            // Note: signals is registered as lichtfeld.ui.signals
            PyObject* ui_module = PyObject_GetAttrString(lf, "ui");
            if (ui_module) {
                PyObject* signals = PyObject_GetAttrString(ui_module, "signals");
                if (signals) {
                    PyObject* init_fn = PyObject_GetAttrString(signals, "init");
                    if (init_fn) {
                        PyObject* result = PyObject_CallNoArgs(init_fn);
                        if (!result) {
                            PyErr_Print();
                            LOG_ERROR("Failed to initialize signal bridge");
                        } else {
                            Py_DECREF(result);
                        }
                        Py_DECREF(init_fn);
                    } else {
                        LOG_ERROR("signals.init function not found");
                    }
                    Py_DECREF(signals);
                } else {
                    LOG_ERROR("signals submodule not found in lichtfeld.ui");
                }
                Py_DECREF(ui_module);
            } else {
                LOG_ERROR("ui submodule not found in lichtfeld module");
            }

            Py_DECREF(lf);
            g_python_bridge_ready.store(true, std::memory_order_release);
            return true;
        }

        std::vector<std::string> discover_enabled_plugins_locked() {
            std::vector<std::string> names;

            PyObject* lf = import_lichtfeld_module("Failed to import lichtfeld for plugin discovery");
            if (!lf) {
                return names;
            }

            PyObject* plugins = PyObject_GetAttrString(lf, "plugins");
            if (!plugins) {
                Py_DECREF(lf);
                return names;
            }

            PyObject* discover = PyObject_GetAttrString(plugins, "discover");
            if (!discover) {
                Py_DECREF(plugins);
                Py_DECREF(lf);
                return names;
            }

            PyObject* discovered = PyObject_CallNoArgs(discover);
            if (!discovered) {
                PyErr_Print();
                Py_DECREF(discover);
                Py_DECREF(plugins);
                Py_DECREF(lf);
                return names;
            }

            // Pre-register all discovered plugins so load() skips re-discovery
            PyObject* mgr_mod = PyImport_ImportModule("lfs_plugins.manager");
            if (mgr_mod) {
                PyObject* mgr_cls = PyObject_GetAttrString(mgr_mod, "PluginManager");
                if (mgr_cls) {
                    PyObject* mgr = PyObject_CallMethod(mgr_cls, "instance", nullptr);
                    if (mgr) {
                        PyObject* result = PyObject_CallMethod(mgr, "pre_register", "O", discovered);
                        Py_XDECREF(result);
                        Py_DECREF(mgr);
                    }
                    Py_DECREF(mgr_cls);
                }
                Py_DECREF(mgr_mod);
            }

            PyObject* settings_mod = PyImport_ImportModule("lfs_plugins.settings");
            PyObject* settings_mgr = nullptr;
            if (settings_mod) {
                PyObject* cls = PyObject_GetAttrString(settings_mod, "SettingsManager");
                if (cls) {
                    PyObject* instance = PyObject_CallMethod(cls, "instance", nullptr);
                    if (instance) {
                        settings_mgr = instance;
                    }
                    Py_DECREF(cls);
                }
                Py_DECREF(settings_mod);
            }

            PyObject* iter = PyObject_GetIter(discovered);
            if (iter) {
                PyObject* item;
                while ((item = PyIter_Next(iter)) != nullptr) {
                    PyObject* name_attr = PyObject_GetAttrString(item, "name");
                    if (name_attr && PyUnicode_Check(name_attr)) {
                        const char* plugin_name = PyUnicode_AsUTF8(name_attr);
                        bool enabled = false;
                        if (settings_mgr && plugin_name) {
                            PyObject* prefs = PyObject_CallMethod(settings_mgr, "get", "s", plugin_name);
                            if (prefs) {
                                PyObject* val = PyObject_CallMethod(prefs, "get", "sO",
                                                                    "load_on_startup", Py_False);
                                if (val) {
                                    enabled = PyObject_IsTrue(val);
                                    Py_DECREF(val);
                                }
                                Py_DECREF(prefs);
                            }
                        }
                        if (enabled && plugin_name) {
                            names.emplace_back(plugin_name);
                        }
                    }
                    Py_XDECREF(name_attr);
                    Py_DECREF(item);
                }
                Py_DECREF(iter);
            }

            Py_XDECREF(settings_mgr);
            Py_DECREF(discovered);
            Py_DECREF(discover);
            Py_DECREF(plugins);
            Py_DECREF(lf);
            return names;
        }

        bool load_single_plugin_locked(const std::string& name) {
            PyObject* lf = import_lichtfeld_module("Failed to import lichtfeld while loading plugin");
            if (!lf)
                return false;

            PyObject* plugins = PyObject_GetAttrString(lf, "plugins");
            if (!plugins) {
                Py_DECREF(lf);
                return false;
            }

            PyObject* py_name = PyUnicode_FromString(name.c_str());
            PyObject* result = PyObject_CallMethod(plugins, "load", "O", py_name);
            const bool success = result && PyObject_IsTrue(result);

            if (!success) {
                PyObject* get_traceback = PyObject_GetAttrString(plugins, "get_traceback");
                if (get_traceback) {
                    PyObject* tb = PyObject_CallOneArg(get_traceback, py_name);
                    if (tb && !Py_IsNone(tb) && PyUnicode_Check(tb)) {
                        LOG_ERROR("Plugin '{}' traceback:\n{}", name, PyUnicode_AsUTF8(tb));
                    }
                    Py_XDECREF(tb);
                    Py_DECREF(get_traceback);
                }
            }

            Py_XDECREF(result);
            Py_DECREF(py_name);
            Py_DECREF(plugins);
            Py_DECREF(lf);
            return success;
        }

    } // namespace

    std::filesystem::path get_user_packages_dir() {
        return PackageManager::instance().site_packages_dir();
    }

    void ensure_initialized() {
        call_once_py_init([] {
            if (!Py_IsInitialized()) {
                PyImport_AppendInittab("_lfs_output", init_capture_module);

                PyConfig config;
                PyConfig_InitPythonConfig(&config);
                config.user_site_directory = 0;

                const auto python_home = lfs::core::getPythonHome();
                if (!python_home.empty()) {
                    const auto home_wstr = python_home.wstring();
                    PyStatus st = PyConfig_SetString(&config, &config.home, home_wstr.c_str());
                    if (PyStatus_Exception(st)) {
                        LOG_ERROR("Failed to set Python home: {}", st.err_msg ? st.err_msg : "unknown");
                        PyConfig_Clear(&config);
                        return;
                    }
                    LOG_INFO("Set Python home: {}", lfs::core::path_to_utf8(python_home));
                }

                PyStatus status = Py_InitializeFromConfig(&config);
                PyConfig_Clear(&config);
                if (PyStatus_Exception(status)) {
                    LOG_ERROR("Failed to initialize Python: {}",
                              status.err_msg ? status.err_msg : "unknown");
                    return;
                }

                g_we_initialized_python = true;
                LOG_INFO("Python interpreter initialized by application");
            } else {
                LOG_WARN("Python already initialized by external code (e.g., .pyd loading)");
                g_we_initialized_python = false;
            }

            register_output_module_post_init();

            // Add user site-packages to sys.path
            std::filesystem::path user_packages = get_user_packages_dir();
            if (!std::filesystem::exists(user_packages)) {
                std::error_code ec;
                std::filesystem::create_directories(user_packages, ec);
                if (ec) {
                    LOG_WARN("Failed to create user packages dir: {}", ec.message());
                }
            }

            PyObject* sys_path = PySys_GetObject("path");
            if (sys_path) {
                prepend_sys_path_once(sys_path, user_packages, "user packages dir");

                const auto python_module_dir = lfs::core::getPythonModuleDir();
                if (!python_module_dir.empty()) {
                    prepend_sys_path_once(sys_path, python_module_dir, "Python module dir");
                } else {
                    const auto exe_dir_utf8 = lfs::core::path_to_utf8(lfs::core::getExecutableDir());
                    LOG_WARN("Python module 'lichtfeld' not found. Expected a lichtfeld*.so/.pyd in: {}/src/python, {}",
                             exe_dir_utf8, exe_dir_utf8);
                }

#ifdef LFS_DEV_PYTHON_SOURCE_DIR
                prepend_dev_python_source_path(sys_path);
#endif
            }

            {
                std::lock_guard lock(g_plugin_init_mutex);
                ensure_python_bridge_ready_locked();
            }

            set_main_thread_state(PyEval_SaveThread());
            set_gil_state_ready(true);
            LOG_DEBUG("GIL released, external_init={}", !g_we_initialized_python);
        });
    }

    void ensure_builtin_ui_registered() {
        ensure_initialized();
        if (!can_acquire_gil()) {
            LOG_WARN("Python GIL state not ready, skipping builtin UI registration");
            return;
        }

        const GilAcquire gil;
        std::lock_guard lock(g_plugin_init_mutex);
        if (!ensure_python_bridge_ready_locked()) {
            return;
        }
        ensure_builtin_ui_ready_locked();
    }

    void ensure_plugins_loaded() {
        ensure_initialized();
        if (!can_acquire_gil()) {
            LOG_WARN("Python GIL state not ready, skipping plugin load");
            return;
        }

        std::vector<std::string> to_load;
        {
            const GilAcquire gil;
            std::lock_guard lock(g_plugin_init_mutex);
            if (!ensure_python_bridge_ready_locked()) {
                LOG_WARN("Python bridge not ready, skipping plugin load");
                return;
            }
            if (are_plugins_loaded()) {
                return;
            }
            to_load = discover_enabled_plugins_locked();
            LOG_INFO("Plugin autoload: {} plugin(s) enabled for startup", to_load.size());
        }

        for (const auto& name : to_load) {
            const GilAcquire gil;
            if (load_single_plugin_locked(name)) {
                LOG_INFO("Loaded plugin: {}", name);
            } else {
                LOG_ERROR("Failed to load plugin: {}", name);
            }
        }

        mark_plugins_loaded();
    }

    void preload_user_plugins_async() {
        if (!env_flag_enabled("LFS_PLUGIN_AUTOLOAD", true)) {
            return;
        }

        bool expected = false;
        if (!g_plugin_preload_scheduled.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
            return;
        }

        g_plugin_preload_thread.thread = std::thread([]() {
            ensure_plugins_loaded();
        });
    }

    bool start_debugpy(const int port) {
        ensure_initialized();

        auto& pm = PackageManager::instance();
        if (!pm.is_installed("debugpy")) {
            LOG_INFO("Installing debugpy...");
            const auto result = pm.install("debugpy");
            if (!result.success) {
                LOG_ERROR("Failed to install debugpy: {}", result.error);
                return false;
            }
            update_python_path();
        }

        int rc;
        {
            const GilAcquire gil;
            const std::string code = std::format("import debugpy; debugpy.listen(('0.0.0.0', {}))", port);
            rc = PyRun_SimpleString(code.c_str());
        }

        if (rc != 0) {
            LOG_ERROR("Failed to start debugpy on port {}", port);
            return false;
        }

        LOG_INFO("debugpy listening on port {}", port);
        return true;
    }

    void join_plugin_preload() {
        if (g_plugin_preload_thread.thread.joinable()) {
            g_plugin_preload_thread.thread.join();
        }
    }

    void finalize() {
        join_plugin_preload();

        if (!Py_IsInitialized()) {
            return;
        }

        set_gil_state_ready(false);

        if (get_main_thread_state()) {
            acquire_gil_main_thread();
        } else {
            LOG_WARN("No saved thread state, using PyGILState_Ensure");
            PyGILState_Ensure();
        }

        // Clear all callbacks that hold Python objects (nanobind::object)
        // This must be done while GIL is held since nanobind::object
        // destructor decrements Python reference counts
        lfs::training::ControlBoundary::instance().clear_all();

        // Clear frame callback if set
        clear_frame_callback();

        // Clear Python UI registries that hold nb::object references
        // These singletons would otherwise destroy nb::objects during
        // static destruction, after Python is gone
        invoke_python_cleanup();

        PyGC_Collect();

        // Skip Py_FinalizeEx() - nanobind static destructors need Python alive
    }

    bool was_python_used() {
        return get_main_thread_state() != nullptr || Py_IsInitialized();
    }

    void install_output_redirect() {
        call_once_redirect([] {
            const GilAcquire gil;
            redirect_output();
        });
    }

    static std::thread g_repl_thread;
    static std::atomic<bool> g_repl_running{false};
    static std::atomic<int> g_repl_read_fd{-1};
    static std::atomic<int> g_repl_write_fd{-1};

    static void close_fd(int fd) {
        if (fd < 0)
            return;
#ifdef _WIN32
        _close(fd);
#else
        ::close(fd);
#endif
    }

    void start_embedded_repl(int read_fd, int write_fd) {
        stop_embedded_repl();
        ensure_initialized();

        auto& pm = PackageManager::instance();
        if (!pm.is_installed("ptpython")) {
            LOG_INFO("Installing ptpython...");
            const auto result = pm.install("ptpython");
            if (!result.success) {
                LOG_WARN("Failed to install ptpython: {} (falling back to code.interact)", result.error);
            } else {
                update_python_path();
            }
        }

        g_repl_read_fd = read_fd;
        g_repl_write_fd = write_fd;
        g_repl_running = true;

        g_repl_thread = std::thread([read_fd, write_fd]() {
            {
                const GilAcquire gil;
                install_output_redirect();

                SceneContextGuard ctx(get_application_scene());

                const std::string setup = std::format(R"(
import sys, os, atexit

_repl_read_fd = {}
_repl_write_fd = {}
_repl_in  = os.fdopen(os.dup(_repl_read_fd), 'r', buffering=1)
_repl_out = os.fdopen(os.dup(_repl_write_fd), 'w', buffering=1)

_saved_stdin  = sys.stdin
_saved_stdout = sys.stdout
_saved_stderr = sys.stderr
sys.stdin  = _repl_in
sys.stdout = _repl_out
sys.stderr = _repl_out

import lichtfeld as lf
_repl_locals = {{"lf": lf, "__name__": "__console__", "__doc__": None}}

_histfile = os.path.join(os.path.expanduser("~"), ".lichtfeld", "repl_history")
os.makedirs(os.path.dirname(_histfile), exist_ok=True)

_used_ptpython = False
try:
    from ptpython.repl import embed as _pt_embed
    from prompt_toolkit.output.vt100 import Vt100_Output
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.application import create_app_session

    _pt_output = Vt100_Output(_repl_out, get_size=lambda: Size(rows=24, columns=80), enable_cpr=False)
    _pt_input = None
    if not _repl_in.isatty():
        from prompt_toolkit.input.vt100 import Vt100Input
        _pt_input = Vt100Input(_repl_in)

    _used_ptpython = True
    with create_app_session(input=_pt_input, output=_pt_output):
        _pt_embed(
            globals=_repl_locals,
            locals=_repl_locals,
            history_filename=_histfile,
            title="LichtFeld Python Console",
        )
except ImportError:
    pass
except SystemExit:
    pass

if not _used_ptpython:
    try:
        import readline
    except ImportError:
        readline = None
    if readline is not None:
        import rlcompleter
        readline.set_completer(rlcompleter.Completer(_repl_locals).complete)
        readline.parse_and_bind("tab: complete")
        readline.set_history_length(1000)
        try:
            readline.read_history_file(_histfile)
        except FileNotFoundError:
            pass
        atexit.register(readline.write_history_file, _histfile)
    try:
        import code
        code.interact(banner="LichtFeld Python Console", local=_repl_locals, exitmsg="")
    except SystemExit:
        pass

sys.stdin  = _saved_stdin
sys.stdout = _saved_stdout
sys.stderr = _saved_stderr
_repl_in.close()
_repl_out.close()
)",
                                                      read_fd, write_fd);

                PyRun_SimpleString(setup.c_str());
            }

            close_fd(read_fd);
            close_fd(write_fd);
            g_repl_read_fd = -1;
            g_repl_write_fd = -1;
            g_repl_running = false;
            LOG_INFO("Embedded REPL thread exited");
        });
    }

    void stop_embedded_repl() {
        if (g_repl_running.load()) {
            const int rfd = g_repl_read_fd.load();
            const int wfd = g_repl_write_fd.load();
            if (rfd >= 0) {
                close_fd(rfd);
                g_repl_read_fd = -1;
            }
            if (wfd >= 0 && wfd != rfd) {
                close_fd(wfd);
                g_repl_write_fd = -1;
            }
        }
        if (g_repl_thread.joinable()) {
            g_repl_thread.join();
        }
    }

    void update_python_path() {
        const auto packages = get_user_packages_dir();
        if (!std::filesystem::exists(packages))
            return;

        const GilAcquire gil;

        PyObject* const sys_path = PySys_GetObject("path");
        if (sys_path) {
            const auto path_str = lfs::core::path_to_utf8(packages);
            PyObject* const py_path = PyUnicode_FromString(path_str.c_str());
            if (PySequence_Contains(sys_path, py_path) == 0) {
                PyList_Insert(sys_path, 0, py_path);
                LOG_INFO("Added to sys.path: {}", path_str);
            }
            Py_DECREF(py_path);
        }
    }

    std::expected<void, std::string> run_scripts(const std::vector<std::filesystem::path>& scripts) {
        if (scripts.empty()) {
            return {};
        }

        ensure_initialized();

        const GilAcquire gil;

        // Install output redirect (calls redirect_output() once)
        call_once_redirect([] { redirect_output(); });

        // Add Python module directory (where lichtfeld.so lives) to sys.path
        {
            const auto python_module_dir = lfs::core::getPythonModuleDir();
            if (!python_module_dir.empty()) {
                const auto python_module_dir_utf8 = lfs::core::path_to_utf8(python_module_dir);
                PyObject* sys_path = PySys_GetObject("path"); // borrowed
                PyObject* py_path = PyUnicode_FromString(python_module_dir_utf8.c_str());
                PyList_Append(sys_path, py_path);
                Py_DECREF(py_path);
                LOG_DEBUG("Added {} to Python path", python_module_dir_utf8);
            }
        }

        // Pre-import lichtfeld module to catch any initialization errors early
        {
            PyObject* lf_module = import_lichtfeld_module("Failed to pre-import lichtfeld module");
            if (!lf_module) {
                return std::unexpected("Failed to import lichtfeld module - see startup log for traceback");
            }
            Py_DECREF(lf_module);
            LOG_INFO("Successfully pre-imported lichtfeld module");
        }

        // Load plugins after lichtfeld is fully imported
        ensure_plugins_loaded();

        for (const auto& script : scripts) {
            const auto script_utf8 = lfs::core::path_to_utf8(script);
            if (!std::filesystem::exists(script)) {
                return std::unexpected(std::format("Python script not found: {}", script_utf8));
            }

            // Ensure script directory is on sys.path
            const auto parent_utf8 = lfs::core::path_to_utf8(script.parent_path());
            if (!parent_utf8.empty()) {
                PyObject* sys_path = PySys_GetObject("path"); // borrowed ref
                PyObject* py_parent = PyUnicode_FromString(parent_utf8.c_str());
                if (sys_path && py_parent) {
                    PyList_Append(sys_path, py_parent);
                }
                Py_XDECREF(py_parent);
            }

#ifdef _WIN32
            FILE* const fp = _wfopen(script.wstring().c_str(), L"r");
#else
            FILE* const fp = fopen(script.c_str(), "r");
#endif
            if (!fp) {
                return std::unexpected(std::format("Failed to open Python script: {}", script_utf8));
            }

            LOG_INFO("Executing Python script: {}", script_utf8);
            const int rc = PyRun_SimpleFileEx(fp, script_utf8.c_str(), /*closeit=*/1);
            if (rc != 0) {
                return std::unexpected(std::format("Python script failed: {} (rc={})", script_utf8, rc));
            }

            LOG_INFO("Python script completed: {}", script_utf8);
        }

        return {};
    }

    FormatResult format_python_code(const std::string& code) {
        if (code.empty())
            return {code, "", true};

        auto& pm = PackageManager::instance();
        if (!pm.is_installed("black")) {
            if (!pm.ensure_venv()) {
                LOG_ERROR("Failed to create venv for black");
                return {code, "Failed to create venv for black", false};
            }
            LOG_INFO("Installing black...");
            const auto install_result = pm.install("black");
            if (!install_result.success) {
                LOG_ERROR("Failed to install black: {}", install_result.error);
                return {code, install_result.error, false};
            }
            update_python_path();
        }

        ensure_initialized();
        const GilAcquire gil;

        static constexpr const char* FORMAT_CODE = R"(
def _lfs_format_code(code):
    import importlib
    import re
    import textwrap
    importlib.invalidate_caches()
    try:
        import black
    except ImportError as e:
        return (None, f"ImportError: {e}")

    def _looks_like_code_line(stripped):
        if not stripped:
            return False
        if stripped.startswith('#'):
            return True
        if stripped.startswith((
            'import ', 'from ', 'def ', 'class ', '@',
            'if ', 'for ', 'while ', 'try', 'with ',
            'async ', 'match ', 'case ', 'return ',
            'raise ', 'pass', 'break', 'continue',
            'global ', 'nonlocal ', 'assert ', 'yield ',
            'del ', 'elif ', 'else:', 'except', 'finally:'
        )):
            return True
        if stripped[:1] in ('"', "'", '(', '[', '{'):
            return True
        if re.match(r'[A-Za-z_][A-Za-z0-9_]*(?:\\.[A-Za-z_][A-Za-z0-9_]*)*\\s*[:=([{.]', stripped):
            return True
        return False

    def _comment_leading_preamble(source):
        lines = source.split('\n')
        changed = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if _looks_like_code_line(stripped):
                break
            indent = line[:len(line) - len(line.lstrip(' '))]
            lines[i] = indent + '# ' + stripped
            changed = True
        return ('\n'.join(lines), changed)

    def _indent_width(line):
        return len(line) - len(line.lstrip(' '))

    def _previous_significant_line(lines, idx):
        for j in range(idx - 1, -1, -1):
            stripped = lines[j].strip()
            if stripped and not stripped.startswith('#'):
                return j, stripped
        return None, ''

    def _expected_indent(lines, idx):
        prev_idx, prev_stripped = _previous_significant_line(lines, idx)
        if prev_idx is None:
            return 0
        prev_indent = _indent_width(lines[prev_idx])
        if prev_stripped.endswith(':'):
            return prev_indent + 4
        return prev_indent

    def _repair_indentation(source):
        lines = source.split('\n')
        changed = False

        for _ in range(len(lines)):
            try:
                compile('\n'.join(lines), '<lfs_formatter>', 'exec')
                return ('\n'.join(lines), changed)
            except IndentationError as err:
                lineno = getattr(err, 'lineno', None)
                if lineno is None:
                    break

                idx = lineno - 1
                if idx < 0 or idx >= len(lines):
                    break

                stripped = lines[idx].lstrip()
                if not stripped:
                    break

                msg = str(err)
                target_indent = _expected_indent(lines, idx)

                if 'expected an indented block' in msg:
                    target_indent = max(target_indent, 4)
                elif 'unexpected indent' not in msg and \
                        'unindent does not match any outer indentation level' not in msg:
                    break

                new_line = (' ' * target_indent) + stripped
                if new_line == lines[idx]:
                    break

                lines[idx] = new_line
                changed = True
            except SyntaxError:
                break

        return ('\n'.join(lines), changed)

    # Normalize unicode characters that break parsing (from copy-paste)
    replacements = {
        '\u201c': '"', '\u201d': '"',  # Smart double quotes
        '\u2018': "'", '\u2019': "'",  # Smart single quotes
        '\u2212': '-',                  # Unicode minus
        '\u2013': '-', '\u2014': '-',  # En-dash, em-dash
        '\u00a0': ' ',                  # Non-breaking space
        '\u2003': ' ', '\u2002': ' ',  # Em space, en space
        '\u2009': ' ',                  # Thin space
    }
    for old, new in replacements.items():
        code = code.replace(old, new)

    # Normalize line endings and remove trailing whitespace
    code = code.replace('\r\n', '\n').replace('\r', '\n')
    lines = [line.rstrip() for line in code.split('\n')]

    # Remove leading empty lines
    while lines and not lines[0].strip():
        lines.pop(0)

    # Remove trailing empty lines
    while lines and not lines[-1].strip():
        lines.pop()

    if not lines:
        return (code, None)

    # Convert tabs to spaces consistently
    cleaned = '\n'.join(line.replace('\t', '    ') for line in lines)
    cleaned, _ = _comment_leading_preamble(cleaned)

    try:
        return (black.format_str(cleaned, mode=black.Mode()), None)
    except Exception as first_error:
        repaired, changed = _repair_indentation(cleaned)
        if changed and repaired != cleaned:
            try:
                return (black.format_str(repaired, mode=black.Mode()), None)
            except Exception:
                pass

        non_empty = [line for line in cleaned.split('\n') if line.strip()]
        first_non_empty = non_empty[0] if non_empty else ''
        dedented = textwrap.dedent(cleaned)

        # Only try to dedent when the snippet itself starts indented.
        if first_non_empty[:1].isspace() and dedented != cleaned:
            try:
                return (black.format_str(dedented, mode=black.Mode()), None)
            except Exception as dedent_error:
                return (None, str(dedent_error))

        return (None, str(first_error))
)";

        PyRun_SimpleString(FORMAT_CODE);

        PyObject* const main_module = PyImport_AddModule("__main__");
        if (!main_module) {
            return {code, "Failed to get __main__ module", false};
        }

        PyObject* const main_dict = PyModule_GetDict(main_module);
        PyObject* const format_func = PyDict_GetItemString(main_dict, "_lfs_format_code");
        if (!format_func || !PyCallable_Check(format_func)) {
            return {code, "Format function not found", false};
        }

        FormatResult result{code, "", false};
        PyObject* const py_code = PyUnicode_FromString(code.c_str());
        if (!py_code) {
            result.error = consume_python_error_detailed();
            return result;
        }
        PyObject* const py_result = PyObject_CallFunctionObjArgs(format_func, py_code, nullptr);
        Py_DECREF(py_code);

        if (!py_result) {
            result.error = consume_python_error_detailed();
            return result;
        }

        if (PyTuple_Check(py_result) && PyTuple_Size(py_result) == 2) {
            PyObject* formatted = PyTuple_GetItem(py_result, 0);
            PyObject* error = PyTuple_GetItem(py_result, 1);

            if (formatted && PyUnicode_Check(formatted)) {
                const char* const str = PyUnicode_AsUTF8(formatted);
                if (str) {
                    result.code = str;
                    result.success = true;
                }
            }

            if (error && !Py_IsNone(error) && PyUnicode_Check(error)) {
                const char* const err = PyUnicode_AsUTF8(error);
                if (err) {
                    result.error = err;
                    result.success = false;
                }
            }

            Py_DECREF(py_result);
        } else {
            result.error = std::format("Format function returned unexpected result of type {}",
                                       Py_TYPE(py_result)->tp_name ? Py_TYPE(py_result)->tp_name : "<unknown>");
            Py_DECREF(py_result);
        }

        return result;
    }

    // Frame callback for animations
    static std::function<void(float)> g_frame_callback;
    static std::mutex g_frame_mutex;

    void set_frame_callback(std::function<void(float)> callback) {
        std::lock_guard lock(g_frame_mutex);
        g_frame_callback = std::move(callback);
    }

    void clear_frame_callback() {
        std::lock_guard lock(g_frame_mutex);
        g_frame_callback = nullptr;
    }

    bool has_frame_callback() {
        std::lock_guard lock(g_frame_mutex);
        return g_frame_callback != nullptr;
    }

    void tick_frame_callback(float dt) {
        std::function<void(float)> cb;
        {
            std::lock_guard lock(g_frame_mutex);
            cb = g_frame_callback;
        }
        if (cb) {
            const GilAcquire gil;
            try {
                cb(dt);
            } catch (const std::exception& e) {
                LOG_ERROR("Frame callback error: {}", e.what());
            }
        }
    }

    CapabilityResult invoke_capability(const std::string& name, const std::string& args_json) {
        ensure_initialized();
        const GilAcquire gil;
        CapabilityResult result;

        PyObject* lichtfeld = import_lichtfeld_module("Failed to import lichtfeld while invoking capability");
        if (!lichtfeld) {
            return {false, "", "Failed to import lichtfeld"};
        }
        Py_DECREF(lichtfeld);

        ensure_plugins_loaded();

        PyObject* lfs_plugins = PyImport_ImportModule("lfs_plugins");
        if (!lfs_plugins) {
            PyErr_Print();
            return {false, "", "Failed to import lfs_plugins"};
        }

        PyObject* registry_class = PyObject_GetAttrString(lfs_plugins, "CapabilityRegistry");
        if (!registry_class) {
            Py_DECREF(lfs_plugins);
            return {false, "", "CapabilityRegistry not found"};
        }

        PyObject* instance_method = PyObject_GetAttrString(registry_class, "instance");
        PyObject* registry = PyObject_CallNoArgs(instance_method);
        Py_DECREF(instance_method);
        Py_DECREF(registry_class);

        if (!registry) {
            Py_DECREF(lfs_plugins);
            return {false, "", "Failed to get capability registry instance"};
        }

        PyObject* json_module = PyImport_ImportModule("json");
        PyObject* loads = PyObject_GetAttrString(json_module, "loads");
        PyObject* dumps = PyObject_GetAttrString(json_module, "dumps");
        PyObject* py_args_str = PyUnicode_FromString(args_json.c_str());
        PyObject* args_dict = PyObject_CallOneArg(loads, py_args_str);
        Py_DECREF(py_args_str);

        if (!args_dict) {
            PyErr_Clear();
            args_dict = PyDict_New();
        }

        PyObject* invoke_method = PyObject_GetAttrString(registry, "invoke");
        PyObject* py_name = PyUnicode_FromString(name.c_str());
        PyObject* py_result = PyObject_CallFunctionObjArgs(invoke_method, py_name, args_dict, nullptr);
        Py_DECREF(py_name);
        Py_DECREF(args_dict);
        Py_DECREF(invoke_method);

        if (py_result && PyDict_Check(py_result)) {
            PyObject* success = PyDict_GetItemString(py_result, "success");
            result.success = success && PyObject_IsTrue(success);

            if (!result.success) {
                PyObject* error = PyDict_GetItemString(py_result, "error");
                if (error && PyUnicode_Check(error)) {
                    result.error = PyUnicode_AsUTF8(error);
                }
            }

            PyObject* json_str = PyObject_CallOneArg(dumps, py_result);
            if (json_str) {
                result.result_json = PyUnicode_AsUTF8(json_str);
                Py_DECREF(json_str);
            }
            Py_DECREF(py_result);
        } else {
            if (PyErr_Occurred())
                PyErr_Print();
            result = {false, "", "Capability invocation failed"};
        }

        Py_DECREF(dumps);
        Py_DECREF(loads);
        Py_DECREF(json_module);
        Py_DECREF(registry);
        Py_DECREF(lfs_plugins);
        return result;
    }

    bool has_capability(const std::string& name) {
        ensure_initialized();
        ensure_plugins_loaded();
        const GilAcquire gil;
        bool result = false;

        PyObject* lfs_plugins = PyImport_ImportModule("lfs_plugins");
        if (lfs_plugins) {
            PyObject* registry_class = PyObject_GetAttrString(lfs_plugins, "CapabilityRegistry");
            if (registry_class) {
                PyObject* instance_method = PyObject_GetAttrString(registry_class, "instance");
                PyObject* registry = PyObject_CallNoArgs(instance_method);
                if (registry) {
                    PyObject* has_method = PyObject_GetAttrString(registry, "has");
                    PyObject* py_name = PyUnicode_FromString(name.c_str());
                    PyObject* py_result = PyObject_CallOneArg(has_method, py_name);
                    if (py_result) {
                        result = PyObject_IsTrue(py_result);
                        Py_DECREF(py_result);
                    }
                    Py_DECREF(py_name);
                    Py_DECREF(has_method);
                    Py_DECREF(registry);
                }
                Py_DECREF(instance_method);
                Py_DECREF(registry_class);
            }
            Py_DECREF(lfs_plugins);
        }

        return result;
    }

    std::vector<CapabilityInfo> list_capabilities() {
        std::vector<CapabilityInfo> result;
        ensure_initialized();
        ensure_plugins_loaded();
        const GilAcquire gil;

        PyObject* lfs_plugins = PyImport_ImportModule("lfs_plugins");
        if (lfs_plugins) {
            PyObject* registry_class = PyObject_GetAttrString(lfs_plugins, "CapabilityRegistry");
            if (registry_class) {
                PyObject* instance_method = PyObject_GetAttrString(registry_class, "instance");
                PyObject* registry = PyObject_CallNoArgs(instance_method);
                if (registry) {
                    PyObject* list_method = PyObject_GetAttrString(registry, "list_all");
                    PyObject* caps = PyObject_CallNoArgs(list_method);
                    if (caps && PyList_Check(caps)) {
                        const Py_ssize_t n = PyList_Size(caps);
                        for (Py_ssize_t i = 0; i < n; ++i) {
                            PyObject* cap = PyList_GetItem(caps, i);
                            CapabilityInfo info;

                            PyObject* name = PyObject_GetAttrString(cap, "name");
                            if (name && PyUnicode_Check(name))
                                info.name = PyUnicode_AsUTF8(name);
                            Py_XDECREF(name);

                            PyObject* desc = PyObject_GetAttrString(cap, "description");
                            if (desc && PyUnicode_Check(desc))
                                info.description = PyUnicode_AsUTF8(desc);
                            Py_XDECREF(desc);

                            PyObject* plugin = PyObject_GetAttrString(cap, "plugin_name");
                            if (plugin && PyUnicode_Check(plugin))
                                info.plugin_name = PyUnicode_AsUTF8(plugin);
                            Py_XDECREF(plugin);

                            result.push_back(info);
                        }
                        Py_DECREF(caps);
                    }
                    Py_DECREF(list_method);
                    Py_DECREF(registry);
                }
                Py_DECREF(instance_method);
                Py_DECREF(registry_class);
            }
            Py_DECREF(lfs_plugins);
        }

        return result;
    }

} // namespace lfs::python
