/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "runner.hpp"
#include "package_manager.hpp"

#include <cstdio>
#include <filesystem>
#include <format>
#include <string>
#include <thread>

#include <core/executable_path.hpp>
#include <core/logger.hpp>
#include <core/path_utils.hpp>

#include "gil.hpp"
#include "python_runtime.hpp"
#include "training/control/control_boundary.hpp"
#include <Python.h>
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

    static void ensure_plugins_loaded() {
        if (are_plugins_loaded())
            return;
        mark_plugins_loaded();

        add_dll_directories();

        LOG_INFO("Attempting to import lichtfeld module...");
        PyObject* lf = PyImport_ImportModule("lichtfeld");
        if (!lf) {
            LOG_ERROR("Failed to import lichtfeld: {}", extract_python_error());
            return;
        }
        LOG_INFO("lichtfeld module imported successfully");

        PyObject* lfs_plugins = PyImport_ImportModule("lfs_plugins");
        if (lfs_plugins) {
            PyObject* register_fn = PyObject_GetAttrString(lfs_plugins, "register_builtin_panels");
            if (register_fn) {
                PyObject* result = PyObject_CallNoArgs(register_fn);
                if (!result) {
                    PyErr_Print();
                    LOG_ERROR("Failed to register builtin panels");
                } else {
                    Py_DECREF(result);
                }
                Py_DECREF(register_fn);
            }
            Py_DECREF(lfs_plugins);
        }

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

        PyObject* plugins = PyObject_GetAttrString(lf, "plugins");
        if (plugins) {
            PyObject* load_all = PyObject_GetAttrString(plugins, "load_all");
            if (load_all) {
                PyObject* results = PyObject_CallNoArgs(load_all);
                if (results && PyDict_Check(results)) {
                    PyObject* key;
                    PyObject* value;
                    Py_ssize_t pos = 0;
                    while (PyDict_Next(results, &pos, &key, &value)) {
                        const char* name = PyUnicode_AsUTF8(key);
                        if (PyObject_IsTrue(value)) {
                            LOG_INFO("Loaded plugin: {}", name ? name : "<unknown>");
                        } else {
                            LOG_ERROR("Failed to load plugin: {}", name ? name : "<unknown>");
                            PyObject* get_traceback = PyObject_GetAttrString(plugins, "get_traceback");
                            if (get_traceback) {
                                PyObject* tb = PyObject_CallOneArg(get_traceback, key);
                                if (tb && !Py_IsNone(tb) && PyUnicode_Check(tb)) {
                                    LOG_ERROR("Traceback:\n{}", PyUnicode_AsUTF8(tb));
                                }
                                Py_XDECREF(tb);
                                Py_DECREF(get_traceback);
                            }
                        }
                    }
                } else if (!results) {
                    PyErr_Print();
                    LOG_ERROR("Plugin load_all() failed");
                }
                Py_XDECREF(results);
                Py_DECREF(load_all);
            }
            Py_DECREF(plugins);
        }

        Py_DECREF(lf);
    }

    std::filesystem::path get_user_packages_dir() {
        return PackageManager::instance().site_packages_dir();
    }

    void ensure_initialized() {
        call_once_py_init([] {
            if (!Py_IsInitialized()) {
                PyImport_AppendInittab("_lfs_output", init_capture_module);

                const auto python_home = lfs::core::getPythonHome();
                static std::wstring python_home_wstr;
                if (!python_home.empty()) {
                    python_home_wstr = python_home.wstring();
                    Py_SetPythonHome(python_home_wstr.c_str());
                    LOG_INFO("Set Python home: {}", lfs::core::path_to_utf8(python_home));
                }

                Py_Initialize();
                g_we_initialized_python = true;
                LOG_INFO("Python interpreter initialized by application");
            } else {
                LOG_WARN("Python already initialized by external code (e.g., .pyd loading)");
                g_we_initialized_python = false;
            }

            PyEval_InitThreads();
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
                const auto user_packages_utf8 = lfs::core::path_to_utf8(user_packages);
                PyObject* py_path = PyUnicode_FromString(user_packages_utf8.c_str());
                PyList_Insert(sys_path, 0, py_path);
                Py_DECREF(py_path);
                LOG_INFO("Added user packages dir to Python path: {}", user_packages_utf8);

                const auto python_module_dir = lfs::core::getPythonModuleDir();
                if (!python_module_dir.empty()) {
                    const auto python_module_dir_utf8 = lfs::core::path_to_utf8(python_module_dir);
                    PyObject* const py_mod_path = PyUnicode_FromString(python_module_dir_utf8.c_str());
                    PyList_Insert(sys_path, 0, py_mod_path);
                    Py_DECREF(py_mod_path);
                    LOG_INFO("Added Python module dir to path: {}", python_module_dir_utf8);
                } else {
                    const auto exe_dir_utf8 = lfs::core::path_to_utf8(lfs::core::getExecutableDir());
                    LOG_WARN("Python module (lichtfeld.pyd) not found. Searched: {}/src/python, {}",
                             exe_dir_utf8, exe_dir_utf8);
                }
            }

            ensure_plugins_loaded();

            set_main_thread_state(PyEval_SaveThread());
            set_gil_state_ready(true);
            LOG_DEBUG("GIL released, external_init={}", !g_we_initialized_python);
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

    void finalize() {
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
            PyObject* lf_module = PyImport_ImportModule("lichtfeld");
            if (!lf_module) {
                PyErr_Print();
                return std::unexpected("Failed to import lichtfeld module - check build output");
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
    import importlib, sys
    importlib.invalidate_caches()
    try:
        import black
    except ImportError as e:
        return (None, f"ImportError: {e}")

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
    lines = [line.replace('\t', '    ') for line in lines]

    # Find indentation levels for all non-empty lines
    indents = []
    for line in lines:
        if line.strip():
            indents.append(len(line) - len(line.lstrip()))

    if not indents:
        return (code, None)

    # If first line has 0 indent but others have consistent indent,
    # this is likely a copy-paste issue - use the mode of other indents
    min_indent = min(indents)
    if min_indent == 0 and len(indents) > 1:
        other_indents = [i for i in indents[1:] if i > 0]
        if other_indents:
            # Find the smallest non-zero indent from other lines
            min_other = min(other_indents)
            # Check if most lines use this or a multiple of it
            consistent = sum(1 for i in other_indents if i >= min_other) / len(other_indents)
            if consistent > 0.5:
                min_indent = min_other

    # Remove common indentation
    dedented = []
    for line in lines:
        if line.strip():
            current_indent = len(line) - len(line.lstrip())
            if current_indent >= min_indent:
                dedented.append(line[min_indent:])
            else:
                dedented.append(line.lstrip())  # Line has less indent, just strip it
        else:
            dedented.append('')

    cleaned = '\n'.join(dedented)

    try:
        return (black.format_str(cleaned, mode=black.Mode()), None)
    except Exception as e:
        return (None, str(e))
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
        PyObject* const py_result = PyObject_CallFunctionObjArgs(format_func, py_code, nullptr);
        Py_DECREF(py_code);

        if (py_result && PyTuple_Check(py_result) && PyTuple_Size(py_result) == 2) {
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
            if (PyErr_Occurred()) {
                PyErr_Clear();
            }
            result.error = "Format function returned unexpected result";
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

        PyObject* lichtfeld = PyImport_ImportModule("lichtfeld");
        if (!lichtfeld) {
            PyErr_Print();
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
