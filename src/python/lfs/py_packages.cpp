/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_packages.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "../package_manager.hpp"
#include "../runner.hpp"
#include "core/executable_path.hpp"
#include <core/logger.hpp>

namespace lfs::python {

    void register_packages(nb::module_& m) {
        auto pkg = m.def_submodule("packages", "Package management via uv");

        nb::class_<PackageInfo>(pkg, "PackageInfo")
            .def_ro("name", &PackageInfo::name)
            .def_ro("version", &PackageInfo::version)
            .def_ro("path", &PackageInfo::path)
            .def("__repr__", [](const PackageInfo& p) { return p.name + "==" + p.version; });

        pkg.def(
            "init",
            []() {
                auto& pm = PackageManager::instance();
                if (!pm.is_uv_available())
                    throw std::runtime_error("uv not found");
                if (!pm.ensure_venv())
                    throw std::runtime_error("Failed to create venv");
                update_python_path();
                return pm.venv_dir().string();
            },
            "Initialize venv at ~/.lichtfeld/venv");

        pkg.def(
            "install",
            [](const std::string& package) {
                auto& pm = PackageManager::instance();
                if (!pm.is_uv_available())
                    throw std::runtime_error("uv not found");
                const auto result = pm.install(package);
                if (!result.success)
                    throw std::runtime_error(result.error);
                return result.output;
            },
            nb::arg("package"), "Install package from PyPI");

        pkg.def(
            "uninstall",
            [](const std::string& package) {
                auto& pm = PackageManager::instance();
                const auto result = pm.uninstall(package);
                if (!result.success)
                    throw std::runtime_error(result.error);
                return result.output;
            },
            nb::arg("package"), "Uninstall package");

        pkg.def(
            "list",
            []() { return PackageManager::instance().list_installed(); },
            "List installed packages");

        pkg.def(
            "is_installed",
            [](const std::string& package) {
                return PackageManager::instance().is_installed(package);
            },
            nb::arg("package"), "Check if package is installed");

        pkg.def(
            "is_uv_available",
            []() { return PackageManager::instance().is_uv_available(); },
            "Check if uv is available");

        pkg.def(
            "uv_path",
            []() -> std::string {
                const auto path = PackageManager::instance().uv_path();
                return path.empty() ? "" : path.string();
            },
            "Get path to uv binary (empty string if not found)");

        pkg.def(
            "embedded_python_path",
            []() -> std::string {
                const auto p = lfs::core::getEmbeddedPython();
                if (p.empty()) {
                    LOG_WARN("Embedded Python not found (exe_dir={})", lfs::core::getExecutableDir().string());
                } else {
                    LOG_INFO("Embedded Python resolved: {}", p.string());
                }
                return p.empty() ? "" : p.string();
            },
            "Get path to embedded Python executable (empty string if not available)");

        pkg.def(
            "install_torch",
            [](const std::string& cuda, const std::string& version) {
                auto& pm = PackageManager::instance();
                if (!pm.is_uv_available())
                    throw std::runtime_error("uv not found");
                const auto result = pm.install_torch(cuda, version);
                if (!result.success)
                    throw std::runtime_error(result.error);
                return result.output;
            },
            nb::arg("cuda") = "auto", nb::arg("version") = "",
            "Install PyTorch with CUDA detection");

        pkg.def(
            "site_packages_dir",
            []() { return PackageManager::instance().site_packages_dir().string(); },
            "Get site-packages path");

        pkg.def(
            "install_async",
            [](const std::string& package) {
                auto& pm = PackageManager::instance();
                if (!pm.is_uv_available())
                    throw std::runtime_error("uv not found");
                if (!pm.ensure_venv())
                    throw std::runtime_error("Failed to create venv");

                return pm.install_async_raw(
                    package,
                    [](const std::string& data) { write_output(data, false); },
                    [package](bool success, int exit_code) {
                        if (success) {
                            write_output("\nInstalled " + package + "\n", false);
                        } else {
                            write_output("\nFailed to install " + package + " (exit " +
                                             std::to_string(exit_code) + ")\n",
                                         true);
                        }
                    });
            },
            nb::arg("package"), "Install package asynchronously (non-blocking)");

        pkg.def(
            "install_torch_async",
            [](const std::string& cuda, const std::string& version) {
                auto& pm = PackageManager::instance();
                if (!pm.is_uv_available())
                    throw std::runtime_error("uv not found");
                if (!pm.ensure_venv())
                    throw std::runtime_error("Failed to create venv");

                return pm.install_torch_async_raw(
                    cuda, version,
                    [](const std::string& data) { write_output(data, false); },
                    [](bool success, int exit_code) {
                        if (success) {
                            write_output("\nPyTorch installed successfully\n", false);
                        } else {
                            write_output("\nFailed to install PyTorch (exit " +
                                             std::to_string(exit_code) + ")\n",
                                         true);
                        }
                    });
            },
            nb::arg("cuda") = "auto", nb::arg("version") = "",
            "Install PyTorch asynchronously (non-blocking)");

        pkg.def(
            "is_busy",
            []() { return PackageManager::instance().has_running_operation(); },
            "Check if async operation is running");

        pkg.def(
            "typings_dir",
            []() -> std::string {
                const auto dir = lfs::core::getTypingsDir();
                return dir.empty() ? "" : dir.string();
            },
            "Get path to type stubs directory (empty if not found)");
    }

} // namespace lfs::python
