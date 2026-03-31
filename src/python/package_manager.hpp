/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "uv_runner.hpp"

#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace lfs::python {

    struct PackageInfo {
        std::string name;
        std::string version;
        std::string path;
    };

    struct InstallResult {
        bool success = false;
        std::string output;
        std::string error;
    };

    class PackageManager {
    public:
        static PackageManager& instance();

        std::filesystem::path uv_path() const;
        bool is_uv_available() const;

        std::filesystem::path root_dir() const;
        std::filesystem::path venv_dir() const;
        std::filesystem::path venv_python() const;
        std::filesystem::path site_packages_dir() const;
        bool ensure_venv();
        bool is_venv_ready() const;

        // Synchronous operations (blocking)
        InstallResult install(const std::string& package);
        InstallResult uninstall(const std::string& package);
        InstallResult install_torch(const std::string& cuda_version = "auto",
                                    const std::string& torch_version = "");

        // Async operations (non-blocking, call poll() in render loop)
        bool install_async(const std::string& package,
                           UvRunner::OutputCallback on_output = nullptr,
                           UvRunner::CompletionCallback on_complete = nullptr);

        bool uninstall_async(const std::string& package,
                             UvRunner::OutputCallback on_output = nullptr,
                             UvRunner::CompletionCallback on_complete = nullptr);

        bool install_torch_async(const std::string& cuda_version = "auto",
                                 const std::string& torch_version = "",
                                 UvRunner::OutputCallback on_output = nullptr,
                                 UvRunner::CompletionCallback on_complete = nullptr);

        bool install_async_raw(const std::string& package,
                               UvRunner::RawOutputCallback on_output = nullptr,
                               UvRunner::CompletionCallback on_complete = nullptr);

        bool install_torch_async_raw(const std::string& cuda_version = "auto",
                                     const std::string& torch_version = "",
                                     UvRunner::RawOutputCallback on_output = nullptr,
                                     UvRunner::CompletionCallback on_complete = nullptr);

        // Poll for async operation progress (call from render loop)
        // Returns true if operation is still running
        bool poll();

        // Cancel current async operation
        void cancel_async();

        // Check if async operation is running
        [[nodiscard]] bool has_running_operation() const;

        std::vector<PackageInfo> list_installed() const;
        bool is_installed(const std::string& package) const;

    private:
        PackageManager();
        InstallResult execute_uv(const std::vector<std::string>& args) const;

        std::filesystem::path m_root_dir;
        std::filesystem::path m_venv_dir;
        mutable std::mutex m_mutex;
        mutable bool m_venv_ready = false;

        std::unique_ptr<UvRunner> m_runner;
    };

} // namespace lfs::python
