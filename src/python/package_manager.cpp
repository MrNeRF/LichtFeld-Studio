/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "package_manager.hpp"

#include <core/cuda_version.hpp>
#include <core/executable_path.hpp>
#include <core/logger.hpp>

#include <cstdio>
#include <regex>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace lfs::python {

    namespace {

#ifdef _WIN32
        constexpr const char* UV_BINARY = "uv.exe";
        constexpr size_t MAX_PATH_LEN = MAX_PATH;
#else
        constexpr const char* UV_BINARY = "uv";
        constexpr size_t MAX_PATH_LEN = 4096;
#endif
        constexpr const char* PYTORCH_INDEX = "https://download.pytorch.org/whl/";

        std::filesystem::path get_executable_dir() {
#ifdef _WIN32
            wchar_t path[MAX_PATH_LEN];
            GetModuleFileNameW(nullptr, path, MAX_PATH_LEN);
            return std::filesystem::path(path).parent_path();
#else
            char path[MAX_PATH_LEN];
            const ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
            if (len != -1) {
                path[len] = '\0';
                return std::filesystem::path(path).parent_path();
            }
            return std::filesystem::current_path();
#endif
        }

        std::pair<int, std::string> execute_command(const std::string& cmd) {
            std::string output;
            int exit_code = -1;

#ifdef _WIN32
            SECURITY_ATTRIBUTES sa;
            sa.nLength = sizeof(SECURITY_ATTRIBUTES);
            sa.bInheritHandle = TRUE;
            sa.lpSecurityDescriptor = nullptr;

            HANDLE hReadPipe, hWritePipe;
            if (!CreatePipe(&hReadPipe, &hWritePipe, &sa, 0)) {
                return {-1, "Failed to create pipe"};
            }

            SetHandleInformation(hReadPipe, HANDLE_FLAG_INHERIT, 0);

            STARTUPINFOW si = {};
            si.cb = sizeof(si);
            si.hStdOutput = hWritePipe;
            si.hStdError = hWritePipe;
            si.dwFlags |= STARTF_USESTDHANDLES;

            PROCESS_INFORMATION pi = {};

            std::wstring wcmd(cmd.begin(), cmd.end());
            if (!CreateProcessW(nullptr, wcmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) {
                CloseHandle(hReadPipe);
                CloseHandle(hWritePipe);
                return {-1, "Failed to create process"};
            }

            CloseHandle(hWritePipe);

            char buffer[4096];
            DWORD bytesRead;
            while (ReadFile(hReadPipe, buffer, sizeof(buffer) - 1, &bytesRead, nullptr) && bytesRead > 0) {
                buffer[bytesRead] = '\0';
                output += buffer;
            }

            WaitForSingleObject(pi.hProcess, INFINITE);

            DWORD exitCodeDword;
            GetExitCodeProcess(pi.hProcess, &exitCodeDword);
            exit_code = static_cast<int>(exitCodeDword);

            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);
            CloseHandle(hReadPipe);
#else
            FILE* pipe = popen((cmd + " 2>&1").c_str(), "r");
            if (!pipe) {
                return {-1, "Failed to execute command"};
            }

            char buffer[4096];
            while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                output += buffer;
            }

            exit_code = pclose(pipe);
            if (WIFEXITED(exit_code)) {
                exit_code = WEXITSTATUS(exit_code);
            }
#endif

            return {exit_code, output};
        }

        std::filesystem::path get_lichtfeld_dir() {
#ifdef _WIN32
            const char* const home = std::getenv("USERPROFILE");
#else
            const char* const home = std::getenv("HOME");
#endif
            return std::filesystem::path(home ? home : "/tmp") / ".lichtfeld";
        }

    } // namespace

    PackageManager::PackageManager() : m_venv_dir(get_lichtfeld_dir() / "venv") {}

    PackageManager& PackageManager::instance() {
        static PackageManager inst;
        return inst;
    }

    std::filesystem::path PackageManager::uv_path() const {
        static std::filesystem::path cached;
        static bool searched = false;

        if (searched)
            return cached;
        searched = true;

        const auto exe_dir = get_executable_dir();
        bool bundled = false;

        if (const auto p = exe_dir / "bin" / UV_BINARY; std::filesystem::exists(p)) {
            cached = p;
            bundled = true;
        } else if (const auto p = exe_dir / UV_BINARY; std::filesystem::exists(p)) {
            cached = p;
            bundled = true;
        } else {
#ifdef _WIN32
            auto [exit_code, result] = execute_command("where uv");
#else
            auto [exit_code, result] = execute_command("which uv");
#endif
            if (exit_code == 0 && !result.empty()) {
                while (!result.empty() && (result.back() == '\n' || result.back() == '\r'))
                    result.pop_back();
                if (std::filesystem::path found(result); std::filesystem::exists(found))
                    cached = found;
            }
        }

        if (cached.empty())
            LOG_WARN("uv not found");
        else if (bundled)
            LOG_INFO("Using uv: {}", cached.string());
        else
            LOG_WARN("Using system uv: {} (bundled version not found)", cached.string());

        return cached;
    }

    bool PackageManager::is_uv_available() const {
        return !uv_path().empty();
    }

    std::filesystem::path PackageManager::venv_dir() const {
        return m_venv_dir;
    }

    std::filesystem::path PackageManager::venv_python() const {
#ifdef _WIN32
        return m_venv_dir / "Scripts" / "python.exe";
#else
        return m_venv_dir / "bin" / "python";
#endif
    }

    bool PackageManager::is_venv_ready() const {
        return m_venv_ready && std::filesystem::exists(venv_python());
    }

    bool PackageManager::ensure_venv() {
        std::lock_guard lock(m_mutex);

        if (m_venv_ready && std::filesystem::exists(venv_python()))
            return true;

        const auto uv = uv_path();
        if (uv.empty()) {
            LOG_ERROR("uv not found");
            return false;
        }

        if (std::filesystem::exists(venv_python())) {
            m_venv_ready = true;
            return true;
        }

        LOG_INFO("Creating venv at {}", m_venv_dir.string());

        std::ostringstream cmd;
        cmd << "\"" << uv.string() << "\" venv \"" << m_venv_dir.string() << "\" --allow-existing";

        const auto embedded_python = lfs::core::getEmbeddedPython();
        if (!embedded_python.empty()) {
            cmd << " --python \"" << embedded_python.string() << "\"";
            LOG_INFO("Using embedded Python for venv: {}", embedded_python.string());
        } else {
            LOG_WARN("Embedded Python not found (exe_dir={}), uv will use system Python",
                     lfs::core::getExecutableDir().string());
        }

        LOG_INFO("Executing: {}", cmd.str());
        const auto [exit_code, output] = execute_command(cmd.str());
        if (exit_code != 0) {
            LOG_ERROR("Failed to create venv: {}", output);
            return false;
        }

        m_venv_ready = true;
        return true;
    }

    std::filesystem::path PackageManager::site_packages_dir() const {
#ifdef _WIN32
        return m_venv_dir / "Lib" / "site-packages";
#else
        const auto lib_dir = m_venv_dir / "lib";
        if (std::filesystem::exists(lib_dir)) {
            std::filesystem::path best_match;
            for (const auto& entry : std::filesystem::directory_iterator(lib_dir)) {
                if (entry.is_directory()) {
                    const auto name = entry.path().filename().string();
                    // Prefer pythonX.Y over pythonX (more specific version)
                    if (name.find("python") == 0) {
                        if (best_match.empty() || name.length() > best_match.filename().string().length()) {
                            best_match = entry.path();
                        }
                    }
                }
            }
            if (!best_match.empty())
                return best_match / "site-packages";
        }
        return m_venv_dir / "lib" / "python3" / "site-packages";
#endif
    }

    InstallResult PackageManager::execute_uv(const std::vector<std::string>& args) const {
        const auto uv = uv_path();
        if (uv.empty())
            return {.error = "uv not found"};

        std::ostringstream cmd;
        cmd << "\"" << uv.string() << "\"";
        for (const auto& arg : args)
            cmd << " " << arg;

        const auto [exit_code, output] = execute_command(cmd.str());

        InstallResult result;
        result.output = output;
        result.success = (exit_code == 0);
        if (!result.success)
            result.error = output.empty() ? "Exit code " + std::to_string(exit_code) : output;
        return result;
    }

    InstallResult PackageManager::install(const std::string& package) {
        if (!ensure_venv())
            return {.error = "Failed to create venv"};

        std::lock_guard lock(m_mutex);
        LOG_INFO("Installing {}", package);
        return execute_uv({"pip", "install", package, "--python", venv_python().string()});
    }

    InstallResult PackageManager::uninstall(const std::string& package) {
        if (!ensure_venv())
            return {.error = "Failed to initialize venv"};

        std::lock_guard lock(m_mutex);
        LOG_INFO("Uninstalling {}", package);
        return execute_uv({"pip", "uninstall", package, "--python", venv_python().string()});
    }

    InstallResult PackageManager::install_torch(const std::string& cuda_version,
                                                const std::string& torch_version) {
        if (!ensure_venv())
            return {.error = "Failed to create venv"};

        const std::string cuda_tag = core::get_pytorch_cuda_tag(cuda_version);
        LOG_INFO("PyTorch CUDA tag: {}", cuda_tag);

        std::string package = "torch";
        if (!torch_version.empty())
            package += "==" + torch_version;

        const std::string index_url = std::string(PYTORCH_INDEX) + cuda_tag;

        std::lock_guard lock(m_mutex);
        LOG_INFO("Installing {} from {}", package, cuda_tag);

        std::vector<std::string> args = {"pip", "install", package, "--extra-index-url", index_url,
                                         "--python", venv_python().string()};
        if (torch_version.empty())
            args.push_back("--upgrade");

        return execute_uv(args);
    }

    std::vector<PackageInfo> PackageManager::list_installed() const {
        std::lock_guard lock(m_mutex);
        std::vector<PackageInfo> packages;

        const auto site_dir = site_packages_dir();
        if (!std::filesystem::exists(site_dir))
            return packages;

        static const std::regex DIST_INFO_PATTERN(R"((.+)-(.+)\.dist-info)");

        for (const auto& entry : std::filesystem::directory_iterator(site_dir)) {
            if (!entry.is_directory())
                continue;

            const auto name = entry.path().filename().string();
            if (name.find(".dist-info") == std::string::npos)
                continue;

            std::smatch match;
            if (std::regex_match(name, match, DIST_INFO_PATTERN)) {
                const std::string pkg_name = match[1].str();
                std::filesystem::path pkg_path = site_dir / pkg_name;

                if (!std::filesystem::exists(pkg_path)) {
                    std::string normalized = pkg_name;
                    std::replace(normalized.begin(), normalized.end(), '-', '_');
                    const auto alt_path = site_dir / normalized;
                    pkg_path = std::filesystem::exists(alt_path) ? alt_path : site_dir;
                }

                packages.push_back(
                    {.name = pkg_name, .version = match[2].str(), .path = pkg_path.string()});
            }
        }
        return packages;
    }

    bool PackageManager::is_installed(const std::string& package) const {
        const auto packages = list_installed();
        auto normalize = [](std::string s) {
            std::replace(s.begin(), s.end(), '-', '_');
            return s;
        };
        const auto normalized = normalize(package);

        for (const auto& pkg : packages) {
            if (pkg.name == package || normalize(pkg.name) == normalized)
                return true;
        }
        return false;
    }

    bool PackageManager::install_async(const std::string& package,
                                       UvRunner::OutputCallback on_output,
                                       UvRunner::CompletionCallback on_complete) {
        if (!ensure_venv())
            return false;

        if (!m_runner) {
            m_runner = std::make_unique<UvRunner>();
        }

        if (m_runner->is_running()) {
            LOG_ERROR("Another UV operation is already running");
            return false;
        }

        LOG_INFO("Installing {} (async)", package);

        m_runner->set_output_callback(std::move(on_output));
        m_runner->set_completion_callback(std::move(on_complete));

        return m_runner->start({"pip", "install", package, "--python", venv_python().string()});
    }

    bool PackageManager::uninstall_async(const std::string& package,
                                         UvRunner::OutputCallback on_output,
                                         UvRunner::CompletionCallback on_complete) {
        if (!ensure_venv())
            return false;

        if (!m_runner) {
            m_runner = std::make_unique<UvRunner>();
        }

        if (m_runner->is_running()) {
            LOG_ERROR("Another UV operation is already running");
            return false;
        }

        LOG_INFO("Uninstalling {} (async)", package);

        m_runner->set_output_callback(std::move(on_output));
        m_runner->set_completion_callback(std::move(on_complete));

        return m_runner->start({"pip", "uninstall", package, "-y", "--python", venv_python().string()});
    }

    bool PackageManager::install_torch_async(const std::string& cuda_version,
                                             const std::string& torch_version,
                                             UvRunner::OutputCallback on_output,
                                             UvRunner::CompletionCallback on_complete) {
        if (!ensure_venv())
            return false;

        if (!m_runner) {
            m_runner = std::make_unique<UvRunner>();
        }

        if (m_runner->is_running()) {
            LOG_ERROR("Another UV operation is already running");
            return false;
        }

        const std::string cuda_tag = core::get_pytorch_cuda_tag(cuda_version);
        LOG_INFO("PyTorch CUDA tag (async): {}", cuda_tag);

        std::string package = "torch";
        if (!torch_version.empty())
            package += "==" + torch_version;

        const std::string index_url = std::string(PYTORCH_INDEX) + cuda_tag;

        LOG_INFO("Installing {} from {} (async)", package, cuda_tag);

        m_runner->set_output_callback(std::move(on_output));
        m_runner->set_completion_callback(std::move(on_complete));

        std::vector<std::string> args = {"pip", "install", package, "--extra-index-url", index_url,
                                         "--python", venv_python().string()};
        if (torch_version.empty())
            args.push_back("--upgrade");

        return m_runner->start(args);
    }

    bool PackageManager::install_async_raw(const std::string& package,
                                           UvRunner::RawOutputCallback on_output,
                                           UvRunner::CompletionCallback on_complete) {
        if (!ensure_venv())
            return false;

        if (!m_runner) {
            m_runner = std::make_unique<UvRunner>();
        }

        if (m_runner->is_running()) {
            LOG_ERROR("Another UV operation is already running");
            return false;
        }

        LOG_INFO("Installing {} (async raw)", package);

        m_runner->set_raw_output_callback(std::move(on_output));
        m_runner->set_completion_callback(std::move(on_complete));

        return m_runner->start({"pip", "install", package, "--python", venv_python().string()});
    }

    bool PackageManager::install_torch_async_raw(const std::string& cuda_version,
                                                 const std::string& torch_version,
                                                 UvRunner::RawOutputCallback on_output,
                                                 UvRunner::CompletionCallback on_complete) {
        if (!ensure_venv())
            return false;

        if (!m_runner) {
            m_runner = std::make_unique<UvRunner>();
        }

        if (m_runner->is_running()) {
            LOG_ERROR("Another UV operation is already running");
            return false;
        }

        const std::string cuda_tag = core::get_pytorch_cuda_tag(cuda_version);
        LOG_INFO("PyTorch CUDA tag (async raw): {}", cuda_tag);

        std::string package = "torch";
        if (!torch_version.empty())
            package += "==" + torch_version;

        const std::string index_url = std::string(PYTORCH_INDEX) + cuda_tag;

        LOG_INFO("Installing {} from {} (async raw)", package, cuda_tag);

        m_runner->set_raw_output_callback(std::move(on_output));
        m_runner->set_completion_callback(std::move(on_complete));

        std::vector<std::string> args = {"pip", "install", package, "--extra-index-url", index_url,
                                         "--python", venv_python().string()};
        if (torch_version.empty())
            args.push_back("--upgrade");

        return m_runner->start(args);
    }

    bool PackageManager::poll() {
        if (!m_runner) {
            return false;
        }
        return m_runner->poll();
    }

    void PackageManager::cancel_async() {
        if (m_runner) {
            m_runner->cancel();
        }
    }

    bool PackageManager::has_running_operation() const {
        return m_runner && m_runner->is_running();
    }

} // namespace lfs::python
