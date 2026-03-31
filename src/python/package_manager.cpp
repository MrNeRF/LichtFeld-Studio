/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "package_manager.hpp"
#include "windows_process_utils.hpp"

#include <core/cuda_version.hpp>
#include <core/executable_path.hpp>
#include <core/logger.hpp>
#include <core/path_utils.hpp>

#include <algorithm>
#include <cstdio>
#include <optional>
#include <regex>
#include <sstream>
#include <vector>

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

        std::pair<int, std::string> execute_process_capture(
            const std::filesystem::path& program,
            const std::vector<std::string>& args,
            const std::optional<std::pair<std::string, std::string>>& env_override = std::nullopt) {
            std::string output;
            int exit_code = -1;

#ifdef _WIN32
            auto build_environment_block =
                [&](const std::optional<std::pair<std::string, std::string>>& override_pair) {
                    std::vector<wchar_t> block;
                    if (!override_pair) {
                        return block;
                    }

                    std::vector<std::wstring> entries;
                    if (LPWCH env = GetEnvironmentStringsW()) {
                        for (const wchar_t* current = env; *current; current += wcslen(current) + 1) {
                            entries.emplace_back(current);
                        }
                        FreeEnvironmentStringsW(env);
                    }

                    const std::wstring key = lfs::core::utf8_to_wstring(override_pair->first);
                    const std::wstring value = lfs::core::utf8_to_wstring(override_pair->second);
                    const std::wstring entry = key + L"=" + value;

                    bool replaced = false;
                    for (auto& existing : entries) {
                        const size_t pos = existing.find(L'=');
                        if (pos == std::wstring::npos) {
                            continue;
                        }
                        const std::wstring name = existing.substr(0, pos);
                        if (_wcsicmp(name.c_str(), key.c_str()) == 0) {
                            existing = entry;
                            replaced = true;
                            break;
                        }
                    }
                    if (!replaced) {
                        entries.push_back(entry);
                    }

                    std::sort(entries.begin(), entries.end(), [](const std::wstring& a, const std::wstring& b) {
                        return _wcsicmp(a.c_str(), b.c_str()) < 0;
                    });

                    for (const auto& existing : entries) {
                        block.insert(block.end(), existing.begin(), existing.end());
                        block.push_back(L'\0');
                    }
                    block.push_back(L'\0');
                    return block;
                };

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
            const std::wstring program_w = program.wstring();
            if (program_w.empty()) {
                CloseHandle(hReadPipe);
                CloseHandle(hWritePipe);
                return {-1, "Failed to resolve process path"};
            }

            std::wstring cmdline = detail::build_win32_cmdline(program, args);

            std::vector<wchar_t> environment = build_environment_block(env_override);
            void* environment_ptr = environment.empty() ? nullptr : environment.data();

            // Keep lpApplicationName and argv[0] aligned to the same executable path.
            if (!CreateProcessW(program_w.c_str(), cmdline.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW,
                                environment_ptr, nullptr, &si, &pi)) {
                CloseHandle(hReadPipe);
                CloseHandle(hWritePipe);
                return {-1, std::format("Failed to create process ({})", GetLastError())};
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
            int pipe_fds[2];
            if (pipe(pipe_fds) != 0) {
                return {-1, "Failed to create pipe"};
            }

            const pid_t pid = fork();
            if (pid == -1) {
                close(pipe_fds[0]);
                close(pipe_fds[1]);
                return {-1, "Failed to fork process"};
            }

            if (pid == 0) {
                dup2(pipe_fds[1], STDOUT_FILENO);
                dup2(pipe_fds[1], STDERR_FILENO);
                close(pipe_fds[0]);
                close(pipe_fds[1]);

                if (env_override) {
                    setenv(env_override->first.c_str(), env_override->second.c_str(), 1);
                }

                const std::string program_utf8 = lfs::core::path_to_utf8(program);
                std::vector<char*> argv;
                argv.reserve(args.size() + 2);
                argv.push_back(const_cast<char*>(program_utf8.c_str()));
                for (const auto& arg : args) {
                    argv.push_back(const_cast<char*>(arg.c_str()));
                }
                argv.push_back(nullptr);

                execvp(program_utf8.c_str(), argv.data());
                _exit(127);
            }

            close(pipe_fds[1]);

            char buffer[4096];
            ssize_t bytes_read = 0;
            while ((bytes_read = read(pipe_fds[0], buffer, sizeof(buffer))) > 0) {
                output.append(buffer, static_cast<size_t>(bytes_read));
            }
            close(pipe_fds[0]);

            int status = 0;
            if (waitpid(pid, &status, 0) == pid && WIFEXITED(status)) {
                exit_code = WEXITSTATUS(status);
            } else if (WIFSIGNALED(status)) {
                exit_code = 128 + WTERMSIG(status);
            }
#endif

            return {exit_code, output};
        }

        std::filesystem::path get_lichtfeld_dir() {
#ifdef _WIN32
            const DWORD size = GetEnvironmentVariableW(L"USERPROFILE", nullptr, 0);
            if (size > 0) {
                std::wstring home(size - 1, L'\0');
                if (GetEnvironmentVariableW(L"USERPROFILE", home.data(), size) == size - 1) {
                    return std::filesystem::path(home) / ".lichtfeld";
                }
            }
            return std::filesystem::temp_directory_path() / "lichtfeld";
#else
            const char* const home = std::getenv("HOME");
            return std::filesystem::path(home ? home : "/tmp") / ".lichtfeld";
#endif
        }

    } // namespace

    PackageManager::PackageManager()
        : m_root_dir(get_lichtfeld_dir()),
          m_venv_dir(m_root_dir / "venv") {}

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
        }

        if (cached.empty())
            LOG_WARN("Bundled uv not found");
        else if (bundled)
            LOG_INFO("Using bundled uv: {}", lfs::core::path_to_utf8(cached));

        return cached;
    }

    bool PackageManager::is_uv_available() const {
        return !uv_path().empty();
    }

    std::filesystem::path PackageManager::root_dir() const {
        return m_root_dir;
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

        if (std::filesystem::exists(venv_python())) {
            LOG_INFO("Existing venv found: {}", lfs::core::path_to_utf8(venv_python()));
            m_venv_ready = true;
            return true;
        }

        if (std::filesystem::exists(m_venv_dir) && !std::filesystem::exists(venv_python())) {
            LOG_WARN("Broken venv (missing python), removing: {}", lfs::core::path_to_utf8(m_venv_dir));
            std::filesystem::remove_all(m_venv_dir);
        }

        const auto uv = uv_path();
        if (uv.empty()) {
            LOG_ERROR("uv not found, cannot create venv");
            return false;
        }

        const auto embedded_python = lfs::core::getEmbeddedPython();
        if (embedded_python.empty()) {
            LOG_ERROR("Embedded Python not found (exe_dir={})",
                      lfs::core::path_to_utf8(lfs::core::getExecutableDir()));
            return false;
        }

        LOG_INFO("Creating venv at {} with {}",
                 lfs::core::path_to_utf8(m_venv_dir),
                 lfs::core::path_to_utf8(embedded_python));

        const auto python_home = lfs::core::getPythonHome();
        LOG_INFO("Python home: {}", python_home.empty() ? "(empty)" : lfs::core::path_to_utf8(python_home));

        std::vector<std::string> args = {
            "venv",
            lfs::core::path_to_utf8(m_venv_dir),
            "--python",
            lfs::core::path_to_utf8(embedded_python),
            "--no-managed-python",
            "--no-python-downloads"};

        std::optional<std::pair<std::string, std::string>> env_override;
        if (!python_home.empty()) {
            env_override = std::make_pair(std::string("PYTHONHOME"), lfs::core::path_to_utf8(python_home));
        }

        LOG_INFO("Executing uv venv create for {}", lfs::core::path_to_utf8(m_venv_dir));
        LOG_DEBUG("UV command: {}", detail::format_command_for_log(uv, args));
        const auto [exit_code, output] = execute_process_capture(uv, args, env_override);

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

        LOG_INFO("Executing uv command with {} args", args.size());
        LOG_DEBUG("UV command: {}", detail::format_command_for_log(uv, args));
        const auto [exit_code, output] = execute_process_capture(uv, args);

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
        return execute_uv({"pip", "install", package, "--python", lfs::core::path_to_utf8(venv_python())});
    }

    InstallResult PackageManager::uninstall(const std::string& package) {
        if (!ensure_venv())
            return {.error = "Failed to initialize venv"};

        std::lock_guard lock(m_mutex);
        LOG_INFO("Uninstalling {}", package);
        return execute_uv({"pip", "uninstall", package, "--python", lfs::core::path_to_utf8(venv_python())});
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
                                         "--python", lfs::core::path_to_utf8(venv_python())};
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
                    {.name = pkg_name, .version = match[2].str(), .path = lfs::core::path_to_utf8(pkg_path)});
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

        return m_runner->start({"pip", "install", package, "--python", lfs::core::path_to_utf8(venv_python())});
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

        return m_runner->start({"pip", "uninstall", package, "-y", "--python", lfs::core::path_to_utf8(venv_python())});
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
                                         "--python", lfs::core::path_to_utf8(venv_python())};
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

        return m_runner->start({"pip", "install", package, "--python", lfs::core::path_to_utf8(venv_python())});
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
                                         "--python", lfs::core::path_to_utf8(venv_python())};
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
