/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/batch_trainer.hpp"

#include "core/executable_path.hpp"
#include "core/logger.hpp"
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"
#include "io/loader.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <print>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include "python/windows_process_utils.hpp"
#else
#include <fcntl.h>
#include <poll.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace lfs::app {

    namespace {

        namespace fs = std::filesystem;
        using json = nlohmann::json;

        std::atomic<bool> g_batch_interrupted{false};
        std::atomic<int> g_interrupt_count{0};
#ifndef _WIN32
        std::atomic<pid_t> g_active_child_pid{0};
#else
        std::atomic<HANDLE> g_active_child_process{nullptr};
#endif

#ifndef _WIN32
        void handle_sigint(int sig) {
            const int count = ++g_interrupt_count;
            g_batch_interrupted.store(true);
            pid_t child = g_active_child_pid.load();
            if (child > 0) {
                if (count == 1) {
                    kill(child, sig);
                } else {
                    kill(child, SIGKILL);
                }
            }
        }
#else
        BOOL WINAPI handle_console_ctrl(DWORD ctrl_type) {
            if (ctrl_type == CTRL_C_EVENT || ctrl_type == CTRL_BREAK_EVENT) {
                const int count = ++g_interrupt_count;
                g_batch_interrupted.store(true);
                HANDLE child = g_active_child_process.load();
                if (child != nullptr && count >= 2) {
                    TerminateProcess(child, 130);
                }
                return TRUE;
            }
            return FALSE;
        }
#endif

        struct ScopedSignalHandler {
            ScopedSignalHandler() {
                g_batch_interrupted.store(false);
                g_interrupt_count.store(0);
#ifndef _WIN32
                struct sigaction sa{};
                sa.sa_handler = handle_sigint;
                sigemptyset(&sa.sa_mask);
                sa.sa_flags = 0;
                sigaction(SIGINT, &sa, &old_sa_int_);
                sigaction(SIGTERM, &sa, &old_sa_term_);
#else
                SetConsoleCtrlHandler(handle_console_ctrl, TRUE);
#endif
            }

            ~ScopedSignalHandler() {
#ifndef _WIN32
                sigaction(SIGINT, &old_sa_int_, nullptr);
                sigaction(SIGTERM, &old_sa_term_, nullptr);
#else
                SetConsoleCtrlHandler(handle_console_ctrl, FALSE);
#endif
            }

        private:
#ifndef _WIN32
            struct sigaction old_sa_int_{};
            struct sigaction old_sa_term_{};
#endif
        };

        std::string format_duration(const double total_seconds) {
            if (total_seconds < 0.0)
                return "00s";
            const auto sec_int = static_cast<int64_t>(total_seconds);
            const int64_t hours = sec_int / 3600;
            const int64_t minutes = (sec_int % 3600) / 60;
            const int64_t seconds = sec_int % 60;

            if (hours > 0) {
                return std::format("{:02d}h {:02d}m {:02d}s", hours, minutes, seconds);
            }
            if (minutes > 0) {
                return std::format("{:02d}m {:02d}s", minutes, seconds);
            }
            return std::format("{:02d}s", seconds);
        }

        std::string current_iso_timestamp() {
            const auto now = std::chrono::system_clock::now();
            const auto in_time_t = std::chrono::system_clock::to_time_t(now);
            std::tm tm_buf{};
#ifdef _WIN32
            localtime_s(&tm_buf, &in_time_t);
#else
            localtime_r(&in_time_t, &tm_buf);
#endif
            std::ostringstream ss;
            ss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S");
            return ss.str();
        }

        bool is_hidden_or_output_dir(const fs::path& path) {
            const auto filename = lfs::core::path_to_utf8(path.filename());
            if (filename.empty())
                return false;
            if (filename.front() == '.')
                return true;
            if (filename == "output" || filename == "outputs")
                return true;
            return false;
        }

        void discover_datasets_recursive(const fs::path& root, int depth, int max_depth,
                                         std::vector<fs::path>& discovered) {
            if (depth > max_depth)
                return;
            if (!lfs::core::native_fs::exists(root) || !lfs::core::native_fs::is_directory(root))
                return;

            try {
                for (const auto& entry : lfs::core::native_fs::directory_iterator(root)) {
                    const auto entry_path = lfs::core::native_fs::entry_path(root, entry.path());
                    if (!lfs::core::native_fs::entry_is_directory(root, entry))
                        continue;
                    if (is_hidden_or_output_dir(entry_path))
                        continue;

                    if (lfs::io::Loader::isDatasetPath(entry_path)) {
                        discovered.push_back(entry_path);
                    } else if (depth < max_depth) {
                        discover_datasets_recursive(entry_path, depth + 1, max_depth, discovered);
                    }
                }
            } catch (const std::exception& e) {
                LOG_WARN("Could not read directory {}: {}", lfs::core::path_to_utf8(root), e.what());
            }
        }

        std::vector<lfs::core::param::BatchJobItem> resolve_jobs_internal(
            const lfs::core::param::BatchTrainingParameters& params) {
            std::vector<lfs::core::param::BatchJobItem> jobs = params.jobs;
            std::vector<fs::path> candidate_paths;

            for (const auto& input : params.input_paths) {
                if (!lfs::core::native_fs::exists(input)) {
                    LOG_WARN("Batch input path does not exist: {}", lfs::core::path_to_utf8(input));
                    continue;
                }

                // If input is a JSON file, load it as a batch manifest
                if (lfs::core::native_fs::is_regular_file(input) && input.extension() == ".json") {
                    auto config_res = lfs::core::param::read_batch_training_parameters_from_json(input, params);
                    if (config_res) {
                        jobs.insert(jobs.end(), config_res->jobs.begin(), config_res->jobs.end());
                    } else {
                        LOG_ERROR("Failed to load batch manifest {}: {}",
                                  lfs::core::path_to_utf8(input), config_res.error().user_message());
                    }
                    continue;
                }

                // If input is directly a dataset (e.g. contains sparse/ or transforms.json)
                if (lfs::io::Loader::isDatasetPath(input)) {
                    candidate_paths.push_back(input);
                    continue;
                }

                // Otherwise, input is a directory containing subdirectories
                if (lfs::core::native_fs::is_directory(input)) {
                    if (params.recursive) {
                        discover_datasets_recursive(input, 1, 5, candidate_paths);
                    } else {
                        try {
                            for (const auto& entry : lfs::core::native_fs::directory_iterator(input)) {
                                const auto entry_path = lfs::core::native_fs::entry_path(input, entry.path());
                                if (!lfs::core::native_fs::entry_is_directory(input, entry))
                                    continue;
                                if (is_hidden_or_output_dir(entry_path))
                                    continue;

                                if (lfs::io::Loader::isDatasetPath(entry_path)) {
                                    candidate_paths.push_back(entry_path);
                                }
                            }
                        } catch (const std::exception& e) {
                            LOG_WARN("Could not read directory {}: {}", lfs::core::path_to_utf8(input), e.what());
                        }
                    }
                }
            }

            // Sort candidate paths alphabetically for reproducible queue ordering
            std::sort(candidate_paths.begin(), candidate_paths.end());

            // Convert discovered dataset paths into BatchJobItems
            for (const auto& path : candidate_paths) {
                lfs::core::param::BatchJobItem item;
                item.dataset_path = path;
                jobs.push_back(std::move(item));
            }

            // Deduplicate jobs by canonical dataset path
            std::vector<lfs::core::param::BatchJobItem> unique_jobs;
            std::vector<std::string> seen_canonical_paths;

            for (auto& job : jobs) {
                std::error_code ec;
                auto canonical = fs::weakly_canonical(job.dataset_path, ec);
                std::string path_str = lfs::core::path_to_utf8(ec ? job.dataset_path : canonical);

                if (std::find(seen_canonical_paths.begin(), seen_canonical_paths.end(), path_str) != seen_canonical_paths.end()) {
                    continue;
                }
                seen_canonical_paths.push_back(path_str);
                unique_jobs.push_back(std::move(job));
            }

            // Check for duplicate folder names among jobs without explicit custom names
            std::unordered_map<std::string, int> base_name_counts;
            for (const auto& job : unique_jobs) {
                if (job.name.empty()) {
                    std::string base = lfs::core::path_to_utf8(job.dataset_path.stem());
                    if (base.empty() || base == ".") {
                        base = "dataset";
                    }
                    base_name_counts[base]++;
                }
            }

            // Assign unique names: qualify colliding names with parent folder or numeric suffix
            std::unordered_map<std::string, int> name_frequencies;
            for (auto& job : unique_jobs) {
                if (job.name.empty()) {
                    std::string base = lfs::core::path_to_utf8(job.dataset_path.stem());
                    if (base.empty() || base == ".") {
                        base = "dataset";
                    }
                    if (base_name_counts[base] > 1) {
                        auto parent = job.dataset_path.parent_path().stem();
                        std::string parent_str = lfs::core::path_to_utf8(parent);
                        if (!parent_str.empty() && parent_str != "." && parent_str != "/" && parent_str != "\\") {
                            job.name = parent_str + "_" + base;
                        } else {
                            job.name = base;
                        }
                    } else {
                        job.name = base;
                    }
                }

                std::string candidate_name = job.name;
                int& count = name_frequencies[candidate_name];
                count++;
                if (count > 1) {
                    job.name = candidate_name + "_" + std::to_string(count);
                }
            }

            // Assign output paths and ensure no collisions
            std::vector<std::string> seen_output_paths;
            for (auto& job : unique_jobs) {
                if (job.output_path.empty()) {
                    if (!params.output_dir.empty()) {
                        job.output_path = params.output_dir / job.name;
                    } else {
                        job.output_path = job.dataset_path / "output";
                    }
                }

                std::error_code ec;
                auto canonical_out = fs::weakly_canonical(job.output_path, ec);
                std::string out_str = lfs::core::path_to_utf8(ec ? job.output_path : canonical_out);

                if (std::find(seen_output_paths.begin(), seen_output_paths.end(), out_str) != seen_output_paths.end()) {
                    int suffix = 2;
                    fs::path base_out = job.output_path;
                    while (true) {
                        fs::path cand = base_out;
                        cand += "_" + std::to_string(suffix);
                        auto cand_canonical = fs::weakly_canonical(cand, ec);
                        std::string cand_str = lfs::core::path_to_utf8(ec ? cand : cand_canonical);
                        if (std::find(seen_output_paths.begin(), seen_output_paths.end(), cand_str) == seen_output_paths.end()) {
                            job.output_path = cand;
                            seen_output_paths.push_back(cand_str);
                            break;
                        }
                        suffix++;
                    }
                } else {
                    seen_output_paths.push_back(out_str);
                }
            }

            return unique_jobs;
        }

        struct JobExecutionResult {
            std::string name;
            fs::path dataset_path;
            fs::path output_path;
            bool success = false;
            bool cancelled = false;
            int exit_code = 0;
            double duration_seconds = 0.0;
            std::string error_message;
        };

        std::vector<std::string> build_child_args(
            const lfs::core::param::BatchJobItem& job,
            const lfs::core::param::BatchTrainingParameters& params) {
            std::vector<std::string> args;
            args.reserve(16 + params.shared_args.size() + job.extra_args.size());

            args.push_back("--headless");
            args.push_back("-d");
            args.push_back(lfs::core::path_to_utf8(job.dataset_path));
            args.push_back("-o");
            args.push_back(lfs::core::path_to_utf8(job.output_path));

            if (job.config_file && !job.config_file->empty()) {
                args.push_back("--config");
                args.push_back(lfs::core::path_to_utf8(*job.config_file));
            }

            // Append shared args, filtering out any conflicting dataset/output flags
            for (size_t i = 0; i < params.shared_args.size(); ++i) {
                const auto& arg = params.shared_args[i];
                if (arg == "-d" || arg == "--data-path" || arg == "-o" || arg == "--output-path") {
                    if (i + 1 < params.shared_args.size() && !params.shared_args[i + 1].starts_with('-')) {
                        ++i;
                    }
                    continue;
                }
                if (arg == "--headless")
                    continue;
                args.push_back(arg);
            }

            // Append per-job extra arguments
            for (const auto& extra : job.extra_args) {
                args.push_back(extra);
            }

            return args;
        }

        JobExecutionResult execute_job(
            const lfs::core::param::BatchJobItem& job,
            const lfs::core::param::BatchTrainingParameters& params,
            const fs::path& executable) {
            JobExecutionResult result;
            result.name = job.name;
            result.dataset_path = job.dataset_path;
            result.output_path = job.output_path;

            std::error_code ec;
            lfs::core::native_fs::create_directories(job.output_path, ec);
            if (ec) {
                result.success = false;
                result.exit_code = 1;
                result.error_message = std::format("Failed to create output directory: {}", ec.message());
                return result;
            }

            const auto log_path = job.output_path / "training_log.txt";
            std::ofstream log_file(log_path, std::ios::out | std::ios::trunc);
            if (!log_file.is_open()) {
                LOG_WARN("Could not open log file at {}", lfs::core::path_to_utf8(log_path));
            }

            const auto child_args = build_child_args(job, params);
            const auto start_time = std::chrono::steady_clock::now();

#ifdef _WIN32
            SECURITY_ATTRIBUTES sa;
            sa.nLength = sizeof(SECURITY_ATTRIBUTES);
            sa.bInheritHandle = TRUE;
            sa.lpSecurityDescriptor = nullptr;

            HANDLE hReadPipe, hWritePipe;
            if (!CreatePipe(&hReadPipe, &hWritePipe, &sa, 0)) {
                result.success = false;
                result.exit_code = 1;
                result.error_message = "Failed to create process pipe";
                return result;
            }
            SetHandleInformation(hReadPipe, HANDLE_FLAG_INHERIT, 0);

            STARTUPINFOW si = {};
            si.cb = sizeof(si);
            si.hStdOutput = hWritePipe;
            si.hStdError = hWritePipe;
            si.dwFlags |= STARTF_USESTDHANDLES;

            PROCESS_INFORMATION pi = {};
            std::wstring cmdline = lfs::python::detail::build_win32_cmdline(executable, child_args);

            if (!CreateProcessW(executable.c_str(), cmdline.data(), nullptr, nullptr, TRUE,
                                0, nullptr, nullptr, &si, &pi)) {
                CloseHandle(hReadPipe);
                CloseHandle(hWritePipe);
                result.success = false;
                result.exit_code = 1;
                result.error_message = std::format("CreateProcessW failed: {}", GetLastError());
                return result;
            }

            CloseHandle(hWritePipe);
            g_active_child_process.store(pi.hProcess);

            HANDLE hStdOut = GetStdHandle(STD_OUTPUT_HANDLE);
            char buffer[4096];
            DWORD bytesRead = 0;
            bool cancel_message_printed = false;

            while (true) {
                DWORD bytesAvailable = 0;
                if (!PeekNamedPipe(hReadPipe, nullptr, 0, nullptr, &bytesAvailable, nullptr)) {
                    break;
                }

                if (bytesAvailable > 0) {
                    DWORD toRead = std::min<DWORD>(static_cast<DWORD>(sizeof(buffer) - 1), bytesAvailable);
                    if (ReadFile(hReadPipe, buffer, toRead, &bytesRead, nullptr) && bytesRead > 0) {
                        buffer[bytesRead] = '\0';
                        DWORD bytesWritten = 0;
                        WriteFile(hStdOut, buffer, bytesRead, &bytesWritten, nullptr);
                        if (log_file.is_open()) {
                            log_file.write(buffer, bytesRead);
                            log_file.flush();
                        }
                    }
                }

                DWORD wait_res = WaitForSingleObject(pi.hProcess, 50);
                if (wait_res == WAIT_OBJECT_0) {
                    while (PeekNamedPipe(hReadPipe, nullptr, 0, nullptr, &bytesAvailable, nullptr) && bytesAvailable > 0) {
                        DWORD toRead = std::min<DWORD>(static_cast<DWORD>(sizeof(buffer) - 1), bytesAvailable);
                        if (ReadFile(hReadPipe, buffer, toRead, &bytesRead, nullptr) && bytesRead > 0) {
                            DWORD bytesWritten = 0;
                            WriteFile(hStdOut, buffer, bytesRead, &bytesWritten, nullptr);
                            if (log_file.is_open()) {
                                log_file.write(buffer, bytesRead);
                                log_file.flush();
                            }
                        } else {
                            break;
                        }
                    }
                    break;
                }

                if (g_batch_interrupted.load()) {
                    result.cancelled = true;
                    result.error_message = "Cancelled by user";
                    if (result.exit_code == 0) {
                        result.exit_code = 130;
                    }

                    const int count = g_interrupt_count.load();
                    if (count >= 2) {
                        std::println(stderr, "\n[Batch] Second interrupt received. Force terminating child process...");
                        TerminateProcess(pi.hProcess, 130);
                        WaitForSingleObject(pi.hProcess, 5000);
                        break;
                    } else if (!cancel_message_printed) {
                        std::println(stderr, "\n[Batch] Interrupt received. Waiting for child process to save and shut down cleanly... (Press Ctrl+C again to force kill)");
                        cancel_message_printed = true;
                    }
                }

                if (params.timeout_minutes > 0) {
                    const auto elapsed_sec = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - start_time).count();
                    if (elapsed_sec > params.timeout_minutes * 60) {
                        TerminateProcess(pi.hProcess, 124);
                        result.exit_code = 124;
                        result.error_message = std::format("Job timed out after {} minutes", params.timeout_minutes);
                        WaitForSingleObject(pi.hProcess, 5000);
                        break;
                    }
                }
            }

            DWORD exitCodeDword = 0;
            GetExitCodeProcess(pi.hProcess, &exitCodeDword);
            if (result.exit_code == 0) {
                result.exit_code = static_cast<int>(exitCodeDword);
            }
            if (result.cancelled && result.exit_code == 0) {
                result.exit_code = 130;
            }

            g_active_child_process.store(nullptr);
            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);
            CloseHandle(hReadPipe);

#else
            int pipe_fds[2];
            if (pipe(pipe_fds) != 0) {
                result.success = false;
                result.exit_code = 1;
                result.error_message = "Failed to create pipe";
                return result;
            }

            fcntl(pipe_fds[0], F_SETFL, O_NONBLOCK);

            const std::string program_utf8 = lfs::core::path_to_utf8(executable);
            std::vector<char*> c_argv;
            c_argv.reserve(child_args.size() + 2);
            c_argv.push_back(const_cast<char*>(program_utf8.c_str()));
            for (const auto& arg : child_args) {
                c_argv.push_back(const_cast<char*>(arg.c_str()));
            }
            c_argv.push_back(nullptr);

            const pid_t pid = fork();
            if (pid == -1) {
                close(pipe_fds[0]);
                close(pipe_fds[1]);
                result.success = false;
                result.exit_code = 1;
                result.error_message = "Failed to fork process";
                return result;
            }

            if (pid == 0) {
                dup2(pipe_fds[1], STDOUT_FILENO);
                dup2(pipe_fds[1], STDERR_FILENO);
                close(pipe_fds[0]);
                close(pipe_fds[1]);

                execvp(program_utf8.c_str(), c_argv.data());
                _exit(127);
            }

            close(pipe_fds[1]);
            g_active_child_pid.store(pid);

            char buffer[4096];
            bool child_done = false;
            bool cancel_message_printed = false;
            bool timeout_triggered = false;
            auto timeout_sigterm_time = std::chrono::steady_clock::time_point{};

            while (!child_done) {
                struct pollfd pfd{};
                pfd.fd = pipe_fds[0];
                pfd.events = POLLIN | POLLHUP;

                int poll_res = poll(&pfd, 1, 100);
                if (poll_res > 0 && (pfd.revents & POLLIN)) {
                    ssize_t bytes_read = 0;
                    while ((bytes_read = read(pipe_fds[0], buffer, sizeof(buffer) - 1)) > 0) {
                        buffer[bytes_read] = '\0';
                        ssize_t written = ::write(STDOUT_FILENO, buffer, bytes_read);
                        static_cast<void>(written);
                        std::fflush(stdout);

                        if (log_file.is_open()) {
                            log_file.write(buffer, bytes_read);
                            log_file.flush();
                        }
                    }
                }

                // Check process status
                int status = 0;
                pid_t wait_res = waitpid(pid, &status, WNOHANG);
                if (wait_res == pid) {
                    child_done = true;
                    // Drain any remaining buffered output
                    ssize_t bytes_read = 0;
                    while ((bytes_read = read(pipe_fds[0], buffer, sizeof(buffer) - 1)) > 0) {
                        buffer[bytes_read] = '\0';
                        ssize_t written = ::write(STDOUT_FILENO, buffer, bytes_read);
                        static_cast<void>(written);
                        std::fflush(stdout);
                        if (log_file.is_open()) {
                            log_file.write(buffer, bytes_read);
                            log_file.flush();
                        }
                    }

                    if (WIFEXITED(status)) {
                        result.exit_code = WEXITSTATUS(status);
                    } else if (WIFSIGNALED(status)) {
                        result.exit_code = 128 + WTERMSIG(status);
                    }

                    if (g_batch_interrupted.load()) {
                        result.cancelled = true;
                        result.error_message = "Cancelled by user";
                        if (result.exit_code == 0) {
                            result.exit_code = 130;
                        }
                    }
                    break;
                }

                if (!child_done && g_batch_interrupted.load()) {
                    result.cancelled = true;
                    result.error_message = "Cancelled by user";
                    if (result.exit_code == 0) {
                        result.exit_code = 130;
                    }

                    const int count = g_interrupt_count.load();
                    if (count >= 2) {
                        std::println(stderr, "\n[Batch] Second interrupt received. Force killing child process (PID {})...", pid);
                        kill(pid, SIGKILL);
                        waitpid(pid, &status, 0);
                        child_done = true;
                        result.exit_code = 130;
                        break;
                    } else if (!cancel_message_printed) {
                        std::println(stderr, "\n[Batch] Interrupt received. Waiting for child process to save and shut down cleanly... (Press Ctrl+C again to force kill)");
                        cancel_message_printed = true;
                    }
                }

                if (!child_done && params.timeout_minutes > 0) {
                    const auto elapsed_sec = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - start_time).count();
                    if (elapsed_sec > params.timeout_minutes * 60) {
                        if (!timeout_triggered) {
                            timeout_triggered = true;
                            timeout_sigterm_time = std::chrono::steady_clock::now();
                            result.exit_code = 124;
                            result.error_message = std::format("Job timed out after {} minutes", params.timeout_minutes);
                            std::println(stderr, "\n[Batch] Job timed out after {} minutes. Sending SIGTERM to PID {}...", params.timeout_minutes, pid);
                            kill(pid, SIGTERM);
                        } else {
                            const auto grace_elapsed = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - timeout_sigterm_time).count();
                            if (grace_elapsed > 30.0) {
                                std::println(stderr, "\n[Batch] Child process did not exit within 30s grace period after timeout. Force killing (PID {})...", pid);
                                kill(pid, SIGKILL);
                                waitpid(pid, &status, 0);
                                child_done = true;
                                break;
                            }
                        }
                    }
                }
            }

            close(pipe_fds[0]);
            g_active_child_pid.store(0);
#endif

            const auto end_time = std::chrono::steady_clock::now();
            result.duration_seconds = std::chrono::duration<double>(end_time - start_time).count();
            result.success = (result.exit_code == 0 && !result.cancelled);
            if (result.cancelled && result.exit_code == 0) {
                result.exit_code = 130;
            }
            if (!result.success && result.error_message.empty()) {
                result.error_message = std::format("Process exited with code {}", result.exit_code);
            }

            return result;
        }

        void write_summary_files(
            const lfs::core::param::BatchTrainingParameters& params,
            const std::vector<JobExecutionResult>& results,
            const double total_duration_seconds,
            const std::string& timestamp) {

            fs::path summary_path = params.summary_file;
            if (summary_path.empty()) {
                if (!params.output_dir.empty()) {
                    summary_path = params.output_dir / "batch_summary.json";
                } else {
                    summary_path = "batch_summary.json";
                }
            }

            fs::path report_path = params.report_file;
            if (report_path.empty()) {
                if (!params.output_dir.empty()) {
                    report_path = params.output_dir / "batch_report.txt";
                } else {
                    report_path = "batch_report.txt";
                }
            }

            json summary_json;
            summary_json["timestamp"] = timestamp;
            summary_json["total_jobs"] = results.size();

            size_t succeeded = 0, failed = 0, cancelled = 0;
            for (const auto& r : results) {
                if (r.cancelled)
                    ++cancelled;
                else if (r.success)
                    ++succeeded;
                else
                    ++failed;
            }

            summary_json["succeeded"] = succeeded;
            summary_json["failed"] = failed;
            summary_json["cancelled"] = cancelled;
            summary_json["total_duration_seconds"] = total_duration_seconds;
            summary_json["total_duration_formatted"] = format_duration(total_duration_seconds);

            json jobs_array = json::array();
            for (size_t i = 0; i < results.size(); ++i) {
                const auto& r = results[i];
                json job_obj;
                job_obj["index"] = i + 1;
                job_obj["name"] = r.name;
                job_obj["dataset_path"] = lfs::core::path_to_utf8(r.dataset_path);
                job_obj["output_path"] = lfs::core::path_to_utf8(r.output_path);
                job_obj["status"] = r.cancelled ? "CANCELLED" : (r.success ? "SUCCESS" : "FAILED");
                job_obj["exit_code"] = r.exit_code;
                job_obj["duration_seconds"] = r.duration_seconds;
                job_obj["duration_formatted"] = format_duration(r.duration_seconds);
                if (!r.error_message.empty()) {
                    job_obj["error"] = r.error_message;
                }
                jobs_array.push_back(std::move(job_obj));
            }
            summary_json["jobs"] = std::move(jobs_array);

            try {
                if (summary_path.has_parent_path()) {
                    std::error_code ec;
                    lfs::core::native_fs::create_directories(summary_path.parent_path(), ec);
                }
                std::ofstream json_out(summary_path);
                if (json_out.is_open()) {
                    json_out << summary_json.dump(2) << std::endl;
                    LOG_INFO("Wrote batch summary to {}", lfs::core::path_to_utf8(summary_path));
                }
            } catch (const std::exception& e) {
                LOG_ERROR("Failed to write summary JSON: {}", e.what());
            }

            try {
                if (report_path.has_parent_path()) {
                    std::error_code ec;
                    lfs::core::native_fs::create_directories(report_path.parent_path(), ec);
                }
                std::ofstream txt_out(report_path);
                if (txt_out.is_open()) {
                    txt_out << "================================================================================\n";
                    txt_out << "LICHTFELD STUDIO - BATCH TRAINING REPORT\n";
                    txt_out << "Timestamp: " << timestamp << "\n";
                    txt_out << "Total Jobs: " << results.size() << " | Succeeded: " << succeeded
                            << " | Failed: " << failed << " | Cancelled: " << cancelled << "\n";
                    txt_out << "Total Duration: " << format_duration(total_duration_seconds) << "\n";
                    txt_out << "================================================================================\n\n";

                    for (size_t i = 0; i < results.size(); ++i) {
                        const auto& r = results[i];
                        const std::string status_str = r.cancelled ? "CANCELLED" : (r.success ? "SUCCESS  " : "FAILED   ");
                        txt_out << std::format("[{:2d}] {} {:<24} ({}) -> {}\n",
                                               i + 1, status_str, r.name,
                                               format_duration(r.duration_seconds),
                                               lfs::core::path_to_utf8(r.output_path));
                        if (!r.success && !r.error_message.empty()) {
                            txt_out << "     Error: " << r.error_message << "\n";
                        }
                    }
                    txt_out << "\n================================================================================\n";
                    LOG_INFO("Wrote batch report to {}", lfs::core::path_to_utf8(report_path));
                }
            } catch (const std::exception& e) {
                LOG_ERROR("Failed to write report text: {}", e.what());
            }
        }

    } // namespace

    std::vector<lfs::core::param::BatchJobItem> resolve_batch_jobs(
        const lfs::core::param::BatchTrainingParameters& params) {
        return resolve_jobs_internal(params);
    }

    int run_batch_training(const lfs::core::param::BatchTrainingParameters& params) {
        ScopedSignalHandler signal_handler;

        fs::path executable;
        try {
            executable = lfs::core::getExecutablePath();
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to locate LichtFeld Studio executable: {}", e.what());
            std::println(stderr, "Error: Failed to locate LichtFeld Studio executable: {}", e.what());
            return 1;
        }

        const auto jobs = resolve_batch_jobs(params);
        if (jobs.empty()) {
            LOG_ERROR("Batch training: No valid datasets found.");
            std::println(stderr, "\nError: No valid datasets found.");
            std::println(stderr, "Specify dataset directories containing sparse/ or transforms.json, or pass a --config manifest.");
            std::println(stderr, "Run 'LichtFeld-Studio batch-train --help' for details.\n");
            return 1;
        }

        if (params.dry_run) {
            std::println("\n================================================================================");
            std::println("BATCH TRAINING - DRY RUN (Resolved Queue: {} jobs)", jobs.size());
            std::println("================================================================================");
            if (!params.shared_args.empty()) {
                std::string shared_str;
                for (const auto& a : params.shared_args) {
                    shared_str += a + " ";
                }
                std::println("Shared arguments: {}", shared_str);
            }
            if (params.timeout_minutes > 0) {
                std::println("Timeout per job: {} minutes", params.timeout_minutes);
            }
            std::println("Continue on failure: {}", params.continue_on_failure ? "yes" : "no");
            std::println("--------------------------------------------------------------------------------");

            for (size_t i = 0; i < jobs.size(); ++i) {
                const auto& job = jobs[i];
                std::println("[{:2d}] Job: {}", i + 1, job.name);
                std::println("     Dataset: {}", lfs::core::path_to_utf8(job.dataset_path));
                std::println("     Output:  {}", lfs::core::path_to_utf8(job.output_path));
                if (job.config_file) {
                    std::println("     Config:  {}", lfs::core::path_to_utf8(*job.config_file));
                }
                if (!job.extra_args.empty()) {
                    std::string extra_str;
                    for (const auto& a : job.extra_args)
                        extra_str += a + " ";
                    std::println("     Extra:   {}", extra_str);
                }
            }
            std::println("================================================================================");
            std::println("Dry run complete. {} job(s) ready to train.\n", jobs.size());
            return 0;
        }

        const auto timestamp = current_iso_timestamp();
        const auto batch_start_time = std::chrono::steady_clock::now();

        std::println("\n================================================================================");
        std::println("LICHTFELD STUDIO - BATCH TRAINING");
        std::println("Queue size: {} dataset(s) | Started at: {}", jobs.size(), timestamp);
        if (!params.shared_args.empty()) {
            std::string shared_str;
            for (const auto& a : params.shared_args) {
                shared_str += a + " ";
            }
            std::println("Shared options: {}", shared_str);
        }
        std::println("Failure policy: {}", params.continue_on_failure ? "Continue next on failure" : "Stop on first failure");
        std::println("================================================================================\n");

        std::vector<JobExecutionResult> results;
        results.reserve(jobs.size());

        for (size_t i = 0; i < jobs.size(); ++i) {
            if (g_batch_interrupted.load()) {
                std::println("\n[Batch] Training interrupted by user. Aborting remaining {} job(s)...", jobs.size() - i);
                for (size_t remaining = i; remaining < jobs.size(); ++remaining) {
                    JobExecutionResult cancelled_res;
                    cancelled_res.name = jobs[remaining].name;
                    cancelled_res.dataset_path = jobs[remaining].dataset_path;
                    cancelled_res.output_path = jobs[remaining].output_path;
                    cancelled_res.cancelled = true;
                    cancelled_res.exit_code = 130;
                    cancelled_res.error_message = "Cancelled before start";
                    results.push_back(std::move(cancelled_res));
                }
                break;
            }

            const auto& job = jobs[i];
            std::println("\n================================================================================");
            std::println("[Batch Job {}/{}] Starting: {}", i + 1, jobs.size(), job.name);
            std::println("Dataset: {}", lfs::core::path_to_utf8(job.dataset_path));
            std::println("Output:  {}", lfs::core::path_to_utf8(job.output_path));
            std::println("Time:    {}", current_iso_timestamp());
            std::println("================================================================================");

            auto res = execute_job(job, params, executable);
            results.push_back(res);

            if (res.success) {
                std::println("\n[Batch Job {}/{}] COMPLETED in {}: {}",
                             i + 1, jobs.size(), format_duration(res.duration_seconds), res.name);
            } else if (res.cancelled) {
                std::println("\n[Batch Job {}/{}] CANCELLED: {}", i + 1, jobs.size(), res.name);
                for (size_t remaining = i + 1; remaining < jobs.size(); ++remaining) {
                    JobExecutionResult cancelled_res;
                    cancelled_res.name = jobs[remaining].name;
                    cancelled_res.dataset_path = jobs[remaining].dataset_path;
                    cancelled_res.output_path = jobs[remaining].output_path;
                    cancelled_res.cancelled = true;
                    cancelled_res.exit_code = 130;
                    cancelled_res.error_message = "Cancelled by user";
                    results.push_back(std::move(cancelled_res));
                }
                break;
            } else {
                std::println("\n[Batch Job {}/{}] FAILED (exit code {}): {}",
                             i + 1, jobs.size(), res.exit_code, res.name);
                if (!res.error_message.empty()) {
                    std::println("  Detail: {}", res.error_message);
                }

                if (!params.continue_on_failure) {
                    std::println("\n[Batch] Job failed and continue-on-failure is disabled. Stopping batch.");
                    for (size_t remaining = i + 1; remaining < jobs.size(); ++remaining) {
                        JobExecutionResult cancelled_res;
                        cancelled_res.name = jobs[remaining].name;
                        cancelled_res.dataset_path = jobs[remaining].dataset_path;
                        cancelled_res.output_path = jobs[remaining].output_path;
                        cancelled_res.cancelled = true;
                        cancelled_res.exit_code = 130;
                        cancelled_res.error_message = "Cancelled due to previous failure";
                        results.push_back(std::move(cancelled_res));
                    }
                    break;
                }
            }
        }

        const auto batch_end_time = std::chrono::steady_clock::now();
        const auto total_duration_seconds = std::chrono::duration<double>(batch_end_time - batch_start_time).count();

        size_t succeeded_count = 0, failed_count = 0, cancelled_count = 0;
        for (const auto& r : results) {
            if (r.cancelled)
                ++cancelled_count;
            else if (r.success)
                ++succeeded_count;
            else
                ++failed_count;
        }

        std::println("\n================================================================================");
        std::println("BATCH TRAINING SUMMARY");
        std::println("Total: {} | Succeeded: {} | Failed: {} | Cancelled: {}",
                     results.size(), succeeded_count, failed_count, cancelled_count);
        std::println("Total Time: {}", format_duration(total_duration_seconds));
        std::println("--------------------------------------------------------------------------------");

        for (size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            const std::string status_label = r.cancelled ? "CANCELLED" : (r.success ? "SUCCESS  " : "FAILED   ");
            std::println("[{:2d}] {} {:<24} ({}) -> {}",
                         i + 1, status_label, r.name,
                         format_duration(r.duration_seconds),
                         lfs::core::path_to_utf8(r.output_path));
        }
        std::println("================================================================================");

        write_summary_files(params, results, total_duration_seconds, timestamp);

        return (failed_count > 0 || cancelled_count > 0) ? 1 : 0;
    }

} // namespace lfs::app
