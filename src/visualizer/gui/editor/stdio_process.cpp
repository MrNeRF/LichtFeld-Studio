/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "stdio_process.hpp"

#include <core/logger.hpp>

#ifndef _WIN32
#include <cerrno>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#else
#include <vector>
#endif

namespace lfs::vis::editor {

    namespace {

#ifdef _WIN32
        std::wstring to_wide(const std::string& value) {
            if (value.empty()) {
                return {};
            }

            const int size = MultiByteToWideChar(CP_UTF8, 0, value.c_str(),
                                                 static_cast<int>(value.size()), nullptr, 0);
            if (size <= 0) {
                return {};
            }

            std::wstring wide(static_cast<size_t>(size), L'\0');
            const int converted = MultiByteToWideChar(CP_UTF8, 0, value.c_str(),
                                                      static_cast<int>(value.size()),
                                                      wide.data(), size);
            if (converted <= 0) {
                return {};
            }

            wide.resize(static_cast<size_t>(converted));
            return wide;
        }

        std::wstring quote_arg(const std::string& arg) {
            std::wstring wide = to_wide(arg);
            const bool needs_quotes =
                wide.empty() || wide.find_first_of(L" \t\"") != std::wstring::npos;
            if (!needs_quotes) {
                return wide;
            }

            std::wstring quoted;
            quoted.push_back(L'"');
            size_t backslashes = 0;
            for (const wchar_t ch : wide) {
                if (ch == L'\\') {
                    ++backslashes;
                    continue;
                }

                if (ch == L'"') {
                    quoted.append(backslashes * 2 + 1, L'\\');
                    quoted.push_back(L'"');
                    backslashes = 0;
                    continue;
                }

                if (backslashes > 0) {
                    quoted.append(backslashes, L'\\');
                    backslashes = 0;
                }
                quoted.push_back(ch);
            }

            if (backslashes > 0) {
                quoted.append(backslashes * 2, L'\\');
            }
            quoted.push_back(L'"');
            return quoted;
        }

        void close_handle(HANDLE& handle) {
            if (handle != INVALID_HANDLE_VALUE) {
                CloseHandle(handle);
                handle = INVALID_HANDLE_VALUE;
            }
        }

#else
        void close_fd(int& fd) {
            if (fd >= 0) {
                close(fd);
                fd = -1;
            }
        }

        void set_non_blocking(const int fd) {
            const int flags = fcntl(fd, F_GETFL, 0);
            if (flags >= 0) {
                fcntl(fd, F_SETFL, flags | O_NONBLOCK);
            }
        }
#endif

    } // namespace

    StdioProcess::~StdioProcess() {
        kill();
    }

#ifndef _WIN32

    bool StdioProcess::start(const std::string& program, const std::vector<std::string>& args) {
        kill();

        int stdin_pipe[2] = {-1, -1};
        int stdout_pipe[2] = {-1, -1};
        int stderr_pipe[2] = {-1, -1};

        if (pipe(stdin_pipe) != 0 || pipe(stdout_pipe) != 0 || pipe(stderr_pipe) != 0) {
            LOG_ERROR("StdioProcess: pipe() failed: {}", strerror(errno));
            if (stdin_pipe[0] >= 0) {
                close(stdin_pipe[0]);
            }
            if (stdin_pipe[1] >= 0) {
                close(stdin_pipe[1]);
            }
            if (stdout_pipe[0] >= 0) {
                close(stdout_pipe[0]);
            }
            if (stdout_pipe[1] >= 0) {
                close(stdout_pipe[1]);
            }
            if (stderr_pipe[0] >= 0) {
                close(stderr_pipe[0]);
            }
            if (stderr_pipe[1] >= 0) {
                close(stderr_pipe[1]);
            }
            return false;
        }

        pid_ = fork();
        if (pid_ < 0) {
            LOG_ERROR("StdioProcess: fork() failed: {}", strerror(errno));
            close(stdin_pipe[0]);
            close(stdin_pipe[1]);
            close(stdout_pipe[0]);
            close(stdout_pipe[1]);
            close(stderr_pipe[0]);
            close(stderr_pipe[1]);
            return false;
        }

        if (pid_ == 0) {
            dup2(stdin_pipe[0], STDIN_FILENO);
            dup2(stdout_pipe[1], STDOUT_FILENO);
            dup2(stderr_pipe[1], STDERR_FILENO);

            close(stdin_pipe[0]);
            close(stdin_pipe[1]);
            close(stdout_pipe[0]);
            close(stdout_pipe[1]);
            close(stderr_pipe[0]);
            close(stderr_pipe[1]);

            std::vector<char*> argv;
            argv.reserve(args.size() + 2);
            argv.push_back(const_cast<char*>(program.c_str()));
            for (const auto& arg : args) {
                argv.push_back(const_cast<char*>(arg.c_str()));
            }
            argv.push_back(nullptr);

            execvp(program.c_str(), argv.data());
            _exit(127);
        }

        close(stdin_pipe[0]);
        close(stdout_pipe[1]);
        close(stderr_pipe[1]);

        stdin_fd_ = stdin_pipe[1];
        stdout_fd_ = stdout_pipe[0];
        stderr_fd_ = stderr_pipe[0];
        exit_code_ = -1;

        set_non_blocking(stdout_fd_);
        set_non_blocking(stderr_fd_);
        return true;
    }

    bool StdioProcess::writeAll(std::string_view data) {
        if (stdin_fd_ < 0) {
            return false;
        }

        size_t written = 0;
        while (written < data.size()) {
            const ssize_t result =
                write(stdin_fd_, data.data() + written, data.size() - written);
            if (result < 0) {
                if (errno == EINTR) {
                    continue;
                }
                if (errno == EPIPE || errno == EBADF) {
                    return false;
                }
                LOG_ERROR("StdioProcess: write() failed: {}", strerror(errno));
                return false;
            }
            written += static_cast<size_t>(result);
        }

        return true;
    }

    ssize_t StdioProcess::readStdout(char* buffer, size_t length) {
        if (stdout_fd_ < 0) {
            return -1;
        }

        const ssize_t result = read(stdout_fd_, buffer, length);
        if (result < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            return 0;
        }
        if (result == 0) {
            return -1;
        }
        return result;
    }

    ssize_t StdioProcess::readStderr(char* buffer, size_t length) {
        if (stderr_fd_ < 0) {
            return -1;
        }

        const ssize_t result = read(stderr_fd_, buffer, length);
        if (result < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            return 0;
        }
        if (result == 0) {
            return -1;
        }
        return result;
    }

    bool StdioProcess::isRunning() const {
        if (pid_ <= 0) {
            return false;
        }

        int status = 0;
        const pid_t result = waitpid(pid_, &status, WNOHANG);
        if (result == 0) {
            return true;
        }

        if (result == pid_) {
            if (WIFEXITED(status)) {
                exit_code_ = WEXITSTATUS(status);
            } else if (WIFSIGNALED(status)) {
                exit_code_ = 128 + WTERMSIG(status);
            }
            pid_ = -1;
        }

        return false;
    }

    void StdioProcess::kill() {
        close_fd(stdin_fd_);
        close_fd(stdout_fd_);
        close_fd(stderr_fd_);

        if (pid_ > 0) {
            const pid_t child = pid_;
            pid_ = -1;

            ::kill(child, SIGTERM);
            usleep(50000);

            int status = 0;
            if (waitpid(child, &status, WNOHANG) == 0) {
                ::kill(child, SIGKILL);
                waitpid(child, &status, 0);
            }

            if (WIFEXITED(status)) {
                exit_code_ = WEXITSTATUS(status);
            } else if (WIFSIGNALED(status)) {
                exit_code_ = 128 + WTERMSIG(status);
            }
        }
    }

#else

    bool StdioProcess::start(const std::string& program, const std::vector<std::string>& args) {
        kill();

        SECURITY_ATTRIBUTES sa = {};
        sa.nLength = sizeof(sa);
        sa.bInheritHandle = TRUE;

        HANDLE stdout_write = INVALID_HANDLE_VALUE;
        HANDLE stderr_write = INVALID_HANDLE_VALUE;
        HANDLE stdin_read = INVALID_HANDLE_VALUE;

        if (!CreatePipe(&stdout_read_, &stdout_write, &sa, 0) ||
            !CreatePipe(&stderr_read_, &stderr_write, &sa, 0) ||
            !CreatePipe(&stdin_read, &stdin_write_, &sa, 0)) {
            LOG_ERROR("StdioProcess: CreatePipe failed");
            close_handle(stdout_read_);
            close_handle(stdout_write);
            close_handle(stderr_read_);
            close_handle(stderr_write);
            close_handle(stdin_read);
            close_handle(stdin_write_);
            return false;
        }

        SetHandleInformation(stdout_read_, HANDLE_FLAG_INHERIT, 0);
        SetHandleInformation(stderr_read_, HANDLE_FLAG_INHERIT, 0);
        SetHandleInformation(stdin_write_, HANDLE_FLAG_INHERIT, 0);

        STARTUPINFOW startup = {};
        startup.cb = sizeof(startup);
        startup.dwFlags = STARTF_USESTDHANDLES;
        startup.hStdInput = stdin_read;
        startup.hStdOutput = stdout_write;
        startup.hStdError = stderr_write;

        std::wstring command_line = quote_arg(program);
        for (const auto& arg : args) {
            command_line.push_back(L' ');
            command_line += quote_arg(arg);
        }

        PROCESS_INFORMATION info = {};
        const BOOL started = CreateProcessW(
            nullptr,
            command_line.data(),
            nullptr,
            nullptr,
            TRUE,
            CREATE_NO_WINDOW,
            nullptr,
            nullptr,
            &startup,
            &info);

        close_handle(stdout_write);
        close_handle(stderr_write);
        close_handle(stdin_read);

        if (!started) {
            LOG_ERROR("StdioProcess: CreateProcessW failed: {}", GetLastError());
            close_handle(stdout_read_);
            close_handle(stderr_read_);
            close_handle(stdin_write_);
            return false;
        }

        CloseHandle(info.hThread);
        process_ = info.hProcess;
        exit_code_ = -1;
        return true;
    }

    bool StdioProcess::writeAll(std::string_view data) {
        if (stdin_write_ == INVALID_HANDLE_VALUE) {
            return false;
        }

        const char* current = data.data();
        size_t remaining = data.size();
        while (remaining > 0) {
            const DWORD chunk = static_cast<DWORD>(std::min<size_t>(remaining, 32 * 1024));
            DWORD written = 0;
            if (!WriteFile(stdin_write_, current, chunk, &written, nullptr)) {
                return false;
            }
            remaining -= static_cast<size_t>(written);
            current += written;
        }

        return true;
    }

    ssize_t StdioProcess::readStdout(char* buffer, size_t length) {
        if (stdout_read_ == INVALID_HANDLE_VALUE) {
            return -1;
        }

        DWORD available = 0;
        if (!PeekNamedPipe(stdout_read_, nullptr, 0, nullptr, &available, nullptr) || available == 0) {
            return 0;
        }

        DWORD bytes_read = 0;
        if (!ReadFile(stdout_read_, buffer, static_cast<DWORD>(length), &bytes_read, nullptr)) {
            return -1;
        }
        return static_cast<ssize_t>(bytes_read);
    }

    ssize_t StdioProcess::readStderr(char* buffer, size_t length) {
        if (stderr_read_ == INVALID_HANDLE_VALUE) {
            return -1;
        }

        DWORD available = 0;
        if (!PeekNamedPipe(stderr_read_, nullptr, 0, nullptr, &available, nullptr) || available == 0) {
            return 0;
        }

        DWORD bytes_read = 0;
        if (!ReadFile(stderr_read_, buffer, static_cast<DWORD>(length), &bytes_read, nullptr)) {
            return -1;
        }
        return static_cast<ssize_t>(bytes_read);
    }

    bool StdioProcess::isRunning() const {
        if (process_ == INVALID_HANDLE_VALUE) {
            return false;
        }

        DWORD code = 0;
        if (!GetExitCodeProcess(process_, &code)) {
            return false;
        }

        if (code == STILL_ACTIVE) {
            return true;
        }

        exit_code_ = static_cast<int>(code);
        return false;
    }

    void StdioProcess::kill() {
        close_handle(stdin_write_);
        close_handle(stdout_read_);
        close_handle(stderr_read_);

        if (process_ != INVALID_HANDLE_VALUE) {
            DWORD code = 0;
            if (GetExitCodeProcess(process_, &code) && code == STILL_ACTIVE) {
                TerminateProcess(process_, 1);
                WaitForSingleObject(process_, 100);
                GetExitCodeProcess(process_, &code);
            }
            exit_code_ = static_cast<int>(code);
            close_handle(process_);
        }
    }

#endif

} // namespace lfs::vis::editor
