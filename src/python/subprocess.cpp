/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "subprocess.hpp"

#include <core/executable_path.hpp>
#include <core/logger.hpp>

#ifndef _WIN32
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <pty.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace lfs::python {

    SubProcess::~SubProcess() {
        kill();
    }

#ifndef _WIN32

    bool SubProcess::start(const std::string& program, const std::vector<std::string>& args) {
        kill();

        constexpr unsigned short PTY_COLS = 120;
        constexpr unsigned short PTY_ROWS = 24;
        struct winsize ws = {};
        ws.ws_col = PTY_COLS;
        ws.ws_row = PTY_ROWS;

        pid_ = forkpty(&stdout_fd_, nullptr, nullptr, &ws);
        if (pid_ == -1) {
            LOG_ERROR("forkpty() failed: {}", strerror(errno));
            return false;
        }

        if (pid_ == 0) {
            setenv("UV_HTTP_TIMEOUT", "300", 1);
            setenv("TERM", "xterm-256color", 1);

            const auto python_home = lfs::core::getPythonHome();
            if (!python_home.empty())
                setenv("PYTHONHOME", python_home.string().c_str(), 1);

            std::vector<const char*> argv;
            argv.push_back(program.c_str());
            for (const auto& arg : args)
                argv.push_back(arg.c_str());
            argv.push_back(nullptr);

            execvp(program.c_str(), const_cast<char* const*>(argv.data()));
            _exit(127);
        }

        const int flags = fcntl(stdout_fd_, F_GETFL, 0);
        fcntl(stdout_fd_, F_SETFL, flags | O_NONBLOCK);

        LOG_INFO("Subprocess started: {} (pid {})", program, pid_);
        return true;
    }

    ssize_t SubProcess::read(char* buf, size_t len) {
        if (stdout_fd_ < 0)
            return -1;

        const ssize_t n = ::read(stdout_fd_, buf, len);
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
            return 0;
        return n;
    }

    bool SubProcess::is_running() const {
        if (pid_ <= 0)
            return false;
        int status;
        const pid_t result = waitpid(pid_, &status, WNOHANG);
        if (result == pid_) {
            if (WIFEXITED(status))
                const_cast<SubProcess*>(this)->exit_code_ = WEXITSTATUS(status);
            else if (WIFSIGNALED(status))
                const_cast<SubProcess*>(this)->exit_code_ = 128 + WTERMSIG(status);
            const_cast<SubProcess*>(this)->pid_ = -1;
            return false;
        }
        return result == 0;
    }

    void SubProcess::kill() {
        if (stdout_fd_ >= 0) {
            close(stdout_fd_);
            stdout_fd_ = -1;
        }
        if (pid_ > 0) {
            ::kill(pid_, SIGTERM);
            usleep(50000);
            if (is_running())
                ::kill(pid_, SIGKILL);
            int status;
            waitpid(pid_, &status, 0);
            if (WIFEXITED(status))
                exit_code_ = WEXITSTATUS(status);
            pid_ = -1;
        }
    }

    int SubProcess::wait() {
        if (pid_ <= 0)
            return exit_code_;

        if (stdout_fd_ >= 0) {
            close(stdout_fd_);
            stdout_fd_ = -1;
        }

        int status;
        if (waitpid(pid_, &status, 0) == pid_) {
            if (WIFEXITED(status))
                exit_code_ = WEXITSTATUS(status);
            else if (WIFSIGNALED(status))
                exit_code_ = 128 + WTERMSIG(status);
        }
        pid_ = -1;
        return exit_code_;
    }

#else // Windows

    bool SubProcess::start(const std::string& program, const std::vector<std::string>& args) {
        kill();

        constexpr COORD PTY_SIZE = {120, 24};
        HANDLE pipe_pty_in = INVALID_HANDLE_VALUE;
        HANDLE pipe_pty_out = INVALID_HANDLE_VALUE;
        HANDLE pipe_in = INVALID_HANDLE_VALUE;

        if (!CreatePipe(&pipe_in, &pipe_pty_in, nullptr, 0)) {
            LOG_ERROR("CreatePipe for input failed");
            return false;
        }

        if (!CreatePipe(&pipe_pty_out, &pipe_stdout_, nullptr, 0)) {
            LOG_ERROR("CreatePipe for output failed");
            CloseHandle(pipe_in);
            CloseHandle(pipe_pty_in);
            return false;
        }

        HPCON hpc = nullptr;
        const HRESULT hr = CreatePseudoConsole(PTY_SIZE, pipe_pty_in, pipe_pty_out, 0, &hpc);
        CloseHandle(pipe_pty_in);
        CloseHandle(pipe_pty_out);
        CloseHandle(pipe_in);

        if (FAILED(hr)) {
            LOG_ERROR("CreatePseudoConsole failed: {}", hr);
            CloseHandle(pipe_stdout_);
            pipe_stdout_ = INVALID_HANDLE_VALUE;
            return false;
        }

        STARTUPINFOEXW si = {};
        si.StartupInfo.cb = sizeof(si);

        SIZE_T attr_size = 0;
        InitializeProcThreadAttributeList(nullptr, 1, 0, &attr_size);
        si.lpAttributeList = static_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(HeapAlloc(GetProcessHeap(), 0, attr_size));

        if (!si.lpAttributeList ||
            !InitializeProcThreadAttributeList(si.lpAttributeList, 1, 0, &attr_size) ||
            !UpdateProcThreadAttribute(si.lpAttributeList, 0, PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE,
                                       hpc, sizeof(hpc), nullptr, nullptr)) {
            LOG_ERROR("Failed to setup process attributes");
            if (si.lpAttributeList)
                HeapFree(GetProcessHeap(), 0, si.lpAttributeList);
            ClosePseudoConsole(hpc);
            CloseHandle(pipe_stdout_);
            pipe_stdout_ = INVALID_HANDLE_VALUE;
            return false;
        }

        std::wstring cmdline;
        cmdline += L"\"";
        cmdline += std::wstring(program.begin(), program.end());
        cmdline += L"\"";
        for (const auto& arg : args) {
            cmdline += L" \"";
            cmdline += std::wstring(arg.begin(), arg.end());
            cmdline += L"\"";
        }

        SetEnvironmentVariableW(L"UV_HTTP_TIMEOUT", L"300");

        PROCESS_INFORMATION pi = {};
        const BOOL success = CreateProcessW(nullptr, cmdline.data(), nullptr, nullptr, FALSE,
                                            EXTENDED_STARTUPINFO_PRESENT, nullptr, nullptr,
                                            &si.StartupInfo, &pi);

        DeleteProcThreadAttributeList(si.lpAttributeList);
        HeapFree(GetProcessHeap(), 0, si.lpAttributeList);
        ClosePseudoConsole(hpc);

        if (!success) {
            LOG_ERROR("CreateProcess failed: {}", GetLastError());
            CloseHandle(pipe_stdout_);
            pipe_stdout_ = INVALID_HANDLE_VALUE;
            return false;
        }

        CloseHandle(pi.hThread);
        process_ = pi.hProcess;

        LOG_INFO("Subprocess started: {}", program);
        return true;
    }

    ssize_t SubProcess::read(char* buf, size_t len) {
        if (pipe_stdout_ == INVALID_HANDLE_VALUE)
            return -1;

        DWORD available = 0;
        if (!PeekNamedPipe(pipe_stdout_, nullptr, 0, nullptr, &available, nullptr) || available == 0)
            return 0;

        DWORD bytesRead = 0;
        if (!ReadFile(pipe_stdout_, buf, static_cast<DWORD>(len), &bytesRead, nullptr))
            return -1;
        return static_cast<ssize_t>(bytesRead);
    }

    bool SubProcess::is_running() const {
        if (process_ == INVALID_HANDLE_VALUE)
            return false;
        DWORD code;
        return GetExitCodeProcess(process_, &code) && code == STILL_ACTIVE;
    }

    void SubProcess::kill() {
        if (pipe_stdout_ != INVALID_HANDLE_VALUE) {
            CloseHandle(pipe_stdout_);
            pipe_stdout_ = INVALID_HANDLE_VALUE;
        }
        if (process_ != INVALID_HANDLE_VALUE) {
            TerminateProcess(process_, 1);
            WaitForSingleObject(process_, 100);
            DWORD code;
            GetExitCodeProcess(process_, &code);
            exit_code_ = static_cast<int>(code);
            CloseHandle(process_);
            process_ = INVALID_HANDLE_VALUE;
        }
    }

    int SubProcess::wait() {
        if (process_ == INVALID_HANDLE_VALUE)
            return exit_code_;

        if (pipe_stdout_ != INVALID_HANDLE_VALUE) {
            CloseHandle(pipe_stdout_);
            pipe_stdout_ = INVALID_HANDLE_VALUE;
        }

        WaitForSingleObject(process_, INFINITE);
        DWORD code;
        GetExitCodeProcess(process_, &code);
        exit_code_ = static_cast<int>(code);
        CloseHandle(process_);
        process_ = INVALID_HANDLE_VALUE;
        return exit_code_;
    }

#endif

} // namespace lfs::python
