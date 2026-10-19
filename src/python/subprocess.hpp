/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "python_runtime_api.hpp"

#include <string>
#include <vector>

#ifdef _WIN32
#include <BaseTsd.h>
#include <windows.h>
using ssize_t = SSIZE_T;
#else
#include <sys/types.h>
#endif

namespace lfs::python {

    class LFS_PYTHON_RUNTIME_API SubProcess {
    public:
        SubProcess() = default;
        ~SubProcess();

        SubProcess(const SubProcess&) = delete;
        SubProcess& operator=(const SubProcess&) = delete;
        SubProcess(SubProcess&&) = delete;
        SubProcess& operator=(SubProcess&&) = delete;

        // Start subprocess with given program and args
        bool start(const std::string& program, const std::vector<std::string>& args);

        // Non-blocking read: returns bytes read, 0 if no data, -1 on error/EOF
        [[nodiscard]] ssize_t read(char* buf, size_t len);

        // Check if process is still running
        [[nodiscard]] bool is_running() const;

        // Get exit code (only valid after process exits)
        [[nodiscard]] int exit_code() const { return exit_code_; }

        // Kill the process (SIGTERM then SIGKILL)
        void kill();

        // Wait for process to exit, returns exit code
        int wait();

    private:
#ifdef _WIN32
        HANDLE process_ = INVALID_HANDLE_VALUE;
        HANDLE pipe_stdout_ = INVALID_HANDLE_VALUE;
#else
        pid_t pid_ = -1;
        int stdout_fd_ = -1;
#endif
        int exit_code_ = -1;
    };

} // namespace lfs::python
