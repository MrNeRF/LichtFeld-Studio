/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <string>
#include <string_view>
#include <vector>

#ifdef _WIN32
#include <BaseTsd.h>
#include <windows.h>
using ssize_t = SSIZE_T;
#else
#include <sys/types.h>
#endif

namespace lfs::vis::editor {

    class StdioProcess {
    public:
        StdioProcess() = default;
        ~StdioProcess();

        StdioProcess(const StdioProcess&) = delete;
        StdioProcess& operator=(const StdioProcess&) = delete;
        StdioProcess(StdioProcess&&) = delete;
        StdioProcess& operator=(StdioProcess&&) = delete;

        bool start(const std::string& program, const std::vector<std::string>& args);
        bool writeAll(std::string_view data);
        [[nodiscard]] ssize_t readStdout(char* buffer, size_t length);
        [[nodiscard]] ssize_t readStderr(char* buffer, size_t length);
        [[nodiscard]] bool isRunning() const;
        [[nodiscard]] int exitCode() const { return exit_code_; }
        void kill();

    private:
#ifdef _WIN32
        HANDLE process_ = INVALID_HANDLE_VALUE;
        HANDLE stdin_write_ = INVALID_HANDLE_VALUE;
        HANDLE stdout_read_ = INVALID_HANDLE_VALUE;
        HANDLE stderr_read_ = INVALID_HANDLE_VALUE;
#else
        mutable pid_t pid_ = -1;
        int stdin_fd_ = -1;
        int stdout_fd_ = -1;
        int stderr_fd_ = -1;
#endif
        mutable int exit_code_ = -1;
    };

} // namespace lfs::vis::editor
