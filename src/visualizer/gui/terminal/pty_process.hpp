/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <string>

#ifdef _WIN32
#include <BaseTsd.h>
#include <windows.h>
using ssize_t = SSIZE_T;
#else
#include <sys/types.h>
#endif

namespace lfs::vis::terminal {

    struct EmbeddedFds {
        int read_fd = -1;
        int write_fd = -1;
        bool valid() const { return read_fd >= 0 && write_fd >= 0; }
    };

    class PtyProcess {
    public:
        static constexpr int DEFAULT_COLS = 80;
        static constexpr int DEFAULT_ROWS = 24;

        PtyProcess() = default;
        ~PtyProcess();

        PtyProcess(const PtyProcess&) = delete;
        PtyProcess& operator=(const PtyProcess&) = delete;
        PtyProcess(PtyProcess&& other) noexcept;
        PtyProcess& operator=(PtyProcess&& other) noexcept;

        bool attach(int fd);
        void close();
        [[nodiscard]] bool is_running() const;

        [[nodiscard]] ssize_t read(char* buf, size_t len);
        [[nodiscard]] ssize_t write(const char* buf, size_t len);

        void resize(int cols, int rows);
        void interrupt();

#ifdef _WIN32
        bool attachPipes(HANDLE read_handle, HANDLE write_handle);
#endif

    private:
        void cleanup();

#ifdef _WIN32
        HPCON hpc_ = nullptr;
        HANDLE pipe_in_ = INVALID_HANDLE_VALUE;
        HANDLE pipe_out_ = INVALID_HANDLE_VALUE;
        HANDLE process_ = INVALID_HANDLE_VALUE;
        HANDLE thread_ = INVALID_HANDLE_VALUE;
        bool pipe_only_ = false;
#else
        int master_fd_ = -1;
        pid_t child_pid_ = -1;
        bool attached_ = false;
#endif
    };

} // namespace lfs::vis::terminal
