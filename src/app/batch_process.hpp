/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <atomic>
#include <chrono>
#include <optional>

#ifdef _WIN32
#include <windows.h>
#else
#include <cerrno>
#include <csignal>
#include <unistd.h>
#endif

namespace lfs::app::detail {

    inline constexpr auto BATCH_STOP_GRACE = std::chrono::seconds{60};

    struct BatchTimeout {
        std::optional<std::chrono::steady_clock::time_point> stopped_at;
        bool forced = false;

        template <typename Stop>
        bool update(const std::chrono::steady_clock::time_point start,
                    const std::chrono::steady_clock::time_point now,
                    const int minutes, Stop stop) {
            if (!stopped_at) {
                // A floating-duration deadline also accommodates INT_MAX minutes.
                const auto deadline = std::chrono::time_point_cast<std::chrono::duration<double>>(start) +
                                      std::chrono::minutes{minutes};
                if (minutes <= 0 || now < deadline)
                    return true;
                stopped_at = now;
                return stop(false);
            }
            if (!forced && now - *stopped_at >= BATCH_STOP_GRACE) {
                if (!stop(true))
                    return false;
                forced = true;
            }
            return true;
        }
    };

    template <typename Stop>
    bool forward_pending_batch_cancel(const int count, int& forwarded, Stop stop) {
        if (count <= forwarded)
            return true;
        if (!stop(count >= 2))
            return false;
        forwarded = count;
        return true;
    }

#ifdef _WIN32
    struct BatchWindowsApi {
        decltype(&::CreateJobObjectW) create_job = ::CreateJobObjectW;
        decltype(&::SetInformationJobObject) configure_job = ::SetInformationJobObject;
        decltype(&::AssignProcessToJobObject) assign_job = ::AssignProcessToJobObject;
        decltype(&::ResumeThread) resume = ::ResumeThread;
        decltype(&::TerminateJobObject) terminate_job = ::TerminateJobObject;
        decltype(&::TerminateProcess) terminate_process = ::TerminateProcess;
        decltype(&::WaitForSingleObject) wait = ::WaitForSingleObject;
    };

    inline bool stop_batch_child(HANDLE process, HANDLE job, const DWORD exit_code,
                                 const BatchWindowsApi& api = {}) {
        if ((job && api.terminate_job(job, exit_code)) || api.terminate_process(process, exit_code))
            return true;
        return api.wait(process, 0) == WAIT_OBJECT_0;
    }

    // The caller creates the child suspended and owns all returned handles.
    inline DWORD prepare_batch_child(const PROCESS_INFORMATION& process, HANDLE& job,
                                     const std::atomic<int>& interrupts, const BatchWindowsApi& api = {}) {
        const auto fail = [&]() {
            const DWORD error = GetLastError();
            // Assignment may have failed, so the direct child is always the fallback.
            if (!api.terminate_process(process.hProcess, 1))
                return GetLastError();
            if (api.wait(process.hProcess, INFINITE) == WAIT_FAILED)
                return GetLastError();
            return error;
        };

        job = api.create_job(nullptr, nullptr);
        if (!job)
            return fail();
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (!api.configure_job(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits)))
            return fail();
        if (!api.assign_job(job, process.hProcess))
            return fail();

        // No work needs saving if cancellation arrived before the child could run.
        if (interrupts.load() > 0) {
            if (!stop_batch_child(process.hProcess, job, 130, api))
                return GetLastError();
        } else if (api.resume(process.hThread) == static_cast<DWORD>(-1)) {
            return fail();
        }
        return ERROR_SUCCESS;
    }
#else
    inline bool prepare_batch_child(const sigset_t& original_mask) {
        if (setpgid(0, 0) != 0)
            return false;
        struct sigaction action {};
        action.sa_handler = SIG_DFL;
        sigemptyset(&action.sa_mask);
        if (sigaction(SIGINT, &action, nullptr) != 0 || sigaction(SIGTERM, &action, nullptr) != 0)
            return false;
        sigset_t child_mask = original_mask;
        sigdelset(&child_mask, SIGINT);
        sigdelset(&child_mask, SIGTERM);
        return sigprocmask(SIG_SETMASK, &child_mask, nullptr) == 0;
    }

    inline bool register_batch_child(const pid_t pid) {
        // Either side of fork may run first; exec/exit can also win this race.
        return setpgid(pid, pid) == 0 || errno == EACCES || errno == ESRCH;
    }

    // Group only: the direct child may already have been reaped.
    inline bool cleanup_batch_group(const pid_t pid) {
        return kill(-pid, SIGKILL) == 0 || errno == ESRCH;
    }

    inline bool signal_batch_child(const pid_t pid, const int signal) {
        if (kill(-pid, signal) == 0)
            return true;
        if (errno != ESRCH)
            return false;
        return kill(pid, signal) == 0 || errno == ESRCH;
    }
#endif

} // namespace lfs::app::detail
