/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/batch_process.hpp"
#include <algorithm>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#ifdef __linux__
#include <sys/prctl.h>
#endif
#endif

namespace {
    using namespace lfs::app::detail;

    TEST(BatchProcessTest, PendingCancelIsDeliveredOnceAfterRegistration) {
        std::atomic<int> interrupts{1};
        int forwarded = 0;
        std::vector<bool> stops;
        const auto stop = [&](bool force) {
            stops.push_back(force);
            return true;
        };
        ASSERT_TRUE(forward_pending_batch_cancel(interrupts.load(), forwarded, stop));
        ASSERT_TRUE(forward_pending_batch_cancel(interrupts.load(), forwarded, stop));
        interrupts.fetch_add(1);
        ASSERT_TRUE(forward_pending_batch_cancel(interrupts.load(), forwarded, stop));
        EXPECT_EQ(stops, (std::vector<bool>{false, true}));
    }

    TEST(BatchProcessTest, PendingDoubleCancelForcesStopImmediately) {
        int forwarded = 0;
        bool forced = false;
        ASSERT_TRUE(forward_pending_batch_cancel(2, forwarded, [&](bool force) {
            forced = force;
            return true;
        }));
        EXPECT_TRUE(forced);
        EXPECT_EQ(forwarded, 2);
    }

    TEST(BatchProcessTest, CancelDuringDeliveryRemainsPending) {
        std::atomic<int> interrupts{1};
        int forwarded = 0;
        ASSERT_TRUE(forward_pending_batch_cancel(interrupts.load(), forwarded, [&](bool force) {
            EXPECT_FALSE(force);
            interrupts.fetch_add(1);
            return true;
        }));
        ASSERT_TRUE(forward_pending_batch_cancel(interrupts.load(), forwarded, [&](bool force) {
            EXPECT_TRUE(force);
            return true;
        }));
        EXPECT_EQ(forwarded, 2);
    }

    TEST(BatchProcessTest, FailedDeliveryRemainsPending) {
        int forwarded = 0;
        EXPECT_FALSE(forward_pending_batch_cancel(1, forwarded, [](bool) { return false; }));
        EXPECT_EQ(forwarded, 0);
        EXPECT_TRUE(forward_pending_batch_cancel(1, forwarded, [](bool) { return true; }));
        EXPECT_EQ(forwarded, 1);
    }

    TEST(BatchProcessTest, TimeoutEscalatesAfterGraceAndOnlyOnce) {
        using namespace std::chrono;
        const steady_clock::time_point start{};
        BatchTimeout timeout;
        std::vector<bool> stops;
        const auto stop = [&](bool force) {
            stops.push_back(force);
            return true;
        };
        EXPECT_TRUE(timeout.update(start, start + seconds{59}, 1, stop));
        EXPECT_TRUE(stops.empty());
        EXPECT_TRUE(timeout.update(start, start + minutes{1}, 1, stop));
        EXPECT_EQ(stops, (std::vector<bool>{false}));
        EXPECT_TRUE(timeout.update(start, start + minutes{1} + BATCH_STOP_GRACE - milliseconds{1}, 1, stop));
        EXPECT_EQ(stops.size(), 1u);
        EXPECT_TRUE(timeout.update(start, start + minutes{1} + BATCH_STOP_GRACE, 1, stop));
        EXPECT_EQ(stops, (std::vector<bool>{false, true}));
        EXPECT_TRUE(timeout.update(start, start + hours{1}, 1, stop));
        EXPECT_EQ(stops.size(), 2u);
    }

    TEST(BatchProcessTest, DisabledAndLargeTimeoutDoNotOverflow) {
        const std::chrono::steady_clock::time_point start{};
        for (const int minutes : {0, std::numeric_limits<int>::max()}) {
            BatchTimeout timeout;
            EXPECT_TRUE(timeout.update(start, start + std::chrono::hours{24}, minutes, [](bool) {
                ADD_FAILURE() << "Unexpected timeout";
                return true;
            }));
            EXPECT_FALSE(timeout.stopped_at);
        }
    }

    TEST(BatchProcessTest, FailedTimeoutForceStopIsReportedAndCanBeRetried) {
        using namespace std::chrono;
        const steady_clock::time_point start{};
        BatchTimeout timeout;
        ASSERT_TRUE(timeout.update(start, start + minutes{1}, 1, [](bool) { return true; }));
        EXPECT_FALSE(timeout.update(start, start + minutes{1} + BATCH_STOP_GRACE, 1, [](bool force) {
            EXPECT_TRUE(force);
            return false;
        }));
        EXPECT_FALSE(timeout.forced);
        EXPECT_TRUE(timeout.update(start, start + minutes{1} + BATCH_STOP_GRACE, 1, [](bool force) {
            EXPECT_TRUE(force);
            return true;
        }));
        EXPECT_TRUE(timeout.forced);
    }

#ifdef _WIN32
    class BatchWindowsProcessTest : public ::testing::Test {
    protected:
        enum class Failure { None,
                             Create,
                             Configure,
                             Assign,
                             Resume,
                             StopJob,
                             StopAll };
        static inline Failure failure;
        static inline std::vector<std::string> calls;
        static inline bool terminated;
        static inline bool reject_direct_stop;
        static inline std::atomic<int>* cancel_on_resume;

        static BOOL succeed_unless(Failure stage) {
            if (failure != stage)
                return TRUE;
            SetLastError(ERROR_ACCESS_DENIED);
            return FALSE;
        }
        static HANDLE WINAPI create(LPSECURITY_ATTRIBUTES, LPCWSTR) {
            calls.emplace_back("create");
            return succeed_unless(Failure::Create) ? reinterpret_cast<HANDLE>(3) : nullptr;
        }
        static BOOL WINAPI configure(HANDLE, JOBOBJECTINFOCLASS, LPVOID info, DWORD) {
            calls.emplace_back("configure");
            EXPECT_EQ(static_cast<JOBOBJECT_EXTENDED_LIMIT_INFORMATION*>(info)->BasicLimitInformation.LimitFlags,
                      JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE);
            return succeed_unless(Failure::Configure);
        }
        static BOOL WINAPI assign(HANDLE, HANDLE) {
            calls.emplace_back("assign");
            return succeed_unless(Failure::Assign);
        }
        static DWORD WINAPI resume(HANDLE) {
            calls.emplace_back("resume");
            if (cancel_on_resume)
                cancel_on_resume->fetch_add(1);
            return succeed_unless(Failure::Resume) ? 1 : static_cast<DWORD>(-1);
        }
        static BOOL WINAPI stop_job(HANDLE, UINT) {
            calls.emplace_back("stop_job");
            if (failure == Failure::StopJob || failure == Failure::StopAll) {
                SetLastError(ERROR_ACCESS_DENIED);
                return FALSE;
            }
            terminated = true;
            return TRUE;
        }
        static BOOL WINAPI stop_process(HANDLE, UINT) {
            calls.emplace_back("stop_process");
            if (reject_direct_stop) {
                SetLastError(ERROR_ACCESS_DENIED);
                return FALSE;
            }
            if (!succeed_unless(Failure::StopAll))
                return FALSE;
            terminated = true;
            return TRUE;
        }
        static DWORD WINAPI wait(HANDLE, DWORD timeout) {
            calls.emplace_back("wait");
            EXPECT_TRUE(terminated || timeout == 0) << "Must not block after a failed termination";
            return terminated ? WAIT_OBJECT_0 : WAIT_TIMEOUT;
        }
        void SetUp() override {
            failure = Failure::None;
            calls.clear();
            terminated = false;
            reject_direct_stop = false;
            cancel_on_resume = nullptr;
        }
        BatchWindowsApi api{create, configure, assign, resume, stop_job, stop_process, wait};
        PROCESS_INFORMATION process{reinterpret_cast<HANDLE>(1), reinterpret_cast<HANDLE>(2), 123, 456};
        HANDLE job = nullptr;
        std::atomic<int> interrupts{0};
    };

    TEST_F(BatchWindowsProcessTest, AssignsToConfiguredJobBeforeResume) {
        EXPECT_EQ(prepare_batch_child(process, job, interrupts, api), ERROR_SUCCESS);
        EXPECT_EQ(calls, (std::vector<std::string>{"create", "configure", "assign", "resume"}));
    }

    TEST_F(BatchWindowsProcessTest, SetupFailuresTerminateDirectChild) {
        for (const auto stage : {Failure::Create, Failure::Configure, Failure::Assign, Failure::Resume}) {
            SCOPED_TRACE(static_cast<int>(stage));
            failure = stage;
            calls.clear();
            terminated = false;
            EXPECT_EQ(prepare_batch_child(process, job, interrupts, api), ERROR_ACCESS_DENIED);
            ASSERT_GE(calls.size(), 3u);
            EXPECT_EQ(calls[calls.size() - 2], "stop_process");
            EXPECT_EQ(calls.back(), "wait");
            EXPECT_TRUE(terminated);
            if (stage != Failure::Resume)
                EXPECT_EQ(std::count(calls.begin(), calls.end(), "resume"), 0);
        }
    }

    TEST_F(BatchWindowsProcessTest, PendingCancelNeverResumesChild) {
        for (const int count : {1, 2}) {
            calls.clear();
            interrupts.store(count);
            EXPECT_EQ(prepare_batch_child(process, job, interrupts, api), ERROR_SUCCESS);
            EXPECT_EQ(calls, (std::vector<std::string>{"create", "configure", "assign", "stop_job"}));
        }
    }

    TEST_F(BatchWindowsProcessTest, FailedAssignmentAndTerminationDoesNotWait) {
        failure = Failure::Assign;
        reject_direct_stop = true;
        EXPECT_EQ(prepare_batch_child(process, job, interrupts, api), ERROR_ACCESS_DENIED);
        EXPECT_EQ(calls, (std::vector<std::string>{"create", "configure", "assign", "stop_process"}));
    }

    TEST_F(BatchWindowsProcessTest, CancelDuringResumeIsDeliveredAfterRegistration) {
        cancel_on_resume = &interrupts;
        ASSERT_EQ(prepare_batch_child(process, job, interrupts, api), ERROR_SUCCESS);
        int forwarded = 0;
        bool delivered = false;
        ASSERT_TRUE(forward_pending_batch_cancel(interrupts.load(), forwarded, [&](bool force) {
            EXPECT_FALSE(force);
            delivered = true;
            return true;
        }));
        EXPECT_TRUE(delivered);
    }

    TEST_F(BatchWindowsProcessTest, JobStopFailureFallsBackToDirectChild) {
        failure = Failure::StopJob;
        ASSERT_TRUE(stop_batch_child(process.hProcess, reinterpret_cast<HANDLE>(3), 130, api));
        EXPECT_EQ(calls, (std::vector<std::string>{"stop_job", "stop_process"}));
    }

    TEST_F(BatchWindowsProcessTest, FailedStopNeverWaitsIndefinitely) {
        failure = Failure::StopAll;
        EXPECT_FALSE(stop_batch_child(process.hProcess, reinterpret_cast<HANDLE>(3), 130, api));
        EXPECT_EQ(calls, (std::vector<std::string>{"stop_job", "stop_process", "wait"}));
    }

    TEST_F(BatchWindowsProcessTest, AlreadyExitedChildCountsAsStopped) {
        failure = Failure::StopAll;
        terminated = true;
        EXPECT_TRUE(stop_batch_child(process.hProcess, reinterpret_cast<HANDLE>(3), 130, api));
    }
#else
    std::atomic<int> pending_interrupts{0};

    void record_interrupt(int) {
        ++pending_interrupts;
    }

    void cancel_before_exec(const int signal, const int count, const bool child_first = false) {
        alarm(10);
        struct sigaction action {};
        action.sa_handler = record_interrupt;
        sigemptyset(&action.sa_mask);
        if (sigaction(SIGINT, &action, nullptr) || sigaction(SIGTERM, &action, nullptr))
            _exit(10);
        sigset_t blocked;
        sigemptyset(&blocked);
        sigaddset(&blocked, SIGINT);
        sigaddset(&blocked, SIGTERM);
        if (sigprocmask(SIG_BLOCK, &blocked, nullptr))
            _exit(11);
        int gate[2], ready[2];
        if (pipe(gate) || pipe(ready))
            _exit(12);
        const pid_t pid = fork();
        if (pid < 0)
            _exit(13);
        if (pid == 0) {
            alarm(5);
            close(gate[1]);
            close(ready[0]);
            if (child_first && !prepare_batch_child(blocked))
                _exit(14);
            if (write(ready[1], "r", 1) != 1)
                _exit(15);
            char byte;
            if (read(gate[0], &byte, 1) != 1)
                _exit(16);
            if (!child_first && !prepare_batch_child(blocked))
                _exit(17);
            execl("/bin/sh", "sh", "-c", "exit 99", static_cast<char*>(nullptr));
            _exit(18);
        }
        close(gate[0]);
        close(ready[1]);
        char byte;
        if (read(ready[0], &byte, 1) != 1)
            _exit(19);
        if (sigprocmask(SIG_UNBLOCK, &blocked, nullptr))
            _exit(20);
        pending_interrupts.store(0);
        for (int i = 0; i < count; ++i)
            raise(signal);
        if (!register_batch_child(pid))
            _exit(21);
        int forwarded = 0;
        if (!forward_pending_batch_cancel(pending_interrupts.load(), forwarded, [&](bool force) {
                return signal_batch_child(pid, force ? SIGKILL : signal);
            }))
            _exit(22);
        // A force-stopped child may have closed its end before this write.
        ::signal(SIGPIPE, SIG_IGN);
        const auto written = write(gate[1], "g", 1);
        static_cast<void>(written);
        int status = 0;
        while (waitpid(pid, &status, 0) < 0) {
            if (errno != EINTR)
                _exit(23);
        }
        _exit(WIFSIGNALED(status) && WTERMSIG(status) == (count >= 2 ? SIGKILL : signal) ? 0 : 24);
    }

    TEST(BatchProcessDeathTest, CancelBeforeRegistrationIsDeliveredBeforeExec) {
        ASSERT_EXIT(cancel_before_exec(SIGINT, 1), ::testing::ExitedWithCode(0), "");
    }

    TEST(BatchProcessDeathTest, SigtermBeforeExecUsesDefaultDisposition) {
        ASSERT_EXIT(cancel_before_exec(SIGTERM, 1), ::testing::ExitedWithCode(0), "");
    }

    TEST(BatchProcessDeathTest, DoubleCancelBeforeExecForcesKill) {
        ASSERT_EXIT(cancel_before_exec(SIGINT, 2), ::testing::ExitedWithCode(0), "");
    }

    TEST(BatchProcessDeathTest, ChildCanEstablishGroupBeforeParent) {
        ASSERT_EXIT(cancel_before_exec(SIGINT, 1, true), ::testing::ExitedWithCode(0), "");
    }

#ifdef __linux__
    void force_stop_process_tree(const int direct_exit = 0) {
        alarm(10);
        if (prctl(PR_SET_CHILD_SUBREAPER, 1) != 0)
            _exit(30);
        int ready[2];
        if (pipe(ready))
            _exit(31);
        const pid_t child = fork();
        if (child < 0)
            _exit(32);
        if (child == 0) {
            alarm(5);
            close(ready[0]);
            sigset_t mask;
            sigemptyset(&mask);
            if (!prepare_batch_child(mask))
                _exit(33);
            const pid_t grandchild = fork();
            if (grandchild < 0)
                _exit(34);
            if (grandchild == 0) {
                alarm(5);
                ::signal(SIGTERM, SIG_IGN);
                ::signal(SIGINT, SIG_IGN);
                const pid_t self = getpid();
                if (write(ready[1], &self, sizeof(self)) != sizeof(self))
                    _exit(35);
            } else if (direct_exit == 1) {
                _exit(0);
            } else if (direct_exit == 2) {
                raise(SIGTERM);
                _exit(41);
            }
            while (true)
                pause();
        }
        close(ready[1]);
        pid_t grandchild = 0;
        if (read(ready[0], &grandchild, sizeof(grandchild)) != sizeof(grandchild))
            _exit(36);
        if (!register_batch_child(child) || getpgid(grandchild) != child)
            _exit(37);
        if (direct_exit != 0) {
            int child_status = 0, grandchild_status = 0;
            if (waitpid(child, &child_status, 0) != child || !cleanup_batch_group(child) ||
                waitpid(grandchild, &grandchild_status, 0) != grandchild)
                _exit(42);
            const bool child_ok = direct_exit == 1 ? WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0
                                                   : WIFSIGNALED(child_status) && WTERMSIG(child_status) == SIGTERM;
            _exit(child_ok && WIFSIGNALED(grandchild_status) && WTERMSIG(grandchild_status) == SIGKILL ? 0 : 43);
        }
        int forwarded = 0;
        if (!forward_pending_batch_cancel(2, forwarded, [&](bool force) {
                return signal_batch_child(child, force ? SIGKILL : SIGINT);
            }))
            _exit(38);
        int child_status = 0, grandchild_status = 0;
        if (waitpid(child, &child_status, 0) != child || waitpid(grandchild, &grandchild_status, 0) != grandchild)
            _exit(39);
        _exit(WIFSIGNALED(child_status) && WTERMSIG(child_status) == SIGKILL &&
                      WIFSIGNALED(grandchild_status) && WTERMSIG(grandchild_status) == SIGKILL
                  ? 0
                  : 40);
    }

    TEST(BatchProcessDeathTest, ForceStopReachesGrandchild) {
        ASSERT_EXIT(force_stop_process_tree(), ::testing::ExitedWithCode(0), "");
    }

    TEST(BatchProcessDeathTest, CleansGrandchildAfterDirectChildExitsNormally) {
        ASSERT_EXIT(force_stop_process_tree(1), ::testing::ExitedWithCode(0), "");
    }

    TEST(BatchProcessDeathTest, CleansGrandchildAfterDirectChildDiesBySignal) {
        ASSERT_EXIT(force_stop_process_tree(2), ::testing::ExitedWithCode(0), "");
    }
#endif
#endif
} // namespace
