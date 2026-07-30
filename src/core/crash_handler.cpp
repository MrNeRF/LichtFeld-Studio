/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/crash_handler.hpp"

#include "core/cuda/memory_arena.hpp"
#include "core/device_fault.hpp"
#include "core/environment.hpp"
#include "core/failure_report.hpp"
#include "core/logger.hpp"
#include "core/pinned_memory_allocator.hpp"
#include "core/tensor.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <functional>
#include <mutex>
#include <string>
#include <typeinfo>

#ifdef _WIN32
#include <windows.h>
#else
#include <csignal>
#include <execinfo.h>
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace lfs::core {

    void flush_diagnostics_noexcept() noexcept {
        try {
            Logger::get().flush();
        } catch (...) {
        }
    }

    void teardown_gpu_before_exit() noexcept {
        try {
            // Phase 6C §9 Ruling 2: drain dedicated DeviceFaultRecord slots
            // (cudaMalloc-owned, never pool memory) BEFORE the tensor memory
            // pool shuts down. device_fault_registry_teardown is no-throw and
            // idempotent (LFS_CUDA_LOG_TEARDOWN on every free).
            device_fault_registry_teardown();
            GlobalArenaManager::instance().shutdown();
            Tensor::shutdown_memory_pool();
            PinnedMemoryAllocator::instance().shutdown();
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): subsystem teardown reports CUDA
            // failures internally; none may escape this sanctioned pre-exit step.
        }
    }

    [[noreturn]] void flush_and_exit(const int code) noexcept {
        flush_diagnostics_noexcept();
        std::_Exit(code);
    }

    namespace {

        constexpr int EXCEPTION_FIREWALL_EXIT_CODE = 70; // EX_SOFTWARE

        std::filesystem::path g_crash_log_path;
        std::once_flag g_install_once;

#ifdef _WIN32
        HANDLE g_crash_log = INVALID_HANDLE_VALUE;

        LONG WINAPI unhandled_exception_filter(EXCEPTION_POINTERS* exception) {
            if (g_crash_log != INVALID_HANDLE_VALUE) {
                std::array<char, 256> header{};
                const DWORD code = exception && exception->ExceptionRecord
                                       ? exception->ExceptionRecord->ExceptionCode
                                       : 0;
                const int length = std::snprintf(
                    header.data(), header.size(),
                    "LichtFeld Studio unhandled exception 0x%08lx\r\n", code);
                DWORD written = 0;
                if (length > 0) {
                    WriteFile(g_crash_log, header.data(), static_cast<DWORD>(length), &written, nullptr);
                }

                std::array<void*, 64> frames{};
                const USHORT count = CaptureStackBackTrace(
                    0, static_cast<DWORD>(frames.size()), frames.data(), nullptr);
                for (USHORT i = 0; i < count; ++i) {
                    const int frame_length = std::snprintf(
                        header.data(), header.size(), "  #%u %p\r\n", i, frames[i]);
                    if (frame_length > 0) {
                        WriteFile(g_crash_log, header.data(),
                                  static_cast<DWORD>(frame_length), &written, nullptr);
                    }
                }
                FlushFileBuffers(g_crash_log);
            }
            return EXCEPTION_CONTINUE_SEARCH;
        }
#else
        int g_crash_log_fd = -1;

        void write_signal_text(const char* text, const size_t length) noexcept {
            if (g_crash_log_fd < 0) {
                return;
            }
            size_t offset = 0;
            while (offset < length) {
                const ssize_t written = ::write(g_crash_log_fd, text + offset, length - offset);
                if (written <= 0) {
                    return;
                }
                offset += static_cast<size_t>(written);
            }
        }

        void fatal_signal_handler(const int signal_number) noexcept {
            static constexpr char SIGSEGV_HEADER[] =
                "LichtFeld Studio fatal signal SIGSEGV; backtrace follows\n";
            static constexpr char SIGABRT_HEADER[] =
                "LichtFeld Studio fatal signal SIGABRT; backtrace follows\n";
            static constexpr char SIGFPE_HEADER[] =
                "LichtFeld Studio fatal signal SIGFPE; backtrace follows\n";
            static constexpr char SIGBUS_HEADER[] =
                "LichtFeld Studio fatal signal SIGBUS; backtrace follows\n";
            static constexpr char UNKNOWN_HEADER[] =
                "LichtFeld Studio fatal signal; backtrace follows\n";

            switch (signal_number) {
            case SIGSEGV:
                write_signal_text(SIGSEGV_HEADER, sizeof(SIGSEGV_HEADER) - 1);
                break;
            case SIGABRT:
                write_signal_text(SIGABRT_HEADER, sizeof(SIGABRT_HEADER) - 1);
                break;
            case SIGFPE:
                write_signal_text(SIGFPE_HEADER, sizeof(SIGFPE_HEADER) - 1);
                break;
            case SIGBUS:
                write_signal_text(SIGBUS_HEADER, sizeof(SIGBUS_HEADER) - 1);
                break;
            default:
                write_signal_text(UNKNOWN_HEADER, sizeof(UNKNOWN_HEADER) - 1);
                break;
            }

            // backtrace_symbols_fd writes directly to the pre-opened descriptor. glibc may
            // lazily load unwind support on its first backtrace(), so installation prewarms it.
            std::array<void*, 128> frames{};
            const int count = ::backtrace(frames.data(), static_cast<int>(frames.size()));
            if (g_crash_log_fd >= 0 && count > 0) {
                ::backtrace_symbols_fd(frames.data(), count, g_crash_log_fd);
            }

            struct sigaction action {};
            action.sa_handler = SIG_DFL;
            sigemptyset(&action.sa_mask);
            action.sa_flags = 0;
            ::sigaction(signal_number, &action, nullptr);

            // A signal is blocked while its handler runs. Unblock it before
            // re-sending so the restored default disposition (and core-dump
            // policy) actually takes effect instead of being bypassed by the
            // fallback _exit below.
            sigset_t unblock_set;
            sigemptyset(&unblock_set);
            sigaddset(&unblock_set, signal_number);
            ::sigprocmask(SIG_UNBLOCK, &unblock_set, nullptr);
            ::kill(::getpid(), signal_number);
            _exit(128 + signal_number);
        }
#endif

        // Shared by the terminate handler and the dispatch firewall: describes
        // whatever exception is active in the calling catch block and emits it
        // through the standard failure-report path. Best effort — swallows its
        // own failures so callers can rely on it never throwing.
        void report_current_exception(const std::string_view family,
                                      const std::string_view contract,
                                      const FailureReportSeverity severity) noexcept {
            std::string exception_type = "<no active exception>";
            std::string what;
            if (const std::exception_ptr exception = std::current_exception()) {
                try {
                    std::rethrow_exception(exception);
                } catch (const std::exception& error) {
                    exception_type = typeid(error).name();
                    what = error.what();
                } catch (...) {
                    exception_type = "<non-std exception>";
                }
            }

            try {
                const auto location = LFS_SOURCE_SITE_CURRENT();
                const std::string message = std::format(
                    "active_exception_type={}, what={}", exception_type,
                    what.empty() ? "<unavailable>" : what);
                emit_failure_report(
                    FailureReport{
                        .family = family,
                        .contract = contract,
                        .expression = "uncaught exception",
                        .message = message,
                        .location = location,
                        .stacktrace_skip_frames = 3,
                    },
                    severity);
            } catch (...) {
            }
        }

        [[noreturn]] void terminate_handler() noexcept {
            report_current_exception("process termination", "std::terminate", FailureReportSeverity::Critical);
            flush_diagnostics_noexcept();
            std::abort();
        }

        [[nodiscard]] bool crash_handlers_disabled() noexcept {
            return environment::flag("LFS_NO_CRASH_HANDLER");
        }

    } // namespace

    void install_crash_handlers() {
        std::call_once(g_install_once, [] {
            if (crash_handlers_disabled()) {
                std::fprintf(stderr, "Crash handlers disabled by LFS_NO_CRASH_HANDLER=1\n");
                return;
            }

#ifdef _WIN32
            std::array<wchar_t, MAX_PATH> temp_path{};
            const DWORD length = GetTempPathW(static_cast<DWORD>(temp_path.size()), temp_path.data());
            const std::filesystem::path base = length > 0 && length < temp_path.size()
                                                   ? std::filesystem::path(temp_path.data())
                                                   : std::filesystem::current_path();
            g_crash_log_path = base / std::format("lichtfeld-studio-crash-{}.log", GetCurrentProcessId());
            g_crash_log = CreateFileW(g_crash_log_path.c_str(), GENERIC_WRITE, FILE_SHARE_READ,
                                      nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
            SetUnhandledExceptionFilter(unhandled_exception_filter);
#else
            g_crash_log_path = std::filesystem::temp_directory_path() /
                               std::format("lichtfeld-studio-crash-{}.log", ::getpid());
            g_crash_log_fd = ::open(g_crash_log_path.c_str(),
                                    O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
            if (g_crash_log_fd >= 0) {
                std::array<void*, 1> warmup{};
                (void)::backtrace(warmup.data(), static_cast<int>(warmup.size()));
            }

            struct sigaction action {};
            action.sa_handler = fatal_signal_handler;
            sigemptyset(&action.sa_mask);
            action.sa_flags = SA_RESETHAND;
            for (const int signal_number : {SIGSEGV, SIGABRT, SIGFPE, SIGBUS}) {
                ::sigaction(signal_number, &action, nullptr);
            }
#endif

            std::set_terminate(terminate_handler);
            const std::string path = g_crash_log_path.string();
            std::fprintf(stderr, "Crash diagnostics: %s\n", path.c_str());
        });
    }

    int run_with_exception_firewall(const std::function<int()>& fn) noexcept {
        try {
            return fn();
        } catch (...) {
            report_current_exception("process dispatch", "exception firewall", FailureReportSeverity::Error);
            flush_diagnostics_noexcept();
            return EXCEPTION_FIREWALL_EXIT_CODE;
        }
    }

} // namespace lfs::core
