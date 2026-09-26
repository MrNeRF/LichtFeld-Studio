/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/crash_handler.hpp"

#if !defined(LFS_UNICODE_TEST_STANDALONE) && LFS_HAS_CUDA
#include "core/cuda/memory_arena.hpp"
#include "core/device_fault.hpp"
#endif
#include "core/environment.hpp"
#include "core/failure_report.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#if !defined(LFS_UNICODE_TEST_STANDALONE) && LFS_HAS_CUDA
#include "core/pinned_memory_allocator.hpp"
#endif
#ifndef LFS_UNICODE_TEST_STANDALONE
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#endif
#include "core/user_paths.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
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

#ifndef LFS_UNICODE_TEST_STANDALONE
    namespace {
        constexpr int kMaxGpuPreShutdownHooks = 32;
        std::array<GpuPreShutdownHook, kMaxGpuPreShutdownHooks> g_gpu_pre_shutdown_hooks{};
        std::atomic<int> g_gpu_pre_shutdown_hook_count{0};
        std::atomic<bool> g_gpu_pre_shutdown_hooks_ran{false};
        std::atomic<bool> g_gpu_process_teardown_started{false};
        std::atomic<bool> g_gpu_pre_shutdown_overflow_logged{false};

        void run_gpu_pre_shutdown_hooks_once() noexcept {
            bool expected = false;
            if (!g_gpu_pre_shutdown_hooks_ran.compare_exchange_strong(
                    expected, true, std::memory_order_acq_rel)) {
                return;
            }
            const int n = g_gpu_pre_shutdown_hook_count.load(std::memory_order_acquire);
            for (int i = 0; i < n && i < kMaxGpuPreShutdownHooks; ++i) {
                if (GpuPreShutdownHook hook = g_gpu_pre_shutdown_hooks[static_cast<size_t>(i)]) {
                    try {
                        hook();
                    } catch (...) {
                        // LFS-CENSUS-OK(empty-catch): hooks must not escape;
                        // continue remaining holders so pool shutdown still runs.
                    }
                }
            }
        }
    } // namespace

    void register_gpu_pre_shutdown_hook(const GpuPreShutdownHook hook) noexcept {
        if (!hook) {
            return;
        }
        const int index = g_gpu_pre_shutdown_hook_count.fetch_add(1, std::memory_order_acq_rel);
        if (index < 0 || index >= kMaxGpuPreShutdownHooks) {
            if (!g_gpu_pre_shutdown_overflow_logged.exchange(true, std::memory_order_relaxed)) {
                try {
                    LOG_ERROR("register_gpu_pre_shutdown_hook: capacity {} exceeded; "
                              "hook dropped during static/TLS release",
                              kMaxGpuPreShutdownHooks);
                } catch (...) {
                }
            }
            return;
        }
        g_gpu_pre_shutdown_hooks[static_cast<size_t>(index)] = hook;
    }

    bool gpu_process_teardown_started() noexcept {
        return g_gpu_process_teardown_started.load(std::memory_order_acquire);
    }
#endif

    void flush_diagnostics_noexcept() noexcept {
        try {
            Logger::get().flush();
        } catch (...) {
        }
    }

#ifndef LFS_UNICODE_TEST_STANDALONE
    void teardown_gpu_before_exit() noexcept {
        try {
            static_cast<void>(shutdown_gpu_backend(GpuBackend::Vulkan));
            // release every registered long-lived CUDA holder
            // (TLS FastGS sort workspaces, rasterizer image caches, PPISP shared
            // statics, mirror mult cache, nan-check scratch,) while the pool
            // and CUDA context are still usable. After this returns, static/TLS
            // dtors must find empty holders — otherwise they free after the
            // Meyers-singleton pool is destroyed → SIGSEGV (exit 139).
#if LFS_HAS_CUDA
            const bool cuda_usable = gpu_backend_available(GpuBackend::CUDA);
            if (cuda_usable) {
                run_gpu_pre_shutdown_hooks_once();
            }
            g_gpu_process_teardown_started.store(true, std::memory_order_release);

            if (!cuda_usable) {
                return;
            }

            // Drain dedicated DeviceFaultRecord slots (cudaMalloc-owned, never
            // pool memory) before the tensor memory
            // pool shuts down. device_fault_registry_teardown is no-throw and
            // idempotent (LFS_CUDA_LOG_TEARDOWN on every free).
            device_fault_registry_teardown();
            GlobalArenaManager::instance().shutdown();
            Tensor::shutdown_memory_pool();
            PinnedMemoryAllocator::instance().shutdown();
#else
            g_gpu_process_teardown_started.store(true, std::memory_order_release);
#endif
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): subsystem teardown reports CUDA
            // failures internally; none may escape this sanctioned pre-exit step.
        }
    }
#endif

    [[noreturn]] void flush_and_exit(const int code) noexcept {
        flush_diagnostics_noexcept();
        std::_Exit(code);
    }

    namespace {

        constexpr int EXCEPTION_FIREWALL_EXIT_CODE = 70; // EX_SOFTWARE

        std::filesystem::path g_crash_log_path;
        std::once_flag g_install_once;
        std::mutex g_crash_log_mutex;

#ifdef _WIN32
        HANDLE open_crash_log() noexcept {
            if (g_crash_log_path.empty())
                return INVALID_HANDLE_VALUE;
            return CreateFileW(g_crash_log_path.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE,
                               nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        }

        LONG WINAPI unhandled_exception_filter(EXCEPTION_POINTERS* exception) {
            const HANDLE crash_log = open_crash_log();
            if (crash_log != INVALID_HANDLE_VALUE) {
                std::array<char, 256> header{};
                const DWORD code = exception && exception->ExceptionRecord
                                       ? exception->ExceptionRecord->ExceptionCode
                                       : 0;
                const int length = std::snprintf(
                    header.data(), header.size(),
                    "LichtFeld Studio unhandled exception 0x%08lx\r\n", code);
                DWORD written = 0;
                if (length > 0) {
                    WriteFile(crash_log, header.data(), static_cast<DWORD>(length), &written, nullptr);
                }

                std::array<void*, 64> frames{};
                const USHORT count = CaptureStackBackTrace(
                    0, static_cast<DWORD>(frames.size()), frames.data(), nullptr);
                for (USHORT i = 0; i < count; ++i) {
                    const int frame_length = std::snprintf(
                        header.data(), header.size(), "  #%u %p\r\n", i, frames[i]);
                    if (frame_length > 0) {
                        WriteFile(crash_log, header.data(),
                                  static_cast<DWORD>(frame_length), &written, nullptr);
                    }
                }
                FlushFileBuffers(crash_log);
                CloseHandle(crash_log);
            }
            return EXCEPTION_CONTINUE_SEARCH;
        }
#else
        int open_crash_log() noexcept {
            if (g_crash_log_path.empty())
                return -1;
            return ::open(g_crash_log_path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
        }

        void write_signal_text(const int fd, const char* text, const size_t length) noexcept {
            if (fd < 0) {
                return;
            }
            size_t offset = 0;
            while (offset < length) {
                const ssize_t written = ::write(fd, text + offset, length - offset);
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

            const int fd = open_crash_log();
            switch (signal_number) {
            case SIGSEGV:
                write_signal_text(fd, SIGSEGV_HEADER, sizeof(SIGSEGV_HEADER) - 1);
                break;
            case SIGABRT:
                write_signal_text(fd, SIGABRT_HEADER, sizeof(SIGABRT_HEADER) - 1);
                break;
            case SIGFPE:
                write_signal_text(fd, SIGFPE_HEADER, sizeof(SIGFPE_HEADER) - 1);
                break;
            case SIGBUS:
                write_signal_text(fd, SIGBUS_HEADER, sizeof(SIGBUS_HEADER) - 1);
                break;
            default:
                write_signal_text(fd, UNKNOWN_HEADER, sizeof(UNKNOWN_HEADER) - 1);
                break;
            }

            // glibc may
            // lazily load unwind support on its first backtrace(), so installation prewarms it.
            std::array<void*, 128> frames{};
            const int count = ::backtrace(frames.data(), static_cast<int>(frames.size()));
            if (fd >= 0 && count > 0) {
                ::backtrace_symbols_fd(frames.data(), count, fd);
                if (fd >= 0)
                    ::close(fd);
            }

            struct sigaction action{};
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
        // through the standard failure-report path. Best effort: swallows its
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

    void write_crash_diagnostic(const std::string_view text) noexcept {
        try {
            const std::lock_guard lock(g_crash_log_mutex);
#ifdef _WIN32
            const HANDLE crash_log = open_crash_log();
            if (crash_log == INVALID_HANDLE_VALUE)
                return;
            size_t offset = 0;
            while (offset < text.size()) {
                const DWORD chunk = static_cast<DWORD>(std::min<size_t>(text.size() - offset, MAXDWORD));
                DWORD written = 0;
                if (!WriteFile(crash_log, text.data() + offset, chunk, &written, nullptr) || written == 0) {
                    CloseHandle(crash_log);
                    return;
                }
                offset += written;
            }
            DWORD written = 0;
            WriteFile(crash_log, "\n", 1, &written, nullptr);
            FlushFileBuffers(crash_log);
            CloseHandle(crash_log);
#else
            const int fd = open_crash_log();
            write_signal_text(fd, text.data(), text.size());
            write_signal_text(fd, "\n", 1);
            if (fd >= 0) {
                (void)::fsync(fd);
                ::close(fd);
            }
#endif
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): A diagnostic sink failure must not recurse
            // into failure reporting or replace the original failure.
        }
    }

    void install_crash_handlers() {
        std::call_once(g_install_once, [] {
            if (crash_handlers_disabled()) {
                std::fprintf(stderr, "Crash handlers disabled by LFS_NO_CRASH_HANDLER=1\n");
                return;
            }

            const auto paths = UserPaths::resolve();
            if (paths) {
                std::error_code error;
                std::filesystem::create_directories(paths->logDir(), error);
                if (!error) {
                    const auto date = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
#ifdef _WIN32
                    const auto pid = GetCurrentProcessId();
#else
                    const auto pid = ::getpid();
#endif
                    g_crash_log_path = paths->logDir() /
                                       std::format("lichtfeld-studio-crash-{:%Y-%m-%d-%H%M%S}-{}.log", date, pid);
                } else {
                    std::fprintf(stderr, "Could not create crash log directory: %s\n", error.message().c_str());
                }
            } else {
                std::fprintf(stderr, "Could not resolve LichtFeld home for crash logs\n");
            }
#ifdef _WIN32
            SetUnhandledExceptionFilter(unhandled_exception_filter);
#else
            std::array<void*, 1> warmup{};
            (void)::backtrace(warmup.data(), static_cast<int>(warmup.size()));

            struct sigaction action {};
            action.sa_handler = fatal_signal_handler;
            sigemptyset(&action.sa_mask);
            action.sa_flags = SA_RESETHAND;
            for (const int signal_number : {SIGSEGV, SIGABRT, SIGFPE, SIGBUS}) {
                ::sigaction(signal_number, &action, nullptr);
            }
#endif

            std::set_terminate(terminate_handler);
            const std::string path = path_to_utf8(g_crash_log_path);
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
