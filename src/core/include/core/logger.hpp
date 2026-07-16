/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "core/export.hpp"
#include "core/source_site.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#if defined(__CUDACC__)
#include <cstdio>
#else
#include <format>
#endif
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lfs::core {

    enum class LogLevel : uint8_t {
        Trace = 0,
        Debug = 1,
        Info = 2,
        Performance = 3,
        Warn = 4,
        Error = 5,
        Critical = 6,
        Off = 7
    };

    enum class LogModule : uint8_t {
        Core = 0,
        Rendering = 1,
        Visualizer = 2,
        Loader = 3,
        Scene = 4,
        Training = 5,
        Input = 6,
        GUI = 7,
        Window = 8,
        Memory = 9,
        Unknown = 10,
        Count = 11
    };

    struct LFS_LOGGER_API LogEntrySnapshot {
        std::chrono::system_clock::time_point timestamp{};
        LogLevel level = LogLevel::Info;
        std::string file;
        int line = 0;
        std::string message;
    };

    using LogHandler = std::function<void(LogLevel level, const SourceSite& loc, std::string_view msg)>;
    using LogHandlerToken = uint32_t;

    class LFS_LOGGER_API Logger {
    public:
        static Logger& get();

        void init();
        void init(LogLevel console_level);
        void init(LogLevel console_level, const std::string& log_file);
        void init(LogLevel console_level,
                  const std::string& log_file,
                  const std::string& filter_pattern);
        void init(LogLevel console_level,
                  const std::string& log_file,
                  const std::string& filter_pattern,
                  bool use_stderr);
        // default_log_dir_override replaces the resolved per-user LichtFeld directory
        // (normally ~/.lichtfeld) used to place the always-on durable log; empty means
        // use the real per-user directory. Exists so tests can redirect it without a
        // process-global env var.
        void init(LogLevel console_level,
                  const std::string& log_file,
                  const std::string& filter_pattern,
                  bool use_stderr,
                  const std::string& default_log_dir_override);

        // Resolves the durable default log file path (<user_dir>/logs/lichtfeld.log).
        // user_dir_override replaces the per-user LichtFeld directory when non-empty.
        [[nodiscard]] static std::string default_log_file_path(const std::string& user_dir_override = {});

        // True once init() has installed a backing spdlog logger. ErrorReporter
        // uses this to detect a not-yet-initialized Logger and fall back to a
        // direct stderr report instead of silently dropping the message.
        [[nodiscard]] bool is_ready() const noexcept;

        // Test-only: returns the Logger singleton to its pre-init() state
        // (is_ready() == false) without touching registered log handlers or
        // the global level. Mirrors force_next_error_allocation_to_fail_for_testing
        // and reset_failure_report_dedup_for_testing elsewhere in core.
        void reset_for_testing() noexcept;

        LogHandlerToken add_log_handler(LogHandler handler);
        void remove_log_handler(LogHandlerToken handler_token);

        // Log a pre-formatted message (called by macros)
        void log(LogLevel level, const SourceSite& loc, std::string_view msg);

        // Module control
        void enable_module(LogModule module, bool enabled = true);
        void set_module_level(LogModule module, LogLevel level);
        void set_level(LogLevel level);
        void flush();
        [[nodiscard]] LogLevel level() const;
        [[nodiscard]] size_t buffered_log_count() const;
        [[nodiscard]] uint64_t buffered_log_generation() const;
        [[nodiscard]] std::vector<LogEntrySnapshot> buffered_logs() const;
        [[nodiscard]] std::string buffered_logs_as_text() const;

        bool is_enabled(LogLevel level) const {
            return should_emit(level);
        }

        // Runtime string logging - no format args, works for both CUDA and non-CUDA
        // Use this when you need to log a dynamically constructed string
        void log_internal(LogLevel level, const SourceSite& loc, const std::string& msg) {
            if (!should_emit(level))
                return;
            log(level, loc, msg);
        }

        // Template wrapper for formatting (header-only for convenience)
        // PERF: Fast-path check BEFORE expensive std::format() call.
        // If the log level is below the global threshold, skip formatting entirely.
        // This reduces LOG_DEBUG overhead from ~1-5μs to <50ns when debug logging is disabled.
        template <typename... Args>
        void log_internal(LogLevel level, const SourceSite& loc,
#ifdef __CUDACC__
                          const char* fmt, Args&&... args) {
            // Fast path: skip formatting if the message would be dropped by all active sinks.
            if (!should_emit(level))
                return;

            // CUDA: use snprintf
            char buffer[1024];

            // Helper lambda to avoid problems with -Wformat-security warnings
            auto safe_snprintf = [&](char* buf, size_t size) {
                if constexpr (sizeof...(Args) == 0) {
                    return std::snprintf(buf, size, "%s", fmt);
                } else {
                    return std::snprintf(buf, size, fmt, std::forward<Args>(args)...);
                }
            };

            const int written = safe_snprintf(buffer, sizeof(buffer));
            if (written < 0)
                return;

            std::string msg;
            if (static_cast<size_t>(written) >= sizeof(buffer)) {
                msg.resize(static_cast<size_t>(written) + 1);
                safe_snprintf(msg.data(), msg.size());
                msg.resize(static_cast<size_t>(written));
            } else {
                msg.assign(buffer, static_cast<size_t>(written));
            }
            log(level, loc, msg);
        }
#else
                          std::format_string<Args...> fmt, Args&&... args) {
            // Fast path: skip formatting if the message would be dropped by all active sinks.
            if (!should_emit(level))
                return;

            log(level, loc, std::format(fmt, std::forward<Args>(args)...));
        }
#endif

        [[nodiscard]] static std::string_view to_string(LogLevel level);

    private:
        Logger();
        ~Logger();
        Logger(const Logger&) = delete;
        Logger& operator=(const Logger&) = delete;

        [[nodiscard]] static constexpr bool passes_display_filter(const LogLevel message_level,
                                                                  const LogLevel display_level) {
            if (message_level == LogLevel::Off || display_level == LogLevel::Off)
                return false;
            // Performance logs are opt-in for the console/UI stream. Keep them out of the
            // normal info/warn/error thresholds unless the user explicitly asks for perf or
            // a fully verbose developer stream.
            if (message_level == LogLevel::Performance)
                return display_level == LogLevel::Performance ||
                       display_level == LogLevel::Trace ||
                       display_level == LogLevel::Debug;
            if (display_level == LogLevel::Performance)
                return static_cast<uint8_t>(message_level) >= static_cast<uint8_t>(LogLevel::Warn);
            return static_cast<uint8_t>(message_level) >= static_cast<uint8_t>(display_level);
        }

        [[nodiscard]] bool should_emit(const LogLevel level) const {
            if (level == LogLevel::Off)
                return false;
            if (capture_all_to_file_.load(std::memory_order_relaxed))
                return true;
            return passes_display_filter(
                level,
                static_cast<LogLevel>(global_level_.load(std::memory_order_relaxed)));
        }

        struct Impl;
        std::unique_ptr<Impl> impl_;

        std::atomic<uint8_t> global_level_{static_cast<uint8_t>(LogLevel::Info)};
        std::atomic<bool> capture_all_to_file_{false};
        std::array<std::atomic<bool>, static_cast<size_t>(LogModule::Count)> module_enabled_{};
        std::array<std::atomic<uint8_t>, static_cast<size_t>(LogModule::Count)> module_level_{};
    };

    // Scoped timer for performance measurement
    class LFS_LOGGER_API ScopedTimer {
    public:
        explicit ScopedTimer(std::string name, LogLevel level, SourceSite loc);
        ScopedTimer(std::string name, double min_log_ms,
                    LogLevel level, SourceSite loc);
        ~ScopedTimer();

    private:
        std::chrono::high_resolution_clock::time_point start_;
        std::string name_;
        double min_log_ms_ = 0.0;
        LogLevel level_;
        SourceSite loc_;
        bool diagnostics_scope_active_ = false;
    };

} // namespace lfs::core

// Global macros
#define LOG_TRACE(...) \
    ::lfs::core::Logger::get().log_internal(::lfs::core::LogLevel::Trace, LFS_SOURCE_SITE_CURRENT(), __VA_ARGS__)

#define LOG_DEBUG(...) \
    ::lfs::core::Logger::get().log_internal(::lfs::core::LogLevel::Debug, LFS_SOURCE_SITE_CURRENT(), __VA_ARGS__)

#define LOG_INFO(...) \
    ::lfs::core::Logger::get().log_internal(::lfs::core::LogLevel::Info, LFS_SOURCE_SITE_CURRENT(), __VA_ARGS__)

#define LOG_PERF(...) \
    ::lfs::core::Logger::get().log_internal(::lfs::core::LogLevel::Performance, LFS_SOURCE_SITE_CURRENT(), __VA_ARGS__)

#define LOG_WARN(...) \
    ::lfs::core::Logger::get().log_internal(::lfs::core::LogLevel::Warn, LFS_SOURCE_SITE_CURRENT(), __VA_ARGS__)

#define LOG_ERROR(...) \
    ::lfs::core::Logger::get().log_internal(::lfs::core::LogLevel::Error, LFS_SOURCE_SITE_CURRENT(), __VA_ARGS__)

#define LOG_CRITICAL(...) \
    ::lfs::core::Logger::get().log_internal(::lfs::core::LogLevel::Critical, LFS_SOURCE_SITE_CURRENT(), __VA_ARGS__)

// Helper macros to force expansion of __COUNTER__ before concatenation
#define _LOG_TIMER_CONCAT_IMPL(x, y)  x##y
#define _LOG_TIMER_MACRO_CONCAT(x, y) _LOG_TIMER_CONCAT_IMPL(x, y)

#define LOG_TIMER(name)                                                     \
    ::lfs::core::ScopedTimer _LOG_TIMER_MACRO_CONCAT(_timer_, __COUNTER__)( \
        (name), ::lfs::core::LogLevel::Performance, LFS_SOURCE_SITE_CURRENT())
#define LOG_TIMER_THRESHOLD(name, min_log_ms)                               \
    ::lfs::core::ScopedTimer _LOG_TIMER_MACRO_CONCAT(_timer_, __COUNTER__)( \
        (name), (min_log_ms), ::lfs::core::LogLevel::Performance,           \
        LFS_SOURCE_SITE_CURRENT())
#define LOG_TIMER_TRACE(name)                                               \
    ::lfs::core::ScopedTimer _LOG_TIMER_MACRO_CONCAT(_timer_, __COUNTER__)( \
        (name), ::lfs::core::LogLevel::Trace, LFS_SOURCE_SITE_CURRENT())
#define LOG_TIMER_DEBUG(name)                                               \
    ::lfs::core::ScopedTimer _LOG_TIMER_MACRO_CONCAT(_timer_, __COUNTER__)( \
        (name), ::lfs::core::LogLevel::Debug, LFS_SOURCE_SITE_CURRENT())

// Memory logging: use LOG_DEBUG("[MEM] ...") and filter with --log-filter "*MEM*"
