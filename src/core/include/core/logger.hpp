/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <source_location>
#include <string>
#include <string_view>

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

    class Logger {
    public:
        static Logger& get();

        void init(LogLevel console_level = LogLevel::Info,
                  const std::string& log_file = "",
                  const std::string& filter_pattern = "");

        // Log a pre-formatted message (called by macros)
        void log(LogLevel level, const std::source_location& loc, std::string_view msg);

        // Module control
        void enable_module(LogModule module, bool enabled = true);
        void set_module_level(LogModule module, LogLevel level);
        void set_level(LogLevel level);
        void flush();

        // Runtime string logging - no format args, works for both CUDA and non-CUDA
        // Use this when you need to log a dynamically constructed string
        void log_internal(LogLevel level, const std::source_location& loc, const std::string& msg) {
            if (static_cast<uint8_t>(level) < global_level_.load(std::memory_order_relaxed))
                return;
            log(level, loc, msg);
        }

        // Template wrapper for formatting (header-only for convenience)
        // PERF: Fast-path check BEFORE expensive std::format() call.
        // If the log level is below the global threshold, skip formatting entirely.
        // This reduces LOG_DEBUG overhead from ~1-5μs to <50ns when debug logging is disabled.
        template <typename... Args>
        void log_internal(LogLevel level, const std::source_location& loc,
#ifdef __CUDACC__
                          const char* fmt, Args&&... args) {
            // Fast path: skip formatting if level is disabled globally
            // (Note: module-specific levels are checked in log(), but this catches most cases)
            if (static_cast<uint8_t>(level) < global_level_.load(std::memory_order_relaxed))
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
            // Fast path: skip formatting if level is disabled globally
            // (Note: module-specific levels are checked in log(), but this catches most cases)
            if (static_cast<uint8_t>(level) < global_level_.load(std::memory_order_relaxed))
                return;

            log(level, loc, std::format(fmt, std::forward<Args>(args)...));
        }
#endif

    private:
        Logger();
        ~Logger();
        Logger(const Logger&) = delete;
        Logger& operator=(const Logger&) = delete;

        struct Impl;
        std::unique_ptr<Impl> impl_;

        std::atomic<uint8_t> global_level_{static_cast<uint8_t>(LogLevel::Info)};
        std::array<std::atomic<bool>, static_cast<size_t>(LogModule::Count)> module_enabled_{};
        std::array<std::atomic<uint8_t>, static_cast<size_t>(LogModule::Count)> module_level_{};
    };

    // Scoped timer for performance measurement
    class ScopedTimer {
    public:
        explicit ScopedTimer(std::string name, LogLevel level = LogLevel::Performance,
                             std::source_location loc = std::source_location::current());
        ~ScopedTimer();

    private:
        std::chrono::high_resolution_clock::time_point start_;
        std::string name_;
        LogLevel level_;
        std::source_location loc_;
    };

} // namespace lfs::core

// Global macros
#define LOG_TRACE(...) \
    ::lfs::core::Logger::get().log_internal(::lfs::core::LogLevel::Trace, std::source_location::current(), __VA_ARGS__)

#define LOG_DEBUG(...) \
    ::lfs::core::Logger::get().log_internal(::lfs::core::LogLevel::Debug, std::source_location::current(), __VA_ARGS__)

#define LOG_INFO(...) \
    ::lfs::core::Logger::get().log_internal(::lfs::core::LogLevel::Info, std::source_location::current(), __VA_ARGS__)

#define LOG_PERF(...) \
    ::lfs::core::Logger::get().log_internal(::lfs::core::LogLevel::Performance, std::source_location::current(), __VA_ARGS__)

#define LOG_WARN(...) \
    ::lfs::core::Logger::get().log_internal(::lfs::core::LogLevel::Warn, std::source_location::current(), __VA_ARGS__)

#define LOG_ERROR(...) \
    ::lfs::core::Logger::get().log_internal(::lfs::core::LogLevel::Error, std::source_location::current(), __VA_ARGS__)

#define LOG_CRITICAL(...) \
    ::lfs::core::Logger::get().log_internal(::lfs::core::LogLevel::Critical, std::source_location::current(), __VA_ARGS__)

// Helper macros to force expansion of __COUNTER__ before concatenation
#define _LOG_TIMER_CONCAT_IMPL(x, y)  x##y
#define _LOG_TIMER_MACRO_CONCAT(x, y) _LOG_TIMER_CONCAT_IMPL(x, y)

#define LOG_TIMER(name)       ::lfs::core::ScopedTimer _LOG_TIMER_MACRO_CONCAT(_timer_, __COUNTER__)(name)
#define LOG_TIMER_TRACE(name) ::lfs::core::ScopedTimer _LOG_TIMER_MACRO_CONCAT(_timer_, __COUNTER__)(name, ::lfs::core::LogLevel::Trace)
#define LOG_TIMER_DEBUG(name) ::lfs::core::ScopedTimer _LOG_TIMER_MACRO_CONCAT(_timer_, __COUNTER__)(name, ::lfs::core::LogLevel::Debug)

// Memory logging: use LOG_DEBUG("[MEM] ...") and filter with --log-filter "*MEM*"
