/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include <array>
#include <atomic>
#include <chrono>
#include <format>
#include <mutex>
#include <source_location>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <string_view>

namespace gs::core {

    enum class LogLevel : uint8_t {
        Trace = 0,
        Debug = 1,
        Info = 2,
        Warn = 3,
        Error = 4,
        Critical = 5,
        Off = 6
    };

    // Module detection from file path
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
        Unknown = 9,
        Count = 10 // Total number of modules
    };

    class Logger {
    public:
        static Logger& get() {
            static Logger instance;
            return instance;
        }

        // Initialize logger
        void init(LogLevel console_level = LogLevel::Info,
                  const std::string& log_file = "") {
            std::lock_guard lock(mutex_);

            std::vector<spdlog::sink_ptr> sinks;

            // Console sink with color
            auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            console_sink->set_level(to_spdlog_level(console_level));
            console_sink->set_pattern("[%H:%M:%S.%e] [%^%l%$] %s:%# %v");
            sinks.push_back(console_sink);

            // Optional file sink
            if (!log_file.empty()) {
                auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file, true);
                file_sink->set_level(spdlog::level::trace);
                file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] %s:%# %v");
                sinks.push_back(file_sink);
            }

            logger_ = std::make_shared<spdlog::logger>("gs", sinks.begin(), sinks.end());
            logger_->set_level(spdlog::level::trace);
            spdlog::set_default_logger(logger_);

            // IMPORTANT: Set the global level to match console level
            global_level_ = static_cast<uint8_t>(console_level);

            // Enable all modules by default at Trace level
            for (size_t i = 0; i < static_cast<size_t>(LogModule::Count); ++i) {
                module_enabled_[i] = true;
                module_level_[i] = static_cast<uint8_t>(LogLevel::Trace);
            }
        }

        // Internal log implementation
        template <typename... Args>
        void log_internal(LogLevel level, const std::source_location& loc,
                          const char* fmt, Args&&... args) {
            if (!logger_)
                return;

            // Detect module from file path
            auto module = detect_module(loc.file_name());

            // Check if module is enabled and level is sufficient
            auto module_idx = static_cast<size_t>(module);
            if (!module_enabled_[module_idx] ||
                static_cast<uint8_t>(level) < module_level_[module_idx] ||
                static_cast<uint8_t>(level) < global_level_) {
                return;
            }

            // Format message using snprintf
            char buffer[1024];
            int written = std::snprintf(buffer, sizeof(buffer), fmt, std::forward<Args>(args)...);
            if (written < 0) {
                // formatting error
                return;
            }
            std::string msg;
            if (static_cast<size_t>(written) >= sizeof(buffer)) {
                // message was truncated — reallocate
                size_t size = written + 1;
                msg.resize(size);
                std::snprintf(msg.data(), size, fmt, std::forward<Args>(args)...);
            } else {
                msg.assign(buffer, written);
            }

            logger_->log(
                spdlog::source_loc{loc.file_name(),
                                   static_cast<int>(loc.line()),
                                   loc.function_name()},
                to_spdlog_level(level),
                msg);
        }

        // Module control
        void enable_module(LogModule module, bool enabled = true) {
            module_enabled_[static_cast<size_t>(module)] = enabled;
        }

        void set_module_level(LogModule module, LogLevel level) {
            module_level_[static_cast<size_t>(module)] = static_cast<uint8_t>(level);
        }

        // Global level control
        void set_level(LogLevel level) {
            if (logger_) {
                logger_->set_level(to_spdlog_level(level));
            }
            global_level_ = static_cast<uint8_t>(level);
        }

        // Flush logs
        void flush() {
            if (logger_)
                logger_->flush();
        }

    private:
        Logger() = default;

        static LogModule detect_module(std::string_view path) {
            // Convert to lowercase for case-insensitive matching
            if (path.find("rendering") != std::string_view::npos ||
                path.find("Rendering") != std::string_view::npos)
                return LogModule::Rendering;
            if (path.find("visualizer") != std::string_view::npos ||
                path.find("Visualizer") != std::string_view::npos)
                return LogModule::Visualizer;
            if (path.find("loader") != std::string_view::npos ||
                path.find("Loader") != std::string_view::npos)
                return LogModule::Loader;
            if (path.find("scene") != std::string_view::npos ||
                path.find("Scene") != std::string_view::npos)
                return LogModule::Scene;
            if (path.find("training") != std::string_view::npos ||
                path.find("Training") != std::string_view::npos)
                return LogModule::Training;
            if (path.find("input") != std::string_view::npos ||
                path.find("Input") != std::string_view::npos)
                return LogModule::Input;
            if (path.find("gui") != std::string_view::npos ||
                path.find("GUI") != std::string_view::npos)
                return LogModule::GUI;
            if (path.find("window") != std::string_view::npos ||
                path.find("Window") != std::string_view::npos)
                return LogModule::Window;
            if (path.find("core") != std::string_view::npos ||
                path.find("Core") != std::string_view::npos)
                return LogModule::Core;
            return LogModule::Unknown;
        }

        static constexpr spdlog::level::level_enum to_spdlog_level(LogLevel level) {
            switch (level) {
            case LogLevel::Trace: return spdlog::level::trace;
            case LogLevel::Debug: return spdlog::level::debug;
            case LogLevel::Info: return spdlog::level::info;
            case LogLevel::Warn: return spdlog::level::warn;
            case LogLevel::Error: return spdlog::level::err;
            case LogLevel::Critical: return spdlog::level::critical;
            case LogLevel::Off: return spdlog::level::off;
            default: return spdlog::level::info;
            }
        }

        std::shared_ptr<spdlog::logger> logger_;
        mutable std::mutex mutex_;
        std::atomic<uint8_t> global_level_{static_cast<uint8_t>(LogLevel::Info)};
        std::array<std::atomic<bool>, static_cast<size_t>(LogModule::Count)> module_enabled_{};
        std::array<std::atomic<uint8_t>, static_cast<size_t>(LogModule::Count)> module_level_{};
    };

    // Scoped timer for performance measurement
    class ScopedTimer {
        std::chrono::high_resolution_clock::time_point start_;
        std::string name_;
        LogLevel level_;
        std::source_location loc_;

    public:
        explicit ScopedTimer(std::string name, LogLevel level = LogLevel::Debug,
                             std::source_location loc = std::source_location::current())
            : start_(std::chrono::high_resolution_clock::now()),
              name_(std::move(name)),
              level_(level),
              loc_(loc) {}

        ~ScopedTimer() {
            auto duration = std::chrono::high_resolution_clock::now() - start_;
            auto ms = std::chrono::duration<double, std::milli>(duration).count();

            switch (level_) {
            case LogLevel::Trace:
                Logger::get().log_internal(LogLevel::Trace, loc_, "{} took {:.2f}ms", name_, ms);
                break;
            case LogLevel::Debug:
                Logger::get().log_internal(LogLevel::Debug, loc_, "{} took {:.2f}ms", name_, ms);
                break;
            default:
                Logger::get().log_internal(LogLevel::Info, loc_, "{} took {:.2f}ms", name_, ms);
                break;
            }
        }
    };

} // namespace gs::core

// Global macros defined OUTSIDE namespace - accessible from anywhere
#define LOG_TRACE(...) \
    ::gs::core::Logger::get().log_internal(::gs::core::LogLevel::Trace, std::source_location::current(), __VA_ARGS__)

#define LOG_DEBUG(...) \
    ::gs::core::Logger::get().log_internal(::gs::core::LogLevel::Debug, std::source_location::current(), __VA_ARGS__)

#define LOG_INFO(...) \
    ::gs::core::Logger::get().log_internal(::gs::core::LogLevel::Info, std::source_location::current(), __VA_ARGS__)

#define LOG_WARN(...) \
    ::gs::core::Logger::get().log_internal(::gs::core::LogLevel::Warn, std::source_location::current(), __VA_ARGS__)

#define LOG_ERROR(...) \
    ::gs::core::Logger::get().log_internal(::gs::core::LogLevel::Error, std::source_location::current(), __VA_ARGS__)

#define LOG_CRITICAL(...) \
    ::gs::core::Logger::get().log_internal(::gs::core::LogLevel::Critical, std::source_location::current(), __VA_ARGS__)

// Timer macros
#define LOG_TIMER(name)       ::gs::core::ScopedTimer _timer##__LINE__(name)
#define LOG_TIMER_TRACE(name) ::gs::core::ScopedTimer _timer##__LINE__(name, ::gs::core::LogLevel::Trace)
