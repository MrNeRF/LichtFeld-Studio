/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include <array>
#include <atomic>
#include <chrono>
#include <format>
#include <iostream>
#include <mutex>
#ifdef WIN32
#define FMT_UNICODE 0
#endif
#include <source_location>
#include <spdlog/sinks/base_sink.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <string_view>

namespace lfs::core {

    // Custom sink that renders Performance logs with custom color and label
    template <typename Mutex>
    class performance_color_sink : public spdlog::sinks::base_sink<Mutex> {
    public:
        performance_color_sink() {
            // ANSI color codes
            colors_[spdlog::level::trace] = "\033[37m";      // white
            colors_[spdlog::level::debug] = "\033[36m";      // cyan
            colors_[spdlog::level::info] = "\033[32m";       // green
            colors_[spdlog::level::warn] = "\033[33m";       // yellow
            colors_[spdlog::level::err] = "\033[31m";        // red
            colors_[spdlog::level::critical] = "\033[1;31m"; // bold red
            colors_[spdlog::level::off] = "\033[0m";         // reset

            // Custom color for performance logs: bright magenta/purple
            perf_color_ = "\033[95m"; // bright magenta
            reset_color_ = "\033[0m";
        }

        void set_level(spdlog::level::level_enum level) {
            level_ = level;
        }

    protected:
        void sink_it_(const spdlog::details::log_msg& msg) override {
            // Extract time components from log_msg
            auto time_point = msg.time;
            auto time_t_val = std::chrono::system_clock::to_time_t(time_point);
            auto tm = *std::localtime(&time_t_val);
            auto duration = time_point.time_since_epoch();
            auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() % 1000;

            // Extract just the filename from the full path
            std::string_view full_path(msg.source.filename);
            auto last_slash = full_path.find_last_of("/\\");
            std::string_view filename = (last_slash != std::string_view::npos)
                                            ? full_path.substr(last_slash + 1)
                                            : full_path;

            // Check if this is a Performance log (has [PERF] prefix)
            std::string_view msg_view(msg.payload.data(), msg.payload.size());
            bool is_perf = msg_view.find("[PERF]") != std::string_view::npos;

            if (is_perf) {
                // Custom formatting for performance logs
                // Remove [PERF] prefix from message
                std::string clean_msg(msg_view);
                size_t perf_pos = clean_msg.find("[PERF] ");
                if (perf_pos != std::string::npos) {
                    clean_msg.erase(perf_pos, 7); // Remove "[PERF] "
                }

                // Format: [timestamp] [perf] file:line message
                char time_buf[64];
                std::snprintf(time_buf, sizeof(time_buf),
                              "[%02d:%02d:%02d.%03d] %s[perf]%s %.*s:%d  ",
                              tm.tm_hour,
                              tm.tm_min,
                              tm.tm_sec,
                              static_cast<int>(millis),
                              perf_color_.c_str(),
                              reset_color_.c_str(),
                              static_cast<int>(filename.size()),
                              filename.data(),
                              msg.source.line);

                std::cout << time_buf << clean_msg << "\n"
                          << std::flush;
            } else {
                // Standard formatting for non-performance logs
                std::string level_str;
                std::string color;

                switch (msg.level) {
                case spdlog::level::trace:
                    level_str = "trace";
                    color = colors_[spdlog::level::trace];
                    break;
                case spdlog::level::debug:
                    level_str = "debug";
                    color = colors_[spdlog::level::debug];
                    break;
                case spdlog::level::info:
                    level_str = "info";
                    color = colors_[spdlog::level::info];
                    break;
                case spdlog::level::warn:
                    level_str = "warn";
                    color = colors_[spdlog::level::warn];
                    break;
                case spdlog::level::err:
                    level_str = "error";
                    color = colors_[spdlog::level::err];
                    break;
                case spdlog::level::critical:
                    level_str = "critical";
                    color = colors_[spdlog::level::critical];
                    break;
                default:
                    level_str = "info";
                    color = colors_[spdlog::level::info];
                    break;
                }

                // Use snprintf for nvcc compatibility
                char time_buf[64];
                std::snprintf(time_buf, sizeof(time_buf),
                              "[%02d:%02d:%02d.%03d] %s[%s]%s %.*s:%d  ",
                              tm.tm_hour,
                              tm.tm_min,
                              tm.tm_sec,
                              static_cast<int>(millis),
                              color.c_str(),
                              level_str.c_str(),
                              reset_color_.c_str(),
                              static_cast<int>(filename.size()),
                              filename.data(),
                              msg.source.line);

                std::cout << time_buf
                          << std::string_view(msg.payload.data(), msg.payload.size())
                          << "\n"
                          << std::flush;
            }
        }

        void flush_() override {
            std::cout << std::flush;
        }

    private:
        spdlog::level::level_enum level_ = spdlog::level::info;
        std::array<std::string, 7> colors_;
        std::string perf_color_;
        std::string reset_color_;
    };

    using performance_color_sink_mt = performance_color_sink<std::mutex>;
    using performance_color_sink_st = performance_color_sink<spdlog::details::null_mutex>;

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

            // Custom console sink with performance log color support
            auto console_sink = std::make_shared<performance_color_sink_mt>();
            console_sink->set_level(to_spdlog_level(console_level));
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
#ifdef __CUDACC__
                          const char* fmt, Args&&... args) {
#else
                          std::format_string<Args...> fmt, Args&&... args) {
#endif
            if (!logger_)
                return;

            // Detect module from file path
            auto module = detect_module(loc.file_name());

            // Check if module is enabled and level is sufficient
            auto module_idx = static_cast<size_t>(module);
            if (!module_enabled_[module_idx] ||
                static_cast<uint8_t>(level) < module_level_[module_idx]) {
                return;
            }

            // Performance level shows ONLY Performance logs
            // When global_level is Performance, filter out everything except Performance
            auto global_lvl = static_cast<LogLevel>(global_level_.load());
            if (global_lvl == LogLevel::Performance) {
                if (level != LogLevel::Performance) {
                    return; // Skip non-performance logs
                }
            } else {
                // Standard filtering: show logs >= global_level (but not Performance unless requested)
                if (level == LogLevel::Performance) {
                    return; // Performance logs only show when explicitly set
                }
                if (static_cast<uint8_t>(level) < global_level_) {
                    return;
                }
            }

#ifdef __CUDACC__
            // Format message using snprintf (for CUDA/nvcc compatibility)
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
#else
            // Format message using std::format (C++20)
            auto msg = std::format(fmt, std::forward<Args>(args)...);
#endif

            // Add performance prefix for better visual distinction
            if (level == LogLevel::Performance) {
                msg = "[PERF] " + msg;
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
            case LogLevel::Performance: return spdlog::level::info; // Map to info but will have custom label
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
        explicit ScopedTimer(std::string name, LogLevel level = LogLevel::Performance,
                             std::source_location loc = std::source_location::current())
            : start_(std::chrono::high_resolution_clock::now()),
              name_(std::move(name)),
              level_(level),
              loc_(loc) {}

        ~ScopedTimer() {
            auto duration = std::chrono::high_resolution_clock::now() - start_;
            auto ms = std::chrono::duration<double, std::milli>(duration).count();

            Logger::get().log_internal(level_, loc_, "{} took {:.2f}ms", name_, ms);
        }
    };

} // namespace lfs::core

// Global macros defined OUTSIDE namespace - accessible from anywhere
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

// Timer macros - LOG_TIMER now uses Performance level by default
#define LOG_TIMER(name)       ::lfs::core::ScopedTimer _timer##__LINE__(name)
#define LOG_TIMER_TRACE(name) ::lfs::core::ScopedTimer _timer##__LINE__(name, ::lfs::core::LogLevel::Trace)
#define LOG_TIMER_DEBUG(name) ::lfs::core::ScopedTimer _timer##__LINE__(name, ::lfs::core::LogLevel::Debug)
