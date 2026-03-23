/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include <array>
#include <cstdio>
#include <mutex>
#include <optional>
#include <regex>
#ifdef WIN32
#define FMT_UNICODE 0
#endif
#include <spdlog/sinks/base_sink.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

namespace lfs::core {

    namespace {
        constexpr const char* ANSI_RESET = "\033[0m";
        constexpr const char* ANSI_PERF = "\033[95m";

        // Convert glob pattern to regex: * -> .*, ? -> .
        std::string glob_to_regex(const std::string& glob) {
            std::string regex;
            regex.reserve(glob.size() * 2);
            for (const char c : glob) {
                switch (c) {
                case '*': regex += ".*"; break;
                case '?': regex += "."; break;
                case '.':
                case '^':
                case '$':
                case '+':
                case '(':
                case ')':
                case '[':
                case ']':
                case '{':
                case '}':
                case '|':
                case '\\':
                    regex += '\\';
                    regex += c;
                    break;
                default: regex += c; break;
                }
            }
            return regex;
        }

        // Check if pattern contains regex-specific chars (not valid in glob)
        // Note: * and ? are valid glob chars, so we don't check for them
        bool is_regex_pattern(const std::string& pattern) {
            for (size_t i = 0; i < pattern.size(); ++i) {
                const char c = pattern[i];
                if (c == '\\' && i + 1 < pattern.size()) {
                    ++i;
                    continue;
                }
                if (c == '+' || c == '[' || c == ']' || c == '(' || c == ')' ||
                    c == '{' || c == '}' || c == '^' || c == '$' || c == '|') {
                    return true;
                }
            }
            return false;
        }

        class ColorSink final : public spdlog::sinks::base_sink<std::mutex> {
        public:
            explicit ColorSink(const std::string& filter = "", FILE* target = stdout)
                : target_(target) {
                if (!filter.empty()) {
                    try {
                        // If it looks like regex, use as-is; otherwise convert glob to regex
                        const std::string pattern = is_regex_pattern(filter) ? filter : glob_to_regex(filter);
                        filter_regex_.emplace(pattern, std::regex::optimize | std::regex::icase);
                    } catch (const std::regex_error& e) {
                        std::fprintf(stderr, "Invalid log filter pattern '%s': %s\n", filter.c_str(), e.what());
                    }
                }
                colors_[spdlog::level::trace] = "\033[37m";
                colors_[spdlog::level::debug] = "\033[36m";
                colors_[spdlog::level::info] = "\033[32m";
                colors_[spdlog::level::warn] = "\033[33m";
                colors_[spdlog::level::err] = "\033[31m";
                colors_[spdlog::level::critical] = "\033[1;31m";
                colors_[spdlog::level::off] = ANSI_RESET;
            }

        protected:
            void sink_it_(const spdlog::details::log_msg& msg) override {
                const std::string_view msg_view(msg.payload.data(), msg.payload.size());

                // Apply regex filter if set
                if (filter_regex_ && !std::regex_search(msg_view.begin(), msg_view.end(), *filter_regex_)) {
                    return;
                }

                const auto time_t_val = std::chrono::system_clock::to_time_t(msg.time);
                std::tm tm{};
#ifdef WIN32
                localtime_s(&tm, &time_t_val);
#else
                localtime_r(&time_t_val, &tm);
#endif
                const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
                                        msg.time.time_since_epoch())
                                        .count() %
                                    1000;

                std::string_view filename;
                if (msg.source.filename) {
                    const std::string_view full_path(msg.source.filename);
                    const auto pos = full_path.find_last_of("/\\");
                    filename = (pos != std::string_view::npos) ? full_path.substr(pos + 1) : full_path;
                }
                const bool is_perf = msg_view.find("[PERF]") != std::string_view::npos;

                const char* color;
                const char* level_str;

                if (is_perf) {
                    color = ANSI_PERF;
                    level_str = "perf";
                } else {
                    switch (msg.level) {
                    case spdlog::level::trace:
                        color = colors_[0].c_str();
                        level_str = "trace";
                        break;
                    case spdlog::level::debug:
                        color = colors_[1].c_str();
                        level_str = "debug";
                        break;
                    case spdlog::level::info:
                        color = colors_[2].c_str();
                        level_str = "info";
                        break;
                    case spdlog::level::warn:
                        color = colors_[3].c_str();
                        level_str = "warn";
                        break;
                    case spdlog::level::err:
                        color = colors_[4].c_str();
                        level_str = "error";
                        break;
                    case spdlog::level::critical:
                        color = colors_[5].c_str();
                        level_str = "critical";
                        break;
                    default:
                        color = colors_[2].c_str();
                        level_str = "info";
                        break;
                    }
                }

                std::string output_msg(msg_view);
                if (is_perf) {
                    if (const auto pos = output_msg.find("[PERF] "); pos != std::string::npos) {
                        output_msg.erase(pos, 7);
                    }
                }

                std::fprintf(target_, "[%02d:%02d:%02d.%03d] %s[%s]%s %.*s:%d  %s\n",
                             tm.tm_hour, tm.tm_min, tm.tm_sec, static_cast<int>(millis),
                             color, level_str, ANSI_RESET,
                             static_cast<int>(filename.size()), filename.data(), msg.source.line,
                             output_msg.c_str());
                std::fflush(target_);
            }

            void flush_() override { std::fflush(target_); }

        private:
            FILE* target_;
            std::array<std::string, 7> colors_;
            std::optional<std::regex> filter_regex_;
        };

        LogModule detect_module(const std::string_view path) {
            if (path.find("rendering") != std::string_view::npos || path.find("Rendering") != std::string_view::npos)
                return LogModule::Rendering;
            if (path.find("visualizer") != std::string_view::npos || path.find("Visualizer") != std::string_view::npos)
                return LogModule::Visualizer;
            if (path.find("loader") != std::string_view::npos || path.find("Loader") != std::string_view::npos)
                return LogModule::Loader;
            if (path.find("scene") != std::string_view::npos || path.find("Scene") != std::string_view::npos)
                return LogModule::Scene;
            if (path.find("training") != std::string_view::npos || path.find("Training") != std::string_view::npos)
                return LogModule::Training;
            if (path.find("input") != std::string_view::npos || path.find("Input") != std::string_view::npos)
                return LogModule::Input;
            if (path.find("gui") != std::string_view::npos || path.find("GUI") != std::string_view::npos)
                return LogModule::GUI;
            if (path.find("window") != std::string_view::npos || path.find("Window") != std::string_view::npos)
                return LogModule::Window;
            if (path.find("memory") != std::string_view::npos || path.find("Memory") != std::string_view::npos)
                return LogModule::Memory;
            if (path.find("core") != std::string_view::npos || path.find("Core") != std::string_view::npos)
                return LogModule::Core;
            return LogModule::Unknown;
        }

        constexpr spdlog::level::level_enum to_spdlog_level(const LogLevel level) {
            switch (level) {
            case LogLevel::Trace: return spdlog::level::trace;
            case LogLevel::Debug: return spdlog::level::debug;
            case LogLevel::Info: return spdlog::level::info;
            case LogLevel::Performance: return spdlog::level::info;
            case LogLevel::Warn: return spdlog::level::warn;
            case LogLevel::Error: return spdlog::level::err;
            case LogLevel::Critical: return spdlog::level::critical;
            case LogLevel::Off: return spdlog::level::off;
            default: return spdlog::level::info;
            }
        }
    } // anonymous namespace

    struct Logger::Impl {
        std::shared_ptr<spdlog::logger> logger;
        std::mutex mutex;
    };

    Logger::Logger() : impl_(std::make_unique<Impl>()) {
        for (size_t i = 0; i < static_cast<size_t>(LogModule::Count); ++i) {
            module_enabled_[i] = true;
            module_level_[i] = static_cast<uint8_t>(LogLevel::Trace);
        }
    }

    Logger::~Logger() = default;

    Logger& Logger::get() {
        static Logger instance;
        return instance;
    }

    void Logger::init(const LogLevel console_level, const std::string& log_file,
                      const std::string& filter_pattern, const bool use_stderr) {
        std::lock_guard lock(impl_->mutex);

        std::vector<spdlog::sink_ptr> sinks;

        auto console_sink = std::make_shared<ColorSink>(filter_pattern, use_stderr ? stderr : stdout);
        console_sink->set_level(to_spdlog_level(console_level));
        sinks.push_back(console_sink);

        if (!log_file.empty()) {
            auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file, true);
            file_sink->set_level(spdlog::level::trace);
            file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] %s:%# %v");
            sinks.push_back(file_sink);
        }

        impl_->logger = std::make_shared<spdlog::logger>("lfs", sinks.begin(), sinks.end());
        impl_->logger->set_level(spdlog::level::trace);
        spdlog::set_default_logger(impl_->logger);

        global_level_ = static_cast<uint8_t>(console_level);
    }

    void Logger::log(const LogLevel level, const std::source_location& loc, const std::string_view msg) {
        if (!impl_->logger)
            return;

        const auto module = detect_module(loc.file_name());
        const auto module_idx = static_cast<size_t>(module);

        if (!module_enabled_[module_idx] || static_cast<uint8_t>(level) < module_level_[module_idx]) {
            return;
        }

        const auto global_lvl = static_cast<LogLevel>(global_level_.load());
        if (global_lvl == LogLevel::Performance) {
            if (level != LogLevel::Performance)
                return;
        } else {
            if (level == LogLevel::Performance)
                return;
            if (static_cast<uint8_t>(level) < global_level_)
                return;
        }

        std::string final_msg(msg);
        if (level == LogLevel::Performance) {
            final_msg = "[PERF] " + final_msg;
        }

        impl_->logger->log(
            spdlog::source_loc{loc.file_name(), static_cast<int>(loc.line()), loc.function_name()},
            to_spdlog_level(level),
            final_msg);
    }

    void Logger::enable_module(const LogModule module, const bool enabled) {
        module_enabled_[static_cast<size_t>(module)] = enabled;
    }

    void Logger::set_module_level(const LogModule module, const LogLevel level) {
        module_level_[static_cast<size_t>(module)] = static_cast<uint8_t>(level);
    }

    void Logger::set_level(const LogLevel level) {
        if (impl_->logger) {
            impl_->logger->set_level(to_spdlog_level(level));
        }
        global_level_ = static_cast<uint8_t>(level);
    }

    void Logger::flush() {
        if (impl_->logger)
            impl_->logger->flush();
    }

    ScopedTimer::ScopedTimer(std::string name, const LogLevel level, const std::source_location loc)
        : start_(std::chrono::high_resolution_clock::now()),
          name_(std::move(name)),
          level_(level),
          loc_(loc) {}

    ScopedTimer::~ScopedTimer() {
        const auto duration = std::chrono::high_resolution_clock::now() - start_;
        const auto ms = std::chrono::duration<double, std::milli>(duration).count();
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%s took %.2fms", name_.c_str(), ms);
        Logger::get().log(level_, loc_, buf);
    }

} // namespace lfs::core
