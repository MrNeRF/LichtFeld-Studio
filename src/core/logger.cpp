/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include <array>
#include <cstdio>
#include <deque>
#include <format>
#include <mutex>
#include <optional>
#include <regex>
#include <vector>
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
        constexpr size_t MAX_BUFFERED_LOG_ENTRIES = 5000;

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

        [[nodiscard]] std::optional<std::regex> compile_filter_regex(const std::string& filter) {
            if (filter.empty())
                return std::nullopt;

            try {
                const std::string pattern = is_regex_pattern(filter) ? filter : glob_to_regex(filter);
                return std::regex(pattern, std::regex::optimize | std::regex::icase);
            } catch (const std::regex_error& e) {
                std::fprintf(stderr, "Invalid log filter pattern '%s': %s\n", filter.c_str(), e.what());
                return std::nullopt;
            }
        }

        [[nodiscard]] std::string_view base_filename(const char* filename) {
            if (!filename)
                return {};

            const std::string_view full_path(filename);
            const auto pos = full_path.find_last_of("/\\");
            return (pos != std::string_view::npos) ? full_path.substr(pos + 1) : full_path;
        }

        [[nodiscard]] std::string strip_perf_prefix(std::string_view msg_view) {
            std::string output_msg(msg_view);
            if (const auto pos = output_msg.find("[PERF] "); pos != std::string::npos)
                output_msg.erase(pos, 7);
            return output_msg;
        }

        [[nodiscard]] LogLevel from_spdlog_level(const spdlog::level::level_enum level,
                                                 const bool is_perf) {
            if (is_perf)
                return LogLevel::Performance;

            switch (level) {
            case spdlog::level::trace: return LogLevel::Trace;
            case spdlog::level::debug: return LogLevel::Debug;
            case spdlog::level::info: return LogLevel::Info;
            case spdlog::level::warn: return LogLevel::Warn;
            case spdlog::level::err: return LogLevel::Error;
            case spdlog::level::critical: return LogLevel::Critical;
            case spdlog::level::off: return LogLevel::Off;
            default: return LogLevel::Info;
            }
        }

        [[nodiscard]] std::string_view log_level_tag(const LogLevel level) {
            switch (level) {
            case LogLevel::Trace: return "trace";
            case LogLevel::Debug: return "debug";
            case LogLevel::Info: return "info";
            case LogLevel::Performance: return "perf";
            case LogLevel::Warn: return "warn";
            case LogLevel::Error: return "error";
            case LogLevel::Critical: return "critical";
            case LogLevel::Off: return "off";
            default: return "info";
            }
        }

        [[nodiscard]] std::string format_timestamp(const std::chrono::system_clock::time_point& timestamp) {
            const auto time_t_val = std::chrono::system_clock::to_time_t(timestamp);
            std::tm tm{};
#ifdef WIN32
            localtime_s(&tm, &time_t_val);
#else
            localtime_r(&time_t_val, &tm);
#endif
            const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    timestamp.time_since_epoch())
                                    .count() %
                                1000;
            return std::format("{:02}:{:02}:{:02}.{:03}",
                               tm.tm_hour,
                               tm.tm_min,
                               tm.tm_sec,
                               static_cast<int>(millis));
        }

        [[nodiscard]] std::string format_log_line(const LogEntrySnapshot& entry) {
            const std::string_view file = entry.file.empty() ? std::string_view("unknown") : std::string_view(entry.file);
            return std::format("[{}] [{}] {}:{}  {}",
                               format_timestamp(entry.timestamp),
                               log_level_tag(entry.level),
                               file,
                               entry.line,
                               entry.message);
        }

        [[nodiscard]] constexpr bool passes_display_filter(const LogLevel message_level,
                                                           const LogLevel display_level) {
            if (message_level == LogLevel::Off || display_level == LogLevel::Off)
                return false;
            if (message_level == LogLevel::Performance)
                return display_level == LogLevel::Performance ||
                       display_level == LogLevel::Trace ||
                       display_level == LogLevel::Debug;
            if (display_level == LogLevel::Performance)
                return static_cast<uint8_t>(message_level) >= static_cast<uint8_t>(LogLevel::Warn);
            return static_cast<uint8_t>(message_level) >= static_cast<uint8_t>(display_level);
        }

        class ColorSink final : public spdlog::sinks::base_sink<std::mutex> {
        public:
            explicit ColorSink(const std::string& filter = "",
                               FILE* target = stdout,
                               const LogLevel display_level = LogLevel::Info)
                : target_(target),
                  display_level_(static_cast<uint8_t>(display_level)),
                  filter_regex_(compile_filter_regex(filter)) {
                colors_[spdlog::level::trace] = "\033[37m";
                colors_[spdlog::level::debug] = "\033[36m";
                colors_[spdlog::level::info] = "\033[32m";
                colors_[spdlog::level::warn] = "\033[33m";
                colors_[spdlog::level::err] = "\033[31m";
                colors_[spdlog::level::critical] = "\033[1;31m";
                colors_[spdlog::level::off] = ANSI_RESET;
            }

            void set_display_level(const LogLevel level) {
                display_level_.store(static_cast<uint8_t>(level), std::memory_order_relaxed);
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

                const std::string_view filename = base_filename(msg.source.filename);
                const bool is_perf = msg_view.find("[PERF]") != std::string_view::npos;
                const LogLevel entry_level = from_spdlog_level(msg.level, is_perf);
                const LogLevel display_level =
                    static_cast<LogLevel>(display_level_.load(std::memory_order_relaxed));
                if (!passes_display_filter(entry_level, display_level))
                    return;

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

                std::string output_msg = is_perf ? strip_perf_prefix(msg_view) : std::string(msg_view);

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
            std::atomic<uint8_t> display_level_{static_cast<uint8_t>(LogLevel::Info)};
            std::optional<std::regex> filter_regex_;
        };

        class MemorySink final : public spdlog::sinks::base_sink<std::mutex> {
        public:
            explicit MemorySink(const std::string& filter = "",
                                const size_t max_entries = MAX_BUFFERED_LOG_ENTRIES,
                                const LogLevel display_level = LogLevel::Info)
                : max_entries_(max_entries),
                  display_level_(static_cast<uint8_t>(display_level)),
                  filter_regex_(compile_filter_regex(filter)) {}

            [[nodiscard]] uint64_t generation() const {
                return generation_.load(std::memory_order_relaxed);
            }

            [[nodiscard]] size_t entry_count() const {
                std::lock_guard lock(entries_mutex_);
                return entries_.size();
            }

            void set_display_level(const LogLevel level) {
                display_level_.store(static_cast<uint8_t>(level), std::memory_order_relaxed);
            }

            [[nodiscard]] std::vector<LogEntrySnapshot> entries() const {
                std::lock_guard lock(entries_mutex_);
                return {entries_.begin(), entries_.end()};
            }

            [[nodiscard]] std::string text() const {
                std::lock_guard lock(entries_mutex_);
                std::string output;
                output.reserve(entries_.size() * 96);
                for (const auto& entry : entries_) {
                    output += format_log_line(entry);
                    output.push_back('\n');
                }
                return output;
            }

        protected:
            void sink_it_(const spdlog::details::log_msg& msg) override {
                const std::string_view msg_view(msg.payload.data(), msg.payload.size());

                if (filter_regex_ && !std::regex_search(msg_view.begin(), msg_view.end(), *filter_regex_))
                    return;

                const bool is_perf = msg_view.find("[PERF]") != std::string_view::npos;
                const LogLevel entry_level = from_spdlog_level(msg.level, is_perf);
                const LogLevel display_level =
                    static_cast<LogLevel>(display_level_.load(std::memory_order_relaxed));
                if (!passes_display_filter(entry_level, display_level))
                    return;

                LogEntrySnapshot entry;
                entry.timestamp = std::chrono::system_clock::time_point(
                    std::chrono::duration_cast<std::chrono::system_clock::duration>(
                        msg.time.time_since_epoch()));
                entry.level = entry_level;
                entry.file = std::string(base_filename(msg.source.filename));
                entry.line = msg.source.line;
                entry.message = is_perf ? strip_perf_prefix(msg_view) : std::string(msg_view);

                std::lock_guard lock(entries_mutex_);
                if (entries_.size() >= max_entries_)
                    entries_.pop_front();
                entries_.push_back(std::move(entry));
                generation_.fetch_add(1, std::memory_order_relaxed);
            }

            void flush_() override {}

        private:
            size_t max_entries_ = MAX_BUFFERED_LOG_ENTRIES;
            mutable std::mutex entries_mutex_;
            std::deque<LogEntrySnapshot> entries_;
            std::atomic<uint8_t> display_level_{static_cast<uint8_t>(LogLevel::Info)};
            std::atomic<uint64_t> generation_{0};
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
        std::shared_ptr<ColorSink> console_sink;
        std::shared_ptr<MemorySink> memory_sink;
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

        impl_->console_sink = std::make_shared<ColorSink>(filter_pattern,
                                                          use_stderr ? stderr : stdout,
                                                          console_level);
        impl_->console_sink->set_level(spdlog::level::trace);
        sinks.push_back(impl_->console_sink);

        impl_->memory_sink = std::make_shared<MemorySink>(filter_pattern,
                                                          MAX_BUFFERED_LOG_ENTRIES,
                                                          console_level);
        impl_->memory_sink->set_level(spdlog::level::trace);
        sinks.push_back(impl_->memory_sink);

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
        capture_all_to_file_ = !log_file.empty();
    }

    void Logger::log(const LogLevel level, const std::source_location& loc, const std::string_view msg) {
        if (!impl_->logger)
            return;

        const auto module = detect_module(loc.file_name());
        const auto module_idx = static_cast<size_t>(module);

        if (!module_enabled_[module_idx] || static_cast<uint8_t>(level) < module_level_[module_idx]) {
            return;
        }

        if (!capture_all_to_file_.load(std::memory_order_relaxed)) {
            const auto global_lvl = static_cast<LogLevel>(global_level_.load(std::memory_order_relaxed));
            if (!passes_display_filter(level, global_lvl))
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
        std::lock_guard lock(impl_->mutex);
        if (impl_->logger) {
            impl_->logger->set_level(spdlog::level::trace);
            if (impl_->console_sink)
                impl_->console_sink->set_display_level(level);
            if (impl_->memory_sink)
                impl_->memory_sink->set_display_level(level);
        }
        global_level_ = static_cast<uint8_t>(level);
    }

    void Logger::flush() {
        if (impl_->logger)
            impl_->logger->flush();
    }

    LogLevel Logger::level() const {
        return static_cast<LogLevel>(global_level_.load(std::memory_order_relaxed));
    }

    size_t Logger::buffered_log_count() const {
        std::lock_guard lock(impl_->mutex);
        return impl_->memory_sink ? impl_->memory_sink->entry_count() : 0;
    }

    uint64_t Logger::buffered_log_generation() const {
        std::lock_guard lock(impl_->mutex);
        return impl_->memory_sink ? impl_->memory_sink->generation() : 0;
    }

    std::vector<LogEntrySnapshot> Logger::buffered_logs() const {
        std::lock_guard lock(impl_->mutex);
        return impl_->memory_sink ? impl_->memory_sink->entries()
                                  : std::vector<LogEntrySnapshot>{};
    }

    std::string Logger::buffered_logs_as_text() const {
        std::lock_guard lock(impl_->mutex);
        return impl_->memory_sink ? impl_->memory_sink->text() : std::string{};
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
