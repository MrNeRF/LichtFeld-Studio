/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "diagnostics/vram_profiler.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <format>
#include <mutex>
#include <optional>
#include <regex>
#include <vector>
#ifdef WIN32
#define FMT_UNICODE 0
#endif
#include <spdlog/sinks/base_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/spdlog.h>

namespace lfs::core {

    namespace {
        namespace fs = std::filesystem;

        constexpr const char* ANSI_RESET = "\033[0m";
        constexpr const char* ANSI_PERF = "\033[95m";
        constexpr size_t MAX_BUFFERED_LOG_ENTRIES = 5000;
        constexpr size_t DEFAULT_LOG_ROTATION_MAX_BYTES = 10 * 1024 * 1024;
        constexpr size_t DEFAULT_LOG_ROTATION_MAX_FILES = 4;
        constexpr const char* DEFAULT_LOG_FILE_PATTERN = "[%Y-%m-%d %H:%M:%S.%e] [%l] %s:%# %v";

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
                if (!is_perf) {
                    std::fflush(target_);
                }
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

        // Mirrors preprocessing/preprocess.cpp's private home_directory() helper.
        // Not shared via core/path_utils.hpp: lfs_logger is a leaf library with
        // deliberately zero upstream includes (see src/core/logger/CMakeLists.txt),
        // and path_utils.hpp sits outside its include root.
        fs::path lichtfeld_home_directory() {
#ifdef _WIN32
            if (const char* profile = std::getenv("USERPROFILE"); profile && profile[0])
                return fs::path(profile);
            const char* drive = std::getenv("HOMEDRIVE");
            const char* homepath = std::getenv("HOMEPATH");
            if (drive && drive[0] && homepath && homepath[0])
                return fs::path(std::string(drive) + homepath);
#else
            if (const char* home = std::getenv("HOME"); home && home[0])
                return fs::path(home);
#endif
            return fs::temp_directory_path();
        }

        fs::path resolve_default_log_directory(const std::string& user_dir_override) {
            return user_dir_override.empty()
                       ? lichtfeld_home_directory() / ".lichtfeld"
                       : fs::path(user_dir_override);
        }

        fs::path resolve_default_log_path(const std::string& user_dir_override) {
            return resolve_default_log_directory(user_dir_override) / "logs" / "lichtfeld.log";
        }

        std::shared_ptr<spdlog::sinks::rotating_file_sink_mt> make_rotating_file_sink(const fs::path& path) {
            auto sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                path.string(), DEFAULT_LOG_ROTATION_MAX_BYTES, DEFAULT_LOG_ROTATION_MAX_FILES);
            sink->set_level(spdlog::level::trace);
            sink->set_pattern(DEFAULT_LOG_FILE_PATTERN);
            return sink;
        }

        // Never throws: startup must not fail because the durable log couldn't be
        // set up. On failure, prints one warning and returns nullptr so init() falls
        // back to console/memory sinks only.
        std::shared_ptr<spdlog::sinks::rotating_file_sink_mt> make_default_file_sink(const fs::path& path) {
            std::error_code ec;
            fs::create_directories(path.parent_path(), ec);
            if (ec) {
                std::fprintf(stderr, "lichtfeld: could not create log directory '%s': %s\n",
                             path.parent_path().string().c_str(), ec.message().c_str());
                return nullptr;
            }
            try {
                return make_rotating_file_sink(path);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "lichtfeld: could not open default log file '%s': %s\n",
                             path.string().c_str(), e.what());
                return nullptr;
            }
        }

        bool same_log_target(const fs::path& a, const fs::path& b) {
            std::error_code ec_a, ec_b;
            const fs::path resolved_a = fs::weakly_canonical(a, ec_a);
            const fs::path resolved_b = fs::weakly_canonical(b, ec_b);
            if (!ec_a && !ec_b)
                return resolved_a == resolved_b;
            return a.lexically_normal() == b.lexically_normal();
        }
    } // anonymous namespace

    struct Logger::Impl {
        std::vector<std::pair<LogHandlerToken, LogHandler>> log_handlers_{};
        LogHandlerToken next_handler_token_{0};
        std::shared_ptr<spdlog::logger> logger;
        std::shared_ptr<ColorSink> console_sink;
        std::shared_ptr<MemorySink> memory_sink;
        std::mutex mutex;
        std::mutex handler_mutex;
    };

    std::string_view Logger::to_string(LogLevel level) {
        switch (level) {
        case LogLevel::Trace: return "trace";
        case LogLevel::Debug: return "debug";
        case LogLevel::Info: return "info";
        case LogLevel::Performance: return "perf";
        case LogLevel::Warn: return "warn";
        case LogLevel::Error: return "error";
        case LogLevel::Critical: return "critical";
        case LogLevel::Off: return "off";
        default: std::unreachable();
        }
    }

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

    void Logger::init() {
        init(LogLevel::Info, std::string{}, std::string{}, false);
    }

    void Logger::init(const LogLevel console_level) {
        init(console_level, std::string{}, std::string{}, false);
    }

    void Logger::init(const LogLevel console_level, const std::string& log_file) {
        init(console_level, log_file, std::string{}, false);
    }

    void Logger::init(const LogLevel console_level,
                      const std::string& log_file,
                      const std::string& filter_pattern) {
        init(console_level, log_file, filter_pattern, false);
    }

    void Logger::init(const LogLevel console_level, const std::string& log_file,
                      const std::string& filter_pattern, const bool use_stderr) {
        init(console_level, log_file, filter_pattern, use_stderr, std::string{});
    }

    void Logger::init(const LogLevel console_level, const std::string& log_file,
                      const std::string& filter_pattern, const bool use_stderr,
                      const std::string& default_log_dir_override) {
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

        const fs::path default_log_path = resolve_default_log_path(default_log_dir_override);
        bool default_sink_added = false;
        if (auto default_sink = make_default_file_sink(default_log_path)) {
            sinks.push_back(default_sink);
            default_sink_added = true;
        }

        if (!log_file.empty()) {
            const fs::path explicit_path(log_file);
            if (!default_sink_added || !same_log_target(default_log_path, explicit_path)) {
                sinks.push_back(make_rotating_file_sink(explicit_path));
            }
        }

        impl_->logger = std::make_shared<spdlog::logger>("lfs", sinks.begin(), sinks.end());
        impl_->logger->set_level(spdlog::level::trace);
        impl_->logger->flush_on(spdlog::level::err);
        spdlog::set_default_logger(impl_->logger);
        spdlog::flush_every(std::chrono::seconds(2));

        global_level_ = static_cast<uint8_t>(console_level);
        capture_all_to_file_ = !log_file.empty();
    }

    std::string Logger::default_log_file_path(const std::string& user_dir_override) {
        return resolve_default_log_path(user_dir_override).string();
    }

    bool Logger::is_ready() const noexcept {
        std::lock_guard lock(impl_->mutex);
        return static_cast<bool>(impl_->logger);
    }

    void Logger::reset_for_testing() noexcept {
        std::lock_guard lock(impl_->mutex);
        impl_->logger.reset();
        impl_->console_sink.reset();
        impl_->memory_sink.reset();
    }

    LogHandlerToken Logger::add_log_handler(LogHandler handler) {
        std::lock_guard lock(impl_->handler_mutex);
        const auto token = impl_->next_handler_token_++;
        impl_->log_handlers_.emplace_back(token, std::move(handler));
        return token;
    }

    void Logger::remove_log_handler(LogHandlerToken handler_token) {
        std::lock_guard lock(impl_->handler_mutex);
        auto& handlers = impl_->log_handlers_;
        handlers.erase(
            std::remove_if(handlers.begin(), handlers.end(),
                           [handler_token](const auto& p) { return p.first == handler_token; }),
            handlers.end());
    }

    void Logger::log(const LogLevel level, const SourceSite& loc, const std::string_view msg) {
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

        std::vector<std::pair<LogHandlerToken, LogHandler>> handlers_snapshot;
        {
            std::lock_guard lock(impl_->handler_mutex);
            handlers_snapshot = impl_->log_handlers_;
        }
        // Contain each handler: a throwing handler must not interrupt the
        // caller of log()/log_internal() or stop later handlers from
        // running. The faulty handler is disabled so it cannot repeat the
        // failure on every subsequent log call.
        std::vector<LogHandlerToken> faulty_tokens;
        for (const auto& [token, handler] : handlers_snapshot) {
            try {
                handler(level, loc, final_msg);
            } catch (const std::exception& e) {
                // LFS-CENSUS-OK(empty-catch): handler containment reports directly to
                // stderr, not LOG_*, because the failure is Logger's own handler misbehaving.
                std::fprintf(stderr, "lichtfeld: log handler %u threw '%s'; disabling it\n",
                             static_cast<unsigned>(token), e.what());
                faulty_tokens.push_back(token);
            } catch (...) {
                // LFS-CENSUS-OK(empty-catch): same rationale as the std::exception branch above.
                std::fprintf(stderr, "lichtfeld: log handler %u threw a non-std exception; disabling it\n",
                             static_cast<unsigned>(token));
                faulty_tokens.push_back(token);
            }
        }
        for (const LogHandlerToken token : faulty_tokens) {
            remove_log_handler(token);
        }
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

    ScopedTimer::ScopedTimer(std::string name, const LogLevel level, const SourceSite loc)
        : start_(std::chrono::high_resolution_clock::now()),
          name_(std::move(name)),
          level_(level),
          loc_(loc) {
        try {
            diagnostics_scope_active_ = lfs::diagnostics::VramProfiler::instance().enabled();
            if (diagnostics_scope_active_) {
                lfs::diagnostics::VramProfiler::instance().pushTimerScope(name_);
            }
        } catch (...) {
            diagnostics_scope_active_ = false;
        }
    }

    ScopedTimer::ScopedTimer(std::string name, const double min_log_ms,
                             const LogLevel level, const SourceSite loc)
        : ScopedTimer(std::move(name), level, loc) {
        min_log_ms_ = min_log_ms;
    }

    ScopedTimer::~ScopedTimer() {
        const auto duration = std::chrono::high_resolution_clock::now() - start_;
        const auto ms = std::chrono::duration<double, std::milli>(duration).count();
        if (diagnostics_scope_active_) {
            try {
                lfs::diagnostics::VramProfiler::instance().popTimerScope(ms);
            } catch (...) {
            }
        }
        if (ms < min_log_ms_)
            return;
        Logger::get().log(level_, loc_, std::format("{} took {:.2f}ms", name_, ms));
    }

} // namespace lfs::core
