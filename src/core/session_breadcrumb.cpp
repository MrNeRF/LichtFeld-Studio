/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/session_breadcrumb.hpp"

#include "core/logger.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string_view>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace lfs::core {

    namespace {
        namespace fs = std::filesystem;
        using json = nlohmann::json;

        constexpr std::uintmax_t PREVIOUS_LOG_MAX_BYTES = 1024ULL * 1024ULL;
        std::mutex breadcrumb_mutex;
        std::optional<SessionBreadcrumb> previous_record;
        std::optional<SessionBreadcrumb> current_record;

        fs::path breadcrumb_path() {
            return lichtfeld_home_directory() / ".lichtfeld" / "logs" / "last_session.json";
        }

        fs::path snapshot_path() {
            return lichtfeld_home_directory() / ".lichtfeld" / "logs" / "previous_session.log";
        }

        std::uint64_t process_id() {
#ifdef _WIN32
            return static_cast<std::uint64_t>(GetCurrentProcessId());
#else
            return static_cast<std::uint64_t>(getpid());
#endif
        }

        std::string utc_now() {
            const auto now = std::chrono::system_clock::now();
            const std::time_t timestamp = std::chrono::system_clock::to_time_t(now);
            std::tm utc{};
#ifdef _WIN32
            gmtime_s(&utc, &timestamp);
#else
            gmtime_r(&timestamp, &utc);
#endif
            std::ostringstream output;
            output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
            return output.str();
        }

        std::optional<SessionBreadcrumb> decode_record(const fs::path& path) {
            try {
                std::ifstream input(path);
                if (!input)
                    return std::nullopt;
                json value;
                input >> value;
                if (!value.is_object())
                    return std::nullopt;

                SessionBreadcrumb record;
                record.pid = value.at("pid").get<std::uint64_t>();
                record.started_at = value.at("started_at").get<std::string>();
                record.log_path = value.at("log_path").get<std::string>();
                record.clean_exit = value.at("clean_exit").get<bool>();
                return record;
            } catch (...) {
                return std::nullopt;
            }
        }

        bool replace_file(const fs::path& source, const fs::path& destination) {
#ifdef _WIN32
            return MoveFileExW(source.c_str(), destination.c_str(),
                               MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
            return std::rename(source.c_str(), destination.c_str()) == 0;
#endif
        }

        bool write_atomically(const fs::path& path, const std::string_view contents) {
            std::error_code ec;
            fs::create_directories(path.parent_path(), ec);
            if (ec)
                return false;

            fs::path temporary = path;
            temporary += ".tmp." + std::to_string(process_id());
            {
                std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
                if (!output)
                    return false;
                output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
                output.flush();
                if (!output)
                    return false;
            }
            if (replace_file(temporary, path))
                return true;
            fs::remove(temporary, ec);
            return false;
        }

        bool write_record(const fs::path& path, const SessionBreadcrumb& record) {
            const json value = {
                {"pid", record.pid},
                {"started_at", record.started_at},
                {"log_path", record.log_path},
                {"clean_exit", record.clean_exit},
            };
            return write_atomically(path, value.dump(2) + "\n");
        }

        std::string read_log_tail(const fs::path& path) {
            std::ifstream input(path, std::ios::binary | std::ios::ate);
            if (!input)
                return {};
            const auto end = input.tellg();
            if (end <= 0)
                return {};

            const auto size = static_cast<std::uintmax_t>(end);
            const bool truncated = size > PREVIOUS_LOG_MAX_BYTES;
            const auto read_size = static_cast<std::size_t>(
                std::min(size, PREVIOUS_LOG_MAX_BYTES));
            input.seekg(static_cast<std::streamoff>(size - read_size));
            std::string tail(read_size, '\0');
            input.read(tail.data(), static_cast<std::streamsize>(read_size));
            tail.resize(static_cast<std::size_t>(input.gcount()));

            if (truncated) {
                const auto newline = tail.find('\n');
                if (newline != std::string::npos)
                    tail.erase(0, newline + 1);
            }
            return tail;
        }

        bool snapshot_previous_log(const std::string& live_log_path) {
            if (live_log_path.empty())
                return false;
            const std::string tail = read_log_tail(fs::path(live_log_path));
            if (tail.empty())
                return false;
            return write_atomically(snapshot_path(), tail);
        }

    } // namespace

    void record_session_start() noexcept {
        try {
            std::lock_guard lock(breadcrumb_mutex);
            previous_record = decode_record(breadcrumb_path());
            if (previous_record && !previous_record->clean_exit) {
                try {
                    if (snapshot_previous_log(previous_record->log_path))
                        previous_record->log_path = snapshot_path().string();
                    else
                        previous_record->log_path.clear();
                } catch (...) {
                    previous_record->log_path.clear();
                }
            } else if (previous_record) {
                previous_record->log_path.clear();
            }

            current_record = SessionBreadcrumb{
                .pid = process_id(),
                .started_at = utc_now(),
                .log_path = Logger::default_log_file_path(),
                .clean_exit = false,
            };
            static_cast<void>(write_record(breadcrumb_path(), *current_record));
        } catch (...) {
        }
    }

    void mark_clean_exit() noexcept {
        try {
            std::lock_guard lock(breadcrumb_mutex);
            if (!current_record)
                return;
            current_record->clean_exit = true;
            static_cast<void>(write_record(breadcrumb_path(), *current_record));
        } catch (...) {
        }
    }

    std::optional<SessionBreadcrumb> previous_session() noexcept {
        try {
            std::lock_guard lock(breadcrumb_mutex);
            return previous_record;
        } catch (...) {
            return std::nullopt;
        }
    }

} // namespace lfs::core
