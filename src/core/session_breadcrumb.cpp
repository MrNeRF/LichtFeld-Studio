/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/session_breadcrumb.hpp"

#include "core/logger.hpp"
#include "core/user_paths.hpp"
#include "core/utc_time.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
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
            const auto paths = UserPaths::resolve();
            return (paths ? paths->logDir()
                          : lichtfeld_home_directory() / ".lichtfeld" / "logs") /
                   "last_session.json";
        }

        fs::path snapshot_path() {
            const auto paths = UserPaths::resolve();
            return (paths ? paths->logDir()
                          : lichtfeld_home_directory() / ".lichtfeld" / "logs") /
                   "previous_session.log";
        }

        std::uint64_t process_id() {
#ifdef _WIN32
            return static_cast<std::uint64_t>(GetCurrentProcessId());
#else
            return static_cast<std::uint64_t>(getpid());
#endif
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

        bool write_atomically(const fs::path& path, const std::string_view contents) {
            return writeTextFileAtomically(path, std::string(contents)).has_value();
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

            const auto paths = UserPaths::resolve();
            const auto live_log_path =
                (paths ? paths->logDir()
                       : lichtfeld_home_directory() / ".lichtfeld" / "logs") /
                "lichtfeld.log";
            current_record = SessionBreadcrumb{
                .pid = process_id(),
                .started_at = utc_now(),
                .log_path = live_log_path.string(),
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
