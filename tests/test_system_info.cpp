/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "core/session_breadcrumb.hpp"
#include "core/system_info.hpp"
#include "core/user_paths.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <regex>
#include <sstream>
#include <string>

namespace {
    namespace fs = std::filesystem;

    fs::path unique_temp_dir(const std::string& label) {
        static std::atomic<std::uint64_t> counter{0};
        return fs::temp_directory_path() /
               ("lfs_system_info_" + label + "_" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "_" +
                std::to_string(counter.fetch_add(1)));
    }

    class HomeDirectoryGuard {
    public:
        explicit HomeDirectoryGuard(const fs::path& home) {
#ifdef _WIN32
            if (const char* existing = std::getenv("USERPROFILE"))
                old_value_ = existing;
            _putenv_s("USERPROFILE", home.string().c_str());
#else
            if (const char* existing = std::getenv("HOME"))
                old_value_ = existing;
            setenv("HOME", home.string().c_str(), 1);
#endif
        }

        ~HomeDirectoryGuard() {
#ifdef _WIN32
            _putenv_s("USERPROFILE", old_value_.value_or("").c_str());
#else
            if (old_value_)
                setenv("HOME", old_value_->c_str(), 1);
            else
                unsetenv("HOME");
#endif
        }

        HomeDirectoryGuard(const HomeDirectoryGuard&) = delete;
        HomeDirectoryGuard& operator=(const HomeDirectoryGuard&) = delete;

    private:
        std::optional<std::string> old_value_;
    };

    nlohmann::json read_json(const fs::path& path) {
        std::ifstream input(path);
        return nlohmann::json::parse(input);
    }

    std::string read_text(const fs::path& path) {
        std::ifstream input(path, std::ios::binary);
        std::ostringstream output;
        output << input.rdbuf();
        return output.str();
    }

} // namespace

TEST(SystemInfoTest, CollectsHostInventoryWithoutThrowing) {
    const auto info = lfs::core::system_info::collect();

    EXPECT_FALSE(info.os.empty());
    EXPECT_FALSE(info.cpu.empty());
    EXPECT_GT(info.ram_mb, 0u);

    int runtime_version = 0;
    if (cudaRuntimeGetVersion(&runtime_version) == cudaSuccess) {
        EXPECT_TRUE(std::regex_match(info.cuda_runtime, std::regex(R"(\d+\.\d+)")));
    } else {
        EXPECT_TRUE(info.cuda_runtime.empty());
    }
}

TEST(LoggerTailTest, PreservesUnalignedSingleLineTail) {
    EXPECT_EQ(lfs::core::truncate_log_tail("0123456789", 4), "6789");
    EXPECT_EQ(lfs::core::truncate_log_tail("prefix\n0123456789", 4), "6789");
    EXPECT_EQ(lfs::core::truncate_log_tail("prefix\n0123456789", 8), "23456789");
}

TEST(SessionBreadcrumbTest, RecordsStartCleanExitAndPreviousSession) {
    const fs::path home = unique_temp_dir("round_trip");
    std::error_code ec;
    fs::remove_all(home, ec);
    HomeDirectoryGuard home_guard(home);

    lfs::core::record_session_start();
    const auto user_paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(user_paths.has_value())
        << lfs::format_for_developer(user_paths.error());
    const fs::path breadcrumb = user_paths->logDir() / "last_session.json";
    ASSERT_TRUE(fs::exists(breadcrumb));
    const auto started = read_json(breadcrumb);
    EXPECT_FALSE(started.at("clean_exit").get<bool>());
    EXPECT_FALSE(started.at("started_at").get<std::string>().empty());

    lfs::core::mark_clean_exit();
    const auto clean = read_json(breadcrumb);
    EXPECT_TRUE(clean.at("clean_exit").get<bool>());

    lfs::core::record_session_start();
    const auto previous = lfs::core::previous_session();
    ASSERT_TRUE(previous.has_value());
    EXPECT_TRUE(previous->clean_exit);
    EXPECT_EQ(previous->pid, clean.at("pid").get<std::uint64_t>());
    EXPECT_EQ(previous->started_at, clean.at("started_at").get<std::string>());
    EXPECT_TRUE(previous->log_path.empty());

    fs::remove_all(home, ec);
}

TEST(SessionBreadcrumbTest, SnapshotsOnlyTheUncleanPreviousLogTail) {
    const fs::path home = unique_temp_dir("snapshot");
    std::error_code ec;
    fs::remove_all(home, ec);
    HomeDirectoryGuard home_guard(home);

    lfs::core::record_session_start();
    const fs::path live_log(lfs::core::Logger::default_log_file_path());
    fs::create_directories(live_log.parent_path());
    constexpr std::string_view marker = "previous-session-marker\n";
    {
        std::ofstream output(live_log, std::ios::binary | std::ios::trunc);
        output << marker;
    }

    lfs::core::record_session_start();
    const auto previous = lfs::core::previous_session();
    ASSERT_TRUE(previous.has_value());
    ASSERT_FALSE(previous->clean_exit);
    ASSERT_FALSE(previous->log_path.empty());
    EXPECT_EQ(fs::path(previous->log_path).filename(), "previous_session.log");

    constexpr std::string_view live_noise = "new-session-noise\n";
    {
        std::ofstream output(live_log, std::ios::binary | std::ios::app);
        output << live_noise;
    }

    const std::string snapshot = read_text(previous->log_path);
    EXPECT_NE(snapshot.find(marker), std::string::npos);
    EXPECT_EQ(snapshot.find(live_noise), std::string::npos);

    fs::remove_all(home, ec);
}
