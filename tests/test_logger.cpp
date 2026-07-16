#include <core/logger.hpp>
#include <gtest/gtest.h>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#ifndef _WIN32
#include <unistd.h>
#endif

namespace {

    class LogHandlerGuard {
    public:
        explicit LogHandlerGuard(lfs::core::LogHandler handler)
            : token_(lfs::core::Logger::get().add_log_handler(std::move(handler))) {}

        ~LogHandlerGuard() {
            lfs::core::Logger::get().remove_log_handler(token_);
        }

        LogHandlerGuard(const LogHandlerGuard&) = delete;
        LogHandlerGuard& operator=(const LogHandlerGuard&) = delete;

    private:
        lfs::core::LogHandlerToken token_;
    };

    // Restores the process-global Logger singleton to its default init state
    // after a test reconfigures it with a temp-dir override.
    class LoggerInitGuard {
    public:
        LoggerInitGuard() = default;
        ~LoggerInitGuard() {
            lfs::core::Logger::get().init();
        }

        LoggerInitGuard(const LoggerInitGuard&) = delete;
        LoggerInitGuard& operator=(const LoggerInitGuard&) = delete;
    };

    std::filesystem::path unique_temp_dir(const std::string& label) {
        static std::atomic<uint64_t> counter{0};
        const auto suffix = std::to_string(counter.fetch_add(1)) + "_" +
                            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        return std::filesystem::temp_directory_path() / ("lfs_logger_test_" + label + "_" + suffix);
    }

    std::string next_marker(const std::string& label) {
        static std::atomic<uint64_t> counter{0};
        return "logger_test_marker_" + label + "_" + std::to_string(counter.fetch_add(1));
    }

    std::string read_file(const std::filesystem::path& path) {
        std::ifstream input(path, std::ios::binary);
        std::ostringstream buffer;
        buffer << input.rdbuf();
        return buffer.str();
    }

    size_t count_occurrences(const std::string& haystack, const std::string& needle) {
        size_t count = 0;
        for (size_t pos = haystack.find(needle); pos != std::string::npos; pos = haystack.find(needle, pos + needle.size()))
            ++count;
        return count;
    }

} // namespace

TEST(LoggerTest, ScopedTimerThresholdSuppressesBelowThresholdPerfLog) {
    auto& logger = lfs::core::Logger::get();
    const auto previous_level = logger.level();
    logger.set_level(lfs::core::LogLevel::Performance);

    std::vector<std::string> messages;
    LogHandlerGuard guard([&messages](lfs::core::LogLevel level,
                                      const lfs::core::SourceSite&,
                                      std::string_view message) {
        if (level == lfs::core::LogLevel::Performance)
            messages.emplace_back(message);
    });

    {
        lfs::core::ScopedTimer timer(
            "logger.threshold.suppressed", 60'000.0,
            lfs::core::LogLevel::Performance, LFS_SOURCE_SITE_CURRENT());
    }

    logger.set_level(previous_level);

    EXPECT_TRUE(messages.empty());
}

TEST(LoggerTest, ScopedTimerThresholdKeepsZeroThresholdCompatible) {
    auto& logger = lfs::core::Logger::get();
    const auto previous_level = logger.level();
    logger.set_level(lfs::core::LogLevel::Performance);

    std::vector<std::string> messages;
    LogHandlerGuard guard([&messages](lfs::core::LogLevel level,
                                      const lfs::core::SourceSite&,
                                      std::string_view message) {
        if (level == lfs::core::LogLevel::Performance)
            messages.emplace_back(message);
    });

    {
        lfs::core::ScopedTimer timer(
            "logger.threshold.compat", 0.0,
            lfs::core::LogLevel::Performance, LFS_SOURCE_SITE_CURRENT());
    }

    logger.set_level(previous_level);

    ASSERT_EQ(messages.size(), 1);
    EXPECT_NE(messages.front().find("logger.threshold.compat took"), std::string::npos);
}

TEST(LoggerTest, DefaultLogFilePathResolvesUnderPerUserDirectory) {
    const std::filesystem::path resolved(lfs::core::Logger::default_log_file_path());

    EXPECT_EQ(resolved.filename(), "lichtfeld.log");
    ASSERT_TRUE(resolved.has_parent_path());
    EXPECT_EQ(resolved.parent_path().filename(), "logs");
    ASSERT_TRUE(resolved.parent_path().has_parent_path());
    EXPECT_EQ(resolved.parent_path().parent_path().filename(), ".lichtfeld");
}

TEST(LoggerTest, DefaultLogFilePathHonorsExplicitOverride) {
    const auto override_dir = std::filesystem::temp_directory_path() / "lfs_logger_test_override_dir";
    const std::filesystem::path resolved(
        lfs::core::Logger::default_log_file_path(override_dir.string()));

    EXPECT_EQ(resolved, override_dir / "logs" / "lichtfeld.log");
}

TEST(LoggerTest, InitOnFreshTempDirCreatesDurableLogFile) {
    LoggerInitGuard reset_guard;

    const std::filesystem::path temp_root = unique_temp_dir("init_fresh");
    std::error_code ec;
    std::filesystem::remove_all(temp_root, ec);

    auto& logger = lfs::core::Logger::get();
    logger.init(lfs::core::LogLevel::Info, "", "", false, temp_root.string());

    const std::string marker = next_marker("init_fresh");
    LOG_INFO("{}", marker);
    logger.flush();

    const std::filesystem::path expected_path(lfs::core::Logger::default_log_file_path(temp_root.string()));
    ASSERT_TRUE(std::filesystem::exists(expected_path));
    EXPECT_NE(read_file(expected_path).find(marker), std::string::npos);

    std::filesystem::remove_all(temp_root, ec);
}

TEST(LoggerTest, ExplicitLogFileAddsAdditionalSink) {
    LoggerInitGuard reset_guard;

    const std::filesystem::path default_root = unique_temp_dir("explicit_default");
    const std::filesystem::path explicit_root = unique_temp_dir("explicit_extra");
    std::error_code ec;
    std::filesystem::remove_all(default_root, ec);
    std::filesystem::remove_all(explicit_root, ec);
    std::filesystem::create_directories(explicit_root, ec);
    ASSERT_FALSE(ec);
    const std::filesystem::path explicit_path = explicit_root / "extra.log";

    auto& logger = lfs::core::Logger::get();
    logger.init(lfs::core::LogLevel::Info, explicit_path.string(), "", false, default_root.string());

    const std::string marker = next_marker("explicit_extra");
    LOG_INFO("{}", marker);
    logger.flush();

    const std::filesystem::path default_path(lfs::core::Logger::default_log_file_path(default_root.string()));
    ASSERT_TRUE(std::filesystem::exists(default_path));
    ASSERT_TRUE(std::filesystem::exists(explicit_path));
    EXPECT_NE(read_file(default_path).find(marker), std::string::npos);
    EXPECT_NE(read_file(explicit_path).find(marker), std::string::npos);

    std::filesystem::remove_all(default_root, ec);
    std::filesystem::remove_all(explicit_root, ec);
}

TEST(LoggerTest, ExplicitLogFileDedupesWhenSameAsDefault) {
    LoggerInitGuard reset_guard;

    const std::filesystem::path default_root = unique_temp_dir("dedupe");
    std::error_code ec;
    std::filesystem::remove_all(default_root, ec);

    const std::string same_path = lfs::core::Logger::default_log_file_path(default_root.string());

    auto& logger = lfs::core::Logger::get();
    logger.init(lfs::core::LogLevel::Info, same_path, "", false, default_root.string());

    const std::string marker = next_marker("dedupe");
    LOG_INFO("{}", marker);
    logger.flush();

    ASSERT_TRUE(std::filesystem::exists(same_path));
    EXPECT_EQ(count_occurrences(read_file(same_path), marker), 1u);

    std::filesystem::remove_all(default_root, ec);
}

TEST(LoggerTest, InitWithUnwritableDirectoryKeepsConsoleAndMemoryWorking) {
#ifndef _WIN32
    if (geteuid() == 0) {
        GTEST_SKIP() << "Running as root bypasses directory permission checks";
    }
#endif
    LoggerInitGuard reset_guard;

    const std::filesystem::path readonly_root = unique_temp_dir("readonly");
    std::error_code ec;
    std::filesystem::remove_all(readonly_root, ec);
    std::filesystem::create_directories(readonly_root, ec);
    ASSERT_FALSE(ec);
    std::filesystem::permissions(readonly_root,
                                 std::filesystem::perms::owner_read | std::filesystem::perms::owner_exec,
                                 std::filesystem::perm_options::replace, ec);
    ASSERT_FALSE(ec);

    auto& logger = lfs::core::Logger::get();
    EXPECT_NO_THROW(logger.init(lfs::core::LogLevel::Info, "", "", false, readonly_root.string()));

    const std::string marker = next_marker("readonly");
    LOG_INFO("{}", marker);
    logger.flush();

    EXPECT_NE(logger.buffered_logs_as_text().find(marker), std::string::npos);

    std::filesystem::permissions(readonly_root, std::filesystem::perms::owner_all,
                                 std::filesystem::perm_options::replace, ec);
    std::filesystem::remove_all(readonly_root, ec);
}
