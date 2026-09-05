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
#ifdef _WIN32
#include "core/windows_console.hpp"
#include <fcntl.h>
#include <memory>
#else
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

#ifdef _WIN32
TEST(LoggerWindowsConsoleTest, UnicodeUsesPrivateScreenBufferWithoutChangingConsoleSettings) {
    // An inactive buffer exercises real WriteConsoleW without touching the user's
    // visible output or changing the console code pages, modes or active buffer.
    const HANDLE buffer = CreateConsoleScreenBuffer(GENERIC_READ | GENERIC_WRITE,
                                                    FILE_SHARE_READ | FILE_SHARE_WRITE,
                                                    nullptr, CONSOLE_TEXTMODE_BUFFER, nullptr);
    if (buffer == INVALID_HANDLE_VALUE)
        GTEST_SKIP() << "Requires an attached Windows console";
    const int fd = _open_osfhandle(reinterpret_cast<intptr_t>(buffer), _O_WRONLY | _O_TEXT);
    if (fd < 0) {
        CloseHandle(buffer);
        FAIL() << "Cannot create console descriptor";
    }
    FILE* raw = _fdopen(fd, "w");
    if (!raw) {
        _close(fd);
        FAIL() << "Cannot create console stream";
    }
    const std::unique_ptr<FILE, decltype(&std::fclose)> stream(raw, &std::fclose);
    const auto input_cp = GetConsoleCP();
    const auto output_cp = GetConsoleOutputCP();
    DWORD mode_before = 0;
    ASSERT_TRUE(GetConsoleMode(buffer, &mode_before));
    EXPECT_EQ(lfs::core::detail::console_output_handle(stream.get()), buffer);

    // Cross the writer's chunk boundary before the visible Unicode suffix.
    std::string message(40'000, '\r');
    message += "caf\xc3\xa9 \xe2\x80\x94 \xce\xa9 \xd0\x96\n";
    ASSERT_TRUE(lfs::core::detail::write_console_utf8(buffer, message));
    const std::wstring expected = L"caf\u00e9 \u2014 \u03a9 \u0416";
    std::wstring actual(expected.size(), L'\0');
    DWORD read = 0;
    ASSERT_TRUE(ReadConsoleOutputCharacterW(buffer, actual.data(), static_cast<DWORD>(actual.size()),
                                            COORD{0, 0}, &read));
    EXPECT_EQ(read, expected.size());
    EXPECT_EQ(actual, expected);
    CONSOLE_SCREEN_BUFFER_INFO info{};
    ASSERT_TRUE(GetConsoleScreenBufferInfo(buffer, &info));
    EXPECT_EQ(info.dwCursorPosition.X, 0);
    EXPECT_EQ(info.dwCursorPosition.Y, 1);
    DWORD mode_after = 0;
    ASSERT_TRUE(GetConsoleMode(buffer, &mode_after));
    EXPECT_EQ(mode_after, mode_before);
    EXPECT_EQ(GetConsoleCP(), input_cp);
    EXPECT_EQ(GetConsoleOutputCP(), output_cp);
}

TEST(LoggerWindowsConsoleTest, RedirectedStdoutAndStderrRetainUtf8Bytes) {
    LoggerInitGuard restore;
    auto& logger = lfs::core::Logger::get();
    const std::string message = "caf\xc3\xa9 \xe2\x80\x94 \xe6\x97\xa5\xe6\x9c\xac \xf0\x9f\x8c\x8d";
    for (const bool use_stderr : {false, true}) {
        logger.init(lfs::core::LogLevel::Info, "", "", use_stderr);
        // Capture after init to guard against caching the old stream handle.
        if (use_stderr)
            testing::internal::CaptureStderr();
        else
            testing::internal::CaptureStdout();
        const auto handle = lfs::core::detail::console_output_handle(use_stderr ? stderr : stdout);
        logger.log(lfs::core::LogLevel::Info, LFS_SOURCE_SITE_CURRENT(), message);
        const auto captured = use_stderr ? testing::internal::GetCapturedStderr()
                                         : testing::internal::GetCapturedStdout();
        EXPECT_EQ(handle, INVALID_HANDLE_VALUE);
        EXPECT_EQ(count_occurrences(captured, message), 1);
        EXPECT_NE(logger.buffered_logs_as_text().find(message), std::string::npos);
    }
}

TEST(LoggerWindowsConsoleTest, InitializationWritesUnicodeProbeToLogFile) {
    LoggerInitGuard restore;
    const auto directory = unique_temp_dir("unicode_probe");
    const auto log_file = directory / "startup.log";
    std::filesystem::create_directories(directory);
    lfs::core::Logger::get().init(lfs::core::LogLevel::Info, log_file.string());
    lfs::core::Logger::get().flush();

    const auto content = read_file(log_file);
    EXPECT_NE(content.find("\xE2\x80\x94 UTF-8 console: \xE2\x9C\x93"), std::string::npos);
    // Release the explicit rotating sink before deleting its Windows file.
    lfs::core::Logger::get().init();
    std::filesystem::remove_all(directory);
}
#endif

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

TEST(LoggerTest, ScopedTimerDisabledPathDoesNotEmit) {
    auto& logger = lfs::core::Logger::get();
    const auto previous_level = logger.level();
    logger.set_level(lfs::core::LogLevel::Info);

    std::vector<std::string> messages;
    LogHandlerGuard guard([&messages](lfs::core::LogLevel level,
                                      const lfs::core::SourceSite&,
                                      std::string_view message) {
        if (level == lfs::core::LogLevel::Performance)
            messages.emplace_back(message);
    });

    {
        lfs::core::ScopedTimer timer(
            "logger.disabled", lfs::core::LogLevel::Performance,
            LFS_SOURCE_SITE_CURRENT());
    }

    logger.set_level(previous_level);
    EXPECT_TRUE(messages.empty());
}

TEST(LoggerTest, DefaultLogFilePathResolvesUnderPerUserDirectory) {
    const std::filesystem::path resolved(lfs::core::Logger::default_log_file_path());

    EXPECT_EQ(resolved.filename(), "lichtfeld.log");
    ASSERT_TRUE(resolved.has_parent_path());
    EXPECT_EQ(resolved.parent_path().filename(), "logs");
    ASSERT_TRUE(resolved.parent_path().has_parent_path());
    EXPECT_EQ(resolved.parent_path().parent_path().filename(), ".lichtfeld");
}

TEST(LoggerTest, BufferedLogsSinceReturnsBoundedMonotonicRingTail) {
    LoggerInitGuard reset_guard;
    const auto temp_root = unique_temp_dir("incremental");
    auto& logger = lfs::core::Logger::get();
    logger.init(lfs::core::LogLevel::Info, "", "", false, temp_root.string());

    const auto initial_generation = logger.buffered_log_generation();
    LOG_INFO("incremental-first");
    LOG_INFO("incremental-second");
    LOG_INFO("incremental-third");

    const auto first_tail = logger.buffered_logs_since(initial_generation, 2);
    ASSERT_EQ(first_tail.size(), 2u);
    EXPECT_EQ(first_tail[0].message, "incremental-second");
    EXPECT_EQ(first_tail[1].message, "incremental-third");
    EXPECT_LT(first_tail[0].sequence, first_tail[1].sequence);

    const auto seen_generation = logger.buffered_log_generation();
    LOG_INFO("incremental-fourth");
    const auto second_tail = logger.buffered_logs_since(seen_generation, 10);
    ASSERT_EQ(second_tail.size(), 1u);
    EXPECT_GT(second_tail.front().sequence, seen_generation);

    for (size_t index = 0; index < 5005; ++index)
        LOG_INFO("incremental-ring-{}", index);

    const auto retained = logger.buffered_logs();
    ASSERT_EQ(retained.size(), 5000u);
    ASSERT_GT(retained.front().sequence, initial_generation);
    const auto older_than_oldest = retained.front().sequence - 1;
    const auto ring_tail = logger.buffered_logs_since(older_than_oldest, 5000);
    ASSERT_EQ(ring_tail.size(), 5000u);
    EXPECT_EQ(ring_tail.front().message, "incremental-ring-5");
    EXPECT_EQ(ring_tail.back().message, "incremental-ring-5004");
    for (size_t index = 1; index < ring_tail.size(); ++index)
        EXPECT_LT(ring_tail[index - 1].sequence, ring_tail[index].sequence);

    const auto bounded_ring_tail = logger.buffered_logs_since(older_than_oldest, 2);
    ASSERT_EQ(bounded_ring_tail.size(), 2u);
    EXPECT_EQ(bounded_ring_tail[0].message, "incremental-ring-5003");
    EXPECT_EQ(bounded_ring_tail[1].message, "incremental-ring-5004");
    EXPECT_LT(bounded_ring_tail[0].sequence, bounded_ring_tail[1].sequence);

    std::error_code error;
    std::filesystem::remove_all(temp_root, error);
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
