#include <core/logger.hpp>
#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <utility>
#include <vector>

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
