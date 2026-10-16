/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/logger/native_rotating_file_sink.hpp"
#include <chrono>
#include <gtest/gtest.h>
#include <spdlog/details/os.h>
#include <spdlog/logger.h>
namespace {
    namespace fs = lfs::core::native_fs;
    class NativeLoggingTest : public ::testing::Test {
    protected:
        void SetUp() override {
            root_ = fs::temp_directory_path() / ("lfs-log-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
            deep_ = root_;
            for (int i = 0; i < 12; ++i)
                deep_ /= "unicode-long-directory-component-0123456789";
            fs::create_directories(deep_);
        }
        void TearDown() override {
            std::error_code ec;
            fs::remove_all(root_, ec);
            EXPECT_FALSE(ec);
        }
        fs::path root_, deep_;
        static std::string read(const fs::path& path) {
            std::ifstream file(fs::native_path(path), std::ios::binary);
            return {std::istreambuf_iterator<char>(file), {}};
        }
    };
    TEST_F(NativeLoggingTest, NativeLogSinkRotatesUnicodeLongPaths) {
        const auto path = deep_ / lfs::core::utf8_to_path("diagnostic_\u6e2c\u8a66.log");
        auto sink = std::make_shared<lfs::core::NativeRotatingFileSink>(path, 16, 2);
        spdlog::logger logger("native-path-test", sink);
        logger.set_pattern("%v");
        logger.info("first-payload");
        logger.info("second-payload");
        logger.info("third-payload");
        logger.flush();
        const auto read = [](const fs::path& value) { std::ifstream file(lfs::core::native_fs::native_path(value), std::ios::binary); return std::string(std::istreambuf_iterator<char>(file), {}); };
        EXPECT_NE(read(path).find("third-payload"), std::string::npos);
        EXPECT_NE(read(lfs::core::NativeRotatingFileSink::rotated_path(path, 1)).find("second-payload"), std::string::npos);
        EXPECT_NE(read(lfs::core::NativeRotatingFileSink::rotated_path(path, 2)).find("first-payload"), std::string::npos);
        const auto long_name = deep_ / (std::string(249, 'x') + ".log");
        const auto rotated = lfs::core::NativeRotatingFileSink::rotated_path(long_name, 12);
        EXPECT_LE(lfs::core::path_to_utf8(rotated.filename()).size(), 255u);
        EXPECT_NE(rotated, lfs::core::NativeRotatingFileSink::rotated_path(deep_ / (std::string(249, 'y') + ".log"), 12));
    }

    TEST_F(NativeLoggingTest, FailedHistoryRotationPreservesLinesAndDoesNotRetry) {
        const auto path = deep_ / "rotation-failure.log";
        const auto history = lfs::core::NativeRotatingFileSink::rotated_path(path, 1);
        const auto blocked = lfs::core::NativeRotatingFileSink::rotated_path(path, 2);
        {
            std::ofstream file(fs::native_path(history));
            file << "previous-history";
        }
        fs::create_directory(blocked);
        {
            std::ofstream file(fs::native_path(blocked / "blocker"));
            file << "occupied";
        }
        auto sink = std::make_shared<lfs::core::NativeRotatingFileSink>(path, 16, 2);
        spdlog::logger logger("failed-rotation-test", sink);
        logger.set_pattern("%v");
        int errors = 0;
        logger.set_error_handler([&](const std::string&) { ++errors; });
        logger.info("first-payload");
        logger.info("rotation-trigger");
        logger.flush();
        fs::remove_all(blocked);
        // Removing the blocker must not cause another history shift on each line.
        logger.info("after-failure");
        logger.info("another-line");
        logger.flush();
        EXPECT_EQ(errors, 1);
        const std::string eol = spdlog::details::os::default_eol;
        EXPECT_EQ(read(path), "first-payload" + eol + "rotation-trigger" + eol + "after-failure" + eol + "another-line" + eol);
        EXPECT_EQ(read(history), "previous-history");
        EXPECT_FALSE(fs::exists(blocked));
    }

#ifndef _WIN32
    TEST_F(NativeLoggingTest, BoundedRotatedNamesKeepNonUtf8BytesDistinct) {
        const auto first = deep_ / fs::path(std::string(250, 'x') + "\xe9.log");
        const auto second = deep_ / fs::path(std::string(250, 'x') + "\xe8.log");
        EXPECT_NE(lfs::core::NativeRotatingFileSink::rotated_path(first, 1),
                  lfs::core::NativeRotatingFileSink::rotated_path(second, 1));
    }
#endif

#ifdef _WIN32
    TEST_F(NativeLoggingTest, SharedActiveLogPreservesAllHistoryAndContinuesAppending) {
        const auto path = deep_ / "shared.log";
        for (size_t i = 1; i <= 4; ++i) {
            std::ofstream file(fs::native_path(lfs::core::NativeRotatingFileSink::rotated_path(path, i)));
            file << "history-" << i;
        }
        auto sink = std::make_shared<lfs::core::NativeRotatingFileSink>(path, 16, 4);
        spdlog::logger logger("shared-rotation-test", sink);
        logger.set_pattern("%v");
        int errors = 0;
        logger.set_error_handler([&](const std::string&) { ++errors; });
        logger.info("first-payload");
        logger.flush();
        const auto handle = CreateFileW(fs::native_path(path).c_str(), GENERIC_READ,
                                        FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                                        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        ASSERT_NE(handle, INVALID_HANDLE_VALUE);
        logger.info("rotation-trigger");
        logger.info("while-shared");
        logger.flush();
        CloseHandle(handle);
        logger.info("after-close");
        logger.flush();
        EXPECT_EQ(errors, 1);
        const std::string eol = spdlog::details::os::default_eol;
        EXPECT_EQ(read(path), "first-payload" + eol + "rotation-trigger" + eol + "while-shared" + eol + "after-close" + eol);
        for (size_t i = 1; i <= 4; ++i)
            EXPECT_EQ(read(lfs::core::NativeRotatingFileSink::rotated_path(path, i)), "history-" + std::to_string(i));
    }
#endif
} // namespace
