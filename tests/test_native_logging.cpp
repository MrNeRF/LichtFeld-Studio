/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/logger/native_rotating_file_sink.hpp"
#include <spdlog/logger.h>
#include <gtest/gtest.h>
#include <chrono>
namespace {
    namespace fs = lfs::core::fs;
    class NativeLoggingTest : public ::testing::Test {
    protected:
        void SetUp() override {
            root_ = fs::temp_directory_path() / ("lfs-log-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
            deep_ = root_;
            for (int i = 0; i < 12; ++i) deep_ /= "unicode-long-directory-component-0123456789";
            fs::create_directories(deep_);
        }
        void TearDown() override { std::error_code ec; fs::remove_all(root_, ec); EXPECT_FALSE(ec); }
        fs::path root_, deep_;
    };
    TEST_F(NativeLoggingTest, NativeLogSinkRotatesUnicodeLongPaths) {
        const auto path = deep_ / lfs::core::utf8_to_path("diagnostic_\u6e2c\u8a66.log");
        auto sink = std::make_shared<lfs::core::NativeRotatingFileSink>(path, 16, 2);
        spdlog::logger logger("native-path-test", sink);
        logger.set_pattern("%v");
        logger.info("first-payload"); logger.info("second-payload"); logger.info("third-payload"); logger.flush();
        const auto read = [](const fs::path& value) { fs::ifstream file(value, std::ios::binary); return std::string(std::istreambuf_iterator<char>(file), {}); };
        EXPECT_NE(read(path).find("third-payload"), std::string::npos);
        EXPECT_NE(read(lfs::core::NativeRotatingFileSink::rotated_path(path, 1)).find("second-payload"), std::string::npos);
        EXPECT_NE(read(lfs::core::NativeRotatingFileSink::rotated_path(path, 2)).find("first-payload"), std::string::npos);
        const auto long_name = deep_ / (std::string(249, 'x') + ".log");
        const auto rotated = lfs::core::NativeRotatingFileSink::rotated_path(long_name, 12);
        EXPECT_LE(lfs::core::path_to_utf8(rotated.filename()).size(), 255u);
        EXPECT_NE(rotated, lfs::core::NativeRotatingFileSink::rotated_path(deep_ / (std::string(249, 'y') + ".log"), 12));
    }
}
