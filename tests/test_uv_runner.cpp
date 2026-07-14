/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "python/package_manager.hpp"
#include "python/uv_runner.hpp"
#include <chrono>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

using namespace lfs::python;

class UvRunnerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Skip test if UV is not available
        if (!PackageManager::instance().is_uv_available()) {
            GTEST_SKIP() << "UV not available, skipping test";
        }
    }
};

TEST_F(UvRunnerTest, RunUvVersion) {
    UvRunner runner;

    std::string output;
    bool completed = false;
    bool success = false;
    int exit_code = -1;

    runner.set_output_callback([&](const std::string& line, bool is_error, bool is_line_update) {
        output += line + "\n";
    });

    runner.set_completion_callback([&](bool s, int ec) {
        completed = true;
        success = s;
        exit_code = ec;
    });

    ASSERT_TRUE(runner.start({"--version"}));
    EXPECT_TRUE(runner.is_running());

    // Poll until complete
    int iterations = 0;
    while (runner.poll() && iterations < 1000) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        iterations++;
    }

    // Should complete
    EXPECT_TRUE(completed);
    EXPECT_TRUE(success);
    EXPECT_EQ(exit_code, 0);
    EXPECT_FALSE(output.empty());
    EXPECT_TRUE(output.find("uv") != std::string::npos);
}

TEST_F(UvRunnerTest, RunUvHelp) {
    UvRunner runner;

    std::string output;
    std::vector<std::string> lines;
    bool completed = false;

    runner.set_output_callback([&](const std::string& line, bool, bool) {
        output += line + "\n";
        lines.push_back(line);
    });

    runner.set_completion_callback([&](bool, int) {
        completed = true;
    });

    ASSERT_TRUE(runner.start({"--help"}));

    // Poll until complete
    int iterations = 0;
    while (runner.poll() && iterations < 1000) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        iterations++;
    }

    EXPECT_TRUE(completed);
    EXPECT_FALSE(output.empty());
    ASSERT_FALSE(lines.empty());
    for (const auto& line : lines) {
        EXPECT_EQ(line.find('\n'), std::string::npos);
        EXPECT_EQ(line.find('\r'), std::string::npos);
    }
}

TEST_F(UvRunnerTest, CancelOperation) {
    UvRunner runner;

    // Start a long-running operation (pip list)
    runner.start({"pip", "list", "--python", PackageManager::instance().venv_python().string()});

    // Should be running
    EXPECT_TRUE(runner.is_running());

    // Cancel it
    runner.cancel();

    // Should not be running after cancel
    EXPECT_FALSE(runner.is_running());
    EXPECT_TRUE(runner.is_complete());
}

TEST_F(UvRunnerTest, NonBlockingPoll) {
    UvRunner runner;

    runner.start({"--version"});

    // poll() should return immediately if no data yet
    auto start = std::chrono::steady_clock::now();
    bool result = runner.poll();
    auto elapsed = std::chrono::steady_clock::now() - start;

    // Should complete quickly (< 100ms for poll call itself)
    EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 100);

    // Wait for completion
    while (runner.poll()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}
