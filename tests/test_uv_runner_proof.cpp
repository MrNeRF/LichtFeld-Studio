/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#ifdef WIN32
#define NOMINMAX
#endif
#include "python/package_manager.hpp"
#include "python/uv_runner.hpp"
#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

using namespace lfs::python;
using namespace std::chrono;

class UvRunnerProofTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!PackageManager::instance().is_uv_available()) {
            GTEST_SKIP() << "UV not available, skipping test";
        }
    }
};

// PROOF 1: Output is correctly captured and displayed
TEST_F(UvRunnerProofTest, OutputIsCorrectlyCaptured) {
    UvRunner runner;

    std::vector<std::string> captured_lines;
    std::string full_output;

    runner.set_output_callback([&](const std::string& line, bool is_error, bool is_line_update) {
        captured_lines.push_back(line);
        full_output += line + "\n";
        // Print to console for visual verification
        std::cout << "[UV OUTPUT] " << (is_line_update ? "(update) " : "") << line << std::endl;
    });

    bool completed = false;
    bool success = false;
    runner.set_completion_callback([&](bool s, int) {
        completed = true;
        success = s;
    });

    // Run "uv --version" which outputs something like "uv 0.5.14"
    ASSERT_TRUE(runner.start({"--version"}));

    while (runner.poll()) {
        std::this_thread::sleep_for(milliseconds(10));
    }

    ASSERT_TRUE(completed);
    ASSERT_TRUE(success);

    // Verify output was captured
    EXPECT_FALSE(captured_lines.empty()) << "No output lines captured";
    EXPECT_FALSE(full_output.empty()) << "No output captured at all";

    // Verify the output contains expected content (uv version string)
    bool found_uv = full_output.find("uv") != std::string::npos;
    EXPECT_TRUE(found_uv) << "Output should contain 'uv', got: " << full_output;

    std::cout << "\n=== PROOF: Output correctly captured ===" << std::endl;
    std::cout << "Total lines captured: " << captured_lines.size() << std::endl;
    std::cout << "Full output:\n"
              << full_output << std::endl;
}

// PROOF 2: UI is not blocked - poll() returns immediately
TEST_F(UvRunnerProofTest, PollDoesNotBlock) {
    UvRunner runner;

    // Start a command that takes some time (pip list)
    ASSERT_TRUE(runner.start({"pip", "list", "--python",
                              PackageManager::instance().venv_python().string()}));

    // Measure poll() call duration - it should return immediately
    std::vector<long> poll_durations_us;
    int poll_count = 0;
    const int max_polls = 1000;

    auto start_time = steady_clock::now();

    while (runner.poll() && poll_count < max_polls) {
        auto poll_start = steady_clock::now();
        runner.poll(); // Extra poll to measure
        auto poll_end = steady_clock::now();

        long duration_us = duration_cast<microseconds>(poll_end - poll_start).count();
        poll_durations_us.push_back(duration_us);

        poll_count++;
        std::this_thread::sleep_for(milliseconds(1));
    }

    auto total_time = steady_clock::now() - start_time;

    // Calculate statistics
    long max_poll_us = 0;
    long total_poll_us = 0;
    for (long d : poll_durations_us) {
        max_poll_us = std::max(max_poll_us, d);
        total_poll_us += d;
    }
    long avg_poll_us = poll_durations_us.empty() ? 0 : total_poll_us / poll_durations_us.size();

    std::cout << "\n=== PROOF: poll() does not block ===" << std::endl;
    std::cout << "Total polls: " << poll_durations_us.size() << std::endl;
    std::cout << "Average poll duration: " << avg_poll_us << " us" << std::endl;
    std::cout << "Max poll duration: " << max_poll_us << " us" << std::endl;
    std::cout << "Total elapsed: " << duration_cast<milliseconds>(total_time).count() << " ms"
              << std::endl;

    // poll() should complete in less than 1ms (1000us) on average
    // This proves it's non-blocking
    EXPECT_LT(avg_poll_us, 1000) << "poll() should average under 1ms";
    EXPECT_LT(max_poll_us, 10000) << "poll() should never take more than 10ms";
}

// PROOF 3: Can do work while UV is running
TEST_F(UvRunnerProofTest, CanDoWorkWhileUvRuns) {
    UvRunner runner;

    std::atomic<int> output_count{0};
    runner.set_output_callback([&](const std::string&, bool, bool) { output_count++; });

    std::atomic<bool> completed{false};
    runner.set_completion_callback([&](bool, int) { completed = true; });

    // Start pip list (takes some time)
    ASSERT_TRUE(runner.start({"pip", "list", "--python",
                              PackageManager::instance().venv_python().string()}));

    // Simulate "UI work" - increment a counter while polling
    std::atomic<int> work_done{0};
    auto start = steady_clock::now();

    while (!completed) {
        // Poll UV (non-blocking)
        runner.poll();

        // Do "UI work" - this simulates rendering frames
        work_done++;

        // Sleep to simulate 60 FPS frame time
        std::this_thread::sleep_for(microseconds(16667)); // ~60 FPS
    }

    auto elapsed = steady_clock::now() - start;
    auto elapsed_ms = duration_cast<milliseconds>(elapsed).count();

    std::cout << "\n=== PROOF: Can do work while UV runs ===" << std::endl;
    std::cout << "UV operation took: " << elapsed_ms << " ms" << std::endl;
    std::cout << "Work iterations (simulated frames): " << work_done.load() << std::endl;
    std::cout << "Output lines received: " << output_count.load() << std::endl;
    std::cout << "Effective FPS: " << (work_done.load() * 1000.0 / elapsed_ms) << std::endl;

    // We should have done significant work while UV was running
    EXPECT_GT(work_done.load(), 0) << "Should have done work while UV ran";

    // If operation took > 100ms, we should have done multiple "frames"
    if (elapsed_ms > 100) {
        EXPECT_GT(work_done.load(), 5) << "Should maintain reasonable frame rate";
    }
}

// PROOF 4: Output lines are complete (not corrupted)
TEST_F(UvRunnerProofTest, OutputLinesAreComplete) {
    UvRunner runner;

    std::vector<std::string> lines;
    runner.set_output_callback([&](const std::string& line, bool, bool) { lines.push_back(line); });

    bool completed = false;
    runner.set_completion_callback([&](bool, int) { completed = true; });

    // Run pip list which outputs multiple lines
    ASSERT_TRUE(runner.start({"pip", "list", "--python",
                              PackageManager::instance().venv_python().string()}));

    while (runner.poll()) {
        std::this_thread::sleep_for(milliseconds(10));
    }

    ASSERT_TRUE(completed);

    std::cout << "\n=== PROOF: Output lines are complete ===" << std::endl;
    std::cout << "Total lines: " << lines.size() << std::endl;

    // Verify no line has embedded newlines (would indicate incorrect parsing)
    for (size_t i = 0; i < lines.size(); i++) {
        EXPECT_EQ(lines[i].find('\n'), std::string::npos)
            << "Line " << i << " contains embedded newline: " << lines[i];
        EXPECT_EQ(lines[i].find('\r'), std::string::npos)
            << "Line " << i << " contains embedded CR: " << lines[i];
    }

    // Print first few lines for verification
    std::cout << "First 10 lines:" << std::endl;
    for (size_t i = 0; i < std::min(lines.size(), size_t(10)); i++) {
        std::cout << "  [" << i << "] " << lines[i] << std::endl;
    }
}

// PROOF 5: Real package install with progress
TEST_F(UvRunnerProofTest, RealPackageInstallShowsProgress) {
    // Ensure venv exists
    ASSERT_TRUE(PackageManager::instance().ensure_venv());

    UvRunner runner;

    std::vector<std::string> output_lines;
    auto start_time = steady_clock::now();
    std::vector<long> timestamps_ms;

    runner.set_output_callback([&](const std::string& line, bool is_error, bool is_line_update) {
        auto now = steady_clock::now();
        long ts = duration_cast<milliseconds>(now - start_time).count();
        timestamps_ms.push_back(ts);
        output_lines.push_back(line);
        std::cout << "[" << ts << "ms] " << (is_error ? "ERR: " : "")
                  << (is_line_update ? "(update) " : "") << line << std::endl;
    });

    bool completed = false;
    bool success = false;
    int exit_code = -1;
    runner.set_completion_callback([&](bool s, int ec) {
        completed = true;
        success = s;
        exit_code = ec;
    });

    // Install a small package (pip is usually fast to install/already installed)
    std::cout << "\n=== Installing 'pip' package (should be quick) ===" << std::endl;
    ASSERT_TRUE(runner.start({"pip", "install", "pip", "--python",
                              PackageManager::instance().venv_python().string()}));

    // Poll and track how many frames we could render
    int frame_count = 0;
    while (runner.poll()) {
        frame_count++;
        std::this_thread::sleep_for(milliseconds(16)); // 60 FPS
    }

    auto total_time = duration_cast<milliseconds>(steady_clock::now() - start_time).count();

    std::cout << "\n=== PROOF: Real install with progress ===" << std::endl;
    std::cout << "Total time: " << total_time << " ms" << std::endl;
    std::cout << "Output lines: " << output_lines.size() << std::endl;
    std::cout << "Frames rendered: " << frame_count << std::endl;
    std::cout << "Success: " << (success ? "yes" : "no") << std::endl;
    std::cout << "Exit code: " << exit_code << std::endl;

    // Verify we got output
    EXPECT_GT(output_lines.size(), 0) << "Should have received output lines";

    // Verify output came in over time (not all at once at the end)
    if (timestamps_ms.size() > 1 && total_time > 100) {
        // Check that not all output arrived in the last 10% of time
        int late_outputs = 0;
        for (long ts : timestamps_ms) {
            if (ts > total_time * 0.9) {
                late_outputs++;
            }
        }
        float late_ratio = float(late_outputs) / timestamps_ms.size();
        std::cout << "Output timing: " << (late_ratio * 100) << "% arrived in last 10% of time"
                  << std::endl;

        // If more than 90% of output arrived in last 10% of time, output wasn't streaming
        EXPECT_LT(late_ratio, 0.9) << "Output should stream progressively, not arrive all at end";
    }
}
