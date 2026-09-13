/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include "core/crash_handler.hpp"

#include "core/failure_report.hpp"
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#ifdef _WIN32
#include <process.h>
#else
#include <csignal>
#include <unistd.h>
#endif

namespace {

    constexpr int FIREWALL_EXIT_CODE = 70; // EX_SOFTWARE, frozen contract

    auto current_process_id() {
#ifdef _WIN32
        return _getpid();
#else
        return getpid();
#endif
    }

} // namespace

TEST(CrashHandlerTest, FlushAndExitExitsWithRequestedCodeZero) {
    EXPECT_EXIT(lfs::core::flush_and_exit(0), ::testing::ExitedWithCode(0), "");
}

TEST(CrashHandlerTest, FlushAndExitExitsWithRequestedCodeSeventy) {
    EXPECT_EXIT(lfs::core::flush_and_exit(70), ::testing::ExitedWithCode(70), "");
}

TEST(CrashHandlerTest, ExceptionFirewallReturnsFirewallCodeForStdException) {
    const int result = lfs::core::run_with_exception_firewall([]() -> int {
        throw std::runtime_error("boom");
    });
    EXPECT_EQ(result, FIREWALL_EXIT_CODE);
}

TEST(CrashHandlerTest, ExceptionFirewallReturnsFirewallCodeForNonStdException) {
    const int result = lfs::core::run_with_exception_firewall([]() -> int {
        throw 42;
    });
    EXPECT_EQ(result, FIREWALL_EXIT_CODE);
}

TEST(CrashHandlerTest, ExceptionFirewallPropagatesNormalReturnValue) {
    const int result = lfs::core::run_with_exception_firewall([]() -> int {
        return 7;
    });
    EXPECT_EQ(result, 7);
}

TEST(CrashHandlerTest, HandledGpuFailureIsSavedOutsideRotatingLogs) {
    EXPECT_EXIT(([] {
                    lfs::core::install_crash_handlers();
                    lfs::core::reset_failure_report_dedup_for_testing();
                    const lfs::core::FailureReport report{
                        .family = "CUDA",
                        .contract = "test handled device failure",
                        .message = "injected driver reset",
                        .location = LFS_SOURCE_SITE_CURRENT(),
                        .capture_stack = false};
                    for (int i = 0; i < 200; ++i)
                        lfs::core::emit_failure_report(report, lfs::core::FailureReportSeverity::Error);
                    const auto pid = current_process_id();
                    const auto path = std::filesystem::temp_directory_path() /
                                      ("lichtfeld-studio-crash-" + std::to_string(pid) + ".log");
                    std::ifstream file(path);
                    const std::string text((std::istreambuf_iterator<char>(file)), {});
                    const auto first = text.find("injected driver reset");
                    const bool valid = text.find("diagnostics initialized") != std::string::npos &&
                                       first != std::string::npos &&
                                       text.find("injected driver reset", first + 1) == std::string::npos;
                    file.close();
                    std::error_code ec;
                    std::filesystem::remove(path, ec);
                    lfs::core::flush_and_exit(valid ? 0 : 1);
                }()),
                ::testing::ExitedWithCode(0), "");
}
