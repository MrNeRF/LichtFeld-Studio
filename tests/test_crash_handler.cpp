/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include "core/crash_handler.hpp"

#include "core/environment.hpp"
#include "core/failure_report.hpp"
#include "core/path_utils.hpp"
#include "core/user_paths.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
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

    class ScopedLfsHome {
    public:
        explicit ScopedLfsHome(const std::string& value)
            : previous_(lfs::core::environment::value("LFS_HOME")) {
            EXPECT_TRUE(lfs::core::environment::set_value("LFS_HOME", value));
        }

        ~ScopedLfsHome() {
            if (previous_) {
                (void)lfs::core::environment::set_value("LFS_HOME", *previous_);
                return;
            }
#ifdef _WIN32
            (void)_wputenv_s(L"LFS_HOME", L"");
#else
            (void)::unsetenv("LFS_HOME");
#endif
        }

        ScopedLfsHome(const ScopedLfsHome&) = delete;
        ScopedLfsHome& operator=(const ScopedLfsHome&) = delete;

    private:
        std::optional<std::string> previous_;
    };

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

TEST(CrashHandlerTest, InstallationReportsUnicodeCrashLogPathAsUtf8) {
    const auto root = std::filesystem::temp_directory_path() /
                      lfs::core::utf8_to_path("lfs_crash_診断");
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    ASSERT_FALSE(ec) << ec.message();
    const ScopedLfsHome home(lfs::core::path_to_utf8(root));

    EXPECT_EXIT(([] {
                    lfs::core::install_crash_handlers();
                    lfs::core::flush_and_exit(0);
                }()),
                ::testing::ExitedWithCode(0), "lfs_crash_診断");

    std::filesystem::remove_all(root, ec);
    EXPECT_FALSE(ec) << ec.message();
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
                    const auto paths = lfs::core::UserPaths::resolve();
                    if (!paths)
                        lfs::core::flush_and_exit(2);
                    std::filesystem::path path;
                    for (const auto& entry : std::filesystem::directory_iterator(paths->logDir())) {
                        const auto name = entry.path().filename().string();
                        if (name.starts_with("lichtfeld-studio-crash-") && name.ends_with("-" + std::to_string(pid) + ".log"))
                            path = entry.path();
                    }
                    std::ifstream file(path);
                    const std::string text((std::istreambuf_iterator<char>(file)), {});
                    const auto first = text.find("injected driver reset");
                    const bool valid = first != std::string::npos &&
                                       text.find("injected driver reset", first + 1) == std::string::npos;
                    file.close();
                    std::error_code ec;
                    std::filesystem::remove(path, ec);
                    lfs::core::flush_and_exit(valid ? 0 : 1);
                }()),
                ::testing::ExitedWithCode(0), "");
}

TEST(CrashHandlerTest, CleanRunCreatesNoCrashFile) {
    EXPECT_EXIT(([] {
                    lfs::core::install_crash_handlers();
                    const auto paths = lfs::core::UserPaths::resolve();
                    if (!paths)
                        lfs::core::flush_and_exit(2);
                    for (const auto& entry : std::filesystem::directory_iterator(paths->logDir())) {
                        const auto name = entry.path().filename().string();
                        if (name.starts_with("lichtfeld-studio-crash-") && name.ends_with("-" + std::to_string(current_process_id()) + ".log"))
                            lfs::core::flush_and_exit(1);
                    }
                    lfs::core::flush_and_exit(0);
                }()),
                ::testing::ExitedWithCode(0), "");
}
