/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include "core/crash_handler.hpp"

#include <stdexcept>

namespace {

    constexpr int FIREWALL_EXIT_CODE = 70; // EX_SOFTWARE, frozen contract

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
