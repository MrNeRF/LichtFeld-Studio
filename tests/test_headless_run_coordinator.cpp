/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/headless_run_coordinator.hpp"

#include <chrono>
#include <csignal>
#include <gtest/gtest.h>
#include <thread>

namespace {

    TEST(HeadlessRunCoordinatorTest, SignalHandlerRequestsStopFromNormalThread) {
        lfs::app::HeadlessRunCoordinator coordinator;

        ASSERT_EQ(std::raise(SIGINT), 0);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
        while (!coordinator.stop_token().stop_requested() &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        EXPECT_TRUE(coordinator.stop_token().stop_requested());
        EXPECT_EQ(coordinator.received_signal(), SIGINT);
        EXPECT_EQ(coordinator.interrupted_exit_code(), 128 + SIGINT);
    }

} // namespace
