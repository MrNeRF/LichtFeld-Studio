/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/host_metrics.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <thread>

TEST(HostMetrics, FirstSampleHasNoCpuDelta) {
    const auto sample = lfs::core::host_metrics::sample();
    EXPECT_TRUE(sample.ram_valid);
    EXPECT_LT(sample.process_cpu_percent, 0.f);
}

TEST(HostMetrics, CpuAndPerCoreBecomeValidAfterSecondSample) {
    (void)lfs::core::host_metrics::sample();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    const auto sample = lfs::core::host_metrics::sample();
    EXPECT_GE(sample.process_cpu_percent, 0.f);
    EXPECT_LE(sample.process_cpu_percent, 100.f);
    EXPECT_EQ(sample.per_core_cpu_percent.size(), std::thread::hardware_concurrency());
}

TEST(HostMetrics, MemAvailableDefinesUsedMemory) {
    const auto sample = lfs::core::host_metrics::sample();
    ASSERT_TRUE(sample.ram_valid);
    EXPECT_GT(sample.system_total_bytes, 0u);
    EXPECT_LT(sample.system_used_bytes, sample.system_total_bytes);
}

TEST(HostMetrics, SamplingStaysWithinBudget) {
    const auto started = std::chrono::steady_clock::now();
    for (int i = 0; i < 100; ++i)
        (void)lfs::core::host_metrics::sample();
    const auto elapsed = std::chrono::steady_clock::now() - started;
    EXPECT_LT(std::chrono::duration_cast<std::chrono::seconds>(elapsed).count(), 10);
}
