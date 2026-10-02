/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "diagnostics/vram_owner_model.hpp"
#include "gui/gpu_memory_query.hpp"

#include <array>
#include <gtest/gtest.h>
#include <limits>

namespace {
    constexpr std::size_t MiB = 1024ull * 1024ull;

    TEST(GpuMemoryQuery, DriverProcessAndPhysicalDevice) {
        const auto memory = lfs::vis::gui::selectGpuMemory(
            381 * MiB, 381 * MiB, 0, 684 * MiB, 24074 * MiB,
            684 * MiB, 24564 * MiB);
        EXPECT_EQ(memory.process_used, 381 * MiB);
        EXPECT_EQ(memory.total_used, 684 * MiB);
        EXPECT_EQ(memory.total, 24564 * MiB);
        EXPECT_TRUE(memory.process_valid);
        EXPECT_FALSE(memory.device_estimated);
        EXPECT_EQ(lfs::vis::gui::formatGpuGiB(memory.total), "23.99");
    }

    TEST(GpuMemoryQuery, ParsesProcessRecordsAndUnavailableSentinel) {
        const std::array<lfs::vis::gui::GpuProcessUsage, 4> records{{
            {42, 381 * MiB},
            {7, 6054 * MiB},
            {42, std::numeric_limits<unsigned long long>::max()},
            {42, 380 * MiB},
        }};
        EXPECT_EQ(lfs::vis::gui::parseGpuProcessBytes(42, records), 381 * MiB);
        EXPECT_EQ(lfs::vis::gui::parseGpuProcessBytes(99, records), 0);
    }

    TEST(GpuMemoryQuery, FallbacksAreMarked) {
        const auto windows = lfs::vis::gui::selectGpuMemory(
            0, 0, 512 * MiB, 768 * MiB, 24074 * MiB, 0, 0);
        EXPECT_EQ(windows.process_used, 512 * MiB);
        EXPECT_TRUE(windows.process_valid);
        EXPECT_TRUE(windows.device_estimated);

        const auto cuda_only = lfs::vis::gui::selectGpuMemory(
            0, 0, 0, 768 * MiB, 24074 * MiB, 0, 0);
        EXPECT_EQ(cuda_only.process_used, 0u);
        EXPECT_FALSE(cuda_only.process_valid);
        EXPECT_TRUE(cuda_only.device_estimated);
        EXPECT_EQ(lfs::vis::gui::formatGpuGiB(768 * MiB), "0.75");
    }

    TEST(GpuMemoryQuery, OversubscriptionIsExplicitAndDeviceUsageIsPhysical) {
        const auto memory = lfs::vis::gui::selectGpuMemory(
            0, 0, 17 * 1024 * MiB, 17 * 1024 * MiB, 10 * 1024 * MiB,
            17 * 1024 * MiB, 10 * 1024 * MiB, 9 * 1024 * MiB, true);
        EXPECT_TRUE(memory.process_valid);
        EXPECT_TRUE(memory.process_over_budget);
        EXPECT_EQ(memory.process_used, 17 * 1024 * MiB);
        EXPECT_EQ(memory.process_budget, 9 * 1024 * MiB);
        EXPECT_EQ(memory.total_used, memory.total);
        EXPECT_EQ(memory.total, 10 * 1024 * MiB);
    }

    TEST(GpuMemoryQuery, ZeroUsageAndZeroBudgetRemainValidDxgiSamples) {
        const auto idle = lfs::vis::gui::selectGpuMemory(0, 0, 0, 500, 1000, 500, 1000, 800, true);
        EXPECT_TRUE(idle.process_valid);
        EXPECT_EQ(idle.process_used, 0u);
        EXPECT_FALSE(idle.process_over_budget);
        const auto no_budget = lfs::vis::gui::selectGpuMemory(0, 0, 100, 500, 1000, 500, 1000, 0, true);
        EXPECT_TRUE(no_budget.process_over_budget);
    }

    TEST(GpuMemoryQuery, DisplayCategoriesCloseWithoutNegativeRemainder) {
        constexpr std::int64_t signed_mib = 1024ll * 1024ll;
        lfs::diagnostics::VramOwnerBreakdown captured;
        captured.process_valid = true;
        captured.process_bytes = 3620 * MiB;
        captured.bytes = {700 * signed_mib, 800 * signed_mib, 1000 * signed_mib,
                          300 * signed_mib, 200 * signed_mib, 250 * signed_mib,
                          500 * signed_mib, 200 * signed_mib, 100 * signed_mib,
                          -430 * signed_mib};
        const auto display = lfs::diagnostics::displayVramOwnerBytes(captured);
        std::int64_t sum = 0;
        for (const auto bytes : display) {
            EXPECT_GE(bytes, 0);
            sum += bytes;
        }
        EXPECT_EQ(sum, 3620 * static_cast<std::int64_t>(MiB));
        EXPECT_EQ(display.back(), 0);
    }

    TEST(GpuMemoryQuery, UnifiedMemoryAvailabilityIsCappedByTheGpuAndTheHost) {
        constexpr std::size_t GiB = std::size_t{1} << 30;
        using lfs::vis::gui::selectUnifiedGpuMemory;

        // Other GPU clients bind first: 28 GiB working set, 10 GiB allocated.
        auto status = selectUnifiedGpuMemory(4 * GiB, 28 * GiB, 10 * GiB, 20 * GiB);
        EXPECT_TRUE(status.unified_memory);
        EXPECT_TRUE(status.process_valid);
        EXPECT_FALSE(status.uses_process_budget);
        EXPECT_EQ(status.process_used, 4 * GiB);
        EXPECT_EQ(status.total, 28 * GiB);
        EXPECT_EQ(status.total_used, 10 * GiB);

        // Host memory held by other processes binds first.
        status = selectUnifiedGpuMemory(4 * GiB, 28 * GiB, 10 * GiB, 3 * GiB);
        EXPECT_EQ(status.total_used, 25 * GiB);

        // Allocation beyond the working set saturates.
        status = selectUnifiedGpuMemory(4 * GiB, 28 * GiB, 40 * GiB, 20 * GiB);
        EXPECT_EQ(status.total_used, 28 * GiB);

        // Device usage never reads below the process's own usage.
        status = selectUnifiedGpuMemory(6 * GiB, 28 * GiB, 2 * GiB, 30 * GiB);
        EXPECT_EQ(status.total_used, 6 * GiB);
    }

} // namespace
