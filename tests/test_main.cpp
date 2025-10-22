/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "core_new/pinned_memory_allocator.hpp"
#include <gtest/gtest.h>

int main(int argc, char** argv) {
    // Initialize logger with Info level
    gs::core::Logger::get().init(gs::core::LogLevel::Info);

    ::testing::InitGoogleTest(&argc, argv);

    // Pre-warm pinned memory cache for fast CPU-GPU transfers
    // This eliminates cold-start penalties (e.g., 23.8ms for 4K allocations)
    lfs::core::PinnedMemoryAllocator::instance().prewarm();

    return RUN_ALL_TESTS();
}
