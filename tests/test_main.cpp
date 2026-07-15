/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/environment.hpp"
#include "core/logger.hpp"
#include "core/pinned_memory_allocator.hpp"
#include "core/tensor.hpp"
#include <gtest/gtest.h>
#include <string>

int main(int argc, char** argv) {
    // Initialize loggers
    auto log_level = lfs::core::LogLevel::Info;
    if (const auto env = lfs::core::environment::value("LFS_LOG_LEVEL")) {
        std::string level(*env);
        if (level == "trace")
            log_level = lfs::core::LogLevel::Trace;
        else if (level == "debug")
            log_level = lfs::core::LogLevel::Debug;
        else if (level == "info")
            log_level = lfs::core::LogLevel::Info;
        else if (level == "perf")
            log_level = lfs::core::LogLevel::Performance;
        else if (level == "warn")
            log_level = lfs::core::LogLevel::Warn;
        else if (level == "error")
            log_level = lfs::core::LogLevel::Error;
    }
    lfs::core::Logger::get().init(log_level);

    ::testing::InitGoogleTest(&argc, argv);

    // Pre-warm pinned memory cache for fast CPU-GPU transfers
    // This eliminates cold-start penalties (e.g., 23.8ms for 4K allocations)
    lfs::core::PinnedMemoryAllocator::instance().prewarm();

    const int result = RUN_ALL_TESTS();

    // Preserve singleton dependency order: pinned blocks return pooled events
    // before the CUDA memory pool shuts that event pool down.
    lfs::core::PinnedMemoryAllocator::instance().shutdown();
    lfs::core::Tensor::shutdown_memory_pool();
    return result;
}
