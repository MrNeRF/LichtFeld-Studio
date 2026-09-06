/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/crash_handler.hpp"
#include "core/environment.hpp"
#include "core/logger.hpp"
#include "core/pinned_memory_allocator.hpp"
#include "tensor_backend_trace_listener.hpp"
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

    // LFS_TENSOR_FACADE_TRACE=path records the backend facade entries each test
    // executes (one JSON line per test) for the tensor-backend manifest, and stops
    // the run naming the test that leaves a sticky CUDA error behind.
    if (const auto trace_path = lfs::core::environment::value("LFS_TENSOR_FACADE_TRACE")) {
        ::testing::UnitTest::GetInstance()->listeners().Append(
            new lfs::testing::FacadeTraceListener(std::string(*trace_path)));
    }

    // Pre-warm pinned memory cache for fast CPU-GPU transfers
    // This eliminates cold-start penalties (e.g., 23.8ms for 4K allocations)
    lfs::core::PinnedMemoryAllocator::instance().prewarm();

    const int result = RUN_ALL_TESTS();

    // ordered GPU release (TLS caches, PPISP statics, mirror mults, …)
    // then pool/arena/pinned shutdown while CUDA is still healthy. After that,
    // do NOT return into C++ static/TLS destruction — those dtors re-enter
    // freed pool storage / half-dead CUDA and produce SIGSEGV (exit 139) or
    // host double-free (exit 134). Same contract as the app binary: teardown
    // then process-terminate without running remaining static destructors.
    lfs::core::teardown_gpu_before_exit();
    lfs::core::flush_and_exit(result);
}
