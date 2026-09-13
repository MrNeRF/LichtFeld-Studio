/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/crash_handler.hpp"
#include "core/environment.hpp"
#include "core/logger.hpp"
#include "core/pinned_memory_allocator.hpp"
#include "core/tensor_backend.hpp"
#include "tensor_backend_trace_listener.hpp"
#include <gtest/gtest.h>
#include <iostream>
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

    lfs::core::TensorBackendOptions options;
    auto backend = lfs::core::GpuBackend::CUDA;
    std::string trace_path;
    int remaining = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg.starts_with("--tensor-backend=")) {
            const auto value = arg.substr(17);
            if (value != "cuda" && value != "vulkan") {
                std::cerr << "Tensor backend must be cuda or vulkan\n";
                return 2;
            }
            backend = value == "vulkan" ? lfs::core::GpuBackend::Vulkan : lfs::core::GpuBackend::CUDA;
        } else if (arg.starts_with("--tensor-validation=")) {
            const auto value = arg.substr(20);
            if (value != "off" && value != "api" && value != "sync") {
                std::cerr << "Tensor validation must be off, api, or sync\n";
                return 2;
            }
            options.vulkan_validation = value == "sync" ? 2 : value == "api" ? 1
                                                                             : 0;
        } else if (arg.starts_with("--tensor-device=")) {
            options.vulkan_device = arg.substr(16);
        } else if (arg == "--tensor-fp32-half") {
            options.force_fp32_half = true;
        } else if (arg == "--tensor-no-atomic-float") {
            options.force_no_atomic_float = true;
        } else if (arg.starts_with("--tensor-facade-trace=")) {
            trace_path = arg.substr(22);
        } else {
            argv[remaining++] = argv[i];
        }
    }
    argc = remaining;
    argv[argc] = nullptr;
    if (!lfs::core::set_tensor_backend_options(options) || !lfs::core::set_default_gpu_backend(backend))
        return 2;
    ::testing::InitGoogleTest(&argc, argv);
    if (!trace_path.empty()) {
        ::testing::UnitTest::GetInstance()->listeners().Append(
            new lfs::testing::FacadeTraceListener(trace_path));
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
