/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "core/pinned_memory_allocator.hpp"
#include "core/tensor.hpp"
#include <c10/cuda/CUDACachingAllocator.h>
#include <cstdlib>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

// Custom event listener to clean GPU memory before AND after each test
class MemoryCleanupListener : public ::testing::EmptyTestEventListener {
private:
    void aggressive_cleanup() {
        cudaSetDevice(0);

        // Synchronize and check for errors
        cudaError_t sync_err = cudaDeviceSynchronize();
        cudaError_t last_err = cudaGetLastError();

        // If there are CUDA errors, log and clear them
        if (sync_err != cudaSuccess || last_err != cudaSuccess) {
            if (sync_err != cudaSuccess) {
                std::cerr << "[CLEANUP] CUDA sync error: " << cudaGetErrorString(sync_err) << std::endl;
            }
            if (last_err != cudaSuccess) {
                std::cerr << "[CLEANUP] CUDA error: " << cudaGetErrorString(last_err) << std::endl;
            }
            // Clear the error
            cudaGetLastError();
        }

        // Empty PyTorch's cache
        try {
            c10::cuda::CUDACachingAllocator::emptyCache();
        } catch (...) {
            std::cerr << "[CLEANUP] Exception while emptying PyTorch cache" << std::endl;
        }

        // Trim custom CUDA memory pool
        try {
            lfs::core::Tensor::trim_memory_pool();
        } catch (...) {
            std::cerr << "[CLEANUP] Exception while trimming custom pool" << std::endl;
        }

        // Final sync
        cudaDeviceSynchronize();
        cudaGetLastError(); // Clear any remaining errors
    }

public:
    void OnTestStart(const ::testing::TestInfo& /*test_info*/) override {
        aggressive_cleanup();
    }

    void OnTestEnd(const ::testing::TestInfo& /*test_info*/) override {
        aggressive_cleanup();
    }
};

int main(int argc, char** argv) {
    // Initialize loggers - check LOG_LEVEL env var
    auto log_level = lfs::core::LogLevel::Info;
    if (const char* env = std::getenv("LOG_LEVEL")) {
        std::string level(env);
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

    // Add custom listener to clean memory before each test
    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new MemoryCleanupListener);

    // Pre-warm pinned memory cache for fast CPU-GPU transfers
    // This eliminates cold-start penalties (e.g., 23.8ms for 4K allocations)
    lfs::core::PinnedMemoryAllocator::instance().prewarm();

    return RUN_ALL_TESTS();
}
