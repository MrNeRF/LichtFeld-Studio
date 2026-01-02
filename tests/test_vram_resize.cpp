/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "core/tensor.hpp"
#include "core/logger.hpp"
#include "core/tensor/internal/memory_pool.hpp"

using namespace lfs::core;

namespace {

// Get current GPU memory usage in MB
std::pair<size_t, size_t> getGPUMemoryMB() {
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    size_t used_bytes = total_bytes - free_bytes;
    return {used_bytes / (1024 * 1024), total_bytes / (1024 * 1024)};
}

// Force CUDA memory cleanup - including memory pool trim
void forceMemoryCleanup() {
    cudaDeviceSynchronize();
    // Trim the CUDA memory pool to release cached memory back to OS
    CudaMemoryPool::instance().trim_cached_memory();
    cudaDeviceSynchronize();
}

} // namespace

class VRAMResizeTest : public ::testing::Test {
protected:
    void SetUp() override {
        forceMemoryCleanup();
        auto [used, total] = getGPUMemoryMB();
        baseline_vram_mb_ = used;
        LOG_INFO("Baseline VRAM: {} MB / {} MB", used, total);
    }

    void TearDown() override {
        forceMemoryCleanup();
        auto [used, total] = getGPUMemoryMB();
        LOG_INFO("Final VRAM: {} MB (delta from baseline: {} MB)",
                 used, static_cast<int>(used) - static_cast<int>(baseline_vram_mb_));
    }

    size_t baseline_vram_mb_ = 0;
};

// Test: Verify tensor allocation and deallocation releases VRAM
TEST_F(VRAMResizeTest, TensorAllocationReleasesMemory) {
    auto [initial_used, total] = getGPUMemoryMB();
    LOG_INFO("Initial VRAM: {} MB", initial_used);

    // Allocate a large tensor (simulating a 4K render buffer: 3840x2160x4 floats = ~127MB)
    // Calculation: 2160 * 3840 * 4 * sizeof(float) = 132,710,400 bytes ≈ 127 MB
    {
        Tensor large_tensor = Tensor::zeros({2160, 3840, 4}, Device::CUDA, DataType::Float32);
        forceMemoryCleanup();

        auto [with_tensor, _] = getGPUMemoryMB();
        size_t tensor_size_mb = with_tensor - initial_used;
        LOG_INFO("With 4K tensor: {} MB (delta: {} MB)", with_tensor, tensor_size_mb);

        // 4K RGBA float32 should be ~127MB (allow some alignment overhead)
        EXPECT_GE(tensor_size_mb, 100) << "4K tensor should use at least 100MB";
        EXPECT_LE(tensor_size_mb, 160) << "4K tensor should use at most 160MB";
    }

    // Tensor goes out of scope - memory should be released
    forceMemoryCleanup();

    auto [after_release, __] = getGPUMemoryMB();
    int delta = static_cast<int>(after_release) - static_cast<int>(initial_used);
    LOG_INFO("After release: {} MB (delta from initial: {} MB)", after_release, delta);

    // Memory should be back to near initial (within 5MB tolerance)
    EXPECT_LE(std::abs(delta), 5) << "Memory should be released after tensor destruction";
}

// Test: Simulate resize pattern - grow then shrink
TEST_F(VRAMResizeTest, ResizePatternGrowShrink) {
    auto [initial_used, total] = getGPUMemoryMB();
    LOG_INFO("Initial VRAM: {} MB", initial_used);

    std::vector<size_t> vram_history;
    vram_history.push_back(initial_used);

    // Simulate growing window - allocate progressively larger tensors
    std::vector<std::pair<int, int>> sizes = {
        {720, 1280},   // 720p
        {1080, 1920},  // 1080p
        {1440, 2560},  // 1440p
        {2160, 3840},  // 4K
    };

    Tensor current_image;
    Tensor current_depth;

    LOG_INFO("=== Growing phase ===");
    for (const auto& [h, w] : sizes) {
        // Simulate rendering: allocate new tensors for this size
        current_image = Tensor::zeros({3, h, w}, Device::CUDA, DataType::Float32);
        current_depth = Tensor::zeros({1, h, w}, Device::CUDA, DataType::Float32);

        forceMemoryCleanup();
        auto [used, _] = getGPUMemoryMB();
        vram_history.push_back(used);
        LOG_INFO("  {}x{}: {} MB (delta: {} MB)",
                 w, h, used, static_cast<int>(used) - static_cast<int>(initial_used));
    }

    size_t peak_vram = vram_history.back();
    LOG_INFO("Peak VRAM at 4K: {} MB", peak_vram);

    LOG_INFO("=== Shrinking phase ===");
    // Shrink back down
    for (auto it = sizes.rbegin(); it != sizes.rend(); ++it) {
        const auto& [h, w] = *it;
        current_image = Tensor::zeros({3, h, w}, Device::CUDA, DataType::Float32);
        current_depth = Tensor::zeros({1, h, w}, Device::CUDA, DataType::Float32);

        forceMemoryCleanup();
        auto [used, _] = getGPUMemoryMB();
        vram_history.push_back(used);
        LOG_INFO("  {}x{}: {} MB", w, h, used);
    }

    // Clear tensors
    current_image = Tensor();
    current_depth = Tensor();
    forceMemoryCleanup();

    auto [final_used, _] = getGPUMemoryMB();
    int final_delta = static_cast<int>(final_used) - static_cast<int>(initial_used);
    LOG_INFO("Final VRAM after clear: {} MB (delta: {} MB)", final_used, final_delta);

    // After clearing, memory should return close to initial
    EXPECT_LE(std::abs(final_delta), 10) << "Memory should be released after clearing tensors";
}

// Test: Tensor reuse vs reallocation
TEST_F(VRAMResizeTest, TensorReuseVsReallocation) {
    auto [initial_used, total] = getGPUMemoryMB();

    LOG_INFO("=== Testing reallocation pattern (BAD) ===");
    size_t peak_realloc = initial_used;

    // Simulate bad pattern: allocate new tensors every frame without reusing
    for (int i = 0; i < 10; ++i) {
        int h = 1080 + (i * 10);  // Slightly different size each time
        int w = 1920 + (i * 10);

        Tensor image = Tensor::zeros({3, h, w}, Device::CUDA, DataType::Float32);
        Tensor depth = Tensor::zeros({1, h, w}, Device::CUDA, DataType::Float32);

        forceMemoryCleanup();
        auto [used, _] = getGPUMemoryMB();
        peak_realloc = std::max(peak_realloc, used);

        // Tensors go out of scope, should be freed
    }

    forceMemoryCleanup();
    auto [after_realloc, _1] = getGPUMemoryMB();
    LOG_INFO("After realloc pattern: {} MB (peak: {} MB)", after_realloc, peak_realloc);

    // Reset
    forceMemoryCleanup();
    auto [reset_used, _2] = getGPUMemoryMB();

    LOG_INFO("=== Testing reuse pattern (GOOD) ===");
    size_t peak_reuse = reset_used;

    // Pre-allocate with over-allocation
    int max_h = 1080 + 100;
    int max_w = 1920 + 100;
    Tensor reused_image = Tensor::zeros({3, max_h, max_w}, Device::CUDA, DataType::Float32);
    Tensor reused_depth = Tensor::zeros({1, max_h, max_w}, Device::CUDA, DataType::Float32);

    forceMemoryCleanup();
    auto [with_prealloc, _3] = getGPUMemoryMB();
    LOG_INFO("With pre-allocated tensors: {} MB", with_prealloc);

    // Simulate using smaller regions (reusing same memory)
    for (int i = 0; i < 10; ++i) {
        int h = 1080 + (i * 10);
        int w = 1920 + (i * 10);

        // Just use a view/slice of the pre-allocated tensors
        // In real code, we'd render to a subregion
        auto image_view = reused_image.slice(1, 0, h).slice(2, 0, w);
        auto depth_view = reused_depth.slice(1, 0, h).slice(2, 0, w);

        forceMemoryCleanup();
        auto [used, _] = getGPUMemoryMB();
        peak_reuse = std::max(peak_reuse, used);
    }

    LOG_INFO("Peak with reuse pattern: {} MB", peak_reuse);

    // Reuse pattern should not grow memory beyond initial pre-allocation
    EXPECT_LE(peak_reuse, with_prealloc + 5) << "Reuse pattern should not allocate more memory";
}

// Test: Measure actual memory per resolution
TEST_F(VRAMResizeTest, MemoryPerResolution) {
    LOG_INFO("=== Memory usage per resolution ===");

    struct Resolution {
        const char* name;
        int width, height;
    };

    std::vector<Resolution> resolutions = {
        {"720p", 1280, 720},
        {"1080p", 1920, 1080},
        {"1440p", 2560, 1440},
        {"4K", 3840, 2160},
        {"5K", 5120, 2880},
        {"8K", 7680, 4320},
    };

    for (const auto& res : resolutions) {
        forceMemoryCleanup();
        auto [before, _1] = getGPUMemoryMB();

        // Allocate typical render buffers:
        // - Color buffer: RGBA float32 (for HDR)
        // - Depth buffer: float32
        // - Screen positions (for selection): float32 x 2
        Tensor color = Tensor::zeros({res.height, res.width, 4}, Device::CUDA, DataType::Float32);
        Tensor depth = Tensor::zeros({res.height, res.width}, Device::CUDA, DataType::Float32);
        Tensor screen_pos = Tensor::zeros({res.height, res.width, 2}, Device::CUDA, DataType::Float32);

        forceMemoryCleanup();
        auto [after, _2] = getGPUMemoryMB();

        size_t delta = after - before;
        size_t expected = (res.width * res.height * (4 + 1 + 2) * sizeof(float)) / (1024 * 1024);

        LOG_INFO("{}: {} MB (expected ~{} MB)", res.name, delta, expected);

        // Verify within reasonable range
        EXPECT_GE(delta, expected * 0.8) << res.name << " should use at least 80% of expected";
        EXPECT_LE(delta, expected * 1.5) << res.name << " should use at most 150% of expected";
    }
}

// Test: Verify multiple resize cycles don't leak
TEST_F(VRAMResizeTest, MultipleResizeCyclesNoLeak) {
    auto [initial_used, total] = getGPUMemoryMB();
    LOG_INFO("Initial VRAM: {} MB", initial_used);

    // Run multiple grow/shrink cycles
    for (int cycle = 0; cycle < 5; ++cycle) {
        LOG_INFO("=== Cycle {} ===", cycle + 1);

        // Grow
        {
            Tensor large = Tensor::zeros({2160, 3840, 4}, Device::CUDA, DataType::Float32);
            forceMemoryCleanup();
            auto [peak, _] = getGPUMemoryMB();
            LOG_INFO("  Peak: {} MB", peak);
        }

        // Shrink
        {
            Tensor small = Tensor::zeros({720, 1280, 4}, Device::CUDA, DataType::Float32);
            forceMemoryCleanup();
            auto [shrunk, _] = getGPUMemoryMB();
            LOG_INFO("  Shrunk: {} MB", shrunk);
        }

        // Clear all
        forceMemoryCleanup();
        auto [after_cycle, _] = getGPUMemoryMB();
        int delta = static_cast<int>(after_cycle) - static_cast<int>(initial_used);
        LOG_INFO("  After clear: {} MB (delta: {} MB)", after_cycle, delta);

        // Should not accumulate memory across cycles
        EXPECT_LE(std::abs(delta), 15) << "Cycle " << (cycle + 1) << " should not leak memory";
    }
}
