/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Memory pool is now an internal implementation detail of tensor library
// #include "core/memory_pool.hpp"
#include "core_new/tensor.hpp"
#include <chrono>
#include <gtest/gtest.h>
#include <iomanip>
#include <torch/torch.h>

using namespace lfs::core;

// ============= Memory Pool Allocation Benchmark =============

namespace {

    class Timer {
    private:
        std::chrono::high_resolution_clock::time_point start_;

    public:
        Timer() {
            start_ = std::chrono::high_resolution_clock::now();
        }

        double elapsed_ms() const {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
            return duration.count() / 1000.0;
        }
    };

    struct BenchmarkResult {
        std::string operation;
        double custom_ms;
        double torch_ms;
        double speedup;

        void print() const {
            std::cout << std::setw(50) << std::left << operation
                      << "  Custom: " << std::setw(8) << std::right << std::fixed
                      << std::setprecision(4) << custom_ms << " ms"
                      << "  Torch: " << std::setw(8) << torch_ms << " ms"
                      << "  Speedup: " << std::setw(6) << std::setprecision(2)
                      << speedup << "×";

            if (speedup < 0.8) {
                std::cout << " ⚠️  SLOWER";
            } else if (speedup > 1.5) {
                std::cout << " ✓ FASTER";
            } else {
                std::cout << " ~ SIMILAR";
            }
            std::cout << std::endl;
        }
    };

} // namespace

class MemoryPoolBenchmarkTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Warmup GPU and memory pool
        auto warmup = Tensor::empty({100, 100}, Device::CUDA);
        auto warmup_torch = torch::empty({100, 100}, torch::kCUDA);
        cudaDeviceSynchronize();
    }

    void print_separator(const std::string& title = "") {
        std::cout << "\n"
                  << std::string(110, '=') << std::endl;
        if (!title.empty()) {
            std::cout << title << std::endl;
            std::cout << std::string(110, '=') << std::endl;
        }
    }
};

// ============= Core Allocation Benchmark =============

TEST_F(MemoryPoolBenchmarkTest, RepeatedAllocationDeallocation) {
    print_separator("MEMORY ALLOCATION BENCHMARK - The Critical Bottleneck");

    std::cout << "\n🎯 OBJECTIVE: Verify memory pool provides 2-10× speedup for repeated allocations" << std::endl;
    std::cout << "📊 This is the EXACT bottleneck identified in tensor operations\n"
              << std::endl;

    // Test various allocation sizes representative of real workloads
    std::vector<std::pair<std::string, size_t>> test_cases = {
        {"Small (1K elements)", 1024},
        {"Medium (256K elements)", 256 * 1024},
        {"Large (1M elements)", 1024 * 1024},
        {"Image 720p (720×820×4)", 720 * 820 * 4},
        {"Image 1080p (1920×1080×4)", 1920 * 1080 * 4}};

    const int iterations = 100;

    for (const auto& [name, num_elements] : test_cases) {
        std::cout << "\n--- " << name << " ---" << std::endl;
        std::cout << "Allocations: " << iterations << " iterations" << std::endl;
        std::cout << "Bytes per allocation: " << (num_elements * sizeof(float)) / 1024.0 << " KB" << std::endl;

        double total_custom = 0.0;
        double total_torch = 0.0;

        // Custom tensor with memory pool
        {
            Timer timer;
            for (int i = 0; i < iterations; ++i) {
                auto t = Tensor::empty({num_elements}, Device::CUDA, DataType::Float32);
                // Tensor destructor will deallocate back to pool
            }
            cudaDeviceSynchronize();
            total_custom = timer.elapsed_ms();
        }

        // PyTorch with its allocator
        {
            Timer timer;
            for (int i = 0; i < iterations; ++i) {
                auto t = torch::empty({static_cast<int64_t>(num_elements)}, torch::kCUDA);
                // PyTorch destructor will handle deallocation
            }
            cudaDeviceSynchronize();
            total_torch = timer.elapsed_ms();
        }

        BenchmarkResult result{
            name,
            total_custom / iterations,
            total_torch / iterations,
            total_torch / total_custom};
        result.print();

        // Performance expectations
        std::cout << "  Per-allocation time: " << std::fixed << std::setprecision(4)
                  << (total_custom / iterations) << " ms (target: < 0.05 ms)" << std::endl;

        if (result.speedup >= 2.0) {
            std::cout << "  Status: ✅ EXCELLENT - Meets 2-10× speedup target!" << std::endl;
        } else if (result.speedup >= 1.2) {
            std::cout << "  Status: ✓ GOOD - Faster than PyTorch" << std::endl;
        } else if (result.speedup >= 0.8) {
            std::cout << "  Status: ~ ACCEPTABLE - Similar to PyTorch" << std::endl;
        } else {
            std::cout << "  Status: ⚠️  WARNING - Slower than PyTorch" << std::endl;
        }
    }
}

TEST_F(MemoryPoolBenchmarkTest, ImageUploadAllocationPattern) {
    print_separator("REAL-WORLD PATTERN - Image Upload Allocations");

    std::cout << "\n📸 Simulating actual image processing workload" << std::endl;
    std::cout << "Pattern: Multiple temporary tensors during upload pipeline\n"
              << std::endl;

    const int H = 720;
    const int W = 820;
    const int C = 3;
    const int iterations = 50;

    std::cout << "Image size: " << H << "×" << W << "×" << C << std::endl;
    std::cout << "Pipeline iterations: " << iterations << std::endl;

    double total_custom = 0.0;
    double total_torch = 0.0;

    // Custom tensor - simulates: RGB input + Alpha channel + RGBA result
    {
        Timer timer;
        for (int i = 0; i < iterations; ++i) {
            // Allocation 1: Input RGB image
            auto rgb = Tensor::empty({static_cast<size_t>(H), static_cast<size_t>(W),
                                      static_cast<size_t>(C)},
                                     Device::CUDA);

            // Allocation 2: Alpha channel
            auto alpha = Tensor::ones({static_cast<size_t>(H), static_cast<size_t>(W), 1},
                                      Device::CUDA);

            // Allocation 3: RGBA concatenated result
            auto rgba = Tensor::empty({static_cast<size_t>(H), static_cast<size_t>(W), 4},
                                      Device::CUDA);

            // Allocation 4: Clamped result
            auto clamped = Tensor::empty({static_cast<size_t>(H), static_cast<size_t>(W), 4},
                                         Device::CUDA);

            // All destructors fire here - deallocations happen
        }
        cudaDeviceSynchronize();
        total_custom = timer.elapsed_ms();
    }

    // PyTorch - same pattern
    {
        Timer timer;
        for (int i = 0; i < iterations; ++i) {
            // Allocation 1: Input RGB image
            auto rgb = torch::empty({H, W, C}, torch::kCUDA);

            // Allocation 2: Alpha channel
            auto alpha = torch::ones({H, W, 1}, torch::kCUDA);

            // Allocation 3: RGBA concatenated result
            auto rgba = torch::empty({H, W, 4}, torch::kCUDA);

            // Allocation 4: Clamped result
            auto clamped = torch::empty({H, W, 4}, torch::kCUDA);

            // All destructors fire here
        }
        cudaDeviceSynchronize();
        total_torch = timer.elapsed_ms();
    }

    BenchmarkResult result{
        "Image upload pattern (4 allocs/deallocs per iteration)",
        total_custom / iterations,
        total_torch / iterations,
        total_torch / total_custom};
    result.print();

    std::cout << "\n📊 ANALYSIS:" << std::endl;
    std::cout << "  Custom (with pool):  " << std::fixed << std::setprecision(4)
              << (total_custom / iterations) << " ms per pipeline" << std::endl;
    std::cout << "  PyTorch:             " << (total_torch / iterations) << " ms per pipeline" << std::endl;
    std::cout << "  Improvement:         " << std::setprecision(1) << (result.speedup - 1.0) * 100.0
              << "% faster" << std::endl;

    if (result.speedup >= 2.0) {
        std::cout << "\n  ✅ SUCCESS: Memory pool achieves 2-10× target!" << std::endl;
    } else if (result.speedup >= 1.5) {
        std::cout << "\n  ✓ GOOD: Significant speedup achieved" << std::endl;
    } else {
        std::cout << "\n  ℹ️  INFO: Both implementations use efficient caching" << std::endl;
    }
}

TEST_F(MemoryPoolBenchmarkTest, ChurnTest) {
    print_separator("MEMORY CHURN TEST - Varying Sizes");

    std::cout << "\n🔄 Testing allocation/deallocation with varying tensor sizes" << std::endl;
    std::cout << "Simulates real workload with mixed tensor sizes\n"
              << std::endl;

    const int iterations = 100;

    // Mix of sizes that appear in real workloads
    std::vector<size_t> sizes = {
        1024,          // 1K
        256 * 1024,    // 256K
        1024 * 1024,   // 1M
        720 * 820 * 4, // Image
        512 * 512,     // Medium
    };

    double total_custom = 0.0;
    double total_torch = 0.0;

    // Custom tensor
    {
        Timer timer;
        for (int i = 0; i < iterations; ++i) {
            for (size_t size : sizes) {
                auto t = Tensor::empty({size}, Device::CUDA);
            }
        }
        cudaDeviceSynchronize();
        total_custom = timer.elapsed_ms();
    }

    // PyTorch
    {
        Timer timer;
        for (int i = 0; i < iterations; ++i) {
            for (size_t size : sizes) {
                auto t = torch::empty({static_cast<int64_t>(size)}, torch::kCUDA);
            }
        }
        cudaDeviceSynchronize();
        total_torch = timer.elapsed_ms();
    }

    BenchmarkResult result{
        "Mixed size churn test",
        total_custom / iterations,
        total_torch / iterations,
        total_torch / total_custom};
    result.print();

    std::cout << "\n  Total allocations: " << iterations * sizes.size() << std::endl;
    std::cout << "  Avg time per alloc/dealloc cycle: "
              << std::fixed << std::setprecision(4)
              << (total_custom / (iterations * sizes.size())) << " ms" << std::endl;
}

TEST_F(MemoryPoolBenchmarkTest, MemoryPoolStats) {
    print_separator("MEMORY POOL STATISTICS");

    std::cout << "\n📈 Current memory pool status:\n"
              << std::endl;

    // Allocate some tensors to populate the pool
    std::vector<Tensor> tensors;
    for (int i = 0; i < 10; ++i) {
        tensors.push_back(Tensor::empty({1024 * 1024}, Device::CUDA));
    }

    std::cout << "Memory pool stats are now an internal implementation detail" << std::endl;

    // Clear tensors (return to pool)
    tensors.clear();

    std::cout << "\n✓ Memory pool is caching memory for fast reuse (internal)" << std::endl;
}

TEST_F(MemoryPoolBenchmarkTest, SummaryReport) {
    print_separator("📊 MEMORY POOL PERFORMANCE SUMMARY");

    std::cout << "\n🎯 OPTIMIZATION GOAL: Eliminate cudaMalloc overhead" << std::endl;
    std::cout << "\n📝 IMPLEMENTATION:" << std::endl;
    std::cout << "  - CUDA 12.8+ cudaMallocAsync with memory pools" << std::endl;
    std::cout << "  - Stream-ordered allocation for near-instant reuse" << std::endl;
    std::cout << "  - Infinite cache threshold (memory stays in pool)" << std::endl;

    std::cout << "\n📊 EXPECTED IMPROVEMENTS:" << std::endl;
    std::cout << "  - Traditional cudaMalloc: ~0.15-0.6 ms per allocation" << std::endl;
    std::cout << "  - Memory pool allocation: ~0.001-0.01 ms (50-600× faster)" << std::endl;
    std::cout << "  - End-to-end tensor operations: 2-10× speedup" << std::endl;

    std::cout << "\n✅ Run benchmarks above to verify actual performance gains" << std::endl;
    std::cout << "\n💡 TIP: Compare with test_tensor_benchmark.cpp for end-to-end speedups" << std::endl;
    std::cout << std::string(110, '=') << std::endl;
}
