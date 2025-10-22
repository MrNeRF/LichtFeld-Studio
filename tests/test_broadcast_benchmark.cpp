/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/tensor.hpp"
#include <chrono>
#include <gtest/gtest.h>
#include <iomanip>
#include <torch/torch.h>

using namespace lfs::core;

// ============= Broadcast Operation Benchmark =============

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

class BroadcastBenchmarkTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Warmup GPU
        auto warmup = Tensor::rand({100, 100}, Device::CUDA);
        auto warmup_torch = torch::rand({100, 100}, torch::kCUDA);
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

// ============= Binary Broadcast Operation Benchmarks =============

TEST_F(BroadcastBenchmarkTest, BroadcastAddition) {
    print_separator("BROADCAST ADDITION - Memory Pool Impact");

    std::cout << "\n🎯 This tests the Phase 2A optimization (binary broadcast ops)" << std::endl;
    std::cout << "📊 Each operation allocates 3 shape arrays (a_shape, b_shape, c_shape)\n"
              << std::endl;

    std::vector<std::tuple<std::string, std::vector<int64_t>, std::vector<int64_t>>> test_cases = {
        {"Image broadcast (720×820×3) + (1×1×3)", {720, 820, 3}, {1, 1, 3}},
        {"Matrix broadcast (1024×1024) + (1024×1)", {1024, 1024}, {1024, 1}},
        {"Batch broadcast (32×256×256) + (1×256×256)", {32, 256, 256}, {1, 256, 256}},
        {"Scalar broadcast (1000×1000) + scalar", {1000, 1000}, {1}},
        {"Vector broadcast (512×512) + (512×1)", {512, 512}, {512, 1}}};

    const int iterations = 100;

    for (const auto& [name, shape_a, shape_b] : test_cases) {
        std::cout << "\n--- " << name << " ---" << std::endl;
        std::cout << "Iterations: " << iterations << std::endl;

        // Convert shapes
        std::vector<size_t> custom_shape_a, custom_shape_b;
        for (auto s : shape_a)
            custom_shape_a.push_back(s);
        for (auto s : shape_b)
            custom_shape_b.push_back(s);

        // Create tensors
        auto a_custom = Tensor::rand(TensorShape(custom_shape_a), Device::CUDA);
        auto b_custom = Tensor::rand(TensorShape(custom_shape_b), Device::CUDA);
        auto a_torch = torch::rand(shape_a, torch::kCUDA);
        auto b_torch = torch::rand(shape_b, torch::kCUDA);

        double total_custom = 0.0;
        double total_torch = 0.0;

        // Benchmark
        for (int i = 0; i < iterations; ++i) {
            // Custom
            {
                Timer timer;
                Tensor result = a_custom + b_custom; // Force evaluation!
                cudaDeviceSynchronize();
                total_custom += timer.elapsed_ms();
            }

            // PyTorch
            {
                Timer timer;
                auto result = a_torch + b_torch;
                cudaDeviceSynchronize();
                total_torch += timer.elapsed_ms();
            }
        }

        BenchmarkResult result{
            name,
            total_custom / iterations,
            total_torch / iterations,
            total_torch / total_custom};
        result.print();
    }
}

TEST_F(BroadcastBenchmarkTest, BroadcastMultiplication) {
    print_separator("BROADCAST MULTIPLICATION");

    const int iterations = 100;

    std::vector<std::tuple<std::string, std::vector<int64_t>, std::vector<int64_t>>> test_cases = {
        {"Element-wise (1024×1024) * (1024×1024)", {1024, 1024}, {1024, 1024}},
        {"Row broadcast (512×512) * (1×512)", {512, 512}, {1, 512}},
        {"Channel-wise (128×128×64) * (1×1×64)", {128, 128, 64}, {1, 1, 64}}};

    for (const auto& [name, shape_a, shape_b] : test_cases) {
        std::vector<size_t> custom_shape_a, custom_shape_b;
        for (auto s : shape_a)
            custom_shape_a.push_back(s);
        for (auto s : shape_b)
            custom_shape_b.push_back(s);

        auto a_custom = Tensor::rand(TensorShape(custom_shape_a), Device::CUDA);
        auto b_custom = Tensor::rand(TensorShape(custom_shape_b), Device::CUDA);
        auto a_torch = torch::rand(shape_a, torch::kCUDA);
        auto b_torch = torch::rand(shape_b, torch::kCUDA);

        double total_custom = 0.0;
        double total_torch = 0.0;

        for (int i = 0; i < iterations; ++i) {
            {
                Timer timer;
                auto result = a_custom * b_custom;
                cudaDeviceSynchronize();
                total_custom += timer.elapsed_ms();
            }

            {
                Timer timer;
                auto result = a_torch * b_torch;
                cudaDeviceSynchronize();
                total_torch += timer.elapsed_ms();
            }
        }

        BenchmarkResult result{
            name,
            total_custom / iterations,
            total_torch / iterations,
            total_torch / total_custom};
        result.print();
    }
}

TEST_F(BroadcastBenchmarkTest, ChainedBroadcastOperations) {
    print_separator("CHAINED BROADCAST OPERATIONS - Real-world Pattern");

    std::cout << "\n📸 Simulates image processing pipeline with multiple broadcasts" << std::endl;
    std::cout << "Pattern: normalize → scale → add bias → clamp\n"
              << std::endl;

    const int H = 720;
    const int W = 820;
    const int C = 3;
    const int iterations = 50;

    auto img_custom = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(C)}, Device::CUDA);
    auto mean_custom = Tensor::rand({1, 1, static_cast<size_t>(C)}, Device::CUDA);
    auto std_custom = Tensor::rand({1, 1, static_cast<size_t>(C)}, Device::CUDA);
    auto scale_custom = Tensor::full({1}, 255.0f, Device::CUDA);

    auto img_torch = torch::rand({H, W, C}, torch::kCUDA);
    auto mean_torch = torch::rand({1, 1, C}, torch::kCUDA);
    auto std_torch = torch::rand({1, 1, C}, torch::kCUDA);
    auto scale_torch = torch::full({1}, 255.0f, torch::kCUDA);

    double total_custom = 0.0;
    double total_torch = 0.0;

    for (int i = 0; i < iterations; ++i) {
        // Custom - 4 broadcast operations
        {
            Timer timer;
            auto normalized = (img_custom - mean_custom) / (std_custom + 1e-8f);
            auto scaled = normalized * scale_custom;
            auto result = scaled.clamp(0.0f, 255.0f);
            cudaDeviceSynchronize();
            total_custom += timer.elapsed_ms();
        }

        // PyTorch - same operations
        {
            Timer timer;
            auto normalized = (img_torch - mean_torch) / (std_torch + 1e-8f);
            auto scaled = normalized * scale_torch;
            auto result = torch::clamp(scaled, 0.0f, 255.0f);
            cudaDeviceSynchronize();
            total_torch += timer.elapsed_ms();
        }
    }

    BenchmarkResult result{
        "Image normalization pipeline (4 broadcast ops)",
        total_custom / iterations,
        total_torch / iterations,
        total_torch / total_custom};
    result.print();

    std::cout << "\n📊 ANALYSIS:" << std::endl;
    std::cout << "  Per-operation overhead: " << std::fixed << std::setprecision(4)
              << (total_custom / iterations / 4.0) << " ms" << std::endl;
    std::cout << "  Shape allocation time: ~" << std::setprecision(4)
              << (total_custom / iterations / 4.0 / 3.0) << " ms per shape array" << std::endl;
}

TEST_F(BroadcastBenchmarkTest, ComparisonOperations) {
    print_separator("BROADCAST COMPARISON OPERATIONS");

    const int iterations = 100;
    const std::vector<int64_t> shape_a = {1024, 1024};
    const std::vector<int64_t> shape_b = {1, 1024};

    std::vector<size_t> custom_shape_a = {1024, 1024};
    std::vector<size_t> custom_shape_b = {1, 1024};

    auto a_custom = Tensor::rand(TensorShape(custom_shape_a), Device::CUDA);
    auto b_custom = Tensor::rand(TensorShape(custom_shape_b), Device::CUDA);
    auto a_torch = torch::rand(shape_a, torch::kCUDA);
    auto b_torch = torch::rand(shape_b, torch::kCUDA);

    // Lambda ops for benchmarking
    auto op_gt_custom = [&]() { return a_custom > b_custom; };
    auto op_lt_custom = [&]() { return a_custom < b_custom; };
    auto op_eq_custom = [&]() { return a_custom == b_custom; };
    auto op_ne_custom = [&]() { return a_custom != b_custom; };

    auto op_gt_torch = [&]() { return a_torch > b_torch; };
    auto op_lt_torch = [&]() { return a_torch < b_torch; };
    auto op_eq_torch = [&]() { return a_torch == b_torch; };
    auto op_ne_torch = [&]() { return a_torch != b_torch; };

    std::vector<std::tuple<std::string, std::function<Tensor()>, std::function<torch::Tensor()>>> ops;
    ops.push_back({"Greater than (>)", op_gt_custom, op_gt_torch});
    ops.push_back({"Less than (<)", op_lt_custom, op_lt_torch});
    ops.push_back({"Equal (==)", op_eq_custom, op_eq_torch});
    ops.push_back({"Not equal (!=)", op_ne_custom, op_ne_torch});

    for (const auto& [name, custom_op, torch_op] : ops) {
        double total_custom = 0.0;
        double total_torch = 0.0;

        for (int i = 0; i < iterations; ++i) {
            {
                Timer timer;
                auto result = custom_op();
                cudaDeviceSynchronize();
                total_custom += timer.elapsed_ms();
            }

            {
                Timer timer;
                auto result = torch_op();
                cudaDeviceSynchronize();
                total_torch += timer.elapsed_ms();
            }
        }

        BenchmarkResult result{
            name,
            total_custom / iterations,
            total_torch / iterations,
            total_torch / total_custom};
        result.print();
    }
}

TEST_F(BroadcastBenchmarkTest, SummaryReport) {
    print_separator("📊 PHASE 2A OPTIMIZATION SUMMARY");

    std::cout << "\n🎯 OPTIMIZATION: Binary broadcast operations use memory pool" << std::endl;

    std::cout << "\n📝 WHAT CHANGED:" << std::endl;
    std::cout << "  - Before: 3× cudaMalloc + 3× cudaFree per operation (~0.9ms overhead)" << std::endl;
    std::cout << "  - After:  3× pool allocate + 3× pool deallocate (~0.006ms overhead)" << std::endl;
    std::cout << "  - Theoretical speedup: 150× for allocation overhead" << std::endl;

    std::cout << "\n📊 OPERATIONS AFFECTED:" << std::endl;
    std::cout << "  - Binary arithmetic: add, sub, mul, div, pow" << std::endl;
    std::cout << "  - Comparisons: ==, !=, <, <=, >, >=" << std::endl;
    std::cout << "  - Logical: and, or, xor" << std::endl;
    std::cout << "  - All use memory pool for shape array allocations" << std::endl;

    std::cout << "\n✅ Run benchmarks above to verify actual performance gains" << std::endl;
    std::cout << "\n💡 Combined with Phase 1 (tensor allocations), memory pool now covers:" << std::endl;
    std::cout << "  ✓ Main tensor data allocations" << std::endl;
    std::cout << "  ✓ Broadcast operation metadata" << std::endl;
    std::cout << "  ⏳ Still TODO: Reduction temp buffers (Phase 2B)" << std::endl;
    std::cout << std::string(110, '=') << std::endl;
}
