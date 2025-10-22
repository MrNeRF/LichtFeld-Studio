/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/tensor.hpp"
#include <chrono>
#include <gtest/gtest.h>
#include <iomanip>
#include <torch/torch.h>

using namespace lfs::core;

// ============= TIER 1 Scalar Reduction Benchmark =============
// Tests warp-parallel optimizations from calm (5-10x expected speedup)

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
                      << speedup << "x";

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

class ScalarReductionBenchmarkTest : public ::testing::Test {
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

// ============= Sum Scalar Benchmarks =============

TEST_F(ScalarReductionBenchmarkTest, SumScalarReduction) {
    print_separator("SUM SCALAR - Warp-Parallel Optimization (TIER 1)");

    std::cout << "\n🎯 OPTIMIZATION: Single-block warp-parallel kernel from calm" << std::endl;
    std::cout << "📊 Pattern: float2 vectorized loads + warp shuffle reductions" << std::endl;
    std::cout << "💡 Expected: 5-10x speedup for small-medium tensors\n"
              << std::endl;

    const int iterations = 100;

    std::vector<std::pair<std::string, std::vector<size_t>>> test_cases = {
        {"Small vector (1K)", {1000}},
        {"Medium vector (100K)", {100'000}},
        {"Large vector (1M)", {1'000'000}},
        {"Small matrix (512×512)", {512, 512}},
        {"Medium matrix (1024×1024)", {1024, 1024}},
        {"Large matrix (2048×2048)", {2048, 2048}},
        {"3D tensor (128×128×64)", {128, 128, 64}},
        {"4D tensor (32×64×64×32)", {32, 64, 64, 32}},
    };

    for (const auto& [name, shape] : test_cases) {
        auto tensor_custom = Tensor::rand(TensorShape(shape), Device::CUDA);

        std::vector<int64_t> torch_shape;
        for (auto s : shape)
            torch_shape.push_back(s);
        auto tensor_torch = torch::rand(torch_shape, torch::kCUDA);

        double total_custom = 0.0;
        double total_torch = 0.0;

        for (int i = 0; i < iterations; ++i) {
            // Custom - optimized warp-parallel kernel
            {
                Timer timer;
                float result = tensor_custom.sum_scalar();
                cudaDeviceSynchronize();
                total_custom += timer.elapsed_ms();
            }

            // PyTorch - standard reduction
            {
                Timer timer;
                auto result = tensor_torch.sum().item<float>();
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

// ============= Mean Scalar Benchmarks =============

TEST_F(ScalarReductionBenchmarkTest, MeanScalarReduction) {
    print_separator("MEAN SCALAR - Warp-Parallel Optimization (TIER 1)");

    const int iterations = 100;

    std::vector<std::pair<std::string, std::vector<size_t>>> test_cases = {
        {"Vector (100K)", {100'000}},
        {"Matrix (1024×1024)", {1024, 1024}},
        {"Matrix (2048×2048)", {2048, 2048}},
        {"3D tensor (256×256×16)", {256, 256, 16}},
    };

    for (const auto& [name, shape] : test_cases) {
        auto tensor_custom = Tensor::rand(TensorShape(shape), Device::CUDA);

        std::vector<int64_t> torch_shape;
        for (auto s : shape)
            torch_shape.push_back(s);
        auto tensor_torch = torch::rand(torch_shape, torch::kCUDA);

        double total_custom = 0.0;
        double total_torch = 0.0;

        for (int i = 0; i < iterations; ++i) {
            {
                Timer timer;
                float result = tensor_custom.mean_scalar();
                cudaDeviceSynchronize();
                total_custom += timer.elapsed_ms();
            }

            {
                Timer timer;
                auto result = tensor_torch.mean().item<float>();
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

// ============= Min/Max Scalar Benchmarks =============

TEST_F(ScalarReductionBenchmarkTest, MinMaxScalarReduction) {
    print_separator("MIN/MAX SCALAR - Warp-Parallel Optimization (TIER 1)");

    const int iterations = 100;

    std::vector<std::pair<std::string, std::vector<size_t>>> test_cases = {
        {"Vector (100K)", {100'000}},
        {"Matrix (1024×1024)", {1024, 1024}},
        {"3D tensor (128×128×64)", {128, 128, 64}},
    };

    for (const auto& [name, shape] : test_cases) {
        auto tensor_custom = Tensor::rand(TensorShape(shape), Device::CUDA);

        std::vector<int64_t> torch_shape;
        for (auto s : shape)
            torch_shape.push_back(s);
        auto tensor_torch = torch::rand(torch_shape, torch::kCUDA);

        double total_custom_min = 0.0;
        double total_torch_min = 0.0;
        double total_custom_max = 0.0;
        double total_torch_max = 0.0;

        for (int i = 0; i < iterations; ++i) {
            // Min
            {
                Timer timer;
                float result = tensor_custom.min_scalar();
                cudaDeviceSynchronize();
                total_custom_min += timer.elapsed_ms();
            }

            {
                Timer timer;
                auto result = tensor_torch.min().item<float>();
                cudaDeviceSynchronize();
                total_torch_min += timer.elapsed_ms();
            }

            // Max
            {
                Timer timer;
                float result = tensor_custom.max_scalar();
                cudaDeviceSynchronize();
                total_custom_max += timer.elapsed_ms();
            }

            {
                Timer timer;
                auto result = tensor_torch.max().item<float>();
                cudaDeviceSynchronize();
                total_torch_max += timer.elapsed_ms();
            }
        }

        BenchmarkResult result_min{
            name + " - MIN",
            total_custom_min / iterations,
            total_torch_min / iterations,
            total_torch_min / total_custom_min};
        result_min.print();

        BenchmarkResult result_max{
            name + " - MAX",
            total_custom_max / iterations,
            total_torch_max / iterations,
            total_torch_max / total_custom_max};
        result_max.print();
    }
}

// ============= Count Nonzero Benchmarks =============

TEST_F(ScalarReductionBenchmarkTest, CountNonzeroReduction) {
    print_separator("COUNT NONZERO - Warp-Parallel Optimization (TIER 1)");

    const int iterations = 100;

    std::vector<std::pair<std::string, std::vector<size_t>>> test_cases = {
        {"Vector (100K)", {100'000}},
        {"Matrix (1024×1024)", {1024, 1024}},
        {"3D tensor (256×256×16)", {256, 256, 16}},
    };

    for (const auto& [name, shape] : test_cases) {
        // Create sparse tensor (50% zeros)
        auto tensor_custom = (Tensor::rand(TensorShape(shape), Device::CUDA) > 0.5f).to(DataType::Float32);

        std::vector<int64_t> torch_shape;
        for (auto s : shape)
            torch_shape.push_back(s);
        auto tensor_torch = (torch::rand(torch_shape, torch::kCUDA) > 0.5f).to(torch::kFloat);

        double total_custom = 0.0;
        double total_torch = 0.0;

        for (int i = 0; i < iterations; ++i) {
            {
                Timer timer;
                size_t result = tensor_custom.count_nonzero();
                cudaDeviceSynchronize();
                total_custom += timer.elapsed_ms();
            }

            {
                Timer timer;
                auto result = tensor_torch.count_nonzero().item<int64_t>();
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

// ============= Norm Benchmarks =============

TEST_F(ScalarReductionBenchmarkTest, NormReduction) {
    print_separator("L1/L2 NORM - Warp-Parallel Optimization (TIER 1)");

    const int iterations = 100;

    std::vector<std::pair<std::string, std::vector<size_t>>> test_cases = {
        {"Vector (100K)", {100'000}},
        {"Matrix (1024×1024)", {1024, 1024}},
        {"3D tensor (128×128×64)", {128, 128, 64}},
    };

    for (const auto& [name, shape] : test_cases) {
        auto tensor_custom = Tensor::rand(TensorShape(shape), Device::CUDA);

        std::vector<int64_t> torch_shape;
        for (auto s : shape)
            torch_shape.push_back(s);
        auto tensor_torch = torch::rand(torch_shape, torch::kCUDA);

        double total_custom_l1 = 0.0;
        double total_torch_l1 = 0.0;
        double total_custom_l2 = 0.0;
        double total_torch_l2 = 0.0;

        for (int i = 0; i < iterations; ++i) {
            // L1 norm
            {
                Timer timer;
                float result = tensor_custom.norm(1.0f);
                cudaDeviceSynchronize();
                total_custom_l1 += timer.elapsed_ms();
            }

            {
                Timer timer;
                auto result = tensor_torch.norm(1.0).item<float>();
                cudaDeviceSynchronize();
                total_torch_l1 += timer.elapsed_ms();
            }

            // L2 norm
            {
                Timer timer;
                float result = tensor_custom.norm(2.0f);
                cudaDeviceSynchronize();
                total_custom_l2 += timer.elapsed_ms();
            }

            {
                Timer timer;
                auto result = tensor_torch.norm(2.0).item<float>();
                cudaDeviceSynchronize();
                total_torch_l2 += timer.elapsed_ms();
            }
        }

        BenchmarkResult result_l1{
            name + " - L1",
            total_custom_l1 / iterations,
            total_torch_l1 / iterations,
            total_torch_l1 / total_custom_l1};
        result_l1.print();

        BenchmarkResult result_l2{
            name + " - L2",
            total_custom_l2 / iterations,
            total_torch_l2 / iterations,
            total_torch_l2 / total_custom_l2};
        result_l2.print();
    }
}

// ============= Dot Product Benchmarks =============

TEST_F(ScalarReductionBenchmarkTest, DotProductReduction) {
    print_separator("DOT PRODUCT - Warp-Parallel Optimization (TIER 1)");

    const int iterations = 100;

    std::vector<std::pair<std::string, size_t>> test_cases = {
        {"Small (1K)", 1'000},
        {"Medium (100K)", 100'000},
        {"Large (1M)", 1'000'000},
        {"Very Large (10M)", 10'000'000},
    };

    for (const auto& [name, n] : test_cases) {
        auto vec1_custom = Tensor::rand({n}, Device::CUDA);
        auto vec2_custom = Tensor::rand({n}, Device::CUDA);

        auto vec1_torch = torch::rand({static_cast<int64_t>(n)}, torch::kCUDA);
        auto vec2_torch = torch::rand({static_cast<int64_t>(n)}, torch::kCUDA);

        double total_custom = 0.0;
        double total_torch = 0.0;

        for (int i = 0; i < iterations; ++i) {
            {
                Timer timer;
                auto result_tensor = vec1_custom.dot(vec2_custom);
                float result = result_tensor.item();
                cudaDeviceSynchronize();
                total_custom += timer.elapsed_ms();
            }

            {
                Timer timer;
                auto result = vec1_torch.dot(vec2_torch).item<float>();
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

// ============= Real-world Pattern Benchmarks =============

TEST_F(ScalarReductionBenchmarkTest, TrainingLossComputation) {
    print_separator("REAL-WORLD PATTERN - Training Loss Computation");

    std::cout << "\n📸 Simulates computing loss + metrics in training loop" << std::endl;
    std::cout << "Pattern: MSE loss + L2 regularization + gradient norm\n"
              << std::endl;

    const int iterations = 50;
    const size_t N = 1024 * 1024; // 1M parameters

    auto predictions = Tensor::rand({N}, Device::CUDA);
    auto targets = Tensor::rand({N}, Device::CUDA);
    auto gradients = Tensor::rand({N}, Device::CUDA);

    auto pred_torch = torch::rand({static_cast<int64_t>(N)}, torch::kCUDA);
    auto targ_torch = torch::rand({static_cast<int64_t>(N)}, torch::kCUDA);
    auto grad_torch = torch::rand({static_cast<int64_t>(N)}, torch::kCUDA);

    double total_custom = 0.0;
    double total_torch = 0.0;

    for (int i = 0; i < iterations; ++i) {
        // Custom - multiple scalar reductions
        {
            Timer timer;
            auto diff = predictions - targets;
            float mse = (diff * diff).mean_scalar();
            float l2_reg = (predictions * predictions).sum_scalar() * 0.01f;
            float grad_norm = gradients.norm(2.0f);
            float total_loss = mse + l2_reg;
            cudaDeviceSynchronize();
            total_custom += timer.elapsed_ms();
        }

        // PyTorch - same operations
        {
            Timer timer;
            auto diff = pred_torch - targ_torch;
            float mse = (diff * diff).mean().item<float>();
            float l2_reg = (pred_torch * pred_torch).sum().item<float>() * 0.01f;
            float grad_norm = grad_torch.norm(2.0).item<float>();
            float total_loss = mse + l2_reg;
            cudaDeviceSynchronize();
            total_torch += timer.elapsed_ms();
        }
    }

    BenchmarkResult result{
        "Loss computation (3 scalar reductions)",
        total_custom / iterations,
        total_torch / iterations,
        total_torch / total_custom};
    result.print();

    std::cout << "\n📊 ANALYSIS:" << std::endl;
    std::cout << "  Per-operation overhead: " << std::fixed << std::setprecision(4)
              << (total_custom / iterations / 3.0) << " ms" << std::endl;
    std::cout << "  This is the EXACT use case for warp-parallel optimizations" << std::endl;
}

// ============= Summary Report =============

TEST_F(ScalarReductionBenchmarkTest, SummaryReport) {
    print_separator("📊 TIER 1 OPTIMIZATION SUMMARY");

    std::cout << "\n🎯 OPTIMIZATION: Warp-parallel scalar reductions from calm" << std::endl;

    std::cout << "\n📝 WHAT CHANGED:" << std::endl;
    std::cout << "  - Before: Multi-block kernels with atomicAdd coordination" << std::endl;
    std::cout << "  - After:  Single-block (256 threads) with warp shuffle reductions" << std::endl;
    std::cout << "  - Key technique: float2 vectorized loads (2× memory bandwidth)" << std::endl;
    std::cout << "  - Expected speedup: 5-10× for small-medium tensors" << std::endl;

    std::cout << "\n📊 OPERATIONS OPTIMIZED:" << std::endl;
    std::cout << "  ✓ sum_scalar()      - Full tensor → single sum" << std::endl;
    std::cout << "  ✓ mean_scalar()     - Full tensor → single mean" << std::endl;
    std::cout << "  ✓ min_scalar()      - Full tensor → single min" << std::endl;
    std::cout << "  ✓ max_scalar()      - Full tensor → single max" << std::endl;
    std::cout << "  ✓ count_nonzero()   - Full tensor → count" << std::endl;
    std::cout << "  ✓ l1_norm()         - Full tensor → L1 norm" << std::endl;
    std::cout << "  ✓ l2_norm()         - Full tensor → L2 norm" << std::endl;
    std::cout << "  ✓ dot()             - Two vectors → dot product" << std::endl;

    std::cout << "\n💡 WHEN TO USE:" << std::endl;
    std::cout << "  - Computing loss values (MSE, MAE, etc.)" << std::endl;
    std::cout << "  - Gradient norm for clipping" << std::endl;
    std::cout << "  - Model parameter statistics" << std::endl;
    std::cout << "  - Any full-tensor-to-scalar reduction" << std::endl;

    std::cout << "\n⚠️  NOT FOR:" << std::endl;
    std::cout << "  - Dimension-wise reductions (use CUB kernels)" << std::endl;
    std::cout << "  - Very large tensors > 100M elements (multi-block better)" << std::endl;

    std::cout << "\n✅ Run benchmarks above to verify actual performance gains" << std::endl;
    std::cout << std::string(110, '=') << std::endl;
}
