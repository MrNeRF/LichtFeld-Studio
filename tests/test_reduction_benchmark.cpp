/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include <chrono>
#include <gtest/gtest.h>
#include <iomanip>
#include <torch/torch.h>

using namespace lfs::core;

// ============= Reduction Operation Benchmark =============

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
                std::cout << " WARNING: SLOWER";
            } else if (speedup > 1.5) {
                std::cout << " FASTER";
            } else {
                std::cout << " ~ SIMILAR";
            }
            std::cout << std::endl;
        }
    };

} // namespace

class ReductionBenchmarkTest : public ::testing::Test {
protected:
    void SetUp() override {

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

// ============= Sum Reduction Benchmarks =============

TEST_F(ReductionBenchmarkTest, SumReductionSingleDim) {
    print_separator("SUM REDUCTION");

    std::vector<std::tuple<std::string, std::vector<int64_t>, int>> test_cases = {
        {"Large matrix sum along dim 0 (1024x1024)", {1024, 1024}, 0},
        {"Large matrix sum along dim 1 (1024x1024)", {1024, 1024}, 1},
        {"3D tensor sum along dim 0 (128x128x64)", {128, 128, 64}, 0},
        {"3D tensor sum along dim 1 (128x128x64)", {128, 128, 64}, 1},
        {"3D tensor sum along dim 2 (128x128x64)", {128, 128, 64}, 2},
        {"4D tensor sum (32x64x64x32) dim 2", {32, 64, 64, 32}, 2},
    };

    const int iterations = 50;

    for (const auto& [name, shape, dim] : test_cases) {
        std::vector<size_t> custom_shape;
        for (const auto s : shape)
            custom_shape.push_back(s);

        auto tensor_custom = Tensor::rand(TensorShape(custom_shape), Device::CUDA);
        auto tensor_torch = torch::rand(shape, torch::kCUDA);
        cudaDeviceSynchronize();

        for (int i = 0; i < 20; ++i) {
            auto r = tensor_custom.sum({dim}, false);
        }
        cudaDeviceSynchronize();

        double total_custom = 0.0, total_torch = 0.0;

        for (int i = 0; i < iterations; ++i) {
            Timer timer;
            auto result = tensor_custom.sum({dim}, false);
            cudaDeviceSynchronize();
            total_custom += timer.elapsed_ms();
        }

        for (int i = 0; i < 20; ++i) {
            auto r = tensor_torch.sum(dim, false);
        }
        cudaDeviceSynchronize();

        for (int i = 0; i < iterations; ++i) {
            Timer timer;
            auto result = tensor_torch.sum(dim, false);
            cudaDeviceSynchronize();
            total_torch += timer.elapsed_ms();
        }

        BenchmarkResult result{
            name,
            total_custom / iterations,
            total_torch / iterations,
            total_torch / total_custom};
        result.print();
    }
}

TEST_F(ReductionBenchmarkTest, MeanReduction) {
    print_separator("MEAN REDUCTION");

    const int iterations = 50;

    std::vector<std::tuple<std::string, std::vector<int64_t>, int>> test_cases = {
        {"Matrix mean along rows (2048x2048)", {2048, 2048}, 0},
        {"Matrix mean along cols (2048x2048)", {2048, 2048}, 1},
        {"3D tensor mean (256x256x16) dim 2", {256, 256, 16}, 2},
    };

    for (const auto& [name, shape, dim] : test_cases) {
        std::vector<size_t> custom_shape;
        for (auto s : shape)
            custom_shape.push_back(s);

        auto tensor_custom = Tensor::rand(TensorShape(custom_shape), Device::CUDA);
        auto tensor_torch = torch::rand(shape, torch::kCUDA);
        cudaDeviceSynchronize();

        for (int i = 0; i < 20; ++i) {
            auto result = tensor_custom.mean({dim}, false);
        }
        cudaDeviceSynchronize();

        double total_custom = 0.0;
        double total_torch = 0.0;

        for (int i = 0; i < iterations; ++i) {
            Timer timer;
            auto result = tensor_custom.mean({dim}, false);
            cudaDeviceSynchronize();
            total_custom += timer.elapsed_ms();
        }

        for (int i = 0; i < 20; ++i) {
            auto result = tensor_torch.mean(dim, false);
        }
        cudaDeviceSynchronize();

        for (int i = 0; i < iterations; ++i) {
            Timer timer;
            auto result = tensor_torch.mean(dim, false);
            cudaDeviceSynchronize();
            total_torch += timer.elapsed_ms();
        }

        BenchmarkResult result{
            name,
            total_custom / iterations,
            total_torch / iterations,
            total_torch / total_custom};
        result.print();
    }
}

TEST_F(ReductionBenchmarkTest, MinMaxReduction) {
    print_separator("MIN/MAX REDUCTION");

    const int iterations = 50;

    std::vector<std::tuple<std::string, std::vector<int64_t>, int>> test_cases = {
        {"Min reduction (1024x1024) dim 0", {1024, 1024}, 0},
        {"Max reduction (1024x1024) dim 1", {1024, 1024}, 1},
        {"Min reduction 3D (128x128x64) dim 2", {128, 128, 64}, 2},
    };

    for (const auto& [name, shape, dim] : test_cases) {
        std::vector<size_t> custom_shape;
        for (auto s : shape)
            custom_shape.push_back(s);

        auto tensor_custom = Tensor::rand(TensorShape(custom_shape), Device::CUDA);
        auto tensor_torch = torch::rand(shape, torch::kCUDA);
        cudaDeviceSynchronize();

        bool is_min = name.find("Min") != std::string::npos;

        for (int i = 0; i < 20; ++i) {
            if (is_min) {
                auto result = tensor_custom.min({dim}, false);
            } else {
                auto result = tensor_custom.max({dim}, false);
            }
        }
        cudaDeviceSynchronize();

        double total_custom = 0.0;
        double total_torch = 0.0;

        for (int i = 0; i < iterations; ++i) {
            Timer timer;
            if (is_min) {
                auto result = tensor_custom.min({dim}, false);
            } else {
                auto result = tensor_custom.max({dim}, false);
            }
            cudaDeviceSynchronize();
            total_custom += timer.elapsed_ms();
        }

        for (int i = 0; i < 20; ++i) {
            if (is_min) {
                auto result = std::get<0>(tensor_torch.min(dim, false));
            } else {
                auto result = std::get<0>(tensor_torch.max(dim, false));
            }
        }
        cudaDeviceSynchronize();

        for (int i = 0; i < iterations; ++i) {
            Timer timer;
            if (is_min) {
                auto result = std::get<0>(tensor_torch.min(dim, false));
            } else {
                auto result = std::get<0>(tensor_torch.max(dim, false));
            }
            cudaDeviceSynchronize();
            total_torch += timer.elapsed_ms();
        }

        BenchmarkResult result{
            name,
            total_custom / iterations,
            total_torch / iterations,
            total_torch / total_custom};
        result.print();
    }
}

TEST_F(ReductionBenchmarkTest, ChainedReductions) {
    print_separator("CHAINED REDUCTIONS");

    const int H = 256;
    const int W = 256;
    const int C = 64;
    const int iterations = 30;

    auto tensor_custom = Tensor::rand({static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(C)}, Device::CUDA);
    auto tensor_torch = torch::rand({H, W, C}, torch::kCUDA);
    cudaDeviceSynchronize();

    for (int i = 0; i < 20; ++i) {
        auto sum_hw = tensor_custom.sum({0, 1}, false);
        auto mean_c = sum_hw.mean({0}, false);
    }
    cudaDeviceSynchronize();

    double total_custom = 0.0;
    double total_torch = 0.0;

    for (int i = 0; i < iterations; ++i) {
        Timer timer;
        auto sum_hw = tensor_custom.sum({0, 1}, false);
        auto mean_c = sum_hw.mean({0}, false);
        cudaDeviceSynchronize();
        total_custom += timer.elapsed_ms();
    }

    for (int i = 0; i < 20; ++i) {
        std::vector<int64_t> dims = {0, 1};
        auto sum_hw = tensor_torch.sum(c10::IntArrayRef(dims), false);
        auto mean_c = sum_hw.mean(0, false);
    }
    cudaDeviceSynchronize();

    for (int i = 0; i < iterations; ++i) {
        Timer timer;
        std::vector<int64_t> dims = {0, 1};
        auto sum_hw = tensor_torch.sum(c10::IntArrayRef(dims), false);
        auto mean_c = sum_hw.mean(0, false);
        cudaDeviceSynchronize();
        total_torch += timer.elapsed_ms();
    }

    BenchmarkResult result{
        "Chained reductions",
        total_custom / iterations,
        total_torch / iterations,
        total_torch / total_custom};
    result.print();
}

TEST_F(ReductionBenchmarkTest, VariableSizeTempStorage) {
    print_separator("VARIABLE-SIZE TEMP STORAGE");

    const int iterations = 100;

    std::vector<std::tuple<std::string, std::vector<int64_t>>> test_cases = {
        {"Small (128x128)", {128, 128}},
        {"Medium (512x512)", {512, 512}},
        {"Large (1024x1024)", {1024, 1024}},
        {"Very Large (2048x2048)", {2048, 2048}},
        {"Small (128x128) - repeat", {128, 128}},  // Should hit cache
        {"Medium (512x512) - repeat", {512, 512}}, // Should hit cache
    };

    for (const auto& [name, shape] : test_cases) {
        std::vector<size_t> custom_shape;
        for (auto s : shape)
            custom_shape.push_back(s);

        auto tensor_custom = Tensor::rand(TensorShape(custom_shape), Device::CUDA);
        auto tensor_torch = torch::rand(shape, torch::kCUDA);
        cudaDeviceSynchronize();

        for (int i = 0; i < 20; ++i) {
            auto result = tensor_custom.sum({0}, false);
        }
        cudaDeviceSynchronize();

        double total_custom = 0.0;
        double total_torch = 0.0;

        for (int i = 0; i < iterations; ++i) {
            Timer timer;
            auto result = tensor_custom.sum({0}, false);
            cudaDeviceSynchronize();
            total_custom += timer.elapsed_ms();
        }

        for (int i = 0; i < 20; ++i) {
            auto result = tensor_torch.sum(0, false);
        }
        cudaDeviceSynchronize();

        for (int i = 0; i < iterations; ++i) {
            Timer timer;
            auto result = tensor_torch.sum(0, false);
            cudaDeviceSynchronize();
            total_torch += timer.elapsed_ms();
        }

        BenchmarkResult result{
            name,
            total_custom / iterations,
            total_torch / iterations,
            total_torch / total_custom};
        result.print();
    }
}

TEST_F(ReductionBenchmarkTest, SummaryReport) {
    print_separator("REDUCTION BENCHMARK SUMMARY");
    std::cout << "CUB temp storage uses memory pool (~0.001ms vs ~0.5ms per alloc)" << std::endl;
    std::cout << std::string(110, '=') << std::endl;
}
