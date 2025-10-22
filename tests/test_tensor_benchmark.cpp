/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/tensor.hpp"
#include <chrono>
#include <gtest/gtest.h>
#include <iomanip>
#include <random>
#include <torch/torch.h>

using namespace lfs::core;

// ============= Benchmark Helper =============

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
            std::cout << std::setw(40) << std::left << operation
                      << "  Custom: " << std::setw(8) << std::right << std::fixed
                      << std::setprecision(3) << custom_ms << " ms"
                      << "  Torch: " << std::setw(8) << torch_ms << " ms"
                      << "  Speedup: " << std::setw(6) << std::setprecision(2)
                      << speedup << "x";

            if (speedup < 0.8) {
                std::cout << " ⚠️  SLOWER";
            } else if (speedup > 1.2) {
                std::cout << " ✓ FASTER";
            } else {
                std::cout << " ~ SIMILAR";
            }
            std::cout << std::endl;
        }
    };

    // Helper to create shared input data
    std::vector<float> create_random_data(size_t n, float min_val = 0.0f, float max_val = 1.0f) {
        static std::mt19937 rng(42); // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dist(min_val, max_val);

        std::vector<float> data(n);
        for (auto& val : data) {
            val = dist(rng);
        }
        return data;
    }

    // Create tensor pair with same data
    std::pair<Tensor, torch::Tensor> create_tensor_pair(
        const std::vector<float>& data,
        const std::vector<int64_t>& shape,
        Device device = Device::CUDA) {
        // Convert shape to size_t for custom tensor
        std::vector<size_t> custom_shape;
        for (auto s : shape) {
            custom_shape.push_back(static_cast<size_t>(s));
        }

        // Create custom tensor
        auto tensor_custom = Tensor::from_vector(data, TensorShape(custom_shape), device);

        // Create torch tensor from same data
        auto tensor_torch = torch::from_blob(
                                const_cast<float*>(data.data()),
                                shape,
                                torch::TensorOptions().dtype(torch::kFloat32))
                                .to(device == Device::CUDA ? torch::kCUDA : torch::kCPU)
                                .clone();

        return {std::move(tensor_custom), tensor_torch};
    }

    void compare_tensors(const Tensor& custom, const torch::Tensor& reference,
                         float rtol = 1e-3f, float atol = 1e-3f) {
        auto ref_cpu = reference.cpu().contiguous();
        auto custom_cpu = custom.cpu();

        ASSERT_EQ(custom_cpu.ndim(), ref_cpu.dim()) << "Rank mismatch";
        ASSERT_EQ(custom_cpu.numel(), static_cast<size_t>(ref_cpu.numel())) << "Element count mismatch";

        if (custom_cpu.dtype() == DataType::Float32) {
            auto custom_vec = custom_cpu.to_vector();
            auto ref_ptr = ref_cpu.data_ptr<float>();

            size_t mismatch_count = 0;
            for (size_t i = 0; i < custom_vec.size(); ++i) {
                float diff = std::abs(custom_vec[i] - ref_ptr[i]);
                float threshold = atol + rtol * std::abs(ref_ptr[i]);
                if (diff > threshold) {
                    if (mismatch_count < 10) { // Only print first 10 mismatches
                        EXPECT_LE(diff, threshold)
                            << "Mismatch at index " << i
                            << " (custom=" << custom_vec[i]
                            << ", torch=" << ref_ptr[i] << ")";
                    }
                    mismatch_count++;
                }
            }
            EXPECT_EQ(mismatch_count, 0) << "Total mismatches: " << mismatch_count;
        } else if (custom_cpu.dtype() == DataType::UInt8) {
            auto custom_vec = custom_cpu.to_vector_uint8();
            auto ref_ptr = ref_cpu.data_ptr<uint8_t>();

            size_t mismatch_count = 0;
            for (size_t i = 0; i < custom_vec.size(); ++i) {
                int diff = std::abs(static_cast<int>(custom_vec[i]) - static_cast<int>(ref_ptr[i]));
                if (diff > 1) { // Allow 1 unit difference for rounding
                    if (mismatch_count < 10) {
                        EXPECT_LE(diff, 1)
                            << "Mismatch at index " << i
                            << " (custom=" << static_cast<int>(custom_vec[i])
                            << ", torch=" << static_cast<int>(ref_ptr[i]) << ")";
                    }
                    mismatch_count++;
                }
            }
            EXPECT_EQ(mismatch_count, 0) << "Total mismatches: " << mismatch_count;
        }
    }

} // namespace

class TensorBenchmarkTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Warmup GPU
        auto warmup = Tensor::rand({100, 100}, Device::CUDA);
        auto warmup_torch = torch::rand({100, 100}, torch::kCUDA);
        cudaDeviceSynchronize();
    }

    void print_separator(const std::string& title = "") {
        std::cout << "\n"
                  << std::string(100, '=') << std::endl;
        if (!title.empty()) {
            std::cout << title << std::endl;
            std::cout << std::string(100, '=') << std::endl;
        }
    }
};

// ============= Exact Bottleneck Scenario =============

TEST_F(TensorBenchmarkTest, ImageUploadBottleneck) {
    print_separator("IMAGE UPLOAD BOTTLENECK - Exact Scenario");

    const int H = 720;
    const int W = 820;
    const int C = 3;
    const int iterations = 10;

    std::cout << "Testing " << H << "x" << W << "x" << C << " image upload scenario" << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << std::endl;

    // Create SAME input data for both implementations
    auto input_data = create_random_data(H * W * C, 0.0f, 1.0f);
    auto [img_custom, img_torch] = create_tensor_pair(input_data, {H, W, C}, Device::CUDA);

    // ===== STEP 1: Add Alpha Channel (Cat) =====
    {
        std::cout << "\n--- STEP 1: Add Alpha Channel (Cat) ---" << std::endl;

        double total_custom = 0.0;
        double total_torch = 0.0;

        Tensor result_custom;
        torch::Tensor result_torch;

        for (int i = 0; i < iterations; ++i) {
            // Custom implementation
            {
                Timer timer;
                auto alpha = Tensor::ones({static_cast<size_t>(H), static_cast<size_t>(W), 1},
                                          Device::CUDA, DataType::Float32);
                result_custom = Tensor::cat({img_custom, alpha}, 2);
                cudaDeviceSynchronize();
                total_custom += timer.elapsed_ms();
            }

            // Torch implementation
            {
                Timer timer;
                auto alpha_torch = torch::ones({H, W, 1}, torch::kCUDA);
                result_torch = torch::cat({img_torch, alpha_torch}, 2);
                cudaDeviceSynchronize();
                total_torch += timer.elapsed_ms();
            }
        }

        BenchmarkResult result{
            "Cat (add alpha channel)",
            total_custom / iterations,
            total_torch / iterations,
            total_torch / total_custom};
        result.print();

        compare_tensors(result_custom, result_torch);
    }

    // ===== STEP 2: Clamp [0, 1] =====
    {
        std::cout << "\n--- STEP 2: Clamp [0, 1] ---" << std::endl;

        auto rgba_custom = Tensor::cat({img_custom,
                                        Tensor::ones({static_cast<size_t>(H), static_cast<size_t>(W), 1},
                                                     Device::CUDA)},
                                       2);
        auto rgba_torch = torch::cat({img_torch, torch::ones({H, W, 1}, torch::kCUDA)}, 2);

        double total_custom = 0.0;
        double total_torch = 0.0;

        Tensor result_custom;
        torch::Tensor result_torch;

        for (int i = 0; i < iterations; ++i) {
            // Custom implementation
            {
                Timer timer;
                result_custom = rgba_custom.clamp(0.0f, 1.0f);
                cudaDeviceSynchronize();
                total_custom += timer.elapsed_ms();
            }

            // Torch implementation
            {
                Timer timer;
                result_torch = torch::clamp(rgba_torch, 0.0f, 1.0f);
                cudaDeviceSynchronize();
                total_torch += timer.elapsed_ms();
            }
        }

        BenchmarkResult result{
            "Clamp [0, 1]",
            total_custom / iterations,
            total_torch / iterations,
            total_torch / total_custom};
        result.print();

        compare_tensors(result_custom, result_torch);
    }

    // ===== STEP 3: Multiply by 255 =====
    {
        std::cout << "\n--- STEP 3: Multiply by 255 ---" << std::endl;

        auto rgba_custom = Tensor::cat({img_custom,
                                        Tensor::ones({static_cast<size_t>(H), static_cast<size_t>(W), 1},
                                                     Device::CUDA)},
                                       2)
                               .clamp(0.0f, 1.0f);
        auto rgba_torch = torch::clamp(
            torch::cat({img_torch, torch::ones({H, W, 1}, torch::kCUDA)}, 2),
            0.0f, 1.0f);

        double total_custom = 0.0;
        double total_torch = 0.0;

        Tensor result_custom;
        torch::Tensor result_torch;

        for (int i = 0; i < iterations; ++i) {
            // Custom implementation
            {
                Timer timer;
                result_custom = rgba_custom * 255.0f;
                cudaDeviceSynchronize();
                total_custom += timer.elapsed_ms();
            }

            // Torch implementation
            {
                Timer timer;
                result_torch = rgba_torch * 255.0f;
                cudaDeviceSynchronize();
                total_torch += timer.elapsed_ms();
            }
        }

        BenchmarkResult result{
            "Multiply by 255",
            total_custom / iterations,
            total_torch / iterations,
            total_torch / total_custom};
        result.print();

        compare_tensors(result_custom, result_torch, 1e-2f, 1e-2f);
    }

    // ===== STEP 4: Convert to UInt8 =====
    {
        std::cout << "\n--- STEP 4: Convert to UInt8 ---" << std::endl;

        auto scaled_custom = Tensor::cat({img_custom,
                                          Tensor::ones({static_cast<size_t>(H), static_cast<size_t>(W), 1},
                                                       Device::CUDA)},
                                         2)
                                 .clamp(0.0f, 1.0f) *
                             255.0f;
        auto scaled_torch = torch::clamp(
                                torch::cat({img_torch, torch::ones({H, W, 1}, torch::kCUDA)}, 2),
                                0.0f, 1.0f) *
                            255.0f;

        double total_custom = 0.0;
        double total_torch = 0.0;

        Tensor result_custom;
        torch::Tensor result_torch;

        for (int i = 0; i < iterations; ++i) {
            // Custom implementation
            {
                Timer timer;
                result_custom = scaled_custom.to(DataType::UInt8);
                cudaDeviceSynchronize();
                total_custom += timer.elapsed_ms();
            }

            // Torch implementation
            {
                Timer timer;
                result_torch = scaled_torch.to(torch::kUInt8);
                cudaDeviceSynchronize();
                total_torch += timer.elapsed_ms();
            }
        }

        BenchmarkResult result{
            "Convert Float32 -> UInt8",
            total_custom / iterations,
            total_torch / iterations,
            total_torch / total_custom};
        result.print();

        compare_tensors(result_custom, result_torch);
    }

    // ===== COMPLETE PIPELINE =====
    {
        std::cout << "\n--- COMPLETE PIPELINE (All Steps) ---" << std::endl;

        double total_custom = 0.0;
        double total_torch = 0.0;

        Tensor result_custom;
        torch::Tensor result_torch;

        for (int i = 0; i < iterations; ++i) {
            // Custom implementation
            {
                Timer timer;
                auto alpha = Tensor::ones({static_cast<size_t>(H), static_cast<size_t>(W), 1},
                                          Device::CUDA);
                auto rgba = Tensor::cat({img_custom, alpha}, 2);
                result_custom = (rgba.clamp(0.0f, 1.0f) * 255.0f).to(DataType::UInt8);
                cudaDeviceSynchronize();
                total_custom += timer.elapsed_ms();
            }

            // Torch implementation
            {
                Timer timer;
                auto alpha = torch::ones({H, W, 1}, torch::kCUDA);
                auto rgba = torch::cat({img_torch, alpha}, 2);
                result_torch = (torch::clamp(rgba, 0.0f, 1.0f) * 255.0f).to(torch::kUInt8);
                cudaDeviceSynchronize();
                total_torch += timer.elapsed_ms();
            }
        }

        BenchmarkResult result{
            "COMPLETE PIPELINE",
            total_custom / iterations,
            total_torch / iterations,
            total_torch / total_custom};
        result.print();

        std::cout << "\n📊 ANALYSIS:" << std::endl;
        std::cout << "  Target: < 1ms total pipeline time" << std::endl;
        std::cout << "  Custom: " << result.custom_ms << " ms" << std::endl;
        std::cout << "  Torch:  " << result.torch_ms << " ms" << std::endl;

        if (result.custom_ms < 1.0) {
            std::cout << "  Status: ✓ EXCELLENT" << std::endl;
        } else if (result.custom_ms < 2.0) {
            std::cout << "  Status: ✓ ACCEPTABLE" << std::endl;
        } else if (result.custom_ms < 5.0) {
            std::cout << "  Status: ⚠️  NEEDS OPTIMIZATION" << std::endl;
        } else {
            std::cout << "  Status: 🔴 CRITICAL - MAJOR BOTTLENECK" << std::endl;
        }

        compare_tensors(result_custom, result_torch);
    }
}

// ============= Individual Operation Benchmarks =============

TEST_F(TensorBenchmarkTest, CatLastDimBenchmark) {
    print_separator("CONCATENATION - Last Dimension Performance");

    std::vector<std::tuple<int, int, int>> sizes = {
        {720, 820, 3},   // Actual image size
        {1080, 1920, 3}, // Full HD
        {2160, 3840, 3}, // 4K
        {256, 256, 3},   // Small
        {512, 512, 3}    // Medium
    };

    for (const auto& [h, w, c] : sizes) {
        std::cout << "\nSize: " << h << "x" << w << "x" << c << std::endl;

        auto input_data = create_random_data(h * w * c);
        auto [img_custom, img_torch] = create_tensor_pair(input_data, {h, w, c}, Device::CUDA);

        const int iters = 20;
        double total_custom = 0.0;
        double total_torch = 0.0;

        Tensor result_custom;
        torch::Tensor result_torch;

        for (int i = 0; i < iters; ++i) {
            // Custom
            {
                Timer timer;
                auto alpha_custom = Tensor::ones({static_cast<size_t>(h), static_cast<size_t>(w), 1},
                                                 Device::CUDA);
                result_custom = Tensor::cat({img_custom, alpha_custom}, 2);
                cudaDeviceSynchronize();
                total_custom += timer.elapsed_ms();
            }

            // Torch
            {
                Timer timer;
                auto alpha_torch = torch::ones({h, w, 1}, torch::kCUDA);
                result_torch = torch::cat({img_torch, alpha_torch}, 2);
                cudaDeviceSynchronize();
                total_torch += timer.elapsed_ms();
            }
        }

        BenchmarkResult result{
            std::to_string(h) + "x" + std::to_string(w) + "x" + std::to_string(c),
            total_custom / iters,
            total_torch / iters,
            total_torch / total_custom};
        result.print();

        // Verify correctness on last iteration
        compare_tensors(result_custom, result_torch);
    }
}

TEST_F(TensorBenchmarkTest, ClampBenchmark) {
    print_separator("CLAMP Operation Performance");

    std::vector<size_t> sizes = {
        720 * 820 * 4,   // RGBA image
        1080 * 1920 * 4, // Full HD
        256 * 256 * 4,   // Small
        1024 * 1024 * 4  // 1K
    };

    for (size_t n : sizes) {
        std::cout << "\nElements: " << n << " (" << (n * 4.0 / 1024 / 1024) << " MB)" << std::endl;

        auto input_data = create_random_data(n, -0.5f, 1.5f); // Range [-0.5, 1.5]
        auto [data_custom, data_torch] = create_tensor_pair(input_data, {static_cast<int64_t>(n)}, Device::CUDA);

        const int iters = 50;
        double total_custom = 0.0;
        double total_torch = 0.0;

        Tensor result_custom;
        torch::Tensor result_torch;

        for (int i = 0; i < iters; ++i) {
            // Custom
            {
                Timer timer;
                result_custom = data_custom.clamp(0.0f, 1.0f);
                cudaDeviceSynchronize();
                total_custom += timer.elapsed_ms();
            }

            // Torch
            {
                Timer timer;
                result_torch = torch::clamp(data_torch, 0.0f, 1.0f);
                cudaDeviceSynchronize();
                total_torch += timer.elapsed_ms();
            }
        }

        BenchmarkResult result{
            std::to_string(n / 1000) + "K elements",
            total_custom / iters,
            total_torch / iters,
            total_torch / total_custom};
        result.print();

        compare_tensors(result_custom, result_torch);
    }
}

TEST_F(TensorBenchmarkTest, TypeConversionBenchmark) {
    print_separator("TYPE CONVERSION Performance");

    const size_t n = 720 * 820 * 4;

    std::cout << "\nFloat32 -> UInt8 conversion" << std::endl;
    std::cout << "Elements: " << n << std::endl;

    auto input_data = create_random_data(n, 0.0f, 255.0f);
    auto [data_custom, data_torch] = create_tensor_pair(input_data, {static_cast<int64_t>(n)}, Device::CUDA);

    const int iters = 50;
    double total_custom = 0.0;
    double total_torch = 0.0;

    Tensor result_custom;
    torch::Tensor result_torch;

    for (int i = 0; i < iters; ++i) {
        // Custom
        {
            Timer timer;
            result_custom = data_custom.to(DataType::UInt8);
            cudaDeviceSynchronize();
            total_custom += timer.elapsed_ms();
        }

        // Torch
        {
            Timer timer;
            result_torch = data_torch.to(torch::kUInt8);
            cudaDeviceSynchronize();
            total_torch += timer.elapsed_ms();
        }
    }

    BenchmarkResult result{
        "Float32 -> UInt8",
        total_custom / iters,
        total_torch / iters,
        total_torch / total_custom};
    result.print();

    compare_tensors(result_custom, result_torch);
}

TEST_F(TensorBenchmarkTest, ScalarMultiplyBenchmark) {
    print_separator("SCALAR MULTIPLICATION Performance");

    const size_t n = 720 * 820 * 4;

    auto input_data = create_random_data(n, 0.0f, 1.0f);
    auto [data_custom, data_torch] = create_tensor_pair(input_data, {static_cast<int64_t>(n)}, Device::CUDA);

    const int iters = 50;
    double total_custom = 0.0;
    double total_torch = 0.0;

    Tensor result_custom;
    torch::Tensor result_torch;

    for (int i = 0; i < iters; ++i) {
        // Custom
        {
            Timer timer;
            result_custom = data_custom * 255.0f;
            cudaDeviceSynchronize();
            total_custom += timer.elapsed_ms();
        }

        // Torch
        {
            Timer timer;
            result_torch = data_torch * 255.0f;
            cudaDeviceSynchronize();
            total_torch += timer.elapsed_ms();
        }
    }

    BenchmarkResult result{
        "Scalar multiply (* 255)",
        total_custom / iters,
        total_torch / iters,
        total_torch / total_custom};
    result.print();

    compare_tensors(result_custom, result_torch, 1e-2f, 1e-2f);
}

// ============= Summary Report =============

TEST_F(TensorBenchmarkTest, SummaryReport) {
    print_separator("📊 PERFORMANCE SUMMARY");

    std::cout << "\n🎯 TARGET: Complete image upload pipeline < 1ms" << std::endl;
    std::cout << "\n📋 OPERATIONS:" << std::endl;
    std::cout << "  1. Cat (add alpha):    Target < 0.2ms" << std::endl;
    std::cout << "  2. Clamp [0,1]:        Target < 0.2ms" << std::endl;
    std::cout << "  3. Multiply by 255:    Target < 0.2ms" << std::endl;
    std::cout << "  4. Float32->UInt8:     Target < 0.2ms" << std::endl;
    std::cout << "  5. cudaMemcpy2DArray:  ~0.1ms (OpenGL interop)" << std::endl;
    std::cout << "\n✅ Run individual benchmarks above to see detailed performance" << std::endl;
    std::cout << std::string(100, '=') << std::endl;
}
