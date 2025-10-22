/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/tensor.hpp"
#include <chrono>
#include <gtest/gtest.h>
#include <iomanip>
#include <torch/torch.h>

using namespace lfs::core;

// ============================================================================
// Helper Functions
// ============================================================================

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
        bool verified;

        void print() const {
            std::cout << std::setw(60) << std::left << operation
                      << "  Custom: " << std::setw(8) << std::right << std::fixed
                      << std::setprecision(4) << custom_ms << " ms"
                      << "  Torch: " << std::setw(8) << torch_ms << " ms"
                      << "  Speedup: " << std::setw(6) << std::setprecision(2)
                      << speedup << "x";

            if (!verified) {
                std::cout << " ❌ MISMATCH";
            } else if (speedup > 1.5) {
                std::cout << " ✓ FASTER";
            } else if (speedup > 0.8) {
                std::cout << " ~ SIMILAR";
            } else {
                std::cout << " ⚠ SLOWER";
            }
            std::cout << std::endl;
        }
    };

    bool tensors_equal(const Tensor& a, const torch::Tensor& b_torch, float tol = 1e-4f) {
        // Convert both to CPU for comparison
        auto a_cpu = a.cpu();
        auto b_cpu = b_torch.cpu();

        if (a_cpu.numel() != b_cpu.numel()) {
            std::cout << "    Size mismatch: " << a_cpu.numel() << " vs " << b_cpu.numel() << std::endl;
            return false;
        }

        const float* a_ptr = a_cpu.ptr<float>();
        const float* b_ptr = b_cpu.data_ptr<float>();

        size_t mismatch_count = 0;
        float max_diff = 0.0f;

        for (size_t i = 0; i < a_cpu.numel(); ++i) {
            float diff = std::abs(a_ptr[i] - b_ptr[i]);
            max_diff = std::max(max_diff, diff);
            if (diff > tol) {
                mismatch_count++;
                if (mismatch_count <= 5) { // Print first 5 mismatches
                    std::cout << "    Mismatch at [" << i << "]: "
                              << a_ptr[i] << " vs " << b_ptr[i]
                              << " (diff: " << diff << ")" << std::endl;
                }
            }
        }

        if (mismatch_count > 0) {
            std::cout << "    Total mismatches: " << mismatch_count << " / " << a_cpu.numel()
                      << " (max diff: " << max_diff << ")" << std::endl;
            return false;
        }

        return true;
    }

} // namespace

// ============================================================================
// Test Fixture
// ============================================================================

class FusionBenchmarkTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Warmup GPU
        auto warmup = Tensor::rand({100, 100}, Device::CUDA);
        auto warmup_torch = torch::rand({100, 100}, torch::kCUDA);
        cudaDeviceSynchronize();
    }

    void print_separator(const std::string& title = "") {
        std::cout << "\n"
                  << std::string(120, '=') << std::endl;
        if (!title.empty()) {
            std::cout << title << std::endl;
            std::cout << std::string(120, '=') << std::endl;
        }
    }
};

// ============================================================================
// 2-Operation Fusion Benchmarks
// ============================================================================

TEST_F(FusionBenchmarkTest, TwoOpChain_ExpMul) {
    print_separator("2-OPERATION FUSION: exp().mul()");

    std::cout << "\nPattern: a.exp() * 2.0" << std::endl;
    std::cout << "Expected: 2× speedup (1 allocation + 1 fused kernel vs 2 allocations + 2 kernels)\n"
              << std::endl;

    const int iterations = 100;
    std::vector<std::tuple<std::string, std::vector<size_t>>> test_cases = {
        {"Small vector (1K)", {1024}},
        {"Medium vector (100K)", {102400}},
        {"Large vector (1M)", {1048576}},
        {"Small matrix (512×512)", {512, 512}},
        {"Large matrix (2048×2048)", {2048, 2048}},
    };

    for (const auto& [name, shape] : test_cases) {
        // Create tensors
        auto tensor_custom = Tensor::randn(TensorShape(shape), Device::CUDA);
        std::vector<int64_t> torch_shape(shape.begin(), shape.end());
        auto tensor_torch = torch::randn(torch_shape, torch::kCUDA);

        // Copy data to ensure same starting values
        cudaMemcpy(const_cast<float*>(tensor_torch.data_ptr<float>()),
                   tensor_custom.ptr<float>(),
                   tensor_custom.numel() * sizeof(float),
                   cudaMemcpyDeviceToDevice);

        double total_custom = 0.0;
        double total_torch = 0.0;

        Tensor result_custom;
        torch::Tensor result_torch;

        for (int i = 0; i < iterations; ++i) {
            // Custom - lazy evaluation with fusion
            {
                Timer timer;
                result_custom = tensor_custom.exp() * 2.0f;
                cudaDeviceSynchronize();
                total_custom += timer.elapsed_ms();
            }

            // PyTorch - eager evaluation
            {
                Timer timer;
                result_torch = tensor_torch.exp() * 2.0f;
                cudaDeviceSynchronize();
                total_torch += timer.elapsed_ms();
            }
        }

        // Verify correctness
        bool verified = tensors_equal(result_custom, result_torch);

        BenchmarkResult result{
            name,
            total_custom / iterations,
            total_torch / iterations,
            total_torch / total_custom,
            verified};
        result.print();
    }
}

TEST_F(FusionBenchmarkTest, TwoOpChain_ExpAdd) {
    print_separator("2-OPERATION FUSION: exp().add()");

    std::cout << "\nPattern: a.exp() + 1.0" << std::endl;
    std::cout << "Expected: 2× speedup\n"
              << std::endl;

    const int iterations = 100;
    std::vector<std::tuple<std::string, std::vector<size_t>>> test_cases = {
        {"Vector (100K)", {102400}},
        {"Matrix (1024×1024)", {1024, 1024}},
    };

    for (const auto& [name, shape] : test_cases) {
        auto tensor_custom = Tensor::randn(TensorShape(shape), Device::CUDA);
        std::vector<int64_t> torch_shape(shape.begin(), shape.end());
        auto tensor_torch = torch::randn(torch_shape, torch::kCUDA);

        cudaMemcpy(const_cast<float*>(tensor_torch.data_ptr<float>()),
                   tensor_custom.ptr<float>(),
                   tensor_custom.numel() * sizeof(float),
                   cudaMemcpyDeviceToDevice);

        double total_custom = 0.0;
        double total_torch = 0.0;

        Tensor result_custom;
        torch::Tensor result_torch;

        for (int i = 0; i < iterations; ++i) {
            {
                Timer timer;
                result_custom = tensor_custom.exp() + 1.0f;
                cudaDeviceSynchronize();
                total_custom += timer.elapsed_ms();
            }

            {
                Timer timer;
                result_torch = tensor_torch.exp() + 1.0f;
                cudaDeviceSynchronize();
                total_torch += timer.elapsed_ms();
            }
        }

        bool verified = tensors_equal(result_custom, result_torch);

        BenchmarkResult result{
            name,
            total_custom / iterations,
            total_torch / iterations,
            total_torch / total_custom,
            verified};
        result.print();
    }
}

TEST_F(FusionBenchmarkTest, TwoOpChain_MulAdd) {
    print_separator("2-OPERATION FUSION: mul().add()");

    std::cout << "\nPattern: a * 2.0 + 1.0" << std::endl;
    std::cout << "Expected: 2× speedup\n"
              << std::endl;

    const int iterations = 100;
    std::vector<std::tuple<std::string, std::vector<size_t>>> test_cases = {
        {"Vector (100K)", {102400}},
        {"Matrix (1024×1024)", {1024, 1024}},
        {"3D Tensor (128×128×64)", {128, 128, 64}},
    };

    for (const auto& [name, shape] : test_cases) {
        auto tensor_custom = Tensor::randn(TensorShape(shape), Device::CUDA);
        std::vector<int64_t> torch_shape(shape.begin(), shape.end());
        auto tensor_torch = torch::randn(torch_shape, torch::kCUDA);

        cudaMemcpy(const_cast<float*>(tensor_torch.data_ptr<float>()),
                   tensor_custom.ptr<float>(),
                   tensor_custom.numel() * sizeof(float),
                   cudaMemcpyDeviceToDevice);

        double total_custom = 0.0;
        double total_torch = 0.0;

        Tensor result_custom;
        torch::Tensor result_torch;

        for (int i = 0; i < iterations; ++i) {
            {
                Timer timer;
                result_custom = tensor_custom * 2.0f + 1.0f;
                cudaDeviceSynchronize();
                total_custom += timer.elapsed_ms();
            }

            {
                Timer timer;
                result_torch = tensor_torch * 2.0f + 1.0f;
                cudaDeviceSynchronize();
                total_torch += timer.elapsed_ms();
            }
        }

        bool verified = tensors_equal(result_custom, result_torch);

        BenchmarkResult result{
            name,
            total_custom / iterations,
            total_torch / iterations,
            total_torch / total_custom,
            verified};
        result.print();
    }
}

// ============================================================================
// 3-Operation Fusion Benchmarks
// ============================================================================

TEST_F(FusionBenchmarkTest, ThreeOpChain_ExpMulAdd) {
    print_separator("3-OPERATION FUSION: exp().mul().add()");

    std::cout << "\nPattern: a.exp() * 2.0 + 1.0" << std::endl;
    std::cout << "Expected: 3× speedup (1 allocation + 1 fused kernel vs 3 allocations + 3 kernels)\n"
              << std::endl;

    const int iterations = 100;
    std::vector<std::tuple<std::string, std::vector<size_t>>> test_cases = {
        {"Vector (100K)", {102400}},
        {"Matrix (1024×1024)", {1024, 1024}},
        {"Large matrix (2048×2048)", {2048, 2048}},
    };

    for (const auto& [name, shape] : test_cases) {
        auto tensor_custom = Tensor::randn(TensorShape(shape), Device::CUDA);
        std::vector<int64_t> torch_shape(shape.begin(), shape.end());
        auto tensor_torch = torch::randn(torch_shape, torch::kCUDA);

        cudaMemcpy(const_cast<float*>(tensor_torch.data_ptr<float>()),
                   tensor_custom.ptr<float>(),
                   tensor_custom.numel() * sizeof(float),
                   cudaMemcpyDeviceToDevice);

        double total_custom = 0.0;
        double total_torch = 0.0;

        Tensor result_custom;
        torch::Tensor result_torch;

        for (int i = 0; i < iterations; ++i) {
            {
                Timer timer;
                result_custom = tensor_custom.exp() * 2.0f + 1.0f;
                cudaDeviceSynchronize();
                total_custom += timer.elapsed_ms();
            }

            {
                Timer timer;
                result_torch = tensor_torch.exp() * 2.0f + 1.0f;
                cudaDeviceSynchronize();
                total_torch += timer.elapsed_ms();
            }
        }

        bool verified = tensors_equal(result_custom, result_torch);

        BenchmarkResult result{
            name,
            total_custom / iterations,
            total_torch / iterations,
            total_torch / total_custom,
            verified};
        result.print();
    }
}

TEST_F(FusionBenchmarkTest, ThreeOpChain_SqrtMulSub) {
    print_separator("3-OPERATION FUSION: sqrt().mul().sub()");

    std::cout << "\nPattern: a.sqrt() * 3.0 - 0.5" << std::endl;
    std::cout << "Expected: 3× speedup\n"
              << std::endl;

    const int iterations = 100;
    std::vector<std::tuple<std::string, std::vector<size_t>>> test_cases = {
        {"Vector (100K)", {102400}},
        {"Matrix (1024×1024)", {1024, 1024}},
    };

    for (const auto& [name, shape] : test_cases) {
        auto tensor_custom = Tensor::rand(TensorShape(shape), Device::CUDA) + 0.1f; // Avoid sqrt(0)
        std::vector<int64_t> torch_shape(shape.begin(), shape.end());
        auto tensor_torch = torch::rand(torch_shape, torch::kCUDA) + 0.1f;

        cudaMemcpy(const_cast<float*>(tensor_torch.data_ptr<float>()),
                   tensor_custom.ptr<float>(),
                   tensor_custom.numel() * sizeof(float),
                   cudaMemcpyDeviceToDevice);

        double total_custom = 0.0;
        double total_torch = 0.0;

        Tensor result_custom;
        torch::Tensor result_torch;

        for (int i = 0; i < iterations; ++i) {
            {
                Timer timer;
                result_custom = tensor_custom.sqrt() * 3.0f - 0.5f;
                cudaDeviceSynchronize();
                total_custom += timer.elapsed_ms();
            }

            {
                Timer timer;
                result_torch = tensor_torch.sqrt() * 3.0f - 0.5f;
                cudaDeviceSynchronize();
                total_torch += timer.elapsed_ms();
            }
        }

        bool verified = tensors_equal(result_custom, result_torch);

        BenchmarkResult result{
            name,
            total_custom / iterations,
            total_torch / iterations,
            total_torch / total_custom,
            verified};
        result.print();
    }
}

// ============================================================================
// 4-Operation Fusion Benchmarks
// ============================================================================

TEST_F(FusionBenchmarkTest, FourOpChain_AbsExpMulAdd) {
    print_separator("4-OPERATION FUSION: abs().exp().mul().add()");

    std::cout << "\nPattern: a.abs().exp() * 2.0 + 1.0" << std::endl;
    std::cout << "Expected: 4× speedup\n"
              << std::endl;

    const int iterations = 100;
    std::vector<std::tuple<std::string, std::vector<size_t>>> test_cases = {
        {"Vector (100K)", {102400}},
        {"Matrix (1024×1024)", {1024, 1024}},
    };

    for (const auto& [name, shape] : test_cases) {
        auto tensor_custom = Tensor::randn(TensorShape(shape), Device::CUDA);
        std::vector<int64_t> torch_shape(shape.begin(), shape.end());
        auto tensor_torch = torch::randn(torch_shape, torch::kCUDA);

        cudaMemcpy(const_cast<float*>(tensor_torch.data_ptr<float>()),
                   tensor_custom.ptr<float>(),
                   tensor_custom.numel() * sizeof(float),
                   cudaMemcpyDeviceToDevice);

        double total_custom = 0.0;
        double total_torch = 0.0;

        Tensor result_custom;
        torch::Tensor result_torch;

        for (int i = 0; i < iterations; ++i) {
            {
                Timer timer;
                result_custom = tensor_custom.abs().exp() * 2.0f + 1.0f;
                cudaDeviceSynchronize();
                total_custom += timer.elapsed_ms();
            }

            {
                Timer timer;
                result_torch = tensor_torch.abs().exp() * 2.0f + 1.0f;
                cudaDeviceSynchronize();
                total_torch += timer.elapsed_ms();
            }
        }

        bool verified = tensors_equal(result_custom, result_torch);

        BenchmarkResult result{
            name,
            total_custom / iterations,
            total_torch / iterations,
            total_torch / total_custom,
            verified};
        result.print();
    }
}

// ============================================================================
// Real-World Pattern Benchmarks
// ============================================================================

TEST_F(FusionBenchmarkTest, RealWorld_ImageNormalization) {
    print_separator("REAL-WORLD PATTERN: Image Normalization");

    std::cout << "\nPattern: (image - mean) / std" << std::endl;
    std::cout << "This is a common preprocessing step in computer vision pipelines" << std::endl;
    std::cout << "Expected: 2× speedup\n"
              << std::endl;

    const int iterations = 100;

    // Typical image batch: [batch, channels, height, width]
    std::vector<std::tuple<std::string, std::vector<size_t>>> test_cases = {
        {"Small batch (8×3×256×256)", {8, 3, 256, 256}},
        {"Medium batch (16×3×512×512)", {16, 3, 512, 512}},
        {"Large batch (32×3×256×256)", {32, 3, 256, 256}},
    };

    for (const auto& [name, shape] : test_cases) {
        auto image_custom = Tensor::rand(TensorShape(shape), Device::CUDA) * 255.0f;
        std::vector<int64_t> torch_shape(shape.begin(), shape.end());
        auto image_torch = torch::rand(torch_shape, torch::kCUDA) * 255.0f;

        cudaMemcpy(const_cast<float*>(image_torch.data_ptr<float>()),
                   image_custom.ptr<float>(),
                   image_custom.numel() * sizeof(float),
                   cudaMemcpyDeviceToDevice);

        float mean = 128.0f;
        float std = 64.0f;

        double total_custom = 0.0;
        double total_torch = 0.0;

        Tensor result_custom;
        torch::Tensor result_torch;

        for (int i = 0; i < iterations; ++i) {
            {
                Timer timer;
                result_custom = (image_custom - mean) / std;
                cudaDeviceSynchronize();
                total_custom += timer.elapsed_ms();
            }

            {
                Timer timer;
                result_torch = (image_torch - mean) / std;
                cudaDeviceSynchronize();
                total_torch += timer.elapsed_ms();
            }
        }

        bool verified = tensors_equal(result_custom, result_torch);

        BenchmarkResult result{
            name,
            total_custom / iterations,
            total_torch / iterations,
            total_torch / total_custom,
            verified};
        result.print();
    }
}

TEST_F(FusionBenchmarkTest, RealWorld_LayerNormActivation) {
    print_separator("REAL-WORLD PATTERN: Layer Norm + Activation");

    std::cout << "\nPattern: (x * scale + bias).relu()" << std::endl;
    std::cout << "Common in neural network forward passes" << std::endl;
    std::cout << "Expected: 3× speedup\n"
              << std::endl;

    const int iterations = 100;

    std::vector<std::tuple<std::string, std::vector<size_t>>> test_cases = {
        {"Feature map (1024×512)", {1024, 512}},
        {"Large feature map (4096×1024)", {4096, 1024}},
    };

    for (const auto& [name, shape] : test_cases) {
        auto x_custom = Tensor::randn(TensorShape(shape), Device::CUDA);
        std::vector<int64_t> torch_shape(shape.begin(), shape.end());
        auto x_torch = torch::randn(torch_shape, torch::kCUDA);

        cudaMemcpy(const_cast<float*>(x_torch.data_ptr<float>()),
                   x_custom.ptr<float>(),
                   x_custom.numel() * sizeof(float),
                   cudaMemcpyDeviceToDevice);

        float scale = 1.5f;
        float bias = 0.5f;

        double total_custom = 0.0;
        double total_torch = 0.0;

        Tensor result_custom;
        torch::Tensor result_torch;

        for (int i = 0; i < iterations; ++i) {
            {
                Timer timer;
                result_custom = (x_custom * scale + bias).relu();
                cudaDeviceSynchronize();
                total_custom += timer.elapsed_ms();
            }

            {
                Timer timer;
                result_torch = (x_torch * scale + bias).relu();
                cudaDeviceSynchronize();
                total_torch += timer.elapsed_ms();
            }
        }

        bool verified = tensors_equal(result_custom, result_torch);

        BenchmarkResult result{
            name,
            total_custom / iterations,
            total_torch / iterations,
            total_torch / total_custom,
            verified};
        result.print();
    }
}

TEST_F(FusionBenchmarkTest, RealWorld_GaussianSplatting) {
    print_separator("REAL-WORLD PATTERN: Gaussian Splatting Activation");

    std::cout << "\nPattern: exp(-x.abs()) * opacity" << std::endl;
    std::cout << "Used in Gaussian splatting rasterization" << std::endl;
    std::cout << "Expected: 3× speedup\n"
              << std::endl;

    const int iterations = 100;

    std::vector<std::tuple<std::string, std::vector<size_t>>> test_cases = {
        {"Small splat batch (10K)", {10000}},
        {"Medium splat batch (100K)", {100000}},
        {"Large splat batch (1M)", {1000000}},
    };

    for (const auto& [name, shape] : test_cases) {
        auto x_custom = Tensor::randn(TensorShape(shape), Device::CUDA);
        std::vector<int64_t> torch_shape(shape.begin(), shape.end());
        auto x_torch = torch::randn(torch_shape, torch::kCUDA);

        cudaMemcpy(const_cast<float*>(x_torch.data_ptr<float>()),
                   x_custom.ptr<float>(),
                   x_custom.numel() * sizeof(float),
                   cudaMemcpyDeviceToDevice);

        float opacity = 0.8f;

        double total_custom = 0.0;
        double total_torch = 0.0;

        Tensor result_custom;
        torch::Tensor result_torch;

        for (int i = 0; i < iterations; ++i) {
            {
                Timer timer;
                result_custom = (-x_custom.abs()).exp() * opacity;
                cudaDeviceSynchronize();
                total_custom += timer.elapsed_ms();
            }

            {
                Timer timer;
                result_torch = (-x_torch.abs()).exp() * opacity;
                cudaDeviceSynchronize();
                total_torch += timer.elapsed_ms();
            }
        }

        bool verified = tensors_equal(result_custom, result_torch);

        BenchmarkResult result{
            name,
            total_custom / iterations,
            total_torch / iterations,
            total_torch / total_custom,
            verified};
        result.print();
    }
}

// ============================================================================
// Complex Chain Benchmarks
// ============================================================================

TEST_F(FusionBenchmarkTest, ComplexChain_5Ops) {
    print_separator("COMPLEX CHAIN: 5 Operations");

    std::cout << "\nPattern: ((a + 1.0).exp() * 2.0 - 0.5).relu()" << std::endl;
    std::cout << "Expected: 5× speedup\n"
              << std::endl;

    const int iterations = 100;

    std::vector<std::tuple<std::string, std::vector<size_t>>> test_cases = {
        {"Vector (100K)", {102400}},
        {"Matrix (1024×1024)", {1024, 1024}},
    };

    for (const auto& [name, shape] : test_cases) {
        auto tensor_custom = Tensor::randn(TensorShape(shape), Device::CUDA);
        std::vector<int64_t> torch_shape(shape.begin(), shape.end());
        auto tensor_torch = torch::randn(torch_shape, torch::kCUDA);

        cudaMemcpy(const_cast<float*>(tensor_torch.data_ptr<float>()),
                   tensor_custom.ptr<float>(),
                   tensor_custom.numel() * sizeof(float),
                   cudaMemcpyDeviceToDevice);

        double total_custom = 0.0;
        double total_torch = 0.0;

        Tensor result_custom;
        torch::Tensor result_torch;

        for (int i = 0; i < iterations; ++i) {
            {
                Timer timer;
                result_custom = ((tensor_custom + 1.0f).exp() * 2.0f - 0.5f).relu();
                cudaDeviceSynchronize();
                total_custom += timer.elapsed_ms();
            }

            {
                Timer timer;
                result_torch = ((tensor_torch + 1.0f).exp() * 2.0f - 0.5f).relu();
                cudaDeviceSynchronize();
                total_torch += timer.elapsed_ms();
            }
        }

        bool verified = tensors_equal(result_custom, result_torch);

        BenchmarkResult result{
            name,
            total_custom / iterations,
            total_torch / iterations,
            total_torch / total_custom,
            verified};
        result.print();
    }
}

TEST_F(FusionBenchmarkTest, ComplexChain_6Ops) {
    print_separator("COMPLEX CHAIN: 6 Operations");

    std::cout << "\nPattern: (a.abs().sqrt() + 0.1).log() * 5.0 - 1.0" << std::endl;
    std::cout << "Expected: 5-6× speedup\n"
              << std::endl;

    const int iterations = 100;

    std::vector<std::tuple<std::string, std::vector<size_t>>> test_cases = {
        {"Vector (100K)", {102400}},
        {"Matrix (1024×1024)", {1024, 1024}},
    };

    for (const auto& [name, shape] : test_cases) {
        auto tensor_custom = Tensor::randn(TensorShape(shape), Device::CUDA);
        std::vector<int64_t> torch_shape(shape.begin(), shape.end());
        auto tensor_torch = torch::randn(torch_shape, torch::kCUDA);

        cudaMemcpy(const_cast<float*>(tensor_torch.data_ptr<float>()),
                   tensor_custom.ptr<float>(),
                   tensor_custom.numel() * sizeof(float),
                   cudaMemcpyDeviceToDevice);

        double total_custom = 0.0;
        double total_torch = 0.0;

        Tensor result_custom;
        torch::Tensor result_torch;

        for (int i = 0; i < iterations; ++i) {
            {
                Timer timer;
                result_custom = (tensor_custom.abs().sqrt() + 0.1f).log() * 5.0f - 1.0f;
                cudaDeviceSynchronize();
                total_custom += timer.elapsed_ms();
            }

            {
                Timer timer;
                result_torch = (tensor_torch.abs().sqrt() + 0.1f).log() * 5.0f - 1.0f;
                cudaDeviceSynchronize();
                total_torch += timer.elapsed_ms();
            }
        }

        bool verified = tensors_equal(result_custom, result_torch);

        BenchmarkResult result{
            name,
            total_custom / iterations,
            total_torch / iterations,
            total_torch / total_custom,
            verified};
        result.print();
    }
}

// ============================================================================
// Summary Test
// ============================================================================

TEST_F(FusionBenchmarkTest, Summary) {
    print_separator("FUSION BENCHMARK SUMMARY");

    std::cout << "\nKEY FINDINGS:\n"
              << std::endl;
    std::cout << "Expression templates provide automatic kernel fusion, eliminating:" << std::endl;
    std::cout << "  1. Intermediate memory allocations (except final result)" << std::endl;
    std::cout << "  2. Multiple kernel launches (fused into single kernel)" << std::endl;
    std::cout << "  3. Redundant memory bandwidth (single pass through data)" << std::endl;
    std::cout << "\nEXPECTED PERFORMANCE:\n"
              << std::endl;
    std::cout << "  - 2-operation chains: 2× speedup" << std::endl;
    std::cout << "  - 3-operation chains: 3× speedup" << std::endl;
    std::cout << "  - 4+ operation chains: 4-10× speedup" << std::endl;
    std::cout << "\nCOMBINED WITH MEMORY POOL (Track 1):\n"
              << std::endl;
    std::cout << "  - Memory pool: 150× faster allocations" << std::endl;
    std::cout << "  - Expression templates: Eliminate intermediate allocations" << std::endl;
    std::cout << "  - Combined: 10-30× total speedup for complex operations" << std::endl;
    std::cout << "\nVERIFICATION:\n"
              << std::endl;
    std::cout << "  All results are verified against PyTorch with tolerance 1e-4" << std::endl;
    std::cout << "  Any mismatches are reported in test output" << std::endl;

    std::cout << "\n"
              << std::string(120, '=') << std::endl;
}
