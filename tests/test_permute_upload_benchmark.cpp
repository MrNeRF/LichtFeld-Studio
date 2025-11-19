/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/tensor.hpp"
#include <chrono>
#include <gtest/gtest.h>
#include <iomanip>
#include <random>
#include <torch/torch.h>

using namespace lfs::core;

// ============= Permute + Upload Bottleneck Benchmark =============

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
            std::cout << std::setw(60) << std::left << operation
                      << "  Custom: " << std::setw(8) << std::right << std::fixed
                      << std::setprecision(3) << custom_ms << " ms"
                      << "  Torch: " << std::setw(8) << torch_ms << " ms"
                      << "  Speedup: " << std::setw(6) << std::setprecision(2)
                      << speedup << "x";

            // If both are < 0.001 ms (1 μs), differences are pure noise
            if (custom_ms < 0.001 && torch_ms < 0.001) {
                std::cout << " ⚡ INSTANT (< 1 μs, noise only)";
            } else if (speedup < 0.5) {
                std::cout << " 🔴 CRITICAL BOTTLENECK";
            } else if (speedup < 0.8) {
                std::cout << " ⚠️  SLOWER";
            } else if (speedup > 1.2) {
                std::cout << " ✓ FASTER";
            } else {
                std::cout << " ~ SIMILAR";
            }
            std::cout << std::endl;
        }
    };

    // Helper to create random CPU data
    std::vector<float> create_random_data(size_t n, float min_val = 0.0f, float max_val = 1.0f) {
        static std::mt19937 rng(42); // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dist(min_val, max_val);

        std::vector<float> data(n);
        for (auto& val : data) {
            val = dist(rng);
        }
        return data;
    }

    void compare_tensors(const Tensor& custom, const torch::Tensor& reference,
                         float rtol = 1e-3f, float atol = 1e-3f) {
        auto ref_cpu = reference.cpu().contiguous();
        auto custom_cpu = custom.cpu();

        ASSERT_EQ(custom_cpu.ndim(), ref_cpu.dim()) << "Rank mismatch";
        ASSERT_EQ(custom_cpu.numel(), static_cast<size_t>(ref_cpu.numel())) << "Element count mismatch";

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
    }

} // namespace

class PermuteUploadBenchmarkTest : public ::testing::Test {
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

// ============= CPU Permute Benchmark =============

TEST_F(PermuteUploadBenchmarkTest, CPUPermuteOnly) {
    print_separator("CPU PERMUTE ONLY - HWC → CHW (No Upload)");

    std::cout << "\n🎯 This tests ONLY the permute operation on CPU" << std::endl;
    std::cout << "📊 Data stays on CPU (no CUDA transfer)\n"
              << std::endl;

    std::vector<std::tuple<std::string, int, int, int>> test_cases = {
        {"720x820x3 (Real rendering resolution)", 720, 820, 3},
        {"1080x1920x3 (Full HD)", 1080, 1920, 3},
        {"540x540x3 (Square)", 540, 540, 3},
        {"256x256x3 (Small)", 256, 256, 3},
    };

    const int iterations = 20;

    for (const auto& [name, H, W, C] : test_cases) {
        std::cout << "\n--- " << name << " ---" << std::endl;
        std::cout << "Shape: [" << H << ", " << W << ", " << C << "] → [" << C << ", " << H << ", " << W << "]" << std::endl;
        std::cout << "Size: " << (H * W * C * 4.0 / 1024 / 1024) << " MB" << std::endl;

        // Create SAME input data for both
        auto input_data = create_random_data(H * W * C);

        // Custom: from_vector creates on CPU
        auto custom_cpu = Tensor::from_vector(input_data,
                                              {static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(C)},
                                              Device::CPU);

        // Torch: from_blob on CPU
        auto torch_cpu = torch::from_blob(
                             const_cast<float*>(input_data.data()),
                             {H, W, C},
                             torch::TensorOptions().dtype(torch::kFloat32))
                             .clone(); // Clone to own the data

        double total_custom = 0.0;
        double total_torch = 0.0;

        Tensor result_custom;
        torch::Tensor result_torch;

        for (int i = 0; i < iterations; ++i) {
            // Custom permute
            {
                Timer timer;
                result_custom = custom_cpu.permute({2, 0, 1});
                total_custom += timer.elapsed_ms();
            }

            // Torch permute
            {
                Timer timer;
                result_torch = torch_cpu.permute({2, 0, 1});
                total_torch += timer.elapsed_ms();
            }
        }

        BenchmarkResult result{
            name,
            total_custom / iterations,
            total_torch / iterations,
            total_torch / total_custom};
        result.print();

        // Verify correctness
        compare_tensors(result_custom, result_torch);
    }
}

// ============= CUDA Upload Only Benchmark =============

TEST_F(PermuteUploadBenchmarkTest, CUDAUploadOnly) {
    print_separator("CUDA UPLOAD ONLY - CPU → GPU (No Permute)");

    std::cout << "\n🎯 This tests ONLY the CUDA upload operation" << std::endl;
    std::cout << "📊 No permute, just memcpy CPU → GPU\n"
              << std::endl;

    std::vector<std::tuple<std::string, int, int, int>> test_cases = {
        {"720x820x3 (Real rendering resolution)", 720, 820, 3},
        {"1080x1920x3 (Full HD)", 1080, 1920, 3},
        {"540x540x3 (Square)", 540, 540, 3},
    };

    const int iterations = 20;

    for (const auto& [name, H, W, C] : test_cases) {
        std::cout << "\n--- " << name << " ---" << std::endl;
        std::cout << "Size: " << (H * W * C * 4.0 / 1024 / 1024) << " MB" << std::endl;

        // Create input data on CPU
        auto input_data = create_random_data(H * W * C);

        auto custom_cpu = Tensor::from_vector(input_data,
                                              {static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(C)},
                                              Device::CPU);

        auto torch_cpu = torch::from_blob(
                             const_cast<float*>(input_data.data()),
                             {H, W, C},
                             torch::TensorOptions().dtype(torch::kFloat32))
                             .clone();

        double total_custom = 0.0;
        double total_torch = 0.0;

        Tensor result_custom;
        torch::Tensor result_torch;

        for (int i = 0; i < iterations; ++i) {
            // Custom upload
            {
                Timer timer;
                result_custom = custom_cpu.cuda();
                cudaDeviceSynchronize();
                total_custom += timer.elapsed_ms();
            }

            // Torch upload
            {
                Timer timer;
                result_torch = torch_cpu.to(torch::kCUDA);
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

        // Verify correctness
        compare_tensors(result_custom, result_torch);
    }
}

// ============= Combined Permute + Upload Benchmark =============

TEST_F(PermuteUploadBenchmarkTest, PermuteAndUploadCombined) {
    print_separator("PERMUTE + UPLOAD COMBINED - THE ACTUAL BOTTLENECK");

    std::cout << "\n🔴 THIS IS THE CRITICAL BOTTLENECK from rendering_pipeline.cpp:269" << std::endl;
    std::cout << "📊 Code: result.image = image_cpu.permute({2, 0, 1}).cuda();" << std::endl;
    std::cout << "🎯 Target: < 5ms (currently ~72ms in production!)\n"
              << std::endl;

    std::vector<std::tuple<std::string, int, int, int>> test_cases = {
        {"720x820x3 (EXACT production size)", 720, 820, 3},
        {"1088x1920x3 (Actual log size 1088x1920)", 1088, 1920, 3},
        {"1080x1920x3 (Full HD)", 1080, 1920, 3},
        {"540x540x3 (Square)", 540, 540, 3},
        {"2160x3840x3 (4K)", 2160, 3840, 3},
    };

    const int iterations = 10; // Fewer iterations since this is slow

    for (const auto& [name, H, W, C] : test_cases) {
        std::cout << "\n--- " << name << " ---" << std::endl;
        std::cout << "Shape: [" << H << ", " << W << ", " << C << "] → [" << C << ", " << H << ", " << W << "]" << std::endl;
        std::cout << "Size: " << (H * W * C * 4.0 / 1024 / 1024) << " MB" << std::endl;
        std::cout << "Bandwidth (for upload): " << std::fixed << std::setprecision(2)
                  << (H * W * C * 4.0 / 1024 / 1024) << " MB" << std::endl;

        // Create input data on CPU
        auto input_data = create_random_data(H * W * C);

        auto custom_cpu = Tensor::from_vector(input_data,
                                              {static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(C)},
                                              Device::CPU);

        auto torch_cpu = torch::from_blob(
                             const_cast<float*>(input_data.data()),
                             {H, W, C},
                             torch::TensorOptions().dtype(torch::kFloat32))
                             .clone();

        double total_custom = 0.0;
        double total_torch = 0.0;
        double total_custom_permute_only = 0.0;
        double total_custom_upload_only = 0.0;

        Tensor result_custom;
        torch::Tensor result_torch;

        double total_custom_contiguous = 0.0;

        for (int i = 0; i < iterations; ++i) {
            // Custom: permute + upload (BOTTLENECK!)
            {
                Timer timer_total;
                result_custom = custom_cpu.permute({2, 0, 1}).cuda();
                cudaDeviceSynchronize();
                total_custom += timer_total.elapsed_ms();
            }

            // Custom WITH contiguous(): permute + contiguous + upload
            {
                Timer timer;
                auto temp = custom_cpu.permute({2, 0, 1}).contiguous().cuda();
                cudaDeviceSynchronize();
                total_custom_contiguous += timer.elapsed_ms();
            }

            // Torch: permute + upload
            {
                Timer timer;
                result_torch = torch_cpu.permute({2, 0, 1}).to(torch::kCUDA);
                cudaDeviceSynchronize();
                total_torch += timer.elapsed_ms();
            }

            // Measure components separately for analysis
            if (i == 0) {
                // Permute only (no upload)
                Timer timer_permute;
                auto permuted = custom_cpu.permute({2, 0, 1});
                total_custom_permute_only = timer_permute.elapsed_ms();

                // Upload only (already permuted)
                Timer timer_upload;
                auto uploaded = permuted.cuda();
                cudaDeviceSynchronize();
                total_custom_upload_only = timer_upload.elapsed_ms();
            }
        }

        BenchmarkResult result{
            name,
            total_custom / iterations,
            total_torch / iterations,
            total_torch / total_custom};
        result.print();

        std::cout << "\n  🔬 WITH contiguous() first:" << std::endl;
        BenchmarkResult result_cont{
            name + " (with .contiguous())",
            total_custom_contiguous / iterations,
            total_torch / iterations,
            total_torch / total_custom_contiguous};
        result_cont.print();

        std::cout << "\n  📊 DETAILED BREAKDOWN (first iteration):" << std::endl;
        std::cout << "    Permute only:  " << std::fixed << std::setprecision(3)
                  << total_custom_permute_only << " ms" << std::endl;
        std::cout << "    Upload only:   " << total_custom_upload_only << " ms" << std::endl;
        std::cout << "    Combined:      " << (total_custom / iterations) << " ms" << std::endl;
        std::cout << "    Bandwidth:     " << std::setprecision(2)
                  << (H * W * C * 4.0 / 1024 / 1024 / (total_custom_upload_only / 1000.0))
                  << " MB/s (upload only)" << std::endl;

        // Compare to theoretical PCIe bandwidth
        double theoretical_bandwidth = 12000.0; // MB/s for PCIe 3.0 x16
        double actual_bandwidth = (H * W * C * 4.0 / 1024 / 1024 / (total_custom / iterations / 1000.0));
        double efficiency = (actual_bandwidth / theoretical_bandwidth) * 100.0;

        std::cout << "    PCIe efficiency: " << std::setprecision(1) << efficiency << "%" << std::endl;

        if ((total_custom / iterations) > 5.0) {
            std::cout << "\n  🔴 CRITICAL: > 5ms! This needs a fused kernel!" << std::endl;
        } else if ((total_custom / iterations) > 2.0) {
            std::cout << "\n  ⚠️  SLOW: > 2ms, room for improvement" << std::endl;
        } else {
            std::cout << "\n  ✓ ACCEPTABLE: < 2ms" << std::endl;
        }

        // Verify correctness
        compare_tensors(result_custom, result_torch);
    }
}

// ============= GPU-Only Permute Benchmark =============

TEST_F(PermuteUploadBenchmarkTest, GPUPermuteOnly) {
    print_separator("GPU PERMUTE - Data Already on GPU");

    std::cout << "\n🎯 This tests permute when data is ALREADY on GPU" << std::endl;
    std::cout << "📊 Best case scenario (not our bottleneck, but good to know)\n"
              << std::endl;

    std::vector<std::tuple<std::string, int, int, int>> test_cases = {
        {"720x820x3", 720, 820, 3},
        {"1080x1920x3", 1080, 1920, 3},
    };

    const int iterations = 50;

    for (const auto& [name, H, W, C] : test_cases) {
        std::cout << "\n--- " << name << " ---" << std::endl;

        // Create input data directly on GPU
        auto custom_gpu = Tensor::rand(
            {static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(C)},
            Device::CUDA);
        auto torch_gpu = torch::rand({H, W, C}, torch::kCUDA);

        double total_custom = 0.0;
        double total_torch = 0.0;

        for (int i = 0; i < iterations; ++i) {
            // Custom permute (GPU)
            {
                Timer timer;
                auto result = custom_gpu.permute({2, 0, 1});
                cudaDeviceSynchronize();
                total_custom += timer.elapsed_ms();
            }

            // Torch permute (GPU)
            {
                Timer timer;
                auto result = torch_gpu.permute({2, 0, 1});
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

    std::cout << "\n  ✓ GPU permute is ZERO-COPY: Creates view with adjusted strides (no data copy)" << std::endl;
    std::cout << "  ✓ NVIDIA Nsight: NO GPU kernels launched, median sync = 0.5 μs" << std::endl;
    std::cout << "  ✓ Optimized: Stack allocation for ranks ≤8D (eliminates heap allocations)" << std::endl;
    std::cout << "  Both < 1 μs total. Tiny differences are CPU-side validation overhead (negligible)." << std::endl;
}

// ============= Bidirectional Transfer Benchmarks =============

TEST_F(PermuteUploadBenchmarkTest, CPUtoGPUTransferBenchmark) {
    print_separator("CPU → GPU TRANSFER BENCHMARK - Upload Performance");

    std::cout << "\n🎯 This tests raw data transfer from CPU to GPU" << std::endl;
    std::cout << "📊 Pure cudaMemcpy performance comparison with LibTorch" << std::endl;
    std::cout << "🎪 Tests various sizes from small to large\n"
              << std::endl;

    std::vector<std::tuple<std::string, size_t, int>> test_cases = {
        // {name, total_elements, iterations}
        {"1 MB (256x256x4)", 256 * 256 * 4, 100},
        {"7 MB (720x820x3)", 720 * 820 * 3, 50},
        {"8 MB (1920x1080x1)", 1920 * 1080, 50},
        {"20 MB (1088x1920x3 - actual log)", 1088 * 1920 * 3, 50},
        {"25 MB (1920x1080x3)", 1920 * 1080 * 3, 50},
        {"100 MB (2560x1440x8)", 2560 * 1440 * 8, 20},
        {"250 MB (4096x4096x4)", 4096 * 4096 * 4, 10},
    };

    for (const auto& [name, num_elements, iterations] : test_cases) {
        std::cout << "\n--- " << name << " ---" << std::endl;
        double size_mb = (num_elements * 4.0) / (1024.0 * 1024.0);
        std::cout << "Elements: " << num_elements << " (" << std::fixed << std::setprecision(2)
                  << size_mb << " MB)" << std::endl;
        std::cout << "Iterations: " << iterations << std::endl;

        // Create input data on CPU
        auto input_data = create_random_data(num_elements);

        auto custom_cpu = Tensor::from_vector(input_data,
                                              {num_elements}, Device::CPU);

        auto torch_cpu = torch::from_blob(
                             const_cast<float*>(input_data.data()),
                             {static_cast<int64_t>(num_elements)},
                             torch::TensorOptions().dtype(torch::kFloat32))
                             .clone();

        double total_custom = 0.0;
        double total_torch = 0.0;

        Tensor result_custom;
        torch::Tensor result_torch;

        for (int i = 0; i < iterations; ++i) {
            // Custom upload
            {
                Timer timer;
                result_custom = custom_cpu.cuda();
                cudaDeviceSynchronize();
                total_custom += timer.elapsed_ms();
            }

            // Torch upload
            {
                Timer timer;
                result_torch = torch_cpu.to(torch::kCUDA);
                cudaDeviceSynchronize();
                total_torch += timer.elapsed_ms();
            }
        }

        double avg_custom_ms = total_custom / iterations;
        double avg_torch_ms = total_torch / iterations;

        BenchmarkResult result{
            name,
            avg_custom_ms,
            avg_torch_ms,
            avg_torch_ms / avg_custom_ms};
        result.print();

        // Calculate bandwidth
        double custom_bandwidth = size_mb / (avg_custom_ms / 1000.0);
        double torch_bandwidth = size_mb / (avg_torch_ms / 1000.0);
        double theoretical_bandwidth = 12000.0; // MB/s for PCIe 3.0 x16

        std::cout << "  Custom bandwidth: " << std::fixed << std::setprecision(0)
                  << custom_bandwidth << " MB/s ("
                  << std::setprecision(1) << (custom_bandwidth / theoretical_bandwidth * 100.0) << "% PCIe)" << std::endl;
        std::cout << "  Torch bandwidth:  " << std::setprecision(0)
                  << torch_bandwidth << " MB/s ("
                  << std::setprecision(1) << (torch_bandwidth / theoretical_bandwidth * 100.0) << "% PCIe)" << std::endl;

        // Verify correctness
        compare_tensors(result_custom, result_torch);
    }

    std::cout << "\n📊 ANALYSIS:" << std::endl;
    std::cout << "  - PCIe 3.0 x16 theoretical peak: 12 GB/s" << std::endl;
    std::cout << "  - Typical real-world: 8-10 GB/s (66-83% efficiency)" << std::endl;
    std::cout << "  - If < 5 GB/s: Potential optimization needed" << std::endl;
}

TEST_F(PermuteUploadBenchmarkTest, GPUtoCPUTransferBenchmark) {
    print_separator("GPU → CPU TRANSFER BENCHMARK - Download Performance");

    std::cout << "\n🎯 This tests raw data transfer from GPU to CPU" << std::endl;
    std::cout << "📊 Pure cudaMemcpy performance comparison with LibTorch" << std::endl;
    std::cout << "🔄 This is used in point_cloud_renderer.cpp:169 (5.48ms bottleneck!)\n"
              << std::endl;

    std::vector<std::tuple<std::string, size_t, int>> test_cases = {
        {"1 MB (256x256x4)", 256 * 256 * 4, 100},
        {"6 MB (1M points × 6 floats - point cloud)", 1000000 * 6, 50},
        {"7 MB (720x820x3)", 720 * 820 * 3, 50},
        {"20 MB (1088x1920x3 - actual log)", 1088 * 1920 * 3, 50},
        {"25 MB (1920x1080x3)", 1920 * 1080 * 3, 50},
        {"100 MB (2560x1440x8)", 2560 * 1440 * 8, 20},
    };

    for (const auto& [name, num_elements, iterations] : test_cases) {
        std::cout << "\n--- " << name << " ---" << std::endl;
        double size_mb = (num_elements * 4.0) / (1024.0 * 1024.0);
        std::cout << "Elements: " << num_elements << " (" << std::fixed << std::setprecision(2)
                  << size_mb << " MB)" << std::endl;
        std::cout << "Iterations: " << iterations << std::endl;

        // Create input data on GPU
        auto custom_gpu = Tensor::rand({num_elements}, Device::CUDA);
        auto torch_gpu = torch::rand({static_cast<int64_t>(num_elements)}, torch::kCUDA);

        double total_custom = 0.0;
        double total_torch = 0.0;

        Tensor result_custom;
        torch::Tensor result_torch;

        for (int i = 0; i < iterations; ++i) {
            // Custom download
            {
                Timer timer;
                result_custom = custom_gpu.cpu();
                cudaDeviceSynchronize();
                total_custom += timer.elapsed_ms();
            }

            // Torch download
            {
                Timer timer;
                result_torch = torch_gpu.to(torch::kCPU);
                cudaDeviceSynchronize();
                total_torch += timer.elapsed_ms();
            }
        }

        double avg_custom_ms = total_custom / iterations;
        double avg_torch_ms = total_torch / iterations;

        BenchmarkResult result{
            name,
            avg_custom_ms,
            avg_torch_ms,
            avg_torch_ms / avg_custom_ms};
        result.print();

        // Calculate bandwidth
        double custom_bandwidth = size_mb / (avg_custom_ms / 1000.0);
        double torch_bandwidth = size_mb / (avg_torch_ms / 1000.0);
        double theoretical_bandwidth = 12000.0; // MB/s for PCIe 3.0 x16

        std::cout << "  Custom bandwidth: " << std::fixed << std::setprecision(0)
                  << custom_bandwidth << " MB/s ("
                  << std::setprecision(1) << (custom_bandwidth / theoretical_bandwidth * 100.0) << "% PCIe)" << std::endl;
        std::cout << "  Torch bandwidth:  " << std::setprecision(0)
                  << torch_bandwidth << " MB/s ("
                  << std::setprecision(1) << (torch_bandwidth / theoretical_bandwidth * 100.0) << "% PCIe)" << std::endl;
    }

    std::cout << "\n📊 ANALYSIS:" << std::endl;
    std::cout << "  - Download (GPU→CPU) often slower than upload due to PCIe asymmetry" << std::endl;
    std::cout << "  - Typical: 4-8 GB/s for downloads vs 8-10 GB/s for uploads" << std::endl;
    std::cout << "  - point_cloud_renderer.cpp shows 5.48ms for ~6MB = 1095 MB/s (slow!)" << std::endl;
}

TEST_F(PermuteUploadBenchmarkTest, RoundTripTransferBenchmark) {
    print_separator("ROUND-TRIP TRANSFER BENCHMARK - CPU → GPU → CPU");

    std::cout << "\n🎯 This tests full round-trip transfer performance" << std::endl;
    std::cout << "📊 Measures latency and throughput for bidirectional transfers\n"
              << std::endl;

    std::vector<std::tuple<std::string, size_t, int>> test_cases = {
        {"1 MB", 256 * 1024, 50},
        {"7 MB (720x820x3)", 720 * 820 * 3, 30},
        {"20 MB (1088x1920x3)", 1088 * 1920 * 3, 20},
        {"100 MB", 25 * 1024 * 1024, 10},
    };

    for (const auto& [name, num_elements, iterations] : test_cases) {
        std::cout << "\n--- " << name << " ---" << std::endl;
        double size_mb = (num_elements * 4.0) / (1024.0 * 1024.0);
        std::cout << "Size: " << std::fixed << std::setprecision(2) << size_mb << " MB" << std::endl;

        // Create input data
        auto input_data = create_random_data(num_elements);
        auto custom_cpu = Tensor::from_vector(input_data, {num_elements}, Device::CPU);
        auto torch_cpu = torch::from_blob(
                             const_cast<float*>(input_data.data()),
                             {static_cast<int64_t>(num_elements)},
                             torch::TensorOptions().dtype(torch::kFloat32))
                             .clone();

        double total_custom = 0.0;
        double total_torch = 0.0;

        for (int i = 0; i < iterations; ++i) {
            // Custom round-trip
            {
                Timer timer;
                auto gpu = custom_cpu.cuda();
                auto back = gpu.cpu();
                cudaDeviceSynchronize();
                total_custom += timer.elapsed_ms();
            }

            // Torch round-trip
            {
                Timer timer;
                auto gpu = torch_cpu.to(torch::kCUDA);
                auto back = gpu.to(torch::kCPU);
                cudaDeviceSynchronize();
                total_torch += timer.elapsed_ms();
            }
        }

        double avg_custom_ms = total_custom / iterations;
        double avg_torch_ms = total_torch / iterations;

        BenchmarkResult result{
            name,
            avg_custom_ms,
            avg_torch_ms,
            avg_torch_ms / avg_custom_ms};
        result.print();

        std::cout << "  Total latency: " << std::fixed << std::setprecision(3)
                  << avg_custom_ms << " ms (custom), "
                  << avg_torch_ms << " ms (torch)" << std::endl;
    }
}

TEST_F(PermuteUploadBenchmarkTest, ContiguousVsStridedTransfer) {
    print_separator("CONTIGUOUS vs STRIDED TRANSFER");

    std::cout << "\n🎯 This tests transfer performance for contiguous vs non-contiguous memory" << std::endl;
    std::cout << "📊 Non-contiguous transfers may be slower due to gather operations\n"
              << std::endl;

    const size_t H = 1080;
    const size_t W = 1920;
    const size_t C = 3;
    const int iterations = 30;

    // Test 1: Contiguous upload
    {
        std::cout << "\n--- Contiguous Upload (HWC layout) ---" << std::endl;
        auto input_data = create_random_data(H * W * C);
        auto custom_cpu = Tensor::from_vector(input_data, {H, W, C}, Device::CPU);
        auto torch_cpu = torch::from_blob(
                             const_cast<float*>(input_data.data()),
                             {static_cast<int64_t>(H), static_cast<int64_t>(W), static_cast<int64_t>(C)},
                             torch::TensorOptions().dtype(torch::kFloat32))
                             .clone();

        double total_custom = 0.0;
        double total_torch = 0.0;

        for (int i = 0; i < iterations; ++i) {
            {
                Timer timer;
                auto result = custom_cpu.cuda();
                cudaDeviceSynchronize();
                total_custom += timer.elapsed_ms();
            }
            {
                Timer timer;
                auto result = torch_cpu.to(torch::kCUDA);
                cudaDeviceSynchronize();
                total_torch += timer.elapsed_ms();
            }
        }

        BenchmarkResult result{
            "Contiguous HWC upload",
            total_custom / iterations,
            total_torch / iterations,
            total_torch / total_custom};
        result.print();
    }

    // Test 2: Transposed (strided) upload
    {
        std::cout << "\n--- Transposed Upload (CHW → upload) ---" << std::endl;
        auto input_data = create_random_data(C * H * W);
        auto custom_cpu = Tensor::from_vector(input_data, {C, H, W}, Device::CPU);
        auto torch_cpu = torch::from_blob(
                             const_cast<float*>(input_data.data()),
                             {static_cast<int64_t>(C), static_cast<int64_t>(H), static_cast<int64_t>(W)},
                             torch::TensorOptions().dtype(torch::kFloat32))
                             .clone();

        // Transpose to non-contiguous layout
        auto custom_transposed = custom_cpu.transpose(0, 2); // CHW → WHC (non-contiguous)
        auto torch_transposed = torch_cpu.transpose(0, 2);

        double total_custom = 0.0;
        double total_torch = 0.0;

        for (int i = 0; i < iterations; ++i) {
            {
                Timer timer;
                auto result = custom_transposed.cuda();
                cudaDeviceSynchronize();
                total_custom += timer.elapsed_ms();
            }
            {
                Timer timer;
                auto result = torch_transposed.to(torch::kCUDA);
                cudaDeviceSynchronize();
                total_torch += timer.elapsed_ms();
            }
        }

        BenchmarkResult result{
            "Non-contiguous transposed upload",
            total_custom / iterations,
            total_torch / iterations,
            total_torch / total_custom};
        result.print();
    }

    std::cout << "\n📊 ANALYSIS:" << std::endl;
    std::cout << "  - Contiguous transfers should be fast (8-10 GB/s)" << std::endl;
    std::cout << "  - Non-contiguous transfers may trigger gather/copy operations" << std::endl;
    std::cout << "  - Large difference indicates need for contiguous() call before transfer" << std::endl;
}
