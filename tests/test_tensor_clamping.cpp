/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/tensor.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <torch/torch.h>

using namespace lfs::core;

// ============= Helper Functions =============

namespace {

    torch::Tensor to_torch(const Tensor& t) {
        auto options = torch::TensorOptions()
                           .dtype([&]() {
                               switch (t.dtype()) {
                               case DataType::Float32: return torch::kFloat32;
                               case DataType::Float16: return torch::kFloat16;
                               case DataType::Int32: return torch::kInt32;
                               case DataType::Int64: return torch::kInt64;
                               case DataType::UInt8: return torch::kUInt8;
                               case DataType::Bool: return torch::kBool;
                               default: return torch::kFloat32;
                               }
                           }())
                           .device(t.device() == Device::CPU ? torch::kCPU : torch::kCUDA);

        std::vector<int64_t> shape;
        for (size_t i = 0; i < t.ndim(); ++i) {
            shape.push_back(static_cast<int64_t>(t.size(i)));
        }

        torch::Tensor result = torch::empty(shape, options);

        if (t.device() == Device::CPU) {
            std::memcpy(result.data_ptr(), t.raw_ptr(), t.bytes());
        } else {
            cudaMemcpy(result.data_ptr(), t.raw_ptr(), t.bytes(), cudaMemcpyDeviceToDevice);
        }

        return result;
    }

    Tensor from_torch(const torch::Tensor& t, Device device = Device::CPU) {
        auto t_cont = t.contiguous();

        DataType dtype;
        switch (t_cont.scalar_type()) {
        case torch::kFloat32: dtype = DataType::Float32; break;
        case torch::kFloat16: dtype = DataType::Float16; break;
        case torch::kInt32: dtype = DataType::Int32; break;
        case torch::kInt64: dtype = DataType::Int64; break;
        case torch::kUInt8: dtype = DataType::UInt8; break;
        case torch::kBool: dtype = DataType::Bool; break;
        default: dtype = DataType::Float32; break;
        }

        std::vector<size_t> shape;
        for (int64_t i = 0; i < t_cont.dim(); ++i) {
            shape.push_back(static_cast<size_t>(t_cont.size(i)));
        }

        Tensor result = Tensor::empty(TensorShape(shape), device, dtype);

        if (device == Device::CPU) {
            std::memcpy(result.raw_ptr(), t_cont.data_ptr(), result.bytes());
        } else {
            if (t_cont.is_cpu()) {
                cudaMemcpy(result.raw_ptr(), t_cont.data_ptr(), result.bytes(), cudaMemcpyHostToDevice);
            } else {
                cudaMemcpy(result.raw_ptr(), t_cont.data_ptr(), result.bytes(), cudaMemcpyDeviceToDevice);
            }
        }

        return result;
    }

    void compare_tensors(const Tensor& custom, const torch::Tensor& reference,
                         float rtol = 1e-5f, float atol = 1e-7f, const std::string& msg = "") {
        auto ref_cpu = reference.cpu();
        auto custom_cpu = custom.cpu();

        ASSERT_EQ(custom_cpu.ndim(), ref_cpu.dim()) << msg << ": Rank mismatch";

        for (size_t i = 0; i < custom_cpu.ndim(); ++i) {
            ASSERT_EQ(custom_cpu.size(i), static_cast<size_t>(ref_cpu.size(i)))
                << msg << ": Shape mismatch at dim " << i;
        }

        ASSERT_EQ(custom_cpu.numel(), static_cast<size_t>(ref_cpu.numel()))
            << msg << ": Element count mismatch";

        if (custom_cpu.dtype() == DataType::Bool || ref_cpu.scalar_type() == torch::kBool) {
            auto custom_vec = custom_cpu.to_vector_bool();
            auto ref_ptr = ref_cpu.data_ptr<bool>();
            for (size_t i = 0; i < custom_vec.size(); ++i) {
                EXPECT_EQ(custom_vec[i], ref_ptr[i]) << msg << ": Mismatch at index " << i;
            }
        } else if (custom_cpu.dtype() == DataType::Int32 || custom_cpu.dtype() == DataType::Int64) {
            auto custom_vec = custom_cpu.to_vector_int();
            if (ref_cpu.scalar_type() == torch::kInt32) {
                auto ref_ptr = ref_cpu.data_ptr<int32_t>();
                for (size_t i = 0; i < custom_vec.size(); ++i) {
                    EXPECT_EQ(custom_vec[i], ref_ptr[i]) << msg << ": Mismatch at index " << i;
                }
            } else if (ref_cpu.scalar_type() == torch::kInt64) {
                auto ref_ptr = ref_cpu.data_ptr<int64_t>();
                for (size_t i = 0; i < custom_vec.size(); ++i) {
                    EXPECT_EQ(custom_vec[i], static_cast<int>(ref_ptr[i]))
                        << msg << ": Mismatch at index " << i;
                }
            }
        } else {
            auto custom_vec = custom_cpu.to_vector();
            auto ref_ptr = ref_cpu.data_ptr<float>();
            for (size_t i = 0; i < custom_vec.size(); ++i) {
                if (std::isnan(ref_ptr[i])) {
                    EXPECT_TRUE(std::isnan(custom_vec[i])) << msg << ": Expected NaN at index " << i;
                } else if (std::isinf(ref_ptr[i])) {
                    EXPECT_TRUE(std::isinf(custom_vec[i])) << msg << ": Expected Inf at index " << i;
                } else {
                    float diff = std::abs(custom_vec[i] - ref_ptr[i]);
                    float threshold = atol + rtol * std::abs(ref_ptr[i]);
                    EXPECT_LE(diff, threshold)
                        << msg << ": Mismatch at index " << i
                        << " (custom=" << custom_vec[i] << ", ref=" << ref_ptr[i] << ")";
                }
            }
        }
    }

    // Helper to check CUDA availability
    bool is_cuda_available() {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            return false;
        }

        // Try to initialize device 0
        err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            return false;
        }

        // Try to allocate and free memory to verify device works
        void* test_ptr = nullptr;
        err = cudaMalloc(&test_ptr, 1024);
        if (err != cudaSuccess) {
            return false;
        }
        cudaFree(test_ptr);

        // Synchronize to catch any lingering errors
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            return false;
        }

        return true;
    }
} // anonymous namespace

class TensorClampTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Clear any previous CUDA errors
        cudaGetLastError();

        Tensor::manual_seed(42);
        torch::manual_seed(42);

        // Initialize CUDA device if available
        if (torch::cuda::is_available()) {
            // Use the correct PyTorch API
            cudaSetDevice(0);
        }
    }
};

// ============= Clamp (non-inplace) Tests =============

TEST_F(TensorClampTest, ClampBasic) {
    std::vector<float> data = {-5.0f, -2.0f, 0.0f, 2.0f, 5.0f};

    auto t_custom = Tensor::from_vector(data, {5}, Device::CPU);
    auto t_torch = torch::tensor(data);

    auto result_custom = t_custom.clamp(-1.0f, 3.0f);
    auto result_torch = t_torch.clamp(-1.0f, 3.0f);

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "ClampBasic");

    // Original should be unchanged
    compare_tensors(t_custom, t_torch, 1e-5f, 1e-7f, "ClampBasicOriginal");
}

TEST_F(TensorClampTest, ClampMinOnly) {
    std::vector<float> data = {-5.0f, -2.0f, 0.0f, 2.0f, 5.0f};

    auto t_custom = Tensor::from_vector(data, {5}, Device::CPU);
    auto t_torch = torch::tensor(data);

    auto result_custom = t_custom.clamp_min(0.0f);
    auto result_torch = t_torch.clamp_min(0.0f);

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "ClampMin");
}

TEST_F(TensorClampTest, ClampMaxOnly) {
    std::vector<float> data = {-5.0f, -2.0f, 0.0f, 2.0f, 5.0f};

    auto t_custom = Tensor::from_vector(data, {5}, Device::CPU);
    auto t_torch = torch::tensor(data);

    auto result_custom = t_custom.clamp_max(3.0f);
    auto result_torch = t_torch.clamp_max(3.0f);

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "ClampMax");
}

// ============= Clamp_ (in-place) Tests =============

TEST_F(TensorClampTest, ClampInPlaceBasic) {
    std::vector<float> data = {-5.0f, -2.0f, 0.0f, 2.0f, 5.0f};

    auto t_custom = Tensor::from_vector(data, {5}, Device::CPU);
    auto t_torch = torch::tensor(data);

    auto& result_custom = t_custom.clamp_(-1.0f, 3.0f);
    t_torch.clamp_(-1.0f, 3.0f);

    EXPECT_EQ(&result_custom, &t_custom); // Should return reference to same object

    compare_tensors(t_custom, t_torch, 1e-5f, 1e-7f, "ClampInPlace");
}

TEST_F(TensorClampTest, ClampMinInPlace) {
    std::vector<float> data = {-5.0f, -2.0f, 0.0f, 2.0f, 5.0f};

    auto t_custom = Tensor::from_vector(data, {5}, Device::CPU);
    auto t_torch = torch::tensor(data);

    t_custom.clamp_min_(0.0f);
    t_torch.clamp_min_(0.0f);

    compare_tensors(t_custom, t_torch, 1e-5f, 1e-7f, "ClampMinInPlace");
}

TEST_F(TensorClampTest, ClampMaxInPlace) {
    std::vector<float> data = {-5.0f, -2.0f, 0.0f, 2.0f, 5.0f};

    auto t_custom = Tensor::from_vector(data, {5}, Device::CPU);
    auto t_torch = torch::tensor(data);

    t_custom.clamp_max_(3.0f);
    t_torch.clamp_max_(3.0f);

    compare_tensors(t_custom, t_torch, 1e-5f, 1e-7f, "ClampMaxInPlace");
}

// ============= CUDA Tests =============

TEST_F(TensorClampTest, ClampCUDA) {
    if (!is_cuda_available()) {
        GTEST_SKIP() << "CUDA not available, skipping test";
    }

    std::vector<float> data = {-5.0f, -2.0f, 0.0f, 2.0f, 5.0f};

    auto t_custom = Tensor::from_vector(data, {5}, Device::CUDA);
    auto t_torch = torch::tensor(data, torch::kCUDA);

    auto result_custom = t_custom.clamp(-1.0f, 3.0f);
    auto result_torch = t_torch.clamp(-1.0f, 3.0f);

    ASSERT_TRUE(result_custom.is_valid());
    EXPECT_EQ(result_custom.device(), Device::CUDA);

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "ClampCUDA");
}

TEST_F(TensorClampTest, ClampInPlaceCUDA) {
    if (!is_cuda_available()) {
        GTEST_SKIP() << "CUDA not available, skipping test";
    }

    std::vector<float> data = {-5.0f, -2.0f, 0.0f, 2.0f, 5.0f};

    auto t_custom = Tensor::from_vector(data, {5}, Device::CUDA);
    auto t_torch = torch::tensor(data, torch::kCUDA);

    t_custom.clamp_(-1.0f, 3.0f);
    t_torch.clamp_(-1.0f, 3.0f);

    compare_tensors(t_custom, t_torch, 1e-5f, 1e-7f, "ClampInPlaceCUDA");
}

TEST_F(TensorClampTest, ClampMinCUDA) {
    if (!is_cuda_available()) {
        GTEST_SKIP() << "CUDA not available, skipping test";
    }

    std::vector<float> data = {-5.0f, 0.0f, 5.0f};

    auto t_custom = Tensor::from_vector(data, {3}, Device::CUDA);
    auto t_torch = torch::tensor(data, torch::kCUDA);

    auto result_custom = t_custom.clamp_min(0.0f);
    auto result_torch = t_torch.clamp_min(0.0f);

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "ClampMinCUDA");
}

TEST_F(TensorClampTest, ClampMaxCUDA) {
    if (!is_cuda_available()) {
        GTEST_SKIP() << "CUDA not available, skipping test";
    }

    std::vector<float> data = {-5.0f, 0.0f, 5.0f};

    auto t_custom = Tensor::from_vector(data, {3}, Device::CUDA);
    auto t_torch = torch::tensor(data, torch::kCUDA);

    auto result_custom = t_custom.clamp_max(3.0f);
    auto result_torch = t_torch.clamp_max(3.0f);

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "ClampMaxCUDA");
}

// ============= Multi-dimensional Tests =============

TEST_F(TensorClampTest, Clamp2D) {
    std::vector<float> data = {-5.0f, -2.0f, 0.0f, 2.0f, 5.0f, 10.0f};

    auto t_custom = Tensor::from_vector(data, {2, 3}, Device::CPU);
    auto t_torch = torch::tensor(data).reshape({2, 3});

    auto result_custom = t_custom.clamp(-1.0f, 3.0f);
    auto result_torch = t_torch.clamp(-1.0f, 3.0f);

    EXPECT_EQ(result_custom.shape(), TensorShape({2, 3}));

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "Clamp2D");
}

TEST_F(TensorClampTest, Clamp3D) {
    // Generate same random data for both
    std::vector<float> data;
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < 2 * 3 * 4; ++i) {
        data.push_back(dist(gen));
    }

    auto t_custom = Tensor::from_vector(data, {2, 3, 4}, Device::CPU);
    auto t_torch = torch::tensor(data).reshape({2, 3, 4});

    auto result_custom = t_custom.clamp(-1.0f, 1.0f);
    auto result_torch = t_torch.clamp(-1.0f, 1.0f);

    EXPECT_EQ(result_custom.shape(), t_custom.shape());

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "Clamp3D");
}

// ============= Edge Cases =============

TEST_F(TensorClampTest, ClampWithEqualMinMax) {
    std::vector<float> data = {-5.0f, 0.0f, 5.0f};

    auto t_custom = Tensor::from_vector(data, {3}, Device::CPU);
    auto t_torch = torch::tensor(data);

    auto result_custom = t_custom.clamp(2.0f, 2.0f);
    auto result_torch = t_torch.clamp(2.0f, 2.0f);

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "ClampEqualMinMax");
}

TEST_F(TensorClampTest, ClampWithInfinity) {
    std::vector<float> data = {-1000.0f, 0.0f, 1000.0f};

    auto t_custom = Tensor::from_vector(data, {3}, Device::CPU);
    auto t_torch = torch::tensor(data);

    // Clamp with infinity as max
    auto result_custom = t_custom.clamp(0.0f, std::numeric_limits<float>::infinity());
    auto result_torch = t_torch.clamp(0.0f, std::numeric_limits<float>::infinity());

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "ClampInfMax");

    // Clamp with -infinity as min
    auto result2_custom = t_custom.clamp(-std::numeric_limits<float>::infinity(), 0.0f);
    auto result2_torch = t_torch.clamp(-std::numeric_limits<float>::infinity(), 0.0f);

    compare_tensors(result2_custom, result2_torch, 1e-5f, 1e-7f, "ClampInfMin");
}

TEST_F(TensorClampTest, ClampEmptyTensor) {
    std::vector<float> empty_data;

    auto t_custom = Tensor::from_vector(empty_data, {0}, Device::CPU);
    auto t_torch = torch::empty({0});

    auto result_custom = t_custom.clamp(-1.0f, 1.0f);
    auto result_torch = t_torch.clamp(-1.0f, 1.0f);

    EXPECT_TRUE(result_custom.is_valid());
    EXPECT_EQ(result_custom.numel(), 0);
    EXPECT_EQ(result_torch.numel(), 0);
}

TEST_F(TensorClampTest, ClampInPlaceEmptyTensor) {
    std::vector<float> empty_data;

    auto t_custom = Tensor::from_vector(empty_data, {0}, Device::CPU);
    auto t_torch = torch::empty({0});

    t_custom.clamp_(-1.0f, 1.0f);
    t_torch.clamp_(-1.0f, 1.0f);

    EXPECT_TRUE(t_custom.is_valid());
    EXPECT_EQ(t_custom.numel(), 0);
    EXPECT_EQ(t_torch.numel(), 0);
}

TEST_F(TensorClampTest, ClampScalarTensor) {
    std::vector<float> data = {5.0f};

    auto t_custom = Tensor::from_vector(data, {1}, Device::CPU);
    auto t_torch = torch::tensor(data);

    auto result_custom = t_custom.clamp(0.0f, 10.0f);
    auto result_torch = t_torch.clamp(0.0f, 10.0f);

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "ClampScalar");
}

// ============= Chain Operations =============

TEST_F(TensorClampTest, ClampChaining) {
    // Generate same random data for both
    std::vector<float> data;
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 5.0f);
    for (size_t i = 0; i < 100; ++i) {
        data.push_back(dist(gen));
    }

    auto t_custom = Tensor::from_vector(data, {100}, Device::CPU);
    auto t_torch = torch::tensor(data);

    // Chain multiple clamps
    auto result_custom = t_custom.clamp(-10.0f, 10.0f)
                             .clamp(-5.0f, 5.0f)
                             .clamp(-1.0f, 1.0f);
    auto result_torch = t_torch.clamp(-10.0f, 10.0f)
                            .clamp(-5.0f, 5.0f)
                            .clamp(-1.0f, 1.0f);

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "ClampChaining");
}

TEST_F(TensorClampTest, ClampInPlaceChaining) {
    // Generate same random data for both
    std::vector<float> data;
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 10.0f);
    for (size_t i = 0; i < 100; ++i) {
        data.push_back(dist(gen));
    }

    auto t_custom = Tensor::from_vector(data, {100}, Device::CPU);
    auto t_torch = torch::tensor(data);

    t_custom.clamp_min_(-5.0f).clamp_max_(5.0f);
    t_torch.clamp_min_(-5.0f).clamp_max_(5.0f);

    compare_tensors(t_custom, t_torch, 1e-5f, 1e-7f, "ClampInPlaceChaining");
}

// ============= Integration with Other Operations =============

TEST_F(TensorClampTest, ClampAfterArithmetic) {
    // Generate same random data for both
    std::vector<float> data_a, data_b;
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 2.0f);
    for (size_t i = 0; i < 100; ++i) {
        data_a.push_back(dist(gen));
        data_b.push_back(dist(gen));
    }

    auto a_custom = Tensor::from_vector(data_a, {100}, Device::CPU);
    auto b_custom = Tensor::from_vector(data_b, {100}, Device::CPU);

    auto a_torch = torch::tensor(data_a);
    auto b_torch = torch::tensor(data_b);

    auto result_custom = (a_custom + b_custom).clamp(-1.0f, 1.0f);
    auto result_torch = (a_torch + b_torch).clamp(-1.0f, 1.0f);

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "ClampAfterArithmetic");
}

TEST_F(TensorClampTest, ClampGradient) {
    // Generate same random data for both
    std::vector<float> data;
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 5.0f);
    for (size_t i = 0; i < 100; ++i) {
        data.push_back(dist(gen));
    }

    auto grad_custom = Tensor::from_vector(data, {100}, Device::CPU);
    auto grad_torch = torch::tensor(data);

    // Clamp gradients (common in optimization)
    auto clamped_custom = grad_custom.clamp(-1.0f, 1.0f);
    auto clamped_torch = grad_torch.clamp(-1.0f, 1.0f);

    compare_tensors(clamped_custom, clamped_torch, 1e-5f, 1e-7f, "ClampGradient");

    // Verify no NaNs or Infs
    EXPECT_FALSE(clamped_custom.has_nan());
    EXPECT_FALSE(clamped_custom.has_inf());
}

// ============= Performance Test =============

TEST_F(TensorClampTest, ClampLargeScale) {
    if (!is_cuda_available()) {
        GTEST_SKIP() << "CUDA not available, skipping test";
    }

    // Generate same random data for both
    std::vector<float> data;
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 2.0f);
    for (size_t i = 0; i < 1000 * 1000; ++i) {
        data.push_back(dist(gen));
    }

    auto t_custom = Tensor::from_vector(data, {1000, 1000}, Device::CUDA);
    auto t_torch = torch::tensor(data, torch::kCUDA).reshape({1000, 1000});

    auto result_custom = t_custom.clamp(-1.0f, 1.0f);
    auto result_torch = t_torch.clamp(-1.0f, 1.0f);

    ASSERT_TRUE(result_custom.is_valid());
    EXPECT_EQ(result_custom.shape(), t_custom.shape());

    // Sample check (compare subset to avoid huge memory copy)
    auto cpu_custom = result_custom.cpu();
    auto cpu_torch = result_torch.cpu();

    auto sample_custom = cpu_custom.debug_values(100);
    auto sample_torch_ptr = cpu_torch.data_ptr<float>();

    for (size_t i = 0; i < std::min(sample_custom.size(), size_t(100)); ++i) {
        EXPECT_GE(sample_custom[i], -1.0f);
        EXPECT_LE(sample_custom[i], 1.0f);
        EXPECT_FLOAT_EQ(sample_custom[i], sample_torch_ptr[i]);
    }
}

// ============= Special Values =============

TEST_F(TensorClampTest, ClampWithNaN) {
    std::vector<float> data = {-5.0f, std::numeric_limits<float>::quiet_NaN(), 5.0f};

    auto t_custom = Tensor::from_vector(data, {3}, Device::CPU);
    auto t_torch = torch::tensor(data);

    auto result_custom = t_custom.clamp(-1.0f, 1.0f);
    auto result_torch = t_torch.clamp(-1.0f, 1.0f);

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "ClampWithNaN");
}

TEST_F(TensorClampTest, ClampWithInfValues) {
    std::vector<float> data = {
        -std::numeric_limits<float>::infinity(),
        0.0f,
        std::numeric_limits<float>::infinity()};

    auto t_custom = Tensor::from_vector(data, {3}, Device::CPU);
    auto t_torch = torch::tensor(data);

    auto result_custom = t_custom.clamp(-1.0f, 1.0f);
    auto result_torch = t_torch.clamp(-1.0f, 1.0f);

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "ClampWithInfValues");
}

// ============= Negative Ranges =============

TEST_F(TensorClampTest, ClampNegativeRange) {
    std::vector<float> data = {-10.0f, -5.0f, -2.0f, 0.0f, 2.0f};

    auto t_custom = Tensor::from_vector(data, {5}, Device::CPU);
    auto t_torch = torch::tensor(data);

    auto result_custom = t_custom.clamp(-6.0f, -3.0f);
    auto result_torch = t_torch.clamp(-6.0f, -3.0f);

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "ClampNegativeRange");
}

// ============= Large Values =============

TEST_F(TensorClampTest, ClampLargeValues) {
    std::vector<float> data = {-1e10f, -1e5f, 0.0f, 1e5f, 1e10f};

    auto t_custom = Tensor::from_vector(data, {5}, Device::CPU);
    auto t_torch = torch::tensor(data);

    auto result_custom = t_custom.clamp(-1e6f, 1e6f);
    auto result_torch = t_torch.clamp(-1e6f, 1e6f);

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "ClampLargeValues");
}

// ============= Broadcasting with Clamp =============

TEST_F(TensorClampTest, ClampAfterBroadcast) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f};

    auto t_custom = Tensor::from_vector(data, {3, 1}, Device::CPU);
    auto t_torch = torch::tensor(data).reshape({3, 1});

    // Expand then clamp
    auto expanded_custom = t_custom.expand({3, 4});
    auto expanded_torch = t_torch.expand({3, 4});

    auto result_custom = expanded_custom.clamp(1.5f, 2.5f);
    auto result_torch = expanded_torch.clamp(1.5f, 2.5f);

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "ClampAfterBroadcast");
}

// ============= Clamp with Different Dtypes =============

TEST_F(TensorClampTest, ClampInt32) {
    std::vector<int> data = {-5, -2, 0, 2, 5};

    auto t_custom = Tensor::from_vector(data, {5}, Device::CPU);
    auto t_torch = torch::tensor(data, torch::kInt32);

    // Clamp should work on int32
    t_custom.clamp_(-1.0f, 3.0f);
    t_torch.clamp_(-1, 3);

    compare_tensors(t_custom, t_torch, 1e-5f, 1e-7f, "ClampInt32");
}

// ============= Verify Clamp Bounds =============

TEST_F(TensorClampTest, ClampBoundsVerification) {
    // Generate random data and verify all values are within bounds
    std::vector<float> data;
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(-10.0f, 10.0f);
    for (size_t i = 0; i < 1000; ++i) {
        data.push_back(dist(gen));
    }

    auto t_custom = Tensor::from_vector(data, {1000}, Device::CPU);
    auto t_torch = torch::tensor(data);

    float min_val = -2.5f;
    float max_val = 2.5f;

    auto result_custom = t_custom.clamp(min_val, max_val);
    auto result_torch = t_torch.clamp(min_val, max_val);

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "ClampBoundsVerification");

    // Additional verification: all values in range
    auto values = result_custom.to_vector();
    for (float v : values) {
        EXPECT_GE(v, min_val);
        EXPECT_LE(v, max_val);
    }
}

// ============= Clamp with Zero Range =============

TEST_F(TensorClampTest, ClampZeroAtBoundary) {
    std::vector<float> data = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};

    auto t_custom = Tensor::from_vector(data, {5}, Device::CPU);
    auto t_torch = torch::tensor(data);

    auto result_custom = t_custom.clamp(0.0f, 0.5f);
    auto result_torch = t_torch.clamp(0.0f, 0.5f);

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "ClampZeroAtBoundary");
}