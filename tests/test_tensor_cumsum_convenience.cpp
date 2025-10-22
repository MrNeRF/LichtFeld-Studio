/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/tensor.hpp"
#include <gtest/gtest.h>
#include <numeric>
#include <torch/torch.h>

using namespace lfs::core;

// ============= Helper Functions =============

namespace {

    // Helper for comparing integer tensors (handles both int32 and int64)
    void compare_int_tensors(const Tensor& custom, const torch::Tensor& reference,
                             const std::string& msg = "") {
        auto ref_cpu = reference.to(torch::kCPU).contiguous();
        auto custom_cpu = custom.cpu();

        ASSERT_EQ(custom_cpu.ndim(), ref_cpu.dim()) << msg << ": Rank mismatch";

        for (size_t i = 0; i < custom_cpu.ndim(); ++i) {
            ASSERT_EQ(custom_cpu.size(i), static_cast<size_t>(ref_cpu.size(i)))
                << msg << ": Shape mismatch at dim " << i;
        }

        ASSERT_EQ(custom_cpu.numel(), static_cast<size_t>(ref_cpu.numel()))
            << msg << ": Element count mismatch";

        auto custom_vec = custom_cpu.to_vector_int();

        // Handle both Int32 and Int64 from PyTorch
        if (ref_cpu.dtype() == torch::kInt32) {
            auto ref_data = ref_cpu.data_ptr<int32_t>();
            for (size_t i = 0; i < custom_vec.size(); ++i) {
                EXPECT_EQ(custom_vec[i], static_cast<int>(ref_data[i]))
                    << msg << ": Mismatch at index " << i
                    << " (custom=" << custom_vec[i] << ", ref=" << ref_data[i] << ")";
            }
        } else if (ref_cpu.dtype() == torch::kInt64 || ref_cpu.dtype() == torch::kLong) {
            auto ref_data = ref_cpu.data_ptr<int64_t>();
            for (size_t i = 0; i < custom_vec.size(); ++i) {
                EXPECT_EQ(custom_vec[i], static_cast<int>(ref_data[i]))
                    << msg << ": Mismatch at index " << i
                    << " (custom=" << custom_vec[i] << ", ref=" << ref_data[i] << ")";
            }
        } else {
            FAIL() << msg << ": Unexpected integer dtype in PyTorch tensor";
        }
    }

    // Helper for comparing boolean tensors
    void compare_bool_tensors(const Tensor& custom, const torch::Tensor& reference,
                              const std::string& msg = "") {
        auto ref_cpu = reference.to(torch::kCPU).contiguous();
        auto custom_cpu = custom.cpu();

        ASSERT_EQ(custom_cpu.ndim(), ref_cpu.dim()) << msg << ": Rank mismatch";

        for (size_t i = 0; i < custom_cpu.ndim(); ++i) {
            ASSERT_EQ(custom_cpu.size(i), static_cast<size_t>(ref_cpu.size(i)))
                << msg << ": Shape mismatch at dim " << i;
        }

        ASSERT_EQ(custom_cpu.numel(), static_cast<size_t>(ref_cpu.numel()))
            << msg << ": Element count mismatch";

        auto custom_vec = custom_cpu.to_vector_bool();
        auto ref_data = ref_cpu.data_ptr<bool>();

        for (size_t i = 0; i < custom_vec.size(); ++i) {
            EXPECT_EQ(custom_vec[i], ref_data[i])
                << msg << ": Mismatch at index " << i
                << " (custom=" << custom_vec[i] << ", ref=" << ref_data[i] << ")";
        }
    }

    void compare_tensors(const Tensor& custom, const torch::Tensor& reference,
                         float rtol = 1e-4f, float atol = 1e-5f, const std::string& msg = "") {
        // Handle boolean tensors specially
        if (reference.dtype() == torch::kBool) {
            compare_bool_tensors(custom, reference, msg);
            return;
        }

        // Handle integer tensors specially (Int32, Int64, Long)
        if (reference.dtype() == torch::kInt32 ||
            reference.dtype() == torch::kInt64 ||
            reference.dtype() == torch::kInt ||
            reference.dtype() == torch::kLong) {
            compare_int_tensors(custom, reference, msg);
            return;
        }

        // Handle float tensors
        auto ref_cpu = reference.to(torch::kCPU).contiguous();
        auto custom_cpu = custom.cpu();

        ASSERT_EQ(custom_cpu.ndim(), ref_cpu.dim()) << msg << ": Rank mismatch";

        for (size_t i = 0; i < custom_cpu.ndim(); ++i) {
            ASSERT_EQ(custom_cpu.size(i), static_cast<size_t>(ref_cpu.size(i)))
                << msg << ": Shape mismatch at dim " << i;
        }

        ASSERT_EQ(custom_cpu.numel(), static_cast<size_t>(ref_cpu.numel()))
            << msg << ": Element count mismatch";

        auto custom_vec = custom_cpu.to_vector();
        auto ref_data = ref_cpu.data_ptr<float>();

        for (size_t i = 0; i < custom_vec.size(); ++i) {
            float ref_val = ref_data[i];
            float custom_val = custom_vec[i];

            if (std::isnan(ref_val)) {
                EXPECT_TRUE(std::isnan(custom_val)) << msg << ": Expected NaN at index " << i;
            } else if (std::isinf(ref_val)) {
                EXPECT_TRUE(std::isinf(custom_val)) << msg << ": Expected Inf at index " << i;
            } else {
                float diff = std::abs(custom_val - ref_val);
                float threshold = atol + rtol * std::abs(ref_val);
                EXPECT_LE(diff, threshold)
                    << msg << ": Mismatch at index " << i
                    << " (custom=" << custom_val << ", ref=" << ref_val << ")";
            }
        }
    }

} // anonymous namespace

class TensorCumsumConvenienceTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA is not available for testing";
        torch::manual_seed(42);
        Tensor::manual_seed(42);
    }
};

// ============= Cumsum Tests =============

TEST_F(TensorCumsumConvenienceTest, Cumsum1DBasic) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    auto custom_t = Tensor::from_vector(data, {5}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_t.cumsum(0);
    auto torch_result = torch_t.cumsum(0);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "Cumsum1D");
}

TEST_F(TensorCumsumConvenienceTest, Cumsum1DNegativeValues) {
    std::vector<float> data = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f};

    auto custom_t = Tensor::from_vector(data, {5}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_t.cumsum(0);
    auto torch_result = torch_t.cumsum(0);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "Cumsum1DNegative");
}

TEST_F(TensorCumsumConvenienceTest, Cumsum2DDim0) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    auto custom_t = Tensor::from_vector(data, {2, 3}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA)).reshape({2, 3});

    auto custom_result = custom_t.cumsum(0);
    auto torch_result = torch_t.cumsum(0);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "Cumsum2DDim0");
}

TEST_F(TensorCumsumConvenienceTest, Cumsum2DDim1) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    auto custom_t = Tensor::from_vector(data, {2, 3}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA)).reshape({2, 3});

    auto custom_result = custom_t.cumsum(1);
    auto torch_result = torch_t.cumsum(1);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "Cumsum2DDim1");
}

TEST_F(TensorCumsumConvenienceTest, Cumsum3D) {
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), 0.0f);

    auto custom_t = Tensor::from_vector(data, {2, 3, 4}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA)).reshape({2, 3, 4});

    // Test cumsum along all dimensions
    for (int dim = 0; dim < 3; ++dim) {
        auto custom_result = custom_t.cumsum(dim);
        auto torch_result = torch_t.cumsum(dim);

        compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f,
                        "Cumsum3D_Dim" + std::to_string(dim));
    }
}

TEST_F(TensorCumsumConvenienceTest, CumsumNegativeDim) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};

    auto custom_t = Tensor::from_vector(data, {4}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_t.cumsum(-1);
    auto torch_result = torch_t.cumsum(-1);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "CumsumNegativeDim");
}

TEST_F(TensorCumsumConvenienceTest, CumsumInt32) {
    std::vector<int> data = {1, 2, 3, 4, 5};

    auto custom_t = Tensor::from_vector(data, {5}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    auto custom_result = custom_t.cumsum(0);
    auto torch_result = torch_t.cumsum(0); // This returns Int64 in PyTorch

    // Verify our result is Int32
    EXPECT_EQ(custom_result.dtype(), DataType::Int32);

    // Convert both to float for comparison to avoid dtype issues
    auto custom_float = custom_result.to(DataType::Float32);
    auto torch_float = torch_result.to(torch::kFloat32);

    compare_tensors(custom_float, torch_float, 1e-6f, 1e-7f, "CumsumInt32");
}

TEST_F(TensorCumsumConvenienceTest, CumsumSingleElement) {
    std::vector<float> data = {42.0f};

    auto custom_t = Tensor::from_vector(data, {1}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_t.cumsum(0);
    auto torch_result = torch_t.cumsum(0);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "CumsumSingle");
}

TEST_F(TensorCumsumConvenienceTest, CumsumLarge) {
    std::vector<float> data(100);
    std::iota(data.begin(), data.end(), 1.0f);

    auto custom_t = Tensor::from_vector(data, {100}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_t.cumsum(0);
    auto torch_result = torch_t.cumsum(0);

    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "CumsumLarge");
}

// ============= Device Transfer Tests =============

TEST_F(TensorCumsumConvenienceTest, CpuToCuda) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    auto custom_cpu = Tensor::from_vector(data, {5}, Device::CPU);
    auto torch_cpu = torch::tensor(data, torch::TensorOptions().device(torch::kCPU));

    auto custom_cuda = custom_cpu.cuda();
    auto torch_cuda = torch_cpu.cuda();

    EXPECT_EQ(custom_cuda.device(), Device::CUDA);
    compare_tensors(custom_cuda, torch_cuda, 1e-6f, 1e-7f, "CpuToCuda");
}

TEST_F(TensorCumsumConvenienceTest, CudaToCpu) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    auto custom_cuda = Tensor::from_vector(data, {5}, Device::CUDA);
    auto torch_cuda = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_cpu = custom_cuda.cpu();
    auto torch_cpu = torch_cuda.cpu();

    EXPECT_EQ(custom_cpu.device(), Device::CPU);
    compare_tensors(custom_cpu, torch_cpu, 1e-6f, 1e-7f, "CudaToCpu");
}

TEST_F(TensorCumsumConvenienceTest, CpuIdempotent) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f};

    auto custom_t = Tensor::from_vector(data, {3}, Device::CPU);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCPU));

    auto custom_result = custom_t.cpu();
    auto torch_result = torch_t.cpu();

    EXPECT_EQ(custom_result.device(), Device::CPU);
    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "CpuIdempotent");

    // Verify independence (should create a copy)
    custom_result.fill_(42.0f);
    EXPECT_FLOAT_EQ(custom_t.to_vector()[0], 1.0f);
}

TEST_F(TensorCumsumConvenienceTest, CudaIdempotent) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f};

    auto custom_t = Tensor::from_vector(data, {3}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_t.cuda();
    auto torch_result = torch_t.cuda();

    EXPECT_EQ(custom_result.device(), Device::CUDA);
    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "CudaIdempotent");
}

TEST_F(TensorCumsumConvenienceTest, DeviceRoundtrip) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};

    auto custom_cpu = Tensor::from_vector(data, {4}, Device::CPU);
    auto torch_cpu = torch::tensor(data, torch::TensorOptions().device(torch::kCPU));

    auto custom_result = custom_cpu.cuda().cpu();
    auto torch_result = torch_cpu.cuda().cpu();

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "DeviceRoundtrip");
}

TEST_F(TensorCumsumConvenienceTest, DeviceTransferPreservesShape) {
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), 0.0f);

    auto custom_cpu = Tensor::from_vector(data, {2, 3, 4}, Device::CPU);
    auto torch_cpu = torch::tensor(data, torch::TensorOptions().device(torch::kCPU)).reshape({2, 3, 4});

    auto custom_cuda = custom_cpu.cuda();
    auto torch_cuda = torch_cpu.cuda();

    EXPECT_EQ(custom_cuda.ndim(), torch_cuda.dim());
    for (size_t i = 0; i < custom_cuda.ndim(); ++i) {
        EXPECT_EQ(custom_cuda.shape()[i], static_cast<size_t>(torch_cuda.size(i)));
    }
}

// ============= item() Tests =============

TEST_F(TensorCumsumConvenienceTest, ItemFloat) {
    std::vector<float> data = {42.5f};

    auto custom_t = Tensor::from_vector(data, {1}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    float custom_value = custom_t.item<float>();
    float torch_value = torch_t.item<float>();

    EXPECT_FLOAT_EQ(custom_value, torch_value);
}

TEST_F(TensorCumsumConvenienceTest, ItemInt) {
    std::vector<int> data = {42};

    auto custom_t = Tensor::from_vector(data, {1}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    int custom_value = custom_t.item<int>();
    int torch_value = torch_t.item<int>();

    EXPECT_EQ(custom_value, torch_value);
}

TEST_F(TensorCumsumConvenienceTest, ItemBool) {
    std::vector<bool> data = {true};

    auto custom_t = Tensor::from_vector(data, {1}, Device::CUDA);
    auto torch_t = torch::tensor({1}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

    unsigned char custom_value = custom_t.item<unsigned char>();
    bool torch_value = torch_t.item<bool>();

    EXPECT_EQ(custom_value, torch_value ? 1 : 0);
}

TEST_F(TensorCumsumConvenienceTest, ItemNegative) {
    std::vector<float> data = {-3.14f};

    auto custom_t = Tensor::from_vector(data, {1}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    float custom_value = custom_t.item<float>();
    float torch_value = torch_t.item<float>();

    EXPECT_FLOAT_EQ(custom_value, torch_value);
}

TEST_F(TensorCumsumConvenienceTest, ItemAfterReduction) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    auto custom_t = Tensor::from_vector(data, {5}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_sum = custom_t.sum();
    auto torch_sum = torch_t.sum();

    float custom_value = custom_sum.item<float>();
    float torch_value = torch_sum.item<float>();

    EXPECT_FLOAT_EQ(custom_value, torch_value);
}

TEST_F(TensorCumsumConvenienceTest, ItemLargeValue) {
    std::vector<float> data = {1e6f};

    auto custom_t = Tensor::from_vector(data, {1}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    float custom_value = custom_t.item<float>();
    float torch_value = torch_t.item<float>();

    EXPECT_FLOAT_EQ(custom_value, torch_value);
}

TEST_F(TensorCumsumConvenienceTest, ItemSmallValue) {
    std::vector<float> data = {1e-6f};

    auto custom_t = Tensor::from_vector(data, {1}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    float custom_value = custom_t.item<float>();
    float torch_value = torch_t.item<float>();

    EXPECT_NEAR(custom_value, torch_value, 1e-9f);
}

// ============= ones_like with dtype Tests =============

TEST_F(TensorCumsumConvenienceTest, OnesLikeFloat32ToInt32) {
    std::vector<float> data(12, 0.0f);

    auto custom_t = Tensor::from_vector(data, {3, 4}, Device::CUDA);
    auto torch_t = torch::zeros({3, 4}, torch::TensorOptions().device(torch::kCUDA));

    auto custom_ones = Tensor::ones_like(custom_t, DataType::Int32);
    auto torch_ones = torch::ones_like(torch_t, torch::TensorOptions().dtype(torch::kInt32));

    EXPECT_EQ(custom_ones.dtype(), DataType::Int32);
    compare_tensors(custom_ones, torch_ones, 1e-6f, 1e-7f, "OnesLikeInt32");
}

TEST_F(TensorCumsumConvenienceTest, OnesLikeFloat32ToBool) {
    std::vector<float> data(5, 0.0f);

    auto custom_t = Tensor::from_vector(data, {5}, Device::CUDA);
    auto torch_t = torch::zeros({5}, torch::TensorOptions().device(torch::kCUDA));

    auto custom_ones = Tensor::ones_like(custom_t, DataType::Bool);
    auto torch_ones = torch::ones_like(torch_t, torch::TensorOptions().dtype(torch::kBool));

    EXPECT_EQ(custom_ones.dtype(), DataType::Bool);
    compare_tensors(custom_ones, torch_ones, 1e-6f, 1e-7f, "OnesLikeBool");
}

TEST_F(TensorCumsumConvenienceTest, OnesLikePreservesDevice) {
    auto custom_cuda = Tensor::zeros({10}, Device::CUDA);
    auto torch_cuda = torch::zeros({10}, torch::TensorOptions().device(torch::kCUDA));

    auto custom_ones = Tensor::ones_like(custom_cuda, DataType::Int32);
    auto torch_ones = torch::ones_like(torch_cuda, torch::TensorOptions().dtype(torch::kInt32));

    EXPECT_EQ(custom_ones.device(), Device::CUDA);
    EXPECT_EQ(custom_ones.dtype(), DataType::Int32);
    compare_tensors(custom_ones, torch_ones, 1e-6f, 1e-7f, "OnesLikeDevice");
}

TEST_F(TensorCumsumConvenienceTest, OnesLikePreservesShape) {
    std::vector<float> data(120, 0.0f);

    auto custom_t = Tensor::from_vector(data, {2, 3, 4, 5}, Device::CUDA);
    auto torch_t = torch::zeros({2, 3, 4, 5}, torch::TensorOptions().device(torch::kCUDA));

    auto custom_ones = Tensor::ones_like(custom_t, DataType::Float32);
    auto torch_ones = torch::ones_like(torch_t);

    EXPECT_EQ(custom_ones.ndim(), torch_ones.dim());
    for (size_t i = 0; i < custom_ones.ndim(); ++i) {
        EXPECT_EQ(custom_ones.shape()[i], static_cast<size_t>(torch_ones.size(i)));
    }

    compare_tensors(custom_ones, torch_ones, 1e-6f, 1e-7f, "OnesLikeShape");
}

// ============= Integration Tests =============

TEST_F(TensorCumsumConvenienceTest, CumsumWithArithmetic) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};

    auto custom_t = Tensor::from_vector(data, {4}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_t.cumsum(0).mul(2.0f);
    auto torch_result = torch_t.cumsum(0) * 2.0f;

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "CumsumArithmetic");
}

TEST_F(TensorCumsumConvenienceTest, CumsumNormalization) {
    std::vector<float> data = {1.0f, 1.0f, 1.0f, 1.0f};

    auto custom_t = Tensor::from_vector(data, {4}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_t.cumsum(0).div(4.0f);
    auto torch_result = torch_t.cumsum(0) / 4.0f;

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "CumsumNormalize");
}

TEST_F(TensorCumsumConvenienceTest, ChainedOperations) {
    std::vector<float> data(10, 1.0f);

    auto custom_cpu = Tensor::from_vector(data, {10}, Device::CPU);
    auto torch_cpu = torch::tensor(data, torch::TensorOptions().device(torch::kCPU));

    auto custom_result = custom_cpu.cuda().mul(2.0f).cpu();
    auto torch_result = torch_cpu.cuda() * 2.0f;
    torch_result = torch_result.cpu();

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "ChainedOps");
}

TEST_F(TensorCumsumConvenienceTest, CumsumThenSlice) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    auto custom_t = Tensor::from_vector(data, {5}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_cumsum = custom_t.cumsum(0);
    auto torch_cumsum = torch_t.cumsum(0);

    auto custom_slice = custom_cumsum.slice(0, 2, 4);
    auto torch_slice = torch_cumsum.slice(0, 2, 4);

    compare_tensors(custom_slice, torch_slice, 1e-6f, 1e-7f, "CumsumSlice");
}

TEST_F(TensorCumsumConvenienceTest, ComplexChain) {
    std::vector<float> data(20);
    std::iota(data.begin(), data.end(), 1.0f);

    auto custom_t = Tensor::from_vector(data, {20}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_t.reshape({4, 5})
                             .cumsum(1)
                             .t()
                             .cumsum(0)
                             .flatten();

    auto torch_result = torch_t.reshape({4, 5})
                            .cumsum(1)
                            .t()
                            .cumsum(0)
                            .flatten();

    compare_tensors(custom_result, torch_result, 1e-5f, 1e-6f, "ComplexChain");
}

// ============= Edge Cases =============

TEST_F(TensorCumsumConvenienceTest, CumsumZeros) {
    auto custom_t = Tensor::zeros({10}, Device::CUDA);
    auto torch_t = torch::zeros({10}, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_t.cumsum(0);
    auto torch_result = torch_t.cumsum(0);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "CumsumZeros");
}

TEST_F(TensorCumsumConvenienceTest, CumsumOnes) {
    auto custom_t = Tensor::ones({8}, Device::CUDA);
    auto torch_t = torch::ones({8}, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_t.cumsum(0);
    auto torch_result = torch_t.cumsum(0);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "CumsumOnes");
}

TEST_F(TensorCumsumConvenienceTest, CumsumAlternating) {
    std::vector<float> data = {1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f};

    auto custom_t = Tensor::from_vector(data, {6}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_t.cumsum(0);
    auto torch_result = torch_t.cumsum(0);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "CumsumAlternating");
}

TEST_F(TensorCumsumConvenienceTest, CumsumMultiDimensional) {
    std::vector<float> data(60);
    std::iota(data.begin(), data.end(), 0.0f);

    auto custom_t = Tensor::from_vector(data, {3, 4, 5}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA)).reshape({3, 4, 5});

    // Test on middle dimension
    auto custom_result = custom_t.cumsum(1);
    auto torch_result = torch_t.cumsum(1);

    compare_tensors(custom_result, torch_result, 1e-5f, 1e-6f, "CumsumMultiDim");
}

TEST_F(TensorCumsumConvenienceTest, DeviceTransferDifferentDtypes) {
    std::vector<int> data = {1, 2, 3, 4, 5};

    auto custom_cpu = Tensor::from_vector(data, {5}, Device::CPU);
    auto torch_cpu = torch::tensor(data, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));

    auto custom_cuda = custom_cpu.cuda();
    auto torch_cuda = torch_cpu.cuda();

    EXPECT_EQ(custom_cuda.dtype(), DataType::Int32);
    compare_tensors(custom_cuda, torch_cuda, 1e-6f, 1e-7f, "DeviceTransferInt");
}