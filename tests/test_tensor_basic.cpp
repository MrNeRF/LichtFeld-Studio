/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/tensor.hpp"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <torch/torch.h>

using namespace lfs::core;

// ============= Helper Functions =============

namespace {

    // Helper to compare tensors
    void compare_tensors(const Tensor& custom, const torch::Tensor& reference,
                         float rtol = 1e-5f, float atol = 1e-7f, const std::string& msg = "") {
        // Flatten and move to CPU for comparison
        auto ref_cpu = reference.to(torch::kCPU).contiguous().flatten();
        auto custom_cpu = custom.cpu();

        ASSERT_EQ(custom_cpu.ndim(), reference.dim()) << msg << ": Rank mismatch";

        for (size_t i = 0; i < custom_cpu.ndim(); ++i) {
            ASSERT_EQ(custom_cpu.size(i), static_cast<size_t>(reference.size(i)))
                << msg << ": Shape mismatch at dim " << i;
        }

        ASSERT_EQ(custom_cpu.numel(), static_cast<size_t>(ref_cpu.numel()))
            << msg << ": Element count mismatch";

        auto custom_vec = custom_cpu.to_vector();
        auto ref_accessor = ref_cpu.accessor<float, 1>();

        for (size_t i = 0; i < custom_vec.size(); ++i) {
            float ref_val = ref_accessor[i];
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

class TensorBasicTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure CUDA is available
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA is not available for testing";

        // Set random seed for reproducibility
        torch::manual_seed(42);
        Tensor::manual_seed(42);
        gen.seed(42);
    }

    std::mt19937 gen;
    std::uniform_real_distribution<float> dist{-10.0f, 10.0f};
};

// ============= Tensor Creation Tests =============

TEST_F(TensorBasicTest, EmptyTensorCreation) {
    // Create empty tensors
    auto tensor_custom = Tensor::empty({2, 3, 4}, Device::CUDA);
    auto tensor_torch = torch::empty({2, 3, 4},
                                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // Check properties match PyTorch
    EXPECT_EQ(tensor_custom.shape().rank(), tensor_torch.dim());
    EXPECT_EQ(tensor_custom.shape()[0], tensor_torch.size(0));
    EXPECT_EQ(tensor_custom.shape()[1], tensor_torch.size(1));
    EXPECT_EQ(tensor_custom.shape()[2], tensor_torch.size(2));
    EXPECT_EQ(tensor_custom.numel(), tensor_torch.numel());
    EXPECT_TRUE(tensor_custom.is_valid());
    EXPECT_FALSE(tensor_custom.is_empty());

    // Both should be contiguous
    EXPECT_TRUE(tensor_custom.is_contiguous());
    EXPECT_TRUE(tensor_torch.is_contiguous());
}

TEST_F(TensorBasicTest, ZerosTensorCreation) {
    // Create zeros tensors
    auto tensor_custom = Tensor::zeros({3, 4}, Device::CUDA);
    auto tensor_torch = torch::zeros({3, 4},
                                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    compare_tensors(tensor_custom, tensor_torch, 1e-6f, 1e-7f, "Zeros");
}

TEST_F(TensorBasicTest, OnesTensorCreation) {
    // Create ones tensors
    auto tensor_custom = Tensor::ones({5, 2}, Device::CUDA);
    auto tensor_torch = torch::ones({5, 2},
                                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    compare_tensors(tensor_custom, tensor_torch, 1e-6f, 1e-7f, "Ones");
}

TEST_F(TensorBasicTest, FullTensorCreation) {
    float fill_value = 3.14f;

    auto tensor_custom = Tensor::full({2, 3, 2}, fill_value, Device::CUDA);
    auto tensor_torch = torch::full({2, 3, 2}, fill_value,
                                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    compare_tensors(tensor_custom, tensor_torch, 1e-6f, 1e-7f, "Full");
}

TEST_F(TensorBasicTest, ArangeTensorCreation) {
    // Test arange with end only
    auto tensor_custom1 = Tensor::arange(10.0f);
    auto tensor_torch1 = torch::arange(0.0f, 10.0f, 1.0f,
                                       torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    compare_tensors(tensor_custom1, tensor_torch1, 1e-6f, 1e-7f, "Arange_End");

    // Test arange with start, end, step
    auto tensor_custom2 = Tensor::arange(2.0f, 10.0f, 0.5f);
    auto tensor_torch2 = torch::arange(2.0f, 10.0f, 0.5f,
                                       torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    compare_tensors(tensor_custom2, tensor_torch2, 1e-5f, 1e-6f, "Arange_StartEndStep");
}

TEST_F(TensorBasicTest, EyeTensorCreation) {
    // Square identity matrix
    auto tensor_custom1 = Tensor::eye(5, Device::CUDA);
    auto tensor_torch1 = torch::eye(5, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    compare_tensors(tensor_custom1, tensor_torch1, 1e-6f, 1e-7f, "Eye_Square");

    // Rectangular identity matrix
    auto tensor_custom2 = Tensor::eye(3, 5, Device::CUDA);
    auto tensor_torch2 = torch::eye(3, 5, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    compare_tensors(tensor_custom2, tensor_torch2, 1e-6f, 1e-7f, "Eye_Rectangular");
}

TEST_F(TensorBasicTest, FromVectorCreation) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    auto tensor_custom = Tensor::from_vector(data, {2, 3}, Device::CUDA);

    // Create PyTorch tensor from same data
    auto tensor_torch = torch::from_blob(
                            const_cast<float*>(data.data()),
                            {2, 3},
                            torch::TensorOptions().dtype(torch::kFloat32))
                            .clone()
                            .to(torch::kCUDA);

    compare_tensors(tensor_custom, tensor_torch, 1e-6f, 1e-7f, "FromVector");
}

TEST_F(TensorBasicTest, FromBlobCreation) {
    // Create data on device
    size_t num_elements = 12;
    float* cuda_data;
    cudaMalloc(&cuda_data, num_elements * sizeof(float));

    // Fill with test data
    std::vector<float> host_data(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
        host_data[i] = static_cast<float>(i);
    }
    cudaMemcpy(cuda_data, host_data.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice);

    // Create tensor from blob
    auto tensor_custom = Tensor::from_blob(cuda_data, {3, 4}, Device::CUDA, DataType::Float32);
    auto tensor_torch = torch::from_blob(cuda_data, {3, 4},
                                         torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    compare_tensors(tensor_custom, tensor_torch, 1e-6f, 1e-7f, "FromBlob");

    EXPECT_FALSE(tensor_custom.owns_memory()); // Should not own the memory

    // Cleanup
    cudaFree(cuda_data);
}

// ============= Device Transfer Tests =============

TEST_F(TensorBasicTest, DeviceTransferCUDAToCPU) {
    // Create CUDA tensor
    auto cuda_custom = Tensor::full({3, 3}, 2.5f, Device::CUDA);
    auto cuda_torch = torch::full({3, 3}, 2.5f,
                                  torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // Transfer to CPU
    auto cpu_custom = cuda_custom.to(Device::CPU);
    auto cpu_torch = cuda_torch.to(torch::kCPU);

    EXPECT_EQ(cpu_custom.device(), Device::CPU);
    EXPECT_TRUE(cpu_torch.device().is_cpu());

    compare_tensors(cpu_custom, cpu_torch, 1e-6f, 1e-7f, "DeviceTransfer_CUDA_to_CPU");
}

TEST_F(TensorBasicTest, DeviceTransferCPUToCUDA) {
    // Create CPU tensor
    auto cpu_custom = Tensor::full({4, 2}, 1.5f, Device::CPU);
    auto cpu_torch = torch::full({4, 2}, 1.5f,
                                 torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

    // Transfer to CUDA
    auto cuda_custom = cpu_custom.to(Device::CUDA);
    auto cuda_torch = cpu_torch.to(torch::kCUDA);

    EXPECT_EQ(cuda_custom.device(), Device::CUDA);
    EXPECT_TRUE(cuda_torch.device().is_cuda());

    compare_tensors(cuda_custom, cuda_torch, 1e-6f, 1e-7f, "DeviceTransfer_CPU_to_CUDA");
}

TEST_F(TensorBasicTest, DeviceTransferRoundTrip) {
    // Create CUDA tensor
    auto cuda_custom = Tensor::full({3, 3}, 2.5f, Device::CUDA);
    auto cuda_torch = torch::full({3, 3}, 2.5f,
                                  torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // CUDA -> CPU -> CUDA
    auto cpu_custom = cuda_custom.cpu();
    auto cuda2_custom = cpu_custom.cuda();

    auto cpu_torch = cuda_torch.cpu();
    auto cuda2_torch = cpu_torch.cuda();

    compare_tensors(cuda2_custom, cuda2_torch, 1e-6f, 1e-7f, "DeviceTransfer_RoundTrip");

    // Verify round-trip preserves values
    EXPECT_TRUE(cuda_custom.all_close(cuda2_custom, 1e-5f, 1e-6f))
        << "Round-trip should preserve values";
    compare_tensors(cuda_custom, cuda_torch, 1e-6f, 1e-7f, "DeviceTransfer_RoundTrip_Original");
}

// ============= Clone Tests =============

TEST_F(TensorBasicTest, Clone) {
    // Create original tensor with data
    std::vector<float> data(20);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = dist(gen);
    }

    auto original_custom = Tensor::from_vector(data, {4, 5}, Device::CUDA);
    auto original_torch = torch::from_blob(
                              const_cast<float*>(data.data()),
                              {4, 5},
                              torch::TensorOptions().dtype(torch::kFloat32))
                              .clone()
                              .to(torch::kCUDA);

    // Clone both
    auto cloned_custom = original_custom.clone();
    auto cloned_torch = original_torch.clone();

    // Check they have same values
    compare_tensors(cloned_custom, cloned_torch, 1e-6f, 1e-7f, "Clone");

    // Verify original and clone match
    EXPECT_TRUE(original_custom.all_close(cloned_custom, 1e-5f, 1e-6f))
        << "Clone should match original";
    compare_tensors(original_custom, original_torch, 1e-6f, 1e-7f, "Clone_Original");

    // Check different memory
    EXPECT_NE(original_custom.raw_ptr(), cloned_custom.raw_ptr());
    EXPECT_NE(original_torch.data_ptr(), cloned_torch.data_ptr());

    EXPECT_TRUE(cloned_custom.owns_memory());
}

TEST_F(TensorBasicTest, CloneModifyIndependence) {
    auto original_custom = Tensor::ones({2, 3}, Device::CUDA);
    auto original_torch = torch::ones({2, 3},
                                      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    auto cloned_custom = original_custom.clone();
    auto cloned_torch = original_torch.clone();

    // Modify clones
    cloned_custom.fill_(0.0f);
    cloned_torch.fill_(0.0f);

    // Originals should be unchanged
    auto original_vals = original_custom.to_vector();
    for (float val : original_vals) {
        EXPECT_FLOAT_EQ(val, 1.0f) << "Original should be unchanged";
    }

    // Clones should be zeros
    auto cloned_vals = cloned_custom.to_vector();
    for (float val : cloned_vals) {
        EXPECT_FLOAT_EQ(val, 0.0f) << "Clone should be modified";
    }

    compare_tensors(cloned_custom, cloned_torch, 1e-6f, 1e-7f, "Clone_Modified");
}

// ============= Copy Tests =============

TEST_F(TensorBasicTest, CopyFrom) {
    auto tensor1_custom = Tensor::ones({2, 3}, Device::CUDA);
    auto tensor2_custom = Tensor::zeros({2, 3}, Device::CUDA);

    auto tensor1_torch = torch::ones({2, 3},
                                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto tensor2_torch = torch::zeros({2, 3},
                                      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // Copy tensor1 to tensor2
    tensor2_custom.copy_from(tensor1_custom);
    tensor2_torch.copy_(tensor1_torch);

    // Check values copied
    compare_tensors(tensor2_custom, tensor2_torch, 1e-6f, 1e-7f, "CopyFrom");

    // Verify tensor1 and tensor2 now match
    EXPECT_TRUE(tensor1_custom.all_close(tensor2_custom, 1e-5f, 1e-6f))
        << "After copy, tensors should match";
    compare_tensors(tensor1_custom, tensor1_torch, 1e-6f, 1e-7f, "CopyFrom_Source");

    // But different memory
    EXPECT_NE(tensor1_custom.raw_ptr(), tensor2_custom.raw_ptr());
    EXPECT_NE(tensor1_torch.data_ptr(), tensor2_torch.data_ptr());
}

// ============= Fill Operations Tests =============

TEST_F(TensorBasicTest, FillOperations) {
    auto tensor_custom = Tensor::empty({3, 4}, Device::CUDA);
    auto tensor_torch = torch::empty({3, 4},
                                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // Fill with value
    tensor_custom.fill_(5.5f);
    tensor_torch.fill_(5.5f);

    compare_tensors(tensor_custom, tensor_torch, 1e-6f, 1e-7f, "Fill");

    // Zero
    tensor_custom.zero_();
    tensor_torch.zero_();

    compare_tensors(tensor_custom, tensor_torch, 1e-6f, 1e-7f, "Zero");
}

TEST_F(TensorBasicTest, UniformFillOperation) {
    auto tensor_custom = Tensor::empty({100}, Device::CUDA);
    auto tensor_torch = torch::empty({100},
                                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // Fill with uniform distribution
    tensor_custom.uniform_(-1.0f, 1.0f);
    tensor_torch.uniform_(-1.0f, 1.0f);

    // Check values are in range
    auto vals = tensor_custom.to_vector();
    for (float val : vals) {
        EXPECT_GE(val, -1.0f);
        EXPECT_LE(val, 1.0f);
    }

    // Both should have values in the same range
    auto torch_cpu = tensor_torch.to(torch::kCPU);
    auto torch_accessor = torch_cpu.accessor<float, 1>();
    for (int64_t i = 0; i < tensor_torch.size(0); ++i) {
        EXPECT_GE(torch_accessor[i], -1.0f);
        EXPECT_LE(torch_accessor[i], 1.0f);
    }
}

TEST_F(TensorBasicTest, NormalFillOperation) {
    auto tensor_custom = Tensor::empty({1000}, Device::CUDA);
    auto tensor_torch = torch::empty({1000},
                                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // Fill with normal distribution
    tensor_custom.normal_(0.0f, 1.0f);
    tensor_torch.normal_(0.0f, 1.0f);

    // Check mean and std are approximately correct for both
    auto custom_mean = tensor_custom.mean_scalar();
    auto custom_std = tensor_custom.std_scalar();

    auto torch_mean = tensor_torch.mean().item<float>();
    auto torch_std = tensor_torch.std().item<float>();

    // Both should be close to target distribution
    EXPECT_NEAR(custom_mean, 0.0f, 0.15f); // Allow some variance
    EXPECT_NEAR(custom_std, 1.0f, 0.15f);

    EXPECT_NEAR(torch_mean, 0.0f, 0.15f);
    EXPECT_NEAR(torch_std, 1.0f, 0.15f);
}

// ============= Move Semantics Tests =============

TEST_F(TensorBasicTest, MoveConstructor) {
    auto tensor1 = Tensor::ones({2, 2}, Device::CUDA);
    void* original_ptr = tensor1.raw_ptr();

    Tensor tensor2(std::move(tensor1));

    EXPECT_EQ(tensor2.raw_ptr(), original_ptr);
    EXPECT_EQ(tensor1.raw_ptr(), nullptr); // tensor1 should be cleared
    EXPECT_TRUE(tensor2.is_valid());
    EXPECT_FALSE(tensor1.is_valid());
}

TEST_F(TensorBasicTest, MoveAssignment) {
    auto tensor1 = Tensor::ones({2, 2}, Device::CUDA);
    void* original_ptr = tensor1.raw_ptr();

    auto tensor2 = Tensor::zeros({2, 2}, Device::CUDA);
    void* tensor2_original = tensor2.raw_ptr();

    tensor2 = std::move(tensor1);

    EXPECT_EQ(tensor2.raw_ptr(), original_ptr);
    EXPECT_EQ(tensor1.raw_ptr(), nullptr);
    EXPECT_TRUE(tensor2.is_valid());
    EXPECT_FALSE(tensor1.is_valid());
}

// ============= Properties Tests =============

TEST_F(TensorBasicTest, Properties) {
    auto tensor_custom = Tensor::full({2, 3, 4}, 1.0f, Device::CUDA);
    auto tensor_torch = torch::full({2, 3, 4}, 1.0f,
                                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    EXPECT_EQ(tensor_custom.numel(), tensor_torch.numel());
    EXPECT_EQ(tensor_custom.bytes(), tensor_torch.nbytes());
    EXPECT_TRUE(tensor_custom.is_valid());
    EXPECT_FALSE(tensor_custom.is_empty());
    EXPECT_TRUE(tensor_custom.is_contiguous());
    EXPECT_TRUE(tensor_torch.is_contiguous());
    EXPECT_TRUE(tensor_custom.owns_memory());

    // Test shape access
    EXPECT_EQ(tensor_custom.shape()[0], tensor_torch.size(0));
    EXPECT_EQ(tensor_custom.shape()[1], tensor_torch.size(1));
    EXPECT_EQ(tensor_custom.shape()[2], tensor_torch.size(2));
    EXPECT_EQ(tensor_custom.shape().rank(), tensor_torch.dim());
    EXPECT_EQ(tensor_custom.ndim(), tensor_torch.dim());
}

TEST_F(TensorBasicTest, SizeMethod) {
    auto tensor_custom = Tensor::full({2, 3, 4}, 1.0f, Device::CUDA);
    auto tensor_torch = torch::full({2, 3, 4}, 1.0f,
                                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    for (size_t i = 0; i < tensor_custom.ndim(); ++i) {
        EXPECT_EQ(tensor_custom.size(i), tensor_torch.size(i));
    }
}

// ============= Invalid Operations Tests =============

TEST_F(TensorBasicTest, InvalidTensor) {
    // Test operations on invalid tensor
    Tensor invalid_tensor;

    EXPECT_FALSE(invalid_tensor.is_valid());
    EXPECT_TRUE(invalid_tensor.is_empty());
    EXPECT_EQ(invalid_tensor.numel(), 0);

    // These should not crash but return invalid tensors
    auto result = invalid_tensor.clone();
    EXPECT_FALSE(result.is_valid());

    result = invalid_tensor.add(1.0f);
    EXPECT_FALSE(result.is_valid());
}

TEST_F(TensorBasicTest, EmptyTensor) {
    auto tensor_custom = Tensor::empty({0}, Device::CUDA);
    auto tensor_torch = torch::empty({0},
                                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    EXPECT_TRUE(tensor_custom.is_valid());
    EXPECT_TRUE(tensor_custom.is_empty());
    EXPECT_EQ(tensor_custom.numel(), 0);
    EXPECT_EQ(tensor_custom.numel(), tensor_torch.numel());
}

// ============= AllClose Tests =============

TEST_F(TensorBasicTest, AllClose) {
    auto tensor1_custom = Tensor::full({3, 3}, 1.0f, Device::CUDA);
    auto tensor2_custom = Tensor::full({3, 3}, 1.0f, Device::CUDA);
    auto tensor3_custom = Tensor::full({3, 3}, 1.00001f, Device::CUDA);
    auto tensor4_custom = Tensor::full({3, 3}, 2.0f, Device::CUDA);

    auto tensor1_torch = torch::full({3, 3}, 1.0f,
                                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto tensor2_torch = torch::full({3, 3}, 1.0f,
                                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto tensor3_torch = torch::full({3, 3}, 1.00001f,
                                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto tensor4_torch = torch::full({3, 3}, 2.0f,
                                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // Test our implementation
    EXPECT_TRUE(tensor1_custom.all_close(tensor2_custom));
    EXPECT_TRUE(tensor1_custom.all_close(tensor3_custom, 1e-4f));
    EXPECT_FALSE(tensor1_custom.all_close(tensor3_custom, 1e-6f));
    EXPECT_FALSE(tensor1_custom.all_close(tensor4_custom));

    // Compare with PyTorch allclose
    EXPECT_TRUE(torch::allclose(tensor1_torch, tensor2_torch));
    EXPECT_TRUE(torch::allclose(tensor1_torch, tensor3_torch, 1e-4f));
    EXPECT_FALSE(torch::allclose(tensor1_torch, tensor3_torch, 1e-6f));
    EXPECT_FALSE(torch::allclose(tensor1_torch, tensor4_torch));
}

// ============= Item Tests =============

TEST_F(TensorBasicTest, ItemScalar) {
    auto tensor_custom = Tensor::full({1}, 3.14f, Device::CUDA);
    auto tensor_torch = torch::full({1}, 3.14f,
                                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    float custom_val = tensor_custom.item();
    float torch_val = tensor_torch.item<float>();

    EXPECT_FLOAT_EQ(custom_val, torch_val);
    EXPECT_FLOAT_EQ(custom_val, 3.14f);
}

TEST_F(TensorBasicTest, ItemTemplate) {
    // Float
    auto tensor_float = Tensor::full({1}, 2.5f, Device::CUDA);
    auto torch_float = torch::full({1}, 2.5f,
                                   torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    EXPECT_FLOAT_EQ(tensor_float.item<float>(), torch_float.item<float>());

    // Int
    auto tensor_int = Tensor::full({1}, 42.0f, Device::CUDA, DataType::Int32);
    auto torch_int = torch::full({1}, 42,
                                 torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    EXPECT_EQ(tensor_int.item<int>(), torch_int.item<int>());
}

// ============= Data Type Tests =============

TEST_F(TensorBasicTest, DataTypes) {
    // Float32
    auto tensor_f32 = Tensor::zeros({2, 2}, Device::CUDA, DataType::Float32);
    auto torch_f32 = torch::zeros({2, 2},
                                  torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    EXPECT_EQ(tensor_f32.dtype(), DataType::Float32);
    EXPECT_EQ(dtype_size(tensor_f32.dtype()), sizeof(float));

    // Int32
    auto tensor_i32 = Tensor::zeros({2, 2}, Device::CUDA, DataType::Int32);
    auto torch_i32 = torch::zeros({2, 2},
                                  torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    EXPECT_EQ(tensor_i32.dtype(), DataType::Int32);
    EXPECT_EQ(dtype_size(tensor_i32.dtype()), sizeof(int32_t));

    // Bool
    auto tensor_bool = Tensor::zeros_bool({2, 2}, Device::CUDA);
    auto torch_bool = torch::zeros({2, 2},
                                   torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
    EXPECT_EQ(tensor_bool.dtype(), DataType::Bool);
    EXPECT_EQ(dtype_size(tensor_bool.dtype()), sizeof(uint8_t));
}

// ============= Conversion Tests =============

TEST_F(TensorBasicTest, ToVector) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};

    auto tensor_custom = Tensor::from_vector(data, {2, 2}, Device::CUDA);
    auto tensor_torch = torch::from_blob(
                            const_cast<float*>(data.data()),
                            {2, 2},
                            torch::TensorOptions().dtype(torch::kFloat32))
                            .clone()
                            .to(torch::kCUDA);

    auto custom_vec = tensor_custom.to_vector();
    auto torch_cpu = tensor_torch.to(torch::kCPU);
    auto torch_vec = std::vector<float>(
        torch_cpu.data_ptr<float>(),
        torch_cpu.data_ptr<float>() + torch_cpu.numel());

    ASSERT_EQ(custom_vec.size(), torch_vec.size());
    for (size_t i = 0; i < custom_vec.size(); ++i) {
        EXPECT_FLOAT_EQ(custom_vec[i], torch_vec[i]);
    }
}

TEST_F(TensorBasicTest, ToVectorInt) {
    std::vector<int> data = {1, 2, 3, 4, 5, 6};

    auto tensor_custom = Tensor::from_vector(data, {2, 3}, Device::CUDA);
    auto tensor_torch = torch::from_blob(
                            const_cast<int*>(data.data()),
                            {2, 3},
                            torch::TensorOptions().dtype(torch::kInt32))
                            .clone()
                            .to(torch::kCUDA);

    auto custom_vec = tensor_custom.to_vector_int();
    auto torch_cpu = tensor_torch.to(torch::kCPU);
    auto torch_vec = std::vector<int>(
        torch_cpu.data_ptr<int>(),
        torch_cpu.data_ptr<int>() + torch_cpu.numel());

    ASSERT_EQ(custom_vec.size(), torch_vec.size());
    for (size_t i = 0; i < custom_vec.size(); ++i) {
        EXPECT_EQ(custom_vec[i], torch_vec[i]);
    }
}

TEST_F(TensorBasicTest, ToVectorBool) {
    std::vector<bool> data = {true, false, true, false, true, false};

    auto tensor_custom = Tensor::from_vector(data, {2, 3}, Device::CUDA);

    // PyTorch bool handling
    std::vector<uint8_t> uint8_data(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        uint8_data[i] = data[i] ? 1 : 0;
    }
    auto tensor_torch = torch::from_blob(
                            uint8_data.data(),
                            {2, 3},
                            torch::TensorOptions().dtype(torch::kBool))
                            .clone()
                            .to(torch::kCUDA);

    auto custom_vec = tensor_custom.to_vector_bool();
    auto torch_cpu = tensor_torch.to(torch::kCPU).flatten(); // Flatten to 1D
    auto torch_accessor = torch_cpu.accessor<bool, 1>();     // Now 1D is correct

    ASSERT_EQ(custom_vec.size(), torch_cpu.numel());
    for (size_t i = 0; i < custom_vec.size(); ++i) {
        EXPECT_EQ(custom_vec[i], torch_accessor[i]);
    }
}

// ============= Comprehensive Test =============

TEST_F(TensorBasicTest, ComprehensiveWorkflow) {
    // Create -> Fill -> Clone -> Transfer -> Compare
    auto tensor_custom = Tensor::empty({10, 10}, Device::CUDA);
    auto tensor_torch = torch::empty({10, 10},
                                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // Generate random values once and copy to both tensors (so they match)
    std::vector<float> random_data(100);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& val : random_data) {
        val = dist(gen);
    }

    // Copy same data to both tensors
    cudaMemcpy(tensor_custom.ptr<float>(), random_data.data(),
               random_data.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(tensor_torch.data_ptr<float>(), random_data.data(),
               random_data.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Clone
    auto cloned_custom = tensor_custom.clone();
    auto cloned_torch = tensor_torch.clone();

    // Transfer to CPU
    auto cpu_custom = cloned_custom.cpu();
    auto cpu_torch = cloned_torch.cpu();

    // Basic checks
    EXPECT_EQ(cpu_custom.device(), Device::CPU);
    EXPECT_TRUE(cpu_torch.device().is_cpu());

    // Verify clones match originals (since they have same data)
    compare_tensors(cloned_custom, cloned_torch, 1e-5f, 1e-6f, "Comprehensive_Clone");

    // Values should be preserved through CPU transfer
    compare_tensors(cpu_custom, cpu_torch, 1e-5f, 1e-6f, "Comprehensive_CPU");
}