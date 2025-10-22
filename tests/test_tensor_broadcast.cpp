/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/tensor.hpp"
#include <gtest/gtest.h>
#include <random>
#include <torch/torch.h>

using namespace lfs::core;

// ============= Helper Functions =============

namespace {

    // Helper to create PyTorch tensor from vector data
    torch::Tensor create_torch_tensor(const std::vector<float>& data,
                                      const std::vector<int64_t>& shape,
                                      torch::Device device = torch::kCUDA) {
        // Create on CPU first with proper cloning
        auto cpu_tensor = torch::from_blob(
                              const_cast<float*>(data.data()),
                              shape.empty() ? std::vector<int64_t>{static_cast<int64_t>(data.size())} : shape,
                              torch::TensorOptions().dtype(torch::kFloat32))
                              .clone(); // Clone to own the memory

        // Now move to target device
        return cpu_tensor.to(device);
    }

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

    // Helper to create a tensor with sequential values
    Tensor create_sequential_tensor(TensorShape shape, Device device = Device::CUDA) {
        auto tensor = Tensor::empty(shape, device);
        size_t n = tensor.numel();

        if (device == Device::CUDA) {
            std::vector<float> data(n);
            for (size_t i = 0; i < n; ++i) {
                data[i] = static_cast<float>(i);
            }
            cudaMemcpy(tensor.ptr<float>(), data.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        } else {
            float* data = tensor.ptr<float>();
            for (size_t i = 0; i < n; ++i) {
                data[i] = static_cast<float>(i);
            }
        }

        return tensor;
    }

    // Helper to create PyTorch sequential tensor
    torch::Tensor create_torch_sequential(const std::vector<int64_t>& shape,
                                          torch::Device device = torch::kCUDA) {
        int64_t n = 1;
        for (auto s : shape)
            n *= s;

        std::vector<float> data(n);
        for (int64_t i = 0; i < n; ++i) {
            data[i] = static_cast<float>(i);
        }

        return create_torch_tensor(data, shape, device);
    }

} // anonymous namespace

class TensorBroadcastTest : public ::testing::Test {
protected:
    void SetUp() override {
        torch::manual_seed(42);
        Tensor::manual_seed(42);
    }
};

// ============= Basic Broadcasting Tests =============

TEST_F(TensorBroadcastTest, CanBroadcast) {
    // Test can_broadcast_to method
    auto a = Tensor::empty({3, 4});
    EXPECT_TRUE(a.can_broadcast_to({3, 4})); // Same shape

    auto b = Tensor::empty({3, 1});
    EXPECT_TRUE(b.can_broadcast_to({3, 4})); // Broadcast dim 1

    auto c = Tensor::empty({1, 4});
    EXPECT_TRUE(c.can_broadcast_to({3, 4})); // Broadcast dim 0

    auto d = Tensor::empty({1});
    EXPECT_TRUE(d.can_broadcast_to({3, 4})); // Broadcast scalar

    auto e = Tensor::empty({4});
    EXPECT_TRUE(e.can_broadcast_to({3, 4})); // Broadcast 1D to 2D

    auto f = Tensor::empty({3, 1, 5});
    EXPECT_TRUE(f.can_broadcast_to({3, 4, 5})); // 3D broadcast

    // Incompatible shapes
    auto g = Tensor::empty({3, 4});
    EXPECT_FALSE(g.can_broadcast_to({3, 5})); // Mismatch in dim 1
    EXPECT_FALSE(g.can_broadcast_to({4, 4})); // Mismatch in dim 0

    auto h = Tensor::empty({3});
    EXPECT_FALSE(h.can_broadcast_to({4})); // Different 1D sizes
}

TEST_F(TensorBroadcastTest, BroadcastShape) {
    // Test broadcast_shape method
    auto a = Tensor::empty({3, 4});
    auto b = Tensor::empty({3, 4});
    EXPECT_EQ(a.broadcast_shape(b.shape()), TensorShape({3, 4}));

    auto c = Tensor::empty({3, 1});
    auto d = Tensor::empty({3, 4});
    EXPECT_EQ(c.broadcast_shape(d.shape()), TensorShape({3, 4}));

    auto e = Tensor::empty({1, 4});
    EXPECT_EQ(e.broadcast_shape(a.shape()), TensorShape({3, 4}));

    auto f = Tensor::empty({1});
    EXPECT_EQ(f.broadcast_shape(a.shape()), TensorShape({3, 4}));

    auto g = Tensor::empty({4});
    EXPECT_EQ(g.broadcast_shape(a.shape()), TensorShape({3, 4}));

    auto h = Tensor::empty({1, 1});
    EXPECT_EQ(h.broadcast_shape(a.shape()), TensorShape({3, 4}));

    auto i = Tensor::empty({5, 1, 4});
    auto j = Tensor::empty({5, 3, 4});
    EXPECT_EQ(i.broadcast_shape(j.shape()), TensorShape({5, 3, 4}));

    auto k = Tensor::empty({1, 3, 1});
    auto l = Tensor::empty({5, 1, 4});
    EXPECT_EQ(k.broadcast_shape(l.shape()), TensorShape({5, 3, 4}));
}

// ============= Scalar Broadcasting Tests =============

TEST_F(TensorBroadcastTest, ScalarBroadcast) {
    // Broadcast scalar to various shapes
    std::vector<float> scalar_data = {2.0f};
    std::vector<float> target_data(12, 1.0f);

    auto scalar_custom = Tensor::from_vector(scalar_data, {1}, Device::CUDA);
    auto target_custom = Tensor::from_vector(target_data, {3, 4}, Device::CUDA);

    auto scalar_torch = create_torch_tensor(scalar_data, {1});
    auto target_torch = create_torch_tensor(target_data, {3, 4});

    auto result_custom = scalar_custom.add(target_custom);
    auto result_torch = scalar_torch + target_torch;

    EXPECT_EQ(result_custom.shape(), target_custom.shape());
    compare_tensors(result_custom, result_torch, 1e-5f, 1e-6f, "ScalarBroadcast");
}

// ============= 1D to 2D Broadcasting Tests =============

TEST_F(TensorBroadcastTest, Broadcast1DRow) {
    // Broadcast row vector to matrix
    std::vector<float> row_data(4, 2.0f);
    std::vector<float> matrix_data(12, 1.0f);

    auto row_custom = Tensor::from_vector(row_data, {4}, Device::CUDA);
    auto matrix_custom = Tensor::from_vector(matrix_data, {3, 4}, Device::CUDA);

    auto row_torch = create_torch_tensor(row_data, {4});
    auto matrix_torch = create_torch_tensor(matrix_data, {3, 4});

    auto result_custom = row_custom.add(matrix_custom);
    auto result_torch = row_torch + matrix_torch;

    EXPECT_EQ(result_custom.shape(), matrix_custom.shape());
    compare_tensors(result_custom, result_torch, 1e-5f, 1e-6f, "Broadcast1DRow");
}

TEST_F(TensorBroadcastTest, Broadcast1DColumn) {
    // Broadcast column vector to matrix
    std::vector<float> col_data(3, 3.0f);
    std::vector<float> matrix_data(12, 2.0f);

    auto col_custom = Tensor::from_vector(col_data, {3, 1}, Device::CUDA);
    auto matrix_custom = Tensor::from_vector(matrix_data, {3, 4}, Device::CUDA);

    auto col_torch = create_torch_tensor(col_data, {3, 1});
    auto matrix_torch = create_torch_tensor(matrix_data, {3, 4});

    auto result_custom = col_custom.add(matrix_custom);
    auto result_torch = col_torch + matrix_torch;

    EXPECT_EQ(result_custom.shape(), matrix_custom.shape());
    compare_tensors(result_custom, result_torch, 1e-5f, 1e-6f, "Broadcast1DColumn");
}

// ============= 2D Broadcasting Tests =============

TEST_F(TensorBroadcastTest, BroadcastMatrixColumn) {
    // Create a column vector [0, 1, 2] and broadcast with a matrix
    auto col_custom = create_sequential_tensor({3, 1}, Device::CUDA);
    auto matrix_custom = Tensor::ones({3, 4}, Device::CUDA);

    auto col_torch = create_torch_sequential({3, 1});
    auto matrix_torch = torch::ones({3, 4}, torch::TensorOptions().device(torch::kCUDA));

    auto result_custom = col_custom.add(matrix_custom);
    auto result_torch = col_torch + matrix_torch;

    EXPECT_EQ(result_custom.shape(), TensorShape({3, 4}));
    compare_tensors(result_custom, result_torch, 1e-5f, 1e-6f, "BroadcastMatrixColumn");
}

TEST_F(TensorBroadcastTest, BroadcastMatrixRow) {
    // Create a row vector [0, 1, 2, 3] and broadcast with a matrix
    auto row_custom = create_sequential_tensor({1, 4}, Device::CUDA);
    auto matrix_custom = Tensor::ones({3, 4}, Device::CUDA);

    auto row_torch = create_torch_sequential({1, 4});
    auto matrix_torch = torch::ones({3, 4}, torch::TensorOptions().device(torch::kCUDA));

    auto result_custom = row_custom.add(matrix_custom);
    auto result_torch = row_torch + matrix_torch;

    EXPECT_EQ(result_custom.shape(), TensorShape({3, 4}));
    compare_tensors(result_custom, result_torch, 1e-5f, 1e-6f, "BroadcastMatrixRow");
}

// ============= 3D Broadcasting Tests =============

TEST_F(TensorBroadcastTest, Broadcast3D) {
    // Test 3D broadcasting
    std::vector<float> a_data(6, 2.0f);  // 2*1*3 = 6
    std::vector<float> b_data(24, 1.0f); // 2*4*3 = 24

    auto a_custom = Tensor::from_vector(a_data, {2, 1, 3}, Device::CUDA);
    auto b_custom = Tensor::from_vector(b_data, {2, 4, 3}, Device::CUDA);

    auto a_torch = create_torch_tensor(a_data, {2, 1, 3});
    auto b_torch = create_torch_tensor(b_data, {2, 4, 3});

    auto result_custom = a_custom.add(b_custom);
    auto result_torch = a_torch + b_torch;

    EXPECT_EQ(result_custom.shape(), TensorShape({2, 4, 3}));
    compare_tensors(result_custom, result_torch, 1e-5f, 1e-6f, "Broadcast3D");
}

TEST_F(TensorBroadcastTest, ComplexBroadcast3D) {
    // More complex 3D broadcasting
    std::vector<float> a_data(3, 5.0f); // 1*3*1 = 3
    std::vector<float> b_data(8, 3.0f); // 2*1*4 = 8

    auto a_custom = Tensor::from_vector(a_data, {1, 3, 1}, Device::CUDA);
    auto b_custom = Tensor::from_vector(b_data, {2, 1, 4}, Device::CUDA);

    auto a_torch = create_torch_tensor(a_data, {1, 3, 1});
    auto b_torch = create_torch_tensor(b_data, {2, 1, 4});

    auto result_custom = a_custom.mul(b_custom);
    auto result_torch = a_torch * b_torch;

    EXPECT_EQ(result_custom.shape(), TensorShape({2, 3, 4}));
    compare_tensors(result_custom, result_torch, 1e-5f, 1e-6f, "ComplexBroadcast3D");
}

// ============= All Operations Broadcasting Tests =============

TEST_F(TensorBroadcastTest, BroadcastAddition) {
    std::vector<float> a_data(3, 2.0f);
    std::vector<float> b_data(4, 3.0f);

    auto a_custom = Tensor::from_vector(a_data, {3, 1}, Device::CUDA);
    auto b_custom = Tensor::from_vector(b_data, {1, 4}, Device::CUDA);

    auto a_torch = create_torch_tensor(a_data, {3, 1});
    auto b_torch = create_torch_tensor(b_data, {1, 4});

    auto result_custom = a_custom + b_custom;
    auto result_torch = a_torch + b_torch;

    EXPECT_EQ(result_custom.shape(), TensorShape({3, 4}));
    compare_tensors(result_custom, result_torch, 1e-5f, 1e-6f, "BroadcastAddition");
}

TEST_F(TensorBroadcastTest, BroadcastSubtraction) {
    std::vector<float> a_data(6, 10.0f); // 2*3*1 = 6
    std::vector<float> b_data(12, 3.0f); // 3*4 = 12

    auto a_custom = Tensor::from_vector(a_data, {2, 3, 1}, Device::CUDA);
    auto b_custom = Tensor::from_vector(b_data, {3, 4}, Device::CUDA);

    auto a_torch = create_torch_tensor(a_data, {2, 3, 1});
    auto b_torch = create_torch_tensor(b_data, {3, 4});

    auto result_custom = a_custom - b_custom;
    auto result_torch = a_torch - b_torch;

    EXPECT_EQ(result_custom.shape(), TensorShape({2, 3, 4}));
    compare_tensors(result_custom, result_torch, 1e-5f, 1e-6f, "BroadcastSubtraction");
}

TEST_F(TensorBroadcastTest, BroadcastMultiplication) {
    auto a_custom = create_sequential_tensor({3, 1}, Device::CUDA); // [0, 1, 2]^T
    auto b_custom = create_sequential_tensor({1, 4}, Device::CUDA); // [0, 1, 2, 3]

    auto a_torch = create_torch_sequential({3, 1});
    auto b_torch = create_torch_sequential({1, 4});

    auto result_custom = a_custom * b_custom;
    auto result_torch = a_torch * b_torch;

    EXPECT_EQ(result_custom.shape(), TensorShape({3, 4}));
    compare_tensors(result_custom, result_torch, 1e-5f, 1e-6f, "BroadcastMultiplication");
}

TEST_F(TensorBroadcastTest, BroadcastDivision) {
    std::vector<float> a_data(6, 12.0f); // 2*3*1 = 6
    std::vector<float> b_data(4, 3.0f);  // 1*4 = 4

    auto a_custom = Tensor::from_vector(a_data, {2, 3, 1}, Device::CUDA);
    auto b_custom = Tensor::from_vector(b_data, {1, 4}, Device::CUDA);

    auto a_torch = create_torch_tensor(a_data, {2, 3, 1});
    auto b_torch = create_torch_tensor(b_data, {1, 4});

    auto result_custom = a_custom / b_custom;
    auto result_torch = a_torch / b_torch;

    EXPECT_EQ(result_custom.shape(), TensorShape({2, 3, 4}));
    compare_tensors(result_custom, result_torch, 1e-5f, 1e-6f, "BroadcastDivision");
}

// ============= Edge Cases =============

TEST_F(TensorBroadcastTest, BroadcastWithEmpty) {
    std::vector<float> empty_data;
    std::vector<float> normal_data(12, 1.0f);

    auto empty_custom = Tensor::from_vector(empty_data, {0}, Device::CUDA);
    auto normal_custom = Tensor::from_vector(normal_data, {3, 4}, Device::CUDA);

    // Broadcasting with empty should fail
    auto result_custom = empty_custom.add(normal_custom);
    EXPECT_FALSE(result_custom.is_valid());

    // Verify PyTorch also has issues with empty tensors
    auto empty_torch = torch::empty({0}, torch::TensorOptions().device(torch::kCUDA));
    auto normal_torch = torch::ones({3, 4}, torch::TensorOptions().device(torch::kCUDA));

    // PyTorch may produce empty result or error - our implementation returns invalid
    LOG_INFO("Empty tensor broadcasting: Our implementation returns invalid tensor");
}

TEST_F(TensorBroadcastTest, BroadcastSingleElement) {
    std::vector<float> single_data = {5.0f};
    std::vector<float> matrix_data(60, 2.0f); // 3*4*5 = 60

    auto single_custom = Tensor::from_vector(single_data, {1, 1, 1}, Device::CUDA);
    auto matrix_custom = Tensor::from_vector(matrix_data, {3, 4, 5}, Device::CUDA);

    auto single_torch = create_torch_tensor(single_data, {1, 1, 1});
    auto matrix_torch = create_torch_tensor(matrix_data, {3, 4, 5});

    auto result_custom = single_custom.add(matrix_custom);
    auto result_torch = single_torch + matrix_torch;

    EXPECT_EQ(result_custom.shape(), matrix_custom.shape());
    compare_tensors(result_custom, result_torch, 1e-5f, 1e-6f, "BroadcastSingleElement");
}

TEST_F(TensorBroadcastTest, BroadcastLargeTensors) {
    // Test with larger tensors
    std::vector<float> a_data(100, 1.0f);
    std::vector<float> b_data(200, 1.0f);

    auto a_custom = Tensor::from_vector(a_data, {100, 1}, Device::CUDA);
    auto b_custom = Tensor::from_vector(b_data, {1, 200}, Device::CUDA);

    auto a_torch = create_torch_tensor(a_data, {100, 1});
    auto b_torch = create_torch_tensor(b_data, {1, 200});

    auto result_custom = a_custom.add(b_custom);
    auto result_torch = a_torch + b_torch;

    EXPECT_EQ(result_custom.shape(), TensorShape({100, 200}));
    EXPECT_EQ(result_custom.numel(), 20000);
    compare_tensors(result_custom, result_torch, 1e-5f, 1e-6f, "BroadcastLargeTensors");
}

// ============= CPU Broadcasting Tests =============

TEST_F(TensorBroadcastTest, BroadcastOnCPU) {
    std::vector<float> a_data(3, 4.0f);
    std::vector<float> b_data(5, 2.0f);

    auto a_custom = Tensor::from_vector(a_data, {3, 1}, Device::CPU);
    auto b_custom = Tensor::from_vector(b_data, {1, 5}, Device::CPU);

    auto a_torch = create_torch_tensor(a_data, {3, 1}, torch::kCPU);
    auto b_torch = create_torch_tensor(b_data, {1, 5}, torch::kCPU);

    auto result_custom = a_custom.add(b_custom);
    auto result_torch = a_torch + b_torch;

    EXPECT_EQ(result_custom.shape(), TensorShape({3, 5}));
    EXPECT_EQ(result_custom.device(), Device::CPU);
    compare_tensors(result_custom, result_torch, 1e-5f, 1e-6f, "BroadcastOnCPU");
}

// ============= Broadcasting Expansion Tests =============

TEST_F(TensorBroadcastTest, ExpandTensor) {
    std::vector<float> data(3, 2.0f);

    auto tensor_custom = Tensor::from_vector(data, {3, 1}, Device::CUDA);
    auto tensor_torch = create_torch_tensor(data, {3, 1});

    auto expanded_custom = tensor_custom.expand({3, 4});
    auto expanded_torch = tensor_torch.expand({3, 4});

    EXPECT_EQ(expanded_custom.shape(), TensorShape({3, 4}));
    compare_tensors(expanded_custom, expanded_torch, 1e-6f, 1e-7f, "ExpandTensor");
}

TEST_F(TensorBroadcastTest, BroadcastToTensor) {
    std::vector<float> data(3, 2.0f);

    auto tensor_custom = Tensor::from_vector(data, {3, 1}, Device::CUDA);
    auto tensor_torch = create_torch_tensor(data, {3, 1});

    auto broadcasted_custom = tensor_custom.broadcast_to({3, 4});
    auto broadcasted_torch = tensor_torch.broadcast_to({3, 4});

    EXPECT_EQ(broadcasted_custom.shape(), TensorShape({3, 4}));
    compare_tensors(broadcasted_custom, broadcasted_torch, 1e-6f, 1e-7f, "BroadcastToTensor");
}

TEST_F(TensorBroadcastTest, ExpandMultipleDimensions) {
    std::vector<float> data(3, 3.0f); // 1*3*1 = 3

    auto tensor_custom = Tensor::from_vector(data, {1, 3, 1}, Device::CUDA);
    auto tensor_torch = create_torch_tensor(data, {1, 3, 1});

    auto expanded_custom = tensor_custom.expand({2, 3, 4});
    auto expanded_torch = tensor_torch.expand({2, 3, 4});

    EXPECT_EQ(expanded_custom.shape(), TensorShape({2, 3, 4}));
    compare_tensors(expanded_custom, expanded_torch, 1e-6f, 1e-7f, "ExpandMultipleDimensions");
}

// ============= Broadcasting with Mixed Shapes Tests =============

TEST_F(TensorBroadcastTest, MixedRankBroadcast) {
    // Test broadcasting between different ranks
    std::vector<float> rank1_data(5, 1.0f);
    std::vector<float> rank2_data(15, 2.0f); // 3*5 = 15
    std::vector<float> rank3_data(30, 3.0f); // 2*3*5 = 30

    auto rank1_custom = Tensor::from_vector(rank1_data, {5}, Device::CUDA);
    auto rank2_custom = Tensor::from_vector(rank2_data, {3, 5}, Device::CUDA);
    auto rank3_custom = Tensor::from_vector(rank3_data, {2, 3, 5}, Device::CUDA);

    auto rank1_torch = create_torch_tensor(rank1_data, {5});
    auto rank2_torch = create_torch_tensor(rank2_data, {3, 5});
    auto rank3_torch = create_torch_tensor(rank3_data, {2, 3, 5});

    // 1D + 2D
    auto result12_custom = rank1_custom.add(rank2_custom);
    auto result12_torch = rank1_torch + rank2_torch;
    EXPECT_EQ(result12_custom.shape(), TensorShape({3, 5}));
    compare_tensors(result12_custom, result12_torch, 1e-5f, 1e-6f, "MixedRank_1D+2D");

    // 1D + 3D
    auto result13_custom = rank1_custom.add(rank3_custom);
    auto result13_torch = rank1_torch + rank3_torch;
    EXPECT_EQ(result13_custom.shape(), TensorShape({2, 3, 5}));
    compare_tensors(result13_custom, result13_torch, 1e-5f, 1e-6f, "MixedRank_1D+3D");

    // 2D + 3D
    auto result23_custom = rank2_custom.add(rank3_custom);
    auto result23_torch = rank2_torch + rank3_torch;
    EXPECT_EQ(result23_custom.shape(), TensorShape({2, 3, 5}));
    compare_tensors(result23_custom, result23_torch, 1e-5f, 1e-6f, "MixedRank_2D+3D");
}

// ============= Performance Test =============

TEST_F(TensorBroadcastTest, BroadcastPerformance) {
    // Test performance of broadcasting large tensors
    const size_t rows = 1000;
    const size_t cols = 1000;

    auto col_vector_custom = Tensor::randn({rows, 1}, Device::CUDA);
    auto row_vector_custom = Tensor::randn({1, cols}, Device::CUDA);

    auto start = std::chrono::high_resolution_clock::now();
    auto result_custom = col_vector_custom.add(row_vector_custom);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    EXPECT_EQ(result_custom.shape(), TensorShape({rows, cols}));

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    LOG_INFO("Broadcast {}x1 + 1x{} took {} ms", rows, cols, duration.count());

    // Also benchmark PyTorch for comparison
    auto col_vector_torch = torch::randn({static_cast<int64_t>(rows), 1},
                                         torch::TensorOptions().device(torch::kCUDA));
    auto row_vector_torch = torch::randn({1, static_cast<int64_t>(cols)},
                                         torch::TensorOptions().device(torch::kCUDA));

    auto torch_start = std::chrono::high_resolution_clock::now();
    auto result_torch = col_vector_torch + row_vector_torch;
    cudaDeviceSynchronize();
    auto torch_end = std::chrono::high_resolution_clock::now();

    auto torch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(torch_end - torch_start);
    LOG_INFO("PyTorch broadcast {}x1 + 1x{} took {} ms", rows, cols, torch_duration.count());
}

// ============= Complex Expression Tests =============

TEST_F(TensorBroadcastTest, ChainedBroadcastOperations) {
    // Test chained operations with broadcasting
    std::vector<float> a_data(3, 2.0f);
    std::vector<float> b_data(4, 3.0f);
    std::vector<float> c_data(12, 1.0f); // 3*4 = 12

    auto a_custom = Tensor::from_vector(a_data, {3, 1}, Device::CUDA);
    auto b_custom = Tensor::from_vector(b_data, {1, 4}, Device::CUDA);
    auto c_custom = Tensor::from_vector(c_data, {3, 4}, Device::CUDA);

    auto a_torch = create_torch_tensor(a_data, {3, 1});
    auto b_torch = create_torch_tensor(b_data, {1, 4});
    auto c_torch = create_torch_tensor(c_data, {3, 4});

    // (a + b) * c
    auto result_custom = (a_custom + b_custom) * c_custom;
    auto result_torch = (a_torch + b_torch) * c_torch;

    EXPECT_EQ(result_custom.shape(), TensorShape({3, 4}));
    compare_tensors(result_custom, result_torch, 1e-5f, 1e-6f, "ChainedBroadcast");
}

TEST_F(TensorBroadcastTest, ComplexBroadcastExpression) {
    // More complex expression with multiple broadcasts
    auto a_custom = create_sequential_tensor({3, 1}, Device::CUDA); // [0, 1, 2]^T
    auto b_custom = create_sequential_tensor({1, 4}, Device::CUDA); // [0, 1, 2, 3]
    auto c_custom = Tensor::ones({3, 4}, Device::CUDA);

    auto a_torch = create_torch_sequential({3, 1});
    auto b_torch = create_torch_sequential({1, 4});
    auto c_torch = torch::ones({3, 4}, torch::TensorOptions().device(torch::kCUDA));

    // a * b + c
    auto result_custom = a_custom * b_custom + c_custom;
    auto result_torch = a_torch * b_torch + c_torch;

    EXPECT_EQ(result_custom.shape(), TensorShape({3, 4}));
    compare_tensors(result_custom, result_torch, 1e-5f, 1e-6f, "ComplexBroadcastExpression");
}

TEST_F(TensorBroadcastTest, ScalarMultiplication) {
    std::vector<float> scalar_data = {3.0f};
    std::vector<float> matrix_data(6, 2.0f); // 2*3 = 6

    auto scalar_custom = Tensor::from_vector(scalar_data, {1}, Device::CUDA);
    auto matrix_custom = Tensor::from_vector(matrix_data, {2, 3}, Device::CUDA);

    auto scalar_torch = create_torch_tensor(scalar_data, {1});
    auto matrix_torch = create_torch_tensor(matrix_data, {2, 3});

    auto result_custom = scalar_custom.mul(matrix_custom);
    auto result_torch = scalar_torch * matrix_torch;

    EXPECT_EQ(result_custom.shape(), matrix_custom.shape());
    compare_tensors(result_custom, result_torch, 1e-5f, 1e-6f, "ScalarMultiplication");
}

// ============= Random Data Tests =============

TEST_F(TensorBroadcastTest, RandomBroadcastOperations) {
    // Generate random data for robust testing
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

    for (int test = 0; test < 3; ++test) {
        std::vector<float> a_data(5);
        std::vector<float> b_data(4);
        for (size_t i = 0; i < a_data.size(); ++i) {
            a_data[i] = dist(gen);
        }
        for (size_t i = 0; i < b_data.size(); ++i) {
            b_data[i] = dist(gen);
        }

        auto a_custom = Tensor::from_vector(a_data, {5, 1}, Device::CUDA);
        auto b_custom = Tensor::from_vector(b_data, {1, 4}, Device::CUDA);

        auto a_torch = create_torch_tensor(a_data, {5, 1});
        auto b_torch = create_torch_tensor(b_data, {1, 4});

        // Test addition
        auto add_custom = a_custom + b_custom;
        auto add_torch = a_torch + b_torch;
        compare_tensors(add_custom, add_torch, 1e-4f, 1e-5f,
                        "RandomBroadcast_Add_" + std::to_string(test));

        // Test multiplication
        auto mul_custom = a_custom * b_custom;
        auto mul_torch = a_torch * b_torch;
        compare_tensors(mul_custom, mul_torch, 1e-4f, 1e-5f,
                        "RandomBroadcast_Mul_" + std::to_string(test));
    }
}

// ============= Comparison Operations with Broadcasting =============

TEST_F(TensorBroadcastTest, BroadcastComparison) {
    std::vector<float> a_data = {1.0f, 2.0f, 3.0f};
    std::vector<float> b_data = {2.0f, 2.0f, 2.0f, 2.0f};

    auto a_custom = Tensor::from_vector(a_data, {3, 1}, Device::CUDA);
    auto b_custom = Tensor::from_vector(b_data, {1, 4}, Device::CUDA);

    auto a_torch = create_torch_tensor(a_data, {3, 1});
    auto b_torch = create_torch_tensor(b_data, {1, 4});

    // Test gt (greater than)
    auto gt_custom = a_custom.gt(b_custom);
    auto gt_torch = a_torch > b_torch;

    EXPECT_EQ(gt_custom.shape(), TensorShape({3, 4}));
    EXPECT_EQ(gt_custom.dtype(), DataType::Bool);

    // Convert both to bool vectors for comparison
    auto gt_custom_vals = gt_custom.to_vector_bool();
    auto gt_torch_cpu = gt_torch.to(torch::kCPU).flatten();
    auto gt_torch_acc = gt_torch_cpu.accessor<bool, 1>();

    for (size_t i = 0; i < gt_custom_vals.size(); ++i) {
        EXPECT_EQ(gt_custom_vals[i], gt_torch_acc[i])
            << "Mismatch at index " << i;
    }
}

// ============= Broadcasting with Maximum/Minimum =============

TEST_F(TensorBroadcastTest, BroadcastMaximum) {
    auto a_custom = create_sequential_tensor({3, 1}, Device::CUDA);
    auto b_custom = Tensor::full({1, 4}, 1.5f, Device::CUDA);

    auto a_torch = create_torch_sequential({3, 1});
    auto b_torch = torch::full({1, 4}, 1.5f, torch::TensorOptions().device(torch::kCUDA));

    auto result_custom = a_custom.maximum(b_custom);
    auto result_torch = torch::maximum(a_torch, b_torch);

    EXPECT_EQ(result_custom.shape(), TensorShape({3, 4}));
    compare_tensors(result_custom, result_torch, 1e-6f, 1e-7f, "BroadcastMaximum");
}

TEST_F(TensorBroadcastTest, BroadcastMinimum) {
    auto a_custom = create_sequential_tensor({3, 1}, Device::CUDA);
    auto b_custom = Tensor::full({1, 4}, 1.5f, Device::CUDA);

    auto a_torch = create_torch_sequential({3, 1});
    auto b_torch = torch::full({1, 4}, 1.5f, torch::TensorOptions().device(torch::kCUDA));

    auto result_custom = a_custom.minimum(b_custom);
    auto result_torch = torch::minimum(a_torch, b_torch);

    EXPECT_EQ(result_custom.shape(), TensorShape({3, 4}));
    compare_tensors(result_custom, result_torch, 1e-6f, 1e-7f, "BroadcastMinimum");
}