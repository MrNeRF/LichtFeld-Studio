/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/tensor.hpp"
#include <chrono>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <torch/torch.h>

using namespace lfs::core;

// ============= Helper Functions =============

namespace {

    void compare_tensors(const Tensor& custom, const torch::Tensor& reference,
                         float rtol = 1e-4f, float atol = 1e-5f, const std::string& msg = "") {
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

class TensorMatrixTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA is not available for testing";
        torch::manual_seed(42);
        Tensor::manual_seed(42);
        gen.seed(42);
    }

    std::mt19937 gen;
};

// ============= Matrix Multiplication Tests =============

TEST_F(TensorMatrixTest, MatMul2D) {
    // Test basic 2D matrix multiplication: (2x3) @ (3x2) = (2x2)
    std::vector<float> data_a = {1, 2, 3,
                                 4, 5, 6}; // 2x3
    std::vector<float> data_b = {7, 8,
                                 9, 10,
                                 11, 12}; // 3x2

    auto custom_a = Tensor::from_vector(data_a, {2, 3}, Device::CUDA);
    auto custom_b = Tensor::from_vector(data_b, {3, 2}, Device::CUDA);

    auto torch_a = torch::tensor(data_a, torch::TensorOptions().device(torch::kCUDA))
                       .reshape({2, 3});
    auto torch_b = torch::tensor(data_b, torch::TensorOptions().device(torch::kCUDA))
                       .reshape({3, 2});

    auto custom_result = custom_a.matmul(custom_b);
    auto torch_result = torch::matmul(torch_a, torch_b);

    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "MatMul2D");

    // Test mm() alias - should give same result
    auto custom_mm = custom_a.mm(custom_b);
    compare_tensors(custom_mm, torch_result, 1e-4f, 1e-5f, "MM_Alias");
}

TEST_F(TensorMatrixTest, MatMulVectorMatrix) {
    // Test vector-matrix multiplication: (3,) @ (3x2) = (2,)
    std::vector<float> vec_data = {1, 2, 3};
    std::vector<float> mat_data = {4, 5,
                                   6, 7,
                                   8, 9}; // 3x2

    auto custom_vec = Tensor::from_vector(vec_data, {3}, Device::CUDA);
    auto custom_mat = Tensor::from_vector(mat_data, {3, 2}, Device::CUDA);

    auto torch_vec = torch::tensor(vec_data, torch::TensorOptions().device(torch::kCUDA));
    auto torch_mat = torch::tensor(mat_data, torch::TensorOptions().device(torch::kCUDA))
                         .reshape({3, 2});

    auto custom_result = custom_vec.matmul(custom_mat);
    auto torch_result = torch::matmul(torch_vec, torch_mat);

    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "VectorMatrix");
}

TEST_F(TensorMatrixTest, MatMulMatrixVector) {
    // Test matrix-vector multiplication: (2x3) @ (3,) = (2,)
    std::vector<float> mat_data = {1, 2, 3,
                                   4, 5, 6}; // 2x3
    std::vector<float> vec_data = {7, 8, 9};

    auto custom_mat = Tensor::from_vector(mat_data, {2, 3}, Device::CUDA);
    auto custom_vec = Tensor::from_vector(vec_data, {3}, Device::CUDA);

    auto torch_mat = torch::tensor(mat_data, torch::TensorOptions().device(torch::kCUDA))
                         .reshape({2, 3});
    auto torch_vec = torch::tensor(vec_data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_mat.matmul(custom_vec);
    auto torch_result = torch::matmul(torch_mat, torch_vec);

    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "MatrixVector");
}

TEST_F(TensorMatrixTest, MatMulVectorVector) {
    // Test vector-vector multiplication (dot product): (4,) @ (4,) = scalar
    std::vector<float> vec1_data = {1, 2, 3, 4};
    std::vector<float> vec2_data = {5, 6, 7, 8};

    auto custom_vec1 = Tensor::from_vector(vec1_data, {4}, Device::CUDA);
    auto custom_vec2 = Tensor::from_vector(vec2_data, {4}, Device::CUDA);

    auto torch_vec1 = torch::tensor(vec1_data, torch::TensorOptions().device(torch::kCUDA));
    auto torch_vec2 = torch::tensor(vec2_data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_vec1.matmul(custom_vec2);
    auto torch_result = torch::matmul(torch_vec1, torch_vec2);

    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "VectorVector");
}

TEST_F(TensorMatrixTest, DotProduct) {
    // Test explicit dot product of two vectors
    std::vector<float> vec1_data = {1, 2, 3, 4};
    std::vector<float> vec2_data = {5, 6, 7, 8};

    auto custom_vec1 = Tensor::from_vector(vec1_data, {4}, Device::CUDA);
    auto custom_vec2 = Tensor::from_vector(vec2_data, {4}, Device::CUDA);

    auto torch_vec1 = torch::tensor(vec1_data, torch::TensorOptions().device(torch::kCUDA));
    auto torch_vec2 = torch::tensor(vec2_data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_vec1.dot(custom_vec2);
    auto torch_result = torch::dot(torch_vec1, torch_vec2);

    // Result should be a scalar
    EXPECT_EQ(custom_result.numel(), 1);
    EXPECT_NEAR(custom_result.item(), torch_result.item<float>(), 1e-4f);
}

TEST_F(TensorMatrixTest, BatchMatMul) {
    // Test batch matrix multiplication: (B, M, N) @ (B, N, P) = (B, M, P)
    // (2, 3, 4) @ (2, 4, 5) = (2, 3, 5)
    std::vector<float> data_a(2 * 3 * 4); // batch=2, 3x4
    std::vector<float> data_b(2 * 4 * 5); // batch=2, 4x5

    // Fill with sequential values
    std::iota(data_a.begin(), data_a.end(), 1.0f);
    std::iota(data_b.begin(), data_b.end(), 1.0f);

    auto custom_a = Tensor::from_vector(data_a, {2, 3, 4}, Device::CUDA);
    auto custom_b = Tensor::from_vector(data_b, {2, 4, 5}, Device::CUDA);

    auto torch_a = torch::tensor(data_a, torch::TensorOptions().device(torch::kCUDA))
                       .reshape({2, 3, 4});
    auto torch_b = torch::tensor(data_b, torch::TensorOptions().device(torch::kCUDA))
                       .reshape({2, 4, 5});

    auto custom_result = custom_a.bmm(custom_b);
    auto torch_result = torch::bmm(torch_a, torch_b);

    compare_tensors(custom_result, torch_result, 1e-3f, 1e-4f, "BatchMatMul");
}

TEST_F(TensorMatrixTest, BatchMatMulBroadcast) {
    // Test batch matmul with broadcasting in batch dimension
    // (3, 4) @ (2, 4, 5) should broadcast to (2, 3, 5)
    std::vector<float> data_a(3 * 4);
    std::vector<float> data_b(2 * 4 * 5);

    std::iota(data_a.begin(), data_a.end(), 1.0f);
    std::iota(data_b.begin(), data_b.end(), 1.0f);

    auto custom_a = Tensor::from_vector(data_a, {3, 4}, Device::CUDA);
    auto custom_b = Tensor::from_vector(data_b, {2, 4, 5}, Device::CUDA);

    auto torch_a = torch::tensor(data_a, torch::TensorOptions().device(torch::kCUDA))
                       .reshape({3, 4});
    auto torch_b = torch::tensor(data_b, torch::TensorOptions().device(torch::kCUDA))
                       .reshape({2, 4, 5});

    auto custom_result = custom_a.matmul(custom_b);
    auto torch_result = torch::matmul(torch_a, torch_b);

    compare_tensors(custom_result, torch_result, 1e-3f, 1e-4f, "BatchMatMul_Broadcast");
}

// ============= Transpose Tests =============

TEST_F(TensorMatrixTest, Transpose2D) {
    // Test 2D transpose: (2x3) -> (3x2)
    std::vector<float> data = {1, 2, 3,
                               4, 5, 6}; // 2x3

    auto custom_tensor = Tensor::from_vector(data, {2, 3}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA))
                            .reshape({2, 3});

    auto custom_result = custom_tensor.t();
    auto torch_result = torch_tensor.t();

    // Check shape
    EXPECT_EQ(custom_result.shape()[0], 3);
    EXPECT_EQ(custom_result.shape()[1], 2);

    compare_tensors(custom_result, torch_result, 1e-5f, 1e-6f, "Transpose2D");
}

TEST_F(TensorMatrixTest, Transpose3D) {
    // Test 3D transpose (transposes last two dimensions)
    // (2, 3, 4) -> (2, 4, 3)
    std::vector<float> data(2 * 3 * 4);
    std::iota(data.begin(), data.end(), 1.0f);

    auto custom_tensor = Tensor::from_vector(data, {2, 3, 4}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA))
                            .reshape({2, 3, 4});

    auto custom_result = custom_tensor.t();
    auto torch_result = torch_tensor.transpose(-2, -1);

    // Check shape
    EXPECT_EQ(custom_result.shape()[0], 2);
    EXPECT_EQ(custom_result.shape()[1], 4);
    EXPECT_EQ(custom_result.shape()[2], 3);

    compare_tensors(custom_result, torch_result, 1e-5f, 1e-6f, "Transpose3D");
}

TEST_F(TensorMatrixTest, TransposeSpecificDims) {
    // Test transpose with specific dimensions: swap dim 0 and 1
    // (2, 3, 4) -> (3, 2, 4)
    std::vector<float> data(2 * 3 * 4);
    std::iota(data.begin(), data.end(), 1.0f);

    auto custom_tensor = Tensor::from_vector(data, {2, 3, 4}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA))
                            .reshape({2, 3, 4});

    auto custom_result = custom_tensor.transpose(0, 1);
    auto torch_result = torch_tensor.transpose(0, 1);

    // Check shape
    EXPECT_EQ(custom_result.shape()[0], 3);
    EXPECT_EQ(custom_result.shape()[1], 2);
    EXPECT_EQ(custom_result.shape()[2], 4);

    compare_tensors(custom_result, torch_result, 1e-5f, 1e-6f, "TransposeSpecificDims");
}

TEST_F(TensorMatrixTest, TransposeNegativeIndices) {
    // Test transpose with negative indices
    // (2, 3, 4, 5) transpose(-1, -2) -> (2, 3, 5, 4)
    std::vector<float> data(2 * 3 * 4 * 5);
    std::iota(data.begin(), data.end(), 1.0f);

    auto custom_tensor = Tensor::from_vector(data, {2, 3, 4, 5}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA))
                            .reshape({2, 3, 4, 5});

    auto custom_result = custom_tensor.transpose(-2, -1);
    auto torch_result = torch_tensor.transpose(-2, -1);

    // Check shape
    EXPECT_EQ(custom_result.shape()[0], 2);
    EXPECT_EQ(custom_result.shape()[1], 3);
    EXPECT_EQ(custom_result.shape()[2], 5);
    EXPECT_EQ(custom_result.shape()[3], 4);

    compare_tensors(custom_result, torch_result, 1e-5f, 1e-6f, "TransposeNegative");
}

// ============= Matrix Creation Tests =============

TEST_F(TensorMatrixTest, Eye) {
    // Test identity matrix creation
    auto custom_result = Tensor::eye(5, Device::CUDA);
    auto torch_result = torch::eye(5, torch::TensorOptions().device(torch::kCUDA));

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "Eye_Square");

    // Test rectangular identity matrix
    auto custom_rect = Tensor::eye(3, 5, Device::CUDA);
    auto torch_rect = torch::eye(3, 5, torch::TensorOptions().device(torch::kCUDA));

    compare_tensors(custom_rect, torch_rect, 1e-6f, 1e-7f, "Eye_Rectangular");
}

TEST_F(TensorMatrixTest, Diag) {
    // Test creating diagonal matrix from vector
    std::vector<float> diag_data = {1, 2, 3, 4, 5};

    auto custom_diag_vec = Tensor::from_vector(diag_data, {5}, Device::CUDA);
    auto torch_diag_vec = torch::tensor(diag_data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = Tensor::diag(custom_diag_vec);
    auto torch_result = torch::diag(torch_diag_vec);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "Diag");

    // Verify it's actually diagonal
    EXPECT_EQ(custom_result.shape()[0], 5);
    EXPECT_EQ(custom_result.shape()[1], 5);
}

// ============= Complex Matrix Operations =============

TEST_F(TensorMatrixTest, MatMulChain) {
    // Test chaining multiple matrix multiplications: A @ B @ C
    std::vector<float> a_data = {1, 2, 3, 4};    // 2x2
    std::vector<float> b_data = {5, 6, 7, 8};    // 2x2
    std::vector<float> c_data = {9, 10, 11, 12}; // 2x2

    auto custom_a = Tensor::from_vector(a_data, {2, 2}, Device::CUDA);
    auto custom_b = Tensor::from_vector(b_data, {2, 2}, Device::CUDA);
    auto custom_c = Tensor::from_vector(c_data, {2, 2}, Device::CUDA);

    auto torch_a = torch::tensor(a_data, torch::TensorOptions().device(torch::kCUDA))
                       .reshape({2, 2});
    auto torch_b = torch::tensor(b_data, torch::TensorOptions().device(torch::kCUDA))
                       .reshape({2, 2});
    auto torch_c = torch::tensor(c_data, torch::TensorOptions().device(torch::kCUDA))
                       .reshape({2, 2});

    auto custom_result = custom_a.matmul(custom_b).matmul(custom_c);
    auto torch_result = torch::matmul(torch::matmul(torch_a, torch_b), torch_c);

    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "MatMulChain");
}

TEST_F(TensorMatrixTest, TransposeMatMul) {
    // Test A^T @ B
    std::vector<float> a_data = {1, 2, 3,
                                 4, 5, 6}; // 2x3
    std::vector<float> b_data = {7, 8,
                                 9, 10}; // 2x2

    auto custom_a = Tensor::from_vector(a_data, {2, 3}, Device::CUDA);
    auto custom_b = Tensor::from_vector(b_data, {2, 2}, Device::CUDA);

    auto torch_a = torch::tensor(a_data, torch::TensorOptions().device(torch::kCUDA))
                       .reshape({2, 3});
    auto torch_b = torch::tensor(b_data, torch::TensorOptions().device(torch::kCUDA))
                       .reshape({2, 2});

    auto custom_result = custom_a.t().matmul(custom_b);
    auto torch_result = torch::matmul(torch_a.t(), torch_b);

    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "TransposeMatMul");
}

TEST_F(TensorMatrixTest, MatMulTranspose) {
    // Test A @ B^T
    std::vector<float> a_data = {1, 2,
                                 3, 4,
                                 5, 6}; // 3x2
    std::vector<float> b_data = {7, 8,
                                 9, 10,
                                 11, 12}; // 3x2

    auto custom_a = Tensor::from_vector(a_data, {3, 2}, Device::CUDA);
    auto custom_b = Tensor::from_vector(b_data, {3, 2}, Device::CUDA);

    auto torch_a = torch::tensor(a_data, torch::TensorOptions().device(torch::kCUDA))
                       .reshape({3, 2});
    auto torch_b = torch::tensor(b_data, torch::TensorOptions().device(torch::kCUDA))
                       .reshape({3, 2});

    auto custom_result = custom_a.matmul(custom_b.t());
    auto torch_result = torch::matmul(torch_a, torch_b.t());

    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "MatMulTranspose");
}

TEST_F(TensorMatrixTest, DoubleTransposeMatMul) {
    // Test A^T @ B^T
    std::vector<float> a_data = {1, 2, 3,
                                 4, 5, 6}; // 2x3
    std::vector<float> b_data = {7, 8,
                                 9, 10,
                                 11, 12}; // 3x2

    auto custom_a = Tensor::from_vector(a_data, {2, 3}, Device::CUDA);
    auto custom_b = Tensor::from_vector(b_data, {3, 2}, Device::CUDA);

    auto torch_a = torch::tensor(a_data, torch::TensorOptions().device(torch::kCUDA))
                       .reshape({2, 3});
    auto torch_b = torch::tensor(b_data, torch::TensorOptions().device(torch::kCUDA))
                       .reshape({3, 2});

    auto custom_result = custom_a.t().matmul(custom_b.t());
    auto torch_result = torch::matmul(torch_a.t(), torch_b.t());

    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "DoubleTransposeMatMul");
}

// ============= Edge Cases =============

TEST_F(TensorMatrixTest, SingleElementMatMul) {
    // Test 1x1 matrix multiplication
    auto custom_a = Tensor::full({1, 1}, 3.0f, Device::CUDA);
    auto custom_b = Tensor::full({1, 1}, 4.0f, Device::CUDA);

    auto torch_a = torch::full({1, 1}, 3.0f, torch::TensorOptions().device(torch::kCUDA));
    auto torch_b = torch::full({1, 1}, 4.0f, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_a.matmul(custom_b);
    auto torch_result = torch::matmul(torch_a, torch_b);

    EXPECT_NEAR(custom_result.item(), torch_result.item<float>(), 1e-5f);
    compare_tensors(custom_result, torch_result, 1e-5f, 1e-6f, "SingleElement");
}

TEST_F(TensorMatrixTest, LargeMatMul) {
    // Test larger matrices
    const size_t n = 128;

    auto custom_a = Tensor::randn({n, n}, Device::CUDA);
    auto custom_b = Tensor::randn({n, n}, Device::CUDA);

    // Get data for torch tensors
    auto a_vec = custom_a.to_vector();
    auto b_vec = custom_b.to_vector();

    auto torch_a = torch::tensor(a_vec, torch::TensorOptions().device(torch::kCUDA))
                       .reshape({static_cast<int64_t>(n), static_cast<int64_t>(n)});
    auto torch_b = torch::tensor(b_vec, torch::TensorOptions().device(torch::kCUDA))
                       .reshape({static_cast<int64_t>(n), static_cast<int64_t>(n)});

    auto custom_result = custom_a.matmul(custom_b);
    auto torch_result = torch::matmul(torch_a, torch_b);

    // Use slightly higher tolerance for larger matrices
    compare_tensors(custom_result, torch_result, 1e-2f, 1e-3f, "LargeMatMul");
}

TEST_F(TensorMatrixTest, VeryLargeMatMul) {
    // Test even larger matrices for robustness
    const size_t n = 512;

    auto custom_a = Tensor::randn({n, n}, Device::CUDA);
    auto custom_b = Tensor::randn({n, n}, Device::CUDA);

    auto a_vec = custom_a.to_vector();
    auto b_vec = custom_b.to_vector();

    auto torch_a = torch::tensor(a_vec, torch::TensorOptions().device(torch::kCUDA))
                       .reshape({static_cast<int64_t>(n), static_cast<int64_t>(n)});
    auto torch_b = torch::tensor(b_vec, torch::TensorOptions().device(torch::kCUDA))
                       .reshape({static_cast<int64_t>(n), static_cast<int64_t>(n)});

    auto custom_result = custom_a.matmul(custom_b);
    auto torch_result = torch::matmul(torch_a, torch_b);

    // Even higher tolerance for very large matrices due to accumulation errors
    compare_tensors(custom_result, torch_result, 1e-1f, 1e-2f, "VeryLargeMatMul");
}

TEST_F(TensorMatrixTest, RectangularMatMul) {
    // Test non-square matrix multiplication
    const size_t m = 47, n = 53, p = 61;

    auto custom_a = Tensor::randn({m, n}, Device::CUDA);
    auto custom_b = Tensor::randn({n, p}, Device::CUDA);

    auto a_vec = custom_a.to_vector();
    auto b_vec = custom_b.to_vector();

    auto torch_a = torch::tensor(a_vec, torch::TensorOptions().device(torch::kCUDA))
                       .reshape({static_cast<int64_t>(m), static_cast<int64_t>(n)});
    auto torch_b = torch::tensor(b_vec, torch::TensorOptions().device(torch::kCUDA))
                       .reshape({static_cast<int64_t>(n), static_cast<int64_t>(p)});

    auto custom_result = custom_a.matmul(custom_b);
    auto torch_result = torch::matmul(torch_a, torch_b);

    compare_tensors(custom_result, torch_result, 1e-2f, 1e-3f, "RectangularMatMul");
}

TEST_F(TensorMatrixTest, InvalidMatMulDimensionMismatch) {
    // Test dimension mismatch
    auto a = Tensor::ones({2, 3}, Device::CUDA);
    auto b = Tensor::ones({4, 5}, Device::CUDA);

    auto result = a.matmul(b);
    EXPECT_FALSE(result.is_valid()) << "MatMul should fail with mismatched dimensions";
}

TEST_F(TensorMatrixTest, InvalidBatchMatMulDimensionMismatch) {
    // Test batch dimension mismatch
    auto a = Tensor::ones({2, 3, 4}, Device::CUDA);
    auto b = Tensor::ones({3, 4, 5}, Device::CUDA);

    auto result = a.bmm(b);
    EXPECT_FALSE(result.is_valid()) << "BMM should fail with mismatched batch dimensions";
}

TEST_F(TensorMatrixTest, InvalidDotProductDimensionMismatch) {
    // Test dot product with mismatched sizes
    auto a = Tensor::ones({3}, Device::CUDA);
    auto b = Tensor::ones({4}, Device::CUDA);

    auto result = a.dot(b);
    EXPECT_FALSE(result.is_valid()) << "Dot product should fail with mismatched sizes";
}

// ============= CPU vs CUDA Tests =============

TEST_F(TensorMatrixTest, MatMulCPU) {
    // Test matrix multiplication on CPU
    std::vector<float> a_data = {1, 2, 3, 4};
    std::vector<float> b_data = {5, 6, 7, 8};

    auto custom_a = Tensor::from_vector(a_data, {2, 2}, Device::CPU);
    auto custom_b = Tensor::from_vector(b_data, {2, 2}, Device::CPU);

    auto torch_a = torch::tensor(a_data, torch::TensorOptions().device(torch::kCPU))
                       .reshape({2, 2});
    auto torch_b = torch::tensor(b_data, torch::TensorOptions().device(torch::kCPU))
                       .reshape({2, 2});

    auto custom_result = custom_a.matmul(custom_b);
    auto torch_result = torch::matmul(torch_a, torch_b);

    EXPECT_EQ(custom_result.device(), Device::CPU);
    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "MatMulCPU");
}

TEST_F(TensorMatrixTest, TransposeCPU) {
    // Test transpose on CPU
    std::vector<float> data = {1, 2, 3, 4, 5, 6};

    auto custom_tensor = Tensor::from_vector(data, {2, 3}, Device::CPU);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCPU))
                            .reshape({2, 3});

    auto custom_result = custom_tensor.t();
    auto torch_result = torch_tensor.t();

    EXPECT_EQ(custom_result.device(), Device::CPU);
    compare_tensors(custom_result, torch_result, 1e-5f, 1e-6f, "TransposeCPU");
}

// ============= Zero and Negative Value Tests =============

TEST_F(TensorMatrixTest, MatMulWithZeros) {
    // Test matrix multiplication with zero matrix
    auto custom_a = Tensor::randn({3, 4}, Device::CUDA);
    auto custom_b = Tensor::zeros({4, 5}, Device::CUDA);

    auto a_vec = custom_a.to_vector();
    auto torch_a = torch::tensor(a_vec, torch::TensorOptions().device(torch::kCUDA))
                       .reshape({3, 4});
    auto torch_b = torch::zeros({4, 5}, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_a.matmul(custom_b);
    auto torch_result = torch::matmul(torch_a, torch_b);

    // Result should be all zeros
    EXPECT_NEAR(custom_result.sum_scalar(), 0.0f, 1e-5f);
    compare_tensors(custom_result, torch_result, 1e-5f, 1e-6f, "MatMulZeros");
}

TEST_F(TensorMatrixTest, MatMulWithNegatives) {
    // Test matrix multiplication with negative values
    std::vector<float> a_data = {-1, 2, -3, 4};
    std::vector<float> b_data = {5, -6, -7, 8};

    auto custom_a = Tensor::from_vector(a_data, {2, 2}, Device::CUDA);
    auto custom_b = Tensor::from_vector(b_data, {2, 2}, Device::CUDA);

    auto torch_a = torch::tensor(a_data, torch::TensorOptions().device(torch::kCUDA))
                       .reshape({2, 2});
    auto torch_b = torch::tensor(b_data, torch::TensorOptions().device(torch::kCUDA))
                       .reshape({2, 2});

    auto custom_result = custom_a.matmul(custom_b);
    auto torch_result = torch::matmul(torch_a, torch_b);

    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "MatMulNegatives");
}

// ============= Identity Matrix Tests =============

TEST_F(TensorMatrixTest, MatMulWithIdentity) {
    // Test A @ I = A
    auto custom_a = Tensor::randn({5, 5}, Device::CUDA);
    auto custom_eye = Tensor::eye(5, Device::CUDA);

    auto a_vec = custom_a.to_vector();
    auto torch_a = torch::tensor(a_vec, torch::TensorOptions().device(torch::kCUDA))
                       .reshape({5, 5});
    auto torch_eye = torch::eye(5, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_a.matmul(custom_eye);
    auto torch_result = torch::matmul(torch_a, torch_eye);

    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "MatMulIdentity");

    // Result should be approximately equal to original
    EXPECT_TRUE(custom_result.all_close(custom_a, 1e-4f, 1e-5f));
}

TEST_F(TensorMatrixTest, DiagMatMul) {
    // Test diagonal matrix multiplication
    std::vector<float> diag_data = {1, 2, 3, 4};
    std::vector<float> mat_data(4 * 5);
    std::iota(mat_data.begin(), mat_data.end(), 1.0f);

    auto custom_diag_vec = Tensor::from_vector(diag_data, {4}, Device::CUDA);
    auto custom_diag = Tensor::diag(custom_diag_vec);
    auto custom_mat = Tensor::from_vector(mat_data, {4, 5}, Device::CUDA);

    auto torch_diag_vec = torch::tensor(diag_data, torch::TensorOptions().device(torch::kCUDA));
    auto torch_diag = torch::diag(torch_diag_vec);
    auto torch_mat = torch::tensor(mat_data, torch::TensorOptions().device(torch::kCUDA))
                         .reshape({4, 5});

    auto custom_result = custom_diag.matmul(custom_mat);
    auto torch_result = torch::matmul(torch_diag, torch_mat);

    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "DiagMatMul");
}

// ============= Performance Benchmark (Optional) =============

TEST_F(TensorMatrixTest, MatMulPerformance) {
    // Compare performance with PyTorch
    const size_t n = 512;
    const int iterations = 10;

    auto custom_a = Tensor::randn({n, n}, Device::CUDA);
    auto custom_b = Tensor::randn({n, n}, Device::CUDA);

    auto a_vec = custom_a.to_vector();
    auto b_vec = custom_b.to_vector();

    auto torch_a = torch::tensor(a_vec, torch::TensorOptions().device(torch::kCUDA))
                       .reshape({static_cast<int64_t>(n), static_cast<int64_t>(n)});
    auto torch_b = torch::tensor(b_vec, torch::TensorOptions().device(torch::kCUDA))
                       .reshape({static_cast<int64_t>(n), static_cast<int64_t>(n)});

    // Warm up
    auto warm_custom = custom_a.matmul(custom_b);
    auto warm_torch = torch::matmul(torch_a, torch_b);
    cudaDeviceSynchronize();

    // Benchmark custom implementation
    auto start_custom = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto result = custom_a.matmul(custom_b);
        cudaDeviceSynchronize();
    }
    auto end_custom = std::chrono::high_resolution_clock::now();
    auto custom_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                           end_custom - start_custom)
                           .count();

    // Benchmark PyTorch
    auto start_torch = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto result = torch::matmul(torch_a, torch_b);
        cudaDeviceSynchronize();
    }
    auto end_torch = std::chrono::high_resolution_clock::now();
    auto torch_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                          end_torch - start_torch)
                          .count();

    LOG_INFO("MatMul {}x{} Performance:", n, n);
    LOG_INFO("  Custom: {:.2f} ms per operation", custom_time / double(iterations));
    LOG_INFO("  PyTorch: {:.2f} ms per operation", torch_time / double(iterations));
    LOG_INFO("  Ratio: {:.2f}x", custom_time / double(torch_time));

    // We don't assert on performance, just log it for information
}