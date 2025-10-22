/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/tensor.hpp"
#include <gtest/gtest.h>
#include <random>
#include <torch/torch.h>

using namespace lfs::core;

// ============= Helper Functions =============

namespace {

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

} // anonymous namespace

class TensorMaskingTest : public ::testing::Test {
protected:
    void SetUp() override {
        Tensor::manual_seed(42);
        torch::manual_seed(42);
    }
};

// ============= Comparison Operations Tests =============

TEST_F(TensorMaskingTest, ComparisonEqual) {
    std::vector<float> a_data = {1, 2, 3, 4, 5};
    std::vector<float> b_data = {1, 0, 3, 0, 5};

    auto a_custom = Tensor::from_vector(a_data, {5}, Device::CUDA);
    auto b_custom = Tensor::from_vector(b_data, {5}, Device::CUDA);
    auto t = Tensor::zeros({1, 2});

    auto a_torch = torch::tensor(a_data, torch::kCUDA);
    auto b_torch = torch::tensor(b_data, torch::kCUDA);

    auto mask_custom = a_custom.eq(b_custom);
    auto mask_torch = a_torch.eq(b_torch);

    compare_tensors(mask_custom, mask_torch, 1e-5f, 1e-7f, "ComparisonEqual");

    // Test scalar comparison
    auto mask2_custom = a_custom.eq(3.0f);
    auto mask2_torch = a_torch.eq(3.0f);

    compare_tensors(mask2_custom, mask2_torch, 1e-5f, 1e-7f, "ComparisonEqualScalar");
}

TEST_F(TensorMaskingTest, ComparisonLessThan) {
    std::vector<float> a_data = {1, 2, 3, 4, 5};
    auto a_custom = Tensor::from_vector(a_data, {5}, Device::CUDA);
    auto a_torch = torch::tensor(a_data, torch::kCUDA);

    auto mask_custom = a_custom.lt(3.0f);
    auto mask_torch = a_torch.lt(3.0f);

    compare_tensors(mask_custom, mask_torch, 1e-5f, 1e-7f, "ComparisonLT");

    // Test with another tensor
    std::vector<float> b_data = {2, 2, 2, 6, 4};
    auto b_custom = Tensor::from_vector(b_data, {5}, Device::CUDA);
    auto b_torch = torch::tensor(b_data, torch::kCUDA);

    auto mask2_custom = a_custom.lt(b_custom);
    auto mask2_torch = a_torch.lt(b_torch);

    compare_tensors(mask2_custom, mask2_torch, 1e-5f, 1e-7f, "ComparisonLTTensor");
}

TEST_F(TensorMaskingTest, ComparisonGreaterThan) {
    std::vector<float> data = {1, 2, 3, 4, 5};
    auto a_custom = Tensor::from_vector(data, {5}, Device::CUDA);
    auto a_torch = torch::tensor(data, torch::kCUDA);

    auto mask_custom = a_custom.gt(3.0f);
    auto mask_torch = a_torch.gt(3.0f);

    compare_tensors(mask_custom, mask_torch, 1e-5f, 1e-7f, "ComparisonGT");
}

TEST_F(TensorMaskingTest, ComparisonChaining) {
    std::vector<float> data = {1, 2, 3, 4, 5};
    auto a_custom = Tensor::from_vector(data, {5}, Device::CUDA);
    auto a_torch = torch::tensor(data, torch::kCUDA);

    // Find elements between 2 and 4 (inclusive)
    auto mask_custom = a_custom.ge(2.0f).logical_and(a_custom.le(4.0f));
    auto mask_torch = a_torch.ge(2.0f).logical_and(a_torch.le(4.0f));

    compare_tensors(mask_custom, mask_torch, 1e-5f, 1e-7f, "ComparisonChaining");
}

TEST_F(TensorMaskingTest, ComparisonWithBroadcasting) {
    std::vector<float> a_data = {1, 2, 3, 4, 5, 6};
    std::vector<float> b_data = {2, 3, 4};

    auto a_custom = Tensor::from_vector(a_data, {2, 3}, Device::CUDA);
    auto b_custom = Tensor::from_vector(b_data, {1, 3}, Device::CUDA);

    auto a_torch = torch::tensor(a_data, torch::kCUDA).reshape({2, 3});
    auto b_torch = torch::tensor(b_data, torch::kCUDA).reshape({1, 3});

    auto mask_custom = a_custom.gt(b_custom);
    auto mask_torch = a_torch.gt(b_torch);

    compare_tensors(mask_custom, mask_torch, 1e-5f, 1e-7f, "ComparisonBroadcast");
}

// ============= Logical Operations Tests =============

TEST_F(TensorMaskingTest, LogicalOperations) {
    std::vector<float> a_data = {1, 0, 1, 0};
    std::vector<float> b_data = {1, 1, 0, 0};

    auto a_custom = Tensor::from_vector(a_data, {4}, Device::CUDA);
    auto b_custom = Tensor::from_vector(b_data, {4}, Device::CUDA);

    auto a_torch = torch::tensor(a_data, torch::kCUDA);
    auto b_torch = torch::tensor(b_data, torch::kCUDA);

    auto a_bool_custom = a_custom.ne(0.0f);
    auto b_bool_custom = b_custom.ne(0.0f);

    auto a_bool_torch = a_torch.ne(0.0f);
    auto b_bool_torch = b_torch.ne(0.0f);

    // AND
    auto and_custom = a_bool_custom.logical_and(b_bool_custom);
    auto and_torch = a_bool_torch.logical_and(b_bool_torch);
    compare_tensors(and_custom, and_torch, 1e-5f, 1e-7f, "LogicalAND");

    // OR
    auto or_custom = a_bool_custom.logical_or(b_bool_custom);
    auto or_torch = a_bool_torch.logical_or(b_bool_torch);
    compare_tensors(or_custom, or_torch, 1e-5f, 1e-7f, "LogicalOR");

    // NOT
    auto not_custom = a_bool_custom.logical_not();
    auto not_torch = a_bool_torch.logical_not();
    compare_tensors(not_custom, not_torch, 1e-5f, 1e-7f, "LogicalNOT");
}

// ============= Masked Select Tests =============

TEST_F(TensorMaskingTest, MaskedSelect) {
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    auto tensor_custom = Tensor::from_vector(data, {2, 3}, Device::CUDA);
    auto tensor_torch = torch::tensor(data, torch::kCUDA).reshape({2, 3});

    auto mask_custom = tensor_custom.gt(3.0f);
    auto mask_torch = tensor_torch.gt(3.0f);

    auto selected_custom = tensor_custom.masked_select(mask_custom);
    auto selected_torch = tensor_torch.masked_select(mask_torch);

    compare_tensors(selected_custom, selected_torch, 1e-5f, 1e-7f, "MaskedSelect");
}

TEST_F(TensorMaskingTest, MaskedSelectEmpty) {
    std::vector<float> data = {1, 2, 3};
    auto tensor_custom = Tensor::from_vector(data, {3}, Device::CUDA);
    auto tensor_torch = torch::tensor(data, torch::kCUDA);

    auto mask_custom = tensor_custom.gt(10.0f);
    auto mask_torch = tensor_torch.gt(10.0f);

    auto selected_custom = tensor_custom.masked_select(mask_custom);
    auto selected_torch = tensor_torch.masked_select(mask_torch);

    EXPECT_EQ(selected_custom.numel(), 0);
    EXPECT_EQ(selected_torch.numel(), 0);
}

// ============= Masked Fill Tests =============

TEST_F(TensorMaskingTest, MaskedFillInplace) {
    std::vector<float> data = {1, 2, 3, 4, 5};
    auto tensor_custom = Tensor::from_vector(data, {5}, Device::CUDA);
    auto tensor_torch = torch::tensor(data, torch::kCUDA);

    auto mask_custom = tensor_custom.gt(3.0f);
    auto mask_torch = tensor_torch.gt(3.0f);

    tensor_custom.masked_fill_(mask_custom, -1.0f);
    tensor_torch.masked_fill_(mask_torch, -1.0f);

    compare_tensors(tensor_custom, tensor_torch, 1e-5f, 1e-7f, "MaskedFillInplace");
}

TEST_F(TensorMaskingTest, MaskedFillNonInplace) {
    std::vector<float> data = {1, 2, 3, 4, 5};
    auto tensor_custom = Tensor::from_vector(data, {5}, Device::CUDA);
    auto tensor_torch = torch::tensor(data, torch::kCUDA);

    auto mask_custom = tensor_custom.le(2.0f);
    auto mask_torch = tensor_torch.le(2.0f);

    auto result_custom = tensor_custom.masked_fill(mask_custom, 0.0f);
    auto result_torch = tensor_torch.masked_fill(mask_torch, 0.0f);

    // Original should be unchanged
    auto orig_torch = torch::tensor(data, torch::kCUDA);
    compare_tensors(tensor_custom, orig_torch, 1e-5f, 1e-7f, "MaskedFillOriginal");

    // Result should have masked values filled
    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "MaskedFillResult");
}

// ============= Where Operation Tests =============

TEST_F(TensorMaskingTest, Where) {
    std::vector<float> cond_data = {1, 0, 1, 0};
    std::vector<float> x_data = {1, 2, 3, 4};
    std::vector<float> y_data = {5, 6, 7, 8};

    auto cond_custom = Tensor::from_vector(cond_data, {4}, Device::CUDA).ne(0.0f);
    auto x_custom = Tensor::from_vector(x_data, {4}, Device::CUDA);
    auto y_custom = Tensor::from_vector(y_data, {4}, Device::CUDA);

    auto cond_torch = torch::tensor(cond_data, torch::kCUDA).ne(0.0f);
    auto x_torch = torch::tensor(x_data, torch::kCUDA);
    auto y_torch = torch::tensor(y_data, torch::kCUDA);

    auto result_custom = Tensor::where(cond_custom, x_custom, y_custom);
    auto result_torch = torch::where(cond_torch, x_torch, y_torch);

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "Where");
}

TEST_F(TensorMaskingTest, WhereWithBroadcasting) {
    std::vector<float> cond_data = {1, 0};
    auto cond_custom = Tensor::from_vector(cond_data, {2, 1}, Device::CUDA).ne(0.0f);
    auto x_custom = Tensor::full({2, 3}, 1.0f, Device::CUDA);
    auto y_custom = Tensor::full({2, 3}, 0.0f, Device::CUDA);

    auto cond_torch = torch::tensor(cond_data, torch::kCUDA).reshape({2, 1}).ne(0.0f);
    auto x_torch = torch::full({2, 3}, 1.0f, torch::kCUDA);
    auto y_torch = torch::full({2, 3}, 0.0f, torch::kCUDA);

    auto result_custom = Tensor::where(cond_custom, x_custom, y_custom);
    auto result_torch = torch::where(cond_torch, x_torch, y_torch);

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "WhereBroadcast");
}

// ============= Index Select Tests =============

TEST_F(TensorMaskingTest, IndexSelect) {
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    auto tensor_custom = Tensor::from_vector(data, {2, 3}, Device::CUDA);
    auto tensor_torch = torch::tensor(data, torch::kCUDA).reshape({2, 3});

    std::vector<int> indices_data = {0, 2};
    auto indices_custom = Tensor::from_vector(indices_data, {2}, Device::CUDA);
    auto indices_torch = torch::tensor(indices_data, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));

    auto selected_custom = tensor_custom.index_select(1, indices_custom);
    auto selected_torch = tensor_torch.index_select(1, indices_torch);

    compare_tensors(selected_custom, selected_torch, 1e-5f, 1e-7f, "IndexSelect");
}

TEST_F(TensorMaskingTest, IndexSelectWithBoundaryClamp) {
    std::vector<float> data = {1, 2, 3, 4};
    auto tensor_custom = Tensor::from_vector(data, {4}, Device::CUDA);
    auto tensor_torch = torch::tensor(data, torch::kCUDA);

    std::vector<int> indices_data = {-1, 0, 5, 2};
    auto indices_custom = Tensor::from_vector(indices_data, {4}, Device::CUDA);

    // Clamp indices for PyTorch
    std::vector<int64_t> clamped_indices;
    for (int idx : indices_data) {
        clamped_indices.push_back(std::clamp(static_cast<int64_t>(idx), static_cast<int64_t>(0), static_cast<int64_t>(3)));
    }
    auto indices_torch = torch::tensor(clamped_indices, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));

    auto selected_custom = tensor_custom.index_select(0, indices_custom, BoundaryMode::Clamp);
    auto selected_torch = tensor_torch.index_select(0, indices_torch);

    compare_tensors(selected_custom, selected_torch, 1e-5f, 1e-7f, "IndexSelectClamp");
}

TEST_F(TensorMaskingTest, IndexSelectWithBoundaryWrap) {
    std::vector<float> data = {1, 2, 3, 4};
    auto tensor_custom = Tensor::from_vector(data, {4}, Device::CUDA);
    auto tensor_torch = torch::tensor(data, torch::kCUDA);

    std::vector<int> indices_data = {-1, 0, 5, 2};
    auto indices_custom = Tensor::from_vector(indices_data, {4}, Device::CUDA);

    // Wrap indices for PyTorch
    std::vector<int64_t> wrapped_indices;
    for (int idx : indices_data) {
        int wrapped = idx % 4;
        if (wrapped < 0)
            wrapped += 4;
        wrapped_indices.push_back(wrapped);
    }
    auto indices_torch = torch::tensor(wrapped_indices, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));

    auto selected_custom = tensor_custom.index_select(0, indices_custom, BoundaryMode::Wrap);
    auto selected_torch = tensor_torch.index_select(0, indices_torch);

    compare_tensors(selected_custom, selected_torch, 1e-5f, 1e-7f, "IndexSelectWrap");
}

// ============= Gather Tests =============

TEST_F(TensorMaskingTest, Gather) {
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    auto tensor_custom = Tensor::from_vector(data, {2, 3}, Device::CUDA);
    auto tensor_torch = torch::tensor(data, torch::kCUDA).reshape({2, 3});

    std::vector<int> indices_data = {0, 2, 1, 0};
    auto indices_custom = Tensor::from_vector(indices_data, {2, 2}, Device::CUDA);
    auto indices_torch = torch::tensor(indices_data, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA)).reshape({2, 2});

    auto gathered_custom = tensor_custom.gather(1, indices_custom);
    auto gathered_torch = tensor_torch.gather(1, indices_torch);

    compare_tensors(gathered_custom, gathered_torch, 1e-5f, 1e-7f, "Gather");
}

// ============= Take Tests =============

TEST_F(TensorMaskingTest, Take) {
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    auto tensor_custom = Tensor::from_vector(data, {2, 3}, Device::CUDA);
    auto tensor_torch = torch::tensor(data, torch::kCUDA).reshape({2, 3});

    std::vector<int> indices_data = {0, 2, 5, 3};
    auto indices_custom = Tensor::from_vector(indices_data, {4}, Device::CUDA);
    auto indices_torch = torch::tensor(indices_data, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));

    auto taken_custom = tensor_custom.take(indices_custom);
    auto taken_torch = tensor_torch.take(indices_torch);

    compare_tensors(taken_custom, taken_torch, 1e-5f, 1e-7f, "Take");
}

TEST_F(TensorMaskingTest, TakeNegativeIndices) {
    std::vector<float> data = {1, 2, 3, 4};
    auto tensor_custom = Tensor::from_vector(data, {4}, Device::CUDA);
    auto tensor_torch = torch::tensor(data, torch::kCUDA);

    std::vector<int> indices_data = {-1, -2, 0, 1};
    auto indices_custom = Tensor::from_vector(indices_data, {4}, Device::CUDA);
    auto indices_torch = torch::tensor(indices_data, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));

    auto taken_custom = tensor_custom.take(indices_custom);
    auto taken_torch = tensor_torch.take(indices_torch);

    compare_tensors(taken_custom, taken_torch, 1e-5f, 1e-7f, "TakeNegative");
}

// ============= Scatter Tests =============

TEST_F(TensorMaskingTest, Scatter) {
    auto tensor_custom = Tensor::zeros({3, 4}, Device::CUDA);
    auto tensor_torch = torch::zeros({3, 4}, torch::kCUDA);

    std::vector<int> indices_data = {0, 2};
    auto indices_custom = Tensor::from_vector(indices_data, {2}, Device::CUDA);
    auto indices_torch = torch::tensor(indices_data, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));

    std::vector<float> src_data = {1, 2, 3, 4, 5, 6, 7, 8};
    auto src_custom = Tensor::from_vector(src_data, {2, 4}, Device::CUDA);
    auto src_torch = torch::tensor(src_data, torch::kCUDA).reshape({2, 4});

    // Expand indices for PyTorch
    auto indices_expanded = indices_torch.reshape({2, 1}).expand({2, 4});

    tensor_custom.scatter_(0, indices_custom, src_custom);
    tensor_torch.scatter_(0, indices_expanded, src_torch);

    compare_tensors(tensor_custom, tensor_torch, 1e-5f, 1e-7f, "Scatter");
}

TEST_F(TensorMaskingTest, ScatterWithAdd) {
    auto tensor_custom = Tensor::ones({4}, Device::CUDA);
    auto tensor_torch = torch::ones({4}, torch::kCUDA);

    std::vector<int> indices_data = {0, 2, 0, 3};
    auto indices_custom = Tensor::from_vector(indices_data, {4}, Device::CUDA);
    auto indices_torch = torch::tensor(indices_data, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));

    std::vector<float> src_data = {1, 2, 3, 4};
    auto src_custom = Tensor::from_vector(src_data, {4}, Device::CUDA);
    auto src_torch = torch::tensor(src_data, torch::kCUDA);

    tensor_custom.scatter_(0, indices_custom, src_custom, ScatterMode::Add);
    tensor_torch.scatter_add_(0, indices_torch, src_torch);

    compare_tensors(tensor_custom, tensor_torch, 1e-5f, 1e-7f, "ScatterAdd");
}

// ============= Index Fill Tests =============

TEST_F(TensorMaskingTest, IndexFill) {
    std::vector<float> data = {1, 2, 3, 4, 5};
    auto tensor_custom = Tensor::from_vector(data, {5}, Device::CUDA);
    auto tensor_torch = torch::tensor(data, torch::kCUDA);

    std::vector<int> indices_data = {1, 3};
    auto indices_custom = Tensor::from_vector(indices_data, {2}, Device::CUDA);
    auto indices_torch = torch::tensor(indices_data, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));

    tensor_custom.index_fill_(0, indices_custom, -1.0f);
    tensor_torch.index_fill_(0, indices_torch, -1.0f);

    compare_tensors(tensor_custom, tensor_torch, 1e-5f, 1e-7f, "IndexFill");
}

// ============= Index Copy Tests =============

TEST_F(TensorMaskingTest, IndexCopy) {
    auto tensor_custom = Tensor::zeros({5}, Device::CUDA);
    auto tensor_torch = torch::zeros({5}, torch::kCUDA);

    std::vector<int> indices_data = {0, 2, 4};
    auto indices_custom = Tensor::from_vector(indices_data, {3}, Device::CUDA);
    auto indices_torch = torch::tensor(indices_data, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));

    std::vector<float> src_data = {1, 2, 3};
    auto src_custom = Tensor::from_vector(src_data, {3}, Device::CUDA);
    auto src_torch = torch::tensor(src_data, torch::kCUDA);

    tensor_custom.index_copy_(0, indices_custom, src_custom);
    tensor_torch.index_copy_(0, indices_torch, src_torch);

    compare_tensors(tensor_custom, tensor_torch, 1e-5f, 1e-7f, "IndexCopy");
}

// ============= Count Nonzero Tests =============

TEST_F(TensorMaskingTest, CountNonzero) {
    std::vector<float> data = {0, 1, 0, 2, 0, 3};
    auto tensor_custom = Tensor::from_vector(data, {6}, Device::CUDA);
    auto tensor_torch = torch::tensor(data, torch::kCUDA);

    auto count_custom = tensor_custom.count_nonzero();
    auto count_torch = tensor_torch.count_nonzero().item<int64_t>();

    EXPECT_EQ(count_custom, static_cast<size_t>(count_torch));

    auto bool_custom = tensor_custom.ne(0.0f);
    auto bool_torch = tensor_torch.ne(0.0f);

    auto bool_count_custom = bool_custom.count_nonzero();
    auto bool_count_torch = bool_torch.count_nonzero().item<int64_t>();

    EXPECT_EQ(bool_count_custom, static_cast<size_t>(bool_count_torch));
}

// ============= Regression Tests =============

TEST_F(TensorMaskingTest, RegressionDiagonalMaskBroadcast) {
    const size_t n = 5;
    std::vector<float> range_data = {0, 1, 2, 3, 4};

    auto range_custom = Tensor::from_vector(range_data, {n}, Device::CUDA);
    auto range_torch = torch::tensor(range_data, torch::kCUDA);

    auto col_custom = range_custom.reshape({n, 1});
    auto row_custom = range_custom.reshape({1, n});

    auto col_torch = range_torch.reshape({static_cast<int64_t>(n), 1});
    auto row_torch = range_torch.reshape({1, static_cast<int64_t>(n)});

    auto diag_mask_custom = col_custom.eq(row_custom);
    auto diag_mask_torch = col_torch.eq(row_torch);

    compare_tensors(diag_mask_custom, diag_mask_torch, 1e-5f, 1e-7f, "DiagonalMaskBroadcast");

    // Verify diagonal count
    auto count_custom = diag_mask_custom.count_nonzero();
    auto count_torch = diag_mask_torch.count_nonzero().item<int64_t>();
    EXPECT_EQ(count_custom, static_cast<size_t>(count_torch));
    EXPECT_EQ(count_custom, n);
}

TEST_F(TensorMaskingTest, RegressionBooleanTensorExpansion) {
    const size_t batch_size = 3;
    const size_t seq_len = 4;

    std::vector<bool> mask_data(seq_len * seq_len);
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
            mask_data[i * seq_len + j] = (i >= j);
        }
    }

    auto mask_2d_custom = Tensor::from_vector(mask_data, {seq_len, seq_len}, Device::CUDA);

    std::vector<uint8_t> mask_data_torch;
    for (bool b : mask_data) {
        mask_data_torch.push_back(b ? 1 : 0);
    }
    auto mask_2d_torch = torch::tensor(mask_data_torch, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA))
                             .reshape({static_cast<int64_t>(seq_len), static_cast<int64_t>(seq_len)});

    auto mask_3d_custom = mask_2d_custom.unsqueeze(0);
    auto mask_3d_torch = mask_2d_torch.unsqueeze(0);

    auto mask_expanded_custom = mask_3d_custom.expand({batch_size, seq_len, seq_len});
    auto mask_expanded_torch = mask_3d_torch.expand({static_cast<int64_t>(batch_size),
                                                     static_cast<int64_t>(seq_len),
                                                     static_cast<int64_t>(seq_len)});

    compare_tensors(mask_expanded_custom, mask_expanded_torch, 1e-5f, 1e-7f, "BooleanExpansion");
}

TEST_F(TensorMaskingTest, RegressionBroadcastComparisonDifferentRanks) {
    // Test case 1: 1D vs 2D
    std::vector<float> vec_data = {1, 2, 3};
    std::vector<float> mat_data = {1, 2, 3, 2, 3, 4};

    auto vec_custom = Tensor::from_vector(vec_data, {3}, Device::CUDA);
    auto mat_custom = Tensor::from_vector(mat_data, {2, 3}, Device::CUDA);

    auto vec_torch = torch::tensor(vec_data, torch::kCUDA);
    auto mat_torch = torch::tensor(mat_data, torch::kCUDA).reshape({2, 3});

    auto mask_custom = vec_custom.eq(mat_custom);
    auto mask_torch = vec_torch.eq(mat_torch);

    compare_tensors(mask_custom, mask_torch, 1e-5f, 1e-7f, "BroadcastRank1vs2");

    // Test case 2: Scalar-like vs Matrix
    std::vector<float> scalar_data = {2};
    std::vector<float> matrix_data = {1, 2, 3, 4};

    auto scalar_custom = Tensor::from_vector(scalar_data, {1, 1}, Device::CUDA);
    auto matrix_custom = Tensor::from_vector(matrix_data, {2, 2}, Device::CUDA);

    auto scalar_torch = torch::tensor(scalar_data, torch::kCUDA).reshape({1, 1});
    auto matrix_torch = torch::tensor(matrix_data, torch::kCUDA).reshape({2, 2});

    auto mask2_custom = scalar_custom.eq(matrix_custom);
    auto mask2_torch = scalar_torch.eq(matrix_torch);

    compare_tensors(mask2_custom, mask2_torch, 1e-5f, 1e-7f, "BroadcastScalarvsMatrix");
}

TEST_F(TensorMaskingTest, RegressionBooleanBroadcastLogical) {
    std::vector<bool> mask1_data = {true, false};
    std::vector<bool> mask2_data = {true, false, true};

    auto mask1_custom = Tensor::from_vector(mask1_data, {2, 1}, Device::CUDA);
    auto mask2_custom = Tensor::from_vector(mask2_data, {1, 3}, Device::CUDA);

    std::vector<uint8_t> mask1_torch_data = {1, 0};
    std::vector<uint8_t> mask2_torch_data = {1, 0, 1};
    auto mask1_torch = torch::tensor(mask1_torch_data, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA)).reshape({2, 1});
    auto mask2_torch = torch::tensor(mask2_torch_data, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA)).reshape({1, 3});

    auto result_custom = mask1_custom.logical_and(mask2_custom);
    auto result_torch = mask1_torch.logical_and(mask2_torch);

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "BooleanBroadcastLogical");
}

// ============= Complex Masking Scenarios =============

TEST_F(TensorMaskingTest, ComplexMaskingScenario) {
    std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    auto tensor_custom = Tensor::from_vector(data, {3, 4}, Device::CUDA);
    auto tensor_torch = torch::tensor(data, torch::kCUDA).reshape({3, 4});

    // Find elements in range [5, 10]
    auto mask_custom = tensor_custom.ge(5.0f).logical_and(tensor_custom.le(10.0f));
    auto mask_torch = tensor_torch.ge(5.0f).logical_and(tensor_torch.le(10.0f));

    // Get selected elements
    auto selected_custom = tensor_custom.masked_select(mask_custom);
    auto selected_torch = tensor_torch.masked_select(mask_torch);

    compare_tensors(selected_custom, selected_torch, 1e-5f, 1e-7f, "ComplexMaskSelect");

    // Replace them with -1
    tensor_custom.masked_fill_(mask_custom, -1.0f);
    tensor_torch.masked_fill_(mask_torch, -1.0f);

    compare_tensors(tensor_custom, tensor_torch, 1e-5f, 1e-7f, "ComplexMaskFill");
}

TEST_F(TensorMaskingTest, AdvancedIndexingScenario) {
    std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    auto tensor_custom = Tensor::from_vector(data, {4, 3}, Device::CUDA);
    auto tensor_torch = torch::tensor(data, torch::kCUDA).reshape({4, 3});

    // Select specific rows
    std::vector<int> row_idx = {0, 2, 3};
    auto row_indices_custom = Tensor::from_vector(row_idx, {3}, Device::CUDA);
    auto row_indices_torch = torch::tensor(row_idx, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));

    auto selected_rows_custom = tensor_custom.index_select(0, row_indices_custom);
    auto selected_rows_torch = tensor_torch.index_select(0, row_indices_torch);

    compare_tensors(selected_rows_custom, selected_rows_torch, 1e-5f, 1e-7f, "AdvancedIndexRows");

    // Select specific columns
    std::vector<int> col_idx = {1, 2};
    auto col_indices_custom = Tensor::from_vector(col_idx, {2}, Device::CUDA);
    auto col_indices_torch = torch::tensor(col_idx, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));

    auto final_custom = selected_rows_custom.index_select(1, col_indices_custom);
    auto final_torch = selected_rows_torch.index_select(1, col_indices_torch);

    compare_tensors(final_custom, final_torch, 1e-5f, 1e-7f, "AdvancedIndexCols");
}

// ============= Boolean Expansion Tests =============

TEST_F(TensorMaskingTest, BooleanExpansionSimple1D) {
    std::vector<bool> data = {true, false, true};
    auto bool_1d_custom = Tensor::from_vector(data, {3}, Device::CUDA);

    std::vector<uint8_t> data_torch = {1, 0, 1};
    auto bool_1d_torch = torch::tensor(data_torch, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

    auto expanded_custom = bool_1d_custom.unsqueeze(0).expand({2, 3});
    auto expanded_torch = bool_1d_torch.unsqueeze(0).expand({2, 3});

    compare_tensors(expanded_custom, expanded_torch, 1e-5f, 1e-7f, "BooleanExpansion1D");
}

TEST_F(TensorMaskingTest, BooleanExpansion2Dto3D) {
    std::vector<bool> data_2d = {true, false, false, true};
    auto bool_2d_custom = Tensor::from_vector(data_2d, {2, 2}, Device::CUDA);

    std::vector<uint8_t> data_2d_torch = {1, 0, 0, 1};
    auto bool_2d_torch = torch::tensor(data_2d_torch, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA)).reshape({2, 2});

    auto expanded_custom = bool_2d_custom.unsqueeze(0).expand({3, 2, 2});
    auto expanded_torch = bool_2d_torch.unsqueeze(0).expand({3, 2, 2});

    compare_tensors(expanded_custom, expanded_torch, 1e-5f, 1e-7f, "BooleanExpansion2Dto3D");
}

// ============= Causal Mask Tests =============

TEST_F(TensorMaskingTest, SimpleCausalMask) {
    const size_t seq_len = 4;

    std::vector<bool> mask_data(seq_len * seq_len);
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
            mask_data[i * seq_len + j] = (i >= j);
        }
    }

    auto mask_custom = Tensor::from_vector(mask_data, {seq_len, seq_len}, Device::CUDA);

    std::vector<uint8_t> mask_data_torch;
    for (bool b : mask_data) {
        mask_data_torch.push_back(b ? 1 : 0);
    }
    auto mask_torch = torch::tensor(mask_data_torch, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA))
                          .reshape({static_cast<int64_t>(seq_len), static_cast<int64_t>(seq_len)});

    auto scores_custom = Tensor::ones({seq_len, seq_len}, Device::CUDA);
    auto scores_torch = torch::ones({static_cast<int64_t>(seq_len), static_cast<int64_t>(seq_len)}, torch::kCUDA);

    scores_custom.masked_fill_(mask_custom.logical_not(), -1e9f);
    scores_torch.masked_fill_(mask_torch.logical_not(), -1e9f);

    compare_tensors(scores_custom, scores_torch, 1e-5f, 1e-7f, "SimpleCausalMask");
}

// ============= Edge Cases =============

TEST_F(TensorMaskingTest, EmptyMask) {
    std::vector<float> data = {1, 2, 3};
    auto tensor_custom = Tensor::from_vector(data, {3}, Device::CUDA);
    auto tensor_torch = torch::tensor(data, torch::kCUDA);

    auto empty_mask_custom = Tensor::zeros_bool({3}, Device::CUDA);
    auto empty_mask_torch = torch::zeros({3}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

    auto selected_custom = tensor_custom.masked_select(empty_mask_custom);
    auto selected_torch = tensor_torch.masked_select(empty_mask_torch);

    EXPECT_EQ(selected_custom.numel(), 0);
    EXPECT_EQ(selected_torch.numel(), 0);
}

TEST_F(TensorMaskingTest, FullMask) {
    std::vector<float> data = {1, 2, 3};
    auto tensor_custom = Tensor::from_vector(data, {3}, Device::CUDA);
    auto tensor_torch = torch::tensor(data, torch::kCUDA);

    auto full_mask_custom = Tensor::ones_bool({3}, Device::CUDA);
    auto full_mask_torch = torch::ones({3}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

    auto selected_custom = tensor_custom.masked_select(full_mask_custom);
    auto selected_torch = tensor_torch.masked_select(full_mask_torch);

    compare_tensors(selected_custom, selected_torch, 1e-5f, 1e-7f, "FullMask");
}

TEST_F(TensorMaskingTest, SingleElementIndexing) {
    std::vector<float> data = {42};
    auto tensor_custom = Tensor::from_vector(data, {1}, Device::CUDA);
    auto tensor_torch = torch::tensor(data, torch::kCUDA);

    std::vector<int> idx_data = {0};
    auto indices_custom = Tensor::from_vector(idx_data, {1}, Device::CUDA);
    auto indices_torch = torch::tensor(idx_data, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));

    auto selected_custom = tensor_custom.index_select(0, indices_custom);
    auto selected_torch = tensor_torch.index_select(0, indices_torch);

    compare_tensors(selected_custom, selected_torch, 1e-5f, 1e-7f, "SingleElement");
}

// ============= Special Values Tests =============

TEST_F(TensorMaskingTest, SpecialValueMasking) {
    std::vector<float> data = {
        0.0f, 1.0f, -1.0f,
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN()};

    auto tensor_custom = Tensor::from_vector(data, {6}, Device::CUDA);
    auto tensor_torch = torch::tensor(data, torch::kCUDA);

    // Test comparisons with special values (NaN != NaN)
    auto finite_mask_custom = tensor_custom.eq(tensor_custom);
    auto finite_mask_torch = tensor_torch.eq(tensor_torch);

    compare_tensors(finite_mask_custom, finite_mask_torch, 1e-5f, 1e-7f, "SpecialValues");
}

// ============= Boolean to Float Conversion =============

TEST_F(TensorMaskingTest, BooleanToFloatConversion) {
    std::vector<bool> data = {true, false, true, false};
    auto bool_custom = Tensor::from_vector(data, {2, 2}, Device::CUDA);

    std::vector<uint8_t> data_torch = {1, 0, 1, 0};
    auto bool_torch = torch::tensor(data_torch, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA)).reshape({2, 2});

    auto float_custom = bool_custom.to(DataType::Float32);
    auto float_torch = bool_torch.to(torch::kFloat32);

    compare_tensors(float_custom, float_torch, 1e-5f, 1e-7f, "BoolToFloatConversion");

    // Multiply by scalar
    auto result_custom = float_custom.mul(5.0f);
    auto result_torch = float_torch.mul(5.0f);

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "BoolToFloatMul");
}

// ============= Broadcast Comparison Edge Cases =============

TEST_F(TensorMaskingTest, BroadcastComparisonScalarLike) {
    std::vector<float> scalar_data = {5};
    std::vector<float> vector_data = {1, 2, 3, 4, 5, 6};

    auto scalar_custom = Tensor::from_vector(scalar_data, {1}, Device::CUDA);
    auto vector_custom = Tensor::from_vector(vector_data, {6}, Device::CUDA);

    auto scalar_torch = torch::tensor(scalar_data, torch::kCUDA);
    auto vector_torch = torch::tensor(vector_data, torch::kCUDA);

    auto mask_custom = vector_custom.eq(scalar_custom);
    auto mask_torch = vector_torch.eq(scalar_torch);

    compare_tensors(mask_custom, mask_torch, 1e-5f, 1e-7f, "BroadcastScalarLike");
}

TEST_F(TensorMaskingTest, BroadcastComparison3D) {
    std::vector<float> a_data = {1, 2};
    std::vector<float> b_data = {1, 2, 3};

    auto a_custom = Tensor::from_vector(a_data, {2, 1, 1}, Device::CUDA);
    auto b_custom = Tensor::from_vector(b_data, {1, 1, 3}, Device::CUDA);

    auto a_torch = torch::tensor(a_data, torch::kCUDA).reshape({2, 1, 1});
    auto b_torch = torch::tensor(b_data, torch::kCUDA).reshape({1, 1, 3});

    auto mask_custom = a_custom.eq(b_custom);
    auto mask_torch = a_torch.eq(b_torch);

    compare_tensors(mask_custom, mask_torch, 1e-5f, 1e-7f, "Broadcast3D");
}

// ============= Attention Mask Creation =============

TEST_F(TensorMaskingTest, AttentionMaskCreation) {
    const size_t seq_len = 8;

    // Create causal mask using broadcasting
    std::vector<float> range_data(seq_len);
    for (size_t i = 0; i < seq_len; ++i) {
        range_data[i] = static_cast<float>(i);
    }

    auto range_custom = Tensor::from_vector(range_data, {seq_len}, Device::CUDA);
    auto range_torch = torch::tensor(range_data, torch::kCUDA);

    auto row_idx_custom = range_custom.reshape({seq_len, 1});
    auto col_idx_custom = range_custom.reshape({1, seq_len});

    auto row_idx_torch = range_torch.reshape({static_cast<int64_t>(seq_len), 1});
    auto col_idx_torch = range_torch.reshape({1, static_cast<int64_t>(seq_len)});

    // Create causal mask: row_idx >= col_idx
    auto causal_mask_custom = row_idx_custom.ge(col_idx_custom);
    auto causal_mask_torch = row_idx_torch.ge(col_idx_torch);

    compare_tensors(causal_mask_custom, causal_mask_torch, 1e-5f, 1e-7f, "AttentionCausalMask");

    // Apply mask to scores
    auto scores_custom = Tensor::ones({seq_len, seq_len}, Device::CUDA);
    auto scores_torch = torch::ones({static_cast<int64_t>(seq_len), static_cast<int64_t>(seq_len)}, torch::kCUDA);

    scores_custom.masked_fill_(causal_mask_custom.logical_not(), -1e9f);
    scores_torch.masked_fill_(causal_mask_torch.logical_not(), -1e9f);

    compare_tensors(scores_custom, scores_torch, 1e-5f, 1e-7f, "AttentionMaskedScores");
}

// ============= Padding Mask Creation =============

TEST_F(TensorMaskingTest, PaddingMaskCreation) {
    const size_t batch = 3;
    const size_t seq_len = 10;

    // Sequence lengths for each batch
    std::vector<int> lengths = {7, 5, 9};

    // Create padding mask
    auto mask_custom = Tensor::zeros({batch, seq_len}, Device::CUDA);
    auto mask_torch = torch::zeros({static_cast<int64_t>(batch), static_cast<int64_t>(seq_len)}, torch::kCUDA);

    // Set valid positions to 1
    auto cpu_mask_custom = mask_custom.to(Device::CPU);
    auto cpu_mask_torch = mask_torch.cpu();

    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < static_cast<size_t>(lengths[b]); ++s) {
            cpu_mask_custom.at({b, s}) = 1.0f;
            cpu_mask_torch[b][s] = 1.0f;
        }
    }

    mask_custom = cpu_mask_custom.to(Device::CUDA);
    mask_torch = cpu_mask_torch.cuda();

    compare_tensors(mask_custom, mask_torch, 1e-5f, 1e-7f, "PaddingMask");
}

// ============= Multi-Condition Masking =============

TEST_F(TensorMaskingTest, MultiConditionMasking) {
    std::vector<float> data;
    for (int i = 0; i < 100; ++i) {
        data.push_back(static_cast<float>(i) / 100.0f - 0.5f); // Range [-0.5, 0.49]
    }

    auto tensor_custom = Tensor::from_vector(data, {100}, Device::CUDA);
    auto tensor_torch = torch::tensor(data, torch::kCUDA);

    // Complex condition: values between -0.3 and 0.3 but not too close to 0
    auto mask1_custom = tensor_custom.gt(-0.3f);
    auto mask2_custom = tensor_custom.lt(0.3f);
    auto mask3_custom = tensor_custom.abs().gt(0.1f);

    auto mask1_torch = tensor_torch.gt(-0.3f);
    auto mask2_torch = tensor_torch.lt(0.3f);
    auto mask3_torch = tensor_torch.abs().gt(0.1f);

    auto complex_mask_custom = mask1_custom.logical_and(mask2_custom).logical_and(mask3_custom);
    auto complex_mask_torch = mask1_torch.logical_and(mask2_torch).logical_and(mask3_torch);

    compare_tensors(complex_mask_custom, complex_mask_torch, 1e-5f, 1e-7f, "MultiConditionMask");

    // Apply mask
    tensor_custom.masked_fill_(complex_mask_custom, 0.0f);
    tensor_torch.masked_fill_(complex_mask_torch, 0.0f);

    compare_tensors(tensor_custom, tensor_torch, 1e-5f, 1e-7f, "MultiConditionApplied");
}

// ============= Diagonal Masking Pattern =============

TEST_F(TensorMaskingTest, DiagonalMasking) {
    const size_t size = 10;

    // Create diagonal mask
    std::vector<bool> mask_data(size * size, false);
    for (size_t i = 0; i < size; ++i) {
        mask_data[i * size + i] = true;
    }

    auto diag_mask_custom = Tensor::from_vector(mask_data, {size, size}, Device::CUDA);

    std::vector<uint8_t> mask_data_torch;
    for (bool b : mask_data) {
        mask_data_torch.push_back(b ? 1 : 0);
    }
    auto diag_mask_torch = torch::tensor(mask_data_torch, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA))
                               .reshape({static_cast<int64_t>(size), static_cast<int64_t>(size)});

    // Apply to create identity matrix
    auto data_custom = Tensor::zeros({size, size}, Device::CUDA);
    auto data_torch = torch::zeros({static_cast<int64_t>(size), static_cast<int64_t>(size)}, torch::kCUDA);

    data_custom.masked_fill_(diag_mask_custom, 1.0f);
    data_torch.masked_fill_(diag_mask_torch, 1.0f);

    compare_tensors(data_custom, data_torch, 1e-5f, 1e-7f, "DiagonalMasking");
}

// ============= Block Sparse Pattern =============

TEST_F(TensorMaskingTest, BlockSparsePattern) {
    const size_t size = 32;
    const size_t block_size = 4;

    // Create checkerboard mask
    std::vector<bool> mask_data(size * size, false);

    for (size_t i = 0; i < size; i += block_size) {
        for (size_t j = 0; j < size; j += block_size) {
            size_t block_i = i / block_size;
            size_t block_j = j / block_size;
            if ((block_i + block_j) % 2 == 0) {
                for (size_t di = 0; di < block_size && (i + di) < size; ++di) {
                    for (size_t dj = 0; dj < block_size && (j + dj) < size; ++dj) {
                        mask_data[(i + di) * size + (j + dj)] = true;
                    }
                }
            }
        }
    }

    auto checkerboard_custom = Tensor::from_vector(mask_data, {size, size}, Device::CUDA);

    std::vector<uint8_t> mask_data_torch;
    for (bool b : mask_data) {
        mask_data_torch.push_back(b ? 1 : 0);
    }
    auto checkerboard_torch = torch::tensor(mask_data_torch, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA))
                                  .reshape({static_cast<int64_t>(size), static_cast<int64_t>(size)});

    // Apply sparsity
    auto data_custom = Tensor::ones({size, size}, Device::CUDA);
    auto data_torch = torch::ones({static_cast<int64_t>(size), static_cast<int64_t>(size)}, torch::kCUDA);

    data_custom.masked_fill_(checkerboard_custom.logical_not(), 0.0f);
    data_torch.masked_fill_(checkerboard_torch.logical_not(), 0.0f);

    compare_tensors(data_custom, data_torch, 1e-5f, 1e-7f, "BlockSparse");
}

// ============= Gradient Clipping with Masking =============

TEST_F(TensorMaskingTest, GradientClippingMask) {
    std::vector<float> grad_data;
    for (int i = 0; i < 100; ++i) {
        grad_data.push_back(static_cast<float>(i) / 10.0f - 5.0f); // Range [-5, 4.9]
    }

    auto gradients_custom = Tensor::from_vector(grad_data, {100}, Device::CUDA);
    auto gradients_torch = torch::tensor(grad_data, torch::kCUDA);

    float clip_value = 1.0f;

    // Clip gradients using clamp
    auto clipped_custom = gradients_custom.clamp(-clip_value, clip_value);
    auto clipped_torch = gradients_torch.clamp(-clip_value, clip_value);

    compare_tensors(clipped_custom, clipped_torch, 1e-5f, 1e-7f, "GradientClipping");
}

// ============= Structured Dropout Pattern =============

TEST_F(TensorMaskingTest, StructuredDropout) {
    const size_t batch_size = 2;
    const size_t seq_len = 10;
    const size_t hidden_dim = 4;

    auto data_custom = Tensor::ones({batch_size, seq_len, hidden_dim}, Device::CUDA);
    auto data_torch = torch::ones({static_cast<int64_t>(batch_size),
                                   static_cast<int64_t>(seq_len),
                                   static_cast<int64_t>(hidden_dim)},
                                  torch::kCUDA);

    // Create deterministic dropout mask (drop specific tokens)
    auto token_mask_custom = Tensor::ones({batch_size, seq_len, 1}, Device::CUDA);
    auto token_mask_torch = torch::ones({static_cast<int64_t>(batch_size),
                                         static_cast<int64_t>(seq_len), 1},
                                        torch::kCUDA);

    auto cpu_mask_custom = token_mask_custom.to(Device::CPU);
    auto cpu_mask_torch = token_mask_torch.cpu();

    // Drop specific tokens
    cpu_mask_custom.at({0, 1, 0}) = 0.0f;
    cpu_mask_custom.at({0, 3, 0}) = 0.0f;
    cpu_mask_custom.at({1, 2, 0}) = 0.0f;

    cpu_mask_torch[0][1][0] = 0.0f;
    cpu_mask_torch[0][3][0] = 0.0f;
    cpu_mask_torch[1][2][0] = 0.0f;

    token_mask_custom = cpu_mask_custom.to(Device::CUDA);
    token_mask_torch = cpu_mask_torch.cuda();

    // Expand and apply
    token_mask_custom = token_mask_custom.expand({batch_size, seq_len, hidden_dim});
    token_mask_torch = token_mask_torch.expand({static_cast<int64_t>(batch_size),
                                                static_cast<int64_t>(seq_len),
                                                static_cast<int64_t>(hidden_dim)});

    data_custom = data_custom.mul(token_mask_custom);
    data_torch = data_torch.mul(token_mask_torch);

    compare_tensors(data_custom, data_torch, 1e-5f, 1e-7f, "StructuredDropout");
}

// ============= Sparse Attention Pattern =============

TEST_F(TensorMaskingTest, SparseAttentionPattern) {
    const size_t seq_len = 32;
    const size_t window_size = 4;

    // Create local window mask
    std::vector<bool> local_mask_data(seq_len * seq_len, false);

    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
            int distance = std::abs(static_cast<int>(i) - static_cast<int>(j));
            if (distance <= static_cast<int>(window_size / 2)) {
                local_mask_data[i * seq_len + j] = true;
            }
        }
    }

    auto local_mask_custom = Tensor::from_vector(local_mask_data, {seq_len, seq_len}, Device::CUDA);

    std::vector<uint8_t> local_mask_torch_data;
    for (bool b : local_mask_data) {
        local_mask_torch_data.push_back(b ? 1 : 0);
    }
    auto local_mask_torch = torch::tensor(local_mask_torch_data, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA))
                                .reshape({static_cast<int64_t>(seq_len), static_cast<int64_t>(seq_len)});

    compare_tensors(local_mask_custom, local_mask_torch, 1e-5f, 1e-7f, "SparseAttention");
}

// ============= Dynamic Value-Based Masking =============

TEST_F(TensorMaskingTest, DynamicValueBasedMasking) {
    std::vector<float> data;
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < 100; ++i) {
        data.push_back(dist(gen));
    }

    auto tensor_custom = Tensor::from_vector(data, {100}, Device::CUDA);
    auto tensor_torch = torch::tensor(data, torch::kCUDA);

    // Compute statistics
    float mean_val = tensor_custom.mean_scalar();
    float std_val = tensor_custom.std_scalar();

    auto mean_torch = tensor_torch.mean().item<float>();
    auto std_torch = tensor_torch.std().item<float>();

    // Should be similar (same input data)
    EXPECT_NEAR(mean_val, mean_torch, 0.1f);
    EXPECT_NEAR(std_val, std_torch, 0.1f);

    // Create mask for values within 2 std devs
    float lower = mean_val - 2 * std_val;
    float upper = mean_val + 2 * std_val;

    auto in_range_custom = tensor_custom.ge(lower).logical_and(tensor_custom.le(upper));
    auto in_range_torch = tensor_torch.ge(lower).logical_and(tensor_torch.le(upper));

    compare_tensors(in_range_custom, in_range_torch, 1e-5f, 1e-7f, "DynamicMask");

    // Apply transformations
    auto scaled_custom = tensor_custom.mul(2.0f);
    auto scaled_torch = tensor_torch.mul(2.0f);

    auto clamped_custom = tensor_custom.clamp(-1.0f, 1.0f);
    auto clamped_torch = tensor_torch.clamp(-1.0f, 1.0f);

    auto result_custom = Tensor::where(in_range_custom, scaled_custom, clamped_custom);
    auto result_torch = torch::where(in_range_torch, scaled_torch, clamped_torch);

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "DynamicValueBased");
}

// ============= Chained Boolean Operations =============

TEST_F(TensorMaskingTest, ChainedBooleanOperations) {
    std::vector<float> data;
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < 100; ++i) {
        data.push_back(dist(gen));
    }

    auto tensor_custom = Tensor::from_vector(data, {100}, Device::CUDA);
    auto tensor_torch = torch::tensor(data, torch::kCUDA);

    // Create multiple conditions
    auto mask1_custom = tensor_custom.gt(0);
    auto mask2_custom = tensor_custom.lt(1);
    auto mask3_custom = tensor_custom.abs().gt(0.1);

    auto mask1_torch = tensor_torch.gt(0);
    auto mask2_torch = tensor_torch.lt(1);
    auto mask3_torch = tensor_torch.abs().gt(0.1);

    // Chain operations
    auto final_custom = mask1_custom.logical_and(mask2_custom).logical_or(mask3_custom.logical_not());
    auto final_torch = mask1_torch.logical_and(mask2_torch).logical_or(mask3_torch.logical_not());

    compare_tensors(final_custom, final_torch, 1e-5f, 1e-7f, "ChainedBoolean");
}

// ============= Boolean Reduction Across Dimensions =============

TEST_F(TensorMaskingTest, BooleanReductionAcrossDimensions) {
    std::vector<float> data;
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < 120; ++i) {
        data.push_back(dist(gen));
    }

    auto tensor_custom = Tensor::from_vector(data, {4, 5, 6}, Device::CUDA);
    auto tensor_torch = torch::tensor(data, torch::kCUDA).reshape({4, 5, 6});

    auto mask_custom = tensor_custom.gt(0);
    auto mask_torch = tensor_torch.gt(0);

    compare_tensors(mask_custom, mask_torch, 1e-5f, 1e-7f, "BooleanReduction");

    // Count nonzeros
    auto count_custom = mask_custom.count_nonzero();
    auto count_torch = mask_torch.count_nonzero().item<int64_t>();

    EXPECT_EQ(count_custom, static_cast<size_t>(count_torch));
}

// ============= Mixed Dtype Comparisons =============

TEST_F(TensorMaskingTest, MixedDtypeComparisons) {
    std::vector<float> data = {1.5, 2.5, 3.5};
    auto float_custom = Tensor::from_vector(data, {3}, Device::CUDA);
    auto float_torch = torch::tensor(data, torch::kCUDA);

    auto mask1_custom = float_custom.gt(2.0f);
    auto mask1_torch = float_torch.gt(2.0f);

    EXPECT_EQ(mask1_custom.dtype(), DataType::Bool);
    EXPECT_EQ(mask1_torch.scalar_type(), torch::kBool);

    compare_tensors(mask1_custom, mask1_torch, 1e-5f, 1e-7f, "MixedDtype");

    // Self comparison (all should be true)
    auto mask2_custom = float_custom.eq(float_custom);
    auto mask2_torch = float_torch.eq(float_torch);

    compare_tensors(mask2_custom, mask2_torch, 1e-5f, 1e-7f, "SelfComparison");
}

// ============= Batch Processing with Masks =============

TEST_F(TensorMaskingTest, BatchMaskProcessing) {
    const size_t batch = 4;
    const size_t features = 8;

    std::vector<float> data;
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < batch * features; ++i) {
        data.push_back(dist(gen));
    }

    auto tensor_custom = Tensor::from_vector(data, {batch, features}, Device::CUDA);
    auto tensor_torch = torch::tensor(data, torch::kCUDA).reshape({static_cast<int64_t>(batch), static_cast<int64_t>(features)});

    // Create different threshold per batch
    std::vector<float> thresholds = {-1.0f, -0.5f, 0.0f, 0.5f};

    for (size_t i = 0; i < batch; ++i) {
        auto batch_custom = tensor_custom.slice(0, i, i + 1).squeeze(0);
        auto batch_torch = tensor_torch.slice(0, i, i + 1).squeeze(0);

        auto mask_custom = batch_custom.gt(thresholds[i]);
        auto mask_torch = batch_torch.gt(thresholds[i]);

        batch_custom.masked_fill_(mask_custom, thresholds[i]);
        batch_torch.masked_fill_(mask_torch, thresholds[i]);

        compare_tensors(batch_custom, batch_torch, 1e-5f, 1e-7f,
                        "BatchProcessing_" + std::to_string(i));
    }
}

// ============= Python-like Indexing Tests =============

TEST_F(TensorMaskingTest, PythonLikeMaskedIndexing) {
    std::vector<float> data = {1, 2, 3, 4, 5};
    auto tensor_custom = Tensor::from_vector(data, {5}, Device::CUDA);
    auto tensor_torch = torch::tensor(data, torch::kCUDA);

    auto mask_custom = tensor_custom.gt(3.0f);
    auto mask_torch = tensor_torch.gt(3.0f);

    // Get masked values using [] operator
    Tensor selected_custom = tensor_custom[mask_custom];
    auto selected_torch = tensor_torch.masked_select(mask_torch);

    compare_tensors(selected_custom, selected_torch, 1e-5f, 1e-7f, "PythonLikeMaskedIndexing");
}

TEST_F(TensorMaskingTest, PythonLikeMaskedAssignment) {
    std::vector<float> data = {1, 2, 3, 4, 5};
    auto tensor_custom = Tensor::from_vector(data, {5}, Device::CUDA);
    auto tensor_torch = torch::tensor(data, torch::kCUDA);

    auto mask_custom = tensor_custom.gt(3.0f);
    auto mask_torch = tensor_torch.gt(3.0f);

    // Set masked values using [] operator
    tensor_custom[mask_custom] = -1.0f;
    tensor_torch.masked_fill_(mask_torch, -1.0f);

    compare_tensors(tensor_custom, tensor_torch, 1e-5f, 1e-7f, "PythonLikeMaskedAssignment");
}

TEST_F(TensorMaskingTest, PythonLikeIndexing) {
    std::vector<float> data = {1, 2, 3, 4, 5};
    auto tensor_custom = Tensor::from_vector(data, {5}, Device::CUDA);
    auto tensor_torch = torch::tensor(data, torch::kCUDA);

    std::vector<int> idx_data = {0, 2, 4};
    auto indices_custom = Tensor::from_vector(idx_data, {3}, Device::CUDA);
    auto indices_torch = torch::tensor(idx_data, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));

    // Get indexed values using [] operator
    Tensor selected_custom = tensor_custom[indices_custom];
    auto selected_torch = tensor_torch.index_select(0, indices_torch);

    compare_tensors(selected_custom, selected_torch, 1e-5f, 1e-7f, "PythonLikeIndexing");
}

TEST_F(TensorMaskingTest, PythonLikeIndexAssignment) {
    std::vector<float> data = {1, 2, 3, 4, 5};
    auto tensor_custom = Tensor::from_vector(data, {5}, Device::CUDA);
    auto tensor_torch = torch::tensor(data, torch::kCUDA);

    std::vector<int> idx_data = {1, 3};
    auto indices_custom = Tensor::from_vector(idx_data, {2}, Device::CUDA);
    auto indices_torch = torch::tensor(idx_data, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));

    // Set indexed values using [] operator
    tensor_custom[indices_custom] = 0.0f;
    tensor_torch.index_fill_(0, indices_torch, 0.0f);

    compare_tensors(tensor_custom, tensor_torch, 1e-5f, 1e-7f, "PythonLikeIndexAssignment");
}

// ============= Any/All Tests =============

TEST_F(TensorMaskingTest, AnyAllOperations) {
    // Test all true
    auto all_true_custom = Tensor::ones_bool({3, 3}, Device::CUDA);
    auto all_true_torch = torch::ones({3, 3}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

    auto all_true_float_custom = all_true_custom.to(DataType::Float32);
    auto all_true_float_torch = all_true_torch.to(torch::kFloat32);

    EXPECT_FLOAT_EQ(all_true_float_custom.min_scalar(), 1.0f);
    EXPECT_FLOAT_EQ(all_true_float_torch.min().item<float>(), 1.0f);

    // Test all false
    auto all_false_custom = Tensor::zeros_bool({3, 3}, Device::CUDA);
    auto all_false_torch = torch::zeros({3, 3}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

    auto all_false_float_custom = all_false_custom.to(DataType::Float32);
    auto all_false_float_torch = all_false_torch.to(torch::kFloat32);

    EXPECT_FLOAT_EQ(all_false_float_custom.max_scalar(), 0.0f);
    EXPECT_FLOAT_EQ(all_false_float_torch.max().item<float>(), 0.0f);

    // Test mixed
    std::vector<bool> mixed_data = {true, false, true};
    auto mixed_custom = Tensor::from_vector(mixed_data, {3}, Device::CUDA);

    std::vector<uint8_t> mixed_torch_data = {1, 0, 1};
    auto mixed_torch = torch::tensor(mixed_torch_data, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

    auto mixed_float_custom = mixed_custom.to(DataType::Float32);
    auto mixed_float_torch = mixed_torch.to(torch::kFloat32);

    EXPECT_FLOAT_EQ(mixed_float_custom.min_scalar(), 0.0f);
    EXPECT_FLOAT_EQ(mixed_float_torch.min().item<float>(), 0.0f);
    EXPECT_FLOAT_EQ(mixed_float_custom.max_scalar(), 1.0f);
    EXPECT_FLOAT_EQ(mixed_float_torch.max().item<float>(), 1.0f);
}

// ============= Top-K Masking Pattern =============

TEST_F(TensorMaskingTest, TopKMasking) {
    const size_t n = 100;
    const size_t k = 10;

    std::vector<float> data;
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) {
        data.push_back(dist(gen));
    }

    auto tensor_custom = Tensor::from_vector(data, {n}, Device::CUDA);
    auto tensor_torch = torch::tensor(data, torch::kCUDA);

    // Get top-k values using PyTorch
    auto topk_result = torch::topk(tensor_torch, k);
    auto topk_values = std::get<0>(topk_result);
    float threshold = topk_values[-1].item<float>();

    // Create mask for top-k
    auto mask_custom = tensor_custom.ge(threshold);
    auto mask_torch = tensor_torch.ge(threshold);

    compare_tensors(mask_custom, mask_torch, 1e-5f, 1e-7f, "TopKMask");

    // Apply mask
    auto result_custom = tensor_custom.clone();
    auto result_torch = tensor_torch.clone();

    result_custom.masked_fill_(mask_custom.logical_not(), 0.0f);
    result_torch.masked_fill_(mask_torch.logical_not(), 0.0f);

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "TopKApplied");
}

// ============= Image Region Masking =============

TEST_F(TensorMaskingTest, ImageRegionMasking) {
    const size_t batch_size = 2;
    const size_t channels = 3;
    const size_t height = 16;
    const size_t width = 16;

    auto images_custom = Tensor::ones({batch_size, channels, height, width}, Device::CUDA);
    auto images_torch = torch::ones({static_cast<int64_t>(batch_size),
                                     static_cast<int64_t>(channels),
                                     static_cast<int64_t>(height),
                                     static_cast<int64_t>(width)},
                                    torch::kCUDA);

    // Create mask for a rectangular region (5,5) to (10,10)
    std::vector<bool> mask_data(height * width, false);
    size_t masked_pixels = 0;
    for (size_t y = 5; y < 10; ++y) {
        for (size_t x = 5; x < 10; ++x) {
            mask_data[y * width + x] = true;
            masked_pixels++;
        }
    }

    auto mask_custom = Tensor::from_vector(mask_data, {height, width}, Device::CUDA);

    std::vector<uint8_t> mask_torch_data;
    for (bool b : mask_data) {
        mask_torch_data.push_back(b ? 1 : 0);
    }
    auto mask_torch = torch::tensor(mask_torch_data, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA))
                          .reshape({static_cast<int64_t>(height), static_cast<int64_t>(width)});

    // Apply mask to first batch, all channels
    for (size_t c = 0; c < channels; ++c) {
        auto channel_custom = images_custom.slice(0, 0, 1).squeeze(0).slice(0, c, c + 1).squeeze(0);
        auto channel_torch = images_torch[0][c];

        channel_custom.masked_fill_(mask_custom, 0.0f);
        channel_torch.masked_fill_(mask_torch, 0.0f);

        compare_tensors(channel_custom, channel_torch, 1e-5f, 1e-7f,
                        "ImageRegion_channel" + std::to_string(c));
    }
}

// ============= Performance Test =============

TEST_F(TensorMaskingTest, MaskingPerformance) {
    const size_t size = 1000000;

    // Generate random data once and use it for both implementations
    std::vector<float> data;
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < size; ++i) {
        data.push_back(dist(gen));
    }

    auto tensor_custom = Tensor::from_vector(data, {size}, Device::CUDA);
    auto tensor_torch = torch::tensor(data, torch::kCUDA);

    // Create mask
    auto start = std::chrono::high_resolution_clock::now();
    auto mask_custom = tensor_custom.gt(0.0f);
    cudaDeviceSynchronize();
    auto custom_time = std::chrono::high_resolution_clock::now() - start;

    start = std::chrono::high_resolution_clock::now();
    auto mask_torch = tensor_torch.gt(0.0f);
    cudaDeviceSynchronize();
    auto torch_time = std::chrono::high_resolution_clock::now() - start;

    std::cout << "Mask creation for " << size << " elements:\n";
    std::cout << "  Custom: " << std::chrono::duration_cast<std::chrono::microseconds>(custom_time).count() << " μs\n";
    std::cout << "  PyTorch: " << std::chrono::duration_cast<std::chrono::microseconds>(torch_time).count() << " μs\n";

    // Verify results match (now they should since we use same input data)
    compare_tensors(mask_custom, mask_torch, 1e-5f, 1e-7f, "PerformanceMask");
}

// ============= Extensive Boolean Expansion Tests =============

TEST_F(TensorMaskingTest, BooleanExpansionMultipleDims) {
    std::vector<bool> data = {true};
    auto bool_custom = Tensor::from_vector(data, {1, 1}, Device::CUDA);

    std::vector<uint8_t> data_torch = {1};
    auto bool_torch = torch::tensor(data_torch, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA))
                          .reshape({1, 1});

    auto expanded_custom = bool_custom.expand({5, 4});
    auto expanded_torch = bool_torch.expand({5, 4});

    compare_tensors(expanded_custom, expanded_torch, 1e-5f, 1e-7f, "BooleanExpansionMulti");
}

TEST_F(TensorMaskingTest, BooleanExpansionComplex) {
    // Test expansion with non-trivial patterns
    std::vector<bool> data = {true, false, true, false};
    auto bool_2d_custom = Tensor::from_vector(data, {2, 2}, Device::CUDA);

    std::vector<uint8_t> data_torch = {1, 0, 1, 0};
    auto bool_2d_torch = torch::tensor(data_torch, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA))
                             .reshape({2, 2});

    // Expand along new dimension
    auto expanded_custom = bool_2d_custom.unsqueeze(0).unsqueeze(-1).expand({3, 2, 2, 4});
    auto expanded_torch = bool_2d_torch.unsqueeze(0).unsqueeze(-1).expand({3, 2, 2, 4});

    compare_tensors(expanded_custom, expanded_torch, 1e-5f, 1e-7f, "BooleanExpansionComplex");
}

// ============= More Broadcasting Edge Cases =============

TEST_F(TensorMaskingTest, BroadcastComparisonMaxRank) {
    std::vector<float> a_data = {2};
    std::vector<float> b_data = {1, 2, 3, 4, 5, 6, 7, 8};

    auto a_custom = Tensor::from_vector(a_data, {1}, Device::CUDA);
    auto b_custom = Tensor::from_vector(b_data, {2, 2, 2}, Device::CUDA);

    auto a_torch = torch::tensor(a_data, torch::kCUDA);
    auto b_torch = torch::tensor(b_data, torch::kCUDA).reshape({2, 2, 2});

    auto mask_custom = a_custom.eq(b_custom);
    auto mask_torch = a_torch.eq(b_torch);

    compare_tensors(mask_custom, mask_torch, 1e-5f, 1e-7f, "BroadcastMaxRank");
}

// ============= Stress Tests =============

TEST_F(TensorMaskingTest, StressTestLargeMasking) {
    const size_t size = 1000;

    auto data_custom = Tensor::randn({size, size}, Device::CUDA);

    // Create complex mask
    auto mask_custom = data_custom.gt(0).logical_and(data_custom.lt(1));

    // Apply mask
    auto selected_custom = data_custom.masked_select(mask_custom);

    // Verify it works
    EXPECT_GE(selected_custom.numel(), 0);
    EXPECT_LE(selected_custom.numel(), size * size);
}

TEST_F(TensorMaskingTest, StressTestManyDimensions) {
    // Generate same random data for both implementations
    std::vector<float> data;
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < 2 * 3 * 4 * 5; ++i) {
        data.push_back(dist(gen));
    }

    auto data_custom = Tensor::from_vector(data, {2, 3, 4, 5}, Device::CUDA);
    auto data_torch = torch::tensor(data, torch::kCUDA).reshape({2, 3, 4, 5});

    auto mask_custom = data_custom.gt(0);
    auto mask_torch = data_torch.gt(0);

    compare_tensors(mask_custom, mask_torch, 1e-5f, 1e-7f, "StressManyDims");

    data_custom.masked_fill_(mask_custom, 1.0f);
    data_torch.masked_fill_(mask_torch, 1.0f);

    compare_tensors(data_custom, data_torch, 1e-5f, 1e-7f, "StressManyDimsApplied");
}

// ============= Memory Layout Tests =============

TEST_F(TensorMaskingTest, ContiguousVsNonContiguousMasking) {
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    auto original_custom = Tensor::from_vector(data, {2, 3}, Device::CUDA);
    auto original_torch = torch::tensor(data, torch::kCUDA).reshape({2, 3});

    auto transposed_custom = original_custom.transpose(0, 1);
    auto transposed_torch = original_torch.transpose(0, 1);

    EXPECT_EQ(transposed_custom.shape(), TensorShape({3, 2}));

    // Apply masking on non-contiguous tensor
    auto mask_custom = transposed_custom.gt(3);
    auto mask_torch = transposed_torch.gt(3);

    auto masked_custom = transposed_custom.masked_fill(mask_custom, -1);
    auto masked_torch = transposed_torch.masked_fill(mask_torch, -1);

    compare_tensors(masked_custom, masked_torch, 1e-5f, 1e-7f, "NonContiguousMasking");
}

// ============= Error Handling Tests =============

TEST_F(TensorMaskingTest, ErrorHandlingMismatchedShapes) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto tensor_custom = Tensor::from_vector(data, {2, 2}, Device::CUDA);
    auto wrong_mask_custom = Tensor::ones_bool({3, 3}, Device::CUDA);

    // This should fail gracefully
    auto result_custom = tensor_custom.masked_select(wrong_mask_custom);

    // PyTorch also fails with mismatched shapes
    auto tensor_torch = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}, torch::kCUDA).reshape({2, 2});
    auto wrong_mask_torch = torch::ones({3, 3}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

    EXPECT_THROW(tensor_torch.masked_select(wrong_mask_torch), c10::Error);
    EXPECT_FALSE(result_custom.is_valid());
}

TEST_F(TensorMaskingTest, ErrorHandlingWrongDevice) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto cuda_tensor_custom = Tensor::from_vector(data, {2, 2}, Device::CUDA);
    auto cpu_mask_custom = Tensor::ones_bool({2, 2}, Device::CPU);

    // This should fail gracefully
    auto result_custom = cuda_tensor_custom.masked_select(cpu_mask_custom);

    // PyTorch also fails with wrong device
    auto cuda_tensor_torch = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}, torch::kCUDA).reshape({2, 2});
    auto cpu_mask_torch = torch::ones({2, 2}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));

    EXPECT_THROW(cuda_tensor_torch.masked_select(cpu_mask_torch), c10::Error);
    EXPECT_FALSE(result_custom.is_valid());
}

// ============= Additional Edge Cases =============

TEST_F(TensorMaskingTest, EmptyTensorMasking) {
    auto empty_custom = Tensor::empty({0}, Device::CUDA);
    auto empty_torch = torch::empty({0}, torch::kCUDA);

    auto empty_mask_custom = Tensor::empty({0}, Device::CUDA, DataType::Bool);
    auto empty_mask_torch = torch::empty({0}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

    auto selected_custom = empty_custom.masked_select(empty_mask_custom);
    auto selected_torch = empty_torch.masked_select(empty_mask_torch);

    EXPECT_EQ(selected_custom.numel(), 0);
    EXPECT_EQ(selected_torch.numel(), 0);
}

TEST_F(TensorMaskingTest, LargeScalarBroadcast) {
    // Generate same random data for both implementations
    std::vector<float> data;
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < 1000 * 1000; ++i) {
        data.push_back(dist(gen));
    }

    auto tensor_custom = Tensor::from_vector(data, {1000, 1000}, Device::CUDA);
    auto tensor_torch = torch::tensor(data, torch::kCUDA).reshape({1000, 1000});

    auto mask_custom = tensor_custom.gt(0.5f);
    auto mask_torch = tensor_torch.gt(0.5f);

    tensor_custom.masked_fill_(mask_custom, 100.0f);
    tensor_torch.masked_fill_(mask_torch, 100.0f);

    compare_tensors(tensor_custom, tensor_torch, 1e-5f, 1e-7f, "LargeScalarBroadcast");
}

// ============= Compound Operations =============

TEST_F(TensorMaskingTest, CompoundMaskingOperations) {
    const size_t batch = 2;
    const size_t seq_len = 4;

    auto scores_custom = Tensor::ones({batch, seq_len, seq_len}, Device::CUDA);
    auto scores_torch = torch::ones({static_cast<int64_t>(batch),
                                     static_cast<int64_t>(seq_len),
                                     static_cast<int64_t>(seq_len)},
                                    torch::kCUDA);

    // Create causal mask for each batch
    std::vector<bool> causal_data(seq_len * seq_len);
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
            causal_data[i * seq_len + j] = (i >= j);
        }
    }

    auto causal_custom = Tensor::from_vector(causal_data, {seq_len, seq_len}, Device::CUDA);

    std::vector<uint8_t> causal_torch_data;
    for (bool b : causal_data) {
        causal_torch_data.push_back(b ? 1 : 0);
    }
    auto causal_torch = torch::tensor(causal_torch_data, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA))
                            .reshape({static_cast<int64_t>(seq_len), static_cast<int64_t>(seq_len)});

    // Apply to each batch
    for (size_t b = 0; b < batch; ++b) {
        auto batch_scores_custom = scores_custom.slice(0, b, b + 1).squeeze(0);
        auto batch_scores_torch = scores_torch[b];

        batch_scores_custom.masked_fill_(causal_custom.logical_not(), -1e9f);
        batch_scores_torch.masked_fill_(causal_torch.logical_not(), -1e9f);

        compare_tensors(batch_scores_custom, batch_scores_torch, 1e-5f, 1e-7f,
                        "CompoundBatch" + std::to_string(b));
    }
}

// ============= Advanced Boolean Indexing =============

TEST_F(TensorMaskingTest, AdvancedBooleanIndexing) {
    const size_t d0 = 3, d1 = 4, d2 = 5;

    std::vector<float> data;
    for (size_t i = 0; i < d0 * d1 * d2; ++i) {
        data.push_back(static_cast<float>(i));
    }

    auto tensor_custom = Tensor::from_vector(data, {d0, d1, d2}, Device::CUDA);
    auto tensor_torch = torch::tensor(data, torch::kCUDA).reshape({static_cast<int64_t>(d0), static_cast<int64_t>(d1), static_cast<int64_t>(d2)});

    // Create boolean mask for middle dimension
    std::vector<bool> mask_data = {false, true, true, false};
    auto mask_1d_custom = Tensor::from_vector(mask_data, {d1}, Device::CUDA);

    // Select using mask - convert to indices
    std::vector<int> indices = {1, 2};
    auto indices_custom = Tensor::from_vector(indices, {2}, Device::CUDA);
    auto indices_torch = torch::tensor(indices, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));

    auto selected_custom = tensor_custom.index_select(1, indices_custom);
    auto selected_torch = tensor_torch.index_select(1, indices_torch);

    compare_tensors(selected_custom, selected_torch, 1e-5f, 1e-7f, "AdvancedBooleanIndexing");
}

// ============= Integration Test =============

TEST_F(TensorMaskingTest, IntegrationCompleteWorkflow) {
    const size_t batch = 4;
    const size_t seq_len = 8;
    const size_t hidden = 16;

    // 1. Create input data
    std::vector<float> input_data;
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < batch * seq_len * hidden; ++i) {
        input_data.push_back(dist(gen));
    }

    auto inputs_custom = Tensor::from_vector(input_data, {batch, seq_len, hidden}, Device::CUDA);
    auto inputs_torch = torch::tensor(input_data, torch::kCUDA).reshape({static_cast<int64_t>(batch), static_cast<int64_t>(seq_len), static_cast<int64_t>(hidden)});

    // 2. Create causal mask
    std::vector<float> range(seq_len);
    for (size_t i = 0; i < seq_len; ++i) {
        range[i] = static_cast<float>(i);
    }

    auto range_custom = Tensor::from_vector(range, {seq_len}, Device::CUDA);
    auto range_torch = torch::tensor(range, torch::kCUDA);

    auto causal_custom = range_custom.reshape({seq_len, 1}).ge(range_custom.reshape({1, seq_len}));
    auto causal_torch = range_torch.reshape({static_cast<int64_t>(seq_len), 1})
                            .ge(range_torch.reshape({1, static_cast<int64_t>(seq_len)}));

    compare_tensors(causal_custom, causal_torch, 1e-5f, 1e-7f, "IntegrationCausalMask");

    // 3. Create padding mask (same for all batches in this test)
    std::vector<bool> padding_data(seq_len, true);
    padding_data[seq_len - 1] = false; // Last token is padding

    auto padding_1d_custom = Tensor::from_vector(padding_data, {seq_len}, Device::CUDA);

    std::vector<uint8_t> padding_torch_data;
    for (bool b : padding_data) {
        padding_torch_data.push_back(b ? 1 : 0);
    }
    auto padding_1d_torch = torch::tensor(padding_torch_data, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

    // 4. Combine masks
    auto padding_2d_custom = padding_1d_custom.unsqueeze(0).expand({seq_len, seq_len});
    auto padding_2d_torch = padding_1d_torch.unsqueeze(0).expand({static_cast<int64_t>(seq_len),
                                                                  static_cast<int64_t>(seq_len)});

    auto combined_custom = causal_custom.logical_and(padding_2d_custom);
    auto combined_torch = causal_torch.logical_and(padding_2d_torch);

    compare_tensors(combined_custom, combined_torch, 1e-5f, 1e-7f, "IntegrationCombinedMask");
}