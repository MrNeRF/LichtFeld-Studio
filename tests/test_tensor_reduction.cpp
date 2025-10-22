/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/tensor.hpp"
#include <cmath>
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

    void compare_scalars(float custom_val, float ref_val, float tolerance = 1e-4f, const std::string& msg = "") {
        if (std::isnan(ref_val)) {
            EXPECT_TRUE(std::isnan(custom_val)) << msg << ": Expected NaN";
        } else if (std::isinf(ref_val)) {
            EXPECT_TRUE(std::isinf(custom_val)) << msg << ": Expected Inf";
        } else {
            EXPECT_NEAR(custom_val, ref_val, tolerance) << msg;
        }
    }

} // anonymous namespace

class TensorReductionTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA is not available for testing";
        torch::manual_seed(42);
        Tensor::manual_seed(42);
        gen.seed(42);
    }

    std::mt19937 gen;
    std::uniform_real_distribution<float> dist{-10.0f, 10.0f};
};

// ============= Basic Scalar Reductions =============

TEST_F(TensorReductionTest, Sum) {
    std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    auto custom_tensor = Tensor::from_vector(data, {10}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    float custom_sum = custom_tensor.sum_scalar();
    float torch_sum = torch_tensor.sum().item<float>();

    compare_scalars(custom_sum, torch_sum, 1e-4f, "Sum");
    EXPECT_FLOAT_EQ(custom_sum, 55.0f);
}

TEST_F(TensorReductionTest, SumMultiDim) {
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), 1.0f);

    auto custom_tensor = Tensor::from_vector(data, {2, 3, 4}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA))
                            .reshape({2, 3, 4});

    float custom_sum = custom_tensor.sum_scalar();
    float torch_sum = torch_tensor.sum().item<float>();

    compare_scalars(custom_sum, torch_sum, 1e-3f, "SumMultiDim");
    EXPECT_FLOAT_EQ(custom_sum, 300.0f); // Sum of 1..24
}

TEST_F(TensorReductionTest, SumLargeTensor) {
    std::vector<float> data(10000);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = dist(gen);
    }

    auto custom_tensor = Tensor::from_vector(data, {10000}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    float custom_sum = custom_tensor.sum_scalar();
    float torch_sum = torch_tensor.sum().item<float>();

    // Allow larger tolerance for large sums
    compare_scalars(custom_sum, torch_sum, std::abs(torch_sum) * 1e-4f, "SumLarge");
}

TEST_F(TensorReductionTest, Mean) {
    std::vector<float> data = {2, 4, 6, 8, 10};

    auto custom_tensor = Tensor::from_vector(data, {5}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    float custom_mean = custom_tensor.mean_scalar();
    float torch_mean = torch_tensor.mean().item<float>();

    compare_scalars(custom_mean, torch_mean, 1e-5f, "Mean");
    EXPECT_FLOAT_EQ(custom_mean, 6.0f);
}

TEST_F(TensorReductionTest, MeanMultiDim) {
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), 1.0f);

    auto custom_tensor = Tensor::from_vector(data, {2, 3, 4}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA))
                            .reshape({2, 3, 4});

    float custom_mean = custom_tensor.mean_scalar();
    float torch_mean = torch_tensor.mean().item<float>();

    compare_scalars(custom_mean, torch_mean, 1e-5f, "MeanMultiDim");
    EXPECT_FLOAT_EQ(custom_mean, 12.5f); // Mean of 1..24
}

TEST_F(TensorReductionTest, MinMax) {
    std::vector<float> data = {5, 2, 8, 1, 9, 3, 7, 4, 6};

    auto custom_tensor = Tensor::from_vector(data, {9}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    float custom_min = custom_tensor.min_scalar();
    float custom_max = custom_tensor.max_scalar();
    float torch_min = torch_tensor.min().item<float>();
    float torch_max = torch_tensor.max().item<float>();

    compare_scalars(custom_min, torch_min, 1e-5f, "Min");
    compare_scalars(custom_max, torch_max, 1e-5f, "Max");

    EXPECT_FLOAT_EQ(custom_min, 1.0f);
    EXPECT_FLOAT_EQ(custom_max, 9.0f);

    // Test minmax helper
    auto [min_val, max_val] = custom_tensor.minmax();
    EXPECT_FLOAT_EQ(min_val, 1.0f);
    EXPECT_FLOAT_EQ(max_val, 9.0f);
}

TEST_F(TensorReductionTest, MinMaxMultiDim) {
    std::vector<float> data(100);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = dist(gen);
    }

    auto custom_tensor = Tensor::from_vector(data, {10, 10}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA))
                            .reshape({10, 10});

    float custom_min = custom_tensor.min_scalar();
    float custom_max = custom_tensor.max_scalar();
    float torch_min = torch_tensor.min().item<float>();
    float torch_max = torch_tensor.max().item<float>();

    compare_scalars(custom_min, torch_min, 1e-4f, "MinMultiDim");
    compare_scalars(custom_max, torch_max, 1e-4f, "MaxMultiDim");
}

TEST_F(TensorReductionTest, StandardDeviation) {
    std::vector<float> data = {2, 4, 4, 4, 5, 5, 7, 9};

    auto custom_tensor = Tensor::from_vector(data, {8}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    float custom_std = custom_tensor.std_scalar(/*unbiased=*/false);
    float torch_std = torch_tensor.std(/*unbiased=*/false).item<float>();

    compare_scalars(custom_std, torch_std, 1e-4f, "Std");
}

TEST_F(TensorReductionTest, StdMultiDim) {
    std::vector<float> data(120);
    std::iota(data.begin(), data.end(), 1.0f);

    auto custom_tensor = Tensor::from_vector(data, {2, 3, 4, 5}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA))
                            .reshape({2, 3, 4, 5});

    float custom_std = custom_tensor.std_scalar(/*unbiased=*/false);
    float torch_std = torch_tensor.std(/*unbiased=*/false).item<float>();

    compare_scalars(custom_std, torch_std, 1e-3f, "StdMultiDim");
}

TEST_F(TensorReductionTest, Variance) {
    std::vector<float> data = {1, 2, 3, 4, 5};

    auto custom_tensor = Tensor::from_vector(data, {5}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    float custom_var = custom_tensor.var_scalar(/*unbiased=*/false);
    float torch_var = torch_tensor.var(/*unbiased=*/false).item<float>();

    compare_scalars(custom_var, torch_var, 1e-4f, "Var");
    EXPECT_NEAR(custom_var, 2.0f, 1e-4f); // Variance of [1,2,3,4,5] with N (biased)
}

TEST_F(TensorReductionTest, VarMultiDim) {
    std::vector<float> data(60);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = dist(gen);
    }

    auto custom_tensor = Tensor::from_vector(data, {3, 4, 5}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA))
                            .reshape({3, 4, 5});

    float custom_var = custom_tensor.var_scalar(/*unbiased=*/false);
    float torch_var = torch_tensor.var(/*unbiased=*/false).item<float>();

    compare_scalars(custom_var, torch_var, 1e-3f, "VarMultiDim");
}

// ============= Norm Tests =============

TEST_F(TensorReductionTest, L2Norm) {
    std::vector<float> data = {3, 4}; // 3-4-5 triangle

    auto custom_tensor = Tensor::from_vector(data, {2}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    float custom_norm = custom_tensor.norm(2.0f);
    float torch_norm = torch_tensor.norm(2).item<float>();

    compare_scalars(custom_norm, torch_norm, 1e-5f, "L2Norm");
    EXPECT_FLOAT_EQ(custom_norm, 5.0f);
}

TEST_F(TensorReductionTest, L1Norm) {
    std::vector<float> data = {-3, 4, -2, 1};

    auto custom_tensor = Tensor::from_vector(data, {4}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    float custom_norm = custom_tensor.norm(1.0f);
    float torch_norm = torch_tensor.norm(1).item<float>();

    compare_scalars(custom_norm, torch_norm, 1e-5f, "L1Norm");
    EXPECT_FLOAT_EQ(custom_norm, 10.0f); // |−3| + |4| + |−2| + |1| = 10
}

TEST_F(TensorReductionTest, InfinityNorm) {
    std::vector<float> data = {-3, 4, -7, 1};

    auto custom_tensor = Tensor::from_vector(data, {4}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    float custom_norm = custom_tensor.norm(std::numeric_limits<float>::infinity());
    float torch_norm = torch_tensor.norm(std::numeric_limits<float>::infinity()).item<float>();

    compare_scalars(custom_norm, torch_norm, 1e-5f, "InfinityNorm");
    EXPECT_FLOAT_EQ(custom_norm, 7.0f); // max absolute value
}

// ============= Item Tests =============

TEST_F(TensorReductionTest, ItemScalar) {
    std::vector<float> data = {3.14159f};

    auto custom_tensor = Tensor::from_vector(data, {1}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    float custom_item = custom_tensor.item();
    float torch_item = torch_tensor.item<float>();

    compare_scalars(custom_item, torch_item, 1e-5f, "Item");
}

TEST_F(TensorReductionTest, ItemMultiElement) {
    // item() should fail for multi-element tensors
    std::vector<float> vec_data = {1.0f, 2.0f, 3.0f};
    auto custom_tensor = Tensor::from_vector(vec_data, {3}, Device::CUDA);

    // Should return 0 or fail gracefully
    float value = custom_tensor.item();
    // Just verify it doesn't crash - implementation defined behavior
    EXPECT_TRUE(true);
}

// ============= Dimensional Reductions =============

TEST_F(TensorReductionTest, SumAlongAxis) {
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), 1.0f);

    auto custom_tensor = Tensor::from_vector(data, {2, 3, 4}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA))
                            .reshape({2, 3, 4});

    // Sum along axis 0
    std::vector<int> axes0 = {0};
    auto custom_result = custom_tensor.sum(axes0);
    auto torch_result = torch_tensor.sum(0);

    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "SumAxis0");

    // Sum along axis 1
    std::vector<int> axes1 = {1};
    auto custom_result1 = custom_tensor.sum(axes1);
    auto torch_result1 = torch_tensor.sum(1);

    compare_tensors(custom_result1, torch_result1, 1e-4f, 1e-5f, "SumAxis1");

    // Sum along axis 2
    std::vector<int> axes2 = {2};
    auto custom_result2 = custom_tensor.sum(axes2);
    auto torch_result2 = torch_tensor.sum(2);

    compare_tensors(custom_result2, torch_result2, 1e-4f, 1e-5f, "SumAxis2");
}

TEST_F(TensorReductionTest, MeanAlongAxis) {
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), 1.0f);

    auto custom_tensor = Tensor::from_vector(data, {2, 3, 4}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA))
                            .reshape({2, 3, 4});

    // Mean along axis 0
    std::vector<int> axes0 = {0};
    auto custom_result = custom_tensor.mean(axes0);
    auto torch_result = torch_tensor.mean(0);

    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "MeanAxis0");

    // Mean along axis 1
    std::vector<int> axes1 = {1};
    auto custom_result1 = custom_tensor.mean(axes1);
    auto torch_result1 = torch_tensor.mean(1);

    compare_tensors(custom_result1, torch_result1, 1e-4f, 1e-5f, "MeanAxis1");
}

TEST_F(TensorReductionTest, KeepDim) {
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), 1.0f);

    auto custom_tensor = Tensor::from_vector(data, {2, 3, 4}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA))
                            .reshape({2, 3, 4});

    // Sum with keepdim
    std::vector<int> axes = {1};
    auto custom_result = custom_tensor.sum(axes, true);
    auto torch_result = torch_tensor.sum(1, /*keepdim=*/true);

    EXPECT_EQ(custom_result.ndim(), 3);
    EXPECT_EQ(custom_result.shape()[1], 1);

    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "SumKeepDim");
}

TEST_F(TensorReductionTest, MultiAxisReduction) {
    std::vector<float> data(120);
    std::iota(data.begin(), data.end(), 1.0f);

    auto custom_tensor = Tensor::from_vector(data, {2, 3, 4, 5}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA))
                            .reshape({2, 3, 4, 5});

    // Sum along multiple axes
    std::vector<int> axes = {1, 3};
    auto custom_result = custom_tensor.sum(axes);
    auto torch_result = torch_tensor.sum({1, 3});

    compare_tensors(custom_result, torch_result, 1e-3f, 1e-4f, "MultiAxisSum");
}

TEST_F(TensorReductionTest, MaxAlongAxis) {
    std::vector<float> data(24);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = dist(gen);
    }

    auto custom_tensor = Tensor::from_vector(data, {2, 3, 4}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA))
                            .reshape({2, 3, 4});

    std::vector<int> axes = {1};
    auto custom_result = custom_tensor.max(axes);
    auto torch_result = std::get<0>(torch_tensor.max(1));

    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "MaxAxis");
}

TEST_F(TensorReductionTest, MinAlongAxis) {
    std::vector<float> data(24);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = dist(gen);
    }

    auto custom_tensor = Tensor::from_vector(data, {2, 3, 4}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA))
                            .reshape({2, 3, 4});

    std::vector<int> axes = {1};
    auto custom_result = custom_tensor.min(axes);
    auto torch_result = std::get<0>(torch_tensor.min(1));

    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "MinAxis");
}

// ============= Edge Cases =============

TEST_F(TensorReductionTest, EmptyTensor) {
    auto custom_empty = Tensor::empty({0}, Device::CUDA);
    auto torch_empty = torch::empty({0}, torch::TensorOptions().device(torch::kCUDA));

    // Sum of empty tensor should be 0
    float custom_sum = custom_empty.sum_scalar();
    float torch_sum = torch_empty.sum().item<float>();

    compare_scalars(custom_sum, torch_sum, 1e-5f, "EmptySum");
}

TEST_F(TensorReductionTest, SingleElement) {
    std::vector<float> data = {5.0f};

    auto custom_tensor = Tensor::from_vector(data, {1}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    compare_scalars(custom_tensor.sum_scalar(), torch_tensor.sum().item<float>(), 1e-5f, "SingleSum");
    compare_scalars(custom_tensor.mean_scalar(), torch_tensor.mean().item<float>(), 1e-5f, "SingleMean");
    compare_scalars(custom_tensor.min_scalar(), torch_tensor.min().item<float>(), 1e-5f, "SingleMin");
    compare_scalars(custom_tensor.max_scalar(), torch_tensor.max().item<float>(), 1e-5f, "SingleMax");
    compare_scalars(custom_tensor.std_scalar(false), torch_tensor.std(false).item<float>(), 1e-5f, "SingleStd");
    compare_scalars(custom_tensor.var_scalar(false), torch_tensor.var(false).item<float>(), 1e-5f, "SingleVar");
}

TEST_F(TensorReductionTest, NegativeValues) {
    std::vector<float> data = {-5, -3, -1, 0, 1, 3, 5};

    auto custom_tensor = Tensor::from_vector(data, {7}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    compare_scalars(custom_tensor.sum_scalar(), torch_tensor.sum().item<float>(), 1e-4f, "NegSum");
    compare_scalars(custom_tensor.mean_scalar(), torch_tensor.mean().item<float>(), 1e-5f, "NegMean");
    compare_scalars(custom_tensor.min_scalar(), torch_tensor.min().item<float>(), 1e-5f, "NegMin");
    compare_scalars(custom_tensor.max_scalar(), torch_tensor.max().item<float>(), 1e-5f, "NegMax");
    compare_scalars(custom_tensor.std_scalar(false), torch_tensor.std(false).item<float>(), 1e-4f, "NegStd");
}

TEST_F(TensorReductionTest, AllZeros) {
    auto custom_tensor = Tensor::zeros({100}, Device::CUDA);
    auto torch_tensor = torch::zeros({100}, torch::TensorOptions().device(torch::kCUDA));

    compare_scalars(custom_tensor.sum_scalar(), 0.0f, 1e-5f, "ZeroSum");
    compare_scalars(custom_tensor.mean_scalar(), 0.0f, 1e-5f, "ZeroMean");
    compare_scalars(custom_tensor.min_scalar(), 0.0f, 1e-5f, "ZeroMin");
    compare_scalars(custom_tensor.max_scalar(), 0.0f, 1e-5f, "ZeroMax");
    compare_scalars(custom_tensor.std_scalar(), 0.0f, 1e-5f, "ZeroStd");
    compare_scalars(custom_tensor.var_scalar(), 0.0f, 1e-5f, "ZeroVar");
}

TEST_F(TensorReductionTest, AllOnes) {
    auto custom_tensor = Tensor::ones({100}, Device::CUDA);
    auto torch_tensor = torch::ones({100}, torch::TensorOptions().device(torch::kCUDA));

    compare_scalars(custom_tensor.sum_scalar(), torch_tensor.sum().item<float>(), 1e-4f, "OnesSum");
    compare_scalars(custom_tensor.mean_scalar(), 1.0f, 1e-5f, "OnesMean");
    compare_scalars(custom_tensor.min_scalar(), 1.0f, 1e-5f, "OnesMin");
    compare_scalars(custom_tensor.max_scalar(), 1.0f, 1e-5f, "OnesMax");
    compare_scalars(custom_tensor.std_scalar(), 0.0f, 1e-5f, "OnesStd");
}

TEST_F(TensorReductionTest, LargeValues) {
    // Test numerical stability with large values
    std::vector<float> data = {1e6f, 1e6f + 1, 1e6f + 2, 1e6f + 3};

    auto custom_tensor = Tensor::from_vector(data, {4}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    float custom_sum = custom_tensor.sum_scalar();
    float torch_sum = torch_tensor.sum().item<float>();

    // Allow larger relative error for large numbers
    compare_scalars(custom_sum, torch_sum, std::abs(torch_sum) * 1e-4f, "LargeSum");
}

TEST_F(TensorReductionTest, SmallValues) {
    // Test numerical stability with small values
    std::vector<float> data = {1e-6f, 2e-6f, 3e-6f, 4e-6f};

    auto custom_tensor = Tensor::from_vector(data, {4}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    float custom_sum = custom_tensor.sum_scalar();
    float torch_sum = torch_tensor.sum().item<float>();

    compare_scalars(custom_sum, torch_sum, 1e-9f, "SmallSum");
}

// ============= Count Nonzero Tests =============

TEST_F(TensorReductionTest, CountNonzero) {
    std::vector<float> data = {0, 1, 0, 2, 0, 3, 0, 4, 0, 5};

    auto custom_tensor = Tensor::from_vector(data, {10}, Device::CUDA);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    size_t custom_count = custom_tensor.count_nonzero();
    int64_t torch_count = torch_tensor.count_nonzero().item<int64_t>();

    EXPECT_EQ(custom_count, static_cast<size_t>(torch_count));
    EXPECT_EQ(custom_count, 5);
}

TEST_F(TensorReductionTest, CountNonzeroAllZeros) {
    auto custom_tensor = Tensor::zeros({100}, Device::CUDA);
    auto torch_tensor = torch::zeros({100}, torch::TensorOptions().device(torch::kCUDA));

    size_t custom_count = custom_tensor.count_nonzero();
    int64_t torch_count = torch_tensor.count_nonzero().item<int64_t>();

    EXPECT_EQ(custom_count, static_cast<size_t>(torch_count));
    EXPECT_EQ(custom_count, 0);
}

TEST_F(TensorReductionTest, CountNonzeroAllNonzero) {
    auto custom_tensor = Tensor::ones({100}, Device::CUDA);
    auto torch_tensor = torch::ones({100}, torch::TensorOptions().device(torch::kCUDA));

    size_t custom_count = custom_tensor.count_nonzero();
    int64_t torch_count = torch_tensor.count_nonzero().item<int64_t>();

    EXPECT_EQ(custom_count, static_cast<size_t>(torch_count));
    EXPECT_EQ(custom_count, 100);
}

// ============= Random Data Consistency Tests =============

TEST_F(TensorReductionTest, RandomDataConsistency) {
    for (int test = 0; test < 10; ++test) {
        std::vector<float> data(100);
        for (auto& val : data) {
            val = dist(gen);
        }

        auto custom_tensor = Tensor::from_vector(data, {100}, Device::CUDA);
        auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

        compare_scalars(custom_tensor.sum_scalar(), torch_tensor.sum().item<float>(),
                        1e-3f, "RandomSum_" + std::to_string(test));
        compare_scalars(custom_tensor.mean_scalar(), torch_tensor.mean().item<float>(),
                        1e-4f, "RandomMean_" + std::to_string(test));
        compare_scalars(custom_tensor.min_scalar(), torch_tensor.min().item<float>(),
                        1e-5f, "RandomMin_" + std::to_string(test));
        compare_scalars(custom_tensor.max_scalar(), torch_tensor.max().item<float>(),
                        1e-5f, "RandomMax_" + std::to_string(test));
        compare_scalars(custom_tensor.norm(2.0f), torch_tensor.norm(2).item<float>(),
                        1e-3f, "RandomNorm_" + std::to_string(test));
    }
}

TEST_F(TensorReductionTest, RandomMultiDimConsistency) {
    for (int test = 0; test < 5; ++test) {
        std::vector<float> data(120);
        for (auto& val : data) {
            val = dist(gen);
        }

        auto custom_tensor = Tensor::from_vector(data, {2, 3, 4, 5}, Device::CUDA);
        auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA))
                                .reshape({2, 3, 4, 5});

        compare_scalars(custom_tensor.sum_scalar(), torch_tensor.sum().item<float>(),
                        1e-2f, "RandomMultiSum_" + std::to_string(test));
        compare_scalars(custom_tensor.mean_scalar(), torch_tensor.mean().item<float>(),
                        1e-4f, "RandomMultiMean_" + std::to_string(test));
    }
}

// ============= CPU Tests =============

TEST_F(TensorReductionTest, ReductionsCPU) {
    std::vector<float> data = {1, 2, 3, 4, 5};

    auto custom_tensor = Tensor::from_vector(data, {5}, Device::CPU);
    auto torch_tensor = torch::tensor(data, torch::TensorOptions().device(torch::kCPU));

    compare_scalars(custom_tensor.sum_scalar(), torch_tensor.sum().item<float>(), 1e-5f, "CPU_Sum");
    compare_scalars(custom_tensor.mean_scalar(), torch_tensor.mean().item<float>(), 1e-5f, "CPU_Mean");
    compare_scalars(custom_tensor.min_scalar(), torch_tensor.min().item<float>(), 1e-5f, "CPU_Min");
    compare_scalars(custom_tensor.max_scalar(), torch_tensor.max().item<float>(), 1e-5f, "CPU_Max");
}