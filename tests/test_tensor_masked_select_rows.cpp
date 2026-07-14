/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include <gtest/gtest.h>
#include <torch/torch.h>

using namespace lfs::core;

namespace {

    void compare_tensors(const Tensor& custom, const torch::Tensor& reference,
                         const float rtol = 1e-5f, const float atol = 1e-7f,
                         const std::string& msg = "") {
        const auto ref_cpu = reference.cpu().contiguous();
        const auto custom_cpu = custom.cpu();

        ASSERT_EQ(custom_cpu.ndim(), static_cast<size_t>(ref_cpu.dim())) << msg << ": Rank mismatch";
        for (size_t i = 0; i < custom_cpu.ndim(); ++i) {
            ASSERT_EQ(custom_cpu.size(i), static_cast<size_t>(ref_cpu.size(i)))
                << msg << ": Shape mismatch at dim " << i;
        }

        const auto custom_vec = custom_cpu.to_vector();
        const auto* ref_ptr = ref_cpu.data_ptr<float>();
        for (size_t i = 0; i < custom_vec.size(); ++i) {
            const float diff = std::abs(custom_vec[i] - ref_ptr[i]);
            const float threshold = atol + rtol * std::abs(ref_ptr[i]);
            EXPECT_LE(diff, threshold) << msg << ": Mismatch at index " << i;
        }
    }

} // namespace

class MaskedSelectRowsTest : public ::testing::Test {
protected:
    void SetUp() override {
        Tensor::manual_seed(42);
        torch::manual_seed(42);
    }
};

TEST_F(MaskedSelectRowsTest, SelectRowsFrom2DTensor_CPU) {
    const std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    const std::vector<bool> mask_data = {true, false, true, false, true};

    const auto tensor = Tensor::from_vector(data, {5, 3}, Device::CPU);
    const auto mask = Tensor::from_vector(mask_data, {5}, Device::CPU);
    const auto result = tensor.index_select(0, mask);

    const auto torch_tensor = torch::tensor(data, torch::kFloat32).reshape({5, 3});
    const auto torch_mask = torch::tensor({true, false, true, false, true});
    const auto torch_result = torch_tensor.index({torch_mask});

    EXPECT_EQ(result.size(0), 3);
    EXPECT_EQ(result.size(1), 3);
    compare_tensors(result, torch_result, 1e-5f, 1e-7f, "CPU");
}

TEST_F(MaskedSelectRowsTest, SelectRowsFrom2DTensor_CUDA) {
    const std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    const std::vector<bool> mask_data = {true, false, true, false, true};

    const auto tensor = Tensor::from_vector(data, {5, 3}, Device::CUDA);
    const auto mask = Tensor::from_vector(mask_data, {5}, Device::CUDA);
    const auto result = tensor.index_select(0, mask);

    const auto torch_tensor = torch::tensor(data, torch::kCUDA).reshape({5, 3});
    const auto torch_mask = torch::tensor({true, false, true, false, true}, torch::kCUDA);
    const auto torch_result = torch_tensor.index({torch_mask});

    EXPECT_EQ(result.size(0), 3);
    EXPECT_EQ(result.size(1), 3);
    compare_tensors(result, torch_result, 1e-5f, 1e-7f, "CUDA");
}

TEST_F(MaskedSelectRowsTest, SelectNoRows) {
    const std::vector<float> data = {1, 2, 3, 4, 5, 6};
    const auto tensor = Tensor::from_vector(data, {2, 3}, Device::CUDA);
    const auto mask = Tensor::from_vector(std::vector<bool>{false, false}, {2}, Device::CUDA);
    const auto result = tensor.index_select(0, mask);

    EXPECT_EQ(result.size(0), 0);
    EXPECT_EQ(result.size(1), 3);
}

TEST_F(MaskedSelectRowsTest, SelectAllRows) {
    const std::vector<float> data = {1, 2, 3, 4, 5, 6};
    const auto tensor = Tensor::from_vector(data, {2, 3}, Device::CUDA);
    const auto mask = Tensor::from_vector(std::vector<bool>{true, true}, {2}, Device::CUDA);
    const auto result = tensor.index_select(0, mask);

    EXPECT_EQ(result.size(0), 2);
    EXPECT_EQ(result.size(1), 3);
}

TEST_F(MaskedSelectRowsTest, SplatDataCropping_CUDA) {
    constexpr size_t N = 2000000;

    const auto means = Tensor::randn({N, 3}, Device::CUDA);
    const auto scaling = Tensor::randn({N, 3}, Device::CUDA);
    const auto rotation = Tensor::randn({N, 4}, Device::CUDA);
    const auto sh0 = Tensor::randn({N, 1, 3}, Device::CUDA);

    auto deleted = Tensor::rand({N}, Device::CUDA) < 0.04f;
    const auto visible_mask = deleted.logical_not();
    const size_t expected = visible_mask.to(DataType::Int32).sum().item<int>();

    EXPECT_EQ(means.index_select(0, visible_mask).size(0), expected);
    EXPECT_EQ(scaling.index_select(0, visible_mask).size(0), expected);
    EXPECT_EQ(rotation.index_select(0, visible_mask).size(0), expected);
    EXPECT_EQ(sh0.index_select(0, visible_mask).size(0), expected);
}

TEST_F(MaskedSelectRowsTest, SelectRowsFrom3DTensor_CUDA) {
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), 1.0f);

    const auto tensor = Tensor::from_vector(data, {4, 2, 3}, Device::CUDA);
    const auto mask = Tensor::from_vector(std::vector<bool>{true, false, true, false}, {4}, Device::CUDA);
    const auto result = tensor.index_select(0, mask);

    const auto torch_tensor = torch::tensor(data, torch::kCUDA).reshape({4, 2, 3});
    const auto torch_mask = torch::tensor(std::vector<int>{1, 0, 1, 0}, torch::kCUDA).to(torch::kBool);

    EXPECT_EQ(result.size(0), 2);
    EXPECT_EQ(result.size(1), 2);
    EXPECT_EQ(result.size(2), 3);
    compare_tensors(result, torch_tensor.index({torch_mask}), 1e-5f, 1e-7f, "3D");
}

TEST_F(MaskedSelectRowsTest, SelectRowsInt32_CPU) {
    const std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    const auto tensor = Tensor::from_vector(data, {3, 3}, Device::CPU);
    const auto mask = Tensor::from_vector(std::vector<bool>{true, false, true}, {3}, Device::CPU);
    const auto result = tensor.index_select(0, mask);

    EXPECT_EQ(result.dtype(), DataType::Int32);
    const auto vec = result.to_vector_int();
    EXPECT_EQ(vec[0], 1);
    EXPECT_EQ(vec[3], 7);
}

TEST_F(MaskedSelectRowsTest, SelectRowsInt32_CUDA) {
    const std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    const auto tensor = Tensor::from_vector(data, {3, 3}, Device::CUDA);
    const auto mask = Tensor::from_vector(std::vector<bool>{true, false, true}, {3}, Device::CUDA);
    const auto result = tensor.index_select(0, mask);

    EXPECT_EQ(result.dtype(), DataType::Int32);
    const auto vec = result.cpu().to_vector_int();
    EXPECT_EQ(vec[0], 1);
    EXPECT_EQ(vec[3], 7);
}

TEST_F(MaskedSelectRowsTest, CPUCUDAConsistency_Float32) {
    const std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    const std::vector<bool> mask_data = {true, false, true, false};

    const auto result_cpu = Tensor::from_vector(data, {4, 4}, Device::CPU)
                                .index_select(0, Tensor::from_vector(mask_data, {4}, Device::CPU));
    const auto result_cuda = Tensor::from_vector(data, {4, 4}, Device::CUDA)
                                 .index_select(0, Tensor::from_vector(mask_data, {4}, Device::CUDA));

    const auto cpu_vec = result_cpu.to_vector();
    const auto cuda_vec = result_cuda.cpu().to_vector();
    for (size_t i = 0; i < cpu_vec.size(); ++i) {
        EXPECT_FLOAT_EQ(cpu_vec[i], cuda_vec[i]);
    }
}

TEST_F(MaskedSelectRowsTest, CPUCUDAConsistency_Int32) {
    const std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const std::vector<bool> mask_data = {true, true, false, true};

    const auto result_cpu = Tensor::from_vector(data, {4, 3}, Device::CPU)
                                .index_select(0, Tensor::from_vector(mask_data, {4}, Device::CPU));
    const auto result_cuda = Tensor::from_vector(data, {4, 3}, Device::CUDA)
                                 .index_select(0, Tensor::from_vector(mask_data, {4}, Device::CUDA));

    const auto cpu_vec = result_cpu.to_vector_int();
    const auto cuda_vec = result_cuda.cpu().to_vector_int();
    for (size_t i = 0; i < cpu_vec.size(); ++i) {
        EXPECT_EQ(cpu_vec[i], cuda_vec[i]);
    }
}

TEST_F(MaskedSelectRowsTest, RandomMaskStress_CUDA) {
    constexpr size_t N = 100000;
    constexpr size_t M = 10;

    const auto tensor = Tensor::randn({N, M}, Device::CUDA);
    const auto mask = Tensor::rand({N}, Device::CUDA) < 0.3f;
    const auto result = tensor.index_select(0, mask);

    const size_t expected = mask.to(DataType::Int32).sum().item<int>();
    EXPECT_EQ(result.size(0), expected);
    EXPECT_EQ(result.size(1), M);
}

TEST_F(MaskedSelectRowsTest, CompareWithPyTorch_LargeScale) {
    constexpr size_t N = 50000;
    constexpr size_t M = 4;

    std::vector<float> data(N * M);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i % 1000) * 0.01f;
    }

    std::vector<bool> mask_data(N);
    for (size_t i = 0; i < N; ++i) {
        mask_data[i] = (i % 5 != 0);
    }

    const auto tensor = Tensor::from_vector(data, {N, M}, Device::CUDA);
    const auto mask = Tensor::from_vector(mask_data, {N}, Device::CUDA);

    const auto torch_tensor = torch::tensor(data, torch::kCUDA).reshape({static_cast<long>(N), static_cast<long>(M)});
    std::vector<int64_t> torch_mask_data(N);
    for (size_t i = 0; i < N; ++i) {
        torch_mask_data[i] = mask_data[i] ? 1 : 0;
    }
    const auto torch_mask = torch::tensor(torch_mask_data, torch::kCUDA).to(torch::kBool);

    compare_tensors(tensor.index_select(0, mask), torch_tensor.index({torch_mask}), 1e-5f, 1e-6f, "LargeScale");
}

// Tests for operator[] with boolean mask (PyTorch-style indexing)
TEST_F(MaskedSelectRowsTest, OperatorBracket_2D_CPU) {
    const std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const std::vector<bool> mask_data = {true, false, true, true};

    const auto tensor = Tensor::from_vector(data, {4, 3}, Device::CPU);
    const auto mask = Tensor::from_vector(mask_data, {4}, Device::CPU);

    // Use operator[] like PyTorch: tensor[mask]
    const Tensor result = tensor[mask];

    const auto torch_tensor = torch::tensor(data, torch::kFloat32).reshape({4, 3});
    const auto torch_mask = torch::tensor({true, false, true, true});
    const auto torch_result = torch_tensor.index({torch_mask});

    EXPECT_EQ(result.size(0), 3);
    EXPECT_EQ(result.size(1), 3);
    compare_tensors(result, torch_result, 1e-5f, 1e-7f, "operator[] CPU");
}

TEST_F(MaskedSelectRowsTest, OperatorBracket_2D_CUDA) {
    const std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const std::vector<bool> mask_data = {true, false, true, true};

    const auto tensor = Tensor::from_vector(data, {4, 3}, Device::CUDA);
    const auto mask = Tensor::from_vector(mask_data, {4}, Device::CUDA);

    // Use operator[] like PyTorch: tensor[mask]
    const Tensor result = tensor[mask];

    const auto torch_tensor = torch::tensor(data, torch::kCUDA).reshape({4, 3});
    const auto torch_mask = torch::tensor({true, false, true, true}, torch::kCUDA);
    const auto torch_result = torch_tensor.index({torch_mask});

    EXPECT_EQ(result.size(0), 3);
    EXPECT_EQ(result.size(1), 3);
    compare_tensors(result, torch_result, 1e-5f, 1e-7f, "operator[] CUDA");
}

TEST_F(MaskedSelectRowsTest, OperatorBracket_3D_CUDA) {
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), 1.0f);

    const auto tensor = Tensor::from_vector(data, {4, 2, 3}, Device::CUDA);
    const auto mask = Tensor::from_vector(std::vector<bool>{false, true, true, false}, {4}, Device::CUDA);

    const Tensor result = tensor[mask];

    const auto torch_tensor = torch::tensor(data, torch::kCUDA).reshape({4, 2, 3});
    const auto torch_mask = torch::tensor(std::vector<int>{0, 1, 1, 0}, torch::kCUDA).to(torch::kBool);
    const auto torch_result = torch_tensor.index({torch_mask});

    EXPECT_EQ(result.size(0), 2);
    EXPECT_EQ(result.size(1), 2);
    EXPECT_EQ(result.size(2), 3);
    compare_tensors(result, torch_result, 1e-5f, 1e-7f, "operator[] 3D CUDA");
}

TEST_F(MaskedSelectRowsTest, OperatorBracket_LargeScale_CUDA) {
    constexpr size_t N = 100000;
    constexpr size_t M = 3;

    const auto tensor = Tensor::randn({N, M}, Device::CUDA);
    const auto mask = Tensor::rand({N}, Device::CUDA) < 0.3f;

    // Test operator[]
    const Tensor result = tensor[mask];

    const size_t expected = mask.to(DataType::Int32).sum().item<int>();
    EXPECT_EQ(result.size(0), expected);
    EXPECT_EQ(result.size(1), M);

    // Verify same result as index_select
    const auto result_direct = tensor.index_select(0, mask);
    EXPECT_EQ(result.size(0), result_direct.size(0));

    const auto vec1 = result.cpu().to_vector();
    const auto vec2 = result_direct.cpu().to_vector();
    for (size_t i = 0; i < vec1.size(); ++i) {
        EXPECT_FLOAT_EQ(vec1[i], vec2[i]);
    }
}

TEST_F(MaskedSelectRowsTest, OperatorBracket_PointCloudFilter_CUDA) {
    // Simulates the trim_distant_points use case
    constexpr size_t N = 50000;

    const auto means = Tensor::randn({N, 3}, Device::CUDA);
    const auto colors = Tensor::randn({N, 3}, Device::CUDA);

    // Create a mask keeping 95% of points
    const auto keep_mask = Tensor::rand({N}, Device::CUDA) < 0.95f;
    const size_t expected = keep_mask.to(DataType::Int32).sum().item<int>();

    // Filter using operator[]
    const Tensor filtered_means = means[keep_mask];
    const Tensor filtered_colors = colors[keep_mask];

    EXPECT_EQ(filtered_means.size(0), expected);
    EXPECT_EQ(filtered_means.size(1), 3);
    EXPECT_EQ(filtered_colors.size(0), expected);
    EXPECT_EQ(filtered_colors.size(1), 3);
}

TEST_F(MaskedSelectRowsTest, LogicalNotFiltersDeletedRows) {
    const auto rows = Tensor::from_vector(
        std::vector<float>{1.0f, 10.0f,
                           2.0f, 20.0f,
                           3.0f, 30.0f,
                           4.0f, 40.0f},
        {4, 2}, Device::CUDA);
    const auto deleted = Tensor::from_vector(
        std::vector<bool>{false, true, false, true}, {4}, Device::CUDA);

    const Tensor kept = rows[~deleted];
    EXPECT_EQ(kept.shape(), TensorShape({2, 2}));
    EXPECT_EQ(kept.cpu().to_vector(),
              (std::vector<float>{1.0f, 10.0f, 3.0f, 30.0f}));
}
