/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <torch/torch.h>

using namespace lfs::core;

namespace {

    void compare_uint8_tensors(const Tensor& lfs, const torch::Tensor& ref, const std::string& ctx = "") {
        const auto torch_cpu = ref.cpu().contiguous();
        const auto lfs_cpu = lfs.cpu();

        ASSERT_EQ(lfs_cpu.ndim(), static_cast<size_t>(torch_cpu.dim())) << ctx;
        for (size_t i = 0; i < lfs_cpu.ndim(); ++i) {
            ASSERT_EQ(lfs_cpu.size(i), static_cast<size_t>(torch_cpu.size(i))) << ctx << " dim " << i;
        }
        if (lfs_cpu.numel() == 0)
            return;

        const auto* lfs_ptr = lfs_cpu.ptr<uint8_t>();
        const auto torch_flat = torch_cpu.flatten();
        const auto acc = torch_flat.accessor<uint8_t, 1>();
        for (size_t i = 0; i < lfs_cpu.numel(); ++i) {
            EXPECT_EQ(lfs_ptr[i], acc[i]) << ctx << " at " << i;
        }
    }

    void compare_float_tensors(const Tensor& lfs, const torch::Tensor& ref, const std::string& ctx = "", float tol = 1e-5f) {
        const auto torch_cpu = ref.cpu().contiguous();
        const auto lfs_cpu = lfs.cpu();

        ASSERT_EQ(lfs_cpu.ndim(), static_cast<size_t>(torch_cpu.dim())) << ctx;
        for (size_t i = 0; i < lfs_cpu.ndim(); ++i) {
            ASSERT_EQ(lfs_cpu.size(i), static_cast<size_t>(torch_cpu.size(i))) << ctx << " dim " << i;
        }
        if (lfs_cpu.numel() == 0)
            return;

        const auto* lfs_ptr = lfs_cpu.ptr<float>();
        const auto torch_flat = torch_cpu.flatten();
        const auto acc = torch_flat.accessor<float, 1>();
        for (size_t i = 0; i < lfs_cpu.numel(); ++i) {
            EXPECT_NEAR(lfs_ptr[i], acc[i], tol) << ctx << " at " << i;
        }
    }

} // namespace

class TensorIndexSelectUInt8Test : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(Tensor::zeros({1}, Device::GPU).is_valid());
    }
};

// ============= Basic UInt8 index_select tests =============

TEST_F(TensorIndexSelectUInt8Test, Basic_1D_CUDA) {
    // Create UInt8 data
    auto data = Tensor::zeros({10}, Device::GPU, DataType::UInt8);
    auto data_cpu = data.cpu();
    auto* p = data_cpu.ptr<uint8_t>();
    for (int i = 0; i < 10; ++i)
        p[i] = static_cast<uint8_t>(i * 10);
    data = data_cpu.cuda();

    // Create indices
    auto indices = Tensor::from_vector(std::vector<int>{0, 2, 4, 6, 8}, {5}, Device::GPU);

    // Select
    auto result = data.index_select(0, indices);

    ASSERT_EQ(result.size(0), 5);
    ASSERT_EQ(result.dtype(), DataType::UInt8);

    auto result_cpu = result.cpu();
    const auto* r = result_cpu.ptr<uint8_t>();
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 20);
    EXPECT_EQ(r[2], 40);
    EXPECT_EQ(r[3], 60);
    EXPECT_EQ(r[4], 80);
}

TEST_F(TensorIndexSelectUInt8Test, Basic_2D_CUDA) {
    // Create UInt8 [4, 3] tensor (like colors)
    auto data = Tensor::zeros({4, 3}, Device::GPU, DataType::UInt8);
    auto data_cpu = data.cpu();
    auto* p = data_cpu.ptr<uint8_t>();
    for (int i = 0; i < 12; ++i)
        p[i] = static_cast<uint8_t>(i * 20);
    data = data_cpu.cuda();

    // Select rows 0 and 2
    auto indices = Tensor::from_vector(std::vector<int>{0, 2}, {2}, Device::GPU);
    auto result = data.index_select(0, indices);

    ASSERT_EQ(result.size(0), 2);
    ASSERT_EQ(result.size(1), 3);
    ASSERT_EQ(result.dtype(), DataType::UInt8);

    auto result_cpu = result.cpu();
    const auto* r = result_cpu.ptr<uint8_t>();
    // Row 0: 0, 20, 40
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 20);
    EXPECT_EQ(r[2], 40);
    // Row 2: 120, 140, 160
    EXPECT_EQ(r[3], 120);
    EXPECT_EQ(r[4], 140);
    EXPECT_EQ(r[5], 160);
}

TEST_F(TensorIndexSelectUInt8Test, Basic_CPU) {
    auto data = Tensor::zeros({6}, Device::CPU, DataType::UInt8);
    auto* p = data.ptr<uint8_t>();
    for (int i = 0; i < 6; ++i)
        p[i] = static_cast<uint8_t>(i + 100);

    auto indices = Tensor::from_vector(std::vector<int>{1, 3, 5}, {3}, Device::CPU);
    auto result = data.index_select(0, indices);

    ASSERT_EQ(result.size(0), 3);
    const auto* r = result.ptr<uint8_t>();
    EXPECT_EQ(r[0], 101);
    EXPECT_EQ(r[1], 103);
    EXPECT_EQ(r[2], 105);
}

TEST_F(TensorIndexSelectUInt8Test, EmptyIndices) {
    auto data = Tensor::zeros({10, 3}, Device::GPU, DataType::UInt8);
    auto indices = Tensor::empty({0}, Device::GPU, DataType::Int32);
    auto result = data.index_select(0, indices);

    ASSERT_EQ(result.size(0), 0);
    ASSERT_EQ(result.size(1), 3);
}

TEST_F(TensorIndexSelectUInt8Test, SelectAll) {
    auto data = Tensor::zeros({5, 3}, Device::GPU, DataType::UInt8);
    auto data_cpu = data.cpu();
    auto* p = data_cpu.ptr<uint8_t>();
    for (int i = 0; i < 15; ++i)
        p[i] = static_cast<uint8_t>(i);
    data = data_cpu.cuda();

    auto indices = Tensor::from_vector(std::vector<int>{0, 1, 2, 3, 4}, {5}, Device::GPU);
    auto result = data.index_select(0, indices);

    ASSERT_EQ(result.numel(), data.numel());

    auto result_cpu = result.cpu();
    for (int i = 0; i < 15; ++i) {
        EXPECT_EQ(result_cpu.ptr<uint8_t>()[i], static_cast<uint8_t>(i));
    }
}

// ============= LibTorch comparison tests =============

TEST_F(TensorIndexSelectUInt8Test, VsTorch_1D) {
    // Create matching data in torch and lfs
    auto t_data = torch::arange(256, torch::kUInt8).cuda();
    auto lfs_data = Tensor::zeros({256}, Device::GPU, DataType::UInt8);
    auto lfs_cpu = lfs_data.cpu();
    auto* p = lfs_cpu.ptr<uint8_t>();
    for (int i = 0; i < 256; ++i)
        p[i] = static_cast<uint8_t>(i);
    lfs_data = lfs_cpu.cuda();

    // Indices
    auto t_indices = torch::tensor({0, 50, 100, 150, 200, 255}, torch::kLong).cuda();
    auto lfs_indices = Tensor::from_vector(std::vector<int>{0, 50, 100, 150, 200, 255}, {6}, Device::GPU);

    // Select
    auto t_result = torch::index_select(t_data, 0, t_indices);
    auto lfs_result = lfs_data.index_select(0, lfs_indices);

    compare_uint8_tensors(lfs_result, t_result, "1D");
}

TEST_F(TensorIndexSelectUInt8Test, VsTorch_2D_RowSelect) {
    // [100, 3] UInt8 - typical color array
    constexpr int N = 100;
    auto t_data = torch::randint(0, 256, {N, 3}, torch::kUInt8).cuda();

    auto lfs_data = Tensor::zeros({N, 3}, Device::GPU, DataType::UInt8);
    auto t_cpu = t_data.cpu().contiguous();
    cudaMemcpy(lfs_data.data_ptr(), t_cpu.data_ptr<uint8_t>(), N * 3, cudaMemcpyHostToDevice);

    // Random indices
    std::vector<int> idx_vec = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99};
    auto t_indices = torch::tensor(std::vector<int64_t>(idx_vec.begin(), idx_vec.end()), torch::kLong).cuda();
    auto lfs_indices = Tensor::from_vector(idx_vec, {idx_vec.size()}, Device::GPU);

    auto t_result = torch::index_select(t_data, 0, t_indices);
    auto lfs_result = lfs_data.index_select(0, lfs_indices);

    compare_uint8_tensors(lfs_result, t_result, "2D row select");
}

TEST_F(TensorIndexSelectUInt8Test, VsTorch_LargeScale) {
    // [100000, 3] - typical point cloud color size
    constexpr int N = 100000;
    auto t_data = torch::randint(0, 256, {N, 3}, torch::kUInt8).cuda();

    auto lfs_data = Tensor::zeros({N, 3}, Device::GPU, DataType::UInt8);
    auto t_cpu = t_data.cpu().contiguous();
    cudaMemcpy(lfs_data.data_ptr(), t_cpu.data_ptr<uint8_t>(), N * 3, cudaMemcpyHostToDevice);

    // Select every 10th point
    std::vector<int> idx_vec;
    for (int i = 0; i < N; i += 10)
        idx_vec.push_back(i);

    auto t_indices = torch::tensor(std::vector<int64_t>(idx_vec.begin(), idx_vec.end()), torch::kLong).cuda();
    auto lfs_indices = Tensor::from_vector(idx_vec, {idx_vec.size()}, Device::GPU);

    auto t_result = torch::index_select(t_data, 0, t_indices);
    auto lfs_result = lfs_data.index_select(0, lfs_indices);

    compare_uint8_tensors(lfs_result, t_result, "large scale");
}

// ============= Point cloud cropping pipeline tests =============

TEST_F(TensorIndexSelectUInt8Test, PointCloudCroppingPipeline_Basic) {
    // Simulate point cloud cropping:
    // 1. Transform points
    // 2. Create mask using comparisons
    // 3. Get indices with nonzero
    // 4. Filter both means (float) and colors (uint8)

    constexpr size_t N = 1000;

    // Create means [N, 3] and colors [N, 3]
    auto means = Tensor::randn({N, 3}, Device::GPU);
    auto colors = Tensor::zeros({N, 3}, Device::GPU, DataType::UInt8);
    auto colors_cpu = colors.cpu();
    auto* cp = colors_cpu.ptr<uint8_t>();
    for (size_t i = 0; i < N * 3; ++i)
        cp[i] = static_cast<uint8_t>(i % 256);
    colors = colors_cpu.cuda();

    // Cropbox: keep points where x in [-0.5, 0.5]
    auto x = means.slice(1, 0, 1).squeeze(1); // [N]
    auto mask = (x >= -0.5f) && (x <= 0.5f);  // [N] Bool

    // Get indices
    auto indices = mask.nonzero().squeeze(1); // [K] Int64
    ASSERT_GT(indices.numel(), 0) << "Should have some points in cropbox";

    // Filter both
    auto filtered_means = means.index_select(0, indices);
    auto filtered_colors = colors.index_select(0, indices);

    ASSERT_EQ(filtered_means.size(0), indices.numel());
    ASSERT_EQ(filtered_colors.size(0), indices.numel());
    ASSERT_EQ(filtered_means.size(1), 3);
    ASSERT_EQ(filtered_colors.size(1), 3);
    ASSERT_EQ(filtered_colors.dtype(), DataType::UInt8);
}

TEST_F(TensorIndexSelectUInt8Test, PointCloudCroppingPipeline_WithTransform) {
    // Full pipeline with matrix transform (like scene_manager.cpp)
    constexpr size_t N = 5000;

    // Create point cloud
    auto means = Tensor::randn({N, 3}, Device::GPU) * 10.0f; // Points in [-10, 10]
    auto colors = Tensor::zeros({N, 3}, Device::GPU, DataType::UInt8);
    auto colors_cpu = colors.cpu();
    for (size_t i = 0; i < N * 3; ++i) {
        colors_cpu.ptr<uint8_t>()[i] = static_cast<uint8_t>(i % 256);
    }
    colors = colors_cpu.cuda();

    // Build transform matrix (GLM column-major -> row-major)
    glm::mat4 m = glm::inverse(glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, 0.0f, 0.0f)));
    const std::vector<float> transform_data = {
        m[0][0], m[1][0], m[2][0], m[3][0],
        m[0][1], m[1][1], m[2][1], m[3][1],
        m[0][2], m[1][2], m[2][2], m[3][2],
        m[0][3], m[1][3], m[2][3], m[3][3]};
    auto transform = Tensor::from_vector(transform_data, {4, 4}, Device::GPU);

    // Transform: [4,4] x [4,N] -> [4,N] -> [N,3]
    auto ones = Tensor::ones({N, 1}, Device::GPU);
    auto means_homo = means.cat(ones, 1);              // [N, 4]
    auto local_pos = transform.mm(means_homo.t()).t(); // [N, 4]

    // Extract xyz
    auto x = local_pos.slice(1, 0, 1).squeeze(1);
    auto y = local_pos.slice(1, 1, 2).squeeze(1);
    auto z = local_pos.slice(1, 2, 3).squeeze(1);

    // Cropbox bounds
    constexpr float MIN = -5.0f, MAX = 5.0f;
    auto mask = (x >= MIN) && (x <= MAX) &&
                (y >= MIN) && (y <= MAX) &&
                (z >= MIN) && (z <= MAX);

    auto indices = mask.nonzero().squeeze(1);
    auto filtered_means = means.index_select(0, indices);
    auto filtered_colors = colors.index_select(0, indices);

    ASSERT_EQ(filtered_means.size(0), filtered_colors.size(0));
    ASSERT_EQ(filtered_means.size(1), 3);
    ASSERT_EQ(filtered_colors.size(1), 3);
    ASSERT_EQ(filtered_colors.dtype(), DataType::UInt8);
}

TEST_F(TensorIndexSelectUInt8Test, PointCloudCroppingPipeline_VsTorch) {
    // Full comparison with LibTorch
    constexpr int64_t N = 10000;

    // Create identical data in both
    auto t_means = torch::randn({N, 3}, torch::kFloat32).cuda();
    auto t_colors = torch::randint(0, 256, {N, 3}, torch::kUInt8).cuda();

    auto lfs_means = Tensor::zeros({static_cast<size_t>(N), 3}, Device::GPU);
    auto lfs_colors = Tensor::zeros({static_cast<size_t>(N), 3}, Device::GPU, DataType::UInt8);

    auto t_means_cpu = t_means.cpu().contiguous();
    auto t_colors_cpu = t_colors.cpu().contiguous();
    cudaMemcpy(lfs_means.data_ptr(), t_means_cpu.data_ptr<float>(), N * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(lfs_colors.data_ptr(), t_colors_cpu.data_ptr<uint8_t>(), N * 3, cudaMemcpyHostToDevice);

    // Create mask: x > 0
    auto t_x = t_means.select(1, 0);
    auto t_mask = t_x > 0;
    auto t_indices = torch::nonzero(t_mask).flatten().to(torch::kLong);

    auto lfs_x = lfs_means.slice(1, 0, 1).squeeze(1);
    auto lfs_mask = lfs_x > 0.0f;
    auto lfs_indices = lfs_mask.nonzero().squeeze(1);

    // Filter
    auto t_filtered_means = torch::index_select(t_means, 0, t_indices);
    auto t_filtered_colors = torch::index_select(t_colors, 0, t_indices);

    auto lfs_filtered_means = lfs_means.index_select(0, lfs_indices);
    auto lfs_filtered_colors = lfs_colors.index_select(0, lfs_indices);

    compare_float_tensors(lfs_filtered_means, t_filtered_means, "filtered means");
    compare_uint8_tensors(lfs_filtered_colors, t_filtered_colors, "filtered colors");
}

// ============= Edge cases =============

TEST_F(TensorIndexSelectUInt8Test, SingleElement) {
    auto data = Tensor::zeros({1, 3}, Device::GPU, DataType::UInt8);
    auto data_cpu = data.cpu();
    data_cpu.ptr<uint8_t>()[0] = 100;
    data_cpu.ptr<uint8_t>()[1] = 150;
    data_cpu.ptr<uint8_t>()[2] = 200;
    data = data_cpu.cuda();

    auto indices = Tensor::from_vector(std::vector<int>{0}, {1}, Device::GPU);
    auto result = data.index_select(0, indices);

    ASSERT_EQ(result.size(0), 1);
    ASSERT_EQ(result.size(1), 3);

    auto r_cpu = result.cpu();
    EXPECT_EQ(r_cpu.ptr<uint8_t>()[0], 100);
    EXPECT_EQ(r_cpu.ptr<uint8_t>()[1], 150);
    EXPECT_EQ(r_cpu.ptr<uint8_t>()[2], 200);
}

TEST_F(TensorIndexSelectUInt8Test, DuplicateIndices) {
    auto data = Tensor::zeros({3, 2}, Device::GPU, DataType::UInt8);
    auto data_cpu = data.cpu();
    for (int i = 0; i < 6; ++i)
        data_cpu.ptr<uint8_t>()[i] = static_cast<uint8_t>(i * 40);
    data = data_cpu.cuda();

    // Select row 1 three times
    auto indices = Tensor::from_vector(std::vector<int>{1, 1, 1}, {3}, Device::GPU);
    auto result = data.index_select(0, indices);

    ASSERT_EQ(result.size(0), 3);
    ASSERT_EQ(result.size(1), 2);

    auto r_cpu = result.cpu();
    // Row 1 is [80, 120]
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(r_cpu.ptr<uint8_t>()[i * 2], 80);
        EXPECT_EQ(r_cpu.ptr<uint8_t>()[i * 2 + 1], 120);
    }
}

TEST_F(TensorIndexSelectUInt8Test, ReversedIndices) {
    auto data = Tensor::zeros({5}, Device::GPU, DataType::UInt8);
    auto data_cpu = data.cpu();
    for (int i = 0; i < 5; ++i)
        data_cpu.ptr<uint8_t>()[i] = static_cast<uint8_t>(i);
    data = data_cpu.cuda();

    auto indices = Tensor::from_vector(std::vector<int>{4, 3, 2, 1, 0}, {5}, Device::GPU);
    auto result = data.index_select(0, indices);

    auto r_cpu = result.cpu();
    EXPECT_EQ(r_cpu.ptr<uint8_t>()[0], 4);
    EXPECT_EQ(r_cpu.ptr<uint8_t>()[1], 3);
    EXPECT_EQ(r_cpu.ptr<uint8_t>()[2], 2);
    EXPECT_EQ(r_cpu.ptr<uint8_t>()[3], 1);
    EXPECT_EQ(r_cpu.ptr<uint8_t>()[4], 0);
}
