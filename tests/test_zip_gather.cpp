/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using namespace lfs::core;

namespace {

    void expect_values(const Tensor& tensor, const std::vector<float>& expected) {
        const auto actual = tensor.cpu().to_vector();
        ASSERT_EQ(actual.size(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_FLOAT_EQ(actual[i], expected[i]) << "index " << i;
        }
    }

    class ZipGatherTest : public ::testing::Test {
    protected:
        void SetUp() override {
            int device_count = 0;
            if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
                GTEST_SKIP() << "CUDA device required";
            }
        }
    };

} // namespace

TEST_F(ZipGatherTest, TwoInputsHonorIndependentStridesAndClampIndices) {
    auto input1 = Tensor::from_vector(
        std::vector<float>{10.0f, 11.0f, 20.0f, 21.0f, 30.0f, 31.0f},
        {6}, Device::CUDA);
    auto input2 = Tensor::from_vector(
        std::vector<float>{100.0f, 101.0f, 102.0f,
                           200.0f, 201.0f, 202.0f,
                           300.0f, 301.0f, 302.0f},
        {9}, Device::CUDA);
    auto indices = Tensor::from_vector(std::vector<int>{-1, 0, 2, 7}, {4}, Device::CUDA);
    auto output1 = Tensor::empty({4}, Device::CUDA);
    auto output2 = Tensor::empty({4}, Device::CUDA);

    tensor_ops::launch_zip_gather_2(
        input1.ptr<float>(), input2.ptr<float>(), indices.ptr<int>(),
        output1.ptr<float>(), output2.ptr<float>(), 3, 4, 2, 3);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    expect_values(output1, {30.0f, 10.0f, 30.0f, 30.0f});
    expect_values(output2, {300.0f, 100.0f, 300.0f, 300.0f});
}

TEST_F(ZipGatherTest, ThreeInputsHonorIndependentStrides) {
    auto input1 = Tensor::from_vector(
        std::vector<float>{10.0f, 20.0f, 30.0f}, {3}, Device::CUDA);
    auto input2 = Tensor::from_vector(
        std::vector<float>{100.0f, 101.0f, 200.0f, 201.0f, 300.0f, 301.0f},
        {6}, Device::CUDA);
    auto input3 = Tensor::from_vector(
        std::vector<float>{1000.0f, 1001.0f, 1002.0f, 1003.0f,
                           2000.0f, 2001.0f, 2002.0f, 2003.0f,
                           3000.0f, 3001.0f, 3002.0f, 3003.0f},
        {12}, Device::CUDA);
    auto indices = Tensor::from_vector(std::vector<int>{1, 0, 2}, {3}, Device::CUDA);
    auto output1 = Tensor::empty({3}, Device::CUDA);
    auto output2 = Tensor::empty({3}, Device::CUDA);
    auto output3 = Tensor::empty({3}, Device::CUDA);

    tensor_ops::launch_zip_gather_3(
        input1.ptr<float>(), input2.ptr<float>(), input3.ptr<float>(), indices.ptr<int>(),
        output1.ptr<float>(), output2.ptr<float>(), output3.ptr<float>(),
        3, 3, 1, 2, 4);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    expect_values(output1, {20.0f, 10.0f, 30.0f});
    expect_values(output2, {200.0f, 100.0f, 300.0f});
    expect_values(output3, {2000.0f, 1000.0f, 3000.0f});
}

TEST_F(ZipGatherTest, EmptyInputLeavesOutputsUntouched) {
    auto input = Tensor::empty({0}, Device::CUDA);
    auto indices = Tensor::from_vector(std::vector<int>{0, 1}, {2}, Device::CUDA);
    auto output1 = Tensor::full({2}, -7.0f, Device::CUDA);
    auto output2 = Tensor::full({2}, -9.0f, Device::CUDA);

    tensor_ops::launch_zip_gather_2(
        input.ptr<float>(), input.ptr<float>(), indices.ptr<int>(),
        output1.ptr<float>(), output2.ptr<float>(), 0, 2, 1, 1);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    expect_values(output1, {-7.0f, -7.0f});
    expect_values(output2, {-9.0f, -9.0f});
}
