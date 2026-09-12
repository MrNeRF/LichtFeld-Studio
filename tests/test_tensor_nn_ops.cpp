/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cmath>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "core/tensor.hpp"

namespace {

    constexpr float RTOL = 1e-4f;
    constexpr float ATOL = 1e-5f;

    lfs::core::Tensor torch_to_lfs(const torch::Tensor& t) {
        auto cpu = t.cpu().contiguous();
        std::vector<size_t> shape;
        for (int i = 0; i < t.dim(); ++i) {
            shape.push_back(t.size(i));
        }
        std::vector<float> data(cpu.data_ptr<float>(), cpu.data_ptr<float>() + cpu.numel());
        return lfs::core::Tensor::from_vector(data, lfs::core::TensorShape(shape), lfs::core::Device::GPU);
    }

    bool tensors_close(const torch::Tensor& expected, const lfs::core::Tensor& actual,
                       float rtol = RTOL, float atol = ATOL) {
        auto exp_cpu = expected.cpu().contiguous();
        auto act_cpu = actual.cpu();

        if (exp_cpu.dim() != static_cast<int64_t>(act_cpu.ndim())) {
            std::cerr << "Dimension mismatch: expected " << exp_cpu.dim()
                      << " got " << act_cpu.ndim() << std::endl;
            return false;
        }

        for (int i = 0; i < exp_cpu.dim(); ++i) {
            if (exp_cpu.size(i) != static_cast<int64_t>(act_cpu.shape()[i])) {
                std::cerr << "Shape mismatch at dim " << i << ": expected "
                          << exp_cpu.size(i) << " got " << act_cpu.shape()[i] << std::endl;
                return false;
            }
        }

        auto exp_data = exp_cpu.data_ptr<float>();
        auto act_data = act_cpu.to_vector();

        int mismatches = 0;
        for (int64_t i = 0; i < exp_cpu.numel(); ++i) {
            float e = exp_data[i];
            float a = act_data[i];
            float tol = atol + rtol * std::abs(e);
            if (std::abs(e - a) > tol) {
                if (mismatches < 5) {
                    std::cerr << "Mismatch at " << i << ": expected " << e << " got " << a
                              << " (diff=" << std::abs(e - a) << ")" << std::endl;
                }
                ++mismatches;
            }
        }

        if (mismatches > 0) {
            std::cerr << "Total mismatches: " << mismatches << " / " << exp_cpu.numel() << std::endl;
            return false;
        }
        return true;
    }

    class TensorNNOpsTest : public ::testing::Test {
    protected:
        void SetUp() override {
            torch::manual_seed(42);
        }
    };

    // ============= Conv1x1 Tests =============

    TEST_F(TensorNNOpsTest, Conv1x1Basic) {
        auto input_torch = torch::randn({1, 3, 32, 32});
        auto weight_torch = torch::randn({16, 3});

        auto input = torch_to_lfs(input_torch);
        auto weight = torch_to_lfs(weight_torch);

        auto output = input.conv1x1(weight);

        // PyTorch equivalent: reshape [1,3,32,32] -> [1024,3], matmul with [3,16], reshape back
        auto input_nhwc = input_torch.permute({0, 2, 3, 1}).contiguous();
        auto input_2d = input_nhwc.reshape({1 * 32 * 32, 3});
        auto output_2d = torch::matmul(input_2d, weight_torch.t());
        auto output_nhwc = output_2d.reshape({1, 32, 32, 16});
        auto expected = output_nhwc.permute({0, 3, 1, 2}).contiguous();

        EXPECT_EQ(output.shape()[0], 1);
        EXPECT_EQ(output.shape()[1], 16);
        EXPECT_EQ(output.shape()[2], 32);
        EXPECT_EQ(output.shape()[3], 32);
        EXPECT_TRUE(tensors_close(expected, output));
    }

    TEST_F(TensorNNOpsTest, Conv1x1Batched) {
        auto input_torch = torch::randn({4, 64, 64, 64});
        auto weight_torch = torch::randn({128, 64});

        auto input = torch_to_lfs(input_torch);
        auto weight = torch_to_lfs(weight_torch);

        auto output = input.conv1x1(weight);

        auto input_nhwc = input_torch.permute({0, 2, 3, 1}).contiguous();
        auto input_2d = input_nhwc.reshape({4 * 64 * 64, 64});
        auto output_2d = torch::matmul(input_2d, weight_torch.t());
        auto output_nhwc = output_2d.reshape({4, 64, 64, 128});
        auto expected = output_nhwc.permute({0, 3, 1, 2}).contiguous();

        EXPECT_EQ(output.shape()[0], 4);
        EXPECT_EQ(output.shape()[1], 128);
        EXPECT_TRUE(tensors_close(expected, output));
    }

    TEST_F(TensorNNOpsTest, Conv1x1WithBias) {
        auto input_torch = torch::randn({2, 3, 16, 16});
        auto weight_torch = torch::randn({8, 3});
        auto bias_torch = torch::randn({8});

        auto input = torch_to_lfs(input_torch);
        auto weight = torch_to_lfs(weight_torch);
        auto bias = torch_to_lfs(bias_torch);

        auto output = input.conv1x1(weight, bias);

        auto input_nhwc = input_torch.permute({0, 2, 3, 1}).contiguous();
        auto input_2d = input_nhwc.reshape({2 * 16 * 16, 3});
        auto output_2d = torch::matmul(input_2d, weight_torch.t()) + bias_torch;
        auto output_nhwc = output_2d.reshape({2, 16, 16, 8});
        auto expected = output_nhwc.permute({0, 3, 1, 2}).contiguous();

        EXPECT_TRUE(tensors_close(expected, output));
    }

    // ============= MaxPool2d Tests =============

    TEST_F(TensorNNOpsTest, MaxPool2dBasic) {
        auto input_torch = torch::randn({1, 64, 32, 32});
        auto input = torch_to_lfs(input_torch);

        auto output = input.max_pool2d(2, 2);

        auto expected = torch::max_pool2d(input_torch, {2, 2}, {2, 2});

        EXPECT_EQ(output.shape()[0], 1);
        EXPECT_EQ(output.shape()[1], 64);
        EXPECT_EQ(output.shape()[2], 16);
        EXPECT_EQ(output.shape()[3], 16);
        EXPECT_TRUE(tensors_close(expected, output));
    }

    TEST_F(TensorNNOpsTest, MaxPool2dStride3) {
        auto input_torch = torch::randn({1, 16, 99, 99});
        auto input = torch_to_lfs(input_torch);

        auto output = input.max_pool2d(3, 3);

        auto expected = torch::max_pool2d(input_torch, {3, 3}, {3, 3});

        EXPECT_EQ(output.shape()[2], 33);
        EXPECT_EQ(output.shape()[3], 33);
        EXPECT_TRUE(tensors_close(expected, output));
    }

    TEST_F(TensorNNOpsTest, MaxPool2dPadding) {
        auto input_torch = torch::randn({2, 32, 28, 28});
        auto input = torch_to_lfs(input_torch);

        auto output = input.max_pool2d(3, 2, 1);

        auto expected = torch::max_pool2d(input_torch, {3, 3}, {2, 2}, {1, 1});

        EXPECT_TRUE(tensors_close(expected, output));
    }

    TEST_F(TensorNNOpsTest, MaxPool2dBatched) {
        auto input_torch = torch::randn({8, 16, 64, 64});
        auto input = torch_to_lfs(input_torch);

        auto output = input.max_pool2d(2, 2);

        auto expected = torch::max_pool2d(input_torch, {2, 2}, {2, 2});

        EXPECT_EQ(output.shape()[0], 8);
        EXPECT_EQ(output.shape()[2], 32);
        EXPECT_TRUE(tensors_close(expected, output));
    }

    // ============= AdaptiveAvgPool2d Tests =============

    TEST_F(TensorNNOpsTest, AdaptiveAvgPool5x5) {
        auto input_torch = torch::randn({1, 64, 33, 33});
        auto input = torch_to_lfs(input_torch);

        auto output = input.adaptive_avg_pool2d(5, 5);

        auto expected = torch::adaptive_avg_pool2d(input_torch, {5, 5});

        EXPECT_EQ(output.shape()[0], 1);
        EXPECT_EQ(output.shape()[1], 64);
        EXPECT_EQ(output.shape()[2], 5);
        EXPECT_EQ(output.shape()[3], 5);
        EXPECT_TRUE(tensors_close(expected, output));
    }

    TEST_F(TensorNNOpsTest, AdaptiveAvgPool1x1) {
        auto input_torch = torch::randn({4, 128, 16, 16});
        auto input = torch_to_lfs(input_torch);

        auto output = input.adaptive_avg_pool2d(1, 1);

        auto expected = torch::adaptive_avg_pool2d(input_torch, {1, 1});

        EXPECT_EQ(output.shape()[2], 1);
        EXPECT_EQ(output.shape()[3], 1);
        EXPECT_TRUE(tensors_close(expected, output));
    }

    TEST_F(TensorNNOpsTest, AdaptiveAvgPoolVaryingSizes) {
        auto input_torch = torch::randn({2, 32, 17, 23});
        auto input = torch_to_lfs(input_torch);

        auto output = input.adaptive_avg_pool2d(5, 5);

        auto expected = torch::adaptive_avg_pool2d(input_torch, {5, 5});

        EXPECT_TRUE(tensors_close(expected, output));
    }

    // ============= Linear Tests =============

    TEST_F(TensorNNOpsTest, LinearBasic) {
        auto input_torch = torch::randn({32, 128});
        auto weight_torch = torch::randn({256, 128});
        auto bias_torch = torch::randn({256});

        auto input = torch_to_lfs(input_torch);
        auto weight = torch_to_lfs(weight_torch);
        auto bias = torch_to_lfs(bias_torch);

        auto output = input.linear(weight, bias);

        auto expected = torch::linear(input_torch, weight_torch, bias_torch);

        EXPECT_EQ(output.shape()[0], 32);
        EXPECT_EQ(output.shape()[1], 256);
        EXPECT_TRUE(tensors_close(expected, output));
    }

    TEST_F(TensorNNOpsTest, LinearNoBias) {
        auto input_torch = torch::randn({16, 64});
        auto weight_torch = torch::randn({128, 64});

        auto input = torch_to_lfs(input_torch);
        auto weight = torch_to_lfs(weight_torch);

        auto output = input.linear(weight);

        auto expected = torch::linear(input_torch, weight_torch);

        EXPECT_TRUE(tensors_close(expected, output));
    }

    TEST_F(TensorNNOpsTest, Linear3D) {
        auto input_torch = torch::randn({4, 8, 64});
        auto weight_torch = torch::randn({32, 64});
        auto bias_torch = torch::randn({32});

        auto input = torch_to_lfs(input_torch);
        auto weight = torch_to_lfs(weight_torch);
        auto bias = torch_to_lfs(bias_torch);

        auto output = input.linear(weight, bias);

        auto expected = torch::linear(input_torch, weight_torch, bias_torch);

        EXPECT_EQ(output.shape()[0], 4);
        EXPECT_EQ(output.shape()[1], 8);
        EXPECT_EQ(output.shape()[2], 32);
        EXPECT_TRUE(tensors_close(expected, output));
    }

    // ============= Integration: PPISP Controller Forward Pass =============

    TEST_F(TensorNNOpsTest, ControllerForward) {
        // Simulate PPISP controller architecture:
        // Input [N,3,H,W] -> Conv1x1 3->16 + ReLU -> MaxPool k=3,s=3
        // -> Conv1x1 16->32 + ReLU -> Conv1x1 32->64 + ReLU
        // -> AdaptiveAvgPool(5,5) -> Flatten -> Linear 1600->128 + ReLU
        // -> Linear 128->128 + ReLU -> Linear 128->9

        const int N = 2;
        const int H = 99, W = 99;

        auto input_torch = torch::randn({N, 3, H, W});
        auto input = torch_to_lfs(input_torch);

        // Layer weights
        auto w1_torch = torch::randn({16, 3});
        auto b1_torch = torch::randn({16});
        auto w2_torch = torch::randn({32, 16});
        auto b2_torch = torch::randn({32});
        auto w3_torch = torch::randn({64, 32});
        auto b3_torch = torch::randn({64});
        auto w4_torch = torch::randn({128, 1600});
        auto b4_torch = torch::randn({128});
        auto w5_torch = torch::randn({128, 128});
        auto b5_torch = torch::randn({128});
        auto w6_torch = torch::randn({9, 128});
        auto b6_torch = torch::randn({9});

        auto w1 = torch_to_lfs(w1_torch);
        auto b1 = torch_to_lfs(b1_torch);
        auto w2 = torch_to_lfs(w2_torch);
        auto b2 = torch_to_lfs(b2_torch);
        auto w3 = torch_to_lfs(w3_torch);
        auto b3 = torch_to_lfs(b3_torch);
        auto w4 = torch_to_lfs(w4_torch);
        auto b4 = torch_to_lfs(b4_torch);
        auto w5 = torch_to_lfs(w5_torch);
        auto b5 = torch_to_lfs(b5_torch);
        auto w6 = torch_to_lfs(w6_torch);
        auto b6 = torch_to_lfs(b6_torch);

        // LFS forward pass
        auto x = input.conv1x1(w1, b1).relu();
        x = x.max_pool2d(3, 3);
        x = x.conv1x1(w2, b2).relu();
        x = x.conv1x1(w3, b3).relu();
        x = x.adaptive_avg_pool2d(5, 5);
        x = x.flatten(1);
        x = x.linear(w4, b4).relu();
        x = x.linear(w5, b5).relu();
        auto lfs_output = x.linear(w6, b6);

        // PyTorch reference forward pass
        auto y_torch = input_torch;

        // Conv1x1 + ReLU
        auto y_nhwc = y_torch.permute({0, 2, 3, 1}).contiguous();
        auto y_2d = y_nhwc.reshape({N * H * W, 3});
        y_2d = torch::relu(torch::matmul(y_2d, w1_torch.t()) + b1_torch);
        y_torch = y_2d.reshape({N, H, W, 16}).permute({0, 3, 1, 2}).contiguous();

        // MaxPool
        y_torch = torch::max_pool2d(y_torch, {3, 3}, {3, 3});
        int H2 = y_torch.size(2), W2 = y_torch.size(3);

        // Conv1x1 16->32 + ReLU
        y_nhwc = y_torch.permute({0, 2, 3, 1}).contiguous();
        y_2d = y_nhwc.reshape({N * H2 * W2, 16});
        y_2d = torch::relu(torch::matmul(y_2d, w2_torch.t()) + b2_torch);
        y_torch = y_2d.reshape({N, H2, W2, 32}).permute({0, 3, 1, 2}).contiguous();

        // Conv1x1 32->64 + ReLU
        y_nhwc = y_torch.permute({0, 2, 3, 1}).contiguous();
        y_2d = y_nhwc.reshape({N * H2 * W2, 32});
        y_2d = torch::relu(torch::matmul(y_2d, w3_torch.t()) + b3_torch);
        y_torch = y_2d.reshape({N, H2, W2, 64}).permute({0, 3, 1, 2}).contiguous();

        // AdaptiveAvgPool
        y_torch = torch::adaptive_avg_pool2d(y_torch, {5, 5});

        // Flatten
        y_torch = y_torch.flatten(1);

        // Linear layers
        y_torch = torch::relu(torch::linear(y_torch, w4_torch, b4_torch));
        y_torch = torch::relu(torch::linear(y_torch, w5_torch, b5_torch));
        auto expected = torch::linear(y_torch, w6_torch, b6_torch);

        EXPECT_EQ(lfs_output.shape()[0], N);
        EXPECT_EQ(lfs_output.shape()[1], 9);
        EXPECT_TRUE(tensors_close(expected, lfs_output, 1e-3f, 1e-4f));
    }

    // ============= CPU Tests =============

    TEST_F(TensorNNOpsTest, MaxPool2dCPU) {
        auto input_torch = torch::randn({1, 16, 32, 32});

        std::vector<size_t> shape = {1, 16, 32, 32};
        auto input_cpu = input_torch.cpu().contiguous();
        std::vector<float> data(input_cpu.data_ptr<float>(),
                                input_cpu.data_ptr<float>() + input_cpu.numel());
        auto input = lfs::core::Tensor::from_vector(data, lfs::core::TensorShape(shape), lfs::core::Device::CPU);

        auto output = input.max_pool2d(2, 2);

        auto expected = torch::max_pool2d(input_torch, {2, 2}, {2, 2});

        EXPECT_EQ(output.device(), lfs::core::Device::CPU);
        EXPECT_TRUE(tensors_close(expected, output));
    }

    TEST_F(TensorNNOpsTest, AdaptiveAvgPool2dCPU) {
        auto input_torch = torch::randn({2, 8, 16, 16});

        std::vector<size_t> shape = {2, 8, 16, 16};
        auto input_cpu = input_torch.cpu().contiguous();
        std::vector<float> data(input_cpu.data_ptr<float>(),
                                input_cpu.data_ptr<float>() + input_cpu.numel());
        auto input = lfs::core::Tensor::from_vector(data, lfs::core::TensorShape(shape), lfs::core::Device::CPU);

        auto output = input.adaptive_avg_pool2d(4, 4);

        auto expected = torch::adaptive_avg_pool2d(input_torch, {4, 4});

        EXPECT_EQ(output.device(), lfs::core::Device::CPU);
        EXPECT_TRUE(tensors_close(expected, output));
    }

} // namespace
