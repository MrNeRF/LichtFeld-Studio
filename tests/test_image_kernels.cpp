/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include "core/tensor.hpp"
#include "kernels/image_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <vector>

using namespace lfs::core;
using namespace lfs::training::kernels;

class ImageKernelsTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA device available";
        }
    }
};

TEST_F(ImageKernelsTest, FusedCannyUInt8MatchesNormalizedFloatInput) {
    constexpr int C = 3;
    constexpr int H = 40;
    constexpr int W = 37;

    std::vector<float> normalized_data(C * H * W);
    std::vector<float> byte_data(C * H * W);
    for (int c = 0; c < C; ++c) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const int idx = (c * H + y) * W + x;
                const int byte_value = (c * 53 + y * 7 + x * 11) % 256;
                normalized_data[idx] = static_cast<float>(byte_value) * (1.0f / 255.0f);
                byte_data[idx] = static_cast<float>(byte_value);
            }
        }
    }

    auto float_input = Tensor::from_vector(normalized_data, TensorShape({C, H, W}), Device::CUDA);
    auto uint8_input = Tensor::from_vector(byte_data, TensorShape({C, H, W}), Device::CUDA)
                           .to(DataType::UInt8);
    auto float_output = Tensor::zeros({H, W}, Device::CUDA, DataType::Float32);
    auto uint8_output = Tensor::zeros({H, W}, Device::CUDA, DataType::Float32);

    launch_fused_canny_edge_filter_chw(
        float_input.ptr<float>(),
        float_output.ptr<float>(),
        H,
        W);
    auto err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    launch_fused_canny_edge_filter_chw(
        uint8_input.ptr<uint8_t>(),
        uint8_output.ptr<float>(),
        H,
        W);
    err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    err = cudaDeviceSynchronize();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    const auto float_cpu = float_output.cpu();
    const auto uint8_cpu = uint8_output.cpu();
    const float* float_ptr = float_cpu.ptr<float>();
    const float* uint8_ptr = uint8_cpu.ptr<float>();

    float max_abs_diff = 0.0f;
    for (int i = 0; i < H * W; ++i) {
        max_abs_diff = std::max(max_abs_diff, std::abs(float_ptr[i] - uint8_ptr[i]));
    }

    EXPECT_LT(max_abs_diff, 1e-5f);
}
