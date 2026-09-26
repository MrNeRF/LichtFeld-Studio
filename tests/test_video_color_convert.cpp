/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "cuda_backend_test.hpp"

#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

#include "core/tensor.hpp"
#include "io/video/color_convert.cuh"
#include <torch/torch.h>

namespace {

    constexpr int TOLERANCE = 2;

    // BT.601 reference implementation
    void rgbToYuvReference(const int r, const int g, const int b, int& y, int& u, int& v) {
        y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
        y = std::clamp(y, 0, 255);
        u = std::clamp(u, 0, 255);
        v = std::clamp(v, 0, 255);
    }

} // namespace

class VideoColorConvertTest : public lfs::test::CudaDeviceTest {};
class VideoColorConvertCudaTest : public lfs::test::CudaBackendTest {};

TEST_F(VideoColorConvertTest, TensorPermuteCHWtoHWC) {
    constexpr int C = 3;
    constexpr int H = 4;
    constexpr int W = 6;

    std::vector<float> data(C * H * W);
    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                data[c * H * W + h * W + w] = static_cast<float>(c * 100 + h * 10 + w);
            }
        }
    }

    auto lfs_chw = lfs::core::Tensor::from_vector(data, {C, H, W}, lfs::core::Device::GPU);
    auto torch_chw = torch::from_blob(data.data(), {C, H, W}, torch::kFloat32).clone().cuda();

    auto lfs_hwc = lfs_chw.permute({1, 2, 0}).contiguous();
    auto torch_hwc = torch_chw.permute({1, 2, 0}).contiguous();

    ASSERT_EQ(lfs_hwc.shape()[0], H);
    ASSERT_EQ(lfs_hwc.shape()[1], W);
    ASSERT_EQ(lfs_hwc.shape()[2], C);

    auto lfs_cpu = lfs_hwc.cpu();
    auto torch_cpu = torch_hwc.cpu();

    const float* const lfs_ptr = lfs_cpu.ptr<float>();
    const float* const torch_ptr = torch_cpu.data_ptr<float>();

    for (int i = 0; i < H * W * C; ++i) {
        EXPECT_FLOAT_EQ(lfs_ptr[i], torch_ptr[i]) << "Mismatch at " << i;
    }
}

// BT.601 YUV to RGB reference implementation
void yuvToRgbReference(const int y, const int u, const int v, int& r, int& g, int& b) {
    const int c = y - 16;
    const int d = u - 128;
    const int e = v - 128;
    r = std::clamp((298 * c + 409 * e + 128) >> 8, 0, 255);
    g = std::clamp((298 * c - 100 * d - 208 * e + 128) >> 8, 0, 255);
    b = std::clamp((298 * c + 516 * d + 128) >> 8, 0, 255);
}

TEST_F(VideoColorConvertCudaTest, Nv12ToRgbSolidRed) {
    constexpr int WIDTH = 4;
    constexpr int HEIGHT = 4;

    int y_val, u_val, v_val;
    rgbToYuvReference(255, 0, 0, y_val, u_val, v_val);

    std::vector<uint8_t> y_host(WIDTH * HEIGHT, static_cast<uint8_t>(y_val));
    std::vector<uint8_t> uv_host((HEIGHT / 2) * WIDTH);
    for (int i = 0; i < (HEIGHT / 2) * (WIDTH / 2); ++i) {
        uv_host[i * 2] = static_cast<uint8_t>(u_val);
        uv_host[i * 2 + 1] = static_cast<uint8_t>(v_val);
    }

    uint8_t* y_gpu = nullptr;
    uint8_t* uv_gpu = nullptr;
    uint8_t* rgb_gpu = nullptr;

    cudaMalloc(&y_gpu, WIDTH * HEIGHT);
    cudaMalloc(&uv_gpu, (HEIGHT / 2) * WIDTH);
    cudaMalloc(&rgb_gpu, WIDTH * HEIGHT * 3);

    cudaMemcpy(y_gpu, y_host.data(), WIDTH * HEIGHT, cudaMemcpyHostToDevice);
    cudaMemcpy(uv_gpu, uv_host.data(), (HEIGHT / 2) * WIDTH, cudaMemcpyHostToDevice);

    lfs::io::video::nv12ToRgbCuda(y_gpu, uv_gpu, rgb_gpu, WIDTH, HEIGHT, 0, 0, nullptr);
    cudaDeviceSynchronize();

    std::vector<uint8_t> rgb_host(WIDTH * HEIGHT * 3);
    cudaMemcpy(rgb_host.data(), rgb_gpu, WIDTH * HEIGHT * 3, cudaMemcpyDeviceToHost);

    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
        EXPECT_NEAR(rgb_host[i * 3], 255, TOLERANCE + 1);
        EXPECT_NEAR(rgb_host[i * 3 + 1], 0, TOLERANCE + 1);
        EXPECT_NEAR(rgb_host[i * 3 + 2], 0, TOLERANCE + 1);
    }

    cudaFree(y_gpu);
    cudaFree(uv_gpu);
    cudaFree(rgb_gpu);
}
