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

TEST_F(VideoColorConvertCudaTest, Nv12RangesWithPitch) {
    constexpr int width = 32, height = 16, pitch = 40;
    std::vector<uint8_t> y(pitch * height), uv(pitch * height / 2), rgb(width * height * 3);
    for (size_t i = 0; i < y.size(); ++i)
        y[i] = static_cast<uint8_t>(i * 13);
    for (size_t i = 0; i < uv.size(); ++i)
        uv[i] = static_cast<uint8_t>(i * 37);
    uint8_t *dy = nullptr, *duv = nullptr, *drgb = nullptr;
    ASSERT_EQ(cudaMalloc(&dy, y.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&duv, uv.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&drgb, rgb.size()), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dy, y.data(), y.size(), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(duv, uv.data(), uv.size(), cudaMemcpyHostToDevice), cudaSuccess);
    for (const bool full : {false, true}) {
        lfs::io::video::nv12ToRgbCuda(dy, duv, drgb, width, height, pitch, pitch, nullptr, full);
        ASSERT_EQ(cudaMemcpy(rgb.data(), drgb, rgb.size(), cudaMemcpyDeviceToHost), cudaSuccess);
        for (int row = 0; row < height; ++row) {
            for (int col = 0; col < width; ++col) {
                const int luma = y[row * pitch + col];
                const int chroma = (row / 2) * pitch + (col / 2) * 2;
                const int u = uv[chroma] - 128, v = uv[chroma + 1] - 128;
                // Keep the old limited-range integer equations as an exact guard.
                const int expected[] = {
                    full ? static_cast<int>(std::round(luma + 1.402 * v)) : (298 * (luma - 16) + 409 * v + 128) >> 8,
                    full ? static_cast<int>(std::round(luma - 0.344136 * u - 0.714136 * v)) : (298 * (luma - 16) - 100 * u - 208 * v + 128) >> 8,
                    full ? static_cast<int>(std::round(luma + 1.772 * u)) : (298 * (luma - 16) + 516 * u + 128) >> 8};
                for (int c = 0; c < 3; ++c)
                    EXPECT_NEAR(rgb[(row * width + col) * 3 + c], std::clamp(expected[c], 0, 255), full ? 1 : 0);
            }
        }
    }
    cudaFree(dy);
    cudaFree(duv);
    cudaFree(drgb);
}
