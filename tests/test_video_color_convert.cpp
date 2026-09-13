/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

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

class VideoColorConvertTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA devices";
        }
    }
};

TEST_F(VideoColorConvertTest, SolidRedYuv420p) {
    constexpr int WIDTH = 4;
    constexpr int HEIGHT = 4;

    std::vector<float> rgb_host(WIDTH * HEIGHT * 3);
    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
        rgb_host[i * 3 + 0] = 1.0f;
        rgb_host[i * 3 + 1] = 0.0f;
        rgb_host[i * 3 + 2] = 0.0f;
    }

    float* rgb_gpu = nullptr;
    uint8_t* y_gpu = nullptr;
    uint8_t* u_gpu = nullptr;
    uint8_t* v_gpu = nullptr;

    cudaMalloc(&rgb_gpu, WIDTH * HEIGHT * 3 * sizeof(float));
    cudaMalloc(&y_gpu, WIDTH * HEIGHT);
    cudaMalloc(&u_gpu, (WIDTH / 2) * (HEIGHT / 2));
    cudaMalloc(&v_gpu, (WIDTH / 2) * (HEIGHT / 2));

    cudaMemcpy(rgb_gpu, rgb_host.data(), WIDTH * HEIGHT * 3 * sizeof(float), cudaMemcpyHostToDevice);
    lfs::io::video::rgbToYuv420pCuda(rgb_gpu, y_gpu, u_gpu, v_gpu, WIDTH, HEIGHT, nullptr);
    cudaDeviceSynchronize();

    std::vector<uint8_t> y_host(WIDTH * HEIGHT);
    std::vector<uint8_t> u_host((WIDTH / 2) * (HEIGHT / 2));
    std::vector<uint8_t> v_host((WIDTH / 2) * (HEIGHT / 2));

    cudaMemcpy(y_host.data(), y_gpu, WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
    cudaMemcpy(u_host.data(), u_gpu, (WIDTH / 2) * (HEIGHT / 2), cudaMemcpyDeviceToHost);
    cudaMemcpy(v_host.data(), v_gpu, (WIDTH / 2) * (HEIGHT / 2), cudaMemcpyDeviceToHost);

    int expected_y, expected_u, expected_v;
    rgbToYuvReference(255, 0, 0, expected_y, expected_u, expected_v);

    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
        EXPECT_NEAR(y_host[i], expected_y, TOLERANCE);
    }
    for (size_t i = 0; i < u_host.size(); ++i) {
        EXPECT_NEAR(u_host[i], expected_u, TOLERANCE);
        EXPECT_NEAR(v_host[i], expected_v, TOLERANCE);
    }

    cudaFree(rgb_gpu);
    cudaFree(y_gpu);
    cudaFree(u_gpu);
    cudaFree(v_gpu);
}

TEST_F(VideoColorConvertTest, SolidGreenYuv420p) {
    constexpr int WIDTH = 4;
    constexpr int HEIGHT = 4;

    std::vector<float> rgb_host(WIDTH * HEIGHT * 3);
    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
        rgb_host[i * 3 + 0] = 0.0f;
        rgb_host[i * 3 + 1] = 1.0f;
        rgb_host[i * 3 + 2] = 0.0f;
    }

    float* rgb_gpu = nullptr;
    uint8_t* y_gpu = nullptr;
    uint8_t* u_gpu = nullptr;
    uint8_t* v_gpu = nullptr;

    cudaMalloc(&rgb_gpu, WIDTH * HEIGHT * 3 * sizeof(float));
    cudaMalloc(&y_gpu, WIDTH * HEIGHT);
    cudaMalloc(&u_gpu, (WIDTH / 2) * (HEIGHT / 2));
    cudaMalloc(&v_gpu, (WIDTH / 2) * (HEIGHT / 2));

    cudaMemcpy(rgb_gpu, rgb_host.data(), WIDTH * HEIGHT * 3 * sizeof(float), cudaMemcpyHostToDevice);
    lfs::io::video::rgbToYuv420pCuda(rgb_gpu, y_gpu, u_gpu, v_gpu, WIDTH, HEIGHT, nullptr);
    cudaDeviceSynchronize();

    std::vector<uint8_t> y_host(WIDTH * HEIGHT);
    cudaMemcpy(y_host.data(), y_gpu, WIDTH * HEIGHT, cudaMemcpyDeviceToHost);

    int expected_y, expected_u, expected_v;
    rgbToYuvReference(0, 255, 0, expected_y, expected_u, expected_v);

    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
        EXPECT_NEAR(y_host[i], expected_y, TOLERANCE);
    }

    cudaFree(rgb_gpu);
    cudaFree(y_gpu);
    cudaFree(u_gpu);
    cudaFree(v_gpu);
}

TEST_F(VideoColorConvertTest, SolidRedNv12) {
    constexpr int WIDTH = 4;
    constexpr int HEIGHT = 4;

    std::vector<float> rgb_host(WIDTH * HEIGHT * 3);
    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
        rgb_host[i * 3 + 0] = 1.0f;
        rgb_host[i * 3 + 1] = 0.0f;
        rgb_host[i * 3 + 2] = 0.0f;
    }

    float* rgb_gpu = nullptr;
    uint8_t* y_gpu = nullptr;
    uint8_t* uv_gpu = nullptr;

    cudaMalloc(&rgb_gpu, WIDTH * HEIGHT * 3 * sizeof(float));
    cudaMalloc(&y_gpu, WIDTH * HEIGHT);
    cudaMalloc(&uv_gpu, (HEIGHT / 2) * WIDTH);

    cudaMemcpy(rgb_gpu, rgb_host.data(), WIDTH * HEIGHT * 3 * sizeof(float), cudaMemcpyHostToDevice);
    lfs::io::video::rgbToNv12Cuda(rgb_gpu, y_gpu, uv_gpu, WIDTH, HEIGHT, 0, 0, nullptr);
    cudaDeviceSynchronize();

    std::vector<uint8_t> y_host(WIDTH * HEIGHT);
    std::vector<uint8_t> uv_host((HEIGHT / 2) * WIDTH);

    cudaMemcpy(y_host.data(), y_gpu, WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
    cudaMemcpy(uv_host.data(), uv_gpu, (HEIGHT / 2) * WIDTH, cudaMemcpyDeviceToHost);

    int expected_y, expected_u, expected_v;
    rgbToYuvReference(255, 0, 0, expected_y, expected_u, expected_v);

    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
        EXPECT_NEAR(y_host[i], expected_y, TOLERANCE);
    }
    // UV interleaved
    for (int i = 0; i < (HEIGHT / 2) * (WIDTH / 2); ++i) {
        EXPECT_NEAR(uv_host[i * 2], expected_u, TOLERANCE);
        EXPECT_NEAR(uv_host[i * 2 + 1], expected_v, TOLERANCE);
    }

    cudaFree(rgb_gpu);
    cudaFree(y_gpu);
    cudaFree(uv_gpu);
}

TEST_F(VideoColorConvertTest, Nv12WithPitch) {
    constexpr int WIDTH = 1920;
    constexpr int HEIGHT = 1080;
    constexpr int Y_PITCH = ((WIDTH + 255) / 256) * 256; // Aligned to 256
    constexpr int UV_PITCH = Y_PITCH;

    std::vector<float> rgb_host(WIDTH * HEIGHT * 3);
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            const int idx = (y * WIDTH + x) * 3;
            rgb_host[idx + 0] = static_cast<float>(x) / WIDTH;
            rgb_host[idx + 1] = static_cast<float>(y) / HEIGHT;
            rgb_host[idx + 2] = 0.0f;
        }
    }

    float* rgb_gpu = nullptr;
    uint8_t* y_gpu = nullptr;
    uint8_t* uv_gpu = nullptr;

    cudaMalloc(&rgb_gpu, WIDTH * HEIGHT * 3 * sizeof(float));
    cudaMalloc(&y_gpu, Y_PITCH * HEIGHT);
    cudaMalloc(&uv_gpu, UV_PITCH * (HEIGHT / 2));
    cudaMemset(y_gpu, 0, Y_PITCH * HEIGHT);
    cudaMemset(uv_gpu, 0, UV_PITCH * (HEIGHT / 2));

    cudaMemcpy(rgb_gpu, rgb_host.data(), WIDTH * HEIGHT * 3 * sizeof(float), cudaMemcpyHostToDevice);
    lfs::io::video::rgbToNv12Cuda(rgb_gpu, y_gpu, uv_gpu, WIDTH, HEIGHT, Y_PITCH, UV_PITCH, nullptr);
    cudaDeviceSynchronize();

    std::vector<uint8_t> y_host_pitched(Y_PITCH * HEIGHT);
    cudaMemcpy(y_host_pitched.data(), y_gpu, Y_PITCH * HEIGHT, cudaMemcpyDeviceToHost);

    // Verify corners
    int y00, y_red, y_green, y_yellow, u, v;
    rgbToYuvReference(0, 0, 0, y00, u, v);
    rgbToYuvReference(255, 0, 0, y_red, u, v);
    rgbToYuvReference(0, 255, 0, y_green, u, v);
    rgbToYuvReference(255, 255, 0, y_yellow, u, v);

    EXPECT_NEAR(y_host_pitched[0], y00, TOLERANCE);
    EXPECT_NEAR(y_host_pitched[WIDTH - 1], y_red, TOLERANCE + 1);
    EXPECT_NEAR(y_host_pitched[(HEIGHT - 1) * Y_PITCH], y_green, TOLERANCE + 1);
    EXPECT_NEAR(y_host_pitched[(HEIGHT - 1) * Y_PITCH + WIDTH - 1], y_yellow, TOLERANCE + 1);

    cudaFree(rgb_gpu);
    cudaFree(y_gpu);
    cudaFree(uv_gpu);
}

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

TEST_F(VideoColorConvertTest, FullHDExportSimulation) {
    constexpr int WIDTH = 1920;
    constexpr int HEIGHT = 1080;
    constexpr int C = 3;

    std::vector<float> chw_data(C * HEIGHT * WIDTH);
    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < HEIGHT; ++h) {
            for (int w = 0; w < WIDTH; ++w) {
                if (c == 0) {
                    chw_data[c * HEIGHT * WIDTH + h * WIDTH + w] = static_cast<float>(w) / WIDTH;
                } else if (c == 1) {
                    chw_data[c * HEIGHT * WIDTH + h * WIDTH + w] = static_cast<float>(h) / HEIGHT;
                } else {
                    chw_data[c * HEIGHT * WIDTH + h * WIDTH + w] = 0.0f;
                }
            }
        }
    }

    auto chw_tensor = lfs::core::Tensor::from_vector(chw_data, {C, HEIGHT, WIDTH}, lfs::core::Device::GPU);
    auto hwc_tensor = chw_tensor.permute({1, 2, 0}).contiguous();

    ASSERT_EQ(hwc_tensor.shape()[0], HEIGHT);
    ASSERT_EQ(hwc_tensor.shape()[1], WIDTH);
    ASSERT_EQ(hwc_tensor.shape()[2], C);

    const float* const rgb_gpu = hwc_tensor.ptr<float>();
    uint8_t* y_gpu = nullptr;
    uint8_t* uv_gpu = nullptr;

    cudaMalloc(&y_gpu, WIDTH * HEIGHT);
    cudaMalloc(&uv_gpu, WIDTH * (HEIGHT / 2));

    lfs::io::video::rgbToNv12Cuda(rgb_gpu, y_gpu, uv_gpu, WIDTH, HEIGHT, 0, 0, nullptr);
    cudaDeviceSynchronize();

    std::vector<uint8_t> y_host(WIDTH * HEIGHT);
    cudaMemcpy(y_host.data(), y_gpu, WIDTH * HEIGHT, cudaMemcpyDeviceToHost);

    int y00, y_red, y_green, y_yellow, u, v;
    rgbToYuvReference(0, 0, 0, y00, u, v);
    rgbToYuvReference(255, 0, 0, y_red, u, v);
    rgbToYuvReference(0, 255, 0, y_green, u, v);
    rgbToYuvReference(255, 255, 0, y_yellow, u, v);

    EXPECT_NEAR(y_host[0], y00, TOLERANCE);
    EXPECT_NEAR(y_host[WIDTH - 1], y_red, TOLERANCE + 1);
    EXPECT_NEAR(y_host[(HEIGHT - 1) * WIDTH], y_green, TOLERANCE + 1);
    EXPECT_NEAR(y_host[(HEIGHT - 1) * WIDTH + WIDTH - 1], y_yellow, TOLERANCE + 1);

    cudaFree(y_gpu);
    cudaFree(uv_gpu);
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

TEST_F(VideoColorConvertTest, Nv12ToRgbSolidRed) {
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

TEST_F(VideoColorConvertTest, Nv12ToRgbRoundtrip) {
    constexpr int WIDTH = 64;
    constexpr int HEIGHT = 64;

    std::vector<float> rgb_original(WIDTH * HEIGHT * 3);
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            const int idx = (y * WIDTH + x) * 3;
            rgb_original[idx] = static_cast<float>(x) / (WIDTH - 1);
            rgb_original[idx + 1] = static_cast<float>(y) / (HEIGHT - 1);
            rgb_original[idx + 2] = 0.5f;
        }
    }

    float* rgb_float_gpu = nullptr;
    uint8_t* y_gpu = nullptr;
    uint8_t* uv_gpu = nullptr;
    uint8_t* rgb_uint8_gpu = nullptr;

    cudaMalloc(&rgb_float_gpu, WIDTH * HEIGHT * 3 * sizeof(float));
    cudaMalloc(&y_gpu, WIDTH * HEIGHT);
    cudaMalloc(&uv_gpu, (HEIGHT / 2) * WIDTH);
    cudaMalloc(&rgb_uint8_gpu, WIDTH * HEIGHT * 3);

    cudaMemcpy(rgb_float_gpu, rgb_original.data(), WIDTH * HEIGHT * 3 * sizeof(float), cudaMemcpyHostToDevice);

    lfs::io::video::rgbToNv12Cuda(rgb_float_gpu, y_gpu, uv_gpu, WIDTH, HEIGHT, 0, 0, nullptr);
    lfs::io::video::nv12ToRgbCuda(y_gpu, uv_gpu, rgb_uint8_gpu, WIDTH, HEIGHT, 0, 0, nullptr);
    cudaDeviceSynchronize();

    std::vector<uint8_t> rgb_result(WIDTH * HEIGHT * 3);
    cudaMemcpy(rgb_result.data(), rgb_uint8_gpu, WIDTH * HEIGHT * 3, cudaMemcpyDeviceToHost);

    int max_diff = 0;
    for (int i = 0; i < WIDTH * HEIGHT * 3; ++i) {
        int expected = static_cast<int>(rgb_original[i] * 255.0f + 0.5f);
        int diff = std::abs(rgb_result[i] - expected);
        max_diff = std::max(max_diff, diff);
    }

    EXPECT_LE(max_diff, 4);

    cudaFree(rgb_float_gpu);
    cudaFree(y_gpu);
    cudaFree(uv_gpu);
    cudaFree(rgb_uint8_gpu);
}
