/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/memory_arena.hpp"
#include "core/parameters.hpp"
#include "core/tensor.hpp"
#include "cuda_backend_test.hpp"
#include "training/kernels/grad_alpha.hpp"
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

using namespace lfs::core;
using namespace lfs::core::param;

class BackgroundImageTest : public lfs::test::CudaBackendTest {
protected:
    void SetUp() override {
        LFS_CUDA_BACKEND_OR_RETURN();
        cudaSetDevice(0);
    }
    void TearDown() override {
        if (IsSkipped()) {
            return;
        }
        GlobalArenaManager::instance().get_arena().full_reset();
    }

    static Tensor createTestImage(const int c, const int h, const int w, const float value) {
        return Tensor::full({static_cast<size_t>(c), static_cast<size_t>(h), static_cast<size_t>(w)},
                            value, Device::GPU, DataType::Float32);
    }

    static Tensor createGradientImage(const int c, const int h, const int w) {
        auto cpu_tensor = Tensor::empty({static_cast<size_t>(c), static_cast<size_t>(h), static_cast<size_t>(w)},
                                        Device::CPU, DataType::Float32);
        float* ptr = cpu_tensor.ptr<float>();
        for (int ch = 0; ch < c; ++ch) {
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    ptr[ch * h * w + y * w + x] = static_cast<float>(x) / static_cast<float>(w - 1);
                }
            }
        }
        return cpu_tensor.to(Device::GPU);
    }
};

TEST_F(BackgroundImageTest, BilinearResize_IdentityWhenSameSize) {
    constexpr int C = 3, H = 64, W = 64;
    const auto src = createTestImage(C, H, W, 0.5f);
    auto dst = Tensor::empty({C, H, W}, Device::GPU, DataType::Float32);

    lfs::training::kernels::launch_bilinear_resize_chw(
        src.ptr<float>(), dst.ptr<float>(), C, H, W, H, W, nullptr);
    cudaDeviceSynchronize();

    const float diff = (src - dst).abs().max().item<float>();
    EXPECT_LT(diff, 1e-5f);
}

TEST_F(BackgroundImageTest, BilinearResize_Upscale2x) {
    constexpr int C = 3, SRC_H = 64, SRC_W = 64, DST_H = 128, DST_W = 128;
    const auto src = createTestImage(C, SRC_H, SRC_W, 0.7f);
    auto dst = Tensor::empty({C, DST_H, DST_W}, Device::GPU, DataType::Float32);

    lfs::training::kernels::launch_bilinear_resize_chw(
        src.ptr<float>(), dst.ptr<float>(), C, SRC_H, SRC_W, DST_H, DST_W, nullptr);
    cudaDeviceSynchronize();

    EXPECT_NEAR(dst.mean().item<float>(), 0.7f, 0.01f);
}

TEST_F(BackgroundImageTest, BilinearResize_Downscale2x) {
    constexpr int C = 3, SRC_H = 128, SRC_W = 128, DST_H = 64, DST_W = 64;
    const auto src = createTestImage(C, SRC_H, SRC_W, 0.3f);
    auto dst = Tensor::empty({C, DST_H, DST_W}, Device::GPU, DataType::Float32);

    lfs::training::kernels::launch_bilinear_resize_chw(
        src.ptr<float>(), dst.ptr<float>(), C, SRC_H, SRC_W, DST_H, DST_W, nullptr);
    cudaDeviceSynchronize();

    EXPECT_NEAR(dst.mean().item<float>(), 0.3f, 0.01f);
}

TEST_F(BackgroundImageTest, BilinearResize_NonSquareAspectRatio) {
    constexpr int C = 3, SRC_H = 64, SRC_W = 128, DST_H = 128, DST_W = 64;
    const auto src = createTestImage(C, SRC_H, SRC_W, 0.5f);
    auto dst = Tensor::empty({C, DST_H, DST_W}, Device::GPU, DataType::Float32);

    lfs::training::kernels::launch_bilinear_resize_chw(
        src.ptr<float>(), dst.ptr<float>(), C, SRC_H, SRC_W, DST_H, DST_W, nullptr);
    cudaDeviceSynchronize();

    EXPECT_EQ(dst.shape()[0], C);
    EXPECT_EQ(dst.shape()[1], DST_H);
    EXPECT_EQ(dst.shape()[2], DST_W);
    EXPECT_NEAR(dst.mean().item<float>(), 0.5f, 0.01f);
}

TEST_F(BackgroundImageTest, BilinearResize_PreservesValueRange) {
    constexpr int C = 3, SRC_H = 64, SRC_W = 64, DST_H = 128, DST_W = 128;
    const auto src = createGradientImage(C, SRC_H, SRC_W);
    auto dst = Tensor::empty({C, DST_H, DST_W}, Device::GPU, DataType::Float32);

    lfs::training::kernels::launch_bilinear_resize_chw(
        src.ptr<float>(), dst.ptr<float>(), C, SRC_H, SRC_W, DST_H, DST_W, nullptr);
    cudaDeviceSynchronize();

    EXPECT_GE(dst.min().item<float>(), -0.01f);
    EXPECT_LE(dst.max().item<float>(), 1.01f);
}

TEST_F(BackgroundImageTest, BilinearResize_LargeImage) {
    constexpr int C = 3, SRC_H = 1080, SRC_W = 1920, DST_H = 540, DST_W = 960;
    const auto src = createTestImage(C, SRC_H, SRC_W, 0.5f);
    auto dst = Tensor::empty({C, DST_H, DST_W}, Device::GPU, DataType::Float32);

    lfs::training::kernels::launch_bilinear_resize_chw(
        src.ptr<float>(), dst.ptr<float>(), C, SRC_H, SRC_W, DST_H, DST_W, nullptr);
    cudaDeviceSynchronize();

    EXPECT_EQ(dst.shape()[1], DST_H);
    EXPECT_EQ(dst.shape()[2], DST_W);
}

TEST_F(BackgroundImageTest, GradAlphaWithImage_ZeroGradImage) {
    constexpr int H = 64, W = 64;
    const auto grad_image = createTestImage(3, H, W, 0.0f);
    const auto bg_image = createTestImage(3, H, W, 0.5f);
    auto grad_alpha = Tensor::empty({static_cast<size_t>(H), static_cast<size_t>(W)}, Device::GPU, DataType::Float32);

    lfs::training::kernels::launch_fused_grad_alpha_with_image(
        grad_image.ptr<float>(), bg_image.ptr<float>(),
        grad_alpha.ptr<float>(), H, W, nullptr);
    cudaDeviceSynchronize();

    EXPECT_LT(grad_alpha.abs().max().item<float>(), 1e-6f);
}

TEST_F(BackgroundImageTest, GradAlphaWithImage_UniformBgMatchesSolid) {
    constexpr int H = 64, W = 64;
    const auto grad_image = createTestImage(3, H, W, 0.3f);
    const auto bg_image = createTestImage(3, H, W, 0.5f);
    const auto bg_color = Tensor::full({3}, 0.5f, Device::GPU);
    auto grad_alpha_image = Tensor::empty({static_cast<size_t>(H), static_cast<size_t>(W)}, Device::GPU, DataType::Float32);
    auto grad_alpha_color = Tensor::empty({static_cast<size_t>(H), static_cast<size_t>(W)}, Device::GPU, DataType::Float32);

    lfs::training::kernels::launch_fused_grad_alpha_with_image(
        grad_image.ptr<float>(), bg_image.ptr<float>(),
        grad_alpha_image.ptr<float>(), H, W, nullptr);
    lfs::training::kernels::launch_fused_grad_alpha(
        grad_image.ptr<float>(), bg_color.ptr<float>(),
        grad_alpha_color.ptr<float>(), H, W, true, nullptr);
    cudaDeviceSynchronize();

    EXPECT_LT((grad_alpha_image - grad_alpha_color).abs().max().item<float>(), 1e-5f);
}

TEST_F(BackgroundImageTest, GradAlphaWithImage_CorrectFormula) {
    constexpr int H = 2, W = 2;

    auto grad_image_cpu = Tensor::empty({3, 2, 2}, Device::CPU, DataType::Float32);
    float* gi = grad_image_cpu.ptr<float>();
    std::fill(gi, gi + 4, 0.1f);
    std::fill(gi + 4, gi + 8, 0.2f);
    std::fill(gi + 8, gi + 12, 0.3f);
    const auto grad_image = grad_image_cpu.to(Device::GPU);

    auto bg_image_cpu = Tensor::empty({3, 2, 2}, Device::CPU, DataType::Float32);
    std::fill(bg_image_cpu.ptr<float>(), bg_image_cpu.ptr<float>() + 12, 1.0f);
    const auto bg_image = bg_image_cpu.to(Device::GPU);

    auto grad_alpha = Tensor::empty({2, 2}, Device::GPU, DataType::Float32);

    lfs::training::kernels::launch_fused_grad_alpha_with_image(
        grad_image.ptr<float>(), bg_image.ptr<float>(),
        grad_alpha.ptr<float>(), H, W, nullptr);
    cudaDeviceSynchronize();

    // grad_alpha = -sum_c(grad_image[c] * bg_image[c]) = -(0.1 + 0.2 + 0.3) = -0.6
    const auto grad_alpha_cpu = grad_alpha.to(Device::CPU);
    const float* ga = grad_alpha_cpu.ptr<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(ga[i], -0.6f, 1e-5f);
    }
}

TEST_F(BackgroundImageTest, GradAlphaHWCUsesChannelLastLayout) {
    constexpr int H = 2;
    constexpr int W = 2;
    const auto grad_image = Tensor::from_vector(
        std::vector<float>{
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            7.0f, 8.0f, 9.0f,
            10.0f, 11.0f, 12.0f},
        {H, W, 3}, Device::GPU);
    const auto background = Tensor::from_vector(
        std::vector<float>{0.5f, 0.25f, 0.125f}, {3}, Device::GPU);
    auto grad_alpha = Tensor::empty({H, W}, Device::GPU, DataType::Float32);

    lfs::training::kernels::launch_fused_grad_alpha(
        grad_image.ptr<float>(), background.ptr<float>(), grad_alpha.ptr<float>(),
        H, W, false, nullptr);

    const auto values = grad_alpha.cpu().to_vector();
    const std::vector<float> expected = {
        -(1.0f * 0.5f + 2.0f * 0.25f + 3.0f * 0.125f),
        -(4.0f * 0.5f + 5.0f * 0.25f + 6.0f * 0.125f),
        -(7.0f * 0.5f + 8.0f * 0.25f + 9.0f * 0.125f),
        -(10.0f * 0.5f + 11.0f * 0.25f + 12.0f * 0.125f)};
    ASSERT_EQ(values.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(values[i], expected[i]);
    }
}

TEST_F(BackgroundImageTest, Checkpoint_BackgroundParamsSerialized) {
    OptimizationParameters params;
    params.bg_mode = BackgroundMode::Image;
    params.bg_color = {0.1f, 0.2f, 0.3f};
    params.bg_image_path = "/path/to/background.png";

    const nlohmann::json j = params.to_json();

    EXPECT_EQ(j["bg_mode"], "image");
    ASSERT_TRUE(j["bg_color"].is_array());
    EXPECT_EQ(j["bg_color"].size(), 3);
    EXPECT_FLOAT_EQ(j["bg_color"][0].get<float>(), 0.1f);
    EXPECT_FLOAT_EQ(j["bg_color"][1].get<float>(), 0.2f);
    EXPECT_FLOAT_EQ(j["bg_color"][2].get<float>(), 0.3f);
    EXPECT_EQ(j["bg_image_path"], "/path/to/background.png");
}

TEST_F(BackgroundImageTest, Checkpoint_BackgroundParamsDeserialized) {
    nlohmann::json j;
    j["iterations"] = 1000;
    j["means_lr"] = 0.001f;
    j["shs_lr"] = 0.001f;
    j["opacity_lr"] = 0.05f;
    j["scaling_lr"] = 0.005f;
    j["rotation_lr"] = 0.001f;
    j["lambda_dssim"] = 0.2f;
    j["min_opacity"] = 0.005f;
    j["refine_every"] = 100;
    j["start_refine"] = 500;
    j["stop_refine"] = 15000;
    j["grad_threshold"] = 0.0002f;
    j["sh_degree"] = 3;
    j["bg_mode"] = "image";
    j["bg_color"] = {0.4f, 0.5f, 0.6f};
    j["bg_image_path"] = "/custom/bg.jpg";

    const auto params = OptimizationParameters::from_json(j);

    EXPECT_EQ(params.bg_mode, BackgroundMode::Image);
    EXPECT_FLOAT_EQ(params.bg_color[0], 0.4f);
    EXPECT_FLOAT_EQ(params.bg_color[1], 0.5f);
    EXPECT_FLOAT_EQ(params.bg_color[2], 0.6f);
    EXPECT_EQ(params.bg_image_path, "/custom/bg.jpg");
}

TEST_F(BackgroundImageTest, Checkpoint_OldCheckpointLoadsWithDefaults) {
    nlohmann::json j;
    j["iterations"] = 1000;
    j["means_lr"] = 0.001f;
    j["shs_lr"] = 0.001f;
    j["opacity_lr"] = 0.05f;
    j["scaling_lr"] = 0.005f;
    j["rotation_lr"] = 0.001f;
    j["lambda_dssim"] = 0.2f;
    j["min_opacity"] = 0.005f;
    j["refine_every"] = 100;
    j["start_refine"] = 500;
    j["stop_refine"] = 15000;
    j["grad_threshold"] = 0.0002f;
    j["sh_degree"] = 3;

    const auto params = OptimizationParameters::from_json(j);

    EXPECT_EQ(params.bg_mode, BackgroundMode::SolidColor);
    EXPECT_FLOAT_EQ(params.bg_color[0], 0.0f);
    EXPECT_FLOAT_EQ(params.bg_color[1], 0.0f);
    EXPECT_FLOAT_EQ(params.bg_color[2], 0.0f);
    EXPECT_TRUE(params.bg_image_path.empty());
}

TEST_F(BackgroundImageTest, Checkpoint_AllBackgroundModesSerialize) {
    static constexpr const char* EXPECTED_NAMES[] = {"solid_color", "modulation", "image", "random"};

    for (int mode_int = 0; mode_int < 4; ++mode_int) {
        OptimizationParameters params;
        params.bg_mode = static_cast<BackgroundMode>(mode_int);

        const nlohmann::json j = params.to_json();
        EXPECT_EQ(j["bg_mode"], EXPECTED_NAMES[mode_int]);

        const auto restored = OptimizationParameters::from_json(j);
        EXPECT_EQ(restored.bg_mode, params.bg_mode);
    }
}

TEST_F(BackgroundImageTest, Checkpoint_EmptyImagePathNotSerialized) {
    OptimizationParameters params;
    params.bg_mode = BackgroundMode::SolidColor;
    params.bg_image_path = "";

    const nlohmann::json j = params.to_json();
    EXPECT_FALSE(j.contains("bg_image_path"));
}

TEST_F(BackgroundImageTest, BilinearResize_SinglePixel) {
    constexpr int C = 3;
    const auto src = createTestImage(C, 1, 1, 0.42f);
    auto dst = Tensor::empty({C, 1, 1}, Device::GPU, DataType::Float32);

    lfs::training::kernels::launch_bilinear_resize_chw(
        src.ptr<float>(), dst.ptr<float>(), C, 1, 1, 1, 1, nullptr);
    cudaDeviceSynchronize();

    EXPECT_NEAR(dst.slice(0, 0, 1).slice(1, 0, 1).slice(2, 0, 1).item<float>(), 0.42f, 1e-5f);
}

TEST_F(BackgroundImageTest, BilinearResize_UpscaleToSingleRow) {
    constexpr int C = 3, SRC_H = 4, SRC_W = 4, DST_H = 1, DST_W = 16;
    const auto src = createTestImage(C, SRC_H, SRC_W, 0.5f);
    auto dst = Tensor::empty({C, DST_H, DST_W}, Device::GPU, DataType::Float32);

    lfs::training::kernels::launch_bilinear_resize_chw(
        src.ptr<float>(), dst.ptr<float>(), C, SRC_H, SRC_W, DST_H, DST_W, nullptr);
    cudaDeviceSynchronize();

    EXPECT_EQ(dst.shape()[1], 1);
    EXPECT_EQ(dst.shape()[2], 16);
}

TEST_F(BackgroundImageTest, MultiSize_ResizeToMultipleDifferentSizes) {
    constexpr int BASE_C = 3, BASE_H = 64, BASE_W = 96;
    const auto base_image = createTestImage(BASE_C, BASE_H, BASE_W, 0.5f);

    const std::vector<std::pair<int, int>> camera_sizes = {
        {64, 96},
        {32, 48},
        {40, 72},
        {24, 36},
        {64, 80},
        {96, 144}};

    for (const auto& [h, w] : camera_sizes) {
        auto resized = Tensor::empty({static_cast<size_t>(BASE_C), static_cast<size_t>(h), static_cast<size_t>(w)}, Device::GPU, DataType::Float32);

        lfs::training::kernels::launch_bilinear_resize_chw(
            base_image.ptr<float>(), resized.ptr<float>(),
            BASE_C, BASE_H, BASE_W, h, w, nullptr);
        cudaDeviceSynchronize();

        EXPECT_EQ(resized.shape()[0], BASE_C);
        EXPECT_EQ(resized.shape()[1], h);
        EXPECT_EQ(resized.shape()[2], w);
        EXPECT_NEAR(resized.mean().item<float>(), 0.5f, 0.02f);
        EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    }
}

TEST_F(BackgroundImageTest, MultiSize_GradientWithDifferentSizes) {
    const std::vector<std::pair<int, int>> cases = {{1, 1}, {8, 13}, {16, 24}, {33, 17}};

    for (const auto& [h, w] : cases) {
        const auto grad_image = createTestImage(3, h, w, 0.2f);
        const auto bg_image = createTestImage(3, h, w, 0.5f);
        auto grad_alpha = Tensor::empty({static_cast<size_t>(h), static_cast<size_t>(w)},
                                        Device::GPU, DataType::Float32);

        lfs::training::kernels::launch_fused_grad_alpha_with_image(
            grad_image.ptr<float>(), bg_image.ptr<float>(),
            grad_alpha.ptr<float>(), h, w, nullptr);
        cudaDeviceSynchronize();

        constexpr float EXPECTED = -3.0f * 0.2f * 0.5f;
        EXPECT_NEAR(grad_alpha.mean().item<float>(), EXPECTED, 0.01f);
    }
}
