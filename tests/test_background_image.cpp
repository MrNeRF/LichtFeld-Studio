/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/memory_arena.hpp"
#include "core/parameters.hpp"
#include "core/tensor.hpp"
#include "training/kernels/grad_alpha.hpp"
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

using namespace lfs::core;
using namespace lfs::core::param;

class BackgroundImageTest : public ::testing::Test {
protected:
    void SetUp() override { cudaSetDevice(0); }
    void TearDown() override { GlobalArenaManager::instance().get_arena().full_reset(); }

    static Tensor createTestImage(const int c, const int h, const int w, const float value) {
        return Tensor::full({static_cast<size_t>(c), static_cast<size_t>(h), static_cast<size_t>(w)},
                            value, Device::CUDA, DataType::Float32);
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
        return cpu_tensor.to(Device::CUDA);
    }
};

TEST_F(BackgroundImageTest, BilinearResize_IdentityWhenSameSize) {
    constexpr int C = 3, H = 64, W = 64;
    const auto src = createTestImage(C, H, W, 0.5f);
    auto dst = Tensor::empty({C, H, W}, Device::CUDA, DataType::Float32);

    lfs::training::kernels::launch_bilinear_resize_chw(
        src.ptr<float>(), dst.ptr<float>(), C, H, W, H, W, nullptr);
    cudaDeviceSynchronize();

    const float diff = (src - dst).abs().max().item<float>();
    EXPECT_LT(diff, 1e-5f);
}

TEST_F(BackgroundImageTest, BilinearResize_Upscale2x) {
    constexpr int C = 3, SRC_H = 64, SRC_W = 64, DST_H = 128, DST_W = 128;
    const auto src = createTestImage(C, SRC_H, SRC_W, 0.7f);
    auto dst = Tensor::empty({C, DST_H, DST_W}, Device::CUDA, DataType::Float32);

    lfs::training::kernels::launch_bilinear_resize_chw(
        src.ptr<float>(), dst.ptr<float>(), C, SRC_H, SRC_W, DST_H, DST_W, nullptr);
    cudaDeviceSynchronize();

    EXPECT_NEAR(dst.mean().item<float>(), 0.7f, 0.01f);
}

TEST_F(BackgroundImageTest, BilinearResize_Downscale2x) {
    constexpr int C = 3, SRC_H = 128, SRC_W = 128, DST_H = 64, DST_W = 64;
    const auto src = createTestImage(C, SRC_H, SRC_W, 0.3f);
    auto dst = Tensor::empty({C, DST_H, DST_W}, Device::CUDA, DataType::Float32);

    lfs::training::kernels::launch_bilinear_resize_chw(
        src.ptr<float>(), dst.ptr<float>(), C, SRC_H, SRC_W, DST_H, DST_W, nullptr);
    cudaDeviceSynchronize();

    EXPECT_NEAR(dst.mean().item<float>(), 0.3f, 0.01f);
}

TEST_F(BackgroundImageTest, BilinearResize_NonSquareAspectRatio) {
    constexpr int C = 3, SRC_H = 64, SRC_W = 128, DST_H = 128, DST_W = 64;
    const auto src = createTestImage(C, SRC_H, SRC_W, 0.5f);
    auto dst = Tensor::empty({C, DST_H, DST_W}, Device::CUDA, DataType::Float32);

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
    auto dst = Tensor::empty({C, DST_H, DST_W}, Device::CUDA, DataType::Float32);

    lfs::training::kernels::launch_bilinear_resize_chw(
        src.ptr<float>(), dst.ptr<float>(), C, SRC_H, SRC_W, DST_H, DST_W, nullptr);
    cudaDeviceSynchronize();

    EXPECT_GE(dst.min().item<float>(), -0.01f);
    EXPECT_LE(dst.max().item<float>(), 1.01f);
}

TEST_F(BackgroundImageTest, BilinearResize_LargeImage) {
    constexpr int C = 3, SRC_H = 1080, SRC_W = 1920, DST_H = 540, DST_W = 960;
    const auto src = createTestImage(C, SRC_H, SRC_W, 0.5f);
    auto dst = Tensor::empty({C, DST_H, DST_W}, Device::CUDA, DataType::Float32);

    lfs::training::kernels::launch_bilinear_resize_chw(
        src.ptr<float>(), dst.ptr<float>(), C, SRC_H, SRC_W, DST_H, DST_W, nullptr);
    cudaDeviceSynchronize();

    EXPECT_EQ(dst.shape()[1], DST_H);
    EXPECT_EQ(dst.shape()[2], DST_W);
}

TEST_F(BackgroundImageTest, BackgroundBlendWithImage_FullyOpaque) {
    constexpr int H = 64, W = 64;
    const auto image = createTestImage(3, H, W, 0.8f);
    const auto alpha = Tensor::full({static_cast<size_t>(H), static_cast<size_t>(W)}, 1.0f, Device::CUDA);
    const auto bg_image = createTestImage(3, H, W, 0.2f);
    auto output = Tensor::empty({3, H, W}, Device::CUDA, DataType::Float32);

    lfs::training::kernels::launch_fused_background_blend_with_image(
        image.ptr<float>(), alpha.ptr<float>(), bg_image.ptr<float>(),
        output.ptr<float>(), H, W, nullptr);
    cudaDeviceSynchronize();

    EXPECT_LT((output - image).abs().max().item<float>(), 1e-5f);
}

TEST_F(BackgroundImageTest, BackgroundBlendWithImage_FullyTransparent) {
    constexpr int H = 64, W = 64;
    const auto image = createTestImage(3, H, W, 0.0f);
    const auto alpha = Tensor::full({static_cast<size_t>(H), static_cast<size_t>(W)}, 0.0f, Device::CUDA);
    const auto bg_image = createTestImage(3, H, W, 0.6f);
    auto output = Tensor::empty({3, H, W}, Device::CUDA, DataType::Float32);

    lfs::training::kernels::launch_fused_background_blend_with_image(
        image.ptr<float>(), alpha.ptr<float>(), bg_image.ptr<float>(),
        output.ptr<float>(), H, W, nullptr);
    cudaDeviceSynchronize();

    EXPECT_LT((output - bg_image).abs().max().item<float>(), 1e-5f);
}

TEST_F(BackgroundImageTest, BackgroundBlendWithImage_HalfTransparent) {
    constexpr int H = 64, W = 64;
    const auto image = createTestImage(3, H, W, 0.4f);
    const auto alpha = Tensor::full({static_cast<size_t>(H), static_cast<size_t>(W)}, 0.5f, Device::CUDA);
    const auto bg_image = createTestImage(3, H, W, 0.8f);
    auto output = Tensor::empty({3, H, W}, Device::CUDA, DataType::Float32);

    lfs::training::kernels::launch_fused_background_blend_with_image(
        image.ptr<float>(), alpha.ptr<float>(), bg_image.ptr<float>(),
        output.ptr<float>(), H, W, nullptr);
    cudaDeviceSynchronize();

    // image + (1-0.5) * bg = 0.4 + 0.5 * 0.8 = 0.8
    EXPECT_NEAR(output.mean().item<float>(), 0.8f, 0.01f);
}

TEST_F(BackgroundImageTest, BackgroundBlendWithImage_MatchesSolidColorUniform) {
    constexpr int H = 64, W = 64;
    const auto image = createTestImage(3, H, W, 0.3f);
    const auto alpha = Tensor::full({static_cast<size_t>(H), static_cast<size_t>(W)}, 0.7f, Device::CUDA);
    const auto bg_image = createTestImage(3, H, W, 0.5f);
    const auto bg_color = Tensor::full({3}, 0.5f, Device::CUDA);
    auto output_image = Tensor::empty({3, H, W}, Device::CUDA, DataType::Float32);
    auto output_color = Tensor::empty({3, H, W}, Device::CUDA, DataType::Float32);

    lfs::training::kernels::launch_fused_background_blend_with_image(
        image.ptr<float>(), alpha.ptr<float>(), bg_image.ptr<float>(),
        output_image.ptr<float>(), H, W, nullptr);
    lfs::training::kernels::launch_fused_background_blend(
        image.ptr<float>(), alpha.ptr<float>(), bg_color.ptr<float>(),
        output_color.ptr<float>(), H, W, nullptr);
    cudaDeviceSynchronize();

    EXPECT_LT((output_image - output_color).abs().max().item<float>(), 1e-5f);
}

TEST_F(BackgroundImageTest, BackgroundBlendRoundTripInPlace_SolidColor) {
    constexpr int H = 8, W = 8;
    const auto raw_image = createGradientImage(3, H, W);

    auto alpha_cpu = Tensor::empty({static_cast<size_t>(H), static_cast<size_t>(W)}, Device::CPU, DataType::Float32);
    float* alpha_ptr = alpha_cpu.ptr<float>();
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            alpha_ptr[y * W + x] = 0.2f + 0.6f * static_cast<float>(x + y) / static_cast<float>((H - 1) + (W - 1));
        }
    }
    const auto alpha = alpha_cpu.to(Device::CUDA);

    auto bg_color_cpu = Tensor::empty({3}, Device::CPU, DataType::Float32);
    float* bg_ptr = bg_color_cpu.ptr<float>();
    bg_ptr[0] = 0.1f;
    bg_ptr[1] = 0.4f;
    bg_ptr[2] = 0.7f;
    const auto bg_color = bg_color_cpu.to(Device::CUDA);

    auto buffer = raw_image.clone();
    lfs::training::kernels::launch_fused_background_blend(
        buffer.ptr<float>(), alpha.ptr<float>(), bg_color.ptr<float>(),
        buffer.ptr<float>(), H, W, nullptr);
    lfs::training::kernels::launch_fused_background_unblend(
        buffer.ptr<float>(), alpha.ptr<float>(), bg_color.ptr<float>(),
        H, W, nullptr);
    cudaDeviceSynchronize();

    EXPECT_LT((buffer - raw_image).abs().max().item<float>(), 1e-5f);
}

TEST_F(BackgroundImageTest, BackgroundBlendRoundTripInPlace_BackgroundImage) {
    constexpr int H = 8, W = 8;
    const auto raw_image = createGradientImage(3, H, W);

    auto alpha_cpu = Tensor::empty({static_cast<size_t>(H), static_cast<size_t>(W)}, Device::CPU, DataType::Float32);
    float* alpha_ptr = alpha_cpu.ptr<float>();
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            alpha_ptr[y * W + x] = 0.1f + 0.8f * static_cast<float>(y) / static_cast<float>(H - 1);
        }
    }
    const auto alpha = alpha_cpu.to(Device::CUDA);

    auto bg_image_cpu = Tensor::empty({3, H, W}, Device::CPU, DataType::Float32);
    float* bg_ptr = bg_image_cpu.ptr<float>();
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                bg_ptr[c * H * W + y * W + x] = 0.05f * static_cast<float>(c + 1) + 0.01f * static_cast<float>(x + y);
            }
        }
    }
    const auto bg_image = bg_image_cpu.to(Device::CUDA);

    auto buffer = raw_image.clone();
    lfs::training::kernels::launch_fused_background_blend_with_image(
        buffer.ptr<float>(), alpha.ptr<float>(), bg_image.ptr<float>(),
        buffer.ptr<float>(), H, W, nullptr);
    lfs::training::kernels::launch_fused_background_unblend_with_image(
        buffer.ptr<float>(), alpha.ptr<float>(), bg_image.ptr<float>(),
        H, W, nullptr);
    cudaDeviceSynchronize();

    EXPECT_LT((buffer - raw_image).abs().max().item<float>(), 1e-5f);
}

TEST_F(BackgroundImageTest, GradAlphaWithImage_ZeroGradImage) {
    constexpr int H = 64, W = 64;
    const auto grad_image = createTestImage(3, H, W, 0.0f);
    const auto bg_image = createTestImage(3, H, W, 0.5f);
    auto grad_alpha = Tensor::empty({static_cast<size_t>(H), static_cast<size_t>(W)}, Device::CUDA, DataType::Float32);

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
    const auto bg_color = Tensor::full({3}, 0.5f, Device::CUDA);
    auto grad_alpha_image = Tensor::empty({static_cast<size_t>(H), static_cast<size_t>(W)}, Device::CUDA, DataType::Float32);
    auto grad_alpha_color = Tensor::empty({static_cast<size_t>(H), static_cast<size_t>(W)}, Device::CUDA, DataType::Float32);

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
    const auto grad_image = grad_image_cpu.to(Device::CUDA);

    auto bg_image_cpu = Tensor::empty({3, 2, 2}, Device::CPU, DataType::Float32);
    std::fill(bg_image_cpu.ptr<float>(), bg_image_cpu.ptr<float>() + 12, 1.0f);
    const auto bg_image = bg_image_cpu.to(Device::CUDA);

    auto grad_alpha = Tensor::empty({2, 2}, Device::CUDA, DataType::Float32);

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
        {H, W, 3}, Device::CUDA);
    const auto background = Tensor::from_vector(
        std::vector<float>{0.5f, 0.25f, 0.125f}, {3}, Device::CUDA);
    auto grad_alpha = Tensor::empty({H, W}, Device::CUDA, DataType::Float32);

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
    auto dst = Tensor::empty({C, 1, 1}, Device::CUDA, DataType::Float32);

    lfs::training::kernels::launch_bilinear_resize_chw(
        src.ptr<float>(), dst.ptr<float>(), C, 1, 1, 1, 1, nullptr);
    cudaDeviceSynchronize();

    EXPECT_NEAR(dst.slice(0, 0, 1).slice(1, 0, 1).slice(2, 0, 1).item<float>(), 0.42f, 1e-5f);
}

TEST_F(BackgroundImageTest, BilinearResize_UpscaleToSingleRow) {
    constexpr int C = 3, SRC_H = 4, SRC_W = 4, DST_H = 1, DST_W = 16;
    const auto src = createTestImage(C, SRC_H, SRC_W, 0.5f);
    auto dst = Tensor::empty({C, DST_H, DST_W}, Device::CUDA, DataType::Float32);

    lfs::training::kernels::launch_bilinear_resize_chw(
        src.ptr<float>(), dst.ptr<float>(), C, SRC_H, SRC_W, DST_H, DST_W, nullptr);
    cudaDeviceSynchronize();

    EXPECT_EQ(dst.shape()[1], 1);
    EXPECT_EQ(dst.shape()[2], 16);
}

TEST_F(BackgroundImageTest, BackgroundBlendWithImage_VaryingAlpha) {
    constexpr int H = 4, W = 4;
    const auto image = createTestImage(3, H, W, 1.0f);
    const auto bg_image = createTestImage(3, H, W, 0.0f);

    auto alpha_cpu = Tensor::empty({static_cast<size_t>(H), static_cast<size_t>(W)}, Device::CPU, DataType::Float32);
    float* a = alpha_cpu.ptr<float>();
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            a[y * W + x] = static_cast<float>(y) / static_cast<float>(H - 1);
        }
    }
    const auto alpha = alpha_cpu.to(Device::CUDA);
    auto output = Tensor::empty({3, H, W}, Device::CUDA, DataType::Float32);

    lfs::training::kernels::launch_fused_background_blend_with_image(
        image.ptr<float>(), alpha.ptr<float>(), bg_image.ptr<float>(),
        output.ptr<float>(), H, W, nullptr);
    cudaDeviceSynchronize();

    const auto output_cpu = output.to(Device::CPU);
    const float* o = output_cpu.ptr<float>();
    for (int i = 0; i < 3 * H * W; ++i) {
        EXPECT_NEAR(o[i], 1.0f, 1e-5f);
    }
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
        auto resized = Tensor::empty({static_cast<size_t>(BASE_C), static_cast<size_t>(h), static_cast<size_t>(w)}, Device::CUDA, DataType::Float32);

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

TEST_F(BackgroundImageTest, MultiSize_BlendWithDifferentSizes) {
    struct TestCase {
        size_t h, w;
        float render_val, bg_val, alpha_val;
    };

    const std::vector<TestCase> cases = {
        {8, 12, 0.3f, 0.7f, 0.5f},
        {16, 24, 0.2f, 0.8f, 0.25f},
        {32, 48, 0.6f, 0.4f, 0.75f},
        {5, 7, 0.0f, 1.0f, 0.0f},
        {12, 20, 1.0f, 0.0f, 1.0f},
    };

    for (const auto& tc : cases) {
        const auto render = createTestImage(3, tc.h, tc.w, tc.render_val);
        const auto bg = createTestImage(3, tc.h, tc.w, tc.bg_val);
        const auto alpha = Tensor::full({tc.h, tc.w},
                                        tc.alpha_val, Device::CUDA);
        auto output = Tensor::empty({3, tc.h, tc.w}, Device::CUDA, DataType::Float32);

        lfs::training::kernels::launch_fused_background_blend_with_image(
            render.ptr<float>(), alpha.ptr<float>(), bg.ptr<float>(),
            output.ptr<float>(), tc.h, tc.w, nullptr);
        cudaDeviceSynchronize();

        const float expected = tc.render_val + (1.0f - tc.alpha_val) * tc.bg_val;
        EXPECT_NEAR(output.mean().item<float>(), expected, 0.01f);
    }
}

TEST_F(BackgroundImageTest, MultiSize_InterleavedSizeChanges) {
    constexpr int BASE_C = 3, BASE_H = 32, BASE_W = 48;
    const auto base_image = createTestImage(BASE_C, BASE_H, BASE_W, 0.42f);

    const std::vector<std::pair<int, int>> size_sequence = {
        {16, 24},
        {32, 48},
        {20, 36},
        {16, 24},
        {12, 18},
        {32, 48},
        {20, 36},
        {16, 24},
        {32, 48},
        {12, 18}};

    for (size_t iter = 0; iter < size_sequence.size(); ++iter) {
        const auto [h, w] = size_sequence[iter];

        auto bg_resized = Tensor::empty({static_cast<size_t>(BASE_C), static_cast<size_t>(h), static_cast<size_t>(w)}, Device::CUDA, DataType::Float32);
        lfs::training::kernels::launch_bilinear_resize_chw(
            base_image.ptr<float>(), bg_resized.ptr<float>(),
            BASE_C, BASE_H, BASE_W, h, w, nullptr);

        const auto render = createTestImage(3, h, w, 0.3f);
        const auto alpha = Tensor::full({static_cast<size_t>(h), static_cast<size_t>(w)}, 0.6f, Device::CUDA);
        auto output = Tensor::empty({3, static_cast<size_t>(h), static_cast<size_t>(w)}, Device::CUDA, DataType::Float32);

        lfs::training::kernels::launch_fused_background_blend_with_image(
            render.ptr<float>(), alpha.ptr<float>(), bg_resized.ptr<float>(),
            output.ptr<float>(), h, w, nullptr);
        cudaDeviceSynchronize();

        constexpr float EXPECTED = 0.3f + 0.4f * 0.42f;
        EXPECT_NEAR(output.mean().item<float>(), EXPECTED, 0.02f);
        EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    }
}

TEST_F(BackgroundImageTest, MultiSize_GradientWithDifferentSizes) {
    const std::vector<std::pair<int, int>> cases = {{1, 1}, {8, 13}, {16, 24}, {33, 17}};

    for (const auto& [h, w] : cases) {
        const auto grad_image = createTestImage(3, h, w, 0.2f);
        const auto bg_image = createTestImage(3, h, w, 0.5f);
        auto grad_alpha = Tensor::empty({static_cast<size_t>(h), static_cast<size_t>(w)},
                                        Device::CUDA, DataType::Float32);

        lfs::training::kernels::launch_fused_grad_alpha_with_image(
            grad_image.ptr<float>(), bg_image.ptr<float>(),
            grad_alpha.ptr<float>(), h, w, nullptr);
        cudaDeviceSynchronize();

        constexpr float EXPECTED = -3.0f * 0.2f * 0.5f;
        EXPECT_NEAR(grad_alpha.mean().item<float>(), EXPECTED, 0.01f);
    }
}
