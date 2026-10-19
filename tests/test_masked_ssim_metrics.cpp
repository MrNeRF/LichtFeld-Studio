/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "lfs/kernels/ssim.cuh"
#include "training/metrics/metrics.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <random>
#include <utility>
#include <vector>

namespace {
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Tensor;
    constexpr int H = 48, W = 64, P = H * W;

    class MaskedSSIMMetricTest : public ::testing::Test {
        void SetUp() override {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
                GTEST_SKIP() << "CUDA unavailable";
        }
    };

    Tensor image(const std::vector<float>& data) {
        return Tensor::from_vector(data, {3, H, W}, Device::CUDA);
    }

    std::vector<float> generated_image() {
        std::mt19937 rng(1010);
        std::vector<float> data(3 * P);
        for (auto& value : data)
            value = 0.15f + 0.7f * static_cast<float>(rng() % 65536) / 65535.0f;
        return data;
    }

    std::vector<float> mask_data(bool thin) {
        std::vector<float> data(P);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                data[y * W + x] = thin ? (x >= 30 && x < 34) : !(x >= 20 && x < 44 && y >= 14 && y < 34);
        return data;
    }

    float legacy_masked(const Tensor& pred, const Tensor& target, const Tensor& mask) {
        const auto map = lfs::training::kernels::ssim_forward_map(pred, target, false).ssim_map;
        const auto expanded = mask.unsqueeze(0).unsqueeze(0).expand({1, 3, H, W});
        return (map * expanded).sum().item<float>() / (3.0f * mask.sum().item<float>());
    }

    void save_raw(const char* name, const std::vector<float>& values) {
        if (const auto* folder = std::getenv("LFS_METRIC_FIGURE_DIR")) {
            std::filesystem::create_directories(folder);
            std::ofstream file(std::filesystem::path(folder) / (std::string(name) + ".f32"), std::ios::binary);
            file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
        }
    }
} // namespace

TEST_F(MaskedSSIMMetricTest, IgnoredHolePaintCannotChangeReportedScore) {
    const auto clean = generated_image();
    auto weights = mask_data(false);
    const auto pred = image(clean);
    lfs::training::SSIM ssim;
    lfs::training::PSNR psnr;
    save_raw("prediction", clean);
    save_raw("mask", weights);
    std::ofstream evidence;
    if (const auto* folder = std::getenv("LFS_METRIC_FIGURE_DIR"))
        evidence.open(std::filesystem::path(folder) / "scores.csv");
    evidence << "soft,paint,ssim,psnr\n"
             << std::setprecision(10);
    for (bool soft : {false, true}) {
        if (soft)
            for (int y = 0; y < H; ++y)
                weights[y * W + 19] = 0.25f;
        const auto mask = Tensor::from_vector(weights, {H, W}, Device::CUDA);
        const float baseline = ssim.compute(pred, pred, mask);
        for (float paint : {0.0f, 0.25f, 0.5f, 0.75f, 1.0f}) {
            auto painted = clean;
            for (int c = 0; c < 3; ++c)
                for (int i = 0; i < P; ++i)
                    if (weights[i] == 0.0f)
                        painted[c * P + i] = (1.0f - paint) * clean[c * P + i] + paint * (c == 0 ? 1.0f : 0.0f);
            const auto target = image(painted);
            const float score = ssim.compute(pred, target, mask);
            const float psnr_score = psnr.compute(pred, target, mask);
            evidence << soft << ',' << paint << ',' << score << ',' << psnr_score << '\n';
            EXPECT_TRUE(std::isfinite(score));
            EXPECT_NEAR(score, baseline, 1e-6f) << "soft=" << soft << " paint=" << paint;
            EXPECT_EQ(psnr_score, psnr.compute(pred, pred, mask));
            if (!soft && paint == 1.0f)
                save_raw("painted", painted);
        }
    }
}

TEST_F(MaskedSSIMMetricTest, ThinKeptStripFallbackExcludesIgnoredPixels) {
    const auto clean = generated_image();
    auto painted = clean;
    const auto weights = mask_data(true);
    for (int c = 0; c < 3; ++c)
        for (int i = 0; i < P; ++i)
            if (weights[i] == 0.0f)
                painted[c * P + i] = c == 0 ? 1.0f : 0.0f;
    const auto mask = Tensor::from_vector(weights, {H, W}, Device::CUDA).to(DataType::UInt8);
    const auto selected = lfs::training::ssim_evaluation_mask(mask, true, "generated thin strip", mask.stream());
    EXPECT_EQ((selected.to(DataType::Float32) - mask.to(DataType::Float32)).abs().max().item<float>(), 0.0f);
    lfs::training::SSIM ssim;
    const auto pred = image(clean);
    const auto score = ssim.compute(pred, image(painted), selected);
    EXPECT_TRUE(std::isfinite(score));
    EXPECT_NEAR(score, ssim.compute(pred, pred, selected), 1e-6f);
}

TEST_F(MaskedSSIMMetricTest, UnmaskedAndAllSupportedScoresRetainTheirValues) {
    const auto data = generated_image();
    auto shifted = data;
    for (auto& value : shifted)
        value = std::min(1.0f, value + 0.07f);
    const auto pred = image(data), target = image(shifted);
    for (bool valid_padding : {false, true}) {
        lfs::training::SSIM ssim(valid_padding);
        const auto reference = lfs::training::kernels::ssim_forward(pred, target, valid_padding).first.mean().item<float>();
        EXPECT_EQ(std::bit_cast<uint32_t>(ssim.compute(pred, target)), std::bit_cast<uint32_t>(reference));
        const auto mask = Tensor::ones({H, W}, Device::CUDA);
        EXPECT_EQ(std::bit_cast<uint32_t>(ssim.compute(pred, target, mask)),
                  std::bit_cast<uint32_t>(legacy_masked(pred, target, mask)));
    }
}

TEST_F(MaskedSSIMMetricTest, AllSupportedMapsMatchUnmaskedMaps) {
    const auto data = generated_image();
    auto shifted = data;
    for (auto& value : shifted)
        value = std::min(1.0f, value + 0.07f);
    const auto pred = image(data);
    for (const auto target_type : {DataType::Float32, DataType::UInt8}) {
        SCOPED_TRACE(static_cast<int>(target_type));
        const auto target = target_type == DataType::UInt8
                                ? (image(shifted) * 255.0f).to(target_type)
                                : image(shifted);
        const auto reference = lfs::training::kernels::ssim_forward_map(pred, target, false)
                                   .ssim_map.to(Device::CPU)
                                   .contiguous();
        for (const auto mask_type : {DataType::Float32, DataType::UInt8, DataType::Bool}) {
            SCOPED_TRACE(static_cast<int>(mask_type));
            const auto mask = Tensor::ones({H, W}, Device::CUDA).to(mask_type);
            const auto actual = lfs::training::kernels::masked_ssim_map_forward(pred, target, mask)
                                    .to(Device::CPU)
                                    .contiguous();
            ASSERT_EQ(actual.shape(), reference.shape());
            for (size_t i = 0; i < actual.numel(); ++i)
                ASSERT_EQ(std::bit_cast<uint32_t>(actual.ptr<float>()[i]),
                          std::bit_cast<uint32_t>(reference.ptr<float>()[i]))
                    << "element=" << i;
        }
    }
}

TEST_F(MaskedSSIMMetricTest, ByteMaskMagnitudeDoesNotChangeBorderWindows) {
    lfs::training::SSIM ssim;
    for (const auto [height, width] : {std::pair{1, 1}, std::pair{7, 9}, std::pair{H, W}}) {
        const int pixels = height * width;
        std::vector<float> pred(3 * pixels), target(3 * pixels);
        std::vector<float> byte_one(pixels, 1), byte_255(pixels, 255);
        std::vector<float> float_one(pixels, 1.0f);
        for (int c = 0; c < 3; ++c)
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x) {
                    const int i = c * pixels + y * width + x;
                    pred[i] = 0.2f + 0.1f * c + 0.03f * x;
                    target[i] = 0.8f - 0.08f * c - 0.02f * y;
                }
        if (pixels > 1) {
            // An ignored hole touching the image edge exercises partial support.
            for (int y = 0; y < std::min(height, 3); ++y)
                for (int x = 0; x < std::min(width, 4); ++x) {
                    byte_one[y * width + x] = 0;
                    byte_255[y * width + x] = 0;
                    float_one[y * width + x] = 0.0f;
                }
        }
        const auto sh = static_cast<size_t>(height), sw = static_cast<size_t>(width);
        const auto p = Tensor::from_vector(pred, {3, sh, sw}, Device::CUDA);
        const auto t = Tensor::from_vector(target, {3, sh, sw}, Device::CUDA);
        const auto one = Tensor::from_vector(byte_one, {sh, sw}, Device::CUDA).to(DataType::UInt8);
        const auto full = Tensor::from_vector(byte_255, {sh, sw}, Device::CUDA).to(DataType::UInt8);
        const auto reference = Tensor::from_vector(float_one, {sh, sw}, Device::CUDA);
        const float score_one = ssim.compute(p, t, one);
        const float score_255 = ssim.compute(p, t, full);
        const float score_float = ssim.compute(p, t, reference);
        EXPECT_TRUE(std::isfinite(score_one));
        EXPECT_TRUE(std::isfinite(score_255));
        EXPECT_TRUE(std::isfinite(score_float));
        EXPECT_NEAR(score_one, score_float, 1e-6f) << height << 'x' << width;
        EXPECT_NEAR(score_255, score_float, 1e-6f) << height << 'x' << width;
    }
}

TEST_F(MaskedSSIMMetricTest, DISABLED_EvaluationBenchmark) {
    constexpr int height = 1080, width = 1920;
    std::vector<float> data(3 * height * width), target(data.size()), weights(height * width, 1.0f);
    std::mt19937 rng(1010);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(rng() % 256) / 255.0f;
        target[i] = std::clamp(data[i] + 0.03f, 0.0f, 1.0f);
    }
    for (int y = 300; y < 780; ++y)
        for (int x = 600; x < 1320; ++x)
            weights[y * width + x] = 0;
    const auto pred = Tensor::from_vector(data, {3, height, width}, Device::CUDA);
    const auto gt = Tensor::from_vector(target, {3, height, width}, Device::CUDA);
    const auto mask = Tensor::from_vector(weights, {height, width}, Device::CUDA);
    for (bool masked : {false, true}) {
        lfs::training::SSIM ssim;
        float score = 0;
        for (int i = 0; i < 10; ++i)
            score = ssim.compute(pred, gt, masked ? mask : Tensor{});
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 100; ++i)
            score = ssim.compute(pred, gt, masked ? mask : Tensor{});
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / 100;
        std::cout << std::setprecision(10) << "EVAL_BENCH masked=" << masked << " ms=" << ms << " score=" << score
                  << " bits=" << std::bit_cast<uint32_t>(score) << '\n';
    }
}

TEST_F(MaskedSSIMMetricTest, CompleteWindowPolicyRetainsExactScoresAndFallbackIsInvariant) {
    const auto data = generated_image();
    auto shifted = data;
    for (auto& value : shifted)
        value = std::min(1.0f, value + 0.07f);
    const auto pred = image(data), target = image(shifted);
    const auto weights = mask_data(false);
    const auto mask = Tensor::from_vector(weights, {H, W}, Device::CUDA).to(DataType::UInt8);
    const auto centers = lfs::training::ssim_evaluation_mask(mask, true, "generated hole", mask.stream());
    lfs::training::SSIM ssim;
    const float reference = legacy_masked(pred, target, centers.to(DataType::Float32));
    EXPECT_EQ(std::bit_cast<uint32_t>(ssim.compute_evaluation(pred, target, mask, true, "generated hole")),
              std::bit_cast<uint32_t>(reference));
    EXPECT_EQ(std::bit_cast<uint32_t>(ssim.compute_evaluation(pred, target, {}, true, "unmasked")),
              std::bit_cast<uint32_t>(ssim.compute(pred, target)));

    const auto thin_weights = mask_data(true);
    const auto thin = Tensor::from_vector(thin_weights, {H, W}, Device::CUDA).to(DataType::UInt8);
    auto painted = data;
    for (int c = 0; c < 3; ++c)
        for (int p = 0; p < P; ++p)
            if (thin_weights[p] == 0)
                painted[c * P + p] = c == 0 ? 1.0f : 0.0f;
    const float clean = ssim.compute_evaluation(pred, pred, thin, true, "generated thin strip");
    const float changed = ssim.compute_evaluation(pred, image(painted), thin, true, "generated thin strip");
    EXPECT_TRUE(std::isfinite(changed));
    EXPECT_NEAR(changed, clean, 1e-6f);
}

TEST_F(MaskedSSIMMetricTest, WeightedStatisticsMatchIndependentDoubleReference) {
    constexpr double gauss[] = {0.001028380123898387, 0.0075987582094967365, 0.036000773310661316,
                                0.10936068743467331, 0.21300552785396576, 0.26601171493530273,
                                0.21300552785396576, 0.10936068743467331, 0.036000773310661316,
                                0.0075987582094967365, 0.001028380123898387};
    auto pred = generated_image(), target = pred, weights = mask_data(false);
    for (int i = 0; i < 3 * P; ++i)
        target[i] = std::clamp(target[i] + 0.05f * std::sin(static_cast<float>(i)), 0.0f, 1.0f);
    for (int i = 0; i < P; ++i)
        if (weights[i] != 0)
            weights[i] = 0.1f + 0.9f * static_cast<float>(i % 17) / 16.0f;
    double score = 0, denominator = 0;
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const double center = weights[y * W + x];
                if (center == 0)
                    continue;
                double mass = 0, mx = 0, my = 0, xx = 0, yy = 0, xy = 0;
                for (int dy = -5; dy <= 5; ++dy) {
                    for (int dx = -5; dx <= 5; ++dx) {
                        const int px = x + dx, py = y + dy;
                        const bool inside = px >= 0 && px < W && py >= 0 && py < H;
                        const double w = gauss[dy + 5] * gauss[dx + 5] * (inside ? weights[py * W + px] : 1.0);
                        const double a = inside ? pred[c * P + py * W + px] : 0;
                        const double b = inside ? target[c * P + py * W + px] : 0;
                        mass += w;
                        mx += w * a;
                        my += w * b;
                        xx += w * a * a;
                        yy += w * b * b;
                        xy += w * a * b;
                    }
                }
                mx /= mass;
                my /= mass;
                const double vx = xx / mass - mx * mx, vy = yy / mass - my * my;
                const double cov = xy / mass - mx * my;
                score += center * (2 * mx * my + 0.0001) * (2 * cov + 0.0009) /
                         ((mx * mx + my * my + 0.0001) * (vx + vy + 0.0009));
                denominator += center;
            }
        }
    }
    lfs::training::SSIM ssim;
    const auto mask = Tensor::from_vector(weights, {H, W}, Device::CUDA);
    const float actual = ssim.compute(image(pred), image(target), mask);
    EXPECT_TRUE(std::isfinite(actual));
    EXPECT_NEAR(actual, score / denominator, 3e-6);
}

TEST_F(MaskedSSIMMetricTest, ByteTargetsBatchAndSparseSupportStayFinite) {
    const auto data = generated_image();
    const auto pred = image(data);
    const auto target = (pred * 255.0f).to(DataType::UInt8);
    const auto weights = mask_data(false);
    const auto mask = Tensor::from_vector(weights, {H, W}, Device::CUDA);
    lfs::training::SSIM ssim;
    const auto float_target = target.to(DataType::Float32) / 255.0f;
    EXPECT_NEAR(ssim.compute(pred, target, mask), ssim.compute(pred, float_target, mask), 1e-6f);
    EXPECT_EQ(ssim.compute(pred, target, mask), ssim.compute(pred, target, mask.to(DataType::UInt8)));
    const auto scaled_mask = mask * 255.0f;
    EXPECT_NEAR(ssim.compute(pred, target, mask),
                ssim.compute(pred, target, scaled_mask.to(DataType::UInt8)), 1e-6f);
    auto byte_weights = std::vector<float>(P);
    for (int i = 0; i < P; ++i)
        byte_weights[i] = weights[i] == 0.0f ? 0.0f : (i % W == 19 ? 32.0f : 255.0f);
    const auto weighted_mask = Tensor::from_vector(byte_weights, {H, W}, Device::CUDA);
    const auto flat_target = Tensor::zeros({3, H, W}, Device::CUDA);
    EXPECT_NEAR(ssim.compute(pred, flat_target, weighted_mask.clamp(0.0f, 1.0f)),
                ssim.compute(pred, flat_target, weighted_mask.to(DataType::UInt8)), 1e-6f);
    auto two_pred = std::vector<float>(3 * P, 0.0f), two_target = two_pred;
    auto two_weights = std::vector<float>(P, 0.0f);
    for (int c = 0; c < 3; ++c) {
        two_pred[c * P + 24 * W + 32] = 1.0f;
        two_target[c * P + 24 * W + 33] = 1.0f;
    }
    two_weights[24 * W + 32] = 255.0f;
    two_weights[24 * W + 33] = 32.0f;
    const auto two_mask = Tensor::from_vector(two_weights, {H, W}, Device::CUDA);
    EXPECT_NEAR(ssim.compute(image(two_pred), image(two_target), two_mask.clamp(0.0f, 1.0f)),
                ssim.compute(image(two_pred), image(two_target), two_mask.to(DataType::UInt8)), 1e-6f);
    EXPECT_NEAR(ssim.compute(pred.unsqueeze(0).expand({2, 3, H, W}).contiguous(), target.unsqueeze(0).expand({2, 3, H, W}).contiguous(), mask),
                ssim.compute(pred, target, mask), 1e-6f);
    for (float weight : {1.0f, 1e-32f}) {
        auto sparse = std::vector<float>(P, 0.0f);
        sparse[24 * W + 32] = weight;
        const float score = ssim.compute(Tensor::full({3, H, W}, 3.0f, Device::CUDA),
                                         Tensor::zeros({3, H, W}, Device::CUDA),
                                         Tensor::from_vector(sparse, {H, W}, Device::CUDA));
        EXPECT_TRUE(std::isfinite(score));
    }
}
