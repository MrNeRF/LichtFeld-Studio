/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "training/kernels/mask_preprocess.hpp"
#include "training/kernels/thin_structure.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace {
    using namespace lfs::core;
    using namespace lfs::training::kernels;
    constexpr size_t height = 97;
    constexpr size_t width = 129;
    constexpr size_t pixels = height * width;

    template <typename Image>
    void ridge_with_optional_mask(const Image& image, const Tensor& mask, Tensor& out,
                                  RidgeWorkspace& workspace, const MaskPhotoMode mode) {
        // The baseline API lacks the mask input; keep the same semantic test executable there.
        if constexpr (requires { ridge_structure_map(image, out, workspace, mask, mode); })
            ridge_structure_map(image, out, workspace, mask, mode);
        else
            ridge_structure_map(image, out, workspace);
    }

    std::vector<float> make_image(const bool change_hidden) {
        std::vector<float> values(3 * pixels);
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                const size_t i = y * width + x;
                const bool hidden = y >= height / 3 && y < 2 * height / 3 &&
                                    x >= width / 3 && x < 2 * width / 3;
                for (size_t channel = 0; channel < 3; ++channel) {
                    const float base = static_cast<float>((x * 17 + y * 31 + channel * 53) % 211) / 255.f;
                    values[channel * pixels + i] = change_hidden && hidden ? (channel == 0 ? 1.f : 0.f) : base;
                }
            }
        }
        return values;
    }

    template <typename T>
    void compare_guidance(const std::vector<T>& mask_values, const MaskPhotoMode mode) {
        const auto source_a = make_image(false);
        const auto source_b = make_image(true);
        const auto image_a = Tensor::from_vector(source_a, {3, height, width}, Device::CUDA);
        const auto image_b = Tensor::from_vector(source_b, {3, height, width}, Device::CUDA);
        Tensor mask;
        if constexpr (std::is_same_v<T, uint8_t>) {
            auto cpu_mask = Tensor::empty({height, width}, Device::CPU, DataType::UInt8);
            std::copy(mask_values.begin(), mask_values.end(), cpu_mask.ptr<uint8_t>());
            mask = cpu_mask.to(Device::CUDA);
        } else {
            mask = Tensor::from_vector(mask_values, {height, width}, Device::CUDA);
        }
        RidgeWorkspace workspace;
        auto ridge_a = Tensor::empty({height, width}, Device::CUDA);
        auto ridge_b = Tensor::empty({height, width}, Device::CUDA);
        ridge_with_optional_mask(image_a, mask, ridge_a, workspace, mode);
        ridge_with_optional_mask(image_b, mask, ridge_b, workspace, mode);
        std::vector<float> keep(pixels, 1.f);
        for (size_t y = height / 3; y < 2 * height / 3; ++y)
            for (size_t x = width / 3; x < 2 * width / 3; ++x)
                keep[y * width + x] = 0.f;
        const auto base_weight = Tensor::from_vector(keep, {height, width}, Device::CUDA);
        Tensor weight_a, weight_b;
        structure_photometric_weight(ridge_a, base_weight, weight_a, 1.f, false);
        structure_photometric_weight(ridge_b, base_weight, weight_b, 1.f, false);
        const auto a = weight_a.cpu().to_vector();
        const auto b = weight_b.cpu().to_vector();
        float maximum = 0.f;
        for (size_t i = 0; i < pixels; ++i) {
            const size_t x = i % width, y = i / width;
            const bool hidden = y >= height / 3 && y < 2 * height / 3 &&
                                x >= width / 3 && x < 2 * width / 3;
            if (!hidden) {
                EXPECT_TRUE(std::isfinite(a[i]));
                EXPECT_TRUE(std::isfinite(b[i]));
                maximum = std::max(maximum, std::abs(a[i] - b[i]));
            }
        }
        EXPECT_LE(maximum, 1e-6f);
    }
} // namespace

TEST(MaskThinStructureGuidance, HiddenFloatPixelsDoNotChangeRetainedWeights) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "CUDA unavailable";
    std::vector<float> mask(pixels, 1.f);
    for (size_t y = height / 3; y < 2 * height / 3; ++y)
        for (size_t x = width / 3; x < 2 * width / 3; ++x)
            mask[y * width + x] = 0.f;
    compare_guidance(mask, MaskPhotoMode::BinaryGt0);
}

TEST(MaskThinStructureGuidance, SegmentAndIgnoreBandDoesNotChangeRetainedWeights) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "CUDA unavailable";
    std::vector<uint8_t> mask(pixels, 255);
    for (size_t y = height / 3; y < 2 * height / 3; ++y)
        for (size_t x = width / 3; x < 2 * width / 3; ++x)
            mask[y * width + x] = 128;
    compare_guidance(mask, MaskPhotoMode::SegmentAndIgnore);
}

TEST(MaskThinStructureGuidance, AllKeepMaskPreservesUnmaskedRidgeValues) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "CUDA unavailable";
    const auto image = Tensor::from_vector(make_image(false), {3, height, width}, Device::CUDA);
    const auto mask = Tensor::ones({height, width}, Device::CUDA);
    RidgeWorkspace reference_workspace, masked_workspace;
    auto reference = Tensor::empty({height, width}, Device::CUDA);
    auto with_mask = Tensor::empty({height, width}, Device::CUDA);
    ridge_structure_map(image, reference, reference_workspace);
    ridge_with_optional_mask(image, mask, with_mask, masked_workspace, MaskPhotoMode::BinaryGt0);
    const auto a = reference.cpu().to_vector();
    const auto b = with_mask.cpu().to_vector();
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i)
        EXPECT_FLOAT_EQ(a[i], b[i]);
}

TEST(MaskThinStructureGuidance, MaskedRidgeMeanUsesRetainedPixels) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "CUDA unavailable";
    const auto image = Tensor::from_vector(make_image(false), {3, height, width}, Device::CUDA);
    std::vector<float> keep(pixels, 0.f);
    for (size_t y = 8; y < 60; ++y)
        for (size_t x = 8; x < 90; ++x)
            keep[y * width + x] = 1.f;
    const auto mask = Tensor::from_vector(keep, {height, width}, Device::CUDA);
    RidgeWorkspace workspace;
    auto ridge = Tensor::empty({height, width}, Device::CUDA);
    ridge_with_optional_mask(image, mask, ridge, workspace, MaskPhotoMode::BinaryGt0);
    const auto values = ridge.cpu().to_vector();
    double retained_sum = 0.0;
    size_t retained_count = 0;
    for (size_t i = 0; i < pixels; ++i) {
        ASSERT_TRUE(std::isfinite(values[i]));
        if (keep[i] > 0.f) {
            retained_sum += values[i];
            ++retained_count;
        }
    }
    ASSERT_GT(retained_sum, 0.0);
    ASSERT_GT(retained_count, 0u);
    EXPECT_NEAR(retained_sum / retained_count, 1.0, 1e-3);
}

TEST(MaskThinStructureGuidance, MaskBoundaryDoesNotCreateAFlatSceneRidge) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "CUDA unavailable";
    const auto image = Tensor::full({3, height, width}, .5f, Device::CUDA);
    std::vector<float> mask_values(pixels, 1.f);
    for (size_t y = height / 3; y < 2 * height / 3; ++y)
        for (size_t x = width / 3; x < 2 * width / 3; ++x)
            mask_values[y * width + x] = 0.f;
    const auto mask = Tensor::from_vector(mask_values, {height, width}, Device::CUDA);
    RidgeWorkspace workspace;
    auto ridge = Tensor::empty({height, width}, Device::CUDA);
    ridge_with_optional_mask(image, mask, ridge, workspace, MaskPhotoMode::BinaryGt0);
    for (const float value : ridge.cpu().to_vector())
        EXPECT_NEAR(value, 0.f, 1e-6f);
}
