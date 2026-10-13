/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "training/kernels/image_kernels.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace {
    using namespace lfs::core;
    using namespace lfs::training::kernels;
    constexpr size_t height = 97;
    constexpr size_t width = 129;
    constexpr size_t pixels = height * width;

    std::vector<float> image_values(const bool changed) {
        std::vector<float> values(3 * pixels);
        for (size_t y = 0; y < height; ++y)
            for (size_t x = 0; x < width; ++x) {
                const size_t i = y * width + x;
                const bool hidden = y >= height / 3 && y < 2 * height / 3 &&
                                    x >= width / 3 && x < 2 * width / 3;
                for (size_t c = 0; c < 3; ++c)
                    values[c * pixels + i] = changed && hidden ? (c == 0 ? 1.f : 0.f)
                                                               : static_cast<float>((x * 17 + y * 31 + c * 53) % 211) / 255.f;
            }
        return values;
    }

    void compare_edges(const bool bands) {
        const auto source_a = image_values(false), source_b = image_values(true);
        const auto image_a = Tensor::from_vector(source_a, {3, height, width}, Device::CUDA);
        const auto image_b = Tensor::from_vector(source_b, {3, height, width}, Device::CUDA);
        std::vector<float> keep(pixels, 1.f);
        for (size_t y = height / 3; y < 2 * height / 3; ++y)
            for (size_t x = width / 3; x < 2 * width / 3; ++x)
                keep[y * width + x] = 0.f;
        Tensor mask;
        if (bands) {
            auto cpu_mask = Tensor::empty({height, width}, Device::CPU, DataType::UInt8);
            for (size_t i = 0; i < pixels; ++i)
                cpu_mask.ptr<uint8_t>()[i] = keep[i] ? 255 : 128;
            mask = cpu_mask.to(Device::CUDA);
        } else {
            mask = Tensor::from_vector(keep, {height, width}, Device::CUDA);
        }
        const auto mode = bands ? MaskPhotoMode::SegmentAndIgnore : MaskPhotoMode::BinaryGt0;
        auto first = Tensor::empty({height, width}, Device::CUDA);
        auto second = Tensor::empty({height, width}, Device::CUDA);
        compute_edge_weight_map(image_a, mask, mode, first, nullptr);
        compute_edge_weight_map(image_b, mask, mode, second, nullptr);
        const auto a = first.cpu().to_vector(), b = second.cpu().to_vector();
        float maximum = 0.f;
        for (size_t i = 0; i < pixels; ++i)
            if (keep[i] != 0) {
                EXPECT_TRUE(std::isfinite(a[i]));
                EXPECT_TRUE(std::isfinite(b[i]));
                maximum = std::max(maximum, std::abs(a[i] - b[i]));
            }
        EXPECT_LE(maximum, 1e-6f);
    }
} // namespace

TEST(MaskEdgeGuidance, IgnoredPixelsDoNotChangeRetainedEdgesAcrossModes) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "CUDA unavailable";

    compare_edges(false);
    compare_edges(true);

    const auto image = Tensor::from_vector(image_values(false), {3, height, width}, Device::CUDA);
    const auto mask = Tensor::ones({height, width}, Device::CUDA);
    auto bare = Tensor::empty({height, width}, Device::CUDA);
    auto kept = Tensor::empty({height, width}, Device::CUDA);
    compute_edge_weight_map(image, {}, MaskPhotoMode::BinaryGt0, bare, nullptr);
    compute_edge_weight_map(image, mask, MaskPhotoMode::BinaryGt0, kept, nullptr);
    EXPECT_EQ(bare.cpu().to_vector(), kept.cpu().to_vector());
}
