/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "training/losses/mask_loss.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace {
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Tensor;
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
        const auto image_a = Tensor::from_vector(image_values(false), {3, height, width}, Device::GPU);
        const auto image_b = Tensor::from_vector(image_values(true), {3, height, width}, Device::GPU);
        std::vector<float> keep(pixels, 1.f);
        for (size_t y = height / 3; y < 2 * height / 3; ++y)
            for (size_t x = width / 3; x < 2 * width / 3; ++x)
                keep[y * width + x] = 0.f;
        Tensor mask;
        if (bands) {
            auto cpu_mask = Tensor::empty({height, width}, Device::CPU, DataType::UInt8);
            for (size_t i = 0; i < pixels; ++i)
                cpu_mask.ptr<uint8_t>()[i] = keep[i] ? 255 : 128;
            mask = cpu_mask.to(Device::GPU);
        } else {
            mask = Tensor::from_vector(keep, {height, width}, Device::GPU);
        }
        auto first = Tensor::empty({height, width}, Device::GPU);
        auto second = Tensor::empty({height, width}, Device::GPU);
        lfs::training::losses::compute_edge_weight_map(image_a, mask, bands, first);
        lfs::training::losses::compute_edge_weight_map(image_b, mask, bands, second);
        const auto a = first.cpu().to_vector(), b = second.cpu().to_vector();
        float maximum = 0.f;
        for (size_t i = 0; i < pixels; ++i)
            if (keep[i] != 0) {
                ASSERT_TRUE(std::isfinite(a[i]));
                ASSERT_TRUE(std::isfinite(b[i]));
                maximum = std::max(maximum, std::abs(a[i] - b[i]));
            }
        EXPECT_LE(maximum, 1e-6f);
    }
} // namespace

TEST(MaskEdgeGuidance, IgnoredPixelsDoNotChangeRetainedEdgesAcrossModes) {
    compare_edges(false);
    compare_edges(true);

    const auto image = Tensor::from_vector(image_values(false), {3, height, width}, Device::GPU);
    const auto mask = Tensor::ones({height, width}, Device::GPU);
    auto bare = Tensor::empty({height, width}, Device::GPU);
    auto kept = Tensor::empty({height, width}, Device::GPU);
    lfs::training::losses::compute_edge_weight_map(image, {}, false, bare);
    lfs::training::losses::compute_edge_weight_map(image, mask, false, kept);
    EXPECT_EQ(bare.cpu().to_vector(), kept.cpu().to_vector());
}
