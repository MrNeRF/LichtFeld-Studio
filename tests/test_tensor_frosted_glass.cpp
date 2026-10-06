/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_backend.hpp"
#include "gui/rmlui/tensor_frosted_glass.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

namespace {
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::GpuBackend;
    using lfs::core::GpuBackendScope;
    using lfs::core::Tensor;
    using lfs::vis::gui::TensorFrostedGlassBackdrop;
    using lfs::vis::gui::TensorFrostedGlassRect;
    using lfs::vis::gui::TensorFrostedGlassRegion;

    constexpr int kWidth = 40;
    constexpr int kHeight = 32;

    std::vector<std::uint8_t> pattern(const int width, const int height) {
        std::vector<std::uint8_t> pixels(std::size_t(width) * height * 4);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const std::size_t i = (std::size_t(y) * width + x) * 4;
                pixels[i] = static_cast<std::uint8_t>((x * 17 + y * 3) & 255);
                pixels[i + 1] = static_cast<std::uint8_t>((x * 5 + y * 23) & 255);
                pixels[i + 2] = static_cast<std::uint8_t>(((x / 3 + y / 2) & 1) ? 240 : 12);
                pixels[i + 3] = 255;
            }
        }
        return pixels;
    }

    std::array<float, 4> texel(const std::vector<std::uint8_t>& source,
                               const int width, const int height,
                               int x, int y) {
        x = std::clamp(x, 0, width - 1);
        y = std::clamp(y, 0, height - 1);
        const std::size_t i = (std::size_t(y) * width + x) * 4;
        return {float(source[i]), float(source[i + 1]),
                float(source[i + 2]), float(source[i + 3])};
    }

    std::array<float, 4> sample(const std::vector<std::uint8_t>& source,
                                const int width, const int height,
                                const float u, const float v) {
        const float px = u * width - 0.5f;
        const float py = v * height - 0.5f;
        const int x = static_cast<int>(std::floor(px));
        const int y = static_cast<int>(std::floor(py));
        const float fx = px - std::floor(px);
        const float fy = py - std::floor(py);
        const auto a = texel(source, width, height, x, y);
        const auto b = texel(source, width, height, x + 1, y);
        const auto c = texel(source, width, height, x, y + 1);
        const auto d = texel(source, width, height, x + 1, y + 1);
        std::array<float, 4> value{};
        for (int channel = 0; channel < 4; ++channel) {
            const float top = a[channel] + (b[channel] - a[channel]) * fx;
            const float bottom = c[channel] + (d[channel] - c[channel]) * fx;
            value[channel] = top + (bottom - top) * fy;
        }
        return value;
    }

    std::vector<std::uint8_t> resample(const std::vector<std::uint8_t>& source,
                                       const int source_width, const int source_height,
                                       const int width, const int height) {
        std::vector<std::uint8_t> output(std::size_t(width) * height * 4);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const auto value = sample(source, source_width, source_height,
                                          (x + 0.5f) / width, (y + 0.5f) / height);
                const std::size_t i = (std::size_t(y) * width + x) * 4;
                for (int channel = 0; channel < 4; ++channel)
                    output[i + channel] = static_cast<std::uint8_t>(
                        std::clamp(value[channel], 0.0f, 255.0f) + 0.5f);
            }
        }
        return output;
    }

    bool inside(const std::vector<TensorFrostedGlassRect>& rectangles,
                const int x, const int y) {
        return std::ranges::any_of(rectangles, [&](const auto& rect) {
            return x >= std::floor(rect.left) && x < std::ceil(rect.right) &&
                   y >= std::floor(rect.top) && y < std::ceil(rect.bottom);
        });
    }

    TEST(TensorFrostedGlassContracts, BilinearChainAndRoundedRegionContract) {
        const auto source = pattern(kWidth, kHeight);
        const int primary_width = std::max(1, kWidth / 4);
        const int primary_height = std::max(1, kHeight / 4);
        const int secondary_width = std::max(1, kWidth / 8);
        const int secondary_height = std::max(1, kHeight / 8);
        const auto down4 = resample(source, kWidth, kHeight,
                                    primary_width, primary_height);
        const auto down8 = resample(down4, primary_width, primary_height,
                                    secondary_width, secondary_height);
        const auto expected = resample(down8, secondary_width, secondary_height,
                                       primary_width, primary_height);

        std::vector<std::pair<GpuBackend, std::vector<std::uint8_t>>> outputs;
        for (const GpuBackend backend : lfs::core::kCompiledGpuBackends) {
            if (!lfs::core::gpu_backend_available(backend))
                continue;
            GpuBackendScope scope(backend);
            const Tensor input = Tensor::from_blob(
                                     const_cast<std::uint8_t*>(source.data()),
                                     {kHeight, kWidth, 4}, Device::CPU, DataType::UInt8)
                                     .to(Device::GPU);
            TensorFrostedGlassBackdrop backdrop;
            auto status = backdrop.update(input);
            ASSERT_TRUE(status) << lfs::core::gpu_backend_name(backend) << ": "
                                << status.error().detail();
            EXPECT_EQ(backdrop.image().dtype(), DataType::UInt8);
            EXPECT_EQ(backdrop.image().shape(),
                      (lfs::core::TensorShape{std::size_t(primary_height),
                                              std::size_t(primary_width), 4}));
            EXPECT_EQ(backdrop.bytes(),
                      std::size_t(primary_width * primary_height +
                                  secondary_width * secondary_height) *
                          4);
            const auto actual = backdrop.image().to_vector_uint8();
            int max_error = 0;
            for (std::size_t i = 0; i < actual.size(); ++i)
                max_error = std::max(max_error, std::abs(int(actual[i]) - int(expected[i])));
            std::cout << "frosted backdrop " << lfs::core::gpu_backend_name(backend)
                      << " CPU max error=" << max_error << '\n';
            EXPECT_LE(max_error, 1);
            outputs.emplace_back(backend, actual);
        }
        ASSERT_FALSE(outputs.empty());
        for (std::size_t i = 1; i < outputs.size(); ++i) {
            int max_error = 0;
            for (std::size_t pixel = 0; pixel < outputs.front().second.size(); ++pixel)
                max_error = std::max(max_error,
                                     std::abs(int(outputs.front().second[pixel]) - int(outputs[i].second[pixel])));
            std::cout << "frosted parity "
                      << lfs::core::gpu_backend_name(outputs.front().first) << " vs "
                      << lfs::core::gpu_backend_name(outputs[i].first)
                      << ": max=" << max_error << '\n';
            EXPECT_LE(max_error, 1);
        }

        const std::array regions{
            TensorFrostedGlassRegion{2, 3, 12, 9, 0},
            TensorFrostedGlassRegion{20, 8, 16, 16, 4},
        };
        const auto rectangles = lfs::vis::gui::frostedGlassClipRects(
            regions, float(kWidth), float(kHeight));
        ASSERT_EQ(rectangles.size(), 3u);
        std::vector<std::uint8_t> composed(std::size_t(kWidth) * kHeight * 4, 37);
        constexpr float inset = 1.25f;
        for (int y = 0; y < kHeight; ++y) {
            for (int x = 0; x < kWidth; ++x) {
                const std::size_t i = (std::size_t(y) * kWidth + x) * 4;
                if (!inside(rectangles, x, y)) {
                    for (int channel = 0; channel < 4; ++channel)
                        EXPECT_EQ(composed[i + channel], 37) << x << ',' << y;
                    continue;
                }
                const auto value = sample(
                    outputs.front().second, primary_width, primary_height,
                    (x + 0.5f + inset) / (kWidth + 2.0f * inset),
                    (y + 0.5f + inset) / (kHeight + 2.0f * inset));
                for (int channel = 0; channel < 4; ++channel) {
                    composed[i + channel] = static_cast<std::uint8_t>(
                        std::clamp(value[channel], 0.0f, 255.0f) + 0.5f);
                    EXPECT_NEAR(composed[i + channel], value[channel], 0.51f)
                        << x << ',' << y << " channel " << channel;
                }
            }
        }
    }
} // namespace
