/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// The export band unpack (bytes HWC to float CHW) and pack (float CHW plus
// alpha to bytes HWC) are tensor programs; both match the byte reference
// exactly, and pack then unpack round-trips the bytes.

#include "core/tensor.hpp"
#include "rendering/export_post_process.hpp"
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {
    using namespace lfs::core;

    constexpr size_t kHeight = 37;
    constexpr size_t kWidth = 53;

    Tensor byte_band(const size_t channels) {
        std::vector<int> values(kHeight * kWidth * channels);
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = static_cast<int>((i * 97 + 13) % 256);
        return Tensor::from_vector(values, TensorShape{kHeight, kWidth, channels}, Device::CPU)
            .to(DataType::UInt8)
            .to(Device::GPU);
    }

    uint8_t reference_byte(const float value) {
        return static_cast<uint8_t>(std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
    }
} // namespace

TEST(ExportBandPack, UnpackMatchesTheByteReferenceWithAndWithoutAlpha) {
    for (const size_t channels : {size_t{3}, size_t{4}}) {
        SCOPED_TRACE(channels);
        const Tensor band = byte_band(channels);
        const auto bytes = band.to_vector_uint8();
        Tensor rgb;
        Tensor alpha;
        const auto status = lfs::rendering::unpackU8HwcBandToChwFloat(band, rgb, &alpha, nullptr);
        ASSERT_TRUE(status.has_value()) << status.error();
        ASSERT_EQ(rgb.shape(), TensorShape({size_t{3}, kHeight, kWidth}));
        const auto rgb_values = rgb.to_vector();
        for (size_t p = 0; p < kHeight * kWidth; ++p) {
            for (size_t c = 0; c < 3; ++c) {
                EXPECT_EQ(rgb_values[c * kHeight * kWidth + p], static_cast<float>(bytes[p * channels + c]) * (1.0f / 255.0f));
            }
        }
        if (channels == 4) {
            ASSERT_EQ(alpha.shape(), TensorShape({kHeight, kWidth}));
            const auto alpha_values = alpha.to_vector();
            for (size_t p = 0; p < kHeight * kWidth; ++p)
                EXPECT_EQ(alpha_values[p], static_cast<float>(bytes[p * 4 + 3]) * (1.0f / 255.0f));
        } else {
            EXPECT_FALSE(alpha.is_valid());
        }
    }
}

TEST(ExportBandPack, PackRoundsClampsAndRoundTripsBytes) {
    std::vector<float> rgb_values(3 * kHeight * kWidth);
    for (size_t i = 0; i < rgb_values.size(); ++i)
        rgb_values[i] = static_cast<float>(static_cast<int>(i % 301) - 20) / 260.0f;
    std::vector<float> alpha_values(kHeight * kWidth);
    for (size_t i = 0; i < alpha_values.size(); ++i)
        alpha_values[i] = static_cast<float>(i % 257) / 256.0f;
    const Tensor rgb = Tensor::from_vector(rgb_values, TensorShape{size_t{3}, kHeight, kWidth}, Device::CPU).to(Device::GPU);
    const Tensor alpha = Tensor::from_vector(alpha_values, TensorShape{kHeight, kWidth}, Device::CPU).to(Device::GPU);
    for (const bool with_alpha : {false, true}) {
        SCOPED_TRACE(with_alpha);
        Tensor band;
        const auto status = lfs::rendering::packChwFloatBandToU8Hwc(rgb, with_alpha ? &alpha : nullptr, band, nullptr);
        ASSERT_TRUE(status.has_value()) << status.error();
        const size_t channels = with_alpha ? 4 : 3;
        ASSERT_EQ(band.shape(), TensorShape({kHeight, kWidth, channels}));
        const auto bytes = band.to_vector_uint8();
        for (size_t p = 0; p < kHeight * kWidth; ++p) {
            for (size_t c = 0; c < 3; ++c)
                EXPECT_EQ(bytes[p * channels + c], reference_byte(rgb_values[c * kHeight * kWidth + p])) << "pixel=" << p;
            if (with_alpha)
                EXPECT_EQ(bytes[p * 4 + 3], reference_byte(alpha_values[p])) << "pixel=" << p;
        }
    }
    const Tensor source = byte_band(4);
    Tensor unpacked_rgb;
    Tensor unpacked_alpha;
    ASSERT_TRUE(lfs::rendering::unpackU8HwcBandToChwFloat(source, unpacked_rgb, &unpacked_alpha, nullptr).has_value());
    Tensor repacked;
    ASSERT_TRUE(lfs::rendering::packChwFloatBandToU8Hwc(unpacked_rgb, &unpacked_alpha, repacked, nullptr).has_value());
    EXPECT_EQ(repacked.to_vector_uint8(), source.to_vector_uint8());
}
