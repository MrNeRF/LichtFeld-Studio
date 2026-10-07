/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mcp/render_capture_utils.hpp"

#include <gtest/gtest.h>
#include <stb_image.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

namespace {

    using lfs::mcp::MAX_CAPTURE_DIMENSION;
    using lfs::mcp::detail::resolve_capture_size;

    std::vector<std::uint8_t> decode_png_rgba(const std::string& base64, int& width, int& height) {
        const auto png = lfs::core::base64_decode(base64);
        int channels = 0;
        const std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> pixels(
            stbi_load_from_memory(png.data(), static_cast<int>(png.size()), &width, &height, &channels, 4),
            &stbi_image_free);
        if (!pixels)
            return {};
        return {pixels.get(), pixels.get() + static_cast<std::size_t>(width) * height * 4};
    }

    TEST(McpRenderCaptureUtilsTest, ResolvesOmittedSidesFromTheSourceAspect) {
        EXPECT_EQ(resolve_capture_size(1600, 900, 0, 0), (std::pair{1600, 900}));
        EXPECT_EQ(resolve_capture_size(1600, 900, 800, 0), (std::pair{800, 450}));
        EXPECT_EQ(resolve_capture_size(1600, 900, 0, 450), (std::pair{800, 450}));
        EXPECT_EQ(resolve_capture_size(1600, 900, 1, 0), (std::pair{1, 1}));
    }

    TEST(McpRenderCaptureUtilsTest, RejectsNegativeAndOversizedCaptures) {
        EXPECT_FALSE(resolve_capture_size(1600, 900, -1, 0));
        EXPECT_FALSE(resolve_capture_size(1600, 900, 0, -1));
        EXPECT_FALSE(resolve_capture_size(1600, 900, MAX_CAPTURE_DIMENSION + 1, 10));
        EXPECT_FALSE(resolve_capture_size(1600, 900, 65536, 65536));
        // A height within the limit can still derive a width past it.
        const auto derived = resolve_capture_size(1600, 900, 0, MAX_CAPTURE_DIMENSION);
        ASSERT_FALSE(derived);
        EXPECT_NE(derived.error().find("29127x16384"), std::string::npos) << derived.error();
        EXPECT_TRUE(resolve_capture_size(1600, 900, MAX_CAPTURE_DIMENSION, 0));
    }

    TEST(McpRenderCaptureUtilsTest, EncodeRejectsOversizedCapturesWithoutAllocating) {
        const std::vector<std::uint8_t> pixels(4 * 4 * 4, 128);
        // 65536^2 RGBA is 16 GiB; the request must fail before any buffer is sized from it.
        const auto huge = lfs::mcp::encode_pixels_to_base64(pixels.data(), 4, 4, 4, 65536, 65536);
        ASSERT_FALSE(huge);
        EXPECT_EQ(huge.error().code(), lfs::ErrorCode::InvalidArgument);
        EXPECT_NE(huge.error().user_message().find("65536x65536"), std::string_view::npos) << huge.error().user_message();

        // A height within the limit whose aspect-derived width is not is the caller's error too.
        const std::vector<std::uint8_t> wide(16 * 4 * 4, 128);
        const auto derived = lfs::mcp::encode_pixels_to_base64(wide.data(), 16, 4, 4, 0, MAX_CAPTURE_DIMENSION);
        ASSERT_FALSE(derived);
        EXPECT_EQ(derived.error().code(), lfs::ErrorCode::InvalidArgument) << derived.error().user_message();

        const auto small = lfs::mcp::encode_pixels_to_base64(pixels.data(), 4, 4, 4, 8, 8);
        ASSERT_TRUE(small) << small.error().user_message();
        EXPECT_FALSE(small->empty());
    }

    // The Metal viewport publishes UInt8 RGBA. Rescaling it as [0, 1] floats saturated
    // every non-zero channel to 255.
    TEST(McpRenderCaptureUtilsTest, EncodesUInt8RenderTensorsWithoutRescaling) {
        const std::vector<std::uint8_t> rgba = {10, 128, 250, 255, 0, 64, 1, 255,
                                                200, 100, 50, 255, 7, 8, 9, 255};
        auto image = lfs::core::Tensor::empty({2, 2, 4}, lfs::core::Device::CPU, lfs::core::DataType::UInt8);
        std::copy(rgba.begin(), rgba.end(), image.ptr<std::uint8_t>());
        const auto encoded = lfs::mcp::encode_render_tensor_to_base64(image);
        ASSERT_TRUE(encoded) << encoded.error().user_message();
        int width = 0, height = 0;
        EXPECT_EQ(decode_png_rgba(*encoded, width, height), rgba);
        EXPECT_EQ(width, 2);
        EXPECT_EQ(height, 2);
    }

    TEST(McpRenderCaptureUtilsTest, EncodesFloatChwRenderTensorsFromUnitRange) {
        // 3x1x2 CHW: pixel 0 = (0.5, 0, 2), pixel 1 = (1, 0.25, -1); out-of-range values clamp.
        const std::vector<float> chw = {0.5f, 1.0f, 0.0f, 0.25f, 2.0f, -1.0f};
        const auto image = lfs::core::Tensor::from_vector(chw, {3, 1, 2}, lfs::core::Device::CPU);
        const auto encoded = lfs::mcp::encode_render_tensor_to_base64(image);
        ASSERT_TRUE(encoded) << encoded.error().user_message();
        int width = 0, height = 0;
        const std::vector<std::uint8_t> expected = {127, 0, 255, 255, 255, 63, 0, 255};
        EXPECT_EQ(decode_png_rgba(*encoded, width, height), expected);
        EXPECT_EQ(width, 2);
        EXPECT_EQ(height, 1);
    }

} // namespace
