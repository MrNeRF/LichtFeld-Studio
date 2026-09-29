/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mcp/render_capture_utils.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace {

    using lfs::mcp::MAX_CAPTURE_DIMENSION;
    using lfs::mcp::detail::resolve_capture_size;

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
        EXPECT_NE(huge.error().find("65536x65536"), std::string::npos) << huge.error();

        const auto small = lfs::mcp::encode_pixels_to_base64(pixels.data(), 4, 4, 4, 8, 8);
        ASSERT_TRUE(small) << small.error();
        EXPECT_FALSE(small->empty());
    }

} // namespace
