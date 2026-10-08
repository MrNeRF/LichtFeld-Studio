/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/image_codecs.hpp"
#include "core/image_io.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace {
    std::filesystem::path real_photo() {
        return std::filesystem::path(PROJECT_ROOT_PATH) / "data" / "bicycle" / "images_2" / "_DSC8679.JPG";
    }

    std::vector<std::uint8_t> decoded_pixels(const std::filesystem::path& path, const bool alpha, int& channels) {
        auto [pixels, width, height, decoded_channels] =
            alpha ? lfs::core::load_image_with_alpha(path) : lfs::core::load_image(path);
        channels = decoded_channels;
        std::vector<std::uint8_t> out(pixels, pixels + static_cast<std::size_t>(width) * height * decoded_channels);
        lfs::core::free_image(pixels);
        return out;
    }
} // namespace

// Catches a striped PNG stream that does not decode to the written pixels: broken stripe joins,
// wrong row filters or a bad combined checksum. Gray covers the single-stripe case.
TEST(ParallelPngWriter, RoundTripsARealPhotoExactly) {
    if (!std::filesystem::exists(real_photo())) {
        GTEST_SKIP() << "missing " << real_photo();
    }
    auto [rgb_pixels, width, height, rgb_channels] = lfs::core::load_image(real_photo());
    ASSERT_EQ(rgb_channels, 3);
    const std::size_t pixel_count = static_cast<std::size_t>(width) * height;
    ASSERT_GE(pixel_count, std::size_t{1} << 20) << "the photo must take the parallel path";
    const std::vector<std::uint8_t> rgb(rgb_pixels, rgb_pixels + pixel_count * 3);
    lfs::core::free_image(rgb_pixels);

    std::vector<std::uint8_t> gray(pixel_count);
    std::vector<std::uint8_t> rgba(pixel_count * 4);
    for (std::size_t i = 0; i < pixel_count; ++i) {
        gray[i] = rgb[i * 3];
        rgba[i * 4 + 0] = rgb[i * 3 + 0];
        rgba[i * 4 + 1] = rgb[i * 3 + 1];
        rgba[i * 4 + 2] = rgb[i * 3 + 2];
        rgba[i * 4 + 3] = rgb[i * 3 + 1];
    }

    const auto dir = std::filesystem::temp_directory_path() / "png_parallel_writer";
    std::filesystem::create_directories(dir);
    for (const int channels : {1, 3, 4}) {
        SCOPED_TRACE(channels);
        const auto& source = channels == 1 ? gray : channels == 3 ? rgb
                                                                  : rgba;
        const auto path = dir / ("photo_" + std::to_string(channels) + ".png");
        std::string error;
        ASSERT_TRUE(lfs::core::image_codecs::write_png(path, source.data(), width, height, channels, 8, 6,
                                                       std::string("parallel writer test"), error))
            << error;

        int decoded_channels = 0;
        const auto decoded = decoded_pixels(path, channels == 4, decoded_channels);
        ASSERT_EQ(decoded.size(), pixel_count * static_cast<std::size_t>(decoded_channels));
        std::size_t mismatches = 0;
        for (std::size_t i = 0; i < pixel_count; ++i) {
            for (int c = 0; c < decoded_channels; ++c) {
                const std::uint8_t expected = channels == 1 ? gray[i] : source[i * channels + c];
                mismatches += decoded[i * decoded_channels + c] != expected;
            }
        }
        EXPECT_EQ(mismatches, 0u);
    }
    std::filesystem::remove_all(dir);
}
