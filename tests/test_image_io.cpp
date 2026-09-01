/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/image_io.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <vector>

namespace {

    void append_u16(std::vector<std::uint8_t>& data, const std::uint16_t value) {
        data.push_back(static_cast<std::uint8_t>(value));
        data.push_back(static_cast<std::uint8_t>(value >> 8));
    }

    void append_u32(std::vector<std::uint8_t>& data, const std::uint32_t value) {
        data.push_back(static_cast<std::uint8_t>(value));
        data.push_back(static_cast<std::uint8_t>(value >> 8));
        data.push_back(static_cast<std::uint8_t>(value >> 16));
        data.push_back(static_cast<std::uint8_t>(value >> 24));
    }

    void append_tiff_tag(std::vector<std::uint8_t>& data,
                         const std::uint16_t tag,
                         const std::uint16_t type,
                         const std::uint32_t count,
                         const std::uint32_t value) {
        append_u16(data, tag);
        append_u16(data, type);
        append_u32(data, count);
        append_u32(data, value);
    }

    void write_float_tiff(const std::filesystem::path& path) {
        std::vector<std::uint8_t> data = {'I', 'I', 42, 0, 8, 0, 0, 0};
        append_u16(data, 10);
        append_tiff_tag(data, 256, 3, 1, 2);
        append_tiff_tag(data, 257, 3, 1, 1);
        append_tiff_tag(data, 258, 3, 1, 32);
        append_tiff_tag(data, 259, 3, 1, 1);
        append_tiff_tag(data, 262, 3, 1, 1);
        append_tiff_tag(data, 273, 4, 1, 134);
        append_tiff_tag(data, 277, 3, 1, 1);
        append_tiff_tag(data, 278, 4, 1, 1);
        append_tiff_tag(data, 279, 4, 1, 8);
        append_tiff_tag(data, 339, 3, 1, 3);
        append_u32(data, 0);
        append_u32(data, 0x43000000);
        append_u32(data, 0x42800000);
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        ASSERT_TRUE(file.good());
    }

    std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        EXPECT_TRUE(file);
        const auto size = static_cast<std::size_t>(file.tellg());
        file.seekg(0);
        std::vector<std::uint8_t> data(size);
        file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));
        EXPECT_TRUE(file.good() || file.eof());
        return data;
    }

} // namespace

TEST(ImageIoTest, ConvertsSixteenBitPngMemoryToUint8) {
    const auto path = std::filesystem::temp_directory_path() / "lfs_image_io_memory16.png";
    const std::vector<std::uint16_t> samples = {0, 257, 32768, 65535};
    ASSERT_TRUE(lfs::core::save_png(path, samples.data(), 2, 2, 1, 16, 0));
    const auto encoded = read_file(path);
    std::error_code ec;
    std::filesystem::remove(path, ec);

    auto [decoded, width, height, channels] =
        lfs::core::load_image_from_memory(encoded.data(), encoded.size());
    ASSERT_NE(decoded, nullptr);
    ASSERT_EQ(width, 2);
    ASSERT_EQ(height, 2);
    ASSERT_EQ(channels, 3);
    EXPECT_EQ(decoded[0], 0);
    EXPECT_EQ(decoded[3], 1);
    EXPECT_EQ(decoded[6], 128);
    EXPECT_EQ(decoded[9], 255);
    lfs::core::free_image(decoded);
}

TEST(ImageIoTest, FloatTiffInferenceRangeNormalization) {
    const auto path = std::filesystem::temp_directory_path() / "lfs_image_io_float_range.tiff";
    write_float_tiff(path);
    auto [decoded, width, height, channels] = lfs::core::load_image_float(path);
    std::error_code ec;
    std::filesystem::remove(path, ec);

    ASSERT_NE(decoded, nullptr);
    ASSERT_EQ(width, 2);
    ASSERT_EQ(height, 1);
    ASSERT_EQ(channels, 1);
    const float max_channel = std::max(decoded[0], decoded[1]);
    const float scale = max_channel > 255.5f ? 1.0f / 65535.0f
                        : max_channel > 1.5f ? 1.0f / 255.0f
                                             : 1.0f;
    decoded[0] = std::clamp(decoded[0] * scale, 0.0f, 1.0f);
    decoded[1] = std::clamp(decoded[1] * scale, 0.0f, 1.0f);
    EXPECT_FLOAT_EQ(decoded[0], 128.0f / 255.0f);
    EXPECT_FLOAT_EQ(decoded[1], 64.0f / 255.0f);
    lfs::core::free_image_float(decoded);
}
