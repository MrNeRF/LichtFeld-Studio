/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/image_io.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <stb_image_write.h>
#include <string>
#include <string_view>
#include <vector>

namespace {

    std::filesystem::path unique_temp_path(const std::string_view stem,
                                           const std::string_view extension) {
        static std::atomic_uint64_t sequence{0};
        const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
        return std::filesystem::temp_directory_path() /
               ("lfs_image_io_" + std::string(stem) + "_" + std::to_string(timestamp) + "_" +
                std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)) +
                std::string(extension));
    }

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

TEST(ImageIoTest, ThumbnailJpegUsesScaledDecode) {
    const auto path = unique_temp_path("thumbnail", ".jpg");
    constexpr int width = 1024;
    constexpr int height = 768;
    std::vector<std::uint8_t> source(static_cast<std::size_t>(width) * height * 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const auto offset = (static_cast<std::size_t>(y) * width + x) * 3;
            source[offset + 0] = static_cast<std::uint8_t>(x * 255 / (width - 1));
            source[offset + 1] = static_cast<std::uint8_t>(y * 255 / (height - 1));
            source[offset + 2] = static_cast<std::uint8_t>((x + y) * 255 / (width + height - 2));
        }
    }
    ASSERT_NE(stbi_write_jpg(path.string().c_str(), width, height, 3, source.data(), 95), 0);

    auto [full, full_width, full_height, full_channels] = lfs::core::load_image(path);
    auto [thumb, thumb_width, thumb_height, thumb_channels] =
        lfs::core::load_image_thumbnail(path, 256);
    std::error_code ec;
    std::filesystem::remove(path, ec);

    ASSERT_NE(full, nullptr);
    ASSERT_NE(thumb, nullptr);
    EXPECT_EQ(full_width, width);
    EXPECT_EQ(full_height, height);
    EXPECT_EQ(full_channels, 3);
    EXPECT_LE(std::max(thumb_width, thumb_height), 256);
    EXPECT_EQ(thumb_channels, 3);
    double mean_absolute_error = 0.0;
    for (int y = 0; y < thumb_height; ++y) {
        const double source_y = (y + 0.5) * full_height / thumb_height - 0.5;
        const double sy = std::clamp(source_y, 0.0, static_cast<double>(full_height - 1));
        const int y0 = static_cast<int>(sy);
        const int y1 = std::min(full_height - 1, y0 + 1);
        const double fy = sy - y0;
        for (int x = 0; x < thumb_width; ++x) {
            const double source_x = (x + 0.5) * full_width / thumb_width - 0.5;
            const double sx = std::clamp(source_x, 0.0, static_cast<double>(full_width - 1));
            const int x0 = static_cast<int>(sx);
            const int x1 = std::min(full_width - 1, x0 + 1);
            const double fx = sx - x0;
            for (int channel = 0; channel < 3; ++channel) {
                const auto at = [&](const int px, const int py) {
                    return static_cast<double>(full[(static_cast<std::size_t>(py) * full_width + px) * 3 + channel]);
                };
                const double top = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
                const double bottom = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
                const auto expected = static_cast<unsigned char>(std::lround(top * (1.0 - fy) + bottom * fy));
                const auto actual = thumb[(static_cast<std::size_t>(y) * thumb_width + x) * 3 + channel];
                mean_absolute_error += std::abs(static_cast<double>(actual) - expected) / 255.0;
            }
        }
    }
    mean_absolute_error /= static_cast<double>(thumb_width) * thumb_height * 3;
    EXPECT_LE(mean_absolute_error, 2.0 / 255.0);
    lfs::core::free_image(full);
    lfs::core::free_image(thumb);
}

TEST(ImageIoTest, ExpandsEightBitPngToUint16) {
    const auto source_path = std::filesystem::path(PROJECT_ROOT_PATH) / "data/bicycle/images_4/_DSC8679.JPG";
    auto [source, source_width, source_height, source_channels] = lfs::core::load_image(source_path);
    ASSERT_NE(source, nullptr);
    ASSERT_EQ(source_channels, 3);

    const auto path = std::filesystem::temp_directory_path() / "lfs_image_io_expand8.png";
    ASSERT_TRUE(lfs::core::save_png(path, source, source_width, source_height, source_channels, 8, 0));
    const auto pixel_count = static_cast<std::size_t>(source_width) * source_height * source_channels;

    auto [decoded, width, height, channels] = lfs::core::load_image_u16(path);
    std::error_code ec;
    std::filesystem::remove(path, ec);

    ASSERT_NE(decoded, nullptr);
    ASSERT_EQ(width, source_width);
    ASSERT_EQ(height, source_height);
    ASSERT_EQ(channels, 3);
    for (std::size_t i = 0; i < pixel_count; ++i)
        ASSERT_EQ(decoded[i], static_cast<std::uint16_t>(static_cast<unsigned int>(source[i]) * 257u)) << "sample " << i;
    lfs::core::free_image(decoded);
    lfs::core::free_image(source);
}

TEST(ImageIoTest, GrayAlphaPathAndMemoryExpansionMatch) {
    const auto source_path = std::filesystem::path(PROJECT_ROOT_PATH) / "data/bicycle/images_4/_DSC8679.JPG";
    auto [source, source_width, source_height, source_channels] = lfs::core::load_image(source_path);
    ASSERT_NE(source, nullptr);
    ASSERT_EQ(source_channels, 3);

    const auto path = std::filesystem::temp_directory_path() / "lfs_image_io_gray_alpha.png";
    const auto pixel_count = static_cast<std::size_t>(source_width) * source_height;
    std::vector<std::uint8_t> gray_alpha(pixel_count * 2);
    for (std::size_t i = 0; i < pixel_count; ++i) {
        const auto red = source[i * 3 + 0];
        const auto green = source[i * 3 + 1];
        const auto blue = source[i * 3 + 2];
        gray_alpha[i * 2 + 0] = static_cast<std::uint8_t>((299u * red + 587u * green + 114u * blue + 500u) / 1000u);
        gray_alpha[i * 2 + 1] = 200;
    }
    lfs::core::free_image(source);
    ASSERT_TRUE(lfs::core::save_png(path, gray_alpha.data(), source_width, source_height, 2, 8, 0));

    auto [from_path, path_width, path_height, path_channels] = lfs::core::load_image(path);
    ASSERT_NE(from_path, nullptr);
    ASSERT_EQ(path_channels, 3);
    const auto encoded = read_file(path);
    auto [from_memory, memory_width, memory_height, memory_channels] =
        lfs::core::load_image_from_memory(encoded.data(), encoded.size());
    std::error_code ec;
    std::filesystem::remove(path, ec);

    ASSERT_NE(from_memory, nullptr);
    EXPECT_EQ(path_width, source_width);
    EXPECT_EQ(path_height, source_height);
    EXPECT_EQ(memory_width, source_width);
    EXPECT_EQ(memory_height, source_height);
    EXPECT_EQ(memory_channels, 3);
    ASSERT_TRUE(std::equal(from_path, from_path + pixel_count * 3, from_memory));
    for (std::size_t i = 0; i < pixel_count; ++i)
        EXPECT_EQ(from_path[i * 3 + 2], (static_cast<unsigned int>(from_path[i * 3 + 0]) + from_path[i * 3 + 1]) / 2);

    lfs::core::free_image(from_path);
    lfs::core::free_image(from_memory);
}

TEST(ImageIoTest, DecodesTgaFromPathAndMemory) {
    const auto source_path = std::filesystem::path(PROJECT_ROOT_PATH) / "data/bicycle/images_4/_DSC8679.JPG";
    auto [source, source_width, source_height, source_channels] = lfs::core::load_image(source_path);
    ASSERT_NE(source, nullptr);
    ASSERT_EQ(source_channels, 3);

    const auto path = std::filesystem::temp_directory_path() / "lfs_image_io.tga";
    ASSERT_NE(stbi_write_tga(path.string().c_str(), source_width, source_height, source_channels, source), 0);
    lfs::core::free_image(source);

    auto [from_path, path_width, path_height, path_channels] = lfs::core::load_image(path);
    ASSERT_NE(from_path, nullptr);
    const auto encoded = read_file(path);
    auto [from_memory, memory_width, memory_height, memory_channels] =
        lfs::core::load_image_from_memory(encoded.data(), encoded.size());
    const auto [probe_width, probe_height, probe_channels] = lfs::core::get_image_info(path);
    std::error_code ec;
    std::filesystem::remove(path, ec);

    ASSERT_NE(from_memory, nullptr);
    EXPECT_EQ(path_width, source_width);
    EXPECT_EQ(path_height, source_height);
    EXPECT_EQ(path_channels, 3);
    EXPECT_EQ(memory_width, source_width);
    EXPECT_EQ(memory_height, source_height);
    EXPECT_EQ(memory_channels, 3);
    EXPECT_EQ(probe_width, source_width);
    EXPECT_EQ(probe_height, source_height);
    EXPECT_EQ(probe_channels, 3);
    EXPECT_TRUE(std::equal(from_path, from_path + static_cast<std::size_t>(source_width) * source_height * 3, from_memory));

    lfs::core::free_image(from_path);
    lfs::core::free_image(from_memory);
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
