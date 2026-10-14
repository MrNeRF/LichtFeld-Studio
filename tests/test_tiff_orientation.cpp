/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/image_codecs.hpp"
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <gtest/gtest.h>
#include <tiffio.h>
#include <vector>

namespace {
    namespace codec = lfs::core::image_codecs;
    constexpr int kW = 6, kH = 4, kC = 3;

    std::vector<std::uint8_t> stored(const int bits) {
        std::vector<std::uint8_t> bytes(static_cast<std::size_t>(kW * kH * kC * (bits / 8)));
        if (bits == 8) {
            for (int i = 0; i < kW * kH; ++i) {
                auto* pixel = bytes.data() + static_cast<std::size_t>(i) * 3;
                const int x = i % kW, y = i / kW;
                pixel[0] = static_cast<std::uint8_t>(x * 40 + 1);
                pixel[1] = static_cast<std::uint8_t>(y * 60 + 3);
                pixel[2] = static_cast<std::uint8_t>(x * 7 + y * 11 + 5);
            }
            return bytes;
        }
        auto* pixel = reinterpret_cast<std::uint16_t*>(bytes.data());
        for (int i = 0; i < kW * kH; ++i) {
            pixel[i * 3] = static_cast<std::uint16_t>(1000 + i);
            pixel[i * 3 + 1] = static_cast<std::uint16_t>(4000 + i * 3);
            pixel[i * 3 + 2] = static_cast<std::uint16_t>(9000 + i * 5);
        }
        return bytes;
    }

    void write_tiff(const std::filesystem::path& path, const int orientation, const bool tiled, const int bits,
                    const std::vector<std::uint8_t>& samples) {
        TIFF* tiff = TIFFOpen(path.string().c_str(), "w");
        ASSERT_NE(tiff, nullptr);
        TIFFSetField(tiff, TIFFTAG_IMAGEWIDTH, kW);
        TIFFSetField(tiff, TIFFTAG_IMAGELENGTH, kH);
        TIFFSetField(tiff, TIFFTAG_SAMPLESPERPIXEL, kC);
        TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, bits);
        TIFFSetField(tiff, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
        TIFFSetField(tiff, TIFFTAG_ORIENTATION, orientation);
        TIFFSetField(tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
        TIFFSetField(tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(tiff, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
        const auto row = static_cast<std::size_t>(kW * kC * (bits / 8));
        if (tiled) {
            TIFFSetField(tiff, TIFFTAG_TILEWIDTH, 16);
            TIFFSetField(tiff, TIFFTAG_TILELENGTH, 16);
            std::vector<std::uint8_t> tile(static_cast<std::size_t>(TIFFTileSize(tiff)));
            const auto stride = static_cast<std::size_t>(TIFFTileRowSize(tiff));
            for (int y = 0; y < kH; ++y)
                std::memcpy(tile.data() + static_cast<std::size_t>(y) * stride, samples.data() + static_cast<std::size_t>(y) * row, row);
            ASSERT_GE(TIFFWriteEncodedTile(tiff, 0, tile.data(), static_cast<tmsize_t>(tile.size())), 0);
        } else {
            TIFFSetField(tiff, TIFFTAG_ROWSPERSTRIP, kH);
            for (int y = 0; y < kH; ++y)
                ASSERT_GE(TIFFWriteScanline(tiff, const_cast<std::uint8_t*>(samples.data() + static_cast<std::size_t>(y) * row), y, 0), 0);
        }
        TIFFClose(tiff);
    }

    void write_palette_tile(const std::filesystem::path& path, const std::vector<std::uint8_t>& rgb) {
        TIFF* tiff = TIFFOpen(path.string().c_str(), "w");
        ASSERT_NE(tiff, nullptr);
        TIFFSetField(tiff, TIFFTAG_IMAGEWIDTH, kW);
        TIFFSetField(tiff, TIFFTAG_IMAGELENGTH, kH);
        TIFFSetField(tiff, TIFFTAG_SAMPLESPERPIXEL, 1);
        TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, 8);
        TIFFSetField(tiff, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
        TIFFSetField(tiff, TIFFTAG_ORIENTATION, ORIENTATION_BOTRIGHT);
        TIFFSetField(tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_PALETTE);
        TIFFSetField(tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(tiff, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
        TIFFSetField(tiff, TIFFTAG_TILEWIDTH, 16);
        TIFFSetField(tiff, TIFFTAG_TILELENGTH, 16);
        std::uint16_t red[256] = {};
        std::uint16_t green[256] = {};
        std::uint16_t blue[256] = {};
        std::vector<std::uint8_t> indices(static_cast<std::size_t>(kW * kH));
        for (int i = 0; i < kW * kH; ++i) {
            indices[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(i);
            red[i] = static_cast<std::uint16_t>(rgb[static_cast<std::size_t>(i) * 3] * 257);
            green[i] = static_cast<std::uint16_t>(rgb[static_cast<std::size_t>(i) * 3 + 1] * 257);
            blue[i] = static_cast<std::uint16_t>(rgb[static_cast<std::size_t>(i) * 3 + 2] * 257);
        }
        TIFFSetField(tiff, TIFFTAG_COLORMAP, red, green, blue);
        std::vector<std::uint8_t> tile(static_cast<std::size_t>(TIFFTileSize(tiff)));
        const auto stride = static_cast<std::size_t>(TIFFTileRowSize(tiff));
        for (int y = 0; y < kH; ++y)
            std::memcpy(tile.data() + static_cast<std::size_t>(y) * stride, indices.data() + static_cast<std::size_t>(y) * kW, kW);
        ASSERT_GE(TIFFWriteEncodedTile(tiff, 0, tile.data(), static_cast<tmsize_t>(tile.size())), 0);
        TIFFClose(tiff);
    }

    TEST(TiffOrientation, StoredPixelsIgnoreOrientationTag) {
        const auto rgb8 = stored(8);
        const auto rgb16 = stored(16);
        const int cases[][2] = {{6, 8}, {3, 8}, {4, 8}, {1, 8}, {6, 16}};
        for (const auto& item : cases) {
            const auto& samples = item[1] == 16 ? rgb16 : rgb8;
            for (const bool tiled : {false, true}) {
                const auto path = std::filesystem::temp_directory_path() /
                                  ("lfs_tiff_or_" + std::to_string(item[0]) + "_" + std::to_string(item[1]) + (tiled ? "t" : "s") + ".tif");
                write_tiff(path, item[0], tiled, item[1], samples);
                codec::Image image;
                std::string error;
                ASSERT_TRUE(codec::decode(path, image, error)) << error;
                EXPECT_EQ(image.channels, kC);
                EXPECT_EQ(image.data, samples) << item[0] << " " << item[1] << " tiled=" << tiled;
                std::vector<std::uint8_t> direct(samples.size());
                codec::DecodeTarget target{kC, item[1] == 16 ? codec::SampleType::UInt16 : codec::SampleType::UInt8, direct.data(), nullptr, nullptr};
                codec::Probe probe;
                ASSERT_TRUE(codec::decode_to_buffer(path, target, probe, error)) << error;
                EXPECT_EQ(direct, samples) << item[0] << " direct tiled=" << tiled;
                std::filesystem::remove(path);
            }
        }

        const auto palette_path = std::filesystem::temp_directory_path() / "lfs_tiff_or_palette.tif";
        const auto strip_path = std::filesystem::temp_directory_path() / "lfs_tiff_or_palette_strip.tif";
        write_palette_tile(palette_path, rgb8);
        write_tiff(strip_path, 1, false, 8, rgb8);
        codec::Image palette;
        codec::Image strip;
        std::string error;
        ASSERT_TRUE(codec::decode(palette_path, palette, error)) << error;
        ASSERT_TRUE(codec::decode(strip_path, strip, error)) << error;
        ASSERT_EQ(palette.channels, 4);
        ASSERT_EQ(strip.data, rgb8);
        ASSERT_EQ(palette.data.size(), rgb8.size() / 3 * 4);
        for (std::size_t i = 0; i < rgb8.size() / 3; ++i) {
            EXPECT_EQ(palette.data[i * 4 + 0], strip.data[i * 3 + 0]) << i;
            EXPECT_EQ(palette.data[i * 4 + 1], strip.data[i * 3 + 1]) << i;
            EXPECT_EQ(palette.data[i * 4 + 2], strip.data[i * 3 + 2]) << i;
            EXPECT_EQ(palette.data[i * 4 + 3], 255) << i;
        }
        std::filesystem::remove(palette_path);
        std::filesystem::remove(strip_path);
    }
} // namespace
