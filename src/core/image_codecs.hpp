/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace lfs::core::image_codecs {

    enum class SampleType { UInt8,
                            UInt16,
                            Float32 };

    struct Image {
        int width = 0;
        int height = 0;
        int channels = 0;
        SampleType sample_type = SampleType::UInt8;
        std::vector<std::uint8_t> data;
    };

    struct Probe {
        int width = 0;
        int height = 0;
        int channels = 0;
        SampleType sample_type = SampleType::UInt8;
    };

    using DecodeAllocator = void* (*)(std::size_t bytes, void* user);

    struct DecodeTarget {
        int channels = 0;
        SampleType sample_type = SampleType::UInt8;
        void* data = nullptr;
        DecodeAllocator allocate = nullptr;
        void* user = nullptr;
        // JPEG decoders may use DCT scaling before writing to the target. Other
        // codecs leave this at zero and the image I/O layer resamples afterward.
        int max_width = 0;
    };

    bool probe(const std::filesystem::path& path, Probe& result, std::string& error);
    bool decode(const std::filesystem::path& path, Image& result, std::string& error);
    bool decode_to_buffer(const std::filesystem::path& path, DecodeTarget& target, Probe& result, std::string& error);
    bool decode_memory(const std::uint8_t* data, size_t size, Image& result, std::string& error);
    bool decode_memory_to_buffer(const std::uint8_t* data, size_t size, DecodeTarget& target, Probe& result, std::string& error);
    bool decode_jpeg_memory(const std::uint8_t* data, size_t size, Image& result, std::string& error);

    bool write_jpeg(const std::filesystem::path& path, const std::uint8_t* data,
                    int width, int height, int channels, int quality,
                    const std::optional<std::string>& comment, std::string& error);
    bool write_png(const std::filesystem::path& path, const void* data,
                   int width, int height, int channels, int bit_depth,
                   int compression_level, const std::optional<std::string>& comment,
                   std::string& error);
    bool write_tiff(const std::filesystem::path& path, const std::uint8_t* data,
                    int width, int height, int channels, std::string& error);

} // namespace lfs::core::image_codecs
