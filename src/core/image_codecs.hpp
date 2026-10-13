/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/atomic_file.hpp"
#include "core/export.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
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

    /// libjpeg scale denominator for a decode that is resampled to max_width afterwards: the largest
    /// reduction that keeps at least max_width pixels along the longer side.
    [[nodiscard]] constexpr unsigned jpeg_scale_denominator(const unsigned max_dimension, const unsigned max_width) {
        for (const unsigned denominator : {8u, 4u, 2u}) {
            if ((max_dimension + denominator - 1) / denominator >= max_width)
                return denominator;
        }
        return 1;
    }

    LFS_IMAGE_CODECS_API bool probe(const std::filesystem::path& path, Probe& result, std::string& error);
    LFS_IMAGE_CODECS_API bool probe_bounded_header(const std::filesystem::path& path, std::string& error);
    // Validate Radiance scanline payload completeness without decoding pixels.
    LFS_IMAGE_CODECS_API bool radiance_scanlines_complete(const std::filesystem::path& path);
    LFS_IMAGE_CODECS_API bool decode(const std::filesystem::path& path, Image& result, std::string& error);
    inline constexpr std::string_view tiff_sdr_encoding = "tiff16-sdr-v1";
    struct TiffMetadata {
        std::optional<std::string> description;
        std::optional<std::array<float, 8>> chromaticities;
        std::vector<std::uint16_t> transfer_function;
        int bits = 0, channels = 0;
        bool unsigned_integer = false;
        bool straight_alpha = false;
        bool matching_rgb_transfer = false;
    };
    // TIFF metadata only. Pixel decode APIs retain the original file encoding.
    LFS_IMAGE_CODECS_API bool read_tiff_metadata(const std::filesystem::path&, TiffMetadata&, std::string& error);
    LFS_IMAGE_CODECS_API bool decode_to_buffer(const std::filesystem::path& path, DecodeTarget& target, Probe& result, std::string& error);
    LFS_IMAGE_CODECS_API bool decode_memory(const std::uint8_t* data, size_t size, Image& result, std::string& error);
    LFS_IMAGE_CODECS_API bool decode_memory_to_buffer(const std::uint8_t* data, size_t size, DecodeTarget& target, Probe& result, std::string& error);

    // RGB uses 4:2:0 by default; full_chroma selects 4:4:4 for high-quality exports.
    LFS_IMAGE_CODECS_API bool write_jpeg(const std::filesystem::path& path, const std::uint8_t* data,
                                         int width, int height, int channels, int quality,
                                         const std::optional<std::string>& comment, std::string& error);
    LFS_IMAGE_CODECS_API bool write_jpeg(const std::filesystem::path& path, const std::uint8_t* data,
                                         int width, int height, int channels, int quality,
                                         const std::optional<std::string>& comment, std::string& error,
                                         bool full_chroma);
    LFS_IMAGE_CODECS_API bool write_png(const std::filesystem::path& path, const void* data,
                                        int width, int height, int channels, int bit_depth,
                                        int compression_level, const std::optional<std::string>& comment,
                                        std::string& error);
    LFS_IMAGE_CODECS_API bool write_tiff(const std::filesystem::path& path, const std::uint8_t* data,
                                         int width, int height, int channels, std::string& error);
    // Native-endian interleaved UINT16 samples, with a byte stride. Shares the
    // TIFF encoder and atomic publication service with legacy UINT8 output.
    struct TiffColorTags {
        std::optional<std::array<float, 8>> chromaticities;
        std::span<const std::uint16_t> transfer_function;
    };
    LFS_IMAGE_CODECS_API Status write_tiff_u16(const std::filesystem::path& path,
                                               std::span<const std::uint8_t> data, int width, int height, int channels,
                                               std::size_t row_stride, const std::optional<std::string>& description,
                                               const AtomicFileOptions& options, const TiffColorTags& color = {});

    // Packed display bytes: preserve samples, PNG/TIFF alpha and legacy codec
    // defaults. JPEG drops RGBA alpha; gray-alpha JPEG is unsupported. This is
    // shared atomic file service; legacy callers keep overwrite enabled and do
    // not create directories. No linear/EXR conversion.
    LFS_IMAGE_CODECS_API bool write_image_u8(const std::filesystem::path& path, const std::uint8_t* data,
                                             int width, int height, int channels, int jpeg_quality,
                                             const std::optional<std::string>& comment, std::string& error,
                                             bool jpeg_full_chroma = false);
    // Explicit publication/cancellation policy; encoding is shared with the
    // path-based codecs and does not close the atomic service's borrowed stream.
    LFS_IMAGE_CODECS_API Status write_image_u8(const std::filesystem::path& path, const std::uint8_t* data,
                                               int width, int height, int channels, int jpeg_quality,
                                               const std::optional<std::string>& comment, const AtomicFileOptions&,
                                               bool jpeg_full_chroma = false);

} // namespace lfs::core::image_codecs
