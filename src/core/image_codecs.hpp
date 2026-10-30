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

    inline constexpr std::size_t max_xmp_packet_bytes = 32u * 1024 * 1024;
    // Packet framing only; native callers remain responsible for valid UTF-8 XML.
    [[nodiscard]] constexpr bool valid_xmp_packet(std::string_view packet) {
        return packet.size() <= max_xmp_packet_bytes && packet.find('\0') == std::string_view::npos;
    }

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
    // Display policy is dispatched here for profiled codecs. JXL defaults to
    // canonical sRGB even without display; native JXL access is explicit.
    LFS_IMAGE_CODECS_API bool decode(const std::filesystem::path& path, Image& result, std::string& error, bool display);
    inline constexpr std::string_view tiff_sdr_encoding = "tiff16-sdr-v1";
    // Immutable, lazily initialized once per supported transfer; shared by
    // export and metadata validation. Unknown transfers return an empty span.
    LFS_IMAGE_CODECS_API std::span<const std::uint16_t> tiff_transfer_function(std::string_view transfer);
    struct TiffMetadata {
        std::optional<std::string> description;
        std::optional<std::array<float, 8>> chromaticities;
        int bits = 0, channels = 0;
        bool unsigned_integer = false;
        bool straight_alpha = false;
        bool matching_rgb_transfer = false;
    };
    // TIFF metadata only. Pixel decode APIs retain the original file encoding.
    LFS_IMAGE_CODECS_API bool read_tiff_metadata(const std::filesystem::path&, TiffMetadata&, std::string& error);
    LFS_IMAGE_CODECS_API bool decode_to_buffer(const std::filesystem::path& path, DecodeTarget& target, Probe& result, std::string& error);
    LFS_IMAGE_CODECS_API bool decode_to_buffer(const std::filesystem::path& path, DecodeTarget& target, Probe& result, std::string& error, bool display);
    // EXR memory reads use the file decoder's Float32 RGBA sample policy.
    // EXR targets require Float32 RGB/RGBA (zero channels selects RGBA) and
    // stage decoding before allocating/writing caller storage.
    LFS_IMAGE_CODECS_API bool decode_memory(const std::uint8_t* data, size_t size, Image& result, std::string& error);
    LFS_IMAGE_CODECS_API bool decode_memory(const std::uint8_t* data, size_t size, Image& result, std::string& error, bool display);
    LFS_IMAGE_CODECS_API bool decode_memory_to_buffer(const std::uint8_t* data, size_t size, DecodeTarget& target, Probe& result, std::string& error);
    LFS_IMAGE_CODECS_API bool decode_memory_to_buffer(const std::uint8_t* data, size_t size, DecodeTarget& target, Probe& result, std::string& error, bool display);

    struct JpegEncodeOptions {
        int quality = 95; // Clamped to [1, 100], matching the file writer.
        bool full_chroma = false;
        std::optional<std::string> comment;
        std::size_t max_output_bytes = 64 * 1024 * 1024;
    };
    // Synchronous encoding of borrowed interleaved UINT8 gray/RGB rows. No color,
    // orientation or alpha conversion. Output is replaced only on success and
    // owns its bytes; the encoded-byte limit is enforced while producing them.
    // Uses the same encoder/settings as write_jpeg; no temporary file or GPU.
    [[nodiscard]] LFS_IMAGE_CODECS_API Status encode_jpeg_memory(
        std::span<const std::uint8_t> data, int width, int height, int channels,
        std::size_t row_stride, std::vector<std::uint8_t>& output,
        const JpegEncodeOptions& options = {});

    // RGB uses 4:2:0 by default; full_chroma selects 4:4:4 for high-quality exports.
    LFS_IMAGE_CODECS_API bool write_jpeg(const std::filesystem::path& path, const std::uint8_t* data,
                                         int width, int height, int channels, int quality,
                                         const std::optional<std::string>& comment, std::string& error);
    LFS_IMAGE_CODECS_API bool write_jpeg(const std::filesystem::path& path, const std::uint8_t* data,
                                         int width, int height, int channels, int quality,
                                         const std::optional<std::string>& comment, std::string& error,
                                         bool full_chroma);
    struct PngEncodeOptions {
        int compression_level = 6;          // Clamped to [0, 9], matching the file writer.
        std::optional<std::string> comment; // Latin-1 tEXt bytes, without NUL.
        std::size_t max_output_bytes = 256 * 1024 * 1024;
    };
    // Borrowed interleaved gray/gray-alpha/RGB/RGBA UINT8 or native-endian
    // UINT16 samples, with an explicit byte stride (unaligned rows supported).
    // No color, alpha or orientation conversion. Owns output on success only;
    // failures preserve the previous output, including input/output aliasing.
    // Uses the file writer's serial/parallel policy. The encoded-byte limit
    // bounds output production, not encoder working memory or concurrent calls.
    [[nodiscard]] LFS_IMAGE_CODECS_API Status encode_png_memory(
        std::span<const std::uint8_t> data, int width, int height, int channels,
        int bit_depth, std::size_t row_stride, std::vector<std::uint8_t>& output,
        const PngEncodeOptions& options = {});

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
    struct TiffEncodeOptions {
        std::optional<std::string> description;
        TiffColorTags color;
        std::size_t max_output_bytes = 256 * 1024 * 1024;
    };
    // Classic TIFF, existing LZW/strip policy, interleaved UINT8 or native-endian
    // UINT16 gray/gray-alpha/RGB/RGBA. Explicit byte stride; unaligned rows and
    // input/output aliasing supported. No sample/alpha/orientation conversion.
    // Owns output only on success; failures preserve the previous output.
    // The byte limit bounds the seekable encoded extent, not codec working RAM.
    [[nodiscard]] LFS_IMAGE_CODECS_API Status encode_tiff_memory(
        std::span<const std::uint8_t> data, int width, int height, int channels,
        int bit_depth, std::size_t row_stride, std::vector<std::uint8_t>& output,
        const TiffEncodeOptions& options = {});
    LFS_IMAGE_CODECS_API Status write_tiff_u16(const std::filesystem::path& path,
                                               std::span<const std::uint8_t> data, int width, int height, int channels,
                                               std::size_t row_stride, const std::optional<std::string>& description,
                                               const AtomicFileOptions& options, const TiffColorTags& color = {});

    // Explicit UTF-8 XMP packet, consumed synchronously and copied by libtiff.
    // Empty packet preserves the existing writer. Maximum 32 MiB; no NUL bytes.
    LFS_IMAGE_CODECS_API Status write_tiff_u16(const std::filesystem::path& path,
                                               std::span<const std::uint8_t> data, int width, int height, int channels,
                                               std::size_t row_stride, const std::optional<std::string>& description,
                                               const AtomicFileOptions& options, const TiffColorTags& color, std::string_view xmp);

    // Packed display bytes: preserve samples, PNG/TIFF/JXL/AVIF alpha and legacy codec
    // defaults. JPEG drops RGBA alpha; gray-alpha JPEG is unsupported. This is
    // shared atomic file service; legacy callers keep overwrite enabled and do
    // not create directories. No linear/EXR conversion. JXL and AVIF are lossless sRGB;
    // JPEG quality is ignored and nonempty metadata comments are rejected.
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

    // PNG-only XMP packet fragments, borrowed for this synchronous call. Emits one
    // uncompressed iTXt before IDAT using the existing serial/parallel encoder.
    // The caller supplies UTF-8 XML; NUL and oversized chunk payloads reject
    // before publication. Empty metadata preserves the existing byte stream.
    LFS_IMAGE_CODECS_API Status write_image_u8(const std::filesystem::path& path, const std::uint8_t* data,
                                               int width, int height, int channels, int jpeg_quality,
                                               const std::optional<std::string>& comment, const AtomicFileOptions&,
                                               bool jpeg_full_chroma, std::span<const std::string_view> png_xmp);

} // namespace lfs::core::image_codecs
