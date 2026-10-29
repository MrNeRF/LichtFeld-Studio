/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace lfs::core::image_codecs {
    enum class MetadataStatus { NotExamined,
                                Absent,
                                Collected,
                                Limited,
                                Malformed,
                                Unsupported,
                                IoError };
    enum class ExifEnvelope { JpegApp1,
                              HeifOffset,
                              PngExif,
                              JxlOffset };
    struct MetadataLimits {
        // Bounds the JPEG header window, not the size of the source image.
        std::size_t jpeg_header_bytes = 1024 * 1024;
        std::size_t block_bytes = 1024 * 1024;
        std::size_t total_bytes = 4 * 1024 * 1024;
        std::size_t blocks = 64;
        // PNG bytes examined; IDAT payloads are skipped without decoding/CRC.
        std::size_t png_scan_bytes = 64 * 1024 * 1024;
        // Non-IDAT chunks; all chunk headers/CRC bytes still count above.
        std::size_t png_chunks = 4096;
        // TIFF library reads, visited directories and copied decoded values.
        std::size_t tiff_reads = 4096;
        std::size_t tiff_read_bytes = 64 * 1024 * 1024;
        std::size_t tiff_directories = 64;
        std::size_t tiff_entries = 4096;
        std::size_t tiff_value_bytes = 8 * 1024 * 1024;
    };
    struct JxlXmpBox {
        std::uint64_t ordinal = 0; // Provider box-event ordinal, including structural boxes.
        bool brotli_compressed = false;
    };
    using JxlBox = JxlXmpBox;
    struct ExifBlock {
        ExifEnvelope envelope = ExifEnvelope::JpegApp1;
        // HEIF image and metadata item identifiers; absent for other carriers.
        std::optional<std::uint32_t> image_item;
        std::optional<std::uint32_t> metadata_item;
        // JPEG/PNG retain original envelopes; HEIF/JXL retain provider-decoded
        // values, including their TIFF-offset prefix, independent of reader lifetime.
        std::vector<std::uint8_t> bytes;
        std::optional<JxlBox> jxl_box;
    };
    struct TiffSourceRange {
        std::uint64_t offset = 0;
        // Exact source bytes read by libtiff, never a relocated EXIF envelope.
        std::vector<std::uint8_t> bytes;
    };
    struct TiffSourceTag {
        std::uint32_t id = 0;
        std::uint16_t type = 0;
        std::uint64_t count = 0;
        // Library-decoded values, NOT original on-disk representations.
        // Rational values have already been converted by libtiff.
        std::vector<std::uint64_t> unsigned_values;
        std::vector<std::int64_t> signed_values;
        std::vector<double> floating_values;
        std::vector<std::uint8_t> bytes;
        std::size_t decoded_storage_bytes = 0;
    };
    struct TiffSourceDirectory {
        std::uint64_t offset = 0;
        std::optional<std::uint64_t> parent;
        std::string kind;
        std::vector<TiffSourceTag> tags;
    };
    struct TiffSourceMetadata {
        bool big_tiff = false;
        bool big_endian = false;
        // An observed sparse source snapshot, not a complete metadata archive.
        // Offsets remain relative to the original file; unvisited vendor data
        // and ignored fields need not be present. Pixel data is not requested.
        std::vector<TiffSourceRange> source_ranges;
        std::vector<TiffSourceDirectory> directories;
    };
    struct SourceExif {
        MetadataStatus status = MetadataStatus::NotExamined;
        // Explicit scope: JPEG before SOS, PNG through IEND, HEIF primary image,
        // JXL container Exif boxes, or the observed TIFF library metadata read set.
        std::string scope;
        std::string diagnostic;
        std::vector<ExifBlock> blocks;
        // TIFF has distributed IFDs, not standalone EXIF blocks. Decoded tags
        // and observed original ranges are represented separately.
        std::optional<TiffSourceMetadata> tiff;
    };
    enum class XmpEnvelope { JpegApp1,
                             JpegExtendedApp1,
                             PngInternationalText,
                             TiffTag,
                             HeifMime,
                             JxlXml,
                             ExrString };
    struct XmpLimits {
        std::size_t blocks = 64;
        std::size_t block_bytes = 1024 * 1024;
        // Aggregate retained envelopes/payloads, inflated PNG text and HEIF MIME strings.
        std::size_t total_bytes = 4 * 1024 * 1024;
        // Decoded output limit for compressed PNG text and JXL XML boxes.
        std::size_t inflated_bytes = 1024 * 1024;
    };
    struct TiffXmpReference {
        std::size_t directory = 0;
        std::size_t tag = 0;
    };
    struct HeifXmpItem {
        std::uint32_t image_item = 0;
        std::uint32_t metadata_item = 0;
        std::string content_type;
        std::string content_encoding;
    };
    struct XmpBlock {
        XmpEnvelope envelope = XmpEnvelope::JpegApp1;
        // Exact APP1/iTXt/EXR string bytes, or provider-decoded HEIF MIME/JXL XML payloads.
        // TIFF packets reference library-decoded bytes owned by SourceExif.
        std::vector<std::uint8_t> bytes;
        // Used only for compressed PNG text; originals remain unchanged.
        std::optional<std::vector<std::uint8_t>> inflated;
        std::size_t payload_offset = 0;
        bool compressed = false;
        MetadataStatus status = MetadataStatus::Collected;
        std::string diagnostic;
        // PNG chunk CRC; JPEG has no corresponding checksum.
        std::optional<bool> crc_valid;
        std::optional<TiffXmpReference> tiff_tag;
        std::optional<HeifXmpItem> heif_item;
        std::optional<JxlXmpBox> jxl_box;
    };
    struct JpegExtendedXmpSegment {
        std::string_view guid;
        std::uint32_t full_length = 0;
        std::uint32_t offset = 0;
        std::span<const std::uint8_t> bytes;
    };
    // Validates existing APP1 framing and borrows its GUID and payload. The
    // returned views live only as long as the block's unchanged owning bytes.
    // Collection and reassembly share this check; it does not parse XML.
    [[nodiscard]] LFS_IMAGE_CODECS_API std::optional<JpegExtendedXmpSegment> source_extended_xmp_segment(
        const XmpBlock&, const char** diagnostic = nullptr) noexcept;
    // Resolves retained bytes; an empty span is a valid empty packet.
    // TIFF references borrow its library-decoded tag bytes; source ranges retain
    // on-disk representations. HEIF/JXL own provider-decoded bytes; JPEG/PNG own originals.
    [[nodiscard]] LFS_IMAGE_CODECS_API std::optional<std::span<const std::uint8_t>> source_xmp_bytes(
        const SourceExif&, const XmpBlock&) noexcept;
    struct SourceXmp {
        MetadataStatus status = MetadataStatus::NotExamined;
        std::string scope;
        std::string diagnostic;
        std::vector<XmpBlock> blocks;
    };
    struct SourceImageMetadata {
        SourceExif exif;
        SourceXmp xmp;
    };
    // Shares the EXIF carrier scan. XMP remains opaque: no XML parsing, entity
    // resolution, property interpretation or extended-JPEG reassembly.
    [[nodiscard]] LFS_IMAGE_CODECS_API SourceImageMetadata read_source_metadata(
        const std::filesystem::path&, const MetadataLimits& = {}, const XmpLimits& = {});
    // Explicit metadata-only operation. Does not validate or decode image pixels,
    // scan auxiliary HEIF images, change training EV, or normalize orientation.
    // Limits bound retained blocks/ranges and decoded TIFF values. TIFF library
    // allocations have separate per-handle ceilings; libheif allocations remain
    // separate. TIFF results are observed source spans, not relocatable EXIF.
    [[nodiscard]] LFS_IMAGE_CODECS_API SourceExif read_source_exif(
        const std::filesystem::path&, const MetadataLimits& = {});
} // namespace lfs::core::image_codecs
