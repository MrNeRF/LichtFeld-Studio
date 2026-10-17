/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
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
                              PngExif };
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
    struct ExifBlock {
        ExifEnvelope envelope = ExifEnvelope::JpegApp1;
        // HEIF image and metadata item identifiers; absent for JPEG/PNG.
        std::optional<std::uint32_t> image_item;
        std::optional<std::uint32_t> metadata_item;
        // Exact original envelope and payload, independent of reader lifetime.
        std::vector<std::uint8_t> bytes;
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
        // or the observed TIFF library metadata read set.
        std::string scope;
        std::string diagnostic;
        std::vector<ExifBlock> blocks;
        // TIFF has distributed IFDs, not standalone EXIF blocks. Decoded tags
        // and observed original ranges are represented separately.
        std::optional<TiffSourceMetadata> tiff;
    };
    // Explicit metadata-only operation. Does not validate or decode image pixels,
    // scan auxiliary HEIF images, change training EV, or normalize orientation.
    // Limits bound retained blocks/ranges and decoded TIFF values. TIFF library
    // allocations have separate per-handle ceilings; libheif allocations remain
    // separate. TIFF results are observed source spans, not relocatable EXIF.
    [[nodiscard]] LFS_IMAGE_CODECS_API SourceExif read_source_exif(
        const std::filesystem::path&, const MetadataLimits& = {});
} // namespace lfs::core::image_codecs
