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
                              HeifOffset };
    struct MetadataLimits {
        // Bounds the JPEG header window, not the size of the source image.
        std::size_t jpeg_header_bytes = 1024 * 1024;
        std::size_t block_bytes = 1024 * 1024;
        std::size_t total_bytes = 4 * 1024 * 1024;
        std::size_t blocks = 64;
    };
    struct ExifBlock {
        ExifEnvelope envelope = ExifEnvelope::JpegApp1;
        // HEIF image and metadata item identifiers; absent for JPEG.
        std::optional<std::uint32_t> image_item;
        std::optional<std::uint32_t> metadata_item;
        // Exact original envelope and payload, independent of reader lifetime.
        std::vector<std::uint8_t> bytes;
    };
    struct SourceExif {
        MetadataStatus status = MetadataStatus::NotExamined;
        // Explicit scope: JPEG markers before SOS, or the HEIF primary image.
        std::string scope;
        std::string diagnostic;
        std::vector<ExifBlock> blocks;
    };
    // Explicit metadata-only operation. Does not validate or decode image pixels,
    // scan auxiliary HEIF images, change training EV, or normalize orientation.
    // Limits bound retained blocks; libheif's container allocations are separate.
    [[nodiscard]] LFS_IMAGE_CODECS_API SourceExif read_source_exif(
        const std::filesystem::path&, const MetadataLimits& = {});
} // namespace lfs::core::image_codecs
